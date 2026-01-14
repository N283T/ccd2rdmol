"""CCD to RDKit molecule converter."""

from __future__ import annotations

from typing import TYPE_CHECKING

import gemmi
from rdkit import Chem
from rdkit.Chem import rdGeometry

from .models import ConformerType, ConversionResult
from .sanitizer import handle_implicit_hydrogens, sanitize

if TYPE_CHECKING:
    pass

# Bond type mapping from gemmi to RDKit
BOND_TYPE_MAP: dict[gemmi.BondType, Chem.BondType] = {
    gemmi.BondType.Unspec: Chem.BondType.UNSPECIFIED,
    gemmi.BondType.Single: Chem.BondType.SINGLE,
    gemmi.BondType.Double: Chem.BondType.DOUBLE,
    gemmi.BondType.Triple: Chem.BondType.TRIPLE,
    gemmi.BondType.Aromatic: Chem.BondType.AROMATIC,
    gemmi.BondType.Deloc: Chem.BondType.OTHER,
    gemmi.BondType.Metal: Chem.BondType.OTHER,
}


def _add_atoms(rwmol: Chem.RWMol, atoms: gemmi.ChemCompAtoms) -> list[str]:
    """Add atoms from chemical component to RDKit molecule.

    Args:
        rwmol: RDKit mutable molecule.
        atoms: Gemmi chemical component atoms.

    Returns:
        List of atom IDs for bond indexing.
    """
    atom_ids: list[str] = []

    for atom in atoms:
        rdkit_atom = Chem.Atom(atom.el.atomic_number)

        if atom.el.name == "D":
            rdkit_atom.SetIsotope(2)

        rdkit_atom.SetProp("name", atom.id)
        rdkit_atom.SetFormalCharge(int(atom.charge))
        rwmol.AddAtom(rdkit_atom)
        atom_ids.append(atom.id)

    return atom_ids


def _add_bonds(
    rwmol: Chem.RWMol,
    bonds: gemmi.RestraintsBonds,
    atom_ids: list[str],
    errors: list[str],
) -> None:
    """Add bonds from chemical component to RDKit molecule.

    Args:
        rwmol: RDKit mutable molecule.
        bonds: Gemmi bond restraints.
        atom_ids: List of atom IDs for index lookup.
        errors: List to append error messages to.
    """
    for bond in bonds:
        try:
            idx1 = atom_ids.index(bond.id1.atom)
            idx2 = atom_ids.index(bond.id2.atom)
            order = BOND_TYPE_MAP.get(bond.type, Chem.BondType.UNSPECIFIED)
            rwmol.AddBond(idx1, idx2, order=order)
        except ValueError:
            errors.append(f"Bond atom not found: {bond.id1.atom} - {bond.id2.atom}")
        except RuntimeError:
            errors.append(f"Duplicate bond: {bond.id1.atom} - {bond.id2.atom}")


def _str_to_float(value: str) -> float:
    """Convert CIF string to float, handling missing values.

    Args:
        value: String value from CIF.

    Returns:
        Float value, 0.0 if missing or invalid.
    """
    if not value or value in ("?", "."):
        return 0.0
    try:
        return float(value)
    except ValueError:
        return 0.0


def _add_conformer(
    rwmol: Chem.RWMol,
    cif_block: gemmi.cif.Block,
    conf_type: ConformerType,
) -> bool:
    """Add conformer with 3D coordinates to molecule.

    Args:
        rwmol: RDKit mutable molecule.
        cif_block: Gemmi CIF block containing coordinate data.
        conf_type: Type of conformer (IDEAL or MODEL).

    Returns:
        True if conformer was added successfully.
    """
    if "_chem_comp_atom." not in cif_block.get_mmcif_category_names():
        return False

    if conf_type == ConformerType.IDEAL:
        coord_fields = [
            "pdbx_model_Cartn_x_ideal",
            "pdbx_model_Cartn_y_ideal",
            "pdbx_model_Cartn_z_ideal",
        ]
    else:
        coord_fields = ["model_Cartn_x", "model_Cartn_y", "model_Cartn_z"]

    try:
        atoms_table = cif_block.find("_chem_comp_atom.", coord_fields)
    except RuntimeError:
        return False

    if not atoms_table:
        return False

    num_atoms = rwmol.GetNumAtoms()
    conformer = Chem.Conformer(num_atoms)

    has_valid_coords = False
    for row in atoms_table:
        x = _str_to_float(row[f"_chem_comp_atom.{coord_fields[0]}"])
        y = _str_to_float(row[f"_chem_comp_atom.{coord_fields[1]}"])
        z = _str_to_float(row[f"_chem_comp_atom.{coord_fields[2]}"])

        if x != 0.0 or y != 0.0 or z != 0.0:
            has_valid_coords = True

        position = rdGeometry.Point3D(x, y, z)
        conformer.SetAtomPosition(row.row_index, position)

    if not has_valid_coords:
        return False

    conformer.SetProp("name", conf_type.name)
    rwmol.AddConformer(conformer, assignId=True)
    return True


def _assign_stereochemistry(mol: Chem.Mol) -> None:
    """Assign stereochemistry from 3D coordinates.

    Args:
        mol: RDKit molecule with conformers.
    """
    conformers = mol.GetConformers()
    if not conformers:
        return

    # Prefer ideal conformer, fall back to model
    conf_id = -1
    for conf in conformers:
        if conf.HasProp("name"):
            name = conf.GetProp("name")
            if name == ConformerType.IDEAL.name:
                conf_id = conf.GetId()
                break
            if name == ConformerType.MODEL.name:
                conf_id = conf.GetId()

    if conf_id >= 0:
        Chem.rdmolops.AssignStereochemistryFrom3D(mol, conf_id)


def chemcomp_to_mol(
    cc: gemmi.ChemComp,
    cif_block: gemmi.cif.Block,
    *,
    sanitize_mol: bool = True,
    add_conformers: bool = True,
    remove_hydrogens: bool = True,
) -> ConversionResult:
    """Convert gemmi ChemComp to RDKit molecule.

    Args:
        cc: Gemmi chemical component.
        cif_block: Gemmi CIF block for coordinate data.
        sanitize_mol: Whether to sanitize the molecule.
        add_conformers: Whether to add 3D conformers.
        remove_hydrogens: Whether to remove hydrogen atoms.

    Returns:
        ConversionResult with molecule and metadata.
    """
    errors: list[str] = []
    warnings: list[str] = []
    sanitized = False

    rwmol = Chem.RWMol()
    atom_ids = _add_atoms(rwmol, cc.atoms)
    _add_bonds(rwmol, cc.rt.bonds, atom_ids, errors)
    handle_implicit_hydrogens(rwmol)

    if add_conformers:
        ideal_added = _add_conformer(rwmol, cif_block, ConformerType.IDEAL)
        model_added = _add_conformer(rwmol, cif_block, ConformerType.MODEL)

        if not ideal_added and not model_added:
            warnings.append("No valid conformer coordinates found")

    if sanitize_mol:
        result = sanitize(rwmol)
        rwmol = Chem.RWMol(result.mol)
        sanitized = result.success

        if sanitized and rwmol.GetNumConformers() > 0:
            _assign_stereochemistry(rwmol)

    mol = rwmol.GetMol()
    if remove_hydrogens:
        mol = Chem.RemoveHs(mol, sanitize=sanitized)

    return ConversionResult(
        mol=mol,
        sanitized=sanitized,
        errors=errors,
        warnings=warnings,
    )


def read_ccd_file(
    path: str,
    *,
    sanitize_mol: bool = True,
    add_conformers: bool = True,
    remove_hydrogens: bool = True,
) -> ConversionResult:
    """Read CCD CIF file and convert to RDKit molecule.

    Args:
        path: Path to CIF file.
        sanitize_mol: Whether to sanitize the molecule.
        add_conformers: Whether to add 3D conformers.
        remove_hydrogens: Whether to remove hydrogen atoms.

    Returns:
        ConversionResult with molecule and metadata.

    Raises:
        FileNotFoundError: If file does not exist.
        ValueError: If file cannot be parsed.
    """
    doc = gemmi.cif.read(path)
    cif_block = doc.sole_block()
    cc = gemmi.make_chemcomp_from_block(cif_block)

    return chemcomp_to_mol(
        cc,
        cif_block,
        sanitize_mol=sanitize_mol,
        add_conformers=add_conformers,
        remove_hydrogens=remove_hydrogens,
    )


def read_ccd_block(
    cif_block: gemmi.cif.Block,
    *,
    sanitize_mol: bool = True,
    add_conformers: bool = True,
    remove_hydrogens: bool = True,
) -> ConversionResult:
    """Convert CCD CIF block to RDKit molecule.

    Args:
        cif_block: Gemmi CIF block.
        sanitize_mol: Whether to sanitize the molecule.
        add_conformers: Whether to add 3D conformers.
        remove_hydrogens: Whether to remove hydrogen atoms.

    Returns:
        ConversionResult with molecule and metadata.
    """
    cc = gemmi.make_chemcomp_from_block(cif_block)

    return chemcomp_to_mol(
        cc,
        cif_block,
        sanitize_mol=sanitize_mol,
        add_conformers=add_conformers,
        remove_hydrogens=remove_hydrogens,
    )
