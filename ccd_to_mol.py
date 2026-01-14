import re
import sys
from io import StringIO

import gemmi  # type: ignore
from rdkit import Chem, rdBase


# ============================================================================
# Constants
# ============================================================================

BOND_TYPE_TO_RDKIT = {
    gemmi.BondType.Unspec:   Chem.BondType.UNSPECIFIED,
    gemmi.BondType.Single:   Chem.BondType.SINGLE,
    gemmi.BondType.Double:   Chem.BondType.DOUBLE,
    gemmi.BondType.Triple:   Chem.BondType.TRIPLE,
    gemmi.BondType.Aromatic: Chem.BondType.AROMATIC,
    gemmi.BondType.Deloc:    Chem.BondType.OTHER,
    gemmi.BondType.Metal:    Chem.BondType.OTHER,
}

METALS_SMART = (
    "[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,"
    "Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi]"
)


# ============================================================================
# Helper functions: Adding atoms and bonds
# ============================================================================

def _add_atoms_from_cc(rwmol: Chem.RWMol, cc_atoms: gemmi.ChemCompAtoms) -> Chem.RWMol:
    """Add atoms from chemical component to RDKit molecule.

    Args:
        rwmol: RDKit mutable molecule object
        cc_atoms: gemmi chemical component atoms list

    Returns:
        RDKit molecule with atoms added
    """
    for atom in cc_atoms:
        a = Chem.Atom(atom.el.atomic_number)  # X case returns 0, which is fine
        if atom.el.name == "D":
            a.SetIsotope(2)
        a.SetProp("name", atom.id)
        a.SetFormalCharge(int(atom.charge))
        rwmol.AddAtom(a)
    return rwmol


def _add_bonds_from_cc(rwmol: Chem.RWMol, cc_bonds: gemmi.RestraintsBonds, atom_ids: list[str]) -> Chem.RWMol:
    """Add bonds from chemical component to RDKit molecule.

    Args:
        rwmol: RDKit mutable molecule object
        cc_bonds: gemmi bond restraints list
        atom_ids: List of atom IDs

    Returns:
        RDKit molecule with bonds added
    """
    for bond in cc_bonds:
        order = BOND_TYPE_TO_RDKIT[bond.type]
        rwmol.AddBond(
            atom_ids.index(bond.id1.atom),
            atom_ids.index(bond.id2.atom),
            order=order
        )
    return rwmol


# ============================================================================
# Molecule processing functions
# ============================================================================

def handle_implicit_hydrogens(mol: Chem.RWMol) -> Chem.RWMol:
    """Forbid atoms without explicit hydrogen partners from getting implicit hydrogens.

    Args:
        mol: RDKit molecule to be modified

    Returns:
        Modified RDKit molecule
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue

        no_Hs = True
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 1:
                no_Hs = False
                break

        atom.SetNoImplicit(no_Hs)
    return mol


def fix_molecule(rwmol: Chem.RWMol) -> bool:
    """Single molecule sanitization process. Currently only handles valence errors.

    Args:
        rwmol: RDKit molecule to be sanitized

    Returns:
        Whether sanitization succeeded
    """
    attempts = 10
    success = False
    saved_std_err = sys.stderr
    rdBase.LogToPythonStderr()

    while (not success) and attempts >= 0:
        # Create new StringIO for each iteration to reset log
        log = sys.stderr = StringIO()
        sanitization_result = Chem.SanitizeMol(rwmol, catchErrors=True)

        if sanitization_result == 0:
            sys.stderr = saved_std_err
            return True

        sanitization_failures = re.findall(
            "[a-zA-Z]{1,2}, \\d+", log.getvalue())

        if not sanitization_failures:
            sys.stderr = saved_std_err
            return False

        for sanitization_failure in sanitization_failures:
            split_object = sanitization_failure.split(
                ",")  # [0] element [1] valency
            element = split_object[0]
            valency = int(split_object[1].strip())

            smarts_metal_check = Chem.MolFromSmarts(
                METALS_SMART + "~[{}]".format(element))
            if smarts_metal_check is None:
                continue

            metal_atom_bonds = rwmol.GetSubstructMatches(smarts_metal_check)
            Chem.SanitizeMol(
                rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)

            for metal_index, other_index in metal_atom_bonds:
                metal_atom = rwmol.GetAtomWithIdx(metal_index)
                other_atom = rwmol.GetAtomWithIdx(other_index)
                # Change bond to dative towards the metal
                if other_atom.GetExplicitValence() == valency:
                    rwmol.RemoveBond(metal_atom.GetIdx(), other_atom.GetIdx())
                    rwmol.AddBond(
                        other_atom.GetIdx(),
                        metal_atom.GetIdx(),
                        Chem.BondType.DATIVE,
                    )
            rwmol.UpdatePropertyCache()  # Regenerate valence records
        attempts -= 1

    sys.stderr = saved_std_err
    return False


def sanitize_molecule(rwmol: Chem.RWMol) -> tuple[Chem.RWMol, bool]:
    """Attempt to sanitize molecule in place. RDKit's standard error can be
    processed to find out what went wrong with sanitization and fix the molecule.

    Args:
        rwmol: RDKit molecule to be sanitized

    Returns:
        Tuple of sanitized molecule and success flag
    """
    success = False
    try:
        mol_copy = Chem.RWMol(rwmol)
        success = fix_molecule(mol_copy)
        if not success:
            Chem.SanitizeMol(
                rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)
            return rwmol, success
        Chem.Kekulize(mol_copy)

    except Exception as e:
        print(e, file=sys.stderr)
        Chem.SanitizeMol(
            rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)
        return rwmol, success

    return mol_copy, success


# ============================================================================
# Conversion functions
# ============================================================================

def chemcomp_to_rdkit(cc: gemmi.ChemComp) -> Chem.Mol:
    """Convert gemmi chemical component to RDKit molecule.

    Args:
        cc: gemmi chemical component object

    Returns:
        RDKit molecule (hydrogens removed)
    """
    cc_atoms = cc.atoms
    cc_bonds = cc.rt.bonds
    atom_ids = [atom.id for atom in cc_atoms]

    rwmol = Chem.RWMol()
    rwmol = _add_atoms_from_cc(rwmol, cc_atoms)
    rwmol = _add_bonds_from_cc(rwmol, cc_bonds, atom_ids)
    rwmol = handle_implicit_hydrogens(rwmol)
    rwmol, success = sanitize_molecule(rwmol)
    no_h = Chem.RemoveHs(rwmol.GetMol(), sanitize=success)
    return no_h


def ccd_to_mol(cc_block: gemmi.cif.Block) -> Chem.Mol:
    """Convert CCD block to RDKit molecule.

    Args:
        cc_block: gemmi CIF block

    Returns:
        RDKit molecule
    """
    cc = gemmi.make_chemcomp_from_block(cc_block)
    mol = chemcomp_to_rdkit(cc)
    return mol
