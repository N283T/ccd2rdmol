"""Molecule sanitization utilities."""

from __future__ import annotations

from rdkit import Chem, rdBase

from .models import SanitizationResult

METALS_SMART = (
    "[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,"
    "Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi]"
)

_MAX_SANITIZE_ATTEMPTS = 11


def handle_implicit_hydrogens(mol: Chem.RWMol) -> None:
    """Forbid atoms without explicit hydrogen partners from getting implicit hydrogens.

    Args:
        mol: RDKit molecule to be modified in place.
    """
    for atom in mol.GetAtoms():
        if atom.GetAtomicNum() == 1:
            continue

        has_hydrogen = False
        for bond in atom.GetBonds():
            other = bond.GetOtherAtom(atom)
            if other.GetAtomicNum() == 1:
                has_hydrogen = True
                break

        atom.SetNoImplicit(not has_hydrogen)


def _fix_valence_errors(rwmol: Chem.RWMol) -> bool:
    """Fix valence errors by converting metal bonds to dative bonds.

    Uses DetectChemistryProblems to identify atoms with valence issues,
    then converts their metal bonds to dative bonds.

    Args:
        rwmol: RDKit molecule to be sanitized in place.

    Returns:
        Whether sanitization succeeded.
    """
    with rdBase.BlockLogs():
        for _ in range(_MAX_SANITIZE_ATTEMPTS):
            sanitization_result = Chem.SanitizeMol(rwmol, catchErrors=True)
            if sanitization_result == 0:
                return True

            problems = Chem.DetectChemistryProblems(rwmol)
            valence_problems = [p for p in problems if p.GetType() == "AtomValenceException"]
            if not valence_problems:
                return False

            for problem in valence_problems:
                atom_idx = problem.GetAtomIdx()
                atom = rwmol.GetAtomWithIdx(atom_idx)
                element = atom.GetSymbol()
                valency = atom.GetExplicitValence()

                smarts_pattern = Chem.MolFromSmarts(f"{METALS_SMART}~[{element}]")
                if smarts_pattern is None:
                    continue

                metal_bonds = rwmol.GetSubstructMatches(smarts_pattern)
                Chem.SanitizeMol(rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)

                for metal_idx, other_idx in metal_bonds:
                    other_atom = rwmol.GetAtomWithIdx(other_idx)
                    if other_atom.GetExplicitValence() == valency:
                        rwmol.RemoveBond(metal_idx, other_idx)
                        rwmol.AddBond(other_idx, metal_idx, Chem.BondType.DATIVE)

                rwmol.UpdatePropertyCache()

    return False


def sanitize(rwmol: Chem.RWMol) -> SanitizationResult:
    """Sanitize molecule and fix common issues.

    Creates a copy of the input molecule; the original is never modified.

    Args:
        rwmol: RDKit molecule to be sanitized.

    Returns:
        SanitizationResult with sanitized molecule and success status.
    """
    mol_copy = Chem.RWMol(rwmol)
    try:
        success = _fix_valence_errors(mol_copy)

        if not success:
            Chem.SanitizeMol(mol_copy, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)
            return SanitizationResult(mol=mol_copy, success=False)

        Chem.Kekulize(mol_copy)
        return SanitizationResult(mol=mol_copy, success=True)

    except Exception:
        mol_fallback = Chem.RWMol(rwmol)
        Chem.SanitizeMol(mol_fallback, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)
        return SanitizationResult(mol=mol_fallback, success=False)
