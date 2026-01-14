"""Molecule sanitization utilities."""

from __future__ import annotations

import re
import sys
from io import StringIO
from typing import TYPE_CHECKING

from rdkit import Chem, rdBase

from .models import SanitizationResult

if TYPE_CHECKING:
    pass

METALS_SMART = (
    "[Li,Na,K,Rb,Cs,Fr,Be,Mg,Ca,Sr,Ba,Ra,Sc,Ti,V,Cr,Mn,Fe,Co,Ni,Cu,Zn,Al,Ga,Y,Zr,Nb,Mo,"
    "Tc,Ru,Rh,Pd,Ag,Cd,In,Sn,Hf,Ta,W,Re,Os,Ir,Pt,Au,Hg,Tl,Pb,Bi]"
)


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

    Args:
        rwmol: RDKit molecule to be sanitized in place.

    Returns:
        Whether sanitization succeeded.
    """
    attempts = 10
    saved_stderr = sys.stderr
    rdBase.LogToPythonStderr()

    while attempts >= 0:
        log = sys.stderr = StringIO()
        sanitization_result = Chem.SanitizeMol(rwmol, catchErrors=True)

        if sanitization_result == 0:
            sys.stderr = saved_stderr
            return True

        sanitization_failures = re.findall(r"[a-zA-Z]{1,2}, \d+", log.getvalue())

        if not sanitization_failures:
            sys.stderr = saved_stderr
            return False

        for failure in sanitization_failures:
            parts = failure.split(",")
            element = parts[0]
            valency = int(parts[1].strip())

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

        attempts -= 1

    sys.stderr = saved_stderr
    return False


def sanitize(rwmol: Chem.RWMol) -> SanitizationResult:
    """Sanitize molecule and fix common issues.

    Args:
        rwmol: RDKit molecule to be sanitized.

    Returns:
        SanitizationResult with sanitized molecule and success status.
    """
    try:
        mol_copy = Chem.RWMol(rwmol)
        success = _fix_valence_errors(mol_copy)

        if not success:
            Chem.SanitizeMol(rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)
            return SanitizationResult(mol=rwmol, success=False)

        Chem.Kekulize(mol_copy)
        return SanitizationResult(mol=mol_copy, success=True)

    except Exception:
        Chem.SanitizeMol(rwmol, sanitizeOps=Chem.SanitizeFlags.SANITIZE_CLEANUP)
        return SanitizationResult(mol=rwmol, success=False)
