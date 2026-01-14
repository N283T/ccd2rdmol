"""Data models for ccd2rdmol."""

from dataclasses import dataclass
from enum import Enum, auto

from rdkit import Chem


class ConformerType(Enum):
    """Type of conformer coordinates."""

    IDEAL = auto()
    MODEL = auto()


@dataclass(frozen=True)
class SanitizationResult:
    """Result of molecule sanitization."""

    mol: Chem.Mol
    success: bool


@dataclass(frozen=True)
class ConversionResult:
    """Result of CCD to RDKit conversion."""

    mol: Chem.Mol
    sanitized: bool
    errors: list[str]
    warnings: list[str]
