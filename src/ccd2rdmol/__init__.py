"""ccd2rdmol - Convert PDB CCD files to RDKit molecules."""

from .converter import (
    chemcomp_to_mol,
    read_ccd_block,
    read_ccd_file,
)
from .models import ConformerType, ConversionResult, SanitizationResult
from .sanitizer import handle_implicit_hydrogens, sanitize

__version__ = "0.1.0"

__all__ = [
    # Main functions
    "read_ccd_file",
    "read_ccd_block",
    "chemcomp_to_mol",
    # Models
    "ConversionResult",
    "SanitizationResult",
    "ConformerType",
    # Utilities
    "sanitize",
    "handle_implicit_hydrogens",
]
