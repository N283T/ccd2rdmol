"""ccd2rdmol - Convert PDB CCD files to RDKit molecules."""

from importlib.metadata import PackageNotFoundError, version

from .converter import (
    chemcomp_to_mol,
    read_ccd_block,
    read_ccd_file,
)
from .models import ConformerType, ConversionResult, SanitizationResult
from .sanitizer import handle_implicit_hydrogens, sanitize

try:
    __version__ = version("ccd2rdmol")
except PackageNotFoundError:
    __version__ = "0.0.0"

__all__ = [
    "ConformerType",
    "ConversionResult",
    "SanitizationResult",
    "chemcomp_to_mol",
    "handle_implicit_hydrogens",
    "read_ccd_block",
    "read_ccd_file",
    "sanitize",
]
