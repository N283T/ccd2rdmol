# ccd2rdmol

A lightweight Python library and CLI tool for converting PDB Chemical Component Dictionary (CCD) files to RDKit molecule objects.

This project is a simplified implementation inspired by [pdbeccdutils](https://github.com/PDBeurope/ccdutils), focusing solely on CCD to RDKit conversion with 3D conformer support.

## Features

- Fast CIF parsing using **gemmi**
- Conversion to **RDKit** molecule objects
- Support for both Ideal and Model 3D conformers
- Automatic metal bond to dative bond conversion
- Stereochemistry assignment from 3D coordinates
- CLI tool with rich output

## Installation

```bash
uv add ccd2rdmol
```

Or for development:

```bash
git clone https://github.com/nagaet/ccd2rdmol.git
cd ccd2rdmol
uv sync
```

## Usage

### As a Library

```python
from ccd2rdmol import read_ccd_file, read_ccd_block
import gemmi

# Read from file
result = read_ccd_file("ATP.cif")
mol = result.mol

print(f"Atoms: {mol.GetNumAtoms()}")
print(f"Bonds: {mol.GetNumBonds()}")
print(f"Conformers: {mol.GetNumConformers()}")  # 2 (IDEAL + MODEL)
print(f"Sanitized: {result.sanitized}")

# With options
result = read_ccd_file(
    "ATP.cif",
    sanitize_mol=True,      # Sanitize molecule (default: True)
    add_conformers=True,    # Add 3D conformers (default: True)
    remove_hydrogens=True,  # Remove hydrogens (default: True)
)

# From gemmi CIF block
doc = gemmi.cif.read("components.cif")
for block in doc:
    result = read_ccd_block(block)
    print(f"{block.name}: {result.mol.GetNumAtoms()} atoms")
```

### As a CLI

```bash
# Output SMILES to stdout
ccd2rdmol convert ATP.cif

# Write to MOL file
ccd2rdmol convert ATP.cif -o ATP.mol

# Write to SDF format
ccd2rdmol convert ATP.cif -o ATP.sdf

# Keep hydrogen atoms
ccd2rdmol convert ATP.cif --keep-hydrogens

# Show verbose information
ccd2rdmol convert ATP.cif -v

# Show molecule information only
ccd2rdmol info ATP.cif
```

### CLI Options

```
ccd2rdmol convert [OPTIONS] INPUT_FILE

Arguments:
  INPUT_FILE  Input CCD CIF file path [required]

Options:
  -o, --output PATH       Output file path (.mol, .sdf)
  -f, --format TEXT       Output format (mol, sdf, smiles, inchi)
  --no-sanitize           Skip sanitization step
  --no-conformers         Skip adding 3D conformers
  -H, --keep-hydrogens    Keep hydrogen atoms
  -v, --verbose           Show detailed information
  --help                  Show help message
```

## Development

This project uses [poethepoet](https://github.com/nat-n/poethepoet) as a task runner.

```bash
# Install dev dependencies
uv sync

# Format code (ruff format)
uv run poe format

# Lint (ruff check)
uv run poe lint

# Lint and auto-fix
uv run poe fix

# Type check (ty)
uv run poe check

# Run tests
uv run poe test

# Run all checks (format, lint, check, test)
uv run poe all

# Clean cache files
uv run poe clean
```

## Acknowledgments

This project is inspired by and built upon concepts from [pdbeccdutils](https://github.com/PDBeurope/ccdutils) by PDBe (Protein Data Bank in Europe). Test data files are derived from the pdbeccdutils test suite.

We thank the PDBe team for their excellent work on chemical component processing tools.

## License

MIT License

Test data files in `tests/data/` are from [pdbeccdutils](https://github.com/PDBeurope/ccdutils) (Apache-2.0 License).
