# ccd2rdmol

[![CI](https://github.com/N283T/ccd2rdmol/actions/workflows/ci.yml/badge.svg)](https://github.com/N283T/ccd2rdmol/actions/workflows/ci.yml)
[![codecov](https://codecov.io/gh/N283T/ccd2rdmol/graph/badge.svg)](https://codecov.io/gh/N283T/ccd2rdmol)
[![PyPI version](https://badge.fury.io/py/ccd2rdmol.svg)](https://badge.fury.io/py/ccd2rdmol)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

A lightweight Python library and CLI tool for converting PDB Chemical Component Dictionary (CCD) files to RDKit molecule objects.

This project is a simplified implementation inspired by [pdbeccdutils](https://github.com/PDBeurope/ccdutils), focusing solely on CCD to RDKit conversion with 3D conformer support.

## Features

- Fast CIF parsing using **gemmi**
- Conversion to **RDKit** molecule objects
- Support for both Ideal and Model 3D conformers
- Automatic metal bond to dative bond conversion
- Stereochemistry assignment from 3D coordinates
- Deuterium isotope handling
- Degenerate conformer detection and rejection
- CLI tool with rich output

## Installation

```bash
# Library only
uv add ccd2rdmol

# With CLI support
uv add ccd2rdmol[cli]
```

Or with pip:

```bash
pip install ccd2rdmol
pip install ccd2rdmol[cli]
```

For development:

```bash
git clone https://github.com/N283T/ccd2rdmol.git
cd ccd2rdmol
uv sync  # CLI is included in dev dependencies
```

## Quick Start

```python
from ccd2rdmol import read_ccd_file

result = read_ccd_file("ATP.cif")
print(f"Atoms: {result.mol.GetNumAtoms()}")
print(f"Sanitized: {result.sanitized}")
```

## Usage

### Reading from a CIF File

```python
from ccd2rdmol import read_ccd_file

# Default: sanitize, add conformers, remove hydrogens
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
```

### Reading from a gemmi CIF Block

```python
import gemmi
from ccd2rdmol import read_ccd_block

doc = gemmi.cif.read("components.cif")
for block in doc:
    result = read_ccd_block(block)
    print(f"{block.name}: {result.mol.GetNumAtoms()} atoms")
```

### Low-Level API: chemcomp_to_mol

```python
import gemmi
from ccd2rdmol import chemcomp_to_mol

doc = gemmi.cif.read("ATP.cif")
block = doc.sole_block()
cc = gemmi.make_chemcomp_from_block(block)

result = chemcomp_to_mol(
    cc, block,
    sanitize_mol=False,       # Skip sanitization
    add_conformers=True,
    remove_hydrogens=False,   # Keep all hydrogens
)
```

### Generating SMILES and InChI

```python
from rdkit import Chem
from rdkit.Chem.inchi import MolToInchi
from ccd2rdmol import read_ccd_file

result = read_ccd_file("ATP.cif")

smiles = Chem.MolToSmiles(result.mol)
inchi = MolToInchi(result.mol)

print(f"SMILES: {smiles}")
print(f"InChI: {inchi}")
```

### Accessing Conformer Coordinates

```python
from ccd2rdmol import read_ccd_file

result = read_ccd_file("ATP.cif", add_conformers=True)
mol = result.mol

for conf in mol.GetConformers():
    name = conf.GetProp("name")  # "IDEAL" or "MODEL"
    print(f"\n{name} conformer:")
    for i in range(mol.GetNumAtoms()):
        pos = conf.GetAtomPosition(i)
        atom = mol.GetAtomWithIdx(i)
        print(f"  {atom.GetSymbol()} ({pos.x:.3f}, {pos.y:.3f}, {pos.z:.3f})")
```

### Handling Conversion Errors

```python
from ccd2rdmol import read_ccd_file

result = read_ccd_file("complex_molecule.cif")

if result.errors:
    print("Errors:", result.errors)

if result.warnings:
    print("Warnings:", result.warnings)

if not result.sanitized:
    print("Sanitization failed — molecule may have valence issues")
```

## API Reference

### Functions

#### `read_ccd_file(path, *, sanitize_mol=True, add_conformers=True, remove_hydrogens=True) → ConversionResult`

Read a CCD CIF file and convert to RDKit molecule.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `path` | `str` | — | Path to CIF file |
| `sanitize_mol` | `bool` | `True` | Sanitize the molecule (fix valence, kekulize) |
| `add_conformers` | `bool` | `True` | Add IDEAL and MODEL 3D conformers |
| `remove_hydrogens` | `bool` | `True` | Remove hydrogen atoms from the molecule |

Raises `FileNotFoundError` if file does not exist.

#### `read_ccd_block(cif_block, *, sanitize_mol=True, add_conformers=True, remove_hydrogens=True) → ConversionResult`

Convert a `gemmi.cif.Block` to RDKit molecule. Same parameters as `read_ccd_file` except takes a pre-parsed CIF block.

#### `chemcomp_to_mol(cc, cif_block, *, sanitize_mol=True, add_conformers=True, remove_hydrogens=True) → ConversionResult`

Convert a `gemmi.ChemComp` and `gemmi.cif.Block` to RDKit molecule. Lowest-level API for maximum control.

### Data Classes

#### `ConversionResult`

Frozen dataclass returned by all conversion functions.

| Field | Type | Description |
|-------|------|-------------|
| `mol` | `Chem.Mol` | RDKit molecule object |
| `sanitized` | `bool` | Whether sanitization succeeded |
| `errors` | `list[str]` | Errors encountered during conversion |
| `warnings` | `list[str]` | Warnings (e.g., missing conformer data) |

#### `SanitizationResult`

Frozen dataclass returned by `sanitize()`.

| Field | Type | Description |
|-------|------|-------------|
| `mol` | `Chem.Mol` | Sanitized molecule (always a copy) |
| `success` | `bool` | Whether sanitization succeeded |

## How It Works

The conversion pipeline:

1. **Parse CIF** — gemmi reads the CIF file and creates a `ChemComp` (atoms, bonds, charges) and a `cif.Block` (coordinate data)
2. **Build molecule** — Atoms are added to an RDKit `RWMol` with element types, charges, and isotope labels (Deuterium → isotope 2). Bonds are mapped from gemmi bond types to RDKit bond types via `BOND_TYPE_MAP`
3. **Set hydrogen flags** — Atoms without explicit hydrogen neighbors are flagged `NoImplicit=True` to prevent RDKit from adding implicit hydrogens
4. **Add conformers** — IDEAL and MODEL 3D coordinates are read from the CIF coordinate columns. Conformers with all-missing coordinates or degenerate positions (>1 atom at origin) are rejected
5. **Sanitize** — The sanitizer fixes valence errors caused by metal-ligand bonds by converting them to dative bonds. Uses `Chem.DetectChemistryProblems()` to identify problematic atoms and iteratively fixes them (up to 11 attempts). The original molecule is never modified
6. **Assign stereochemistry** — `AssignStereochemistryFrom3D` is called using the IDEAL conformer (preferred) or MODEL conformer
7. **Remove hydrogens** — Optionally strips hydrogen atoms from the final molecule

## Comparison with pdbeccdutils

| | ccd2rdmol | pdbeccdutils |
|---|---|---|
| **Focus** | CCD → RDKit conversion only | Full CCD processing toolkit |
| **Dependencies** | gemmi + rdkit | gemmi + rdkit + scipy + numpy + ... |
| **Scope** | Single molecules from CIF | Depictions, scaffolds, fragments, PDB integration |
| **Install size** | Minimal | ~50+ transitive dependencies |
| **Use case** | "I just need an RDKit Mol from a CCD entry" | Full cheminformatics pipeline |

If you only need to convert CCD entries to RDKit molecules, ccd2rdmol provides a simpler, lighter alternative.

## CLI

> **Note**: CLI requires extra dependencies. Install with `pip install ccd2rdmol[cli]`

```bash
# Output SMILES to stdout
ccd2rdmol convert ATP.cif

# Write to MOL file
ccd2rdmol convert ATP.cif -o ATP.mol

# Write to SDF format
ccd2rdmol convert ATP.cif -o ATP.sdf

# Output InChI
ccd2rdmol convert ATP.cif -f inchi

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

# Run tests with coverage
uv run poe test-cov

# Multi-version testing with nox (3.10, 3.11, 3.12, 3.13, 3.14)
uv run poe nox

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
