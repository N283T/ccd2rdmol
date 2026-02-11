# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Development Commands

This project uses **uv** for package management and **poethepoet** (poe) as the task runner. All commands run via `uv run poe <task>`.

```bash
uv run poe test          # Run tests with pytest
uv run poe test-cov      # Run tests with coverage report
uv run poe format        # Format with ruff
uv run poe lint          # Lint with ruff
uv run poe fix           # Lint + autofix with ruff
uv run poe check         # Type check with ty
uv run poe all           # format + lint + check + test
uv run poe nox           # Multi-version tests (Python 3.10-3.14)
```

Run a single test file or test function:
```bash
uv run pytest tests/test_converter.py -v
uv run pytest tests/test_converter.py::TestReadCcdFile::test_read_atp -v
```

## Architecture

The library converts PDB Chemical Component Dictionary (CCD) CIF files into RDKit molecule objects. The pipeline flows through four modules in `src/ccd2rdmol/`:

**converter.py** — Core pipeline orchestrating the full conversion:
1. Parse CIF with `gemmi` → `gemmi.ChemComp` + `gemmi.cif.Block`
2. Build RDKit `RWMol`: add atoms (`_add_atoms` returns `dict[str, int]` for O(1) bond lookup), add bonds (`_add_bonds` using `BOND_TYPE_MAP`)
3. Set implicit hydrogen flags via `sanitizer.handle_implicit_hydrogens`
4. Add IDEAL and/or MODEL 3D conformers from CIF coordinate columns (`_add_conformer`), rejecting degenerate conformers (>1 atom at origin)
5. Sanitize the molecule (metal bond → dative bond conversion, kekulization)
6. Assign stereochemistry from 3D coordinates (preferring IDEAL conformer)
7. Optionally remove hydrogens

Three public entry points: `read_ccd_file(path)`, `read_ccd_block(cif_block)`, `chemcomp_to_mol(cc, cif_block)`.

**sanitizer.py** — Handles RDKit sanitization failures caused by metal-ligand bonds. `_fix_valence_errors` uses `rdBase.BlockLogs()` to suppress RDKit noise and `Chem.DetectChemistryProblems()` to identify atoms with valence issues, then converts their metal bonds to dative bonds (up to 11 attempts). `sanitize()` always works on a copy; the original input is never modified.

**models.py** — Frozen dataclasses: `ConversionResult` (mol, sanitized, errors, warnings), `SanitizationResult` (mol, success), and `ConformerType` enum (IDEAL, MODEL).

**cli.py** — Optional Typer CLI (`[cli]` extra). Commands: `convert` (output SMILES/InChI/MOL/SDF) and `info` (rich table display).

## Testing

- Test CIF files live in `tests/data/random_sample/` (ATP, HEM, GOL, etc.)
- `test_cli.py` requires the `[cli]` extras (typer, rich); nox installs them automatically
- Coverage is collected only on Python 3.12 (in both nox and CI)
- CI runs ruff lint + ty type check on Python 3.12, then nox test matrix across 3.10-3.14

## Key Conventions

- All result types are **frozen dataclasses** (immutable)
- `sanitize()` never mutates the input molecule; always works on internal copies
- `BOND_TYPE_MAP` maps `gemmi.BondType` → `Chem.BondType`; `Deloc` and `Metal` map to `OTHER`
- `_str_to_float()` returns `float | None` to distinguish missing CIF values from real zeros
- Version is read dynamically from package metadata via `importlib.metadata.version()`
