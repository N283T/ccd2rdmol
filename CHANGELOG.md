# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.2] - 2026-02-11

### Changed

- Rewrote sanitizer to use `rdBase.BlockLogs()` + `Chem.DetectChemistryProblems()` instead of unsafe `sys.stderr` redirection
- `_add_atoms` returns `dict[str, int]` for O(1) bond lookup (was O(n) list scan)
- `_str_to_float` returns `float | None` to distinguish missing CIF values from real zeros
- Reject degenerate conformers (>1 atom at origin) instead of checking single-atom origin
- `sanitize()` always works on a copy; the original input molecule is never modified

### Added

- `ty` type checker in dev dependencies and CI lint job
- Release version validation (tag vs pyproject.toml) in release workflow
- Ruff rules: SIM, PERF, RUF, FURB
- Dependency upper bounds (`gemmi<1`, `rdkit<2026`, `rich<15`, `typer<1`)
- 38 new tests (73 → 95 total), overall coverage 82% → 91%
- API reference and advanced usage examples in README

### Fixed

- Version mismatch between `__init__.py` and `pyproject.toml` (now uses `importlib.metadata`)
- `sanitize()` no longer mutates the input molecule on failure/exception paths
- Bond error handling: KeyError for missing atoms, RuntimeError for duplicate bonds

## [0.2.1] - 2025-01-16

### Fixed

- Updated README with optional CLI installation instructions
- Updated CHANGELOG for v0.2.0 changes

## [0.2.0] - 2025-01-16

### Changed

- CLI is now an optional dependency (`pip install ccd2rdmol[cli]`)
- Core library (`ccd2rdmol`) no longer requires `rich` and `typer`

## [0.1.0] - 2025-01-15

### Added

- Initial release
- CIF file parsing using gemmi
- Conversion to RDKit molecule objects
- Support for Ideal and Model 3D conformers
- Automatic metal bond to dative bond conversion
- Stereochemistry assignment from 3D coordinates
- CLI tool with `convert` and `info` commands
- Rich terminal output support
- Options for sanitization, conformer handling, and hydrogen removal

[Unreleased]: https://github.com/N283T/ccd2rdmol/compare/v0.2.2...HEAD
[0.2.2]: https://github.com/N283T/ccd2rdmol/compare/v0.2.1...v0.2.2
[0.2.1]: https://github.com/N283T/ccd2rdmol/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/N283T/ccd2rdmol/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/N283T/ccd2rdmol/releases/tag/v0.1.0
