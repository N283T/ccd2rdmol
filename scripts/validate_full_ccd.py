#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ccd2rdmol",
#     "gemmi>=0.7.0,<1",
#     "rdkit>=2024.3.1,<2026",
# ]
# ///
"""Validate ccd2rdmol against the full PDB CCD (components.cif.gz).

Iterates all entries, converts each to RDKit mol, generates SMILES,
and reports success/failure statistics.

Usage:
    uv run scripts/validate_full_ccd.py /path/to/components.cif.gz
    uv run scripts/validate_full_ccd.py /path/to/components.cif.gz -o results.json
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gemmi
from rdkit import Chem, RDLogger

from ccd2rdmol.converter import read_ccd_block

# Suppress RDKit warnings during bulk processing
RDLogger.logger().setLevel(RDLogger.ERROR)

PROGRESS_INTERVAL = 5000


@dataclass
class EntryResult:
    """Result for a single CCD entry."""

    comp_id: str
    success: bool
    smiles: str = ""
    sanitized: bool = False
    num_atoms: int = 0
    num_heavy_atoms: int = 0
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    failure_reason: str = ""


@dataclass
class ValidationSummary:
    """Summary statistics for the full validation run."""

    total: int = 0
    converted: int = 0
    valid_smiles: int = 0
    sanitized: int = 0
    round_trip_ok: int = 0
    failed_conversion: int = 0
    failed_smiles: int = 0
    failed_round_trip: int = 0
    elapsed_seconds: float = 0.0


def validate_entry(cif_block: gemmi.cif.Block) -> EntryResult:
    """Validate a single CCD entry."""
    comp_id = cif_block.name
    result = EntryResult(comp_id=comp_id, success=False)

    try:
        conv = read_ccd_block(cif_block)
    except Exception as e:
        result.failure_reason = f"conversion_error: {e}"
        return result

    result.errors = list(conv.errors)
    result.warnings = list(conv.warnings)
    result.sanitized = conv.sanitized

    if conv.mol is None:
        result.failure_reason = "mol_is_none"
        return result

    result.num_atoms = conv.mol.GetNumAtoms()
    result.num_heavy_atoms = conv.mol.GetNumHeavyAtoms()

    if result.num_atoms == 0:
        result.failure_reason = "zero_atoms"
        return result

    try:
        smiles = Chem.MolToSmiles(conv.mol)
    except Exception as e:
        result.failure_reason = f"smiles_generation_error: {e}"
        return result

    if not smiles:
        result.failure_reason = "empty_smiles"
        return result

    result.smiles = smiles

    # Round-trip check: SMILES → Mol → SMILES
    rt_mol = Chem.MolFromSmiles(smiles)
    if rt_mol is None:
        result.failure_reason = "round_trip_failed"
        result.success = True  # SMILES was generated, just can't round-trip
        return result

    result.success = True
    return result


def run_validation(cif_path: str) -> tuple[ValidationSummary, list[EntryResult]]:
    """Run validation on all entries in components.cif.gz."""
    print(f"Reading {cif_path} ...")
    t0 = time.time()
    doc = gemmi.cif.read(cif_path)
    t_read = time.time() - t0
    print(f"  Loaded {len(doc)} blocks in {t_read:.1f}s")

    summary = ValidationSummary(total=len(doc))
    failures: list[EntryResult] = []

    t_start = time.time()
    for i, block in enumerate(doc):
        entry = validate_entry(block)

        if entry.smiles:
            summary.valid_smiles += 1
            rt_mol = Chem.MolFromSmiles(entry.smiles)
            if rt_mol is not None:
                summary.round_trip_ok += 1
            else:
                summary.failed_round_trip += 1
        elif entry.num_atoms > 0:
            summary.failed_smiles += 1
        else:
            summary.failed_conversion += 1

        if entry.sanitized:
            summary.sanitized += 1

        if entry.num_atoms > 0:
            summary.converted += 1

        if not entry.success or entry.failure_reason:
            failures.append(entry)

        if (i + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (summary.total - i - 1) / rate
            print(
                f"  [{i + 1:>6}/{summary.total}] "
                f"smiles={summary.valid_smiles} "
                f"fail={summary.failed_conversion + summary.failed_smiles} "
                f"({rate:.0f} entries/s, ETA {eta:.0f}s)"
            )

    summary.elapsed_seconds = time.time() - t_start
    return summary, failures


def print_summary(summary: ValidationSummary) -> None:
    """Print validation summary."""
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"Total entries:        {summary.total:>8}")
    print(f"Converted (>0 atoms): {summary.converted:>8}  ({pct(summary.converted, summary.total)})")
    print(f"Sanitized:            {summary.sanitized:>8}  ({pct(summary.sanitized, summary.total)})")
    print(f"Valid SMILES:         {summary.valid_smiles:>8}  ({pct(summary.valid_smiles, summary.total)})")
    print(
        f"Round-trip OK:        {summary.round_trip_ok:>8}  ({pct(summary.round_trip_ok, summary.total)})"
    )
    print("-" * 60)
    print(f"Failed conversion:    {summary.failed_conversion:>8}")
    print(f"Failed SMILES gen:    {summary.failed_smiles:>8}")
    print(f"Failed round-trip:    {summary.failed_round_trip:>8}")
    print(f"Elapsed:              {summary.elapsed_seconds:>7.1f}s")
    print(
        f"Rate:                 {summary.total / max(summary.elapsed_seconds, 0.001):>7.0f} entries/s"
    )
    print("=" * 60)


def pct(n: int, total: int) -> str:
    """Format percentage."""
    if total == 0:
        return "0.0%"
    return f"{100.0 * n / total:.1f}%"


def categorize_failures(failures: list[EntryResult]) -> dict[str, int]:
    """Group failures by reason."""
    categories: dict[str, int] = {}
    for f in failures:
        reason = f.failure_reason.split(":")[0] if f.failure_reason else "unknown"
        categories[reason] = categories.get(reason, 0) + 1
    return dict(sorted(categories.items(), key=lambda x: -x[1]))


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Validate ccd2rdmol against full CCD")
    parser.add_argument("cif_path", help="Path to components.cif.gz")
    parser.add_argument("-o", "--output", help="Save detailed failure results to JSON")
    args = parser.parse_args()

    cif_path = args.cif_path
    if not Path(cif_path).exists():
        print(f"Error: {cif_path} not found", file=sys.stderr)
        sys.exit(1)

    summary, failures = run_validation(cif_path)
    print_summary(summary)

    # Failure categories
    categories = categorize_failures(failures)
    if categories:
        print("\nFailure categories:")
        for reason, count in categories.items():
            print(f"  {reason:.<40} {count:>6}")

    # Show sample failures (top 10 per category)
    if failures:
        print(f"\nTotal entries with issues: {len(failures)}")
        print("\nSample failures (first 20):")
        for f in failures[:20]:
            line = f"  {f.comp_id}: {f.failure_reason}"
            if f.errors:
                line += f" | errors: {f.errors}"
            print(line)

    # Save to JSON
    if args.output:
        output_data = {
            "summary": asdict(summary),
            "failure_categories": categories,
            "failures": [asdict(f) for f in failures],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\nDetailed results saved to {args.output}")


if __name__ == "__main__":
    main()
