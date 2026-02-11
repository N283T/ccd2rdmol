#!/usr/bin/env -S uv run --script
# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "ccd2rdmol",
#     "gemmi>=0.7.0,<1",
#     "rdkit>=2024.3.1,<2026",
# ]
# ///
"""Compare ccd2rdmol SMILES generation vs CCD reference SMILES.

Parses _pdbx_chem_comp_descriptor from each CCD entry, attempts to
read each reference SMILES with RDKit, and compares success rates.

Usage:
    uv run scripts/compare_ccd_smiles.py /path/to/components.cif.gz
    uv run scripts/compare_ccd_smiles.py /path/to/components.cif.gz -o results.json
"""

from __future__ import annotations

import json
import sys
import time
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import gemmi
from rdkit import Chem, RDLogger

from ccd2rdmol.converter import read_ccd_block

RDLogger.logger().setLevel(RDLogger.ERROR)

PROGRESS_INTERVAL = 5000

# SMILES descriptor types in CCD
SMILES_TYPES = [
    ("SMILES_CANONICAL", "CACTVS"),
    ("SMILES", "CACTVS"),
    ("SMILES_CANONICAL", "OpenEye OEToolkits"),
    ("SMILES", "OpenEye OEToolkits"),
    ("SMILES", "ACDLabs"),
]


@dataclass
class SourceStats:
    """Stats for a single SMILES source."""

    name: str
    total_present: int = 0
    rdkit_parseable: int = 0
    rdkit_failed: int = 0


@dataclass
class EntryComparison:
    """Comparison result for a single entry."""

    comp_id: str
    ccd2rdmol_smiles: str = ""
    ccd2rdmol_ok: bool = False
    ref_smiles: dict[str, str] = field(default_factory=dict)
    ref_parseable: dict[str, bool] = field(default_factory=dict)


def _strip_cif_quotes(value: str) -> str:
    """Strip CIF quoting from a string value."""
    if len(value) >= 2 and value[0] == '"' and value[-1] == '"':
        return value[1:-1]
    if len(value) >= 2 and value[0] == "'" and value[-1] == "'":
        return value[1:-1]
    return value


def _extract_ref_smiles(cif_block: gemmi.cif.Block) -> dict[str, str]:
    """Extract reference SMILES from _pdbx_chem_comp_descriptor."""
    result: dict[str, str] = {}
    try:
        table = cif_block.find(
            [
                "_pdbx_chem_comp_descriptor.type",
                "_pdbx_chem_comp_descriptor.program",
                "_pdbx_chem_comp_descriptor.descriptor",
            ]
        )
    except RuntimeError:
        return result

    for row in table:
        desc_type = _strip_cif_quotes(row[0])
        program = _strip_cif_quotes(row[1])
        descriptor = _strip_cif_quotes(row[2])
        if "SMILES" in desc_type and descriptor and descriptor not in ("?", "."):
            key = f"{desc_type} ({program})"
            result[key] = descriptor

    return result


def run_comparison(cif_path: str) -> tuple[dict[str, SourceStats], list[EntryComparison]]:
    """Run comparison across all CCD entries."""
    print(f"Reading {cif_path} ...")
    t0 = time.time()
    doc = gemmi.cif.read(cif_path)
    print(f"  Loaded {len(doc)} blocks in {time.time() - t0:.1f}s")

    # Initialize stats for each source
    stats: dict[str, SourceStats] = {}
    stats["ccd2rdmol"] = SourceStats(name="ccd2rdmol")

    # Collect known SMILES source names as we encounter them
    interesting: list[EntryComparison] = []

    t_start = time.time()
    for i, block in enumerate(doc):
        comp_id = block.name
        entry = EntryComparison(comp_id=comp_id)

        # ccd2rdmol conversion
        try:
            conv = read_ccd_block(block)
            if conv.mol is not None and conv.mol.GetNumAtoms() > 0:
                smiles = Chem.MolToSmiles(conv.mol)
                if smiles:
                    # Verify round-trip
                    rt = Chem.MolFromSmiles(smiles)
                    if rt is not None:
                        entry.ccd2rdmol_smiles = smiles
                        entry.ccd2rdmol_ok = True
        except Exception:
            pass

        stats["ccd2rdmol"].total_present += 1
        if entry.ccd2rdmol_ok:
            stats["ccd2rdmol"].rdkit_parseable += 1
        else:
            stats["ccd2rdmol"].rdkit_failed += 1

        # Reference SMILES
        ref_smiles = _extract_ref_smiles(block)
        entry.ref_smiles = ref_smiles

        for key, smi in ref_smiles.items():
            if key not in stats:
                stats[key] = SourceStats(name=key)
            stats[key].total_present += 1

            mol = Chem.MolFromSmiles(smi)
            parseable = mol is not None
            entry.ref_parseable[key] = parseable

            if parseable:
                stats[key].rdkit_parseable += 1
            else:
                stats[key].rdkit_failed += 1

        # Track entries where ccd2rdmol succeeds but all refs fail (or vice versa)
        all_ref_fail = ref_smiles and all(not v for v in entry.ref_parseable.values())
        if (entry.ccd2rdmol_ok and all_ref_fail) or (
            not entry.ccd2rdmol_ok and entry.ref_parseable
        ):
            interesting.append(entry)

        if (i + 1) % PROGRESS_INTERVAL == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            print(f"  [{i + 1:>6}/{len(doc)}] ({rate:.0f} entries/s)")

    elapsed = time.time() - t_start
    print(f"  Done in {elapsed:.1f}s")

    return stats, interesting


def print_comparison(stats: dict[str, SourceStats]) -> None:
    """Print comparison table."""
    print("\n" + "=" * 80)
    print("SMILES SOURCE COMPARISON")
    print("=" * 80)
    print(f"{'Source':<45} {'Present':>8} {'Parseable':>10} {'Failed':>8} {'Rate':>8}")
    print("-" * 80)

    # ccd2rdmol first
    s = stats["ccd2rdmol"]
    rate = 100.0 * s.rdkit_parseable / max(s.total_present, 1)
    print(
        f"{'ccd2rdmol (this library)':<45} {s.total_present:>8} {s.rdkit_parseable:>10} {s.rdkit_failed:>8} {rate:>7.1f}%"
    )
    print("-" * 80)

    # Reference sources
    for key, s in sorted(stats.items()):
        if key == "ccd2rdmol":
            continue
        rate = 100.0 * s.rdkit_parseable / max(s.total_present, 1)
        print(
            f"{key:<45} {s.total_present:>8} {s.rdkit_parseable:>10} {s.rdkit_failed:>8} {rate:>7.1f}%"
        )

    print("=" * 80)


def main() -> None:
    """Entry point."""
    import argparse

    parser = argparse.ArgumentParser(description="Compare ccd2rdmol vs CCD reference SMILES")
    parser.add_argument("cif_path", help="Path to components.cif.gz")
    parser.add_argument("-o", "--output", help="Save results to JSON")
    args = parser.parse_args()

    if not Path(args.cif_path).exists():
        print(f"Error: {args.cif_path} not found", file=sys.stderr)
        sys.exit(1)

    stats, interesting = run_comparison(args.cif_path)
    print_comparison(stats)

    # Show interesting cases
    ccd2rdmol_wins = [e for e in interesting if e.ccd2rdmol_ok]
    ccd2rdmol_loses = [e for e in interesting if not e.ccd2rdmol_ok]

    print(f"\nccd2rdmol succeeds but ALL refs fail: {len(ccd2rdmol_wins)}")
    if ccd2rdmol_wins:
        print("  Sample:")
        for e in ccd2rdmol_wins[:10]:
            print(f"    {e.comp_id}")

    print(f"\nccd2rdmol fails but some ref succeeds: {len(ccd2rdmol_loses)}")
    if ccd2rdmol_loses:
        print("  Sample:")
        for e in ccd2rdmol_loses[:10]:
            ref_ok = [k for k, v in e.ref_parseable.items() if v]
            print(f"    {e.comp_id} (ref OK: {', '.join(ref_ok)})")

    if args.output:
        output_data = {
            "stats": {k: asdict(v) for k, v in stats.items()},
            "ccd2rdmol_wins_count": len(ccd2rdmol_wins),
            "ccd2rdmol_loses_count": len(ccd2rdmol_loses),
            "ccd2rdmol_wins_sample": [e.comp_id for e in ccd2rdmol_wins[:50]],
            "ccd2rdmol_loses_sample": [e.comp_id for e in ccd2rdmol_loses[:50]],
        }
        Path(args.output).write_text(json.dumps(output_data, indent=2))
        print(f"\nResults saved to {args.output}")


if __name__ == "__main__":
    main()
