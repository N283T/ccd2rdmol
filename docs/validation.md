# Full CCD Validation Report

Validation of ccd2rdmol against the complete PDB Chemical Component Dictionary (CCD).

- **CCD version**: `components.cif.gz` (2025-01-24)
- **Total entries**: 49,282
- **ccd2rdmol version**: 0.2.2
- **RDKit version**: 2024.09.6
- **Python**: 3.12

## Summary

ccd2rdmol achieves **99.53%** round-trip SMILES success rate across all 49,282 CCD entries,
outperforming all reference SMILES sources bundled in the CCD itself.

## Conversion Pipeline Results

| Step | Count | Rate |
|------|------:|-----:|
| Total CCD entries | 49,282 | — |
| Converted to RDKit Mol (>0 atoms) | 49,281 | 100.0% |
| Sanitization successful | 49,022 | 99.5% |
| Valid SMILES generated | 49,281 | 100.0% |
| **SMILES round-trip OK** (SMILES → Mol → SMILES) | **49,050** | **99.5%** |

- Processing speed: ~1,300 entries/s (38s total)
- Only 1 entry has zero atoms: `UNL` (Unknown Ligand placeholder)

## Comparison with CCD Reference SMILES

The CCD bundles pre-computed SMILES from multiple software packages. We tested whether
RDKit can parse each reference SMILES via `Chem.MolFromSmiles()`.

| Source | Present | RDKit Parseable | Rate |
|--------|--------:|----------------:|-----:|
| **ccd2rdmol** (this library) | **49,282** | **49,050** | **99.53%** |
| SMILES_CANONICAL (CACTVS) | 49,271 | 48,848 | 99.14% |
| SMILES_CANONICAL (OpenEye OEToolkits) | 49,250 | 48,964 | 99.42% |
| SMILES (ACDLabs) | 33,818 | 31,150 | 92.11% |

### Venn Diagram: ccd2rdmol vs Best-of-CCD-References

Taking the best result from any CCD reference source per entry:

| Category | Count | Rate |
|----------|------:|-----:|
| Both succeed | 49,010 | 99.45% |
| **ccd2rdmol only** (refs all fail) | **40** | 0.08% |
| Ref only (ccd2rdmol fails) | 21 | 0.04% |
| Both fail | 211 | 0.43% |
| **ccd2rdmol total** | **49,050** | **99.53%** |
| Ref total (best-of) | 49,031 | 99.49% |

ccd2rdmol produces valid SMILES for **40 entries** that no CCD reference source can provide
as RDKit-parseable SMILES, while only missing **21 entries** where at least one reference succeeds.

## Failure Analysis

All 232 failures (231 round-trip + 1 zero-atoms) fall into well-defined categories:

| Category | Count | Description |
|----------|------:|-------------|
| Metal complexes | 101 | Fe, V, Hf, etc. — valence exceeds RDKit model |
| Boron clusters | 40 | Polyhedral boranes — complex cage topology |
| Other unsanitized | 90 | N-oxides, hypervalent P/S/B, exotic charges |
| Zero atoms | 1 | `UNL` (empty placeholder) |

All failures share a common root cause: **RDKit sanitization failure** due to non-standard
valence states. The molecules are correctly parsed from CIF and bonds are properly assigned,
but RDKit's valence model rejects them. The generated SMILES is structurally correct but
cannot be round-tripped through `MolFromSmiles()`.

### Common Patterns in Failures

- **Metal clusters**: `[V+4]`, `[Fe]`, `[Hf]` — multi-center bonding
- **Boron cages**: polyhedral boranes with 3D ring systems (e.g., carboranes)
- **N-oxides**: `=N(O)` notation — valence issue on nitrogen
- **Hypervalent atoms**: `FP(F)(F)(F)(F)F` (PF6), `[N+2](=O)=O`, `[B+]`

## Reproducing

```bash
# Full validation
uv run scripts/validate_full_ccd.py /path/to/components.cif.gz -o results.json

# Comparison with CCD references
uv run scripts/compare_ccd_smiles.py /path/to/components.cif.gz -o comparison.json
```

Download `components.cif.gz` from:
https://files.wwpdb.org/pub/pdb/data/monomers/components.cif.gz
