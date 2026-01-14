"""Command-line interface for ccd2rdmol."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer
from rdkit import Chem
from rich.console import Console
from rich.table import Table

from .converter import read_ccd_file

app = typer.Typer(
    name="ccd2rdmol",
    help="Convert PDB Chemical Component Dictionary (CCD) files to RDKit molecules.",
    add_completion=False,
)
console = Console()


@app.command()
def convert(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input CCD CIF file path.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
    output: Annotated[
        Path | None,
        typer.Option(
            "-o",
            "--output",
            help="Output file path. Supports .mol, .sdf, .mol2 formats.",
        ),
    ] = None,
    format: Annotated[
        str | None,
        typer.Option(
            "-f",
            "--format",
            help="Output format (mol, sdf, smiles, inchi).",
        ),
    ] = None,
    no_sanitize: Annotated[
        bool,
        typer.Option(
            "--no-sanitize",
            help="Skip sanitization step.",
        ),
    ] = False,
    no_conformers: Annotated[
        bool,
        typer.Option(
            "--no-conformers",
            help="Skip adding 3D conformers.",
        ),
    ] = False,
    keep_hydrogens: Annotated[
        bool,
        typer.Option(
            "--keep-hydrogens",
            "-H",
            help="Keep hydrogen atoms in output.",
        ),
    ] = False,
    verbose: Annotated[
        bool,
        typer.Option(
            "--verbose",
            "-v",
            help="Show detailed conversion information.",
        ),
    ] = False,
) -> None:
    """Convert a CCD CIF file to RDKit molecule format."""
    try:
        result = read_ccd_file(
            str(input_file),
            sanitize_mol=not no_sanitize,
            add_conformers=not no_conformers,
            remove_hydrogens=not keep_hydrogens,
        )
    except FileNotFoundError:
        console.print(f"[red]Error:[/red] File not found: {input_file}")
        raise typer.Exit(1) from None
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read file: {e}")
        raise typer.Exit(1) from None

    mol = result.mol

    if verbose:
        _print_info(result, input_file)

    if result.errors:
        for error in result.errors:
            console.print(f"[yellow]Warning:[/yellow] {error}")

    # Determine output format
    output_format = format
    if output_format is None and output is not None:
        suffix = output.suffix.lower()
        format_map = {".mol": "mol", ".sdf": "sdf"}
        output_format = format_map.get(suffix, "mol")
    elif output_format is None:
        output_format = "smiles"

    # Generate output
    output_text = _generate_output(mol, output_format)

    if output_text is None:
        console.print(f"[red]Error:[/red] Failed to generate {output_format} output")
        raise typer.Exit(1)

    if output is not None:
        output.write_text(output_text)
        console.print(f"[green]Written:[/green] {output}")
    else:
        console.print(output_text)


@app.command()
def info(
    input_file: Annotated[
        Path,
        typer.Argument(
            help="Input CCD CIF file path.",
            exists=True,
            dir_okay=False,
            readable=True,
        ),
    ],
) -> None:
    """Show information about a CCD CIF file."""
    try:
        result = read_ccd_file(str(input_file))
    except Exception as e:
        console.print(f"[red]Error:[/red] Failed to read file: {e}")
        raise typer.Exit(1) from None

    _print_info(result, input_file)


def _print_info(result, input_file: Path) -> None:
    """Print detailed information about conversion result."""
    mol = result.mol

    table = Table(title=f"CCD: {input_file.stem}")
    table.add_column("Property", style="cyan")
    table.add_column("Value", style="green")

    table.add_row("File", str(input_file))
    table.add_row("Atoms", str(mol.GetNumAtoms()))
    table.add_row("Bonds", str(mol.GetNumBonds()))
    table.add_row("Conformers", str(mol.GetNumConformers()))
    table.add_row("Sanitized", "Yes" if result.sanitized else "No")

    smiles = Chem.MolToSmiles(mol)
    if smiles:
        table.add_row("SMILES", smiles)

    if result.warnings:
        table.add_row("Warnings", ", ".join(result.warnings))
    if result.errors:
        table.add_row("Errors", ", ".join(result.errors))

    console.print(table)


def _generate_output(mol: Chem.Mol, fmt: str) -> str | None:
    """Generate output in specified format."""
    if fmt == "smiles":
        return Chem.MolToSmiles(mol)
    if fmt == "inchi":
        from rdkit.Chem.inchi import MolToInchi

        result = MolToInchi(mol)
        if isinstance(result, tuple):
            return result[0]
        return result
    if fmt in ("mol", "sdf"):
        return Chem.MolToMolBlock(mol)
    return None


def main() -> None:
    """Entry point for CLI."""
    app()


if __name__ == "__main__":
    main()
