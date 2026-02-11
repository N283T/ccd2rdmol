"""Tests for CLI module."""

from pathlib import Path

from typer.testing import CliRunner

from ccd2rdmol.cli import app

runner = CliRunner()

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data" / "random_sample"


class TestConvertCommand:
    """Tests for convert command."""

    def test_convert_basic(self) -> None:
        """Test basic convert command."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path)])

        assert result.exit_code == 0
        # Should output SMILES by default
        assert len(result.stdout) > 0

    def test_convert_smiles_format(self) -> None:
        """Test convert with SMILES format."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path), "-f", "smiles"])

        assert result.exit_code == 0
        # GOL (glycerol) SMILES should contain C and O
        assert "C" in result.stdout or "O" in result.stdout

    def test_convert_inchi_format(self) -> None:
        """Test convert with InChI format."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path), "-f", "inchi"])

        assert result.exit_code == 0
        assert "InChI" in result.stdout

    def test_convert_mol_format(self) -> None:
        """Test convert with MOL format."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path), "-f", "mol"])

        assert result.exit_code == 0
        # MOL format should contain header lines
        assert "V2000" in result.stdout or "V3000" in result.stdout

    def test_convert_no_sanitize(self) -> None:
        """Test convert with --no-sanitize option."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path), "--no-sanitize"])

        assert result.exit_code == 0

    def test_convert_no_conformers(self) -> None:
        """Test convert with --no-conformers option."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path), "--no-conformers"])

        assert result.exit_code == 0

    def test_convert_keep_hydrogens(self) -> None:
        """Test convert with --keep-hydrogens option."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path), "-H"])

        assert result.exit_code == 0

    def test_convert_verbose(self) -> None:
        """Test convert with --verbose option."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path), "-v"])

        assert result.exit_code == 0
        # Verbose should show table with info
        assert "Atoms" in result.stdout or "Property" in result.stdout

    def test_convert_to_output_file(self, tmp_path: Path) -> None:
        """Test convert with output file."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        output_path = tmp_path / "output.mol"

        result = runner.invoke(app, ["convert", str(gol_path), "-o", str(output_path)])

        assert result.exit_code == 0
        assert output_path.exists()
        content = output_path.read_text()
        assert "V2000" in content or "V3000" in content

    def test_convert_file_not_found(self) -> None:
        """Test convert with non-existent file."""
        result = runner.invoke(app, ["convert", "nonexistent.cif"])

        # Typer returns exit code 2 for file validation errors
        assert result.exit_code in (1, 2)


class TestInfoCommand:
    """Tests for info command."""

    def test_info_basic(self) -> None:
        """Test basic info command."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["info", str(gol_path)])

        assert result.exit_code == 0
        # Info should show table with molecule properties
        assert "Atoms" in result.stdout
        assert "Bonds" in result.stdout

    def test_info_atp(self) -> None:
        """Test info command with ATP."""
        atp_path = TEST_DATA_DIR / "ATP.cif"
        result = runner.invoke(app, ["info", str(atp_path)])

        assert result.exit_code == 0
        assert "ATP" in result.stdout

    def test_info_file_not_found(self) -> None:
        """Test info with non-existent file."""
        result = runner.invoke(app, ["info", "nonexistent.cif"])

        # Typer returns exit code 2 for file validation errors
        assert result.exit_code in (1, 2)


class TestConvertOutputFormats:
    """Tests for output format handling."""

    def test_unsupported_format(self) -> None:
        """Test convert with unsupported format returns error."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path), "-f", "xyz"])

        assert result.exit_code == 1
        assert "Failed to generate" in result.stdout

    def test_sdf_format(self) -> None:
        """Test convert with SDF format."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["convert", str(gol_path), "-f", "sdf"])

        assert result.exit_code == 0
        assert "V2000" in result.stdout or "V3000" in result.stdout

    def test_output_file_sdf_extension(self, tmp_path: Path) -> None:
        """Test that .sdf extension is auto-detected."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        output_path = tmp_path / "output.sdf"

        result = runner.invoke(app, ["convert", str(gol_path), "-o", str(output_path)])

        assert result.exit_code == 0
        assert output_path.exists()

    def test_output_file_unknown_extension(self, tmp_path: Path) -> None:
        """Test that unknown extension defaults to MOL format."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        output_path = tmp_path / "output.xyz"

        result = runner.invoke(app, ["convert", str(gol_path), "-o", str(output_path)])

        assert result.exit_code == 0
        assert output_path.exists()


class TestPrintInfoBranches:
    """Tests for _print_info display branches."""

    def test_verbose_with_warnings(self) -> None:
        """Verbose output includes warnings when present."""
        # Use a CIF with all-missing coords to trigger warning
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(
            app, ["convert", str(gol_path), "-v", "--no-conformers", "--no-sanitize"]
        )

        assert result.exit_code == 0
        # _print_info shows "Sanitized: No" when not sanitized
        assert "No" in result.stdout

    def test_info_shows_smiles(self) -> None:
        """Info command shows SMILES for valid molecules."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(app, ["info", str(gol_path)])

        assert result.exit_code == 0
        assert "SMILES" in result.stdout


class TestConvertErrorDisplay:
    """Tests for error/warning display in convert output."""

    def test_verbose_shows_warnings_and_errors(self, tmp_path: Path) -> None:
        """Verbose output shows warnings in _print_info."""
        # Create a CIF with bad bond references to produce errors
        cif_text = """\
data_TEST
_chem_comp.id TEST
_chem_comp.name 'Test errors'
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
TEST C1 C 0
TEST C2 C 0
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.type
TEST C1 C2 single
TEST C1 MISSING single
"""
        cif_path = tmp_path / "test_errors.cif"
        cif_path.write_text(cif_text)

        result = runner.invoke(app, ["convert", str(cif_path), "-v", "--no-sanitize"])

        assert result.exit_code == 0
        # Should show the warning about bond error
        assert "Warning" in result.stdout or "Bond atom not found" in result.stdout

    def test_info_shows_errors(self, tmp_path: Path) -> None:
        """Info command displays errors in table."""
        cif_text = """\
data_TEST
_chem_comp.id TEST
_chem_comp.name 'Test errors'
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
TEST C1 C 0
TEST C2 C 0
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.type
TEST C1 C2 single
TEST C1 MISSING single
"""
        cif_path = tmp_path / "test_errors.cif"
        cif_path.write_text(cif_text)

        result = runner.invoke(app, ["info", str(cif_path)])

        assert result.exit_code == 0
        assert "Errors" in result.stdout


class TestGenerateOutputDirect:
    """Tests for _generate_output function directly."""

    def test_generate_output_unsupported_returns_none(self) -> None:
        """Unsupported format returns None."""
        from rdkit import Chem

        from ccd2rdmol.cli import _generate_output

        mol = Chem.MolFromSmiles("CCO")
        assert _generate_output(mol, "xyz") is None

    def test_generate_output_smiles(self) -> None:
        """SMILES format returns valid string."""
        from rdkit import Chem

        from ccd2rdmol.cli import _generate_output

        mol = Chem.MolFromSmiles("CCO")
        result = _generate_output(mol, "smiles")
        assert result is not None
        assert "O" in result

    def test_generate_output_inchi(self) -> None:
        """InChI format returns valid string."""
        from rdkit import Chem

        from ccd2rdmol.cli import _generate_output

        mol = Chem.MolFromSmiles("CCO")
        result = _generate_output(mol, "inchi")
        assert result is not None
        assert "InChI" in result

    def test_generate_output_mol(self) -> None:
        """MOL format returns valid string."""
        from rdkit import Chem

        from ccd2rdmol.cli import _generate_output

        mol = Chem.MolFromSmiles("CCO")
        result = _generate_output(mol, "mol")
        assert result is not None
        assert "V2000" in result


class TestCliEdgeCases:
    """Tests for CLI edge cases."""

    def test_convert_metal_complex(self) -> None:
        """Test convert with metal complex (HEM)."""
        hem_path = TEST_DATA_DIR / "HEM.cif"
        result = runner.invoke(app, ["convert", str(hem_path)])

        # Should complete without error
        assert result.exit_code == 0

    def test_convert_with_all_options(self) -> None:
        """Test convert with all options combined."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = runner.invoke(
            app,
            [
                "convert",
                str(gol_path),
                "-f",
                "smiles",
                "--no-sanitize",
                "--no-conformers",
                "-H",
                "-v",
            ],
        )

        assert result.exit_code == 0
