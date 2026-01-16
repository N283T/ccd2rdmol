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
