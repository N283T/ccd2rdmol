"""Tests for converter module."""

from pathlib import Path

import pytest
from rdkit import Chem

from ccd2rdmol import read_ccd_file

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data"


@pytest.fixture
def atp_cif() -> Path:
    """Path to ATP CIF file."""
    return TEST_DATA_DIR / "random_sample" / "ATP.cif"


@pytest.fixture
def hem_cif() -> Path:
    """Path to HEM CIF file."""
    return TEST_DATA_DIR / "random_sample" / "HEM.cif"


@pytest.fixture
def gol_cif() -> Path:
    """Path to GOL CIF file."""
    return TEST_DATA_DIR / "random_sample" / "GOL.cif"


class TestReadCcdFile:
    """Tests for read_ccd_file function."""

    def test_read_atp(self, atp_cif: Path) -> None:
        """Test reading ATP CIF file."""
        result = read_ccd_file(str(atp_cif))

        assert result.mol is not None
        assert result.mol.GetNumAtoms() > 0
        assert result.mol.GetNumBonds() > 0

    def test_read_with_conformers(self, atp_cif: Path) -> None:
        """Test that conformers are added."""
        result = read_ccd_file(str(atp_cif), add_conformers=True)

        assert result.mol.GetNumConformers() > 0

    def test_read_without_conformers(self, atp_cif: Path) -> None:
        """Test reading without conformers."""
        result = read_ccd_file(str(atp_cif), add_conformers=False)

        assert result.mol.GetNumConformers() == 0

    def test_read_with_hydrogens(self, gol_cif: Path) -> None:
        """Test reading with hydrogens kept."""
        result_with_h = read_ccd_file(str(gol_cif), remove_hydrogens=False)
        result_without_h = read_ccd_file(str(gol_cif), remove_hydrogens=True)

        assert result_with_h.mol.GetNumAtoms() > result_without_h.mol.GetNumAtoms()

    def test_sanitization_success(self, atp_cif: Path) -> None:
        """Test that sanitization succeeds for normal molecules."""
        result = read_ccd_file(str(atp_cif), sanitize_mol=True)

        assert result.sanitized is True

    def test_smiles_generation(self, gol_cif: Path) -> None:
        """Test that valid SMILES can be generated."""
        result = read_ccd_file(str(gol_cif))
        smiles = Chem.MolToSmiles(result.mol)

        assert smiles is not None
        assert len(smiles) > 0

    def test_metal_complex_hem(self, hem_cif: Path) -> None:
        """Test reading HEM (metal complex)."""
        result = read_ccd_file(str(hem_cif))

        assert result.mol is not None
        # HEM has Fe, check it's present
        has_fe = any(atom.GetAtomicNum() == 26 for atom in result.mol.GetAtoms())
        assert has_fe

    def test_file_not_found(self) -> None:
        """Test error handling for non-existent file."""
        with pytest.raises(FileNotFoundError):
            read_ccd_file("nonexistent.cif")


class TestConformerType:
    """Tests for conformer handling."""

    def test_ideal_conformer_exists(self, atp_cif: Path) -> None:
        """Test that ideal conformer is added."""
        result = read_ccd_file(str(atp_cif), add_conformers=True)
        conformers = result.mol.GetConformers()

        conformer_names = [conf.GetProp("name") for conf in conformers if conf.HasProp("name")]
        assert "IDEAL" in conformer_names or "MODEL" in conformer_names

    def test_conformer_coordinates(self, atp_cif: Path) -> None:
        """Test that conformer has valid coordinates."""
        result = read_ccd_file(str(atp_cif), add_conformers=True)

        if result.mol.GetNumConformers() > 0:
            conf = result.mol.GetConformer(0)
            pos = conf.GetAtomPosition(0)
            # At least one coordinate should be non-zero
            assert pos.x != 0.0 or pos.y != 0.0 or pos.z != 0.0
