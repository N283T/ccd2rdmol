"""Tests for converter module."""

from pathlib import Path

import gemmi
import pytest
from rdkit import Chem

from ccd2rdmol import chemcomp_to_mol, read_ccd_file
from ccd2rdmol.converter import _is_degenerate_conformer, _str_to_float

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


class TestReadCcdBlock:
    """Tests for read_ccd_block function."""

    def test_read_block_from_file(self, atp_cif: Path) -> None:
        """Test reading CIF block directly."""
        import gemmi

        from ccd2rdmol import read_ccd_block

        doc = gemmi.cif.read(str(atp_cif))
        block = doc.sole_block()

        result = read_ccd_block(block)

        assert result.mol is not None
        assert result.mol.GetNumAtoms() > 0

    def test_read_block_with_options(self, gol_cif: Path) -> None:
        """Test read_ccd_block with various options."""
        import gemmi

        from ccd2rdmol import read_ccd_block

        doc = gemmi.cif.read(str(gol_cif))
        block = doc.sole_block()

        # Test with all options
        result = read_ccd_block(
            block,
            sanitize_mol=True,
            add_conformers=True,
            remove_hydrogens=False,
        )

        assert result.mol is not None
        assert result.sanitized is True
        assert result.mol.GetNumConformers() > 0

    def test_read_block_no_sanitize(self, atp_cif: Path) -> None:
        """Test read_ccd_block without sanitization."""
        import gemmi

        from ccd2rdmol import read_ccd_block

        doc = gemmi.cif.read(str(atp_cif))
        block = doc.sole_block()

        result = read_ccd_block(block, sanitize_mol=False)

        assert result.mol is not None
        # Sanitized should be False when sanitize_mol=False
        assert result.sanitized is False


class TestChemcompToMol:
    """Tests for chemcomp_to_mol function."""

    def test_chemcomp_basic(self, gol_cif: Path) -> None:
        """Test chemcomp_to_mol basic usage."""
        import gemmi

        from ccd2rdmol import chemcomp_to_mol

        doc = gemmi.cif.read(str(gol_cif))
        block = doc.sole_block()
        cc = gemmi.make_chemcomp_from_block(block)

        result = chemcomp_to_mol(cc, block)

        assert result.mol is not None
        assert result.mol.GetNumAtoms() > 0

    def test_chemcomp_with_conformers(self, atp_cif: Path) -> None:
        """Test chemcomp_to_mol with conformers."""
        import gemmi

        from ccd2rdmol import chemcomp_to_mol

        doc = gemmi.cif.read(str(atp_cif))
        block = doc.sole_block()
        cc = gemmi.make_chemcomp_from_block(block)

        result = chemcomp_to_mol(cc, block, add_conformers=True)

        assert result.mol.GetNumConformers() > 0

    def test_chemcomp_without_conformers(self, atp_cif: Path) -> None:
        """Test chemcomp_to_mol without conformers."""
        import gemmi

        from ccd2rdmol import chemcomp_to_mol

        doc = gemmi.cif.read(str(atp_cif))
        block = doc.sole_block()
        cc = gemmi.make_chemcomp_from_block(block)

        result = chemcomp_to_mol(cc, block, add_conformers=False)

        assert result.mol.GetNumConformers() == 0


class TestStrToFloat:
    """Tests for _str_to_float helper."""

    def test_valid_float(self) -> None:
        assert _str_to_float("1.23") == 1.23

    def test_negative_float(self) -> None:
        assert _str_to_float("-4.56") == -4.56

    def test_zero(self) -> None:
        assert _str_to_float("0.0") == 0.0

    def test_missing_question_mark(self) -> None:
        assert _str_to_float("?") is None

    def test_missing_dot(self) -> None:
        assert _str_to_float(".") is None

    def test_empty_string(self) -> None:
        assert _str_to_float("") is None

    def test_invalid_text(self) -> None:
        assert _str_to_float("abc") is None


class TestDegenerateConformer:
    """Tests for _is_degenerate_conformer."""

    def test_valid_conformer(self) -> None:
        """Non-degenerate conformer with distinct positions."""
        conf = Chem.Conformer(3)
        conf.SetAtomPosition(0, Chem.rdGeometry.Point3D(1.0, 0.0, 0.0))
        conf.SetAtomPosition(1, Chem.rdGeometry.Point3D(0.0, 1.0, 0.0))
        conf.SetAtomPosition(2, Chem.rdGeometry.Point3D(0.0, 0.0, 1.0))

        assert _is_degenerate_conformer(conf) is False

    def test_single_atom_at_origin(self) -> None:
        """One atom at origin is acceptable."""
        conf = Chem.Conformer(3)
        conf.SetAtomPosition(0, Chem.rdGeometry.Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, Chem.rdGeometry.Point3D(1.0, 2.0, 3.0))
        conf.SetAtomPosition(2, Chem.rdGeometry.Point3D(4.0, 5.0, 6.0))

        assert _is_degenerate_conformer(conf) is False

    def test_multiple_atoms_at_origin(self) -> None:
        """More than one atom at origin = degenerate."""
        conf = Chem.Conformer(3)
        conf.SetAtomPosition(0, Chem.rdGeometry.Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, Chem.rdGeometry.Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(2, Chem.rdGeometry.Point3D(1.0, 2.0, 3.0))

        assert _is_degenerate_conformer(conf) is True

    def test_all_atoms_at_origin(self) -> None:
        """All atoms at origin = degenerate."""
        conf = Chem.Conformer(3)
        conf.SetAtomPosition(0, Chem.rdGeometry.Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(1, Chem.rdGeometry.Point3D(0.0, 0.0, 0.0))
        conf.SetAtomPosition(2, Chem.rdGeometry.Point3D(0.0, 0.0, 0.0))

        assert _is_degenerate_conformer(conf) is True


class TestDeuteriumIsotope:
    """Test Deuterium (D) isotope handling."""

    def test_deuterium_isotope_set(self) -> None:
        """Verify Deuterium atoms get isotope=2."""
        cif_text = """
data_TEST
_chem_comp.id TEST
_chem_comp.name 'Test deuterium'
loop_
_chem_comp_atom.comp_id
_chem_comp_atom.atom_id
_chem_comp_atom.type_symbol
_chem_comp_atom.charge
TEST C1 C 0
TEST D1 D 0
loop_
_chem_comp_bond.comp_id
_chem_comp_bond.atom_id_1
_chem_comp_bond.atom_id_2
_chem_comp_bond.type
TEST C1 D1 single
"""
        doc = gemmi.cif.read_string(cif_text)
        block = doc.sole_block()
        cc = gemmi.make_chemcomp_from_block(block)

        result = chemcomp_to_mol(
            cc, block, sanitize_mol=False, add_conformers=False, remove_hydrogens=False
        )

        deuterium_found = False
        for atom in result.mol.GetAtoms():
            if atom.GetAtomicNum() == 1 and atom.GetIsotope() == 2:
                deuterium_found = True
                break

        assert deuterium_found


class TestSanitizeInputImmutability:
    """Test that sanitize() does not modify its input."""

    def test_sanitize_does_not_mutate_input(self) -> None:
        """Original molecule should remain unchanged after sanitize()."""
        from ccd2rdmol.sanitizer import sanitize

        mol = Chem.MolFromSmiles("CCO", sanitize=False)
        rwmol = Chem.RWMol(mol)
        original_smiles = Chem.MolToSmiles(rwmol.GetMol())
        original_num_atoms = rwmol.GetNumAtoms()

        sanitize(rwmol)

        assert rwmol.GetNumAtoms() == original_num_atoms
        assert Chem.MolToSmiles(rwmol.GetMol()) == original_smiles

    def test_sanitize_does_not_mutate_metal_complex(self) -> None:
        """Metal complex input should remain unchanged after sanitize()."""
        from ccd2rdmol.sanitizer import sanitize

        hem_path = TEST_DATA_DIR / "random_sample" / "HEM.cif"
        result = read_ccd_file(str(hem_path), sanitize_mol=False)
        rwmol = Chem.RWMol(result.mol)
        original_num_bonds = rwmol.GetNumBonds()

        sanitize(rwmol)

        assert rwmol.GetNumBonds() == original_num_bonds
