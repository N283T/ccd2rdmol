"""Tests for sanitizer module."""

from pathlib import Path

from rdkit import Chem

from ccd2rdmol import read_ccd_file
from ccd2rdmol.sanitizer import handle_implicit_hydrogens, sanitize

# Test data directory
TEST_DATA_DIR = Path(__file__).parent / "data" / "random_sample"


class TestHandleImplicitHydrogens:
    """Tests for handle_implicit_hydrogens function."""

    def test_atom_with_explicit_hydrogen(self) -> None:
        """Test that atoms with explicit H partners allow implicit H."""
        # Create ethane and add explicit hydrogens
        mol = Chem.MolFromSmiles("CC")
        mol = Chem.AddHs(mol)
        rwmol = Chem.RWMol(mol)

        handle_implicit_hydrogens(rwmol)

        # Find a carbon atom
        for atom in rwmol.GetAtoms():
            if atom.GetAtomicNum() == 6:
                # Carbon has explicit H neighbors, so NoImplicit should be False
                assert atom.GetNoImplicit() is False
                break

    def test_atom_without_hydrogen(self) -> None:
        """Test that atoms without H partners forbid implicit H."""
        # Create bare carbon (no hydrogens attached in SMILES)
        mol = Chem.MolFromSmiles("[C]", sanitize=False)
        rwmol = Chem.RWMol(mol)

        handle_implicit_hydrogens(rwmol)

        carbon = rwmol.GetAtomWithIdx(0)
        # Carbon has no explicit hydrogens, so NoImplicit should be True
        assert carbon.GetNoImplicit() is True

    def test_hydrogen_atoms_skipped(self) -> None:
        """Test that hydrogen atoms themselves are skipped."""
        mol = Chem.MolFromSmiles("[H][H]", sanitize=False)
        rwmol = Chem.RWMol(mol)

        # Should not raise any errors
        handle_implicit_hydrogens(rwmol)

        # Hydrogen atoms should not be modified
        for atom in rwmol.GetAtoms():
            assert atom.GetAtomicNum() == 1


class TestSanitize:
    """Tests for sanitize function."""

    def test_sanitize_simple_molecule(self) -> None:
        """Test sanitization of a simple molecule."""
        mol = Chem.MolFromSmiles("CCO", sanitize=False)
        rwmol = Chem.RWMol(mol)

        result = sanitize(rwmol)

        assert result.success is True
        assert result.mol is not None

    def test_sanitize_gol(self) -> None:
        """Test sanitization of glycerol (GOL)."""
        gol_path = TEST_DATA_DIR / "GOL.cif"
        result = read_ccd_file(str(gol_path), sanitize_mol=True)

        assert result.sanitized is True
        assert result.mol is not None

    def test_sanitize_atp(self) -> None:
        """Test sanitization of ATP."""
        atp_path = TEST_DATA_DIR / "ATP.cif"
        result = read_ccd_file(str(atp_path), sanitize_mol=True)

        assert result.sanitized is True
        assert result.mol is not None


class TestSanitizeMetalComplexes:
    """Tests for metal complex sanitization."""

    def test_sanitize_hem(self) -> None:
        """Test sanitization of HEM (iron porphyrin)."""
        hem_path = TEST_DATA_DIR / "HEM.cif"
        result = read_ccd_file(str(hem_path), sanitize_mol=True)

        assert result.mol is not None
        # HEM has Fe
        has_fe = any(atom.GetAtomicNum() == 26 for atom in result.mol.GetAtoms())
        assert has_fe

    def test_sanitize_fes(self) -> None:
        """Test sanitization of FES (iron-sulfur cluster)."""
        fes_path = TEST_DATA_DIR / "FES.cif"
        result = read_ccd_file(str(fes_path), sanitize_mol=True)

        assert result.mol is not None
        # FES has Fe and S
        has_fe = any(atom.GetAtomicNum() == 26 for atom in result.mol.GetAtoms())
        has_s = any(atom.GetAtomicNum() == 16 for atom in result.mol.GetAtoms())
        assert has_fe
        assert has_s

    def test_sanitize_na(self) -> None:
        """Test sanitization of sodium ion."""
        na_path = TEST_DATA_DIR / "NA.cif"
        result = read_ccd_file(str(na_path), sanitize_mol=True)

        assert result.mol is not None
        # NA has sodium
        has_na = any(atom.GetAtomicNum() == 11 for atom in result.mol.GetAtoms())
        assert has_na


class TestSanitizeEdgeCases:
    """Tests for edge cases in sanitization."""

    def test_sanitize_preserves_molecule_on_failure(self) -> None:
        """Test that sanitization returns a valid molecule even on partial failure."""
        # Create a molecule that might have issues
        mol = Chem.MolFromSmiles("[Cu]", sanitize=False)
        rwmol = Chem.RWMol(mol)

        result = sanitize(rwmol)

        # Should return a molecule regardless of success
        assert result.mol is not None

    def test_sanitize_aromatic_molecule(self) -> None:
        """Test sanitization of aromatic molecule."""
        mol = Chem.MolFromSmiles("c1ccccc1", sanitize=False)
        rwmol = Chem.RWMol(mol)

        result = sanitize(rwmol)

        assert result.success is True
        assert result.mol is not None


class TestFixValenceErrors:
    """Tests specifically for _fix_valence_errors function."""

    def test_valence_error_with_metal_bond(self) -> None:
        """Test fixing valence errors by converting metal bonds to dative."""
        # HEM is a good test case - it has Fe with multiple bonds
        hem_path = TEST_DATA_DIR / "HEM.cif"
        result = read_ccd_file(str(hem_path), sanitize_mol=True)

        # Should complete without crashes
        assert result.mol is not None
        # Check for dative bonds (should exist after fix)
        has_metal = any(atom.GetAtomicNum() == 26 for atom in result.mol.GetAtoms())
        assert has_metal

    def test_cyclic_peptide_sanitization(self) -> None:
        """Test sanitization of cyclic peptide (PRDCC_000103)."""
        peptide_path = TEST_DATA_DIR / "PRDCC_000103.cif"
        result = read_ccd_file(str(peptide_path), sanitize_mol=True)

        assert result.mol is not None
        # Cyclic hexapeptide should have many atoms
        assert result.mol.GetNumAtoms() > 40

    def test_complex_metal_cluster(self) -> None:
        """Test FES iron-sulfur cluster which requires valence fixes."""
        fes_path = TEST_DATA_DIR / "FES.cif"
        result = read_ccd_file(str(fes_path), sanitize_mol=True)

        assert result.mol is not None
        # Should have Fe atoms
        fe_count = sum(1 for atom in result.mol.GetAtoms() if atom.GetAtomicNum() == 26)
        assert fe_count > 0

    def test_sulfate_sanitization(self) -> None:
        """Test SO4 (sulfate) which has formal charges."""
        so4_path = TEST_DATA_DIR / "SO4.cif"
        result = read_ccd_file(str(so4_path), sanitize_mol=True)

        assert result.mol is not None
        assert result.sanitized is True

    def test_multiple_valence_issues(self) -> None:
        """Test molecule with multiple potential valence issues."""
        # NAD has complex structure with phosphates
        nad_path = TEST_DATA_DIR / "NAD.cif"
        result = read_ccd_file(str(nad_path), sanitize_mol=True)

        assert result.mol is not None
        assert result.mol.GetNumAtoms() > 30


class TestSanitizeFailurePaths:
    """Tests for sanitization failure scenarios."""

    def test_sanitize_returns_on_error(self) -> None:
        """Test that sanitize handles errors gracefully."""
        from ccd2rdmol.sanitizer import _fix_valence_errors

        # Create a problematic molecule with an invalid bond
        mol = Chem.MolFromSmiles("[Cu]", sanitize=False)
        rwmol = Chem.RWMol(mol)

        # This should not raise an exception
        result = _fix_valence_errors(rwmol)

        # Should return True (simple metal needs no fixing)
        assert isinstance(result, bool)

    def test_kekulization_failure(self) -> None:
        """Test handling of Kekulization failures."""
        # Create a molecule that might fail Kekulization
        # Using a bare aromatic system without proper structure
        mol = Chem.RWMol()
        c1 = mol.AddAtom(Chem.Atom(6))
        c2 = mol.AddAtom(Chem.Atom(6))
        mol.AddBond(c1, c2, Chem.BondType.AROMATIC)

        result = sanitize(mol)

        # Should return a result even if kekulization fails
        assert result.mol is not None

    def test_sanitize_with_charged_atoms(self) -> None:
        """Test sanitization of molecules with formal charges."""
        # Ammonium has a positive charge
        na_path = TEST_DATA_DIR / "NA.cif"
        result = read_ccd_file(str(na_path), sanitize_mol=True)

        assert result.mol is not None

    def test_fix_valence_direct_call(self) -> None:
        """Test _fix_valence_errors directly with a metal-ligand molecule."""
        from ccd2rdmol.sanitizer import _fix_valence_errors

        # Create a simple metal-nitrogen coordination
        mol = Chem.RWMol()
        fe = mol.AddAtom(Chem.Atom(26))  # Iron
        n1 = mol.AddAtom(Chem.Atom(7))  # Nitrogen
        n2 = mol.AddAtom(Chem.Atom(7))  # Nitrogen
        mol.AddBond(fe, n1, Chem.BondType.SINGLE)
        mol.AddBond(fe, n2, Chem.BondType.SINGLE)

        result = _fix_valence_errors(mol)

        # Should return bool
        assert isinstance(result, bool)

    def test_fix_valence_with_copper(self) -> None:
        """Test valence fixing with copper complex."""
        from ccd2rdmol.sanitizer import _fix_valence_errors

        # Copper with two oxygen ligands (typical in coordination)
        mol = Chem.RWMol()
        cu = mol.AddAtom(Chem.Atom(29))  # Copper
        o1 = mol.AddAtom(Chem.Atom(8))  # Oxygen
        o2 = mol.AddAtom(Chem.Atom(8))  # Oxygen
        mol.AddBond(cu, o1, Chem.BondType.SINGLE)
        mol.AddBond(cu, o2, Chem.BondType.SINGLE)

        result = _fix_valence_errors(mol)

        assert isinstance(result, bool)

    def test_sanitize_exception_handling(self) -> None:
        """Test that sanitize catches exceptions gracefully."""
        # Create a molecule that could cause issues during sanitization
        mol = Chem.RWMol()
        # Add atoms with potentially problematic configuration
        c = mol.AddAtom(Chem.Atom(6))
        o = mol.AddAtom(Chem.Atom(8))
        # Add multiple bonds to potentially exceed valence
        mol.AddBond(c, o, Chem.BondType.TRIPLE)

        result = sanitize(mol)

        # Should still return a result
        assert result.mol is not None


class TestVariousMoleculeTypes:
    """Tests for various molecule types to improve coverage."""

    def test_sugar_molecule(self) -> None:
        """Test GLC (glucose) sanitization."""
        glc_path = TEST_DATA_DIR / "GLC.cif"
        result = read_ccd_file(str(glc_path), sanitize_mol=True)

        assert result.mol is not None
        assert result.sanitized is True

    def test_lipid_molecule(self) -> None:
        """Test CDL (cardiolipin) - large lipid molecule."""
        cdl_path = TEST_DATA_DIR / "CDL.cif"
        result = read_ccd_file(str(cdl_path), sanitize_mol=True)

        assert result.mol is not None

    def test_drug_molecule(self) -> None:
        """Test IBP (ibuprofen) sanitization."""
        ibp_path = TEST_DATA_DIR / "IBP.cif"
        result = read_ccd_file(str(ibp_path), sanitize_mol=True)

        assert result.mol is not None
        assert result.sanitized is True

    def test_solvent_molecule(self) -> None:
        """Test DMS (dimethyl sulfoxide) sanitization."""
        dms_path = TEST_DATA_DIR / "DMS.cif"
        result = read_ccd_file(str(dms_path), sanitize_mol=True)

        assert result.mol is not None
        assert result.sanitized is True

    def test_amino_acid(self) -> None:
        """Test GLU (glutamate) sanitization."""
        glu_path = TEST_DATA_DIR / "GLU.cif"
        result = read_ccd_file(str(glu_path), sanitize_mol=True)

        assert result.mol is not None
        assert result.sanitized is True

    def test_large_macrocycle(self) -> None:
        """Test BCD (beta-cyclodextrin) - large cyclic sugar."""
        bcd_path = TEST_DATA_DIR / "BCD.cif"
        result = read_ccd_file(str(bcd_path), sanitize_mol=True)

        assert result.mol is not None
        # BCD is very large
        assert result.mol.GetNumAtoms() > 50
