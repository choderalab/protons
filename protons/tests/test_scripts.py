"""This module tests the scripts included with protons."""
import os
import pytest
from protons.scripts import run_parametrize_ligand, run_prep_ffxml, run_simulation, cli
from .utilities import hasOpenEye, hasCUDA, is_schrodinger_suite_installed as hasEpik
from . import get_test_data


# TODO make temporary input dir, copy input files and clean afterwards for these tests.


class TestParameterizationScript:
    """Test cli scripts for parametrizing molecules."""

    input_dir = get_test_data("param-cli", "testsystems/cli-tests")

    @pytest.mark.skipif(
        not (hasOpenEye and hasEpik()), reason="Needs Schrödinger and OpenEye."
    )
    def test_from_mae_with_epik(self):
        """Test parametrizing a molecule from a maestro file"""
        input_file = os.path.join(
            TestParameterizationScript.input_dir, "ligand-setup-with-epik.json"
        )
        os.chdir(TestParameterizationScript.input_dir)
        run_parametrize_ligand(input_file)

    @pytest.mark.skipif(
        not (hasOpenEye and hasEpik()), reason="Needs Schrödinger and OpenEye."
    )
    def test_from_smiles_with_epik(self):
        assert False

    @pytest.mark.skipif(not hasOpenEye, reason="Needs OpenEye.")
    def test_from_mae_without_epik(self):
        assert False

    @pytest.mark.slowtest
    @pytest.mark.skipif(not hasOpenEye, reason="Needs OpenEye.")
    def test_dense_omega(self):
        """Test dense conformer script."""
        assert False


class TestPreparationScript:
    input_dir = get_test_data("prep-cli", "testsystems/cli-tests")

    def test_prepare_sams(self):
        assert False

    def test_prepare_implicit_solvent(self):
        assert False

    def test_prepare_equil(self):
        assert False

    def test_prepare_ais(self):
        assert False

    def test_prepare_ais_systematic(self):
        assert False


class TestRunScript:
    input_dir = get_test_data("run-cli", "testsystems/cli-tests")

    def test_prepare_sams(self):
        assert False

    def test_prepare_implicit_solvent(self):
        assert False

    def test_prepare_equil(self):
        assert False

    def test_prepare_ais(self):
        assert False


class TestCLI:
    pass
