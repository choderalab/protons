"""This module tests the scripts included with protons."""
import os
import pytest
from protons.scripts import run_parametrize_ligand, run_prep_ffxml, run_simulation, cli
from .utilities import (
    hasOpenEye,
    hasCUDA,
    is_schrodinger_suite_installed as hasEpik,
    files_to_tempdir,
)
from . import get_test_data
from shutil import copy, rmtree


@pytest.mark.slowtest
class TestParameterizationScript:
    """Test cli scripts for parametrizing molecules."""

    top_input_dir = get_test_data("param-cli", "cli-tests")

    # Example template for ligand file names to copy over
    trypsin_example = {
        "structure": "1D-source.pdb",
        "epik": "1D-epik-out.mae",
        "mae": "1D-input.mae",
    }

    @pytest.mark.skipif(
        not (hasOpenEye and hasEpik()), reason="Needs Schrödinger and OpenEye."
    )
    def test_from_mae_with_epik(self):
        """Test parametrizing a molecule from a maestro file"""
        tmpdir = files_to_tempdir(
            [
                os.path.join(TestParameterizationScript.top_input_dir, filename)
                for filename in TestParameterizationScript.trypsin_example.values()
            ]
        )
        json_input = "ligand-setup-with-epik.json"
        copy(os.path.join(TestParameterizationScript.top_input_dir, json_input), tmpdir)
        olddir = os.getcwd()
        os.chdir(tmpdir)

        # Perform parametrization in the temp directory
        run_parametrize_ligand.run_parametrize_main(json_input)

        # check if output files were produced
        assert os.path.isfile(
            "output/1D-7.8-10kT-epik.mae"
        ), "No epik output was produced"
        assert os.path.isfile("output/1D.xml"), "No forcefield file was produced"
        assert os.path.isfile(
            "output/1D-h.xml"
        ), "No hydrogen definitions file was produced"
        assert os.path.isfile(
            "output/1D-vacuum.cif"
        ), "No vacuum system file was produced"
        assert os.path.isfile(
            "output/1D-water.cif"
        ), "No water system file was produced"
        os.chdir(olddir)
        rmtree(tmpdir)  # clean files

    @pytest.mark.skipif(
        not (hasOpenEye and hasEpik()), reason="Needs Schrödinger and OpenEye."
    )
    def test_from_smiles_with_epik(self):
        """Test parametrizing a molecule specified by smiles."""

        # Has no input files beside the json
        tmpdir = files_to_tempdir([])
        json_input = "ligand-setup-from-smiles.json"
        copy(os.path.join(TestParameterizationScript.top_input_dir, json_input), tmpdir)
        olddir = os.getcwd()
        os.chdir(tmpdir)

        # Perform parametrization in the temp directory
        run_parametrize_ligand.run_parametrize_main(json_input)

        # check if output files were produced
        assert os.path.isfile(
            "output/crizotinib-epik.mae"
        ), "No epik output was produced"
        assert os.path.isfile(
            "output/crizotinib.xml"
        ), "No forcefield file was produced"
        assert os.path.isfile(
            "output/crizotinib-h.xml"
        ), "No hydrogen definitions file was produced"
        os.chdir(olddir)
        rmtree(tmpdir)  # clean files

    @pytest.mark.skipif(not hasOpenEye, reason="Needs OpenEye.")
    def test_from_mae_without_epik(self):
        tmpdir = files_to_tempdir(
            [
                os.path.join(TestParameterizationScript.top_input_dir, filename)
                for filename in TestParameterizationScript.trypsin_example.values()
            ]
        )
        json_input = "ligand-setup-without-epik.json"
        copy(os.path.join(TestParameterizationScript.top_input_dir, json_input), tmpdir)
        olddir = os.getcwd()
        os.chdir(tmpdir)

        # Perform parametrization in the temp directory
        run_parametrize_ligand.run_parametrize_main(json_input)

        # check if output files were produced
        assert os.path.isfile("output/1D.xml"), "No forcefield file was produced"
        assert os.path.isfile(
            "output/1D-h.xml"
        ), "No hydrogen definitions file was produced"
        assert os.path.isfile(
            "output/1D-vacuum.cif"
        ), "No vacuum system file was produced"
        assert os.path.isfile(
            "output/1D-water.cif"
        ), "No water system file was produced"
        os.chdir(olddir)
        rmtree(tmpdir)  # clean files

    @pytest.mark.skipif(
        not (hasOpenEye and hasEpik()), reason="Needs Schrödinger and OpenEye."
    )
    def test_dense_omega(self):
        """Test dense conformer script."""
        # Has no input files beside the json
        tmpdir = files_to_tempdir([])
        json_input = "ligand-setup-from-smiles-dense.json"
        copy(os.path.join(TestParameterizationScript.top_input_dir, json_input), tmpdir)
        olddir = os.getcwd()
        os.chdir(tmpdir)

        # Perform parametrization in the temp directory
        run_parametrize_ligand.run_parametrize_main(json_input)

        # check if output files were produced
        assert os.path.isfile(
            "output/crizotinib-epik.mae"
        ), "No epik output was produced"
        assert os.path.isfile(
            "output/crizotinib.xml"
        ), "No forcefield file was produced"
        assert os.path.isfile(
            "output/crizotinib-h.xml"
        ), "No hydrogen definitions file was produced"
        os.chdir(olddir)
        rmtree(tmpdir)


@pytest.mark.slowtest
class TestPreparationScript:
    top_input_dir = get_test_data("prep-cli", "cli-tests")

    # Example template for ligand file names to copy over
    trypsin_example = {
        "water": "1D-water.cif",
        "vacuum": "1D-vacuum.cif",
        "xml": "1D.xml",
    }

    def test_prepare_sams(self):
        """Prepare a SAMS/calibration run."""
        tmpdir = files_to_tempdir(
            [
                os.path.join(TestPreparationScript.top_input_dir, filename)
                for filename in TestPreparationScript.trypsin_example.values()
            ]
        )
        json_input = "prepare_calibration.json"
        copy(os.path.join(TestPreparationScript.top_input_dir, json_input), tmpdir)
        olddir = os.getcwd()
        os.chdir(tmpdir)

        # Perform preparation in temp dir
        run_prep_ffxml.run_prep_ffxml_main(json_input)

        # check if output files were produced
        assert os.path.isfile(
            "output/1D-checkpoint-0.xml"
        ), f"No checkpoint file was produced in {tmpdir}"
        os.chdir(olddir)
        rmtree(tmpdir)

    def test_prepare_implicit_solvent(self):
        """Prepare a SAMS/calibration run in implicit solvent."""
        json_input = "prepare_calibration_implicit.yaml"
        example = TestPreparationScript.trypsin_example
        tmpdir = files_to_tempdir(
            [
                os.path.join(TestPreparationScript.top_input_dir, filename)
                for filename in example.values()
            ]
        )

        copy(os.path.join(TestPreparationScript.top_input_dir, json_input), tmpdir)
        olddir = os.getcwd()
        os.chdir(tmpdir)

        # Perform preparation in temp dir
        run_prep_ffxml.run_prep_ffxml_main(json_input)

        # check if output files were produced
        assert os.path.isfile(
            "output/1D-checkpoint-0.xml"
        ), f"No checkpoint file was produced in {tmpdir}"
        os.chdir(olddir)
        rmtree(tmpdir)

    def test_prepare_equil(self):
        tmpdir = files_to_tempdir(
            [
                os.path.join(TestPreparationScript.top_input_dir, filename)
                for filename in TestPreparationScript.trypsin_example.values()
            ]
        )
        json_input = "prepare_simulation.json"
        copy(os.path.join(TestPreparationScript.top_input_dir, json_input), tmpdir)
        olddir = os.getcwd()
        os.chdir(tmpdir)

        # Perform preparation in temp dir
        run_prep_ffxml.run_prep_ffxml_main(json_input)

        # check if output files were produced
        assert os.path.isfile(
            "output/1D-checkpoint-0.xml"
        ), f"No checkpoint file was produced in {tmpdir}"
        os.chdir(olddir)
        rmtree(tmpdir)

    def test_prepare_ais(self):
        json_input = "prepare_importance_sampling.json"
        example = TestPreparationScript.trypsin_example
        tmpdir = files_to_tempdir(
            [
                os.path.join(TestPreparationScript.top_input_dir, filename)
                for filename in example.values()
            ]
        )

        copy(os.path.join(TestPreparationScript.top_input_dir, json_input), tmpdir)
        olddir = os.getcwd()
        os.chdir(tmpdir)

        # Perform preparation in temp dir
        run_prep_ffxml.run_prep_ffxml_main(json_input)

        # check if output files were produced
        assert os.path.isfile(
            "output/1D-checkpoint-0.xml"
        ), f"No checkpoint file was produced in {tmpdir}"
        os.chdir(olddir)
        rmtree(tmpdir)

    @pytest.mark.slowtest
    def test_prepare_ais_systematic(self):
        json_input = "prepare_systematic_importance_sampling.json"
        example = TestPreparationScript.trypsin_example
        tmpdir = files_to_tempdir(
            [
                os.path.join(TestPreparationScript.top_input_dir, filename)
                for filename in example.values()
            ]
        )

        copy(os.path.join(TestPreparationScript.top_input_dir, json_input), tmpdir)
        olddir = os.getcwd()
        os.chdir(tmpdir)

        # Perform preparation in temp dir
        run_prep_ffxml.run_prep_ffxml_main(json_input)

        # check if output files were produced
        for index in range(8):
            assert os.path.isfile(
                f"output/1D-importance-state-{index}-checkpoint-0.xml"
            ), f"No checkpoint file was produced in {tmpdir}"
        os.chdir(olddir)
        rmtree(tmpdir)


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
