from collections import OrderedDict

import pytest
from lxml import etree
from uuid import uuid4
from openmoltools import forcefield_generators as omtff
from openmoltools.schrodinger import is_schrodinger_suite_installed
from openmoltools.amber import find_gaff_dat
from protons import app as protons_app
from simtk import unit
from simtk.openmm import app
from os import path, remove

try:
    from protons.app.ligands import generate_protons_ffxml, _TitratableForceFieldCompiler, _write_ffxml
    ligands_success = True
except ImportError:
    ligands_success = False

from protons import ForceFieldProtonDrive
from protons.tests import get_test_data
from protons.tests.utilities import hasOpenEye, SystemSetup

try:
    find_gaff_dat()
    found_gaff = True
except ValueError:
    found_gaff = False

# Ligands module import will fail if OpenEye not present, but this should be optional for pass/failing the code
pytestmark = pytest.mark.skipif(not ligands_success, reason="These tests can't run if ligands module can't be imported.")


class TestTrypsinLigandParameterization:
    """Test the parameterization of the 1D ligand"""
    messy_input = get_test_data("1D_allH.mae", "testsystems/trypsin1d") # typical hand generated input for an epik calculation
    messy_epik_output = get_test_data("1D_epik.mae", "testsystems/trypsin1d") # some messy epik generated output with scrambled and duplicated hydrogen names
    preprocessed_mol2 = get_test_data("1D_preprocessed.mol2", "testsystems/trypsin1d") # A cleaned up mol2 file. Parameterization should work for this one
    ffxml_file = get_test_data("1D.ffxml", "testsystems/trypsin1d") # A cleaned up mol2 file. Parameterization should work for this one

    def test_mapping_states(self):
        """Test the generation of a clean mol2 file from an epik result."""
        unique_filename = "{}.mol2".format(str(uuid4()))
        protons_app.ligands.epik_results_to_mol2(TestTrypsinLigandParameterization.messy_epik_output, unique_filename)
        assert path.isfile(unique_filename), "No output mol2 file was produced"
        remove(unique_filename)  # clean up after ourselves

    @pytest.mark.skipif(not is_schrodinger_suite_installed() or not found_gaff or not hasOpenEye,
                        reason="This test requires Schrodinger's suite, OpenEye, and gaff")
    def test_running_epik(self):
        """Test if Epik is run successfully on the molecule."""
        unique_id = str(uuid4())
        unique_filename = "{}.mae".format(unique_id)
        log_name = "{}.log".format(unique_id)
        protons_app.ligands.generate_epik_states(TestTrypsinLigandParameterization.messy_input,
                                                    unique_filename,
                                                    pH=7.8,
                                                    max_penalty= 10.0,
                                                    workdir= None,
                                                    tautomerize= False)

        assert path.isfile(unique_filename), "No Epik output file was produced"
        remove(unique_filename) # clean up after ourselves
        remove(log_name)

    @pytest.mark.skipif(not is_schrodinger_suite_installed() or not found_gaff or not hasOpenEye,
                        reason="This test requires Schrodinger's suite, OpenEye, and gaff")
    def test_extracting_epik_results(self):
        """Test if Epik results can be extracted from file."""

        result = protons_app.ligands.retrieve_epik_info(TestTrypsinLigandParameterization.messy_epik_output)
        assert len(result) > 0, "No epik data was found."
        print(result)

    @pytest.mark.skipif(not found_gaff or not hasOpenEye, reason="This test requires OpenEye, and gaff")
    def test_parametrize_molecule(self):
        """Generate parameters for a series of protonation states."""
        isomer_dicts = [{'net_charge': 2, 'log_population': -0.20928808598192242},
                        {'net_charge': 1, 'log_population': -1.7617330979671824},
                        {'net_charge': 1, 'log_population': -4.334626309828816},
                        {'net_charge': 0, 'log_population': -6.526412668345948},
                        {'net_charge': 1, 'log_population': -6.862455070918535},
                        {'net_charge': 1, 'log_population': -6.862455070918535},
                        {'net_charge': 0, 'log_population': -8.414900082903793},
                        {'net_charge': 0, 'log_population': -8.414900082903793}]

        unique_filename = "{}.ffxml".format(str(uuid4()))
        protons_app.ligands.generate_protons_ffxml(TestTrypsinLigandParameterization.preprocessed_mol2,
                                                   isomer_dicts, unique_filename, 7.8, resname="1D")
        assert path.isfile(unique_filename), "No Epik output file was produced"
        remove(unique_filename)  # clean up after ourselves



