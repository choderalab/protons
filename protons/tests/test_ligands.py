from collections import OrderedDict

import pytest
from lxml import etree
from openmoltools import forcefield_generators as omtff
from openmoltools.schrodinger import is_schrodinger_suite_installed
from openmoltools.amber import find_gaff_dat
from simtk import unit
from simtk.openmm import app

try:
    from protons.ligands import generate_protons_ffxml, _TitratableForceFieldCompiler, write_ffxml
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


import os

# Ligands module import will fail if OpenEye not present, but this should be optional for pass/failing the code
pytestmark = pytest.mark.skipif(not ligands_success, reason="These tests can't run if ligands module can't be imported.")


@pytest.mark.skip(reason="Currently not supporting implicit solvent until we can add GB parameters for gaff types.")
class TestLigandParameterizationImplicit(object):
    """Test the epik and antechamber parametrization procedure, and ffxml files that are generated"""
    @pytest.mark.skipif(not is_schrodinger_suite_installed() or not found_gaff or not hasOpenEye,
                        reason="This test requires Schrodinger's suite, OpenEye, and gaff")
    def test_ligand_cphxml(self):
        """
        Run epik on a ligand and parametrize its isomers using antechamber
        """

        generate_protons_ffxml(get_test_data("imidazole.mol2", "testsystems/imidazole_implicit"), "/tmp/protons-imidazole-parameterization-test-implicit.xml",
                               pH=7.0)

    @pytest.mark.skipif(not hasOpenEye, reason="This test requires OpenEye.")
    def test_xml_compilation(self):
        """
        Compile an xml file for the isomers and read it in OpenMM
        """
        from openeye import oechem
        isomers = OrderedDict()
        isomer_index = 0
        store = False

        for line in open(get_test_data("epik.sdf", "testsystems/imidazole_implicit"), 'r'):
            # for line in open('/tmp/tmp3qp7lep7/epik.sdf', 'r'):
            if store:
                epik_penalty = line.strip()

                if store == "log_population":
                    isomers[isomer_index]['epik_penalty'] = epik_penalty
                    epik_penalty = float(epik_penalty)
                    # Epik reports -RT ln p
                    # Divide by -RT in kcal/mol/K at 25 Celsius (Epik default)
                    isomers[isomer_index]['log_population'] = epik_penalty / (-298.15 * 1.9872036e-3)

                # NOTE: relies on state penalty coming before charge
                if store == "net_charge":
                    isomers[isomer_index]['net_charge'] = int(epik_penalty)
                    isomer_index += 1

                store = ""

            elif "r_epik_State_Penalty" in line:
                # Next line contains epik state penalty
                store = "log_population"
                isomers[isomer_index] = dict()

            elif "i_epik_Tot_Q" in line:
                # Next line contains charge
                store = "net_charge"

        ifs = oechem.oemolistream()
        ifs.open(get_test_data("epik.mol2", "testsystems/imidazole_implicit"))

        xmlparser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        for isomer_index, oemolecule in enumerate(ifs.GetOEMols()):
            # generateForceFieldFromMolecules takes a list
            ffxml = omtff.generateForceFieldFromMolecules([oemolecule], normalize=False)
            isomers[isomer_index]['ffxml'] = etree.fromstring(ffxml, parser=xmlparser)

        compiler = _TitratableForceFieldCompiler(isomers)
        output_xml = '/tmp/imidazole-implicit.cph.xml'
        compiler.write(output_xml)
        forcefield = app.ForceField(output_xml)

    def test_reading_validated_xml_file_using_forcefield(self):
        """
        Read the xmlfile using app.ForceField

        Notes
        -----
        Using a pregenerated, manually validated xml file.
        This can detect failure because of changes to OpenMM ForceField.
        """
        xmlfile = get_test_data("imidazole.xml", "testsystems/imidazole_implicit")
        forcefield = app.ForceField(xmlfile)


class TestLigandParameterizationExplicit(object):
    """Test the epik and antechamber parametrization procedure, and ffxml files that are generated"""

    @pytest.mark.skipif(not is_schrodinger_suite_installed() or not found_gaff or not hasOpenEye,
                        reason="This test requires Schrodinger's suite, OpenEye, and gaff")
    def test_ligand_cphxml(self):
        """
        Run epik on a ligand and parametrize its isomers using antechamber
        """

        generate_protons_ffxml(get_test_data("imidazole.mol2", "testsystems/imidazole_explicit"), "/tmp/protons-imidazole-parameterization-test-explicit.xml", remove_temp_files=True, pH=7.0, resname="LIG")

    @pytest.mark.skipif(not hasOpenEye, reason="This test requires OpenEye.")
    def test_xml_compilation(self):
        """
        Compile an xml file for the isomers and read it in OpenMM
        """
        from openeye import oechem
        isomers = OrderedDict()
        isomer_index = 0
        store = False

        for line in open(get_test_data("epik.sdf", "testsystems/imidazole_explicit"), 'r'):
            # for line in open('/tmp/tmp3qp7lep7/epik.sdf', 'r'):
            if store:
                epik_penalty = line.strip()

                if store == "log_population":
                    isomers[isomer_index]['epik_penalty'] = epik_penalty
                    epik_penalty = float(epik_penalty)
                    # Epik reports -RT ln p
                    # Divide by -RT in kcal/mol/K at 25 Celsius (Epik default)
                    isomers[isomer_index]['log_population'] = epik_penalty / (-298.15 * 1.9872036e-3)

                # NOTE: relies on state penalty coming before charge
                if store == "net_charge":
                    isomers[isomer_index]['net_charge'] = int(epik_penalty)
                    isomer_index += 1

                store = ""

            elif "r_epik_State_Penalty" in line:
                # Next line contains epik state penalty
                store = "log_population"
                isomers[isomer_index] = dict()

            elif "i_epik_Tot_Q" in line:
                # Next line contains charge
                store = "net_charge"

        ifs = oechem.oemolistream()
        ifs.open(get_test_data("epik.mol2", "testsystems/imidazole_explicit"))

        xmlparser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        for isomer_index, oemolecule in enumerate(ifs.GetOEMols()):
            # generateForceFieldFromMolecules takes a list
            ffxml = omtff.generateForceFieldFromMolecules([oemolecule], normalize=False)
            isomers[isomer_index]['ffxml'] = etree.fromstring(ffxml, parser=xmlparser)

        compiler = _TitratableForceFieldCompiler(isomers)
        output_xml = '/tmp/imidazole-explicit.cph.xml'

        write_ffxml(compiler, filename=output_xml)
        gaff = get_test_data("gaff.xml", "../forcefields/")
        forcefield = app.ForceField(gaff, output_xml)

    def test_reading_validated_xml_file_using_forcefield(self):
        """
        Read the xmlfile using app.ForceField

        Notes
        -----
        Using a pregenerated, manually validated xml file.
        This can detect failure because of changes to OpenMM ForceField.
        """
        xmlfile = get_test_data("protons-imidazole.xml", "testsystems/imidazole_explicit")
        gaff = get_test_data("gaff.xml", "../forcefields/")
        forcefield = app.ForceField(gaff, xmlfile)

    def test_creating_ligand_system(self):
        """Create an OpenMM system using a pdbfile, and a ligand force field"""
        xmlfile = get_test_data("protons-imidazole.xml", "testsystems/imidazole_explicit")
        gaff = get_test_data("gaff.xml", "../forcefields/")

        forcefield = app.ForceField(gaff, xmlfile, 'amber99sbildn.xml', 'tip3p.xml')
        pdb = app.PDBFile(get_test_data("imidazole-solvated-minimized.pdb", "testsystems/imidazole_explicit"))
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.Ewald,
                                         nonbondedCutoff=1.0 * unit.nanometers, constraints=app.HBonds,
                                         rigidWater=False,
                                         ewaldErrorTolerance=0.0005)


