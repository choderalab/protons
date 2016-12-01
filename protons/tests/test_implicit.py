from __future__ import print_function

import openmmtools
import pytest
from openmoltools.amber import find_gaff_dat
from openmoltools.schrodinger import is_schrodinger_suite_installed
import openeye
import openmoltools as omt
from openmoltools import forcefield_generators as omtff
from lxml import etree
from simtk import unit, openmm
from simtk.openmm import app
from protons import AmberProtonDrive
from protons.calibration import SelfAdjustedMixtureSampling, AmberCalibrationSystem
from protons.ligands import generate_protons_ffxml, _TitratableForceFieldCompiler

from . import get_test_data
from .helper_func import SystemSetup, hasOpenEye
from collections import OrderedDict

try:
    find_gaff_dat()
    found_gaff = True
except ValueError:
    found_gaff = False

found_schrodinger = is_schrodinger_suite_installed()


class TestTyrosineImplicit(object):
    """Simulating a tyrosine in implicit solvent"""
    default_platform = 'Reference'

    @staticmethod
    def setup_tyrosine_implicit():
        """
        Set up a tyrosine in implicit solvent

        """
        tyr_system = SystemSetup()
        tyr_system.temperature = 300.0 * unit.kelvin
        tyr_system.pressure = 1.0 * unit.atmospheres
        tyr_system.timestep = 1.0 * unit.femtoseconds
        tyr_system.collision_rate = 9.1 / unit.picoseconds
        tyr_system.pH = 9.6
        testsystems = get_test_data('tyr_implicit', 'testsystems')
        tyr_system.positions = openmm.XmlSerializer.deserialize(open('{}/tyr.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        tyr_system.system = openmm.XmlSerializer.deserialize(open('{}/tyr.sys.xml'.format(testsystems)).read())
        tyr_system.prmtop = app.AmberPrmtopFile('{}/tyr.prmtop'.format(testsystems))
        tyr_system.cpin_filename = '{}/tyr.cpin'.format(testsystems)
        return tyr_system

    def test_tyrosine_instantaneous(self):
        """
        Run tyrosine in implicit solvent with an instanteneous state switch.
        """
        testsystem = self.setup_tyrosine_implicit()
        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename, integrator, debug=False,
                                        pressure=None, ncmc_steps_per_trial=0, implicit=True)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_instantaneous_calibrated(self):
        """
        Run tyrosine in implicit solvent with an instanteneous state switch, using the calibration feature.
        """
        testsystem = self.setup_tyrosine_implicit()
        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename,
                                        integrator, debug=False,
                                        pressure=None, ncmc_steps_per_trial=0, implicit=True)

        mc_titration.calibrate(max_iter=2)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) tyrosine in implicit solvent with an instantaneous state switch
        """
        testsystem = self.setup_tyrosine_implicit()
        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename,
                                                   integrator, debug=False,
                                                   pressure=None, ncmc_steps_per_trial=0, implicit=True)
        sams_sampler = SelfAdjustedMixtureSampling(mc_titration)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'binary')

    def test_tyrosine_sams_instantaneous_global(self):
        """
        Run SAMS (global update) tyrosine in implicit solvent with an instantaneous state switch
        """
        testsystem = self.setup_tyrosine_implicit()
        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop,
                                        testsystem.cpin_filename,
                                        integrator, debug=False,
                                        pressure=None, ncmc_steps_per_trial=0, implicit=True)
        sams_sampler = SelfAdjustedMixtureSampling(mc_titration)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'global')

    def test_tyrosine_ncmc(self):
        """
        Run tyrosine in implicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_implicit()

        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename, integrator, debug=False,
                                        pressure=None, ncmc_steps_per_trial=10, implicit=True)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_sams_ncmc_binary(self):
        """
        Run SAMS (binary update) tyrosine in implicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_implicit()
        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename,
                                                   integrator, debug=False,
                                                   pressure=None, ncmc_steps_per_trial=10, implicit=True)
        sams_sampler = SelfAdjustedMixtureSampling(mc_titration)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'binary')

    def test_tyrosine_sams_ncmc_global(self):
        """
        Run SAMS (global update) tyrosine in implicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_implicit()
        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop,
                                        testsystem.cpin_filename,
                                        integrator, debug=False,
                                        pressure=None, ncmc_steps_per_trial=10, implicit=True)
        sams_sampler = SelfAdjustedMixtureSampling(mc_titration)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'global')


class TestAminoAcidsImplicitCalibration(object):
    """Testing of the AmberCalibrationSystem API for implicit solvent systems"""
    @classmethod
    def setup(cls):
        settings = dict()
        settings["temperature"] = 300.0 * unit.kelvin
        settings["timestep"] = 1.0 * unit.femtosecond
        settings["pressure"] = 1013.25 * unit.hectopascal
        settings["collision_rate"] = 9.1 / unit.picoseconds
        settings["pH"] = 7.4
        settings["solvent"] = "implicit"
        settings["nsteps_per_trial"] = 0
        settings["platform_name"] = "Reference"
        cls.settings = settings

    def test_lys_calibration(self):
        """
        Calibrate a single lysine in implicit solvent
        """
        self.calibrate("lys")

    def test_cys_calibration(self):
        """
        Calibrate a single cysteine in implicit solvent
        """
        self.calibrate("cys")

    def test_tyr_calibration(self):
        """
        Calibrate a single tyrosine in implicit solvent
        """

        self.calibrate("tyr")

    def test_as4_calibration(self):
        """
        Calibrate a single aspartic acid in implicit solvent
        """

        self.calibrate("as4")

    def test_gl4_calibration(self):
        """
        Calibrate a single glutamic acid in implicit solvent
        """

        self.calibrate("gl4")

    def test_hip_calibration(self):
        """
        Calibrate a single histidine in implicit solvent
        """
        self.calibrate("hip")

    def calibrate(self, resname):
        print(resname)
        aac = AmberCalibrationSystem(resname, self.settings, minimize=False)
        aac.sams_till_converged(max_iter=10, platform_name=self.settings["platform_name"])


class TestPeptideImplicit(object):
    """Implicit solvent tests for a peptide with the sequence EDYCHK"""
    default_platform = 'Reference'

    @staticmethod
    def setup_edchky_peptide():
        """Sets up a peptide with the sequence EDYCHK"""
        edchky_peptide_system = SystemSetup()
        edchky_peptide_system.temperature = 300.0 * unit.kelvin
        edchky_peptide_system.pressure = 1.0 * unit.atmospheres
        edchky_peptide_system.timestep = 1.0 * unit.femtoseconds
        edchky_peptide_system.collision_rate = 9.1 / unit.picoseconds
        edchky_peptide_system.pH = 7.4
        testsystems = get_test_data('edchky_implicit', 'testsystems')
        edchky_peptide_system.positions = openmm.XmlSerializer.deserialize(
            open('{}/edchky-implicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        edchky_peptide_system.system = openmm.XmlSerializer.deserialize(open('{}/edchky-implicit.sys.xml'.format(testsystems)).read())
        edchky_peptide_system.prmtop = app.AmberPrmtopFile('{}/edchky-implicit.prmtop'.format(testsystems))
        edchky_peptide_system.cpin_filename = '{}/edchky-implicit.cpin'.format(testsystems)
        return edchky_peptide_system

    def test_peptide_instantaneous_calibrated(self):
        """
        Run edchky peptide in implicit solvent with an instanteneous state switch. with calibration
        """
        testsystem = self.setup_edchky_peptide()
        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename,
                                        integrator, debug=False,
                                        pressure=None, ncmc_steps_per_trial=0, implicit=True)

        mc_titration.calibrate(max_iter=10, platform_name=self.default_platform)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation


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


class TestImidazoleImplicit(object):
    """Tests for imidazole in implicit solvent"""

    def test_creating_ligand_system(self):
        """Create an OpenMM system using a pdbfile, and a ligand force field"""
        xmlfile = get_test_data("imidazole.xml", "testsystems/imidazole_implicit")
        forcefield = app.ForceField(xmlfile)
        pdb = app.PDBFile(get_test_data("imidazole.pdb", "testsystems/imidazole_implicit"))
        system = forcefield.createSystem(pdb.topology, implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Test not finished")
    @pytest.mark.skipif(not is_schrodinger_suite_installed() or not found_gaff, reason="This test requires Schrodinger's suite and gaff")
    def test_full_procedure(self):
        """
        Run through an entire parametrization procedure and start a simulation

        """
        xml_output_file = "/tmp/full-proceduretest-implicit.xml"
        generate_protons_ffxml(get_test_data("imidazole.mol2", "testsystems/imidazole_implicit"), xml_output_file, pH=7.0)

        forcefield = app.ForceField(xml_output_file)
        pdb = app.PDBFile(get_test_data("imidazole.pdb", "testsystems/imidazole_implicit"))
        system = forcefield.createSystem(pdb.topology, implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

        raise NotImplementedError("This test is unfinished.")

        # Need to implement the API for reading FFXML and use it here.


