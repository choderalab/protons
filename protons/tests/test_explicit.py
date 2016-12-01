from __future__ import print_function

import os

import openmmtools
import pytest
from openmoltools.amber import find_gaff_dat
from openmoltools.schrodinger import is_schrodinger_suite_installed
from simtk import unit, openmm
from simtk.openmm import app
from protons import AmberProtonDrive
from protons.calibration import SelfAdjustedMixtureSampling, AmberCalibrationSystem
from protons.ligands import generate_protons_ffxml, _TitratableForceFieldCompiler, write_ffxml
import openeye
import openmoltools as omt
from openmoltools import forcefield_generators as omtff
from lxml import etree
from collections import OrderedDict

from . import get_test_data
from .helper_func import hasCUDA, hasOpenEye, SystemSetup

try:
    find_gaff_dat()
    found_gaff = True
except ValueError:
    found_gaff = False

found_schrodinger = is_schrodinger_suite_installed()


class TestTyrosineExplicit(object):
    """
    Simulating a tyrosine in explicit solvent
    """

    default_platform = 'CPU'

    @staticmethod
    def setup_tyrosine_explicit():
        """
        Set up a tyrosine in explicit solvent
        """
        tyrosine_explicit_system = SystemSetup()
        tyrosine_explicit_system.temperature = 300.0 * unit.kelvin
        tyrosine_explicit_system.pressure = 1.0 * unit.atmospheres
        tyrosine_explicit_system.timestep = 1.0 * unit.femtoseconds
        tyrosine_explicit_system.collision_rate = 9.1 / unit.picoseconds
        tyrosine_explicit_system.pH = 9.6
        testsystems = get_test_data('tyr_explicit', 'testsystems')
        tyrosine_explicit_system.positions = openmm.XmlSerializer.deserialize(open('{}/tyr.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        tyrosine_explicit_system.system = openmm.XmlSerializer.deserialize(open('{}/tyr.sys.xml'.format(testsystems)).read())
        tyrosine_explicit_system.prmtop = app.AmberPrmtopFile('{}/tyr.prmtop'.format(testsystems))
        tyrosine_explicit_system.cpin_filename = '{}/tyr.cpin'.format(testsystems)
        return tyrosine_explicit_system

    def test_tyrosine_instantaneous(self):
        """
        Run tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()
        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename, integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename,
                                                   integrator, debug=False,
                                                   pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False)
        sams_sampler = SelfAdjustedMixtureSampling(mc_titration)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, sams_sampler.driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'binary')

    def test_tyrosine_sams_instantaneous_global(self):
        """
        Run SAMS (global update) tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop,
                                        testsystem.cpin_filename,
                                        integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False)
        sams_sampler = SelfAdjustedMixtureSampling(mc_titration)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, sams_sampler.driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'global')

    def test_tyrosine_ncmc(self):
        """
        Run tyrosine in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename, integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=10, implicit=False)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_sams_ncmc_binary(self):
        """
        Run SAMS (binary update) tyrosine in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename,
                                                   integrator, debug=False,
                                                   pressure=testsystem.pressure, ncmc_steps_per_trial=10, implicit=False)
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
        Run SAMS (global update) tyrosine in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop,
                                        testsystem.cpin_filename,
                                        integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=10, implicit=False)
        sams_sampler = SelfAdjustedMixtureSampling(mc_titration)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'global')


class TestAminoAcidsExplicitCalibration(object):
    """Testing of the AmberCalibrationSystem API for explicit solvent systems"""
    @classmethod
    def setup(cls):
        settings = dict()
        settings["temperature"] = 300.0 * unit.kelvin
        settings["timestep"] = 1.0 * unit.femtosecond
        settings["pressure"] = 1013.25 * unit.hectopascal
        settings["collision_rate"] = 9.1 / unit.picoseconds
        settings["nsteps_per_trial"] = 5
        settings["pH"] = 7.4
        settings["solvent"] = "explicit"
        settings["platform_name"] = "CPU"
        cls.settings = settings

    def test_lys_calibration(self):
        """
        Calibrate a single lysine in explicit solvent
        """
        self.calibrate("lys")

    @pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
    def test_cys_calibration(self):
        """
        Calibrate a single cysteine in explicit solvent
        """
        self.calibrate("cys")

    @pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
    def test_tyr_calibration(self):
        """
        Calibrate a single tyrosine in explicit solvent
        """

        self.calibrate("tyr")

    @pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
    def test_as4_calibration(self):
        """
        Calibrate a single aspartic acid in explicit solvent
        """

        self.calibrate("as4")

    @pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
    def test_gl4_calibration(self):
        """
        Calibrate a single glutamic acid in explicit solvent
        """

        self.calibrate("gl4")

    @pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
    def test_hip_calibration(self):
        """
        Calibrate a single histidine in explicit solvent
        """
        self.calibrate("hip")

    def calibrate(self, resname):
        """
        Sets up a calibration system for a given amino acid and runs it.
        """
        aac = AmberCalibrationSystem(resname, self.settings, minimize=False)
        aac.sams_till_converged(max_iter=10, platform_name=self.settings["platform_name"])


@pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
class TestPeptideExplicit(object):
    """Explicit solvent tests for a peptide with the sequence EDYCHK"""
    default_platform = 'CUDA'

    def setup_peptide_explicit_system(self):
        peptide_explicit_system = SystemSetup()

        peptide_explicit_system.temperature = 300.0 * unit.kelvin
        peptide_explicit_system.pressure = 1.0 * unit.atmospheres
        peptide_explicit_system.timestep = 1.0 * unit.femtoseconds
        peptide_explicit_system.collision_rate = 9.1 / unit.picoseconds
        peptide_explicit_system.pH = 7.4

        testsystems = get_test_data('edchky_explicit', 'testsystems')
        peptide_explicit_system.positions = openmm.XmlSerializer.deserialize(open('{}/edchky-explicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        peptide_explicit_system.system = openmm.XmlSerializer.deserialize(open('{}/edchky-explicit.sys.xml'.format(testsystems)).read())
        peptide_explicit_system.prmtop = app.AmberPrmtopFile('{}/edchky-explicit.prmtop'.format(testsystems))
        peptide_explicit_system.cpin_filename = '{}/edchky-explicit.cpin'.format(testsystems)

        return peptide_explicit_system

    @pytest.mark.skipif(hasCUDA == False, reason="Test depends on CUDA. Make sure the right version is installed.")
    def test_peptide_ncmc_calibrated(self):
        """
        Run edchky peptide in explicit solvent with an ncmc state switch and calibration
        """

        testsystem = self.setup_peptide_explicit_system()
        integrator = openmmtools.integrators.VelocityVerletIntegrator(testsystem.timestep)
        mc_titration = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop, testsystem.cpin_filename,
                                        integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=1, implicit=False)
        mc_titration.calibrate(max_iter=1, platform_name=self.default_platform)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, mc_titration.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation


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


class TestImidazoleExplicit(object):
    """Tests for imidazole in explict solvent (TIP3P)"""

    def test_creating_ligand_system(self):
        """Create an OpenMM system using a pdbfile, and a ligand force field"""
        gaff = get_test_data("gaff.xml", "../forcefields/")
        xmlfile = get_test_data("protons-imidazole.xml", "testsystems/imidazole_explicit")
        pdbfile = get_test_data("imidazole_solvated.pdb", "testsystems/imidazole_explicit")
        forcefield = app.ForceField(gaff, xmlfile, 'amber99sbildn.xml', 'tip3p.xml')
        pdb = app.PDBFile(pdbfile)
        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.Ewald,
                    nonbondedCutoff=1.0*unit.nanometers, constraints=app.HBonds, rigidWater=False,
                    ewaldErrorTolerance=0.0005)

    @pytest.mark.xfail(raises=NotImplementedError, reason="Test not finished")
    @pytest.mark.skipif(not is_schrodinger_suite_installed() or not found_gaff,
                        reason="This test requires Schrodinger's suite and gaff")
    def test_full_procedure(self):
        """
        Run through an entire parametrization procedure and start a simulation

        """
        gaff = get_test_data("gaff.xml", "../forcefields/")
        xml_output_file = "/tmp/full-proceduretest-explicit.xml"
        generate_protons_ffxml(get_test_data("imidazole.mol2", "testsystems/imidazole_explicit"), xml_output_file, pH=7.0)

        forcefield = app.ForceField(gaff, xml_output_file, 'amber99sbildn.xml', 'tip3p.xml')
        pdb = app.PDBFile(get_test_data("imidazole_solvated.pdb", "testsystems/imidazole_explicit"))
        system = forcefield.createSystem(pdb.topology, implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds)

        raise NotImplementedError("This test is unfinished.")

        # Need to implement the API for reading FFXML and use it here.

