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
from protons.ligands import parametrize_ligand, _TitratableForceFieldCompiler

from . import get_data
from .helper_func import hasCUDA, SystemSetup

try:
    find_gaff_dat()
    found_gaff = True
except ValueError:
    found_gaff = False

found_schrodinger = is_schrodinger_suite_installed()

class TestTyrosineExplicit(object):

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
        testsystems = get_data('tyr_explicit', 'testsystems')
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
        aac = AmberCalibrationSystem(resname, self.settings, minimize=False)
        aac.sams_till_converged(max_iter=10, platform_name=self.settings["platform_name"])


@pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
class TestPeptideExplicit(object):

    default_platform = 'CUDA'

    def setup_peptide_explicit_system(self):
        peptide_explicit_system = SystemSetup()

        peptide_explicit_system.temperature = 300.0 * unit.kelvin
        peptide_explicit_system.pressure = 1.0 * unit.atmospheres
        peptide_explicit_system.timestep = 1.0 * unit.femtoseconds
        peptide_explicit_system.collision_rate = 9.1 / unit.picoseconds
        peptide_explicit_system.pH = 7.4

        testsystems = get_data('edchky_explicit', 'testsystems')
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

    @pytest.mark.skipif(not is_schrodinger_suite_installed() or not found_gaff,
                        reason="This test requires Schrodinger's suite and gaff")
    def test_ligand_cphxml(self):
        """
        Run epik on a ligand and parametrize its isomers using antechamber
        """

        parametrize_ligand(get_data("imidazole.mol2", "testsystems/imidazole_explicit"), "/tmp/ligand-isomers-explicit.xml",
                           pH=7.0, write_gb_params=False)

    def test_xml_compilation(self):
        """
        Compile an xml file for the isomers and read it in OpenMM
        """
        xmlfile = get_data("intermediate.xml", "testsystems/imidazole_explicit")
        m = _TitratableForceFieldCompiler(xmlfile)
        output_xml = '/tmp/isomers-explicit.cph.xml'
        m.write(output_xml)
        forcefield = app.ForceField(output_xml)

    def test_reading_validated_xml_file_using_forcefield(self):
        """
        Read the xmlfile using app.ForceField

        Notes
        -----
        Using a pregenerated, manually validated xml file.
        This can detect failure because of changes to OpenMM ForceField.
        """
        xmlfile = get_data("imidazole.xml", "testsystems/imidazole_explicit")
        forcefield = app.ForceField(xmlfile)


class TestImidazoleExplicit(object):
    """Tests for imidazole in explict solvent (TIP3P)"""

    def test_creating_ligand_system(self):
        """Create an OpenMM system using a pdbfile, and a ligand force field"""
        xmlfile = get_data("imidazole.xml", "testsystems/imidazole_explicit")
        pdbfile = get_data("imidazole_solvated.pdb", "testsystems/imidazole_explicit")
        forcefield = app.ForceField(xmlfile, 'amber99sbildn.xml', 'tip3p.xml')
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
        xml_output_file = "/tmp/full-proceduretest-implicit.xml"
        parametrize_ligand(get_data("imidazole.mol2", "testsystems/imidazole_implicit"), xml_output_file, pH=7.0)

        forcefield = app.ForceField(xml_output_file)
        pdb = app.PDBFile(get_data("imidazole.pdb", "testsystems/imidazole_implicit"))
        system = forcefield.createSystem(pdb.topology, implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds)

        raise NotImplementedError("This test is unfinished.")

        # Need to implement the API for reading FFXML and use it here.

