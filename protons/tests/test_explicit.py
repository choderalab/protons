from __future__ import print_function

import os
import pytest

from simtk import unit, openmm
from simtk.openmm import app

from protons import AmberProtonDrive, ForceFieldProtonDrive
from protons.calibration import SelfAdjustedMixtureSampling, AmberCalibrationSystem
from . import get_test_data
from .utilities import hasCUDA, SystemSetup, create_compound_gbaoab_integrator, create_compound_ghmc_integrator


class TestAmberTyrosineExplicit(object):
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
        tyrosine_explicit_system.collision_rate = 1.0 / unit.picoseconds
        tyrosine_explicit_system.constraint_tolerance = 1e-7
        tyrosine_explicit_system.pH = 9.6
        testsystems = get_test_data('tyr_explicit', 'testsystems')
        tyrosine_explicit_system.positions = openmm.XmlSerializer.deserialize(open('{}/tyr.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        tyrosine_explicit_system.system = openmm.XmlSerializer.deserialize(open('{}/tyr.sys.xml'.format(testsystems)).read())
        tyrosine_explicit_system.prmtop = app.AmberPrmtopFile('{}/tyr.prmtop'.format(testsystems))
        tyrosine_explicit_system.cpin_filename = '{}/tyr.cpin'.format(testsystems)
        tyrosine_explicit_system.nsteps_per_ghmc = 1
        return tyrosine_explicit_system

    def test_tyrosine_instantaneous(self):
        """
        Run tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_ghmc_integrator(testsystem)

        driver = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop.topology, testsystem.cpin_filename, compound_integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        driver.update(context)  # protonation

    def test_tyrosine_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop.topology, testsystem.cpin_filename,
                                                   compound_integrator, debug=False,
                                                   pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False)
        sams_sampler = SelfAdjustedMixtureSampling(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, sams_sampler.driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'binary')

    def test_tyrosine_sams_instantaneous_global(self):
        """
        Run SAMS (global update) tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop.topology,
                                        testsystem.cpin_filename,
                                        compound_integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False)
        sams_sampler = SelfAdjustedMixtureSampling(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, sams_sampler.driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'global')

    def test_tyrosine_ncmc(self):
        """
        Run tyrosine in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop.topology, testsystem.cpin_filename, compound_integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=10, implicit=False)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        driver.update(context)  # protonation

    def test_tyrosine_ncmc_gbaoab(self):
        """
        Run tyrosine in explicit solvent with an ncmc state switch using a gBAOAB integrator
        """
        testsystem = self.setup_tyrosine_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop.topology, testsystem.cpin_filename, compound_integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=10, implicit=False)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        driver.update(context)

    def test_tyrosine_sams_ncmc_binary(self):
        """
        Run SAMS (binary update) tyrosine in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop.topology, testsystem.cpin_filename,
                                                   compound_integrator, debug=False,
                                                   pressure=testsystem.pressure, ncmc_steps_per_trial=10, implicit=False)
        sams_sampler = SelfAdjustedMixtureSampling(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'binary')

    @pytest.mark.skip(reason="NCMC global scheme is invalid without NCMC sams.")
    def test_tyrosine_sams_ncmc_global(self):
        """
        Run SAMS (global update) tyrosine in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop.topology,
                                        testsystem.cpin_filename,
                                        compound_integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=10, implicit=False)
        sams_sampler = SelfAdjustedMixtureSampling(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'global')


class TestAmberAminoAcidsExplicitCalibration(object):
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
class TestAmberPeptideExplicit(object):
    """Explicit solvent tests for a peptide with the sequence EDYCHK"""
    default_platform = 'CUDA'

    def setup_peptide_explicit_system(self):
        peptide_explicit_system = SystemSetup()

        peptide_explicit_system.temperature = 300.0 * unit.kelvin
        peptide_explicit_system.pressure = 1.0 * unit.atmospheres
        peptide_explicit_system.timestep = 1.0 * unit.femtoseconds
        peptide_explicit_system.collision_rate = 1.0 / unit.picoseconds
        peptide_explicit_system.pH = 7.4
        peptide_explicit_system.constraint_tolerance = 1e-7
        testsystems = get_test_data('edchky_explicit', 'testsystems')
        peptide_explicit_system.positions = openmm.XmlSerializer.deserialize(open('{}/edchky-explicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        peptide_explicit_system.system = openmm.XmlSerializer.deserialize(open('{}/edchky-explicit.sys.xml'.format(testsystems)).read())
        peptide_explicit_system.prmtop = app.AmberPrmtopFile('{}/edchky-explicit.prmtop'.format(testsystems))
        peptide_explicit_system.cpin_filename = '{}/edchky-explicit.cpin'.format(testsystems)
        peptide_explicit_system.nsteps_per_ghmc = 1
        return peptide_explicit_system

    @pytest.mark.skipif(hasCUDA == False, reason="Test depends on CUDA. Make sure the right version is installed.")
    def test_peptide_ncmc_calibrated(self):
        """
        Run edchky peptide in explicit solvent with an ncmc state switch and calibration
        """

        testsystem = self.setup_peptide_explicit_system()
        compound_integrator = compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, testsystem.prmtop.topology, testsystem.cpin_filename,
                                        compound_integrator, debug=False,
                                        pressure=testsystem.pressure, ncmc_steps_per_trial=1, implicit=False)
        driver.calibrate(max_iter=1, platform_name=self.default_platform)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        compound_integrator.step(10)  # MD
        driver.update(context)  # protonation


class TestForceFieldImidazoleExplicit(object):
    """Tests for imidazole in explict solvent (TIP3P)"""

    default_platform = 'CPU'

    @staticmethod
    def setup_imidazole_explicit():
        """
        Set up a tyrosine in explicit solvent
        """
        imidazole_explicit_system = SystemSetup()
        imidazole_explicit_system.temperature = 300.0 * unit.kelvin
        imidazole_explicit_system.pressure = 1.0 * unit.atmospheres
        imidazole_explicit_system.timestep = 1.0 * unit.femtoseconds
        imidazole_explicit_system.collision_rate = 1.0 / unit.picoseconds
        imidazole_explicit_system.pH = 9.6
        testsystems = get_test_data('imidazole_explicit', 'testsystems')
        imidazole_explicit_system.positions = openmm.XmlSerializer.deserialize(
            open('{}/imidazole-explicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        imidazole_explicit_system.system = openmm.XmlSerializer.deserialize(
            open('{}/imidazole-explicit.sys.xml'.format(testsystems)).read())
        imidazole_explicit_system.ffxml_filename = '{}/protons-imidazole.xml'.format(testsystems)
        imidazole_explicit_system.gaff = get_test_data("gaff.xml", "../forcefields/")
        imidazole_explicit_system.pdbfile = app.PDBFile(get_test_data("imidazole-solvated-minimized.pdb", "testsystems/imidazole_explicit"))
        imidazole_explicit_system.topology = imidazole_explicit_system.pdbfile.topology
        imidazole_explicit_system.nsteps_per_ghmc = 1
        return imidazole_explicit_system

    def test_imidazole_instantaneous(self):
        """
        Run imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH, [testsystem.ffxml_filename],
                                  testsystem.topology, compound_integrator, debug=False,
                                  pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False, residues_by_name=['LIG'])
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        driver.update(context)  # protonation

    def test_imidazole_ncmc(self):
        """
        Run imidazole in explicit solvent with an NCMC state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_ghmc_integrator(testsystem)

        driver = ForceFieldProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH,
                                       [testsystem.ffxml_filename],
                                       testsystem.topology, compound_integrator, debug=False,
                                       pressure=testsystem.pressure, ncmc_steps_per_trial=10, implicit=False,
                                       residues_by_name=['LIG'])
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        driver.update(context)  # protonation

    def test_imidazole_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()

        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH,
                                       [testsystem.ffxml_filename],
                                       testsystem.topology, compound_integrator, debug=False,
                                       pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False,
                                       residues_by_name=['LIG'])
        sams_sampler = SelfAdjustedMixtureSampling(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, sams_sampler.driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'binary')

    def test_imidazole_sams_instantaneous_global(self):
        """
        Run SAMS (global update) imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()

        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH,
                                       [testsystem.ffxml_filename],
                                       testsystem.topology, compound_integrator, debug=False,
                                       pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False,
                                       residues_by_name=['LIG'])

        sams_sampler = SelfAdjustedMixtureSampling(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, sams_sampler.driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'global')

    def test_imidazole_sams_ncmc_binary(self):
        """
        Run SAMS (binary update) imidazole in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_ghmc_integrator(testsystem)

        driver = ForceFieldProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH,
                                       [testsystem.ffxml_filename],
                                       testsystem.topology, compound_integrator, debug=False,
                                       pressure=testsystem.pressure, ncmc_steps_per_trial=10, implicit=False,
                                       residues_by_name=['LIG'])

        sams_sampler = SelfAdjustedMixtureSampling(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'binary')
