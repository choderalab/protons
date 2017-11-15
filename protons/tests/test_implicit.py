from __future__ import print_function

import pytest
import os
from openmoltools.amber import find_gaff_dat
from openmoltools.schrodinger import is_schrodinger_suite_installed
from simtk import unit, openmm
from simtk.openmm import app

from protons.app.driver import AmberProtonDrive
from protons.app.calibration import SelfAdjustedMixtureSampling
from protons.app.proposals import UniformProposal
from . import get_test_data
from .utilities import SystemSetup, create_compound_gbaoab_integrator, create_compound_ghmc_integrator

try:
    find_gaff_dat()
    found_gaff = True
except ValueError:
    found_gaff = False

found_schrodinger = is_schrodinger_suite_installed()


@pytest.mark.skip(reason="Implicit solvent support is disabled.")
class TestAmberTyrosineImplicit(object):
    """Simulating a tyrosine in implicit solvent"""
    default_platform = 'CPU'

    @staticmethod
    def setup_tyrosine_implicit():
        """
        Set up a tyrosine in implicit solvent

        """
        tyr_system = SystemSetup()
        tyr_system.temperature = 300.0 * unit.kelvin
        tyr_system.pressure = 1.0 * unit.atmospheres
        tyr_system.timestep = 1.0 * unit.femtoseconds
        tyr_system.collision_rate = 1.0 / unit.picoseconds
        tyr_system.pH = 9.6
        testsystems = get_test_data('tyr_implicit', 'testsystems')
        tyr_system.positions = openmm.XmlSerializer.deserialize(open('{}/tyr.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        tyr_system.system = openmm.XmlSerializer.deserialize(open('{}/tyr.sys.xml'.format(testsystems)).read())
        tyr_system.prmtop = app.AmberPrmtopFile('{}/tyr.prmtop'.format(testsystems))
        tyr_system.topology = tyr_system.prmtop.topology
        tyr_system.cpin_filename = '{}/tyr.cpin'.format(testsystems)
        tyr_system.nsteps_per_ghmc = 1
        tyr_system.constraint_tolerance = 1.e-7
        return tyr_system

    def test_tyrosine_instantaneous(self):
        """
        Run tyrosine in implicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_implicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, perturbations_per_trial=0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal())  # protonation

    def test_tyrosine_import_gk(self):
        """
        Import calibrated values for tyrosine
        """
        testsystem = self.setup_tyrosine_implicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, perturbations_per_trial=0)
        driver.import_gk_values(dict(TYR=[0.0,1.0]))

    def test_tyrosine_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) tyrosine in implicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_implicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, perturbations_per_trial=0)
        sams_sampler = SelfAdjustedMixtureSampling(driver, 0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal())  # protonation
        sams_sampler.adapt_zetas('binary')

    def test_tyrosine_sams_instantaneous_global(self):
        """
        Run SAMS (global update) tyrosine in implicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_implicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system,
                                  testsystem.cpin_filename, perturbations_per_trial=0)
        sams_sampler = SelfAdjustedMixtureSampling(driver,0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal())  # protonation
        sams_sampler.adapt_zetas('global')

    def test_tyrosine_ncmc(self):
        """
        Run tyrosine in implicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_implicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system,
                                  testsystem.cpin_filename, perturbations_per_trial=2)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal(), nattempts=10)  # protonation


@pytest.mark.skip(reason="Implicit solvent support is disabled.")
class TestAmberPeptideImplicit(object):
    """Implicit solvent tests for a peptide with the sequence EDYCHK"""
    default_platform = 'Reference'

    @staticmethod
    def setup_edchky_implicit():
        """Sets up a peptide with the sequence EDYCHK"""
        edchky_peptide_system = SystemSetup()
        edchky_peptide_system.temperature = 300.0 * unit.kelvin
        edchky_peptide_system.pressure = None
        edchky_peptide_system.timestep = 1.0 * unit.femtoseconds
        edchky_peptide_system.collision_rate = 1.0 / unit.picoseconds
        edchky_peptide_system.constraint_tolerance=1.e-7
        edchky_peptide_system.pH = 7.4
        testsystems = get_test_data('edchky_implicit', 'testsystems')
        edchky_peptide_system.positions = openmm.XmlSerializer.deserialize(
            open('{}/edchky-implicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        edchky_peptide_system.system = openmm.XmlSerializer.deserialize(open('{}/edchky-implicit.sys.xml'.format(testsystems)).read())
        edchky_peptide_system.prmtop = app.AmberPrmtopFile('{}/edchky-implicit.prmtop'.format(testsystems))
        edchky_peptide_system.topology = edchky_peptide_system.prmtop.topology
        edchky_peptide_system.cpin_filename = '{}/edchky-implicit.cpin'.format(testsystems)
        edchky_peptide_system.nsteps_per_ghmc = 1
        return edchky_peptide_system

    def test_peptide_instantaneous(self):
        """
        Run peptide in implicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_edchky_implicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)
        driver.define_pools({
            'group1': [0, 2, 4],
            'group2': [1, 3, 5],
            'GLU': [0],
            'ASP': [1],
            'CYS': [2],
            'HIS': [3],
            'LYS': [4],
            'TYR': [5],
        })

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal(), residue_pool='group2', nattempts=10)  # protonation

    def test_peptide_import_gk(self):
        """
        Import calibrated values for tyrosine
        """
        testsystem = self.setup_edchky_implicit()
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
        driver.import_gk_values(dict(TYR=[0.0,1.0]))

    def test_peptide_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) on LYS residue in peptide in implicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_edchky_implicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
        sams_sampler = SelfAdjustedMixtureSampling(driver, 4)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)
        sams_sampler.driver.define_pools({
            'group1': [0, 2, 4],
            'group2': [1, 3, 5],
            'GLU': [0],
            'ASP': [1],
            'CYS': [2],
            'HIS': [3],
            'LYS': [4],
            'TYR': [5],
        })

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal(), residue_pool="LYS")  # protonation
        sams_sampler.adapt_zetas('binary')

    @pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
    def test_peptide_ncmc(self):
        """
        Run peptide in implicit solvent with an ncmc state switch
        """
        testsystem = self.setup_edchky_implicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=2)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal(), nattempts=10)  # protonation

