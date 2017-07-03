from __future__ import print_function

import pytest
from simtk import unit, openmm

from protons import AmberProtonDrive, ForceFieldProtonDrive
from protons import SelfAdjustedMixtureSampling
from protons import UniformProposal
from protons import app
from . import get_test_data
from .utilities import SystemSetup, create_compound_gbaoab_integrator, create_compound_ghmc_integrator


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

        driver = AmberProtonDrive(testsystem.temperature, testsystem.pressure, testsystem.topology,testsystem.system, testsystem.cpin_filename, ncmc_steps_per_trial=0)
        proposal = UniformProposal()
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(proposal)  # protonation

    def test_tyrosine_import_gk(self):
        """
        Import calibrated values for tyrosine
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_ghmc_integrator(testsystem)

        driver = AmberProtonDrive(testsystem.temperature, testsystem.pressure, testsystem.topology,testsystem.system, testsystem.cpin_filename, ncmc_steps_per_trial=0)
        driver.import_gk_values(dict(TYR=[0.0,1.0]))

    def test_tyrosine_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.pressure, testsystem.topology,testsystem.system, testsystem.cpin_filename, ncmc_steps_per_trial=0)
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
        driver = AmberProtonDrive(testsystem.temperature, testsystem.pressure, testsystem.topology,testsystem.system, testsystem.cpin_filename, ncmc_steps_per_trial=0)
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
        driver = AmberProtonDrive(testsystem.pH, testsystem.system, testsystem.temperature,
                                  pressure=testsystem.pressure,
                                  simultaneous_proposal_probability=testsystem.cpin_filename, ncmc_steps_per_trial=10)
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
        driver = AmberProtonDrive(testsystem.pH, testsystem.system, testsystem.temperature,
                                  pressure=testsystem.pressure,
                                  simultaneous_proposal_probability=testsystem.cpin_filename, ncmc_steps_per_trial=10)
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
        driver = AmberProtonDrive(testsystem.pH, testsystem.system, testsystem.temperature,
                                  pressure=testsystem.pressure,
                                  simultaneous_proposal_probability=testsystem.cpin_filename, ncmc_steps_per_trial=10)
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
        driver = AmberProtonDrive(testsystem.pH, testsystem.system, testsystem.temperature,
                                  pressure=testsystem.pressure,
                                  simultaneous_proposal_probability=testsystem.cpin_filename, ncmc_steps_per_trial=10)
        sams_sampler = SelfAdjustedMixtureSampling(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'global')


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
        imidazole_explicit_system.forcefield = app.ForceField('gaff.xml', imidazole_explicit_system.ffxml_filename)
        imidazole_explicit_system.gaff = 'gaff.xml'
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
        driver = ForceFieldProtonDrive(testsystem.system, pressure=testsystem.pressure, topology=testsystem.forcefield,
                                       system=testsystem.pH, forcefield=testsystem.pressure,
                                       ffxml_files=testsystem.temperature,
                                       simultaneous_proposal_probability=testsystem.forcefield, ncmc_steps_per_trial=0,
                                       ncmc_prop_per_step=compound_integrator)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        driver.update(context)  # protonation

    @staticmethod
    def pattern_from_multiline(multiline, pattern):
        """Return only lines that contain the pattern
        
        Parameters
        ----------
        multiline - multiline str        
        pattern - str

        Returns
        -------
        multiline str containing pattern
        """

        return '\n'.join([line for line in multiline.splitlines() if pattern in line])

    def test_system_integrity(self):
        """
        Set up imidazole, and assure that the systems particles have not been modified after driver instantiation.
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.system, pressure=testsystem.pressure, topology=testsystem.forcefield,
                                       system=testsystem.pH, forcefield=testsystem.pressure,
                                       ffxml_files=testsystem.temperature,
                                       simultaneous_proposal_probability=testsystem.forcefield, ncmc_steps_per_trial=0,
                                       ncmc_prop_per_step=compound_integrator)

        _system_folder = get_test_data('imidazole_explicit', 'testsystems')
        # Only grab the particles from each system. It is okay if versions et cetera mismatch.
        original_system = self.pattern_from_multiline(open('{}/imidazole-explicit.sys.xml'.format(_system_folder)).read(), '<Particle')
        after_driver = self.pattern_from_multiline(openmm.XmlSerializer.serialize(testsystem.system),'<Particle')

        # Make sure there are no differences between the particles in each system
        assert original_system == after_driver

    def test_imidazole_import_gk(self):
        """
        Import calibrated values for imidazole weights
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.system, pressure=testsystem.pressure, topology=testsystem.forcefield,
                                       system=testsystem.pH, forcefield=testsystem.pressure,
                                       ffxml_files=testsystem.temperature,
                                       simultaneous_proposal_probability=testsystem.forcefield, ncmc_steps_per_trial=0,
                                       ncmc_prop_per_step=compound_integrator)
        driver.import_gk_values(gk_dict=dict(LIG=[0.0,1.0]))

    def test_imidazole_ncmc(self):
        """
        Run imidazole in explicit solvent with an NCMC state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_ghmc_integrator(testsystem)

        driver = ForceFieldProtonDrive(testsystem.system, pressure=testsystem.pressure, topology=testsystem.forcefield,
                                       system=testsystem.pH, forcefield=testsystem.pressure,
                                       ffxml_files=testsystem.temperature,
                                       simultaneous_proposal_probability=testsystem.forcefield, ncmc_steps_per_trial=10,
                                       ncmc_prop_per_step=compound_integrator)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        driver.update(context)  # protonation

    def test_imidazole_attempts(self):
        """
        Run multiple attempts of imidazole in explicit solvent with an NCMC state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_ghmc_integrator(testsystem)

        driver = ForceFieldProtonDrive(testsystem.system, pressure=testsystem.pressure, topology=testsystem.forcefield,
                                       system=testsystem.pH, forcefield=testsystem.pressure,
                                       ffxml_files=testsystem.temperature,
                                       simultaneous_proposal_probability=testsystem.forcefield, ncmc_steps_per_trial=10,
                                       ncmc_prop_per_step=compound_integrator)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)

        compound_integrator.step(10)  # MD
        driver.update(context, nattempts=15)  # protonation

    def test_imidazole_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()

        compound_integrator = create_compound_ghmc_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.system, pressure=testsystem.pressure, topology=testsystem.forcefield,
                                       system=testsystem.pH, forcefield=testsystem.pressure,
                                       ffxml_files=testsystem.temperature,
                                       simultaneous_proposal_probability=testsystem.forcefield, ncmc_steps_per_trial=0,
                                       ncmc_prop_per_step=compound_integrator)
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
        driver = ForceFieldProtonDrive(testsystem.system, pressure=testsystem.pressure, topology=testsystem.forcefield,
                                       system=testsystem.pH, forcefield=testsystem.pressure,
                                       ffxml_files=testsystem.temperature,
                                       simultaneous_proposal_probability=testsystem.forcefield, ncmc_steps_per_trial=0,
                                       ncmc_prop_per_step=compound_integrator)

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

        driver = ForceFieldProtonDrive(testsystem.system, pressure=testsystem.pressure, topology=testsystem.forcefield,
                                       system=testsystem.pH, forcefield=testsystem.pressure,
                                       ffxml_files=testsystem.temperature,
                                       simultaneous_proposal_probability=testsystem.forcefield, ncmc_steps_per_trial=10,
                                       ncmc_prop_per_step=compound_integrator)

        sams_sampler = SelfAdjustedMixtureSampling(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, driver.compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(context)  # protonation
        sams_sampler.adapt_zetas(context, 'binary')
