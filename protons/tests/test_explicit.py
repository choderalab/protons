from __future__ import print_function

import pytest
import os
from simtk import unit, openmm
from protons import app
from protons.app import AmberProtonDrive, ForceFieldProtonDrive
from protons.app import SelfAdjustedMixtureSampling
from protons.app import UniformProposal
from protons.app import Topology
from protons.app import ForceField

from . import get_test_data
from .utilities import SystemSetup, create_compound_gbaoab_integrator


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
        tyrosine_explicit_system.topology = tyrosine_explicit_system.prmtop.topology
        tyrosine_explicit_system.cpin_filename = '{}/tyr.cpin'.format(testsystems)
        tyrosine_explicit_system.nsteps_per_ghmc = 1
        return tyrosine_explicit_system

    def test_tyrosine_instantaneous(self):
        """
        Run tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal())  # protonation

    def test_tyrosine_instantaneous_pH_adjusted(self):
        """
        Run tyrosine in explicit solvent with an instanteneous state switch at pH 7.4 using a pKa based adjustment
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal())  # protonation
        old_values = list(driver.titrationGroups[0].g_k_values)
        driver.adjust_to_ph(7.4)
        new_values = list(driver.titrationGroups[0].g_k_values)
        assert old_values != new_values, "Values are not adjusted"


    def test_tyrosine_import_gk(self):
        """
        Import calibrated values for tyrosine
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
        driver.import_gk_values(dict(TYR=[0.0,1.0]))

    def test_tyrosine_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
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
        Run SAMS (global update) tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
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
        Run tyrosine in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=2)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal(), nattempts=10)  # protonation


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
        imidazole_explicit_system.ffxml_filename = os.path.join(testsystems,'protons-imidazole.xml')
        imidazole_explicit_system.forcefield = ForceField('gaff.xml', imidazole_explicit_system.ffxml_filename)
        imidazole_explicit_system.gaff = 'gaff.xml'
        imidazole_explicit_system.pdbfile = app.PDBFile(os.path.join(testsystems, "imidazole-solvated-minimized.pdb"))
        imidazole_explicit_system.topology = imidazole_explicit_system.pdbfile.topology
        imidazole_explicit_system.nsteps_per_ghmc = 1
        imidazole_explicit_system.constraint_tolerance = 1.e-7
        return imidazole_explicit_system

    def test_imidazole_instantaneous(self):
        """
        Run imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.forcefield, testsystem.ffxml_filename, pressure= testsystem.pressure, perturbations_per_trial=0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal())  # protonation

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
        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.forcefield, testsystem.ffxml_filename, pressure=testsystem.pressure, perturbations_per_trial=0)

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
        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.forcefield, testsystem.ffxml_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
        driver.import_gk_values(gk_dict=dict(LIG=[0.0,1.0]))

    def test_imidazole_ncmc(self):
        """
        Run imidazole in explicit solvent with an NCMC state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.forcefield, testsystem.ffxml_filename, pressure=testsystem.pressure, perturbations_per_trial=10)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal())  # protonation

    def test_imidazole_attempts(self):
        """
        Run multiple attempts of imidazole in explicit solvent with an NCMC state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology,
                                       testsystem.system, testsystem.forcefield, testsystem.ffxml_filename, pressure=testsystem.pressure,
                                       perturbations_per_trial=10)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal(), nattempts=3)  # protonation

    def test_imidazole_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology,
                                       testsystem.system, testsystem.forcefield, testsystem.ffxml_filename, pressure=testsystem.pressure,
                                       perturbations_per_trial=0)
        sams_sampler = SelfAdjustedMixtureSampling(driver,0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal())  # protonation
        sams_sampler.adapt_zetas('binary')

    def test_imidazole_sams_instantaneous_global(self):
        """
        Run SAMS (global update) imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology,
                                       testsystem.system, testsystem.forcefield, testsystem.ffxml_filename, pressure= testsystem.pressure,
                                       perturbations_per_trial=0)
        sams_sampler = SelfAdjustedMixtureSampling(driver,0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal())  # protonation
        sams_sampler.adapt_zetas('global')


class TestAmberPeptide(object):
    """
    Simulating a peptide with sequence 'EDCHKY' in explicit solvent
    """

    default_platform = 'CPU'

    @staticmethod
    def setup_edchky_explicit():
        """
        Set up a tyrosine in explicit solvent
        """
        test_system = SystemSetup()
        test_system.temperature = 300.0 * unit.kelvin
        test_system.pressure = 1.0 * unit.atmospheres
        test_system.timestep = 1.0 * unit.femtoseconds
        test_system.collision_rate = 1.0 / unit.picoseconds
        test_system.constraint_tolerance = 1e-7
        test_system.pH = 9.6
        testsystems = get_test_data('edchky_explicit', 'testsystems')
        test_system.positions = openmm.XmlSerializer.deserialize(open('{}/edchky-explicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        test_system.system = openmm.XmlSerializer.deserialize(open('{}/edchky-explicit.sys.xml'.format(testsystems)).read())
        test_system.prmtop = app.AmberPrmtopFile('{}/edchky-explicit.prmtop'.format(testsystems))
        test_system.topology = test_system.prmtop.topology
        test_system.cpin_filename = '{}/edchky-explicit.cpin'.format(testsystems)
        return test_system

    def test_peptide_instantaneous(self):
        """
        Run peptide in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_edchky_explicit()
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
        testsystem = self.setup_edchky_explicit()
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=0)
        driver.import_gk_values(dict(TYR=[0.0,1.0]))

    def test_peptide_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) on LYS residue in peptide in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_edchky_explicit()
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

    @pytest.mark.slowtest
    @pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
    def test_peptide_ncmc(self):
        """
        Run peptide in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_edchky_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.cpin_filename, pressure=testsystem.pressure, perturbations_per_trial=2)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal(), nattempts=10)  # protonation


class TestForceFieldImidazoleExplicitpHAdjusted(object):
    """Tests for pH adjusting imidazole weights in explict solvent (TIP3P)"""

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
        imidazole_explicit_system.ffxml_filename = os.path.join(testsystems,'protons-imidazole-ph-feature.xml')
        imidazole_explicit_system.forcefield = ForceField('gaff.xml', imidazole_explicit_system.ffxml_filename)
        imidazole_explicit_system.gaff = 'gaff.xml'
        imidazole_explicit_system.pdbfile = app.PDBFile(os.path.join(testsystems, "imidazole-solvated-minimized.pdb"))
        imidazole_explicit_system.topology = imidazole_explicit_system.pdbfile.topology
        imidazole_explicit_system.nsteps_per_ghmc = 1
        imidazole_explicit_system.constraint_tolerance = 1.e-7
        return imidazole_explicit_system

    def test_imidazole_instantaneous_pH_adjust(self):
        """
        Run imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.forcefield, testsystem.ffxml_filename, pressure= testsystem.pressure, perturbations_per_trial=0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        old_values = list(driver.titrationGroups[0].g_k_values)
        driver.adjust_to_ph(7.4)
        new_values = list(driver.titrationGroups[0].g_k_values)
        assert old_values != new_values, "Values are not adjusted"

    def test_imidazole_instantaneous_wrongpH_adjust(self):
        """
        Run imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system, testsystem.forcefield, testsystem.ffxml_filename, pressure= testsystem.pressure, perturbations_per_trial=0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)
        with pytest.raises(ValueError) as exception_info:
            driver.adjust_to_ph(4.0)

    def test_imidazole_instantaneous_serialize(self):
        """
        Test the serialization of an imidazole titration group.
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(testsystem.temperature, testsystem.topology, testsystem.system,
                                       testsystem.forcefield, testsystem.ffxml_filename, pressure=testsystem.pressure,
                                       perturbations_per_trial=0)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)
        driver.adjust_to_ph(7.4)
        x = driver.titrationGroups[0].serialize()

