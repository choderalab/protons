from __future__ import print_function

import os
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
from lxml import etree
from numpy.random import choice
from saltswap.swapper import Swapper
from saltswap.wrappers import Salinator
from simtk import unit, openmm

from protons import app
from protons.app import AmberProtonDrive, ForceFieldProtonDrive, NCMCProtonDrive
from protons.app import ForceField
from protons.app import SAMSCalibrationEngine
from protons.app import UniformProposal
from protons.app.proposals import OneDirectionChargeProposal
from . import get_test_data
from .utilities import SystemSetup, create_compound_gbaoab_integrator, hasCUDA


class TestAmberTyrosineExplicit(object):
    """
    Simulating a tyrosine in explicit solvent
    """

    default_platform = "CPU"

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
        testsystems = get_test_data("tyr_explicit", "testsystems")
        tyrosine_explicit_system.positions = openmm.XmlSerializer.deserialize(
            open("{}/tyr.state.xml".format(testsystems)).read()
        ).getPositions(asNumpy=True)
        tyrosine_explicit_system.system = openmm.XmlSerializer.deserialize(
            open("{}/tyr.sys.xml".format(testsystems)).read()
        )
        tyrosine_explicit_system.prmtop = app.AmberPrmtopFile(
            "{}/tyr.prmtop".format(testsystems)
        )
        tyrosine_explicit_system.topology = tyrosine_explicit_system.prmtop.topology
        tyrosine_explicit_system.cpin_filename = "{}/tyr.cpin".format(testsystems)
        tyrosine_explicit_system.nsteps_per_ghmc = 1
        return tyrosine_explicit_system

    def test_tyrosine_instantaneous(self):
        """
        Run tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
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

        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
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

        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        driver.import_gk_values(dict(TYR=[0.0, 1.0]))

    def test_tyrosine_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        driver.enable_calibration(app.driver.SAMSApproach.ONESITE, group_index=0)
        sams_sampler = SAMSCalibrationEngine(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal())  # protonation
        sams_sampler.adapt_zetas(app.driver.UpdateRule.BINARY)

    def test_tyrosine_sams_instantaneous_global(self):
        """
        Run SAMS (global update) tyrosine in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_tyrosine_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        driver.enable_calibration(app.driver.SAMSApproach.ONESITE, group_index=0)
        sams_sampler = SAMSCalibrationEngine(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal())  # protonation
        sams_sampler.adapt_zetas(app.driver.UpdateRule.GLOBAL)

    def test_tyrosine_ncmc(self):
        """
        Run tyrosine in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_tyrosine_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=2,
        )
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal(), nattempts=10)  # protonation


class TestForceFieldImidazoleExplicit(object):
    """Tests for imidazole in explict solvent (TIP3P)"""

    default_platform = "CPU"

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
        testsystems = get_test_data("imidazole_explicit", "testsystems")
        imidazole_explicit_system.positions = openmm.XmlSerializer.deserialize(
            open("{}/imidazole-explicit.state.xml".format(testsystems)).read()
        ).getPositions(asNumpy=True)
        imidazole_explicit_system.system = openmm.XmlSerializer.deserialize(
            open("{}/imidazole-explicit.sys.xml".format(testsystems)).read()
        )
        imidazole_explicit_system.ffxml_filename = os.path.join(
            testsystems, "protons-imidazole.xml"
        )
        imidazole_explicit_system.forcefield = ForceField(
            "gaff.xml", imidazole_explicit_system.ffxml_filename
        )
        imidazole_explicit_system.gaff = "gaff.xml"
        imidazole_explicit_system.pdbfile = app.PDBFile(
            os.path.join(testsystems, "imidazole-solvated-minimized.pdb")
        )
        imidazole_explicit_system.topology = imidazole_explicit_system.pdbfile.topology
        imidazole_explicit_system.nsteps_per_ghmc = 1
        imidazole_explicit_system.constraint_tolerance = 1.0e-7
        return imidazole_explicit_system

    def test_imidazole_instantaneous(self):
        """
        Run imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
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

        return "\n".join([line for line in multiline.splitlines() if pattern in line])

    def test_system_integrity(self):
        """
        Set up imidazole, and assure that the systems particles have not been modified after driver instantiation.
        """
        testsystem = self.setup_imidazole_explicit()
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )

        _system_folder = get_test_data("imidazole_explicit", "testsystems")
        # Only grab the particles from each system. It is okay if versions et cetera mismatch.
        original_system = self.pattern_from_multiline(
            open("{}/imidazole-explicit.sys.xml".format(_system_folder)).read(),
            "<Particle ",
        )
        after_driver = self.pattern_from_multiline(
            openmm.XmlSerializer.serialize(testsystem.system), "<Particle "
        )

        # Make sure there are no differences between the particles in each system
        assert original_system == after_driver

    def test_imidazole_import_gk(self):
        """
        Import calibrated values for imidazole weights
        """
        testsystem = self.setup_imidazole_explicit()
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        driver.import_gk_values(gk_dict=dict(LIG=[0.0, 1.0]))

    def test_imidazole_ncmc(self):
        """
        Run imidazole in explicit solvent with an NCMC state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=10,
        )
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

        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=10,
        )
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
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        driver.enable_calibration(app.driver.SAMSApproach.ONESITE, group_index=0)
        sams_sampler = SAMSCalibrationEngine(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal())  # protonation
        sams_sampler.adapt_zetas(app.driver.UpdateRule.BINARY)

    def test_imidazole_sams_instantaneous_global(self):
        """
        Run SAMS (global update) imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        driver.enable_calibration(app.driver.SAMSApproach.ONESITE, group_index=0)
        sams_sampler = SAMSCalibrationEngine(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal())  # protonation
        sams_sampler.adapt_zetas(app.driver.UpdateRule.GLOBAL)


class TestAmberPeptide(object):
    """
    Simulating a peptide with sequence 'EDCHKY' in explicit solvent
    """

    default_platform = "CPU"

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
        testsystems = get_test_data("edchky_explicit", "testsystems")
        test_system.positions = openmm.XmlSerializer.deserialize(
            open("{}/edchky-explicit.state.xml".format(testsystems)).read()
        ).getPositions(asNumpy=True)
        test_system.system = openmm.XmlSerializer.deserialize(
            open("{}/edchky-explicit.sys.xml".format(testsystems)).read()
        )
        test_system.prmtop = app.AmberPrmtopFile(
            "{}/edchky-explicit.prmtop".format(testsystems)
        )
        test_system.topology = test_system.prmtop.topology
        test_system.cpin_filename = "{}/edchky-explicit.cpin".format(testsystems)
        return test_system

    def test_peptide_instantaneous(self):
        """
        Run peptide in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_edchky_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)

        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)
        driver.define_pools(
            {
                "group1": [0, 2, 4],
                "group2": [1, 3, 5],
                "GLU": [0],
                "ASP": [1],
                "CYS": [2],
                "HIS": [3],
                "LYS": [4],
                "TYR": [5],
            }
        )

        compound_integrator.step(10)  # MD
        driver.update(
            UniformProposal(), residue_pool="group2", nattempts=10
        )  # protonation

    def test_peptide_import_gk(self):
        """
        Import calibrated values for tyrosine
        """
        testsystem = self.setup_edchky_explicit()
        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        driver.import_gk_values(dict(TYR=[0.0, 1.0]))

    def test_peptide_sams_instantaneous_binary(self):
        """
        Run SAMS (binary update) on LYS residue in peptide in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_edchky_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        driver.enable_calibration(app.driver.SAMSApproach.ONESITE, group_index=4)
        sams_sampler = SAMSCalibrationEngine(driver)
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        sams_sampler.driver.attach_context(context)
        sams_sampler.driver.define_pools(
            {
                "group1": [0, 2, 4],
                "group2": [1, 3, 5],
                "GLU": [0],
                "ASP": [1],
                "CYS": [2],
                "HIS": [3],
                "LYS": [4],
                "TYR": [5],
            }
        )

        compound_integrator.step(10)  # MD
        sams_sampler.driver.update(UniformProposal(), residue_pool="LYS")  # protonation
        sams_sampler.adapt_zetas(app.driver.UpdateRule.BINARY)

    @pytest.mark.slowtest
    @pytest.mark.skipif(
        os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
    )
    def test_peptide_ncmc(self):
        """
        Run peptide in explicit solvent with an ncmc state switch
        """
        testsystem = self.setup_edchky_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=2,
        )
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        compound_integrator.step(10)  # MD
        driver.update(UniformProposal(), nattempts=10)  # protonation

    def test_peptide_serialization(self):
        """
        Set up a peptide system and serialize it.
        """
        testsystem = self.setup_edchky_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=2,
        )
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)
        driver.adjust_to_ph(7.4)
        x = driver.state_to_xml()

    @pytest.mark.slowtest
    @pytest.mark.skipif(
        os.environ.get("TRAVIS", None) == "true", reason="Skip slow test on travis."
    )
    def test_peptide_deserialization(self):
        """
        Set up a peptide system and serialize it, then deserialize it.
        """
        testsystem = self.setup_edchky_explicit()

        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        olddrive = AmberProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.cpin_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=2,
        )
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        olddrive.attach_context(context)
        olddrive.adjust_to_ph(7.4)

        x = olddrive.state_to_xml()
        newdrive = NCMCProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            pressure=testsystem.pressure,
            perturbations_per_trial=2,
        )
        newdrive.state_from_xml_tree(etree.fromstring(x))

        for old_res, new_res in zip(olddrive.titrationGroups, newdrive.titrationGroups):
            assert old_res == new_res, "Residues don't match. {} :: {}".format(
                old_res.name, new_res.name
            )

        newdrive.attach_context(context)
        compound_integrator.step(10)
        newdrive.update(UniformProposal(), nattempts=1)


class TestForceFieldImidazoleExplicitpHAdjusted:
    """Tests for pH adjusting imidazole weights in explict solvent (TIP3P)"""

    default_platform = "CPU"

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
        testsystems = get_test_data("imidazole_explicit", "testsystems")
        imidazole_explicit_system.positions = openmm.XmlSerializer.deserialize(
            open("{}/imidazole-explicit.state.xml".format(testsystems)).read()
        ).getPositions(asNumpy=True)
        imidazole_explicit_system.system = openmm.XmlSerializer.deserialize(
            open("{}/imidazole-explicit.sys.xml".format(testsystems)).read()
        )
        imidazole_explicit_system.ffxml_filename = os.path.join(
            testsystems, "protons-imidazole-ph-feature.xml"
        )
        imidazole_explicit_system.forcefield = ForceField(
            "gaff.xml", imidazole_explicit_system.ffxml_filename
        )
        imidazole_explicit_system.gaff = "gaff.xml"
        imidazole_explicit_system.pdbfile = app.PDBFile(
            os.path.join(testsystems, "imidazole-solvated-minimized.pdb")
        )
        imidazole_explicit_system.topology = imidazole_explicit_system.pdbfile.topology
        imidazole_explicit_system.nsteps_per_ghmc = 1
        imidazole_explicit_system.constraint_tolerance = 1.0e-7
        return imidazole_explicit_system

    def test_imidazole_instantaneous_pH_adjust(self):
        """
        Run imidazole in explicit solvent with an instanteneous state switch
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
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
        Run imidazole in explicit solvent with an instanteneous state switch, trying to set to an unavailable pH.
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
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
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)
        driver.adjust_to_ph(7.4)
        x = driver.titrationGroups[0].serialize()

    def test_imidazole_instantaneous_deserialize(self):
        """
        Test the deserialization of an imidazole titration group.
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)
        driver.adjust_to_ph(7.4)
        x = driver.titrationGroups[0].serialize()
        y = app.driver._TitratableResidue.from_serialized_xml(x)

    def test_imidazole_serialization_correctness(self):
        """
        Test the correctness of a deserialized imidazole titration group.
        """
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)
        driver.adjust_to_ph(7.4)
        before_group = driver.titrationGroups[0]
        x = before_group.serialize()
        after_group = app.driver._TitratableResidue.from_serialized_xml(x)
        assert (
            before_group == after_group
        ), "The deserialized group does not match what was serialized."


class TestForceFieldImidazoleSaltswap:
    """Tests for saltswap dependent functionality on imidazole in explicit solvent."""

    default_platform = "CPU"

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
        testsystems = get_test_data("imidazole_explicit", "testsystems")
        imidazole_explicit_system.positions = openmm.XmlSerializer.deserialize(
            open("{}/imidazole-explicit.state.xml".format(testsystems)).read()
        ).getPositions(asNumpy=True)
        imidazole_explicit_system.system = openmm.XmlSerializer.deserialize(
            open("{}/imidazole-explicit.sys.xml".format(testsystems)).read()
        )
        imidazole_explicit_system.ffxml_filename = os.path.join(
            testsystems, "protons-imidazole-ph-feature.xml"
        )
        imidazole_explicit_system.forcefield = ForceField(
            "gaff.xml", imidazole_explicit_system.ffxml_filename
        )
        imidazole_explicit_system.gaff = "gaff.xml"
        imidazole_explicit_system.pdbfile = app.PDBFile(
            os.path.join(testsystems, "imidazole-solvated-minimized.pdb")
        )
        imidazole_explicit_system.topology = imidazole_explicit_system.pdbfile.topology
        imidazole_explicit_system.nsteps_per_ghmc = 1
        imidazole_explicit_system.constraint_tolerance = 1.0e-7
        return imidazole_explicit_system

    def test_saltswap_incorporation(self):
        """Test if the attachment of a swapper works."""
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=0,
        )
        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        # The salinator initializes the system salts
        salinator = Salinator(
            context=context,
            system=testsystem.system,
            topology=testsystem.topology,
            ncmc_integrator=compound_integrator.getIntegrator(1),
            salt_concentration=0.2 * unit.molar,
            pressure=testsystem.pressure,
            temperature=testsystem.temperature,
        )
        salinator.neutralize()
        salinator.initialize_concentration()
        swapper = salinator.swapper
        driver.attach_swapper(swapper)

        driver.adjust_to_ph(7.4)
        compound_integrator.step(1)
        driver.update(UniformProposal())

    def test_saltswap_total_charge(self):
        """Test if the system charge is neutral when perturbing the system using ncmc, with saltswap."""
        testsystem = self.setup_imidazole_explicit()
        compound_integrator = create_compound_gbaoab_integrator(testsystem)
        driver = ForceFieldProtonDrive(
            testsystem.temperature,
            testsystem.topology,
            testsystem.system,
            testsystem.forcefield,
            testsystem.ffxml_filename,
            pressure=testsystem.pressure,
            perturbations_per_trial=1,
        )

        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(testsystem.system, compound_integrator, platform)
        context.setPositions(testsystem.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(testsystem.temperature)
        driver.attach_context(context)

        # The salinator initializes the system salts
        salinator = Salinator(
            context=context,
            system=testsystem.system,
            topology=testsystem.topology,
            ncmc_integrator=compound_integrator.getIntegrator(1),
            salt_concentration=0.2 * unit.molar,
            pressure=testsystem.pressure,
            temperature=testsystem.temperature,
        )
        salinator.neutralize()
        salinator.initialize_concentration()
        swapper = salinator.swapper
        driver.attach_swapper(swapper)

        driver.adjust_to_ph(7.4)
        swap_proposal = OneDirectionChargeProposal()
        old_charge = self.calculate_explicit_solvent_system_charge(driver.system)
        # The initial state is neutral, the new state is +1
        # Pick swaps using the proposal method explicitly.
        saltswap_residue_indices, saltswap_state_pairs, log_ratio = swap_proposal.propose_swaps(
            driver, 0, 1
        )
        # First residue is updates from state 0, to state 1, and the previously selected salt swap is added to the protocol
        driver._perform_ncmc_protocol(
            [0],
            np.asarray([0]),
            np.asarray([1]),
            salt_residue_indices=saltswap_residue_indices,
            salt_states=saltswap_state_pairs,
        )

        # This should be the same as the old charge
        new_charge = self.calculate_explicit_solvent_system_charge(driver.system)
        # Bookkeeping
        driver.excess_ions -= 1

        # The saltswap indices are updated to indicate the change of species
        for saltswap_residue, (from_ion_state, to_ion_state) in zip(
            saltswap_residue_indices, saltswap_state_pairs
        ):
            swapper.stateVector[saltswap_residue] = to_ion_state

        # If the total charge has changed during the protocol, something is wrong.
        assert (
            pytest.approx(0.0, rel=0.0, abs=1.0e-12) == old_charge - new_charge
        ), "The total charge changed after NCMC."

    @staticmethod
    def calculate_explicit_solvent_system_charge(system):
        """Calculate the total charge of an explicit solvent system."""

        serializer = openmm.XmlSerializer
        xml = serializer.serialize(system)
        tree = etree.fromstring(xml)
        tot_charge = np.float64(0.0)
        for q in tree.xpath(
            "/System/Forces/Force[@type='NonbondedForce']/Particles/Particle/@q"
        ):
            tot_charge += np.float64(q)

        return tot_charge


class TestIonSwapping:
    """This class contains some simulation-independent testing features of schemes for selecting what ions need to be
    added/removed from a simulation to facilitate charge changes."""

    histidine = np.asarray(
        [0, 0, +1], dtype=int
    )  # Has two neutral states, and one positive state
    aspartate = np.asarray(
        [-1, 0, 0, 0, 0], dtype=int
    )  # Has 4 neutral syn/anti hydrogen positions, also covers glutamate
    lysine = np.asarray([0, +1], dtype=int)
    tyrosine = np.asarray([0, -1], dtype=int)

    diprotic_acid = np.asarray([-2, -1, -1, 0], dtype=int)
    diprotic_base = np.asarray([0, 1, 1, 2], dtype=int)
    zwitter_one = np.asarray([1, 0, 0, -1], dtype=int)
    zwitter_two = np.asarray(
        [-2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=int
    )

    residues = {
        "His": histidine,
        "Asp": aspartate,
        "Glu": deepcopy(aspartate),
        "Cys": deepcopy(tyrosine),
        "Tyr": tyrosine,
        "Lys": lysine,
        "Diprotic acid": diprotic_acid,
        "Diprotic base": diprotic_base,
        "Zwitter ion": zwitter_one,
        "Double zwitter ion": zwitter_two,
    }

    def test_accumulation_chen_roux_swap_proposals(self):
        """
        Select ion swaps for a hypothetical chain of protonation state changes.
        """

        n_samples = 1000
        for resname, residue in TestIonSwapping.residues.items():
            max_anions, max_cations, min_anions, min_cations, span_q = self.sample_residue_trajectory_chenroux(
                n_samples, residue
            )

            self._check_for_accumulation_depletion(
                max_anions, max_cations, min_anions, min_cations, resname, span_q
            )

    @staticmethod
    def sample_residue_trajectory_chenroux(n_samples: int, residue: np.ndarray):
        """
        For a given residue, randomly sample a trajectory of states from it.
        """
        # initial charge, and initial counterions
        res_charge = [residue[0]]
        # If residue is negative, add cations
        ncat = [-res_charge[0]] if res_charge[0] < 0 else [0]
        # If residue is positive, add anions
        nani = [-res_charge[0]] if res_charge[0] > 0 else [0]
        for x in range(n_samples):
            initial_charge = res_charge[x]
            final_charge = choice(residue)
            swaps = OneDirectionChargeProposal._select_swaps_chenroux(
                initial_charge, final_charge
            )
            cat = 0
            cat += swaps["water_to_cation"]
            cat -= swaps["cation_to_water"]
            ani = 0
            ani += swaps["water_to_anion"]
            ani -= swaps["anion_to_water"]

            ncat.append(ncat[x] + cat)
            nani.append(nani[x] + ani)
            res_charge.append(final_charge)

        deltaq = [0]  # first delta is 0
        deltaq.extend(list(res_charge[x + 1] - res_charge[x] for x in range(n_samples)))
        delta_cat = [0]
        delta_cat.extend(list(ncat[x + 1] - ncat[x] for x in range(n_samples)))
        delta_ani = [0]
        delta_ani.extend(list(nani[x + 1] - nani[x] for x in range(n_samples)))

        max_q = np.max(res_charge)
        min_q = np.min(res_charge)
        span_q = max_q - min_q
        max_anions = np.max(nani)
        min_anions = np.min(nani)
        max_cations = np.max(ncat)
        min_cations = np.min(ncat)
        return max_anions, max_cations, min_anions, min_cations, span_q

    @staticmethod
    def _check_for_accumulation_depletion(
        max_anions: int,
        max_cations: int,
        min_anions: int,
        min_cations: int,
        resname: str,
        span_q: int,
    ):
        """
        Given some properties of the residue, and the ion depletion, assert whether accumulation or depletion of ions is occuring
        """

        assert span_q >= max_anions, (
            "The number of anions for residue {} is too large, could indicate "
            "accumulation of ions. Largest q_span: {} anions: {}".format(
                resname, span_q, max_anions
            )
        )
        assert span_q >= max_cations, (
            "The number of cations for residue {} is too large, could indicate "
            "accumulation of ions. Largest_q_span q: {} cations: {}".format(
                resname, span_q, max_cations
            )
        )
        assert span_q >= abs(min_anions), (
            "The number of anions for residue {} is too small, could indicate "
            "depletion of ions. Largest q_span: {} anions: {}".format(
                resname, span_q, min_anions
            )
        )
        assert span_q >= abs(min_cations), (
            "The number of cations for residue {} is too small, could indicate "
            "depletion of ions. Largest_q_span q: {} cations: {}".format(
                resname, span_q, max_cations
            )
        )
