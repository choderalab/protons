from __future__ import print_function

import os
from collections import Counter
from copy import deepcopy

import uuid


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
from protons.app import SelfAdjustedMixtureSampler
from protons.app import UniformProposal
from protons.app.template_patches import patch_cooh
from protons.app.proposals import OneDirectionChargeProposal, COOHDummyMover
from protons.tests import get_test_data
from protons.tests.utilities import (
    SystemSetup,
    create_compound_gbaoab_integrator,
    hasCUDA,
)
import logging
from random import choice, uniform, sample
from protons.app import log
from math import exp

log.setLevel(logging.DEBUG)


class TestCarboxylicAcid:
    if hasCUDA:
        default_platform_name = "CUDA"
    else:
        default_platform_name = "CPU"

    platform = openmm.Platform.getPlatformByName(default_platform_name)

    @staticmethod
    def setup_viologen_vacuum():
        """
        Set up viologen in vacuum
        """
        viologen = SystemSetup()
        viologen.temperature = 300.0 * unit.kelvin
        viologen.pressure = 1.0 * unit.atmospheres
        viologen.timestep = 1.0 * unit.femtoseconds
        viologen.collision_rate = 1.0 / unit.picoseconds
        viologen.pH = 7.0
        testsystems = get_test_data("viologen", "testsystems")
        viologen.ffxml_files = os.path.join(testsystems, "viologen-protons.ffxml")
        viologen.gaff = os.path.join(testsystems, "gaff.xml")
        viologen.forcefield = ForceField(viologen.gaff, viologen.ffxml_files)

        viologen.pdbfile = app.PDBFile(os.path.join(testsystems, "viologen-vacuum.pdb"))
        viologen.topology = viologen.pdbfile.topology
        viologen.positions = viologen.pdbfile.getPositions(asNumpy=True)
        viologen.constraint_tolerance = 1.e-7

        viologen.integrator = openmm.LangevinIntegrator(
            viologen.temperature, viologen.collision_rate, viologen.timestep
        )

        viologen.integrator.setConstraintTolerance(viologen.constraint_tolerance)
        viologen.system = viologen.forcefield.createSystem(
            viologen.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )
        viologen.cooh1 = {  # indices in topology of the first cooh group
            "HO": 56,
            "OH": 0,
            "CO": 1,
            "OC": 2,
            "R": 3,
        }
        viologen.cooh2 = {  # indices in topology of the second cooh group
            "HO": 57,
            "OH": 27,
            "CO": 25,
            "OC": 26,
            "R": 24,
        }

        viologen.simulation = app.Simulation(
            viologen.topology,
            viologen.system,
            viologen.integrator,
            TestCarboxylicAcid.platform,
        )
        viologen.simulation.context.setPositions(viologen.positions)
        viologen.context = viologen.simulation.context
        viologen.perturbations_per_trial = 10
        viologen.propagations_per_step = 1

        return viologen

    @staticmethod
    def setup_viologen_implicit():
        """
        Set up viologen in implicit solvent
        """
        viologen = SystemSetup()
        viologen.temperature = 300.0 * unit.kelvin
        viologen.pressure = 1.0 * unit.atmospheres
        viologen.timestep = 1.0 * unit.femtoseconds
        viologen.collision_rate = 1.0 / unit.picoseconds
        viologen.pH = 7.0
        testsystems = get_test_data("viologen", "testsystems")
        viologen.ffxml_files = os.path.join(testsystems, "viologen-protons.ffxml")
        viologen.gaff = os.path.join(testsystems, "gaff.xml")
        viologen.forcefield = ForceField(viologen.gaff, "gaff-obc2.xml", viologen.ffxml_files)

        viologen.pdbfile = app.PDBFile(os.path.join(testsystems, "viologen-vacuum.pdb"))
        viologen.topology = viologen.pdbfile.topology
        viologen.positions = viologen.pdbfile.getPositions(asNumpy=True)
        viologen.constraint_tolerance = 1.e-7

        viologen.integrator = openmm.LangevinIntegrator(
            viologen.temperature, viologen.collision_rate, viologen.timestep
        )

        viologen.integrator.setConstraintTolerance(viologen.constraint_tolerance)
        viologen.system = viologen.forcefield.createSystem(
            viologen.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )
        viologen.cooh1 = {  # indices in topology of the first cooh group
            "HO": 56,
            "OH": 0,
            "CO": 1,
            "OC": 2,
            "R": 3,
        }
        viologen.cooh2 = {  # indices in topology of the second cooh group
            "HO": 57,
            "OH": 27,
            "CO": 25,
            "OC": 26,
            "R": 24,
        }

        viologen.simulation = app.Simulation(
            viologen.topology,
            viologen.system,
            viologen.integrator,
            TestCarboxylicAcid.platform,
        )
        viologen.simulation.context.setPositions(viologen.positions)
        viologen.context = viologen.simulation.context
        viologen.perturbations_per_trial = 10
        viologen.propagations_per_step = 1

        return viologen

    @staticmethod
    def setup_viologen_water():
        """
        Set up viologen in water
        """
        viologen = SystemSetup()
        viologen.temperature = 300.0 * unit.kelvin
        viologen.pressure = 1.0 * unit.atmospheres
        viologen.timestep = 1.0 * unit.femtoseconds
        viologen.collision_rate = 1.0 / unit.picoseconds
        viologen.pH = 7.0
        testsystems = get_test_data("viologen", "testsystems")
        viologen.ffxml_files = os.path.join(testsystems, "viologen-protons-cooh.ffxml")
        viologen.gaff = os.path.join(testsystems, "gaff.xml")
        viologen.forcefield = ForceField(
            viologen.gaff, viologen.ffxml_files, "tip3p.xml"
        )

        viologen.pdbfile = app.PDBFile(
            os.path.join(testsystems, "viologen-solvated.pdb")
        )
        viologen.topology = viologen.pdbfile.topology
        viologen.positions = viologen.pdbfile.getPositions(asNumpy=True)
        viologen.constraint_tolerance = 1.e-7

        viologen.integrator = create_compound_gbaoab_integrator(viologen)

        viologen.integrator.setConstraintTolerance(viologen.constraint_tolerance)
        viologen.system = viologen.forcefield.createSystem(
            viologen.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )
        viologen.system.addForce(
            openmm.MonteCarloBarostat(viologen.pressure, viologen.temperature, 25)
        )
        viologen.cooh1 = {  # indices in topology of the first cooh group
            "HO": 56,
            "OH": 1,
            "CO": 0,
            "OC": 2,
            "R": 3,
        }
        viologen.cooh2 = {  # indices in topology of the second cooh group
            "HO": 57,
            "OH": 27,
            "CO": 25,
            "OC": 26,
            "R": 24,
        }

        viologen.simulation = app.Simulation(
            viologen.topology,
            viologen.system,
            viologen.integrator,
            TestCarboxylicAcid.platform,
        )
        viologen.simulation.context.setPositions(viologen.positions)
        viologen.context = viologen.simulation.context
        viologen.perturbations_per_trial = 1000
        viologen.propagations_per_step = 1

        return viologen

    @staticmethod
    def setup_amino_acid_water(three_letter_code):
        """
        Set up glutamic acid in water
        """
        if three_letter_code not in ["glh", "ash"]:
            raise ValueError("Amino acid not available.")

        aa = SystemSetup()
        aa.temperature = 300.0 * unit.kelvin
        aa.pressure = 1.0 * unit.atmospheres
        aa.timestep = 1.0 * unit.femtoseconds
        aa.collision_rate = 1.0 / unit.picoseconds
        aa.pH = 7.0
        testsystems = get_test_data("amino_acid", "testsystems")
        aa.ffxml_files = "amber10-constph.xml"
        aa.forcefield = ForceField(aa.ffxml_files, "tip3p.xml")

        aa.pdbfile = app.PDBFile(os.path.join(testsystems, "{}.pdb".format(three_letter_code))
        )
        aa.topology = aa.pdbfile.topology
        aa.positions = aa.pdbfile.getPositions(asNumpy=True)
        aa.constraint_tolerance = 1.e-7

        aa.integrator = create_compound_gbaoab_integrator(aa)

        aa.integrator.setConstraintTolerance(aa.constraint_tolerance)
        aa.system = aa.forcefield.createSystem(
            aa.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )
        aa.system.addForce(
            openmm.MonteCarloBarostat(aa.pressure, aa.temperature, 25)
        )

        aa.simulation = app.Simulation(
            aa.topology,
            aa.system,
            aa.integrator,
            TestCarboxylicAcid.platform,
        )
        aa.simulation.context.setPositions(aa.positions)
        aa.context = aa.simulation.context
        aa.perturbations_per_trial = 1000
        aa.propagations_per_step = 1

        return aa

    @staticmethod
    def setup_amino_acid_implicit(three_letter_code):
        """
        Set up glutamic acid in water
        """
        if three_letter_code not in ["glh", "ash"]:
            raise ValueError("Amino acid not available.")

        aa = SystemSetup()
        aa.temperature = 300.0 * unit.kelvin
        aa.timestep = 1.0 * unit.femtoseconds
        aa.collision_rate = 1.0 / unit.picoseconds
        aa.pH = 7.0
        testsystems = get_test_data("amino_acid", "testsystems")
        aa.ffxml_files = "amber10-constph.xml"
        aa.forcefield = ForceField(aa.ffxml_files, "amber10-constph-obc2.xml")

        aa.pdbfile = app.PDBFile(os.path.join(testsystems, "{}_vacuum.pdb".format(three_letter_code))
                                 )
        aa.topology = aa.pdbfile.topology
        aa.positions = aa.pdbfile.getPositions(asNumpy=True)
        aa.constraint_tolerance = 1.e-7

        aa.integrator = create_compound_gbaoab_integrator(aa)

        aa.integrator.setConstraintTolerance(aa.constraint_tolerance)
        aa.system = aa.forcefield.createSystem(
            aa.topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=True,
        )

        aa.simulation = app.Simulation(
            aa.topology,
            aa.system,
            aa.integrator,
            TestCarboxylicAcid.platform,
        )
        aa.simulation.context.setPositions(aa.positions)
        aa.context = aa.simulation.context
        aa.perturbations_per_trial = 1000
        aa.propagations_per_step = 1

        return aa

    def test_dummy_moving(self) -> None:
        """Move dummies without accepting and evaluate the energy differences."""

        viologen = self.setup_viologen_vacuum()

        cooh1 = COOHDummyMover.from_system(viologen.system, viologen.cooh1)
        cooh2 = COOHDummyMover.from_system(viologen.system, viologen.cooh2)

        for iteration in range(50):
            viologen.simulation.step(10)
            new_pos, logp = cooh1.mirror_oxygens(viologen.positions)
            if not logp == pytest.approx(0.0, abs=0.1):
                raise ValueError("Proposal should be very favorable.")

        for iteration in range(50):
            viologen.simulation.step(10)
            new_pos, logp = cooh1.mirror_syn_anti(viologen.positions)
            if not logp == pytest.approx(0.0, abs=1e-12):
                raise ValueError("Proposal should be 100% accepted.")

        for iteration in range(50):
            viologen.simulation.step(10)
            new_pos, logp = cooh2.mirror_oxygens(viologen.positions)
            if not logp == pytest.approx(0.0, abs=0.1):
                raise ValueError("Proposal should be very favorable.")

        for iteration in range(50):
            viologen.simulation.step(10)
            new_pos, logp = cooh2.mirror_syn_anti(viologen.positions)
            if not logp == pytest.approx(0.0, abs=1e-12):
                raise ValueError("Proposal should be 100% accepted.")

        return

    def test_dummy_serialization(self) -> None:
        """Move dummies without accepting and evaluate the energy differences."""

        viologen = self.setup_viologen_vacuum()

        cooh1 = COOHDummyMover.from_system(viologen.system, viologen.cooh1)
        cooh2 = COOHDummyMover.from_system(viologen.system, viologen.cooh2)

        cooh1_xml = cooh1.to_xml()
        cooh2_xml = cooh2.to_xml()

        cooh1_tree = etree.fromstring(cooh1_xml)
        cooh2_tree = etree.fromstring(cooh2_xml)

        cooh3 = COOHDummyMover.from_xml(cooh1_tree)
        cooh4 = COOHDummyMover.from_xml(cooh2_tree)

        for atom in ["HO", "OH", "CO", "OC", "R"]:
            assert getattr(cooh1, atom) ==  getattr(cooh3, atom), 'Atom {} does not match.'.format(atom)

        for atom in ["HO", "OH", "CO", "OC", "R"]:
            assert getattr(cooh2, atom) ==  getattr(cooh4, atom), 'Atom {} does not match.'.format(atom)

        for angle1, angle3 in zip(cooh1.angles, cooh3.angles):
            assert angle1 == angle3, "Angles do not match."

        for angle2, angle4 in zip(cooh2.angles, cooh4.angles):
            assert angle2 == angle4, "Angles do not match."

        for dihedral1, dihedral3 in zip(cooh1.dihedrals, cooh3.dihedrals):
            assert dihedral1 == dihedral3, "Dihedrals do not match."

        for dihedral2, dihedral4 in zip(cooh2.dihedrals, cooh4.dihedrals):
            assert dihedral2 == dihedral4, "Dihedrals do not match."

        for iteration in range(50):
            viologen.simulation.step(10)
            new_pos, logp = cooh3.mirror_oxygens(viologen.positions)
            if not logp == pytest.approx(0.0, abs=0.1):
                raise ValueError("Proposal should be very favorable.")

        for iteration in range(50):
            viologen.simulation.step(10)
            new_pos, logp = cooh3.mirror_syn_anti(viologen.positions)
            if not logp == pytest.approx(0.0, abs=1e-12):
                raise ValueError("Proposal should be 100% accepted.")

        for iteration in range(50):
            viologen.simulation.step(10)
            new_pos, logp = cooh4.mirror_oxygens(viologen.positions)
            if not logp == pytest.approx(0.0, abs=0.1):
                raise ValueError("Proposal should be very favorable.")

        for iteration in range(50):
            viologen.simulation.step(10)
            new_pos, logp = cooh4.mirror_syn_anti(viologen.positions)
            if not logp == pytest.approx(0.0, abs=1e-12):
                raise ValueError("Proposal should be 100% accepted.")

        return

    def test_dummy_moving_mc(self) -> None:
        """Move dummies with monte carlo and evaluate the energy differences."""

        md_steps_between_mc = 100
        total_loops = 100
        # viologen = self.setup_viologen_vacuum()
        viologen = self.setup_viologen_water()

        if log.getEffectiveLevel() == logging.DEBUG:
            viologen.simulation.reporters.append(
                app.DCDReporter(
                    "cooh-viologen-{}.dcd".format(str(uuid.uuid4())),
                    md_steps_between_mc // 10,
                )
            )

        cooh1 = COOHDummyMover.from_system(viologen.system, viologen.cooh1)
        cooh2 = COOHDummyMover.from_system(viologen.system, viologen.cooh2)

        do_nothing = lambda positions: (positions, 0.0)

        moveset = {
            do_nothing,
            cooh1.mirror_syn_anti,
            cooh1.mirror_oxygens,
            cooh2.mirror_syn_anti,
            cooh2.mirror_oxygens,
        }
        n_accept = 0
        for iteration in range(total_loops):
            viologen.simulation.step(md_steps_between_mc)

            state = viologen.context.getState(getPositions=True, getVelocities=True)
            pos = state.getPositions(asNumpy=True)

            # perform a move.
            move = sample(moveset, 1)[0]
            log.debug(move.__name__)
            new_pos, logp = move(pos)
            if exp(logp) > uniform(0.0, 1.0):
                log.debug("Accepted: logp %f", logp)
                viologen.context.setPositions(new_pos)
                n_accept += 1
            else:
                log.debug("Rejected: logp %f", logp)
                # Resample velocities if rejected to maintain detailed balance
                viologen.context.setVelocitiesToTemperature(viologen.temperature)

        if not logp == pytest.approx(0.0, abs=0.1):
            raise ValueError("Proposal should be very favorable, log p was {}.".format(logp))

        acceptance_rate = n_accept / (iteration + 1)
        log.info("Acceptance rate was %f", acceptance_rate)
        if acceptance_rate < 0.9:
            raise ValueError("Acceptance rate was lower than expected.")
        return

    def test_dummy_moving_protondrive(self) -> None:
        """Move dummies within a proton drive."""

        md_steps_between_mc = 10
        total_loops = 25
        viologen = self.setup_viologen_water()

        drive = ForceFieldProtonDrive(
            viologen.temperature,
            viologen.topology,
            viologen.system,
            viologen.forcefield,
            viologen.ffxml_files,
            pressure=viologen.pressure,
            perturbations_per_trial=viologen.perturbations_per_trial,
            propagations_per_step=viologen.propagations_per_step,
            residues_by_name=None,
            residues_by_index=None,
        )

        assert len(drive.titrationGroups[0].titration_states[0]._mc_moves["COOH"]) == 2, "The first state should have two movers"
        assert len(drive.titrationGroups[0].titration_states[1]._mc_moves["COOH"]) == 1, "The second state should have one mover"
        assert len(drive.titrationGroups[0].titration_states[2]._mc_moves["COOH"]) == 1, "The third state should have one mover"
        # There should be no COOH in this state
        with pytest.raises(KeyError):
            assert len(drive.titrationGroups[0].titration_states[3]._mc_moves["COOH"])
        drive.attach_context(viologen.context)
        drive.adjust_to_ph(7.0)
        for iteration in range(50):
            viologen.integrator.step(100)
            drive.update("COOH", nattempts=1)

        return

    def test_dummy_moving_protondrive_serialization(self) -> None:
        """Move dummies using a deserialized proton drive."""

        viologen = self.setup_viologen_water()

        drive = ForceFieldProtonDrive(
            viologen.temperature,
            viologen.topology,
            viologen.system,
            viologen.forcefield,
            viologen.ffxml_files,
            pressure=viologen.pressure,
            perturbations_per_trial=viologen.perturbations_per_trial,
            propagations_per_step=viologen.propagations_per_step,
            residues_by_name=None,
            residues_by_index=None,
        )


        drive.attach_context(viologen.context)
        drive.adjust_to_ph(7.0)
        for iteration in range(1):
            viologen.integrator.step(100)
            drive.update("COOH", nattempts=1)


        x = drive.serialize_titration_groups()
        newdrive = NCMCProtonDrive(viologen.temperature, viologen.topology, viologen.system,
                                   pressure=viologen.pressure, perturbations_per_trial=viologen.perturbations_per_trial)
        newdrive.add_residues_from_serialized_xml(etree.fromstring(x))

        for old_res, new_res in zip(drive.titrationGroups, newdrive.titrationGroups):
            assert old_res == new_res, "Residues don't match. {} :: {}".format(old_res.name, new_res.name)

        assert len(newdrive.titrationGroups[0].titration_states[0]._mc_moves[
                       "COOH"]) == 2, "The first state should have two movers"
        assert len(newdrive.titrationGroups[0].titration_states[1]._mc_moves[
                       "COOH"]) == 1, "The second state should have one mover"
        assert len(newdrive.titrationGroups[0].titration_states[2]._mc_moves[
                       "COOH"]) == 1, "The third state should have one mover"
        # There should be no COOH in this state
        with pytest.raises(KeyError):
            assert len(newdrive.titrationGroups[0].titration_states[3]._mc_moves["COOH"])

        newdrive.attach_context(viologen.context)
        for iteration in range(50):
            viologen.integrator.step(100)
            newdrive.update("COOH", nattempts=1)
        newdrive.update(UniformProposal(), nattempts=1)

        return

    def test_residue_patch(self):
        """Add COOH statements and fix COOH atom types for small molecules."""
        viologen = self.setup_viologen_vacuum()
        source = viologen.ffxml_files
        output = patch_cooh(source, "VIO")
        tree = etree.fromstring(output)
        assert len([cooh for cooh in tree.xpath('//COOH')]) == 4, "There should be a total of 4 COOH statements for viologen."
        atoms = [atom.get('type') for atom in tree.xpath('//Atom') ]
        assert atoms.count('oh') == 20, "There should be 20 atoms with oh types total (4 per state, and 4 in the main template)."

    def test_glutamic_acid_cooh(self):
        """Use the dummy mover with the amino acid glutamic acid"""

        md_steps_between_mc = 1000
        total_loops = 50

        glh = self.setup_amino_acid_water("glh")
        drive = ForceFieldProtonDrive(
            glh.temperature,
            glh.topology,
            glh.system,
            glh.forcefield,
            glh.ffxml_files,
            pressure=glh.pressure,
            perturbations_per_trial=glh.perturbations_per_trial,
            propagations_per_step=glh.propagations_per_step,
            residues_by_name=None,
            residues_by_index=None,
        )

        if log.getEffectiveLevel() == logging.DEBUG:
            glh.simulation.reporters.append(
                app.DCDReporter(
                    "cooh-glh-{}.dcd".format(str(uuid.uuid4())),
                    md_steps_between_mc // 100,
                    enforcePeriodicBox=True
                )
            )

        drive.attach_context(glh.context)
        drive.adjust_to_ph(4.6)

        for iteration in range(total_loops):
            glh.simulation.step(md_steps_between_mc)
            drive.update("COOH", nattempts=1)

        return

