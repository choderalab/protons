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
from protons.app import SelfAdjustedMixtureSampling
from protons.app import UniformProposal
from protons.app.proposals import OneDirectionChargeProposal, COOHDummyMover
from protons.tests import get_test_data
from protons.tests.utilities import SystemSetup, create_compound_gbaoab_integrator, hasCUDA
import logging
from random import choice, uniform, sample
from protons.app import log
from math import exp
log.setLevel(logging.INFO)

class TestCarboxylicAcid:
    default_platform_name = 'CPU'
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
        testsystems = get_test_data('viologen', 'testsystems')
        viologen.ffxml_filename = os.path.join(testsystems, 'viologen-protons.ffxml')
        viologen.gaff = os.path.join(testsystems, 'gaff.xml')
        viologen.forcefield = ForceField(viologen.gaff, viologen.ffxml_filename)

        viologen.pdbfile = app.PDBFile(
            os.path.join(testsystems, "viologen-vacuum.pdb"))
        viologen.topology = viologen.pdbfile.topology
        viologen.positions = viologen.pdbfile.getPositions(asNumpy=True)
        viologen.constraint_tolerance = 1.e-7

        viologen.integrator = openmm.LangevinIntegrator(viologen.temperature, viologen.collision_rate,
                                               viologen.timestep)

        viologen.integrator.setConstraintTolerance(viologen.constraint_tolerance)
        viologen.system = viologen.forcefield.createSystem(viologen.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        viologen.cooh1 = { # indices in topology of the first cooh group
            "HO" : 56,
            "OH" : 0,
            "CO": 1,
            "OC": 2,
            "R" : 3
        }
        viologen.cooh2 = { # indices in topology of the second cooh group
            "HO": 57,
            "OH": 27,
            "CO": 25,
            "OC": 26,
            "R": 24
        }

        viologen.simulation = app.Simulation(viologen.topology, viologen.system, viologen.integrator, TestCarboxylicAcid.platform)
        viologen.simulation.context.setPositions(viologen.positions)
        viologen.context = viologen.simulation.context


        return viologen

    def test_dummy_moving(self) -> None:
        """Move dummies without accepting and evaluate the energy differences."""

        viologen = self.setup_viologen_vacuum()

        cooh1 = COOHDummyMover(viologen.system, viologen.cooh1)
        cooh2 = COOHDummyMover(viologen.system, viologen.cooh2)

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


    def test_dummy_moving_mc(self) -> None:
        """Move dummies with monte carlo and evaluate the energy differences."""

        viologen = self.setup_viologen_vacuum()

        cooh1 = COOHDummyMover(viologen.system, viologen.cooh1)
        cooh2 = COOHDummyMover(viologen.system, viologen.cooh2)

        do_nothing = lambda positions: (positions, 0.0)

        moveset = {do_nothing, cooh1.mirror_syn_anti, cooh1.mirror_oxygens, cooh2.mirror_syn_anti, cooh2.mirror_oxygens}
        n_accept = 0
        for iteration in range(100):
            viologen.simulation.step(10)

            state = viologen.context.getState(getPositions=True, getVelocities=True)
            pos = state.getPositions(asNumpy=True)

            # perform a move.
            move = sample(moveset,1)[0]
            log.info(move.__name__)
            new_pos, logp = move(pos)
            if exp(logp) > uniform(0.0,1.0):
                log.debug("Accepted: logp %f", logp)
                viologen.context.setPositions(new_pos)
                n_accept +=1
            else:
                log.debug("Rejected: logp %f", logp)
                # Resample velocities if rejected to maintain detailed balance
                # PS: rejection unlikely
                viologen.context.setVelocitiesToTemperature(viologen.temperature)

        log.info("Acceptance rate was %f", n_accept / iteration )
        return
