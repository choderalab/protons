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


class TestCarboxylicAcid:
    default_platform = 'CPU'

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

        viologen.constraint_tolerance = 1.e-7

        integrator = openmm.LangevinIntegrator(viologen.temperature, viologen.collision_rate,
                                               viologen.timestep)

        integrator.setConstraintTolerance(viologen.constraint_tolerance)
        viologen.system = viologen.forcefield.createSystem(viologen.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

        return viologen

    def test_vacuum_dummy(self) -> None:

        viologen = self.setup_viologen_vacuum()

        return

