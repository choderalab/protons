"""Tests specific to ProtonDrive objects."""
from __future__ import print_function

import os
from collections import Counter
from copy import deepcopy

import numpy as np
import pytest
from lxml import etree
from numpy.random import choice
from protons import app
from protons.app import (AmberProtonDrive, ForceField, ForceFieldProtonDrive,
                         GBAOABIntegrator, NCMCProtonDrive,
                         SelfAdjustedMixtureSampling, UniformProposal)
from protons.app.proposals import OneDirectionChargeProposal
from saltswap.swapper import Swapper
from saltswap.wrappers import Salinator
from simtk import openmm, unit

from . import get_test_data
from .utilities import SystemSetup, create_compound_gbaoab_integrator, hasCUDA


class TestVacuumSystem(object):
    """Tests for imidazole in vacuum"""

    default_platform = 'CPU'

    def test_imidazole_instantaneous(self):
        """
        
        """
        pdb = app.PDBFile(get_test_data('imidazole.pdb', 'testsystems/imidazole_explicit'))
        imidazole_xml = get_test_data('protons-imidazole-ph-feature.xml', 'testsystems/imidazole_explicit')
        forcefield = app.ForceField('gaff.xml', imidazole_xml)

        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

        temperature = 300 * unit.kelvin
        integrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds,
                                      timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=False)
        ncmcintegrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds,
                                          timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7,
                                          external_work=True)

        compound_integrator = openmm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmcintegrator)

        driver = ForceFieldProtonDrive(temperature, pdb.topology, system, forcefield, [imidazole_xml],
                                       pressure=None,
                                       perturbations_per_trial=0)

        platform = openmm.Platform.getPlatformByName(self.default_platform)
        context = openmm.Context(system, compound_integrator, platform)
        context.setPositions(pdb.positions)  # set to minimized positions
        context.setVelocitiesToTemperature(temperature)
        driver.attach_context(context)
        driver.adjust_to_ph(7.4)

        for i in range(42):
            compound_integrator.step(10)  # MD
            driver.update(UniformProposal())  # protonation

        drive_str = driver.to_xml()
        new_drv = NCMCProtonDrive.from_xml(drive_str, system, pdb.topology)

        # TODO come up with a comparison method.
        
        assert new_drv.titrationGroups == driver.titrationGroups, "Titration groups between drives are not equal."
