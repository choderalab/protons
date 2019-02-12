"""Tests for self adjusted mixture sampling."""

from protons import app
from lxml import etree
from simtk import unit
from simtk.openmm import openmm as mm
from . import get_test_data
from uuid import uuid4
import os
import pytest
import numpy as np
from protons.app import AmberProtonDrive, ForceFieldProtonDrive, NCMCProtonDrive
from protons.app import ForceField
from protons.app import UniformProposal
from protons.app.calibration import MultiSiteSAMSSampler
from protons.app.driver import SAMSApproach, Stage, UpdateRule
from protons.app import log
import logging

from protons.tests.utilities import (
    SystemSetup,
    create_compound_gbaoab_integrator,
    hasCUDA,
)

log.setLevel(logging.INFO)

class TestSAMS:
    """Test the functionality of the ``app.calibration.WeighsTable`` class."""

    if hasCUDA:
        default_platform_name = "CUDA"
    else:
        default_platform_name = "CPU"

    platform = mm.Platform.getPlatformByName(default_platform_name)


    @staticmethod
    def setup_peptide_implicit(name:str,minimize=True, createsim=True):
        """
        Set up implicit solvent peptide

        name - name of the peptide file. The folder "name_implicit" needs to exist,
         and "name".pdb needs to exist in the folder.
        minimize = Minimize the system before running (recommended)
            Works only if simulation is created
        createsim - instantiate simulation class
            If False, minimization is not performed.

        """
        peptide = SystemSetup()
        peptide.temperature = 300.0 * unit.kelvin
        # hahaha.pressure = 1.0 * unit.atmospheres
        peptide.timestep = 2.0 * unit.femtoseconds
        peptide.collision_rate = 1.0 / unit.picoseconds
        peptide.pH = 7.0
        peptide.perturbations_per_trial = 0 # instantaneous monte carlo
        peptide.propagations_per_step = 1

        testsystems = get_test_data("{}_implicit".format(name), "testsystems")
        peptide.ffxml_files = os.path.join("amber10-constph.xml")
        peptide.forcefield = ForceField(peptide.ffxml_files, "amber10-constph-obc2.xml")

        peptide.pdbfile = app.PDBFile(os.path.join(testsystems, "{}.pdb".format(name)))
        peptide.topology = peptide.pdbfile.topology
        peptide.positions = peptide.pdbfile.getPositions(asNumpy=True)

        # Quick fix for histidines in topology
        # Openmm relabels them HIS, which leads to them not being detected as
        # titratable. Renaming them fixes this.
        for residue in peptide.topology.residues():
            if residue.name == 'HIS':
                residue.name = 'HIP'

        peptide.constraint_tolerance = 1.e-7

        peptide.integrator = create_compound_gbaoab_integrator(peptide)

        peptide.system = peptide.forcefield.createSystem(
            peptide.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
        )

        peptide.drive = ForceFieldProtonDrive(
            peptide.temperature,
            peptide.topology,
            peptide.system,
            peptide.forcefield,
            peptide.ffxml_files,
            perturbations_per_trial=peptide.perturbations_per_trial,
            propagations_per_step=peptide.propagations_per_step,
            residues_by_name=None,
            residues_by_index=None,
        )

        if createsim:
            peptide.simulation = app.ConstantPHSimulation(
                peptide.topology,
                peptide.system,
                peptide.integrator,
                peptide.drive,
                platform=TestSAMS.platform,
            )
            peptide.simulation.context.setPositions(peptide.positions)
            peptide.context = peptide.simulation.context
            if minimize:
                peptide.simulation.minimizeEnergy()

        return peptide

    def test_make_onesite_table(self):
        """Test the creation of a weights table for one site sams."""

        pep = self.setup_peptide_implicit("yeah")
        pep.drive.adjust_to_ph(7.4) # ensure nonequal target weights
        for group_index in range(len(pep.drive.titrationGroups)):
            pep.drive.enable_calibration(SAMSApproach.ONESITE, group_index=group_index)
            table = pep.drive.calibration_state

            # The one site table should have exactly one entry for
            assert len(table) == 7, "The number of g_k values does not equal the number of available (independent) states."
            assert table.free_energy(pep.drive.titrationStates) == pep.drive.titrationGroups[group_index].state.g_k, "The weight should be the weight of the current state."
            assert np.all(table.targets == pep.drive.titrationGroups[group_index].target_weights)
            assert np.all(table.free_energies == pep.drive.titrationGroups[group_index].g_k_values)

        # Test that using None treats the last residue.
        group_index = None
        pep.drive.enable_calibration(SAMSApproach.ONESITE, group_index=group_index)
        table = pep.drive.calibration_state

        assert len(table) == 7, "The number of g_k values does not equal the number of available (independent) states."
        assert table.free_energy(pep.drive.titrationStates) == pep.drive.titrationGroups[
            -1].state.g_k, "The weight should be the weight of the current state."
        assert np.all(table.targets == pep.drive.titrationGroups[-1].target_weights)
        assert np.all(table.free_energies == pep.drive.titrationGroups[-1].g_k_values)

        xml = etree.tostring(table.to_xml()).decode()
        assert xml.count("<Residue") == 3, "The number of states in the XML output is wrong."
        assert xml.count("<State") == 7, "The number of states in the XML output is wrong."

        new_table = app.driver._SAMSState.from_xml(etree.fromstring(xml))
        return

    def test_make_multisite_table(self):
        """Test the creation of a weights table for multisite sams."""

        pep = self.setup_peptide_implicit("yeah")
        pep.drive.adjust_to_ph(7.4) # ensure nonequal target weights for tests

        pep.drive.enable_calibration(SAMSApproach.MULTISITE)
        table = pep.drive.calibration_state
        # The multi site table should have exactly one entry for
        assert len(table) == 12, "The number of g_k values does not equal the product of the number of available (independent) states."
        assert table.free_energy(pep.drive.titrationStates) ==  pytest.approx(pep.drive.sum_of_gk()), "The weight should be the same as the initial weight at this stage."
        assert np.all(table.targets == 1./12.), "Targets should be 1/num of states."

        xml = etree.tostring(table.to_xml()).decode()
        assert xml.count("<Dimension") == 3, "The number of dimensions in the XML output is wrong."
        assert xml.count("<State") == 12, "The number of states in the XML output is wrong."

        new_table = app.driver._SAMSState.from_xml(etree.fromstring(xml))
        return


    def test_onesite_sams_sampling_binary(self):
        """Test the one site sams sampling approach with binary updates"""
        old_log_level = log.getEffectiveLevel()

        pep = self.setup_peptide_implicit("yeah")
        pep.drive.enable_calibration(SAMSApproach.ONESITE, group_index=1)
        sampler = MultiSiteSAMSSampler(pep.drive)
        total_iterations = 1500
        for x in range(total_iterations):
            if x == total_iterations - 1:
                log.setLevel(logging.DEBUG)
            pep.simulation.step(1)
            pep.simulation.update(1)

            sampler.adapt_zetas(UpdateRule.BINARY, b=0.51, stage=Stage.SLOWDECAY)
        log.setLevel(old_log_level)

    def test_onesite_sams_sampling_global(self):
        """Test the one site sams sampling approach with global updates"""
        old_log_level = log.getEffectiveLevel()

        pep = self.setup_peptide_implicit("yeah")
        pep.drive.enable_calibration(SAMSApproach.ONESITE, group_index=1)
        sampler = MultiSiteSAMSSampler(pep.drive)
        total_iterations = 1500
        for x in range(total_iterations):
            if x == total_iterations - 1:
                log.setLevel(logging.DEBUG)
            pep.simulation.step(1)
            pep.simulation.update(1)
            sampler.adapt_zetas(UpdateRule.GLOBAL, b=0.51, stage=Stage.SLOWDECAY)
        log.setLevel(old_log_level)


    def test_multisite_sams_sampling(self):
        """Test the multisite SAMS sampling approach."""
        old_log_level = log.getEffectiveLevel()
        pep = self.setup_peptide_implicit("yeah")
        pep.drive.enable_calibration(SAMSApproach.MULTISITE)
        sampler = MultiSiteSAMSSampler(pep.drive)
        total_iterations = 1500
        for x in range(total_iterations):
            if x == total_iterations -1:
                log.setLevel(logging.DEBUG)
            pep.simulation.step(1)
            pep.simulation.update(1)
            sampler.adapt_zetas(UpdateRule.BINARY, b=0.51, stage=Stage.SLOWDECAY)
        log.setLevel(old_log_level)
        return

    def test_calibration_class_multisite(self):
        """Test the multisite SAMS sampling approach."""
        old_log_level = log.getEffectiveLevel()
        pep = self.setup_peptide_implicit("yeah", createsim=True)
        pep.drive.enable_calibration(SAMSApproach.MULTISITE)
        sampler = MultiSiteSAMSSampler(pep.drive)
        total_iterations = 1500
        for x in range(total_iterations):
            if x == total_iterations -1:
                log.setLevel(logging.DEBUG)
            pep.simulation.step(1)
            pep.simulation.update(1)
            sampler.adapt_zetas(UpdateRule.BINARY, b=0.51, stage=Stage.SLOWDECAY)
        log.setLevel(old_log_level)
        return