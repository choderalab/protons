"""Tests for self adjusted mixture sampling."""


from protons import app
from simtk import unit
from simtk.openmm import openmm as mm
from . import get_test_data
from uuid import uuid4
import os
import pytest

from protons.app import AmberProtonDrive, ForceFieldProtonDrive, NCMCProtonDrive
from protons.app import ForceField
from protons.app import SelfAdjustedMixtureSampling
from protons.app import UniformProposal
from protons.app.calibration import WeightTable, SAMSApproach

from protons.tests.utilities import (
    SystemSetup,
    create_compound_gbaoab_integrator,
    hasCUDA,
)



class TestWeightTable:
    """Test the functionality of the ``app.calibration.WeighsTable`` class."""

    if hasCUDA:
        default_platform_name = "CUDA"
    else:
        default_platform_name = "CPU"

    platform = mm.Platform.getPlatformByName(default_platform_name)


    @staticmethod
    def setup_peptide_implicit(name:str):
        """
        Set up implicit solvent peptide

        Note
        ----
        folder "name_implicit" needs to exist, and "name".pdb needs to exist in the folder.

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
        peptide.drive.adjust_to_ph(7.4)
        peptide.simulation = app.ConstantPHSimulation(
            peptide.topology,
            peptide.system,
            peptide.integrator,
            peptide.drive,
            TestWeightTable.platform,
        )
        peptide.simulation.context.setPositions(peptide.positions)

        peptide.context = peptide.simulation.context

        return peptide

    def test_make_onesite_table(self):
        """Test the creation of a weights table for one site sams."""

        pep = self.setup_peptide_implicit("yeah")
        table = WeightTable(pep.drive, SAMSApproach.ONESITE)

        # The one site table should have exactly one entry for
        assert len(table) == 7, "The number of g_k values does not equal the number of available (independent) states."
        assert table.weight(pep.drive.titrationStates) == pep.drive.sum_of_gk(), "The weight should be the sum of independent weights."
        return

    def test_make_multisite_table(self):
        """Test the creation of a weights table for multisite sams."""

        pep = self.setup_peptide_implicit("yeah")
        table = WeightTable(pep.drive, SAMSApproach.MULTISITE)

        # The multi site table should have exactly one entry for
        assert len(table) == 12, "The number of g_k values does not equal the product of the number of available (independent) states."
        assert pytest.approx(table.weight(pep.drive.titrationStates), 0.0), "The weight should be 0 at this stage."
        return