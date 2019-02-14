# coding=utf-8
"""
Tests the augmented app layer classes, including ForceField, Topology, Modeller.
"""

from protons import app
from simtk import unit, openmm
from . import get_test_data


class TestTopology:
    """Tests reading of topology using protons app layer"""

    def test_read_pdbx_file(self):
        """Read a pdbx/mmCif file using protons.app."""

        cif = app.PDBxFile(
            get_test_data(
                "glu_ala_his-solvated-minimized-renamed.cif", "testsystems/tripeptides/"
            )
        )

    def test_create_system(self):
        """Create a system using the amber10 constph force field."""
        cif = app.PDBxFile(
            get_test_data(
                "glu_ala_his-solvated-minimized-renamed.cif", "testsystems/tripeptides/"
            )
        )
        forcefield = app.ForceField(
            "amber10-constph.xml", "ions_tip3p.xml", "tip3p.xml"
        )
        system = forcefield.createSystem(
            cif.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )

    def test_modeller(self):
        """Test addition of hydrogens to a PDB file using modeller."""
        pdb = app.PDBFile(
            get_test_data("glu_ala_his_noH.pdb", "testsystems/tripeptides/")
        )
        forcefield = app.ForceField(
            "amber10-constph.xml", "ions_tip3p.xml", "tip3p.xml"
        )
        modeller = app.Modeller(pdb.topology, pdb.positions)
        modeller.addHydrogens(forcefield)
        # Should have 59 hydrogens after adding them
        assert modeller.topology.getNumAtoms() == 59

        system = forcefield.createSystem(modeller.topology)
