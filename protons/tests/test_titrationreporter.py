# coding=utf-8
"""Test functionality of the TitrationReporter."""
from protons import app
from protons.app import titrationreporter as tr
from simtk import unit, openmm as mm
from protons.app import GBAOABIntegrator, ForceFieldProtonDrive
from . import get_test_data
import uuid
import numpy as np
import netCDF4
import pytest

class TestTitrationReporter(object):
    """Tests use cases for ConstantPHSimulation"""

    _default_platform = mm.Platform.getPlatformByName('Reference')

    def test_reports(self):
        """Instantiate a ConstantPHSimulation at 300K/1 atm for a small peptide."""

        pdb = app.PDBxFile(get_test_data('glu_ala_his-solvated-minimized-renamed.cif', 'testsystems/tripeptides'))
        num_atoms = pdb.topology.getNumAtoms()
        forcefield = app.ForceField('amber10-constph.xml', 'ions_tip3p.xml', 'tip3p.xml')

        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.0 * unit.nanometers, constraints=app.HBonds, rigidWater=True,
                                         ewaldErrorTolerance=0.0005)

        temperature = 300 * unit.kelvin
        integrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds, timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=False)
        ncmcintegrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds, timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=True)

        compound_integrator = mm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmcintegrator)
        pressure = 1.0 * unit.atmosphere

        system.addForce(mm.MonteCarloBarostat(pressure, temperature))
        driver = ForceFieldProtonDrive(temperature, pdb.topology, system, forcefield, ['amber10-constph.xml'], pressure=pressure,
                                       perturbations_per_trial=0)

        num_titratable = len(driver.titrationGroups)
        simulation = app.ConstantPHSimulation(pdb.topology, system, compound_integrator, driver, platform=self._default_platform)
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        filename = uuid.uuid4().hex + ".nc"
        print("Temporary file: ",filename)
        newreporter = tr.TitrationReporter(filename, 2, shared=False)
        simulation.update_reporters.append(newreporter)

        # Regular MD step
        simulation.step(1)
        # Update the titration states using the uniform proposal
        simulation.update(6)
        # Basic checks for dimension
        assert newreporter.ncfile['Protons/Titration'].dimensions['update'].size == 3, "There should be 3 updates recorded."
        assert newreporter.ncfile['Protons/Titration'].dimensions['residue'].size == num_titratable, "There should be {} residues recorded.".format(num_titratable)
        assert newreporter.ncfile['Protons/Titration'].dimensions['atom'].size == num_atoms, "There should be {} atoms recorded.".format(num_atoms)

    def test_state_reporting(self):
        """Test if the titration state is correctly reported."""

        pdb = app.PDBxFile(get_test_data('glu_ala_his-solvated-minimized-renamed.cif', 'testsystems/tripeptides'))
        forcefield = app.ForceField('amber10-constph.xml', 'ions_tip3p.xml', 'tip3p.xml')

        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.0 * unit.nanometers, constraints=app.HBonds, rigidWater=True,
                                         ewaldErrorTolerance=0.0005)

        temperature = 300 * unit.kelvin
        integrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds, timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=False)
        ncmcintegrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds, timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=True)

        compound_integrator = mm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmcintegrator)
        pressure = 1.0 * unit.atmosphere

        system.addForce(mm.MonteCarloBarostat(pressure, temperature))
        driver = ForceFieldProtonDrive(temperature, pdb.topology, system, forcefield, ['amber10-constph.xml'], pressure=pressure,
                                       perturbations_per_trial=0)

        simulation = app.ConstantPHSimulation(pdb.topology, system, compound_integrator, driver, platform=self._default_platform)
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        filename = uuid.uuid4().hex + ".nc"
        ncfile = netCDF4.Dataset(filename, 'w')
        print("Temporary file: ",filename)
        newreporter = tr.TitrationReporter(ncfile, 2, shared=False)
        simulation.update_reporters.append(newreporter)

        # Regular MD step
        simulation.step(1)
        # Update the titration states forcibly from a pregenerated series
        glu_states = np.asarray([1, 4, 4, 1, 0, 2, 0, 2, 4, 3, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 1])
        his_states = np.asarray([2, 1, 0, 1, 2, 2, 1, 2, 2, 1, 0, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1])

        for i in range(len(glu_states)):
            simulation.drive._set_titration_state(0, glu_states[i], updateParameters=False)
            simulation.drive._set_titration_state(1, his_states[i], updateParameters=False)
            assert simulation.drive.titrationGroups[0].state_index == glu_states[i], "Glu state not set correctly."
            assert simulation.drive.titrationGroups[1].state_index == his_states[i], "His state not set correctly."
            simulation.update_reporters[0].report(simulation)

        recorded_glu_states = np.asarray(ncfile['Protons/Titration/state'][:,0])
        recorded_his_states = np.asarray(ncfile['Protons/Titration/state'][:,1])
        assert np.array_equal(glu_states, recorded_glu_states), "Glutamate states are not recorded correctly."
        assert np.array_equal(his_states, recorded_his_states), "Histidine states are not recorded correctly."

    def test_atom_status_reporting(self):
        """Test if the atom_status is correctly reported."""

        pdb = app.PDBxFile(get_test_data('glu_ala_his-solvated-minimized-renamed.cif', 'testsystems/tripeptides'))
        forcefield = app.ForceField('amber10-constph.xml', 'ions_tip3p.xml', 'tip3p.xml')

        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.0 * unit.nanometers, constraints=app.HBonds, rigidWater=True,
                                         ewaldErrorTolerance=0.0005)

        temperature = 300 * unit.kelvin
        integrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds, timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=False)
        ncmcintegrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds, timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=True)

        compound_integrator = mm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmcintegrator)
        pressure = 1.0 * unit.atmosphere

        system.addForce(mm.MonteCarloBarostat(pressure, temperature))
        driver = ForceFieldProtonDrive(temperature, pdb.topology, system, forcefield, ['amber10-constph.xml'], pressure=pressure,
                                       perturbations_per_trial=0)

        simulation = app.ConstantPHSimulation(pdb.topology, system, compound_integrator, driver, platform=self._default_platform)
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        filename = uuid.uuid4().hex + ".nc"
        ncfile = netCDF4.Dataset(filename, 'w')
        print("Temporary file: ",filename)
        newreporter = tr.TitrationReporter(ncfile, 2, shared=False)
        simulation.update_reporters.append(newreporter)

        # Regular MD step
        simulation.step(1)
        # Update the titration states forcibly from a pregenerated series
        glu_states = np.asarray([1, 4, 4, 1, 0, 2, 0, 2, 4, 3, 2, 1, 0, 2, 0, 0, 0, 0, 0, 0, 1, 2, 1, 3, 1])
        his_states = np.asarray([2, 1, 0, 1, 2, 2, 1, 2, 2, 1, 0, 2, 2, 1, 2, 2, 2, 2, 2, 2, 0, 2, 2, 2, 1])

        for i in range(len(glu_states)):
            simulation.drive._set_titration_state(0, glu_states[i], updateParameters=False)
            simulation.drive._set_titration_state(1, his_states[i], updateParameters=False)
            simulation.update_reporters[0].report(simulation)
            # check glu
            self._verify_atom_status(i, 0, ncfile, simulation)
            # check his
            self._verify_atom_status(i, 1, ncfile, simulation)

    def _verify_atom_status(self, iteration, group_index, ncfile, simulation):
        """Check the atom status for all atoms in one residue for a single iteration."""
        residue = simulation.drive.titrationGroups[group_index]
        atom_status = residue.atom_status
        atom_ids = residue.atom_indices
        for id, status in zip(atom_ids, atom_status):
            assert status == ncfile['Protons/Titration/atom_status'][iteration, id],\
                "Residue {} atom status recorded should match the current status.".format(group_index)
