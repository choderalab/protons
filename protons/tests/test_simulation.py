# coding=utf-8
"""
Test functionality of app.Simulation analogues.
"""
from protons import app, GBAOABIntegrator, ForceFieldProtonDrive

from simtk import unit
from simtk.openmm import openmm as mm
from . import get_test_data

class TestConstantPHSimulation(object):
    """Tests use cases for ConstantPHSimulation"""

    _default_platform = mm.Platform.getPlatformByName('Reference')

    def test_create_constantphsimulation(self):
        """Instantiate a ConstantPHSimulation at 300K/1 atm for a small peptide."""

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

        # Regular MD step
        simulation.step(1)
        # Update the titration states using the uniform proposal
        simulation.update(1)
        print('Done!')


class TestConstantPHCalibration:
    """Tests the functionality of the ConstantPHCalibration class."""

    _default_platform =  mm.Platform.getPlatformByName('Reference')

    def test_create_constantphcalibration(self):
        """Test running a calibration using constant-pH."""
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
        simulation = app.ConstantPHCalibration(pdb.topology, system, compound_integrator, driver,group_index=1, platform=self._default_platform)
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(1)
        # Update the titration states using the uniform proposal
        simulation.update(1)
        # Adapt the weights using binary update.
        simulation.adapt()
