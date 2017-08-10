# coding=utf-8
"""Testing of the SAMSReporter class"""

from protons import app
from protons.app import samsreporter as sr
from simtk import unit, openmm as mm
from protons.app import GBAOABIntegrator, ForceFieldProtonDrive
from . import get_test_data
import uuid
import os
import pytest

travis = os.environ.get("TRAVIS", None)


@pytest.mark.skipif(travis == 'true', reason="Travis segfaulting risk.")
class TestSAMSReporter(object):
    """Tests use cases for ConstantPHCalibration"""

    _default_platform = mm.Platform.getPlatformByName('Reference')

    def test_reports(self):
        """Instantiate a ConstantPHCalibration at 300K/1 atm for a small peptide."""

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
                                       perturbations_per_trial=3)

        calibration = app.ConstantPHCalibration(pdb.topology, system, compound_integrator, driver, group_index=1, platform=self._default_platform)
        calibration.context.setPositions(pdb.positions)
        calibration.context.setVelocitiesToTemperature(temperature)
        filename = uuid.uuid4().hex + ".nc"
        print("Temporary file: ",filename)
        newreporter = sr.SAMSReporter(filename, 2, shared=False)
        calibration.calibration_reporters.append(newreporter)

        # Regular MD step
        for iteration in range(4):
            calibration.step(1)
            # Update the titration states using the uniform proposal
            calibration.update(1)
            # adapt sams weights
            calibration.adapt()

        # Basic checks for dimension
        assert newreporter.ncfile['Protons/SAMS'].dimensions['adaptation'].size == 2, "There should be 2 updates recorded."
        assert newreporter.ncfile['Protons/SAMS'].dimensions['state'].size == 3, "There should be 3 states reported."

        newreporter.ncfile.close()

    def test_burn_in_sams(self):
        """
        Tests a case of a SAMS reporter that finished burn_in
        """
        pdb = app.PDBxFile(get_test_data('glu_ala_his-solvated-minimized-renamed.cif', 'testsystems/tripeptides'))
        forcefield = app.ForceField('amber10-constph.xml', 'ions_tip3p.xml', 'tip3p.xml')

        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.0 * unit.nanometers, constraints=app.HBonds, rigidWater=True,
                                         ewaldErrorTolerance=0.0005)

        temperature = 300 * unit.kelvin
        integrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds,
                                      timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=False)
        ncmcintegrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds,
                                          timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7,
                                          external_work=True)

        compound_integrator = mm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmcintegrator)
        pressure = 1.0 * unit.atmosphere

        system.addForce(mm.MonteCarloBarostat(pressure, temperature))
        driver = ForceFieldProtonDrive(temperature, pdb.topology, system, forcefield, ['amber10-constph.xml'],
                                       pressure=pressure,
                                       perturbations_per_trial=3)

        calibration = app.ConstantPHCalibration(pdb.topology, system, compound_integrator, driver, group_index=1,
                                                platform=self._default_platform)
        calibration.context.setPositions(pdb.positions)
        calibration.context.setVelocitiesToTemperature(temperature)
        filename = uuid.uuid4().hex + ".nc"
        print("Temporary file: ", filename)
        newreporter = sr.SAMSReporter(filename, 2, shared=False)
        calibration.calibration_reporters.append(newreporter)
        calibration.stage = "slow-gain"

        # Regular MD step
        for iteration in range(4):
            calibration.step(1)
            # Update the titration states using the uniform proposal
            calibration.update(1)
            # adapt sams weights
            calibration.adapt()

        # close files to avoid segfaults, possibly
        newreporter.ncfile.close()

