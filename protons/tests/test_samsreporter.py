# coding=utf-8
"""Testing of the SAMSReporter class"""

from protons import app
from protons.app import samsreporter as sr
from protons.app.driver import SAMSApproach, Stage, UpdateRule
from simtk import unit, openmm as mm
from protons.app import GBAOABIntegrator, ForceFieldProtonDrive
from . import get_test_data
import uuid
import os
import pytest
from protons.tests.utilities import hasCUDA

travis = os.environ.get("TRAVIS", None)


@pytest.mark.skipif(travis == "true", reason="Travis segfaulting risk.")
class TestSAMSReporter(object):
    """Tests use cases for ConstantPHCalibration"""

    # _default_platform = mm.Platform.getPlatformByName('Reference')
    if hasCUDA:
        default_platform_name = "CUDA"
    else:
        default_platform_name = "CPU"

    _default_platform = mm.Platform.getPlatformByName(default_platform_name)

    def test_reports(self):
        """Instantiate a simulation at 300K/1 atm for a small peptide with reporter."""

        pdb = app.PDBxFile(
            get_test_data(
                "glu_ala_his-solvated-minimized-renamed.cif", "testsystems/tripeptides"
            )
        )
        forcefield = app.ForceField(
            "amber10-constph.xml", "ions_tip3p.xml", "tip3p.xml"
        )

        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )

        temperature = 300 * unit.kelvin
        integrator = GBAOABIntegrator(
            temperature=temperature,
            collision_rate=1.0 / unit.picoseconds,
            timestep=2.0 * unit.femtoseconds,
            constraint_tolerance=1.0e-7,
            external_work=False,
        )
        ncmcintegrator = GBAOABIntegrator(
            temperature=temperature,
            collision_rate=1.0 / unit.picoseconds,
            timestep=2.0 * unit.femtoseconds,
            constraint_tolerance=1.0e-7,
            external_work=True,
        )

        compound_integrator = mm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmcintegrator)
        pressure = 1.0 * unit.atmosphere

        system.addForce(mm.MonteCarloBarostat(pressure, temperature))
        driver = ForceFieldProtonDrive(
            temperature,
            pdb.topology,
            system,
            forcefield,
            ["amber10-constph.xml"],
            pressure=pressure,
            perturbations_per_trial=3,
        )

        # prep the driver for calibration
        driver.enable_calibration(
            SAMSApproach.ONESITE, group_index=1, update_rule=UpdateRule.BINARY
        )

        calibration = app.ConstantPHSimulation(
            pdb.topology,
            system,
            compound_integrator,
            driver,
            platform=self._default_platform,
        )
        calibration.context.setPositions(pdb.positions)
        calibration.context.setVelocitiesToTemperature(temperature)
        filename = uuid.uuid4().hex + ".nc"
        print("Temporary file: ", filename)
        newreporter = sr.SAMSReporter(filename, 2)
        calibration.calibration_reporters.append(newreporter)

        # Regular MD step
        for iteration in range(4):
            calibration.step(1)
            # Update the titration states using the uniform proposal
            calibration.update(1)
            # adapt sams weights
            calibration.adapt()

        # Basic checks for dimension
        assert (
            newreporter.ncfile["Protons/SAMS"].dimensions["adaptation"].size == 2
        ), "There should be 2 updates recorded."
        assert (
            newreporter.ncfile["Protons/SAMS"].dimensions["state"].size == 3
        ), "There should be 3 states reported."

        newreporter.ncfile.close()

    def test_reports_multisite(self):
        """Instantiate a simulation at 300K/1 atm for a small peptide with reporter."""

        pdb = app.PDBxFile(
            get_test_data(
                "glu_ala_his-solvated-minimized-renamed.cif", "testsystems/tripeptides"
            )
        )
        forcefield = app.ForceField(
            "amber10-constph.xml", "ions_tip3p.xml", "tip3p.xml"
        )

        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )

        temperature = 300 * unit.kelvin
        integrator = GBAOABIntegrator(
            temperature=temperature,
            collision_rate=1.0 / unit.picoseconds,
            timestep=2.0 * unit.femtoseconds,
            constraint_tolerance=1.0e-7,
            external_work=False,
        )
        ncmcintegrator = GBAOABIntegrator(
            temperature=temperature,
            collision_rate=1.0 / unit.picoseconds,
            timestep=2.0 * unit.femtoseconds,
            constraint_tolerance=1.0e-7,
            external_work=True,
        )

        compound_integrator = mm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmcintegrator)
        pressure = 1.0 * unit.atmosphere

        system.addForce(mm.MonteCarloBarostat(pressure, temperature))
        driver = ForceFieldProtonDrive(
            temperature,
            pdb.topology,
            system,
            forcefield,
            ["amber10-constph.xml"],
            pressure=pressure,
            perturbations_per_trial=3,
        )

        # prep the driver for calibration
        driver.enable_calibration(SAMSApproach.MULTISITE, update_rule=UpdateRule.BINARY)

        calibration = app.ConstantPHSimulation(
            pdb.topology,
            system,
            compound_integrator,
            driver,
            platform=self._default_platform,
        )
        calibration.context.setPositions(pdb.positions)
        calibration.minimizeEnergy()
        calibration.context.setVelocitiesToTemperature(temperature)

        filename = uuid.uuid4().hex + ".nc"
        print("Temporary file: ", filename)
        newreporter = sr.SAMSReporter(filename, 2)
        calibration.calibration_reporters.append(newreporter)

        # Regular MD step
        for iteration in range(4):
            calibration.step(1)
            # Update the titration states using the uniform proposal
            calibration.update(1)
            # adapt sams weights
            calibration.adapt()

        # Basic checks for dimension
        assert (
            newreporter.ncfile["Protons/SAMS"].dimensions["adaptation"].size == 2
        ), "There should be 2 updates recorded."
        assert (
            newreporter.ncfile["Protons/SAMS"].dimensions["state"].size == 15
        ), "There should be 15 states reported."

        newreporter.ncfile.close()

    def test_burn_in_sams(self):
        """
        Tests a case of a SAMS reporter that finished burn_in
        """
        pdb = app.PDBxFile(
            get_test_data(
                "glu_ala_his-solvated-minimized-renamed.cif", "testsystems/tripeptides"
            )
        )
        forcefield = app.ForceField(
            "amber10-constph.xml", "ions_tip3p.xml", "tip3p.xml"
        )

        system = forcefield.createSystem(
            pdb.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * unit.nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )

        temperature = 300 * unit.kelvin
        integrator = GBAOABIntegrator(
            temperature=temperature,
            collision_rate=1.0 / unit.picoseconds,
            timestep=2.0 * unit.femtoseconds,
            constraint_tolerance=1.0e-7,
            external_work=False,
        )
        ncmcintegrator = GBAOABIntegrator(
            temperature=temperature,
            collision_rate=1.0 / unit.picoseconds,
            timestep=2.0 * unit.femtoseconds,
            constraint_tolerance=1.0e-7,
            external_work=True,
        )

        compound_integrator = mm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmcintegrator)
        pressure = 1.0 * unit.atmosphere

        system.addForce(mm.MonteCarloBarostat(pressure, temperature))
        driver = ForceFieldProtonDrive(
            temperature,
            pdb.topology,
            system,
            forcefield,
            ["amber10-constph.xml"],
            pressure=pressure,
            perturbations_per_trial=3,
        )
        driver.enable_calibration(SAMSApproach.MULTISITE, update_rule=UpdateRule.BINARY)

        calibration = app.ConstantPHSimulation(
            pdb.topology,
            system,
            compound_integrator,
            driver,
            platform=self._default_platform,
        )
        calibration.context.setPositions(pdb.positions)
        calibration.minimizeEnergy()
        calibration.context.setVelocitiesToTemperature(temperature)
        filename = uuid.uuid4().hex + ".nc"
        print("Temporary file: ", filename)
        newreporter = sr.SAMSReporter(filename, 2)
        calibration.calibration_reporters.append(newreporter)
        driver.calibration_state._stage = Stage.FASTDECAY

        # Regular MD step
        for iteration in range(20):
            calibration.step(1)
            # Update the titration states using the uniform proposal
            calibration.update(1)
            # adapt sams weights
            calibration.adapt()

        # Basic checks for dimension
        assert (
            newreporter.ncfile["Protons/SAMS"].dimensions["adaptation"].size == 10
        ), "There should be 2 updates recorded."
        assert (
            newreporter.ncfile["Protons/SAMS"].dimensions["state"].size == 15
        ), "There should be 15 states reported."
        assert (
            newreporter.ncfile["Protons/SAMS/stage"][9] == 1
        ), "Stage should be 1 at this point"

        # close files to avoid segfaults, possibly
        newreporter.ncfile.close()
