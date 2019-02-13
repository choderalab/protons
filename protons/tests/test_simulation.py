# coding=utf-8
"""
Test functionality of app.Simulation analogues.
"""
from protons import app, GBAOABIntegrator, ForceFieldProtonDrive

from simtk import unit
from simtk.openmm import openmm as mm
from . import get_test_data
import netCDF4
from protons.app import MetadataReporter, TitrationReporter, NCMCReporter, SAMSReporter
from protons.app.driver import SAMSApproach, NCMCProtonDrive
from uuid import uuid4
from lxml import etree
import os


class TestConstantPHSimulation(object):
    """Tests use cases for ConstantPHSimulation"""

    _default_platform = mm.Platform.getPlatformByName("Reference")

    def test_create_constantphsimulation_with_reporters(self):
        """Instantiate a ConstantPHSimulation at 300K/1 atm for a small peptide, with reporters."""

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
            perturbations_per_trial=1,
        )

        simulation = app.ConstantPHSimulation(
            pdb.topology,
            system,
            compound_integrator,
            driver,
            platform=self._default_platform,
        )
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        # Outputfile for reporters
        newname = uuid4().hex + ".nc"
        ncfile = netCDF4.Dataset(newname, "w")
        tr = TitrationReporter(ncfile, 1)
        mr = MetadataReporter(ncfile)
        nr = NCMCReporter(ncfile, 1)
        simulation.update_reporters.append(tr)
        simulation.update_reporters.append(mr)
        simulation.update_reporters.append(nr)

        # Regular MD step
        simulation.step(1)
        # Update the titration states using the uniform proposal
        simulation.update(1)

    def test_create_simulation(self):
        """Instantiate a ConstantPHSimulation at 300K/1 atm for a small peptide."""

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
            perturbations_per_trial=0,
        )

        simulation = app.ConstantPHSimulation(
            pdb.topology,
            system,
            compound_integrator,
            driver,
            platform=self._default_platform,
        )
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)

        # Regular MD step
        simulation.step(1)
        # Update the titration states using the uniform proposal
        simulation.update(1)
        print("Done!")


class TestConstantPHFreeEnergyCalculation:
    """Tests the functionality of the ConstantpHSimulation class when using SAMS class."""

    _default_platform = mm.Platform.getPlatformByName("Reference")

    def test_create_constantphcalibration(self):
        """Test running a calibration using constant-pH."""
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
            perturbations_per_trial=0,
        )

        # prep the driver for calibration
        driver.enable_calibration(SAMSApproach.ONESITE, group_index=-1)
        simulation = app.ConstantPHSimulation(
            pdb.topology,
            system,
            compound_integrator,
            driver,
            platform=self._default_platform,
        )
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(1)
        # Update the titration states using the uniform proposal
        simulation.update(1)
        # Adapt the weights using binary update.
        simulation.adapt()

    def test_create_constantphcalibration_resume(self):
        """Test running a calibration using constant-pH."""
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
            perturbations_per_trial=0,
        )
        driver.enable_calibration(SAMSApproach.ONESITE, group_index=-1)

        simulation = app.ConstantPHSimulation(
            pdb.topology,
            system,
            compound_integrator,
            driver,
            platform=self._default_platform,
        )
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        simulation.step(1)
        # Update the titration states using the uniform proposal
        simulation.update(1)
        # Adapt the weights using binary update, a few times just to rack up the counters.
        for x in range(5):
            simulation.adapt()
        # retrieve the samsProperties
        # TODO use deserialize proton drive for this
        xml = simulation.drive.state_to_xml()

        driver2 = NCMCProtonDrive(
            temperature,
            pdb.topology,
            system,
            pressure=pressure,
            perturbations_per_trial=0,
        )
        driver2.state_from_xml_tree(etree.fromstring(xml))
        integrator2 = GBAOABIntegrator(
            temperature=temperature,
            collision_rate=1.0 / unit.picoseconds,
            timestep=2.0 * unit.femtoseconds,
            constraint_tolerance=1.0e-7,
            external_work=False,
        )
        ncmcintegrator2 = GBAOABIntegrator(
            temperature=temperature,
            collision_rate=1.0 / unit.picoseconds,
            timestep=2.0 * unit.femtoseconds,
            constraint_tolerance=1.0e-7,
            external_work=True,
        )
        compound_integrator2 = mm.CompoundIntegrator()
        compound_integrator2.addIntegrator(integrator2)
        compound_integrator2.addIntegrator(ncmcintegrator2)

        # Make a new calibration and do another step, ignore the state for this example
        simulation2 = app.ConstantPHSimulation(
            pdb.topology,
            system,
            compound_integrator2,
            driver2,
            platform=self._default_platform,
        )

        assert (
            simulation2.drive.calibration_state._stage
            is simulation.drive.calibration_state._stage
        )
        # get going then
        simulation2.context.setPositions(pdb.positions)
        simulation2.context.setVelocitiesToTemperature(temperature)
        simulation2.step(1)
        # Update the titration states using the uniform proposal
        simulation2.update(1)
        # Adapt the weights using binary update.
        simulation2.adapt()

        assert (
            simulation2.drive.calibration_state._current_adaptation
            == simulation.drive.calibration_state.min_burn + 6
        ), "The resumed calibration does not have the right adaptation_index"

    def test_create_constantphcalibration_with_reporters(self):
        """Test running a calibration using constant-pH with reporters."""
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
            perturbations_per_trial=2,
            propagations_per_step=1,
        )

        # prep the driver for calibration
        driver.enable_calibration(SAMSApproach.ONESITE, group_index=-1)
        simulation = app.ConstantPHSimulation(
            pdb.topology,
            system,
            compound_integrator,
            driver,
            platform=self._default_platform,
        )
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        # Outputfile for reporters
        newname = uuid4().hex + ".nc"
        ncfile = netCDF4.Dataset(newname, "w")
        tr = TitrationReporter(ncfile, 1)
        mr = MetadataReporter(ncfile)
        nr = NCMCReporter(ncfile, 1)
        sr = SAMSReporter(ncfile, 1)
        simulation.update_reporters.append(tr)
        simulation.update_reporters.append(mr)
        simulation.update_reporters.append(nr)
        simulation.calibration_reporters.append(sr)

        simulation.step(1)
        # Update the titration states using the uniform proposal
        simulation.update(1)
        # Adapt the weights using binary update.
        simulation.adapt()
        # Attempt to prevent segfaults by closing files
        ncfile.close()
