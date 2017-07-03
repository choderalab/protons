# coding=utf-8
"""
Test functionality of app.Simulation analogues.
"""
from simtk import unit
from simtk.openmm import openmm as mm

from protons import app, integrators, ForceFieldProtonDrive

from . import get_test_data


class TestConstantPHSimulation(object):
    """Tests use cases for ConstantPHSimulation"""

    _default_platform =  mm.Platform.getPlatformByName('Reference')

    def test_create_constantphsimulation(self):
        """Instantiate a ConstantPHSimulation at 300K/1 atm for a small peptide."""

        # Integrator and system settings
        timestep = 2.0 * unit.femtoseconds
        temperature = 300. * unit.kelvin
        collision_rate = 90. / unit.picosecond
        pressure = 1.0 * unit.atmospheres
        barostatInterval = 25
        constraint_tolerance = 1.e-7

        # Load pdb file with protons compatible residue names
        pdbx = app.PDBxFile(get_test_data('glu_ala_his-solvated-minimized-renamed.cif', 'testsystems/tripeptides'))
        forcefield = app.ForceField('amber10-constph.xml', 'ions_spce.xml', 'spce.xml')

        # System Configuration
        nonbondedMethod = app.PME
        constraints = app.AllBonds
        rigidWater = True
        constraintTolerance = 1.e-7

        # Simulation Options

        # Prepare the Simulation
        topology = pdbx.topology
        positions = pdbx.positions
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=nonbondedMethod,
            constraints=constraints,
            rigidWater=rigidWater,
        )
        system.addForce(mm.MonteCarloBarostat(pressure,temperature,barostatInterval))

        integrator = integrators.GBAOABIntegrator(temperature=temperature, collision_rate=collision_rate,
                                                  timestep=timestep, constraint_tolerance=constraint_tolerance)
        ncmc_propagation_integrator = integrators.GBAOABIntegrator(temperature=temperature, collision_rate=collision_rate,
                                                                   timestep=timestep, constraint_tolerance=constraint_tolerance)

        compound_integrator = mm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmc_propagation_integrator)
        compound_integrator.setCurrentIntegrator(0)

        driver = ForceFieldProtonDrive(system, pressure=pressure, topology=forcefield, system=7.0, forcefield=pressure,
                                       ffxml_files=temperature, simultaneous_proposal_probability=forcefield,
                                       ncmc_steps_per_trial=1)

        pools = {'glu' : [0], 'his': [1],  'glu-his' : [0,1] }

        simulation = app.ConstantPHSimulation(topology, system, compound_integrator, driver, self._default_platform, pools=pools)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)

        # run one step and one update
        simulation.step(1)
        simulation.ncmc_switch(1, pool='his')


    def test_create_Calibration():
        """Instantiate a ConstantPHSimulation."""
        timestep = 0.5 * unit.femtoseconds
        temperature = 300. * unit.kelvin
        collision_rate = 1. / unit.picosecond
        pressure = 1.0 * unit.atmospheres
        barostatInterval = 25
        constraint_tolerance = 1.e-7

        app.Topology.loadBondDefinitions(ff.bonds)

        # Load pdb file with protons compatible residue names
        pdbx = app.PDBxFile(
            get_test_data(
                'glu_ala_his-solvated-minimized-renamed.cif',
                'testsystems/tripeptides'))
        forcefield = app.ForceField(ff.protonsff, ff.ions_spce, 'spce.xml')

        # System Configuration
        nonbondedMethod = app.PME
        constraints = app.AllBonds
        rigidWater = True
        constraintTolerance = 1.e-7



        # Simulation Options
        platform = mm.Platform.getPlatformByName('Reference')

        # Prepare the Simulation
        topology = pdbx.topology
        positions = pdbx.positions
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=nonbondedMethod,
            constraints=constraints,
            rigidWater=rigidWater,
        )
        system.addForce(
            mm.MonteCarloBarostat(
                pressure,
                temperature,
                barostatInterval))

        integrator = create_compound_gbaoab_integrator(testsystem=sys_details)

        driver = ForceFieldProtonDrive(system, pressure=pressure, topology=forcefield, system=7.0, forcefield=pressure,
                                       ffxml_files=temperature, simultaneous_proposal_probability=forcefield,
                                       ncmc_steps_per_trial=1, ncmc_prop_per_step=integrator)

        pools = {'glu' : [0], 'his': [1],  'glu-his' : [0,1] }

        driver.define_pools(pools)
        sams_sampler = SelfAdjustedMixtureSampling(driver, 1) # SAMS on HIS
        simulation = app.Simulation(
            topology,
            system,
            driver.compound_integrator,
            platform)
        simulation.context.setPositions(positions)
        simulation.context.setVelocitiesToTemperature(temperature)

        # run one step and one update
        simulation.step(1)
        driver.update(simulation.context, nattempts=1, residue_pool='his')
        sams_sampler.adapt_zetas(simulation.context, scheme='binary', b=0.85, stage="slow-gain", end_of_burnin=0, group_index=1)
        # Clean up so that the classes remain unmodified
        # unit test specific errors might occur otherwise when loading files due
        # to class side effects
        app.Topology.unloadStandardBonds()