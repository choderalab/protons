# coding=utf-8
"""Test the reading of forcefield files included with the package"""

import pytest
from simtk import unit
from simtk.openmm import app, openmm as mm

from protons import ff, integrators, ForceFieldProtonDrive
from protons.calibration import SelfAdjustedMixtureSampling
from . import get_test_data
from .utilities import create_compound_gbaoab_integrator, SystemSetup


# Patch topology to unload standard bond definitions
def unloadStandardBonds(cls):
    """
    Resets _standardBonds and _hasLoadedStandardBonds to original state.
    """

    cls._hasLoadedStandardBonds = False
    cls._standardBonds = dict()


app.Topology.unloadStandardBonds = classmethod(unloadStandardBonds)


def test_reading_protons():
    """Read parameters and templates protons.xml using OpenMM."""
    parsed = app.ForceField(ff.protonsff)


def test_reading_bonds():
    """Read bond definitions in bonds-protons.xml using OpenMM."""

    app.Topology.loadBondDefinitions(ff.bonds)
    # unit test specific errors might occur otherwise when loading files due
    # to class side effects
    app.Topology.unloadStandardBonds()


def test_create_peptide_system_using_protons_xml():
    """Test if protons.xml can be used to successfully create a peptide System object in OpenMM."""
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

    # Integration Options
    dt = 0.5 * unit.femtoseconds
    temperature = 300. * unit.kelvin
    friction = 1. / unit.picosecond
    pressure = 1.0 * unit.atmospheres
    barostatInterval = 25

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

    integrator = integrators.GHMCIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)

    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due
    # to class side effects
    app.Topology.unloadStandardBonds()


def test_create_peptide_simulation_using_protons_xml():
    """Test if protons.xml can be used to successfully create a peptide Simulation object in OpenMM and
    Instantiate a ForceFieldProtonDrive."""

    sys_details = SystemSetup()
    sys_details.timestep = 0.5 * unit.femtoseconds
    sys_details.temperature = 300. * unit.kelvin
    sys_details.collision_rate = 1. / unit.picosecond
    sys_details.pressure = 1.0 * unit.atmospheres
    sys_details.barostatInterval = 25
    sys_details.constraint_tolerance = 1.e-7

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
    sys_details.constraintTolerance = 1.e-7

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
            sys_details.pressure,
            sys_details.temperature,
            sys_details.barostatInterval))

    integrator = create_compound_gbaoab_integrator(testsystem=sys_details)


    driver = ForceFieldProtonDrive(
        system, sys_details.temperature, 7.0, [ff.protonsff], forcefield,
        topology, integrator, debug=False, pressure=sys_details.pressure,
        ncmc_steps_per_trial=1, implicit=False, cationName="NA",
        anionName="CL")

    simulation = app.Simulation(
        topology,
        system,
        driver.compound_integrator,
        platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(sys_details.temperature)

    # run one step and one update
    simulation.step(1)
    driver.update(simulation.context, nattempts=1)
    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due
    # to class side effects
    app.Topology.unloadStandardBonds()


def test_create_peptide_simulation_with_residue_pools_using_protons_xml():
    """Test if protons.xml can be used to successfully create a peptide Simulation object in OpenMM and
    Instantiate a ForceFieldProtonDrive, while using pools of residues to sample from."""

    sys_details = SystemSetup()
    sys_details.timestep = 0.5 * unit.femtoseconds
    sys_details.temperature = 300. * unit.kelvin
    sys_details.collision_rate = 1. / unit.picosecond
    sys_details.pressure = 1.0 * unit.atmospheres
    sys_details.barostatInterval = 25
    sys_details.constraint_tolerance = 1.e-7

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
    sys_details.constraintTolerance = 1.e-7



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
            sys_details.pressure,
            sys_details.temperature,
            sys_details.barostatInterval))

    integrator = create_compound_gbaoab_integrator(testsystem=sys_details)

    driver = ForceFieldProtonDrive(
        system, sys_details.temperature, 7.0, [ff.protonsff], forcefield,
        topology, integrator, debug=False, pressure=sys_details.pressure,
        ncmc_steps_per_trial=1, implicit=False, cationName="NA",
        anionName="CL")

    pools = {'glu' : [0], 'his': [1],  'glu-his' : [0,1] }

    driver.define_pools(pools)

    simulation = app.Simulation(
        topology,
        system,
        driver.compound_integrator,
        platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(sys_details.temperature)

    # run one step and one update
    simulation.step(1)
    driver.update(simulation.context, nattempts=1, residue_pool='his')
    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due
    # to class side effects
    app.Topology.unloadStandardBonds()


def pattern_from_multiline(multiline, pattern):
    """Return only lines that contain the pattern

    Parameters
    ----------
    multiline - multiline str        
    pattern - str

    Returns
    -------
    multiline str containing pattern
    """

    return '\n'.join([line for line in multiline.splitlines() if pattern in line])


def test_create_peptide_calibration_with_residue_pools_using_protons_xml():
    """Test if protons.xml can be used to successfully create a peptide Simulation object in OpenMM and
    Instantiate a ForceFieldProtonDrive, while using pools of residues to sample from,
    and calibrate histidine."""

    sys_details = SystemSetup()
    sys_details.timestep = 0.5 * unit.femtoseconds
    sys_details.temperature = 300. * unit.kelvin
    sys_details.collision_rate = 1. / unit.picosecond
    sys_details.pressure = 1.0 * unit.atmospheres
    sys_details.barostatInterval = 25
    sys_details.constraint_tolerance = 1.e-7

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
    sys_details.constraintTolerance = 1.e-7



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
            sys_details.pressure,
            sys_details.temperature,
            sys_details.barostatInterval))

    integrator = create_compound_gbaoab_integrator(testsystem=sys_details)

    driver = ForceFieldProtonDrive(
        system, sys_details.temperature, 7.0, [ff.protonsff], forcefield,
        topology, integrator, debug=False, pressure=sys_details.pressure,
        ncmc_steps_per_trial=1, implicit=False, cationName="NA",
        anionName="CL")

    pools = {'glu' : [0], 'his': [1],  'glu-his' : [0,1] }

    driver.define_pools(pools)
    sams_sampler = SelfAdjustedMixtureSampling(driver, 1) # SAMS on HIS
    simulation = app.Simulation(
        topology,
        system,
        driver.compound_integrator,
        platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(sys_details.temperature)

    # run one step and one update
    simulation.step(1)
    driver.update(simulation.context, nattempts=1, residue_pool='his')
    sams_sampler.adapt_zetas(simulation.context, scheme='binary', b=0.85, stage="slow-gain", end_of_burnin=0, group_index=1)
    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due
    # to class side effects
    app.Topology.unloadStandardBonds()

def test_peptide_system_integrity():
    """
    Set up peptide, and assure that the systems particles have not been modified after driver instantiation.
    """

    sys_details = SystemSetup()
    sys_details.timestep = 0.5 * unit.femtoseconds
    sys_details.temperature = 300. * unit.kelvin
    sys_details.collision_rate = 1. / unit.picosecond
    sys_details.pressure = 1.0 * unit.atmospheres
    sys_details.barostatInterval = 25
    sys_details.constraint_tolerance = 1.e-7

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
    sys_details.constraintTolerance = 1.e-7

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
            sys_details.pressure,
            sys_details.temperature,
            sys_details.barostatInterval))

    integrator = create_compound_gbaoab_integrator(testsystem=sys_details)

    original_system = pattern_from_multiline(mm.XmlSerializer.serialize(system), '<Particle')

    driver = ForceFieldProtonDrive(
        system, sys_details.temperature, 7.0, [ff.protonsff], forcefield,
        topology, integrator, debug=False, pressure=sys_details.pressure,
        ncmc_steps_per_trial=1, implicit=False, cationName="NA",
        anionName="CL")

    after_driver = pattern_from_multiline(mm.XmlSerializer.serialize(system), '<Particle')

    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due
    # to class side effects
    app.Topology.unloadStandardBonds()

    # Make sure there are no differences between the particles in each system
    assert original_system == after_driver


@pytest.mark.skip(reason="Still creating input file.")
def test_create_hewl_simulation_using_protons_xml():
    """Test if protons.xml can be used to successfully create a Simulation object in OpenMM."""

    app.Topology.loadBondDefinitions(ff.bonds)

    # Input Files
    pdbx = app.PDBxFile(
        get_test_data(
            '4lzt-protons.cif',
            'testsystems/hewl-explicit'))
    forcefield = app.ForceField(ff.protonsff, ff.ions_spce, 'spce.xml')

    # System Configuration
    nonbondedMethod = app.PME
    nonbondedCutoff = 10.0 * unit.nanometers
    ewaldErrorTolerance = 1.e-5
    constraints = app.AllBonds
    rigidWater = True
    constraintTolerance = 1.e-7

    # Integration Options
    dt = 0.5 * unit.femtoseconds
    temperature = 300. * unit.kelvin
    friction = 1. / unit.picosecond
    pressure = 1.0 * unit.atmospheres
    barostatInterval = 25

    # Simulation Options
    platform = mm.Platform.getPlatformByName('Reference')

    # Prepare the Simulation
    topology = pdbx.topology
    positions = pdbx.positions
    system = forcefield.createSystem(
        topology,
        nonbondedMethod=nonbondedMethod,
        nonbondedCutoff=nonbondedCutoff,
        constraints=constraints,
        rigidWater=rigidWater,
        ewaldErrorTolerance=ewaldErrorTolerance)
    system.addForce(
        mm.MonteCarloBarostat(
            pressure,
            temperature,
            barostatInterval))
    integrator = integrators.GHMCIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due
    # to class side effects
    app.Topology.unloadStandardBonds()


def test_reading_hydrogens():
    """Read hydrogen definitions in hydrogens-protons.xml using OpenMM."""
    app.Modeller.loadHydrogenDefinitions(ff.hydrogens)
    # Clean up so that the classes remain unmodified
    pass  # implement unloadhydrogendefinitions if necessary


def test_reading_ions_spce():
    """Read parameters and templates in ions_spce.xml using OpenMM."""
    parsed = app.ForceField(ff.ions_spce)


def test_reading_ions_tip3p():
    """Read parameters and templates in ions_tip3p.xml using OpenMM."""

    parsed = app.ForceField(ff.ions_tip3p)


def test_reading_ions_tip4pew():
    """Read parameters and templates in ions_tip4pew.xml using OpenMM."""

    parsed = app.ForceField(ff.ions_tip4pew)


def test_reading_gaff():
    """Read parameters and templates in gaff.xml using OpenMM."""

    parsed = app.ForceField(ff.gaff)


def test_reading_gaff2():
    """Read parameters and templates in gaff2.xml using OpenMM."""

    parsed = app.ForceField(ff.gaff2)
