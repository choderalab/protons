# coding=utf-8
"""Test the reading of forcefield files included with the package.
Developer Notes
---------------
Do not use protons.app classes for this test module. These files need to be tested to be compatible with original OpenMM.
Note that the Z in the filename is necessary so that functions with class side effects get loaded last.
"""

import pytest
from simtk import unit
from simtk.openmm import app, openmm as mm
import os
from protons import GBAOABIntegrator, GHMCIntegrator
from protons import ForceFieldProtonDrive
from protons.app.proposals import UniformProposal
from protons import app as protonsapp
from protons.app.driver import SAMSApproach

from protons import SAMSCalibrationEngine
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

app_location = os.path.dirname(protonsapp.__file__)
bonds_path = os.path.join(app_location, 'data', 'bonds-amber10-constph.xml')
ffxml_path = os.path.join(app_location, 'data', 'amber10-constph.xml')
ions_spce_path = os.path.join(app_location, 'data', 'ions_spce.xml')
ions_tip3p_path = os.path.join(app_location, 'data', 'ions_tip3p.xml')
ions_tip4pew_path = os.path.join(app_location, 'data', 'ions_tip4pew.xml')
hydrogen_path = os.path.join(app_location, 'data', 'hydrogens-amber10-constph.xml')
gaff_path = os.path.join(app_location, 'data', 'gaff.xml')
gaff2_path = os.path.join(app_location, 'data', 'gaff2.xml')

def test_reading_protons():
    """Read parameters and templates protons.xml using OpenMM."""
    parsed = app.ForceField(ffxml_path)

def test_reading_bonds():
    """Read bond definitions in bonds-protons.xml using OpenMM."""

    app.Topology.loadBondDefinitions(bonds_path)

    # unit test specific errors might occur otherwise when loading files due
    # to class side effects
    app.Topology.unloadStandardBonds()


def test_create_peptide_system_using_protons_xml():
    """Test if protons.xml can be used to successfully create a peptide System object in OpenMM."""
    app.Topology.loadBondDefinitions(bonds_path)

    # Load pdb file with protons compatible residue names
    pdbx = app.PDBxFile(
       get_test_data(
            'glu_ala_his-solvated-minimized-renamed.cif',
            'testsystems/tripeptides'))
    forcefield = app.ForceField(ffxml_path, ions_spce_path, 'spce.xml')

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

    integrator = GHMCIntegrator(temperature, friction, dt)
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

    app.Topology.loadBondDefinitions(bonds_path)

    # Load pdb file with protons compatible residue names
    pdbx = app.PDBxFile(
        get_test_data(
            'glu_ala_his-solvated-minimized-renamed.cif',
            'testsystems/tripeptides'))
    forcefield = app.ForceField(ffxml_path, ions_spce_path, 'spce.xml')

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
    driver = ForceFieldProtonDrive(sys_details.temperature, topology, system, forcefield, ffxml_path, pressure=sys_details.pressure, perturbations_per_trial=1)

    simulation = app.Simulation(
        topology,
        system,
        integrator,
        platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(sys_details.temperature)

    # run one step and one update
    simulation.step(1)
    driver.attach_context(simulation.context)
    driver.update(UniformProposal(), nattempts=1)
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

    app.Topology.loadBondDefinitions(bonds_path)

    # Load pdb file with protons compatible residue names
    pdbx = app.PDBxFile(
        get_test_data(
            'glu_ala_his-solvated-minimized-renamed.cif',
            'testsystems/tripeptides'))
    forcefield = app.ForceField(ffxml_path, ions_spce_path, 'spce.xml')

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
    driver = ForceFieldProtonDrive(sys_details.temperature, topology, system, forcefield, ffxml_path, pressure=sys_details.pressure, perturbations_per_trial=1)

    pools = {'glu' : [0], 'his': [1],  'glu-his' : [0,1] }

    driver.define_pools(pools)

    simulation = app.Simulation(
        topology,
        system,
        integrator,
        platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(sys_details.temperature)
    driver.attach_context(simulation.context)

    # run one step and one update
    simulation.step(1)
    driver.update(UniformProposal(), nattempts=1, residue_pool='his')
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

    app.Topology.loadBondDefinitions(bonds_path)

    # Load pdb file with protons compatible residue names
    pdbx = app.PDBxFile(
        get_test_data(
            'glu_ala_his-solvated-minimized-renamed.cif',
            'testsystems/tripeptides'))
    forcefield = app.ForceField(ffxml_path, ions_spce_path, 'spce.xml')

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

    driver = ForceFieldProtonDrive(sys_details.temperature, topology, system, forcefield, ffxml_path, pressure=sys_details.pressure, perturbations_per_trial=1)

    pools = {'glu' : [0], 'his': [1],  'glu-his' : [0,1] }

    driver.define_pools(pools)
    driver.enable_calibration(SAMSApproach.ONESITE, group_index=1)
    sams_sampler = SAMSCalibrationEngine(driver) # SAMS on HIS
    simulation = app.Simulation(
        topology,
        system,
        integrator,
        platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(sys_details.temperature)
    driver.attach_context(simulation.context)
    # run one step and one update
    simulation.step(1)
    driver.update(UniformProposal(), nattempts=1, residue_pool='his')
    sams_sampler.adapt_zetas()
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

    app.Topology.loadBondDefinitions(bonds_path)

    # Load pdb file with protons compatible residue names
    pdbx = app.PDBxFile(
        get_test_data(
            'glu_ala_his-solvated-minimized-renamed.cif',
            'testsystems/tripeptides'))
    forcefield = app.ForceField(ffxml_path, ions_spce_path, 'spce.xml')

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

    driver = ForceFieldProtonDrive(sys_details.temperature, topology, system, forcefield, ffxml_path, pressure=sys_details.pressure, perturbations_per_trial=1)

    after_driver = pattern_from_multiline(mm.XmlSerializer.serialize(system), '<Particle')

    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due
    # to class side effects
    app.Topology.unloadStandardBonds()

    # Make sure there are no differences between the particles in each system
    assert original_system == after_driver


def test_reading_hydrogens():
    """Read hydrogen definitions in hydrogens-protons.xml using OpenMM."""
    app.Modeller.loadHydrogenDefinitions(hydrogen_path)
    # Clean up so that the classes remain unmodified
    pass  # implement unloadhydrogendefinitions if necessary


def test_reading_ions_spce():
    """Read parameters and templates in ions_spce.xml using OpenMM."""
    parsed = app.ForceField(ions_spce_path)


def test_reading_ions_tip3p():
    """Read parameters and templates in ions_tip3p.xml using OpenMM."""

    parsed = app.ForceField(ions_tip3p_path)


def test_reading_ions_tip4pew():
    """Read parameters and templates in ions_tip4pew.xml using OpenMM."""

    parsed = app.ForceField(ions_tip4pew_path)


def test_reading_gaff():
    """Read parameters and templates in gaff.xml using OpenMM."""

    parsed = app.ForceField(gaff_path)


def test_reading_gaff2():
    """Read parameters and templates in gaff2.xml using OpenMM."""

    parsed = app.ForceField(gaff2_path)
