# coding=utf-8
"""Test the reading of forcefield files included with the package"""

from protons import ff, integrators, ForceFieldProtonDrive
import simtk
from simtk.openmm import app, openmm as mm
from simtk import unit
from . import get_test_data
import pytest


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
    # unit test specific errors might occur otherwise when loading files due to class side effects
    app.Topology.unloadStandardBonds()


def test_create_peptide_system_using_protons_xml():
    """Test if protons.xml can be used to successfully create a peptide System object in OpenMM."""
    app.Topology.loadBondDefinitions(ff.bonds)

    # Load pdb file with protons compatible residue names
    pdbx = app.PDBxFile(get_test_data('glu_ala_his-solvated-minimized-renamed.cif', 'testsystems/tripeptides'))
    forcefield = app.ForceField(ff.protonsff, ff.ions_spce, 'spce.xml')

    # System Configuration
    nonbondedMethod = app.NoCutoff
    constraints = app.AllBonds
    rigidWater = True
    constraintTolerance = 1.e-7

    # Integration Options
    dt = 0.5 * unit.picoseconds
    temperature = 300. * unit.kelvin
    friction = 1. / unit.picosecond
    pressure = 1.0 * unit.atmospheres
    barostatInterval = 25

    # Simulation Options
    platform = mm.Platform.getPlatformByName('Reference')

    # Prepare the Simulation
    topology = pdbx.topology
    positions = pdbx.positions
    system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod,
                                     constraints=constraints, rigidWater=rigidWater, )
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostatInterval))

    integrator = integrators.GHMCIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)

    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due to class side effects
    app.Topology.unloadStandardBonds()


def test_create_peptide_simulation_using_protons_xml():
    """Test if protons.xml can be used to successfully create a peptide Simulation object in OpenMM and
    Instantiate a ForceFieldProtonDrive."""

    app.Topology.loadBondDefinitions(ff.bonds)

    # Load pdb file with protons compatible residue names
    pdbx = app.PDBxFile(get_test_data('glu_ala_his-solvated-minimized-renamed.cif', 'testsystems/tripeptides'))
    forcefield = app.ForceField(ff.protonsff, ff.ions_spce, 'spce.xml')

    # System Configuration
    nonbondedMethod = app.NoCutoff
    constraints = app.AllBonds
    rigidWater = True
    constraintTolerance = 1.e-7

    # Integration Options
    dt = 0.5 * unit.picoseconds
    temperature = 300. * unit.kelvin
    friction = 1. / unit.picosecond
    pressure = 1.0 * unit.atmospheres
    barostatInterval = 25

    # Simulation Options
    platform = mm.Platform.getPlatformByName('Reference')

    # Prepare the Simulation
    topology = pdbx.topology
    positions = pdbx.positions
    system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod,
                                     constraints=constraints, rigidWater=rigidWater, )
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostatInterval))

    integrator = integrators.GHMCIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)

    driver = ForceFieldProtonDrive(system, temperature, 7.0, [ff.protonsff],
                                   topology, integrator, debug=False,
                                   pressure=pressure, ncmc_steps_per_trial=1, implicit=False,
                                   cationName="NA", anionName="CL")

    simulation = app.Simulation(topology, system, driver.compound_integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    # run one step and one update
    simulation.step(1)
    driver.update(simulation.context, nattempts=1)
    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due to class side effects
    app.Topology.unloadStandardBonds()


@pytest.mark.skip(reason="Still creating input file.")
def test_create_hewl_simulation_using_protons_xml():
    """Test if protons.xml can be used to successfully create a Simulation object in OpenMM."""

    app.Topology.loadBondDefinitions(ff.bonds)

    # Input Files
    pdbx = app.PDBxFile(get_test_data('4lzt-protons.cif', 'testsystems/hewl-explicit'))
    forcefield = app.ForceField(ff.protonsff, ff.ions_spce, 'spce.xml')

    # System Configuration
    nonbondedMethod = app.PME
    nonbondedCutoff = 10.0 * unit.nanometers
    ewaldErrorTolerance = 1.e-5
    constraints = app.AllBonds
    rigidWater = True
    constraintTolerance = 1.e-7

    # Integration Options
    dt = 0.5 * unit.picoseconds
    temperature = 300. * unit.kelvin
    friction = 1. / unit.picosecond
    pressure = 1.0 * unit.atmospheres
    barostatInterval = 25

    # Simulation Options
    platform = mm.Platform.getPlatformByName('Reference')

    # Prepare the Simulation
    topology = pdbx.topology
    positions = pdbx.positions
    system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod, nonbondedCutoff=nonbondedCutoff,
                                     constraints=constraints, rigidWater=rigidWater,
                                     ewaldErrorTolerance=ewaldErrorTolerance)
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostatInterval))
    integrator = integrators.GHMCIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)

    # Clean up so that the classes remain unmodified
    # unit test specific errors might occur otherwise when loading files due to class side effects
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
