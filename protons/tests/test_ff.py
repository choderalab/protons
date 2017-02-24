# coding=utf-8
"""Test the reading of forcefield files included with the package"""

from protons import ff, integrators
from simtk.openmm import app, openmm as mm
from simtk import unit
from . import get_test_data
import pytest


def test_reading_protons():
    """Read parameters and templates protons.xml using OpenMM."""
    parsed = app.ForceField(ff.protonsff)


def test_create_peptide_simulation_using_protons_xml():
    """Test if protons.xml can be used to successfully create a peptide Simulation object in OpenMM."""

    # Input Files
    app.Topology.loadBondDefinitions(ff.bonds)
    pdbx = app.PDBFile(get_test_data('glu_ala_his.pdb','testsystems/tripeptides'))
    forcefield = app.ForceField(ff.protonsff, ff.ions_spce, 'spce.xml')

    # System Configuration
    nonbondedMethod = app.NoCutoff
    constraints = app.AllBonds
    rigidWater = True
    constraintTolerance = 1.e-7

    # Integration Options
    dt = 0.5 *unit.picoseconds
    temperature = 300.* unit.kelvin
    friction = 1. / unit.picosecond
    pressure = 1.0 * unit.atmospheres
    barostatInterval = 25

    # Simulation Options
    platform = mm.Platform.getPlatformByName('Reference')

    # Prepare the Simulation
    topology = pdbx.topology
    positions = pdbx.positions
    system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod,
                                     constraints=constraints, rigidWater=rigidWater,                                    )
    system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostatInterval))
    integrator = integrators.GHMCIntegrator(temperature, friction, dt)
    integrator.setConstraintTolerance(constraintTolerance)
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)


@pytest.mark.skip(reason="Still creating input file.")
def test_create_hewl_simulation_using_protons_xml():
    """Test if protons.xml can be used to successfully create a Simulation object in OpenMM."""

    # Input Files
    app.Topology.loadBondDefinitions(ff.bonds)
    pdbx = app.PDBxFile(get_test_data('4lzt-protons.cif','testsystems/hewl-explicit'))
    forcefield = app.ForceField(ff.protonsff, ff.ions_spce, 'spce.xml')

    # System Configuration
    nonbondedMethod = app.PME
    nonbondedCutoff = 10.0 * unit.nanometers
    ewaldErrorTolerance = 1.e-5
    constraints = app.AllBonds
    rigidWater = True
    constraintTolerance = 1.e-7

    # Integration Options
    dt = 0.5 *unit.picoseconds
    temperature = 300.* unit.kelvin
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


def test_reading_hydrogens():
    """Read hydrogen definitions in hydrogens-protons.xml using OpenMM."""
    parsed = app.Modeller.loadHydrogenDefinitions(ff.hydrogens)


def test_reading_bonds():
    """Read bond definitions in bonds-protons.xml using OpenMM."""
    parsed = app.Topology.loadBondDefinitions(ff.bonds)


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

