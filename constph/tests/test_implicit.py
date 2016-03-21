from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph.constph import *
import time

def test_tyrosine_implict():
    from openmmtools.integrators import VVVRIntegrator, GHMCIntegrator        
    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    timestep = 1.0 * unit.femtoseconds
    collision_rate = 9.1 / unit.picoseconds
    platform_name = 'CPU'
    cpin_filename = 'examples/calibration-implicit/tyr.cpin'
    pH = 9.6
    positions, prmtop, incprd, system = create_implicit_amber_system('examples/calibration-implicit/tyr.inpcrd', 'examples/calibration-implicit/tyr.prmtop')
    
    # Minimize energy.
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context, 1.0, 1000)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    del context, integrator
    
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    
    # Tighten bond constraint tolerance.
    CONSTRAINT_TOLERANCE = 1.0e-5 # can't go tighter than this or LocalEnergyMinimizer will fail.
    integrator.setConstraintTolerance(CONSTRAINT_TOLERANCE)

    # Initialize Monte Carlo titration.
    mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, integrator, debug=False, pressure=None, nsteps_per_trial=1, implicit=True)

    # Create Context (using compound integrator from MonteCarloTitration).
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, mc_titration.compound_integrator, platform)
    context.setPositions(positions) # set to minimized positions

    # Minimize energy.
    openmm.LocalEnergyMinimizer.minimize(context, 1.0, 1000)

    # Run dynamics.
    integrator.step(1)
    # Attempt protonation state changes.
    mc_titration.update(context)


def create_implicit_amber_system(inpcrd_filename,prmtop_filename):
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    positions = inpcrd.getPositions()
    system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    return positions, prmtop, inpcrd, system
    
