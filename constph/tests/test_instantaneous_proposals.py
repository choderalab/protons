from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph.constph import *


def test_tyrosine_implicit():
    _tyrosine_implicit(False)  # no minimization is performed


def test_tyrosine_explicit():
    _tyrosine_explicit(False)  # no minimization performed


def _tyrosine_implicit(minimize):
    """
    Perform a single timestep and single instantenous titration attempt.
    """

    collision_rate, pH, platform_name, pressure, temperature, timestep = standard_state()

    cpin_filename = 'constph/examples/calibration-implicit/tyr.cpin'
    positions, prmtop, incprd, system = create_implicit_amber_system('constph/examples/calibration-implicit/tyr.inpcrd',
                                                                     'constph/examples/calibration-implicit/tyr.prmtop')
    if minimize:
        positions = minimizer(platform_name, system, positions)

    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, integrator, debug=False,
                                       pressure=None, nsteps_per_trial=0, implicit=True)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, mc_titration.compound_integrator, platform)
    context.setPositions(positions)  # set to minimized positions
    integrator.step(1)  # MD
    mc_titration.update(context)  # protonation


def _tyrosine_explicit(minimize):
    """
    Perform a single timestep and single instantenous titration attempt.
    """
    collision_rate, pH, platform_name, pressure, temperature, timestep = standard_state()
    cpin_filename = 'constph/examples/calibration-explicit/tyr.cpin'
    positions, prmtop, incprd, system = create_explicit_amber_system('constph/examples/calibration-explicit/tyr.inpcrd',
                                                                     'constph/examples/calibration-explicit/tyr.prmtop',
                                                                     pressure, temperature)
    if minimize:
        positions = minimizer(platform_name, system, positions)
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, integrator, debug=False,
                                       nsteps_per_trial=0, pressure=pressure, implicit=False)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, mc_titration.compound_integrator, platform)
    context.setPositions(positions)  # set to minimized positions
    integrator.step(1)  # MD
    mc_titration.update(context)  # protonation


def standard_state():
    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    timestep = 1.0 * unit.femtoseconds
    collision_rate = 9.1 / unit.picoseconds
    platform_name = 'CPU'
    pH = 9.6
    return collision_rate, pH, platform_name, pressure, temperature, timestep


def create_implicit_amber_system(inpcrd_filename, prmtop_filename):
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    positions = inpcrd.getPositions()
    system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    return positions, prmtop, inpcrd, system


def create_explicit_amber_system(inpcrd_filename, prmtop_filename, pressure, temperature):
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    positions = inpcrd.getPositions()
    system = prmtop.createSystem(implicitSolvent=None, nonbondedMethod=app.CutoffPeriodic, constraints=app.HBonds)
    system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

    return positions, prmtop, inpcrd, system


def minimizer(platform_name, system, positions, nsteps=1000):
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    CONSTRAINT_TOLERANCE = 1.0e-5
    integrator.setConstraintTolerance(CONSTRAINT_TOLERANCE)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    openmm.LocalEnergyMinimizer.minimize(context, 1.0, nsteps)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    return positions
