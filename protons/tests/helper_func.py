from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from protons import *
import logging


try:
    openmm.Platform.getPlatformByName('CUDA')
    hasCUDA = True
except Exception:
    logging.info("CUDA unavailable on this system.")
    hasCUDA = False


class SystemSetup:
    """Empty class for storing systems and relevant attributes"""
    pass


def make_method(func, input):
    # http://blog.kevinastone.com/generate-your-tests.html
    def test_input(self):
        func(self, input)
    test_input.__name__ = 'test_{func}_{input}'.format(func=func.__name__, input=input)
    return test_input


def generate(func, *inputs):
    """
    Take a TestCase and add a test method for each input
    """
    # http://blog.kevinastone.com/generate-your-tests.html
    def decorator(testcase):
        for input in inputs:
            test_input = make_method(func, input)
            setattr(testcase, test_input.__name__, test_input)
        return testcase

    return decorator


def make_xml_explicit_tyr(inpcrd_filename='constph/examples/calibration-explicit/tyr.inpcrd',
                          prmtop_filename='constph/examples/calibration-explicit/tyr.prmtop',
                          outfile='tyrosine_explicit'):

    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    outfile1=open('{}.sys.xml'.format(outfile), 'w')
    outfile2=open('{}.state.xml'.format(outfile), 'w')
    platform_name = 'CPU'
    pH = 9.6
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    positions = inpcrd.getPositions()
    system = prmtop.createSystem(implicitSolvent=None, nonbondedMethod=app.CutoffPeriodic, constraints=app.HBonds)
    system.addForce(openmm.MonteCarloBarostat(pressure, temperature))
    context = minimizer(platform_name, system, positions)
    outfile1.write(openmm.XmlSerializer.serialize(system))
    outfile2.write(openmm.XmlSerializer.serialize(context.getState(getPositions=True)))

def make_xml_implicit_tyr(inpcrd_filename='constph/examples/calibration-implicit/tyr.inpcrd',
                          prmtop_filename='constph/examples/calibration-implicit/tyr.prmtop',
                          outfile='tyrosine_implicit'):

    temperature = 300.0 * unit.kelvin
    outfile1=open('{}.sys.xml'.format(outfile), 'w')
    outfile2=open('{}.state.xml'.format(outfile), 'w')
    platform_name = 'CPU'
    pH = 9.6
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    positions = inpcrd.getPositions()
    system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    context = minimizer(platform_name, system, positions)
    outfile1.write(openmm.XmlSerializer.serialize(system))
    outfile2.write(openmm.XmlSerializer.serialize(context.getState(getPositions=True)))


def compute_potential_components(context):
    """
    Compute potential energy, raising an exception if it is not finite.

    Parameters
    ----------
    context : simtk.openmm.Context
        The context from which to extract, System, parameters, and positions.

    """
    import copy
    system = context.getSystem()
    system = copy.deepcopy(system)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    parameters = context.getParameters()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName('Reference')
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    for (parameter, value) in parameters.items():
        context.setParameter(parameter, value)
    energy_components = list()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        forcename = force.__class__.__name__
        groups = 1<<index
        potential = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        energy_components.append((forcename, potential))
    del context, integrator
    return energy_components


def minimizer(platform_name, system, positions, nsteps=1000):
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    CONSTRAINT_TOLERANCE = 1.0e-5
    integrator.setConstraintTolerance(CONSTRAINT_TOLERANCE)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    logging.info("Initial energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
    openmm.LocalEnergyMinimizer.minimize(context, 1.0, nsteps)
    logging.info("Final energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    return context, positions
