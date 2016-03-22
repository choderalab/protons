from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph.constph import *
import logging

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
    return context
