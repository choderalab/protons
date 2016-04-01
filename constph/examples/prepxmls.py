from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
import logging


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

def setup_systems_thuper_serially():
    
    for solvent in ["implicit"]: # , "explicit"]:
        for aa in ["cys", "lys", "glu", "his", "tyr", "asp"]:
            foldername = "calibration-{}".format(solvent)
            prmtop = "{}/{}.prmtop".format(foldername, aa)
            inpcrd = "{}/{}.inpcrd".format(foldername, aa)
            outname = "calibration-systems/{}-{}".format(aa,solvent)
            
            if solvent == "explicit":
                callf = make_xml_explicit
            else:
                callf = make_xml_implicit
            print(aa,solvent)
            callf(inpcrd, prmtop, outname)
            

def make_xml_explicit(inpcrd_filename,prmtop_filename,outfile):

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

def make_xml_implicit(inpcrd_filename,prmtop_filename,outfile):

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


def minimizer(platform_name, system, positions, nsteps=10000):
    integrator = openmm.VerletIntegrator(0.1 * unit.femtoseconds)
    CONSTRAINT_TOLERANCE = 1.0e-7
    integrator.setConstraintTolerance(CONSTRAINT_TOLERANCE)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    logging.info("Initial energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
    openmm.LocalEnergyMinimizer.minimize(context, 1.0, nsteps)
    logging.info("Final energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    return context

if __name__ == "__main__":
    setup_systems_thuper_serially()
  
