from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
import openmmtools
from protons import AmberProtonDrive
from protons.calibration import SelfAdjustedMixtureSampling
from time import time
import numpy as np

def prepare_system(prmtop, cpin, inpcrd=None, xml=None, pH = 7.0, platform='CPU', nsteps=0, implicit=True):
    """
    Function to prepare a system specified by AMBER topology files for a constant-ph calibration simulation with protons.
    Calibration is performed with self adjusted mixture sampling (SAMS).

    Parameters
    ----------
    prmtop: str
        the name of the AMBER prmtop file
    inpcrd: str
        the name of the AMBER inpcrd file
    xml: str
        the name of the XML file that contains the positions of the system
    cpin: str
        the name of the AMBER cpin file
    pH: float
        the pH at which the calibration will be performed
    platform: str
        OpenMM platform with which the simulation will be performed. Choices are 'CPU' or 'OpenCL'
    nsteps: int
        The number of NCMC steps to perform for proton creation/annihilation. If nstesp=0, instant exchanges will be done.
    implicit: bool
        Whether to calibrated in implicit or explicit solvent

    Return
    ------
    simulation: simtk.openmm.Simulation
        OpenMM simulation object
    driver: protons.AmberProtonDrive
        Driver for constant pH simulation
    sams_sampler: protons.SelfAdjustedMixtureSampling
        Wrapper for calibrating with SAMS
    integrator:  simtk.openmm.integrator
        Integrator for sampling the configuration of the system
    """
    # Loading system and initializing driver
    temperature = 300.0*unit.kelvin

    # Loading the system and forcefield
    prmtop = app.AmberPrmtopFile(prmtop)
    topology = prmtop.topology

    # Loading the positions
    if inpcrd is not None:
        inpcrd = app.AmberInpcrdFile(inpcrd)
        positions = inpcrd.getPositions()
    elif xml is not None:
        lines = open(xml, 'r').read()
        state = openmm.openmm.XmlSerializer.deserialize(lines)
        positions = state.getPositions()
    else:
        raise ('Please input either an inpcrd or xml that contains the system coordinates')

    # Create system
    if implicit == True:
        system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    else:
        system = prmtop.createSystem(nonbondedMethod=app.PME, constraints=app.HBonds)
        system.addForce(openmm.MonteCarloBarostat(1*unit.atmospheres, temperature, 25))

    # Create protons integrator
    integrator = openmmtools.integrators.GHMCIntegrator(temperature, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)

    # Create protons proton driver
    if implicit == True:
        driver = AmberProtonDrive(system, temperature, pH, prmtop, cpin, integrator, debug=False,
                              pressure=None, ncmc_steps_per_trial=nsteps, implicit=implicit)
    else:
        driver = AmberProtonDrive(system, temperature, pH, prmtop, cpin, integrator, debug=False,
                              pressure=1*unit.atmospheres, ncmc_steps_per_trial=nsteps, implicit=implicit)

    # Create SAMS sampler
    sams_sampler = SelfAdjustedMixtureSampling(driver)

    # Create simulation
    if platform == 'OpenCL':
        platform = openmm.Platform.getPlatformByName('OpenCL')
        properties = {'OpenCLPrecision': 'mixed'}
        simulation = app.Simulation(topology, system, driver.compound_integrator, platform, properties)
    else:
        simulation = app.Simulation(topology, system, driver.compound_integrator)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)
    return simulation, driver, sams_sampler, integrator

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Run a calibration simulation with protons from AMBER topology files")
    parser.add_argument('-p', '--prmtop',type=str, help="AMBER prmtop file")
    parser.add_argument('-i', '--inpcrd', type=str, help="AMBER inpcrd file, default=None", default=None)
    parser.add_argument('-x', '--xml', type=str, help="xml file containing the system state, default=None", default=None)
    parser.add_argument('-c', '--cpin', type=str, help="AMBER cpin file")
    parser.add_argument('-o', '--out', type=str, help="the naming scheme of the output files, default='out'", default='out')
    parser.add_argument("--explicit",action='store_true',help="whether the simulation is of explicit water, default=False",default=False)
    parser.add_argument('--iterations', type=int, help="the number of iterations of MD and proton moves, default=500", default=500)
    parser.add_argument('--md_steps', type=int, help="the number of MD steps at each iteration, default=500", default=500)
    parser.add_argument('--ncmc_steps', type=int, help="the number of NCMC steps at each iteration, default=0", default=0)
    parser.add_argument('--platform', type=str, choices = ['CPU','OpenCL'], help="the platform where the simulation will be run, default=CPU", default='CPU')
    args = parser.parse_args()

    if args.explicit:
        implicit = False
    else:
        implicit = True

    simulation, driver, sams_sampler, integrator = prepare_system(args.prmtop, args.cpin, inpcrd=args.inpcrd,
                                                                  xml=args.xml, platform=args.platform,
                                                                  nsteps=args.ncmc_steps, implicit=implicit)
    # Minimization
    simulation.minimizeEnergy(maxIterations=1000)

    # Pre-assignment
    deviation = []    # The deviation between the target weight and actual counts
    weights = []      # The bias applied by SAMS to reach target weight
    delta_t = []      # To record the time (in seconds) for each iteration

    g_initial = {'lys': [0.0, -6.8],
                 'tyr': [0.0, 126.7],
                 'as4': [0.0, -63.2, -65.1, -63.1, -69.5],
                 'gl4': [0.0, -33.8, -39.7, -36.1, -38.5],
                 'hip': [0.0, 27.5, 29.6],
                 'cys': [0.0, 154.4]}
    driver.import_gk_values(g_initial)

    # Naming the output file
    out_text = args.out + '.txt'

    # Initialize pickle for saving data
    f = open(out_text, "w")
    s = "# iterations = {:4}, # MD steps = {:7}, # perturbations = {:5}\n".format(args.iterations, args.md_steps, args.ncmc_steps)
    f.write(s)
    s = 'Iter      State      AccProb      Time\n'
    f.write(s)
    f.close()

    # Run protons for a specified number of iterations
    t0 = time()
    for i in range(args.iterations):
        integrator.step(args.md_steps)
        driver.update(simulation.context, nattempts=1)  # protonation
        if i % 20 == 0:
            try:
                accprob = driver._get_acceptance_probability()
            except ZeroDivisionError:
                accprob = 0.0
            f = open(out_text, "a")
            s = "{:7}     {:3}       {:0.2f}     {:8.1f}\n".format(i, driver.work_history[i][0][0], accprob, time() - t0)
            f.write(s)
            f.close()
            work = [wrk[2] for wrk in driver.work_history]
            np.savetxt(fname=args.out +'_work.txt', X=work)