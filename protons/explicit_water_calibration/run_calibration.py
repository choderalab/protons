from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from protons import AmberProtonDrive
from protons.calibration import SelfAdjustedMixtureSampling
import pickle
import shutil
from time import time

def prepare_system(prmtop, inpcrd, cpin, pH = 7.0, platform='CPU', nsteps=0, implicit=True):
    """
    Function to prepare a system specified by AMBER topology files for a constant-ph calibration simulation with protons.
    Calibration is performed with self adjusted mixture sampling (SAMS).

    Parameters
    ----------
    prmtop: str
        the name of the AMBER prmtop file
    inpcrd: str
        the name of the AMBER inpcrd file
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
    temperature = 300.0*unit.kelvin
    prmtop = app.AmberPrmtopFile(prmtop)
    inpcrd = app.AmberInpcrdFile(inpcrd)
    positions = inpcrd.getPositions()
    topology = prmtop.topology
    # Create system
    system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    # Create protons integrator
    integrator = openmm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
    # Create protons proton driver
    driver = AmberProtonDrive(system, temperature, pH, prmtop, cpin, integrator, debug=False,
                              pressure=None, ncmc_steps_per_trial=nsteps, implicit=implicit)
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
    parser.add_argument('-i', '--inpcrd', type=str, help="AMBER inpcrd file")
    parser.add_argument('-c', '--cpin', type=str, help="AMBER cpin file")
    parser.add_argument('-o', '--out', type=str, help="the naming scheme of the output files, default='out'", default='out.pickle')
    parser.add_argument('--iterations', type=int, help="the number of iterations of MD and proton moves, default=100000", default=100000)
    parser.add_argument('--md_steps', type=int, help="the number of MD steps at each iteration, default=1000", default=1000)
    parser.add_argument('--ncmc_steps', type=int, help="the number of NCMC steps at each iteration, default=0", default=0)
    parser.add_argument('--platform', type=str, choices = ['CPU','OpenCL'], help="the platform where the simulation will be run, default=CPU", default='CPU')
    args = parser.parse_args()

    simulation, driver, sams_sampler, integrator = prepare_system(args.prmtop, args.inpcrd, args.cpin, platform=args.platform, nsteps=args.ncmc_steps)
    simulation.minimizeEnergy(maxIterations=1000)

    deviation = []    # The deviation between the target weight and actual counts
    weights = []      # The bias applied by SAMS to reach target weight
    delta_t = []      # To record the time (in seconds) for each iteration

    # Open file ready for saving data
    filename = args.out
    f = open(filename, "wb")
    pickle.dump((deviation, weights, delta_t), open(filename, "wb"))
    f.close()

    N = args.iterations
    for i in range(N):
        t0 = time()
        integrator.step(args.md_steps)
        sams_sampler.driver.update(simulation.context)  # protonation
        deviation.append(sams_sampler.adapt_zetas(simulation.context, 'binary', end_of_burnin=N/2))
        delta_t.append(time() - t0)
        weights.append(sams_sampler.get_gk())
        if i % 5 == 0:
            shutil.copyfile(filename, 'prev_'.format(i) + filename)
            f = open(filename, "wb")
            pickle.dump((deviation, weights, delta_t), open(filename, "wb"))
            f.close()

    shutil.copyfile(filename, 'prev_'.format(i) + filename)
    pickle.dump((deviation, weights, delta_t), open(filename, "wb"))
