from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
import openmmtools
from protons import AmberProtonDrive
from protons.calibration import SelfAdjustedMixtureSampling
import pickle
import shutil
from time import time
from simtk.openmm.app import PDBFile
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
    parser.add_argument('-i', '--inpcrd', type=str, help="AMBER inpcrd file, default=None", default=None)
    parser.add_argument('-x', '--xml', type=str, help="xml file containing the system state, default=None", default=None)
    parser.add_argument('-c', '--cpin', type=str, help="AMBER cpin file")
    parser.add_argument('-o', '--out', type=str, help="the naming scheme of the output files, default='out'", default='out')
    parser.add_argument("--explicit",action='store_true',help="whether the simulation is of explicit water, default=False",default=False)
    parser.add_argument('--iterations', type=int, help="the number of iterations of MD and proton moves, default=4000", default=4000)
    parser.add_argument('--burnin', type=float, help="the fraction of iterations that are in the burn-in phase, default=0.5", default=0.5)
    parser.add_argument('--md_steps', type=int, help="the number of MD steps at each iteration, default=1000", default=1000)
    parser.add_argument('--ncmc_steps', type=int, help="the number of NCMC steps at each iteration, default=0", default=0)
    parser.add_argument('--platform', type=str, choices = ['CPU','OpenCL'], help="the platform where the simulation will be run, default=CPU", default='CPU')
    args = parser.parse_args()

    if args.explicit == True:
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

    # Initializing the weights using the last values from previous simulations
    g_initial = {'lys': [0.0, -6.94177861],
                 'tyr': [0.0, 113.80143471],
                 'as4': [0.0, -56.35949002, -56.90458235, -53.66854317, -58.70386813],
                 'gl4': [0.0, -30.89810794, -31.58412605, -33.78753743, -31.04032087],
                 'hip': [0.0, 26.66488525, 32.07581],
                 'cys': [0.0, 136.00398956]}
    driver.import_gk_values(g_initial)

    # Naming the output files
    out_pdb = args.out + '.pdb'
    out_pickle = args.out + '.pickle'
    prev_pickle = 'prev_' + args.out + '.pickle'

    # Initialize pickle for saving data
    f = open(out_pickle, "wb")
    pickle.dump((deviation, weights, delta_t), f)
    f.close()

    pdbfile = open(out_pdb, 'w')
    positions = simulation.context.getState(getPositions=True).getPositions()
    PDBFile.writeModel(simulation.topology, positions, file=pdbfile, modelIndex=0)
    pdbfile.close()

    # Run SAMS for a specified number of iterations
    N = args.iterations
    for i in range(N):
        t0 = time()
        integrator.step(args.md_steps)
        sams_sampler.driver.update(simulation.context)  # protonation
        deviation.append(sams_sampler.adapt_zetas(simulation.context, 'binary', stage='burn-in', b=0.5,
                                                  end_of_burnin=int(N * args.burnin)))
        delta_t.append(time() - t0)
        weights.append(sams_sampler.get_gk())
        if i % 5 == 0:
            shutil.copyfile(out_pickle, prev_pickle)
            f = open(out_pickle, "wb")
            pickle.dump((deviation, weights, driver.work_history, delta_t), f)
            f.close()
            np.savetxt(fname='weights.txt', X=sams_sampler.get_gk())
        if i % 1000 == 0:
            pdbfile = open(out_pdb, 'a')
            positions = simulation.context.getState(getPositions=True).getPositions()
            PDBFile.writeModel(simulation.topology, positions, file=pdbfile, modelIndex=i)
            pdbfile.close()

    shutil.copyfile(out_pickle, prev_pickle)
    f = open(out_pickle, "wb")
    pickle.dump((deviation, weights, driver.work_history, delta_t), f)
    f.close()
    np.savetxt(fname='weights.txt', X=sams_sampler.get_gk())
    