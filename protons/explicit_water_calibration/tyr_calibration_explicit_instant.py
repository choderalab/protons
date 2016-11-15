from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from protons import AmberProtonDrive
from protons.calibration import SelfAdjustedMixtureSampling
import pickle
from time import time

# Create system
directory = '/cbio/jclab/home/rossg/protons/protons/tests/testsystems/tyr_explicit/'
filename = 'sams_explicit_instant.pickle'

def load_tyrosine_implicit(directory, pH = 7.0, platform='CPU', nsteps=0, implicit=True):
    temperature = 300.0*unit.kelvin
    prmtop = app.AmberPrmtopFile(directory + 'tyr.prmtop')
    inpcrd = app.AmberInpcrdFile(directory + 'tyr.inpcrd')
    positions = inpcrd.getPositions()
    topology = prmtop.topology
    cpin_filename = directory + 'tyr.cpin'
    # Create system
    system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    # Create protons integrator
    integrator = openmm.LangevinIntegrator(temperature, 1.0/unit.picoseconds, 2.0*unit.femtoseconds)
    # Create protons proton driver
    driver = AmberProtonDrive(system, temperature, pH, prmtop, cpin_filename, integrator, debug=False,
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


simulation, driver, sams_sampler, integrator = load_tyrosine_implicit(directory,platform='OpenCL', nsteps=0, implicit=False)
simulation.minimizeEnergy(maxIterations=10000)

deviation = []    # The deviation between the target weight and actual counts
weights = []      # The bias applied by SAMS to reach target weight
delta_t = []      # To record the time (in seconds) for each iteration
N = 100000
for i in range(N):
    t0 = time()
    integrator.step(1000)
    sams_sampler.driver.update(simulation.context)  # protonation
    deviation.append(sams_sampler.adapt_zetas(simulation.context, 'binary', end_of_burnin=50000))
    delta_t.append(time() - t0)
    weights.append(sams_sampler.get_gk())
    if N % 5 == 0:
        pickle.dump((deviation, weights, delta_t), open(filename, "wb"))
pickle.dump((deviation, weights, delta_t), open(filename, "wb"))
