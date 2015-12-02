from constph import *
import simtk.openmm.app as app
import pandas as pd

niterations = 5000 # number of dynamics/titration cycles to run
nsteps = 500  # number of timesteps of dynamics per iteration
temperature = 300.0 * units.kelvin
timestep = 1.0 * units.femtoseconds
collision_rate = 9.1 / units.picoseconds
log = open("states.log", "w")

# Filenames.
# prmtop_filename = 'amber-example/prmtop'
# inpcrd_filename = 'amber-example/min.x'
# cpin_filename = 'amber-example/cpin'
# pH = 7.0

# Calibration on a terminally-blocked amino acid in implicit solvent
prmtop_filename = 'calibration-implicit/tyr.prmtop'
inpcrd_filename = 'calibration-implicit/tyr.inpcrd'
cpin_filename =   'calibration-implicit/tyr.cpin'
pH = 9.6

#prmtop_filename = 'calibration-explicit/his.prmtop'
#inpcrd_filename = 'calibration-explicit/his.inpcrd'
#cpin_filename =   'calibration-explicit/his.cpin'
#pH = 6.5

#prmtop_filename = 'calibration-implicit/his.prmtop'
#inpcrd_filename = 'calibration-implicit/his.inpcrd'
#cpin_filename =   'calibration-implicit/his.cpin'
#pH = 6.5

# Load the AMBER system.

print("Creating AMBER system...")
inpcrd = app.AmberInpcrdFile(inpcrd_filename)
prmtop = app.AmberPrmtopFile(prmtop_filename)
system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
debuglogger = pd.DataFrame()

# Initialize Monte Carlo titration.
print("Initializing Monte Carlo titration...")
# mc_titration = CalibrationTitration(system, temperature, pH, prmtop, cpin_filename, debug=True)
# mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, debug=True)

# Create integrator and context.
platform_name = 'OpenCL'
platform = openmm.Platform.getPlatformByName(platform_name)
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
context = openmm.Context(system, integrator, platform)
context.setPositions(inpcrd.getPositions())
# MBar Titration needs a context to initialize

mc_titration = MBarCalibrationTitration(system, temperature, pH, prmtop, cpin_filename, context, debug=True)
# Minimize energy.
print("Minimizing energy...")
openmm.LocalEnergyMinimizer.minimize(context, 10.0)

# Run dynamics.
state = context.getState(getEnergy=True)
potential_energy = state.getPotentialEnergy()
print("Initial protonation states: %s   %12.3f kcal/mol" % (str(mc_titration.getTitrationStates()), potential_energy / units.kilocalories_per_mole))
for iteration in range(niterations):
    # Run some dynamics.
    initial_time = time.time()
    integrator.step(nsteps)
    state = context.getState(getEnergy=True)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    print("  %.3f s elapsed for %d steps of dynamics" % (elapsed_time, nsteps))

    # Attempt protonation state changes.
    initial_time = time.time()
    mc_titration.update(context)
    debuglogger = debuglogger.append(mc_titration.adapt_weights(context, debuglogger=True), ignore_index=True)
    # mc_titration.adapt_weights(context)
    state = context.getState(getEnergy=True)
    final_time = time.time()
    elapsed_time = final_time - initial_time
    print("  %.3f s elapsed for %d titration trials" % (elapsed_time, mc_titration.getNumAttemptsPerUpdate()))

    # Show titration states.
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    print("Iteration %5d / %5d:    %s   %12.3f kcal/mol (%d / %d accepted)" % (
        iteration, niterations, str(mc_titration.getTitrationStates()), potential_energy / units.kilocalories_per_mole,
        mc_titration.naccepted, mc_titration.nattempted))
    log.write(str(mc_titration.getTitrationStates()[0]) + "\n")

debuglogger.to_csv("Tyrosinewl.csv")
