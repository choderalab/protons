from simtk import unit, openmm
from simtk.openmm import app
from protons import ProtonDrive
import numpy as np
from openmmtools.integrators import VelocityVerletIntegrator
from protons import log
import pickle
import time

# Import one of the standard systems.
temperature = 300.0 * unit.kelvin
timestep = 1.0 * unit.femtoseconds
pH = 7.4

platform = openmm.Platform.getPlatformByName('CUDA')

prmtop = app.AmberPrmtopFile('complex.prmtop')
inpcrd = app.AmberInpcrdFile('complex.inpcrd')
positions = inpcrd.getPositions()
topology = prmtop.topology
cpin_filename = 'complex.cpin'
integrator = VelocityVerletIntegrator(timestep)

log.info("Creating system")
# Create a system from the AMBER prmtop file
system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

log.info("Done creating system")
# Create the driver that will track the state of the simulation and provides the updating API
log.info("Creating protondrive")
driver = ProtonDrive(system, temperature, pH, prmtop, cpin_filename, integrator, pressure=None, ncmc_steps_per_trial=0,
                     implicit=True)
log.info("Finished drive setup")

# Create an OpenMM simulation object as one normally would.
log.info("Creating simulation object")
simulation = app.Simulation(topology, system, driver.compound_integrator, platform)
simulation.context.setPositions(positions)
simulation.context.setVelocitiesToTemperature(temperature)


try:
    with open("calibration.pickle", "rb") as precalibrated:
        calibration_results = pickle.load(precalibrated)
        log.info("Found precalibrated results. ")
except IOError:
    log.info("Calibrating")
    calibration_results = driver.calibrate()
    pickle.dump(calibration_results, open("calibration.pickle", "wb"))
    log.info("Finished calibration")

driver.import_gk_values(calibration_results)

# 60 ns, 10000 state updates
log.info("Entering main md loop")
niter, mc_frequency = 10000, 6000
simulation.reporters.append(app.DCDReporter('trajectory.dcd', mc_frequency))

try:
    for iteration in range(1, niter):
        simulation.step(mc_frequency)  # MD
        simulation.saveState('running.xml')
        driver.update(simulation.context)  # protonation
        if iteration % 10 == 0:
            log.info("%.1f", float(iteration) / float(niter))

except:
    log.error("Your simulation has crashed, writing out error files")
    simulation.saveState(time.strftime('crash.xml@%X_%x'))
    pickle.dump(driver, open(time.strftime("protondrive.pickle@%X_%x"), "wb"))
    raise
