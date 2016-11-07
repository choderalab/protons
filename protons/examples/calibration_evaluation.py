"""Script to run the complete API including calibration """
from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from protons import get_data
from protons.logger import log
from protons import AmberProtonDrive
from protons.calibration import Histidine
import numpy as np
import openmmtools
import logging

log.setLevel(logging.INFO)

# Import one of the standard systems.
temperature = 300.0 * unit.kelvin
pressure = None
timestep = 1.0 * unit.femtoseconds
collision_rate = 9.1 / unit.picoseconds
pH = 7.0
platform_name = 'CUDA'

# Get histidine files from the calibration system directory
testsystems = get_data('.', 'calibration-systems')
positions = openmm.XmlSerializer.deserialize(open('{}/hip-implicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
system = openmm.XmlSerializer.deserialize(open('{}/hip-implicit.sys.xml'.format(testsystems)).read())
prmtop = app.AmberPrmtopFile('{}/hip-implicit.prmtop'.format(testsystems))
cpin_filename = '{}/hip-implicit.cpin'.format(testsystems)

# integrator = openmmtools.integrators.VVVRIntegrator(temperature, collision_rate, timestep)
integrator = openmmtools.integrators.VelocityVerletIntegrator(timestep)
mc_titration = AmberProtonDrive(system, temperature, pH, prmtop, cpin_filename, integrator, debug=False, pressure=pressure, ncmc_steps_per_trial=0, implicit=True)
if platform_name:
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, mc_titration.compound_integrator, platform)
else:
    context = openmm.Context(system, mc_titration.compound_integrator)

context.setPositions(positions)
context.setVelocitiesToTemperature(temperature)

log.info("Calibrating")
log.info("Restypes by index %s", mc_titration._detect_residues()[0])
mc_titration.calibrate(platform_name=platform_name)

# Example of how to read in pre-equilibrated values.
#mc_titration.import_gk_values({'hip': np.array([1.60072121,   5.50168959,  13.32546669])})



niter = 20000 # .6 ns
mc_freq = 6
counts = {0: 0, 1: 0, 2: 0}
for iteration in range(1, niter):
    if iteration % (niter/100) == 0:
        print(100* iteration/niter, "%")
    integrator.step(mc_freq)
    mc_titration.update(context)  # protonation
    counts[mc_titration._get_titration_state(0)] += 1

totcounts = sum(counts.values())
for key in counts.keys():
    counts[key] /= totcounts


print("Target:", Histidine(pH).populations())
print("Observed", counts)
