"""Script to run the complete API including calibration """
from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph import get_data
from constph.logger import logger
from constph.constph import MonteCarloTitration
from constph.calibration import Histidine
import openmmtools
import logging

logger.setLevel(logging.INFO)

# Import one of the standard systems.
temperature = 300.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
timestep = 1.0 * unit.femtoseconds
collision_rate = 9.1 / unit.picoseconds
pH = 7.0
platform_name = 'OpenCL'

# Get histidine files from the calibration system directory
testsystems = get_data('.', 'calibration-systems')
positions = openmm.XmlSerializer.deserialize(open('{}/hip-implicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
system = openmm.XmlSerializer.deserialize(open('{}/hip-implicit.sys.xml'.format(testsystems)).read())
prmtop = app.AmberPrmtopFile('{}/hip-implicit.prmtop'.format(testsystems))
cpin_filename = '{}/hip-implicit.cpin'.format(testsystems)

integrator = openmmtools.integrators.VVVRIntegrator(temperature, collision_rate, timestep)
mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, integrator, debug=False, pressure=None, nsteps_per_trial=0, implicit=True)
if platform_name:
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, mc_titration.compound_integrator, platform)
else:
    context = openmm.Context(system, mc_titration.compound_integrator)
context.setPositions(positions)  # set to minimized positions

logger.debug("Calibrating")
mc_titration.calibrate(platform_name=platform_name, threshold=1.e-7, t0=100)


niter = 1000 # 1 ns
mc_freq = 1000
counts = {0: 0, 1 : 0, 2 : 0}
for iteration in range(1, niter):
    if iteration % (niter/100) == 0:
        print(iteration/niter, "\%")
    integrator.step(mc_freq)
    mc_titration.update(context)  # protonation
    counts[mc_titration.getTitrationState(0)] += 1

totcounts = sum(counts.values())
for key in counts.keys():
    counts[key] /= totcounts


print("Target:", Histidine(pH).weights())
print("Observed", counts)
