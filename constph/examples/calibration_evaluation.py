from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph import get_data
from constph.constph import MonteCarloTitration
import logging

# Import one of the standard systems.
temperature = 300.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
timestep = 1.0 * unit.femtoseconds
collision_rate = 9.1 / unit.picoseconds
pH = 8.6
platform_name = 'CUDA'
testsystems = get_data('.', 'calibration-systems')
positions = openmm.XmlSerializer.deserialize(open('{}/tyr-implicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
system = openmm.XmlSerializer.deserialize(open('{}/tyr-implicit.sys.xml'.format(testsystems)).read())
prmtop = app.AmberPrmtopFile('{}/tyr-implicit.prmtop'.format(testsystems))
cpin_filename = '{}/tyr-implicit.cpin'.format(testsystems)
integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, integrator, debug=True, pressure=None, nsteps_per_trial=50, implicit=True)
datapoints = list()
frens = dict(tyr=[0, 108.0])
dps = 3

for x in range(dps):
    print(x)
    frens = dict(mc_titration.calibrate(platform_name="CUDA", iterations=100000, mc_every=100, weights_every=1, scheme='global'))
    datapoints.append(frens["tyr"])


import matplotlib
matplotlib.use('Agg')
import seaborn as sns

sns.plt.plot(range(dps), datapoints)
sns.plt.title("tyr calibration")
sns.plt.xlabel("updates * 1000")
sns.plt.ylabel("Reference energies")

sns.plt.legend()
sns.plt.savefig("tyr-calibrated_energies.png")


