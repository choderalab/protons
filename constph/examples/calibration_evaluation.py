from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph import get_data
from constph.constph import MonteCarloTitration

# Import one of the standard systems.
temperature = 300.0 * unit.kelvin
pressure = 1.0 * unit.atmospheres
timestep = 1.0 * unit.femtoseconds
collision_rate = 9.1 / unit.picoseconds
pH = 5.0
platform_name = 'CPU'
testsystems = get_data('.','calibration-systems')
positions = openmm.XmlSerializer.deserialize(open('{}/as4-implicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
system = openmm.XmlSerializer.deserialize(open('{}/as4-implicit.sys.xml'.format(testsystems)).read())
prmtop = app.AmberPrmtopFile('{}/as4-implicit.prmtop'.format(testsystems))
cpin_filename = '{}/as4-implicit.cpin'.format(testsystems)
calibration_settings = dict()
calibration_settings["temperature"] = temperature
calibration_settings["timestep"] = timestep
calibration_settings["pressure"] = pressure
calibration_settings["collision_rate"] = collision_rate
calibration_settings["pH"] = pH
calibration_settings["solvent"] = "explicit"
calibration_settings["nsteps_per_trial"] = 25

integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename,
                                   integrator, debug=False,
                                   pressure=None, nsteps_per_trial=25, implicit=True)

print(mc_titration.calibrate(calibration_settings, iterations=1000, mc_every=10, weights_every=1))
# platform = openmm.Platform.getPlatformByName('CPU')
# context = openmm.Context(system, mc_titration.compound_integrator, platform)
# context.setPositions(positions)  # set to minimized positions
# integrator.step(10)  # MD
# mc_titration.update(context)  # protonation