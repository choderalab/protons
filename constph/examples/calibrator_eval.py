"""Script to run the calibrator"""
from __future__ import print_function
from simtk import unit, openmm
from constph.calibration import CalibrationSystem
import logging
from constph.constph import logger
import numpy as np

logger.setLevel(logging.INFO)

settings = dict()
settings["temperature"] = 300.0 * unit.kelvin
settings["timestep"] = 1.0 * unit.femtosecond
settings["pressure"] = 1.0 * unit.atmospheres
settings["collision_rate"] = 9.1 / unit.picoseconds
settings["pH"] = 7.4
settings["solvent"] = "implicit"
settings["nsteps_per_trial"] = 0
settings["platform_name"] = "CUDA"
datapoints = dict(HIP=[], HID=[], HIE=[],idx=[])
aac = CalibrationSystem("hip", settings, minimize=True, guess_free_energy=[0.0, 0.0, 0.0])
print(aac.target_weights)

window = 1000
for i,x in enumerate(aac.sams_till_converged(threshold=1.e-6, mc_every=100, gk_every=1, window=window, scheme='global'), start=1):
    datapoints['HIP'].append(x[0])
    datapoints['HID'].append(x[1])
    datapoints['HIE'].append(x[2])
    datapoints['idx'].append(i)

print(aac.titration.naccepted / aac.titration.nattempted)

from constph.diagnostics import plot_sams_trace

plot_sams_trace(datapoints["HID"], title="His-Delta Tautomer", ylabel="beta * zeta_2", window=window, filename="hid-calibrated.png")
plot_sams_trace(datapoints["HIE"], window=window,title="His-Eps Tautomer", ylabel="beta * zeta_3", filename="hie-calibrated.png")

