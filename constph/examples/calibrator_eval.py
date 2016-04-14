from __future__ import print_function
from simtk import unit, openmm
from constph.calibration import AminoAcidCalibrator
import logging
import numpy as np

settings = dict()
settings["temperature"] = 300.0 * unit.kelvin
settings["timestep"] = 1.0 * unit.femtosecond
settings["pressure"] = 1.0 * unit.atmospheres
settings["collision_rate"] = 9.1 / unit.picoseconds
settings["pH"] = 3.4
settings["solvent"] = "implicit"
settings["nsteps_per_trial"] = 0
settings["platform_name"] = "CUDA"
datapoints = dict(HIP=[], HID=[], HIE=[],idx=[])
aac = AminoAcidCalibrator("tyr", settings, minimize=False, guess_free_energy=[0.0, 104.5])
print(aac.target_weights)
for i,x in enumerate(aac.calibrate(iterations=10000000, mc_every=100, weights_every=1, scheme='global'), start=1):
    datapoints['HIP'].append(x[0])
    datapoints['HID'].append(x[1])
    # datapoints['HIE'].append(x[2])
    datapoints['idx'].append(i)
    if i % 5000 == 0:
        print("detecting equil")
        d_equil = np.average(np.gradient(datapoints['HID'][-5000:], 2))
        print(d_equil)
        if abs(d_equil) <= 1.e-6:
            print(np.average(datapoints["HID"][-5000:]))
            break


print(aac.titration.naccepted / aac.titration.nattempted)

import matplotlib
matplotlib.use('Agg')
import seaborn as sns

f, axarr = sns.plt.subplots(3, sharex=True)
axarr[0].plot(datapoints['idx'], datapoints['HIP'], label='HIP')
sns.plt.legend()
axarr[1].plot(datapoints['idx'], datapoints['HID'], label='HID')
axarr[1].scatter(datapoints['idx'][-5000:], datapoints['HID'][-5000:], label='HID', color='red')

sns.plt.legend()
# axarr[2].plot(datapoints['idx'], datapoints['HIE'], label='HIE')
sns.plt.legend()
sns.plt.savefig("his-calibrated_energies.png")
