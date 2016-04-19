from __future__ import print_function
from simtk import unit, openmm
from constph.calibration import AminoAcidCalibrator
import logging
from constph.constph import logger
import numpy as np

logger.setLevel(logging.DEBUG)

settings = dict()
settings["temperature"] = 300.0 * unit.kelvin
settings["timestep"] = 1.0 * unit.femtosecond
settings["pressure"] = 1.0 * unit.atmospheres
settings["collision_rate"] = 9.1 / unit.picoseconds
settings["pH"] = 7.4
settings["solvent"] = "explicit"
settings["nsteps_per_trial"] = 0
settings["platform_name"] = "CUDA"
datapoints = dict(HIP=[], HID=[], HIE=[],idx=[])
aac = AminoAcidCalibrator("hip", settings, minimize=True, guess_free_energy=[0.0, 0.0, 0.0])
print(aac.target_weights)
# for i,x in enumerate(aac.calibrate(iterations=10000000, mc_every=100, zeta_every=1, scheme='global'), start=1):
#     datapoints['HIP'].append(x[0])
#     datapoints['HID'].append(x[1])
#     # datapoints['HIE'].append(x[2])
#     datapoints['idx'].append(i)
#     if i % 5000 == 0:
#         print("detecting equil")
#         d_equil = np.average(np.gradient(datapoints['HID'][-5000:], 2))
#         print(d_equil)
#         if abs(d_equil) <= 1.e-6:
#             print(np.average(datapoints["HID"][-5000:]))
#             break
window = 1000
for i,x in enumerate(aac.calibrate_till_converged(threshold=1.e-6, mc_every=100, zeta_every=1, window=window, scheme='global'), start=1):
    datapoints['HIP'].append(x[0])
    datapoints['HID'].append(x[1])
    datapoints['HIE'].append(x[2])
    datapoints['idx'].append(i)

print(aac.titration.naccepted / aac.titration.nattempted)

import matplotlib
matplotlib.use('Agg')
import seaborn as sns


def add_subplot_axes(ax,rect):
    #http://stackoverflow.com/a/17479417
    with sns.axes_style("white"):
        fig = sns.plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position  = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]  # <= Typo was here
        subax = fig.add_axes([x,y,width,height])
        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2]**0.25
        y_labelsize *= rect[3]**0.25
        subax.xaxis.set_tick_params(labelsize=x_labelsize)
        subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


f, axarr = sns.plt.subplots(2, sharex=True)
# axarr[0].plot(datapoints['idx'], datapoints['HIP'], label='HIP')
# sns.plt.legend()
axarr[0].plot(datapoints['idx'], datapoints['HID'],)
axarr[0].scatter(datapoints['idx'][-window:], datapoints['HID'][-window:],  color='red')
axarr[0].set_title("Histidine-delta tautomer")
axarr[0].set_ylabel("g_2, ref", rotation=0)
axarr[0].yaxis.set_label_position('right')
axarr[0].set_ylim(datapoints['HID'][-1]-.5,datapoints['HID'][-1]+.5)
a0 = add_subplot_axes(axarr[0], [0.1,0.1,0.25,0.2])
a0.plot(datapoints['idx'], datapoints['HID'], )
a0.set_xlim(datapoints['idx'][0]-1, datapoints['idx'][-1]+1)
a0.scatter(datapoints['idx'][-window:], datapoints['HID'][-window:], color='red')
sns.plt.legend()
axarr[1].plot(datapoints['idx'], datapoints['HIE'], )
axarr[1].scatter(datapoints['idx'][-window:], datapoints['HIE'][-window:], color='red')
axarr[1].set_title("Histidine-epsilon tautomer")
axarr[1].set_ylabel("g_3, ref", rotation=0)
axarr[1].yaxis.set_label_position('right')
axarr[1].set_xlim(datapoints['idx'][-window]-1, datapoints['idx'][-1] + 1)
axarr[1].set_ylim(datapoints['HIE'][-1]-.5,datapoints['HIE'][-1]+.5)
a1 = add_subplot_axes(axarr[1], [0.1,0.1,0.25,0.2])
a1.plot(datapoints['idx'], datapoints['HIE'], )
a1.scatter(datapoints['idx'][-window:], datapoints['HIE'][-window:], color='red')
a1.set_xlim(datapoints['idx'][0]-1, datapoints['idx'][-1]+1)
sns.plt.legend()
sns.plt.savefig("his-calibrated_energies.png")
