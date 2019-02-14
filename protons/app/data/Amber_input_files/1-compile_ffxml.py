# coding: utf-8
"""
This script takes all the lib, frcmod and dat files from the current directory, and compiles them into a single
ffxml file. It doesn't do any validation, and it might throw a lot of warnings. It is up to the discretion of the user to validate
the output

Common problems with the output:
You may run into residues for which atom types types are not available. This can happen if you load a .lib file,
but you lack the necessary .dat or frcmod file with the parameters.
"""

from parmed.amber import AmberParameterSet
from parmed.openmm import OpenMMParameterSet
from glob import glob

# Get ALL the amber source files in this directory.
libs = glob("Amber_input_files/*.lib")
dats = glob("Amber_input_files/*.dat")
frcmods = glob("Amber_input_files/frcmod*")
all_files = list()
all_files.extend(dats)
all_files.extend(libs)
all_files.extend(frcmods)
# For logging purposes
print(all_files)
amberset = AmberParameterSet(*all_files)
openmmset = OpenMMParameterSet.from_parameterset(amberset)
openmmset.write("Amber_input_files/raw-amber10-constph-tmp.xml")
