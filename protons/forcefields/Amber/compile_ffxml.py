# coding: utf-8
from parmed.amber import AmberParameterSet
from parmed.openmm import OpenMMParameterSet
from glob import glob
libs = glob("*.lib")
dats = glob("*.dat")
frcmods = glob("frcmod*")
all_files = list()
all_files.extend(dats)
all_files.extend(libs)
all_files.extend(frcmods)
print(all_files)
amberset = AmberParameterSet(*all_files)
openmmset = OpenMMParameterSet.from_parameterset(amberset)
openmmset.write("raw-constph.xml")
