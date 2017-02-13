# coding: utf-8
from parmed.amber import AmberParameterSet
from parmed.openmm import OpenMMParameterSet
from glob import glob
lib = "atomic_ions.lib"

frcmods = glob("frcmod*")
for frcmod in frcmods:
    solvent_model = frcmod.split(sep="_")[-1]
    amberset = AmberParameterSet(lib, frcmod)
    openmmset = OpenMMParameterSet.from_parameterset(amberset)
    openmmset.write("raw_ions_{}.xml".format(solvent_model))
