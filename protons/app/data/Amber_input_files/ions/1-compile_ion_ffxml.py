# coding: utf-8
from parmed.amber import AmberParameterSet
from parmed.openmm import OpenMMParameterSet
from glob import glob

lib = "Amber_input_files/ions/atomic_ions.lib"

frcmods = glob("Amber_input_files/ions/frcmod*")
for frcmod in frcmods:
    solvent_model = frcmod.split("_")[-1]
    amberset = AmberParameterSet(lib, frcmod)
    openmmset = OpenMMParameterSet.from_parameterset(amberset)
    openmmset.write("Amber_input_files/ions/raw_ions_{}-tmp.xml".format(solvent_model))
