# Simulation directories, content description

## tleap input files

protein.pdb
tleap.in
ligand.gaff.mol2
ligand.gaff.frcmod

## simulation input files
complex.cpin
complex.pdb
complex.inpcrd
complex.prmtop
run_simulation.py - the script that ran everything, set up to run for 60 ns total, though a decent portion will not be finished by that time.
submit.sh - batch script

## output files
trajectory.dcd - system trajectory written out by simulation
states_output.dat - csv file with residue names in the headers, current state index as value per row. If you need a reminder of each state index physical meaning,
use `cpinutil.py --describe HIP` for example.

failed.txt - appears if simulation ended in an error instead of completing
completed.txt - appears if simulation ended successfully.
