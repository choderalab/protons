#!/usr/bin/env python
"""
Example implementation of a simulation of a box of water where the number of counterions can fuctuate in a box of water using the semi-grand canonical ensemble.

Constant salt class incomplete, and script demonstrates desired usage.

"""

import numpy as np
from numpy import random
from simtk import openmm, unit
from simtk.openmm import app
from sys import stdout
import constsalt            # Package still under constrution!!!!

# CONSTANTS
kB = unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA
pressure = 1*unit.atmospheres
temperature = 300*unit.kelvin
delta_chem = -160*unit.kilojoule_per_mole       # The free energy penalty to turn 2 waters into Na and Cl. To be calibrated against concentration.


#Loading a premade water box:
pdb = app.PDBFile('waterbox.pdb')
forcefield = app.ForceField('tip3p.xml')
system = forcefield.createSystem(pdb.topology,nonbondedMethod=app.PME, nonbondedCutoff=1.0*unit.nanometer, constraints=app.HBonds)
integrator = openmm.LangevinIntegrator(temperature, 1/unit.picosecond, 0.002*unit.picoseconds)
system.addForce(openmm.MonteCarloBarostat(pressure, temperature, 25))
simulation = app.Simulation(pdb.topology, system, integrator)
simulation.context.setPositions(pdb.positions)

iterations = 100       # Number of rounds of MD and constant salt moves
nsteps = 100           # Amount of MD steps per iteration (not to be confused with number of steps in NCMC)
nattempts = 10       # Number of identity exchanges for water and ions.

print "Initializing constant salt class"
mc_constant_salt = constsalt.ConstantSalt(system=system,topology=pdb.topology,temperature=temperature,delta_chem=delta_chem,integrator=integrator,pressure=pressure,nattempts_per_update=nattempts, debug=False)

print "Minimizing energy..."
simulation.minimizeEnergy(maxIterations=25)

simulation.context.setVelocitiesToTemperature(temperature)

print "Running simulation..."
for i in range(iterations):
    simulation.reporters.append(app.PDBReporter('output.pdb',10))
    simulation.reporters.append(app.StateDataReporter(stdout,10,step=True,potentialEnergy=True,volume=True))
    simulation.step(nsteps)
    mc_constant_salt.update(simulation.context,nattempts=nattempts)

