"""Example ligand parametrization script."""
from constph.ligutils import parametrize_ligand
from constph.constph import MonteCarloTitration
from constph.tests import get_data
from constph.logger import logger
from openmmtools.integrators import VelocityVerletIntegrator
from simtk import openmm, unit
from simtk.openmm import app
from sys import stdout
import logging, os

logger.setLevel(logging.INFO)

# System conditions
temperature = 300.0 * unit.kelvin
pH = 7.0
pressure = None

outfile = parametrize_ligand(get_data("imidazole.mol2", "testsystems"), "ligand-isomers.xml", pH=7.4, max_antechambers=1)
pdb = app.PDBFile(get_data("imidazole.pdb", "testsystems"))
forcefield = app.ForceField("ligand-isomers.xml")

system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
integrator = VelocityVerletIntegrator(1.0*unit.femtoseconds)

mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, integrator, debug=False, pressure=pressure, ncmc_steps_per_trial=0, implicit=True)



platform = openmm.Platform.getPlatformByName('CPU')

simulation = app.Simulation(pdb.topology, system, integrator, platform,)
simulation.context.setPositions(pdb.positions)

print('Minimizing...')
simulation.minimizeEnergy(tolerance=0.001*unit.kilojoule/unit.mole)


simulation.context.setVelocitiesToTemperature(temperature)
print('Equilibrating...')
simulation.reporters.append(app.PDBReporter('trajectory.pdb', 1))
simulation.reporters.append(app.StateDataReporter(stdout, 1, step=True,
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True,
    speed=True, totalSteps=1000, separator='\t'))

print('Running Production...')
simulation.step(1000)
print('Done!')