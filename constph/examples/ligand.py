"""Example ligand parametrization script."""
from constph.ligutils import parametrize_ligand, _TitratableForceFieldCompiler
from constph.tests import get_data
from constph.logger import logger
from simtk import openmm, unit
from simtk.openmm import app
from sys import stdout
import logging

logger.setLevel(logging.DEBUG)


outfile = parametrize_ligand(get_data("dasatinib_allH.mol2", "testsystems"), "ligand-isomers.xml", pH=7.4, max_antechambers=4)
# _TitratableForceFieldCompiler(get_data("intermediate.xml", "testsystems")).write("ligand-isomers.xml")
pdb = app.PDBFile(get_data("dasatinib_allH.pdb", "testsystems"))
forcefield = app.ForceField("ligand-isomers.xml")

system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
#
integrator = openmm.LangevinIntegrator(300*unit.kelvin, 1.0/unit.picoseconds,
    1.0*unit.femtoseconds)
integrator.setConstraintTolerance(0.00001)

platform = openmm.Platform.getPlatformByName('CPU')

simulation = app.Simulation(pdb.topology, system, integrator, platform,)
simulation.context.setPositions(pdb.positions)

print('Minimizing...')
simulation.minimizeEnergy()

simulation.context.setVelocitiesToTemperature(300*unit.kelvin)
print('Equilibrating...')
simulation.step(5000)

simulation.reporters.append(app.DCDReporter('trajectory.dcd', 1000))
simulation.reporters.append(app.StateDataReporter(stdout, 1000, step=True,
    potentialEnergy=True, temperature=True, progress=True, remainingTime=True,
    speed=True, totalSteps=1000, separator='\t'))

print('Running Production...')
simulation.step(1000)
print('Done!')