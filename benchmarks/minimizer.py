from simtk.openmm.app import *
from simtk.openmm import *
from simtk.unit import *
from sys import stdout, argv
from os import getcwd

prmtop = AmberPrmtopFile('complex.prmtop')
inpcrd = AmberInpcrdFile('complex.inpcrd')

if getcwd().split(sep='-')[-1] == 'explicit':
    print("explicit system, using PME")
    system = prmtop.createSystem(nonbondedMethod=PME, nonbondedCutoff=1 * nanometer,
                                 constraints=HBonds)
else:
    print("implicit system, using NoCutoff")
    system = prmtop.createSystem(nonbondedMethod=NoCutoff, constraints=HBonds, implicitSolvent=OBC2)

integrator = LangevinIntegrator(300 * kelvin, 1 / picosecond, 0.002 * picoseconds)
platform = Platform.getPlatformByName('CUDA')
properties = {'CudaPrecision': 'single'}
simulation = Simulation(prmtop.topology, system, integrator, platform, properties)
simulation.context.setPositions(inpcrd.positions)
if inpcrd.boxVectors is not None:
    simulation.context.setPeriodicBoxVectors(*inpcrd.boxVectors)
simulation.minimizeEnergy()
state = simulation.context.getState(getPositions=True)
pos = state.getPositions()

PDBFile.writeFile(prmtop.topology, pos, open("min.pdb",'w'))
