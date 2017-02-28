from simtk import unit
from simtk.openmm import app, openmm
from protons.ff import protonsff, ions_tip3p, bonds


def make_explicit_system(pdb_filename='his_ala_his.pdb', outfile='his_ala_his'):
    """Solvate a pdb file and minimize the system"""

    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    outfile1 = open('{}.sys.xml'.format(outfile), 'w')
    outfile2 = open('{}.state.xml'.format(outfile), 'w')
    app.Topology.loadBondDefinitions(bonds)
    pdb = app.PDBFile(pdb_filename)
    forcefield = app.ForceField(protonsff, ions_tip3p, 'tip3p.xml')
    integrator = openmm.LangevinIntegrator(300 * unit.kelvin, 1.0 / unit.picoseconds,
                                           2.0 * unit.femtoseconds)

    integrator.setConstraintTolerance(0.00001)

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(forcefield, boxSize=openmm.Vec3(3.5, 3.5, 3.5) * unit.nanometers, model='tip3p', ionicStrength=0.1*unit.molar, positiveIon="Na+", negativeIon="Cl-")
    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME,
                                     nonbondedCutoff=1.0 * unit.nanometers, constraints=app.HBonds, rigidWater=True,
                                     ewaldErrorTolerance=0.0005)
    system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    simulation.minimizeEnergy()

    outfile1.write(openmm.XmlSerializer.serialize(simulation.system))
    positions = simulation.context.getState(getPositions=True)
    outfile2.write(openmm.XmlSerializer.serialize(positions))
    app.PDBxFile.writeFile(simulation.topology, modeller.positions, open('{}-solvated-minimized.cif'.format(outfile), 'w'))

if __name__ == "__main__":
    make_explicit_system()
