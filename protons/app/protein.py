from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
from .logger import log
import os
from collections import OrderedDict
from .. import app
from simtk.openmm import openmm
from simtk.unit import *
from ..app.integrators import GBAOABIntegrator
from typing import List, Dict, Tuple


def prepare_protein(ifile, ofile):


    log.info('Preparing protein file.')
    fixer = PDBFixer(filename=ifile)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    fixer.replaceNonstandardResidues()
    log.info('Writing new file to {}.'.format(ofile))
    PDBFile.writeFile(fixer.topology, fixer.positions, open(ofile, 'w'))


def prepare_protein_simulation_systems(
    protein_name : str,
    protein_ligand: str,
    ligand_resname : str,
    output_basename: str,
    ffxml: str = None,
    hxml: str = None,
    bxml: str = None,
    delete_old_H: bool = True,
    minimize: bool = True,
    box_size: app.modeller.Vec3 = None,
):
    """Add hydrogens to a residue based on forcefield and hydrogen definitons, and then solvate.

    Note that no salt is added. We use saltswap for this.

    Parameters
    ----------
    protein_ligand - a protein ligand pdb file to add hydrogens to and solvate.
    output_file - the basename for an output mmCIF file with the solvated system.
    ffxml - the forcefield file containing the residue definition,
        optional for CDEHKY amino acids, required for ligands.
    hxml - the hydrogen definition xml file,
        optional for CDEHKY amino acids, required for ligands.
    delete_old_H - delete old hydrogen atoms and add in new ones.
        Typically necessary for ligands, where hydrogen names will have changed during parameterization to match up
        different protonation states.
    minimize - perform an energy minimization on the system. Recommended.
    box_size - Vec3 of box vectors specified as in ``simtk.openmm.app.modeller.addSolvent``. 
    """

    # Load relevant template definitions for modeller, forcefield and topology
    if hxml is not None:
        app.Modeller.loadHydrogenDefinitions(hxml)
    if ffxml is not None:
        forcefield = app.ForceField(
            "amber10-constph.xml", "gaff.xml", ffxml, "tip3p.xml", "ions_tip3p.xml"
        )
    else:
        forcefield = app.ForceField(
            "amber10-constph.xml", "gaff.xml", "tip3p.xml", "ions_tip3p.xml"
        )

    _, extension = os.path.splitext(protein_ligand)
    if extension == ".pdb":
        pdb = app.PDBFile(protein_ligand)
    elif extension == ".cif":
        pdb = app.PDBxFile(protein_ligand)
    else:
        raise ValueError(
            f"Unsupported file extension {extension} for vacuum file. Currently supported: pdb, cif."
        )
    modeller = app.Modeller(pdb.topology, pdb.positions)

    log.info(pdb.topology)

    # The system will likely have different hydrogen names.
    # In this case its easiest to just delete and re-add with the right names based on hydrogen files
    if delete_old_H:
        log.info('Deleting hydrogens for ligand.')
        to_delete = [
            atom for atom in modeller.topology.atoms() if atom.element.symbol in ["H"] and str(atom.residue.name).upper() == str(ligand_resname).upper()
        ]
        modeller.delete(to_delete)
    log.info(modeller.topology)

    modeller.addHydrogens(forcefield=forcefield)
    log.info('New ligand hydrogen names.')
    for atom in modeller.topology.atoms():
        if str(atom.residue.name).upper() == str(ligand_resname).upper():
            log.info(atom)

    if box_size == None:
        padding = 1.2 * nanometers
    else:
        padding = None

    modeller.addSolvent(
        forcefield, model="tip3p", padding=padding, neutralize=False, boxSize=box_size
    )

    if minimize:
        system = forcefield.createSystem(
            modeller.topology,
            nonbondedMethod=app.PME,
            nonbondedCutoff=1.0 * nanometers,
            constraints=app.HBonds,
            rigidWater=True,
            ewaldErrorTolerance=0.0005,
        )
        system.addForce(openmm.MonteCarloBarostat(1.0 * atmosphere, 300.0 * kelvin))
        simulation = app.Simulation(modeller.topology, system, GBAOABIntegrator())
        simulation.context.setPositions(modeller.positions)

        simulation.minimizeEnergy()
        positions = simulation.context.getState(getPositions=True).getPositions()
    else:
        positions = modeller.positions

    app.PDBxFile.writeFile(
        modeller.topology, positions, open(f"{output_basename}/{protein_name}-{ligand_resname}.cif", "w")
    )
