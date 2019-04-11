from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile


def prepare_protein(ifile, ofile):

    fixer = PDBFixer(filename=ifile)
    fixer.findMissingResidues()
    fixer.findNonstandardResidues()
    fixer.findMissingAtoms()
    fixer.addMissingAtoms()
    fixer.addMissingHydrogens(7.0)
    fixer.findNonstandardResidues()
    fixer.replaceNonstandardResidues()
    PDBFile.writeFile(fixer.topology, fixer.positions, open(ofile, 'w'))
