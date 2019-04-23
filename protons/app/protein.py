from pdbfixer import PDBFixer
from simtk.openmm.app import PDBFile
from .logger import log


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
