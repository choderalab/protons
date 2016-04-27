"""Example ligand parametrization script."""
from constph.ligutils import parametrize_ligand, _TitratableForceFieldCompiler
from constph.tests import get_data
from constph.logger import logger
import logging

logger.setLevel(logging.DEBUG)

outfile = parametrize_ligand(get_data("ligand_allH.mol2", "testsystems"), "ligand-isomers.xml", pH=4.5)

