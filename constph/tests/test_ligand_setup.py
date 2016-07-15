from constph.ligutils import parametrize_ligand, _TitratableForceFieldCompiler
from . import get_data
from openmoltools.schrodinger import is_schrodinger_suite_installed
from openmoltools.amber import find_gaff_dat

try:
    find_gaff_dat()
    found_gaff = True
except ValueError:
    found_gaff = False

from unittest import skipIf, TestCase


class LigandXmlTestCase(TestCase):
    
    @skipIf(not is_schrodinger_suite_installed() or not found_gaff, "This test requires Schrodinger's suite and gaff")
    def test_ligand_cphxml(self):
        """
        Run epik on a ligand and parametrize its isomers
        """
        import sys

        parametrize_ligand(get_data("ligand_allH.mol2", "testsystems"), "/tmp/ligand-isomers.xml", pH=4.5)

    def test_xml_compilation(self):
        """
        Compile an xml file for the isomers
        """
        xmlfile = get_data("isomers.xml", "testsystems/ligand_xml")
        m = _TitratableForceFieldCompiler(xmlfile)
        m.write('/tmp/isomers.cph.xml')
