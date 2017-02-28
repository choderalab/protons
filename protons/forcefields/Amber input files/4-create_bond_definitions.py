# coding=utf-8
"""
Create bond definitions for protons residues.
"""
# coding=utf-8
"""
This script parses an ffxml file, and creates a bonds definitions file for all residues.
"""
from lxml import etree
xmltree = etree.parse("../protons.xml", etree.XMLParser(remove_blank_text=True, remove_comments=True))
bond_definitions_tree = etree.fromstring('<Residues/>')

# Loop through each residue and add one entry for the N-terminal/P-terminal bond, and all other bonds
for residue in xmltree.xpath('Residues/Residue'):
    bond_file_residue = etree.fromstring("<Residue/>")
    bond_file_residue.set('name', residue.get('name'))

    # Loop through external bonds and look for the bonds we recognize
    for extbond in residue.xpath('ExternalBond'):
        if extbond.get("atomName") == "N":
            bond_file_residue.append(etree.fromstring('<Bond from="-C" to="N"/>'))
        elif extbond.get("atomName") == "P":
            bond_file_residue.append(etree.fromstring('<Bond from="-O3\'" to="P"/>'))


    # Loop through bonds
    for bond in residue.xpath('Bond'):
        bond_xml = etree.fromstring('<Bond/>')
        bond_xml.set('from', bond.get('atomName1'))
        bond_xml.set('to', bond.get('atomName2'))
        bond_file_residue.append(bond_xml)

    bond_definitions_tree.append(bond_file_residue)

# Write output
xmlstring = etree.tostring(bond_definitions_tree,encoding="utf-8", pretty_print=True, xml_declaration=False)
xmlstring = xmlstring.decode("utf-8")
with open('../bonds-protons.xml', 'w') as fstream:
    fstream.write(xmlstring)