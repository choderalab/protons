# coding=utf-8
"""
This script parses an ffxml file, and creates a hydrogen definitions file for all residues.
Note that it doesn't add terminal specifications
"""
from lxml import etree
xmltree = etree.parse("../protons.xml", etree.XMLParser(remove_blank_text=True, remove_comments=True))
hydrogen_definitions_tree = etree.fromstring('<Residues/>')
# Detect all hydrogens by element and store them in a set
hydrogen_types = set()
for atomtype in xmltree.xpath('AtomTypes/Type'):
    if atomtype.get('element') == "H":
        hydrogen_types.add(atomtype.get('name'))

# Loop through each reisdue and find all the hydrogens
for residue in xmltree.xpath('Residues/Residue'):
    hydrogen_file_residue = etree.fromstring("<Residue/>")
    hydrogen_file_residue.set('name', residue.get('name'))
    # enumerate hydrogens in this list
    hydrogens = list()
    # Loop through atoms to find all hydrogens
    for atom in residue.xpath('Atom'):
        if atom.get('type') in hydrogen_types:
            # Find the parent atom
            for bond in residue.xpath('Bond'):
                atomname1 = bond.get('atomName1')
                atomname2 = bond.get('atomName2')
                # There should be only one bond containing this hydrogen
                if atom.get('name') == atomname1:
                    hydrogens.append(tuple([atomname1, atomname2]))
                    break
                elif atom.get('name') == atomname2:
                    hydrogens.append(tuple([atomname2, atomname1]))
                    break

    # Loop through all hydrogens, and create definitions
    for name, parent in hydrogens:
        h_xml = etree.fromstring("<H/>")
        h_xml.set("name", name)
        h_xml.set("parent", parent)
        # Not using terminal, amber FF has special residues for these.
        # Leaving this section in case useful
        # Terminal atoms can be recognized if they're bond to these atoms
        # if parent in ["O5'", "C"]:
        #     h_xml.set("terminal", "C")
        # elif parent in ["O3'", "N"] and name != "H":
        #     h_xml.set("terminal", "N")
        hydrogen_file_residue.append(h_xml)
    hydrogen_definitions_tree.append(hydrogen_file_residue)
# Write output
xmlstring = etree.tostring(hydrogen_definitions_tree,encoding="utf-8", pretty_print=True, xml_declaration=False)
xmlstring = xmlstring.decode("utf-8")
with open('../hydrogens-protons.xml', 'w') as fstream:
    fstream.write(xmlstring)