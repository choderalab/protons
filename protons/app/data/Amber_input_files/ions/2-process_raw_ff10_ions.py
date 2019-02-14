# coding=utf-8
"""
This script parses the raw parmed output, and removes residues for which parameters don't exist.
"""
from lxml import etree

from simtk.openmm import app

for inputfile in ["raw_ions_spce.xml", "raw_ions_tip3p.xml", "raw_ions_tip4pew.xml"]:
    shortnames = inputfile[4:-4]  # slice off raw tag and .xml
    xml_parser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
    xmltree = etree.parse("Amber_input_files/ions/" + inputfile, xml_parser)

    # Create set of all available atom types
    types = set()
    for atomtype in xmltree.xpath("/ForceField/AtomTypes/Type"):
        types.add(atomtype.get("name"))

    # Delete residues if the type does not exist
    for residue in xmltree.xpath("/ForceField/Residues/Residue"):
        for atom in residue.xpath("Atom"):
            if atom.get("type") not in types:
                residue.getparent().remove(residue)

    # Write out to file
    xmlstring = etree.tostring(
        xmltree, encoding="utf-8", pretty_print=True, xml_declaration=False
    )
    xmlstring = xmlstring.decode("utf-8")

    output_name = "{}-tmp.xml".format(shortnames)
    with open(output_name, "w") as fstream:
        fstream.write(xmlstring)

    # Validate that the resulting file can be read by openmm
    y = app.ForceField(output_name)
