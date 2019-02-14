# coding=utf-8
"""
This script parses an ffxml file, and creates a hydrogen definitions file for all residues.
Note that it doesn't add terminal specifications
"""
from lxml import etree

xmltree = etree.parse(
    "amber10-constph-tmp.xml",
    etree.XMLParser(remove_blank_text=True, remove_comments=True),
)
hydrogen_definitions_tree = etree.fromstring("<Residues/>")
# Detect all hydrogens by element and store them in a set
hydrogen_types = set()
for atomtype in xmltree.xpath("AtomTypes/Type"):
    if atomtype.get("element") == "H":
        hydrogen_types.add(atomtype.get("name"))


# Variants to add for histidine
his_variants = ["HIP", "HIE", "HID"]
# glutamic acid
glu_variants = ["GL4", "GLU", "GLH"]
# aspartic acid
asp_variants = ["AS4", "ASH", "AS2", "ASP"]


# Atoms that need variants specified for histidine
his_atoms = dict()
his_atoms["HE2"] = "HIP,HIE"
his_atoms["HD1"] = "HIP,HID"
# glutamic acid
glu_atoms = dict()
glu_atoms["HE2"] = "GLH"
glu_atoms["HE11"] = "GL4"
glu_atoms["HE12"] = "GL4"
glu_atoms["HE21"] = "GL4"
glu_atoms["HE22"] = "GL4"
# aspartic acid
asp_atoms = dict()
asp_atoms["HD2"] = "ASH"
asp_atoms["HD11"] = "AS4"
asp_atoms["HD12"] = "AS4"
asp_atoms["HD21"] = "AS4"
asp_atoms["HD22"] = "AS4, AS2"


# Loop through each reisdue and find all the hydrogens
for residue in xmltree.xpath("Residues/Residue"):
    hydrogen_file_residue = etree.fromstring("<Residue/>")
    resname = residue.get("name")
    if resname in his_variants:
        if not resname == "HIP":
            continue
        else:
            resname = "HIS"
    elif resname in glu_variants:
        if not resname == "GL4":
            continue
        else:
            resname = "GLU"
    elif resname in asp_variants:
        if not resname == "AS4":
            continue
        else:
            resname = "ASP"

    hydrogen_file_residue.set("name", resname)
    # enumerate hydrogens in this list
    hydrogens = list()
    # Loop through atoms to find all hydrogens
    for atom in residue.xpath("Atom"):
        if atom.get("type") in hydrogen_types:
            # Find the parent atom
            for bond in residue.xpath("Bond"):
                atomname1 = bond.get("atomName1")
                atomname2 = bond.get("atomName2")
                # There should be only one bond containing this hydrogen
                if atom.get("name") == atomname1:
                    hydrogens.append(tuple([atomname1, atomname2]))
                    break
                elif atom.get("name") == atomname2:
                    hydrogens.append(tuple([atomname2, atomname1]))
                    break

    if resname == "HIS":
        for variant in his_variants:
            var_element = etree.fromstring("<Variant/>")
            var_element.set("name", variant)
            hydrogen_file_residue.append(var_element)
    elif resname == "GLU":
        for variant in glu_variants:
            var_element = etree.fromstring("<Variant/>")
            var_element.set("name", variant)
            hydrogen_file_residue.append(var_element)
    elif resname == "ASP":
        for variant in asp_variants:
            var_element = etree.fromstring("<Variant/>")
            var_element.set("name", variant)
            hydrogen_file_residue.append(var_element)

    # Loop through all hydrogens, and create definitions
    for name, parent in hydrogens:
        h_xml = etree.fromstring("<H/>")
        h_xml.set("name", name)
        h_xml.set("parent", parent)

        if resname == "HIS" and name in his_atoms:
            h_xml.set("variant", his_atoms[name])
        elif resname == "GLU" and name in glu_atoms:
            h_xml.set("variant", glu_atoms[name])
        elif resname == "ASP" and name in asp_atoms:
            h_xml.set("variant", asp_atoms[name])

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
xmlstring = etree.tostring(
    hydrogen_definitions_tree,
    encoding="utf-8",
    pretty_print=True,
    xml_declaration=False,
)
xmlstring = xmlstring.decode("utf-8")
with open("hydrogens-amber10-constph-tmp.xml", "w") as fstream:
    fstream.write(xmlstring)
