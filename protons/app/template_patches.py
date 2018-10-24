"""Module provides a way of patching COOH in forcefield xml templates."""
from lxml import etree, objectify
import networkx as nx
import math

def patch_cooh(source:str, residue_name:str) -> str:
    """Add COOH statements to protons ffxml templates and fix hydroxy atom types.

    Parameters
    ----------

    source - location of the original ffxml file
    residue_name - name of the residue to patch

    Returns
    -------
    xml string containing the patched ffxml templates (as well as the remaining residues).

    """

    with open(source, 'r') as ffxml:
        ffxml = etree.fromstring(ffxml.read())

    # Keep track of all relevant atoms
    hydrogens = list()
    oxygens = list()
    carbons = list()

    # This list will contain dictionaries pointing to all detected COOH moieties in the molecule

    coohs = list()


    # undirected graph represents the complete openmm residue (ignoring protonation states)
    G = nx.Graph()
    for atom in ffxml.xpath("/ForceField/Residues/Residue[@name='{}']/Atom".format(residue_name)):
        G.add_node(atom.get("name"), type=atom.get('type'), charge=float(atom.get("charge")))

        # For convenience, assume only hydroxy hydrogen types are part of lowercase COOH
        if atom.attrib['type'] == 'ho':
            hydrogens.append(atom.get('name'))
        # No restriction besides the first letter in the type should be an lowercase  o
        elif atom.attrib['type'][0] == 'o':
            oxygens.append(atom.get('name'))
        # No restriction on carbon type besides first letter should be a lowercase c
        elif atom.attrib['type'][0] == 'c':
            carbons.append(atom.get('name'))

    # Add edges to graph using the bond records
    for bond in ffxml.xpath("/ForceField/Residues/Residue[@name='{}']/Bond".format(residue_name)):
        G.add_edge(bond.get("atomName1"), bond.get("atomName2"))

    # A COOH moiety is always a H-O=C=O path, with a branch on the C
    # Detecting COOH moiety in graph by going through all H and O atoms
    for h in hydrogens:
        for o in oxygens:

            # Find shortest path from H to O, and if length 4, check if its a HOCO moiety
            path = nx.shortest_path(G, source=h, target=o)
            if len(path) == 4:
                if path[1] in oxygens and path[2] in carbons:

                    # Discover the R group
                    neighbors = list(G.neighbors(path[2]))
                    neighbors.remove(path[1])
                    neighbors.remove(path[3])
                    # If no R group, or more than 1 R group, not sure what to do, dont call it as COOH
                    if len(neighbors) != 1:
                        continue
                    # Defined the moiety based on template atom names
                    else:
                        cooh = {
                            'HO' : path[0],
                            'OH' : path[1],
                            'CO' : path[2],
                            'OC' : path[3],
                            'R' : neighbors[0]
                        }
                        # Add to molecule wide list of templates
                        coohs.append(cooh)

    # Now that all COOH moieties were found in the main template, ensure oxygens have equal types
    # Then, detect dummy status
    for cooh in coohs:
        for atom in ffxml.xpath("/ForceField/Residues/Residue[@name='{}']/Atom".format(residue_name)):
            # Ensure oxygens have same type in deprotonated state.
            # Using oh because otherwise this results in lack of a C-O-H angle term
            if atom.get("name") in (cooh["OH"], cooh["OC"]):
                atom.set("type", 'oh')

        for state in ffxml.xpath("/ForceField/Residues/Residue[@name='{}']/Protons/State".format(residue_name)):
            for atom in state.xpath('Atom'):
                # Ensure oxygens have same type in deprotonated state.
                # Using oh because otherwise this results in lack of a C-O-H angle term
                if atom.get("name") in (cooh["OH"], cooh["OC"]):
                    atom.set("type", 'oh')

                elif atom.get("name") == cooh["HO"]:
                    # Determine whether the hydrogen is a dummy in this particular state.
                    # If it is a dummy, add a COOH statement to tell protons to insert a dummy mover.
                    if math.isclose(float(atom.get("charge")), 0.000):
                        cooh_element = etree.fromstring("<COOH/>\n")
                        for key, name in cooh.items():
                            cooh_element.set(key, name)
                        state.insert(0, cooh_element)

    # remove type annotations that clutter file.
    objectify.deannotate(ffxml)
    etree.cleanup_namespaces(ffxml)

    # Initial result has messy newlines
    result = etree.tostring(ffxml, pretty_print=True, encoding="UTF-8").decode()
    # reread and reprint to remove messy syntax
    re_tree = etree.fromstring(result,etree.XMLParser(remove_blank_text=True, remove_comments=False))
    result = etree.tostring(re_tree, pretty_print=True, encoding="UTF-8").decode()

    return result


