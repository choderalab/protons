# coding=utf-8
"""
Library for parametrizing small molecules for simulation
"""

from __future__ import print_function
from itertools import permutations
import os, re
from collections import OrderedDict, Counter
import openmoltools as omt
from lxml import etree, objectify
from openeye import oechem
from openmoltools import forcefield_generators as omtff
from .logger import log
import numpy as np
import networkx as nx
import lxml
from .. import app
from simtk.openmm import openmm
from simtk.unit import *
from ..app.integrators import GBAOABIntegrator
from copy import copy, deepcopy
import matplotlib.pyplot as plt
import logging
import itertools
from io import StringIO
from simtk import unit

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))
gaff_default = os.path.join(PACKAGE_ROOT, 'data', 'gaff.xml')

class Default(dict):
    def __missing__(self, key):
        return 0.0

def strip_in_unit_system(quant, unit_system=unit.md_unit_system, compatible_with=None):
    """Strips the unit from a simtk.units.Quantity object and returns it's value conforming to a unit system

    Parameters
    ----------
    quant : simtk.unit.Quantity
        object from which units are to be stripped
    unit_system : simtk.unit.UnitSystem:
        unit system to which the unit needs to be converted, default is the OpenMM unit system (md_unit_system)
    compatible_with : simtk.unit.Unit
        Supply to make sure that the unit is compatible with an expected unit

    Returns
    -------
    quant : object with no units attached
    """
    if unit.is_quantity(quant):
        if compatible_with is not None:
            quant = quant.in_units_of(compatible_with)
        return quant.value_in_unit_system(unit_system)
    else:
        return quant


class _State(object):
    """
    Private class representing a template of a single isomeric state of the molecule.
    """
    def __init__(self, index, log_population, g_k, net_charge, pH):
        """

        Parameters
        ----------
        index - int
            Index of the isomeric state
        log_population - str
            Solvent population of the isomeric state
        g_k - str
            The penalty for this state( i.e. returned from Epik (kcal/mol))
        net_charge - str
            Net charge of the isomeric state
        """
        self.index = index
        self.log_population = log_population
        self.g_k = g_k
        self.net_charge = net_charge
        self.atoms=OrderedDict()
        self.proton_count = -1
        self.pH = pH

    def set_number_of_protons(self, min_charge):
        """
        Set the number of acidic protons for this state

        Parameters
        ----------
        min_charge - int
            The net charge of the least protonated state.
        """
        self.proton_count = int(self.net_charge) - min_charge

    def __str__(self):
        return """<State index="{index}" log_population="{log_population}" g_k="{g_k}" proton_count="{proton_count}">
                <Condition pH="{pH}" log_population="{log_population}" temperature_kelvin="298.15"/>
                </State>""".format(**self.__dict__)

    __repr__ = __str__




def prepare_mol2_for_parametrization(input_mol2: str, output_mol2: str):
    """
    Map the hydrogen atoms between Epik states, and return a mol2 file that
    should be ready to parametrize.

    Parameters
    ----------
    input_mol2: location of the multistate mol2 file.

    Notes
    -----
    This renames the hydrogen atoms in your molecule so that
     no ambiguity can exist between protonation states.
    """
    if not output_mol2[-5:] == ".mol2":
        output_mol2 += ".mol2"
    # Generate a file format that Openeye can read

    ifs = oechem.oemolistream()
    ifs.open(input_mol2)

    # make oemols for mapping
    graphmols = [oechem.OEGraphMol(mol) for mol in ifs.GetOEGraphMols()]
    ifs.close()

    # Make graph for keeping track of which atoms are the same
    graph = nx.Graph()

    # Some hydrogens within one molecule may be chemically identical, and would otherwise be indistinguishable
    # And some hydrogens accidentally get the same name
    # Therefore, give every hydrogen a unique identifier.
    # One labelling the molecule, the other labeling the position in the molecule.
    for imol, mol in enumerate(graphmols):
        h_count = 0
        for atom in mol.GetAtoms():
            if atom.GetAtomicNum() == 1:
                h_count += 1
                # H for hydrogen, M for mol
                atom.SetName("H{}-M{}".format(h_count,imol+1))
                # Add hydrogen atom to the graph
                graph.add_node(atom, mol=imol)

    # Connect atoms that are the same
    # No need to avoid self maps for now. Code is fast enough
    for i1, mol1 in enumerate(graphmols):
        for i2, mol2 in enumerate(graphmols):

            mol1_atoms = [atom for atom in mol1.GetAtoms()]
            mol2_atoms = [atom for atom in mol2.GetAtoms()]

            # operate on a copy to avoid modifying molecule
            pattern = oechem.OEGraphMol(mol1)
            target = oechem.OEGraphMol(mol2)

            # Element should be enough to map
            atomexpr = oechem.OEExprOpts_AtomicNumber
            # Ignore aromaticity et cetera
            bondexpr = oechem.OEExprOpts_EqSingleDouble

            # create maximum common substructure object
            mcss = oechem.OEMCSSearch(pattern, atomexpr, bondexpr, oechem.OEMCSType_Approximate)
            # set scoring function
            mcss.SetMCSFunc(oechem.OEMCSMaxAtoms())
            mcss.SetMinAtoms(oechem.OECount(pattern, oechem.OEIsHeavy()))
            mcss.SetMaxMatches(10)

            # Constrain all heavy atoms, so the search goes faster.
            # These should not be different anyways
            for at1 in pattern.GetAtoms():
                # skip H
                if at1.GetAtomicNum() < 2:
                    continue
                for at2 in target.GetAtoms():
                    # skip H
                    if at2.GetAtomicNum() < 2:
                        continue
                    if at1.GetName() == at2.GetName():
                        pat_idx = mcss.GetPattern().GetAtom(oechem.HasAtomIdx(at1.GetIdx()))
                        tar_idx = target.GetAtom(oechem.HasAtomIdx(at2.GetIdx()))
                        if not mcss.AddConstraint(oechem.OEMatchPairAtom(pat_idx, tar_idx)):
                            raise ValueError("Could not constrain {} {}.".format(at1.GetName(), at2.GetName()))

            unique = True
            matches = mcss.Match(target, unique)
            # We should only use the top one match.
            for count, match in enumerate(matches):
                for ma in match.GetAtoms():
                    idx1 = ma.pattern.GetIdx()
                    idx2 = ma.target.GetIdx()
                    # Add edges between all hydrogens
                    if mol1_atoms[idx1].GetAtomicNum() == 1:
                        if mol2_atoms[idx2].GetAtomicNum() == 1:
                            graph.add_edge(mol1_atoms[idx1], mol2_atoms[idx2])
                        # Sanity check, we should never see two elements mixed
                        else:
                            raise RuntimeError("Two atoms of different elements were matched.")
                # stop after one match
                break

    # Assign unique but matching ID's per atom/state

    # The current H counter
    h_count = 0

    for cc in nx.connected_components(graph):
        # All of these atoms are chemically identical, but there could be more than one per molecule.
        atomgraph = graph.subgraph(cc)
        # Keep track of the unique H count
        h_count += 1
        names = [at.GetName() for at in atomgraph.nodes]
        # last part says which molecule the atom belongs to
        mol_identifiers = [int(name.split('-M')[1]) for name in names ]
        # Number
        counters = {i+1: 0 for i,mol in enumerate(graphmols)}
        for atom, mol_id in zip(atomgraph.nodes, mol_identifiers):
            h_num = h_count + counters[mol_id]
            atom.SetName("H{}".format(h_num))
            counters[mol_id] += 1

        # If more than one hydrogen per mol found, add it to the count.
        extra_h_count = max(counters.values()) - 1
        if extra_h_count < 0:
            raise ValueError("Found 0 hydrogens in graph, is there a bug?")
        h_count += extra_h_count

    _mols_to_file(graphmols, output_mol2)



def _mols_to_file(graphmols: list, output_mol2:str):
    """Take a list of OEGraphMols and write it to a mol2 file."""
    ofs = oechem.oemolostream()
    ofs.open(output_mol2)
    for mol in graphmols:
        oechem.OEWriteMol2File(ofs, mol)
    ofs.close()



def _visualise_graphs(graph):
    """Visualize the connected subcomponents of an atom graph"""
    import matplotlib.pyplot as plt
    nx.draw(graph, pos=nx.spring_layout(graph))
    nx.draw_networkx_labels(graph, pos=nx.spring_layout(graph), labels=dict(zip(graph.nodes, [at.GetName() for at in graph.nodes])))
    plt.show()


def generate_protons_ffxml(inputmol2: str, isomer_dicts: list, outputffxml: str, pH: float, icalib : str, resname: str="LIG"):
    """
    Compile a protons ffxml file from a preprocessed mol2 file, and a dictionary of states and charges.

    Parameters
    ----------
    inputmol2
        Location of mol2 file with protonation states results. Ensure that the names of atoms matches between protonation
         states, otherwise you will end up with atoms being duplicated erroneously. The `epik_results_to_mol2` function
          provides a handy preprocessing to clean up epik output.
    isomer_dicts: list of dicts
        One dict is necessary for every isomer. Dict should contain 'log_population' and 'net_charge'
    outputffxml : str
        location for output xml file containing all ligand states and their parameters
    pH : float
        The pH that these states are valid for.

    Other Parameters
    ----------------
    resname : str, optional (default : "LIG")
        Residue name in output files.
    

    TODO
    ----
    * Atom matching for protons based on bonded atoms?.

    Returns
    -------
    str : The absolute path of the outputfile

    """

    # Grab data from sdf file and make a file containing the charge and penalty
    isomers = isomer_dicts

    log.info("Parametrizing the isomers...")
    xmlparser = etree.XMLParser(remove_blank_text=True, remove_comments=True)

    # Open the Epik output into OEMols
    ifs = oechem.oemolistream()
    ifs.open(inputmol2)
    base = inputmol2.split('.')[-2]

    for isomer_index, oemolecule in enumerate(ifs.GetOEMols()):
        # generateForceFieldFromMolecules needs a list
        # Make new ffxml for each isomer
        log.info("ffxml generation for {}".format(isomer_index))
        ffxml = omtff.generateForceFieldFromMolecules([oemolecule], normalize=False)
        log.info(ffxml)

        ffxml_xml = etree.fromstring(ffxml, parser=xmlparser)
        name_type_mapping = {}
        name_charge_mapping = {}
        atom_index_mapping = {}
        for residue in ffxml_xml.xpath('Residues/Residue'):
            for idx, atom in enumerate(residue.xpath('Atom')):
                name_type_mapping[atom.get('name')] = atom.get('type')
                name_charge_mapping[atom.get('name')] = atom.get('charge')
                atom_index_mapping[atom.get('name')] = idx

        isomers[isomer_index]['ffxml'] = ffxml_xml
        isomers[isomer_index]['pH'] = pH
        # write open-eye mol2 file
        fileIO = str(base) + '_tmp_'+ str(isomer_index) + '.mol2'
        ofs = oechem.oemolostream()
        ofs.open(fileIO)
        oechem.OEWriteMol2File(ofs, oemolecule)
        ofs.close()

        # generate atom graph representation
        G = nx.Graph()      
        atom_name_to_unique_atom_name = dict()
  
        # set atoms
        for atom in oemolecule.GetAtoms():
            atom_name = atom.GetName()
            atom_type = name_type_mapping[atom_name]
            atom_charge = name_charge_mapping[atom_name]
            atom_index = atom_index_mapping[atom_name]
            atom_name_to_unique_atom_name[atom_name] = atom_name

            if int(atom.GetAtomicNum()) == 1:
                #found hydrogen
                heavy_atom_name = None
                for atom in atom.GetAtoms():
                    if int(atom.GetAtomicNum()) != 1:
                        heavy_atom_name = atom.GetName()
                        break
                atom_name_to_unique_atom_name[atom_name] = atom_name + name_type_mapping[heavy_atom_name]
                atom_name = atom_name + name_type_mapping[heavy_atom_name]
            
            G.add_node(atom_name, atom_type=atom_type, atom_charge=atom_charge, atom_index=atom_index)
        
        # enumerate hydrogens in this list
        hydrogens = list()
        # set bonds
        for bond in oemolecule.GetBonds():
            a1 = bond.GetBgn()
            a1_atom_name = a1.GetName()
            a2 = bond.GetEnd()
            a2_atom_name = a2.GetName()
            if int(a1.GetAtomicNum()) == 1:
                #found hydrogen
                heavy_atom_name = None
                for atom in a1.GetAtoms():
                    if int(atom.GetAtomicNum()) != 1:
                        heavy_atom_name = atom.GetName()
                        break
                hydrogens.append(tuple([a1_atom_name, a2_atom_name]))
                a1_atom_name = a1_atom_name + name_type_mapping[heavy_atom_name]
            if int(a2.GetAtomicNum()) == 1:
                #found hydrogen
                heavy_atom_name = None
                for atom in a2.GetAtoms():
                    if int(atom.GetAtomicNum()) != 1:
                        heavy_atom_name = atom.GetName()
                        break
                hydrogens.append(tuple([a2_atom_name, a1_atom_name]))
                a2_atom_name = a2_atom_name + name_type_mapping[heavy_atom_name]
            G.add_edge(a1_atom_name, a2_atom_name)
        
        # generate hydrogen definitions for each tautomer state
        # this is used to generate an openMM system of the tautomer
        isomers[isomer_index]['hxml'] = generate_tautomer_hydrogen_definitions(hydrogens, resname, isomer_index)
        isomers[isomer_index]['mol-graph'] = G
        isomers[isomer_index]['atom_name_dict'] = atom_name_to_unique_atom_name
        generate_tautomer_torsion_definitions(isomers, icalib, isomer_index)

    ifs.close()
    compiler = _TitratableForceFieldCompiler(isomers, residue_name=resname)
    _write_ffxml(compiler, outputffxml)
    log.info("Done!  Your result is located here: {}".format(outputffxml))

def generate_tautomer_hydrogen_definitions(hydrogens, residue_name, isomer_index):
    
    """
    Creates a hxml file that is used to add hydrogens for a specific tautomer to the heavy-atom skeleton 

    Parameters
    ----------
    hydrogens: list of tuple
        Tuple contains two atom names: (hydrogen-atom-name, heavy-atom-atom-name)
    residue_name : str
        name of the residue to fill the Residues entry in the xml tree
    isomer_index : int
        
    """


    hydrogen_definitions_tree = etree.fromstring('<Residues/>')
    hydrogen_file_residue = etree.fromstring("<Residue/>")
    hydrogen_file_residue.set('name', residue_name)

    for name, parent in hydrogens:
        h_xml = etree.fromstring("<H/>")
        h_xml.set("name", name)
        h_xml.set("parent", parent)
        hydrogen_file_residue.append(h_xml)
    hydrogen_definitions_tree.append(hydrogen_file_residue)
    
    return hydrogen_definitions_tree

def generate_tautomer_torsion_definitions(isomers : dict, vacuum_file:str, isomer_index:int):
    """
    Creates torsion entries (proper and improper) in the isomer dictionary. It does this 
    by creating an openMM system using the heavy-atom skeleton defined in the pdb file (vacuum_file) and the ffxml file 
    of the specific isomer. Hydrogens are added using the hxml file of the specific isomer. 

    Parameters
    ----------
    isomers: list of dicts
        One dict is necessary for every isomer. Dict should contain 'ffxml' and 'hxml' 
    vacumm_file : str
        location for heavy-atom skeleton pdb file
    isomer_index : int
        
    """

    log.info("Generate tautomer torsion definitions ...")
    ffxml = StringIO(etree.tostring(isomers[isomer_index]['ffxml']).decode("utf-8"))
    hxml = StringIO(etree.tostring(isomers[isomer_index]['hxml']).decode("utf-8"))
    forcefield = app.ForceField('amber10.xml', 'gaff.xml', ffxml, 'tip3p.xml')
    pdb = app.PDBFile(vacuum_file)
    app.Modeller.loadHydrogenDefinitions(hxml)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addHydrogens()
    index_to_name = dict()
    for a in modeller.topology.atoms():
        index_to_name[a.index] = isomers[isomer_index]['atom_name_dict'][a.name]

    system = forcefield.createSystem(modeller.topology)
    isomers[isomer_index]['torsion-forces'] = []

    for force in system.getForces():
        force_classname = force.__class__.__name__
    
        if force_classname == 'PeriodicTorsionForce':
            for torsion_index in range(force.getNumTorsions()):
                a1, a2, a3, a4, periodicity, phase, k = force.getTorsionParameters(torsion_index)
                par = { 'index': torsion_index,
                        'atom1-name' : index_to_name[a1],
                        'atom2-name' : index_to_name[a2],
                        'atom3-name' : index_to_name[a3],
                        'atom4-name' : index_to_name[a4],
                        'periodicity' : int(strip_in_unit_system(periodicity)),
                        'phase' : strip_in_unit_system(phase),
                        'k' : strip_in_unit_system(k)}
                isomers[isomer_index]['torsion-forces'].append(par)

    logging.info('Forces for tautomer: {}'.format(isomer_index))
    _log_forces(system.getForces(), index_to_name)


def _log_forces(forces, atom_index_to_atom_name):

    """
    Helper function that outputs all forces defined in a system and displays
    atom names using a mapping dictionary.

    Parameters
    ----------
    forces: openMM.Force
        list of forces present in a system 
    atom_name_by_atom_index : str
        location for heavy-atom skeleton pdb file
       
    """


    for force in forces:
        # Get name of force class.
        force_classname = force.__class__.__name__
        logging.info('############################')
        logging.info('{}'.format(force_classname))
        logging.info('############################')

        if force_classname == 'NonbondedForce':
            logging.info('#######################')
            logging.info('NonbondedForce')
            logging.info('#######################')
            for atom_idx in sorted(atom_index_to_atom_name):
                charge, sigma, eps = map(strip_in_unit_system, force.getParticleParameters(atom_idx))
                logging.info('Idx:{} Name: {} Charge:{} Sigma:{} Eps:{}'.format(atom_idx, atom_index_to_atom_name[atom_idx], charge, sigma, eps))

        elif force_classname == 'HarmonicBondForce':
            logging.info('#######################')
            logging.info('HarmonicBondForce')
            logging.info('#######################')
            for bond_idx in range(force.getNumBonds()): 
                a1, a2, length, k = map(strip_in_unit_system, force.getBondParameters(bond_idx))
                if a1 not in atom_index_to_atom_name or a2 not in atom_index_to_atom_name:
                    continue
                logging.info('Idx:{} Atom1:{} Atom2:{} l:{} k:{}'.format(bond_idx, atom_index_to_atom_name[a1], atom_index_to_atom_name[a2], length, k))
                    
        elif force_classname == 'HarmonicAngleForce':
            for angle_idx in range(force.getNumAngles()):
                a1, a2, a3, angle, k = map(strip_in_unit_system, force.getAngleParameters(angle_idx))
                if a1 not in atom_index_to_atom_name or a2 not in atom_index_to_atom_name or a3 not in atom_index_to_atom_name:
                    continue
                logging.info('Idx:{} Atom1:{} Atom2:{} Atom3:{} Angle:{} k:{}'.format(angle_idx, atom_index_to_atom_name[a1], atom_index_to_atom_name[a2], atom_index_to_atom_name[a3], angle, k))
                
        elif force_classname == 'PeriodicTorsionForce':
            for torsion_idx in range(force.getNumTorsions()):
                a1, a2, a3, a4, periodicity, phase, k = map(strip_in_unit_system, force.getTorsionParameters(torsion_idx))
                if a1 not in atom_index_to_atom_name or a2 not in atom_index_to_atom_name or a3 not in atom_index_to_atom_name or a4 not in atom_index_to_atom_name:
                    continue
                logging.info('Idx:{} Atom1:{} Atom2:{} Atom3:{} Atom4:{} Per:{} Phase:{} k:{}'.format(torsion_idx, atom_index_to_atom_name[a1], atom_index_to_atom_name[a2], atom_index_to_atom_name[a3], atom_index_to_atom_name[a4], periodicity, phase, k))


def create_hydrogen_definitions(inputfile: str, outputfile: str, gaff: str=gaff_default):
    """
    Generates hydrogen definitions for a small molecule residue template.

    Parameters
    ----------
    inputfile - a forcefield XML file defined using Gaff atom types
    outputfile - Name for the XML output file
    gaff - optional.
        The location of your gaff.xml file. By default uses the one included with protons.
    """

    gafftree = etree.parse(gaff, etree.XMLParser(remove_blank_text=True, remove_comments=True))
    xmltree = etree.parse(inputfile, etree.XMLParser(remove_blank_text=True, remove_comments=True))
    # Output tree
    hydrogen_definitions_tree = etree.fromstring('<Residues/>')
    hydrogen_types = _find_hydrogen_types(gafftree, xmltree)

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
                        # H is the first, parent is the second atom
                        hydrogens.append(tuple([atomname1, atomname2]))
                        break
                    elif atom.get('name') == atomname2:
                        # H is the second, parent is the first atom
                        hydrogens.append(tuple([atomname2, atomname1]))
                        break

        # Loop through all hydrogens, and create definitions
        for name, parent in hydrogens:
            h_xml = etree.fromstring("<H/>")
            h_xml.set("name", name)
            h_xml.set("parent", parent)
            hydrogen_file_residue.append(h_xml)
        hydrogen_definitions_tree.append(hydrogen_file_residue)
    # Write output
    xmlstring = etree.tostring(hydrogen_definitions_tree, encoding="utf-8", pretty_print=True, xml_declaration=False)
    xmlstring = xmlstring.decode("utf-8")
    with open(outputfile, 'w') as fstream:
        fstream.write(xmlstring)


def _find_hydrogen_types(gafftree: lxml.etree.ElementTree, xmlfftree: lxml.etree.ElementTree) -> set:
    """
    Find all atom types that describe hydrogen atoms.

    Parameters
    ----------
    gafftree - A GAFF input xml file that contains atom type definitions.
    xmlfftree - the customized force field template generated that contains the dummy hydrogen definitions

    Returns
    -------
    set - names of all atom types that correspond to hydrogen
    """

    # Detect all hydrogen types by element and store them in a set
    hydrogen_types = set()
    for atomtype in gafftree.xpath('AtomTypes/Type'):
        if atomtype.get('element') == "H":
            hydrogen_types.add(atomtype.get('name'))

    for atomtype in xmlfftree.xpath('AtomTypes/Type'):
        # adds dummy atome types
        if atomtype.get('name').startswith("d"):
            hydrogen_types.add(atomtype.get('name'))


    return hydrogen_types


def _write_ffxml(xml_compiler, filename=None):
    """Generate an ffxml file from a compiler object.

    Parameters
    ----------
    xml_compiler : _TitratableForceFieldCompiler
        The object that contains all the ffxml template data
    filename : str, optional
        Location and name of the file to save. If not supplied, returns the ffxml template as a string.

    Returns
    -------
    str or None
    """

    # Generate the string version.
    xmlstring = etree.tostring(xml_compiler.ffxml, encoding="utf-8", pretty_print=True, xml_declaration=False)
    xmlstring = xmlstring.decode("utf-8")

    if filename is not None:
        with open(filename, 'w') as fstream:
            fstream.write(xmlstring)
    else:
        return xmlstring


def _generate_xml_template(residue_name="LIG"):
    """
    Generate an empty template xml file in the constph format.

    Parameters
    ----------
    residue_name : str
        Name attribute of the residue

    Returns
    -------
    An xml tree  object
    """
    forcefield = _make_xml_object("ForceField")
    residues = _make_xml_object("Residues")
    residue = _make_xml_object("Residue")
    atomtypes = _make_xml_object("AtomTypes")

    hbondforce = _make_xml_object("HarmonicBondForce")
    hangleforce = _make_xml_object("HarmonicAngleForce")
    pertorsionforce = _make_xml_object("PeriodicTorsionForce")
    nonbondforce = _make_xml_object("NonbondedForce", coulomb14scale="0.833333333333", lj14scale="0.5")

    residue.attrib["name"] = residue_name
    residues.append(residue)
    forcefield.append(residues)
    forcefield.append(atomtypes)
    forcefield.append(hbondforce)
    forcefield.append(hangleforce)
    forcefield.append(pertorsionforce)
    forcefield.append(nonbondforce)

    return forcefield



class _TitratableForceFieldCompiler(object):
    """
    Compiles intermediate ffxml data to the final constant-ph ffxml file.
    """
    def __init__(self, input_state_data: list, gaff_xml:str=None, residue_name: str="LIG"):
        """
        Compiles the intermediate ffxml files into a constant-pH compatible ffxml file.

        Parameters
        ----------
        input_state_data : dict
            Contains the ffxml of the Epik isomers, net charge, population and mol graphs
        gaff_xml : string, optional
            File location of a gaff.xml file. If specified, read gaff parameters from here.
            Otherwise, gaff parameters are taken from protons/forcefields/gaff.xml
        residue_name : str, optional, default = "LIG"
            name of the residue in the output template
        
        """
        self._input_state_data = input_state_data
        self._state_templates = list()
        self.ffxml = _generate_xml_template(residue_name=residue_name)

        # including gaff file that is included with this package
        if gaff_xml is None:
            gaff_xml = gaff_default

        # list of all xml files containing relevant parameters that may be used to construct template,
        self._xml_parameter_trees = [etree.parse(gaff_xml,
                                                 etree.XMLParser(remove_blank_text=True, remove_comments=True)
                                                 )
                                     ]
        for state in self._input_state_data:
            self._xml_parameter_trees.append(state['ffxml'])

        # Compile all information into the output structure
        self._make_output_tree()


    def _make_output_tree(self):
        """
        Store all contents of a compiled ffxml file of all isomers, and add dummies for all missing hydrogens.
        """

        # Register the states
        self._complete_state_registry()
        # Set the initial state of the template that is read by OpenMM
        self._initialize_forcefield_template()
        # Add isomer specific information
        self._add_isomers()
        # Append extra parameters from frcmod
        self._append_extra_gaff_types()
        # Append dummy parameters
        self._append_dummy_parameters()
        # Remove empty blocks, and unnecessary information in the ffxml tree
        self._sanitize_ffxml()

        return

    def _append_dummy_parameters(self):

        def _unique(element):
            
            if 'type3' in element.attrib:
                a1, a2, a3 = element.attrib['type1'], element.attrib['type2'], element.attrib['type3']
                if (a1, a2, a3) in unique_set:
                    return False
                else:
                    unique_set.add((a1,a2,a3))
                    unique_set.add((a3,a2,a1))
                    return True
            else:
                a1, a2 = element.attrib['type1'], element.attrib['type2']
                if (a1, a2) in unique_set:
                    return False
                else:
                    unique_set.add((a1,a2))
                    unique_set.add((a2,a1))
                    return True

        unique_set = set()
        logging.info('################################')
        logging.info('Appending dummy parameters')
        logging.info('################################')

        # add dummy atom types present in the isomers
        for element_string in set(self.dummy_atom_type_strings):
            self._add_to_output(etree.fromstring(element_string), "/ForceField/AtomTypes")
            logging.info('Adding dummy atom element: \n{}'.format(element_string))
        
        for element_string in set(self.dummy_atom_nb_strings):
            self._add_to_output(etree.fromstring(element_string), "/ForceField/NonbondedForce")
            logging.info('Adding dummy atom nonbonded parameters: \n{}'.format(element_string))
            
        for element_string in set(self.dummy_bond_strings):
            e = etree.fromstring(element_string)
            if _unique(e):
                self._add_to_output(e, "/ForceField/HarmonicBondForce")
                logging.info(element_string)

        for element_string in set(self.dummy_angle_strings):
            e = etree.fromstring(element_string)
            if _unique(e):
                self._add_to_output(etree.fromstring(element_string), "/ForceField/HarmonicAngleForce")
                logging.info(element_string)

       
    def _sanitize_ffxml(self):
        """
        Clean up the structure of the ffxml file by removing unnecessary blocks and information.
        """
        # Get rid of extra junk information that is added to the xml files.
        objectify.deannotate(self.ffxml)
        etree.cleanup_namespaces(self.ffxml)
        # Get rid of empty blocks directly under ForceField
        for empty_block in self.ffxml.xpath('/ForceField/*[count(child::*) = 0]'):
            empty_block.getparent().remove(empty_block)

    
    def register_tautomers(self, mol_graphs):


        # generate union mol graph
        # and register the atom_types, bonds, angles and torsions for the different mol_graphs (tautomers)
        superset_graph = nx.Graph()
        
        #####################
        #atoms
        #####################
        atom_types_dict = dict()
        atom_charge_dict = dict()
        for state_idx in mol_graphs:
            atom_types_dict_tmp = dict()
            atom_charges_dict_tmp = dict()
            for node, data in mol_graphs[state_idx].nodes(data=True):
                superset_graph.add_node(node)
                atom_types_dict_tmp[node] = data['atom_type']
                atom_charges_dict_tmp[node] = data['atom_charge']

            atom_types_dict[state_idx] = atom_types_dict_tmp
            atom_charge_dict[state_idx] = atom_charges_dict_tmp

        #####################
        #bonds
        #####################
        complete_list_of_bonds = dict()
        for state_idx in mol_graphs:
            list_of_bonds_atom_types = []
            list_of_bonds_atom_names = []

            for edge in mol_graphs[state_idx].edges(data=True):
                a1_name = (list(edge)[0])
                a2_name = (list(edge)[1])
                superset_graph.add_edge(a1_name, a2_name)
                list_of_bonds_atom_types.append([atom_types_dict[state_idx][a1_name], atom_types_dict[state_idx][a2_name]])
                list_of_bonds_atom_names.append([a1_name, a2_name])
            
            complete_list_of_bonds[state_idx] = {'bonds_atom_types' : list_of_bonds_atom_types, 'bonds_atom_names' : list_of_bonds_atom_names}

        nx.draw(superset_graph, pos=nx.kamada_kawai_layout(superset_graph), with_labels=True, font_weight='bold', node_size=1400, alpha=0.5, font_size=12)
        plt.show()       

        #####################
        #angles
        #####################
        complete_list_of_angles = dict()
        for state_idx in mol_graphs:
            list_of_angles_atom_types = []
            list_of_angles_atom_names = []
            for node1 in mol_graphs[state_idx]:
                for node2 in mol_graphs[state_idx]:
                    if node2.startswith('H') or not mol_graphs[state_idx].has_edge(node1, node2):
                        continue
                    else:
                        for node3 in mol_graphs[state_idx]:
                            if not mol_graphs[state_idx].has_edge(node2, node3) or node3 == node1:
                                continue
                            else:
                                atom_types = []
                                for node in [node1, node2, node3]:
                                    if node in atom_types_dict[state_idx]:
                                        atom_types.append(atom_types_dict[state_idx][node])
                                    else: 
                                        atom_types.append('d' + node)                                
                                list_of_angles_atom_types.append(atom_types)
                                list_of_angles_atom_names.append([node1, node2, node3])
            
            complete_list_of_angles[state_idx] = {'angles_atom_types' : list_of_angles_atom_types, 'angles_atom_names' : list_of_angles_atom_names}


        self.superset_graph = superset_graph
        self.atom_types_dict = atom_types_dict
        self.atom_charge_dict = atom_charge_dict
        self.complete_list_of_bonds = complete_list_of_bonds
        self.complete_list_of_angles = complete_list_of_angles

 
    def _complete_state_registry(self):
        """
        Store all the properties that are specific to each state
        """
        mol_graphs = {}
        for index, state in enumerate(self._input_state_data):
            mapping_atom_name_to_charge = {}
            for xml_atom in state['ffxml'].xpath('/ForceField/Residues/Residue/Atom'):
                mapping_atom_name_to_charge[xml_atom.attrib['name']] = xml_atom.attrib['charge']
            mol_graphs[index] = state['mol-graph']

        self.register_tautomers(mol_graphs)            

        charges = list()
        for index, state in enumerate(self._input_state_data):
            net_charge = state['net_charge']
            charges.append(int(net_charge))
            template = _State(index,
                              state['log_population'],
                              0.0, # set g_k defaults to 0 for now
                              net_charge,
                              state['pH']
                              )

            self._state_templates.append(template)

        min_charge = min(charges)
        for state in self._state_templates:
            state.set_number_of_protons(min_charge)
        return                


    def _initialize_forcefield_template(self):
        """
        Set up the residue template using the superset graph.
        The residue includes all atom names of all the states, only the atoms of 
        the first state have read atom types.
        """

        residue = self.ffxml.xpath('/ForceField/Residues/Residue')[0]
        atom_string = '<Atom name="{name}" type="{atom_type}" charge="{charge}"/>'
        bond_string = '<Bond atomName1="{atomName1}" atomName2="{atomName2}"/>'

        for node in self.superset_graph:
            if node in self.atom_types_dict[0]:
                atom_charge = self.atom_charge_dict[0][node]
                atom_type = self.atom_types_dict[0][node]
                residue.append(etree.fromstring(atom_string.format(name=node, atom_type=atom_type, charge=atom_charge)))
            else:
                atom_type='d' + str(node)
                residue.append(etree.fromstring(atom_string.format(name=node, atom_type=atom_type, charge=0.0)))
        
        for bond in self.superset_graph.edges:
            atom1_name = bond[0]
            atom2_name = bond[1]
            residue.append(etree.fromstring(bond_string.format(atomName1=atom1_name, atomName2=atom2_name)))


    def _add_isomers(self):
        """
        Add all the isomer specific data to the xml template.
        """
        logging.info('Add isomer information ...')

        protonsdata = etree.fromstring("<Protons/>")
        protonsdata.attrib['number_of_states'] = str(len(self._state_templates))

        atom_string = '<Atom name="{node1}" type="{atom_type}" charge="{charge}" epsilon="{epsilon}" sigma="{sigma}" />'
        dummy_atom_string = '<Type name="{atom_type}" class="{atom_type}" element="{element}" mass="{mass}"/>'
        dummy_nb_string = '<Atom type="{atom_type}" sigma="{sigma}" epsilon="{epsilon}" charge="{charge}"/>'
               
        bond_string = '<Bond name1="{node1}" name2="{node2}" length="{bond_length}" k="{k}"/>'
        dummy_bond_string = '<Bond type1="{atomType1}" type2="{atomType2}" length="{bond_length}" k="{k}"/>'

        angle_string = '<Angle name1="{node1}" name2="{node2}" name3="{node3}" angle="{angle}" k="{k}"/>'
        dummy_angle_string = '<Angle type1="{atomType1}" type2="{atomType2}" type3="{atomType3}" angle="{angle}" k="{k}"/>'

        torsion_string = '<Torsion name1="{node1}" name2="{node2}" name3="{node3}" name4="{node4}" periodicity="{periodicity}" phase="{phase}" k="{k}"/>'

        # pregenerating all angles and torsions present
        # generate all angles present in the superset
        list_of_angles = []
        for node1 in self.superset_graph:
            for node2 in self.superset_graph:
                if node2.startswith('H') or not self.superset_graph.has_edge(node1, node2):
                    continue
                else:
                    for node3 in self.superset_graph:
                        if not self.superset_graph.has_edge(node2, node3) or node3 == node1:
                            continue
                        else:
                            list_of_angles.append([node1, node2, node3])
               

        # here the entries for all atoms, bonds, angles and torsions are generated 
        # and added to the xml file using ATOM NAMES (not atom type) for each residue
        # also dummy types are generated
        self.dummy_atom_type_strings = list()
        self.dummy_atom_nb_strings = list()
        self.dummy_bond_strings = list()
        self.dummy_angle_strings = list()

        atom_name_to_atom_type_inclusive_dummy_types = dict()

        for residue in self.ffxml.xpath('/ForceField/Residues/Residue'):
            for isomer_index, isomer in enumerate(self._state_templates):
                ##############################################
                # atom entries
                ##############################################
                logging.info('ISOMER: {}'.format(isomer_index))
                isomer_str = str(isomer)
                logging.info(isomer_str)
                isomer_xml = etree.fromstring(isomer_str)

                # atom_types_dict for specific isomer
                isomer_atom_name_to_atom_type = self.atom_types_dict[isomer_index]
                isomer_atom_name_to_atom_charge = self.atom_charge_dict[isomer_index]
                tmp_atom_name_to_atom_type_inclusive_dummy_types = dict()
                # iterate over all nodes in superset graph
                for node in self.superset_graph:
                    parm = None
                    e = None
                    # if node is in isomer_atom_types dict it has an assigned atom type in this isomer
                    if node in isomer_atom_name_to_atom_type:
                        atom_type = isomer_atom_name_to_atom_type[node]
                        parm = self._retrieve_parameters(atom_type1=atom_type)                       
                        e = atom_string.format(node1=node, atom_type=atom_type, charge=isomer_atom_name_to_atom_charge[node],epsilon=parm['nonbonds'].attrib['epsilon'],sigma=parm['nonbonds'].attrib['sigma'])
                    else:
                        atom_type = 'd' + node
                        e = atom_string.format(node1=node, atom_type=atom_type, charge=0.0,epsilon=0,sigma=0)

                        if int(isomer_index) == 0:
                            # add dummy nb and atom parameters for first isomer
                            self.dummy_atom_type_strings.append(dummy_atom_string.format(atom_type=atom_type, element='H', mass=1.008))
                            self.dummy_atom_nb_strings.append(dummy_nb_string.format(atom_type=atom_type, charge=0.0,epsilon=0,sigma=0))
                    
                    tmp_atom_name_to_atom_type_inclusive_dummy_types[node] = atom_type
                    isomer_xml.append(etree.fromstring(e))
                    logging.info(e)

                atom_name_to_atom_type_inclusive_dummy_types[isomer_index] = tmp_atom_name_to_atom_type_inclusive_dummy_types
                ##############################################
                # bond entries
                ##############################################
                isomer_bond_atom_names = self.complete_list_of_bonds[isomer_index]['bonds_atom_names']
                dummy_bond_type_list = []
                for edge in self.superset_graph.edges:
                    node1 = edge[0]
                    node2 = edge[1]
                    # there is never a bond that has not 
                    # in some isomer only real atoms - therefore we can search through the atom_types_dict  
                    atom_types = []
                    # if node is not in iserom_atom_types_dict it does not have an assigned atom type in this isomer
                    if [node1, node2] in isomer_bond_atom_names or [node2, node1] in isomer_bond_atom_names:
                        atom_types.extend([isomer_atom_name_to_atom_type[node1], isomer_atom_name_to_atom_type[node2]])
                        parm = self._retrieve_parameters(atom_type1=atom_types[0], atom_type2=atom_types[1])
                        bond_length = parm['bonds'].attrib['length']
                        k = parm['bonds'].attrib['k']
                        found_parameter = True
                    else:
                        # search through all atom_types_dict to find real atom type
                        found_parameter = False

                        for tmp_isomer_index in range(len(self._state_templates)):
                            if [node1, node2] in self.complete_list_of_bonds[tmp_isomer_index]['bonds_atom_names'] or [node2, node1] in self.complete_list_of_bonds[tmp_isomer_index]['bonds_atom_names']:
                                atom_types.extend([self.atom_types_dict[tmp_isomer_index][node1], self.atom_types_dict[tmp_isomer_index][node2]])
                                found_parameter = True
                                # add dummy bond parameters for all superset bonds for starting residue
                                parm = self._retrieve_parameters(atom_type1=atom_types[0], atom_type2=atom_types[1])
                                bond_length = parm['bonds'].attrib['length']
                                k = parm['bonds'].attrib['k']
                                if int(isomer_index) == 0:
                                    # add dummy bond entries for first isomer
                                    d = dummy_bond_string.format(atomType1=atom_name_to_atom_type_inclusive_dummy_types[0][node1], atomType2=atom_name_to_atom_type_inclusive_dummy_types[0][node2], bond_length=bond_length, k=k)
                                    self.dummy_bond_strings.append(d)
                                    d = None
                                break
                    
                    if found_parameter is False:
                        print('CAREFUL! parameter not found')
                        raise NotImplementedError('This case is not covered so far.')
                    
                    e = bond_string.format(node1=node1, node2=node2, bond_length=bond_length, k=k)
                    logging.info(e)
                    isomer_xml.append(etree.fromstring(e))

                ##############################################
                # angle entries
                ##############################################
                isomer_angle_atom_names = self.complete_list_of_angles[isomer_index]['angles_atom_names']
                
                for nodes in list_of_angles:
                    node1, node2, node3 = nodes
                    found_parameters = False
                    parm = None
                    # angle between three atoms that are real in this isomer 
                    if [node1, node2, node3] in isomer_angle_atom_names or [node3, node2, node1] in isomer_angle_atom_names:
                        atom_type1, atom_type2, atom_type3 = isomer_atom_name_to_atom_type[node1], isomer_atom_name_to_atom_type[node2],isomer_atom_name_to_atom_type[node3]  
                        parm = self._retrieve_parameters(atom_type1=atom_type1,atom_type2=atom_type2,atom_type3=atom_type3)
                        angle = parm['angle'].attrib['angle']
                        k = parm['angle'].attrib['k']
                        found_parameters = True
                    else:
                        # not real in this isomer - look at other isomers and get parameters from isomer 
                        # in which these 3 atoms form real angle
                        for tmp_isomer_index in range(len(self._state_templates)):
                            if [node1, node2, node3] in self.complete_list_of_angles[tmp_isomer_index]['angles_atom_names'] or [node3, node2, node1] in self.complete_list_of_angles[tmp_isomer_index]['angles_atom_names']:
                                parm = self._retrieve_parameters(atom_type1=self.atom_types_dict[tmp_isomer_index][node1],atom_type2=self.atom_types_dict[tmp_isomer_index][node2],atom_type3=self.atom_types_dict[tmp_isomer_index][node3])
                                angle = parm['angle'].attrib['angle']
                                k = parm['angle'].attrib['k']
                                found_parameters = True
                                if int(isomer_index) == 0:
                                    # add dummy angle parameters for first isomer
                                    d = dummy_angle_string.format(atomType1=atom_name_to_atom_type_inclusive_dummy_types[0][node1], atomType2=atom_name_to_atom_type_inclusive_dummy_types[0][node2], atomType3=atom_name_to_atom_type_inclusive_dummy_types[0][node3], angle=angle, k=k)
                                    self.dummy_angle_strings.append(d)
                                    d = None
                                break
                    
                    # this is not a real angle - there are never only real atoms involved in this angle
                    if not found_parameters:
                        angle = 0.0
                        k = 0.0
                        e = angle_string.format(node1=node1, node2=node2, node3=node3, angle=angle, k=k)
                        if int(isomer_index) == 0:
                            # add dummy angle parameters for first isomer
                            d = dummy_angle_string.format(atomType1=atom_name_to_atom_type_inclusive_dummy_types[0][node1], atomType2=atom_name_to_atom_type_inclusive_dummy_types[0][node2], atomType3=atom_name_to_atom_type_inclusive_dummy_types[0][node3], angle=angle, k=k)
                            self.dummy_angle_strings.append(d)
                            d = None

                    e = angle_string.format(node1=node1, node2=node2, node3=node3, angle=angle, k=k)
                    logging.info(e)

                    isomer_xml.append(etree.fromstring(e))

                ##############################################
                # torsion entries
                ##############################################
                for torsion_force in self._input_state_data[isomer_index]['torsion-forces']:
                    e = torsion_string.format(node1=torsion_force['atom1-name'], node2=torsion_force['atom2-name'], node3=torsion_force['atom3-name'], node4=torsion_force['atom4-name'], periodicity=torsion_force['periodicity'], phase=torsion_force['phase'], k=torsion_force['k'])
                    isomer_xml.append(etree.fromstring(e))
                    logging.info(e)

                protonsdata.append(isomer_xml)
            residue.append(protonsdata)

    def _append_extra_gaff_types(self):
        """
        Add additional parameters generated by antechamber/parmchk for the individual isomers
        """
        added_parameters = list()  # for bookkeeping of duplicates

        # in order to avoid overwriting the proper and improper parameter entries 
        # in self._xml_parameter_trees a copy is created
        # mw: I have no idea, why this overwrites, this is a ugly hack to avoid this issue
        xml_trees = deepcopy(self._xml_parameter_trees)

        # All xml sources except the entire gaff.xml
        for idx, xmltree in enumerate(xml_trees[1:]):
            # Match the type of the atom in the AtomTypes block
            for atomtype in xmltree.xpath("/ForceField/AtomTypes/Type"):
                items = set(atomtype.items())
                type_element = tuple(["AtomTypes", "Type", items])
                # Make sure the type wasn't already added by a previous state
                if type_element not in added_parameters:
                    added_parameters.append(type_element)
                    self._add_to_output(atomtype, "/ForceField/AtomTypes")

            # Match the bonds of the atom in the HarmonicBondForce block
            for bond in xmltree.xpath("/ForceField/HarmonicBondForce/Bond"):
                items = set(bond.items())
                bond_element = tuple(["HarmonicBondForce", "Bond", items])
                # Make sure the force wasn't already added by a previous state
                if bond_element not in added_parameters:
                    added_parameters.append(bond_element)
                    self._add_to_output(bond, "/Forcefield/HarmonicBondForce")

            # Match the angles of the atom in the HarmonicAngleForce block
            for angle in xmltree.xpath("/ForceField/HarmonicAngleForce/Angle"):
                items = set(angle.items())
                angle_element = tuple(["HarmonicAngleForce", "Angle", items])
                # Make sure the force wasn't already added by a previous state
                if angle_element not in added_parameters:
                    added_parameters.append(angle_element)
                    self._add_to_output(angle, "/Forcefield/HarmonicAngleForce")


            # Match nonbonded type of the atom in NonbondedForce block
            for nonbond in xmltree.xpath("/ForceField/NonbondedForce/Atom"):
                items = set(nonbond.items())
                nb_element = tuple(["NonbondedForce", "Atom", items])
                # Make sure the force wasn't already added by a previous state
                if nb_element not in added_parameters:
                    added_parameters.append(nb_element)
                    self._add_to_output(nonbond, "/ForceField/NonbondedForce")


            # Match improper dihedral of the atom in PeriodicTorsionForce block
            for improper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Improper"):
                items = set(improper.items())
                improper_element = tuple(["PeriodicTorsionForce", "Improper", items])
                # Make sure the force wasn't already added by a previous state
                if improper_element not in added_parameters:
                    added_parameters.append(improper_element)
                    # all impropers should be 
                    self._add_to_output(improper, "/ForceField/PeriodicTorsionForce")

            # Match proper dihedral of the atom in PeriodicTorsionForce block
            for proper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Proper"):
                items = set(proper.items())
                proper_element = tuple(["PeriodicTorsionForce", "Proper", items])
                # Make sure the force wasn't already added by a previous state
                if proper_element not in added_parameters:
                    added_parameters.append(proper_element)
                    self._add_to_output(proper, "/ForceField/PeriodicTorsionForce")



    def _add_to_output(self, element, xpath):
        """
        Insert elements into the output tree at a location specified using XPATH

        Parameters
        ----------
        element - the element to append
        xpath - XPATH specification of the location to append the element

        Returns
        -------

        """
        for item in self.ffxml.xpath(xpath):
            item.append(element)
        return



    def _validate_states(self):
        """
        Check the validity of all states.
        """
        for state in self._state_templates:
            state.validate()

      
    
    def _retrieve_parameters(self, **kwargs):
        """ Look through FFXML files and find all parameters pertaining to the supplied atom type.
        Looks either for atom, bond, angle or torsion parameters depending on the number of arguments provided.
        Returns
        -------
        input : atom_type1:str, atom_type2[opt]:str, atom_type3[opt]:str, atom_type4[opt]:str, 
        """
        
        
        # Storing all the detected parameters here
        params = {}
        # Loop through different sources of parameters

        if len(kwargs) == 1:
            # Loop through different sources of parameters
            for xmltree in self._xml_parameter_trees:
                # Match the type of the atom in the AtomTypes block
                for atomtype in xmltree.xpath("/ForceField/AtomTypes/Type"):
                    if atomtype.attrib['name'] == kwargs['atom_type1']:
                        params['type'] = atomtype
                for nonbond in xmltree.xpath("/ForceField/NonbondedForce/Atom"):
                    if nonbond.attrib['type'] == kwargs['atom_type1']:
                        params['nonbonds'] = nonbond

            return params

        elif len(kwargs) == 2:
            for xmltree in self._xml_parameter_trees:
                # Match the bonds of the atom in the HarmonicBondForce block
                for bond in xmltree.xpath("/ForceField/HarmonicBondForce/Bond"):
                    if sorted([kwargs['atom_type1'], kwargs['atom_type2']]) == sorted([bond.attrib['type1'], bond.attrib['type2']]):
                        params['bonds'] = bond
            return params
                    

        elif len(kwargs) == 3:
            search_list = [kwargs['atom_type1'], kwargs['atom_type2'], kwargs['atom_type3']]
            for xmltree in self._xml_parameter_trees:
                # Match the angles of the atom in the HarmonicAngleForce block
                for angle in xmltree.xpath("/ForceField/HarmonicAngleForce/Angle"):
                    angle_atom_types_list = [angle.attrib['type1'], angle.attrib['type2'], angle.attrib['type3']]
                    if search_list == angle_atom_types_list or search_list == angle_atom_types_list[::-1]:
                        params['angle'] = angle
                        return params
                    else:
                        continue
            return params
            
  
def _make_xml_object(root_name, **attributes):
    """
    Create a new xml root object with a given root name, and attributes

    Parameters
    ----------
    root_name - str
        The name of the xml root.
    attributes - dict
        Dictionary of attributes and values (as strings) for the xml file

    Returns
    -------
    ObjectifiedElement

    """
    xml = '<{0}></{0}>'.format(root_name)
    root = objectify.fromstring(xml)
    for attribute, value in attributes.items():
        root.set(attribute, value)

    return root

def prepare_calibration_system(vacuum_file:str, output_file:str, ofolder:str, ffxml: str=None, hxml:str=None,  delete_old_H:bool=True):
    """Add hydrogens to a residue based on forcefield and hydrogen definitons, and then solvate.

    Note that no salt is added. We use saltswap for this.

    Parameters
    ----------
    vacuum_file - a single residue in vacuum to add hydrogens to and solvate.
    output_file - the basename for an output mmCIF file with the solvated system.
    ffxml - the forcefield file containing the residue definition,
        optional for CDEHKY amino acids, required for ligands.
    hxml - the hydrogen definition xml file,
        optional for CDEHKY amino acids, required for ligands.
    delete_old_H - delete old hydrogen atoms and add in new ones.
        Typically necessary for ligands, where hydrogen names will have changed during parameterization to match up
        different protonation states.
    """

    # Load relevant template definitions for modeller, forcefield and topology
    if hxml is not None:
        app.Modeller.loadHydrogenDefinitions(hxml)
        
    if ffxml is not None:
        # for tautomers we will for the moment use the regular amber10 ff and not the constantph
        forcefield = app.ForceField('amber10.xml', 'gaff.xml', ffxml, 'tip3p.xml')
        #forcefield = app.ForceField('amber10-constph.xml', 'gaff.xml', ffxml, 'tip3p.xml', 'ions_tip3p.xml')
    else:
        forcefield = app.ForceField('amber10-constph.xml', 'gaff.xml', 'tip3p.xml', 'ions_tip3p.xml')

    pdb = app.PDBFile(vacuum_file)
    modeller = app.Modeller(pdb.topology, pdb.positions)
    # The system will likely have different hydrogen names.
    # In this case its easiest to just delete and re-add with the right names based on hydrogen files
    #if delete_old_H:
    #    to_delete = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H']]
    #    print(to_delete)

    modeller.addHydrogens(forcefield=forcefield)
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * nanometers, neutralize=False)

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * nanometers,
                                     constraints=app.HBonds, rigidWater=True,
                                     ewaldErrorTolerance=0.0005)
    system.addForce(openmm.MonteCarloBarostat(1.0 * atmosphere, 300.0 * kelvin))
    simulation = app.Simulation(modeller.topology, system, GBAOABIntegrator())
    simulation.context.setPositions(modeller.positions)
    simulation.minimizeEnergy()

    app.PDBxFile.writeFile(modeller.topology, simulation.context.getState(getPositions=True).getPositions(),
                           open(output_file, 'w'))
    
    app.PDBFile.writeFile(modeller.topology, simulation.context.getState(getPositions=True).getPositions(),
                           open(ofolder + '/input.pdb', 'w'))

    
    return simulation