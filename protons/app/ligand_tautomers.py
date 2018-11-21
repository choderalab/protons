# coding=utf-8
"""
Library for parametrizing small molecules for simulation
"""

from __future__ import print_function

import os
import shutil
import tempfile
import mdtraj
import uuid
from collections import OrderedDict
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
from rdkit import Chem
from rdkit.Chem import MCS
from copy import copy, deepcopy
from rdkit.Chem import rdMolAlign
import matplotlib.pyplot as plt
import logging
from collections import defaultdict
logging.basicConfig(level=logging.DEBUG)

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))


gaff_default = os.path.join(PACKAGE_ROOT, 'data', 'gaff.xml')


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


def generate_protons_ffxml(inputmol2: str, isomer_dicts: list, outputffxml: str, pH: float, resname: str="LIG"):
    """
    Compile a protons ffxml file from a preprocessed mol2 file, and a dictionary of states and charges.
    Also populates the isomer_dicts with rdkit mol objects for every state.

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
    log.info("Processing Epik output...")
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
        for residue in ffxml_xml.xpath('Residues/Residue'):
            for atom in residue.xpath('Atom'):
                name_type_mapping[atom.get('name')] = atom.get('type')


        isomers[isomer_index]['ffxml'] = ffxml_xml
        isomers[isomer_index]['pH'] = pH
        # write open-eye mol2 file
        fileIO = str(base) + '_tmp_'+ str(isomer_index) + '.mol2'
        ofs = oechem.oemolostream()
        ofs.open(fileIO)
        oechem.OEWriteMol2File(ofs, oemolecule)
        ofs.close()
        # read in using rdkit
        rdmol = Chem.MolFromMol2File(fileIO, removeHs=False)

        # set atom-names and types for atoms in rdkit mol
        # according to open-eye atom types
        for a in oemolecule.GetAtoms():
            rdmol.GetAtomWithIdx(a.GetIdx()).SetProp('name', a.GetName())
            rdmol.GetAtomWithIdx(a.GetIdx()).SetProp('type', name_type_mapping[a.GetName()])           
        
        # save rdmol in isomers map
        isomers[isomer_index]['mol'] = rdmol
    
    ifs.close()
    compiler = _TitratableForceFieldCompiler(isomers, residue_name=resname)
    _write_ffxml(compiler, outputffxml)
    log.info("Done!  Your result is located here: {}".format(outputffxml))


    return outputffxml, compiler


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

    print('forcefield')
    print(type(forcefield))
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
        input_state_data : list
            Contains the ffxml of the Epik isomers, net charge, population and mol objects (rdkit)
        gaff_xml : string, optional
            File location of a gaff.xml file. If specified, read gaff parameters from here.
            Otherwise, gaff parameters are taken from protons/forcefields/gaff.xml
        residue_name : str, optional, default = "LIG"
            name of the residue in the output template
        
        """
        self._input_state_data = input_state_data
        self._state_templates = list()
        self._unison_clone = None     
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


    def _make_output_tree(self, chimera=True):
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

        print('Appending dummy parameters')
        print('################################')

        atom_string = '<Type name="{atom_type}" class="{atom_type}" charge="{charge}" element="{element}" mass="{mass}"/>'
        nb_string = '<Atom type="{atom_type}" sigma="{sigma}" epsilon="{epsilon}" charge="{charge}"/>'

        unique_atom_set = set()
        for state in range(len(self.mol_array)):
            for node in self.network:
                atom_type = self.atom_types_dict[node][state]
                if str(atom_type) == '0':
                    idx,atom_type = _return_real_atom_type(self.atom_types_dict, node)
                    atom_charge = 0.0
                    atom_type='d'+ str(node) +str(atom_type)
                    if atom_type in unique_atom_set:
                        continue
                    else:
                        unique_atom_set.add(atom_type)
                    element_string = etree.fromstring(atom_string.format(name=node, atom_type=atom_type, charge=atom_charge, element='H', mass=1.008))
                    nb_element_string = etree.fromstring(nb_string.format(atom_type=atom_type, sigma=0.0, epsilon=0.0, charge=0.0))

                    self._add_to_output(element_string, "/ForceField/AtomTypes")
                    self._add_to_output(nb_element_string, "/ForceField/NonbondedForce")

        # Now add all dummy bonds
        unique_bond_set = set()
        dummy_bond_string = '<Bond type1="{atomType1}" type2="{atomType2}" length="{bond_length}" k="{k}"/>'
        for state in range(len(self.mol_array)):
            print('$$$$$$$$$$$$$$$$$$$$')
            print('State: ' + str(state))
            print('$$$$$$$$$$$$$$$$$$$$')
            
            for bond in self.network.edges:
                atomName1 = bond[0]
                atomName2 = bond[1]
                
                atom_type1 = self.atom_types_dict[atomName1][state]
                atom_type2 = self.atom_types_dict[atomName2][state]
                
                if str(atom_type1) == '0':
                    print('#############')
                    idx, atom_type1 = _return_real_atom_type(self.atom_types_dict, atomName1)
                    helper_atom_type2 = (self.atom_binds_to_atom_type[atomName1])
                    print('Found dummy bond between ' + str(atomName1) + ' and ' + str(atomName2))
                    print('Found dummy bond between ' + str(atom_type1) + ' and ' + str(helper_atom_type2))
                    parm = self._retrieve_parameters(atom_type1=atom_type1, atom_type2=helper_atom_type2)
                    print(parm)
                    atom_type1 = 'd'+ str(atomName1) + str(atom_type1)
                    if (atom_type1, atom_type2) in unique_bond_set or (atom_type2, atom_type1) in unique_bond_set:
                        print('Found duplicate')
                    else:
                        unique_bond_set.add((atom_type1, atom_type2))
             
                elif str(atom_type2) == '0':
                    print('#############')
                    idx, atom_type2 = _return_real_atom_type(self.atom_types_dict, atomName2)
                    helper_atom_type1 = (self.atom_binds_to_atom_type[atomName2])
                    print('Found dummy bond between ' + str(atomName1) + ' and ' + str(atomName2))
                    print('Real bond between ' + str(helper_atom_type1) + ' and ' + str(atom_type2))

                    parm = self._retrieve_parameters(atom_type1=helper_atom_type1, atom_type2=atom_type2)
                    atom_type2 = 'd' + str(atomName2) + str(atom_type2)              
                    print(parm)

                    if (atom_type1, atom_type2) in unique_bond_set or (atom_type2, atom_type1) in unique_bond_set:
                        print('Found duplicate')
                    else:
                        unique_bond_set.add((atom_type2, atom_type1))
                else:
                    continue
                    
                length= parm['bonds'].attrib['length']
                k= parm['bonds'].attrib['k']

                print(dummy_bond_string.format(atomType1=atom_type1, atomType2=atom_type2, bond_length=length, k=k))
                element_string= etree.fromstring(dummy_bond_string.format(atomType1=atom_type1, atomType2=atom_type2, bond_length=length, k=k))
                self._add_to_output(element_string, "/ForceField/HarmonicBondForce")
            

        unique_angle_set = set()
        angle_string = '<Angle type1="{atomType1}" type2="{atomType2}" type3="{atomType3}" angle="{angle}" k="{k}"/>'
        print('Generating dummy angle entries ...')
        for state in range(len(self.mol_array)):
            print('@@@@@@@@@@@@')

            print('State: ', state)
            list_of_angles_atom_types, list_of_angles_atom_names = _get_all_angles(self.network, self.atom_types_dict, state)

            for i, nodes in enumerate(list_of_angles_atom_names):
                node1, node2, node3 = nodes
                key = hash((node1, node2, node3))
                # angle between two dummy atoms
                if list_of_angles_atom_types[i].count(0) > 1:
                    #print('Dummy to dummy')
                    continue
                # only one dummy atom
                elif 0 in list_of_angles_atom_types[i]:
                    original_atom_type1, original_atom_type2, original_atom_type3 = list_of_angles_atom_types[i]
                    for angles_types in self.all_angles_at_all_states[key].values():
                        if 0 in angles_types:
                            continue
                        else:
                            new_atom_type1, new_atom_type2, new_atom_type3 = angles_types
                            parm = self._retrieve_parameters(atom_type1=new_atom_type1, atom_type2=new_atom_type2, atom_type3=new_atom_type3)

                            if str(original_atom_type1) == '0':
                                t, real_atom_type = _return_real_atom_type(self.atom_types_dict, node1)
                                original_atom_type1 = 'd' + str(node1) + real_atom_type
                            elif str(original_atom_type2) == '0':
                                t, real_atom_type = _return_real_atom_type(self.atom_types_dict, node2)
                                original_atom_type2 = 'd' + str(node2) + real_atom_type
                            else:
                                t, real_atom_type = _return_real_atom_type(self.atom_types_dict, node3)
                                original_atom_type3 = 'd' + str(node3) + real_atom_type

                            if (original_atom_type1, original_atom_type2, original_atom_type3) in unique_angle_set or (original_atom_type3, original_atom_type2, original_atom_type1) in unique_angle_set:
                                #print('Repetition!')
                                continue
                            else:
                                unique_angle_set.add((original_atom_type1, original_atom_type2, original_atom_type3))

                            angle= parm['angle'].attrib['angle']
                            k= parm['angle'].attrib['k']
                            #print(original_atom_type1, original_atom_type2, original_atom_type3)
                            element_string= etree.fromstring(angle_string.format(atomType1=original_atom_type1, atomType2=original_atom_type2, atomType3=original_atom_type3, angle=angle, k=k))
                            self._add_to_output(element_string, "/ForceField/HarmonicAngleForce")
               
        # # Last are all TORSIONS
        print('Generating dummy torsion entries ...')

        unique_torsion_set = set()
        proper_string = '<Proper type1="{atomType1}" type2="{atomType2}" type3="{atomType3}" type4="{atomType4}" periodicity1="{periodicity}" phase1="{phase}" k1="{k}"/>'
        for state in range(len(self.mol_array)):

            print('@@@@@@@@@@@@')
            print('State: ', state)

            list_of_torsion_atom_names, list_of_torsion_atom_types = _get_all_torsion(self.network, self.atom_types_dict, state)
            for i, nodes in enumerate(list_of_torsion_atom_names):

                node1, node2, node3, node4 = nodes
                key = hash((node1, node2, node3, node4))


                # torsion between two dummy atoms
                if list_of_torsion_atom_types[i].count(0) > 1:
                    #print('##############')
                    #print(node1, node2, node3, node4)

                    #print('Dummy to dummy')
                    continue

                elif 0 in list_of_torsion_atom_types[i]:
                    #print('##############')
                    #print(node1, node2, node3, node4)

                    original_atom_type1, original_atom_type2, original_atom_type3, original_atom_type4 = list_of_torsion_atom_types[i]
                    for torsion_types in self.all_torsionss_at_all_states[key].values():
                        if 0 in torsion_types:
                            continue
                        else:
                            new_atom_type1, new_atom_type2, new_atom_type3, new_atom_type4 = torsion_types
                            parm = self._retrieve_parameters(atom_type1=new_atom_type1, atom_type2=new_atom_type2, atom_type3=new_atom_type3, atom_type4=new_atom_type4)

                            if str(original_atom_type1) == '0':
                                t, real_atom_type = _return_real_atom_type(self.atom_types_dict, node1)
                                original_atom_type1 = 'd' + str(node1) + real_atom_type
                            elif str(original_atom_type2) == '0':
                                t, real_atom_type = _return_real_atom_type(self.atom_types_dict, node2)
                                original_atom_type2 = 'd' + str(node2) + real_atom_type
                            elif str(original_atom_type3) == '0':
                                t, real_atom_type = _return_real_atom_type(self.atom_types_dict, node3)
                                original_atom_type3 = 'd' + str(node3) + real_atom_type
                            else:
                                t, real_atom_type = _return_real_atom_type(self.atom_types_dict, node4)
                                original_atom_type4 = 'd' + str(node4) + real_atom_type

                            if (original_atom_type1, original_atom_type2, original_atom_type3, original_atom_type4) in unique_torsion_set:
                                #print('Repetition!')
                                continue
                            else:
                                unique_torsion_set.add((original_atom_type1, original_atom_type2, original_atom_type3, original_atom_type4))

                                for par in parm['proper']:
                                    periodicity1 = par.attrib['periodicity1']
                                    phase1 = par.attrib['phase1']
                                    k1 = par.attrib['k1']

                                    element_string= etree.fromstring(proper_string.format(atomType1=original_atom_type1, atomType2=original_atom_type2, atomType3=original_atom_type3, atomType4=original_atom_type4, periodicity=periodicity1, phase=phase1, k=k1))
                                    self._add_to_output(element_string, "/ForceField/PeriodicTorsionForce")


        print('@@@@@@@@@@@@')      
        print('Finished ...')
        


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

    
    def _register_tautomers(self, mols:list):
        def _generate_hydrogen_atom_name(mol, atom):
            # all hydrogen atom names consist of the assigned atom name (e.g. H7) and the atom TYPE of the 
            # bonded heavy atom (e.g. ca) => H7ca

            atom_name = atom.GetProp('name')

            for bond in mol.GetBonds():
                if bond.GetBeginAtom().GetProp('name') == atom_name:
                    heavy_atom = bond.GetEndAtom()
                    break
                if bond.GetEndAtom().GetProp('name') == atom_name:
                    heavy_atom = bond.GetBeginAtom()
                    break
            
            atom_name = atom_name + heavy_atom.GetProp('type')
            return atom_name, heavy_atom.GetProp('type')
            
       
        G = nx.Graph()
        # generate associated state dictionaries for nodes
        atom_types_dict = defaultdict(list)
        atom_charge_dict = defaultdict(list)
        atom_binds_to_atom_type = defaultdict()
        
        for mol in mols:
            for atom in mol.GetAtoms():
                atom_name = atom.GetProp('name')
                if atom_name.startswith('H'):
                    atom_name, heavy_atom_type = _generate_hydrogen_atom_name(mol, atom)
                    # every hydrogen atom binds only to a single heavy_atom_type
                    atom_binds_to_atom_type[atom_name] = heavy_atom_type 

                atom_types_dict[atom_name] = [0]
                atom_charge_dict[atom_name] = [0]

        # write arrays of zeros for all atom names
        for key in atom_types_dict:
            for mol in mols[1:]:
                atom_types_dict[key].append(0)
                atom_charge_dict[key].append(0)


        # generate union mol graph
        for index, mol in enumerate(mols):
            # set nodes
            for atom in mol.GetAtoms():
                atom_name = atom.GetProp('name')
                if atom_name.startswith('H'):
                    atom_name, heavy_atom_type = _generate_hydrogen_atom_name(mol, atom)

                atom_type = atom.GetProp('type')
                atom_charge = atom.GetProp('charge')
                atom_types_dict[atom_name][index] = atom_type
                atom_charge_dict[atom_name][index] = atom_charge

                G.add_node(atom_name)

            # set bonds
            for bond in mol.GetBonds():
                a1 = bond.GetBeginAtom()
                a2 = bond.GetEndAtom()
                
                a1_name = a1.GetProp('name')
                if a1_name.startswith('H'):
                    a1_name, heavy_atom_type = _generate_hydrogen_atom_name(mol, a1)

                a2_name = a2.GetProp('name')
                if a2_name.startswith('H'):
                    a2_name, heavy_atom_type = _generate_hydrogen_atom_name(mol, a2)


                G.add_edge(a1_name, a2_name)


        nx.draw(G, pos=nx.kamada_kawai_layout(G), with_labels=True, font_weight='bold', node_size=1400, alpha=0.5, font_size=12)
        plt.show()       
        all_angles_at_all_states = defaultdict(dict)

        #generate all angles dict
        for state in range(len(mols)):
            list_of_angles_atom_types, list_of_angles_atom_names = _get_all_angles(G, atom_types_dict, state)

            for i, nodes in enumerate(list_of_angles_atom_names):
                node1, node2, node3 = nodes

                key = hash((node1, node2, node3))
                atom_type1, atom_type2, atom_type3 = list_of_angles_atom_types[i]
                all_angles_at_all_states[key][state] = [atom_type1, atom_type2, atom_type3]


        all_torsionss_at_all_states = defaultdict(dict)

        #generate all dihedral dict
        for state in range(len(mols)):
            list_of_torsion_atom_names, list_of_torsion_atom_types = _get_all_torsion(G, atom_types_dict, state)

            for i, nodes in enumerate(list_of_torsion_atom_names):
                node1, node2, node3, node4 = nodes

                key = hash((node1, node2, node3, node4))
                atom_type1, atom_type2, atom_type3, atom_type4 = list_of_torsion_atom_types[i]
                all_torsionss_at_all_states[key][state] = [atom_type1, atom_type2, atom_type3, atom_type4]


        self.network = G
        self.atom_types_dict = atom_types_dict
        self.atom_charge_dict = atom_charge_dict
        self.atom_binds_to_atom_type = atom_binds_to_atom_type
        self.all_angles_at_all_states = all_angles_at_all_states
        self.all_torsionss_at_all_states = all_torsionss_at_all_states

    
 
    def _complete_state_registry(self):
        """
        Store all the properties that are specific to each state
        """

        # mol array
        mol_array = []
        # get charge property for each mol from ForceField
        for index, state in enumerate(self._input_state_data):
            mapping_atom_name_to_charge = {}
            for xml_atom in state['ffxml'].xpath('/ForceField/Residues/Residue/Atom'):
                mapping_atom_name_to_charge[xml_atom.attrib['name']] = xml_atom.attrib['charge']

            # set charges for each atom of rdkit mol object
            mol = state['mol']
            for atom in mol.GetAtoms():
                atom_name = atom.GetProp('name')
                atom.SetProp('charge', mapping_atom_name_to_charge[atom_name]) 
            mol_array.append(mol)

        self._register_tautomers(mol_array)
        self.mol_array = mol_array
           
            

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
        Set up the residue template using the first state of the molecule
        """

        residue = self.ffxml.xpath('/ForceField/Residues/Residue')[0]

        atom_string = '<Atom name="{name}" type="{atom_type}" charge="{charge}"/>'
        bond_string = '<Bond atomName1="{atomName1}" atomName2="{atomName2}"/>'

        for node in self.network:
            atom_type = self.atom_types_dict[node][0]
            if str(atom_type) == '0':
                idx,atom_type = _return_real_atom_type(self.atom_types_dict, node)
                atom_charge = 0.0
                atom_type='d' + str(node) + str(atom_type)
                residue.append(etree.fromstring(atom_string.format(name=node, atom_type=atom_type, charge=atom_charge)))
            else:
                atom_charge = self.atom_charge_dict[node][0]
                residue.append(etree.fromstring(atom_string.format(name=node, atom_type=atom_type, charge=atom_charge)))
         
        
        for bond in self.network.edges:
            atomName1 = bond[0]
            atomName2 = bond[1]
            atom_type1 = self.atom_types_dict[atomName1][0]
            atom_type2 = self.atom_types_dict[atomName2][0]
            if str(atom_type1) == '0':
                idx,atom_type1 = _return_real_atom_type(self.atom_types_dict, atomName1)
                atom_type1='d' + str(atomName1) + str(atom_type1)              
            elif str(atom_type2) == '0':
                idx,atom_type2 = _return_real_atom_type(self.atom_types_dict, atomName2)
                atom_type2='d'+ str(atomName2) + str(atom_type2)


            residue.append(etree.fromstring(bond_string.format(atomName1=atomName1, atomName2=atomName2)))

    def _add_isomers(self):
        def _generate_hydrogen_dummy_atom_type(node):
            # hydrogen dummy atom type includes information when in which environment the dummy becomes real
            # e.g. dH5caha means that H5 becomes real when the heavy atom type is 'ca' - then H5 becomes atom type 'ha'
            idx, atom_type = _return_real_atom_type(self.atom_types_dict, node)
            return 'd' + str(node) + atom_type

        """
        Add all the isomer specific data to the xml template.
        """
        logging.info('Add isomer information ...')

        protonsdata = etree.fromstring("<Protons/>")
        protonsdata.attrib['number_of_states'] = str(len(self._state_templates))

        atom_string = '<Atom name="{name}" type="{atom_type}" charge="{charge}" epsilon="{epsilon}" sigma="{sigma}" />'
        bond_string = '<Bond name1="{atomName1}" name2="{atomName2}" length="{bond_length}" k="{k}"/>'
        angle_string = '<Angle name1="{atomName1}" name2="{atomName2}" name3="{atomName3}" angle="{angle}" k="{k}"/>'
        proper_string = '<Proper name1="{atomName1}" name2="{atomName2}" name3="{atomName3}" name4="{atomName4}" periodicity1="{periodicity1}" phase1="{phase1}" k1="{k1}" periodicity2="{periodicity2}" phase2="{phase2}" k2="{k2}" periodicity3="{periodicity3}" phase3="{phase3}" k3="{k3}" periodicity4="{periodicity4}" phase4="{phase4}" k4="{k4}" periodicity5="{periodicity5}" phase5="{phase5}" k5="{k5}" periodicity6="{periodicity6}" phase6="{phase6}" k6="{k6}" />'      
        improper_string = '<Improper name1="{atomName1}" name2="{atomName2}" name3="{atomName3}" name4="{atomName4}" periodicity1="{periodicity}" phase1="{phase}" k1="{k}"/>'


        for residue in self.ffxml.xpath('/ForceField/Residues/Residue'):
    
            for isomer_index, isomer in enumerate(self._state_templates):

                ##############################################
                # atom entries
                logging.info('ISOMER: {}'.format(isomer_index))
                isomer_str = str(isomer)
                logging.info(isomer_str)
                isomer_xml = etree.fromstring(isomer_str)

                for node in self.network:
                    atom_type = self.atom_types_dict[node][isomer_index]                   
                    if str(atom_type) == '0':
                        e = atom_string.format(name=node, atom_type=_generate_hydrogen_dummy_atom_type(node), charge=0,epsilon=0,sigma=0)
                    else:
                        parm = self._retrieve_parameters(atom_type1=atom_type)                       
                        sigma= parm['nonbonds'].attrib['sigma']
                        epsilon = parm['nonbonds'].attrib['epsilon']
                        e = atom_string.format(name=node, atom_type=atom_type, charge=self.atom_charge_dict[node][isomer_index],epsilon=epsilon,sigma=sigma)
                    
                    isomer_xml.append(etree.fromstring(e))
                    logging.info(e)

                ##############################################
                # bond entries
                for bond in self.network.edges:
                    atomName1 = bond[0]
                    atomName2 = bond[1]
                    
                    atom_type1 = self.atom_types_dict[atomName1][isomer_index]
                    atom_type2 = self.atom_types_dict[atomName2][isomer_index]

                    if str(atom_type1) == '0':
                        idx, atom_type1 = _return_real_atom_type(self.atom_types_dict, atomName1)
                        helper_atom_type2 = (self.atom_binds_to_atom_type[atomName1])
                        parm = self._retrieve_parameters(atom_type1=atom_type1, atom_type2=helper_atom_type2)
                
                    elif str(atom_type2) == '0':
                        idx, atom_type2 = _return_real_atom_type(self.atom_types_dict, atomName2)
                        helper_atom_type1 = (self.atom_binds_to_atom_type[atomName2])
                        parm = self._retrieve_parameters(atom_type1=helper_atom_type1, atom_type2=atom_type2)
                    else:
                        parm = self._retrieve_parameters(atom_type1=atom_type1, atom_type2=atom_type2)

                    length= parm['bonds'].attrib['length']
                    k= parm['bonds'].attrib['k']

                    e = bond_string.format(atomName1=atomName1, atomName2=atomName2, bond_length=length, k=k)
                    logging.info(e)
                    isomer_xml.append(etree.fromstring(e))

                ##############################################
                # angle entries
                angle_string_for_debug = dict()
                angle_string_for_debug['no-dummy'] = []
                angle_string_for_debug['one-dummy'] = []
                angle_string_for_debug['two-dummy'] = []
                list_of_angles_atom_types, list_of_angles_atom_names = _get_all_angles(self.network, self.atom_types_dict, isomer_index)

                for i, nodes in enumerate(list_of_angles_atom_names):
                    node1, node2, node3 = nodes
                    key = hash((node1, node2, node3))
                    # angle between two dummy atoms
                    if list_of_angles_atom_types[i].count(0) == 2:
                        original_atom_type1, original_atom_type2, original_atom_type3 = list_of_angles_atom_types[i]
                        angle= float(0.0)
                        k= float(0.0)
                        e = angle_string.format(atomName1=node1, atomName2=node2, atomName3=node3, angle=angle, k=k)
                        angle_string_for_debug['two-dummy'].append(e)
                        isomer_xml.append(etree.fromstring(e))
                        continue

                    # angle with a single dummy atom 
                    elif list_of_angles_atom_types[i].count(0) == 1:
                        original_atom_type1, original_atom_type2, original_atom_type3 = list_of_angles_atom_types[i]
                        for angles_types in self.all_angles_at_all_states[key].values():
                            if 0 in angles_types:
                                continue
                            else:
                                new_atom_type1, new_atom_type2, new_atom_type3 = angles_types
                                parm = self._retrieve_parameters(atom_type1=new_atom_type1, atom_type2=new_atom_type2, atom_type3=new_atom_type3)

                                angle= parm['angle'].attrib['angle']
                                k= parm['angle'].attrib['k']
                                e= angle_string.format(atomName1=node1, atomName2=node2, atomName3=node3, angle=angle, k=k)
                                angle_string_for_debug['one-dummy'].append(e)
                                isomer_xml.append(etree.fromstring(e))

                    # angles between real atoms
                    elif list_of_angles_atom_types[i].count(0) == 0:
                        original_atom_type1, original_atom_type2, original_atom_type3 = list_of_angles_atom_types[i]
                        parm = self._retrieve_parameters(atom_type1=original_atom_type1, atom_type2=original_atom_type2, atom_type3=original_atom_type3)
                        angle= parm['angle'].attrib['angle']
                        k= parm['angle'].attrib['k']
                        e = angle_string.format(atomName1=node1, atomName2=node2, atomName3=node3, angle=angle, k=k)
                        angle_string_for_debug['no-dummy'].append(e)
                        isomer_xml.append(etree.fromstring(e))
                    else:
                        logging.warning('WHAT IS GOING ON? MORE THAN 3 DUMMY TYPES FOR ANGLE!??!??')

                # printing debug info
                for k in angle_string_for_debug:
                    logging.info(' - Angles {} ...'.format(k))
                    for e in angle_string_for_debug[k]:
                        logging.info('  : {}'.format(e))

                ##############################################
                # torsion entries
                proper_string_for_debug = dict()
                proper_string_for_debug['no-dummy'] = []
                proper_string_for_debug['one-dummy'] = []
                proper_string_for_debug['two-dummy'] = []
                proper_string_for_debug['always-dummy'] = []

                list_of_torsion_atom_names, list_of_torsion_atom_types = _get_all_torsion(self.network, self.atom_types_dict, isomer_index)

                for i, nodes in enumerate(list_of_torsion_atom_names):
                    node1, node2, node3, node4 = nodes
                    key = hash((node1, node2, node3, node4))
                    # torsion between two dummy atoms
                    if list_of_torsion_atom_types[i].count(0) > 1:
                        periodicity_list = [1] * 6
                        phase_list = [0.0] * 6
                        k_list = [0.0] * 6

                        e = proper_string.format(atomName1=node1, atomName2=node2, atomName3=node3, atomName4=node4, 
                        periodicity1=str(periodicity_list[0]), phase1=str(phase_list[0]), k1=str(k_list[0]), 
                        periodicity2=str(periodicity_list[1]), phase2=str(phase_list[1]), k2=str(k_list[1]), 
                        periodicity3=str(periodicity_list[2]), phase3=str(phase_list[2]), k3=str(k_list[2]), 
                        periodicity4=str(periodicity_list[3]), phase4=str(phase_list[3]), k4=str(k_list[3]),
                        periodicity5=str(periodicity_list[4]), phase5=str(phase_list[4]), k5=str(k_list[4]), 
                        periodicity6=str(periodicity_list[5]), phase6=str(phase_list[5]), k6=str(k_list[5]) )

                        proper_string_for_debug['two-dummy'].append(e)
                        isomer_xml.append(etree.fromstring(e))

                    # torsion between real atoms
                    elif 0 not in list_of_torsion_atom_types[i]:
                        atom_type1, atom_type2, atom_type3, atom_type4 = list_of_torsion_atom_types[i]
                        parm = self._retrieve_parameters(atom_type1=atom_type1, atom_type2=atom_type2, atom_type3=atom_type3, atom_type4=atom_type4)

                        periodicity_list = [1] * 6
                        phase_list = [0.0] * 6
                        k_list = [0.0] * 6
                        offset = 0
                        #print(node1, node2, node3, node4)
                        #print(parm['proper'])
                        for par in parm['proper']:
                            nr_of_par = ['period' in x for x in list(par.attrib)].count(True)    
                            #print(nr_of_par)                       
                            for i in range(1, nr_of_par+1):
                                periodicity_list[offset + i-1] = (par.attrib['periodicity' + str(i)])
                                phase_list[offset+i-1] = (par.attrib['phase' + str(i)])
                                k_list[offset+i-1] = (par.attrib['k' + str(i)])
                            offset = nr_of_par
                        
                        e = proper_string.format(atomName1=node1, atomName2=node2, atomName3=node3, atomName4=node4, 
                        periodicity1=str(periodicity_list[0]), phase1=str(phase_list[0]), k1=str(k_list[0]), 
                        periodicity2=str(periodicity_list[1]), phase2=str(phase_list[1]), k2=str(k_list[1]), 
                        periodicity3=str(periodicity_list[2]), phase3=str(phase_list[2]), k3=str(k_list[2]), 
                        periodicity4=str(periodicity_list[3]), phase4=str(phase_list[3]), k4=str(k_list[3]),
                        periodicity5=str(periodicity_list[4]), phase5=str(phase_list[4]), k5=str(k_list[4]), 
                        periodicity6=str(periodicity_list[5]), phase6=str(phase_list[5]), k6=str(k_list[5]))

                        isomer_xml.append(etree.fromstring(e))
                        proper_string_for_debug['no-dummy'].append(e)

                    # torsion which includes a single dummy atom
                    else:
                        found_real_torsion = False
                        for torsion_types in self.all_torsionss_at_all_states[key].values():
                            if 0 in torsion_types:
                                continue
                            else:
                                found_real_torsion = True
                                new_atom_type1, new_atom_type2, new_atom_type3, new_atom_type4 = torsion_types
                                parm = self._retrieve_parameters(atom_type1=new_atom_type1, atom_type2=new_atom_type2, atom_type3=new_atom_type3, atom_type4=new_atom_type4)

                        periodicity_list = [1] * 10
                        phase_list = [0.0] * 10
                        k_list = [0.0] * 10
                        offset = 0
                        for par in parm['proper']:
                            nr_of_par = ['period' in x for x in list(par.attrib)].count(True)    
                            for i in range(1, nr_of_par+1):
                                periodicity_list[offset + i-1] = (par.attrib['periodicity' + str(i)])
                                phase_list[offset+i-1] = (par.attrib['phase' + str(i)])
                                k_list[offset+i-1] = (par.attrib['k' + str(i)])
                            offset = nr_of_par
                                                  
                        e = proper_string.format(atomName1=node1, atomName2=node2, atomName3=node3, atomName4=node4, 
                        periodicity1=str(periodicity_list[0]), phase1=str(phase_list[0]), k1=str(k_list[0]), 
                        periodicity2=str(periodicity_list[1]), phase2=str(phase_list[1]), k2=str(k_list[1]), 
                        periodicity3=str(periodicity_list[2]), phase3=str(phase_list[2]), k3=str(k_list[2]), 
                        periodicity4=str(periodicity_list[3]), phase4=str(phase_list[3]), k4=str(k_list[3]),
                        periodicity5=str(periodicity_list[4]), phase5=str(phase_list[4]), k5=str(k_list[4]), 
                        periodicity6=str(periodicity_list[5]), phase6=str(phase_list[5]), k6=str(k_list[5]))

                        isomer_xml.append(etree.fromstring(e))
                        proper_string_for_debug['one-dummy'].append(e)
                           
                        # there might be 4 atoms that always contain a dummy at each state
                        # - these torsions are not real therefore everything is set to zero
                        if found_real_torsion == False:
                            periodicity_list = [1] * 10
                            phase_list = [0.0] * 10
                            k_list = [0.0] * 10

                            e = proper_string.format(atomName1=node1, atomName2=node2, atomName3=node3, atomName4=node4, 
                            periodicity1=str(periodicity_list[0]), phase1=str(phase_list[0]), k1=str(k_list[0]), 
                            periodicity2=str(periodicity_list[1]), phase2=str(phase_list[1]), k2=str(k_list[1]), 
                            periodicity3=str(periodicity_list[2]), phase3=str(phase_list[2]), k3=str(k_list[2]), 
                            periodicity4=str(periodicity_list[3]), phase4=str(phase_list[3]), k4=str(k_list[3]),
                            periodicity5=str(periodicity_list[4]), phase5=str(phase_list[4]), k5=str(k_list[4]), 
                            periodicity6=str(periodicity_list[5]), phase6=str(phase_list[5]), k6=str(k_list[5]))
                            proper_string_for_debug['always-dummy'].append(e)
                            isomer_xml.append(etree.fromstring(e))



                for k in proper_string_for_debug:
                    logging.info(' - Proper {}...'.format(k))
                    for e in proper_string_for_debug[k]:
                        logging.info('  : {}'.format(e))


                #http://alma.karlov.mff.cuni.cz/bio/99_Studenti/00_Dalsi/ParamFit/2013_ParamFit_AmberTools13.pdf

                #list_of_improper_atom_names, list_of_improper_atom_types = _get_all_improper(self.network, self.atom_types_dict, isomer_index)
                # print('############')
                # print(list_of_improper_atom_names)
                # print('############')
                # print(list_of_improper_atom_types)
                # print('############')
                # set_of_improper = set()
                # for xmltree in self._xml_parameter_trees[1:]:
                #     for improper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Improper"):
                #         print(improper)
                #         periodicity = improper.attrib['periodicity1']
                #         phase = improper.attrib['phase1']
                #         k = improper.attrib['k1']
                #         a1_type = improper.attrib['type1']
                #         a2_type = improper.attrib['type2']
                #         a3_type = improper.attrib['type3']
                #         a4_type = improper.attrib['type4']
                #         for i, atom_types in enumerate(list_of_improper_atom_types):
                #             node1_type, node2_type, node3_type, node4_type = atom_types
                #             if (a1_type, a2_type, a3_type, a4_type) == (node1_type, node2_type, node3_type, node4_type):

                #                 node1, node2, node3, node4 = list_of_improper_atom_names[i]

                #                 if (node1, node2, node3, node4) in set_of_improper:
                #                     continue
                #                 else:
                #                     print('##############################')
                #                     set_of_improper.add((node1, node2, node3, node4))
                #                     e = improper_string.format(atomName1=node1, atomName2=node2, atomName3=node3, atomName4=node4, periodicity=str(periodicity), phase=str(phase), k=str(k))
                #                     isomer_xml.append(etree.fromstring(e))
                #                     print('##############################')

                

                protonsdata.append(isomer_xml)
            residue.append(protonsdata)

    def _append_extra_gaff_types(self):
        """
        Add additional parameters generated by antechamber/parmchk for the individual isomers
        """
        added_parameters = list()  # for bookkeeping of duplicates
        improper_dict = dict()

        # in order to avoid overwriting the proper and improper parameter entries 
        # in self._xml_parameter_trees a copy is created
        # mw: I have no idea, why this overwrites, this is a ugly hack to avoid this issue
        xml_trees = deepcopy(self._xml_parameter_trees)

        # All xml sources except the entire gaff.xml
        for idx, xmltree in enumerate(xml_trees[1:]):
            improper_dict[idx] = []
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


            #print('#####################')
            #print('First iteration')
            #for xmltree in self._xml_parameter_trees[1:]:
            #    for proper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Proper"):
            #        items = set(proper.items())
            #        proper_element = tuple(["PeriodicTorsionForce", "Proper", items])
            #        print(proper_element)
            #print('!!!!!!!!!!!!!!')

            # Match proper dihedral of the atom in PeriodicTorsionForce block
            for proper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Proper"):
                items = set(proper.items())
                proper_element = tuple(["PeriodicTorsionForce", "Proper", items])
                # Make sure the force wasn't already added by a previous state
                if proper_element not in added_parameters:
                    added_parameters.append(proper_element)
                    self._add_to_output(proper, "/ForceField/PeriodicTorsionForce")
                self._add_to_output(proper, "/ForceField/PeriodicTorsionForce")
                pass

            #print('#####################')
            #print('Second iteration')
            #for xmltree in self._xml_parameter_trees[1:]:
            #    for proper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Proper"):
            #        items = set(proper.items())
            #        proper_element = tuple(["PeriodicTorsionForce", "Proper", items])
            #        print(proper_element)
            #print('!!!!!!!!!!!!!!')
            #print('#####################')


            # Match improper dihedral of the atom in PeriodicTorsionForce block
            for improper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Improper"):
                g = dict()
                for t in improper.items():
                    g[t[0]] = t[1]
                items = set(improper.items())
                improper_element = tuple(["PeriodicTorsionForce", "Improper", items])
                # Make sure the force wasn't already added by a previous state
                if improper_element not in added_parameters:
                    added_parameters.append(improper_element)
                    # all impropers should be 
                    self._add_to_output(improper, "/ForceField/PeriodicTorsionForce")
                    improper_dict[idx].append(g)

            # Match nonbonded type of the atom in NonbondedForce block
            for nonbond in xmltree.xpath("/ForceField/NonbondedForce/Atom"):
                items = set(nonbond.items())
                nb_element = tuple(["NonbondedForce", "Atom", items])
                # Make sure the force wasn't already added by a previous state
                if nb_element not in added_parameters:
                    added_parameters.append(nb_element)
                    self._add_to_output(nonbond, "/ForceField/NonbondedForce")
        




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
                    if (kwargs['atom_type1'] == bond.attrib['type1'] and kwargs['atom_type2'] == bond.attrib['type2']) or (kwargs['atom_type2'] == bond.attrib['type1'] and kwargs['atom_type1'] == bond.attrib['type2']):
                        params['bonds'] = bond
            return params
                    

        elif len(kwargs) == 3:
            for xmltree in self._xml_parameter_trees:
                # Match the angles of the atom in the HarmonicAngleForce block
                for angle in xmltree.xpath("/ForceField/HarmonicAngleForce/Angle"):
                    angle_atom_types_list = [angle.attrib['type1'], angle.attrib['type2'], angle.attrib['type3']]
                    search_list = [kwargs['atom_type1'], kwargs['atom_type2'], kwargs['atom_type3']]
                    if search_list[0] == angle_atom_types_list[0] and search_list[1] == angle_atom_types_list[1] and search_list[2] == angle_atom_types_list[2]:
                        params['angle'] = angle
                        return params
                    elif search_list[2] == angle_atom_types_list[0] and search_list[1] == angle_atom_types_list[1] and search_list[0] == angle_atom_types_list[2]:
                        params['angle'] = angle
                        return params
                    else:
                        continue
            return params
            
        
        elif len(kwargs) == 4:
            #for xmltree in self._xml_parameter_trees[1:]:
            #   # Match proper dihedral of the atom in PeriodicTorsionForce block
            #    print('all propers that I see')
            #    print(etree.tostring(xmltree))
            #    print('$$$$$$$$$')

            # match torsion parameters
            par = []
            generic = []
            search_list = [kwargs['atom_type1'], kwargs['atom_type2'], kwargs['atom_type3'], kwargs['atom_type4']]
            for xmltree in self._xml_parameter_trees:
                # Match proper dihedral of the atom in PeriodicTorsionForce block
                for proper in xmltree.xpath("*/Proper"):
                    # create matching list of torsion atom types
                    torsion_types_list = [proper.attrib['type1'], proper.attrib['type2'], proper.attrib['type3'], proper.attrib['type4']]
                    # Start with matching the two central atoms of the torsion - this could now apply to either wildcard torsion or specific torsions
                    if search_list[1] == torsion_types_list[1] and search_list[2] == torsion_types_list[2]:
                        if (torsion_types_list[0] == search_list[0]) and (torsion_types_list[3] == search_list[3]):
                            par.append(proper)
                        elif torsion_types_list[0] == '' and torsion_types_list[3] == '':
                            generic.append(proper)

                    elif search_list[1] == torsion_types_list[2] and search_list[2] == torsion_types_list[1]:
                        if (search_list[3] == torsion_types_list[0]) and (search_list[0] == torsion_types_list[3]):
                            par.append(proper)
                        elif torsion_types_list[0] == '' and torsion_types_list[3] == '':
                            generic.append(proper)
                            
            params['proper'] = generic + par
            #print('$$$$$$$$$$$$$$$$$$')
            #for i in generic + par:
            #    print(etree.tostring(i))
            #print('$$$$$$$$$$$$$$$$$$')

            for xmltree in self._xml_parameter_trees:

                par = []
                # Match improper dihedral of the atom in PeriodicTorsionForce block
                for improper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Improper"):
                    improp_types_list = [improper.attrib['type1'], improper.attrib['type2'], improper.attrib['type3'], improper.attrib['type4']]
                    search_list = [kwargs['atom_type1'], kwargs['atom_type2'], kwargs['atom_type3'], kwargs['atom_type4']]
                    # Start with matching the two central atoms of the torsion - this could now apply to either wildcard torsion or specific torsions
                    if search_list[1] ==  improp_types_list[1] and search_list[2] ==  improp_types_list[2]:
                        if improp_types_list[0] == '' and improp_types_list[3] == '':
                            par.append(improper)
                        elif search_list[0] == improp_types_list[0] and search_list[3] == improp_types_list[3]:
                            # foudn a specific torsion!
                            par.append(improper)
                        else:
                            continue

                    elif search_list[2] ==  improp_types_list[1] and search_list[1] ==  improp_types_list[2]:

                        if improp_types_list[0] == '' and improp_types_list[3] == '':
                            # found an unspecific improp! will use it!
                            par.append(improper)
                        elif search_list[0] == improp_types_list[0] and search_list[3] == improp_types_list[3]:
                            # foudn a specific torsion!
                            par.append(improper)
                        else:
                            continue

            params['improper'] = par


            return params
  

def _get_all_angles(network, atom_types_dict, state):

    list_of_angles_atom_types = []
    list_of_angles_atom_names = []
    for node1 in network:
        for node2 in network:
            if node2.startswith('H'):
                continue
            elif not network.has_edge(node1, node2):
                continue
            else:
                for node3 in network:
                    if not network.has_edge(node2, node3):
                        continue
                    elif node3 == node1:
                        continue
                    else:
                        atom_type1 = atom_types_dict[node1][state]
                        atom_type2 = atom_types_dict[node2][state]
                        atom_type3 = atom_types_dict[node3][state]
                        list_of_angles_atom_types.append([atom_type1, atom_type2, atom_type3])
                        list_of_angles_atom_names.append([node1, node2, node3])
    
    return list_of_angles_atom_types, list_of_angles_atom_names


def _get_all_torsion(network, atom_types_dict, state):

    list_of_torsion_atom_types = []
    list_of_torsion_atom_names = []

    for node1 in network:
        for node2 in network:
            if node2.startswith('H'):
                continue
            elif not network.has_edge(node1, node2):
                continue
            else:
                for node3 in network:
                    if not network.has_edge(node2, node3):
                        continue
                    elif node3 == node1:
                        continue
                    else:
                        for node4 in network:
                            if not network.has_edge(node3, node4):
                                continue
                            elif node4 == node2:
                                continue
                            else:

                                atom_type1 = atom_types_dict[node1][state]
                                atom_type2 = atom_types_dict[node2][state]
                                atom_type3 = atom_types_dict[node3][state]
                                atom_type4 = atom_types_dict[node4][state]

                                list_of_torsion_atom_types.append([atom_type1, atom_type2, atom_type3, atom_type4])
                                list_of_torsion_atom_names.append([node1, node2, node3, node4])

    return list_of_torsion_atom_names, list_of_torsion_atom_types

def _get_all_improper(network, atom_types_dict, state):

    list_of_improper_atom_types = []
    list_of_improper_atom_names = []



    for node1 in network:
        if node1.startswith('H'):
            continue
        else:
            for node2 in network:
                if node2 == node1:
                    continue
                elif not network.has_edge(node1, node2):
                    continue
                else:
                    for node3 in network:
                        if node3 == node1 or node3 == node2:
                            continue
                        elif not network.has_edge(node1, node3):
                            continue
                        else:
                            for node4 in network:
                                if node4 == node1 or node4 == node2 or node4 == node3:
                                    continue
                                elif not network.has_edge(node1, node4):
                                    continue
                                else:
                                    atom_type1 = atom_types_dict[node1][state]
                                    atom_type2 = atom_types_dict[node2][state]
                                    atom_type3 = atom_types_dict[node3][state]
                                    atom_type4 = atom_types_dict[node4][state]

                                    if 0 in [atom_type2, atom_type3, atom_type4]:
                                        continue

                                    # Yes, sorted by atom type! Therefore undefined if 2 atom types are the same
                                    # ref: https://github.com/pandegroup/openmm/issues/220

                                    sorting_dict = dict()
                                    sorting_dict[node1] = atom_type1
                                    sorting_dict[node2] = atom_type2
                                    sorting_dict[node3] = atom_type3
                                    sorting_dict[node4] = atom_type4
                                    print('!!!!!!!!!!!!!!')
                                    print([node1, node2, node3, node4])
                                    print([atom_type1, atom_type2, atom_type3, atom_type4])
                                    print('!!!!!!!!!!!!!!')
                                    sorted_atom_names = []
                                    sorted_atom_types = sorted([atom_type2, atom_type3, atom_type4])
                                    sorted_atom_types.insert(2,atom_type1)


                                    list_of_improper_atom_names.append(sorted_atom_names)
                                    list_of_improper_atom_types.append(sorted_atom_types)


    return list_of_improper_atom_names, list_of_improper_atom_types

def _return_adjacent_node(network, node):

    return network.neighbors(node)

def _return_real_atom_type(atom_types_dict, node_name):
    for idx, atom_type in enumerate(atom_types_dict[node_name]):
        if str(atom_type) != '0':
            return idx, atom_type


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

def prepare_calibration_system(vacuum_file:str, output_file:str, ffxml: str=None, hxml:str=None, delete_old_H:bool=True):
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
    #    modeller.delete(to_delete)

    modeller.addHydrogens(forcefield=forcefield)
    modeller.addSolvent(forcefield, model='tip3p', padding=1.0 * nanometers, neutralize=False)

    system = forcefield.createSystem(modeller.topology, nonbondedMethod=app.PME, nonbondedCutoff=1.0 * nanometers,
                                     constraints=app.HBonds, rigidWater=True,
                                     ewaldErrorTolerance=0.0005)
    system.addForce(openmm.MonteCarloBarostat(1.0 * atmosphere, 300.0 * kelvin))
    simulation = app.Simulation(modeller.topology, system, GBAOABIntegrator())
    simulation.context.setPositions(modeller.positions)
    #simulation.minimizeEnergy()

    app.PDBxFile.writeFile(modeller.topology, simulation.context.getState(getPositions=True).getPositions(),
                           open(output_file, 'w'))
    
    app.PDBFile.writeFile(modeller.topology, simulation.context.getState(getPositions=True).getPositions(),
                           open('/home/mwieder/input.pdb', 'w'))

    
    return simulation