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

PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))


gaff_default = os.path.join(PACKAGE_ROOT, 'data', 'gaff.xml')


class _State(object):
    """
    Private class representing a template of a single isomeric state of the molecule.
    """
    def __init__(self, index, log_population, g_k, atom_name, net_charge, pH):
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

    return outputffxml


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

    print('FIND HYDROGEN TYPES!!!')
    # Detect all hydrogen types by element and store them in a set
    hydrogen_types = set()
    for atomtype in gafftree.xpath('AtomTypes/Type'):
        if atomtype.get('element') == "H":
            hydrogen_types.add(atomtype.get('name'))

    for atomtype in xmlfftree.xpath('AtomTypes/Type'):
        # adds dummy atome types
        if atomtype.get('name').startswith("d_"):
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

class _Unison_States_Mol(object):
    """
    Class that takes care of the different atom/bond types at different states of the atoms.
    It also creates dummy definitions and adds dummy angle and torsion entries to the custom force field. 
    """  

    def __init__(self, mol):
        self.mol = mol
    """
    Parameters
    ----------
    mol : rdkit Mol object with all heavy atoms
        
    
    """

    @staticmethod
    def _get_real_atom_type(atom, nr_of_states):
        for state in range(int(nr_of_states)):
            atomtype = atom.GetProp('atom_type_at_state_'+str(state))
            if atomtype.startswith('d_'):
                continue
            else:
                return atomtype
    @staticmethod
    def _get_real_torsion_type(a1, a2, a3, a4, nr_of_states):
        found_real_torsion = False
        for state in range(int(nr_of_states)):
            type_a1 = a1.GetProp('atom_type_at_state_'+str(state))
            type_a2 = a2.GetProp('atom_type_at_state_'+str(state))
            type_a3 = a3.GetProp('atom_type_at_state_'+str(state))
            type_a4 = a4.GetProp('atom_type_at_state_'+str(state))

            if type_a1.startswith('d_') or type_a2.startswith('d_') or type_a3.startswith('d_') or type_a4.startswith('d_') :
                continue
            else:
                found_real_torsion = True
                #print(found_real_angle, type_a1, type_a2, type_a3)
                return(found_real_torsion, type_a1, type_a2, type_a3, type_a4)

        return(found_real_torsion, type_a1, type_a2, type_a3, type_a4)
    
    @staticmethod
    def _get_real_angle_type(a1, a2, a3, nr_of_states):
        found_real_angle = False
        for state in range(int(nr_of_states)):
            type_a1 = a1.GetProp('atom_type_at_state_'+str(state))
            type_a2 = a2.GetProp('atom_type_at_state_'+str(state))
            type_a3 = a3.GetProp('atom_type_at_state_'+str(state))

            if type_a1.startswith('d_') or type_a2.startswith('d_') or type_a3.startswith('d_') :
                #print(found_real_angle, type_a1, type_a2, type_a3)
                continue
            else:
                found_real_angle = True
                #print(found_real_angle, type_a1, type_a2, type_a3)
                return(found_real_angle, type_a1, type_a2, type_a3)

        return(found_real_angle, type_a1, type_a2, type_a3)

    @staticmethod
    def _print_xml_atom_string(atom, state, string):
        """
        Generate a populated xml string. Given an rdkit atom and a state it 
        generates either (depending on the atom type of the atom at the given state)
        a xml string with dummy parameters or real parameters. 

        Parameters
        ----------
        atom : rdAtom
        state : int
        string : string
            
        Returns
        -------
        A formated xml string
        """
        atom_type = atom.GetProp('atom_type_at_state_' + str(state))
        name = atom.GetProp('name')
        atom_charge = atom.GetProp('charge_at_state_'+str(state))
        sigma = atom.GetProp('sigma_at_state_' + str(state))
        epsilon = atom.GetProp('epsilon_at_state_' + str(state))
        element = atom.GetSymbol()
        mass = atom.GetMass()
        return string.format(name=name, atom_type=atom_type, charge=atom_charge, sigma=sigma, epsilon=epsilon, element=element, mass=mass)


    @staticmethod 
    def _print_xml_angle_string(atomType1, atomType2, atomType3, parameter, string):
        """
        Generate a populated xml string.

        Parameters
        ----------
        atomType1, atomType2, atomType3 : str
        parameter
        string : xml string
            
        Returns
        -------
        An xml string
        """

        angle = parameter['angle'].attrib['angle']
        k = parameter['angle'].attrib['k']
        return string.format(atomType1=atomType1, atomType2=atomType2, atomType3=atomType3, angle=angle, k=k)

    @staticmethod 
    def _print_xml_torsion_string(atomType1, atomType2, atomType3, atomType4, parameter, string):
        """
        Generate a populated xml string.

        Parameters
        ----------
        atomType1, atomType2, atomType3 : str
        parameter
        string : xml string
            
        Returns
        -------
        An xml string
        """

        periodicity = parameter.attrib['periodicity1']
        phase = parameter.attrib['phase1']
        k = parameter.attrib['k1']
        return string.format(atomType1=atomType1, atomType2=atomType2, atomType3=atomType3, atomType4=atomType4, phase=phase, periodicity=periodicity, k=k)



    @staticmethod
    def _print_xml_bond_string(bond, state, string):
        atomName1 = bond.GetBeginAtom().GetProp('name')
        atomName2 = bond.GetEndAtom().GetProp('name')
        atomType1 = bond.GetBeginAtom().GetProp('atom_type_at_state_'+str(state))
        atomType2 = bond.GetEndAtom().GetProp('atom_type_at_state_'+str(state))
        bond_length = bond.GetProp('bond_length_at_state_' + str(state))
        k = bond.GetProp('k_at_state_' + str(state))

        return string.format(atomName1=atomName1, atomName2=atomName2, atomType1=atomType1, atomType2=atomType2,bond_length=bond_length, k=k)


    def _print_state_of_unison_mol(self):
        mol = self.mol
        nr_of_state = int(self.mol.GetProp('nr_of_states'))
        print('How many states included in unison clone: ', nr_of_state)

        print('Showing atom name mappings for each state ...')
        for atom in mol.GetAtoms():
            print('#################################')
            
            print('Atom-Name: ', atom.GetProp('name'))
            for state in range(nr_of_state):
                print('State: ', str(state), end=' ')
                print('Atom-Type: ',atom.GetProp('atom_type_at_state_'+str(state)), end=' ')
                print('Charge: ', atom.GetProp('charge_at_state_' +str(state)), end=' ')
                print('Sigma: ', atom.GetProp('sigma_at_state_'+str(state)), end=' ')
                print('Epsilon: ', atom.GetProp('epsilon_at_state_'+str(state)))
    
        print('#################################')
        print('#################################')
        for bond in mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            print('#################################')

            print('Bond: between ', a1.GetProp('name'), 'and ', a2.GetProp('name'))
            for state in range(nr_of_state):
                print('State: ', str(state))
                print('Involved Atom-Types: ',a1.GetProp('atom_type_at_state_'+str(state)), end=' ')
                print(a2.GetProp('atom_type_at_state_'+str(state)))
                print('Bond length: ', bond.GetProp('bond_length_at_state_' + str(state)), end=' ')
                print('K: ', bond.GetProp('k_at_state_' + str(state)))

        

    def generate_atom_name_list_for_state(self, state):

        atom_list = []
        for atom in self.mol.GetAtoms():
            if(atom.GetProp('atom_type_at_state_'+str(state)) == 'dummy'):
                continue
            else:
                atom_list.append(atom.GetProp('name'))
        return atom_list


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

        unison_mol = self._unison_clone.mol
        print('Append dummy atoms')
        # Start with adding all dummy atom type definitions
        atom_string = '<Type name="{atom_type}" class="{atom_type}" charge="{charge}" element="{element}" mass="{mass}"/>'
        nb_string = '<Atom type="{atom_type}" sigma="{sigma}" epsilon="{epsilon}" charge="{charge}"/>'
        # get atom parameters
        nr_of_states = int(unison_mol.GetProp('nr_of_states'))
        for atom in unison_mol.GetAtoms():
            for state in range(nr_of_states):
                if atom.GetProp('atom_type_at_state_'+str(state)).startswith('d_'):
                    element_string= etree.fromstring(_Unison_States_Mol._print_xml_atom_string(atom, state, atom_string ))
                    nb_element_string= etree.fromstring(_Unison_States_Mol._print_xml_atom_string(atom, state, nb_string ))
                    self._add_to_output(element_string, "/ForceField/AtomTypes")
                    self._add_to_output(nb_element_string, "/ForceField/NonbondedForce")
        
           
        print('################################')

        # Now add all dummy bonds
        bond_string = '<Bond type1="{atomType1}" type2="{atomType2}" length="{bond_length}" k="{k}"/>'
        # get bond parameters
        for bond in unison_mol.GetBonds():
            for state in range(nr_of_states):
                a1, a2 = bond.GetBeginAtom(), bond.GetEndAtom()
                atom_type1 = a1.GetProp('atom_type_at_state_'+str(state))
                atom_type2 = a2.GetProp('atom_type_at_state_'+str(state))
                if atom_type1.startswith('d_') or atom_type2.startswith('d_'):
                    # the bond of interest is identified
                    element_string= etree.fromstring(_Unison_States_Mol._print_xml_bond_string(bond, state, bond_string ))
                    self._add_to_output(element_string, "/ForceField/HarmonicBondForce")
            


        # Now add all dummy ANGLES
        angle_string = '<Angle type1="{atomType1}" type2="{atomType2}" type3="{atomType3}" angle="{angle}" k="{k}"/>'
        # get angles involving dummy atoms
        patt = Chem.MolFromSmarts('*~*~*')

        angle_list = unison_mol.GetSubstructMatches(patt)
        for state in range(nr_of_states):
            for angle in angle_list:
                a1, a2, a3 = unison_mol.GetAtomWithIdx(angle[0]), unison_mol.GetAtomWithIdx(angle[1]), unison_mol.GetAtomWithIdx(angle[2])
                atomType1 = a1.GetProp('atom_type_at_state_'+str(state))
                atomType2 = a2.GetProp('atom_type_at_state_'+str(state))
                atomType3 = a3.GetProp('atom_type_at_state_'+str(state))
                if atomType1.startswith('d_') or atomType2.startswith('d_') or atomType3.startswith('d_'):
                    real_angle_bool, real_atomType1, real_atomType2, real_atomType3 = _Unison_States_Mol._get_real_angle_type(a1, a2, a3, nr_of_states)
                    if real_angle_bool:
                        parameter = self._retrieve_parameters(atom_type1=real_atomType1, atom_type2=real_atomType2, atom_type3=real_atomType3)
                        # build xml string with real parameters and dummy atom types
                        element_string= etree.fromstring(_Unison_States_Mol._print_xml_angle_string(atomType1, atomType2, atomType3, parameter, angle_string))
                        self._add_to_output(element_string, "/ForceField/HarmonicAngleForce")
                    else:
                        # there is no real angle between these atom types (because at each state there are dummy atoms)
                        print('COuld not find real angle atom types')
                        print(a1.GetProp('name'), ' - ', a2.GetProp('name'), ' - ', a3.GetProp('name') )
                        angle_specific_string = angle_string.format(atomType1=atomType1, atomType2=atomType2, atomType3=atomType3, angle='0.0', k='0.0')
                        element_string = etree.fromstring(angle_specific_string)
                        self._add_to_output(element_string, "/ForceField/HarmonicAngleForce")
                                          
        # Last are all TORSIONS

        proper_string = '<Proper type1="{atomType1}" type2="{atomType2}" type3="{atomType3}" type4="{atomType3}" periodicity1="{periodicity}" phase1="{phase}" k1="{k}"/>'
        improper_string = '<Proper type1="{atomType1}" type2="{atomType2}" type3="{atomType3}" type4="{atomType3}" periodicity1="{periodicity}" phase1="{phase}" k1="{k}"/>'
        patt = Chem.MolFromSmarts('*~*~*~*')

        dihedrals = unison_mol.GetSubstructMatches(patt)
        for state in range(nr_of_states):
            print('#####################')
            print(state)
            print('#####################')
            for torsion in dihedrals:
                a1, a2, a3, a4 = unison_mol.GetAtomWithIdx(torsion[0]), unison_mol.GetAtomWithIdx(torsion[1]), unison_mol.GetAtomWithIdx(torsion[2]), unison_mol.GetAtomWithIdx(torsion[3])
                atomType1, atomType2, atomType3, atomType4 = a1.GetProp('atom_type_at_state_'+str(state)), a2.GetProp('atom_type_at_state_'+str(state)), a3.GetProp('atom_type_at_state_'+str(state)), a4.GetProp('atom_type_at_state_'+str(state))
                
                if atomType1.startswith('d_') or atomType2.startswith('d_') or atomType3.startswith('d_') or atomType4.startswith('d_'):
                    element_string= etree.fromstring(proper_string.format(atomType1=atomType1, atomType2=atomType2, atomType3=atomType3, atomType4=atomType4, periodicity='0', phase='0', k='0'))
                    self._add_to_output(element_string, "/ForceField/PeriodicTorsionForce")

                    real_torsion_bool, real_atomType1, real_atomType2, real_atomType3, real_atomType4 = _Unison_States_Mol._get_real_torsion_type(a1, a2, a3, a4, nr_of_states)
                    print('Involves dummy ...')
                    # print(atomType1, ' - ', atomType2, ' - ', atomType3, ' - ', atomType4)
                    # print(real_atomType1, ' - ', real_atomType2, ' - ', real_atomType3, ' - ', real_atomType4)
                    # if real_torsion_bool:
                    #     parameters = self._retrieve_parameters(atom_type1=real_atomType1, atom_type2=real_atomType2, atom_type3=real_atomType3, atom_type4=real_atomType4)                          
                    #     for torsion_variety in parameters:                              
                    #         for parameter in parameters[torsion_variety]:
                    #             if torsion_variety == 'proper':
                    #                 element_string= etree.fromstring(_Unison_States_Mol._print_xml_torsion_string(atomType1, atomType2, atomType3, atomType4, parameter, proper_string))
                    #             else:
                    #                 element_string= etree.fromstring(_Unison_States_Mol._print_xml_torsion_string(atomType1, atomType2, atomType3, atomType4, parameter, improper_string))                                        
                    #             self._add_to_output(element_string, "/ForceField/PeriodicTorsionForce")

                    # else:
                    #     print('Found torsion between dummy atom types ...')
                        
                    #     element_string= etree.fromstring(proper_string.format(atomType1=atomType1, atomType2=atomType2, atomType3=atomType3, atomType4=atomType4, periodicity='0', phase='0', k='0'))
                    #     self._add_to_output(element_string, "/ForceField/PeriodicTorsionForce")
                            
        return 1


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

    
    def _generate_unison_clone(self, mols:list):
        """
        Generates the Unison_Mol object.

        Parameters
        ----------
        mols : list
            List of rdMol objects
        
        """

        # constructs all atom type changes in the atoms relative to a ref
        # comparing everything to a reference state 
        ref = mols[0]     
        # generate a unison_mol on which all the work is done
        unison_mol = deepcopy(ref)
        # set global mol property on unison_mol
        nr_of_states = len(mols)
        unison_mol.SetProp('nr_of_states', str(nr_of_states))
        # set reference atom_type in unison_mol
        for unison_atom in unison_mol.GetAtoms():
            unison_atom_type = unison_atom.GetProp('type')
            atom_parameters = self._retrieve_parameters(atom_type1=unison_atom_type)
            unison_atom_charge = unison_atom.GetProp('charge')
            unison_atom.SetProp('atom_type_at_state_0', str(unison_atom_type))
            unison_atom.SetProp('charge_at_state_0', str(unison_atom_charge))
            unison_sigma = atom_parameters['nonbonds'].attrib['sigma']
            unison_epsilon = atom_parameters['nonbonds'].attrib['epsilon']
            unison_atom.SetProp('sigma_at_state_0', str(unison_sigma))
            unison_atom.SetProp('epsilon_at_state_0', str(unison_epsilon))

        # set reference bond parameters in unison_mol
        for unison_bond in unison_mol.GetBonds():
            a1 = unison_bond.GetBeginAtom()
            a2 = unison_bond.GetEndAtom()
            unison_a1_type = a1.GetProp('type')
            unison_a2_type = a2.GetProp('type')
            bond_parameter = self._retrieve_parameters(atom_type1=unison_a1_type, atom_type2=unison_a2_type)
            unison_length = bond_parameter['bonds'].attrib['length']
            unison_k = bond_parameter['bonds'].attrib['k']
            unison_bond.SetProp('bond_length_at_state_0', str(unison_length))
            unison_bond.SetProp('k_at_state_0', str(unison_k))

        unison_mol = Chem.RWMol(unison_mol)

        # iterate over all molecules except reference mol
        # and set atom parameters
        state = 0
        for query in mols[1:]:
            state += 1
            # starting by finding all atoms that are the same in ref and query mol
            # and set atom_type_at_state_${state} property
            for unison_atom in unison_mol.GetAtoms():
                unison_atom_name = unison_atom.GetProp('name')
                unison_atom_type = unison_atom.GetProp('type')
                
                for query_atom in query.GetAtoms():
                    query_atom_name = query_atom.GetProp('name')
                    query_atom_type = query_atom.GetProp('type')

                    # adding parameters for all atoms                 
                    if (query_atom_name == unison_atom_name):
                        parmeter = self._retrieve_parameters(atom_type1=query_atom_type)                       
                        query_atom_charge = query_atom.GetProp('charge')
                        # set properties for unison atom in next state (i.e. query state)
                        unison_atom.SetProp('atom_type_at_state_'+str(state), str(query_atom_type))
                        unison_atom.SetProp('charge_at_state_'+str(state), str(query_atom_charge))
                        query_sigma = parmeter['nonbonds'].attrib['sigma']
                        query_epsilon = parmeter['nonbonds'].attrib['epsilon']
                        unison_atom.SetProp('sigma_at_state_'+str(state), str(query_sigma))
                        unison_atom.SetProp('epsilon_at_state_'+str(state), str(query_epsilon))               
            
            # set also all bonded terms for all atoms that have the same atom name
            for unison_bond in unison_mol.GetBonds():
                unison_a1 = unison_bond.GetBeginAtom()
                unison_a2 = unison_bond.GetEndAtom()
                unison_a1_name = unison_a1.GetProp('name')
                unison_a2_name = unison_a2.GetProp('name')

                for query_bond in query.GetBonds():
                    query_a1 = query_bond.GetBeginAtom()
                    query_a2 = query_bond.GetEndAtom()
                    query_a1_name = query_a1.GetProp('name')
                    query_a2_name = query_a2.GetProp('name')
                    query_a1_type = query_a1.GetProp('type')
                    query_a2_type = query_a2.GetProp('type')

                    # if atoms are the same on bond
                    if((unison_a1_name == query_a1_name and unison_a2_name == query_a2_name) or 
                        (unison_a2_name == query_a1_name and unison_a1_name == query_a2_name)):
                        bond_param = self._retrieve_parameters(atom_type1=query_a1_type, atom_type2=query_a2_type)
                        unison_length = bond_param['bonds'].attrib['length']
                        unison_k = bond_param['bonds'].attrib['k']
                        unison_bond.SetProp('bond_length_at_state_'+str(state), str(unison_length))
                        unison_bond.SetProp('k_at_state_'+str(state), str(unison_k))

            # if an atom of the unison molecule does not have a corresponding atom type 
            # in this state, then this atom will become a dummy atom in this particular state
            for unison_atom in unison_mol.GetAtoms():
                if not unison_atom.HasProp('atom_type_at_state_'+str(state)):
                    # this if clause is fullfilled if an atom of the reference molecule 
                    # will become a dummy at this state 
                    dummy_atom_type = 'd_'+unison_atom.GetProp('name') + '_' + unison_atom.GetProp('atom_type_at_state_0')
                    print('Hydrogen present in reference state that will become dummy atom: ', str(unison_atom.GetProp('name')))
                    unison_atom.SetProp('atom_type_at_state_'+str(state), dummy_atom_type)
                    unison_atom.SetProp('charge_at_state_'+str(state), str(0))
                    unison_atom.SetProp('sigma_at_state_'+str(state), str(0))
                    unison_atom.SetProp('epsilon_at_state_'+str(state), str(0))
                    
                    bonded_heavy_atom = self._return_bonded_heavy_atom(unison_atom, unison_mol)
                    unison_bond = unison_mol.GetBondBetweenAtoms(unison_atom.GetIdx(), bonded_heavy_atom.GetIdx())
                    query_a1 = unison_bond.GetBeginAtom()
                    query_a2 = unison_bond.GetEndAtom()
                    atomType_a1 = _Unison_States_Mol._get_real_atom_type(query_a1, nr_of_states )
                    atomType_a2 = _Unison_States_Mol._get_real_atom_type(query_a2, nr_of_states )
                    parameter = self._retrieve_parameters(atom_type1=atomType_a1, atom_type2=atomType_a2)

                    unison_bond.SetProp('bond_length_at_state_'+str(state), parameter['bonds'].attrib['length'])
                    unison_bond.SetProp('k_at_state_'+str(state), parameter['bonds'].attrib['k'])


        # add dummy atom to unison mol, add dummy bonds and set dummy/atom parameters
        # bonds are retrieved and stored in newly_added_bonds 
        unison_atom_names = [atom.GetProp('name') for atom in unison_mol.GetAtoms()]       
        state_index = 0
        newly_added_bonds = []

        for query in mols[1:]:
            state_index += 1
            for query_atom in query.GetAtoms():
                # find dummy atom in query mol
                if query_atom.GetProp('name') not in unison_atom_names:
                    # get atom that binds to atom in query mol
                    query_bonded_atom = self._return_bonded_heavy_atom(query_atom, query)
                    # add query atom to the unison mol
                    unison_atom = Chem.Atom('H')
                    unison_atom.SetProp('name', str(query_atom.GetProp('name')))
                    print('Dummy atom that will be added to reference state: ', str(unison_atom.GetProp('name')))
                    # set atom type on newly added unison atom - it has to have the same parameters as 
                    # the query atom at this particular satete
                    parmeter = self._retrieve_parameters(atom_type1=query_atom.GetProp('type'))
                    unison_sigma = parmeter['nonbonds'].attrib['sigma']
                    unison_epsilon = parmeter['nonbonds'].attrib['epsilon']

                    unison_atom.SetProp('atom_type_at_state_' + str(state_index), query_atom.GetProp('type'))
                    unison_atom.SetProp('charge_at_state_' + str(state_index), query_atom.GetProp('charge'))
                    unison_atom.SetProp('sigma_at_state_'+str(state_index), str(unison_sigma))
                    unison_atom.SetProp('epsilon_at_state_'+str(state_index), str(unison_epsilon))

                    # set dummy property in unison_mol on dummy atom
                    for state in range(len(mols)):
                        if not unison_atom.HasProp('atom_type_at_state_' + str(state)):
                            dummy_atom_type = 'd_'+unison_atom.GetProp('name') + '_' + unison_atom.GetProp('atom_type_at_state_' + str(state_index))
                            unison_atom.SetProp('atom_type_at_state_' + str(state), dummy_atom_type)
                            unison_atom.SetProp('charge_at_state_' + str(state), str(0))
                            unison_atom.SetProp('sigma_at_state_'+str(state), str(0))
                            unison_atom.SetProp('epsilon_at_state_'+str(state), str(0))

                    idx = unison_mol.AddAtom(unison_atom)

                    for atom in unison_mol.GetAtoms():
                        if query_bonded_atom.GetProp('name') == atom.GetProp('name'):
                            # create dummy-heavy atom bond
                            unison_mol.AddBond(atom.GetIdx(),idx)
                            newly_added_bonds.append(unison_mol.GetBondBetweenAtoms(atom.GetIdx(), idx))
                            break
                    

        # creat hydrogen dummy atoms for each hydrogen that changes bond length
        for atom in unison_mol.GetAtoms():
            print(atom.GetProp('name'))
            # CASE 1: hydrogens change atom type
            if atom.GetSymbol() == 'H' and self._check_if_hydrogen_changes_atom_type(atom, len(mols)):
                # get all hydrogen atom types
                hydrogen_atom_types = self._return_all_atom_types_in_different_states(atom, 2)

                # get bonded heavy atom
                bond = atom.GetBonds()[0]
                bonded_heavy_atom = bond.GetOtherAtom(atom)
                b1_idx = bonded_heavy_atom.GetIdx()

                newly_added_atoms = []
                for unique_atom_type in set(hydrogen_atom_types[1:]):
                    # for each unique hydrogen atom type (except the original) a new hydrogen atom will be generated
                    # that is bound to the heavy atom
                    new_atom = Chem.Atom('H')

                    b2_idx = unison_mol.AddAtom(new_atom)
                    unison_mol.AddBond(b1_idx, b2_idx)
                    unison_bond = unison_mol.GetBondBetweenAtoms(b1_idx, b2_idx)
                    newly_added_bonds.append(unison_bond)

                    unison_atom = unison_mol.GetAtomWithIdx(b2_idx)
                    # the name of the new hydrogen atom is D+the old name+the unique atom type, e.g. (DH5_ha)
                    unison_atom.SetProp('name', 'D'+str(atom.GetProp('name')) +'_' + str(unique_atom_type))
                    newly_added_atoms.append(unison_atom)

                for newly_added_atom in newly_added_atoms:
                    # for every new atom 
                    for state, unique_atom_type in enumerate(hydrogen_atom_types):
                        # find the real atom type it represents
                        dummy_type = newly_added_atom.GetProp('name').split('_')[-1]
                        if dummy_type != unique_atom_type:
                            # at this state it is a dummy atom
                            dummy_atom_type = 'd_D'+atom.GetProp('name') + '_' + dummy_type
                            newly_added_atom.SetProp('atom_type_at_state_' + str(state), dummy_atom_type)
                            newly_added_atom.SetProp('charge_at_state_' + str(state), '0')
                            newly_added_atom.SetProp('sigma_at_state_'+str(state), '0')
                            newly_added_atom.SetProp('epsilon_at_state_'+str(state), '0')
                        else:
                            # this is the real state
                            parmeter = self._retrieve_parameters(atom_type1=dummy_type)
                            unison_sigma = parmeter['nonbonds'].attrib['sigma']
                            unison_epsilon = parmeter['nonbonds'].attrib['epsilon']
                            newly_added_atom.SetProp('atom_type_at_state_' + str(state), dummy_type)
                            newly_added_atom.SetProp('sigma_at_state_'+str(state), unison_sigma)
                            newly_added_atom.SetProp('epsilon_at_state_'+str(state), unison_epsilon)
                            newly_added_atom.SetProp('charge_at_state_' + str(state), atom.GetProp('charge_at_state_' + str(state)))
                
                # this atom will become a dummy at all other states
                dummy_atom_type = 'd_'+atom.GetProp('name') + '_' + atom.GetProp('atom_type_at_state_' + str(0))
                for state in range(1, len(mols)):
                    atom.SetProp('atom_type_at_state_' + str(state), dummy_atom_type)
                    atom.SetProp('charge_at_state_' + str(state), '0')
                    atom.SetProp('sigma_at_state_'+str(state), '0')
                    atom.SetProp('epsilon_at_state_'+str(state), '0')
                    bonded_atom = self._return_bonded_heavy_atom(atom, unison_mol)
                    bond = unison_mol.GetBondBetweenAtoms(bonded_atom.GetIdx(), atom.GetIdx())
                    parameter = self._retrieve_parameters(atom_type1=bonded_atom.GetProp('atom_type_at_state_0'), atom_type2=atom.GetProp('atom_type_at_state_0'))

                    bond.SetProp('bond_length_at_state_'+str(state), parameter['bonds'].attrib['length'])
                    bond.SetProp('k_at_state_'+str(state), parameter['bonds'].attrib['k'])


            # CASE 2: heavy atom changes atom type
            elif atom.GetSymbol() == 'H' and self._check_if_bond_heavy_atom_changes_atom_type(atom, unison_mol, len(mols)):               

                # generate for each atom type a dummy atom (except for the first, since this is the already present hydrogen atom)
                for real_state, atom_type in enumerate(self._return_all_bonded_heavy_atom_types(atom, unison_mol, len(mols))[1:]):

                    new_dummy_name = 'D' + atom.GetProp('name') + '_' + atom_type
                    dummy_atom = Chem.Atom('H')
                    dummy_atom.SetProp('name', new_dummy_name)
                    bonded_idx = self._return_bonded_heavy_atom(atom, unison_mol).GetIdx()
                    dummy_atom_type = 'd_' + atom.GetProp('name') + '_' + atom_type

                    # real atom type does not change
                    dummy_atom.SetProp('atom_type_at_state_' + str(real_state+1), atom.GetProp('atom_type_at_state_0'))
                    dummy_atom.SetProp('charge_at_state_' + str(real_state+1), atom.GetProp('charge_at_state_' + str(real_state + 1)))
                    dummy_atom.SetProp('sigma_at_state_'+str(real_state+1), atom.GetProp('sigma_at_state_' + str(real_state + 1)))
                    dummy_atom.SetProp('epsilon_at_state_'+str(real_state+1), atom.GetProp('epsilon_at_state_' + str(real_state + 1)))

                    # dummy type for all others
                    dummy_atom_type = 'd_D' + atom.GetProp('name') + '_' + atom_type
                    for state in range(len(mols)):
                        if not dummy_atom.HasProp('atom_type_at_state_' + str(state)):
                            dummy_atom.SetProp('atom_type_at_state_' + str(state), dummy_atom_type)
                            dummy_atom.SetProp('charge_at_state_' + str(state), str(0))
                            dummy_atom.SetProp('sigma_at_state_'+str(state), str(0))
                            dummy_atom.SetProp('epsilon_at_state_'+str(state), str(0))

                    idx = unison_mol.AddAtom(dummy_atom)
                    bond_idx = unison_mol.AddBond(idx, bonded_idx)                    
                    newly_added_bonds.append(unison_mol.GetBondWithIdx(bond_idx-1))

                # for the real atom correct atom and bond parameters
                dummy_atom_type = 'd_' + atom.GetProp('name') + '_' + atom_type

                for state in range(1, nr_of_states):
                    atom.SetProp('atom_type_at_state_' + str(state), dummy_atom_type)
                    atom.SetProp('charge_at_state_' + str(state), '0')
                    atom.SetProp('sigma_at_state_'+str(state), '0')
                    atom.SetProp('epsilon_at_state_'+str(state), '0')
                    bonded_atom = self._return_bonded_heavy_atom(atom, unison_mol)
                    bond = unison_mol.GetBondBetweenAtoms(bonded_atom.GetIdx(), atom.GetIdx())
                    parameter = self._retrieve_parameters(atom_type1=bonded_atom.GetProp('atom_type_at_state_0'), atom_type2=atom.GetProp('atom_type_at_state_0'))

                    bond.SetProp('bond_length_at_state_'+str(state), parameter['bonds'].attrib['length'])
                    bond.SetProp('k_at_state_'+str(state), parameter['bonds'].attrib['k'])

      
        # set the parameters for all new bonds
        for bond in newly_added_bonds:
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            print('Newly added bond ... between: ', a1.GetProp('name'), a2.GetProp('name'))
            for state in range(len(mols)):
                atomType_a1 = a1.GetProp('atom_type_at_state_' + str(state))
                atomType_a2 = a2.GetProp('atom_type_at_state_' + str(state))
                if atomType_a1.startswith('d_') or atomType_a2.startswith('d_'):
                    continue
                else:
                    parameter = self._retrieve_parameters(atom_type1=atomType_a1, atom_type2=atomType_a2)
            for state in range(len(mols)):
                bond.SetProp('bond_length_at_state_'+str(state), parameter['bonds'].attrib['length'])
                bond.SetProp('k_at_state_'+str(state), parameter['bonds'].attrib['k'])


        # write out the unison mol as pdb to check sanity
        Chem.MolToPDBFile(unison_mol, 'unison_mol.pdb')
        unison_mol = _Unison_States_Mol(unison_mol)
        unison_mol._print_state_of_unison_mol()

        self._unison_clone = unison_mol

 
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

        # generate unison clone of all mols provided and 
        # save it in self._unison_clone
        self._generate_unison_clone(mol_array)

        for bond in self._unison_clone.mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            print(a1.GetProp('name') ,' - ', a2.GetProp('name'))
            
            

        charges = list()
        for index, state in enumerate(self._input_state_data):
            net_charge = state['net_charge']
            charges.append(int(net_charge))
            template = _State(index,
                              state['log_population'],
                              0.0, # set g_k defaults to 0 for now
                              self._unison_clone.generate_atom_name_list_for_state(index),
                              net_charge,
                              state['pH']
                              )

            self._state_templates.append(template)

        min_charge = min(charges)
        for state in self._state_templates:
            state.set_number_of_protons(min_charge)
        return

    
    def _return_all_atom_types_in_different_states(self, atom, nr_of_states):

        hydrogen_type = []
        for state in range(nr_of_states):
            hydrogen_type.append(atom.GetProp('atom_type_at_state_' + str(state)))

        return hydrogen_type

    def _return_all_bonded_heavy_atom_types(self, atom, mol, nr_of_states):
        
        heavy_atom_types = []
        heavy_atom_type_changes_flag = False
        for bond in mol.GetBonds():
            if atom.GetProp('name') == bond.GetBeginAtom().GetProp('name') or atom.GetProp('name') == bond.GetEndAtom().GetProp('name'):
                heavy_atom = bond.GetOtherAtom(atom)

        for state in range(nr_of_states):
            heavy_atom_types.append(heavy_atom.GetProp('atom_type_at_state_' + str(state)))
       
        return heavy_atom_types

    def _return_bonded_heavy_atom(self, atom, mol):
        
        for bond in mol.GetBonds():
            if atom.GetProp('name') == bond.GetBeginAtom().GetProp('name') or atom.GetProp('name') == bond.GetEndAtom().GetProp('name'):
                heavy_atom = bond.GetOtherAtom(atom)
                return heavy_atom
        
        print('WARNING! COULD NOT RETURN HEAVY ATOM!')


    def _check_if_bond_heavy_atom_changes_atom_type(self, atom, mol, nr_of_states):
        
        heavy_atom_types = []
        heavy_atom_type_changes_flag = False
        for bond in mol.GetBonds():
            if atom.GetProp('name') == bond.GetBeginAtom().GetProp('name') or atom.GetProp('name') == bond.GetEndAtom().GetProp('name'):
                heavy_atom = bond.GetOtherAtom(atom)

        for state in range(nr_of_states):
            heavy_atom_types.append(heavy_atom.GetProp('atom_type_at_state_' + str(state)))
            if atom.GetProp('atom_type_at_state_' + str(state)).startswith('d_'):
                return False

        if len(set(heavy_atom_types)) != 1:
            heavy_atom_type_changes_flag = True
        
        return heavy_atom_type_changes_flag
         

    def _check_if_hydrogen_changes_atom_type(self, atom, nr_of_states):

        # this method checks if the hydrogen atom type 
        # changes in the different states for a particular 
        # hydorgen
        # dummy atom types are excluded from this check

        hydrogen_type_changes_flag = False
        hydrogen_type = []
        for state in range(nr_of_states):
            if(atom.GetProp('atom_type_at_state_' + str(state)).startswith('d_')):
                continue
            else:
                hydrogen_type.append(atom.GetProp('atom_type_at_state_' + str(state)))


        if len(set(hydrogen_type)) != 1:
            hydrogen_type_changes_flag = True
        
        return hydrogen_type_changes_flag
                  


    def _initialize_forcefield_template(self):
        """
        Set up the residue template using the first state of the molecule
        """

        residue = self.ffxml.xpath('/ForceField/Residues/Residue')[0]

        atom_string = '<Atom name="{name}" type="{atom_type}" charge="{charge}"/>'
        bond_string = '<Bond atomName1="{atomName1}" atomName2="{atomName2}"/>'

        for atom in self._unison_clone.mol.GetAtoms():
            name = atom.GetProp('name')
            atom_type = atom.GetProp('atom_type_at_state_0')
            charge = atom.GetProp('charge_at_state_0')
            residue.append(etree.fromstring(atom_string.format(name=name, atom_type=atom_type, charge=charge)))
  
        
        for bond in self._unison_clone.mol.GetBonds():
            a1 = bond.GetBeginAtom()
            a2 = bond.GetEndAtom()
            residue.append(etree.fromstring(bond_string.format(atomName1=a1.GetProp('name'), atomName2=a2.GetProp('name'))))

    def _add_isomers(self):
        """
        Add all the isomer specific data to the xml template.
        """

        patt = Chem.MolFromSmarts('*~*~*')
        unison_mol = self._unison_clone.mol
        angle_list = unison_mol.GetSubstructMatches(patt)
        for residue in self.ffxml.xpath('/ForceField/Residues/Residue'):
            atom_string = '<Atom name="{name}" type="{atom_type}" charge="{charge}" epsilon="{epsilon}" sigma="{sigma}" />'
            bond_string = '<Bond type1="{atomType1}" type2="{atomType2}" length="{bond_length}" k="{k}"/>'
            angle_string = '<Angle type1="{atomType1}" type2="{atomType2}" type3="{atomType3}" angle="{angle}" k="{k}"/>'

            protonsdata = etree.fromstring("<Protons/>")
            nr_of_states = int(len(self._state_templates))
            protonsdata.attrib['number_of_states'] = str(len(self._state_templates))
            for isomer_index, isomer in enumerate(self._state_templates):
                isomer_str = str(isomer)
                isomer_xml = etree.fromstring(isomer_str)
                for atom in unison_mol.GetAtoms():
                    isomer_xml.append(etree.fromstring(_Unison_States_Mol._print_xml_atom_string(atom, isomer_index, atom_string)))

                for bond in unison_mol.GetBonds():
                    isomer_xml.append(etree.fromstring(_Unison_States_Mol._print_xml_bond_string(bond, isomer_index, bond_string)))

                
                # for angle in angle_list:
                #     a1 = unison_mol.GetAtomWithIdx(angle[0])
                #     a2 = unison_mol.GetAtomWithIdx(angle[1])
                #     a3 = unison_mol.GetAtomWithIdx(angle[2])
                #     atomType1 = a1.GetProp('atom_type_at_state_'+str(isomer_index))
                #     atomType2 = a2.GetProp('atom_type_at_state_'+str(isomer_index))
                #     atomType3 = a3.GetProp('atom_type_at_state_'+str(isomer_index))
                #     print('Angle between: ', atomType1, atomType2, atomType3)
                #     parameter = self._retrieve_parameters(atom_type1=atomType1, atom_type2=atomType2, atom_type3=atomType3)
                #     isomer_xml.append(etree.fromstring(_Unison_States_Mol._print_xml_angle_string(atomType1, atomType2, atomType3, parameter, angle_string)))



                protonsdata.append(isomer_xml)
            residue.append(protonsdata)

    def _append_extra_gaff_types(self):
        """
        Add additional parameters generated by antechamber/parmchk for the individual isomers
        """
        added_parameters = list()  # for bookkeeping of duplicates

        # All xml sources except the entire gaff.xml
        for xmltree in self._xml_parameter_trees[1:]:
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

            # Match proper dihedral of the atom in PeriodicTorsionForce block
            for proper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Proper"):
                items = set(proper.items())
                proper_element = tuple(["PeriodicTorsionForce", "Proper", items])
                # Make sure the force wasn't already added by a previous state
                if proper_element not in added_parameters:
                    added_parameters.append(proper_element)
                    self._add_to_output(proper, "/ForceField/PeriodicTorsionForce")

            # Match improper dihedral of the atom in PeriodicTorsionForce block
            for improper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Improper"):
                items = set(improper.items())
                improper_element = tuple(["PeriodicTorsionForce", "Improper", items])
                # Make sure the force wasn't already added by a previous state
                if improper_element not in added_parameters:
                    added_parameters.append(improper_element)
                    self._add_to_output(improper, "/ForceField/PeriodicTorsionForce")

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
                    angle_atom_types_list = sorted([angle.attrib['type1'], angle.attrib['type2'], angle.attrib['type3']])
                    search_list = sorted([kwargs['atom_type1'], kwargs['atom_type2'], kwargs['atom_type3']])
                    if search_list[0] in angle_atom_types_list[0] and search_list[1] in angle_atom_types_list[1] and search_list[2] in angle_atom_types_list[2]:
                        params['angle'] = angle
            return params
            
        
        elif len(kwargs) == 4:
            # match torsion parameters
            params['proper'] = []
            params['improper'] = []
            found_torsion = False 
            for xmltree in self._xml_parameter_trees:
                # Match proper dihedral of the atom in PeriodicTorsionForce block
                for proper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Proper"):
                    
                    # create matching list of torsion atom types
                    torsion_types_list = [proper.attrib['type1'], proper.attrib['type2'], proper.attrib['type3'], proper.attrib['type4']]
                    search_list = [kwargs['atom_type1'], kwargs['atom_type2'], kwargs['atom_type3'], kwargs['atom_type4']]
                    # Start with matching the two central atoms of the torsion - this could now apply to either wildcard torsion or specific torsions
                    if search_list[1] == torsion_types_list[1] and search_list[2] == torsion_types_list[2] :

                        if torsion_types_list[0] == '' and torsion_types_list[3] == '':
                            params['proper'].append(proper)
                            found_torsion = True

                        elif search_list[0] == torsion_types_list[0] and search_list[3] == torsion_types_list[3]:
                            # foudn a specific torsion!
                            #  
                            params['proper'].append(proper)
                            found_torsion = True
                        else:
                            continue
                            
                    elif search_list[2] == torsion_types_list[1] and search_list[1] == torsion_types_list[2] :

                        if torsion_types_list[0] == '' and torsion_types_list[3] == '':
                            params['proper'].append(proper)
                            found_torsion = True

                        elif search_list[3] == torsion_types_list[0] and search_list[0] == torsion_types_list[3]:
                            # foudn a specific torsion!
                            #  
                            params['proper'].append(proper)
                            found_torsion = True
                        else:
                            continue
                         
                
                # Match improper dihedral of the atom in PeriodicTorsionForce block
                for improper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Improper"):
                    # create matching list of torsion atom types
                    improp_types_list = [improper.attrib['type1'], improper.attrib['type2'], improper.attrib['type3'], improper.attrib['type4']]
                    search_list = [kwargs['atom_type1'], kwargs['atom_type2'], kwargs['atom_type3'], kwargs['atom_type4']]
                    # Start with matching the two central atoms of the torsion - this could now apply to either wildcard torsion or specific torsions
                    if search_list[1] ==  improp_types_list[1] and search_list[2] ==  improp_types_list[2]:

                        if improp_types_list[0] == '' and improp_types_list[3] == '':
                            # found an unspecific improp! will use it!
                            params['improper'].append(proper)
                        elif search_list[0] == improp_types_list[0] and search_list[3] == improp_types_list[3]:
                            # foudn a specific torsion!
                            params['improper'].append(proper)
                        else:
                            continue

                    elif search_list[2] ==  improp_types_list[1] and search_list[1] ==  improp_types_list[2]:

                        if improp_types_list[0] == '' and improp_types_list[3] == '':
                            # found an unspecific improp! will use it!
                            params['improper'].append(proper)
                        elif search_list[0] == improp_types_list[0] and search_list[3] == improp_types_list[3]:
                            # foudn a specific torsion!
                            params['improper'].append(proper)
                        else:
                            continue


            if found_torsion == False:
                print('Could not find torsion parameter for dummy torsion! Trouble ahead!')
                print('Looking for: ', search_list)

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
    if delete_old_H:
        to_delete = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H']]
        modeller.delete(to_delete)

    modeller.addHydrogens(forcefield=forcefield)
    #NOTE: MW we should add the option to use either charmm-gui output (fully solvated system, either using pdb or 
    # the ideal case would actually be that it generates the system using psf and pdb/crd files 
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
    return simulation
