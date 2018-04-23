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
from collections import defaultdict
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

    def validate(self):
        """
        Checks to see if the isomeric state is valid.

        Raises
        ------
        ValueError
            If any atom has not been instantiated, or is instantiated wrongly.

        """
        issues = "The following issues need to be resolved:\r\n"

        for atom in self.atoms.values():

            if atom is None:
                issues += "Atom '{}' has not been instantiated.\r\n".format(atom.name)
            elif not isinstance(atom, _Atom):
                issues += "Invalid atom found '{}'.\r\n".format(atom.name)
            elif atom.is_dummy():
                issues += "Atom is a dummy, please assign proper types."
            elif hasattr(atom, 'half_life'):
                issues += "Atom '{}' is radioactive.\r\n".format(atom.name)

        if self.proton_count < 0:
            issues += "Invalid number of acidic protons: {}.".format(self.proton_count)

        raise ValueError(issues)

    def get_dummies(self):
        """
        Return the list of atoms that currently are None
        """

        dummies = list()

        for name, atom in self.atoms.items():
            if atom is None:
                dummies.append(name)
            elif atom.is_dummy():
                dummies.append(name)

        return dummies

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

    # NOTE: MW: entry point
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
        # NOTE: MW: Here I want to write out mols to 

        # write open-eye mol2 file
        fileIO = str(base) + '_tmp_'+ str(isomer_index) + '.mol2'
        ofs = oechem.oemolostream()
        ofs.open(fileIO)
        oechem.OEWriteMol2File(ofs, oemolecule)
        ofs.close()
        # read in using rdkit
        rdmol = Chem.MolFromMol2File(fileIO, removeHs=False)

        # set atom-names and types for atoms in rdkit mol
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

def generate_rdkit_mol_from_oemol_and_ff(ffxml:str, oemolecule):


    pass



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
    hydrogen_types = _find_hydrogen_types(gafftree)

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


def _find_hydrogen_types(gafftree: lxml.etree.ElementTree) -> set:
    """
    Find all atom types that describe hydrogen atoms.

    Parameters
    ----------
    gafftree - A GAFF input xml file that contains atom type definitions.

    Returns
    -------
    set - names of all atom types that correspond to hydrogen
    """

    # Detect all hydrogen types by element and store them in a set
    hydrogen_types = set()
    for atomtype in gafftree.xpath('AtomTypes/Type'):
        if atomtype.get('element') == "H":
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

class _State_mol(object):

    def __init__(self, mol):
        self.mol = mol
    
    @classmethod
    def _print_xml_atom_string(self, atom, state):
        string = '<Atom name="{name}" type="{atom_type}" charge="{charge} sigma={sigma} epsilon={epsilon}"/>'
        name = atom.GetProp('name')
        atom_type = atom.GetProp('atom_type_at_state_'+str(state))
        atom_charge = atom.GetProp('charge_at_state_'+str(state))
        sigma = atom.GetProp('sigma_at_state_' + str(state))
        epsilon = atom.GetProp('epsilon_at_state_' + str(state))
        return string.format(name=name, atom_type=atom_type, charge=atom_charge, sigma=sigma, epsilon=epsilon)
        
    @classmethod
    def _print_xml_bond_string(self, bond, state):
        string = '<Bond atomName1="{atomName1}" atomName2="{atomName2}" bond_length="{bond_length}" k="{k}" />'
        atomName1 = bond.GetBeginAtom().GetProp('name')
        atomName2 = bond.GetEndAtom().GetProp('name')
        bond_length = bond.GetProp('bond_length_at_state_' + str(state))
        k = bond.GetProp('k_at_state_' + str(state))
        return string.format(atomName1=atomName1, atomName2=atomName2, bond_length=bond_length, k=k)
        

    def _print_state_of_shadow_mol(self):
        mol = self.mol
        nr_of_state = int(self.mol.GetProp('nr_of_states'))
        print('How many states included in shadow clone: ', nr_of_state)

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

    def generate_atom_type_list_for_state(self):

            for atom in self.mol.GetAtoms:
                pass

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
        self._atoms_for_each_state = None
        self._bonds_for_each_state = None
        self._mol_for_each_state = defaultdict()
        self._shadow_clone = None


        
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
        # Remove empty blocks, and unnecessary information in the ffxml tree
        self._sanitize_ffxml()
        return


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

    def _generate_shadow_clone(self, mols:list):
        # constructs all atom type changes in the atoms relative to a ref
        # comparing everything to a reference state 
        ref = mols[0]     
        state = 0
        # generate a shadow_mol on which all the work is done
        shadow_mol = deepcopy(ref)
        shadow_mol.SetProp('nr_of_states', str(len(mols)))
        # set reference atom_type in shadow_mol
        for shadow_atom in shadow_mol.GetAtoms():
            shadow_atom_type = shadow_atom.GetProp('type')
            shadow_atom_charge = shadow_atom.GetProp('charge')
            shadow_atom.SetProp('atom_type_at_state_0', str(shadow_atom_type))
            shadow_atom.SetProp('charge_at_state_0', str(shadow_atom_charge))
            a = self._retrieve_parameters(atom_type1=shadow_atom_type)
            shadow_sigma = a['nonbonds'].attrib['sigma']
            shadow_epsilon = a['nonbonds'].attrib['epsilon']
            shadow_atom.SetProp('sigma_at_state_0', str(shadow_sigma))
            shadow_atom.SetProp('epsilon_at_state_0', str(shadow_epsilon))


        # set reference bond parameters in shadow_mol
        for shadow_bond in shadow_mol.GetBonds():
            a1 = shadow_bond.GetBeginAtom()
            a2 = shadow_bond.GetEndAtom()
            shadow_a1_type = a1.GetProp('type')
            shadow_a2_type = a2.GetProp('type')
            a = self._retrieve_parameters(atom_type1=shadow_a1_type, atom_type2=shadow_a2_type)
            shadow_length = a['bonds'].attrib['length']
            shadow_k = a['bonds'].attrib['k']
            shadow_bond.SetProp('bond_length_at_state_0', str(shadow_length))
            shadow_bond.SetProp('k_at_state_0', str(shadow_k))

        # iterate over all molecules except reference mol
        for query in mols[1:]:
            state += 1
            # starting by finding all atoms that are the same in ref and query mol
            # and set atom_type_at_state_${state} propertie
            for shadow_atom in shadow_mol.GetAtoms():
                shadow_atom_name = shadow_atom.GetProp('name')
                
                for query_atom in query.GetAtoms():
                    query_atom_name = query_atom.GetProp('name')
                    query_atom_type = query_atom.GetProp('type')
                    query_atom_type = query_atom.GetProp('type')

                    if (query_atom_name == shadow_atom_name):
                        query_atom_charge = shadow_atom.GetProp('charge')
                        # set properties for shadow atom in next state (i.e. query state)
                        shadow_atom.SetProp('atom_type_at_state_'+str(state), str(query_atom_type))
                        shadow_atom.SetProp('charge_at_state_'+str(state), str(query_atom_charge))
                        parmeter = self._retrieve_parameters(atom_type1=shadow_atom_type)
                        shadow_sigma = parmeter['nonbonds'].attrib['sigma']
                        shadow_epsilon = parmeter['nonbonds'].attrib['epsilon']
                        shadow_atom.SetProp('sigma_at_state_'+str(state), str(shadow_sigma))
                        shadow_atom.SetProp('epsilon_at_state_'+str(state), str(shadow_epsilon))
            
            # set also all bonded terms for all atoms that have the same atom name
            for shadow_bond in shadow_mol.GetBonds():
                shadow_a1 = shadow_bond.GetBeginAtom()
                shadow_a2 = shadow_bond.GetEndAtom()
                shadow_a1_name = shadow_a1.GetProp('name')
                shadow_a2_name = shadow_a2.GetProp('name')

                
                for query_bond in query.GetBonds():
                    query_a1 = query_bond.GetBeginAtom()
                    query_a2 = query_bond.GetEndAtom()
                    query_a1_name = query_a1.GetProp('name')
                    query_a2_name = query_a2.GetProp('name')
                    query_a1_type = query_a1.GetProp('type')
                    query_a2_type = query_a2.GetProp('type')

                    if((shadow_a1_name == query_a1_name and shadow_a2_name == query_a2_name) or (shadow_a2_name == query_a1_name and shadow_a1_name == query_a2_name)):
                        bond_param = self._retrieve_parameters(atom_type1=query_a1_type, atom_type2=query_a2_type)
                        shadow_length = bond_param['bonds'].attrib['length']
                        shadow_k = bond_param['bonds'].attrib['k']
                        shadow_bond.SetProp('bond_length_at_state_'+str(state), str(shadow_length))
                        shadow_bond.SetProp('k_at_state_'+str(state), str(shadow_k))


            # if an atom of the shadow molecule does not have a corresponding atom type 
            # in this state, then this atom will become a dummy atom in this particular state
            for shadow_atom in shadow_mol.GetAtoms():
                if not shadow_atom.HasProp('atom_type_at_state_'+str(state)):
                    shadow_atom.SetProp('atom_type_at_state_'+str(state), 'dummy')
                    shadow_atom.SetProp('charge_at_state_'+str(state), str(0))
                    shadow_atom.SetProp('sigma_at_state_'+str(state), str(0))
                    shadow_atom.SetProp('epsilon_at_state_'+str(state), str(0))

                    idx = shadow_atom.GetIdx()

                    for shadow_bond in shadow_mol.GetBonds():
                        if shadow_bond.GetBeginAtomIdx() == idx or shadow_bond.GetEndAtomIdx() == idx:
                            shadow_bond.SetProp('bond_length_at_state_'+str(state), 'None')
                            shadow_bond.SetProp('k_at_state_'+str(state), 'None')
                                     
        shadow_mol = Chem.RWMol(shadow_mol)
        shadow_atom_names = [atom.GetProp('name') for atom in shadow_mol.GetAtoms()]
        
        state_index = 0
        for query in mols[1:]:
            state_index += 1
            # since I don't want to manually interfer the correct 
            # 3D coorindates I will use alignemnt to the reference 
            # atom to get proximatly correct coordinates for the new
            # atom
            # NOTE: This might not be the best strategy - since 
            # I know that only one atom is added I could also 
            # generate internal coordinates for this atom
            # and subsequently calculate cartesian coord
            pyO3A = rdMolAlign.GetO3A(shadow_mol, query)
            pyO3A.Align()

            for query_atom in query.GetAtoms():
                # find dummy atom in query mol
                if query_atom.GetProp('name') not in shadow_atom_names:
                    # add query atom that is shadow mol at reference state
                    idx = shadow_mol.AddAtom(query_atom)
                    shadow_atom = shadow_mol.GetAtomWithIdx(idx)
                    # set atom type on newly added shadow atom to query atom
                    shadow_atom.SetProp('atom_type_at_state_' + str(state_index), query_atom.GetProp('type'))
                    shadow_atom.SetProp('charge_at_state_' + str(state_index), query_atom.GetProp('charge'))
                    parmeter = self._retrieve_parameters(atom_type1=query_atom.GetProp('type'))
                    shadow_sigma = parmeter['nonbonds'].attrib['sigma']
                    shadow_epsilon = parmeter['nonbonds'].attrib['epsilon']
                    shadow_atom.SetProp('sigma_at_state_'+str(state_index), str(shadow_sigma))
                    shadow_atom.SetProp('epsilon_at_state_'+str(state_index), str(shadow_epsilon))

                    involved_bond = list(query_atom.GetBonds())[0]
                    other_atom_bound_to_dummy = involved_bond.GetOtherAtom(query_atom)
                    new_bond_idx = -1
                    for shadow_atom in shadow_mol.GetAtoms():
                        if shadow_atom.GetProp('name') == other_atom_bound_to_dummy.GetProp('name'):
                            new_bond_idx = shadow_atom.GetIdx()
                    
                    # set dummy property in shadow_mol on dummy atom
                    dummy_shadow_atom = shadow_mol.GetAtomWithIdx(idx)
                    print(dummy_shadow_atom)
                    for state in range(len(mols)):
                        if not dummy_shadow_atom.HasProp('atom_type_at_state_' + str(state)):
                            dummy_shadow_atom.SetProp('atom_type_at_state_' + str(state), 'dummy')
                            shadow_atom.SetProp('charge_at_state_' + str(state), str(0))
                            shadow_atom.SetProp('sigma_at_state_'+str(state), str(0))
                            shadow_atom.SetProp('epsilon_at_state_'+str(state), str(0))

                    # create dummy-heavy atom bond
                    shadow_mol.AddBond(new_bond_idx,dummy_shadow_atom.GetIdx())
                    # get this bond
                    for shadow_bond in shadow_mol.GetBonds():
                        if shadow_bond.GetBeginAtomIdx() == new_bond_idx or shadow_bond.GetEndAtomIdx() == new_bond_idx:
                            for state in range(len(mols)):
                                a1_type = shadow_bond.GetBeginAtom().GetProp('atom_type_at_state_' + str(state))
                                a2_type = shadow_bond.GetEndAtom().GetProp('atom_type_at_state_' + str(state))
                                if( a1_type == 'dummy' or a2_type == 'dummy'):
                                    shadow_bond.SetProp('bond_length_at_state_' + str(state), 'None')
                                    shadow_bond.SetProp('k_at_state_'+ str(state), 'None')
                                else:
                                    bond_param = self._retrieve_parameters(atom_type1=a1_type, atom_type2=a2_type )
                                    shadow_length = bond_param['bonds'].attrib['length']
                                    shadow_k = bond_param['bonds'].attrib['k']
                                    shadow_bond.SetProp('bond_length_at_state_'+str(state), str(shadow_length))
                                    shadow_bond.SetProp('k_at_state_'+str(state), str(shadow_k))



        Chem.MolToPDBFile(shadow_mol, 'shadow_mol.pdb')
        
        shadow_mol = _State_mol(shadow_mol)
        shadow_mol._print_state_of_shadow_mol()

        self._shadow_clone = shadow_mol


    def _generate_pdb_for_state(self):
        # TODO: looks through mol and removes everything with dummy stats at 
        # this particular state
        pass
        


    
    def _return_all_atom_for_state(self, state:int, returnListOfAtomObjects=False):

        if self._shadow_clone == None:
            print('States have to be registered!')
            return None
        
        mol = self._shadow_clone
        
        for atom in mol.GetAtoms():
            pass



    def _return_all_bonds_for_state(self, state:int):
        pass
        
    def _return_all_angles_for_state(self, state:int):
        pass
        
    def _return_all_torsion_angles_for_state(self, state:int):

        pass

    def _complete_state_registry(self):
        """
        Store all the properties that are specific to each state
        """

        # mol array
        mol_array = []
        # set charge property for each mol
        for index, state in enumerate(self._input_state_data):
            mapping_atom_name_to_charge = {}
            for xml_atom in state['ffxml'].xpath('/ForceField/Residues/Residue/Atom'):
                mapping_atom_name_to_charge[xml_atom.attrib['name']] = xml_atom.attrib['charge']

            mol = state['mol']
            self._mol_for_each_state['state' + str(index)] = mol
            for atom in mol.GetAtoms():
                atom_name = atom.GetProp('name')
                atom.SetProp('charge', mapping_atom_name_to_charge[atom_name]) 
            mol_array.append(mol)

        self._generate_shadow_clone(mol_array)


        charges = list()
        for index, state in enumerate(self._input_state_data):
            net_charge = state['net_charge']
            charges.append(int(net_charge))
            template = _State(index,
                              state['log_population'],
                              0.0, # set g_k defaults to 0 for now
                              self._shadow_clone.generate_atom_name_list_for_state(index),
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

        for atom in self._shadow_clone.mol.GetAtoms():
            print(etree.fromstring(atom_string.format(name=atom.GetProp('name'), atom_type=atom.GetProp('atom_type_at_state_0'), charge=atom.GetProp('charge_at_state_0'))))
            residue.append(etree.fromstring(atom_string.format(name=atom.GetProp('name'), atom_type=atom.GetProp('atom_type_at_state_0'), charge=atom.GetProp('charge_at_state_0'))))
        for bond in self._shadow_clone.mol.GetBonds():
            print(etree.fromstring(bond_string.format(atomName1=bond.GetBeginAtom().GetProp('name'), atomName2=bond.GetEndAtom().GetProp('name'))))
            residue.append(etree.fromstring(bond_string.format(atomName1=bond.GetBeginAtom().GetProp('name'), atomName2=bond.GetEndAtom().GetProp('name'))))

    def _add_isomers(self):
        """
        Add all the isomer specific data to the xml template.
        """

        for residue in self.ffxml.xpath('/ForceField/Residues/Residue'):
            protonsdata = etree.fromstring("<Protons/>")
            protonsdata.attrib['number_of_states'] = str(len(self._state_templates))
            for isomer_index, isomer in enumerate(self._state_templates):
                isomer_str = str(isomer)
                isomer_xml = etree.fromstring(isomer_str)
                for atom in self._shadow_clone.mol.GetAtoms():
                    print(etree.fromstring(_State_mol._print_xml_atom_string(atom, isomer_index)))
                    isomer_xml.append(etree.fromstring(_State_mol._print_xml_atom_string(atom, isomer_index)))
                for bond in self._shadow_clone.mol.GetBonds():
                    print(etree.fromstring(_State_mol._print_xml_bond_string(bond, isomer_index)))
                    isomer_xml.append(etree.fromstring(_State_mol._print_xml_bond_string(bond, isomer_index)))

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



    def _atom_type_match_angle_types(atom_types, angle_parameter):
        """ Looks through the given xml element angle_parameter and matches atom_types in correct order.
        Return true if match.
        """

        #
        atom1_to_match = atom_types['atom_type1']
        atom2_to_match = atom_types['atom_type2']
        atom3_to_match = atom_types['atom_type3']

        atom1_in_parameter_set = angle_parameter.attrib['type1']
        atom2_in_parameter_set = angle_parameter.attrib['type2']
        atom3_in_parameter_set = angle_parameter.attrib['type3']


        if atom1_to_match == atom1_in_parameter_set and atom2_to_match == atom2_in_parameter_set and atom3_to_match == atom3_in_parameter_set:
            print(atom1_to_match, ' - ',atom2_to_match, ' - ',atom3_to_match, ' matches', )
            return True
        
        elif atom3_to_match == atom1_in_parameter_set and atom2_to_match == atom2_in_parameter_set and atom1_to_match == atom3_in_parameter_set:
            print(atom1_to_match, ' - ',atom2_to_match, ' - ',atom3_to_match, ' matches', )
            return True
        else:
            return False
        

    def _retrieve_parameters(self, **kwargs):
        """ Look through FFXML files and find all parameters pertaining to the supplied atom type.
        Returns
        -------
        input : atom_type1:str, atom_type2[opt]:str, atom_type3[opt]:str, atom_type4[opt]:str, 
        """
        

        # Storing all the detected parameters here
        params = {}

        # Loop through different sources of parameters
        
            # Match nonbonded type of the atom in NonbondedForce block

        if len(kwargs) == 1:
            print('Searching for atom parameters for: ', kwargs['atom_type1'])
            # Loop through different sources of parameters
            for xmltree in self._xml_parameter_trees:
                # Match the type of the atom in the AtomTypes block
                for atomtype in xmltree.xpath("/ForceField/AtomTypes/Type"):
                    if atomtype.attrib['name'] == kwargs['atom_type1']:
                        params['atomtypes'] = atomtype
                for nonbond in xmltree.xpath("/ForceField/NonbondedForce/Atom"):
                    if nonbond.attrib['type'] == kwargs['atom_type1']:
                        params['nonbonds'] = nonbond

            return params

        elif len(kwargs) == 2:
            print('Searching for bond parameters for: ', kwargs['atom_type1'], ' - ', kwargs['atom_type2'] )
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
                    if _atom_type_match_angle_types(kwargs, angle) == True:
                            params['angles'] = (angle)
        
        
        
        elif len(kwargs) == 4:
            for xmltree in self._xml_parameter_trees:
            
                # Match proper dihedral of the atom in PeriodicTorsionForce block
                for proper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Proper"):
                    if atom_type1 in (proper.attrib['type1'], proper.attrib['type2'], proper.attrib['type3'], proper.attrib['type4']) and atom_type2 in (proper.attrib['type1'], proper.attrib['type2'], proper.attrib['type3'], proper.attrib['type4']) and atom_type3 in (proper.attrib['type1'], proper.attrib['type2'], proper.attrib['type3'], proper.attrib['type4']) and atom_type4 in (proper.attrib['type1'], proper.attrib['type2'], proper.attrib['type3'], proper.attrib['type4']):
                        params['propers'].append(proper)

                # Match improper dihedral of the atom in PeriodicTorsionForce block
                for improper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Improper"):
                    if atom_type1 in (improper.attrib['type1'], improper.attrib['type2'], improper.attrib['type3'], improper.attrib['type4']) and atom_type2 in (improper.attrib['type1'], improper.attrib['type2'], improper.attrib['type3'], improper.attrib['type4']) and  atom_type3 in (improper.attrib['type1'], improper.attrib['type2'], improper.attrib['type3'], improper.attrib['type4']) and atom_type4 in (improper.attrib['type1'], improper.attrib['type2'], improper.attrib['type3'], improper.attrib['type4']):
                        params['impropers'].append(improper)






    @staticmethod
    def _bonds_including_type(atom_type, available_parameters_per_type):
        bond_params = available_parameters_per_type[atom_type]['bonds']
        list_of_bond_params = list()
        for bond_type in bond_params:
            list_of_bond_params.append(_BondType(bond_type.get('type1'), bond_type.get('type2')))
        return list_of_bond_params


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
