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


PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))


gaff_default = os.path.join(PACKAGE_ROOT, 'data', 'gaff.xml')


class _Atom(object):
    """
    Private class representing GAFF parameters for a single atom
    """
    def __init__(self, name, atom_type, charge):
        """
        Parameters
        ----------
        name : str
            Name of the atom
        atom_type : str
            Gaff LJ type of the atom, or "dummy"
        charge : str
            point charge of the atom

        """
        self.name = name
        self.atom_type = atom_type
        self.charge = charge

    def __str__(self):
        return '<Atom name="{name}" type="{atom_type}" charge="{charge}"/>'.format(**self.__dict__)

    def __eq__(self, other):
        """
        Check that all attributes are the same, EXCEPT for charge
        """
        name_eq = self.name == other.name
        type_eq = self.atom_type == other.atom_type
        return name_eq and type_eq

    def is_dummy(self):
        """Check if this atom is a dummy atom."""
        return self.atom_type == "dummy"


class _BondType(object):
    """
    Private class representing a bond between two atom types.
    """
    def __init__(self, atomtype1, atomtype2):
            # The exact order does not matter
            atoms = sorted([atomtype1, atomtype2])
            self.atomType1 = atoms[0]
            self.atomType2 = atoms[1]

    def __eq__(self, other):
        """Two bonds are the same if all their attributes are the same."""
        return self.__dict__ == other.__dict__


class _Bond(object):
    """
    Private class representing a bond between two atoms. Supports comparisons.
    """
    def __init__(self, atom1, atom2):
        if atom1 == atom2:
            raise ValueError("Can't define a bond from one atom to itself!")
        else:
            # Ensure that bonds between two atoms are always in the same order
            # The exact order does not matter
            atoms = sorted([atom1, atom2])
            self.atomName1 = atoms[0]
            self.atomName2 = atoms[1]

    def __eq__(self, other):
        """Two bonds are the same if all their attributes are the same."""
        return self.__dict__ == other.__dict__

    def __repr__(self):
        """FFXML representation of the bond"""
        return '<Bond atomName1="{atomName1}" atomName2="{atomName2}"/>'.format(**self.__dict__)

    __str__ = __repr__

    def has_atom(self, atomname):
        """
        Check if a particular atom is part of this bond.

        Parameters
        ----------
        atomname - str
            Name of the atom

        Returns
        -------
        bool - True if the atom is part of the bond
        """
        return atomname in (self.atomName1, self.atomName2)


class _State(object):
    """
    Private class representing a template of a single isomeric state of the molecule.
    """
    def __init__(self, index, log_population, g_k, net_charge, atom_list, pH):
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
        atom_list - list of str
            Atoms that need to be included in this isomer

        """
        self.index = index
        self.log_population = log_population
        self.g_k = g_k
        self.net_charge = net_charge
        self.atoms=OrderedDict()
        self.proton_count = -1
        for atom in atom_list:
            self.atoms[atom] = None
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

    def set_atom(self, atom):
        """
        Set the parameters for a single atom

        Parameters
        ----------
        atom : _Atom
            The parameters of the atom
        """
        if not isinstance(atom, _Atom):
            raise ValueError("Input needs to be an instance of class '_Atom'.")

        if atom.name not in self.atoms.keys():
            raise ValueError("Atom '{}' could not be found".format(atom.name))
        self.atoms[atom.name] = atom

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
            Contains the ffxml of the Epik isomers, net charge, and population
        gaff_xml : string, optional
            File location of a gaff.xml file. If specified, read gaff parameters from here.
            Otherwise, gaff parameters are taken from protons/forcefields/gaff.xml
        residue_name : str, optional, default = "LIG"
            name of the residue in the output template
        """
        self._input_state_data = input_state_data
        self._atom_names = list()
        self._bonds = list()
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

    def _make_output_tree(self, chimera=True):
        """
        Store all contents of a compiled ffxml file of all isomers, and add dummies for all missing hydrogens.
        """

        # Obtain information about all the atoms
        self._complete_atom_registry()
        # Obtain information about all the bonds
        self._complete_bond_registry()
        # Register the states
        self._complete_state_registry()
        # Interpolate differing atom types between states to create a single template state.
        if chimera:
            # Chimera takes the most populated state, and then adds missing parameters from the other states
            self._create_chimera_template()
        else:
            # Hybrid template does not favor one state over the other, but may not always yield a solution
            self._create_hybrid_template()

        # Set the initial state of the template that is read by OpenMM
        self._initialize_forcefield_template()
        # Add isomer specific information
        self._add_isomers()
        # Append extra parameters from frcmod
        self._append_extra_gaff_types()
        # Remove empty blocks, and unnecessary information in the ffxml tree
        self._sanitize_ffxml()

        return

    def _initialize_forcefield_template(self):
        """
        Set up the residue template using the first state of the molecule
        """

        residue = self.ffxml.xpath('/ForceField/Residues/Residue')[0]
        for atom in self._state_templates[0].atoms.values():
            residue.append(etree.fromstring(str(atom)))
        for bond in self._bonds:
            residue.append(etree.fromstring(str(bond)))

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
                for atom in isomer.atoms.values():
                    isomer_xml.append(etree.fromstring(str(atom)))
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

    def _create_chimera_template(self):
        """
        Start with atom types from the most populated state, and attempt to fill in the remaining atoms from the other
        states.
        
        Checks if bonded terms exist.
        If not, creates a new bond type with the properties of the same bond in the state where the atom existed.
        Bonds, angle, Torsions?
        """
        possible_types_per_atom = dict()  # Dictionary of all possible types for each atom, taken from all isomers
        available_parameters_per_type = dict()  # The GAFF parameters for all the atomtypes that may be used.

        # The final, uniform set of atomtypes that will be used
        final_types = dict()

        # Collect all possible types by looping through states
        for atomname in self._atom_names:
            possible_types_per_atom[atomname] = set()
            for state in self._state_templates:
                if state.atoms[atomname] is not None:
                    # Store the atomtype for this state as a possible pick
                    if atomname not in final_types:
                        final_types[atomname] = state.atoms[atomname].atom_type

                    # In case we need alternatives later, store potential types
                    possible_types_per_atom[atomname].add(state.atoms[atomname].atom_type)
                else:
                    # add missing atoms, using the placeholder type "dummy' for now and a net charge of 0.0
                    state.atoms[atomname] = _Atom(atomname, 'dummy', '0.0')

            # look up the parameters for the encountered atom types from all available parameter sources
            for atom_type in possible_types_per_atom[atomname]:
                if atom_type not in available_parameters_per_type.keys():
                    available_parameters_per_type[atom_type] = self._retrieve_atom_type_parameters(atom_type)

        # Make sure we haven't missed any atom types
        if len(final_types) != len(self._atom_names):
            missing = set(self._atom_names) - set(final_types.keys())
            raise RuntimeError("Did not find an atom type for {}".format(', '.join(missing)))

        # Keep looping until solution has not changed, which means all bonds have been found
        old_types = dict()
        while old_types != final_types:
            old_types = dict(final_types)
            # Validate that bonds exist for the atomtypes that were assigned
            for atomname in self._atom_names:
                atom_type = final_types[atomname]

                bonded_to = self._find_bond_partner_types(atomname, final_types)
                # Search from gaff and frcmod contents for all bonds that contain this atom type
                list_of_bond_params = self._bonds_including_type(atom_type, available_parameters_per_type)

                # Loop through all bonds to check if the bond types are defined
                for bond_partner_name, bond_partner_type in bonded_to.items():
                    this_bond_type = _BondType(atom_type, bond_partner_type)

                    # If there is no bond definition for these types
                    # propose a change of type to a different type from another state
                    # TODO If hydrogen atom involved,
                    # TODO could try to first update the type of the hydrogen
                    # TODO since that should not affect any of the other bonds in the system.
                    if this_bond_type not in list_of_bond_params:
                        # Keep track of whether a fix has been proposed
                        update_made = False
                        # Change the current atoms type to see if a bond exist
                        for possible_type in possible_types_per_atom[atomname]:
                            # Find all bonds that contain this new atom type
                            alternate_list_of_bond_params = self._bonds_including_type(possible_type, available_parameters_per_type)
                            if _BondType(possible_type, bond_partner_type) in alternate_list_of_bond_params:
                                log.debug(
                                    "Atom: %s type changed %s -> %s to facilitate binding to Atom: %s, with type %s",
                                          atomname, atom_type, possible_type, bond_partner_name, bond_partner_type)

                                # Update the current selection
                                final_types[atomname] = possible_type
                                update_made = True
                                break

                        # If the current atom could not be updated, attempt changing the partner
                        if not update_made:

                            # Loop through types of the bond partner found in each state
                            for possible_type in possible_types_per_atom[bond_partner_name]:
                                if _BondType(atom_type, possible_type) in list_of_bond_params:
                                    log.debug(
                                        "Atom: %s type changed %s -> %s to facilitate binding to Atom: %s, with type %s",
                                        bond_partner_name, bond_partner_type, possible_type, atomname, atom_type)

                                    # Update the current selection with the new type
                                    final_types[bond_partner_name] = possible_type
                                    update_made = True
                                    break

                        # If neither current atom, or partner atom types could be updated to match,
                        # both atoms will need to be changed to facilitate a bond between them
                        if not update_made:

                            # All possible types from each state
                            for possible_type_atom in possible_types_per_atom[atomname]:
                                # Find the bonds to this atom type
                                alternate_list_of_bond_params = self._bonds_including_type(possible_type_atom, available_parameters_per_type)

                                # All possible types for the partner from each state
                                for possible_type_partner in possible_types_per_atom[bond_partner_name]:


                                    if _BondType(possible_type_atom, possible_type_partner) in alternate_list_of_bond_params:
                                        log.debug(
                                            "Atom: %s type changed %s -> %s and \n "
                                            "Atom: %s type changed %s -> %s to facilitate bond.",
                                            atomname, atom_type, possible_type_atom, bond_partner_name, bond_partner_type, possible_type_partner)

                                        # Update both types with the new selection
                                        final_types[atomname] = possible_type_atom
                                        final_types[bond_partner_name] = possible_type_partner
                                        update_made = True
                                        break

                        # There are no bond parameters for this bond anywhere
                        # If you run into this error, likely, GAFF does not cover the protonation state you provided
                        if not update_made:
                            raise RuntimeError("Can not resolve bonds between Atoms {} - {}.\n"
                                               "Gaff types may not suffice to describe this molecule/protonation state.".format(atomname, bond_partner_name))

        # Assign the final atom types to each state
        for state_index in range(len(self._state_templates)):
            for atomname in self._atom_names:
                self._state_templates[state_index].atoms[atomname].atom_type = final_types[atomname]
        return

    @staticmethod
    def _bonds_including_type(atom_type, available_parameters_per_type):
        bond_params = available_parameters_per_type[atom_type]['bonds']
        list_of_bond_params = list()
        for bond_type in bond_params:
            list_of_bond_params.append(_BondType(bond_type.get('type1'), bond_type.get('type2')))
        return list_of_bond_params

    def _create_hybrid_template(self):
        """
        Interpolate differing atom types to create a single template state that has proper bonded terms for all atoms.
        """

        possible_types_per_atom = dict()  # Dictionary of all possible types for each atom, taken from all isomers
        available_parameters_per_type = dict()  # The GAFF parameters for all the atomtypes that may be used.

        # Collect all possible types by looping through states
        for atomname in self._atom_names:
            possible_types_per_atom[atomname] = set()
            for state in self._state_templates:
                if state.atoms[atomname] is not None:
                    # Store the atomtype for this state as a possible pick
                    possible_types_per_atom[atomname].add(state.atoms[atomname].atom_type)
                else:
                    # add missing atoms, using the placeholder type "dummy' for now and a net charge of 0.0
                    state.atoms[atomname] = _Atom(atomname, 'dummy', '0.0')

            # look up the parameters for the encountered atom types from all available parameter sources
            for atom_type in possible_types_per_atom[atomname]:
                if atom_type not in available_parameters_per_type.keys():
                    available_parameters_per_type[atom_type] = self._retrieve_atom_type_parameters(atom_type)

        # The final, uniform set of atomtypes that will be used
        final_types = dict()

        # Keep looping until all types have been assigned
        number_of_attempts = 0
        while len(final_types) != len(self._atom_names):

            # Deepcopy
            old_types = dict(final_types)
            # For those that need to be resolved
            for atomname, possible_types_for_this_atom in possible_types_per_atom.items():

                # Already assigned this atom, skip it.
                if atomname in final_types:
                    continue
                # If only one option available
                elif len(possible_types_for_this_atom) == 1:
                    final_types[atomname] = next(iter(possible_types_for_this_atom))
                # Not in the list of final assignments, and still has more than one option
                else:
                    # Dictionary of all the bonds that could/would have to be available when picking an atom type
                    bonded_to = self._find_all_potential_bond_types_to_atom(atomname, final_types, possible_types_per_atom)
                    # The atom types that could be compatible with at least one of the possible atom types for each atom
                    solutions = self._resolve_types(bonded_to, available_parameters_per_type,
                                                    possible_types_for_this_atom)
                    # Pick a solution
                    solution_one = next(iter(solutions.values()))
                    # If there is only one solution, that is the final solution
                    if len(solutions) == 1:
                        final_types[atomname] = list(solutions.keys())[0]
                    elif len(solutions) == 0:
                        # If this happens, you may manually need to assign atomtypes. The available types won't do.
                        raise ValueError("Cannot come up with a single set of atom types that describes all states in bonded form.")

                    # If more than one atomtype is possible for this atom, but all partner atom options are the same, just pick one.
                    elif all(solution_one == solution_value for solution_value in solutions.values()):
                        final_types[atomname] = list(solutions.keys())[0]

                    else:
                        # Some partner atoms might still be variable
                        # kick out invalid ones, and repeat procedure afterwards

                        old_possibilities = dict(possible_types_per_atom)
                        for partner_name in bonded_to.keys():
                            if partner_name in final_types.keys():
                                continue
                            else:
                                for partner_type in possible_types_per_atom[partner_name]:
                                    # if an atomtype in the current list of options did not match
                                    # any of the possible combinations with our potential solutions for the current atom
                                    if not any(partner_type in valid_match[partner_name] for valid_match in
                                               solutions.values()):
                                        # kick it out of the options
                                        possible_types_per_atom[partner_name].remove(partner_type)

                        # If this hasn't changed anything, remove an arbitrary option, and see if the molecule can be resolved in the next iteration.
                        if old_possibilities == possible_types_per_atom:
                            possible_types_per_atom[atomname].remove(next(iter(possible_types_per_atom[atomname])))


            # If there is more than one unique solution to the entire thing, this may result in an infinite loop
            # It is not completely obvious what would be the right thing to do in such a case.
            # It takes two iterations, one to identify what atoms are now invalid, and one to check whether the number
            # of (effective) solutions is equal to 1. If after two iterations, there are no changes, the algorithm is stuck
            if final_types == old_types and number_of_attempts % 2 == 0 and number_of_attempts > 20:
                raise RuntimeError("Can't seem to resolve atom types, there might be more than 1 unique solution.")
            number_of_attempts += 1

        log.debug("Final atom types have been selected.")

        # Assign the final atom types to each state
        for state_index in range(len(self._state_templates)):
            for atomname in self._atom_names:
                self._state_templates[state_index].atoms[atomname].atom_type = final_types[atomname]

        return

    def _find_all_potential_bond_types_to_atom(self, atomname, final_types, potential_types):
        """ Find all the atoms it is bonded to, and collect their types

        Parameters
        ----------
        atomname : str
            Name of the atom
        final_types : dict
            dictionary by atom name of the types that have already been determined
        potential_types : dict
            dictionary by atom name of the types that could be chosen as the final solution

        Returns
        -------
        bonded_to : dict
            The atoms that this atom is bonded to, and the types/potential type of the bond partners

        """
        bonded_to = dict()
        for bond in self._bonds:
            if bond.has_atom(atomname):
                atomname1 = bond.atomName1
                atomname2 = bond.atomName2
                if atomname1 == atomname:
                    if atomname2 in final_types:
                        bonded_to[atomname2] = [final_types[atomname2]]
                    else:
                        bonded_to[atomname2] = potential_types[atomname2]
                else:
                    if atomname1 in final_types:
                        bonded_to[atomname1] = [final_types[atomname1]]
                    else:
                        bonded_to[atomname1] = potential_types[atomname1]
        return bonded_to

    def _find_bond_partner_types(self, atomname, final_types):
        """ Find all the atoms it is bonded to, and collect their types

        Parameters
        ----------
        atomname : str
            Name of the atom
        final_types : dict
            dictionary by atom name of the types that have already been determined
        Returns
        -------
        bonded_to : dict
            The atoms that this atom is bonded to, and the types/potential type of the bond partners

        """
        bonded_to = dict()
        for bond in self._bonds:
            if bond.has_atom(atomname):
                atomname1 = bond.atomName1
                atomname2 = bond.atomName2
                if atomname1 == atomname:
                    bonded_to[atomname2] = final_types[atomname2]
                else:

                    bonded_to[atomname1] = final_types[atomname1]
        return bonded_to

    @staticmethod
    def _resolve_types(bonded_to, params, type_list):
        """

        Parameters
        ----------
        bonded_to - dictionary of the atoms its bonded to, and their potential types
        params - gaff parameters for all atomtypes
        type_list - all possible types for this atom

        Returns
        -------

        solutions : dict
            keys in the dictionary are atom_types that have bonded parameters for binding to at least one of the potential atom
            types of its bond partners

        """
        # After collecting the necessary bonded types, loop through possible types
        # for the current atom, and pick the first one that has a bond for all of them.

        # dictionary of atom_types that have bonded parameters for binding to at least one of the potential atom
        # types of its bond partners
        solutions = dict()
        for atom_type in type_list:
            bond_params = params[atom_type]['bonds']

            valid_match = dict()
            have_found_valid_type = True

            # The partner atoms that the current atom is bonded to, and a list of potential atom types for each partner
            for bonded_atom, types_of_bonded_atom in bonded_to.items():
                matched_this_atom = False
                # The bonded atom could have several types, go through all of them
                for type_of_bonded_atom in types_of_bonded_atom:

                    # Go through the list of bonds and try to find if there is a bond for the current two types
                    for bond_param in bond_params:
                        bond_param_atoms = set()
                        bond_param_atoms.add(bond_param.attrib['type1'])
                        bond_param_atoms.add(bond_param.attrib['type2'])

                        # If there is a bond that describes binding between the proposed atom type, and
                        # one of the candidates of its bonding partner, store which atom type the
                        # partner has, so we can potentially reduce the list of atom types for the
                        # partner at a later stage
                        if atom_type in bond_param_atoms and type_of_bonded_atom in bond_param_atoms:
                            matched_this_atom = True
                            if bonded_atom in valid_match:
                                valid_match[bonded_atom].append(type_of_bonded_atom)
                            else:
                                valid_match[bonded_atom] = [type_of_bonded_atom]

                # Could not detect bonds parameters between this atom type, and the types of atoms that
                # it needs to be bonded to
                if not matched_this_atom:
                    have_found_valid_type = False
                    valid_match = dict()
                    break

            if have_found_valid_type:
                # Store the information on the potential atom types of the bond partners
                # This can help narrow down which types these atoms can take on in the final solution
                solutions[atom_type] = valid_match

        return solutions

    def _retrieve_atom_type_parameters(self, atom_type_name):
        """ Look through FFXML files and find all parameters pertaining to the supplied atom type.
        Returns
        -------
        params : dict(atomtypes=[], bonds=[], angles=[], propers=[], impropers=[], nonbonds=[])
            Dictionary of lists by force type
        """

        # Storing all the detected parameters here
        params = dict(atomtypes=[], bonds=[], angles=[], propers=[], impropers=[], nonbonds=[])

        if atom_type_name is None:
            return params

        # Loop through different sources of parameters
        for xmltree in self._xml_parameter_trees:
            # Match the type of the atom in the AtomTypes block
            for atomtype in xmltree.xpath("/ForceField/AtomTypes/Type"):
                if atomtype.attrib['name'] == atom_type_name:
                    params['atomtypes'].append(atomtype)

            # Match the bonds of the atom in the HarmonicBondForce block
            for bond in xmltree.xpath("/ForceField/HarmonicBondForce/Bond"):
                if atom_type_name in (bond.attrib['type1'], bond.attrib['type2']):
                    params['bonds'].append(bond)

            # Match the angles of the atom in the HarmonicAngleForce block
            for angle in xmltree.xpath("/ForceField/HarmonicAngleForce/Angle"):
                if atom_type_name in (angle.attrib['type1'], angle.attrib['type2'], angle.attrib['type3']):
                    params['angles'].append(angle)

            # Match proper dihedral of the atom in PeriodicTorsionForce block
            for proper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Proper"):
                if atom_type_name in (proper.attrib['type1'], proper.attrib['type2'], proper.attrib['type3'], proper.attrib['type4']):
                    params['propers'].append(proper)

            # Match improper dihedral of the atom in PeriodicTorsionForce block
            for improper in xmltree.xpath("/ForceField/PeriodicTorsionForce/Improper"):
                if atom_type_name in (improper.attrib['type1'], improper.attrib['type2'], improper.attrib['type3'], improper.attrib['type4']):
                    params['impropers'].append(improper)

            # Match nonbonded type of the atom in NonbondedForce block
            for nonbond in xmltree.xpath("/ForceField/NonbondedForce/Atom"):
                if nonbond.attrib['type'] == atom_type_name:
                    params['nonbonds'].append(nonbond)

        return params

    def _complete_bond_registry(self):
        """
        Register all bonds.
        """

        for state in self._input_state_data:
            for bond in state['ffxml'].xpath('/ForceField/Residues/Residue/Bond'):
                self._bonds.append(_Bond(bond.attrib['atomName1'], bond.attrib['atomName2']))
        self._unique_bonds()

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

    def _complete_atom_registry(self):
        """
        Registers unique atom names. Store in self._atom_names from Residue
        """
        for state in self._input_state_data:
            for atom in state['ffxml'].xpath('/ForceField/Residues/Residue/Atom'):
                atom_name = atom.attrib['name']
                if atom_name not in self._atom_names:
                    self._atom_names.append(atom_name)

        return

    def _complete_state_registry(self):
        """
        Store all the properties that are specific to each state
        """
        charges = list()
        for index, state in enumerate(self._input_state_data):
            net_charge = state['net_charge']
            charges.append(int(net_charge))
            template = _State(index,
                              state['log_population'],
                              0.0, # set g_k defaults to 0 for now
                              net_charge,
                              self._atom_names,
                              state['pH']
                              )
            for xml_atom in state['ffxml'].xpath('/ForceField/Residues/Residue/Atom'):
                template.set_atom(_Atom(xml_atom.attrib['name'], xml_atom.attrib['type'], xml_atom.attrib['charge']))

            self._state_templates.append(template)

        min_charge = min(charges)
        for state in self._state_templates:
            state.set_number_of_protons(min_charge)

        return

    def _unique_bonds(self):
        """Ensure only unique bonds are kept."""
        bonds = list()
        for bond in self._bonds:
            if bond not in bonds:
                bonds.append(bond)

        self._bonds = bonds

    def _sort_bonds(self):
        self._bonds = sorted(self._bonds)  # Sort by first, then second atom in bond


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


def generate_epik_states(inputmae: str, outputmae: str, pH: float, max_penalty: float=10.0, workdir: str=None, tautomerize: bool=False, **kwargs):
    """Generate protonation states using Epik, with shortcuts to a few useful settings.

    Parameters
    ----------
    inputmae - location of a maestro input file for Epik.
    outputmae - location for the output file containing protonation states
    pH - the pH value
    max_penalty - the max energy penalty in kT, default=10.0
    workdir - Path/directory to place output files, including logs. If `outputmae` is a relative path, it will be placed here.
    tautomerize = If too, besides protonation states generate tautomers

    Notes
    -----
    Epik doesn't retain the input protonation if it's non-relevant.

    """
    log.info("Running Epik to detect protomers and tautomers...")
    inputmae = os.path.abspath(inputmae)
    oldwd = os.getcwd()
    try:
        if workdir is not None:            
            os.chdir(workdir)
            log.info("Log files can be found in {}".format(workdir))
        omt.schrodinger.run_epik(inputmae, outputmae, ph=pH, min_probability=np.exp(-max_penalty), tautomerize=tautomerize, **kwargs)
    finally:
        os.chdir(oldwd)


def retrieve_epik_info(epik_mae: str) -> list:
    """
    Retrieve the state populations and charges from the Epik output maestro file

    Parameters
    ----------
    epik_mae - location of the Epik output (a maestro file)

    Returns
    -------
    list of dicts
        has the keys log_population, net_charge
    """

    penalty_tag = "r_epik_State_Penalty"
    net_charge_tag = "i_epik_Tot_Q"
    props = omt.schrodinger.run_proplister(epik_mae)

    all_info = list()

    for state in props:
        state_info = dict()
        epik_penalty = state[penalty_tag]
        state_info["log_population"] = float(epik_penalty) / (-298.15 * 1.9872036e-3)
        state_info['net_charge'] = int(state[net_charge_tag])
        all_info.append(state_info)

    return all_info


def epik_results_to_mol2(epik_mae: str, output_mol2: str):
    """
    Map the hydrogen atoms between Epik states, and return a mol2 file that
    should be ready to parametrize.

    Parameters
    ----------
    epik_mae: location of the maestro file produced by Epik.

    Notes
    -----
    This renames the hydrogen atoms in your molecule so that
     no ambiguity can exist between protonation states.
    """
    if not output_mol2[-5:] == ".mol2":
        output_mol2 += ".mol2"
    # Generate a file format that Openeye can read
    unique_filename = str(uuid.uuid4())
    tmpfilename = "{}.mol2".format(unique_filename)
    omt.schrodinger.run_structconvert(epik_mae, tmpfilename)

    ifs = oechem.oemolistream()
    ifs.open(tmpfilename)

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
    os.remove(tmpfilename)


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
        One dict is necessary for every isomer. Dict should contain 'log_population' and 'net_charge' keys.
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
    for isomer_index, oemolecule in enumerate(ifs.GetOEMols()):
        # generateForceFieldFromMolecules needs a list
        # Make new ffxml for each isomer
        log.info("ffxml generation for {}".format(isomer_index))
        ffxml = omtff.generateForceFieldFromMolecules([oemolecule], normalize=False)
        log.info(ffxml)
        isomers[isomer_index]['ffxml'] = etree.fromstring(ffxml, parser=xmlparser)
        isomers[isomer_index]['pH'] = pH

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


def extract_residue(inputfile:str, outputfile:str, resname:str ):
    """Extract a specific residue from a file and write to new file. Useful for setting up a calibration.

    Parameters
    ----------
    inputfile - A file that is compatible with MDtraj (see mdtraj.org)
    outputfile - Filename for the output.
    resname - the residue name in the system

    """
    input_traj = mdtraj.load(inputfile)
    res = input_traj.topology.select("resn {}".format(resname))
    input_traj.restrict_atoms(res)
    input_traj.save(outputfile)


def prepare_calibration_system(vacuum_file:str, output_file:str, ffxml:str, hxml:str):
    """Add hydrogens to a residue based on forcefield and hydrogen definitons, and then solvate.

    Note that no salt is added. We use saltswap for this.

    Parameters
    ----------
    vacuum_file - a single residue in vacuum to add hydrogens to and solvate.
    output_file - the basename for an output mmCIF file with the solvated system.
    ffxml - the forcefield file containing the residue definition
    hxml - the hydrogen definition xml file

    """

    # Load relevant template definitions for modeller, forcefield and topology
    app.Modeller.loadHydrogenDefinitions(hxml)
    forcefield = app.ForceField('amber10-constph.xml', 'gaff.xml', ffxml, 'tip3p.xml', 'ions_tip3p.xml')

    pdb = app.PDBFile(vacuum_file)
    modeller = app.Modeller(pdb.topology, pdb.positions)

    # The system will likely have different hydrogen names.
    # In this case its easiest to just delete and re-add with the right names based on hydrogen files
    to_delete = [atom for atom in modeller.topology.atoms() if atom.element.symbol in ['H']]
    modeller.delete(to_delete)

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

