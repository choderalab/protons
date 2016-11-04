from __future__ import print_function

import logging
import os
import random
import re
import shutil
import tempfile
from collections import OrderedDict
from copy import deepcopy
from glob import glob
from math import exp

import openmoltools as omt
import parmed
from joblib import Parallel, delayed
from lxml import etree, objectify
from six.moves import input

from protons.logger import log


def _make_xml_root(root_name):
    """
    Create a new xml root object with a given root name

    Parameters
    ----------
    root_name - str
        The name of the xml root.

    Returns
    -------
        Xml tree
    """
    xml = '<{0}></{0}>'.format(root_name)
    root = objectify.fromstring(xml)
    return root


def _make_cph_xml():
    """
    Returns an empty xml file template for constant-pH simulation
    """

    forcefield_block = _make_xml_root("ForceField")
    isomerdata_block = _make_xml_root("IsomerData")
    atomtypes_block = _make_xml_root("AtomTypes")
    residues_block = _make_xml_root("Residues")
    residue_block = _make_xml_root("Residue")
    harmonicbondforce_block = _make_xml_root("HarmonicBondForce")
    harmonicangleforce_block = _make_xml_root("HarmonicAngleForce")
    periodictorsionforce_block = _make_xml_root("PeriodicTorsionForce")
    # Blocks for NonbondedForce and GBSAOBC forces are generated elsewhere on the fly

    residue_block.append(isomerdata_block)
    residues_block.append(residue_block)
    forcefield_block.append(atomtypes_block)
    forcefield_block.append(residues_block)
    forcefield_block.append(harmonicbondforce_block)
    forcefield_block.append(harmonicangleforce_block)
    forcefield_block.append(periodictorsionforce_block)
    return forcefield_block


class _Bond(object):
    """
    Private class representing a bond between two atoms. Supports comparisons.
    """
    def __init__(self, atom1, atom2):
        atom1 = int(atom1)
        atom2 = int(atom2)
        if atom1 < atom2:
            self.from_atom = atom1
            self.to_atom = atom2
        elif atom2 < atom1:
            self.from_atom = atom2
            self.to_atom = atom1
        else:
            raise ValueError("Can't define a bond from one atom to itself!")

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __gt__(self, other):
        if self.from_atom > other.from_atom:
            return True
        elif self.from_atom == other.from_atom and self.to_atom > other.to_atom:
            return True
        else:
            return False

    def __lt__(self, other):
        if self.from_atom < other.from_atom:
            return True
        elif self.from_atom == other.from_atom and self.to_atom < other.to_atom:
            return True
        else:
            return False

    def __le__(self, other):
        if self.from_atom < other.from_atom:
            return True
        elif self.from_atom == other.from_atom and self.to_atom <= other.to_atom:
            return True
        else:
            return False

    def __ge__(self, other):
        if self.from_atom > other.from_atom:
            return True
        elif self.from_atom == other.from_atom and self.to_atom >= other.to_atom:
            return True
        else:
            return False

    def __str__(self):
        return '<Bond from="{from_atom}" to="{to_atom}"/>'.format(**self.__dict__)

    __repr__ = __str__


class _Atom(object):
    """
    Private class representing an atom that is part of a single residue
    """

    def __init__(self, resname, name, number=None):
        """

        Parameters
        ----------
        resname - str, name of the residue that the atom is in
        name - name of the atom
        number - str or int,
            optional, the atomic number
        """
        self.resname = resname
        self.name = name

        #  If number not supplied, guess by looking for a number at the end of the atom name
        # numerical component in name. NOT equivalent to index in Residue block
        if number is None:
            self.number = int(re.findall(r"\d+", name)[-1])
        else:
            self.number = number
        self.type = "{}-{}".format(resname,name)

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def _same_res(self, other):
        """Compare resnames"""
        return self.resname == other.resname

    def __int__(self):
        """Convert atom to integer."""
        return int(self.number) - 1

    def __gt__(self, other):
        if not self._same_res(other):
            raise ValueError("Cannot compare atoms between residues.")
        if self.number > other.number:
            return True
        else:
            return False

    def __lt__(self, other):
        if not self._same_res(other):
            raise ValueError("Cannot compare atoms between residues.")
        if self.number < other.number:
            return True
        else:
            return False

    def __le__(self, other):
        if not self._same_res(other):
            raise ValueError("Cannot compare atoms between residues.")
        if self.number <= other.number:
            return True
        else:
            return False

    def __ge__(self, other):
        if not self._same_res(other):
            raise ValueError("Cannot compare atoms between residues.")
        if self.number >= other.number:
            return True
        else:
            return False

    def __str__(self):
        return '<Atom name="{name}" type="{type}"/>'.format(**self.__dict__)

    def __repr__ (self):
        self.__str__() + str(self.number)


class _AtomType(object):
    """Private class representing an atomtype"""
    def __init__(self, resname, atomname, aclass, element, mass):
        """

        Parameters
        ----------
        resname - str
            name of the residue that the Atom belongs to
        atomname - str
            Name of the atom that the type refers to
        aclass - str
            Class of the atom type
        element - str
            Element of the atom type
        mass - str or float
            Mass of the atom type
        """

        self.name = "{}-{}".format(resname, atomname)
        self.aclass = aclass
        self.element = element
        self.mass = float(mass)
        self.number = int(re.findall(r"\d+", atomname)[-1])

    def __eq__(self, other):
        return self.__dict__ == other.__dict__

    def __str__(self):
        return '<Type name="{name}" class="{aclass}" element="{element}" mass="{mass}"/>'.format(**self.__dict__)

    __repr__ = __str__


class _NonbondedForce(object):
    """
    Private class representing the parameters for a nonbonded atom type
    """
    def __init__(self, resname, state, atomname, epsilon, sigma, charge):
        """

        Parameters
        ----------
        resname - str
            Name of the residue
        state - int
            Index of the isomer state
        atomname - str
            Name of the atom that the Nonbonded force points to
        epsilon - str
            Value of epsilon LJ parameter
        sigma - str
            Value of sigma LJ parameter
        charge - str
            Value of partial charge parameter
        """
        self.resname = resname
        self.state = state
        self.atomname = atomname
        self.epsilon = epsilon
        self.sigma = sigma
        self.charge = charge
        self.type = self.resname + "-" + self.atomname

    def __eq__(self, other):
        if not self.type == other.type:
            raise ValueError("Can only compare identical-atom types.")

        return self.sigma == other.sigma and self.epsilon == other.epsilon

    def __str__(self):
        return '<Atom type="{type}" charge="{charge}" sigma="{sigma}" epsilon="{epsilon}"/>'.format(**self.__dict__)

    __repr__ = __str__


class _GBSAOBCForce(object):
    """
    Private class representing the parameters of a GBSAOBCForce type
    """
    def __init__(self, resname, state, atomname, charge, radius, scale):
        """

        Parameters
        ----------
        Parameters
        ----------
        resname - str
            Name of the residue
        state - int
            Index of the isomer state
        atomname - str
            Name of the atom that the Nonbonded force points to
        charge - str
            Value of partial charge parameter
        radius - str
            Value of the GBSA radius parameter
        scale - str
            Value of the GBSA scale parameter

        """
        self.resname = resname
        self.state = state
        self.atomname = atomname
        self.radius = radius
        self.scale = scale
        self.charge = charge
        self.type = self.resname + "-" + self.atomname

    def __eq__(self, other):
        if not self.type == other.type:
            raise ValueError("Can only compare identical-atom types.")

        return self.radius == other.radius and self.scale == other.scale and self.charge == other.charge

    def __str__(self):
        return '<Atom type="{type}" charge="{charge}" radius="{radius}" scale="{scale}"/>'.format(**self.__dict__)

    __repr__ = __str__


class _Isomer(object):
    """Private class representing a single isomeric state of the molecule.
    """
    def __init__(self, index, population, charge, coulomb14scale, lj14scale):
        """

        Parameters
        ----------
        index - int
            Index of the isomeric state
        population - str
            Solvent population of the isomeric state
        charge - str
            Net charge of the isomeric state
        coulomb14scale - str
            1-4 interaction scaling factor for Coulomb interactions
        lj14scale - str
            1-4 interaction scaling factor for LJ interactions
        """
        self.index = index
        self.population = population
        self.charge = charge
        self.nonbondedtypes = OrderedDict()
        self.gbsaobctypes = OrderedDict()
        self.coulomb14sale = coulomb14scale
        self.lj14scale = lj14scale

    def __str__(self):
        return '<IsomericState index="{index}" population="{population}" netcharge="{charge}"/>'.format(**self.__dict__)

    def __len__(self):
        return len(self.nonbondedtypes)

    def to_xml(self, gb_params):
        """
        Generate the IsomericState blocks for the output file.
        """

        xml = etree.fromstring(str(self))
        nonbondblock = _make_xml_root("NonbondedForce")
        if gb_params:
            gbsaobc_block = _make_xml_root("GBSAOBCForce")

        nonbondblock.attrib['coulomb14scale'] = self.coulomb14sale
        nonbondblock.attrib['lj14scale'] = self.lj14scale

        for nonbondtype in self.nonbondedtypes.values():
            nonbondblock.append(etree.fromstring(str(nonbondtype)))

        if gb_params:
            for gbsaobc_type in self.gbsaobctypes.values():
                gbsaobc_block.append(etree.fromstring(str(gbsaobc_type)))

        xml.append(nonbondblock)
        if gb_params:
            xml.append(gbsaobc_block)

        return xml

    __repr__ = __str__

    def to_placeholder_xml(self):
        """
        Dummy placeholders for Nonbonded and GBSAOBC forces xml file
        """
        nonbond_block = _make_xml_root("NonbondedForce")
        nonbond_block.attrib['coulomb14scale'] = self.coulomb14sale
        nonbond_block.attrib['lj14scale'] = self.lj14scale

        for nonbondtype in self.nonbondedtypes.values():
            dummytype = deepcopy(nonbondtype)
            dummytype.charge = 0.0
            dummytype.epsilon = 1.0
            dummytype.sigma = 1.0
            nonbond_block.append(etree.fromstring(str(dummytype)))

        gbsaobc_block = _make_xml_root("GBSAOBCForce")

        for gbsaobc_type in self.gbsaobctypes.values():
            dummytype = deepcopy(gbsaobc_type)
            dummytype.charge = 0.0
            dummytype.radius = 0.0
            dummytype.scale = 1.0
            gbsaobc_block.append(etree.fromstring(str(dummytype)))

        return nonbond_block, gbsaobc_block


class _TitratableForceFieldCompiler(object):
    """
    Compiles an intermediate xml file to the final constant-ph ffxml file.
    """
    # format of atomtype name
    # Groups 1: resname, 2: isomer number, 3: atom letter, 4 atom number
    typename_format = re.compile(r"^(\w+)-(\d+)-(\D+)(\d+)$")

    def __init__(self, xmlfile, autoresolve=1, write_gb_params=True):
        """
        Compliles the intermediate ffxml files into a constant-pH compatible ffxml file.

        Parameters
        ----------
        xmlfile - str
            Path of the intermediate xml file.
        autoresolve - int, optional
            Automatically resolve any type conflicts by picking the entry numbered. Unpredictable.
            Choose 0 unless you know exactly what you are doing.
            0 leads to an interactive resolution of conflicts (recommended!).
        write_gb_params - bool, optional
            Write GB parameters into output xml file.

        """
        self._atoms = OrderedDict()
        self._bonds = list()
        self._atom_types = list()
        self._isomeric_states = OrderedDict()
        xmlparser = etree.XMLParser(remove_blank_text=True, remove_comments=True)
        self._input_tree = etree.parse(xmlfile, parser=xmlparser)
        self._make_output_tree(autoresolve, write_gb_params)

    def _make_output_tree(self, autoresolve, write_gb_params):
        """
        Store all contents of a compiled ffxml file of all isomers, and add dummies for all missing hydrogens.

        autoresolve - int
            Automatically pick all parameters from the numbered entery.
            Use 0 for interactive picking (recommended.)

        write_gb_params - bool
            Include GB parameters in output.

        """
        self._resname = None
        # Create a registry of atom names in self._atoms and atom types in self._atom_types
        self._complete_atom_registry(autoresolve=autoresolve)
        self._output_tree = _make_cph_xml()

        # Add missing atoms to shift indices of atoms in bonds correctly, then register bonds
        self._complete_bond_registry()

        # Register the isomeric states (without nonbonded parameters)
        self._complete_state_registry()

        # Add nonbonded terms for each state
        self._complete_nonbonded_registry(autoresolve=autoresolve)

        # Add GB parameters if applicable
        if write_gb_params: self._complete_gbparams_registry()

        self._sort_bonds()
        self._sort_atypes()

        # Add atoms and bonds to the output
        for residue in self._output_tree.xpath('/ForceField/Residues/Residue'):
            residue.attrib["name"] = self._resname

            for atom in self._atoms.values():
                residue.append(etree.fromstring(str(atom)))

            for bond in self._bonds:
                residue.append(etree.fromstring(str(bond)))

        # Add atomtypes to the output
        for atypes in self._output_tree.xpath('/ForceField/AtomTypes'):
            for atype in self._atom_types:
                atypes.append(etree.fromstring(str(atype)))

        # Copy bonded definitions from input directly to output
        for hbfblock in self._output_tree.xpath('/ForceField/HarmonicBondForce'):
            for harmonicbondforce in self._input_tree.xpath('/IntermediateResidueTemplate/ForceField/HarmonicBondForce/Bond'):
                hbfblock.append(harmonicbondforce)

        for hafblock in self._output_tree.xpath('/ForceField/HarmonicAngleForce'):
            for harmonicangleforce in self._input_tree.xpath('/IntermediateResidueTemplate/ForceField/HarmonicAngleForce/Angle'):
                hafblock.append(harmonicangleforce)

        for pdtblock in self._output_tree.xpath('/ForceField/PeriodicTorsionForce'):
            for propertorsionforce in self._input_tree.xpath('/IntermediateResidueTemplate/ForceField/PeriodicTorsionForce/Proper'):
                pdtblock.append(propertorsionforce)
            for impropertorsionforce in self._input_tree.xpath('/IntermediateResidueTemplate/ForceField/PeriodicTorsionForce/Improper'):
                pdtblock.append(impropertorsionforce)

        # Fill in nonbonded  and gbsaobc atomtype placeholders
        for forcefieldblock in self._output_tree.xpath('/ForceField'):
            nonbondplaceholder, gbsaobcplaceholder = self._isomeric_states[1].to_placeholder_xml()

            forcefieldblock.append(nonbondplaceholder)
            if write_gb_params:
                forcefieldblock.append(gbsaobcplaceholder)

        # Fill in isomer specific nonbonded and gbsaobctypes
        for isodatablock in self._output_tree.xpath('/ForceField/Residues/Residue/IsomerData'):
            for isoindex, isostate in self._isomeric_states.items():
                isodatablock.append(isostate.to_xml(write_gb_params))

    def _complete_bond_registry(self):
        """
        Register all bonds.
        """

        for residue in self._input_tree.xpath('/IntermediateResidueTemplate/ForceField/Residues/Residue'):
            per_state_index = OrderedDict()
            for aix, atom in enumerate(residue.xpath('Atom')):

                per_state_index[aix] = atom.attrib["name"]

            for bond in residue.xpath('Bond'):
                atomnames = list(self._atoms.keys())
                from_atom = per_state_index[int(bond.attrib["from"])]
                to_atom = per_state_index[int(bond.attrib["to"])]

                self._bonds.append(_Bond(atomnames.index(from_atom), atomnames.index(to_atom)))

        self._unique_bonds()

    def write(self, filename=None, moleculename="LIG"):
        """Generate the output xml."""

        # Get rid of extra junk information that is added to the xml files.
        objectify.deannotate(self._output_tree)
        etree.cleanup_namespaces(self._output_tree)

        # Generate the string version. Replace ligand name that comes out of antechamber/tleap
        xmlstring = etree.tostring(self._output_tree, encoding="utf-8", pretty_print=True, xml_declaration=False)
        xmlstring = xmlstring.decode("utf-8")
        xmlstring = xmlstring.replace("isom-", "{}-".format(moleculename))
        xmlstring = xmlstring.replace('name="isom"', 'name="{}"'.format(moleculename))

        if filename is not None:
            with open(filename, 'w') as fstream:
                fstream.write(xmlstring)

        return xmlstring

    def _complete_atom_registry(self, autoresolve):
        """
        Registers unique atom names. Store in self._atomnames from atom type list.
        """
        for atype in self._input_tree.xpath('/IntermediateResidueTemplate/ForceField/AtomTypes/Type'):
            matched_names = _TitratableForceFieldCompiler.typename_format.search(atype.attrib['name'])
            if self._resname is None:
                self._resname = matched_names.group(1)
            atomname = matched_names.group(3, 4)
            atomname = ''.join(atomname)
            if atomname not in self._atoms:
                self._atoms[atomname] = _Atom(self._resname, atomname)
            self._atom_types.append(_AtomType(self._resname, atomname, atype.attrib['class'], atype.attrib['element'], atype.attrib['mass']))

        self._unique_atom_types(autoresolve=autoresolve)
        self._sort_atoms()

    def _complete_state_registry(self):

        # Default scale in case we can't find it.
        coulomb14scale = 1.0
        lj14scale = 1.0

        for nonbondblock in self._input_tree.xpath("/IntermediateResidueTemplate/ForceField/NonbondedForce"):
            coulomb14scale = nonbondblock.attrib['coulomb14scale']
            lj14scale = nonbondblock.attrib['lj14scale']

        # Create new states for each proto/tautomer
        for isostate in self._input_tree.xpath('/IntermediateResidueTemplate/IsomerData/IsomericState'):
            isomer = _Isomer(int(isostate.attrib["index"]), isostate.attrib["population"], isostate.attrib["netcharge"], coulomb14scale, lj14scale)
            self._isomeric_states[int(isostate.attrib["index"])] = isomer

    def _complete_nonbonded_registry(self, autoresolve):
        """
        Register the non_bonded interactions for all states.
        """

        for isomer_state_index, isomer_state in self._isomeric_states.items():
            # Keep track of atoms that haven't been observed for this state, so we can add dummies later.
            unmatched_atoms = list(self._atoms.keys())

            # Atom types are pulled from the NonbondedForce block in the intermediate xml file
            for nonbond_atom in self._input_tree.xpath("/IntermediateResidueTemplate/ForceField/NonbondedForce/Atom"):
                matched_names = _TitratableForceFieldCompiler.typename_format.search(nonbond_atom.attrib['type'])
                atomname = matched_names.group(3, 4)
                atomname = ''.join(atomname)
                # atom types have the isomer in them, also, converting zero to one based
                state_index_nonbonded_atom = int(matched_names.group(2)) + 1

                # If the isomer matches the state that the atom type belongs to, store it as part of the state.
                if state_index_nonbonded_atom == isomer_state_index:
                    unmatched_atoms.remove(atomname)
                    self._isomeric_states[isomer_state_index].nonbondedtypes[atomname] = _NonbondedForce(self._resname, isomer_state_index, atomname, nonbond_atom.attrib['epsilon'], nonbond_atom.attrib['sigma'], nonbond_atom.attrib['charge'])

            # Add dummies for any atom undetected in the entire nonbonded block
            for unmatched in unmatched_atoms:
                self._isomeric_states[isomer_state_index].nonbondedtypes[unmatched] = _NonbondedForce(self._resname,
                                                                                                      isomer_state_index,
                                                                                                      unmatched, "0.0",
                                                                                                      "0.0",
                                                                                                      "0.0")

        # We need to resolve LJ types so they don't change between atoms, and fill in LJ for dummies.
        for atom in self._atoms.values():
            self._resolve_lj_params(atom, autoresolve=autoresolve)

    def _complete_gbparams_registry(self):
        """
        Register the GBSAOBC parameters for each isomeric state.
        """

        for isomeric_state_index, isomeric_state in self._isomeric_states.items():
            unmatched_atoms = list(self._atoms.keys())
            # Keep track of atoms that haven't been observed for this state, so we can add dummies later.
            for gbsaobcatom in self._input_tree.xpath("/IntermediateResidueTemplate/GBSAOBCForce/Atom"):

                matched_names = _TitratableForceFieldCompiler.typename_format.search(gbsaobcatom.attrib['type'])
                atomname = matched_names.group(3, 4)
                atomname = ''.join(atomname)
                # atom types have the isomer encoded in them, also, converting zero to one based
                state_index_gbsa_atom = int(matched_names.group(2)) + 1
                if state_index_gbsa_atom == isomeric_state_index:
                    unmatched_atoms.remove(atomname)

                    # If the isomer matches the state that the atom type belongs to, store it as part of the state.
                    # Ensure charge is same as nonbonded type charge.
                    self._isomeric_states[isomeric_state_index].gbsaobctypes[atomname] = _GBSAOBCForce(self._resname, isomeric_state_index, atomname,
                                                                                           self._isomeric_states[isomeric_state_index].nonbondedtypes[atomname].charge,
                                                                                           gbsaobcatom.attrib['radius'],
                                                                                           gbsaobcatom.attrib['scale'])

            # Add dummies for any atom undetected in the entire GBSAOBC block
            for unmatched in unmatched_atoms:
                # charge should be 0, but this also checks if we already knew about the atom from nonbonded forces,
                # in case there is a discrepancy.
                self._isomeric_states[isomeric_state_index].gbsaobctypes[unmatched] = _GBSAOBCForce(self._resname,
                                                                                                    isomeric_state_index,
                                                                                                    unmatched,
                                                                                                    self._isomeric_states[
                                                                                                        isomeric_state_index].nonbondedtypes[
                                                                                                        unmatched].charge,
                                                                                                    0.0,
                                                                                                    1.0)

    def _resolve_lj_params(self, atom, autoresolve):
        typelist = OrderedDict()
        for isomeric_state_index, isomeric_state in self._isomeric_states.items():
            nbtype = isomeric_state.nonbondedtypes[atom.name]
            # Assume zero values for all are newly added dummy atoms, and should not be suggested as types.
            if (float(nbtype.epsilon) > 0.0 and float(nbtype.sigma) > 0.0) or abs(float(nbtype.charge)) > 0.0:
               typelist[isomeric_state_index] = nbtype

        # Select an arbitrary type to compare all to
        randomindex = random.choice(list(typelist.keys()))
        first_state_type = typelist[randomindex]
        # Check if all remaining types have the same LJ parameters
        if all(atype == first_state_type for atype in typelist.values()):
            log.debug("All types shared for {}".format(atom))
            # Still need to standardize the dummies.
            self._standardize_lj_params(atom, randomindex)
        elif autoresolve:
            if autoresolve not in typelist.keys():
                raise UserWarning("Nonbonded type reference provided a dummy atom, this is probably a bad idea.")
            for key, value in typelist.items():
                log.debug("{}:{}".format(key, value))
            self._standardize_lj_params(atom, autoresolve)
        else:
            self._manual_lj_params(atom, typelist)

    def _standardize_lj_params(self, atom, reference_state_index):
        """
        Set the epsilon and sigma for the atom to that of the reference state, for all other states.
        """
        epsilon = self._isomeric_states[reference_state_index].nonbondedtypes[atom.name].epsilon
        sigma = self._isomeric_states[reference_state_index].nonbondedtypes[atom.name].sigma
        log.info("Setting epsilon/sigma for {} to {} / {}".format(atom.name, epsilon, sigma))
        for isoindex in self._isomeric_states.keys():
            self._isomeric_states[isoindex].nonbondedtypes[atom.name].epsilon = epsilon
            self._isomeric_states[isoindex].nonbondedtypes[atom.name].sigma = sigma

    def _manual_lj_params(self, atom, typelist):
        """Interactive choice between LJ parameter types."""
        print("Atom types for {}".format(atom.name))
        for isoindex, atype in typelist.items():
            print("{}), {}".format(isoindex, atype))
        choice = -1
        while choice not in typelist.keys():
            print("Please pick a number from the list:")
            choice = int(input("Which sigma/epsilon would you like to keep?").strip())

        self._standardize_lj_params(atom, choice)

    def _unique_bonds(self):
        """Ensure only unique bonds are kept."""
        bonds = list()
        for bond in self._bonds:
            if bond not in bonds:
                bonds.append(bond)

        self._bonds = bonds

    def _unique_atom_types(self, autoresolve):
        """Ensure only unique atomtypes are kept. Provide conflict resolution."""
        atypes = list()
        for atype in self._atom_types:
            if atype not in atypes:
                atypes.append(atype)
        self._atom_types = atypes

        # Atomtypes might be different between isomers. We can only support one type at the moment
        while len(self._atom_types) != len(self._atoms):
            for atype1 in self._atom_types:
                conflicts = [atype1]
                for atype2 in self._atom_types:
                    if atype1 == atype2:
                        continue
                    elif atype1.name == atype2.name:
                        conflicts.append(atype2)

                    if len(conflicts) > 1:
                        self._resolve_type_conflict(conflicts, autoresolve=autoresolve)

    def _resolve_type_conflict(self, conflicts, autoresolve):
        """Keep all but one of the conflicting bonded atom types.

        Parameters
        ----------
        conflicts - list
            Conflicting atom types
        autoresolve - int
            one based index of the type to keep in each case.
            If not provided, interactive choice will be provided.

        Notes
        -----
        If the autoresolve index is out of bounds, will fallback to interactive mode.
        """
        if autoresolve:
            try:
                log.info("Using atomtype {}".format(conflicts[autoresolve-1]))
                conflicts.remove(conflicts[autoresolve-1])
                log.debug("Deleting atomtypes {}".format(conflicts))
                for conf in conflicts:
                    self._atom_types.remove(conf)
            except IndexError:
                log.warning("Automatic type resolution failure (out of bounds.) Switching to manual.")
                self._manual_type_resolution(conflicts)
        else:
            self._manual_type_resolution(conflicts)

    def _manual_type_resolution(self, conflicts):
        while True:
            for c, conf in enumerate(conflicts, start=1):
                print("Type {}) : {}".format(c,conf))
            keep = int(input("Which type do you want to keep? (1/2)").strip())
            if keep not in range(1, len(conflicts)+1):
                print("Please pick a number from the list.")
            else:
                log.info("Using atomtype {}".format(conflicts[keep-1]))
                conflicts.remove(conflicts[keep-1])
                log.debug("Deleting atomtypes {}".format(conflicts))
                for conf in conflicts:
                    self._atom_types.remove(conf)
                break

    def _sort_atoms(self):
        self._atoms = OrderedDict(sorted(self._atoms.items(), key=lambda t: t[1]))  # Sort by atom number

    def _sort_bonds(self):
        self._bonds = sorted(self._bonds)  # Sort by first, then second atom in bond

    def _sort_atypes(self):
        self._atom_types = sorted(self._atom_types, key=lambda at: at.number)


def _param_isomer(isomer, tmpdir, q, gb_params, shortname="LIG"):
    """
    Run a single ligand isomer through antechamber and tleap

    Parameters
    ----------
    isomer - int
        Isomer state index ( will be used to find filenames)
    tmpdir - str
        Working directory
    q - int
        Net charge of molecule
    shortname - str
        Short (3 character ) name for ligand to use in output files.
    """

    # TODO hardcoded using gasteiger charges for debugging speed
    molecule_name = 'isomer_{}'.format(isomer)
    omt.amber.run_antechamber(molecule_name, '{}/{}.mol2'.format(tmpdir, isomer), charge_method="gas", net_charge=q, resname=shortname)
    omt.utils.run_tleap(molecule_name, "{}.gaff.mol2".format(molecule_name), "{}.frcmod".format(molecule_name), prmtop_filename=None, inpcrd_filename=None, log_debug_output=False)
    if gb_params: _extract_gbparms(isomer, "{}.prmtop".format(molecule_name))
    logging.info("Parametrized isomer{}".format(isomer))


def _extract_gbparms(isomer, prmtop_filename=None):
    """
    Extract GB parameters from a prmtop file and return xml string containing them.

    Parameters
    ----------
    isomer - int
        Index of the isomer from which to fetch the prmtop file
    prmtop_filename - str
        override the prmtop filename

    Returns
    -------
    str - Name of xml file containing parameters


    """

    # Go for the standard filename
    if prmtop_filename is None:
        prmtop_filename = "isomer_{}.prmtop".format(isomer)

    prmtop = parmed.load_file(prmtop_filename)

    # TODO is there an official way to extract information from a prmtop using parmed?
    atoms = prmtop.parm_data['ATOM_NAME']
    # TODO do these match the other charges in the mol2 files?
    charges = prmtop.parm_data['CHARGE']
    radii = [ x / 10.0 for x in prmtop.parm_data['RADII'] ]  # nanometers for the openmm god
    scale = prmtop.parm_data['SCREEN']

    # Compile data into an xml format
    combined = zip(atoms, charges, radii, scale)
    template = '  <Atom type="isom-{0}-{1}" charge="{2}" radius="{3}" scale="{4}"/>\n'
    block = '<GBSAOBCForce id="{}">\n'.format(isomer)
    for atom in combined:
        block += template.format(isomer -1, *atom) # zero based isomer index for intermediate files
    block += '</GBSAOBCForce>\n'

    # Write xml to intermediate files
    xml_output = "isomer_{}.gbparams.xml".format(isomer)
    xml_parameters = open(xml_output, 'w')
    xml_parameters.write(block)
    xml_parameters.close()

    # Return the file name for convenience
    return xml_output


def line_prepender(filename, line):
    """Prepend a line to a file.
    http://stackoverflow.com/a/5917395
    """

    with open(filename, 'r+') as f:
        content = f.read() # current content of the file
        f.seek(0, 0)
        f.write(line.rstrip('\r\n') + '\n' + content) # add new line before the current content


def parametrize_ligand(inputmol2, outputffxml, max_antechambers=1, write_gb_params=True, tmpdir=None, remove_temp_files=False, pH=7.4, resname="LIG"):
    """
    Parametrize a ligand for constant-pH simulation using Epik.

    Parameters
    ----------
    inputmol2 : str
        location of mol2 file with all possible atoms included.
    outputffxml : str
        location for output xml file containing all ligand states and their parameters

    Other Parameters
    ----------------
    pH : float
        The pH that Epik should use
    max_antechambers : int, optional (default : 1)
        Maximal number of concurrent antechamber processes.
    write_gb_params : bool (default : True)
        If true, add GB params to the xml files.
    tmpdir : str, optional
        Temporary directory for storing intermediate files.
    remove_temp_files : bool, optional (default : True)
        Remove temporary files when done.

    resname : str, optional (default : "LIG")
        Residue name in output files.

    Notes
    -----
    The supplied mol2 file needs to have ALL possible atoms included, with unique names.
    This could be non-physical, also don't worry about bond order.
    If you're not sure, better to overprotonate. Epik doesn't retain the input protonation if it's non-physical.

    Todo
    ----

    * Currently hardcoded to use Gasteiger charges for debugging purposes.
        See  _param_isomer

    Returns
    -------
    str : The absolute path of the outputfile

    """
    log.setLevel(logging.DEBUG)
    log.info("Running Epik to detect protomers and tautomers...")
    inputmol2 = os.path.abspath(inputmol2)
    outputffxml = os.path.abspath(outputffxml)
    oldwd = os.getcwd()
    if tmpdir is None:
        tmpdir = tempfile.mkdtemp()
    os.chdir(tmpdir)
    log.debug("Running in {}".format(tmpdir))

    # Using very tolerant settings, since we can't predict probability from just ref states.
    omt.schrodinger.run_epik(inputmol2, "epik.mae", ph=pH, min_probability=0.00001, ph_tolerance=5.0)
    omt.schrodinger.run_structconvert("epik.mae", "epik.sdf")
    omt.schrodinger.run_structconvert("epik.mae", "epik.mol2")
    log.info("Done with Epik run!")

    # Grab data from sdf file and make a file containing the charge and penalty
    log.info("Processing Epik output...")
    epikxml = _make_xml_root("IsomerData")
    isomer_index = 0
    store = False
    charges = dict()
    for line in open('epik.sdf', 'r'):
        if store:
            value = line.strip()

            if store == "population":
                value = float(value)
                value /= 298.15 * 1.9872036e-3 # / RT in kcal/mol/K at 25 Celsius (Epik default)
                value = exp(-value)

            obj.set(store, str(value))

            # NOTE: relies on state penalty coming before charge
            if store == "netcharge":
                epikxml.append(obj)
                charges[isomer_index] = int(value)
                del(obj)

            store = ""

        elif "r_epik_State_Penalty" in line:
            # Next line contains epik state penalty
            store = "population"
            isomer_index += 1
            obj = objectify.Element("IsomericState")
            obj.set("index", str(isomer_index))

        elif "i_epik_Tot_Q" in line:
            # Next line contains charge
            store = "netcharge"

    # remove lxml annotation
    objectify.deannotate(epikxml)
    etree.cleanup_namespaces(epikxml)
    epikxmlstring = etree.tostring(epikxml, pretty_print=True, xml_declaration=False)

    # Make new mol2 file for each isomer
    with open('epik.mol2', 'r') as molfile:
        isomer_index = 0
        out = None
        for line in molfile:
            if line.strip() =='@<TRIPOS>MOLECULE':
                if out:
                    out.close()
                isomer_index += 1
                out = open('{}.mol2'.format(isomer_index), 'w')
            out.write(line)
        out.close()
    log.info("Done! Processed {} isomers.".format(isomer_index))
    log.info("Calculating isomer parameters {} processes at a time... (This may take a while!)".format(max_antechambers))
    Parallel(n_jobs=max_antechambers)(delayed(_param_isomer)(i, tmpdir, charges[i], write_gb_params) for i in range(1, isomer_index + 1))
    log.info("Done!")
    log.info("Combining isomers into one XML file.")

    omt.utils.create_ffxml_file(glob("isomer*.mol2"), glob("isomer*.frcmod"), ffxml_filename="intermediate.xml")

    # Look for the xml files containing gbparameters
    if write_gb_params:
        gb_xml_files = glob("isomer_*.gbparams.xml")
    else:
        gb_xml_files = []

    # Append epik information to the end of the file
    with open("intermediate.xml", mode='ab') as outfile:
        outfile.write(epikxmlstring)
        for gb_xml_file in gb_xml_files:
            gb_xml_contents = open(gb_xml_file, 'rb').read()
            outfile.write(gb_xml_contents)

        # Wrap in new root tag
        outfile.write(b"</IntermediateResidueTemplate>")
    line_prepender("intermediate.xml", "<IntermediateResidueTemplate>")

    _TitratableForceFieldCompiler("intermediate.xml", write_gb_params=write_gb_params).write(outputffxml, moleculename=resname)
    log.info("Done, your result is located here: {}!".format(outputffxml))
    if remove_temp_files:
        shutil.rmtree(tmpdir)
    os.chdir(oldwd)
    return outputffxml
