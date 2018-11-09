# coding=utf-8
"""
Drivers for Monte Carlo sampling of chemical states, such as tautomers and protomers.
"""
import copy
import logging
import math
import random
from pandas import DataFrame
import pandas as pd
from pandas.util.testing import assert_frame_equal
import sys
import numpy as np
import os
from simtk import unit
from simtk import openmm as mm
import saltswap
from saltswap.swapper import Swapper
from .proposals import _StateProposal, SaltSwapProposal, OneDirectionChargeProposal
from .topology import Topology
from .pka import available_pkas
from simtk.openmm import app
from numbers import Number
import re
from collections import defaultdict
from .logger import log
from abc import ABCMeta, abstractmethod
from lxml import etree, objectify
from .integrators import GHMCIntegrator, GBAOABIntegrator

kB = (1.0 * unit.BOLTZMANN_CONSTANT_kB * unit.AVOGADRO_CONSTANT_NA).in_units_of(unit.kilocalorie_per_mole  / unit.kelvin)
np.set_printoptions(precision=15)


class _TitratableResidue:
    """Representation of a single residue with multiple titration states."""

    def __init__(self):
        """
        Instantiate a _TitratableResidue

        Notes
        -----
        This class should not be instantiated directly. Use `from_lists` or `from_serialized_xml` instead.

        """
        # The indices of the residue atoms in the system
        self.atom_indices = list()
        # List to store titration states
        self.titration_states = list()
        self.index = None
        self.name = None
        self.residue_type = None
        # NonbondedForce exceptions associated with this titration state
        self.exception_indices = list()
        self._state = None
        self._pka_data = None
        self._residue_pka = None

        return

    def __eq__(self, other):

        for own_state, other_state in zip(self.titration_states, other.titration_states):
            if own_state != other_state:
                return False

        if self.name != other.name:
            return False

        if self.residue_type != other.residue_type:
            return False

        if self.index != other.index:
            return False

        if self._state != other._state:
            return False

        if self._pka_data is not None and  other._pka_data is not None:
            try:
                assert_frame_equal(self._pka_data, other._pka_data)
            except AssertionError:
                return False

        if self._residue_pka != other._residue_pka:
            return False

        return True


    @classmethod
    def from_lists(cls, atom_indices, group_index, name, residue_type, exception_indices, pka_data=None, residue_pka=None):
        """
        Instantiate a _TitratableResidue from lists and strings that contain all necessary information

        Parameters
        ----------
        atom_indices - list of system indices of the residue atoms
        group_index - the index of the residue in the list of titratable residues
        name - str, an identifier for this residue
        residue_type - str, the 3 letter residue type in the forcefield specification (e.g. AS4).
        exception_indices - list of NonbondedForce exceptions associated with this titration state
        pka_data - dict, optional, dict of weights, with pH as key. floats as keys. Not compatible with residue_pka option.
        residue_pka - PopulationCalculator, optional. Can be used to provide target weights at a given pH. Not compatible with pka_data option.
        """
        # The indices of the residue atoms in the system
        obj = cls()

        obj.atom_indices = list(atom_indices)  # deep copy
        # List to store titration states
        obj.titration_states = list()
        obj.index = group_index
        obj.name = name
        obj.residue_type = residue_type
        # NonbondedForce exceptions associated with this titration state
        obj.exception_indices = exception_indices
        obj._state = None
        obj._pka_data = None
        obj._residue_pka = None

        if pka_data is not None and residue_pka is not None:
            raise ValueError("You can only provide pka_data, or residue_pka, not both.")
        elif pka_data is not None:
            obj._pka_data = pka_data

        elif residue_pka is not None:
            obj._residue_pka = residue_pka

        return obj

    @classmethod
    def from_serialized_xml(cls, xmltree):
        """Create a titratable residue from a serialized titratable residue.

        Parameters
        ----------
        xmltree - etree.ElementTree or compatible lxml class, should only contain one residue.

        Returns
        -------
        obj - a newly instantiated _TitratableResidue object.

        """

        # prevent accidental modification of the user supplied file.
        xmltree = copy.deepcopy(xmltree)
        obj = cls()

        # The indices of the residue atoms in the system
        atom_indices = list()

        res = xmltree.xpath('/TitratableResidue')[0]
        for atom in xmltree.xpath('/TitratableResidue/atom'):
            atom_indices.append(int(atom.get('index')))
        obj.atom_indices = atom_indices

        # List to store titration states
        obj.titration_states = list()
        obj.index = int(res.get('index'))
        obj.name = str(res.get('name'))

        obj.residue_type = str(res.get('type'))
        # NonbondedForce exceptions associated with this titration state
        exception_indices = list()
        for exception in xmltree.xpath('/TitratableResidue/exception'):
            exception_indices.append(int(exception.get('index')))

        obj.exception_indices = exception_indices
        obj._state = None
        obj._pka_data = None
        obj._residue_pka = None

        # parse the pka data block as if an html table
        pka_data = xmltree.xpath('/TitratableResidue/pka_data')
        if len(pka_data):
            pka_data = copy.deepcopy(pka_data[0])
            pka_data.tag = 'table'
            obj._pka_data = pd.read_html(etree.tostring(pka_data))[0]

        res_pka = res.get("residue_pka")
        if res_pka is not None:
            obj._residue_pka = available_pkas[res_pka]

        if obj._pka_data is not None and obj._residue_pka is not None:
            raise ValueError("You can only provide pka_data, or residue_pka, not both.")

        states = xmltree.xpath('/TitratableResidue/TitrationState')
        obj.titration_states = [None] * len(states)

        for state in states:
            state_index = int(state.get('index'))
            obj.titration_states[state_index] = _TitrationState.from_serialized_xml(state)

        # Set the titration state of this residue
        obj.state = int(res.get('state'))

        return obj

    def add_state(self, state):
        """Adds a _TitrationState to the residue."""
        self.titration_states.append(state)

    def serialize(self):
        """
        Create an xml representation of this residue.

        Returns
        -------
        res - lxml tree containing residue information

        """
        # xml factory
        E = objectify.E
        res = E.TitratableResidue(
            name=self.name,
            type=self.residue_type,
            index=str(self.index),
            state=str(self.state_index))

        if self._residue_pka is not None:
            # residue_pka holds a reference to the base class.
            # Storing the name of the type, which can be looked to find it from the available_pkas dict
            res.set('residue_pka', self.residue_type)

        for atom_index in self.atom_indices:
            objectify.SubElement(res, 'atom', index=str(atom_index))

        for exception_index in self.exception_indices:
            objectify.SubElement(res, 'exception', index=str(exception_index))

        if self._pka_data is not None:
            res.pka_data = objectify.fromstring(self._pka_data.to_html(index=False))


        res.TitrationState = E.TitrationState()
        res.TitrationState[:] = [state.serialize(index) for index,state in enumerate(self.titration_states)][:]

        return res

    def get_populations(self, pH, temperature=None, ionic_strength=None, strict=True):
        """Return the state populations for a given pH.

        Parameters
        ----------
        pH - float, the pH for which populations should be returned
        temperature - float, the temperature in Kelvin that should be used  to find the log populations if available.
            Optional, soft requirement. Won't throw error if not matched.
        ionic_strength - float, the ionic strength in millimolar that should be used to find the log populations if available.
            Optional, soft requirement. Won't throw error if not matched.
        strict - bool, default True. If there are no pH dependent weights, throw an error. Else, just return default weights.

        Notes
        -----
        Temperature, and ionic strength are soft requirements.

        """
        log_weights = np.empty(len(self))
        # look up weights in the dictionary
        if self._pka_data is not None:
            # Search an appropriate log population value from the dataframe that was constructed.
            # Temperature and Ionic strength aren't always provided.
            for state, group in self._pka_data.groupby("State"):
                # Get the first element where the pH matches, the temperature potentially matches, and the ionic strenght potentially matches
                state = int(state)
                pH_match = (group['pH'] == pH)
                temperature_match = True
                ionic_strength_match = True
                if temperature is not None:
                    temperature_match = (group['Temperature (K)'].isin([temperature, None]))

                if ionic_strength is not None:
                    ionic_strength_match = (group['Ionic strength (mM)'].isin(ionic_strength, None))

                matches = group.loc[pH_match & temperature_match & ionic_strength_match]

                # If there are no matches, throw an error
                if len(matches) == 0:
                    raise ValueError("There is no matching pH/temperature condition available for residue {}.".format(self.name))
                # get the first one
                else:
                    first_row = next(matches.iterrows())
                    # index 1 is the row values, get the log population
                    log_population = first_row[1]['log population']

                log_weights[state] = log_population

        # calculate residue weights from pka object
        elif self._residue_pka is not None:
            log_weights = self._residue_pka(pH).populations()

        # If there is no pH dependent population specified, return the current target populations.
        # This will be equal if this was never specified previously. See the target_weights property.
        else:
            if strict:
                raise RuntimeError("Residue is not adjustable by pH. {}".format(self.name))
            else:
                log_weights = np.log(np.asarray(self.target_weights))

        return np.asarray(log_weights)

    def set_populations(self, pH):
        """
        Set the target weights using the pH

        Parameters
        ----------
        pH - float, the pH of the simulation.
        """
        # old_weights = np.asarray(self.target_weights)
        self.target_weights = np.exp(self.get_populations(pH, strict=True))
        ph_correction = -np.log(self.target_weights)
        self.g_k_values = np.asarray(self.g_k_values) + ph_correction

    @property
    def state(self):
        """
        Returns
        -------
        _TitrationState
        """
        return self.titration_states[self._state]

    @property
    def state_index(self):
        """
        The index of the current state of the residue.
        """
        return self._state

    @state.setter
    def state(self, state):
        """
        state - int
        """

        if state > len(self):
            raise IndexError("Titration state index out of bounds. ( > {}".format(len(self)))
        self._state = state

    @property
    def target_weights(self):
        """Target weight of each state. Default is equal weights."""
        target_weights = [state.target_weight for state in self.titration_states]
        if None in target_weights:
            return [1.0 / len(self)] * len(self)
        else:
            return target_weights

    @target_weights.setter
    def target_weights(self, weights):
        """Set sampling target weights for all states."""
        if not len(weights) == len(self):
            raise ValueError("The number of weights needs to be equal to the number of states.")

        for id, state in enumerate(self):
            state.target_weight = weights[id]

    @property
    def g_k_values(self):
        """A list containing the g_k value for each state."""
        return [state.g_k for state in self]

    @g_k_values.setter
    def g_k_values(self, g_klist):
        """Set sampling target weights for all states."""
        if not len(g_klist) == len(self):
            raise ValueError("The number of g_k values needs to be equal to the number of states.")

        for id, state in enumerate(self):
            state.g_k = g_klist[id]

    @property
    def proton_count(self):
        """Number of titratable protons in current state."""
        return self.state.proton_count

    @property
    def proton_counts(self):
        """Number of titratable protons active in each state."""
        return [state.proton_count for state in self]

    def __len__(self):
        """Return length of group."""
        return len(self.titration_states)

    def __getitem__(self, item):
        """Retrieve state by index.
        Parameters
        ----------

        item - int
            Titration state to be accessed.
        """
        if item >= len(self.titration_states):
            raise IndexError("Titration state outside of range.")
        else:
            return self.titration_states[item]

    @property
    def atom_status(self):
        """Returns boolean array of atoms, and if they're switched on.
        Defined as charge equal to 0 (to precision of 1.e-9
        """
        return [0 if abs(charge) < 1.e-9 else 1 for charge in self.state.charges]

    @property
    def total_charge(self) -> int:
        """Total charge of the current titration state."""
        return self.state.total_charge

    @property
    def total_charges(self):
        """Total charge of each state."""
        return [state.total_charge for state in self]


class _TitrationState:
    """Representation of a titration state"""

    def __init__(self):
        """Instantiate a _TitrationState"""

        self.g_k = None  # dimensionless quantity
        self.charges = list()
        self.proton_count = None
        self._forces = list()
        self._target_weight = None
        self.epsilon = list()
        self.sigma = list()
        self.bonded_par = None

    @classmethod
    def from_lists(cls, g_k, atom_charges, atom_epsilon, atom_sigma, atom_type, atom_name, bonded_par, angle_par, proper_par, improper_par, proton_count):
        """Instantiate a _TitrationState from g_k, proton count and a list of the charges

        Returns
        -------
        obj - a new _TitrationState instance
        """
        obj = cls()
        obj.g_k = g_k  # dimensionless quantity
        obj.charges = copy.deepcopy(atom_charges)
        obj.epsilon = copy.deepcopy(atom_epsilon)
        obj.sigma = copy.deepcopy(atom_sigma)
        obj.bonded_par = copy.deepcopy(bonded_par)
        obj.atom_type = copy.deepcopy(atom_type)
        obj.atom_name = copy.deepcopy(atom_name)
        obj.angle_par = copy.deepcopy(angle_par)
        obj.proper_par = copy.deepcopy(proper_par)
        obj.improper_par = copy.deepcopy(improper_par)
        obj.proton_count = proton_count
        # Note that forces are to be manually added by force caching functionality in ProtonDrives
        obj._forces = list()
        obj._target_weight = None
        return obj

    @classmethod
    def from_serialized_xml(cls, state_element):
        """
        Deserialize a _TitrationState from a previously serialized xml tree

        Parameters
        ----------
        xmltree - etree.Element or compatible lxml class containing one single titration state

        Returns
        -------
        obj - a new _TitrationState instance
        """

        obj = cls()

        # prevent accidental modification
        state = copy.deepcopy(state_element)
        obj.proton_count = int(state.get('proton_count'))
        obj._target_weight = np.float64(state.get('target_weight'))
        obj.g_k = np.float64(state.get('g_k'))

        charges = state.xpath('charge')
        obj.charges = [None] * len(charges)
        for charge in charges:
            # Get the array index
            charge_index = int(charge.get('charge_index'))
            charge_value = np.float64(charge.text)
            obj.charges[charge_index] = charge_value

        # forces is a list of forces, though currently in practice its of length one and contains only nonbonded force
        # TODO implement GBSA forces or custom forces here
        # Inside each force is a dict containing 'atoms', and 'exceptions'
        # 'atoms' and 'exceptions' are lists
        # Inside of the list are dicts.
        # Each dictionary contains the parameters for either an atom, or an exception.
        # For atom it contains 'charge', 'sigma', 'epsilon', and 'atom_index'.
        # For exception it contains  'exception_index' 'particle1' 'particle2' 'chargeProd' 'sigma', and 'epsilon'
        forces = state.xpath('force')
        obj._forces = [None] * len(forces)
        for f_index, force in enumerate(forces):
            force_dict = dict(atoms=list(), exceptions=list())

            for atom in force.xpath('atom'):
                atom_dict = dict()
                for key in ['atom_index', 'charge', 'epsilon', 'sigma']:
                    if key == 'atom_index':
                        atom_dict[key] = int(atom.get(key))
                    else:
                        atom_dict[key] = np.float64(atom.get(key))
                force_dict['atoms'].append(atom_dict)

            for exception in force.xpath('exception'):
                exc_dict = dict()
                for key in ["chargeProd", "epsilon", "exception_index", "particle1", "particle2" , "sigma"]:
                    if key in ["particle1", "particle2", "exception_index"]:
                        exc_dict[key] = int(exception.get(key))
                    else:
                        exc_dict[key] = np.float64(exception.get(key))
                force_dict['exceptions'].append(exc_dict)
            obj._forces[f_index] = force_dict

        return obj

    @property
    def total_charge(self) -> int:
        """Return the total charge of the state."""
        return int(round(sum(self.charges)))

    @property
    def forces(self):
        return self._forces

    @forces.setter
    def forces(self, force_params):
        self._forces = copy.deepcopy(force_params)

    @property
    def target_weight(self):
        return self._target_weight

    @target_weight.setter
    def target_weight(self, weight):
        self._target_weight = weight

    def serialize(self, index=None):
        """Serialize a state into xml etree.

        Returns
        -------
        state - objectify tree
        """
        E = objectify.E
        if index is not None:
            index = str(index)
        # Only serializing values that are not properties.
        state = E.TitrationState(proton_count=str(self.proton_count), target_weight=str(self.target_weight), index=index, g_k=str(self.g_k))

        q_tags = list()
        for q_index, q in enumerate(self.charges):
            # Ensure float is numpy type for print precision as specified in numpy print options
            q = np.float64(q)
            q_tags.append(E.charge("{:.15f}".format(q), charge_index=str(q_index)))

        state.charge = E.charge
        state.charge[:] = q_tags[:]

        # forces is a list of forces, though currently in practice its of length one and contains only nonbonded force
        # Other forces will get serialized correctly, but deserialization may be an issue.
        # Inside each force is a dict containing 'atoms', and 'exceptions'
        # 'atoms' and 'exceptions' are lists
        # Inside of the list are dicts.
        # Each dictionary contains the parameters for either an atom, or an exception.
        # For atom it contains 'charge', 'sigma', 'epsilon', and 'atom_index'.
        # For exception it contains  'exception_index' 'particle1' 'particle2' 'chargeProd' 'sigma', and 'epsilon'
        for f_index, force in enumerate(self._forces):
            force_xml = E.force(index=str(f_index), # the force index in the internal state, not the force index in openmm
                                )
            atoms = force['atoms']
            exceptions = force['exceptions']
            for atom in atoms:
                # Convert to string for xml storage
                atom_strings = dict(atom)
                for key in atom.keys():
                    if key == 'atom_index':
                        atom_strings[key] = str(atom[key])
                    else:
                        # Ensure numpy type for print precision
                        atom_strings[key] = "{:.15f}".format(np.float64(atom[key]))
                atom_tag = objectify.SubElement(force_xml, 'atom', **atom_strings)
            for exception in exceptions:
                exception_strings = dict(exception)
                for key in exception.keys():
                    if key in ["particle1", "particle2", "exception_index"]:
                        exception_strings[key] = str(exception[key])
                    else:
                        # Ensure numpy type for print precision
                        exception_strings[key] = "{:.15f}".format(np.float64(exception[key]))
                exception_tag = objectify.SubElement(force_xml, 'exception', **exception_strings)

            state.force = force_xml

        return state

    def __eq__(self, other):
        """Compare the equality of two _TitrationState objects."""
        if not isinstance(other, _TitrationState):
            return False

        float_atol = 1.e-10
        if not np.isclose(self._target_weight, other._target_weight, rtol=0.0, atol=float_atol):
            return False

        if not np.isclose(self.proton_count, other.proton_count, rtol=0.0, atol=float_atol):
            return False

        if not np.isclose(self.g_k, other.g_k, rtol=0.0, atol=float_atol):
            return False

        if len(self.charges) != len(other.charges):
            return False

        # Check if all stored charges are equal
        if not np.all(np.isclose(self.charges, other.charges, atol=float_atol, rtol=0.0)):
            return False

        # check if all force parameters are equal
        for own_force, other_force in zip(self._forces, other._forces):
            own_atoms, other_atoms = own_force['atoms'], other_force['atoms']
            own_exceptions, other_exceptions = own_force['exceptions'], other_force['exceptions']

            for own_atom, other_atom in zip(own_atoms, other_atoms):
                for key in own_atom.keys():
                    if not np.isclose(own_atom[key], other_atom[key], rtol=0.0, atol=float_atol):
                        return False

            for own_exception, other_exception in zip(own_exceptions, other_exceptions):
                for key in own_exception.keys():
                    if not np.isclose(own_exception[key], other_exception[key], rtol=0.0, atol=float_atol):
                        return False

        # Everything that was checked seems equal.
        return True


class _TitrationAttemptData(object):
    """Private class for bookkeeping information regarding a single titration state update."""

    def __init__(self):
        """Set up all internal variables for tracking."""

        self._accepted = None
        self._logp_ratio_residue_proposal = None
        self._logp_ratio_salt_proposal = None
        self._logp_accept = None
        self._work = None
   
        self._initial_charge = None
        self._initial_states = None
        self._initial_ion_states = None

        self._proposed_charge = None
        self._proposed_states = None
        self._proposed_ion_states = None

        return

    @property
    def initial_charge(self) -> int:
        """Initial charge of titratable residues."""
        return int(self._initial_charge)

    @initial_charge.setter
    def initial_charge(self, initial_charge: int):
        """Initial charge of titratable residues."""
        self._initial_charge = initial_charge

    @property
    def proposed_charge(self) -> int:
        """proposed charge of titratable residues."""
        return int(self._proposed_charge)
    
    @proposed_charge.setter
    def proposed_charge(self, proposed_charge: int):
        """proposed charge of titratable residues."""
        self._proposed_charge = proposed_charge

    @property
    def accepted(self) -> bool:
        """True if the proposal was accepted, false if rejected."""
        return self._accepted

    @accepted.setter
    def accepted(self, accepted:bool):
        """True if proposal was accepted, false if rejected."""
        self._accepted = accepted

    @property
    def rejected(self) -> bool:
        """True if the proposal was rejected, false if accepted."""
        return not self._accepted

    @rejected.setter
    def rejected(self, rejected: bool):
        """True if the proposal was rejected false if accepted."""
        self._accepted = not rejected

    @property
    def initial_states(self) -> np.ndarray:
        """The titration state at the start of the attempt."""
        return self._initial_states

    @initial_states.setter
    def initial_states(self, initial_states: np.ndarray):
        """The titration state at the start of the attempt."""
        self._initial_states = np.asarray(initial_states)

    @property
    def proposed_states(self) -> np.ndarray:
        """The titration state at the end of the attempt."""
        return self._proposed_states

    @proposed_states.setter
    def proposed_states(self, proposed_states: np.ndarray):
        """
        The titration state at the end of the attempt.
        """
        self._proposed_states = np.asarray(proposed_states)

    @property
    def work(self) -> np.float64:
        """The total work performed during the attempt."""
        return self._work
    
    @work.setter
    def work(self,work: np.float64):
        """The total work performed during the attempt."""
        self._work = work

    @property
    def logp_ratio_residue_proposal(self) -> np.float64:
        """The reverse/forward ratio of the probability of picking the residue,
         and its state."""
        return self._logp_ratio_residue_proposal

    @logp_ratio_residue_proposal.setter
    def logp_ratio_residue_proposal(self, logp_ratio_residue_proposal: np.float64):
        """The reverse/forward ratio of the probability of picking the residue,
         and its state."""
        self._logp_ratio_residue_proposal = logp_ratio_residue_proposal

    @property
    def logp_ratio_salt_proposal(self) -> np.float64:
        """The reverse/forward ratio of the probability of picking a water 
        molecule, and its ionic state."""
        return self._logp_ratio_salt_proposal

    @logp_ratio_salt_proposal.setter
    def logp_ratio_salt_proposal(self, logp_ratio_salt_proposal: np.float64):
        """The reverse/forward ratio of the probability of picking a water 
        molecule, and its ionic state."""
        self._logp_ratio_salt_proposal = logp_ratio_salt_proposal

    @property
    def logp_accept(self) -> np.float64:
        """The acceptance probability of the entire proposal."""
        return self._logp_accept

    @logp_accept.setter
    def logp_accept(self, logp_accept: np.float64):
        """The acceptance probability of the entire proposal."""
        self._logp_accept = logp_accept

    @property
    def initial_ion_states(self) -> np.ndarray:
        """The initial state of water molecules treated by saltswap."""
        return self._initial_ion_states

    @initial_ion_states.setter
    def initial_ion_states(self, initial_ion_states: np.ndarray):
        """The initial state of water molecules treated by saltswap."""
        self._initial_ion_states = np.asarray(initial_ion_states)

    @property
    def proposed_ion_states(self) -> np.ndarray:
        """The proposed state of water molecules treated by saltswap."""
        return self._proposed_ion_states

    @proposed_ion_states.setter
    def proposed_ion_states(self, proposed_ion_states: np.ndarray):
        """The proposed state of water molecules treated by saltswap."""
        self._proposed_ion_states = np.asarray(proposed_ion_states)


class _BaseDrive(metaclass=ABCMeta):
    """An abstract base class describing the common public interface of Drive-type classes

    .. note::

        Examples of a Drive class would include the NCMCProtonDrive, which has instantaneous MC, and NCMC updates of
        protonation states of the system in its ``update`` method, and provides tracking tools, and calibration tools for
        the relative weights of the protonation states.
    """

    @abstractmethod
    def update(self, proposal):
        """
        Update the state of the system using some kind of Monte Carlo move
        """
        pass

    @abstractmethod
    def import_gk_values(self, gk_dict):
        """
        Import the relative weights, gk, of the different states of the residues that are part of the system

        Parameters
        ----------
        gk_dict : dict
            dict of starting value g_k estimates in numpy arrays, with residue names as keys.
        """
        pass

    @abstractmethod
    def reset_statistics(self):
        """
        Reset statistics of titration state tracking.
        """
        pass

    @abstractmethod
    def attach_context(self,context):
        """
        Attach a context containing a compoundintegrator for use with NCMC
        Parameters
        ----------
        context - simtk.openmm.Context
        """

        pass

    @abstractmethod
    def define_pools(self, dict_of_pools):
        """
        Defines a dictionary of indices that describe different parts of the simulation system,
        such as 'protein' or 'ligand'.

        Parameters
        ----------
        dict_of_pools - dict of lists
        """
        pass

    @abstractmethod
    def adjust_to_ph(self, pH):
        """
        Apply the pH target weight correction for the specified pH.

        For amino acids, the pKa values in app.pkas will be used to calculate the correction.
        For ligands, target populations at the specified pH need to be available, otherwise an exception will be thrown.

        Parameters
        ----------
        pH - float, the pH to which target populations should be adjusted.

        Raises
        ------
        ValueError - if the target weight for a given pH is not supplied.
        """
        pass

    @abstractmethod
    def attach_swapper(self, swapper: Swapper, proposal=None):
        """Attach a saltswap.swapper object that is used for maintaining total charges.

        The `swapper` will be used for bookkeeping of solvent/buffer ions in the system. In order to
         maintain the charge neutrality of the system, the swapper is used to randomly take water molecules, and
         change it into an anion or a cation.

        It should take an optional argument called `proposal` that determines how ions are selected.

        """

        pass


class NCMCProtonDrive(_BaseDrive):
    """
    The NCMCProtonDrive is a base class Monte Carlo driver for protonation state changes and tautomerism in OpenMM.

    Protonation state changes, and additionally, tautomers are treated using the constant-pH dynamics method of Mongan, Case and McCammon [Mongan2004]_, or Stern [Stern2007]_ and NCMC methods from Nilmeier [Nilmeier2011]_.

    References
    ----------

    .. [Mongan2004] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born implicit solvent. J Comput Chem 25:2038, 2004.
        http://dx.doi.org/10.1002/jcc.20139

    .. [Stern2007] Stern HA. Molecular simulation with variable protonation states at constant pH. JCP 126:164112, 2007.
        http://link.aip.org/link/doi/10.1063/1.2731781

    .. [Nilmeier2011] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. PNAS 108:E1009, 2011.
        http://dx.doi.org/10.1073/pnas.1106094108

    .. todo::

      * Add alternative proposal types, including schemes that avoid proposing self-transitions (or always accept them):
        - Parallel Monte Carlo schemes: Compute N proposals at once, and pick using Gibbs sampling or Metropolized Gibbs?
      * Add automatic tuning of switching times for optimal acceptance.
    """

    def __init__(self, temperature, topology, system, pressure=None, perturbations_per_trial=0, propagations_per_step=1):
        """
        Initialize a Monte Carlo titration driver for simulation of protonation states and tautomers.

        Parameters
        ----------
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature at which the system is to be simulated.
        topology : protons.app.Topology
            OpenMM object containing the topology of system
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        pressure : simtk.unit.Quantity compatible with atmospheres, optional
            For explicit solvent simulations, the pressure.
        perturbations_per_trial : int, optional, default=0
            Number of perturbation steps per NCMC switching trial, or 0 if instantaneous Monte Carlo is to be used.
        propagations_per_step : int, optional, default=1
            Number of propagation steps in between perturbation steps.
        """
        # Store parameters.
        self.system = system
        self.temperature = temperature
        kT = kB * temperature  # thermal energy
        self.beta = 1.0 / kT  # inverse temperature
        # For more efficient calculation of the work (in multiples of KT) during NCMC
        self.beta_unitless = strip_in_unit_system(self.beta)
        self.pressure = pressure
        self._attempt_number = 0 # Internal tracker for current iteration attempt
        self.perturbations_per_trial = perturbations_per_trial
        # Keeps track of the last ncmc protocol attempt work.
        self.ncmc_stats_per_step = [None] * perturbations_per_trial
        self.propagations_per_step = propagations_per_step
        self._last_attempt_data = _TitrationAttemptData()
        self.nattempted = 0
        self.naccepted = 0
        self.nrejected = 0
        self.topology = topology

        # Sets of residues that are pooled together to sample exclusively from them
        self.residue_pools = dict()

        # The compound integrator used for simulation
        # Needs to be added after instantiation using ``attach_context``
        self.compound_integrator = None
        self.ncmc_integrator = None
        self.context = None

        # A salt swap swapper can later be attached to enable counterion coupling to protonation state changes
        # Using the `attach_swapper` method
        self.swapper = None
        # The total excess charge from ions, applied as counter-charge to protonation state changes.
        # Positive indicates the amount of cations that have been added to the system
        # Negative indicates the amount of anions that have been added to the system
        # The drive should never add cations and anions at the same time.
        self.excess_ions = 0

        # A dict of ion parameters, indexed by integers. Set from the swapper in attach_swapper.
        self._ion_parameters = None

        # The method used to select ions. Should be a subclass of SaltSwapProposal
        # This variable is set using the `attach_swapper` method.
        self.swap_proposal = None

        # Record the forces that need to be switched off for NCMC
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in
                  range(system.getNumForces())}

        # Control center mass remover
        if 'CMMotionRemover' in forces:
            self.cm_remover = forces['CMMotionRemover']
            self.cm_remover_freq = self.cm_remover.getFrequency()
        else:
            self.cm_remover = None
            self.cm_remover_freq = None

        # Check that system has MonteCarloBarostat if pressure is specified
        if pressure is not None:
            if 'MonteCarloBarostat' not in forces:
                raise Exception("`pressure` is specified, but `system` object lacks a `MonteCarloBarostat`")

        # Initialize titration group records.
        self.titrationGroups = list()

        # Keep track of forces and whether they've been cached.
        self.precached_forces = False

        # Determine 14 Coulomb and Lennard-Jones scaling from system.
        self.coulomb14scale = self._get14scaling(system)

        # Store list of exceptions that may need to be modified.
        self.atomExceptions = [list() for index in range(topology.getNumAtoms())]
        self._set14exceptions(system)

        # Store force object pointers.
        # TODO: Add Custom forces.
        force_classes_to_update = ['NonbondedForce', 'GBSAOBCForce', 'HarmonicBondForce', 'HarmonicAngleForce', 'PeriodicTorsionForce']
        #force_classes_to_update = ['NonbondedForce', 'HarmonicBondForce']
        #force_classes_to_update = ['NonbondedForce']
        self.forces_to_update = list()
        for force_index in range(self.system.getNumForces()):
            force = self.system.getForce(force_index)
            if force.__class__.__name__ in force_classes_to_update:
                self.forces_to_update.append(force)
            

        return

    def serialize_titration_groups(self):
        """Store residues handled by the drive as xml.

        Returns
        -------
        str - xml representation of the residues inside of the drive.
        """
        xmltree = etree.Element("NCMCProtonDrive")
        for res in self.titrationGroups:
            xmltree.append(res.serialize())

        return etree.tostring(xmltree, encoding="utf-8", pretty_print=True)

    def add_residues_from_serialized_xml(self, xmltree):
        """Add residues from previously serialized residues."""
        if type(xmltree) == str:
            xmltree = etree.fromstring(xmltree)
        drive_xml = xmltree.xpath("/NCMCProtonDrive")[0]
        for res in drive_xml.xpath("TitratableResidue"):
            self.titrationGroups.append(_TitratableResidue.from_serialized_xml(res))

    @property
    def titrationStates(self):
        return [group.state_index for group in self.titrationGroups]

    def attach_context(self, context):
        """Attaches a context to the Drive. The Drive requires a context with an NCMC integrator to be attached before it is functional.

        Parameters
        ----------
        context : simtk.openmm.Context
            Context that has a compound integrator bound to it. The integrator with index 1 is used for NCMC.

            The NCMC integrator needs to be a CustomIntegrator with the following two properties defined:
            first_step: 0 or 1. 0 indicates the first step in an NCMC protocol and can be used for special actions
                required such as computing the energy prior to perturbation.
                protocol_work: double, the protocol work performed by external moves in between steps.

        Returns
        -------

        """

        self.compound_integrator = context._integrator
        self.context = context

        # Check compatibility of integrator.
        if not isinstance(self.compound_integrator, mm.CompoundIntegrator):
            raise ValueError("The integrator provided is not a CompoundIntegrator.")
        try:
            self.ncmc_integrator = self.compound_integrator.getIntegrator(1)
        except IndexError:
            raise IndexError("Could not find a second integrator for use in NCMC.")

        # Check the attributes of the NCMC integrator
        try:
            self.ncmc_integrator.getGlobalVariableByName('protocol_work')
        except:
            raise ValueError("The NCMC integrator does not have a 'protocol_work' attribute.")

        try:
            self.ncmc_integrator.getGlobalVariableByName('first_step')
        except:
            raise ValueError("The NCMC integrator does not have a 'first_step' attribute.")

        for force_index, force in enumerate(self.forces_to_update):
            force.updateParametersInContext(self.context)

    def attach_swapper(self, swapper: Swapper, proposal: SaltSwapProposal=None):
        """
        Provide a saltswapper to enable maintaining charge neutrality.

        Parameters
        ----------
        swapper - a saltswap.Swapper object that is used for ion manipulation and bookkeeping.
        proposal - optional, a SaltSwapProposal derived class that is used to select ions. If not provided it uses
        the OneDirectionChargeProposal

        """
        if not isinstance(swapper, Swapper):
            raise TypeError("Please provide a Swapper object.")

        self.swapper = swapper

        nwat, ncat, nani = swapper.get_identity_counts()

        # The excess amount of ions, positive if cations, negative if more anions
        self.excess_ions = ncat - nani

        self._ion_parameters = {0: self.swapper.water_parameters,
                                1: self.swapper.cation_parameters,
                                2: self.swapper.anion_parameters
                                }

        if proposal is not None:
            self.swap_proposal = proposal
        else:
            self.swap_proposal = OneDirectionChargeProposal()
        return

    def define_pools(self, dict_of_pools):
        """
        Specify named residue_pools/subgroups of residues that can be sampled from separately.

        For instance, it might be useful to separate the protein from the ligand so you can sample the 
        protonation state of one component of the system at a time. 

        Note that the indices are dependent on self.titrationGroups, not a residue index in the PDB or in
        OpenMM topology. 

        Parameters
        ----------

        dict_of_pools : dict of list of int
            Provide a dictionary with named groups of residue indices.

        Examples
        --------       

        residue_pools = dict{protein=list(range(34)),ligand=[34])

        """

        # TODO alter residue specification by openmm topology index?

        # Validate user input
        if not (isinstance(dict_of_pools, dict)):
            raise TypeError("Please provide a dict of the different residue_pools.")

        # Make sure residues exist
        for group, indices in dict_of_pools.items():

            if not(isinstance(indices, list) or isinstance(indices, np.ndarray)):
                raise ValueError("Indices must be supplied as list or ndarrays.")

            if not all(index < len(self.titrationGroups) for index in indices):
                raise ValueError("Residue in {} specified is outside of range.".format(group))

        self.residue_pools = dict_of_pools

    def update(self, proposal, residue_pool=None, nattempts=1):
        """
        Perform a number of Monte Carlo update trials for the system protonation/tautomer states of multiple residues.

        Parameters
        ----------
        proposal : _StateProposal derived class
            Defines how to select residues for updating

        residue_pool : str
            The set of titration group incides to propose from. Groups can be defined using self.define_pools.
            If None, select from all titration groups uniformly.

        nattempts: int, optional
            Number of individual attempts per update.

        Notes
        -----
        The titration state actually present in the given context is not checked; it is assumed the ProtonDrive internal state is correct.

        """

        if not issubclass(type(proposal), _StateProposal):
            raise ValueError("Move needs to be a _StateProposal derived class.")

        if self.context is None:
            raise RuntimeError("Driver has no context attached.")

        # Perform a number of protonation state update trials.
        for attempt in range(nattempts):
            self._attempt_number = attempt
            accepted = self._attempt_state_change(proposal, residue_pool=residue_pool)

        return accepted

    def import_gk_values(self, gk_dict, strict=False):
        """Import precalibrated gk values. Only use this if your simulation settings are exactly the same.

        If you changed any details, rerun calibrate instead!

        Parameters
        ----------
        gk_dict : dict
            dict of starting value g_k estimates in numpy arrays, with residue names as keys.
        strict: bool, default False
            If True, raises an error if gk values are specified for nonexistent residue.

        """

        all_restypes = {group.residue_type for group in self.titrationGroups}

        # If gk_dict contains entry not in
        supplied_residues = set(gk_dict.keys())
        if not supplied_residues <= all_restypes:
            if strict:
                raise ValueError("Weights were supplied for a residue that was not in the system.\n"
                                 "{}".format(", ".join(supplied_residues-all_restypes)))

        for residue_type, weights in gk_dict.items():
            # Set the g_k values to the user supplied values.
            for group_index, group in enumerate(self.titrationGroups):
                if group.residue_type == residue_type:

                    # Make sure the right number of weights are specified
                    num_weights = len(weights)
                    num_states = len(self.titrationGroups[group_index])
                    if not num_weights == num_states:
                        raise ValueError("The number of weights ({}) supplied does not match the number of states ({}) for this residue.".format(num_weights, num_states))

                    for state_index, state in enumerate(self.titrationGroups[group_index]):
                        self.titrationGroups[group_index][state_index].g_k = gk_dict[residue_type][state_index]

    def reset_statistics(self):
        """
        Reset statistics of ncmc trials.
        """
        self.nattempted = 0
        self.naccepted = 0
        self.nrejected = 0

        return

    def adjust_to_ph(self, pH):
        """
        Apply the pH target weight correction for the specified pH.

        For amino acids, the pKa values in app.pkas will be used to calculate the correction.
        For ligands, target populations at the specified pH need to be available, otherwise an exception will be thrown.

        Parameters
        ----------
        pH - float, the pH to which target populations should be adjusted.

        Raises
        ------
        ValueError - if the target weight for a given pH is not supplied.
        """

        for residue in self.titrationGroups:
            residue.set_populations(pH)

    def _get14scaling(self, system):
        """
        Determine Coulomb 14 scaling.

        Parameters
        ----------

        system : simtk.openmm.System
            the system to examine

        Returns
        -------

        coulomb14scale (float) - degree to which 1,4 coulomb interactions are scaled

        """
        # Look for a NonbondedForce.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        force = forces['NonbondedForce']
        # Determine coulomb14scale from first exception with nonzero chargeprod.
        for index in range(force.getNumExceptions()):
            [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
            [charge1, sigma1, epsilon1] = force.getParticleParameters(particle1)
            [charge2, sigma2, epsilon2] = force.getParticleParameters(particle2)
            # Using 1.e-15 as necessary precision for establishing greater than 0
            # Needs to be slightly larger than sys.float_info.epsilon to prevent numerical errors.
            if (abs(charge1 / (unit.elementary_charge)) > 1.e-15) and (abs(charge2 / unit.elementary_charge) > 1.e-15) and (abs(chargeProd/(unit.elementary_charge ** 2)) > 1.e-15):
                coulomb14scale = chargeProd / (charge1 * charge2)
                return coulomb14scale

        return None

    def _get14exceptions(self, system, particle_indices):
        """
        Return a list of all 1,4 exceptions involving the specified particles that are not exclusions.

        Parameters
        ----------

        system : simtk.openmm.System
            the system to examine
        particle_indices :list of int
            only exceptions involving at least one of these particles are returned

        Returns
        -------

        exception_indices : list
            list of exception indices for NonbondedForce

        Todo
        ----

        * Deal with the case where there may be multiple NonbondedForce objects.
        * Deal with electrostatics implmented as CustomForce objects (by CustomNonbondedForce + CustomBondForce)

        """
        # Locate NonbondedForce object.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        force = forces['NonbondedForce']
        # Build a list of exception indices involving any of the specified particles.
        exception_indices = list()
        for exception_index in range(force.getNumExceptions()):
            # TODO this call to getExceptionParameters is expensive. Perhaps this could be cached somewhere per force.
            [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
            if (particle1 in particle_indices) or (particle2 in particle_indices):
                if (particle2 in self.atomExceptions[particle1]) or (particle1 in self.atomExceptions[particle2]):
                    exception_indices.append(exception_index)
                    # BEGIN UGLY HACK
                    # chargeprod and sigma cannot be identically zero or else we risk the error:
                    # Exception: updateParametersInContext: The number of non-excluded exceptions has changed
                    # TODO: Once OpenMM interface permits this, omit this code.
                    [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
                    if (2 * chargeProd == chargeProd):
                        chargeProd = sys.float_info.epsilon
                    if (2 * epsilon == epsilon):
                        epsilon = sys.float_info.epsilon
                    force.setExceptionParameters(exception_index, particle1, particle2, chargeProd, sigma, epsilon)
                    # END UGLY HACK

        return exception_indices

    def _set14exceptions(self, system):
        """
        Collect all the NonbondedForce exceptions that pertain to 1-4 interactions.

        Parameters
        ----------
        system - OpenMM System object

        Returns
        -------

        """
        for force in system.getForces():
            if force.__class__.__name__ == "NonbondedForce":
                for index in range(force.getNumExceptions()):
                    [atom1, atom2, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
                    unitless_epsilon = epsilon / unit.kilojoule_per_mole
                    # 1-2 and 1-3 should be 0 for both chargeProd and episilon, whereas a 1-4 interaction is scaled.
                    # Potentially, chargeProd is 0, but epsilon should never be 0.
                    # Using > 1.e-15 as a reasonable float precision for being greater than 0
                    if abs(unitless_epsilon) > 1.e-15:
                        self.atomExceptions[atom1].append(atom2)
                        self.atomExceptions[atom2].append(atom1)
        return

    @staticmethod
    def _parse_fortran_namelist(filename, namelist_name):
        """
        Parse a fortran namelist generated by AMBER 11 constant-pH python scripts.

        Parameters
        ----------

        filename : string
            the name of the file containing the fortran namelist
        namelist_name : string
            name of the namelist section to parse

        Returns
        -------

        namelist : dict
            namelist[key] indexes read values, converted to Python types

        Notes
        -----

        This code is not fully general for all Fortran namelists---it is specialized to the cpin files.

        """
        # Read file contents.
        infile = open(filename, 'r')
        lines = infile.readlines()
        infile.close()

        # Concatenate all text.
        contents = ''
        for line in lines:
            contents += line.strip()

        # Extract section corresponding to keyword.
        key = '&' + namelist_name
        terminator = '/'
        match = re.match(key + '(.*)' + terminator, contents)
        contents = match.groups(1)[0]

        # Parse contents.
        # These regexp match strings come from fortran-namelist from Stephane Chamberland (stephane.chamberland@ec.gc.ca) [LGPL].
        valueInt = re.compile(r'[+-]?[0-9]+')
        valueReal = re.compile(r'[+-]?([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)')
        valueString = re.compile(r'^[\'\"](.*)[\'\"]$')

        # Parse contents.
        namelist = dict()
        while len(contents) > 0:
            # Peel off variable name.
            match = re.match(r'^([^,]+)=(.+)$', contents)
            if not match:
                break
            name = match.group(1).strip()
            contents = match.group(2).strip()

            # Peel off value, which extends to either next variable name or end of section.
            match = re.match(r'^([^=]+),([^,]+)=(.+)$', contents)
            if match:
                value = match.group(1).strip()
                contents = match.group(2) + '=' + match.group(3)
            else:
                value = contents
                contents = ''

            # Split value on commas.
            elements = value.split(',')
            value = list()
            for element in elements:
                if valueReal.match(element):
                    element = float(element)
                elif valueInt.match(element):
                    element = int(element)
                elif valueString.match(element):
                    element = element[1:-1]
                if element != '':
                    value.append(element)
            if len(value) == 1:
                value = value[0]

            namelist[name] = value

        return namelist

    def _get_num_titratable_groups(self):
        """
        Return the number of titratable groups.

        Returns
        -------

        ngroups : int
            the number of titratable groups that have been defined

        """

        return len(self.titrationGroups)

    def _add_titratable_group(self, atom_indices, residue_type, name='', residue_pka=None, pka_data=None):
        """
        Define a new titratable group.

        Parameters
        ----------

        atom_indices : list of int
            the atom indices defining the titration group

        residue_type: str
            The type of residue, e.g. LYS for lysine, HIP for histine, STI for imatinib.

        Other Parameters
        ----------------
        name : str
            name of the group, e.g. Residue: LYS 13.

        Notes
        -----

        No two titration groups may share atoms.

        """
        # Check to make sure the requested group does not share atoms with any existing titration group.
        for group in self.titrationGroups:
            if set(group.atom_indices).intersection(atom_indices):
                raise Exception("Titration groups cannot share atoms. The requested atoms of new titration group (%s) share atoms with another group (%s)." % (
                    str(atom_indices), str(group.atom_indices)))

        # Define the new group.
        group_index = len(self.titrationGroups) + 1
        group = _TitratableResidue.from_lists(list(atom_indices), group_index, name, residue_type, self._get14exceptions(self.system, atom_indices), residue_pka=residue_pka, pka_data=pka_data)
        self.titrationGroups.append(group)
        return group_index

    def get_num_titration_states(self, titration_group_index):
        """
        Return the number of titration states defined for the specified titratable group.

        Parameters
        ----------

        titration_group_index : int
            the titration group to be queried

        Returns
        -------

        nstates : int
            the number of titration states defined for the specified titration group

        """
        if titration_group_index not in range(self._get_num_titratable_groups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." %
                            (titration_group_index, self._get_num_titratable_groups()))

        return len(self.titrationGroups[titration_group_index])

    def _add_titration_state(self, titration_group_index, relative_energy, atom_charges, atom_epsilon, atom_sigma, atom_type, atom_name, bonded_par, angle_par, proper_par, improper_par, proton_count):
        """
        Add a titration state to a titratable group.

        Parameters
        ----------

        titration_group_index : int
            the index of the titration group to which a new titration state is to be added
        relative_energy : simtk.unit.Quantity with units compatible with simtk.unit.kilojoules_per_mole
            the relative energy of this protonation state
        charges : list or numpy array of simtk.unit.Quantity with units compatible with simtk.unit.elementary_charge
            the atomic charges for this titration state
        proton_count : int
            number of protons in this titration state

        Notes
        -----

        The number of charges specified must match the number (and order) of atoms in the defined titration group.
        """

        print('Adding titration state ...')
        # Check input arguments.
        if titration_group_index not in range(self._get_num_titratable_groups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." %
                            (titration_group_index, self._get_num_titratable_groups()))
        if len(atom_charges) != len(self.titrationGroups[titration_group_index].atom_indices):
            raise Exception('The number of charges must match the number (and order) of atoms in the defined titration group.')

        state = _TitrationState.from_lists(relative_energy * self.beta, atom_charges, atom_epsilon, atom_sigma, atom_type, atom_name, bonded_par, angle_par, proper_par, improper_par, proton_count)
        self.titrationGroups[titration_group_index].add_state(state)
        return

    def _get_titration_state(self, titration_group_index):
        """
        Return the current titration state for the specified titratable group.

        Parameters
        ----------

        titration_group_index : int
            the titration group to be queried

        Returns
        -------

        state : int
            the titration state for the specified titration group

        """
        if titration_group_index not in range(self._get_num_titratable_groups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." %
                            (titration_group_index, self._get_num_titratable_groups()))

        return self.titrationGroups[titration_group_index].state_index

    def _get_titration_state_total_charge(self, titration_group_index, titration_state_index):
        """
        Return the total charge for the specified titration state.

        Parameters
        ----------

        titration_group_index : int
            the titration group to be queried

        titration_state_index : int
            the titration state to be queried

        Returns
        -------

        charge : simtk.openmm.Quantity compatible with simtk.unit.elementary_charge
            total charge for the specified titration state

        """
        self._validate_indices(titration_group_index, titration_state_index)

        return self.titrationGroups[titration_group_index][titration_state_index].total_charge

    def _validate_indices(self, titration_group_index, titration_state_index):
        """
        Checks if group and state indexes provided exist.
        Parameters
        ----------
        titration_group_index -  int
        titration_state_index - int

        Returns
        -------

        """
        if titration_group_index not in range(self._get_num_titratable_groups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." %
                            (titration_group_index, self._get_num_titratable_groups()))
        if titration_state_index not in range(len(self.titrationGroups[titration_group_index])):
            raise Exception("Invalid titration state requested.  Requested %d, valid states are in range(%d)." %
                            (titration_state_index, self.get_num_titration_states(titration_group_index)))

    def _set_titration_state(self, titration_group_index, titration_state_index, updateParameters=True):
        """
        Change the titration state of the designated group for the provided state.

        Parameters
        ----------

        titration_group_index : int
            the index of the titratable group whose titration state should be updated
        titration_state_index : int
            the titration state to set as active
        """

        # Check parameters for validity.
        self._validate_indices(titration_group_index, titration_state_index)
        print('Titration group index: ', str(titration_group_index), 'Titration state index:', str(titration_state_index), 'update parameter flag: ', str(updateParameters))

        self._update_forces(titration_group_index, titration_state_index)
        # The context needs to be updated after the force parameters are updated
        if self.context is not None and updateParameters:
            for force_index, force in enumerate(self.forces_to_update):
                force.updateParametersInContext(self.context)
        self.titrationGroups[titration_group_index].state = titration_state_index

        return

    def _update_forces(self, titration_group_index, final_titration_state_index, initial_titration_state_index=None, verbose=0, fractional_titration_state=1.0):
        """
        Update the force parameters to a new titration state by reading them from the cache.

        Notes
        -----
        * Please ensure that the context is updated after calling this function, by using
        `force.updateParametersInContext(context)` for each force that has been updated.

        Parameters
        ----------
        titration_group_index : int
            Index of the group that is changing state
        final_titration_state_index : int
            Index of the state of the chosen residue
        initial_titration_state_index : int, optional, default=None
            If blending two titration states, the initial titration state to blend.
            If `None`, set to `titration_state_index`
        fractional_titration_state : float, optional, default=1.0
            Fraction of `titration_state_index` to be blended with `initial_titration_state_index`.
            If 0.0, `initial_titration_state_index` is fully active; if 1.0, `titration_state_index` is fully active.

        Notes
        -----
        * Every titration state has a list called forces, which stores parameters for all forces that need updating.
        * Inside each list entry is a dictionary that always contains an entry called `atoms`, with single atom parameters by name.
        * NonbondedForces also have an entry called `exceptions`, containing exception parameters.

        """
        # `initial_titration_state_index` should have no effect if not specified, so set it identical to
        # `final_titration_state_index` in that case
        if initial_titration_state_index is None:
            initial_titration_state_index = final_titration_state_index

        # Retrieve cached force parameters fro this titration state.
        cache_initial_forces = self.titrationGroups[titration_group_index][initial_titration_state_index].forces
        cache_final_forces = self.titrationGroups[titration_group_index][final_titration_state_index].forces
        
        if fractional_titration_state == 0.0001:
            print('#######################')
            print('Update forces from {:>4} to {:>4}'.format(initial_titration_state_index, final_titration_state_index))
            print('#######################')
        # Modify charges and exceptions.
        for force_index, force in enumerate(self.forces_to_update):
            # Get name of force class.
            force_classname = force.__class__.__name__


            if force_classname == 'NonbondedForce' or force_classname == 'GBSAOBCForce':
                # Update forces using appropriately blended parameters
                for (atom_initial, atom_final) in zip(cache_initial_forces[force_index]['atoms'], cache_final_forces[force_index]['atoms']):
                    atom = {key: atom_initial[key] for key in ['atom_index']}
                    if force_classname == 'NonbondedForce':
                        # TODO : if we ever change LJ parameters, we need to look into softcore potentials
                        # and separate out the changes in charge, and sigma/eps into different steps.
                        for parameter_name in ['sigma', 'epsilon']:
                            #if charges increase epsilon and sigma have to increase faster to shield charges
                            if float(atom_final['charge']) > float(atom_initial['charge']):
                                ch_fractional_titration_state = fractional_titration_state* 2.0
                                if ch_fractional_titration_state > 1.0:
                                    ch_fractional_titration_state = 1.0 
                                atom[parameter_name] = (1.0 - ch_fractional_titration_state) * atom_initial[parameter_name] + ch_fractional_titration_state * atom_final[parameter_name]
                            else:
                                atom[parameter_name] = (1.0 - fractional_titration_state) * atom_initial[parameter_name] + fractional_titration_state * atom_final[parameter_name]
                        
                        for parameter_name in ['charge']:
                            # change charges seperatly
                            # if charges are decresed they should decrese faster to avoid unshielded charges
                            if float(atom_final[parameter_name]) < float(atom_initial[parameter_name]):
                                ch_fractional_titration_state = fractional_titration_state* 2.0
                                if ch_fractional_titration_state > 1.0:
                                    ch_fractional_titration_state = 1.0 
                                atom[parameter_name] = (1.0 - ch_fractional_titration_state) * atom_initial[parameter_name] + ch_fractional_titration_state * atom_final[parameter_name]
                            else:
                                atom[parameter_name] = (1.0 - fractional_titration_state) * atom_initial[parameter_name] + fractional_titration_state * atom_final[parameter_name]


                        if atom_initial['charge'] != atom_final['charge'] or atom_initial['sigma'] != atom_final['sigma'] or atom_initial['epsilon'] != atom_final['epsilon']:
                            if fractional_titration_state == 0.0001:
                                print('Updating nonbonded parameters for: {:3d}'.format(atom['atom_index']))                          
                                print('Atom-ID: {:3d} atom-current: ch:{:01.4f} si:{:01.4f} ep:{:01.4f} atom-final: ch:{:01.4f} si:{:01.4f} ep:{:01.4f}'.format(atom['atom_index'], float(atom_initial['charge']), float(atom_initial['sigma']), float(atom_initial['epsilon']), float(atom_final['charge']), float(atom_final['sigma']), float(atom_final['epsilon'])))

                        force.setParticleParameters(atom['atom_index'], atom['charge'], atom['sigma'], atom['epsilon'])

                        # Update exceptions
                        # TODO: Handle Custom forces.
                        for (exc_initial, exc_final) in zip(cache_initial_forces[force_index]['exceptions'], cache_final_forces[force_index]['exceptions']):
                                                    
                            exc = {key: exc_initial[key] for key in ['exception_index', 'particle1', 'particle2']}
                            for parameter_name in ['chargeProd', 'sigma', 'epsilon']:
                                exc[parameter_name] = (1.0 - fractional_titration_state) * exc_initial[parameter_name] + \
                                    fractional_titration_state * exc_final[parameter_name]
                            force.setExceptionParameters(
                                exc['exception_index'], exc['particle1'], exc['particle2'], exc['chargeProd'], exc['sigma'], exc['epsilon'])
                    
                    elif force_classname == 'GBSAOBCForce':                        
                        for parameter_name in ['charge', 'radius', 'scaleFactor']:
                            atom[parameter_name] = (1.0 - fractional_titration_state) * atom_initial[parameter_name] + \
                                fractional_titration_state * atom_final[parameter_name]
                        force.setParticleParameters(atom['atom_index'], atom['charge'], atom['radius'], atom['scaleFactor'])
                    
            
            elif force_classname == 'HarmonicBondForce':  
                for bond_index, (bond_initial, bond_final) in enumerate(zip(cache_initial_forces[force_index]['bonds'], cache_final_forces[force_index]['bonds'])):
                    bond = dict()
                    
                    for parameter_name in ['length', 'k']:
                        # generate new, interpolated parameters
                        new_parameter = (1.0 - fractional_titration_state) * float(bond_initial[parameter_name]) + fractional_titration_state * float(bond_final[parameter_name])
                        bond[parameter_name] = new_parameter                    
                   
                    if bond_initial['length'] != bond_final['length'] or bond_initial['k'] != bond_final['k']:
                        if fractional_titration_state == 0.0001:
                            print('Updating bond between: {:3d} and {:3d}'.format(bond_initial['a1'], bond_initial['a2']))
                            print('bond current: {:1.4f} {:5.4f} bond final: {:1.4f} {:5.4f}'.format(float(bond['length']), float(bond['k']), float(bond_final['length']), float(bond_final['k'])))                  

                    # set new parameters using atom indices
                    force.setBondParameters(bond_index, bond_initial['a1'], bond_initial['a2'], bond['length'], bond['k'])
                        
            elif force_classname == 'HarmonicAngleForce':
                if fractional_titration_state == 0.0001:
                    print('#######################')
                    print('#######################')

                for angle_index, (angle_initial, angle_final) in enumerate(zip(cache_initial_forces[force_index]['angles'], cache_final_forces[force_index]['angles'])):
                    angle = dict()

                    for parameter_name in ['angle', 'k']:
                        new_parameter = (1.0 - fractional_titration_state) * float(angle_initial[parameter_name]) + fractional_titration_state * float(angle_final[parameter_name])
                        angle[parameter_name] = new_parameter
                    
                    if angle_initial['angle'] != angle_final['angle'] or angle_initial['k'] != angle_final['k']:
                        if fractional_titration_state == 0.0001:
                            print('Updating angle between: {:3d} {:3d} {:3d}'.format(angle_initial['a1'], angle_initial['a2'], angle_initial['a3']))
                            print('angle initial: {:1.4f} {:5.4f} angle final: {:1.4f} {:5.4f}'.format(float(angle_initial['angle']), float(angle_initial['k']), float(angle_final['angle']), float(angle_final['k'])))                  


                    force.setAngleParameters(angle_index, angle_initial['a1'], angle_initial['a2'], angle_initial['a3'], angle['angle'], angle['k'])

            elif force_classname == 'PeriodicTorsionForce':
                if fractional_titration_state == 0.0001:
                    print('#######################')
                    print('#######################')

                for torsion_index, (torsion_initial, torsion_final) in enumerate(zip(cache_initial_forces[force_index]['torsion'], cache_final_forces[force_index]['torsion'])):
                    torsion = dict()

                    for parameter_name in ['phase1', 'k1']:
                        if fractional_titration_state < 0.5:
                            new_parameter = (1.0 - (2* fractional_titration_state)) * float(torsion_initial[parameter_name])
                            torsion[parameter_name] = new_parameter
                    
                        else:
                            new_parameter = (0.5 - fractional_titration_state) * 2 * float(torsion_final[parameter_name])
                            torsion[parameter_name] = new_parameter
                    

                    for parameter_name in ['periodicity1']:
                        if fractional_titration_state < 0.5:
                            torsion[parameter_name] = torsion_initial[parameter_name]
                        else:
                            torsion[parameter_name] = torsion_final[parameter_name]
                    
                    
                    
                    if torsion_initial['phase1'] != torsion_final['phase1'] or torsion_initial['k1'] != torsion_final['k1']:
                        print('Updating torsion between: {:3d} {:3d} {:3d} {:3d}'.format(torsion_initial['a1'], torsion_initial['a2'], torsion_initial['a3'], torsion_initial['a4']))
                        print('torsion initial: {:1.4f} {:5.4f} torsion final: {:1.4f} {:5.4f}'.format(float(torsion_initial['phase1']), float(torsion_initial['periodicity1']), float(torsion_final['phase2']), float(torsion_final['periodicity1'])))                  

                    force.setTorsionParameters(torsion_index, torsion_initial['a1'], torsion_initial['a2'], torsion_initial['a3'], torsion_initial['a4'], torsion['periodicity1'], torsion['phase1'], torsion['k1'])


            else:
                raise Exception("Don't know how to update force type '%s'" % force_classname)


    def _cache_force(self, titration_group_index, titration_state_index):
        """
        Cache the force parameters for a single titration state.

        Parameters
        ----------
        titration_group_index : int
            Index of the group
        titration_state_index : int

        Call this function to set up the 'forces' information for a single titration state.
        Every titration state has a list called forces, which stores parameters for all forces that need updating.
        Inside each list entry is a dictionary that always contains an entry called `atoms`, with single atom parameters by name.
        NonbondedForces also have an entry called `exceptions`, containing exception parameters.

        Returns
        -------

        """

        print('#########################')
        print('defined in topology ....')
        print('Titration group index', titration_group_index)
        print('titration state index', titration_state_index)
        print('#########################')

        titration_group = self.titrationGroups[titration_group_index]
        titration_state = self.titrationGroups[titration_group_index][titration_state_index]
        atom_indices = titration_group.atom_indices
        atom_type_by_atom_index = dict(zip(atom_indices, titration_state.atom_type))
        atom_name_by_atom_index = dict(zip(atom_indices, titration_state.atom_name))
        atom_charge_by_atom_index = dict(zip(atom_indices, titration_state.charges))
        atom_epsilon_by_atom_index = dict(zip(atom_indices, titration_state.epsilon))
        atom_sigma_by_atom_index = dict(zip(atom_indices, titration_state.sigma))

        bonded_par = titration_state.bonded_par
        angle_par = titration_state.angle_par
        proper_par = titration_state.proper_par

        charge_by_atom_index = dict(zip(atom_indices, titration_state.charges))
        # Store the parameters per individual force
        f_params = list()

        for force_index, force in enumerate(self.forces_to_update):

            # Get name of force class.
            force_classname = force.__class__.__name__
            # update atom parameters
            print('########################')
            print(force_classname)
            print('########################')

            if force_classname == 'NonbondedForce' or force_classname == 'GBSAOBCForce':
                f_params.append(dict(atoms=list()))
                atom_string = 'Atom-ID: {:>3d} Atom-Name: {:>4} Atom-Type: {:>3} Charge: {: 02.4f} Epsilon: {: 02.4f}'
                for atom_index in atom_indices:

                    if force_classname == 'GBSAOBCForce':
                        current_parameters = {key: value for (key, value) in zip(['charge', 'radius', 'scaleFactor'], map(strip_in_unit_system, force.getParticleParameters(atom_index)))}
                        f_params[force_index]['atoms'].append(current_parameters)
                    else:                        
                        charge, sigma, epsilon = map(strip_in_unit_system, force.getParticleParameters(atom_index))
                        print(atom_string.format(atom_index, atom_name_by_atom_index[atom_index], atom_type_by_atom_index[atom_index], atom_charge_by_atom_index[atom_index], atom_epsilon_by_atom_index[atom_index]))

                        current_parameters = {key: value for (key, value) in zip(['charge', 'sigma', 'epsilon'], map(strip_in_unit_system, force.getParticleParameters(atom_index)))}
                        current_parameters['atom_index'] = atom_index
                        current_parameters['atom_name'] = atom_name_by_atom_index[atom_index]
                        current_parameters['atom_type'] = atom_type_by_atom_index[atom_index]
                        current_parameters['charge'] = atom_charge_by_atom_index[atom_index]
                        current_parameters['sigma'] = atom_sigma_by_atom_index[atom_index]
                        current_parameters['epsilon'] = atom_epsilon_by_atom_index[atom_index]

                        f_params[force_index]['atoms'].append(current_parameters)

            elif force_classname == 'HarmonicBondForce':
                f_params.append(dict(bonds=list()))
                bond_string_from_openmm = 'Bond-ID: {:>3d} Atom-ID1: {:>3d} Atom-ID2: {:>3d} length: {: 02.4f} k: {: 02.4f}'

                # only heavy atom - heavy atom bonds are regarded
                for bond_index in range(force.getNumBonds()):
                    a1, a2, length, k = force.getBondParameters(bond_index)     
                    # update current parameters with particular titration state
                    current_parameters = {key: value for (key, value) in zip(['a1', 'a2', 'length', 'k'], map(strip_in_unit_system, force.getBondParameters(bond_index)))}
                    a1_atom_name = atom_name_by_atom_index[a1]
                    a2_atom_name = atom_name_by_atom_index[a2]
                    length = bonded_par[tuple([a1_atom_name, a2_atom_name])]['bond_length']
                    k = bonded_par[tuple([a1_atom_name, a2_atom_name])]['bond_k']                   

                    current_parameters['length'] = length
                    current_parameters['k'] = k

                    f_params[force_index]['bonds'].append(current_parameters)
                    print(bond_string_from_openmm.format(bond_index, a1, a2, float(length), float(k)))


            elif force_classname == 'HarmonicAngleForce':
                f_params.append(dict(angles=list()))
                angle_string_from_openmm = 'Angle-ID: {:>3d} Atom-ID1: {:>3d} Atom-ID2: {:>3d} Atom-ID3: {:>3d} angle: {: 02.4f} k: {: 02.4f}'

                for angle_index in range(force.getNumAngles()):
                    a1, a2, a3, angle_value, k = force.getAngleParameters(angle_index)
                    # update current parameters with particular titration state
                    current_parameters = {key: value for (key, value) in zip(['a1', 'a2', 'a3', 'angle', 'k'], map(strip_in_unit_system, force.getAngleParameters(angle_index)))}
                    
                    a1_atom_name = atom_name_by_atom_index[a1]
                    a2_atom_name = atom_name_by_atom_index[a2]
                    a3_atom_name = atom_name_by_atom_index[a3]
                    angle = angle_par[tuple([a1_atom_name, a2_atom_name, a3_atom_name])]['angle']
                    k = angle_par[tuple([a1_atom_name, a2_atom_name, a3_atom_name])]['k']                   

                    current_parameters['angle'] = angle
                    current_parameters['k'] = k
                    
                    f_params[force_index]['angles'].append(current_parameters)
                    print(angle_string_from_openmm.format(angle_index, a1, a2, a3, float(strip_in_unit_system(angle_value)), float(strip_in_unit_system(k))))

            
            # set torsion parameters
            elif force_classname == 'PeriodicTorsionForce':
                f_params.append(dict(torsion=list()))
                proper_string_from_openmm = 'Proper-ID: {:>3d} Atom-ID1: {:>2d} Atom-ID2: {:>2d} Atom-ID3: {:>2d} Atom-ID4: {:>2d} periodicity: {: 02.4f} phase: {: 02.4f} k: {: 02.4f}'

                for torsion_index in range(force.getNumTorsions()):
                    a1, a2, a3, a4, periodicity, phase, k = force.getTorsionParameters(torsion_index)

                    current_parameters = {key: value for (key, value) in zip(['a1', 'a2', 'a3', 'a4', 'periodicity1', 'phase1', 'k1'], map(strip_in_unit_system, force.getTorsionParameters(torsion_index)))}
                    
                    a1_atom_name = atom_name_by_atom_index[a1]
                    a2_atom_name = atom_name_by_atom_index[a2]
                    a3_atom_name = atom_name_by_atom_index[a3]
                    a4_atom_name = atom_name_by_atom_index[a4]
                    if tuple([a1_atom_name, a2_atom_name, a3_atom_name, a4_atom_name]) in proper_par:
                        periodicity = proper_par[tuple([a1_atom_name, a2_atom_name, a3_atom_name, a4_atom_name])]['periodicity1']
                        phase = proper_par[tuple([a1_atom_name, a2_atom_name, a3_atom_name, a4_atom_name])]['phase1']
                        k = proper_par[tuple([a1_atom_name, a2_atom_name, a3_atom_name, a4_atom_name])]['k1']

                        current_parameters['periodicity1'] = periodicity
                        current_parameters['phase1'] = phase
                        current_parameters['k1'] = k
                        f_params[force_index]['torsion'].append(current_parameters)
                        print(proper_string_from_openmm.format(torsion_index, a1, a2, a3, a4, float(strip_in_unit_system(periodicity)), float(strip_in_unit_system(phase)), float(strip_in_unit_system(k))))


                    else:
                        # improper parameters
                        print("Don't know where to find this torsion : " , tuple([a1_atom_name, a2_atom_name, a3_atom_name, a4_atom_name]))
                        print(a1, a2, a3, a4)
                        print(str([atom_type_by_atom_index[a1], atom_type_by_atom_index[a2], atom_type_by_atom_index[a3], atom_type_by_atom_index[a4]]))
                        print('Assuming this is an improper - setting everything to zero.')

                        #periodicity = proper_par[tuple([a1_atom_name, a2_atom_name, a3_atom_name, a4_atom_name])]['periodicity1']
                        #phase = proper_par[tuple([a1_atom_name, a2_atom_name, a3_atom_name, a4_atom_name])]['phase1']
                        #k = improper_par[tuple([a1_atom_name, a2_atom_name, a3_atom_name, a4_atom_name])]['k1']
                        #print('Periodicity: {} phase: {} k: {}'.format(periodicity, phase, k))

                        current_parameters['periodicity1'] = 0
                        current_parameters['phase1'] = 0.0
                        current_parameters['k1'] = 0.0
                        f_params[force_index]['torsion'].append(current_parameters)

            else:
                raise Exception("Don't know how to update force type '%s'" % force_classname)

            # Update exceptions
            # TODO: Handle Custom forces.
            if force_classname == 'NonbondedForce':
                f_params[force_index]['exceptions'] = list()
                for e_ix, exception_index in enumerate(titration_group.exception_indices):
                    [particle1, particle2, chargeProd, sigma, epsilon] = map(
                        strip_in_unit_system, force.getExceptionParameters(exception_index))

                    # Deal with exceptions between atoms outside of titratable residue
                    try:
                        charge_1 = charge_by_atom_index[particle1]
                    except KeyError:
                        charge_1 = strip_in_unit_system(force.getParticleParameters(particle1)[0])
                    try:
                        charge_2 = charge_by_atom_index[particle2]
                    except KeyError:
                        charge_2 = strip_in_unit_system(force.getParticleParameters(particle2)[0])

                    chargeProd = self.coulomb14scale * charge_1 * charge_2

                    # chargeprod and sigma cannot be identically zero or else we risk the error:
                    # Exception: updateParametersInContext: The number of non-excluded exceptions has changed
                    # TODO: Once OpenMM interface permits this, omit this code.
                    if 2 * chargeProd == chargeProd:
                        chargeProd = sys.float_info.epsilon
                    if 2 * epsilon == epsilon:
                        epsilon = sys.float_info.epsilon

                    # store specific local variables in dict by name
                    exc_dict = dict()
                    for i in ('exception_index', 'particle1', 'particle2', 'chargeProd', 'sigma', 'epsilon'):
                        exc_dict[i] = locals()[i]
                    f_params[force_index]['exceptions'].append(exc_dict)

        self.titrationGroups[titration_group_index][titration_state_index].forces = f_params


    def _perform_ncmc_protocol(self, titration_group_indices, initial_titration_states, final_titration_states, salt_residue_indices=None, salt_states=None):
        """
        Performs non-equilibrium candidate Monte Carlo (NCMC) for attempting an change from the initial protonation
        states to the final protonation states. This functions changes the system's states and returns the work for the
        transformation. Parameters are linearly interpolated between the initial and final states.
        
        Notes
        -----
        The integrator is an simtk.openmm.CustomIntegrator object that calculates the protocol work internally.

        To ensure the NCMC protocol is time symmetric, it has the form
            propagation --> perturbation --> propagation

        Parameters
        ----------
        titration_group_indices :
            The indices of the titratable groups that will be perturbed

        initial_titration_states :
            The initial protonation state of the titration groups

        final_titration_states :
            The final protonation state of the titration groups

        salt_residue_indices: optional, list of int
            The indices of saltswap residues that are to be updated during the ncmc protocol.

        salt_states: optional, list of tuples(int,int)
            The indices of the initial and final states of the specified salt residues.


        Returns
        -------
        work : float
          the protocol work of the NCMC procedure in multiples of kT.
        """
        # Turn the center of mass remover off, otherwise it contributes to the work
        if self.cm_remover is not None:
            self.cm_remover.setFrequency(0)


        print('_perform_ncmc_protocol')
        ncmc_integrator = self.ncmc_integrator
        update_salt = False

        if salt_residue_indices is not None and salt_states is not None:
            update_salt = True
        elif salt_residue_indices is not None and salt_states is None:
                raise ValueError("Need to provide states of the salt changes when specifying salt indices.")
        elif salt_states is not None and salt_residue_indices is None:
                raise ValueError("Need to specify the salt_residue_indices when specifying salt state changes.")

        # Reset integrator statistics
        try:
            # This case covers the GHMCIntegrator
            ncmc_integrator.setGlobalVariableByName("ntrials", 0)  # Reset the internally accumulated work
            ncmc_integrator.setGlobalVariableByName("naccept", 0)  # Reset the GHMC acceptance rate counter

        # Not a GHMCIntegrator
        except:
            try:
                # This case handles the GBAOABIntegrator, and ExternalPerturbationLangevinIntegrator
                ncmc_integrator.setGlobalVariableByName("first_step", 0)
                ncmc_integrator.setGlobalVariableByName("protocol_work", 0)
            except:
                raise RuntimeError("Could not reset the integrator work, this integrator is not supported.")

        # The "work" in the acceptance test has a contribution from the titratable group weights.
        g_initial = 0
        for titration_group_index, (titration_group, titration_state_index) in enumerate(zip(self.titrationGroups, self.titrationStates)):
            titration_state = titration_group[titration_state_index]
            g_initial += titration_state.g_k

        # PROPAGATION
        ncmc_integrator.step(self.propagations_per_step)
        print('Initial titration state: ', str(initial_titration_states))
        print('Final titration state: ', str(final_titration_states))
        #pdb = mm.app.PDBFile('/home/mwieder/input.pdb')
        #pos = self.context.getState(getPositions=True).getPositions() 
        #mm.app.PDBFile.writeFile(pdb.topology, pos, open('/home/mwieder/tmp_wd/00_start_'+str(r) + '_test.pdb', 'w'))


        for step in log_progress(range(self.perturbations_per_trial)):

            # Get the fractional stage of the the protocol
            titration_lambda = float(step + 1) / float(self.perturbations_per_trial)
            # perturbation
            for titration_group_index in titration_group_indices:
                self._update_forces(titration_group_index, final_titration_states[titration_group_index],
                                    initial_titration_state_index=initial_titration_states[titration_group_index],
                                    fractional_titration_state=titration_lambda)

            if update_salt:
                for salt_residue, (from_state, to_state) in zip(salt_residue_indices, salt_states):
                    self.swapper.update_fractional_ion(salt_residue, self._ion_parameters[from_state], self._ion_parameters[to_state], titration_lambda)
            for force_index, force in enumerate(self.forces_to_update):
                force.updateParametersInContext(self.context)
            # propagation
            ncmc_integrator.step(self.propagations_per_step)
           
            # logging of statistics
            if isinstance(ncmc_integrator, GHMCIntegrator):
                self.ncmc_stats_per_step[step] = (ncmc_integrator.getGlobalVariableByName('protocol_work') * self.beta_unitless, ncmc_integrator.getGlobalVariableByName('naccept'), ncmc_integrator.getGlobalVariableByName('ntrials'))
            else:
                self.ncmc_stats_per_step[step] = (ncmc_integrator.getGlobalVariableByName('protocol_work') * self.beta_unitless, 0, 0)

        # Extract the internally calculated work from the integrator
        work = ncmc_integrator.getGlobalVariableByName('protocol_work') * self.beta_unitless

        # Setting the titratable group to the final state so that the appropriate weight can be extracted
        for titration_group_index in titration_group_indices:
            self.titrationGroups[titration_group_index].state = final_titration_states[titration_group_index]

        # Extracting the final state's weight.
        g_final = 0
        for titration_group_index, (titration_group, titration_state_index) in enumerate(zip(self.titrationGroups, self.titrationStates)):
            titration_state = titration_group[titration_state_index]
            g_final += titration_state.g_k

        # Extract the internally calculated work from the integrator
        work += (g_final - g_initial)

        # Turn center of mass remover on again
        if self.cm_remover is not None:
            self.cm_remover.setFrequency(self.cm_remover_freq)

        return work

    def _attempt_state_change(self, proposal, residue_pool=None, reject_on_nan=False):
        """
        Attempt a single Monte Carlo protonation state change.

        move : _StateProposal derived class
            Method of selecting residues to update, and their states.
        residue_pool : str, default None
            The set of titration group incides to propose from. See self.titrationGroups for the list of groups.
            If None, select from all groups uniformly.
        reject_on_nan: bool, (default=False)
            Reject proposal if NaN. Not recommended since NaN typically indicates issues with the simulation.

        """
        attempt_data = _TitrationAttemptData()

        initial_positions = initial_velocities = initial_box_vectors = None

        # If using NCMC, store initial positions.
        if self.perturbations_per_trial > 0:
            initial_openmm_state = self.context.getState(getPositions=True, getVelocities=True)
            initial_positions = initial_openmm_state.getPositions(asNumpy=True)
            initial_velocities = initial_openmm_state.getVelocities(asNumpy=True)
            initial_box_vectors = initial_openmm_state.getPeriodicBoxVectors(asNumpy=True)

        # Select which titratible residues to update.
        if residue_pool is None:
            residue_pool_indices = range(self._get_num_titratable_groups())
        else:
            try:
                residue_pool_indices = self.residue_pools[residue_pool]
            except KeyError:
                raise KeyError("The residue pool '{}' does not exist.".format(residue_pool))

        # Compute initial probability of this protonation state. Used in the acceptance test for instantaneous
        # attempts, and to record potential and kinetic energy.
        log_P_initial, pot1, kin1 = self._compute_log_probability()

        # Store current titration state indices.
        initial_titration_states = copy.deepcopy(self.titrationStates)
        final_titration_states, titration_group_indices, logp_ratio_residue_proposal = proposal.propose_states(self, residue_pool_indices)
        initial_charge = 0
        final_charge = 0
        for idx in titration_group_indices:
            initial_state = initial_titration_states[idx]
            initial_charge += self.titrationGroups[idx].total_charges[initial_state]
            final_state = final_titration_states[idx]
            final_charge += self.titrationGroups[idx].total_charges[final_state]
        
        attempt_data.initial_charge = initial_charge
        attempt_data.initial_states = initial_titration_states
        attempt_data.proposed_charge = final_charge
        attempt_data.proposed_states = final_titration_states
        attempt_data.logp_ratio_residue_proposal = logp_ratio_residue_proposal

        if self.swapper is not None:
            initial_ion_states = copy.deepcopy(self.swapper.stateVector)
            proposed_ion_states = copy.deepcopy(self.swapper.stateVector)        
            net_charge_difference = self._calculate_charge_differences(initial_titration_states, final_titration_states, titration_group_indices)
            saltswap_residue_indices, saltswap_states, logp_ratio_salt_proposal = self.swap_proposal.propose_swaps(self, initial_charge, final_charge)
            
            # The saltswap indices are updated to indicate the change of species
            for saltswap_residue, (from_ion_state, to_ion_state) in zip(saltswap_residue_indices, saltswap_states):
                        proposed_ion_states[saltswap_residue] = to_ion_state
            
            attempt_data.initial_ion_states = initial_ion_states
            attempt_data.proposed_ion_states = proposed_ion_states
            attempt_data.logp_ratio_salt_proposal = logp_ratio_salt_proposal
        

        try:
            # Compute work for switching to new protonation states.            
            # 0 is the shortcut for instantaneous Monte Carlo
            if self.perturbations_per_trial == 0:
                # Use instantaneous switching.
                for titration_group_index in titration_group_indices:
                    self._set_titration_state(titration_group_index, final_titration_states[titration_group_index], updateParameters=False)

                    # If maintaining charge neutrality.
                    if self.swapper is not None:
                        for saltswap_residue, (from_ion_state, to_ion_state) in zip(saltswap_residue_indices, saltswap_states):
                            from_parameter = self._ion_parameters[from_ion_state]
                            to_parameter = self._ion_parameters[to_ion_state]
                            self.swapper.update_fractional_ion(saltswap_residue, from_parameter, to_parameter, 1.0)                
                
                # Push parameter updates to the context
                for force in self.forces_to_update:
                    force.updateParametersInContext(self.context)

                log_P_final, pot2, kin2 = self._compute_log_probability()
                work = - (log_P_final - log_P_initial)

            else:
                # Only perform NCMC when the proposed state is different from the current state
                if initial_titration_states != final_titration_states:
                    # Run NCMC integration.
                    if self.swapper is not None:
                        work = self._perform_ncmc_protocol(titration_group_indices, initial_titration_states, final_titration_states, saltswap_residue_indices, saltswap_states)
                    else:
                        work = self._perform_ncmc_protocol(titration_group_indices, initial_titration_states, final_titration_states)
                else:
                    work = 0.0
                    for step in range(self.perturbations_per_trial):
                        self.ncmc_stats_per_step[step] = (0.0, 0.0, 0.0)

            # Store work history                        
            attempt_data.work = work

            log_P_accept = -work
            log_P_accept += logp_ratio_residue_proposal


            # If maintaining charge neutrality using saltswap
            if self.swapper is not None:
                # The acceptance criterion is extended with the ratio of salt proposal probabilities (reverse/forward)
                log_P_accept += logp_ratio_salt_proposal

            # Only record acceptance statistics for exchanges to different protonation states
            if initial_titration_states != final_titration_states:
                self.nattempted += 1
            print('Calculate work for change from ' + str(initial_titration_states) + ' to ' + str(final_titration_states))
            
            print('Work: ' + str(work))
            
            # Accept or reject with Metropolis criteria.
            attempt_data.logp_accept = log_P_accept
            accept_move = self._accept_reject(log_P_accept)
            attempt_data.accepted = accept_move

            if accept_move:
                # Accept.
                print('accept ncmc titration state change ...')
                if initial_titration_states != final_titration_states:
                    self.naccepted += 1
                # Update titration states.
                for titration_group_index in titration_group_indices:
                    self._set_titration_state(titration_group_index, final_titration_states[titration_group_index], updateParameters=False)
                for force in self.forces_to_update:
                    force.updateParametersInContext(self.context)
                print('New titration state: ' + str(self.titrationStates))

                # If using NCMC, flip velocities to satisfy super-detailed balance.
                if self.perturbations_per_trial > 0:
                    self.context.setVelocities(-self.context.getState(getVelocities=True).getVelocities(asNumpy=True))

                # If maintaining charge neutrality using saltswap
                if self.swapper is not None:
                    # The excess ion count is updated with the change in counterions
                    self.excess_ions -= net_charge_difference
                    # The saltswap indices are updated to indicate the change of species
                    for saltswap_residue, (from_ion_state, to_ion_state) in zip(saltswap_residue_indices,
                                                                                saltswap_states):
                        self.swapper.stateVector[saltswap_residue] = to_ion_state

            else:
                # Reject.
                print('reject ncmc titration state change ...')
                if initial_titration_states != final_titration_states:
                    self.nrejected += 1
                # Restore titration states.
                for titration_group_index in titration_group_indices:
                    self._set_titration_state(titration_group_index, initial_titration_states[titration_group_index], updateParameters=False)

                # If maintaining charge neutrality using saltswap
                if self.swapper is not None:
                    # Restore the salt species parameters
                    for saltswap_residue, (from_ion_state, to_ion_state) in zip(saltswap_residue_indices, saltswap_states):
                        from_parameter = self._ion_parameters[from_ion_state]
                        to_parameter = self._ion_parameters[to_ion_state]
                        self.swapper.update_fractional_ion(saltswap_residue, from_parameter, to_parameter, 0.0)

                for force in self.forces_to_update:
                    force.updateParametersInContext(self.context)

                # If using NCMC, restore coordinates and velocities.
                if self.perturbations_per_trial > 0:
                    self.context.setPositions(initial_positions)
                    self.context.setVelocities(initial_velocities)
                    self.context.setPeriodicBoxVectors(*initial_box_vectors)

        except Exception as err:
            print("nan found")
            if str(err) == 'Particle coordinate is nan' and reject_on_nan:

                logging.warning("NaN during NCMC move, rejecting")
                # Reject.
                if initial_titration_states != final_titration_states:
                    self.nrejected += 1
                # Restore titration states.
                for titration_group_index in titration_group_indices:
                    self._set_titration_state(titration_group_index, initial_titration_states[titration_group_index], updateParameters=False)

                # If maintaining charge neutrality using saltswap
                if self.swapper is not None:
                    # Restore the salt species parameters
                    for saltswap_residue, (from_ion_state, to_ion_state) in zip(saltswap_residue_indices,
                                                                                saltswap_states):
                        from_parameter = self._ion_parameters[from_ion_state]
                        to_parameter = self._ion_parameters[to_ion_state]
                        self.swapper.update_fractional_ion(saltswap_residue, from_parameter, to_parameter, 0.0)

                for force in self.forces_to_update:
                    force.updateParametersInContext(self.context)
                # If using NCMC, restore coordinates and flip velocities.
                if self.perturbations_per_trial > 0:
                    self.context.setPositions(initial_positions)
            else:
                raise
        finally:
            # Restore user integrator
            self._last_attempt_data = attempt_data
            self.compound_integrator.setCurrentIntegrator(0)

        return accepted

    def _accept_reject(self, log_P_accept) -> bool:
        """Perform acceptance/rejection check according to the Metropolis-Hastings acceptance criterium."""
        return (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept))

    def _get_acceptance_probability(self):
        """
        Return the fraction of accepted moves

        Returns
        -------
        fraction : float
            the fraction of accepted moves

        """
        return float(self.naccepted) / float(self.nattempted)

    def _compute_log_probability(self):
        """
        Compute log probability of current configuration and protonation state.

        Returns
        -------
        log_P : float
            log probability of the current context
        pot_energy : float
            potential energy of the current context
        kin_energy : float
            kinetic energy of the current context

        TODO
        ----
        * Generalize this to use ThermodynamicState concept of reduced potential (from repex)
        """

        # Add energetic contribution to log probability.
        state = self.context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy()
        kin_energy = state.getKineticEnergy()
        total_energy = pot_energy + kin_energy
        log_P = - self.beta * total_energy

        if self.pressure is not None:
            # Add pressure contribution for periodic simulations.
            volume = self.context.getState().getPeriodicBoxVolume()
            log.debug('beta = %s, pressure = %s, volume = %s, multiple = %s', str(self.beta), str(self.pressure), str(volume), str(-self.beta * self.pressure * volume * unit.AVOGADRO_CONSTANT_NA))
            log_P -= self.beta * self.pressure * volume * unit.AVOGADRO_CONSTANT_NA

        # Add reference free energy contributions.
        for titration_group_index, (titration_group, titration_state_index) in enumerate(zip(self.titrationGroups, self.titrationStates)):
            titration_state = titration_group[titration_state_index]
            g_k = titration_state.g_k
            log.debug("g_k: %.2f", g_k)
            log_P -= g_k

        # Return the log probability.
        return log_P, pot_energy, kin_energy

    def _get_reduced_potentials(self, group_index=0):
        """Retrieve the reduced potentials for all states of the system given a context.

        Parameters
        ----------
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        """
        # beta * U(x)_j

        ub_j = np.empty(len(self.titrationGroups[group_index]))
        for j in range(ub_j.size):
            ub_j[j] = self._reduced_potential(j)

        # Reset to current state
        return ub_j

    def _reduced_potential(self, state_index):
        """Retrieve the reduced potential for a given state (specified by index) in the given context.

        Parameters
        ----------
        state_index : int
            Index of the state for which the reduced potential needs to be calculated.

        """
        potential_energy = self._get_potential_energy(state_index)
        red_pot = self.beta * potential_energy

        if self.pressure is not None:
            volume = self.context.getState().getPeriodicBoxVolume()
            red_pot -= self.beta * self.pressure * volume * unit.AVOGADRO_CONSTANT_NA

        return red_pot

    def _get_potential_energy(self, state_index, group_index=0):
        """ Retrieve the potential energy for a given state (specified by index) in the given context.

        Parameters
        ----------
        state_index : int
            Index of the state for which the reduced potential needs to be calculated.
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        Things to do
        ------------
         * Implement an NCMC version of this?

        """
        current_state = self._get_titration_state(group_index)
        self._set_titration_state(group_index, state_index, updateParameters=True)
        temp_state = self.context.getState(getEnergy=True)
        potential_energy = temp_state.getPotentialEnergy()
        self._set_titration_state(group_index, current_state)
        return potential_energy

    def _calculate_charge_differences(self, from_states, to_states, indices=None):
        """Calculate the net charge difference between states.

        Parameters
        ----------
        from_states - list of states before state change
        to_states - list after state change
        indices - optional, set of indices over which to calculate charge change.

        Returns
        -------
        int - charge
        """
        charge = 0
        if indices is None:
            indices = range(len(self.titrationGroups))

        for index in indices:
            group = self.titrationGroups[index]
            from_state = from_states[index]
            to_state = to_states[index]
            charge += group.titration_states[to_state].proton_count - group.titration_states[from_state].proton_count

        return charge


class AmberProtonDrive(NCMCProtonDrive):
    """
    The AmberProtonDrive is a Monte Carlo driver for protonation state changes and tautomerism in OpenMM.
    It relies on Ambertools to set up a simulation system, and requires a ``.cpin`` input file with protonation states.

    Protonation state changes, and additionally, tautomers are treated using the constant-pH dynamics method of Mongan, Case and McCammon [Mongan2004]_, or Stern [Stern2007]_ and NCMC methods from Nilmeier [Nilmeier2011]_.

    References
    ----------

    .. [Mongan2004] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born implicit solvent. J Comput Chem 25:2038, 2004.
        http://dx.doi.org/10.1002/jcc.20139

    .. [Stern2007] Stern HA. Molecular simulation with variable protonation states at constant pH. JCP 126:164112, 2007.
        http://link.aip.org/link/doi/10.1063/1.2731781

    .. [Nilmeier2011] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. PNAS 108:E1009, 2011.
        http://dx.doi.org/10.1073/pnas.1106094108

    .. todo::

      * Add alternative proposal types, including schemes that avoid proposing self-transitions (or always accept them):
        - Parallel Monte Carlo schemes: Compute N proposals at once, and pick using Gibbs sampling or Metropolized Gibbs?
      * Add automatic tuning of switching times for optimal acceptance.


    """
    def __init__(self, temperature, topology, system, cpin_filename, pressure=None,
                 perturbations_per_trial=0, propagations_per_step=1):
        """
        Initialize a Monte Carlo titration driver for simulation of protonation states and tautomers.

        Parameters
        ----------
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature at which the system is to be simulated.
        topology : protons.app.Topology
            OpenMM object containing the topology of system
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        cpin_filename : string
            AMBER 'cpin' file defining protonation charge states and energies of amino acids
        pressure : simtk.unit.Quantity compatible with atmospheres, optional
            For explicit solvent simulations, the pressure.
        perturbations_per_trial : int, optional, default=0
            Number of perturbation steps per NCMC switching trial, or 0 if instantaneous Monte Carlo is to be used.
        propagations_per_step : int, optional, default=1
            Number of propagation steps in between perturbation steps.

        Things to do
        ------------
        * Generalize simultaneous_proposal_probability to allow probability of single, double, triple, etc. proposals to be specified?

        """

        super(AmberProtonDrive, self).__init__(temperature, topology, system, pressure, perturbations_per_trial=perturbations_per_trial, propagations_per_step=propagations_per_step)

        # Load AMBER cpin file defining protonation states.
        namelist = self._parse_fortran_namelist(cpin_filename, 'CNSTPH')

        # Make sure RESSTATE is a list.
        if type(namelist['RESSTATE']) == int:
            namelist['RESSTATE'] = [namelist['RESSTATE']]

        # Make sure RESNAME is a list.
        if type(namelist['RESNAME']) == str:
            namelist['RESNAME'] = [namelist['RESNAME']]

        # Extract number of titratable groups.
        ngroups = len(namelist['RESSTATE'])
        # Define titratable groups and titration states.
        for group_index in range(ngroups):
            # Extract information about this titration group.
            name = namelist['RESNAME'][group_index + 1]
            first_atom = namelist['STATEINF(%d)%%FIRST_ATOM' % group_index] - 1
            first_charge = namelist['STATEINF(%d)%%FIRST_CHARGE' % group_index]
            first_state = namelist['STATEINF(%d)%%FIRST_STATE' % group_index]
            num_atoms = namelist['STATEINF(%d)%%NUM_ATOMS' % group_index]
            num_states = namelist['STATEINF(%d)%%NUM_STATES' % group_index]

            # Define titratable group.
            atom_indices = range(first_atom, first_atom + num_atoms)
            residue_type = str.split(name)[1] #  Should grab AS4
            if not len(residue_type) == 3:
                example = 'Residue: AS4 2'
                log.warn("Residue type '{}' has unusual length, verify residue name"
                         " in CPIN file has format like this one: '{}'".format(residue_type, example))
            if residue_type in available_pkas:
                residue_pka = available_pkas[residue_type]
            else:
                residue_pka = None

            self._add_titratable_group(atom_indices, residue_type, name=name, residue_pka=residue_pka)

            # Define titration states.
            for titration_state in range(num_states):
                # Extract charges for this titration state.
                # is defined in elementary_charge units
                charges = namelist['CHRGDAT'][(first_charge+num_atoms*titration_state):(first_charge+num_atoms*(titration_state+1))]

                # Extract relative energy for this titration state.
                relative_energy = namelist['STATENE'][first_state + titration_state] * unit.kilocalories_per_mole
                relative_energy = 0.0 * unit.kilocalories_per_mole
                # Get proton count.
                proton_count = namelist['PROTCNT'][first_state + titration_state]
                # Create titration state.

                self._add_titration_state(group_index, relative_energy, charges, proton_count)
                self._cache_force(group_index, titration_state)
            # Set default state for this group.

            self._set_titration_state(group_index, namelist['RESSTATE'][group_index])

        return


class ForceFieldProtonDrive(NCMCProtonDrive):
    """
    The ForceFieldProtonDrive is a Monte Carlo driver for protonation state changes and tautomerism in OpenMM.
    It relies on ffxml files to set up a simulation system.

    Protonation state changes, and additionally, tautomers are treated using the constant-pH dynamics method of Mongan, Case and McCammon [Mongan2004]_, or Stern [Stern2007]_ and NCMC methods from Nilmeier [Nilmeier2011]_ and Chen and Roux [Chen2015]_ .

    References
    ----------

    .. [Mongan2004] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born implicit solvent. J Comput Chem 25:2038, 2004.
        http://dx.doi.org/10.1002/jcc.20139

    .. [Stern2007] Stern HA. Molecular simulation with variable protonation states at constant pH. JCP 126:164112, 2007.
        http://link.aip.org/link/doi/10.1063/1.2731781

    .. [Nilmeier2011] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. PNAS 108:E1009, 2011.
        http://dx.doi.org/10.1073/pnas.1106094108

    .. [Chen2015] Chen, Yunjie, and Benoît Roux. "Constant-pH hybrid nonequilibrium molecular dynamics–monte carlo simulation method." Journal of chemical theory and computation 11.8 (2015): 3919-3931.

    .. todo::

      * Add NCMC switching moves to allow this scheme to be efficient in explicit solvent.
      * Add alternative proposal types, including schemes that avoid proposing self-transitions (or always accept them):
        - Parallel Monte Carlo schemes: Compute N proposals at once, and pick using Gibbs sampling or Metropolized Gibbs?
      * Allow specification of probabilities for selecting N residues to change protonation state at once.
      * Add calibrate() method to automagically adjust relative energies of protonation states of titratable groups in molecule.
      * Add automatic tuning of switching times for optimal acceptance.
      * Extend to handle systems set up via OpenMM app Forcefield class.
      * Make integrator optional if not using NCMC

    """

    def __init__(self, temperature, topology, system, forcefield, ffxml_files, compiler, pressure=None, perturbations_per_trial=0, propagations_per_step=1, residues_by_name=None, residues_by_index=None):


        """
        Initialize a Monte Carlo titration driver for simulation of protonation states and tautomers.

        Parameters
        ----------
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature at which the system is to be simulated.
        topology : protons.app.Topology
            Topology of the system
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        ffxml_files : str or list of str
            Single ffxml filename, or list of ffxml filenames containing protons information.
        forcefield : simtk.openmm.app.ForceField
            ForceField parameters used to make a system.
        pressure : simtk.unit.Quantity compatible with atmospheres, optional
            For explicit solvent simulations, the pressure.
        perturbations_per_trial : int, optional, default=0
            Number of steps per NCMC switching trial, or 0 if instantaneous Monte Carlo is to be used.
        propagations_per_step : int, optional, default=1
            Number of propagation steps in between perturbation steps.

        #NOTE: here we indicate the residue that should be treated as titratable 

        residues_by_index : list of int
            Residues in topology by index that should be treated as titratable
        residues_by_name : list of str
            Residues by name in topology that should be treated as titratable

        Notes
        -----
        If neither residues_by_index, or residues_by_name are specified, all possible residues with Protons parameters
        will be treated.

        """
        # Input validation
        if residues_by_name is not None:
            if not isinstance(residues_by_name, list):
                raise TypeError("residues_by_name needs to be a list")

        if residues_by_index is not None:
            if not isinstance(residues_by_index, list):
                raise TypeError("residues_by_index needs to be a list")

        super(ForceFieldProtonDrive, self).__init__(temperature, topology, system, pressure, perturbations_per_trial=perturbations_per_trial, propagations_per_step=propagations_per_step)

        ffxml_residues = self._parse_ffxml_files(ffxml_files)

        # Collect all of the residues that need to be treated
        all_residues = list(topology.residues())
        selected_residue_indices = list()

        # Validate user specified indices
        if residues_by_index is not None:
            for residue_index in residues_by_index:
                residue = all_residues[residue_index]
                if residue.name not in ffxml_residues:
                    raise ValueError("Residue '{}:{}' is not treatable using protons. Please provide Protons parameters using an ffxml file, or deselect it.".format(residue.name, residue.index))
            selected_residue_indices.extend(residues_by_index)

        # Validate user specified residue names
        if residues_by_name is not None:
            for residue_name in residues_by_name:
                if residue_name not in ffxml_residues:
                    raise ValueError("Residue type '{}' is not a protons compatible residue. Please provide Protons parameters using an ffxml file, or deselect it.".format(residue_name))

            for residue in all_residues:
                if residue.name in residues_by_name:
                    selected_residue_indices.append(residue.index)
                    print('Selected residue indeces: ', selected_residue_indices)
        # If no names or indices are specified, make all compatible residues titratable
        if residues_by_name is None and residues_by_index is None:
            for residue in all_residues:
                if residue.name in ffxml_residues:
                    selected_residue_indices.append(residue.index)

        # Remove duplicate indices and sort
        selected_residue_indices = sorted(list(set(selected_residue_indices)))
        self._compiler = compiler
        self._add_xml_titration_groups(topology, forcefield, ffxml_residues, selected_residue_indices)

        return

    def _add_xml_titration_groups(self, topology, forcefield, ffxml_residues, selected_residue_indices):
        """
        Create titration groups for the selected residues in the topology, using ffxml information gathered earlier.
        Parameters
        ----------
        topology - OpenMM Topology object
        forcefield - OpenMM ForceField object
        ffxml_residues - dict of residue ffxml templates
        selected_residue_indices - Residues to treat using Protons.

        Returns
        -------

        """

        all_residues = list(topology.residues())
        bonded_to_atoms_list = forcefield._buildBondedToAtomList(topology)

        # Extract number of titratable groups.
        ngroups = len(selected_residue_indices)
        # Define titratable groups and titration states.
        for group_index in range(ngroups):
            # Extract information about this titration group.
            residue_index = selected_residue_indices[group_index]
            residue = all_residues[residue_index]

            template = forcefield._templates[residue.name]
            # Find the system indices of the template atoms for this residue
            matches = app.forcefield._matchResidue(residue, template, bonded_to_atoms_list)

            if matches is None:
                raise ValueError("Could not match residue atoms to template.")

            atom_indices = [atom.index for atom in residue.atoms()]

            # Sort the atom indices in the template in the same order as the topology indices.
            atom_indices = [id for (match, id) in sorted(zip(matches, atom_indices))]
            protons_block = ffxml_residues[residue.name].xpath('Protons')[0]

            residue_pka = None
            pka_data = None
            # Add pka adjustment features
            if residue.name in available_pkas:
                residue_pka = available_pkas[residue.name]

            if len(protons_block.findall('State/Condition')) > 0 and residue_pka is None:
                pka_data = DataFrame(columns=["pH", 'Temperature (K)', 'Ionic strength (mM)', 'log population'])
                for state_index, state_block in enumerate(protons_block.xpath('State')):
                    for condition in state_block.xpath('Condition'):
                        row = dict(State=state_index)
                        try:
                            row['pH'] = float(condition.get('pH'))
                        except TypeError:
                            row['pH'] = None
                        try:
                            row['Temperature (K)'] = float(condition.get('temperature_kelvin'))
                        except TypeError:
                            row['Temperature (K)'] = None
                        try:
                            row['Ionic strength (mM)'] = float(condition.get('ionic_strength_mM'))
                        except TypeError:
                            row['Ionic strength (mM)'] = None
                        logpop = condition.get('log_population')
                        try:
                            row['log population'] = float(logpop)
                        except TypeError:
                            raise ValueError("The log population provided can not be converted to a number :'{}'".format(logpop))
                        pka_data = pka_data.append(row, ignore_index=True)

            # Create a new group with the given indices
            self._add_titratable_group(atom_indices,
                                       residue.name,
                                       name="Chain {} Residue {} {}".format(residue.chain.id, residue.name, residue.id),
                                       residue_pka=residue_pka, pka_data=pka_data)

            # Define titration states.

            print(' -- Parameters for the different states as defined in ffxml')
            # mw entry point!
            bonded_par = dict()
            angle_par = dict()
            torsion_par = dict()
            proper_par = dict()
            improper_par = dict()

                
            for state_block in protons_block.xpath("State"):
                # Extract charges for this titration state.
                # is defined in elementary_charge units
                print('###############')
                state_index = int(state_block.get("index"))
                print(   '-- State: ', state_index)
                # Parameters for state
                # 
                atom_charges = []
                atom_epsilon = []
                atom_sigma = []
                atom_type = []
                atom_name = []
                #print('ATOMS:')
                for index, atom in enumerate(state_block.xpath("Atom")):
                    atom_charges.append(float(atom.get("charge")))
                    atom_epsilon.append(float(atom.get("epsilon")))
                    atom_sigma.append(float(atom.get("sigma")))
                    atom_type.append(atom.get('type'))
                    atom_name.append(atom.get('name'))

                #print('BONDS')
                for index, bond in enumerate(state_block.xpath("Bond")):
                    key = tuple([bond.get('name1'), bond.get('name2')])
                    bond_length = (bond.get('length'))
                    bond_k = (bond.get('k'))
                    d = dict()
                    d['bond_length'] = bond_length
                    d['bond_k'] = bond_k
                    bonded_par[key] = d

                #print('Angle')
                for index, angle in enumerate(state_block.xpath("Angle")):
                    key = tuple([angle.get('name1'), angle.get('name2'), angle.get('name3')])
                    angle_value = (angle.get('angle'))
                    angle_k = (angle.get('k'))
                    d = dict()
                    d['angle'] = angle_value
                    d['k'] = angle_k
                    angle_par[key] = d


                #print('PROPER')
                for index, proper in enumerate(state_block.xpath("Proper")):
                    #print(index, ':', proper.get('name1'), proper.get('name2'), proper.get('name3'), proper.get('name4'))
                    key = tuple(([proper.get('name1'), proper.get('name2'), proper.get('name3'), proper.get('name4')]))
                    d = dict()

                    d['periodicity1'] = (proper.get('periodicity1'))
                    d['phase1'] = (proper.get('phase1'))
                    d['k1'] = (proper.get('k1'))
                    proper_par[key] = d

                #print('IMPROPER')
                for index, improper in enumerate(state_block.xpath("Improper")):
                    #print(index, ':', improper.get('name1'), improper.get('name2'), improper.get('name3'), improper.get('name4'))
                    key = tuple(([improper.get('name1'), improper.get('name2'), improper.get('name3'), improper.get('name4')]))
                    d = dict()

                    d['periodicity1'] = (improper.get('periodicity1'))
                    d['phase1'] = (improper.get('phase1'))
                    d['k1'] = (improper.get('k1'))
                    improper_par[key] = d

                relative_energy = float(state_block.get("g_k")) * unit.kilocalories_per_mole
                # Get proton count.
                proton_count = int(state_block.get("proton_count"))
                # Create titration state.
                self._add_titration_state(group_index, relative_energy, atom_charges, atom_epsilon, atom_sigma, atom_type, atom_name, bonded_par, angle_par, proper_par, improper_par, proton_count)
                self._cache_force(group_index, state_index)

            # Set default state for this group.
            self._set_titration_state(group_index, 0)


    def _parse_ffxml_files(self, ffxml_files):
        """
        Read an ffxml file, or a list of ffxml files, and extract the residues that have Protons information.

        Parameters
        ----------
        ffxml_files single object, or list of
            - a file name/path
            - a file object
            - a file-like object
            - a URL using the HTTP or FTP protocol
        The file should contain ffxml residues that have a <Protons> block.

        Returns
        -------
        ffxml_residues - dict of all residue blocks that were detected, with residue names as keys.

        """
        if not isinstance(ffxml_files, list):
            ffxml_files = [ffxml_files]

        xmltrees = list()
        ffxml_residues = dict()
        # Collect xml parameters from provided input files
        for file in ffxml_files:
            try:
                tree = etree.parse(file)
                xmltrees.append(tree)
            except IOError:
                full_path = os.path.join(os.path.dirname(__file__), 'data', file)
                tree = etree.parse(full_path)
                xmltrees.append(tree)

        for xmltree in xmltrees:
            # All residues that contain a protons block
            for xml_residue in xmltree.xpath('/ForceField/Residues/Residue[Protons]'):
                xml_resname = xml_residue.get("name")
                if not xml_resname in ffxml_residues:
                    # Store the protons block of the residue
                    ffxml_residues[xml_resname] = xml_residue
                else:
                    raise ValueError("Duplicate residue name found in parameters: {}".format(xml_resname))

        return ffxml_residues


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



def log_progress(sequence, every=None, size=None, name='Items'):
    from ipywidgets import IntProgress, HTML, VBox
    from IPython.display import display

    is_iterator = False
    if size is None:
        try:
            size = len(sequence)
        except TypeError:
            is_iterator = True
    if size is not None:
        if every is None:
            if size <= 200:
                every = 1
            else:
                every = int(size / 200)     # every 0.5%
    else:
        assert every is not None, 'sequence is iterator, set every'

    if is_iterator:
        progress = IntProgress(min=0, max=1, value=1)
        progress.bar_style = 'info'
    else:
        progress = IntProgress(min=0, max=size, value=0)
    label = HTML()
    box = VBox(children=[label, progress])
    display(box)

    index = 0
    try:
        for index, record in enumerate(sequence, 1):
            if index == 1 or index % every == 0:
                if is_iterator:
                    label.value = '{name}: {index} / ?'.format(
                        name=name,
                        index=index
                    )
                else:
                    progress.value = index
                    label.value = u'{name}: {index} / {size}'.format(
                        name=name,
                        index=index,
                        size=size
                    )
            yield record
    except:
        progress.bar_style = 'danger'
        raise
    else:
        progress.bar_style = 'success'
        progress.value = index
        label.value = "{name}: {index}".format(
            name=name,
            index=str(index or '?')
        )
