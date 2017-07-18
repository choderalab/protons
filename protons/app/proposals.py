# coding=utf-8
"""Residue selection moves for protons MC"""
from abc import ABCMeta, abstractmethod
from .logger import log
import copy
import random
import numpy as np


class _StateProposal(metaclass=ABCMeta):
    """An abstract base class describing the common public interface of residue selection moves."""

    @abstractmethod
    def propose_states(self, drive, residue_pool_indices):
        """ Pick new states for a set of titration groups.

            Parameters
            ----------
            drive - subclass of NCMCProtonDrive
                A proton drive with a set of titratable residues.
            residue_pool_indices - list of int
                List of the residues that could be titrated

            Returns
            -------
            final_titration_states - list of the final titration state of every residue
            titration_group_indices - the indices of the residues that are changing
            float, log (probability of reverse proposal)/(probability of forward proposal)
        """
        return list(), list(), float()


class UniformProposal(_StateProposal):
    """Selects residues uniformly from the supplied residue pool."""

    def __init__(self):
        """Instantiate a UniformProposal"""
        pass

    def propose_states(self, drive, residue_pool_indices):
        """ Pick new states for a set of titration groups.

            Parameters
            ----------
            drive - subclass of NCMCProtonDrive
                A protondrive to update
            residue_pool_indices - list of int
                List of the residues that could be titrated

            Returns
            -------
            final_titration_states - list of the final titration state of every residue
            titration_group_indices - the indices of the residues that are changing
            float, log (probability of reverse proposal)/(probability of forward proposal)

        """
        final_titration_states = copy.deepcopy(drive.titrationStates)
        titration_group_indices = random.sample(residue_pool_indices, 1)
        # Select new titration states.
        for titration_group_index in titration_group_indices:
            # Choose a titration state with uniform probability (even if it is the same as the current state).
            titration_state_index = random.choice(range(drive.get_num_titration_states(titration_group_index)))
            final_titration_states[titration_group_index] = titration_state_index
        return final_titration_states, titration_group_indices, 0.0


class DoubleProposal(_StateProposal):
    """Uniformly selects one, or two residues with some probability from the supplied residue pool."""

    def __init__(self, simultaneous_proposal_probability):
        """Instantiate a DoubleProposal

        Parameters
        ----------
        simultaneous_proposal_probability - float
            The probability of drawing two residues instead of one. Must be between 0 and 1.
        """

        self.simultaneous_proposal_probability = simultaneous_proposal_probability

        return

    def propose_states(self, drive, residue_pool_indices):
        """Uniformly select new titrationstates from the provided residue pool, with a probability of selecting
        two residues.

        Parameters
        ----------
        drive - subclass of NCMCProtonDrive
            A protondrive
        residue_pool_indices - list of int
            List of the residues that could be titrated

        Returns
        -------
        final_titration_states - list of the final titration state of every residue
        titration_group_indices - the indices of the residues that are changing
        float - log (probability of reverse proposal)/(probability of forward proposal)

        """
        final_titration_states = copy.deepcopy(drive.titrationStates)

        # Choose how many titratable groups to simultaneously attempt to update.

        # Update one residue by default
        ndraw = 1
        # Draw two residues with some probability
        if (len(residue_pool_indices) > 1) and (random.random() < self.simultaneous_proposal_probability):
            ndraw = 2

        log.debug("Updating %i residues.", ndraw)

        titration_group_indices = random.sample(residue_pool_indices, ndraw)
        # Select new titration states.
        for titration_group_index in titration_group_indices:
            # Choose a titration state with uniform probability (even if it is the same as the current state).
            titration_state_index = random.choice(range(drive.get_num_titration_states(titration_group_index)))
            final_titration_states[titration_group_index] = titration_state_index
        return final_titration_states, titration_group_indices, 0.0


class CategoricalProposal(_StateProposal):
    """Select N residues, with probability p_N"""

    def __init__(self, p_N):
        """
        Instantiate a CategoricalProposal.

        Parameters
        ----------
        p_N -  sequence of floats, length N
        Probability of updating 1...N  states. These should sum to 1.
        """

        if not sum(p_N) == 1.0:
            raise ValueError("p_N should sum to 1.0")

        self.p_N = p_N
        self.N = len(p_N)

    def propose_states(self, drive, residue_pool_indices):
        """Propose new titration states."""

        final_titration_states = copy.deepcopy(drive.titrationStates)

        # Choose how many titratable groups to simultaneously attempt to update.

        # Update one residue by default
        ndraw = np.random.choice(self.N, 1, p=self.p_N) + 1
        log.debug("Updating %i residues.", ndraw)

        titration_group_indices = random.sample(residue_pool_indices, ndraw)
        # Select new titration states.
        for titration_group_index in titration_group_indices:
            # Choose a titration state with uniform probability (even if it is the same as the current state).
            titration_state_index = random.choice(range(drive.get_num_titration_states(titration_group_index)))
            final_titration_states[titration_group_index] = titration_state_index

        return final_titration_states, titration_group_indices, 0.0


