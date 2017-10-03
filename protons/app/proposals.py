# coding=utf-8
"""Residue selection moves for protons MC"""
from abc import ABCMeta, abstractmethod
from .logger import log
import copy
import random
import numpy as np
from scipy.misc import comb

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


class _SaltSwapProposal:
    """This class is a baseclass for selecting water molecules or ions to maintain charge neutrality."""

    def propose_swaps(self, drive, net_charge_difference):
        """Select a series of water molecules/ions to swap.

        Parameters
        ----------

        drive - a ProtonDrive object with a swapper attached.
        net_charge_difference - int, the total charge that needs to be countered by ions.

        Nota bene
        ---------
        The net charge difference will have the opposite charge of the ions that will be added to the system,
        such that the resulting total charge change will be 0.

        Returns
        -------
        list(int) - indices of water molecule in saltswap that are swapped
        list(tuple(dict,dict),) -  list of tuples with (initial, final) parameters of the water molecule/ion as dict, one tuple for each requested swap
        float - log (probability of reverse proposal)/(probability of forward proposal)
        """

        return list(), list(), float()


class UniformSwapProposal(_SaltSwapProposal):
    """Uniformly selects as many water/ions as needed from the array of available waters/ions."""

    def __init__(self):
        """Instantiate a UniformSwapProposal."""
        pass

    @staticmethod
    def _select_ion_water_swaps(drive, charge_to_counter):
        """Select ions or water molecule swap procedure to facilitate maintaining charge neutrality while changing protonation states.

        Notes
        -----
        Technically, for charge differences of 2 we could change one anion into a cation or vice versa.
        This code would instead remove one anion, and change one water into a cation.

        Parameters
        ----------
        drive - a ProtonDrive object
        charge_to_counter - the total difference in charge that needs to be countered.

        Returns
        -------
        dict(wat2cat, wat2ani, cat2wat, ani2wat)

        """

        # Note that we don't allow for direct transitions between ions of different charge.
        swaps = dict(wat2cat=0, wat2ani=0, cat2wat=0, ani2wat=0)
        excess_ions = int(drive.excess_ions)  # copy

        while abs(charge_to_counter) > 0:

            # A net amount of cations were previously added to the system
            if excess_ions > 0:
                charge_to_counter, excess_ions = UniformSwapProposal._swap_excess_cations(charge_to_counter, excess_ions, swaps)

            # No additional ions were previously added by the swapper.
            elif excess_ions == 0:
                charge_to_counter, excess_ions = UniformSwapProposal._swap_no_excess_ions(charge_to_counter, excess_ions, swaps)

            # A net amount of anions were previously added to the system
            elif excess_ions < 0:
                charge_to_counter, excess_ions = UniformSwapProposal._swap_excess_anions(charge_to_counter, excess_ions, swaps)

        return swaps

    @staticmethod
    def _swap_excess_anions(charge_to_counter, net_ions, swaps):
        """Adds a single unit of charge swap in the case of excess anions"""

        # Need to counter positive charge by adding anion
        if charge_to_counter > 0:
            swaps['wat2ani'] += 1
            # One negative charge was added
            net_ions -= 1
            # One positive charge was countered
            charge_to_counter -= 1
        # Need to counter negative charge by removing anion
        elif charge_to_counter < 0:
            swaps['ani2wat'] += 1
            # One negative charge was removed
            net_ions += 1
            # One negative charge was countered
            charge_to_counter += 1
        return charge_to_counter, net_ions

    @staticmethod
    def _swap_no_excess_ions(charge_to_counter, net_ions, swaps):
        """Add a neutralizing swap operation in the case of no excess ions."""
        # Need to counter positive charge by adding anion
        if charge_to_counter > 0:
            swaps['wat2ani'] += 1
            # One negative charge was added
            net_ions -= 1
            # One positive charge was countered
            charge_to_counter -= 1
        # Need to counter negative charge by adding cation
        elif charge_to_counter < 0:
            swaps['wat2cat'] += 1
            # One positive charge was added
            net_ions += 1
            # One negative charge was countered
            charge_to_counter += 1

        return charge_to_counter, net_ions

    @staticmethod
    def _swap_excess_cations(charge_to_counter, net_ions, swaps):
        """Add a swap of water/ions in the case of excess cations."""
        # Need to counter positive charge by removing cation
        if charge_to_counter > 0:
            swaps['cat2wat'] += 1
            # One positive charge was removed
            net_ions -= 1
            # One positive charge was countered
            charge_to_counter -= 1
        # Need to counter negative charge by adding cation
        elif charge_to_counter < 0:
            swaps['wat2cat'] += 1
            # One positive charge was added
            net_ions += 1
            # One negative charge was countered
            charge_to_counter += 1

        return charge_to_counter, net_ions

    @staticmethod
    def _validate_swaps(swaps):
        """Perform sanity checks. Swaps should only add charge in one direction.

        Raises
        ------
        RuntimeError - in case conflicting swaps occur at the same time.
            The error message will detail what the conflict is.
        """

        if swaps['wat2cat'] != 0 and swaps['wat2ani'] != 0:
            raise RuntimeError("Opposing charge ions are added. This is a bug in the code.")
        elif swaps['cat2wat'] != 0 and swaps['ani2wat'] != 0:
            raise RuntimeError("Opposing charge ions are removed. This is a bug in the code.")
        elif swaps['cat2wat'] != 0 and swaps['wat2cat'] != 0:
            raise RuntimeError("Cations are being added and removed at the same time. This is a bug in the code.")
        elif swaps['ani2wat'] != 0 and swaps['wat2ani'] != 0:
            raise RuntimeError("Anions are being added and removed at the same time. This is a bug in the code.")

    def propose_swaps(self, drive, net_charge_difference):
        """Select a series of water molecules/ions to swap uniformly from all waters/ions.

            Parameters
            ----------

            drive - a ProtonDrive object with a swapper attached.
            net_charge_difference - int, the total charge that needs to be countered.

            Returns
            -------
            list(int) - indices of water molecule in saltswap that are swapped
            list(tuple(dict,dict),) -  list of tuples with (initial, final) parameters of the water molecule/ion as dict, one tuple for each requested swap
            float - log (probability of reverse proposal)/(probability of forward proposal)
        """

        # Defaults. If no swaps are necessary, this will be all that is needed.
        saltswap_residue_indices = list()
        saltswap_parameter_pairs = list()
        log_ratio = 0.0 # fully symmetrical proposal if no swaps occur.

        # If swaps are needed
        if net_charge_difference != 0:

            water_params = drive.swapper.water_parameters
            cation_params = drive.swapper.cation_parameters
            anion_params = drive.swapper.anion_parameters

            # There is a net charge difference, find which swaps are necessary to compute.
            swaps = self._select_ion_water_swaps(drive, net_charge_difference)

            # Apply sanity checks
            UniformSwapProposal._validate_swaps(swaps)

            all_waters = np.where(drive.swapper.stateVector == 0)[0]
            all_cations = np.where(drive.swapper.stateVector == 1)[0]
            all_anions = np.where(drive.swapper.stateVector == 2)[0]


            # This code should only perform wat2cat OR wat2ani, not both.
            # The sanity check should prevent the same waters/ions from being selected twice.
            # individual types of swaps should be completely independent for the purpose of calculating
            # the proposal probabilities.

            if swaps['wat2cat'] > 0:
                for water_index in np.random.choice(a=all_waters, size=swaps['wat2cat'], replace=False):
                    saltswap_residue_indices.append(water_index)
                    saltswap_parameter_pairs.append(tuple([water_params, cation_params]))

                # Forward: choose m water to change into cations, probability of one pick is
                # 1.0 / (n_water choose m); e.g. from all waters select m (the wat2cat count).
                log_p_forward = -np.log(comb(all_waters.size, swaps['wat2cat'], exact=True))
                # Reverse: choose m cations to change into water, probability of one pick is
                # 1.0 / (n_cation + m choose m); e.g. from current cations plus m (the wat2cat count), select m
                log_p_reverse = -np.log(comb(all_cations.size + swaps['wat2cat'], swaps['wat2cat'], exact=True))
                log_ratio += (log_p_reverse - log_p_forward)

            if swaps['wat2ani'] > 0:
                for water_index in np.random.choice(a=all_waters, size=swaps['wat2ani'], replace=False):
                    saltswap_residue_indices.append(water_index)
                    saltswap_parameter_pairs.append(tuple([water_params, anion_params]))

                # Forward: probability of one pick is
                # 1.0 / (n_water choose m); e.g. from all waters select m (the wat2ani count).
                log_p_forward = -np.log(comb(all_waters.size, swaps['wat2ani'], exact=True))
                # Reverse: probability of one pick is
                # 1.0 / (n_anion + m choose m); e.g. from all current anions plus m (the wat2ani count), select m
                log_p_reverse = -np.log(comb(all_anions.size + swaps['wat2ani'], swaps['wat2ani'], exact=True))
                log_ratio += (log_p_reverse - log_p_forward)

            if swaps['cat2wat'] > 0:
                for cation_index in np.random.choice(a=all_cations, size=swaps['cat2wat'], replace=False):
                    saltswap_residue_indices.append(cation_index)
                    saltswap_parameter_pairs.append(tuple([cation_params, water_params]))

                # Forward: choose m cations to change into water, probability of one pick is
                # 1.0 / (n_cations choose m); e.g. from all cations select m (the cat2wat count).
                log_p_forward = -np.log(comb(all_cations.size, swaps['cat2wat'], exact=True))
                # Reverse: choose m water to change into cations, probability of one pick is
                # 1.0 / (n_water + m choose m); e.g. from current waters plus m (the ani2wat count), select m
                log_p_reverse = -np.log(comb(all_cations.size + swaps['cat2wat'], swaps['cat2wat'], exact=True))
                log_ratio += (log_p_reverse - log_p_forward)

            if swaps['ani2wat'] > 0:
                for anion_index in np.random.choice(a=all_anions, size=swaps['ani2wat'], replace=False):
                    saltswap_residue_indices.append(anion_index)
                    saltswap_parameter_pairs.append(tuple([anion_params, water_params]))

                # Forward: probability of one pick is
                # 1.0 / (n_anions choose m); e.g. from all anions select m (the ani2wat count).
                log_p_forward = -np.log(comb(all_anions.size, swaps['ani2wat'], exact=True))
                # Reverse: probability of one pick is
                # 1.0 / (n_water + m choose m); e.g. from water plus m (the ani2wat count), select m
                log_p_reverse = -np.log(comb(all_waters.size + swaps['ani2wat'], swaps['ani2wat'], exact=True))
                log_ratio += (log_p_reverse - log_p_forward)

        return saltswap_residue_indices, saltswap_parameter_pairs, log_ratio
