# coding=utf-8
"""Residue selection moves for protons MC"""
from abc import ABCMeta, abstractmethod
from .logger import log
import copy
import random
import numpy as np
import math
from simtk import unit, openmm
from scipy.misc import comb
from typing import Dict, Tuple, Callable, List, Optional
from lxml import etree


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
            logp_ratio_proposal - float (probability of reverse proposal)/(probability of forward proposal)
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
            titration_state_index = random.choice(
                range(drive.get_num_titration_states(titration_group_index))
            )
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
        if (len(residue_pool_indices) > 1) and (
            random.random() < self.simultaneous_proposal_probability
        ):
            ndraw = 2

        log.debug("Updating %i residues.", ndraw)

        titration_group_indices = random.sample(residue_pool_indices, ndraw)
        # Select new titration states.
        for titration_group_index in titration_group_indices:
            # Choose a titration state with uniform probability (even if it is the same as the current state).
            titration_state_index = random.choice(
                range(drive.get_num_titration_states(titration_group_index))
            )
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
            titration_state_index = random.choice(
                range(drive.get_num_titration_states(titration_group_index))
            )
            final_titration_states[titration_group_index] = titration_state_index

        return final_titration_states, titration_group_indices, 0.0


class SaltSwapProposal:
    """This class is a baseclass for selecting water molecules or ions to maintain charge neutrality."""

    def propose_swaps(self, drive, initial_charge: int, final_charge: int):
        """Select a series of water molecules/ions to swap.

        Parameters
        ----------

        drive - a ProtonDrive object with a swapper attached.
        initial_charge - the total charge of the changing residue before the state change
        total_charge - the total charge of the changing residue after the state change

        Please note
        -----------
        The net charge difference will have the opposite charge of the ions that will be added to the system,
        such that the resulting total charge change will be 0.

        Returns
        -------
        list(int) - indices of water molecule in saltswap that are swapped
        list(tuple(int,int),) -  list of tuples with (initial, final) states of the water molecule/ion as int, one tuple for each requested swap
            By saltswap convention
             - 0 is water
             - 1 is cation
             - 2 is anion
        float - log (probability of reverse proposal)/(probability of forward proposal)
        """

        return list(), list(), float()

    @staticmethod
    def _validate_swaps(swaps):
        """Perform sanity checks. Swaps should only add charge in one direction.

        Raises
        ------
        RuntimeError - in case conflicting swaps occur at the same time.
            The error message will detail what the conflict is.
        """

        if swaps["water_to_cation"] != 0 and swaps["water_to_anion"] != 0:
            raise RuntimeError(
                "Opposing charge ions are added. This is a bug in the code."
            )
        elif swaps["cation_to_water"] != 0 and swaps["anion_to_water"] != 0:
            raise RuntimeError(
                "Opposing charge ions are removed. This is a bug in the code."
            )
        elif swaps["cation_to_water"] != 0 and swaps["water_to_cation"] != 0:
            raise RuntimeError(
                "Cations are being added and removed at the same time. This is a bug in the code."
            )
        elif swaps["anion_to_water"] != 0 and swaps["water_to_anion"] != 0:
            raise RuntimeError(
                "Anions are being added and removed at the same time. This is a bug in the code."
            )


class OneDirectionChargeProposal(SaltSwapProposal):
    """Swaps ions in a way that does not create opposite charges during the alchemical protocol.
    This is an implementation of the method outlined in Chen and Roux 2015
    """

    def __init__(self, cation_coefficient: float = 0.5, err_on_depletion: bool = True):
        """Instantiate a UniformSwapProposal.

        Parameters
        ----------
        cation_coefficient - optional. Must be between 0 and 1, default 0.5.
            The fraction of the chemical potential of a water pair -> salt pair transformation that is attributed to the
            cation -> water transformation.
        err_on_depletion - optional.
            If ions get depleted, raise an error. If false, add opposite charge ion to fix.


        """
        if not 0.0 <= cation_coefficient <= 1.0:
            raise ValueError("The cation coefficient should be between 0 and 1.")

        self._cation_weight = cation_coefficient
        self._anion_weight = 1.0 - cation_coefficient

        self._err_on_depletion = err_on_depletion

    def select_ions(
        self,
        chem_potential: float,
        drive,
        log_ratio: float,
        saltswap_residue_indices: list,
        saltswap_state_pairs: list,
        swaps: dict,
    ):
        """

        Parameters
        ----------
        chem_potential - used to calculate the probability of the swap
        drive - ProtonDrive object that has a swapper attached
        log_ratio - starting log_ratio estimate
        saltswap_residue_indices - list to append residue indices to
        saltswap_state_pairs - list to append initial and final states of residues to
        swaps - dict of the type of swaps to perform

        Returns
        -------

        """
        all_waters = np.where(drive.swapper.stateVector == 0)[0]
        all_cations = np.where(drive.swapper.stateVector == 1)[0]
        all_anions = np.where(drive.swapper.stateVector == 2)[0]
        # This code should only perform water_to_cation OR water_to_anion, not both.
        # The sanity check should prevent the same waters/ions from being selected twice.
        # individual types of swaps should be completely independent for the purpose of calculating
        # the proposal probabilities.
        if swaps["water_to_cation"] > 0:
            for water_index in np.random.choice(
                a=all_waters, size=swaps["water_to_cation"], replace=False
            ):
                saltswap_residue_indices.append(water_index)
                saltswap_state_pairs.append(tuple([0, 1]))

            # Forward: choose m water to change into cations, probability of one pick is
            # 1.0 / (n_water choose m); e.g. from all waters select m (the water_to_cation count).
            log_p_forward = -np.log(
                comb(all_waters.size, swaps["water_to_cation"], exact=True)
            )
            # Reverse: choose m cations to change into water, probability of one pick is
            # 1.0 / (n_cation + m choose m); e.g. from current cations plus m (the water_to_cation count), select m
            log_p_reverse = -np.log(
                comb(
                    all_cations.size + swaps["water_to_cation"],
                    swaps["water_to_cation"],
                    exact=True,
                )
            )
            log_ratio += log_p_reverse - log_p_forward
            # Calculate the work of transforming one water molecule into a cation
            work = chem_potential * self._cation_weight
            # Subtract the work from the acceptance probability
            log_ratio -= work
        if swaps["water_to_anion"] > 0:
            for water_index in np.random.choice(
                a=all_waters, size=swaps["water_to_anion"], replace=False
            ):
                saltswap_residue_indices.append(water_index)
                saltswap_state_pairs.append(tuple([0, 2]))

            # Forward: probability of one pick is
            # 1.0 / (n_water choose m); e.g. from all waters select m (the water_to_anion count).
            log_p_forward = -np.log(
                comb(all_waters.size, swaps["water_to_anion"], exact=True)
            )
            # Reverse: probability of one pick is
            # 1.0 / (n_anion + m choose m); e.g. from all current anions plus m (the water_to_anion count), select m
            log_p_reverse = -np.log(
                comb(
                    all_anions.size + swaps["water_to_anion"],
                    swaps["water_to_anion"],
                    exact=True,
                )
            )
            log_ratio += log_p_reverse - log_p_forward
            # Calculate the work of transforming one water into one anion
            work = chem_potential * self._anion_weight
            # Subtract the work from the acceptance probability
            log_ratio -= work
        if swaps["cation_to_water"] > 0:
            for cation_index in np.random.choice(
                a=all_cations, size=swaps["cation_to_water"], replace=False
            ):
                saltswap_residue_indices.append(cation_index)
                saltswap_state_pairs.append(tuple([1, 0]))

            # Forward: choose m cations to change into water, probability of one pick is
            # 1.0 / (n_cations choose m); e.g. from all cations select m (the cation_to_water count).
            log_p_forward = -np.log(
                comb(all_cations.size, swaps["cation_to_water"], exact=True)
            )
            # Reverse: choose m water to change into cations, probability of one pick is
            # 1.0 / (n_water + m choose m); e.g. from current waters plus m (the anion_to_water count), select m
            log_p_reverse = -np.log(
                comb(
                    all_cations.size + swaps["cation_to_water"],
                    swaps["cation_to_water"],
                    exact=True,
                )
            )
            log_ratio += log_p_reverse - log_p_forward
            # Calculate the work of transforming one cation into one water molecule
            work = -chem_potential * self._cation_weight
            # Subtract the work from the acceptance probability
            log_ratio -= work
        if swaps["anion_to_water"] > 0:
            for anion_index in np.random.choice(
                a=all_anions, size=swaps["anion_to_water"], replace=False
            ):
                saltswap_residue_indices.append(anion_index)
                saltswap_state_pairs.append(tuple([2, 0]))

            # Forward: probability of one pick is
            # 1.0 / (n_anions choose m); e.g. from all anions select m (the anion_to_water count).
            log_p_forward = -np.log(
                comb(all_anions.size, swaps["anion_to_water"], exact=True)
            )
            # Reverse: probability of one pick is
            # 1.0 / (n_water + m choose m); e.g. from water plus m (the anion_to_water count), select m
            log_p_reverse = -np.log(
                comb(
                    all_waters.size + swaps["anion_to_water"],
                    swaps["anion_to_water"],
                    exact=True,
                )
            )
            log_ratio += log_p_reverse - log_p_forward
            # Calculate the work of transforming one anion into water based on the chemical potential
            work = -chem_potential * self._anion_weight
            # Subtract the work from the acceptance probability
            log_ratio -= work
        return log_ratio

    def propose_swaps(self, drive, initial_charge: int, final_charge: int):
        """Select a series of water molecules/ions to swap uniformly from all waters/ions.

            Parameters
            ----------

            drive - a ProtonDrive object with a swapper attached.
            initial_charge - the total charge of the changing residue before the state change
            total_charge - the total charge of the changing residue after the state change

            Returns
            -------
            list(int) - indices of water molecule in saltswap that are swapped
            list(tuple(int,int),) -  list of tuples with (initial, final) states of the water molecule/ion as dict, one tuple for each requested swap
                By saltswap convention:
                - 0 is water
                - 1 is cation
                - 2 is anion

            float - log (probability of reverse proposal)/(probability of forward proposal)
        """
        net_charge_difference = final_charge - initial_charge
        # Defaults. If no swaps are necessary, this will be all that is needed.
        saltswap_residue_indices = list()
        saltswap_state_pairs = list()
        log_ratio = 0.0  # fully symmetrical proposal if no swaps occur.

        # the chemical potential for switching two water molecules into cation + anion
        chem_potential = drive.swapper.delta_chem

        # If no cost is supplied, use the supplied chemical potential

        # Check the type of the chemical potential, and reduce units if necessary
        if isinstance(chem_potential, unit.Quantity):
            chem_potential *= drive.beta
            if unit.is_quantity(chem_potential):
                raise ValueError(
                    "The chemical potential has irreducible units ({}).".format(
                        str(chem_potential.unit)
                    )
                )

        # If swaps are needed
        if net_charge_difference != 0:

            # There is a net charge difference, find which swaps are necessary to compute.
            swaps = self._select_swaps_chenroux(initial_charge, final_charge)

            # Apply sanity checks
            SaltSwapProposal._validate_swaps(swaps)

            log_ratio = self.select_ions(
                chem_potential,
                drive,
                log_ratio,
                saltswap_residue_indices,
                saltswap_state_pairs,
                swaps,
            )

        return saltswap_residue_indices, saltswap_state_pairs, log_ratio

    @staticmethod
    def _select_swaps_chenroux(initial_charge: int, final_charge: int) -> dict:
        """Select ions or water molecule swap procedure to facilitate maintaining charge neutrality while changing protonation states.

        Notes
        -----
        Based on the method from Chen and Roux 2015.

        Parameters
        ----------
        initial_charge - the initial charge of the residue
        final_charge - the state of the residue after changing protonation states

        Returns
        -------
        dict(water_to_cation, water_to_anion, cation_to_water, anion_to_water)

        Raises
        ------
        RuntimeError - if the swaps cannot be resolved within 1000 iterations
            Usually this means there is a bug in the algorithm, or charges may
            have been passed in as float (which was a bug in the past).

        """

        # Note that we don't allow for direct transitions between ions of different charge.
        swaps = dict(
            water_to_cation=0, water_to_anion=0, cation_to_water=0, anion_to_water=0
        )
        charge_to_counter = final_charge - initial_charge

        counter = 0
        while abs(charge_to_counter) > 0:
            # The protonation state change annihilates a positive charge
            if (initial_charge > 0 >= final_charge) or (
                0 < final_charge < initial_charge
            ):
                swaps["water_to_cation"] += 1
                charge_to_counter += 1
                initial_charge -= 1  # One part of the initial charge has been countered

            # The protonation state change annihilates a negative charge
            elif initial_charge < 0 <= final_charge or (
                0 > final_charge > initial_charge
            ):
                swaps["water_to_anion"] += 1
                charge_to_counter -= 1
                initial_charge += 1
            # The protonation state change adds a negative charge
            elif initial_charge == 0 > final_charge or (
                0 > initial_charge > final_charge
            ):
                swaps["anion_to_water"] += 1
                charge_to_counter += 1
                initial_charge -= 1
            # The protonation state adds a positive charge
            elif (initial_charge == 0 < final_charge) or (
                0 < initial_charge < final_charge
            ):
                swaps["cation_to_water"] += 1
                charge_to_counter -= 1
                initial_charge += 1
            else:
                raise ValueError("Impossible scenario reached.")

            counter += 1
            if counter > 1000:
                raise RuntimeError(
                    "Infinite while loop predicted for salt resolution. Bailing out."
                )
        return swaps


class COOHDummyMover:
    """This class performs a deterministic moves of the dummy atom in a carboxylic acid among symmetrical directions.

    Notes
    -----
    This class performs symmetric moves (meaning `f(f(x)) = x`)  on the dummy atom of a deprotonated carboxylic acid.

    Two moves are available:
        mirror_oxygens - exchange the two oxygen positions, and mirror + move the dummy hydrogen along to maintain the bond length.
        mirror_syn_anti - this mirrors the location of the hydrogen between a syn/anti conformation.

    Due the the assymetry of a real carboxylic acid conformation, mirror_oxygens results in the C-O-H angle terms changing.
    It also can change the dihedral energy terms regarding the hydrogen and oxygen atoms.
    Likewise mirror_syn_anti could potentially change the dihedral energy of the hydrogen.
    Bond lengths are NOT affected.

    To facilitate fast computation, the following approximations are made in the energy computation:

    - There is negligible non-bonded interaction between the hydrogen and the rest of the system.
    - Both oxygen atoms have negligible differences in non-bonded parameters.

    If you don't want to make these approximations, you can still feed the new positions into OpenMM and recompute the full energy.

    Using these approximations, it means the only energy terms that have to be re-evaluated are the few angle and dihedral terms that involve the
    moving atoms. This is easily and efficiently done outside of OpenMM.

    This is typically true for a deprotonated carboxylic acid with a dummy hydrogen. Be careful if you want to repurpose this
    class for other classes of molecules.

    Attributes
    ----------
    HO - index in the system of the hydroxyl hydrogen.
    OH - index in the system of the hydroxyl oxygen
    OC - index in the system of the carbonyl oxygen
    CO - index in the system of the carbonyl carbon
    R  - index in the system of the atom "X"  this COOH group is connected to.

    movable - indices of the atoms that can move (OH, HO, OC)
    angles - list of angle parameters that go into the energy function
    dihedrals - list of dihedral parameters that go into the energy function
    """

    # Keep track of which force index corresponds to angles and dihedral between instances
    angleforceindex = None
    torsionforceindex = None

    def __init__(self, system: openmm.System, indices: Dict[str, int]):
        """Instantiate a COOHDummyMover for a single C-COOH moiety in your system.

        Parameters
        ----------
        system - The OpenMM system containting the COOH moiety
        indices - a dictionary labeling the indices of the C-COOH atoms with keys:
            HO - index in the system of the hydroxyl hydrogen.
            OH - index in the system of the hydroxyl oxygen
            OC - index in the system of the carbonyl oxygen
            CO - index in the system of the carbonyl carbon
            R -  index in the system of the atom "R"  this COOH group is connected to.
        """

        # Hydroxyl hydrogen
        self.HO = indices["HO"]
        # Hydroxyl oxygen
        self.OH = indices["OH"]
        # Carbonyl oxygen
        self.OC = indices["OC"]
        # The carbons are used as reference points for reflection
        self.CO = indices["CO"]
        self.R = indices["R"]

        # All atoms that this class may decide to move
        self.movable = [indices["OC"], indices["OH"], indices["HO"]]
        # The parameters for angles
        self.angles = []
        # The parameters for dihedrals
        self.dihedrals = []

        # Instantiate the class variable
        # This is to keep track of angle and torsion force indices for all future instances
        if (
            COOHDummyMover.angleforceindex is None
            or COOHDummyMover.torsionforceindex is None
        ):
            for force_index in range(system.getNumForces()):
                force = system.getForce(force_index)
                if force.__class__.__name__ == "HarmonicAngleForce":
                    COOHDummyMover.angleforceindex = force_index
                elif force.__class__.__name__ == "PeriodicTorsionForce":
                    COOHDummyMover.torsionforceindex = force_index
            if COOHDummyMover.angleforceindex is None:
                raise RuntimeError(
                    "{} requires the system to have a HarmonicAngleForce!".format(
                        COOHDummyMover.__name__
                    )
                )
            if COOHDummyMover.torsionforceindex is None:
                raise RuntimeError(
                    "{} requires the system to have a PeriodicTorsionForce!".format(
                        COOHDummyMover.__name__
                    )
                )

        angleforce = system.getForce(COOHDummyMover.angleforceindex)
        torsionforce = system.getForce(COOHDummyMover.torsionforceindex)

        # Loop through and collect all angle energy terms that include moving atoms
        for angle_index in range(angleforce.getNumAngles()):
            *particles, theta0, k = angleforce.getAngleParameters(angle_index)
            if any(particle in self.movable for particle in particles):
                # Energy function for this angle.
                params = [k._value, theta0._value, *particles]
                log.debug("Found this COOH angle: %s", params)
                self.angles.append(params)

        # Loop through and collect all torsion energy terms that include moving atoms
        for torsion_index in range(torsionforce.getNumTorsions()):
            *particles, n, theta0, k = torsionforce.getTorsionParameters(torsion_index)
            if any(particle in self.movable for particle in particles):
                # Energy function for this dihedral.
                params = [k._value, n, theta0._value, *particles]
                log.debug("Found this COOH dihedral: %s", params)
                self.dihedrals.append(params)
        return

    @staticmethod
    def e_angle(
        positions: np.ndarray,
        k: float,
        theta0: float,
        particle1: int,
        particle2: int,
        particle3: int,
    ) -> float:
        """Angle energy function as defined in OpenMM documentation."""
        return (
            0.5
            * k
            * (
                COOHDummyMover.angle_between_vectors(
                    positions[particle1] - positions[particle2],
                    positions[particle3] - positions[particle2],
                )
                - theta0
            )
            ** 2
        )

    @staticmethod
    def e_dihedral(
        positions: np.ndarray,
        k: float,
        n: int,
        theta0: float,
        particle1: int,
        particle2: int,
        particle3: int,
        particle4: int,
    ) -> float:
        """Dihedral energy function as defined in OpenMM documentation. Deals with proper and improper equivalently."""
        return k * (
            1
            + math.cos(
                n
                * COOHDummyMover.angle_between_vectors(
                    positions[particle1] - positions[particle2],
                    positions[particle4] - positions[particle3],
                )
                - theta0
            )
        )

    @staticmethod
    def reflect(n: np.ndarray, x0: np.ndarray, x1: np.ndarray) -> np.ndarray:
        """Reflect in a mirror with normal vector n.

        Parameters
        ----------
        n - normal vector of a mirror
        x0 - point on mirror.
        x1 - original point

        Returns
        -------
        reflection of x1
        """
        v = x0 - x1
        n /= np.linalg.norm(n)
        return v + x0 - 2 * (np.dot(v, n) * n)

    def log_probability(self, positions: np.ndarray) -> float:
        """Return the log probability of the angles and dihedrals for a given set of positions."""
        e_angles = [
            COOHDummyMover.e_angle(positions, *params) for params in self.angles
        ]
        e_dihedrals = [
            COOHDummyMover.e_dihedral(positions, *params) for params in self.dihedrals
        ]
        log.debug("COOH angle energies: %s", e_angles)
        log.debug("COOH dihedral energies: %s", e_dihedrals)
        return -1.0 * (sum(e_angles) + sum(e_dihedrals))

    @staticmethod
    def angle_between_vectors(v1: np.ndarray, v2: np.ndarray) -> float:
        """Returns the angle (in radians) between two vectors

        Parameters
        ----------
        v1 - first vector
        v2 - second vector
        """
        n_v1 = np.linalg.norm(v1)
        n_v2 = np.linalg.norm(v2)
        return np.arccos(np.dot(v1, v2) / (n_v1 * n_v2))

    def mirror_oxygens(self, positions: np.ndarray) -> Tuple[np.ndarray, float]:
        """Mirror around the CC bound to swap oxygen positions and move hydrogen along without changing bonds.

        Parameters
        ----------
        positions - array of positions of all the atoms in the openmm system

        Returns
        -------
        new_positions, log_accept_mirror

        """
        new_positions = copy.deepcopy(positions)
        hydroxyl_o = positions[self.OH]
        carbonyl_o = positions[self.OC]
        hydroxyl_h = positions[self.HO]
        carbonyl_c = positions[self.CO]
        r_group_atom = positions[self.R]

        # Mirror normal vector along c-c bond
        norm = carbonyl_c - r_group_atom
        # x0 (carbonyl c) lies on norm
        # x1 is the atom that needs to be reflected
        ho_reflection = self.reflect(norm, carbonyl_c, hydroxyl_h)
        oh_reflection = self.reflect(norm, carbonyl_c, hydroxyl_o)
        # Correct for assymmetry between oxygen atoms
        new_positions[self.HO] = ho_reflection - oh_reflection + carbonyl_o
        new_positions[self.OC] = hydroxyl_o
        new_positions[self.OH] = carbonyl_o

        new_state_probability = self.log_probability(new_positions)
        old_state_probability = self.log_probability(positions)
        logp_accept_mirror = new_state_probability - old_state_probability

        log.debug("%s", new_positions - positions)
        log.debug("E new: %.10f", new_state_probability)
        log.debug("E old: %.10f", old_state_probability)
        return new_positions, logp_accept_mirror

    def mirror_syn_anti(self, positions: np.ndarray) -> Tuple[np.ndarray, float]:
        """Mirror around the O-C bond to peform a syn/anti coordinate flip.

        Parameters
        ----------
        positions - array of positions of all the atoms in the openmm system

        Returns
        -------
        new_positions, log_accept_mirror
        """
        new_positions = copy.deepcopy(positions)
        hydroxyl_o = positions[self.OH]
        hydroxyl_h = positions[self.HO]
        carbonyl_c = positions[self.CO]

        # Mirror normal vector along O-C bond
        norm = hydroxyl_o - carbonyl_c
        # x0 (hydroxyl O) lies on norm
        # x1 is the atom that needs to be reflected
        new_positions[self.HO] = self.reflect(norm, hydroxyl_o, hydroxyl_h)

        new_state_probability = self.log_probability(new_positions)
        old_state_probability = self.log_probability(positions)
        logp_accept_mirror = new_state_probability - old_state_probability

        log.debug("%s", new_positions - positions)
        log.debug("E new: %.10f", new_state_probability)
        log.debug("E old: %.10f", old_state_probability)

        return new_positions, logp_accept_mirror

    def to_xml(self):
        """Return an xml representation of the dummy mover."""
        tree = etree.fromstring(
            '<COOHDummyMover OH="{}" HO="{}" OC="{}" CO="{}" R="{}"/>'.format(
                self.OH, self.HO, self.OC, self.CO, self.R
            )
        )
        for angle in self.angles:
            angle_xml = '<Angle k="{}" theta0="{}" particle1="{}"  particle2="{}" particle3="{}"/>'.format(
                *angle
            )
            tree.append(etree.fromstring(angle_xml))

        for dihedral in self.dihedrals:
            dihedral_xml = '<Dihedral k="{}" n="{}" theta0="{}" particle1="{}" particle2="{}" particle3="{}" particle4="{}" />'.format(
                *dihedral
            )
            tree.append(etree.fromstring(dihedral_xml))

        return etree.tostring(tree, pretty_print=True)
