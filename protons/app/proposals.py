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
from scipy.special import logsumexp
from typing import Dict, Tuple, Callable, List, Optional
from lxml import etree
from saltswap.wrappers import Swapper


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


class SaltSwapProposal(metaclass=ABCMeta):
    """Base class defining interface for proposing ion swaps."""

    @abstractmethod
    def propose_swaps(
        self, swapper: Swapper, delta_cations: int, delta_anions: int
    ) -> Tuple[List[int], List[Tuple[int, int]], float]:
        """Abstract method,

        Parameters
        ----------
        swapper - saltswap swapper object associated with the simulations
        delta_cations - number of cations to add/remove
        delta_anions - number of anions to add/remove


        Returns
        -------
        list of state vector elements being updated
        List of the corresponding state updates (from state, to state), 0 for water 1 for cation 2 for anion.
        The log_probability of the update."""
        return list(), list(), 0.0


class UniformSwapProposal(SaltSwapProposal):
    """This class can select ions based on specification and returns the probability of swapping."""

    def __init__(self, cation_coefficient: float = 0.5):
        """Instantiate a UniformSwapProposal.

        Parameters
        ----------
        cation_coefficient - optional. Must be between 0 and 1, default 0.5.
            The fraction of the chemical potential of a water pair -> salt pair transformation that is attributed to the
            cation -> water transformation.
        """

        if not 0.0 <= cation_coefficient <= 1.0:
            raise ValueError("The cation coefficient should be between 0 and 1.")

        self._cation_weight = cation_coefficient
        self._anion_weight = 1.0 - cation_coefficient

    def propose_swaps(
        self, swapper: Swapper, delta_cations: int, delta_anions: int
    ) -> Tuple[List[int], List[Tuple[int, int]], float]:
        """Propose ions/waters to swap.

        Parameters
        ----------
        swapper - saltswap swapper object associated with the simulations
        delta_cations - number of cations to add/remove
        delta_anions - number of anions to add/remove


        Returns
        -------
        list of state vector elements being updated
        List of the corresponding state updates (from state, to state), 0 for water 1 for cation 2 for anion.
        The log_probability of the update.
        """
        all_waters = np.where(swapper.stateVector == 0)[0]
        all_cations = np.where(swapper.stateVector == 1)[0]
        all_anions = np.where(swapper.stateVector == 2)[0]

        saltswap_residue_indices: List[int] = list()
        saltswap_state_pairs: List[Tuple[int, int]] = list()

        # Check the type of the chemical potential, and reduce units if necessary
        chem_potential = swapper.delta_chem
        if isinstance(chem_potential, unit.Quantity):
            chem_potential /= swapper.kT
            if unit.is_quantity(chem_potential):
                raise ValueError(
                    "The chemical potential has irreducible units ({}).".format(
                        str(chem_potential.unit)
                    )
                )

        log_ratio = 0

        # individual types of swaps should be completely independent for the purpose of calculating
        # the proposal probabilities.
        if delta_cations > 0:
            for water_index in np.random.choice(
                a=all_waters, size=abs(delta_cations), replace=False
            ):
                saltswap_residue_indices.append(water_index)
                saltswap_state_pairs.append(tuple([0, 1]))

            # Forward: choose m water to change into cations, probability of one pick is
            # 1.0 / (n_water choose m); e.g. from all waters select m (the water_to_cation count).
            log_p_forward = -np.log(comb(all_waters.size, delta_cations, exact=True))
            # Reverse: choose m cations to change into water, probability of one pick is
            # 1.0 / (n_cation + m choose m); e.g. from current cations plus m (the water_to_cation count), select m
            log_p_reverse = -np.log(
                comb(all_cations.size + delta_cations, delta_cations, exact=True)
            )
            log_ratio += log_p_reverse - log_p_forward
            # Calculate the work of transforming one water molecule into a cation
            work = chem_potential * self._cation_weight
            # Subtract the work from the acceptance probability
            log_ratio -= work

        if delta_anions > 0:
            for water_index in np.random.choice(
                a=all_waters, size=delta_anions, replace=False
            ):
                saltswap_residue_indices.append(water_index)
                saltswap_state_pairs.append(tuple([0, 2]))

            # Forward: probability of one pick is
            # 1.0 / (n_water choose m); e.g. from all waters select m (the water_to_anion count).
            log_p_forward = -np.log(comb(all_waters.size, delta_anions, exact=True))
            # Reverse: probability of one pick is
            # 1.0 / (n_anion + m choose m); e.g. from all current anions plus m (the water_to_anion count), select m
            log_p_reverse = -np.log(
                comb(all_anions.size + delta_anions, delta_anions, exact=True)
            )
            log_ratio += log_p_reverse - log_p_forward
            # Calculate the work of transforming one water into one anion
            work = chem_potential * self._anion_weight
            # Subtract the work from the acceptance probability
            log_ratio -= work

        if delta_cations < 0:
            delta_water = abs(delta_cations)
            for cation_index in np.random.choice(
                a=all_cations, size=delta_water, replace=False
            ):
                saltswap_residue_indices.append(cation_index)
                saltswap_state_pairs.append(tuple([1, 0]))

            # Forward: choose m cations to change into water, probability of one pick is
            # 1.0 / (n_cations choose m); e.g. from all cations select m (the cation_to_water count).
            log_p_forward = -np.log(comb(all_cations.size, delta_water, exact=True))
            # Reverse: choose m water to change into cations, probability of one pick is
            # 1.0 / (n_water + m choose m); e.g. from current waters plus m (the anion_to_water count), select m
            log_p_reverse = -np.log(
                comb(all_waters.size + delta_water, delta_water, exact=True)
            )
            log_ratio += log_p_reverse - log_p_forward
            # Calculate the work of transforming one cation into one water molecule
            work = -chem_potential * self._cation_weight
            # Subtract the work from the acceptance probability
            log_ratio -= work

        if delta_anions < 0:
            delta_water = abs(delta_anions)
            for anion_index in np.random.choice(
                a=all_anions, size=delta_water, replace=False
            ):
                saltswap_residue_indices.append(anion_index)
                saltswap_state_pairs.append(tuple([2, 0]))

            # Forward: probability of one pick is
            # 1.0 / (n_anions choose m); e.g. from all anions select m (the anion_to_water count).
            log_p_forward = -np.log(comb(all_anions.size, delta_water, exact=True))
            # Reverse: probability of one pick is
            # 1.0 / (n_water + m choose m); e.g. from water plus m (the anion_to_water count), select m
            log_p_reverse = -np.log(
                comb(all_waters.size + delta_water, delta_water, exact=True)
            )
            log_ratio += log_p_reverse - log_p_forward
            # Calculate the work of transforming one anion into water based on the chemical potential
            work = -chem_potential * self._anion_weight
            # Subtract the work from the acceptance probability
            log_ratio -= work

        return saltswap_residue_indices, saltswap_state_pairs, log_ratio


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
    dihedralforceindex = None
    # Number of different phi angles to propose for moves
    num_phi_proposals = 50

    def __init__(self):
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
        self.HO = None
        # Hydroxyl oxygen
        self.OH = None
        # Carbonyl oxygen
        self.OC = None
        # The carbons are used as reference points for reflection
        self.CO = None
        self.R = None

        # All atoms that this class may decide to move
        self.movable = []
        # The parameters for angles
        self.angles = []
        # The parameters for dihedrals
        self.dihedrals = []

        # Cached arrays for reuse to prevent copying
        self._position_proposals = None
        self._log_weights = None

    @classmethod
    def from_system(cls, system: openmm.System, indices: Dict[str, int]):
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
        obj = cls()
        # Hydroxyl hydrogen
        obj.HO = indices["HO"]
        # Hydroxyl oxygen
        obj.OH = indices["OH"]
        # Carbonyl oxygen
        obj.OC = indices["OC"]
        # The carbons are used as reference points for reflection
        obj.CO = indices["CO"]
        obj.R = indices["R"]

        # All atoms that this class may decide to move
        obj.movable = [indices["OC"], indices["OH"], indices["HO"]]
        # The parameters for angles
        obj.angles = []
        # The parameters for dihedrals
        obj.dihedrals = []

        # Instantiate the class variable
        # This is to keep track of angle and torsion force indices for all future instances
        if (
            COOHDummyMover.angleforceindex is None
            or COOHDummyMover.dihedralforceindex is None
        ):
            for force_index in range(system.getNumForces()):
                force = system.getForce(force_index)
                if force.__class__.__name__ == "HarmonicAngleForce":
                    COOHDummyMover.angleforceindex = force_index
                elif force.__class__.__name__ == "PeriodicTorsionForce":
                    COOHDummyMover.dihedralforceindex = force_index
            if COOHDummyMover.angleforceindex is None:
                raise RuntimeError(
                    "{} requires the system to have a HarmonicAngleForce!".format(
                        COOHDummyMover.__name__
                    )
                )
            if COOHDummyMover.dihedralforceindex is None:
                raise RuntimeError(
                    "{} requires the system to have a PeriodicTorsionForce!".format(
                        COOHDummyMover.__name__
                    )
                )

        angleforce = system.getForce(COOHDummyMover.angleforceindex)
        torsionforce = system.getForce(COOHDummyMover.dihedralforceindex)

        # Loop through and collect all angle energy terms that include moving atoms
        for angle_index in range(angleforce.getNumAngles()):
            *particles, theta0, k = angleforce.getAngleParameters(angle_index)
            if any(particle in obj.movable for particle in particles):
                # Energy function for this angle.
                params = [k._value, theta0._value, *particles]
                log.debug("Found this COOH angle: %s", params)
                obj.angles.append(params)

        # Loop through and collect all torsion energy terms that include moving atoms
        for torsion_index in range(torsionforce.getNumTorsions()):
            *particles, n, theta0, k = torsionforce.getTorsionParameters(torsion_index)
            if any(particle in obj.movable for particle in particles):
                # Energy function for this dihedral.
                params = [k._value, n, theta0._value, *particles]
                log.debug("Found this COOH dihedral: %s", params)
                obj.dihedrals.append(params)
        return obj

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
        theta = COOHDummyMover.bond_angle(
            positions[particle1], positions[particle2], positions[particle3]
        )
        log.debug("Angle for %i %i %i: %f", particle1, particle2, particle3, theta)
        return 0.5 * k * (theta - theta0) ** 2

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

        theta = COOHDummyMover.dihedral_angle(
            positions[particle1],
            positions[particle2],
            positions[particle3],
            positions[particle4],
        )
        log.debug(
            "Angle for %i %i %i %i: %f",
            particle1,
            particle2,
            particle3,
            particle4,
            theta,
        )
        return k * (1 + math.cos(n * theta - theta0))

    def log_probability(self, positions: np.ndarray, calc_angle=False) -> float:
        """Return the log probability of the angles and dihedrals for a given set of positions."""
        if calc_angle:
            e_angles = [
                COOHDummyMover.e_angle(positions, *params) for params in self.angles
            ]
        else:
            e_angles = [0.0]
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
        dot = np.dot(v1, v2)
        y = dot / (n_v1 * n_v2)

        # Limit to the domain of the arccos to deal with float precision issues.
        if y > 1.0:
            y = 1.0
        elif y < -1.0:
            y = -1.0

        return np.arccos(y)

    @staticmethod
    def plane_norm(p1: np.ndarray, p2: np.ndarray, p3: np.ndarray):
        return np.cross(p2 - p1, p3 - p1)

    @staticmethod
    def bond_angle(p1, p2, p3):
        return COOHDummyMover.angle_between_vectors(p1 - p2, p3 - p2)

    @staticmethod
    def dihedral_angle(p1, p2, p3, p4):
        plane1 = COOHDummyMover.plane_norm(p1, p2, p3)
        plane2 = COOHDummyMover.plane_norm(p4, p3, p2)
        return COOHDummyMover.angle_between_vectors(plane1, plane2)

    @staticmethod
    def rodrigues_rotation_vectorized(
        theta: np.ndarray, k: np.ndarray, v: np.ndarray
    ) -> np.ndarray:
        """Return rotated vector v around axis k, using angle theta.

        Note
        ----
        Ensure k is a unit vector
        """

        a = v[:, np.newaxis] * np.cos(theta)
        b = np.cross(k, v)[:, np.newaxis] * np.sin(theta)
        c = (k * np.dot(k, v))[:, np.newaxis] * (1 - np.cos(theta))
        return a + b + c

    @staticmethod
    def rodrigues_rotation(theta: float, k: np.ndarray, v: np.ndarray) -> np.ndarray:
        """Return rotated vector v around axis k, using angle theta.

        Note
        ----
        Ensure k is a unit vector
        """

        return (
            v * np.cos(theta)
            + (np.cross(k, v) * np.sin(theta))
            + (k * np.dot(k, v) * (1 - np.cos(theta)))
        )

    @staticmethod
    def internal_to_cartesian(
        r: float,
        theta: float,
        phi: np.ndarray,
        pos2: np.ndarray,
        pos3: np.ndarray,
        pos4: np.ndarray,
    ) -> np.ndarray:
        """Find the xyz coordinates of atom 1 in the dihedral 1-2-3-4 based on internal coordinates, and cartesian
        coordinates of the other atoms

        Parameters
        ----------
        r - distance between atoms 1-2
        theta - angle between 1-2-3
        phi - array of dihedral angles between 1-2 and 3-4
        pos2 - position (x,y,z) of atom 2
        pos3 - position (x,y,z) of atom 3
        pos4 - position (x,y,z) of atom 4

        Returns
        -------
        pos1 -  the position of atom 1 in the dihedral 1-2-3-4
            [xyz, proposal]

        Notes
        -----
        Based on https://github.com/choderalab/perses/blob/c9d5f0b14355da9cd58221f28b384ea5f215f5aa/perses/rjmc/coordinate_numba.py

        """
        if not np.isscalar(r):
            raise ValueError("r should be a scalar")
        if not np.isscalar(theta):
            raise ValueError("theta should be a scalar")

        bond23_vec = (pos3 - pos2) / np.linalg.norm(pos3 - pos2)
        bond34_vec = (pos3 - pos4) / np.linalg.norm(pos3 - pos4)

        angle234_vec = np.cross(bond23_vec, bond34_vec)
        angle234_vec /= np.linalg.norm(angle234_vec)

        bond = r * bond23_vec
        bond_plus_angle = COOHDummyMover.rodrigues_rotation(theta, angle234_vec, bond)
        bond_plus_angle_plus_torsion = COOHDummyMover.rodrigues_rotation_vectorized(
            phi, bond23_vec, bond_plus_angle
        )
        pos1 = pos2[:, np.newaxis] + bond_plus_angle_plus_torsion
        return pos1.T

    def propose_configurations(
        self,
        current_positions: np.ndarray,
        proposed_positions: np.ndarray,
        log_weights: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """

        Parameters
        ----------
        current_positions - numpy array of current positions
            Shape dimensions [atom, xyz]
        proposed_positions - numpy array for new positions
            Shape dimensions [proposal, atom, xyz], where proposal dimension is an odd number
            Will get modified in place
        log_weights - numpy array for log_weights
            Will be modified in place dimensions should match proposed_positions

        Returns
        -------
        (proposed_positions, log_weights)
            positions have dimensions [proposal, atom, xyz]
            log_weights have dimensions [proposal]

            the first entry in proposal dimension is the old configuration
        """
        if proposed_positions.shape[0] % 2 != 1:
            raise ValueError(
                "Proposed positions needs to have an odd-sized first dimension"
            )

        elif log_weights.size != proposed_positions.shape[0]:
            raise ValueError(
                "Weights need to have the same dimension as the first dimension of proposed_positions."
            )

        hydroxy_hydrogen = current_positions[self.HO]
        hydroxy_oxygen = current_positions[self.OH]
        carbonyl_oxygen = current_positions[self.OC]
        carbon = current_positions[self.CO]
        substituent = current_positions[self.R]

        # Bond length, constrained in proposal
        r = np.linalg.norm(hydroxy_oxygen - hydroxy_hydrogen)
        # Bond angle, constrained in proposal
        theta = COOHDummyMover.bond_angle(hydroxy_hydrogen, hydroxy_oxygen, carbon)

        num_phi = COOHDummyMover.num_phi_proposals
        number_of_proposals = COOHDummyMover.num_phi_proposals * 2

        # Duplicate all positions
        proposed_positions[:, :, :] = current_positions

        phi_angles = np.linspace(-np.pi, np.pi, num_phi, endpoint=False)

        # New phi on same oxygen
        proposed_positions[
            1 : 1 + num_phi, self.HO, :
        ] = COOHDummyMover.internal_to_cartesian(
            r, theta, phi_angles, hydroxy_oxygen, carbon, carbonyl_oxygen
        )

        # New phi on new oxygen
        proposed_positions[
            1 + num_phi :, self.HO, :
        ] = COOHDummyMover.internal_to_cartesian(
            r, theta, phi_angles, carbonyl_oxygen, carbon, hydroxy_oxygen
        )
        proposed_positions[1 + num_phi :, self.OH, :] = carbonyl_oxygen[:, np.newaxis].T
        proposed_positions[1 + num_phi :, self.OC, :] = hydroxy_oxygen[:, np.newaxis].T

        for i in range(number_of_proposals):
            log_weights[i] = self.log_probability(proposed_positions[i + 1, :, :])

        return proposed_positions, log_weights

    def to_xml(self):
        """Return an xml representation of the dummy mover."""
        tree = etree.fromstring(
            '<COOHDummyMover OH="{}" HO="{}" OC="{}" CO="{}" R="{}"/>'.format(
                self.OH, self.HO, self.OC, self.CO, self.R
            )
        )
        tree.set("AngleForceIndex", str(self.angleforceindex))
        tree.set("DihedralForceIndex", str(self.dihedralforceindex))

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

    @classmethod
    def from_xml(cls, xmltree: etree.Element):
        """Instantiate a COOHDummyMover class from an lxml Element."""
        obj = cls()
        if not xmltree.tag == "COOHDummyMover":
            raise ValueError(
                "Wrong xml element provided, was expecting a COOHDummyMover"
            )
        obj.OH = int(xmltree.get("OH"))
        obj.HO = int(xmltree.get("HO"))
        obj.OC = int(xmltree.get("OC"))
        obj.CO = int(xmltree.get("CO"))
        obj.R = int(xmltree.get("R"))
        cls.angleforceindex = int(xmltree.get("AngleForceIndex"))
        cls.dihedralforceindex = int(xmltree.get("DihedralForceIndex"))

        for angle in xmltree.xpath("//Angle"):
            angle_params = []
            for param, ptype in [
                ("k", float),
                ("theta0", float),
                ("particle1", int),
                ("particle2", int),
                ("particle3", int),
            ]:
                angle_params.append(ptype(angle.get(param)))
            obj.angles.append(angle_params)

        for dihedral in xmltree.xpath("//Dihedral"):
            dihedral_params = []
            for param, ptype in [
                ("k", float),
                ("n", int),
                ("theta0", float),
                ("particle1", int),
                ("particle2", int),
                ("particle3", int),
                ("particle4", int),
            ]:
                dihedral_params.append(ptype(dihedral.get(param)))
            obj.dihedrals.append(dihedral_params)

        return obj

    def random_move(self, current_positions):
        """Use importance sampling to propose a new position."""
        num_proposals = 1 + (2 * COOHDummyMover.num_phi_proposals)
        if self._position_proposals is None:
            self._position_proposals = np.empty(
                [num_proposals, *(current_positions.shape)]
            )
            self._log_weights = np.empty([num_proposals])

        self.propose_configurations(
            current_positions, self._position_proposals, self._log_weights
        )
        # Draw a new configuration.
        # Accept probability is 1 because weights are equal to -u(x)
        log_normalizing_constant = logsumexp(self._log_weights)
        chosen = np.random.choice(
            np.arange(num_proposals),
            p=np.exp(self._log_weights) / np.exp(log_normalizing_constant),
        )

        return self._position_proposals[chosen, :, :], 0.0
