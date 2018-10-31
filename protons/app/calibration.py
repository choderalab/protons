# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from .driver import NCMCProtonDrive, _SAMSState, SAMSApproach
from warnings import warn
import simtk.unit as units
from .logger import log
from scipy.special import logsumexp
kB = (1.0 * units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA).in_units_of(units.kilocalories_per_mole / units.kelvin)
from enum import Enum
from typing import Optional, Union, List


class SelfAdjustedMixtureSampler:

    """Implementation of self-adjusted mixture sampling for calibrating titratable residues or ligands.

    Attributes
    ----------
    n_adaptations : int
        Number of times the relative free energies have been adapted.

    state_counts : np.array
        Histogram of the expected weights of current states.

    References
    ----------
    .. [1] Z. Tan, Optimally adjusted mixture sampling and locally weighted histogram analysis
        DOI: 10.1080/10618600.2015.1113975

    """

    def __init__(self, driver, group_index):
        """
        Initialize a Self-adjusted mixture sampling (SAMS) simulation engine for a given
        ProtonDrive object.
        
        Parameters
        ----------
        driver : NCMCProtonDrive derived class
        group_index : int
            Index of the titration group that is being sampled.
        
        """

        # Check if driver is of the right type.
        assert issubclass(type(driver), NCMCProtonDrive)
        self.driver = driver
        self.n_adaptations = 0
        self.group_index = group_index

        target_weights = None
        for i, group in enumerate(self.driver.titrationGroups):
            for j, state in enumerate(self.driver.titrationGroups[i]):
                if target_weights is not None:
                    self.driver.titrationGroups[i].target_weights = target_weights[i]
                else:
                    self.driver.titrationGroups[i].target_weights  = [1.0 / len(self.driver.titrationGroups[i])] * len(self.driver.titrationGroups[i])

        nstates = len(self.driver.titrationGroups[group_index])
        self.state_counts = np.zeros(nstates, np.float64)
        log.debug('There are %d titration states' % nstates)

    def adapt_zetas(self, scheme='binary', b=0.85, stage="slow-gain", end_of_burnin=0):
        """
        Update the relative free energy of titration states of the specified titratable group
        using self-adjusted mixture sampling (SAMS)

        Parameters
        ----------
        scheme : str, optional (default : 'binary')
            Scheme from Tan paper ('binary' or 'global').
        b : float, optional (default : 0.85)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : str, optional (default : "slow-gain")
            Sams two-stage phase. Options : "burn-in" or "slow-gain"
        end_of_burnin: int, optional (default : 0)
            Iteration at which burn-in phase was ended.
        Returns
        -------
        target_deviation - np.array of the deviation of the sampled histogram weights from the target pi_j's

        """
        self.n_adaptations += 1

        # zeta^{t-1}
        zeta = self.get_gk(group_index=self.group_index)

        if scheme == 'binary':
            update = self._binary_update(group_index=self.group_index, b=b, stage=stage, end_of_burnin=end_of_burnin)
        elif scheme == 'global':
            update = self._global_update(group_index=self.group_index, b=b, stage=stage, end_of_burnin=end_of_burnin)
        else:
            raise ValueError("Unknown adaptation scheme: {}!".format(scheme))

        # zeta^{t-1/2}
        zeta += update
        # zeta^{t} = zeta^{t-1/2} - zeta_1^{t-1/2}
        zeta_t = zeta - zeta[0]

        # Set reference free energy based on new zeta
        self.set_gk(zeta_t, group_index=self.group_index)

        Nk = self.state_counts / self.state_counts.sum()
        target = self._get_target_weights(self.group_index)
        target_deviation = sum(abs(target - Nk))
        log.debug('Adaptation step %8d : zeta_t = %s, N_k = %s, %2f%% deviation' % (self.n_adaptations, str(zeta_t), str(Nk), target_deviation * 100))
        return target_deviation

    def set_gk(self, zetas, group_index=0):
        """
        Set g_k based on provided zetas
        Parameters
        ----------
        zetas : list of float
            Zeta values for each titration state
        group_index : int, optional
            Index of the group that needs updating, defaults to 0
        """

        for i, titr_state_zeta in enumerate(zetas):
            self.driver.titrationGroups[group_index][i].g_k = titr_state_zeta

    def get_gk(self, group_index=0):
        """Retrieve g_k/zeta for specified titratable group.

        Parameters
        ----------
        group_index : int, optional
            Index of the group that needs updating, defaults to 0

        Returns
        -------
        np.ndarray - zeta of states
        """
        zeta = np.asarray(self.driver.titrationGroups[group_index].g_k_values)
        return zeta

    def _get_target_weights(self, group_index=0):
        """Retrieve target weights pi_j for specified titratable group.
        Parameters
        ----------
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.group_index

        Returns
        -------
        np.ndarray - target population of the states.

        """
        return np.asarray(self.driver.titrationGroups[group_index].target_weights)

    def _binary_update(self, group_index=0, b=1.0, stage="slow-gain", end_of_burnin=0):
        """
        Binary update scheme (equation 9) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : str, optional (default : "burn-in")
            Sams two-stage phase. Options : "burn-in" or "slow-gain"
        end_of_burnin: int, optional (default : 0)
            Iteration at which burn-in phase was ended.
        Returns
        -------
        np.ndarray - free energy updates
        """
        # [1/pi_1...1/pi_i]
        update = np.asarray(list(map(lambda x: 1 / x, self.driver.titrationGroups[group_index].target_weights)))
        # delta(Lt)
        delta = np.zeros_like(update)
        delta[self.driver._get_titration_state(group_index)] = 1
        update *= delta
        update = np.dot(self._gain_factor(b=b, stage=stage, group_index=group_index, end_of_burnin=end_of_burnin), update)

        # Update count of current state weights.
        current_state = self.driver._get_titration_state(group_index)
        #  Use sqrt to make recent states count more
        self.state_counts[current_state] += np.sqrt(self.n_adaptations)

        return update

    def _global_update(self, b=1.0, stage="slow-gain", end_of_burnin=0, group_index=0):
        """
        Global update scheme (equation 12) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : str, optional (default : "burn-in")
            Sams two-stage phase. Options : "burn-in" or "slow-gain"
        end_of_burnin: int, optional (default : 0)
            Iteration at which burn-in phase was ended.
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.
        Returns
        -------
        np.ndarray : free energy updates
        """
        zeta = self.get_gk(group_index)
        pi_j = self._get_target_weights(group_index)
        # [1/pi_1...1/pi_i]
        update = 1.0 / pi_j
        ub_j = self.driver._get_reduced_potentials(group_index)
        # w_j(X;ζ⁽ᵗ⁻¹⁾)
        log_w_j = np.log(pi_j) - zeta - ub_j
        log_w_j -= logsumexp(log_w_j)
        w_j = np.exp(log_w_j)
        update *= w_j / pi_j
        update = np.dot(self._gain_factor(b=b, stage=stage, group_index=group_index, end_of_burnin=end_of_burnin), update)

        # Update count of current state weights.
        #  Use sqrt to make recent states count more
        self.state_counts += np.sqrt(self.n_adaptations) * w_j

        return update

    def _gain_factor(self, b=1.0, stage="slow-gain", end_of_burnin=0, group_index=0):
        """
        Two stage update scheme (equation 15) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : str, optional (default : "burn-in")
            Sams two-stage phase. Options : "burn-in" or "slow-gain"
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.
        end_of_burnin: int, optional (default : 0)
            Iteration at which burn-in phase was ended.

        Returns
        -------
        np.ndarray - gain factor matrix
        """

        if not 0.5 <= b <= 1.0:
            raise ValueError("β needs to be between 1/2 and 1")

        pi_j = self._get_target_weights(group_index)

        gain = np.zeros_like(pi_j)
        for j in range(gain.size):
            if stage == "burn-in":
                gain[j] = min(pi_j[j], 1.0/pow(self.n_adaptations, b))
            elif stage == "slow-gain":
                gain[j] = min(pi_j[j], 1.0/(self.n_adaptations - end_of_burnin + pow(end_of_burnin, b)))
            else:
                raise ValueError("Invalid SAMS adaptation stage specified %s. Choose 'burn-in' or 'slow-gain'.")

        return np.diag(gain)



class Stage(Enum):
    """Two stages of a sams run."""
    BURNIN = 0
    SLOWGAIN = 1


class UpdateRule(Enum):
    """SAMS update rule."""
    BINARY = 0
    GLOBAL = 1


class MultiSiteSAMSSampler:
    """An implementation of Self Adjusted mixture sampling that can sample multiple sites at once."""

    def __init__(self, driver: NCMCProtonDrive):
        """
        Initialize a Self-adjusted mixture sampling (SAMS) simulation engine for a given ProtonDrive object.

        Parameters
        ----------
        driver : NCMCProtonDrive derived class
        approach : The SAMS approach (SAMSApproach.ONESITE or SAMSApproach.MULTISITE)
        group_index : int
            Index of the titration group that is being sampled.

        """
        # Type checks
        assert issubclass(type(driver), NCMCProtonDrive)

        if driver.calibration_state is None:
            raise ValueError("Drive has not been prepared for calibration. Please call driver.attach_calibration.")

        self.driver = driver
        self.n_adaptations = 0
        self.approach: SAMSApproach = driver.calibration_state.approach
        self.group_index = driver.calibration_state.group_index
        self._calibration_state: _SAMSState = driver.calibration_state
        self.nstates = len(self._calibration_state.free_energies)
        self.state_counts = np.zeros(self.nstates, np.float64)
        log.debug('There are %d titration states', self.nstates)
        return

    def adapt_zetas(self, update_rule=UpdateRule.BINARY, b:float=0.85, stage: Stage=Stage.SLOWGAIN, end_of_burnin: int=0):
        """
        Update the relative free energy of titration states of the specified titratable group
        using self-adjusted mixture sampling (SAMS)

        Parameters
        ----------
        update_rule : str, optional (default : 'binary')
            Scheme from Tan paper ('binary' or 'global').
        b : float, optional (default : 0.85)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage:
            Sams two-stage phase. Options : "burn-in" or "slow-gain"
        end_of_burnin: int, optional (default : 0)
            Iteration at which burn-in phase was ended.
        Returns
        -------
        target_deviation - np.array of the deviation of the sampled histogram weights from the target pi_j's

        """
        self.n_adaptations += 1

        # zeta^{t-1}
        zeta = self._calibration_state.free_energies

        if update_rule == UpdateRule.BINARY:
            update = self._binary_update(group_index=self.group_index, b=b, stage=stage, end_of_burnin=end_of_burnin)
        elif update_rule == UpdateRule.GLOBAL:
            if self.approach is SAMSApproach.MULTISITE:
                raise NotImplementedError("Global updates only implemented for one site at this time.")
            update = self._global_update(group_index=self.group_index, b=b, stage=stage, end_of_burnin=end_of_burnin)
        else:
            raise ValueError("Unknown adaptation update_rule: {}!".format(update_rule))

        # zeta^{t-1/2}
        zeta += update
        # zeta^{t} = zeta^{t-1/2} - zeta_1^{t-1/2}
        zeta_t = zeta - zeta[0]

        # Set reference free energy based on new zeta
        self._calibration_state.free_energies=(zeta_t)

        Nk = self.state_counts / self.state_counts.sum()
        target = self._calibration_state.targets
        target_deviation = sum(abs(target - Nk))
        log.debug('Adaptation step %8d : zeta_t = %s, N_k = %s, %2f%% deviation' % (self.n_adaptations, str(zeta_t), str(Nk), target_deviation * 100))
        return target_deviation

    def _binary_update(self, group_index: int=0, b:float=1.0, stage=Stage.SLOWGAIN, end_of_burnin:int=0):
        """
        Binary update scheme (equation 9) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : Sams two-stage phase. Options : Stage.BURNIN or Stage.SLOWGAIN
        end_of_burnin: int, optional (default : 0)
            Iteration at which burn-in phase was ended.
        Returns
        -------
        np.ndarray - free energy updates
        """
        # [1/pi_1...1/pi_i]
        update = np.asarray(list(map(lambda x: 1 / x, self._calibration_state.targets)))
        # delta(Lt)
        delta = np.zeros_like(update)
        current_state = self._calibration_state.state_index(self.driver.titrationStates)
        delta[current_state] = 1
        update *= delta
        update = np.dot(self._gain_factor(b=b, stage=stage, end_of_burnin=end_of_burnin), update)

        # Update count of current state weights.
        #  Use sqrt to make recent states count more
        self.state_counts[current_state] += np.sqrt(self.n_adaptations)

        return update

    def _global_update(self, b=1.0, stage:Stage=Stage.SLOWGAIN, end_of_burnin=0, group_index=0):
        """
        Global update scheme (equation 12) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : Stage, optional (default : "burn-in")
            Sams two-stage phase. Options : "burn-in" or "slow-gain"
        end_of_burnin: int, optional (default : 0)
            Iteration at which burn-in phase was ended.
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.
        Returns
        -------
        np.ndarray : free energy updates
        """
        zeta = self._calibration_state.free_energies
        pi_j = self._calibration_state.targets
        # [1/pi_1...1/pi_i]
        update = 1.0 / pi_j
        ub_j = self.driver._get_reduced_potentials(group_index)
        # w_j(X;ζ⁽ᵗ⁻¹⁾)
        log_w_j = np.log(pi_j) - zeta - ub_j
        log_w_j -= logsumexp(log_w_j)
        w_j = np.exp(log_w_j)
        update *= w_j / pi_j
        update = np.dot(self._gain_factor(b=b, stage=stage, end_of_burnin=end_of_burnin), update)

        # Update count of current state weights.
        #  Use sqrt to make recent states count more
        self.state_counts += np.sqrt(self.n_adaptations) * w_j

        return update

    def _gain_factor(self, b=1.0, stage=Stage.SLOWGAIN, end_of_burnin=0):
        """
        Two stage update scheme (equation 15) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : Sams two-stage phase. Options : Stage.BURNIN or Stage.SLOWGAIN
        end_of_burnin: int, optional (default : 0)
            Iteration at which burn-in phase was ended.

        Returns
        -------
        np.ndarray - gain factor matrix
        """

        if not 0.5 <= b <= 1.0:
            raise ValueError("β needs to be between 1/2 and 1")

        pi_j = self._calibration_state.targets

        gain = np.zeros_like(pi_j)
        for j in range(gain.size):
            if stage is Stage.BURNIN:
                gain[j] = min(pi_j[j], 1.0/pow(self.n_adaptations, b))
            elif stage is Stage.SLOWGAIN:
                gain[j] = min(pi_j[j], 1.0/(self.n_adaptations - end_of_burnin + pow(end_of_burnin, b)))
            else:
                raise ValueError("Invalid SAMS adaptation stage specified %s. Choose Stage.BURNIN or Stage.SLOWGAIN.")

        return np.diag(gain)