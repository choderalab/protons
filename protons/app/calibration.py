# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np

from .driver import NCMCProtonDrive, _SAMSState, SAMSApproach, Stage, UpdateRule
import simtk.unit as units
from .logger import log
from scipy.special import logsumexp

kB = (1.0 * units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA).in_units_of(
    units.kilocalories_per_mole / units.kelvin
)


class SAMSCalibrationEngine:
    """An implementation of Self Adjusted mixture sampling that can sample multiple sites at once."""

    def __init__(self, driver: NCMCProtonDrive):
        """
        Initialize a Self-adjusted mixture sampling (SAMS) simulation engine for a given ProtonDrive object.

        Parameters
        ----------
        driver : NCMCProtonDrive derived class
        """
        # Type checks
        assert issubclass(type(driver), NCMCProtonDrive)

        if driver.calibration_state is None:
            raise ValueError(
                "Drive has not been prepared for calibration. Please call driver.enable_calibration."
            )

        self.driver = driver
        self.approach: SAMSApproach = driver.calibration_state.approach
        self.group_index = driver.calibration_state.group_index
        self._calibration_state: _SAMSState = driver.calibration_state
        self.nstates = len(self._calibration_state.free_energies)
        return

    def adapt_zetas(
        self,
        update_rule=UpdateRule.BINARY,
        b: float = 0.85,
        stage: Stage = Stage.FASTDECAY,
        end_of_slowdecay: int = 0,
    ):
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
        end_of_slowdecay: int, optional (default : 0)
            Iteration at which slow-decay phase was ended.
        Returns
        -------
        target_deviation - np.array of the deviation of the sampled histogram weights from the target pi_j's

        """

        # zeta^{t-1}
        zeta = self._calibration_state.free_energies

        # Stage dependend prep work

        # Only increase adaptation number if adaptation is expected
        if stage in [Stage.NODECAY, Stage.SLOWDECAY, Stage.FASTDECAY]:
            self._calibration_state._current_adaptation += 1

        elif stage is Stage.EQUILIBRIUM:
            # Update the histogram with a single count
            current_state = self._calibration_state.state_index(
                self.driver.titrationStates
            )
            counts = self._calibration_state.observed_counts
            counts[current_state] += 1
            self._calibration_state.observed_counts = counts
            return self._calibration_state.max_absolute_deviation

        # Perform updates for all stages other than equilibrium
        if update_rule is UpdateRule.BINARY:
            update = self._binary_update(
                group_index=self.group_index,
                b=b,
                stage=stage,
                end_of_slowdecay=end_of_slowdecay,
            )
        elif update_rule is UpdateRule.GLOBAL:
            if self.approach is SAMSApproach.MULTISITE:
                raise NotImplementedError(
                    "Global updates only implemented for one site at this time."
                )
            update = self._global_update(
                group_index=self.group_index,
                b=b,
                stage=stage,
                end_of_slowdecay=end_of_slowdecay,
            )
        else:
            raise ValueError("Unknown adaptation update_rule: {}!".format(update_rule))

        # zeta^{t-1/2}
        zeta += update
        # zeta^{t} = zeta^{t-1/2} - zeta_1^{t-1/2}
        zeta_t = zeta - zeta[0]

        # Set reference free energy based on new zeta
        self._calibration_state.free_energies = zeta_t

        # These need to be taken from full calibration run
        if self._calibration_state._current_adaptation > 0:
            target_deviation = self._calibration_state.max_absolute_deviation
            log.debug(
                "Adaptation step %8d : zeta_t = %s, N_k = %s, %2f%% deviation"
                % (
                    self._calibration_state._current_adaptation,
                    str(zeta_t),
                    str(self._calibration_state.observed_counts),
                    target_deviation * 100,
                )
            )
            return target_deviation
        else:
            log.debug(
                "Adaptation step %8d : still in burn-in."
                % self._calibration_state._current_adaptation
            )
            return 1.0

    def _binary_update(
        self,
        group_index: int = 0,
        b: float = 1.0,
        stage=Stage.FASTDECAY,
        end_of_slowdecay: int = 0,
    ):
        """
        Binary update scheme (equation 9) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : Sams two-stage phase. Options : Stage.SLOWDECAY or Stage.FASTDECAY
        end_of_slowdecay: int, optional (default : 0)
            Iteration at which slow-decay phase was ended.
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
        update = np.dot(
            self._gain_factor(b=b, stage=stage, end_of_slowdecay=end_of_slowdecay),
            update,
        )

        # Update count of current state weights.
        # Use sqrt to make recent states count more
        # Only performed if past burn-in
        if self._calibration_state._current_adaptation > 0:
            counts = self._calibration_state.observed_counts
            counts[current_state] += np.sqrt(
                self._calibration_state._current_adaptation
            )
            self._calibration_state.observed_counts = counts

        return update

    def _global_update(
        self, b=1.0, stage: Stage = Stage.FASTDECAY, end_of_slowdecay=0, group_index=0
    ):
        """
        Global update scheme (equation 12) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : Stage, optional (default : "burn-in")
            Sams two-stage phase. Options : "burn-in" or "slow-gain"
        end_of_slowdecay: int, optional (default : 0)
            Iteration at which slow-decay phase was ended.
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
        update = np.dot(
            self._gain_factor(b=b, stage=stage, end_of_slowdecay=end_of_slowdecay),
            update,
        )

        # Update count of current state weights.
        # Use sqrt to make recent states count more
        # Only use if past the burn in phase
        if self._calibration_state._current_adaptation > 0:
            self._calibration_state.observed_counts += (
                np.sqrt(self._calibration_state._current_adaptation) * w_j
            )
        return update

    def _gain_factor(self, b=1.0, stage=Stage.FASTDECAY, end_of_slowdecay=0):
        """
        Two stage update scheme (equation 15) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : Sams two-stage phase. Options : Stage.SLOWDECAY or Stage.FASTDECAY
        end_of_slowdecay: int, optional (default : 0)
            Iteration at which slow_decay phase was ended.

        Returns
        -------
        np.ndarray - gain factor matrix
        """

        if not 0.5 <= b <= 1.0:
            raise ValueError("β needs to be between 1/2 and 1")

        pi_j = self._calibration_state.targets
        n_adapt = self._calibration_state._current_adaptation

        gain = np.zeros_like(pi_j)
        for j in range(gain.size):
            # Adaptation with a slow decay of the gain factor
            if stage is Stage.SLOWDECAY:
                gain[j] = min(pi_j[j], 1.0 / pow(n_adapt, b))
            # Adaptation with a fast decay of the gain factor (asymptotic optimal convergence)
            elif stage is Stage.FASTDECAY:
                gain[j] = min(
                    pi_j[j],
                    1.0 / (n_adapt - end_of_slowdecay + pow(end_of_slowdecay, b)),
                )
            # Adaptation with no decay of the gain factor (sub-optimal, not proven to converge, for initial guess)
            elif stage is Stage.NODECAY:
                gain[j] = min(pi_j[j], 1.0)
            # No adaptation, for equilibrium free energy estimates
            elif stage is Stage.EQUILIBRIUM:
                gain[j] = 0.0
            else:
                raise ValueError(
                    "Invalid SAMS adaptation stage specified %s. Choose Stage.SLOWDECAY or Stage.FASTDECAY."
                )

        return np.diag(gain)
