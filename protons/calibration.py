# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from protons.driver import _BaseProtonDrive, AmberProtonDrive
import simtk.openmm.app as app
from simtk import openmm
import simtk.unit as units
from .logger import log
import abc
from . import get_data
from scipy.misc import logsumexp
from collections import deque
import openmmtools

kB = (1.0 * units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA).in_units_of(units.kilocalories_per_mole / units.kelvin)


class SelfAdjustedMixtureSampling(object):
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

    def __init__(self, driver):
        """
        Initialize a Self-adjusted mixture sampling (SAMS) simulation engine for a given
        ProtonDrive object.
        """

        # Check if driver is of the right type.
        assert issubclass(type(driver), _BaseProtonDrive)
        self.driver = driver
        self.n_adaptations = 0

        target_weights = None
        for i, group in enumerate(self.driver.titrationGroups):
            for j, state in enumerate(self.driver.titrationGroups[i]['titration_states']):
                if target_weights is not None:
                    self.driver.titrationGroups[i]['titration_states'][j]['target_weight'] = target_weights[i][j]
                else:
                    self.driver.titrationGroups[i]['titration_states'][j]['target_weight'] = 1.0 / len(self.driver.titrationGroups[i]['titration_states'])

        group_index = 0
        nstates = len(self.driver.titrationGroups[group_index]['titration_states'])
        self.state_counts = np.zeros(nstates, np.float64)
        log.info('There are %d sams_sampler states' % nstates)

    def adapt_zetas(self, context, scheme='binary', b=0.85, stage="slow-gain", end_of_burnin=0, group_index=0):
        """
        Update the relative free energy of sams_sampler states of the specified titratable group
        using self-adjusted mixture sampling (SAMS)
        Parameters
        ----------
        context :  (simtk.openmm.Context)
            The context to update
        scheme : str, optional (default : 'binary')
            Scheme from Tan paper ('binary' or 'global').
        b : float, optional (default : 0.85)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        stage : str, optional (default : "slow-gain")
            Sams two-stage phase. Options : "burn-in" or "slow-gain"
        end_of_burnin: int, optional (default : 0)
            Iteration at which burn-in phase was ended.
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        Returns
        -------
        target_deviation - np.array of the deviation of the sampled histogram weights from the target pi_j's

        """
        self.n_adaptations += 1

        # zeta^{t-1}
        zeta = self.get_gk()

        if scheme == 'binary':
            update = self._binary_update(group_index=group_index, b=b, stage=stage, end_of_burnin=end_of_burnin)
        elif scheme == 'global':
            update = self._global_update(context, group_index=group_index, b=b, stage=stage, end_of_burnin=end_of_burnin)
        else:
            raise ValueError("Unknown adaptation scheme: {}!".format(scheme))

        # zeta^{t-1/2}
        zeta += update
        # zeta^{t} = zeta^{t-1/2} - zeta_1^{t-1/2}
        zeta_t = zeta - zeta[0]

        # Set reference free energy based on new zeta
        self.set_gk(zeta_t, group_index=group_index)

        Nk = self.state_counts / self.state_counts.sum()
        target = self._get_target_weights(group_index)
        target_deviation = sum(abs(target - Nk))
        log.debug('Adaptation step %8d : zeta_t = %s, N_k = %s, %2f%% deviation' % (self.n_adaptations, str(zeta_t), str(Nk), target_deviation * 100))
        return target_deviation

    def set_gk(self, zetas, group_index=0):
        """
        Set g_k based on provided zetas
        Parameters
        ----------
        zetas : list of float
            Zeta values for each sams_sampler state
        group_index : int, optional
            Index of the group that needs updating, defaults to 0
        """

        for i, titr_state_zeta in enumerate(zetas):
            # Zeta has opposite sign of relative energies
            self.driver.titrationGroups[group_index]['titration_states'][i]['g_k'] = titr_state_zeta

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
        zeta = np.asarray(list(map(lambda x: x['g_k'], self.driver.titrationGroups[group_index]['titration_states'][:])))
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
        return np.asarray(list(map(lambda x: x['target_weight'], self.driver.titrationGroups[group_index]['titration_states'][:])))

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
        update = np.asarray(list(map(lambda x: 1 / x['target_weight'], self.driver.titrationGroups[group_index]['titration_states'][:])))
        # delta(Lt)
        delta = np.zeros_like(update)
        delta[self.driver._get_titration_state(group_index)] = 1
        update *= delta
        update = np.dot(self._gain_factor(b=b, stage=stage, group_index=group_index, end_of_burnin=end_of_burnin), update)

        # Update count of current state weights.
        group_index = 0
        current_state = self.driver._get_titration_state(group_index)
        #  Use sqrt to make recent states count more
        self.state_counts[current_state] += np.sqrt(self.n_adaptations)

        return update

    def _global_update(self, context, b=1.0, stage="slow-gain", end_of_burnin=0, group_index=0):
        """
        Global update scheme (equation 12) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        context :  (simtk.openmm.Context)
            The context to update
        zeta : np.ndarray
            Current estimate of free energies ζ⁽ᵗ⁾
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
        np.ndarray : free energy updates
        """
        zeta = self.get_gk(group_index)
        pi_j = self._get_target_weights(group_index)
        # [1/pi_1...1/pi_i]
        update = 1.0 / pi_j
        ub_j = self.driver._get_reduced_potentials(context, group_index)
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
            raise ValueError("β needs to be between 1/2 and 1.0")

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


class PopulationCalculator(object):
    """
    Abstract base class for determining state populations from a pH curve
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def populations(self):
        """ Return population of each state of the amino acid.

        Returns
        -------
        list of float
        """
        return NotImplemented


class Histidine(PopulationCalculator):
    """
    Amber constant-pH HIP residue state weights at given pH
    """
    pka_d = 6.5
    pka_e = 7.1

    def __init__(self, pH):
        self.kd = pow(10.0, pH - Histidine.pka_d)
        self.ke = pow(10.0, pH - Histidine.pka_e)

    def hip_concentration(self):
        """
        Concentration of the doubly protonated form
        """
        return 1.0/(self.ke + self.kd + 1.0)

    def hie_concentration(self):
        """
        Concentration of the epsilon protonated form
        """
        return self.ke / (self.ke + self.kd + 1.0)

    def hid_concentration(self):
        """
        Concentration of the delta pronated form
        """
        return self.kd / (self.ke + self.kd + 1.0)

    def populations(self):
        """
        Returns
        -------
        list of float : state weights in order of AMBER cpH residue
        """
        return [self.hip_concentration(), self.hid_concentration(), self.hie_concentration()]


class Aspartic4(PopulationCalculator):
    """
    Amber constant-pH AS4 residue state weights at given pH
    """
    pka = 4.0

    def __init__(self, pH):
        self.k = pow(10.0, pH - self.pka)

    def protonated_concentration(self):
        """
        Concentration of protonated form
        """
        return 1.0/(self.k + 1.0)

    def deprotonated_concenration(self):
        """
        Concentration of deprotonated form
        """
        return self.k / (self.k + 1.0)

    def populations(self):
        """
        Returns
        -------
        list of float : state weights in order of AMBER cpH residue
        """
        acid = self.protonated_concentration() / 4.0
        return [self.deprotonated_concenration(), acid, acid, acid, acid]


class Glutamic4(Aspartic4):
    """
    Amber constant-pH GL4 residue state weights at given pH
    """
    pka = 4.4


class Lysine(Aspartic4):
    """
    Amber constant-pH LYS residue state weights at given pH
    """
    pka = 10.4

    def populations(self):
        return [self.protonated_concentration(), self.deprotonated_concenration()]


class Tyrosine(Lysine):
    """
    Amber constant-pH TYR residue state weights at given pH
    """
    pka = 9.6


class Cysteine(Lysine):
    """
    Amber constant-pH CYS residue state weights at given pH
    """
    pka = 8.5


class AmberCalibrationSystem(object):
    """
    Set up the reference system for one of the AMBER supported amino acids and provide an interface for calibration.
    """
    supported_aminoacids = {"lys": Lysine,
                 "cys": Cysteine,
                 "tyr": Tyrosine,
                 "as4": Aspartic4,
                 "gl4": Glutamic4,
                 "hip": Histidine
                            }

    def __init__(self, residue_name, settings, guess_free_energy=None, minimize=False):
        """Calibrate a single amino acid in a reference system for a given pH.

        Parameters
        ----------
        residue_name : str
            Three letter abbreviation of the amino acid
        settings : dict
            pH : float, the pH of the system
            temperature : float, temperature of the system
            solvent : str, implicit or explicit (currently no choice of solvent models)
            pressure : float, pressure of the system for an explicit solvent system
            collision_rate : collision rate for certain integrators
            timestep : int, timestep for Langevin integration
            nsteps_per_trial : int, number of ncmc timesteps (0 for instantaneous MC)
            platform_name : str, name of the OpenMM Platform (e.g. 'CPU', 'OpenCL')
        guess_free_energy : list, optional
            Reference free energy values (g_k) for the single amino acid from a previous calibration.
        minimize: bool, optional (default: False)
            Minimize system before equilibration.

        Notes
        -----
        The weights for each amino acid are predetermined by the pKas, which are hardcoded.
        All that is necessary to supply is the pH.

        See Also
        --------
        `Histidine` : Histidine pKas based weights
        `Tyrosine` : Tyrosine pKa based weights
        `Lysine` : Lysine pKa based weights
        `Cysteine` : Cysteine pKa based weights
        `Aspartic4` : Aspartic acid pKa based weights
        `Glutamic4` : Glutamic acid pKa based weights

        Todo
        ----
         - Add choice of solvent model beyond just "explicit" vs "implicit"

        """
        residue_name = residue_name.lower()

        # Retrieve the prepared calibration files for calibration on a terminally-blocked amino acid in implicit/explicit solvent
        positions = openmm.XmlSerializer.deserialize(open(get_data('calibration-systems/{}-{}.state.xml'.format(residue_name, settings["solvent"]), '')).read()).getPositions(asNumpy=True)
        system = openmm.XmlSerializer.deserialize(open(get_data('calibration-systems/{}-{}.sys.xml'.format(residue_name, settings["solvent"]), '')).read())
        prmtop = app.AmberPrmtopFile(get_data('calibration-systems/{}-{}.prmtop'.format(residue_name,settings["solvent"]), ''))
        cpin_filename = get_data('calibration-systems/{}-{}.cpin'.format(residue_name,settings["solvent"]), '')

        # Retrieve system conditions
        temperature = settings["temperature"]
        integrator_timestep = settings["timestep"]
        pressure = settings["pressure"]
        pH = settings["pH"]
        if not "collision_rate" in settings:
            integrator_collision_rate = 9.1 / units.picoseconds
        else:
            integrator_collision_rate = settings["collision_rate"]

        ncmc_steps_per_trial = settings["nsteps_per_trial"]

        # TODO detect platforms automatically?
        platform_name = settings["platform_name"]

        # TODO Confirm choice of GHMC integrator
        integrator = openmmtools.integrators.GHMCIntegrator(temperature=temperature, collision_rate=integrator_collision_rate, timestep=integrator_timestep)
        self.log_state_probabilities = np.log(np.array(AmberCalibrationSystem.supported_aminoacids[residue_name](pH).populations()))

        # Use SAMS to determine free energies of each protonation state under uniform state target weights.
        if settings["solvent"] == "explicit":
            system.addForce(openmm.MonteCarloBarostat(pressure, temperature))
            mc_titration = AmberProtonDrive(system, temperature, pH, prmtop, cpin_filename, integrator, pressure=pressure, ncmc_steps_per_trial=ncmc_steps_per_trial, implicit=False)
        elif settings["solvent"] == "implicit":
            system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            mc_titration = AmberProtonDrive(system, temperature, pH, prmtop, cpin_filename, integrator, pressure=None, ncmc_steps_per_trial=ncmc_steps_per_trial, implicit=True)
        else:
            raise ValueError("Solvent not recognized")

        sams_sampler = SelfAdjustedMixtureSampling(mc_titration)

        if guess_free_energy is not None:
            sams_sampler.set_gk(guess_free_energy)
        if platform_name:
            platform = openmm.Platform.getPlatformByName(platform_name)
            context = openmm.Context(system, sams_sampler.driver.compound_integrator, platform)
        else:
            context = openmm.Context(system, sams_sampler.driver.compound_integrator)

        if minimize:
            minimized_context, positions = self._minimizer(platform_name, system, positions)
            del minimized_context

        context.setPositions(positions)  # set to minimized positions
        context.setVelocitiesToTemperature(temperature)  # optional for some integrators
        self.context = context
        self.integrator = integrator
        self.integrator.step(1)
        self.system = system
        self.sams_sampler = sams_sampler
        self.settings = settings

    def sams_till_converged(self, threshold=1.e-5, mc_every=500, gk_every=1, check_frequency=100, window=200,
                            max_iter=None, min_iter=1, **kwargs):
        """
        Calibrate the amino acid using SAMS,  until converged to below the gradient threshold.

        Parameters
        ----------
        threshold : float, optional (default: 1.e-7)
            Maximum absolute gradient in gk to assume convergence.
        mc_every : int, optional (default: 100)
            Update sams_sampler state every `mc_every` dynamics steps.
        gk_every : int, optional (default: 1)
            Adapt the gk values every `gk_every` sams_sampler state updates
        check_frequency: int, optional (default: 100)
            Check for convergence for this amount of gk updates
        window : int, optional (default: 200)
            Gradient is evaluated over the last `window` samples.
        max_iter : int, optional
            Maxmimum number of iterations to run.
        min_iter: int, optional (default: 200)
            Minimum number of steps of burn-in.

        Other Parameters
        ----------------
        scheme : str
            'global' for global update (not recommended for ncmc)
            'binary' for binary update, default
        b : float, optional (default: 0.9)
            beta factor for gain factor calculation (SAMS equation 15).

        kwargs : optional keyword arguments are passed to underlying SAMS sampler.
            See `constph.calibration.SelfAdjustedMixtureSampling#adapt_zetas`.

        Yields
        -------
        np.ndarray - log bias weight g_k from calibration to give log state populations in solvent
        """

        # Default value for optional arguments to weight adaptation scheme
        b = kwargs.pop("b", 0.85)
        scheme = kwargs.pop("scheme", "binary")

        if kwargs:
            raise TypeError('"{}" are not valid keyword arguments.'.format('", "'.join(kwargs.keys())))

        gk_updates = 0
        iteration = 1
        # Stack of last updates to keep track of. First in first out.
        gk_deque = deque(maxlen=window)
        stage = "burn-in"
        end_of_burnin = None
        log.info("Starting calibration burn-in phase.")

        while True:

            # Regular MD is performed
            self.integrator.step(mc_every)
            iteration += 1

            # Attempt changing the protonation/tautomer state
            self.sams_sampler.driver.update(self.context)

            # Update gk using SAMS
            if iteration % gk_every == 0:
                gk_updates += 1
                target_deviation = self.sams_sampler.adapt_zetas(self.context, stage=stage, end_of_burnin=end_of_burnin, b=b, scheme=scheme)

                # Once we're within 20 percent of the target, switch to slow stage
                if target_deviation < 0.2 and end_of_burnin is None and iteration >= min_iter:
                    log.info("Burn-in complete in %d iterations! Switching to calibration slow-gain phase." % iteration)
                    end_of_burnin = iteration
                    stage="slow-gain"

                # We sample uniformly with SAMS, so we need to subtract log pi_j's from the g_k to get our targets.
                g_k_uniform = self.sams_sampler.get_gk()
                g_k = g_k_uniform - self.log_state_probabilities
                gk_deque.append(g_k)

                # Latest estimates are yielded with every iteration
                yield g_k

                # Mechanism to stop the loop once convergence is achieved to within the threshold

                if gk_updates % check_frequency == 0 and end_of_burnin is not None:
                    grad = np.average(np.gradient(gk_deque, 10), axis=1)[0]  # average gradient for each state
                    log.info("Gradient magnitude: {}".format(["{:.3f}".format(np.log10(abs(g))) for g in grad]))
                    # Absolute gradient of all states is equal/below threshold
                    if (abs(grad) <= threshold).all() and gk_updates >= min_iter + window:
                        break

            # Quit the loop if we exceed the max number of iterations
            if max_iter is not None and iteration == max_iter:
                log.warning("Calibration reached maximum number of iterations without converging.")
                break

    @staticmethod
    def _minimizer(platform_name, system, positions, nsteps=1000):
        """
        Convenient minimizer in case extra minimization is preferred
        """
        integrator = openmm.VerletIntegrator(1.0 * units.femtoseconds)
        constraint_tolerance = 1.0e-4
        integrator.setConstraintTolerance(constraint_tolerance)
        platform = openmm.Platform.getPlatformByName(platform_name)
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        log.info("Initial energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
        openmm.LocalEnergyMinimizer.minimize(context, 1.0, nsteps)
        log.info("Final energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        return context, positions
