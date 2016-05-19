# -*- coding: utf-8 -*-
from __future__ import print_function
import numpy as np
from .constph import MonteCarloTitration
import simtk.openmm.app as app
from simtk import openmm
import simtk.unit as units
from .logger import logger
import pymbar
from . import get_data
from scipy.misc import logsumexp
from collections import deque


# MODULE CONSTANTS
kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
kB = kB.in_units_of(units.kilocalories_per_mole / units.kelvin)


class CalibrationTitration(MonteCarloTitration):
    """Implementation of self-adjusted mixture sampling for calibrating titratable residues.

    Attributes
    ----------
    n_adaptations : int
        Number of times the relative free energies have been adapted.

    References
    ----------
    .. [1] Z. Tan, Optimally adjusted mixture sampling and locally weighted histogram analysis
        DOI: 10.1080/10618600.2015.1113975

    """

    def __init__(self, system, temperature, pH, prmtop, cpin_filename, integrator, pressure=None,
                 nattempts_per_update=None, simultaneous_proposal_probability=0.1,
                 nsteps_per_trial=0, ncmc_timestep=1.0 * units.femtoseconds,
                 maintainChargeNeutrality=False, cationName='Na+', anionName='Cl-', implicit=False, debug=False):
        """
        Initialize a Monte Carlo titration driver for constant pH simulation.

        Parameters
        ----------
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature to be simulated.
        pH : float
            The pH to be simulated.
        prmtop : simtk.openmm.app.Prmtop
            Parsed AMBER 'prmtop' file (necessary to provide information on exclusions
        cpin_filename : string
            AMBER 'cpin' file defining protonation charge states and energies
        integrator : simtk.openmm.integrator
            The integrator used for dynamics
        pressure : simtk.unit.Quantity compatible with atmospheres, optional, default=None
            For explicit solvent simulations, the pressure.
        nattempts_per_update : int, optional, default=None
            Number of protonation state change attempts per update call;
            if None, set automatically based on number of titratible groups (default: None)
        simultaneous_proposal_probability : float, optional, default=0.1
            Probability of simultaneously proposing two updates
        debug : bool, optional, default=False
            Turn debug information on/off.
        nsteps_per_trial : int, optional, default=0
            Number of steps per NCMC switching trial, or 0 if instantaneous Monte Carlo is to be used.
        ncmc_timestep : simtk.unit.Quantity with units compatible with femtoseconds
            Timestep to use for NCMC switching
        maintainChargeNeutrality : bool, optional, default=True
            If True, waters will be converted to monovalent counterions and vice-versa.
        cationName : str, optional, default='Na+'
            Name of cation residue from which parameters are to be taken.
        anionName : str, optional, default='Cl-'
            Name of anion residue from which parameters are to be taken.
        implicit: bool, optional, default=False
            Flag for implicit simulation. Skips ion parameter lookup.

        Other Parameters
        ----------------
        debug : bool, optional
            turn debug information on/off

        """

        super(CalibrationTitration, self).__init__(system, temperature, pH, prmtop, cpin_filename, integrator,
                                                   nattempts_per_update=nattempts_per_update,
                                                   simultaneous_proposal_probability=simultaneous_proposal_probability,
                                                   pressure=pressure,
                                                   nsteps_per_trial=nsteps_per_trial, ncmc_timestep=ncmc_timestep,
                                                   maintainChargeNeutrality=maintainChargeNeutrality,
                                                   cationName=cationName, anionName=anionName,
                                                   implicit=implicit,
                                                   debug=debug)

        self.n_adaptations = 0

        target_weights = None
        for i, group in enumerate(self.titrationGroups):
            for j, state in enumerate(self.titrationGroups[i]['titration_states']):
                if target_weights is not None:
                    self.titrationGroups[i]['titration_states'][j]['target_weight'] = target_weights[i][j]
                else:
                    self.titrationGroups[i]['titration_states'][j]['target_weight'] = 1.0 / len(self.titrationGroups[i]['titration_states'])

        group_index = 0
        nstates = len(self.titrationGroups[group_index]['titration_states'])
        self.state_counts = np.zeros(nstates, np.float64)
        logger.info('There are %d titration states' % nstates)

    def adapt_weights(self, context, scheme='global', b=0.85, t0=10000, group_index=0):
        """
        Update the relative free energy of titration states of the specified titratable group
        using self-adjusted mixture sampling (SAMS)
        Parameters
        ----------
        context :  (simtk.openmm.Context)
            The context to update
        scheme : str ('binary' or 'global')
            Scheme from Tan paper.
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        t0 : int, optional (default : 1.0)
            Burn-in size for adaptation.
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.
        """
        self.n_adaptations += 1

        # zeta^{t-1}
        zeta = self.get_zeta()

        if scheme == 'binary':
            update = self._binary_update(group_index=group_index, b=b, t0=t0)
        elif scheme == 'global':
            update = self._global_update(context, group_index=group_index, b=b, t0=t0)
        else:
            raise ValueError("Unknown adaptation scheme: {}!".format(scheme))

        # zeta^{t-1/2}
        zeta += update
        # zeta^{t} = zeta^{t-1/2} - zeta_1^{t-1/2}
        zeta_t = zeta - zeta[0]

        # Set reference energy based on new zeta
        self.set_relative_free_energy(zeta_t, group_index=group_index)

        logger.info('Adaptation step %8d : zeta_t = %s, N_k = %s' % (self.n_adaptations, str(zeta_t), str(self.state_counts / self.state_counts.sum())))

        return

    def set_relative_free_energy(self, zetas, group_index=0):
        """
        Set relative free energies based on provided zetas
        Parameters
        ----------
        zetas : list of float
            Zeta values for each titration state
        group_index : int, optional
            Index of the group that needs updating, defaults to 0
        """

        for i, titr_state_zeta in enumerate(zetas):
            # Zeta has opposite sign of relative energies
            self.titrationGroups[group_index]['titration_states'][i]['relative_frenergy'] = titr_state_zeta

    def get_zeta(self, group_index=0):
        """Retrieve zeta for specified titratable group.
        Parameters
        ----------
        beta : simtk.unit.Quantity compatible with simtk.unit.mole/simtk.unit.kcal
            inverse temperature
        group_index : int, optional
            Index of the group that needs updating, defaults to 0

        Returns
        -------
        np.ndarray - zeta of states
        """
        zeta = np.asarray(list(map(lambda x: x['relative_frenergy'], self.titrationGroups[group_index]['titration_states'][:])))
        return zeta

    def _get_target_weights(self, group_index=0):
        """Retrieve target weights for specified titratable group.
        Parameters
        ----------
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.group_index

        Returns
        -------
        np.ndarray - relative free energy of states
        """
        return np.asarray(list(map(lambda x: x['target_weight'], self.titrationGroups[group_index]['titration_states'][:])))

    def _binary_update(self, group_index=0, b=1.0, t0=0):
        """
        Binary update scheme (equation 9) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        t0 : int, optional (default : 1.0)
            Burn-in size for adapation.
        Returns
        -------
        np.ndarray - free energy updates
        """
        # [1/pi_1...1/pi_i]
        update = np.asarray(list(map(lambda x: 1 / x['target_weight'], self.titrationGroups[group_index]['titration_states'][:])))
        # delta(Lt)
        delta = np.zeros_like(update)
        delta[self.getTitrationState(group_index)] = 1
        update *= delta
        update = np.dot(self._gain_factor(b=b, t0=t0, group_index=group_index), update)

        # Update histogram count of current state.
        group_index = 0
        current_state = self.getTitrationState(group_index)
        self.state_counts[current_state] += np.sqrt(self.n_adaptations)

        return update

    def _global_update(self, context, b=1.0, t0=0, group_index=0):
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
        t0 : int, optional (default : 1.0)
            Burn-in size for adapation.

        Returns
        -------
        np.ndarray : free energy updates
        """
        zeta = self.get_zeta(group_index)
        pi_j = self._get_target_weights(group_index)
        # [1/pi_1...1/pi_i]
        update = 1.0 / pi_j
        ub_j = self._get_reduced_potentials(context, group_index)
        # w_j(X;ζ⁽ᵗ⁻¹⁾)
        log_w_j = np.log(pi_j) - zeta - ub_j
        log_w_j -= logsumexp(log_w_j)
        w_j = np.exp(log_w_j)
        update *= w_j / pi_j
        update = np.dot(self._gain_factor(b=b, t0=t0, group_index=group_index), update)

        # Update histogram count of current state.
        self.state_counts += np.sqrt(self.n_adaptations) * w_j

        return update

    def _gain_factor(self, b=1.0, t0=0, group_index=0):
        """
        Two stage update scheme (equation 15) from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        b : float, optional (default : 1.0)
             Decay factor β in two stage SAMS update scheme. Must be between 0.5 and 1.0.
        t0 : int, optional (default : 1.0)
            Burn-in size for adapation.
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        Returns
        -------
        np.ndarray - gain factor matrix
        """

        if not 0.5 <= b <= 1.0:
            raise ValueError("β needs to be between 1/2 and 1.0")

        pi_j = self._get_target_weights(group_index)

        gain = np.zeros_like(pi_j)
        for j in range(gain.size):
            if self.n_adaptations <= t0:
                gain[j] = min(pi_j[j], 1.0/pow(self.n_adaptations, b))
            elif self.n_adaptations > t0:
                gain[j] = min(pi_j[j], 1.0/(self.n_adaptations - t0 + pow(t0, b)))

        return np.diag(gain)


class Histidine(object):
    """
    Amber constant-pH HIP residue state weights at given pH
    """
    pKa_d = 6.5
    pKa_e = 7.1

    def __init__(self, pH):
        self.kd = pow(10.0, pH - Histidine.pKa_d)
        self.ke = pow(10.0, pH - Histidine.pKa_e)

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

    def weights(self):
        """
        Returns
        -------
        list of float : state weights in order of AMBER cpH residue
        """
        return [self.hip_concentration(), self.hid_concentration(), self.hie_concentration()]


class Aspartic4(object):
    """
    Amber constant-pH AS4 residue state weights at given pH
    """
    pKa = 4.0

    def __init__(self, pH):
        self.k = pow(10.0, pH - self.pKa)

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

    def weights(self):
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
    pKa = 4.4


class Lysine(Aspartic4):
    """
    Amber constant-pH LYS residue state weights at given pH
    """
    pKa = 10.4

    def weights(self):
        return [self.protonated_concentration(), self.deprotonated_concenration()]


class Tyrosine(Lysine):
    """
    Amber constant-pH TYR residue state weights at given pH
    """
    pKa = 9.6


class Cysteine(Lysine):
    """
    Amber constant-pH CYS residue state weights at given pH
    """
    pKa = 8.5


class AminoAcidCalibrator(object):
    """
    Set up the reference system for one of the AMBER supported amino acids and provide an interface for calibration.
    """
    supported = {"lys": Lysine,
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
            collision_rate : collision rate for Langevin integration
            timestep : int, timestep for Langevin integration
            nsteps_per_trial : int, number of ncmc timesteps (0 for instantaneous MC)
            platform_name : str, name of the OpenMM Platform (e.g. 'CPU', 'OpenCL')
        guess_free_energy : list, optional
            Reference free energies for the single amino acid from a previous calibration, to continue where one left off.

        Notes
        -----
        The weights for each amino acid are predetermined by the pKas. All that is necessary to supply is the pH.

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
        # Calibration on a terminally-blocked amino acid in implicit/explicit solvent
        positions = openmm.XmlSerializer.deserialize(open(get_data('calibration-systems/{}-{}.state.xml'.format(residue_name, settings["solvent"]), '')).read()).getPositions(asNumpy=True)
        system = openmm.XmlSerializer.deserialize(open(get_data('calibration-systems/{}-{}.sys.xml'.format(residue_name, settings["solvent"]), '')).read())
        prmtop = app.AmberPrmtopFile(get_data('calibration-systems/{}-{}.prmtop'.format(residue_name,settings["solvent"]), ''))
        cpin_filename = get_data('calibration-systems/{}-{}.cpin'.format(residue_name,settings["solvent"]), '')
        temp = settings["temperature"]
        ts = settings["timestep"]
        press = settings["pressure"]
        pH = settings["pH"]
        #crate = settings["collision_rate"]
        crate = 9.1 / units.picoseconds
        nspt = settings["nsteps_per_trial"]
        platform_name = settings["platform_name"]
        integrator = openmm.LangevinIntegrator(temp, crate, ts)

        # TODO: Change to recording state log probabilities
        self.log_state_probabilities = [np.log(np.array(AminoAcidCalibrator.supported[residue_name](pH).weights()))]

        # Use SAMS to determine free energies of each protonation state under uniform state target weights.
        if settings["solvent"] == "explicit":
            system.addForce(openmm.MonteCarloBarostat(press, temp))
            mc_titration = CalibrationTitration(system, temp, pH, prmtop, cpin_filename, integrator, pressure=press, nsteps_per_trial=nspt, implicit=False)
        elif settings["solvent"] == "implicit":
            system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            mc_titration = CalibrationTitration(system, temp, pH, prmtop, cpin_filename, integrator, pressure=None, nsteps_per_trial=nspt, implicit=True)
        else:
            raise ValueError("Solvent not recognized")

        if guess_free_energy is not None:
            # Zeta has opposite sign of relative free energy
            mc_titration.set_relative_free_energy(guess_free_energy)
        if platform_name:
            platform = openmm.Platform.getPlatformByName(platform_name)
            context = openmm.Context(system, mc_titration.compound_integrator, platform)
        else:
            context = openmm.Context(system, mc_titration.compound_integrator)

        # platform.setPropertyValue(context, "CudaDeviceIndex", "0")

        if minimize:
            minimized_context, positions = self._minimizer(platform_name, system, positions) # dont use minimized_context
        context.setPositions(positions)  # set to minimized positions

        self.context = context
        self.integrator = integrator
        self.integrator.step(1)
        self.system = system
        self.titration = mc_titration
        self.settings = settings


    def calibrate_till_converged(self, threshold=1.e-5, mc_every=500, zeta_every=1, convergence_frequency=500, window=2000, max_iter=None, **kwargs):
        """
        Calibrate the amino acid until converged to below the gradient threshold

        Parameters
        ----------
        threshold : float, optional (default: 1.e-7)
            Maximum absolute gradient to assume convergence.
        mc_every : int, optional (default: 100)
            Update titration state every `mc_every` steps.
        zeta_every : int, optional (default: 1)
            Adapt the SAMS zeta every `zeta_every` titration state updates
        convergence_frequency: int, optional (default: 500)
            Check for convergence for this amount of zeta updates
        window : int, optional (default: 2000)
            Gradient is evaluated over the last `window` samples.
        max_iter : int, optional
            Maxmimum number of iterations to run.
        scheme : str
            'global' for global update
            'binary' for binary update (not recommended)
        kwargs : optional keyword arguments are passed to underlying SAMS sampler.
            See `constph.calibration.CalibrationTitration#adapt_weights`.

        Yields
        -------
        np.ndarray - log bias weight g_k from calibration to give log state populations in solvent
        """

        # Default value for optional arguments to weight adaptation scheme
        t0 = kwargs.pop("t0", 1500)
        b = kwargs.pop("b", .9)
        scheme=kwargs.pop("scheme", "global")

        if kwargs:
            raise TypeError('"{}" are not valid keyword arguments.'.format('", "'.join(kwargs.keys())))

        state_updates = 0
        zeta_updates = 0
        iteration = 1
        zeta_window = deque(maxlen=window)
        while True:
            self.integrator.step(mc_every)
            iteration += 1
            self.titration.update(self.context)
            if iteration % zeta_every == 0:
                zeta_updates += 1
                self.titration.adapt_weights(self.context, t0=t0, b=b, scheme=scheme)
                zeta = self.titration.get_zeta()
                g_k = zeta - self.log_state_probabilities
                zeta_window.append(zeta)
                yield g_k

                if zeta_updates % convergence_frequency == 0:
                    grad = np.average(np.gradient(zeta_window, 10), axis=1)[0]  # average gradient for each state
                    logger.info("Gradient magnitude: {}".format([ "{:.3f}".format(np.log10(abs(g))) for g in grad]))
                    # Absolute gradient of all states is equal/below threshold
                    if (abs(grad) <= threshold).all() and zeta_updates >= t0 + window:
                        break


            if max_iter is not None and iteration == max_iter:
                break

    @staticmethod
    def _minimizer(platform_name, system, positions, nsteps=1000):
        """
        Convenient minimizer in case extra minimization is preferred
        """
        integrator = openmm.VerletIntegrator(1.0 * units.femtoseconds)
        CONSTRAINT_TOLERANCE = 1.0e-4
        integrator.setConstraintTolerance(CONSTRAINT_TOLERANCE)
        platform = openmm.Platform.getPlatformByName(platform_name)
        context = openmm.Context(system, integrator, platform)
        context.setPositions(positions)
        logger.info("Initial energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
        openmm.LocalEnergyMinimizer.minimize(context, 1.0, nsteps)
        logger.info("Final energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        return context, positions
