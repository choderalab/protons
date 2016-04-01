from __future__ import print_function
import numpy as np
from constph.constph import MonteCarloTitration
import simtk.openmm.app as app
from simtk import openmm
import simtk.unit as units
import logging
import pymbar
from . import get_data
from scipy.misc import logsumexp
import joblib

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
                 maintainChargeNeutrality=False, cationName='Na+', anionName='Cl-', implicit=False, target_weights=None, debug=False):
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
        target_weights : list, optional
            Nested list indexed [group][state] of relative weights (pi) for SAMS method
            If unspecified, all target weights are set to equally sample all states.

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

        for i, group in enumerate(self.titrationGroups):
            for j, state in enumerate(self.titrationGroups[i]['titration_states']):
                if target_weights is not None:
                    self.titrationGroups[i]['titration_states'][j]['target_weight'] = target_weights[i][j]
                else:
                    self.titrationGroups[i]['titration_states'][j]['target_weight'] = 1.0 / len(self.titrationGroups[i]['titration_states'])

    def adapt_weights(self, context, scheme, group_index=0, debuglogger=False):
        """
        Update the relative free energy of titration states of the specified titratable group
        Parameters
        ----------
        context :  (simtk.openmm.Context)
            The context to update
        scheme : str ('eq9' or 'eq12')
            Scheme from Tan paper.
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        """
        self.n_adaptations += 1
        temperature = self.temperature
        kT = kB * temperature  # thermal energy
        beta = 1.0 / kT  # inverse temperature
        # zeta^{t-1}
        zeta = self.get_zeta(beta)
        if debuglogger:
            dlogger = dict()
            dlogger['L'] = self.getTitrationState(group_index) + 1
        else:
            dlogger = None

        if scheme == 'eq9':
            update = self._equation9(group_index, dlogger)
        elif scheme == 'eq12':
            update = self._equation12(context, beta, zeta, group_index, dlogger)
        else:
            raise ValueError("Unknown adaptation scheme!")

        # zeta^{t-1/2}
        zeta += update
        # zeta^{t} = zeta^{t-1/2} - zeta_1^{t-1/2}
        zeta_t = zeta - zeta[0]
        if debuglogger:
            for j, z in enumerate(zeta_t):
                dlogger['zeta_t %d' % (j + 1)] = z

        # Set reference energy based on new zeta
        for i, titr_state in enumerate(zeta_t):
            self.titrationGroups[group_index]['titration_states'][i]['relative_energy'] = titr_state / -beta
        return dlogger

    def get_zeta(self, beta=None, group_index=0):
        """Retrieve relative free energies for specified titratable group.
        Parameters
        ----------
        beta : simtk.unit.Quantity compatible with simtk.unit.mole/simtk.unit.kcal
            inverse temperature
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.group_index

        Returns
        -------
        np.ndarray - relative free energy of states
        """
        if beta is None:
            beta = self.beta
        zeta = np.asarray(
            list(map(lambda x: np.float64(x['relative_energy'] * -beta), self.titrationGroups[group_index]['titration_states'][:])))
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

    def _equation9(self, group_index, dlogger=None):
        """
        Equation 9 from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

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
        if dlogger is not None:
            for j, d in enumerate(delta):
                dlogger['δ %d' % (j + 1)] = d
        update /= self.n_adaptations  # t^{-1}
        return update

    def _equation12(self, context, beta, zeta, group_index=0, dlogger=None):
        """
        Equation 12 from DOI: 10.1080/10618600.2015.1113975

        Parameters
        ----------
        context :  (simtk.openmm.Context)
            The context to update
        beta : simtk.unit.Quantity compatible with simtk.unit.mole/simtk.unit.kcal
            inverse temperature
        zeta : np.ndarray
            Current estimate of free energies ζ⁽ᵗ⁾
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        Returns
        -------
        np.ndarray - free energy updates
        """
        # target weights
        pi_j = self._get_target_weights(group_index)
        # [1/pi_1...1/pi_i]
        update = np.apply_along_axis(lambda x: 1 / x, 0, pi_j)

        ub_j = self._get_reduced_potentials(context, beta,   group_index)

        # w_j(X;ζ⁽ᵗ⁻¹⁾)
        log_w_j = np.log(pi_j) - zeta - ub_j
        if dlogger is not None:
            for j, z in enumerate(log_w_j):
                dlogger['-ln(π_{0}) - zeta_{0} - U_{0}(x)'.format(j + 1)] = z
        log_w_j -= logsumexp(log_w_j)
        w_j = np.exp(log_w_j)
        update *= w_j
        update /= self.n_adaptations  # t^{-1}
        if dlogger is not None:
            for j, w in enumerate(w_j):
                dlogger['beta * U_%d(x)' % (j + 1)] = ub_j[j]
                dlogger['w_%d' % (j + 1)] = w

        return update


class MBarCalibrationTitration(MonteCarloTitration):

    def __init__(self, system, temperature, pH, prmtop, cpin_filename, context, integrator, pressure=None, nattempts_per_update=None,
                 simultaneous_proposal_probability=0.1, nsteps_per_trial=0, ncmc_timestep=1.0 * units.femtoseconds,
                 maintainChargeNeutrality=False, cationName='Na+', anionName='Cl-', implicit=False,
                 debug=False):
        """
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

        """
        super(MBarCalibrationTitration, self).__init__(system, temperature, pH, prmtop, cpin_filename, integrator,
                                                       nattempts_per_update=nattempts_per_update,
                                                       simultaneous_proposal_probability=simultaneous_proposal_probability,
                                                       pressure=pressure,
                                                       nsteps_per_trial=nsteps_per_trial, ncmc_timestep=ncmc_timestep,
                                                       maintainChargeNeutrality=maintainChargeNeutrality,
                                                       cationName=cationName, anionName=anionName,
                                                       implicit=implicit,
                                                       debug=debug)

        self.n_adaptations = 0
        temperature = self.temperature
        kT = kB * temperature  # thermal energy
        beta = 1.0 / kT  # inverse temperature
        for i, group in enumerate(self.titrationGroups):
            self.titrationGroups[i]['adaptation_tracker'] = dict(label=[self.getTitrationState(i)], red_potential=[self._get_reduced_potentials(context, beta, i)])

    def adapt_weights(self, context, group_index=0, debuglogger=False):
        """
        Update the relative free energy of titration states of the specified titratable group
        Parameters
        ----------
        context :  (simtk.openmm.Context)
            The context to update
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        """

        if debuglogger:
            dlogger = dict()
        self.n_adaptations += 1
        temperature = self.temperature
        kT = kB * temperature  # thermal energy
        beta = 1.0 / kT  # inverse temperature
        # zeta^{t-1}

        self.titrationGroups[group_index]['adaptation_tracker']['label'].append(self.getTitrationState(group_index))
        self.titrationGroups[group_index]['adaptation_tracker']['red_potential'].append(
            self._get_reduced_potentials(context, beta, group_index))
        states = range(len(self.titrationGroups[group_index]['titration_states']))
        N_k = [self.titrationGroups[group_index]['adaptation_tracker']['label'].count(s) for s in states]
        U_k = zip(*self.titrationGroups[group_index]['adaptation_tracker']['red_potential'])
        mbar = pymbar.MBAR(U_k, N_k)
        frenergy = mbar.getFreeEnergyDifferences()[0][0]

        if debuglogger:
            dlogger['L'] = self.titrationGroups[group_index]['adaptation_tracker']['label'][-1] + 1
            for j, z in enumerate(frenergy):
                dlogger['beta * U_%d(x)' % (j + 1)] = self.titrationGroups[group_index]['adaptation_tracker']['red_potential'][-1][j]
                dlogger['zeta_t %d' % (j + 1)] = z

        # Set reference energy based on new zeta
        for i, titr_state in enumerate(frenergy):
            self.titrationGroups[group_index]['titration_states'][i]['relative_energy'] = titr_state / beta

        if debuglogger:
            return dlogger


class AminoAcidCalibrator(object):
    def __init__(self, residue_name, settings, platform_name="CPU", weights=None, minimize=False):
        """ Calibrate a single amino acid in a reference system for a given pH.
        Parameters
        ----------
        residue_name - str
            Three letter abbreviation of the amino acid
        settings - dict
            pH - float, the pH of the system
            temperature - float, temperature of the system
            solvent - str, implicit or explicit (currently no choice of solvent models)
            pressure - float, pressure of the system for an explicit solvent system
            collision_rate - collision rate for Langevin integration
            timestep - int, timestep for Langevin integration
            nsteps_per_trial - int, number of ncmc timesteps (0 for instantaneous MC)


        weights - list/tuple
            List of weights for each state


        Todo
        ----
         - Add choice of solvent model beyond just "explicit" vs "implicit"
         - What order should weights be defined?
         - Define states as nodes on a graph?
         - Convenience function to calculate weights as function of the pKa

        """

        residue_name = residue_name.lower()
        # Calibration on a terminally-blocked amino acid in implicit/explicit solvent
        positions = openmm.XmlSerializer.deserialize(open(get_data('calibration-systems/{}-{}.state.xml'.format(residue_name,settings["solvent"]), '')).read()).getPositions(asNumpy=True)
        system = openmm.XmlSerializer.deserialize(open(get_data('calibration-systems/{}-{}.sys.xml'.format(residue_name,settings["solvent"]), '')).read())
        prmtop = app.AmberPrmtopFile(get_data('calibration-systems/{}-{}.prmtop'.format(residue_name,settings["solvent"]), ''))
        cpin_filename = get_data('calibration-systems/{}-{}.cpin'.format(residue_name,settings["solvent"]), '')
        temp = settings["temperature"]
        ts = settings["timestep"]
        press = settings["pressure"]
        pH = settings["pH"]
        crate = settings["collision_rate"]
        nspt = settings["nsteps_per_trial"]
        integrator = openmm.LangevinIntegrator(temp, crate, ts)

        if settings["solvent"] == "explicit":
            system.addForce(openmm.MonteCarloBarostat(press, temp))
            mc_titration = CalibrationTitration(system, temp, pH, prmtop, cpin_filename, integrator, pressure=press,
                                                nsteps_per_trial=nspt, implicit=False)
        elif settings["solvent"] == "implicit":
            system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
            mc_titration = CalibrationTitration(system, temp, pH, prmtop, cpin_filename, integrator, pressure=None,
                                                nsteps_per_trial=nspt, implicit=True)
        else:
            raise ValueError("Solvent not recognized")

        platform = openmm.Platform.getPlatformByName(platform_name)
        context = openmm.Context(system, mc_titration.compound_integrator, platform)

        if minimize:
            xcontext, positions = self._minimizer(platform_name, system, positions)
        context.setPositions(positions)  # set to minimized positions

        self.context = context
        self.integrator = integrator
        self.integrator.step(1)
        self.system = system
        self.titration = mc_titration
        self.settings = settings

    def calibrate(self, iterations=10000, mc_every=100, weights_every=1, scheme='eq9'):
        state_updates = 0
        for iteration in range(iterations):
            self.integrator.step(1)
            if iteration % mc_every == 0:
                state_updates += 1
                self.titration.update(self.context)
                if state_updates % weights_every == 0:
                    self.titration.adapt_weights(self.context, scheme)

        return self.titration.get_zeta()

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
        logging.info("Initial energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
        openmm.LocalEnergyMinimizer.minimize(context, 1.0, nsteps)
        logging.info("Final energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
        positions = context.getState(getPositions=True).getPositions(asNumpy=True)
        return context, positions
