# coding=utf-8
"""Augmentation of the OpenMM app layer for constant-pH sampling."""

from simtk.openmm.app import Simulation
from simtk import openmm
from .driver import ForceFieldProtonDrive
from .calibration import SelfAdjustedMixtureSampling
from protons.app import proposals
from protons.app.logger import log
import sys
from datetime import datetime


class ConstantPHSimulation(Simulation):
    """ConstantPHSimulation is an API for running constant-pH simulation in OpenMM analogous to app.Simulation."""

    def __init__(self, topology, system, compound_integrator, drive, move=None, pools=None, platform=None, platformProperties=None, state=None):
        """Create a ConstantPHSimulation.

        Parameters
        ----------
        topology : Topology
            A Topology describing the the system to simulate
        system : System or XML file name
            The OpenMM System object to simulate (or the name of an XML file
            with a serialized System)
        compound_integrator : openmm.CompoundIntegrator or XML file name
            The OpenMM Integrator to use for simulating the System (or the name of an XML file with a serialized System)
            Needs to have 2 integrators. The first integrator is used for MD, the second integrator is used for NCMC.
            The NCMC integrator needs to be a CustomIntegrator with the following two properties defined:
                first_step: 0 or 1. 0 indicates the first step in an NCMC protocol and can be used for special actions
                required such as computing the energy prior to perturbation.
                protocol_work: double, the protocol work performed by external moves in between steps.
        drive : protons ProtonDrive
            A ProtonDrive object that can manipulate the protonation states of the system.
        move: StateProposal, default None
            An MC proposal move for updating titration states.
        pools: dict, default is None
            A dictionary of titration group indices to group together.
        platform : Platform=None
            If not None, the OpenMM Platform to use
        platformProperties : map=None
            If not None, a set of platform-specific properties to pass to the
            Context's constructor
        state : XML file name=None
            The name of an XML file containing a serialized State. If not None,
            the information stored in state will be transferred to the generated
            Simulation object.
        """

        super(ConstantPHSimulation, self).__init__(topology, system, compound_integrator, platform=platform, platformProperties=platformProperties, state=state)

        self.drive = drive
        self.drive.attach_context(self.context)

        if move is None:
            move = proposals.UniformProposal()
        self.move = move

        # The index of the current protonation state update
        self.currentUpdate = 0

        # A list of reporters specifically for protonation state updates
        self.update_reporters = []

        if pools is not None:
            drive.define_pools(pools)

        return

    def step(self, steps):
        """Advance the simulation by integrating a specified number of time steps."""
        self._simulate(endStep=self.currentStep + steps)

    def update(self, updates, move=None, pool=None):
        """Advance the simulation by propagating the protonation states a specified number of times.

        Parameters
        ----------

        updates : int
            The number of independent updates to perform.
        move : StateProposal
            Type of move to update system. Uses pre-specified move if not given.
        pool : str
            The identifier for the pool of residues to update.
        """
        self._update(endUpdate=self.currentUpdate + updates, pool=pool, move=move)

    def _update(self, pool=None, move=None, endUpdate=None, endTime=None):

        if move is None:
            move = self.move
        if endUpdate is None:
            endUpdate = sys.maxsize
        nextReport = [None] * len(self.update_reporters)
        while self.currentUpdate < endUpdate and (endTime is None or datetime.now() < endTime):
            nextUpdates = endUpdate - self.currentUpdate
            anyReport = False
            for i, reporter in enumerate(self.update_reporters):
                nextReport[i] = reporter.describeNextReport(self)
                if 0 < nextReport[i][0] <= nextUpdates:
                    nextUpdates = nextReport[i][0]
                    anyReport = True
            updatesToGo = nextUpdates
            while updatesToGo > 1:
                # Only take 1 steps at a time, since each ncmc move is assumed to take a long time
                self.drive.update(move, residue_pool=pool, nattempts=1)
                updatesToGo -= 1
                if endTime is not None and datetime.now() >= endTime:
                    return
            self.drive.update(move, residue_pool=pool, nattempts=updatesToGo)

            self.currentUpdate += nextUpdates
            if anyReport:
                for reporter, nextR in zip(self.update_reporters, nextReport):
                    if nextR[0] == nextUpdates:
                        reporter.report(self)


class ConstantPHCalibration(ConstantPHSimulation):
    """ConstantPHCalibration is an API for calibrating a constant-pH free energy calculation that uses
    self-adjusted mixture sampling (SAMS) to calculate the relative free energy of different protonation states
    in a simulation system."""


    def __init__(self, topology, system, compound_integrator, drive, group_index=0, move=None, pools=None, samsProperties=None, platform=None, platformProperties=None, state=None):
        """Create a ConstantPHCalibration.

        Parameters
        ----------
        topology : Topology
            A Topology describing the the system to simulate
        system : System or XML file name
            The OpenMM System object to simulate (or the name of an XML file
            with a serialized System)
        compound_integrator : openmm.CompoundIntegrator or XML file name
            The OpenMM Integrator to use for simulating the System (or the name of an XML file with a serialized System)
            Needs to have 2 integrators. The first integrator is used for MD, the second integrator is used for NCMC.
            The NCMC integrator needs to be a CustomIntegrator with the following two properties defined:
                first_step: 0 or 1. 0 indicates the first step in an NCMC protocol and can be used for special actions
                required such as computing the energy prior to perturbation.
                protocol_work: double, the protocol work performed by external moves in between steps.
        drive : protons ProtonDrive
            A ProtonDrive object that can manipulate the protonation states of the system.
        group_index: int, default 0
            The index of the titratable group in Drive.titrationGroups.
        move: StateProposal, default None
            An MC proposal move for updating titration states.
        pools: dict, default is None
            A dictionary of titration group indices to group together.
        samsProperties: dict, default is None
            A dictionary with properties for the sams calibration.
        platform : Platform=None
            If not None, the OpenMM Platform to use
        platformProperties : map=None
            If not None, a set of platform-specific properties to pass to the
            Context's constructor
        state : XML file name, default None
            The name of an XML file containing a serialized State. If not None,
            the information stored in state will be transferred to the generated
            Simulation object.
        """

        super(ConstantPHCalibration, self).__init__(topology, system, compound_integrator, drive,  move=move, pools=pools, platform=platform, platformProperties=platformProperties, state=state)
        self.sams = SelfAdjustedMixtureSampling(self.drive, group_index=group_index)

        self.group_index = group_index
        self.scheme = "binary"
        self.beta_sams = 0.5
        self.stage = "burn-in"
        self.flatness_criterion = 0.20
        self.end_of_burnin = 0 # t0, end of the burn in period.
        self.current_adaptation = 0
        self.calibration_reporters = list() # keeps track of calibration results
        self.min_burn = 0
        self.last_dev = None # deviation at last iteration
        self.last_gk = None # weights at last iteration

        if samsProperties is not None:
            if 'scheme' in samsProperties:
                self.scheme = samsProperties['scheme']
            if 'beta' in samsProperties:
                self.beta_sams = samsProperties['beta']
            if 'flatness_criterion' in samsProperties:
                self.flatness_criterion = samsProperties['flatness_criterion']
            if 'min_burn' in samsProperties:
                self.min_burn = samsProperties['min_burn']

    def adapt(self):
        """
        Adapt the weights for the residue that is being calibrated.
        """

        nextReport = [None] * len(self.calibration_reporters)
        anyReport = False
        for i, reporter in enumerate(self.calibration_reporters):
            nextReport[i] = reporter.describeNextReport(self)
            if 0 < nextReport[i][0] <= 1:
                anyReport = True

        self.last_dev = self.sams.adapt_zetas(self.scheme, stage=self.stage, b=self.beta_sams, end_of_burnin=self.end_of_burnin)
        self.last_gk = self.sams.get_gk(group_index=self.group_index)
        self.current_adaptation += 1
        # If the histogram is flat below the criterion
        # Flatness is defined as the sum of absolute deviations
        if self.last_dev < self.flatness_criterion and self.stage == "burn-in" and self.current_adaptation > self.min_burn:
            log.info("Histogram flat below {}. Initiating the slow-gain phase.".format(self.flatness_criterion))
            self.stage = "slow-gain"
            self.end_of_burnin = int(self.sams.n_adaptations)

        if anyReport:
            for reporter, nextR in zip(self.calibration_reporters, nextReport):
                if nextR[0] == 1:
                    reporter.report(self)

        return self.last_dev, self.last_gk

