# coding=utf-8
"""Augmentation of the OpenMM app layer for constant-pH sampling."""

from simtk.openmm.app import Simulation
from simtk import openmm
from ..driver import ForceFieldProtonDrive

import sys
from datetime import datetime


class ConstantPHSimulation(Simulation):
    """ConstantPHSimulation is an API for running constant-pH simulation in OpenMM analogous to app.Simulation."""

    def __init__(self, topology, system, compound_integrator, drive, move, pools=None, platform=None, platformProperties=None, state=None):
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

    def ncmc(self, updates, pool=None):
        """Advance the simulation by propagating the protonation states a specified number of times.

        Parameters
        ----------

        updates : int
            The number of independent updates to perform.
        pool : str
            The identifier for the pool of residues to update.
        """
        self._update(endUpdate=self.currentUpdate + updates, pool=pool)

    def _update(self, pool=None, endUpdate=None, endTime=None):
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
                self.drive.update(self.move, residue_pool=pool, nattempts=1)
                updatesToGo -= 1
                if endTime is not None and datetime.now() >= endTime:
                    return
            self.drive.update(residue_pool=pool, nattempts=updatesToGo)

            self.currentUpdate += nextUpdates
            if anyReport:
                for reporter, nextR in zip(self.reporters, nextReport):
                    if nextR[0] == nextUpdates:
                        reporter.report(self, self.drive)


class ConstantPHCalibration(ConstantPHSimulation):
    """ConstantPHCalibration is an API for calibrating a constant-pH free energy calculation that uses
    self-adjusted mixture sampling (SAMS) to calculate the relative free energy of different protonation states
    in a simulation system."""


    def __init__(self, topology, system, compound_integrator, samsdriver, platform=None, platformProperties=None, state=None):

        pass
