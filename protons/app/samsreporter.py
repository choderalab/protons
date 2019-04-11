# coding=utf-8
"""Reporter for recording SAMS data."""

import netCDF4
import time
import numpy as np
from copy import deepcopy
from protons.app.simulation import ConstantPHSimulation
from protons.app.driver import Stage, UpdateRule, SAMSApproach


class SAMSReporter:
    """SamsReporter outputs SAMS data from a ConstantPHCalibration to a netCDF4 file."""

    def __init__(self, netcdffile, reportInterval):
        """Create a TitrationReporter.

        Parameters
        ----------
        netcdffile : string
            The netcdffile to write to
        reportInterval : int
            The interval (in adaptation steps) at which to write frames
        
        """
        self._reportInterval = reportInterval
        if isinstance(netcdffile, str):
            self._out = netCDF4.Dataset(netcdffile, mode="w")
        elif isinstance(netcdffile, netCDF4.Dataset):
            self._out = netcdffile
            self._out.sync()  # check if writing works
        else:
            raise ValueError(
                "Please provide a string with the filename location,"
                " or an opened netCDF4 file with write access."
            )
        self._grp = None  # netcdf group that will contain all data.
        self._hasInitialized = False
        self._adaptation = 0  # Number of adaptations written to the file.
        self._group_index = None  # group that is being calibrated
        self._nstates = 0  # number of states of the calibration residue

    @property
    def ncfile(self):
        """The netCDF file currently being written to."""
        return self._out

    def describeNextReport(self, calibration: ConstantPHSimulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        calibration : ConstantPHCalibration
            The calibration to generate a report for

        Returns
        -------
        tuple
            A tuple. The first element is the number of steps
            until the next report.
        """
        updates = (
            self._reportInterval
            - calibration.drive.calibration_state._current_adaptation
            % self._reportInterval
        )
        return tuple([updates])

    def report(self, calibration: ConstantPHSimulation):
        """Generate a report.

        Parameters
        ----------
        calibration : ConstantPHCalibration
            The Simulation to generate a report for
        """
        if not self._hasInitialized:
            self._initialize_constants(calibration)
            self._create_netcdf_structure()
            self._record_metadata(calibration)
            self._hasInitialized = True

        # Gather and record all data for the current update
        self._write_adaptation(calibration)
        # Update number of written updates
        self._adaptation += 1

        # Write the values.
        self._out.sync()

    def _write_adaptation(self, simulation: ConstantPHSimulation):
        """Record data for the current update in the netCDF file.

        Parameters
        ----------
        simulation : ConstantPHSimulation
            The calibration to generate a report for
        """
        drv = simulation.drive
        sams = simulation.sams
        iadapt = self._adaptation
        # The iteration of the protonation state update attempt. [update]
        self._grp["adaptation"][
            iadapt
        ] = simulation.drive.calibration_state._current_adaptation
        self._grp["g_k"][iadapt, :] = deepcopy(
            simulation.drive.calibration_state.free_energies[:]
        )
        self._grp["flatness"][iadapt] = simulation.last_dev
        self._grp["stage"][iadapt] = simulation.drive.calibration_state._stage.value
        if simulation.drive.calibration_state._stage == Stage.FASTDECAY:
            self._grp["end_of_slowdecay"][
                0
            ] = simulation.drive.calibration_state._end_of_slowdecay
            self._grp["end_of_slowdecay"][
                0
            ] = simulation.drive.calibration_state._end_of_slowdecay

    def _initialize_constants(self, simulation: ConstantPHSimulation):
        """Initialize a set of constants required for the reports

        Parameters
        - simulation (ConstantPHSimulation) The simulation to generate a report for
        """
        system = simulation.context.getSystem()
        drive = simulation.drive
        if drive.calibration_state.approach is SAMSApproach.ONE_RESIDUE:
            group_index = drive.calibration_state.group_index
        else:
            group_index = np.nan
        self._group_index = group_index
        self._ngroups = len(simulation.drive.titrationGroups)
        self._nstates = drive.calibration_state.free_energies.size
        self._perturbation_steps = drive.perturbations_per_trial

    def _create_netcdf_structure(self):
        """Construct the netCDF directory structure and variables
        """

        grp = self._out.createGroup("Protons/SAMS")
        grp.description = (
            "This group contains data stored by a SamsReporter object from protons."
        )
        grp.history = "This group was created on UTC [{}].".format(
            time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime())
        )

        adapt_dim = grp.createDimension("adaptation")
        state_dim = grp.createDimension("state", self._nstates)

        # Variables written every adaptation
        adaptation = grp.createVariable("adaptation", int, ("adaptation",))
        adaptation.description = "The current adaptation step of the SAMS algorithm."

        g_k = grp.createVariable("g_k", float, ("adaptation", "state"))
        g_k.description = (
            "Weight of each individual state, relative to g_0. [adaptation, state]"
        )
        flatness = grp.createVariable("flatness", float, ("adaptation",))
        flatness.description = "Measure of how flat the histogram is. (Sum over all deviations from target)."

        approach = grp.createVariable("approach", int)
        approach.description = (
            "The approach used with sams, one-site (0) or multi-site (1)"
        )

        # Metadata
        # burn-in or slow-gain
        group = grp.createVariable("group_index", int)
        group.description = "The index of the titration group that is being calibrated in the drive. (not the topology index)."

        stage = grp.createVariable("stage", int, ("adaptation"))
        stage.description = (
            "Current stage in the SAMS protocol (burn-in (0) or slow-gain(1))."
        )
        # binary vs global
        scheme = grp.createVariable("update_rule", int)
        scheme.description = (
            "The SAMS update rule that is being used (binary(0)/global(1))."
        )
        # Two stage beta value
        beta = grp.createVariable("beta", float)
        beta.description = (
            "SAMS parameter beta, determining the scale of the adaptation (0.5< b <1.0"
        )
        # t0, end of the burn in period
        t0 = grp.createVariable("end_of_slowdecay", int)
        t0.description = (
            "The end of the burn-in period, necessary for calculating gain-factor."
        )
        # The criterion for flatness
        flatness_cr = grp.createVariable("flatness_criterion", float)
        flatness_cr.description = "The criterion for flatness, after which the sams stage switches to slow gain."
        # Minimum period of burn-in
        minburn = grp.createVariable("min_burn", int)
        minburn.description = "The minimum number of steps of burn-in that are run before flatness is estimated."

        self._grp = grp
        self._out.sync()

        return

    def _record_metadata(self, calibration):
        """Records all metadata that doesn't depend on the adaptation dimension, into the netCDF file.

        Parameters
        ----------
        calibration - ConstantPHCalibration
        """
        sams = calibration.sams
        grp = self._grp

        grp["approach"][0] = calibration.drive.calibration_state.approach.value
        # only set group index when onesite is used
        if calibration.drive.calibration_state.approach is SAMSApproach.ONE_RESIDUE:
            grp["group_index"][0] = calibration.drive.calibration_state.group_index

        grp["update_rule"][0] = calibration.drive.calibration_state._update_rule.value
        grp["beta"][0] = calibration.drive.calibration_state._beta_sams
        grp["end_of_slowdecay"][
            0
        ] = calibration.drive.calibration_state._end_of_slowdecay
        grp["min_burn"][0] = calibration.drive.calibration_state._min_burn
        grp["flatness_criterion"][
            0
        ] = calibration.drive.calibration_state._flatness_criterion
        return
