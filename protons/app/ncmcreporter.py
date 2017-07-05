# coding=utf-8
"""A reporter for ConstantPHSimulation that tracks NCMC statistics."""

import netCDF4
import time
import numpy as np

class NCMCReporter(object):
    """NCMCReporter outputs NCMC statistics from a ConstantPHSimulation to a netCDF4 file."""

    def __init__(self, netcdffile, reportInterval, shared=False):
        """Create a TitrationReporter.

        Parameters
        ----------
        netcdffile : string
            The netcdffile to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        shared: bool, default False
            Indicate whether the netcdf file is shared by other

        """
        self._reportInterval = reportInterval
        if isinstance(netcdffile, str):
            self._out = netCDF4.Dataset(netcdffile, mode="w")
        elif isinstance(netcdffile, netCDF4.Dataset):
            self._out = netcdffile
            self._out.sync() # check if writing works
        else:
            raise ValueError("Please provide a string with the filename location,"
                             " or an opened netCDF4 file with write access.")
        self._grp = None # netcdf group that will contain all data.
        self._hasInitialized = False
        self._update = 0 # Number of updates written to the file.
        self._ngroups = 0 # number of residues
        self._perturbation_steps = 0 # number of perturbation steps per ncmc trial

        if shared:
            self._close_file = False # close the file on deletion of this reporter.
        else:
            self._close_file = True

    @property
    def ncfile(self):
        """The netCDF file currently being written to."""
        return self._out

    def describeNextReport(self, simulation):
        """Get information about the next report this object will generate.

        Parameters
        ----------
        simulation : ConstantPHSimulation
            The Simulation to generate a report for

        Returns
        -------
        tuple
            A tuple. The first element is the number of steps
            until the next report.
        """
        updates = self._reportInterval - simulation.currentUpdate % self._reportInterval
        return tuple([updates])

    def report(self, simulation):
        """Generate a report.

        Parameters
        ----------
        simulation : ConstantPHSimulation
            The Simulation to generate a report for
        """
        if not self._hasInitialized:
            self._initialize_constants(simulation)
            self._create_netcdf_structure()
            self._record_metadata(simulation)
            self._hasInitialized = True

        # Gather and record all data for the current update
        self._write_update(simulation)
        # Update number of written updates
        self._update += 1

        # Write the values.
        self._out.sync()

    def _write_update(self, simulation):
        """Record data for the current update in the netCDF file.

        Parameters
        ----------
        simulation : ConstantPHSimulation
            The Simulation to generate a report for
        """
        drv = simulation.drive
        iupdate = self._update
        # The iteration of the protonation state update attempt. [update]
        self._grp['update'][iupdate] = simulation.currentUpdate
        for ires, residue in enumerate(drv.titrationGroups):
            # The present state of the residue. [update,residue]
            self._grp['state'][iupdate,] = residue.state_index

        # first array is initial states, second is proposed state, last is work
        self._grp['initial_state'][iupdate,:] = drv.last_proposal[0]
        self._grp['proposed_state'][iupdate,:] = drv.last_proposal[1]
        self._grp['total_work'][iupdate,] = drv.last_proposal[2]
        self._grp['cumulative_work'][iupdate,:] = np.asarray(drv.ncmc_stats_per_step)[:,0]

    def _initialize_constants(self, simulation):
        """Initialize a set of constants required for the reports

        Parameters
        - simulation (ConstantPHSimulation) The simulation to generate a report for
        """
        system = simulation.system
        driver = simulation.drive
        self._ngroups = len(simulation.drive.titrationGroups)
        self._perturbation_steps = driver.perturbations_per_trial

    def _create_netcdf_structure(self):
        """Construct the netCDF directory structure and variables
        """

        grp = self._out.createGroup("NCMCReporter")
        grp.description = "This group contains data stored by a NCMCReporter object from protons."
        grp.history = "This group was created on UTC [{}].".format(
            time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

        update_dim = grp.createDimension('update')
        residue_dim = grp.createDimension('residue', self._ngroups)
        perturbation_dim = grp.createDimension('perturbation', self._perturbation_steps)

        # Variables written every update
        update = grp.createVariable('update', int, ('update',))
        update.description = "The iteration of the protonation state update attempt. [update]"
        residue_state = grp.createVariable('state', int, ('update', 'residue',))
        residue_state.description = "The present state of the residue. [update,residue]"
        residue_initial_state = grp.createVariable('initial_state', int, ('update', 'residue',))
        residue_initial_state.description = "The state of the residue, before switching.[update,residue]"
        residue_proposed_state = grp.createVariable('proposed_state', int, ('update', 'residue',))
        residue_proposed_state.description = "The proposed state of the residue. [update,residue]"
        total_work = grp.createVariable('total_work', float, ('update',))
        total_work.description = "The work of the protocol, including Δg_k. [update]"
        total_work.unit = "unitless (W/kT)"
        cumulative_work = grp.createVariable('cumulative_work', float, ('update', 'perturbation',), zlib=True)
        cumulative_work.description = "Cumulative work at the end of each NCMC perturbation step.[update,perturbation]"
        cumulative_work.unit = "unitless (W/kT)"

        self._grp = grp
        self._out.sync()

        return

    def _record_metadata(self, simulation):
        """Records all metadata that doesn't depend on the update dimension, into the netCDF file.

        Parameters
        ----------
        simulation - ConstantPHSimulation
        """
        return

    def __del__(self):
        """Clean up on deletion of object."""
        if self._close_file:
            self._out.close()
