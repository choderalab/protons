# coding=utf-8
"""A reporter for ConstantPHSimulation that tracks NCMC statistics."""

import netCDF4
import time
import numpy as np


class NCMCReporter:
    """NCMCReporter outputs NCMC statistics from a ConstantPHSimulation to a netCDF4 file."""

    def __init__(self, netcdffile, reportInterval: int, cumulativeworkInterval: int=0):
        """Create a TitrationReporter.

        Parameters
        ----------
        netcdffile : string
            The netcdffile to write to
        reportInterval : int
            The interval (in update steps) at which to write frames
        cumulativeworkInterval : 
            Store cumulative work every m perturbation steps (default 0) in the NCMC protocol.
            Set to 0 for not storing.
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
        self._perturbation_steps = np.empty([]) # indices of perturbation steps stored per ncmc trial
        self._cumulative_work_interval = cumulativeworkInterval # store cumulative work ever m perturbations
        self._has_swapper = False # True if using saltswap

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
        self._grp['initial_state'][iupdate,:] = drv._last_attempt_data.initial_states
        self._grp['proposed_state'][iupdate,:] = drv._last_attempt_data.proposed_states
        self._grp['total_work'][iupdate,] = drv._last_attempt_data.work
        self._grp['logp_ratio_residue_proposal'][iupdate,]= drv._last_attempt_data.logp_ratio_residue_proposal

        if self._has_swapper:
            self._grp['logp_ratio_salt_proposal'][iupdate,] = drv._last_attempt_data.logp_ratio_salt_proposal
            self._grp['initial_ion_states'][iupdate,:] = drv._last_attempt_data.initial_ion_states
            self._grp['proposed_ion_states'][iupdate,:] = drv._last_attempt_data.proposed_ion_states

        self._grp['logp_accept'][iupdate,] = drv._last_attempt_data.logp_accept

        if self._cumulative_work_interval > 0:
            self._grp['cumulative_work'][iupdate,:] = \
             np.asarray(drv.ncmc_stats_per_step)[self._perturbation_steps,0]

    def _initialize_constants(self, simulation):
        """Initialize a set of constants required for the reports

        Parameters
        - simulation (ConstantPHSimulation) The simulation to generate a report for
        """
        system = simulation.context.getSystem()
        driver = simulation.drive
        self._ngroups = len(simulation.drive.titrationGroups)
        if self._cumulative_work_interval > 0:            
            self._perturbation_steps = np.arange(0,driver.perturbations_per_trial, self._cumulative_work_interval)
        
        # If ions are being swapped as part of this simulation
        if driver.swapper is not None:
            self._has_swapper = True
            self._nsaltsites = len(driver.swapper.stateVector)

    def _create_netcdf_structure(self):
        """Construct the netCDF directory structure and variables
        """

        grp = self._out.createGroup("Protons/NCMC")
        grp.description = "This group contains data stored by a NCMCReporter object from protons."
        grp.history = "This group was created on UTC [{}].".format(
            time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

        update_dim = grp.createDimension('update')
        residue_dim = grp.createDimension('residue', self._ngroups)
        if self._has_swapper:
            ion_dim = grp.createDimension('ion_site', self._nsaltsites)

        # Variables written every update
        update = grp.createVariable('update', int, ('update',))
        update.description = "The iteration of the protonation state update attempt. [update]"
        residue_state = grp.createVariable('state', int, ('update', 'residue',))
        residue_state.description = "The present state of the residue. [update,residue]"
        residue_initial_state = grp.createVariable('initial_state', int, ('update', 'residue',))
        residue_initial_state.description = "The state of the residue, before switching.[update,residue]"
        residue_proposed_state = grp.createVariable('proposed_state', int, ('update', 'residue',))
        residue_proposed_state.description = "The proposed state of the residue. [update,residue]"
        if self._has_swapper:
            salt_initial_states = grp.createVariable('initial_ion_states', int, ('update', 'ion_site',))
            salt_proposed_states = grp.createVariable('proposed_ion_states', int, ('update', 'ion_site',))
            salt_initial_states.description = "The initial state of water molecules treated by saltswap. [update, ion_site]"
            salt_proposed_states.description = "The proposed state of water molecules treated by saltswap. [update,ion_site]"

        total_work = grp.createVariable('total_work', float, ('update',))
        total_work.description = "The work of the protocol, including Δg_k. [update]"
        total_work.unit = "unitless (W/kT)"
        logp_ratio_residue = grp.createVariable('logp_ratio_residue_proposal', float, ('update',))
        logp_ratio_residue.description = "The log P ratio (reverse/forward) of proposing this residue and state. [update]"
        if self._has_swapper:
            logp_ratio_salt = grp.createVariable('logp_ratio_salt_proposal', float, ('update',))
            logp_ratio_salt.description = "The log P ratio (reverse/forward) of proposing this ion swap. [update]"
        logp_accept = grp.createVariable('logp_accept', float, ('update',))
        logp_accept.description = "The log P acceptance ratio for accepting the full NCMC move. [update]"

        # Variables written per ncmc step
        if self._cumulative_work_interval > 0:
            perturbation_dim = grp.createDimension('perturbation', len(self._perturbation_steps))
            cumulative_work = grp.createVariable('cumulative_work', float, ('update', 'perturbation',), zlib=True)
            cumulative_work.description = "Cumulative work at the end of each NCMC perturbation step.[update,perturbation]"
            cumulative_work.unit = "unitless (W/kT)"
            perturbation = grp.createVariable('perturbation', int, ('perturbation',))
            perturbation.description = "The step indices of the NCMC protocol. [perturbation]"

        self._grp = grp
        self._out.sync()

        return

    def _record_metadata(self, simulation):
        """Records all metadata that doesn't depend on the update dimension, into the netCDF file.

        Parameters
        ----------
        simulation - ConstantPHSimulation
        """
        if self._cumulative_work_interval > 0:
            self._grp["perturbation"][:] = self._perturbation_steps[:]
        return
