# coding=utf-8
"""Reporters for Constant-pH simulations."""

import netCDF4
import time
import numpy as np


class TitrationReporter:
    """TitrationReporter outputs protonation states of residues in the system to a netCDF4 file."""

    def __init__(self, netcdffile, reportInterval, shared=False):
        """Create a TitrationReporter.

        Parameters
        ----------
        netcdffile : string
            The netcdffile to write to
        reportInterval : int
            The interval (in time steps) at which to write frames
        shared: bool, default False
            Indicate whether the netcdf file is shared by other reporters. Prevents file closing.

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
            self._hasInitialized = True

        # Gather and record all data for the current update
        self._write_update(simulation)
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
        all_stat = np.ones(self._natoms_topology, dtype=np.bool_)
        for ires, residue in enumerate(drv.titrationGroups):
            # The present state of the residue. [update,residue]
            self._grp['state'][iupdate, ires] = residue.state_index
            # Indicator of whether an atom is on/off (charged) at present. [update,residue,atom]
            for iatom, status in enumerate(residue.atom_status):
                if status != 1:
                    topo_index = residue.atom_indices[iatom]
                    all_stat[topo_index] = False
        self._grp['atom_status'][iupdate, :] = all_stat[:]

    def _initialize_constants(self, simulation):
        """Initialize a set of constants required for the reports

        Parameters
        - simulation (ConstantPHSimulation) The simulation to generate a report for
        """
        system = simulation.system
        driver = simulation.drive
        self._ngroups = len(simulation.drive.titrationGroups)
        self._natoms_topology = simulation.topology.getNumAtoms()

    def _create_netcdf_structure(self):
        """Construct the netCDF directory structure and variables
        """

        grp = self._out.createGroup("Protons/Titration")
        grp.description = "This group contains data stored by a TitrationReporter object from protons."
        grp.history = "This group was created on UTC [{}].".format(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

        update_dim = grp.createDimension('update')
        residue_dim = grp.createDimension('residue', self._ngroups)
        atom_dim = grp.createDimension('atom', self._natoms_topology)

        # Variables written every update
        update = grp.createVariable('update', int, ('update',))
        update.description = "The iteration of the protonation state update attempt. [update]"

        residue_state = grp.createVariable("state", int, ('update', 'residue',))
        residue_state.description = "The present state of the residue. [update,residue]"

        atom_status = grp.createVariable("atom_status", 'u1', ('update', 'atom'), zlib=True)
        atom_status.description = "Byte indicator (1/0) of whether an atom is on/off (charged) at present, where the atom index is equal to the topology.[update,atom]"

        self._grp = grp
        self._out.sync()

        return

    def __del__(self):
        """Clean up on deletion of object."""
        if self._close_file:
            self._out.close()
