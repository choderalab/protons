# coding=utf-8
"""Reporters for Constant-pH simulations."""

import netCDF4
import time
import numpy as np


class MetadataReporter:
    """MetadataReporter outputs protonation state metadata for the system to a netCDF4 file."""

    def __init__(self, netcdffile, shared=False):
        """Create a MetadataReporter.

        Parameters
        ----------
        netcdffile : string
            The netcdffile to write to
        shared: bool, default False
            Indicate whether the netcdf file is shared by other reporters. Prevents file closing.

        """
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
        # never reports
        if self._hasInitialized:
            return tuple([np.infty])
        else:
            return tuple([1])

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

        # Write the values.
        self._out.sync()

    def _initialize_constants(self, simulation):
        """Initialize a set of constants required for the reports

        Parameters
        - simulation (ConstantPHSimulation) The simulation to generate a report for
        """
        system = simulation.system
        driver = simulation.drive
        self._ngroups = len(simulation.drive.titrationGroups)

    def _create_netcdf_structure(self):
        """Construct the netCDF directory structure and variables
        """

        grp = self._out.createGroup("MetadataReporter")
        grp.description = "This group contains data stored by a MetadataReporter object from protons."
        grp.history = "This group was created on UTC [{}].".format(time.strftime("%a, %d %b %Y %H:%M:%S +0000", time.gmtime()))

        residue_dim = grp.createDimension('residue', self._ngroups)
        state_dim = grp.createDimension('state')
        atom_dim = grp.createDimension('atom')

        # Metadata, written once
        residue_idx = grp.createVariable("residue_index", int, ('residue',))
        residue_idx.description = "The index of the titratable residue group in the drive (NOT the topology index)."

        residue_types = grp.createVariable("residue_type", str, ('residue',))
        residue_types.description = "The type of residue, e.g. pdb code/ffxml residue name. [residue]"

        residue_name = grp.createVariable("residue_name", str, ('residue',))
        residue_name.description = "A name to recognize the residue by. [residue]"

        state_gk = grp.createVariable("g_k", float, ('residue', 'state',))
        state_gk.description = "The free energy bias g_k for each titration state. [residue,state]"

        state_proton_count = grp.createVariable("proton_count", int, ('residue', 'state',))
        state_proton_count.description = "The amount of (titratable) protons active in the state. [residue,state]"

        state_charge = grp.createVariable("total_charge", float, ('residue', 'state',))
        state_charge.description = "The total charge of a state. [residue, state]"

        atom_index = grp.createVariable('atom_index', int, ('residue', 'atom',))
        atom_index.description = "The index of the residue atoms in the topology. [residue,atom]"

        charge = grp.createVariable('charge', float, ('residue', 'atom', 'state',))
        charge.description = "The charge of residue atoms per state. [residue,atom,state]"

        self._grp = grp
        self._out.sync()

        return

    def _record_metadata(self, simulation):
        """Records all metadata that doesn't depend on the update dimension, into the netCDF file.

        Parameters
        ----------
        simulation - ConstantPHSimulation
        """
        drv = simulation.drive

        # Per residue variable
        for ires, residue in enumerate(drv.titrationGroups):
            self._grp['residue_index'][ires] = residue.index
            self._grp['residue_type'][ires] = residue.residue_type
            self._grp['residue_name'][ires] = residue.name
            # Per residue, per atom variable
            for iatom,atom_index in enumerate(residue.atom_indices):
                self._grp['atom_index'][ires,iatom] = atom_index
            # Per residue, per state variable
            for istate, state in enumerate(residue):
                self._grp['g_k'][ires, istate] = state.g_k
                self._grp['proton_count'][ires, istate] = state.proton_count
                self._grp['total_charge'][ires, istate] = state.total_charge
                # Per residue, per state, per atom
                for iatom, charge in enumerate(state.charges):
                    self._grp['charge'][ires, iatom, istate] = charge
        return

    def __del__(self):
        """Clean up on deletion of object."""
        if self._close_file:
            self._out.close()
