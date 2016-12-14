# coding=utf-8
"""
Functionality to store simulation output by means of netCDF4 files
"""
import netCDF4
import numpy as np
import protons
from simtk import unit, openmm

# Specification of attributes and variable types to be stored and read
# tuples of (name, type, unit)

# Single value attributes of ProtonDrive (states are defined separately)
drive_attributes = [('naccepted',int, None), ("nrejected",int, None), ("nattempted", int, None)]
# Global variables of GHMC integrators
ghmc_global_variables = [('naccept', int, None), ('ntrials', int, None), ('work', np.float64, 'kilojoule_per_mole')]
# Context state variables
state_vars = [('total_energy', np.float64, 'kilojoule_per_mole'),
              ('kinetic_energy', np.float64, 'kilojoule_per_mole'),
              ('potential_energy', np.float64, 'kilojoule_per_mole'),
              ('temperature', np.float64, 'kelvin'),]


def netcdf_file(filename, num_titratable_groups, num_iterations=None, degrees_of_freedom=-1):
    """ Creates an NC file with groups and variables for ProtonDrives, Systems, and GHMCIntegrator

    Parameters
    ----------
    filename - str
        Name of the nc file that is to be created
    num_titratable_groups - int
        Number of titratable groups in the system
    num_iterations - int, default is None
        Number of iterations, leave None for unlimited

    Returns
    -------
    Dataset - netCDF4 dataset
    """
    ncfile = netCDF4.Dataset(filename, mode='w', format='NETCDF4')
    ncfile.degrees_of_freedom = degrees_of_freedom
    ncfile.version = protons.__version__
    ncfile.createDimension("iteration", size=num_iterations)
    ncfile.createVariable("iteration", int, ("iteration",))
    ncfile.createDimension('group', size=num_titratable_groups)

    # System variable group
    system = ncfile.createGroup('State')
    for state_var, state_type, state_unit in state_vars:
        newvar = system.createVariable(state_var, state_type, ('iteration',))
        if state_unit is not None:
            newvar.unit = state_unit

    # Proton drive variable group
    proton_drive = ncfile.createGroup('ProtonDrive')
    for attribute,attrtype, attrunit in drive_attributes:
        newvar = proton_drive.createVariable(attribute, attrtype, ('iteration',))
        if attrunit is not None:
            newvar.unit = attrunit

    # Titration state at each iteration, for each residue
    proton_drive.createVariable('titration_states', int, ('iteration','group'))

    # NCMC integrator variable subgroup
    ncmc_integrator = proton_drive.createGroup('NCMCIntegrator')

    for globvar, vartype, varunit in ghmc_global_variables:
        newvar = ncmc_integrator.createVariable(globvar, vartype, ('iteration',))
        if varunit is not None:
            newvar.unit = varunit

    # GHMC integrator variable group
    ghmc_integrator = ncfile.createGroup('GHMCIntegrator')
    for globvar, vartype, varunit in ghmc_global_variables:
        newvar = ghmc_integrator.createVariable(globvar, vartype, ('iteration',))
        if varunit is not None:
            newvar.unit = varunit

    return ncfile


def save_drive_data(ncfile, drive, iteration):
    """
    Store all relevant properties of a ProtonDrive type object in a netcdf dataset.

    Parameters
    ----------
    ncfile - netCDF4.Dataset
        An opened netCDF4 dataset object.
    drive - ProtonDrive object
        This function will save all relevant properties of the supplied drive
    iteration - the current iteration

    """
    # Append new iteration to the ProtonDrive variable group
    for attribute, attrtype, attrunit in drive_attributes:
        ncfile['ProtonDrive/{}'.format(attribute)][iteration] = getattr(drive,attribute)

    # Append the titrationstate for each group
    for titration_group_index, titration_state in enumerate(drive.titrationStates):
        ncfile['ProtonDrive/titration_states'][iteration, titration_group_index] = titration_state

    ncmc_integrator = drive.ncmc_propagation_integrator

    # Append new iteration to the NCMC integrator variable group
    for globvar, vartype, varunit in ghmc_global_variables:
        ncfile['ProtonDrive/NCMCIntegrator/{}'.format(globvar)][iteration] = ncmc_integrator.getGlobalVariableByName(globvar)

    return


def save_ghmc_integrator_data(ncfile, integrator, iteration):
    """
    Store relevant properties of a GHMCIntegrator object in a netcdf dataset

    Parameters
    ----------
    ncfile - netCDF4.Dataset
        An opened netCDF4 Dataset object.
    integrator - protons.integrators.GHMCIntegrator
        Custom GHMCIntegrator class to save variables from
    iteration - the current iteration
    """
    # Append new iteration to the GHMCIntegrator variable group
    for globvar, vartype, varunit in ghmc_global_variables:
        ncfile['GHMCIntegrator/{}'.format(globvar)][iteration] = integrator.getGlobalVariableByName(globvar)


def save_state_data(ncfile, context, system, iteration):
    """
    Store state properties from a simulation Context, and system object

    Parameters
    ----------

    ncfile - netCDF4.Dataset
        An opened netCDF4 Dataset object
    context - openmm simulation Context object
        The context from which to save the state
    system - openmm simulation System object
        The simulation system
    iteration - int
        The current iteration

    Returns
    -------

    """

    # See if degrees of freedom has been instantiated, if not calculate
    if ncfile.degrees_of_freedom < 0:
        ncfile.degrees_of_freedom = _calculate_degrees_of_freedom(system)

    # Extract the data from state

    state = context.getState(getEnergy=True)
    statedata = {'potential_energy': state.getPotentialEnergy().value_in_unit(unit.kilojoule_per_mole),
            'kinetic_energy' : state.getKineticEnergy().value_in_unit(unit.kilojoule_per_mole)}
    statedata['total_energy'] = statedata['potential_energy'] + statedata['kinetic_energy']
    # T = 2 * Kin / ( R * dof)
    statedata['temperature'] = (2 * statedata['kinetic_energy'] * unit.kilojoule_per_mole / (ncfile.degrees_of_freedom * unit.MOLAR_GAS_CONSTANT_R)).value_in_unit(unit.kelvin)

    for state_var, state_type, state_unit in state_vars:
        ncfile['State/{}'.format(state_var)][iteration] = statedata[state_var]


def _calculate_degrees_of_freedom(system):
    """
    Calculate degrees of freedom of an OpenMM system

    Based on simtk/openmm/app/statedatareporter.py

    Parameters
    ----------
    system - OpenMM simulation System

    Returns
    -------
    int - degrees of freedom
    """
    degrees_of_freedom = 0
    for i in range(system.getNumParticles()):
        # If particle is real (has mass), add 3 degrees for each coordinate
        if system.getParticleMass(i) > 0 * unit.dalton:
            degrees_of_freedom += 3
    # Every constraint removes a degree of freedom
    degrees_of_freedom -= system.getNumConstraints()
    # Adding a CMMotionRemover removes 3 degrees of freedom from the system
    if any(type(system.getForce(i)) == openmm.CMMotionRemover for i in range(system.getNumForces())):
        degrees_of_freedom -= 3

    return degrees_of_freedom


def save_all(ncfile, drive, integrator, context, system, iteration):
    """
    Save relevant properties from ProtonDrive, GHMCIntegrator, and context plus system with one function


    Parameters
    ----------

    ncfile - netCDF4.Dataset
        An opened netCDF4 Dataset object
    drive - ProtonDrive object
        This function will save all relevant properties of the supplied drive
    integrator - protons.integrators.GHMCIntegrator
        Custom GHMCIntegrator class to save variables from
    context - openmm simulation Context object
        The context from which to save the state
    system - openmm simulation System object
        The simulation system
    iteration - int
        The current iteration

    Returns
    -------

    """
    # Extend the iteration variable ( and dimension )
    ncfile['iteration'][iteration] = iteration
    save_drive_data(ncfile, drive, iteration)
    save_ghmc_integrator_data(ncfile, integrator, iteration)
    save_state_data(ncfile, context, system, iteration)


def walk_netcdf_tree(top):
    """
    Recurse through a given netcdf Dataset or Group and yield children

    Parameters
    ----------
    top - top branch of a netcdf file (Group of Dataset)

    Yields
    ------
    netCDF4.Groups

    """
    values = top.groups.values()
    yield values
    for value in top.groups.values():
        for children in walk_netcdf_tree(value):
            yield children


def display_content_structure(rootgrp):
    """
    Display the contents of a netcdf directory

    Parameters
    ----------
    rootgrp - the directory to walk through
    """
    for children in walk_netcdf_tree(rootgrp):
        for child in children:
            print(child)
            for var in child.variables:
                print('\t', var)