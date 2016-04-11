#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""
Constant salt dynamics in OpenMM.

Description
-----------

This module implements a pure python constant salt functionality in OpenMM, based on OpenMM's constant pH implementation.
Constant salt simulations are achieved using the semi grand canonical enemble, which molecules/particles to change indentity.

Non-equilibrium candidate Monte Carlo (NCMC) can be used to increase acceptance rates of switching.

Notes
-----

The code is still being written.


References
----------

[1] Frenkel and Smit, Understanding Molecular Simulation, from algorithms to applications, second edition, 2002 Academic Press.
    (Chapter 9, page 225 to 231)
[2] Nilmeir, Crooks, Minh, Chodera, Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation,PNAS,108,E1009

Examples
--------

Coming soon to an interpreter near you!

TODO
----

    * Add NCMC switching moves.

Copyright and license
---------------------

@author Gregory A. Ross <gregoryross.uk@gmail.com>

"""
from __future__ import print_function
import re
import sys
import math
import random
import copy
import time
import numpy as np
from scipy.misc import logsumexp
import simtk
import simtk.openmm as openmm
import simtk.unit as units
import pymbar
from openmmtools.integrators import VelocityVerletIntegrator

# MODULE CONSTANTS
kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA
kB = kB.in_units_of(units.kilojoule_per_mole / units.kelvin)

def strip_in_unit_system(quant, unit_system=units.md_unit_system, compatible_with=None):
    """Strips the unit from a simtk.units.Quantity object and returns it's value conforming to a unit system

    Parameters
    ----------
    quant : simtk.unit.Quantity
        object from which units are to be stripped
    unit_system : simtk.unit.UnitSystem:
        unit system to which the unit needs to be converted, default is the OpenMM unit system (md_unit_system)
    compatible_with : simtk.unit.Unit
        Supply to make sure that the unit is compatible with an expected unit

    Returns
    -------
    quant : object with no units attached
    """
    if units.is_quantity(quant):
        if compatible_with is not None:
            quant = quant.in_units_of(compatible_with)
        return quant.value_in_unit_system(unit_system)
    else:
        return quant


class ConstantSalt(object):
    """
    Monte Carlo driver for semi-grand canonical ensemble moves.

    Class that allows for particles and/or molecules to change identities and forcefield.

    """

    def __init__(self, system, topology,temperature, delta_chem, integrator, pressure=None, nattempts_per_update=50, debug=False,
        nsteps_per_trial=0, ncmc_timestep=1.0*units.femtoseconds, waterName="HOH", cationName='Na+', anionName='Cl-', cationparams = None, anionparams = None,implicit=False):
        """
        Initialize a Monte Carlo titration driver for semi-grand ensemble simulation.

        Parameters
        ----------
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature to be simulated.
        delta_chem : float
            The difference in chemical potential for swapping 2 water molecules for Na Cl.
        chemical_names : list of strings
            Names of each of the residues whose parameters will be exchanged.
        integrator : simtk.openmm.integrator
            The integrator used for dynamics
        pressure : simtk.unit.Quantity compatible with atmospheres, optional, default=None
            For explicit solvent simulations, the pressure.
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

        Todo
        ----
        """

        # Set defaults.

        # Store parameters.
        self.system = system
        self.topology = topology
        self.temperature = temperature
        self.pressure = pressure
        self.delta_chem = delta_chem
        self.debug = debug

        self.anionName = anionName
        self.cationName = cationName
        self.waterName = waterName

        # Create a Verlet integrator to handle NCMC integration
        self.compound_integrator = openmm.CompoundIntegrator()
        self.compound_integrator.addIntegrator(integrator)
        self.verlet_integrator = VelocityVerletIntegrator(ncmc_timestep)
        self.compound_integrator.addIntegrator(self.verlet_integrator)
        self.compound_integrator.setCurrentIntegrator(0)  # make user integrator active

        # Set constraint tolerance.
        self.verlet_integrator.setConstraintTolerance(integrator.getConstraintTolerance())

        # Store force object pointer.
        # Unlike constph, only altering nonbonded force (like dual topology calculations).
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if force.__class__.__name__ == 'NonbondedForce':
                self.forces_to_update = force

        # Check that system has MonteCarloBarostat if pressure is specified.
        if pressure is not None:
            forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
            if 'MonteCarloBarostat' not in forces:
                raise Exception("`pressure` is specified, but `system` object lacks a `MonteCarloBarostat`")

        self.mutable_residues = self.identifyResidues(self.topology,residue_names=(self.waterName,self.anionName,self.cationName))

        self.stateVector = self.initializeStateVector()
        self.water_parameters = self.retrieveResidueParameters(self.topology,self.system,self.waterName)
        self.cation_parameters = self.initializeIonParameters(ion_name=self.cationName,ion_params=None)
        self.anion_parameters = self.initializeIonParameters(ion_name=self.anionName,ion_params=None)

        # Describing the identities of water and ions with numpy vectors


        if implicit and maintainChargeNeutrality:
            raise ValueError("Implicit solvent and charge neutrality are mutually exclusive.")

        # Keep track of forces and whether they're cached.
        self.precached_forces = False

        # Track simulation state
        self.kin_energies = units.Quantity(list(), units.kilocalorie_per_mole)
        self.pot_energies = units.Quantity(list(), units.kilocalorie_per_mole)
        self.states_per_update = list()

        # Store list of exceptions that may need to be modified.
        self.nattempts_per_update = nattempts_per_update

        # Reset statistics.
        self.resetStatistics()

        return

    def retrieveResidueParameters(self, topology, system, resname):
        """
        Retrieves the non-bonded parameters for a specified residue.

        Parameters
        ----------
        topology : simtk.openmm.app.topology
            The topology from which water residues are to be identified.
        system : simtk.openmm.System
            The System object from which parameters are to be extracted.
        resname : str
            The residue name of the residue from which parameters are to be retrieved.

        Returns
        -------
        param_list : list of dict of str:float
            List of NonbondedForce parameter dict ('charge', 'sigma', 'epsilon') for each atom.

        Warnings
        --------
        * Only `NonbondedForce` parameters are returned
        * If the system contains more than one `NonbondedForce`, behavior is undefined

        """
        # Find the NonbondedForce in the system
        #forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        #nonbonded_force = forces['NonbondedForce']
        # Return the first occurrence of NonbondedForce particle parameters matching `resname`
        param_list = []
        for residue in topology.residues():
            if residue.name == resname:
                atoms = [atom for atom in residue.atoms()]
                for atm in atoms:
                    [charge, sigma, epsilon] = self.forces_to_update.getParticleParameters(atm.index)
                    parameters = {'charge': charge, 'sigma': sigma, 'epsilon': epsilon}
                    param_list.append(parameters)
                    #if self.debug: print('retrieveResidueParameters: %s : %s' % (resname, str(parameters)))
                return param_list
        raise Exception("resname '%s' not found in topology" % resname)

    def initializeIonParameters(self,ion_name,ion_params=None):
        '''
        Initialize the set of ion non-bonded parameters so that they match the number of atoms of the water model.

        Parameters
        ----------
        water_name : str
            The residue name of the water molecule
        ion_name : str
            The residue name of the ion
        ion_params : dict of str:float
            NonbondedForce parameter dict ('charge', 'sigma', 'epsilon') for ion.
        Returns
        -------
        '''

        # Creating a list of non-bonded parameters that matches the size of the water model.
        num_wat_atoms = len(self.water_parameters)

        # Initialising dummy atoms to having the smallest float that's not zero, due to a bug
        eps = sys.float_info.epsilon
        ion_param_list = num_wat_atoms*[{'charge': eps*units.elementary_charge,'sigma': eps*units.nanometer,'epsilon':eps*units.kilojoule_per_mole}]

        # Making the first element of list of parameter dictionaries the ion. This means that ions will be centered
        # on the water oxygen atoms.
        # If ion parameters are not supplied, use Joung and Cheatham parameters.
        if ion_name == self.cationName:
            if ion_params == None:
                ion_param_list[0] = {'charge': 1.0*units.elementary_charge,'sigma': 0.4477657*units.nanometer,'epsilon':0.148912744*units.kilojoule_per_mole}
            else:
                ion_param_list[0] = ion_params
        elif ion_name == self.anionName:
            if ion_params == None:
                ion_param_list[0] = {'charge': -1.0*units.elementary_charge, 'sigma': 0.2439281*units.nanometer, 'epsilon': 0.3658460312*units.kilojoule_per_mole}
            else:
                ion_parm_list[0] = ion_params
        else:
            raise NameError('Ion name %s does not match known cation or anion names' % ion_name)

        return  ion_param_list

    def identifyResidues(self, topology, residue_names):
        """
        Compile a list of residues that could be converted to/from another chemical species.

        Parameters
        ----------
        topology : simtk.openmm.app.topology
            The topology from which water residues are to be identified.
        residue_names : list of str
            Residues identified as water molecules.

        Returns
        -------
        water_residues : list of simtk.openmm.app.Residue
            Water residues.

        TODO
        ----
        * Can this feature be added to simt.openmm.app.Topology?

        """
        target_residues = list()
        for residue in topology.residues():
            if residue.name in residue_names:
                target_residues.append(residue)

        if self.debug: print('identifyResidues: %d %s molecules identified.' % (len(target_residues),residue_names[0]))
        return target_residues

    def getResidueIdentities(self,topology):
        '''
        Describe the identities of water, cations, and anions with a 3 by N matrix of ones and zeros.
        Parameters
        ----------
        topology : simtk.openmm.app.topology
            The topology from which water and ion residues are to be identified.
        Returns
        -------
        stateMatrix : numpy array
            The identity of residues described by a matrix of ones and zeros. One row for each exchnageable species.
        '''

        stateMatrix = np.zeros((3,topology.residues.im_self.getNumResidues()))

        # The first row indicates which residues are water molecules
        water_indices = [res.index for res in self.water_residues]
        stateMatrix[0,water_indices] = 1
        # The second row indicates cations
        cation_indices = [res.index for res in self.cation_residues]
        stateMatrix[1,cation_indices] = 1
        # The third row indicates anions
        anion_indices = [res.index for res in self.anion_residues]
        stateMatrix[2,anion_indices] = 1
        return stateMatrix

    def initializeStateVector(self):
        '''
        Stores the identity of the mutabable residues in a numpy array for efficient seaching and updating of
        residue identies.

        Returns
        -------
        stateVector : numpy array
            Array of 0s, 1s, and 2s to indicate water, sodium, and chlorine.

        '''
        names = [res.name for res in self.mutable_residues]
        stateVector = np.zeros(len(names))
        for i in range(len(names)):
            if names[i] == self.waterName:  stateVector[i] = 0
            elif names[i] == self.cationName: stateVector[i] = 1
            elif names[i] == self.anionName: stateVector[i] = 2
        return stateVector


    def resetStatistics(self):
        """
        Reset statistics of titration state tracking.

        Todo
        ----

        * Keep track of more statistics regarding history of individual protonation states.
        * Keep track of work values for individual trials to use for calibration.

        """

        self.nattempted = 0
        self.naccepted = 0

        return

    def attempt_identity_swap(self,context):
        '''
        Attempt the exchange of (possibly multiple) chemical species.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        temperature : quantity in units of Kelvin
            The inverse temperature of the simulation

        Notes
        -----
        Code currently written specifically for exchanging two water molecules for Na and Cl, with generalisation to follow.
        Currently without NCMC, to be added.
        '''
        kT=kB * self.temperature
        self.nattempted += 1
        # Getting the current potential energy
        state = context.getState(getEnergy=True)
        pot_energy_old = state.getPotentialEnergy()
        if self.debug: print("old potential energy =", pot_energy_old)

        # Saving and assigning the initial and final state matrix respectively.
        initial_identities = copy.deepcopy(self.mutable_residues)
        final_identities = copy.deepcopy(self.mutable_residues)

        # Whether to delete or add salt by selecting random water molecules to turn into a cation and an anion or vice versa.
        # Here could add the option to create multiple pairs of waters options for configurational biasing
        if (sum(self.stateVector==1)==0) or (sum(self.stateVector==1) < sum(self.stateVector==0)*0.5) and (np.random.random() < 0.5):
            change_indices = np.random.choice(a=np.where(self.stateVector == 0)[0],size=2,replace=False)
            mode_forward = "add salt"
            mode_backward ="remove salt"
        else:
            mode_forward = "remove salt"
            mode_backward = "add salt"
            cation_index = np.random.choice(a=np.where(self.stateVector==1)[0],size=1)
            anion_index = np.random.choice(a=np.where(self.stateVector==2)[0],size=1)
            change_indices = np.array([cation_index,anion_index])

        # TODO: option for NCMC. The section below should go into a separete function
        # Update the names and forces of the selected mutable residues.
        self.updateForces(mode_forward,change_indices)
        # Get the new energy of the state
        # .updateParameters may be very slow, as ALL of the parameters are updated.
        self.forces_to_update.updateParametersInContext(context)
        state = context.getState(getEnergy=True)
        pot_energy_new = state.getPotentialEnergy()

        log_accept = (pot_energy_old - pot_energy_new - self.delta_chem)/kT
        # Accept or reject:
        if (log_accept > 0.0) or (random.random() < math.exp(log_accept)):
            # Accept :D
            self.naccepted += 1
            self.setIdentity(mode_forward,change_indices)
            if self.debug==True: print(mode_forward,"accepted")
        else:
            # Reject :(
            # Revert parameters to their previous value
            self.updateForces(mode_backward,change_indices)
            # As above, optimise this step so only the relavent paramaters are updated.
            self.forces_to_update.updateParametersInContext(context)
            if self.debug==True: print(mode_forward,"rejected")

    def setIdentity(self,mode,exchange_indices):
        '''
        Function to set the names of the mutated residues and update the state vector. Called after a transformation
        of the forcefield parameters has been accepted.

        Parameters
        ----------
        mode : string
            Either 'add salt' or 'remove  salt'
        exchange_indices : numpy array
            Two element vector containing the residue indices that have been changed

        '''

        if mode == "add salt":
            self.mutable_residues[exchange_indices[0]].name = self.cationName
            self.stateVector[exchange_indices[0]] = 1
            self.mutable_residues[exchange_indices[1]].name = self.anionName
            self.stateVector[exchange_indices[1]] = 2
        if mode == "remove salt":
            self.mutable_residues[exchange_indices[0]].name = self.waterName
            self.mutable_residues[exchange_indices[1]].name = self.waterName
            self.stateVector[exchange_indices] = 0

    def updateForces(self,mode,exchange_indices):
        '''
        Update the forcefield parameters and names according the state vector.

        Parameters
        ----------
        mode : string
            Whether the supplied indices will be used to 'add salt' or 'remove salt'
        exchange_indices : numpy array
            Which water residues will be converted to cation and anion, or which cation and anion will be turned
            into 2 water residue.
        Returns
        -------

        '''
        # TODO: add fractional state for NCMC.

        if mode == 'add salt':
            molecule = [atom for atom in self.mutable_residues[exchange_indices[0]].atoms()]
            atm_index = 0
            for atom in molecule:
                target_force = self.cation_parameters[atm_index]
                self.forces_to_update.setParticleParameters(atom.index,charge=target_force["charge"],sigma=target_force["sigma"],epsilon=target_force["epsilon"])
                atm_index += 1
            molecule = [atom for atom in self.mutable_residues[exchange_indices[1]].atoms()]
            atm_index = 0
            for atom in molecule:
                target_force = self.anion_parameters[atm_index]
                self.forces_to_update.setParticleParameters(atom.index,charge=target_force["charge"],sigma=target_force["sigma"],epsilon=target_force["epsilon"])
                atm_index += 1
        if mode == 'remove salt':
            molecule = [atom for atom in self.mutable_residues[exchange_indices[0]].atoms()]
            atm_index = 0
            for atom in molecule:
                target_force = self.water_parameters[atm_index]
                self.forces_to_update.setParticleParameters(atom.index,charge=target_force["charge"],sigma=target_force["sigma"],epsilon=target_force["epsilon"])
                atm_index += 1
            molecule = [atom for atom in self.mutable_residues[exchange_indices[1]].atoms()]
            atm_index = 0
            for atom in molecule:
                target_force = self.water_parameters[atm_index]
                self.forces_to_update.setParticleParameters(atom.index,charge=target_force["charge"],sigma=target_force["sigma"],epsilon=target_force["epsilon"])
                atm_index += 1

    def update(self, context,nattempts=None):
        """
        Perform a number of Monte Carlo update trials for the titration state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        nattempts : integer
            Number of salt insertion and deletion moves to attempt.

        Notes
        -----
        The titration state actually present in the given context is not checked; it is assumed the MonteCarloTitration internal state is correct.

        """
        if nattempts == None: nattempts = self.nattempts_per_update
        # Perform a number of protonation state update trials.
        for attempt in range(nattempts):
            self.attempt_identity_swap(context)
        return

    def getAcceptanceProbability(self):
        """
        Return the fraction of accepted moves

        Returns
        -------

        fraction : float
            the fraction of accepted moves

        """
        return float(self.naccepted) / float(self.nattempted)
