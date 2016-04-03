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

The code is still being written. The algorithm for constant salt dynamics should rougthly follow:
    Load system and free energy cost (difference in chemical potential) for switching process
    Initialize non-bonded parameters for water, cations, and anions
To update the identities of the molecules:
    Select whether to add or remove salt with equal probability (remove salt has zero probability when no ions present)
    If adding ions: identify water molecule indices
    If removing ions: identify ion indices
    Calculate energy prior to switch
    Update non-bonded parameters
    Calculate new energy
    Accept or reject new move based on change in energy and input chemical potential difference.

NCMC step to be added as a seperate option and function (based on constph) within the update step and accept step.


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

    * Add NCMC switching moves to allow this scheme to be efficient in explicit solvent.

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
kB = kB.in_units_of(units.kilocalories_per_mole / units.kelvin)


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

    def __init__(self, system, temperature, chemical_potentials, chemical_names, integrator, pressure=None, nattempts_per_update=None, simultaneous_proposal_probability=0.1, debug=False,
        nsteps_per_trial=0, ncmc_timestep=1.0*units.femtoseconds, maintainChargeNeutrality=True, cationName='Na+', anionName='Cl-', cationparams = None, anionparams = None,implicit=False):
        """
        Initialize a Monte Carlo titration driver for semi-grand ensemble simulation.

        Parameters
        ----------
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature to be simulated.
        chemical_potentials : numpy array
            The chemical potentials of each chemical species that can be exchanged.
        chemical_names : list of strings
            Names of each of the residues whose parameters will be exchanged.
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

        Todo
        ----
        * Create a set of ion parameters that has the same number of dimentions as the water model
        """

        # Set defaults.
        # probability of proposing two simultaneous protonation state changes
        self.simultaneous_proposal_probability = simultaneous_proposal_probability

        # Store parameters.
        self.system = system
        self.temperature = temperature
        self.pressure = pressure
        self.chemical_potentials = chemical_potentials
        self.chemical_names = chemical_names
        self.debug = debug
        self.nsteps_per_trial = nsteps_per_trial

        # Create a Verlet integrator to handle NCMC integration
        self.compound_integrator = openmm.CompoundIntegrator()
        self.compound_integrator.addIntegrator(integrator)
        self.verlet_integrator = VelocityVerletIntegrator(ncmc_timestep)
        self.compound_integrator.addIntegrator(self.verlet_integrator)
        self.compound_integrator.setCurrentIntegrator(0)  # make user integrator active

        # Set constraint tolerance.
        self.verlet_integrator.setConstraintTolerance(integrator.getConstraintTolerance())

        # Check that system has MonteCarloBarostat if pressure is specified.
        if pressure is not None:
            forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
            if 'MonteCarloBarostat' not in forces:
                raise Exception("`pressure` is specified, but `system` object lacks a `MonteCarloBarostat`")

        # Store options for maintaining charge neutrality by converting waters to/from monovalent ions.
        self.maintainChargeNeutrality = maintainChargeNeutrality

        self.water_residues = self.IdentifyResidues(prmtop.topology, residue_names = ('WAT', 'HOH', 'TP4', 'TP5', 'T4E'))
        self.cation_residues = self.IdentifyResidues(prmtop.topology, residue_names = ('Na', 'NA', 'Na+'))
        self.anion_residues = self.IdentifyResidues(prmtop.topology, residue_names = ('Cl', 'CL', 'Cl-'))

        # Describe which residues are waters, cations, or anions with a matrix of ones and zeros.
        self.stateMatrix = self.getResidueIdentities(topology)

        self.cation_parameters = initializeIonParameters(self,water_name,cationName,ion_params=None)
        self.anion_parameters = initializeIonParameters(self,water_name,anionName,ion_params=None)

        # Describing the identities of water and ions with numpy vectors


        if implicit and maintainChargeNeutrality:
            raise ValueError("Implicit solvent and charge neutrality are mutually exclusive.")

        # Keep track of forces and whether they're cached.
        self.precached_forces = False

        # Track simulation state
        self.kin_energies = units.Quantity(list(), units.kilocalorie_per_mole)
        self.pot_energies = units.Quantity(list(), units.kilocalorie_per_mole)
        self.states_per_update = list()

        # Determine 14 Coulomb and Lennard-Jones scaling from system.
        # TODO: Get this from prmtop file?
        self.coulomb14scale = self.get14scaling(system)

        # Store list of exceptions that may need to be modified.
        self.atomExceptions = [list() for index in range(prmtop._prmtop.getNumAtoms())]
        for (atom1, atom2, chargeProd, rMin, epsilon, iScee, iScnb) in prmtop._prmtop.get14Interactions():
            self.atomExceptions[atom1].append(atom2)
            self.atomExceptions[atom2].append(atom1)

        # Store force object pointer.
        # Unlike constph, only altering nonbonded force (like dual topology calculations).
        for force_index in range(system.getNumForces()):
            force = system.getForce(force_index)
            if force.__class__.__name__ == 'NonbondedForce':
            self.forces_to_update = force

        self.setNumAttemptsPerUpdate(nattempts_per_update)

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
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        nonbonded_force = forces['NonbondedForce']

        # Return the first occurrence of NonbondedForce particle parameters matching `resname`
        param_list = []
        for residue in topology.residues():
            if residue.name == resname:
                atoms = [atom for atom in residue.atoms()]
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atoms[0].index)
                parameters = {'charge': charge, 'sigma': sigma, 'epsilon': epsilon}
                param_list.apped(paramters)
                if self.debug: print('retrieveResidueParameters: %s : %s' % (resname, str(parameters)))
                return param_list

        raise Exception("resname '%s' not found in topology" % resname)

    def initializeIonParameters(self,water_name,ion_name,ion_params=None):
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
        num_wat_atoms = len(retrieveResidueParameters(topology,system,water_name))
        ion_param_list = [{'charge': 0.0, 'sigma': 0.0, 'epsilon': 0.0}]*num_wat_atoms

        # Making the first element of list of parameter dictionaries the ion. This means that ions will be centered
        # on the water oxygen atoms.
        # If ion parameters are not supplied, use Joung and Cheatham parameters.
        if ion_name == cationName:
            if ion_params == None:
                ion_param_list[0] = {'charge': 1.0, 'sigma': 0.4477657, 'epsilon': 0.148912744}
            else:
                ion_param_list[0] = ion_params
        elif ion_name == anionName:
            if ion_params == None:
                ion_param_list[0] = {'charge': -1.0, 'sigma': 0.2439281, 'epsilon': 0.3658460312}
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
        self.work_history = list()

        return

    def _update_forces(self, titration_group_index, final_titration_state_index, initial_titration_state_index=None, fractional_titration_state=1.0, context=None):
        """
        Update the force parameters to a new titration state by reading them from the cache

        Parameters
        ----------
        titration_group_index : int
            Index of the group that is changing state
        titration_state_index : int
            Index of the state of the chosen residue
        initial_titration_state_index : int, optional, default=None
            If blending two titration states, the initial titration state to blend.
            If `None`, set to `titration_state_index`
        fractional_titration_state : float, optional, default=1.0
            Fraction of `titration_state_index` to be blended with `initial_titration_state_index`.
            If 0.0, `initial_titration_state_index` is fully active; if 1.0, `titration_state_index` is fully active.
        context : simtk.openmm.Context, optional, default=None
            If provided, will update forces state in the specified Context

        Notes
        -----
        * Every titration state has a list called forces, which stores parameters for all forces that need updating.
        * Inside each list entry is a dictionary that always contains an entry called `atoms`, with single atom parameters by name.
        * NonbondedForces also have an entry called `exceptions`, containing exception parameters.

        """
        # `initial_titration_state_index` should have no effect if not specified, so set it identical to `final_titration_state_index` in that case
        if initial_titration_state_index is None:
            initial_titration_state_index = final_titration_state_index

        # Retrieve cached force parameters fro this titration state.
        cache_initial = self.titrationGroups[titration_group_index]['titration_states'][initial_titration_state_index]['forces']
        cache_final = self.titrationGroups[titration_group_index]['titration_states'][final_titration_state_index]['forces']

        # Modify charges and exceptions.
        for force_index, force in enumerate(self.forces_to_update):
            # Get name of force class.
            force_classname = force.__class__.__name__
            # Get atom indices and charges.

            # Update forces using appropriately blended parameters
            for (atom_initial, atom_final) in zip(cache_initial[force_index]['atoms'], cache_final[force_index]['atoms']):
                atom = {key: atom_initial[key] for key in ['atom_index']}
                if force_classname == 'NonbondedForce':
                    for parameter_name in ['charge', 'sigma', 'epsilon']:
                        atom[parameter_name] = (1.0 - fractional_titration_state) * atom_initial[parameter_name] + \
                            fractional_titration_state * atom_final[parameter_name]
                    force.setParticleParameters(atom['atom_index'], atom['charge'], atom['sigma'], atom['epsilon'])
                elif force_classname == 'GBSAOBCForce':
                    for parameter_name in ['charge', 'radius', 'scaleFactor']:
                        atom[parameter_name] = (1.0 - fractional_titration_state) * atom_initial[parameter_name] + \
                            fractional_titration_state * atom_final[parameter_name]
                    force.setParticleParameters(atom['atom_index'], atom['charge'], atom['radius'], atom['scaleFactor'])
                else:
                    raise Exception("Don't know how to update force type '%s'" % force_classname)

            # Update exceptions
            # TODO: Handle Custom forces.
            if force_classname == 'NonbondedForce':
                for (exc_initial, exc_final) in zip(cache_initial[force_index]['exceptions'], cache_final[force_index]['exceptions']):
                    exc = {key: exc_initial[key] for key in ['exception_index', 'particle1', 'particle2']}
                    for parameter_name in ['chargeProd', 'sigma', 'epsilon']:
                        exc[parameter_name] = (1.0 - fractional_titration_state) * exc_initial[parameter_name] + \
                            fractional_titration_state * exc_final[parameter_name]
                    force.setExceptionParameters(
                        exc['exception_index'], exc['particle1'], exc['particle2'], exc['chargeProd'], exc['sigma'], exc['epsilon'])

            # Update parameters in Context, if specified.
            if context and hasattr(force, 'updateParametersInContext'):
                force.updateParametersInContext(context)

    def _cache_force(self, titration_group_index, titration_state_index):
        """
        Cache the force parameters for a single titration state.

        Parameters
        ----------
        titration_group_index : int
            Index of the group
        titration_state_index : int
            Index of the titration state of the group

        Notes
        -----

        Call this function to set up the 'forces' information for a single titration state.
        Every titration state has a list called forces, which stores parameters for all forces that need updating.
        Inside each list entry is a dictionary that always contains an entry called `atoms`, with single atom parameters by name.
        NonbondedForces also have an entry called `exceptions`, containing exception parameters.

        Returns
        -------

        """

        titration_group = self.titrationGroups[titration_group_index]
        titration_state = self.titrationGroups[titration_group_index]['titration_states'][titration_state_index]

        # Store the parameters per individual force
        f_params = list()
        for force_index, force in enumerate(self.forces_to_update):
            # Store parameters for this particular force
            f_params.append(dict(atoms=list()))
            # Get name of force class.
            force_classname = force.__class__.__name__
            # Get atom indices and charges.
            charges = titration_state['charges']
            atom_indices = titration_group['atom_indices']

            charge_by_atom_index = dict(zip(atom_indices, charges))

            # Update charges.
            # TODO: Handle Custom forces, looking for "charge" and "chargeProd".
            for atom_index in atom_indices:
                if force_classname == 'NonbondedForce':
                    f_params[force_index]['atoms'].append(
                        {key: value for (key, value) in zip(['charge', 'sigma', 'epsilon'], map(strip_in_unit_system, force.getParticleParameters(atom_index)))})
                elif force_classname == 'GBSAOBCForce':
                    f_params[force_index]['atoms'].append(
                        {key: value for (key, value) in zip(['charge', 'radius', 'scaleFactor'], map(strip_in_unit_system, force.getParticleParameters(atom_index)))})
                else:
                    raise Exception("Don't know how to update force type '%s'" % force_classname)
                f_params[force_index]['atoms'][-1]['charge'] = charge_by_atom_index[atom_index]
                f_params[force_index]['atoms'][-1]['atom_index'] = atom_index

            # Update exceptions
            # TODO: Handle Custom forces.
            if force_classname == 'NonbondedForce':
                f_params[force_index]['exceptions'] = list()
                for e_ix, exception_index in enumerate(titration_group['exception_indices']):
                    [particle1, particle2, chargeProd, sigma, epsilon] = map(
                        strip_in_unit_system, force.getExceptionParameters(exception_index))

                    # Deal with exceptions between atoms outside of titratable residue
                    try:
                        charge_1 = charge_by_atom_index[particle1]
                    except KeyError:
                        charge_1 = strip_in_unit_system(force.getParticleParameters(particle1)[0])
                    try:
                        charge_2 = charge_by_atom_index[particle2]
                    except KeyError:
                        charge_2 = strip_in_unit_system(force.getParticleParameters(particle2)[0])

                    chargeProd = self.coulomb14scale * charge_1 * charge_2

                    # chargeprod and sigma cannot be identically zero or else we risk the error:
                    # Exception: updateParametersInContext: The number of non-excluded exceptions has changed
                    # TODO: Once OpenMM interface permits this, omit this code.
                    if (2 * chargeProd == chargeProd):
                        chargeProd = sys.float_info.epsilon
                    if (2 * epsilon == epsilon):
                        epsilon = sys.float_info.epsilon

                    # store specific local variables in dict by name
                    exc_dict = dict()
                    for i in ('exception_index', 'particle1', 'particle2', 'chargeProd', 'sigma', 'epsilon'):
                        exc_dict[i] = locals()[i]
                    f_params[force_index]['exceptions'].append(exc_dict)

        self.titrationGroups[titration_group_index]['titration_states'][titration_state_index]['forces'] = f_params

    def attempt_identity_swap(self,context):
        '''
        Attempt the exchange of (possibly multiple) chemical species.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update

        Notes
        -----
        Code currently written specifically for exchanging two water molecules for Na and Cl, with generalisation to follow.
        Currently without NCMC, to be added.
        '''

        # Getting the current potential energy
        state = context.getState(getEnergy=True)
        pot_energy_old = state.getPotentialEnergy()
        if self.debug: print "old potential energy = ", pot_energy_old

        # Saving and assigning the initial and final state matrix respectively.
        initial_identities = copy.deepcopy(self.stateMatrix)
        final_identities = copy.deepcopy(self.stateMatrix)

        # Whether to delete or add salt by selecting random water molecules to turn into a cation and an anion or vice versa.
        # Here could add the option to create multiple pairs of waters options for configurational biasing
        if (len(self.cation_residues)==0) or ((len(self.cation_residues) < len(self.water_residues)*0.5) and (np.random.random() < 0.5)):
            indices = np.random.choice(a=np.where(self.stateMatrix[0] == 1)[0],size=2,replace=False)
            final_identities[0][indices] = 0
            final_identities[1][indices[0]] = 1
            final_identities[2][indices[1]] = 1
        else:
            cation_index = np.random.choice(a=np.where(self.stateMatrix[1]==1)[0],size=1)
            anion_index = np.random.choice(a=np.where(self.stateMatrix[2]==1)[0],size=1)
            final_identities[0][cation_index] = 1
            final_identities[0][anion_index] = 1
            final_identities[1][cation_index] = 0
            final_identities[2][anion_index] = 0

        # TODO: set forcefield parameters to new values, calculate new energy, accept or reject.
        # TODO: option for NCMC.

    def attempt_protonation_state_change(self, context):
        """
        Attempt a single Monte Carlo protonation state change.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update

        Notes
        -----
        The titration state actually present in the given context is not checked; it is assumed the MonteCarloTitration internal state is correct.

        """

        # Activate velocity Verlet integrator
        self.compound_integrator.setCurrentIntegrator(1)

        # If using NCMC, store initial positions.
        if self.nsteps_per_trial > 0:
            initial_positions = context.getState(getPositions=True).getPositions(asNumpy=True)

        # Compute initial probability of this protonation state.
        log_P_initial, pot1, kin1 = self._compute_log_probability(context)

        if self.debug:
            state = context.getState(getEnergy=True)
            initial_potential = state.getPotentialEnergy()
            print("   initial %s   %12.3f kcal/mol" % (str(self.getTitrationStates()), initial_potential / units.kilocalories_per_mole))

        # Store current titration state indices.
        initial_titration_states = copy.deepcopy(self.titrationStates)  # deep copy

        # Select new titration states.
        final_titration_states = copy.deepcopy(self.titrationStates)  # deep copy
        # Choose how many titratable groups to simultaneously attempt to update.
        # TODO: Refine how we select residues and groups of residues to titrate to increase efficiency.
        ndraw = 1
        if (self.getNumTitratableGroups() > 1) and (random.random() < self.simultaneous_proposal_probability):
            ndraw = 2
        # Select which titratible residues to update.
        titration_group_indices = random.sample(range(self.getNumTitratableGroups()), ndraw)
        # Select new titration states.
        for titration_group_index in titration_group_indices:
            # Choose a titration state with uniform probability (even if it is the same as the current state).
            titration_state_index = random.choice(range(self.getNumTitrationStates(titration_group_index)))
            final_titration_states[titration_group_index] = titration_state_index
        # TODO: Always accept self transitions, or avoid them altogether.

        if self.maintainChargeNeutrality:
            # TODO: Designate waters/ions to switch to maintain charge neutrality
            raise Exception('maintainChargeNeutrality feature not yet supported')

        # Compute work for switching to new protonation states.
        if self.nsteps_per_trial == 0:
            # Use instantaneous switching.
            for titration_group_index in titration_group_indices:
                self.setTitrationState(titration_group_index, final_titration_states[titration_group_index], context)
        else:
            # Run NCMC integration.
            for step in range(self.nsteps_per_trial):
                # Take a Verlet integrator step.
                self.verlet_integrator.step(1)
                # Update the titration state.
                titration_lambda = float(step + 1) / float(self.nsteps_per_trial)
                # TODO: Using a VerletIntegrator together with half-kicks on either side would save one force evaluation per iteration,
                # since parameter update would occur in the middle of a velocity Verlet step.
                # TODO: This could be optimized by only calling
                # context.updateParametersInContext once rather than after every titration
                # state update.
                for titration_group_index in titration_group_indices:
                    self._update_forces(titration_group_index, final_titration_states[titration_group_index], initial_titration_state_index=initial_titration_states[
                                        titration_group_index], fractional_titration_state=titration_lambda, context=context)
                    # TODO: Optimize where integrator.step() is called
                    self.verlet_integrator.step(1)

        # Compute final probability of this protonation state.
        log_P_final, pot2, kin2 = self._compute_log_probability(context)

        # Compute work and store work history.
        work = - (log_P_final - log_P_initial)
        self.work_history.append((initial_titration_states, final_titration_states, work))

        # Accept or reject with Metropolis criteria.
        log_P_accept = -work
        if self.debug:
            print("LOGP" + str(log_P_accept))
        if self.debug:
            print("   proposed log probability change: %f -> %f | work %f\n" % (log_P_initial, log_P_final, work))
        self.nattempted += 1
        if (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept)):
            # Accept.
            self.naccepted += 1
            self.pot_energies.append(pot2)
            self.kin_energies.append(kin2)
            # Update titration states.
            for titration_group_index in titration_group_indices:
                self.setTitrationState(titration_group_index, final_titration_states[titration_group_index], context)
            # If using NCMC, flip velocities to satisfy super-detailed balance.
            if self.nsteps_per_trial > 0:
                context.setVelocities(-context.getState(getVelocities=True).getVelocities(asNumpy=True))
        else:
            # Reject.
            self.pot_energies.append(pot1)
            self.kin_energies.append(kin1)
            # Restore titration states.
            for titration_group_index in titration_group_indices:
                self.setTitrationState(titration_group_index, initial_titration_states[titration_group_index], context)
            # If using NCMC, restore coordinates and flip velocities.
            if self.nsteps_per_trial > 0:
                context.setPositions(initial_positions)

        # Restore user integrator
        self.compound_integrator.setCurrentIntegrator(0)

        return

    def update(self, context):
        """
        Perform a number of Monte Carlo update trials for the titration state.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update

        Notes
        -----
        The titration state actually present in the given context is not checked; it is assumed the MonteCarloTitration internal state is correct.

        """

        # Perform a number of protonation state update trials.
        for attempt in range(self.nattempts_per_update):
            self.attempt_protonation_state_change(context)

        self.states_per_update.append(self.getTitrationStates())

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

    def _compute_log_probability(self, context):
        """
        Compute log probability of current configuration and protonation state.

        Parameters
        ----------

        context : simtk.openmm.Context
            the context

        Returns
        -------
        log_P : float
            log probability of the current context
        pot_energy : float
            potential energy of the current context
        kin_energy : float
            kinetic energy of the current context

        TODO
        ----
        * Generalize this to use ThermodynamicState concept of reduced potential (from repex)


        """

        temperature = self.temperature
        pressure = self.pressure
        kT = kB * temperature  # thermal energy
        beta = 1.0 / kT  # inverse temperature

        # Add energetic contribution to log probability.
        state = context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy()
        kin_energy = state.getKineticEnergy()
        total_energy = pot_energy + kin_energy
        log_P = - beta * total_energy

        if pressure is not None:
            # Add pressure contribution for periodic simulations.
            volume = context.getState().getPeriodicBoxVolume()
            if self.debug:
                print('beta = %s, pressure = %s, volume = %s, multiple = %s' % (str(beta), str(pressure), str(volume), str(-beta*pressure*volume*units.AVOGADRO_CONSTANT_NA)))
            log_P += -beta * pressure * volume * units.AVOGADRO_CONSTANT_NA

        # Include reference energy and pH-dependent contributions.
        for titration_group_index, (titration_group, titration_state_index) in enumerate(zip(self.titrationGroups, self.titrationStates)):
            titration_state = titration_group['titration_states'][titration_state_index]
            relative_energy = titration_state['relative_energy']
            if self.debug:
                print("beta * relative_energy: %.2f",  +beta * relative_energy)
            log_P += - self._get_proton_chemical_potential(titration_group_index, titration_state_index) + beta * relative_energy

        # Return the log probability.
        return log_P, pot_energy, kin_energy

    def getNumAttemptsPerUpdate(self):
        """
        Get the number of Monte Carlo titration state change attempts per call to update().

        Returns
        -------

        nattempts_per_iteration : int
            the number of attempts to be made per iteration

        """
        return self.nattempts_per_update

    def setNumAttemptsPerUpdate(self, nattempts=None):
        """
        Set the number of Monte Carlo titration state change attempts per call to update().

        Parameters
        ----------

        nattempts : int
            the number to attempts to make per iteration;
            if None, this value is computed automatically based on the number of titratable groups (default None)

        """
        self.nattempts_per_update = nattempts
        if nattempts is None:
            # TODO: Perform enough titration attempts to ensure thorough mixing without taking too long per update.
            # TODO: Cache already-visited states to avoid recomputing?
            self.nattempts_per_update = self.getNumTitratableGroups()

    def _get_reduced_potentials(self, context, beta, group_index=0):
        """Retrieve the reduced potentials for all states of the system given a context.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        beta : simtk.unit.Quantity compatible with simtk.unit.mole/simtk.unit.kcal
            inverse temperature
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        """
        # beta * U(x)_j

        ub_j = np.empty(len(self.titrationGroups[group_index]['titration_states']))
        for j in range(ub_j.size):
            ub_j[j] = self._reduced_potential(context, beta, j, group_index)

        # Reset to current state
        return ub_j

    def _reduced_potential(self, context, beta, state_index, group_index=0):
        """Retrieve the reduced potential for a given state (specified by index) in the given context.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        beta : simtk.unit.Quantity compatible with simtk.unit.mole/simtk.unit.kcal
            inverse temperature
        state_index : int
            Index of the state for which the reduced potential needs to be calculated.
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        """
        potential_energy = self._get_potential_energy(context, state_index)
        return self._get_proton_chemical_potential(group_index, state_index) + beta * potential_energy

    def _get_potential_energy(self, context, state_index, group_index=0):
        """ Retrieve the potential energy for a given state (specified by index) in the given context.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        state_index : int
            Index of the state for which the reduced potential needs to be calculated.
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        """
        current_state = self.getTitrationState(group_index)
        self.setTitrationState(group_index, state_index, context)
        temp_state = context.getState(getEnergy=True)
        potential_energy = temp_state.getPotentialEnergy()
        self.setTitrationState(group_index, current_state, context)
        return potential_energy