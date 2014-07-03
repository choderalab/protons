#!/usr/local/bin/env python

#======================================================================================
# MODULE DOCSTRING
#======================================================================================

"""
Constant pH dynamics test.

DESCRIPTION

This module tests the constant pH functionality in OpenMM


NOTES
p
This is still in development.

REFERENCES

[1] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born implicit solvent. J Comput Chem 25:2038, 2004.
http://dx.doi.org/10.1002/jcc.20139

[2] Stern HA. Molecular simulation with variable protonation states at constant pH. JCP 126:164112, 2007.
http://link.aip.org/link/doi/10.1063/1.2731781

[3] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. PNAS 108:E1009, 2011.
http://dx.doi.org/10.1073/pnas.1106094108

EXAMPLES

TODO

* Add NCMC switching moves to allow this scheme to be efficient in explicit solvent.
* Add alternative proposal types, including schemes that avoid proposing self-transitions (or always accept them):
  - Parallel Monte Carlo schemes: Compute N proposals at once, and pick using Gibbs sampling or Metropolized Gibbs?
* Allow specification of probabilities for selecting N residues to change protonation state at once.
* Add calibrate() method to automagically adjust relative energies of protonation states of titratable groups in molecule.
* Add automatic tuning of switching times for optimal acceptance.
* Extend to handle systems set up via OpenMM app Forcefield class.

COPYRIGHT AND LICENSE

@author John D. Chodera <jchodera@gmail.com>

"""

#=============================================================================================
# GLOBAL IMPORTS
#=============================================================================================

import re
import os
import sys
import math
import random
import copy
import time

import simtk
import simtk.openmm as openmm
import simtk.unit as units
import numpy


from db import * 


#=============================================================================================
# MODULE CONSTANTS
#=============================================================================================

kB = units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA

#=============================================================================================
# Monte Carlo titration.
#=============================================================================================

class MonteCarloTitration(object):
    """
    Monte Carlo titration driver for constnat-pH dynamics.

    This move type implements the constant-pH dynamics of Mongan and Case [1].

    REFERENCES

    [1] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born implicit solvent. J Comput Chem 25:2038, 2004.
    http://dx.doi.org/10.1002/jcc.20139
    
    [2] Stern HA. Molecular simulation with variable protonation states at constant pH. JCP 126:164112, 2007.
    http://link.aip.org/link/doi/10.1063/1.2731781
    
    [3] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. PNAS 108:E1009, 2011.
    http://dx.doi.org/10.1073/pnas.1106094108

    TODO

    * Add methods to keep track of history of protonation states.

    """

    #=============================================================================================
    # Initialization.
    #=============================================================================================

    def __init__(self, system, temperature, pH, prmtop, cpin_filename, nattempts_per_update=None, simultaneous_proposal_probability=0.1, debug=False):
        """
        Initialize a Monte Carlo titration driver for constant pH simulation.

        ARGUMENTS

        system (simtk.openmm.System) - system to be titrated, containing all possible protonation sites
        temperature (simtk.unit.Quantity compatible with simtk.unit.kelvin) - temperature to be simulated
        pH (float) - the pH to be simulated 
        prmtop (Prmtop) - parsed AMBER 'prmtop' file (necessary to provide information on exclusions)
        cpin_filename (string) - AMBER 'cpin' file defining protonation charge states and energies

        OPTIONAL ARGUMENTS
        
        nattempts_per_update (int) - number of protonation state change attempts per update call; 
                                   if None, set automatically based on number of titratible groups (default: None)
        simultaneous_proposal_probability (float) - probability of simultaneously proposing two updates
        debug (boolean) - turn debug information on/off

        TODO

        * Allow constant-pH dynamics to be initialized in other ways than using the AMBER cpin file (e.g. from OpenMM app; automatically).
        * Generalize simultaneous_proposal_probability to allow probability of single, double, triple, etc. proposals to be specified?

        """

        # Set defaults.
        self.simultaneous_proposal_probability = simultaneous_proposal_probability # probability of proposing two simultaneous protonation state changes

        # Store parameters.
        self.system = system
        self.temperature = temperature
        self.pH = pH
        self.cpin_filename = cpin_filename
        self.debug = debug

        # Initialize titration group records.
        self.titrationGroups = list()
        self.titrationStates = list()

        # Determine 14 Coulomb and Lennard-Jones scaling from system.
        # TODO: Get this from prmtop file?
        self.coulomb14scale = self.get14scaling(system)

        # Store list of exceptions that may need to be modified.        
        self.atomExceptions = [ list() for index in range(prmtop._prmtop.getNumAtoms()) ]
        for (atom1, atom2, chargeProd, rMin, epsilon, iScee, iScnb) in prmtop._prmtop.get14Interactions():
            self.atomExceptions[atom1].append(atom2)
            self.atomExceptions[atom2].append(atom1)                                        

        # Store force object pointers.
        # TODO: Add Custom forces.
        force_classes_to_update = ['NonbondedForce', 'GBSAOBCForce']
        self.forces_to_update = list()
        for force_index in range(self.system.getNumForces()):
            force = self.system.getForce(force_index)
            if force.__class__.__name__ in force_classes_to_update:
                self.forces_to_update.append(force)            

        if cpin_filename:
            # Load AMBER cpin file defining protonation states.
            namelist = self._parse_fortran_namelist(cpin_filename, 'CNSTPH')

            # Make sure RESSTATE is a list.
            if type(namelist['RESSTATE'])==int:
                namelist['RESSTATE'] = [namelist['RESSTATE']]    

            # Extract number of titratable groups.
            self.ngroups = len(namelist['RESSTATE'])

            # Define titratable groups and titration states.
            for group_index in range(self.ngroups):
                # Extract information about this titration group.
                first_atom = namelist['STATEINF(%d)%%FIRST_ATOM' % group_index] - 1
                first_charge = namelist['STATEINF(%d)%%FIRST_CHARGE' % group_index]
                first_state = namelist['STATEINF(%d)%%FIRST_STATE' % group_index]
                num_atoms = namelist['STATEINF(%d)%%NUM_ATOMS' % group_index]
                num_states = namelist['STATEINF(%d)%%NUM_STATES' % group_index]
                
                # Define titratable group.
                atom_indices = range(first_atom, first_atom+num_atoms)
                self.addTitratableGroup(atom_indices)
                
                # Define titration states.
                for titration_state in range(num_states):
                    # Extract charges for this titration state.
                    charges = namelist['CHRGDAT'][(first_charge+num_atoms*titration_state):(first_charge+num_atoms*(titration_state+1))]
                    charges = units.Quantity(charges, units.elementary_charge)
                    # Extract relative energy for this titration state.
                    relative_energy = namelist['STATENE'][first_state+titration_state] * units.kilocalories_per_mole
                    # Don't use pKref for AMBER cpin files---reference pKa contribution is already included in relative_energy.
                    pKref = 0.0
                    # Get proton count.
                    proton_count = namelist['PROTCNT'][first_state+titration_state]
                    # Create titration state.
                    self.addTitrationState(group_index, pKref, relative_energy, charges, proton_count)

                # Set default state for this group.
                self.setTitrationState(group_index, namelist['RESSTATE'][group_index])

        self.setNumAttemptsPerUpdate(nattempts_per_update)

        # Reset statistics.
        self.resetStatistics()
                
        return

    def get14scaling(self, system):
        """
		Determine Coulomb 14 scaling.
        
        ARGUMENTS

        system (simtk.openmm.System) - the system to examine

        RETURNS

        coulomb14scale (float) - degree to which 1,4 coulomb interactions are scaled

        """
        # Look for a NonbondedForce.
        forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
        force = forces['NonbondedForce']         
        # Determine coulomb14scale from first exception with nonzero chargeprod.
        for index in range(force.getNumExceptions()):
            [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
            [charge1, sigma1, epsilon1] = force.getParticleParameters(particle1)
            [charge2, sigma2, epsilon2] = force.getParticleParameters(particle2)
            if (abs(charge1/units.elementary_charge) > 0) and (abs(charge2/units.elementary_charge)>0):
                coulomb14scale = chargeProd / (charge1*charge2)
                return coulomb14scale
        
        return None

    def get14exceptions(self, system, particle_indices):
        """
        Return a list of all 1,4 exceptions involving the specified particles that are not exclusions.
        
        ARGUMENTS
        
        system (simtk.openmm.System) - the system to examine
        particle_indices (list of int) - only exceptions involving at least one of these particles are returned
        
        RETURNS

        exception_indices (list) - list of exception indices for NonbondedForce

        TODO

        * Deal with the case where there may be multiple NonbondedForce objects.
        * Deal with electrostatics implmented as CustomForce objects (by CustomNonbondedForce + CustomBondForce)
        
        """
        # Locate NonbondedForce object.
        forces = { system.getForce(index).__class__.__name__ : system.getForce(index) for index in range(system.getNumForces()) }
        force = forces['NonbondedForce']         
        # Build a list of exception indices involving any of the specified particles.
        exception_indices = list()
        for exception_index in range(force.getNumExceptions()):
            [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
            if (particle1 in particle_indices) or (particle2 in particle_indices):
                if (particle2 in self.atomExceptions[particle1]) or (particle1 in self.atomExceptions[particle2]):
                    exception_indices.append(exception_index)
                    # BEGIN UGLY HACK
                    # chargeprod and sigma cannot be identically zero or else we risk the error:
                    # Exception: updateParametersInContext: The number of non-excluded exceptions has changed
                    # TODO: Once OpenMM interface permits this, omit this code.
                    [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
                    if (2*chargeProd == chargeProd): chargeProd = sys.float_info.epsilon                        
                    if (2*epsilon == epsilon): epsilon = sys.float_info.epsilon
                    force.setExceptionParameters(exception_index, particle1, particle2, chargeProd, sigma, epsilon)
                    # END UGLY HACK

        return exception_indices

    def resetStatistics(self):
        """
        Reset statistics of titration state tracking.

        TODO

        * Keep track of more statistics regarding history of individual protonation states.
        * Keep track of work values for individual trials to use for calibration.
        
        """
                
        self.nattempted = 0
        self.naccepted = 0 
        self.work_history = list()
        
        return

    def _parse_fortran_namelist(self, filename, namelist_name):
        """
        Parse a fortran namelist generated by AMBER 11 constant-pH python scripts.

        ARGUMENTS

        filename (string) - the name of the file containing the fortran namelist
        namelist_name (string) - name of the namelist section to parse

        RETURNS

        namelist (dict) - namelist[key] indexes read values, converted to Python types

        NOTES

        This code is not fully general for all Fortran namelists---it is specialized to the cpin files.

        """
        # Read file contents.
        infile = open(filename, 'r')
        lines = infile.readlines()
        infile.close()
        
        # Concatenate all text.
        contents = ''
        for line in lines:
            contents += line.strip()

        # Extract section corresponding to keyword.
        key = '&' + namelist_name
        terminator = '/'
        match = re.match(key + '(.*)' + terminator, contents)
        contents = match.groups(1)[0]

        # Parse contents.
        # These regexp match strings come from fortran-namelist from Stephane Chamberland (stephane.chamberland@ec.gc.ca) [LGPL].
        valueInt  = re.compile(r'[+-]?[0-9]+')
        valueReal = re.compile(r'[+-]?([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)')
        valueString = re.compile(r'^[\'\"](.*)[\'\"]$')

        # Parse contents.
        namelist = dict()
        while len(contents) > 0:
            # Peel off variable name.
            match = re.match(r'^([^,]+)=(.+)$', contents)
            if not match: break
            name = match.group(1).strip()
            contents = match.group(2).strip()

            # Peel off value, which extends to either next variable name or end of section.
            match = re.match(r'^([^=]+),([^,]+)=(.+)$', contents)
            if match:
                value = match.group(1).strip()
                contents = match.group(2) + '=' + match.group(3)
            else:
                value = contents
                contents = ''

            # Split value on commas.
            elements = value.split(',')            
            value = list()
            for element in elements:
                if valueReal.match(element):
                    element = float(element)
                elif valueInt.match(element):
                    element = int(element)
                elif valueString.match(element):
                    element = element[1:-1] 
                if element != '':
                    value.append(element)
            if len(value) == 1:
                value = value[0]

            namelist[name] = value

        return namelist


    #=============================================================================================
    # This functionality takes the place of a C++ MonteCarloTitration Force object.
    #=============================================================================================

    def getNumTitratableGroups(self):
        """
        Return the number of titratable groups.

        RETURNS

        ngroups (int) - the number of titratable groups that have been defined

        """

        return len(self.titrationGroups)

    def addTitratableGroup(self, atom_indices):
        """
        Define a new titratable group.
        
        ARGUMENTS

        atom_indices (list of int) - the atom indices defining the titration group

        NOTE

        No two titration groups may share atoms.

        """
        # Check to make sure the requested group does not share atoms with any existing titration group.
        for group in self.titrationGroups:
            if set(group['atom_indices']).intersection(atom_indices):
                raise Exception("Titration groups cannot share atoms.  The requested atoms of new titration group (%s) share atoms with another group (%s)." % (str(atom_indices), str(group['atom_indices'])))
        
        # Define the new group.
        group = dict()
        group['atom_indices'] = list(atom_indices) # deep copy
        group['titration_states'] = list()
        group_index = len(self.titrationGroups) + 1
        group['index'] = group_index
        group['nstates'] = 0
        group['exception_indices'] = self.get14exceptions(system, atom_indices) # NonbondedForce exceptions associated with this titration state

        self.titrationGroups.append(group)

        # Note that we haven't yet defined any titration states, so current state is set to None.
        self.titrationStates.append(None)

        return group_index

    def getNumTitrationStates(self, titration_group_index):
        """
        Return the number of titration states defined for the specified titratable group.

        ARGUMENTS

        titration_group_index (int) - the titration group to be queried

        RETURNS

        nstates (int) - the number of titration states defined for the specified titration group
        
        """
        if titration_group_index not in range(self.getNumTitratableGroups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." % (titration_group_index, self.getNumTitratableGroups()))        

        return len(self.titrationGroups[titration_group_index]['titration_states'])

    def addTitrationState(self, titration_group_index, pKref, relative_energy, charges, proton_count):
        """
        Add a titration state to a titratable group.

        ARGUMENTS

        titration_group_index (int) - the index of the titration group to which a new titration state is to be added
        pKref (float) - the pKa for the reference compound used in calibration
        relative_energy (simtk.unit.Quantity with units compatible with simtk.unit.kilojoules_per_mole) - the relative energy of this protonation state
        charges (list or numpy array of simtk.unit.Quantity with units compatible with simtk.unit.elementary_charge) - the atomic charges for this titration state
        proton_count (int) - number of protons in this titration state

        NOTE

        The relative free energy of a titration state is computed as
        
        relative_energy + kT * proton_count * ln (10^(pH - pKa))
        = relative_energy + kT * proton_count * (pH - pKa) * ln 10
        
        The number of charges specified must match the number (and order) of atoms in the defined titration group.

        """

        # Check input arguments.
        if titration_group_index not in range(self.getNumTitratableGroups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." % (titration_group_index, self.getNumTitratableGroups()))        
        if len(charges) != len(self.titrationGroups[titration_group_index]['atom_indices']):
            raise Exception('The number of charges must match the number (and order) of atoms in the defined titration group.')
        
        state = dict()
        state['pKref'] = pKref
        state['relative_energy'] = relative_energy
        state['charges'] = copy.deepcopy(charges) 
        state['proton_count'] = proton_count
        self.titrationGroups[titration_group_index]['titration_states'].append(state)

        # Increment count of titration states and set current state to last defined state.
        self.titrationStates[titration_group_index] = self.titrationGroups[titration_group_index]['nstates']
        self.titrationGroups[titration_group_index]['nstates'] += 1

        return

    def getTitrationState(self, titration_group_index):        
        """
        Return the current titration state for the specified titratable group.
        
        ARGUMENTS

        titration_group_index (int) - the titration group to be queried
        
        RETURNS

        state (int) - the titration state for the specified titration group

        """
        if titration_group_index not in range(self.getNumTitratableGroups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." % (titration_group_index, self.getNumTitratableGroups()))        

        return self.titrationStates[titration_group_index]

    def getTitrationStates(self):        
        """
        Return the current titration states for all titratable groups.        
        
        RETURNS

        states (list of int) - the titration states for all titratable groups

        """
        return list(self.titrationStates) # deep copy

    def getTitrationStateTotalCharge(self, titration_group_index):
        """
        Return the total charge for the specified titration state.
        
        ARGUMENTS

        titration_group_index (int) - the titration group to be queried

        RETURNS

        charge (simtk.openmm.Quantity compatible with simtk.unit.elementary_charge) - total charge for the specified titration state

        """
        if titration_group_index not in range(self.getNumTitratableGroups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." % (titration_group_index, self.getNumTitratableGroups()))        
        if titration_state_index not in range(self.getNumTitrationStates(titratable_group_index)):
            raise Exception("Invalid titration state requested.  Requested %d, valid states are in range(%d)." % (titration_state_index, self.getNumTitrationStates(titration_group_index)))

        charges = self.titrationGroups[titration_group_index]['titration_states'][titration_state_index]['charges'][:]
        return simtk.unit.Quantity((charges / charges.unit).sum(), charges.unit)


    def setTitrationState(self, titration_group_index, titration_state_index, context=None, debug=False):
        """
        Change the titration state of the designated group for the provided state.

        ARGUMENTS
        
        titration_group_index (int) - the index of the titratable group whose titration state should be updated
        titration_state_index (int) - the titration state to set as active
        
        OPTIONAL ARGUMENTS

        context (simtk.openmm.Context) - if provided, will update protonation state in the specified Context (default: None)
        debug (boolean) - if True, will print debug information

        """

        # Check parameters for validity.
        if titration_group_index not in range(self.getNumTitratableGroups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." % (titration_group_index, self.getNumTitratableGroups()))        
        if titration_state_index not in range(self.getNumTitrationStates(titration_group_index)):
            raise Exception("Invalid titration state requested.  Requested %d, valid states are in range(%d)." % (titration_state_index, self.getNumTitrationStates(titration_group_index)))


        # Get titration group and state.
        titration_group = self.titrationGroups[titration_group_index]
        titration_state = self.titrationGroups[titration_group_index]['titration_states'][titration_state_index]
        
        # Modify charges and exceptions.
        for force in self.forces_to_update:
            # Get name of force class.
            force_classname = force.__class__.__name__
            # Get atom indices and charges.
            charges = titration_state['charges']
            atom_indices = titration_group['atom_indices']
            # Update charges.
            # TODO: Handle Custom forces, looking for "charge" and "chargeProd".
            for (charge_index, atom_index) in enumerate(atom_indices):
                if force_classname == 'NonbondedForce':
                    [charge, sigma, epsilon] = force.getParticleParameters(atom_index)
                    if debug: print " modifying NonbondedForce atom %d : (charge, sigma, epsilon) : (%s, %s, %s) -> (%s, %s, %s)" % (atom_index, str(charge), str(sigma), str(epsilon), str(charges[charge_index]), str(sigma), str(epsilon))
                    force.setParticleParameters(atom_index, charges[charge_index], sigma, epsilon)
                elif force_classname == 'GBSAOBCForce':
                    if debug: print " modifying GBSAOBCForce atom %d : (charge, radius, scaleFactor) : (%s, %s, %s) -> (%s, %s, %s)" % (atom_index, str(charge), str(radius), scaleFactor, str(charges[charge_index]), str(radius), scaleFactor)
                    [charge, radius, scaleFactor] = force.getParticleParameters(atom_index)
                    force.setParticleParameters(atom_index, charges[charge_index], radius, scaleFactor)
                else:
                    raise Exception("Don't know how to update force type '%s'" % force_classname)
            # Update exceptions
            # TODO: Handle Custom forces.
            if force_classname == 'NonbondedForce':
                for exception_index in titration_group['exception_indices']:
                    [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
                    [charge1, sigma1, epsilon1] = force.getParticleParameters(particle1)
                    [charge2, sigma2, epsilon2] = force.getParticleParameters(particle2)
                    #print "chargeProd: old %s new %s" % (str(chargeProd), str(self.coulomb14scale * charge1 * charge2))
                    chargeProd = self.coulomb14scale * charge1 * charge2
                    # BEGIN UGLY HACK
                    # chargeprod and sigma cannot be identically zero or else we risk the error:
                    # Exception: updateParametersInContext: The number of non-excluded exceptions has changed
                    # TODO: Once OpenMM interface permits this, omit this code.
                    if (2*chargeProd == chargeProd): chargeProd = sys.float_info.epsilon                        
                    if (2*epsilon == epsilon): epsilon = sys.float_info.epsilon
                    # END UGLY HACK
                    force.setExceptionParameters(exception_index, particle1, particle2, chargeProd, sigma, epsilon)

            # Update parameters in Context, if specified.
            if context and hasattr(force, 'updateParametersInContext'): 
                force.updateParametersInContext(context)                

        # Update titration state records.
        self.titrationStates[titration_group_index] = titration_state_index

        return

    def update(self, context):
        """
        Perform a Monte Carlo update of the titration state.

        ARGUMENTS

        context (simtk.openmm.Context) - the context to update

        NOTE

        The titration state actually present in the given context is not checked; it is assumed the MonteCarloTitration internal state is correct.

        """

        # Perform a number of protonation state update trials.
        for attempt in range(self.nattempts_per_update):
            # Choose how many titratable groups to simultaneously attempt to update.
            ndraw = 1
            if (self.getNumTitratableGroups() > 1) and (random.random() < self.simultaneous_proposal_probability):
                ndraw = 2
                
            # Choose groups to update.
            # TODO: Use Gibbs or Metropolized Gibbs sampling?  Or always accept proposals to same state?
            titration_group_indices = random.sample(range(self.getNumTitratableGroups()), ndraw)
            
            # Compute initial probability of this protonation state.
            log_P_initial = self._compute_log_probability(context)

            if self.debug:
                state = context.getState(getEnergy=True)
                initial_potential = state.getPotentialEnergy()
                print "   initial %s   %12.3f kcal/mol" % (str(self.getTitrationStates()), initial_potential / units.kilocalories_per_mole)

            # Perform update attempt.
            initial_titration_states = copy.deepcopy(self.titrationStates) # deep copy
            for titration_group_index in titration_group_indices:
                # Choose a titration state with uniform probability (even if it is the same as the current state).
                titration_state_index = random.choice(range(self.getNumTitrationStates(titration_group_index)))
                self.setTitrationState(titration_group_index, titration_state_index, context)
            final_titration_states = copy.deepcopy(self.titrationStates) # deep copy

            # TODO: Always accept self transitions, or avoid them altogether.
            
            # Compute final probability of this protonation state.
            log_P_final = self._compute_log_probability(context)
            
            # Compute work and store work history.
            work = - (log_P_final - log_P_initial)
            self.work_history.append( (initial_titration_states, final_titration_states, work) )

            # Accept or reject with Metropolis criteria.
            log_P_accept = -work
            if self.debug:
                print "   proposed log probability change: %f -> %f | work %f" % (log_P_initial, log_P_final, work)
                print ""
            self.nattempted += 1
            if (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept)):
                # Accept.
                self.naccepted += 1
            else:
                # Reject.
                # Restore titration states.
                for titration_group_index in titration_group_indices:
                    self.setTitrationState(titration_group_index, initial_titration_states[titration_group_index], context)
                # TODO: If using NCMC, restore coordinates.
        
        return


# jn routines: setPartialTitrationState and update_ncmc + + + + + + + + + + + + + + + + + + + + + +
    def local_verlet(state,m,v,q,dt):
		#velocity half-step update
		f = state.getForces( asNumpy=True )        
		v += f/m * (0.5*dt)
		#position full step update
		q += v*dt
		#velocity half step (with updated positions)
		context.setPositions(q)
		f = state.getForces( asNumpy=True ) 
		v += f/m * (0.5*dt)
      #not working yet!


    def setPartialTitrationState(self, titration_group_index, titration_state_index_initial, titration_state_index_final, lambda_t, context=None, debug=False):
        """
        Change the titration state of the designated group for the provided state.

        ARGUMENTS
        
        titration_group_index (int) - the index of the titratable group whose titration state should be updated
        titration_state_index_initial (int) - the initial titration state 
        titration_state_index_final (int)   - the final titration state 
        lambda_t (float) - value used to assign superposition of charge states 
        OPTIONAL ARGUMENTS

        context (simtk.openmm.Context) - if provided, will update protonation state in the specified Context (default: None)
        debug (boolean) - if True, will print debug information

        """

        # Check parameters for validity.
        if titration_group_index not in range(self.getNumTitratableGroups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." % (titration_group_index, self.getNumTitratableGroups()))        
        if titration_state_index_final not in range(self.getNumTitrationStates(titration_group_index)):
            raise Exception("Invalid titration state requested.  Requested %d, valid states are in range(%d)." % (titration_state_index, self.getNumTitrationStates(titration_group_index)))

#jn: only need the charges (for now from the initial group
        charges_initial= self.titrationGroups[titration_group_index]['titration_states'][titration_state_index_initial]['charges']

        # Get (final) titration group and state.  
	titration_group = self.titrationGroups[titration_group_index]
        titration_state = self.titrationGroups[titration_group_index]['titration_states'][titration_state_index_final]

        
#jndb        print("jn in Partial update");
        # Modify charges and exceptions.
        # Update charges.
	charges = titration_state['charges']  #jn NOTE: this refers to the 'final' charge state!
	atom_indices = titration_group['atom_indices']

	for (charge_index, atom_index) in enumerate(atom_indices):
	    #jn: getting parameters from previous lambda state (charge, namely)
	    #jn: getting new charge and mixing: 
	    charge_initial = charges_initial[ charge_index ] 
	    charge_final   = charges[        charge_index ]  #note how charge_final is assigned
	    charge_mixture=(1-lambda_t)*charge_initial + lambda_t*charge_final
	    
	    for force in self.forces_to_update:  #this loop should go inside 
		# Get name of force class.
		force_classname = force.__class__.__name__
		# Get atom indices and charges.
                if force_classname == 'NonbondedForce':
                    #get the parameters from NonbondedForce
		    [charge_previous, sigma, epsilon] = force.getParticleParameters(atom_index)
                    #print some debugs
                    if debug: print " partially modifying NonbondedForce atom %d : (charge, sigma, epsilon) : (%s, %s, %s) -> (%s, %s, %s)" % (atom_index, str(charge_previous), str(sigma), str(epsilon), str(charge_mixture), str(sigma), str(epsilon))
		    if debug: print "   lambda=(%s) , q(t->T)= (%s -> %s),  q(t-1)= %s, q(t)= %s " % (lambda_t,charge_initial,charge_final,charge_previous,charge_mixture)
                    #update nonbonded force
		    force.setParticleParameters(atom_index, charge_mixture, sigma, epsilon)

                elif force_classname == 'GBSAOBCForce':
                    #get the parameters from GBSAOBCForce
		    [charge, radius, scaleFactor] = force.getParticleParameters(atom_index)
                    #print some debugs
		    if debug: print " modifying GBSAOBCForce atom %d : (charge, radius, scaleFactor) : (%s, %s, %s) -> (%s, %s, %s)" % (atom_index, str(charge_previous), str(radius), scaleFactor, str(charge_mixture), str(radius), scaleFactor)
		    #update GBSAOBCForce
		    force.setParticleParameters(atom_index, charge_mixture, sigma, epsilon)

                else:
                    raise Exception("Don't know how to update force type '%s'" % force_classname)
            # Update exceptions
            # TODO: Handle Custom forces.
            if force_classname == 'NonbondedForce':
                for exception_index in titration_group['exception_indices']:
                    [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
                    [charge1, sigma1, epsilon1] = force.getParticleParameters(particle1)
                    [charge2, sigma2, epsilon2] = force.getParticleParameters(particle2)
                    #print "chargeProd: old %s new %s" % (str(chargeProd), str(self.coulomb14scale * charge1 * charge2))
                    chargeProd = self.coulomb14scale * charge1 * charge2
                    # BEGIN UGLY HACK
                    # chargeprod and sigma cannot be identically zero or else we risk the error:
                    # Exception: updateParametersInContext: The number of non-excluded exceptions has changed
                    # TODO: Once OpenMM interface permits this, omit this code.
                    if (2*chargeProd == chargeProd): chargeProd = sys.float_info.epsilon                        
                    if (2*epsilon == epsilon): epsilon = sys.float_info.epsilon
                    # END UGLY HACK
                    force.setExceptionParameters(exception_index, particle1, particle2, chargeProd, sigma, epsilon)

            # Update parameters in Context, if specified.
            if context and hasattr(force, 'updateParametersInContext'):
                force.updateParametersInContext(context) 
                             

        # Update titration state records.
        # jn note: this sets the titration state to the 'initial state', so that titratStates doesn't know about lambda or the final state.  
        self.titrationStates[titration_group_index] = titration_state_index_initial
        # jn note: adjusting titration state to final state if lambda_t=1
        if (lambda_t==1.0):
       #     print "jn setting final state for lambda=1"
	    self.titrationStates[titration_group_index] = titration_state_index_final     
               
        if debug:        print("end partial titration state update")
#        code.interact(local=locals())
#        sys.exit()
        return
#====================================================================================================================

    def update_ncmc(self, context,m,temperature,n_work_steps,n_prop_steps,timestep):
        """
        Perform a Monte Carlo update of the titration state.
        JN: making a copy of update and working from that

        ARGUMENTS

        context (simtk.openmm.Context) - the context to update
        temperature    - the temperature in kelvin (required for MB distribution of velocities and Langevin reinitialization)
        m              - masses numpy array
        n_work_steps   - number of work steps ( lambda(t) = t / n_work_steps )   
        n_prop_steps   - number of simulation steps at fixed lambda

n
        The titration state actually present in the given context is not checked; it is assumed the MonteCarloTitration internal state is correct.

        """

#jndb pre-trial momentum bookkeepping---------------------------------
# jn:  don't forget to add a 'momentum reversal branch' (you can do that inside update_ncmc)


# getting state:
	state=context.getState(getVelocities=True,getPositions=True)
	velocities_restore=state.getVelocities()
        positions_restore=state.getPositions()
	print ("velocities to restore")
        print(velocities_restore[0]) 
	print("positions to restore")
	print(positions_restore[0])

        # Maxwell Boltzmann distribution of velocities (for now)
        context.setVelocitiesToTemperature(temperature)        
        print("jn after first velocity initialization (in update_ncmc):")
	state=context.getState(getVelocities=True,getPositions=True,getForces=True,getEnergy=True)
	# printing velocites to test randomization step 
        velocity_test=state.getVelocities()
        print(velocity_test[0])


        q = state.getPositions(  asNumpy=True )
        v = state.getVelocities( asNumpy=True )
        print "jn before titration site loop"
        print"q: "+str(q[0,:])
        print"v: "+str(v[0,:])

        # Perform a number of protonation state update trials.

        for attempt in range(self.nattempts_per_update):
            # Choose how many titratable groups to simultaneously attempt to update.
            ndraw = 1
            if (self.getNumTitratableGroups() > 1) and (random.random() < self.simultaneous_proposal_probability):
                ndraw = 2
                
            # Choose groups to update.
            # TODO: Use Gibbs or Metropolized Gibbs sampling?  Or always accept proposals to same state?
            
            #maybe create an array of energies to bias selection to low energy states? (This will take a bit of doing, btw)
            titration_group_indices = random.sample(range(self.getNumTitratableGroups()), ndraw)
            
            # Compute initial probability of this protonation state.
            log_P_initial = self._compute_log_probability(context)

            if self.debug:
                state = context.getState(getVelocities=True,getPositions=True,getForces=True,getEnergy=True)
                initial_potential = state.getPotentialEnergy()
                print "   initial %s   %12.3f kcal/mol" % (str(self.getTitrationStates()), initial_potential / units.kilocalories_per_mole)               
                print "U_init(kJ/mol)=", str(initial_potential)

            # Perform update attempt.
            initial_titration_states = copy.deepcopy(self.titrationStates) # deep copy
            for titration_group_index in titration_group_indices:
                # new setPartialTitrationState needs initial and final states
                titration_state_index_initial  = self.titrationStates[titration_group_index]

                # Choose a titration state with uniform probability (jn now excluding self transitions).
                titration_state_list=range(self.getNumTitrationStates(titration_group_index))
                index_to_remove=titration_state_list.index(titration_state_index_initial)
                del titration_state_list[index_to_remove]
                titration_state_index_final = random.choice(titration_state_list)

                print"jn check titration states for self transitions (maybe an Exceptions should go here?)" 

#jn: adding the lambda loop: 
 
	    print("jn before first propagation and lambda loop")
#jndb	    code.interact(local=locals())
	    print"q: "+str(q[0,:])
	    print"v: "+str(v[0,:])

#jn: adding the first propagation step to symmetrize the ncmc trial:
            state=context.getState(getForces=True,getEnergy=True,getVelocities=True,getPositions=True)
	    f  = state.getForces( asNumpy=True )    
            for j_i in range(n_prop_steps):
                v=state.getVelocities(asNumpy=True)
                q=state.getPositions(asNumpy=True)

		v += f/m * (0.5*timestep)
		#position full step update
		q += v*timestep
		#velocity half step (with updated positions)
		context.setPositions(q)
                state=context.getState(getForces=True,getEnergy=True,getVelocities=True,getPositions=True)
		f = state.getForces( asNumpy=True ) 
		v += f/m * (0.5*timestep)
                #update state for next round

                context.setVelocities(v) #for debugs only
                 
                state=context.getState(getForces=True,getEnergy=True,getVelocities=True,getPositions=True)
 
                if (self.debug==True):
                        U=state.getPotentialEnergy()
                        K=state.getKineticEnergy()

			H=U.value_in_unit(U.unit)  #units=kJ/mol
    #                   jn fun fact:  (nm^2 dalton) / ps^2 (KE units)  = 1 kJ/mol (U units)  
			H+=0.5*(m*v**2).in_units_of(U.unit).sum()
		#	print "j) "+ str(j_i) + "H: "+str(H) +" "+ str(U.unit) + " U: "+ str(U.value_in_unit(U.unit)) + " K: " + str( 0.5*(m*v**2).sum() )
          #              print "    f: "+str(f[0,0]) + "   q: "+str(q[0,0]) + "   v: "+ str(v[0,0])

                        log_P_test=self._compute_log_probability(context)

                        #code.interact(local=locals())
                        print " H(org): " + str(H) + "  H(alt): " + str(U+K) + " H(from routine): " + str(log_P_test)
                      

	    print("jn 03jul2014: line 925 after first propagation and before lambda loop")
#jndb	    code.interact(local=locals())
	    print"q: "+str(q[0,:])
	    print"v: "+str(v[0,:])
            print "\n"; code.interact(local=locals())
            import sys; sys.exit()


	    for i_t in range(1,n_work_steps+1): # range needs to be 1:n_w for lambda calc
		lambda_t = float(i_t)/n_work_steps 
                # perturbation step with partial titration
		self.setPartialTitrationState(titration_group_index, titration_state_index_initial,titration_state_index_final, lambda_t, context,debug=False)
# original function 
# 		self.setTitrationState(titration_group_index, titration_state_index, context)
                print "jn: positions and velocities: lambda_t="+str(lambda_t)
                print q[0,:] 
                print v[0,:]

                if (self.debug==True):
                        U=state.getPotentialEnergy()
			H=U.value_in_unit(U.unit)  #units=kJ/mol
    #                   jn fun fact:  (nm^2 dalton) / ps^2 (KE units)  = 1 kJ/mol (U units)  
			H+=0.5*(m*v**2).sum()
			print "i) "+ str(i_t) + "H: "+str(H) +" "+ str(U.unit) + " U: "+ str(U.value_in_unit(U.unit)) + " K: " + str( 0.5*(m*v**2).sum() )






                for j_i in range(n_prop_steps):
             #       print"jndb getting forces\n"
                #    code.interact(local=locals())
                    #velocity half-step update
		    f = state.getForces( asNumpy=True )        
		    v += f/m * (0.5*timestep)
                    #position full step update
                    q += v*timestep
                    #velocity half step (with updated positions)
                    context.setPositions(q)
                    f = state.getForces( asNumpy=True ) 
	            v += f/m * (0.5*timestep)

    
		    if (self.debug==True):
			H=state.getPotentialEnergy().value_in_unit(U.unit)  #units=kJ/mol
    #                   jn fun fact:  (nm^2 dalton) / ps^2 (KE units)  = 1 kJ/mol (U units)  
			H+=0.5*(m*v**2).sum()
			print "(i,j)= ("+ str(i_t) + ","+ str(j_i) +")  H: "+str(H) +" "+ str(U.unit) + " U: "+ str(U.value_in_unit(U.unit)) + " K: "+ str( 0.5*(m*v**2).sum() )   

                print "\n"; code.interact(local=locals())
                import sys; sys.exit()

#jn: I am not sure that we need this final_titration_state assignment currently:
	    final_titration_states = copy.deepcopy(self.titrationStates) # deep copy

            # Compute final probability of this protonation state.
            log_P_final = self._compute_log_probability(context)
            
            # Compute work and store work history.
            work = - (log_P_final - log_P_initial)
            self.work_history.append( (initial_titration_states, final_titration_states, work) )

            # Accept or reject with Metropolis criteria.
            log_P_accept = -work
            if self.debug:
                print "   proposed log probability change: %f -> %f | work %f" % (log_P_initial, log_P_final, work)
                print ""
            self.nattempted += 1

#jndb            print "jn check out states after ncmc trajectory"
#jndb            print "don't forget to set reference energies too"
#jndb            code.interact(local=locals())
#	    log_P_accept=1.0
            if (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept)):
                # Accept.
                self.naccepted += 1
                acc=True
            else:
                # Reject.
                acc=False
                # Restore titration states.
                for titration_group_index in titration_group_indices:
                    self.setTitrationState(titration_group_index, initial_titration_states[titration_group_index], context)
                context.setPositions(positions_restore)
                # TODO: If using NCMC, restore coordinates.


# drawing velocities from Maxwell-Boltzmann distribution
         # Maxwell Boltzmann distribution of velocities
        integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
	positions  = context.getState(getPositions=True).getPositions()
        context    = openmm.Context(system, integrator, platform)
#this section will be different if we use the 'flipping' approach:
	context.setPositions(positions)
        context.setVelocitiesToTemperature(temperature)        
	state_test=context.getState(getVelocities=True,getPositions=True)
	velocities_test=state_test.getVelocities()
        positions_test=state_test.getPositions()       


 #       print "compare to t=0 velocity:"
 #       print(velocities_restore[0]) 
 #       print("random velocity")
 #       print(velocities_test[0]) 
#        print "compare to t=0 coords:"
#        print(positions_restore[0])
#        print ("coordinate states: accepted = "+ str(acc) )
#        print(positions_test[0])

#jndb            code.interact(local=locals())

        return context 

#end jn routines:  #psetPartialTitrationState and update_ncmc +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++


        
    def getAcceptanceProbability(self): 
        """
        Return the fraction of accepted moves
        
        RETURNS
        
        fraction (float) - the fraction of accepted moves

        """
        return float(self.naccepted) / float(self.nattempted)

    def _compute_log_probability(self, context):
        """
        Compute log probability of current configuration and protonation state.
        
        """
        temperature = self.temperature 
        kT = kB * temperature # thermal energy
        beta = 1.0 / kT # inverse temperature

        # Add energetic contribution to log probability.
        state = context.getState(getEnergy=True)
        total_energy = state.getPotentialEnergy() + state.getKineticEnergy()
        log_P = - beta * total_energy

        # TODO: Add pressure contribution for periodic simulations.

        # Correct for reference states.
        for (titration_group, titration_state_index) in zip(self.titrationGroups, self.titrationStates):
            titration_state = titration_group['titration_states'][titration_state_index]
            pKref = titration_state['pKref']
            proton_count = titration_state['proton_count']
            relative_energy = titration_state['relative_energy']
            #print "proton_count = %d | pH = %.1f | pKref = %.1f | %.1f | %.1f | beta*relative_energy = %.1f" % (proton_count, self.pH, pKref, -beta*total_energy , - proton_count * (self.pH - pKref) * math.log(10), +beta*relative_energy)
            log_P += - proton_count * (self.pH - pKref) * math.log(10) + beta * relative_energy 
            
        # Return the log probability.
        return total_energy #log_P
    
    def getNumAttemptsPerUpdate(self):
        """
        Get the number of Monte Carlo titration state change attempts per call to update().
	
        RETURNS

        nattempts_per_iteration (int) - the number of attempts to be made per iteration
        
        """
        return self.nattempts_per_update

    def setNumAttemptsPerUpdate(self, nattempts=None):
        """
        Set the number of Monte Carlo titration state change attempts per call to update().
        
        ARGUMENTS 
          nattempts (int) - the number to attempts to make per iteration;
                            if None, this value is computed automatically based on the number of titratable groups (default None)
        """
        self.nattempts_per_update = nattempts
        if nattempts is None:
            # TODO: Perform enough titration attempts to ensure thorough mixing without taking too long per update.
            # TODO: Cache already-visited states to avoid recomputing?
            self.nattempts_per_update = self.getNumTitratableGroups()

        
#=================================================================================
# MAIN AND TESTS
#=================================================================================

if __name__ == "__main__":
    import doctest
    doctest.testmod()
    
    #
    # Test with an example from the Amber 11 distribution.
    #
    
    # Parameters.
    niterations = 500 # number of dynamics/titration cycles to run
    nsteps = 500 # number of timesteps of dynamics per iteration
    temperature = 300.0 * units.kelvin
    timestep = 1.0 * units.femtoseconds
    collision_rate = 9.1 / units.picoseconds
    pH = 7.0

    timestep_ncmc=0.5*timestep #size of nMD timestep in ncmc trials (VV with no shake)

#   jn: NCMC variables added
    n_work_steps=10 # number of slices for lamba...T in paper: (l(t)=t/T)
    n_prop_steps=150   # number of md steps per l(t) may need to be adjusted
                     # for optimal performance when considering latency 
                     # in context updating

    # Filenames.
    prmtop_filename = 'amber-example/prmtop'
    inpcrd_filename = 'amber-example/min.x'
    cpin_filename = 'amber-example/cpin'
   
 
    # Calibration on a terminally-blocked amino acid in implicit solvent
    #prmtop_filename = 'calibration-implicit/tyr.prmtop'
    #inpcrd_filename = 'calibration-implicit/tyr.inpcrd'
    #cpin_filename =   'calibration-implicit/tyr.cpin'
    #pH = 9.6

    #prmtop_filename = 'calibration-explicit/his.prmtop'
    #inpcrd_filename = 'calibration-explicit/his.inpcrd'
    #cpin_filename =   'calibration-explicit/his.cpin'
    #pH = 6.5

    #prmtop_filename = 'calibration-implicit/his.prmtop'
    #inpcrd_filename = 'calibration-implicit/his.inpcrd'
    #cpin_filename =   'calibration-implicit/his.cpin'
    #pH = 6.5
    
    # Load the AMBER system.
    import simtk.openmm.app as app
    print "Creating AMBER system..."
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)

#jndb    print("\n ==jn: generating system output before titration call==\n")
#jndb    import code; code.interact(local=locals())
#jndb    print (str(openmm.XmlSerializer.serialize(state)))   
        
    # Initialize Monte Carlo titration.
    print "Initializing Monte Carlo titration..."
    mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, debug=True)

   
#jndb    print("\n ==jn: printing xml here and quitting==\n")
#jndb    print (str(openmm.XmlSerializer.serialize(system)))  
#jndb    code.interact(local=locals())


    # Create integrator and context.
  #  platform_name = 'OpenCL'
    platform_name = 'CUDA'
    platform = openmm.Platform.getPlatformByName(platform_name)
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(inpcrd.getPositions())

#jn: generating a mass array 
    print "Creating masses array..."
    nparticles = system.getNumParticles()   
    masses = units.Quantity(numpy.zeros([nparticles,3], numpy.float64), units.amu)
    for particle_index in range(nparticles):
        masses[particle_index,:] = system.getParticleMass(particle_index)
#question: does this numpy array get passed as a pointer to update_ncmc?
   
 

    # Initialize PDB output.
    # jn adding new name
    pdboutfile = open('trajectory_jn.pdb', 'w')
   


 
    # Minimize energy
    print("jn disabling minimizer for debugging")
    print "Minimizing energy..."
#    openmm.LocalEnergyMinimizer.minimize(context, 10.0)

    # Run dynamics.
    state = context.getState(getEnergy=True)
   


    potential_energy = state.getPotentialEnergy()
    print "Initial protonation states: %s   %12.3f kcal/mol" % (str(mc_titration.getTitrationStates()), potential_energy/units.kilocalories_per_mole)

   
    for iteration in range(niterations):
        # Run some dynamics.
        initial_time = time.time()
        integrator.step(nsteps)
        state = context.getState(getEnergy=True)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        print "  %.3f s elapsed for %d steps of dynamics" % (elapsed_time, nsteps)

        # Attempt protonation state changes.
        initial_time = time.time()
#        mc_titration.update(context)   

#=======jn: new ncmc routine and wrappers===============
 
        context=mc_titration.update_ncmc(context,masses,temperature,n_work_steps,n_prop_steps,timestep_ncmc)
        print"jn: do we need to pass context back?"
#	integrator=context.getIntegrator()

        state_test=context.getState(getVelocities=True, getPositions=True)
        position_test=state_test.getPositions()
 
#        print ("coordinate states after update_ncmc" )
#        print(position_test[0])

#	context.setVelocitiesToTemperature(temperature)        
#        print("random velocity after update_ncmc")
#        velocities_test=state_test.getVelocities()
#        print(velocities_test[0]) 
                

#        code.interact(local=locals())
#        sys.exit()
#========================================================


        state = context.getState(getEnergy=True)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        print "  %.3f s elapsed for %d titration trials" % (elapsed_time, mc_titration.getNumAttemptsPerUpdate())

        # Show titration states.
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()
#        code.interact(local=locals())

        print "Iteration %5d / %5d:    %s   %12.3f kcal/mol (%d / %d accepted)" % (iteration, niterations, str(mc_titration.getTitrationStates()), potential_energy/units.kilocalories_per_mole, mc_titration.naccepted, mc_titration.nattempted)

#        print("jndb check work_history")
#        code.interact(local=locals())

        # Write trajectory frame.
        state = context.getState(getPositions=True)
        positions = state.getPositions()
        app.PDBFile.writeModel(prmtop.topology, positions, pdboutfile, modelIndex=iteration)

        # TODO: Write out protonation states to a file

