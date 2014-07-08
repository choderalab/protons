#!/usr/local/bin/env python

#======================================================================================
# MODULE DOCSTRING
#======================================================================================

"""
Constant pH dynamics test.

DESCRIPTION
/
This module tests the constant pH functionality in OpenMM


NOTES

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

#from db import * 


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
#        """
#        Change the titration state of the designated group for the provided state.
#
#        ARGUMENTS
#        
#        titration_group_index (int) - the index of the titratable group whose titration state should be updated
#        titration_state_index (int) - the titration state to set as active
#        
#        OPTIONAL ARGUMENTS
#
#        context (simtk.openmm.Context) - if provided, will update protonation state in the specified Context (default: None)
#        debug (boolean) - if True, will print debug information
#
#        """
#
#        # Check parameters for validity.
#        if titration_group_index not in range(self.getNumTitratableGroups()):
#            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." % (titration_group_index, self.getNumTitratableGroups()))        
#        if titration_state_index not in range(self.getNumTitrationStates(titration_group_index)):
#            raise Exception("Invalid titration state requested.  Requested %d, valid states are in range(%d)." % (titration_state_index, self.getNumTitrationStates(titration_group_index)))
#
#
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
                     [charge, radius, scaleFactor] = force.getParticleParameters(atom_index)
                     force.setParticleParameters(atom_index, charges[charge_index], radius, scaleFactor)
                     if debug: print " modifying GBSAOBCForce atom %d : (charge, radius, scaleFactor) : (%s, %s, %s) -> (%s, %s, %s)" % (atom_index, str(charge), str(radius), scaleFactor, str(charges[charge_index]), str(radius), scaleFactor)

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
#
#  def update(self, context):
#      """
#      Perform a Monte Carlo update of the titration state.
#
#      ARGUMENTS
#
#      context (simtk.openmm.Context) - the context to update
#
#      NOTE
#
#      The titration state actually present in the given context is not checked; it is assumed the MonteCarloTitration internal state is correct.
#
#      """
#
#      # Perform a number of protonation state update trials.
#      for attempt in range(self.nattempts_per_update):
#          # Choose how many titratable groups to simultaneously attempt to update.
#          ndraw = 1
#          if (self.getNumTitratableGroups() > 1) and (random.random() < self.simultaneous_proposal_probability):
#              ndraw = 2
#              
#          # Choose groups to update.
#          # TODO: Use Gibbs or Metropolized Gibbs sampling?  Or always accept proposals to same state?
#          titration_group_indices = random.sample(range(self.getNumTitratableGroups()), ndraw)
#          
#          # Compute initial probability of this protonation state.
#          log_P_initial = self._compute_log_probability(context)
#
#          if self.debug:
#              state = context.getState(getEnergy=True)
#              initial_potential = state.getPotentialEnergy()
#              print "   initial %s   %12.3f kcal/mol" % (str(self.getTitrationStates()), initial_potential / units.kilocalories_per_mole)
#
#          # Perform update attempt.
#          initial_titration_states = copy.deepcopy(self.titrationStates) # deep copy
#          for titration_group_index in titration_group_indices:
#              # Choose a titration state with uniform probability (even if it is the same as the current state).
#              titration_state_index = random.choice(range(self.getNumTitrationStates(titration_group_index)))
#              self.setTitrationState(titration_group_index, titration_state_index, context)
#          final_titration_states = copy.deepcopy(self.titrationStates) # deep copy
#
#          # TODO: Always accept self transitions, or avoid them altogether.
#          
#          # Compute final probability of this protonation state.
#          log_P_final = self._compute_log_probability(context)
#          
#          # Compute work and store work history.
#          work = - (log_P_final - log_P_initial)
#          self.work_history.append( (initial_titration_states, final_titration_states, work) )
#
#          # Accept or reject with Metropolis criteria.
#          log_P_accept = -work
#
#          self.nattempted += 1
#          acc=False
#          if (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept)):
#              # Accept.
#              self.naccepted += 1
#              acc=True
#          else:
#              # Reject.
#              # Restore titration states.
#              for titration_group_index in titration_group_indices:
#                  self.setTitrationState(titration_group_index, initial_titration_states[titration_group_index], context)
#              # TODO: If using NCMC, restore coordinates.
#      
#          if self.debug:
#              print "   proposed log probability change: %f -> %f | work %f acc? %s" % (log_P_initial, log_P_final, work,str(acc))
#              print ""
#      
#      return
#

#jn routines: setPartialTitrationState and update_ncmc +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    def setPartialTitrationState(self, titration_group_indices, titration_state_index_initial, titration_state_index_final, lambda_t, context=None, debug=False):
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
        drawcount=0
        for titration_group_index in titration_group_indices:
            if titration_group_index not in range(self.getNumTitratableGroups()):
                raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." % (titration_group_index, self.getNumTitratableGroups()))        
            if titration_state_index_final[drawcount] not in range(self.getNumTitrationStates(titration_group_index)):
                raise Exception("Invalid titration state requested.  Requested %d, valid states are in range(%d)." % (titration_state_index, self.getNumTitrationStates(titration_group_index)))
            drawcount+=1

        #begin looping through titration group indices:
        drawcount=0
        for titration_group_index in titration_group_indices: 
            # The MonteCarloTitration object stores the initial charge state until lambda=1
            charges_initial= self.titrationGroups[titration_group_index]['titration_states'][titration_state_index_initial[drawcount]]['charges']

            # Get (final) titration group and state using the index of the final state.  
            titration_group = self.titrationGroups[titration_group_index]
            titration_state = self.titrationGroups[titration_group_index]['titration_states'][titration_state_index_final[drawcount]]
            
            # Modify charges and exceptions.
            # Update charges.
            charges = titration_state['charges']  #jn NOTE: this refers to the 'final' charge state!
            atom_indices = titration_group['atom_indices']

            for (charge_index, atom_index) in enumerate(atom_indices):
            # getting initial, final charges charge and mixing charges for fractional states: 
                charge_initial = charges_initial[ charge_index ] 
                charge_final   = charges[        charge_index ]  #note how charge_final is assigned
                charge_mixture=(1-lambda_t)*charge_initial + lambda_t*charge_final
            
                for force in self.forces_to_update:  #this loop should go inside 
                    # Get name of force class.
                    force_classname = force.__class__.__name__
                    # Get atom indices and charges.
                    if force_classname == 'NonbondedForce':
                        # get the parameters from NonbondedForce
                        [charge_previous, sigma, epsilon] = force.getParticleParameters(atom_index)
                        # debugs
                        if debug: print " partially modifying NonbondedForce atom %d : (charge, sigma, epsilon) : (%s, %s, %s) -> (%s, %s, %s)" % (atom_index, str(charge_previous), str(sigma), str(epsilon), str(charge_mixture), str(sigma), str(epsilon))
                        if debug: print "   lambda=(%s) , q(t->T)= (%s -> %s),  q(t-1)= %s, q(t)= %s " % (lambda_t,charge_initial,charge_final,charge_previous,charge_mixture)
                      
                        # update nonbonded force
                        force.setParticleParameters(atom_index, charge_mixture, sigma, epsilon)

                    elif force_classname == 'GBSAOBCForce':
                        # get the parameters from GBSAOBCForce
                        [charge, radius, scaleFactor] = force.getParticleParameters(atom_index)
                        # print some debugs
                        if debug: 
                            print " modifying GBSAOBCForce atom %d : (charge, radius, scaleFactor) : (%s, %s, %s) -> (%s, %s, %s)" % (atom_index, str(charge_previous), str(radius), scaleFactor, str(charge_mixture), str(radius), scaleFactor)
                        #jn Question:  How are Born Radii computed here?
                            print "do we know how Born Radii are computed/updated?"
                        #update GBSAOBCForce
                            force.setParticleParameters(atom_index, charge_mixture, radius, epsilon)

                    else:
                        raise Exception("Don't know how to update force type '%s'" % force_classname)
                # Update exceptions
                # TODO: Handle Custom forces.
                if force_classname == 'NonbondedForce':
                    for exception_index in titration_group['exception_indices']:
                        [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(exception_index)
                        [charge1, sigma1, epsilon1] = force.getParticleParameters(particle1)
                        [charge2, sigma2, epsilon2] = force.getParticleParameters(particle2)
                        # print "chargeProd: old %s new %s" % (str(chargeProd), str(self.coulomb14scale * charge1 * charge2))
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
            # note: titration state is set to initial state until last step (lambda==1).  
            self.titrationStates[titration_group_index] = titration_state_index_initial[drawcount]
            # jn note: adjusting titration state to final state if lambda_t=1
            if (lambda_t==1.0):
                self.titrationStates[titration_group_index] = titration_state_index_final[drawcount]     
        
            drawcount+=1
        #end looping throug titration group indices:
               
        if debug:        print("end partial titration state update")

        return
#====================================================================================================================

    def update_ncmc(self, context,temperature,timestep,n_work_steps,n_prop_steps):
        """
        Perform a Monte Carlo update of the titration state.
        JN: making a copy of update and working from that

        ARGUMENTS

        context (simtk.openmm.Context) - the context to update
        temperature    - the temperature in kelvin (required for MB distribution of velocities and Langevin reinitialization)
        collision_rate - required for Langevin reinitialization
        n_work_steps   - number of work steps ( lambda(t) = t / n_work_steps )   
        n_prop_steps   - number of simulation steps at fixed lambda


        The titration state actually present in the given context is not checked; it is assumed the MonteCarloTitration internal state is correct.

        """
        velocity_option='randomize' # ' randomize (str) : at beginning of each trial and at each accepted step'
        #velocity_option='reversal'  # ' reversal  (str) : no randomization...reverse at each accepted step'
 
        # for both options, rejected steps restore to previous positions and (nonreversed)v

        
        if velocity_option=='reversal':
            velocities=context.getState(getVelocities=True).getVelocities()
                  
        integrator = openmm.VerletIntegrator(timestep)
        positions  = context.getState(getPositions=True).getPositions()
        context    = openmm.Context(system, integrator, platform)
        context.setPositions(positions)

        #assigning velocities:
        # random initialization if option is selected:
        if velocity_option=='randomize': context.setVelocitiesToTemperature(temperature)        
        # restoring velocities from last step before context change:
        elif velocity_option=='reversal': context.setVelocities(velocities)
       
        # Perform a number of protonation state update 'attempts' .
        acc=False #initializing acceptance flag
       #	state=context.getState(getVelocities=True,getPositions=True,getEnergy=True)
        for attempt in range(self.nattempts_per_update):
            # Choose how many titratable groups to simultaneously attempt to update.
            ndraw = 1
            if (self.getNumTitratableGroups() > 1) and (random.random() < self.simultaneous_proposal_probability):
                ndraw = 2
            
            
            if self.debug:
                print  "-------------------------------------------------------------"
                print "attempt # ", str(attempt),  "ndraw: "+ str(ndraw)+ "----------------------"

 
            # Choose groups to update.
            # TODO: Use Gibbs or Metropolized Gibbs sampling?  Or always accept proposals to same state?
            # maybe create an array of energies to bias selection to low energy states?
            # selection probability may end up being complex enough to warrant a function.

            titration_group_indices = random.sample(range(self.getNumTitratableGroups()), ndraw)

            state=context.getState(getVelocities=True, getPositions=True,getEnergy=True)
            velocities_restore=state.getVelocities()
            positions_restore=state.getPositions()

            if self.debug:
                print ("velocities to restore ( velocity option = "+velocity_option+" )")
                print(velocities_restore[0]) 
                print("positions to restore")
                print(positions_restore[0])
                U_restore=state.getPotentialEnergy()
                K_restore=state.getKineticEnergy()
                H_restore=U_restore+K_restore;
                print"energies: U= "+ str(U_restore) +" K= "+ str(K_restore) + "H= ",str(H_restore)

            # Compute initial probability of this protonation state.
            log_P_initial = self._compute_log_probability(context)

            if self.debug:
                state = context.getState(getEnergy=True)
                initial_potential = state.getPotentialEnergy()
                print "   initial %s   %12.3f kcal/mol" % (str(self.getTitrationStates()), initial_potential / units.kilocalories_per_mole)

            # Perform update attempt.
            drawcount=-1
            titration_state_index_initial=list()
            titration_state_index_final=list()
            initial_titration_states = copy.deepcopy(self.titrationStates) # deep copy of initial state
            for titration_group_index in titration_group_indices:
                drawcount+=1
                if self.debug: print "draw #:", str(drawcount)
                # new setPartialTitrationState needs initial and final states
                
                titration_state_index_initial.append( self.titrationStates[titration_group_index] )
                # Choose a titration state with uniform probability (now excluding self transitions).
                titration_state_list = range(self.getNumTitrationStates(titration_group_index))
                index_to_remove = titration_state_list.index(titration_state_index_initial[drawcount])
                del titration_state_list[index_to_remove]
                titration_state_index_final.append( random.choice(titration_state_list) )

                if self.debug:
                    state = context.getState(getEnergy=True)
                    initial_potential = state.getPotentialEnergy()
                    # print "   initial %s   %12.3f kcal/mol" % (str(self.getTitrationStates()), initial_potential / units.kilocalories_per_mole)
                    print " draw #: ", str(drawcount+1), ")"+ "residue # "+str(titration_group_index)\
                          + ": state change: ( " + str(titration_state_index_initial[drawcount])+ "->"+ \
                           str( titration_state_index_final[drawcount]  )  + " )"                
                
 
            if self.debug:
                H_init=log_P_initial #using a shorter name for dbugs
                print "starting energy: "+ str(log_P_initial)

           # adding the first propagation step to symmetrize the ncmc trial:
            integrator.step(n_prop_steps)
            if self.debug:
                print "conservation of energy after initial propagation step:"
                H_t = self._compute_log_probability(context)
                print "attempt: %i)  l(t=0) = 0 , step:, %i  b*H= %f b*H(t=0): %f delt: %f"%(attempt,n_prop_steps,H_t,H_init,H_t-H_init)

            # lambda loop:
            if self.debug:  print "energy conservation in lambda loop"
            for i_t in range(n_work_steps):
                lambda_t = float(i_t+1)/n_work_steps
                # perturbation step
                # new partial titration state routine
                self.setPartialTitrationState(titration_group_indices, titration_state_index_initial,titration_state_index_final, lambda_t, context,debug=False)
                if self.debug: H0_db = self._compute_log_probability(context)  #computing initial energy for conservation
                # propagation step
                integrator.step(n_prop_steps)
                if self.debug:
                    H_t = self._compute_log_probability(context)
                    print "attempt: %i)  l(t) = %f , step:, %i  b*H= %f b*H(t=0): %f delt: %f"%(attempt,lambda_t,n_prop_steps,H_t,H0_db,H_t-H0_db)

            # end lambda loop
            #----------------------------------
            
            final_titration_states = copy.deepcopy(self.titrationStates) # deep copy

            # Compute final probability of this protonation state.
            log_P_final = self._compute_log_probability(context)

            # Compute work and store work history.
            work = - (log_P_final - log_P_initial)
            self.work_history.append( (initial_titration_states, final_titration_states, work) )
            # Accept or reject with Metropolis criteria.
            log_P_accept = -work
            self.nattempted += 1


            if self.debug:
                #getting last velocity:
                state = context.getState(getVelocities=True, getPositions=True,getEnergy=True)
                velocities_last_step=state.getVelocities()
                print "velocity at end of ncmc trial:"
                print (str(velocities_last_step[0]))
                U_test=state.getPotentialEnergy()
                K_test=state.getKineticEnergy()
                H_test=U_test+K_test;
                print"trial end energies: U= "+ str(U_test) +" K= "+ str(K_test) + "H= ",str(H_test)
                print ""

          #  log_P_accept=1000.0  toggles for branch testing

            if (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept)):
            # Accept ==================
                self.naccepted += 1
                acc=True
                # jn modify velocities upon acceptance:  also reverse!
                if velocity_option=='randomize':
                    context.setVelocitiesToTemperature(temperature)        
                elif velocity_option=='reversal':
                    velocities_last_step = context.getState(getVelocities=True).getVelocities(asNumpy=True)
                    context.setVelocities(-velocities_last_step)
                      
                    if self.debug:
                        print"jndb kinetic energy conservation?"
                        state = context.getState(getVelocities=True, getPositions=True,getEnergy=True)
                        velocities_test=state.getVelocities()
                        U_rev=state.getPotentialEnergy()
                        K_rev=state.getKineticEnergy()
                        H_rev=U_test+K_test;
                        print "reversed velocities:"
                        print velocities_test[0]
                        print"trial end energies: U= "+ str(U_rev) +" K= "+ str(K_rev) + "H= ",str(H_rev)
                        print "difference in KE (s/b 0):"+ str(K_rev-K_test)



            else:
            # Reject  ================
                acc=False
                # Restore titration states.
                for titration_group_index in titration_group_indices:
                    #self.setTitrationState(titration_group_index, titration_state_index_initial, context,debug=False)
                    # we could use the partial state function if needed, but it will be slightly slower
                    self.setPartialTitrationState(titration_group_indices, titration_state_index_initial,titration_state_index_initial, 1.0, context,debug=False)
                    context.setPositions(positions_restore)
                    context.setVelocities(velocities_restore)
            # end accept/reject ====


            #updating state here (needed for accepted?)
            state = context.getState(getVelocities=True, getPositions=True,getEnergy=True)

            if self.debug:
                print "attempt:"+str(attempt)+") proposed log probability change: %f -> %f | work %f, acc= %s" % (log_P_initial, log_P_final, work,str(acc))
                print ""
                log_P_from_current = self._compute_log_probability(context)
                if (velocity_option=='randomize'):
                     print "current state has updated (random) velocities (different H)" 
                print "energy of current state: %f "%(log_P_from_current)
                if(acc): print "final energy(accepted): %f "%(log_P_final)
                if(not acc): print "final energy(rejected): %f"%(log_P_initial) 
            if self.debug:
                state=context.getState(getVelocities=True, getPositions=True,getEnergy=True)
                initial_potential = state.getPotentialEnergy()
                print "   final %s   %12.3f kcal/mol" % (str(self.getTitrationStates()), initial_potential / units.kilocalories_per_mole)

            if self.debug:
                print "state space for next iteration:"     
                state=context.getState(getVelocities=True, getPositions=True,getEnergy=True)
                velocities_test=state.getVelocities()
                positions_test=state.getPositions()
                print ("velocities:")
                print(velocities_test[0]) 
                print("positions:")
                print(positions_test[0])
                U_restore=state.getPotentialEnergy()
                K_restore=state.getKineticEnergy()
                H_restore=U_restore+K_restore;
                print"energies into next iteration: U= "+ str(U_restore) +" K= "+ str(K_restore) + "H= ",str(H_restore)
#                import code; code.interact(local=locals())

         # end attempt loop
        #----------------------------------


        # restoring langevin integrator
        integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
        positions  = context.getState(getPositions=True).getPositions()
        context    = openmm.Context(system, integrator, platform)
        context.setPositions(positions)

        context.setVelocitiesToTemperature(temperature)        
#        state_test=context.getState(getVelocities=True,getPositions=True)
#        velocities_test=state_test.getVelocities()
#        positions_test=state_test.getPositions()       

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
        return (beta*total_energy) #log_P
    
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
    niterations    = 500 # number of dynamics/titration cycles to run
    nsteps         = 5 #500 # number of timesteps of dynamics per iteration
    temperature    = 300.0 * units.kelvin
    timestep       = 1.0 * units.femtoseconds
    collision_rate = 9.1 / units.picoseconds
    pH = 7.0

#   jn: NCMC variables added
    n_work_steps=5 #10  # number of slices for lamba...T in paper: (l(t)=t/T)
    n_prop_steps=5 #50   # number of md steps per l(t) may need to be adjusted
                     # for optimal performance when considering latency 
                     # in context updating
    timestep_ncmc = 0.5* units.femtoseconds

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


    # Create integrator and context.
  #  platform_name = 'OpenCL'
    platform_name = 'CUDA'
  #  platform_name = "CPU"
    platform = openmm.Platform.getPlatformByName(platform_name)
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(inpcrd.getPositions())



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
        state = context.getState(getEnergy=True,getVelocities=True)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        print "  %.3f s elapsed for %d steps of dynamics" % (elapsed_time, nsteps)

        # Attempt protonation state changes.
        initial_time = time.time()
#        mc_titration.update(context)   



#=======jn: new ncmc routine and wrappers===============
        #velocity_test=state.getVelocities()
        #print"jndb: velocities before titration:", velocity_test[0]
         
        context=mc_titration.update_ncmc(context,temperature,timestep_ncmc,n_work_steps,n_prop_steps)

        state_test=context.getState(getVelocities=True, getPositions=True)
        position_test=state_test.getPositions()
 
#========================================================

        integrator=context.getIntegrator()
        state = context.getState(getEnergy=True)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        print "  %.3f s elapsed for %d titration trials" % (elapsed_time, mc_titration.getNumAttemptsPerUpdate())

        # Show titration states.
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()

        print "Iteration %5d / %5d:    %s   %12.3f kcal/mol (%d / %d accepted)" % (iteration, niterations, str(mc_titration.getTitrationStates()), potential_energy/units.kilocalories_per_mole, mc_titration.naccepted, mc_titration.nattempted)

        # Write trajectory frame.
        state = context.getState(getPositions=True)
        positions = state.getPositions()
        app.PDBFile.writeModel(prmtop.topology, positions, pdboutfile, modelIndex=iteration)

        # TODO: Write out protonation states to a file
