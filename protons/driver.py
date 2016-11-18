import copy
import logging
import math
import random
import re
import sys
import numpy as np
import simtk
from openmmtools.integrators import VelocityVerletIntegrator
from simtk import unit as units, openmm
from .logger import log
from abc import ABCMeta, abstractmethod
from .integrators import GHMCIntegrator

kB = (1.0 * units.BOLTZMANN_CONSTANT_kB * units.AVOGADRO_CONSTANT_NA).in_units_of(units.kilocalories_per_mole / units.kelvin)


class _BaseDrive(object):
    """An abstract base class describing the common public interface of Drive-type classes

    .. note::

        Examples of a Drive class would include the _BaseProtonDrive, which has instantaneous MC, and NCMC updates of
        protonation states of the system in its ``update`` method, and provides tracking tools, and calibration tools for
        the relative weights of the protonation states.
    """
    __metaclass__ = ABCMeta

    @abstractmethod
    def update(self):
        """
        Update the state of the system using some kind of Monte Carlo move
        """
        pass

    @abstractmethod
    def calibrate(self):
        """
        Calibrate the relative weights, gk, of the different states of the residues that are part of the system
        """
        pass

    @abstractmethod
    def import_gk_values(self):
        """
        Import the relative weights, gk, of the different states of the residues that are part of the system
        """
        pass

    @abstractmethod
    def reset_statistics(self):
        """
        Reset statistics of sams_sampler state tracking.
        """
        pass


class _BaseProtonDrive(_BaseDrive):
    """
    The _BaseProtonDrive is an abstract base class Monte Carlo driver for protonation state changes and tautomerism in OpenMM.

    Protonation state changes, and additionally, tautomers are treated using the constant-pH dynamics method of Mongan, Case and McCammon [Mongan2004]_, or Stern [Stern2007]_ and NCMC methods from Nilmeier [Nilmeier2011]_.

    References
    ----------

    .. [Mongan2004] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born implicit solvent. J Comput Chem 25:2038, 2004.
        http://dx.doi.org/10.1002/jcc.20139

    .. [Stern2007] Stern HA. Molecular simulation with variable protonation states at constant pH. JCP 126:164112, 2007.
        http://link.aip.org/link/doi/10.1063/1.2731781

    .. [Nilmeier2011] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. PNAS 108:E1009, 2011.
        http://dx.doi.org/10.1073/pnas.1106094108

    .. todo::

      * Add NCMC switching moves to allow this scheme to be efficient in explicit solvent.
      * Add alternative proposal types, including schemes that avoid proposing self-transitions (or always accept them):
        - Parallel Monte Carlo schemes: Compute N proposals at once, and pick using Gibbs sampling or Metropolized Gibbs?
      * Allow specification of probabilities for selecting N residues to change protonation state at once.
      * Add automatic tuning of switching times for optimal acceptance.
      * Extend to handle systems set up via OpenMM app Forcefield class.
      * Make integrator optional if not using NCMC

    """
    __metaclass__ = ABCMeta

    def _retrieve_ion_parameters(self, topology, system, resname):
        """
        Retrieve parameters from specified monovalent atomic ion.

        Parameters
        ----------
        topology : simtk.openmm.app.topology
            The topology from which water residues are to be identified.
        system : simtk.openmm.System
            The System object from which parameters are to be extracted.
        resname : str
            The residue name of the monovalent atomic anion or cation from which parameters are to be retrieved.

        Returns
        -------
        parameters : dict of str:float
            NonbondedForce parameter dict ('charge', 'sigma', 'epsilon') for ion parameters.

        Warnings
        --------
        * Only `NonbondedForce` parameters are returned
        * If the system contains more than one `NonbondedForce`, behavior is undefined

        """
        # Find the NonbondedForce in the system
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        nonbonded_force = forces['NonbondedForce']

        # Return the first occurrence of NonbondedForce particle parameters matching `resname`
        for residue in topology.residues():
            if residue.name == resname:
                atoms = [atom for atom in residue.atoms()]
                [charge, sigma, epsilon] = nonbonded_force.getParticleParameters(atoms[0].index)
                parameters = {'charge': charge, 'sigma': sigma, 'epsilon': epsilon}
                if self.debug: print('_retrieve_ion_parameters: %s : %s' % (resname, str(parameters)))
                return parameters

        raise Exception("resname '%s' not found in topology" % resname)

    def _identify_water_residues(self, topology, water_residue_names=('WAT', 'HOH', 'TP4', 'TP5', 'T4E')):
        """
        Compile a list of water residues that could be converted to/from monovalent ions.

        Parameters
        ----------
        topology : simtk.openmm.app.topology
            The topology from which water residues are to be identified.
        water_residue_names : list of str
            Residues identified as water molecules.

        Returns
        -------
        water_residues : list of simtk.openmm.app.Residue
            Water residues.

        TODO
        ----
        * Can this feature be added to simt.openmm.app.Topology?

        """
        water_residues = list()
        for residue in topology.residues():
            if residue.name in water_residue_names:
                water_residues.append(residue)

        if self.debug: print('_identify_water_residues: %d water molecules identified.' % len(water_residues))
        return water_residues

    def _get14scaling(self, system):
        """
        Determine Coulomb 14 scaling.

        Parameters
        ----------

        system : simtk.openmm.System
            the system to examine

        Returns
        -------

        coulomb14scale (float) - degree to which 1,4 coulomb interactions are scaled

        """
        # Look for a NonbondedForce.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
        force = forces['NonbondedForce']
        # Determine coulomb14scale from first exception with nonzero chargeprod.
        for index in range(force.getNumExceptions()):
            [particle1, particle2, chargeProd, sigma, epsilon] = force.getExceptionParameters(index)
            [charge1, sigma1, epsilon1] = force.getParticleParameters(particle1)
            [charge2, sigma2, epsilon2] = force.getParticleParameters(particle2)
            if (abs(charge1 / units.elementary_charge) > 0) and (abs(charge2 / units.elementary_charge) > 0):
                coulomb14scale = chargeProd / (charge1 * charge2)
                return coulomb14scale

        return None

    def _get14exceptions(self, system, particle_indices):
        """
        Return a list of all 1,4 exceptions involving the specified particles that are not exclusions.

        Parameters
        ----------

        system : simtk.openmm.System
            the system to examine
        particle_indices :list of int
            only exceptions involving at least one of these particles are returned

        Returns
        -------

        exception_indices : list
            list of exception indices for NonbondedForce

        Todo
        ----

        * Deal with the case where there may be multiple NonbondedForce objects.
        * Deal with electrostatics implmented as CustomForce objects (by CustomNonbondedForce + CustomBondForce)

        """
        # Locate NonbondedForce object.
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in range(system.getNumForces())}
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
                    if (2 * chargeProd == chargeProd):
                        chargeProd = sys.float_info.epsilon
                    if (2 * epsilon == epsilon):
                        epsilon = sys.float_info.epsilon
                    force.setExceptionParameters(exception_index, particle1, particle2, chargeProd, sigma, epsilon)
                    # END UGLY HACK

        return exception_indices

    def reset_statistics(self):
        """
        Reset statistics of sams_sampler state tracking.

        Todo
        ----

        * Keep track of more statistics regarding history of individual protonation states.
        * Keep track of work values for individual trials to use for calibration.

        """

        self.nattempted = 0
        self.naccepted = 0
        self.nrejected = 0
        self.work_history = list()

        return

    def _parse_fortran_namelist(self, filename, namelist_name):
        """
        Parse a fortran namelist generated by AMBER 11 constant-pH python scripts.

        Parameters
        ----------

        filename : string
            the name of the file containing the fortran namelist
        namelist_name : string
            name of the namelist section to parse

        Returns
        -------

        namelist : dict
            namelist[key] indexes read values, converted to Python types

        Notes
        -----

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
        valueInt = re.compile(r'[+-]?[0-9]+')
        valueReal = re.compile(r'[+-]?([0-9]+\.[0-9]*|[0-9]*\.[0-9]+)')
        valueString = re.compile(r'^[\'\"](.*)[\'\"]$')

        # Parse contents.
        namelist = dict()
        while len(contents) > 0:
            # Peel off variable name.
            match = re.match(r'^([^,]+)=(.+)$', contents)
            if not match:
                break
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

    def _get_proton_chemical_potential(self, titration_group_index, titration_state_index):
        """Calculate the chemical potential contribution of protons of individual titratable sites.

        Parameters
        ----------
        titration_group_index : int
            Index of the group
        titration_state_index : int
            Index of the state

        Returns
        -------
        float
        """
        titration_state = self.titrationGroups[titration_group_index]['titration_states'][titration_state_index]
        proton_count = titration_state['proton_count']
        pKref = titration_state['pKref']

        if self.debug:
            print("proton_count = %d | pH = %.1f | pKref = %.1f | %.1f " % (
                proton_count, self.pH, pKref, - proton_count * (self.pH - pKref) * math.log(10)))

        return proton_count * (self.pH - pKref) * math.log(10)

    def _get_num_titratable_groups(self):
        """
        Return the number of titratable groups.

        Returns
        -------

        ngroups : int
            the number of titratable groups that have been defined

        """

        return len(self.titrationGroups)

    def _add_titratable_group(self, atom_indices, name=''):
        """
        Define a new titratable group.

        Parameters
        ----------

        atom_indices : list of int
            the atom indices defining the sams_sampler group

        Other Parameters
        ----------------
        name : str
            name of the group, e.g. Residue: LYS 13.

        Notes
        -----

        No two sams_sampler groups may share atoms.

        """
        # Check to make sure the requested group does not share atoms with any existing sams_sampler group.
        for group in self.titrationGroups:
            if set(group['atom_indices']).intersection(atom_indices):
                raise Exception("Titration groups cannot share atoms.  The requested atoms of new sams_sampler group (%s) share atoms with another group (%s)." % (
                    str(atom_indices), str(group['atom_indices'])))

        # Define the new group.
        group = dict()
        group['atom_indices'] = list(atom_indices)  # deep copy
        group['titration_states'] = list()
        group_index = len(self.titrationGroups) + 1
        group['index'] = group_index
        group['name'] = name
        group['nstates'] = 0
        # NonbondedForce exceptions associated with this sams_sampler state
        group['exception_indices'] = self._get14exceptions(self.system, atom_indices)

        self.titrationGroups.append(group)

        # Note that we haven't yet defined any sams_sampler states, so current state is set to None.
        self.titrationStates.append(None)

        return group_index

    def _get_num_titration_states(self, titration_group_index):
        """
        Return the number of sams_sampler states defined for the specified titratable group.

        Parameters
        ----------

        titration_group_index : int
            the sams_sampler group to be queried

        Returns
        -------

        nstates : int
            the number of sams_sampler states defined for the specified sams_sampler group

        """
        if titration_group_index not in range(self._get_num_titratable_groups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." %
                            (titration_group_index, self._get_num_titratable_groups()))

        return len(self.titrationGroups[titration_group_index]['titration_states'])

    def _add_titration_state(self, titration_group_index, pKref, relative_energy, charges, proton_count):
        """
        Add a sams_sampler state to a titratable group.

        Parameters
        ----------

        titration_group_index : int
            the index of the sams_sampler group to which a new sams_sampler state is to be added
        pKref : float
            the pKa for the reference compound used in calibration
        relative_energy : simtk.unit.Quantity with units compatible with simtk.unit.kilojoules_per_mole
            the relative energy of this protonation state
        charges : list or numpy array of simtk.unit.Quantity with units compatible with simtk.unit.elementary_charge
            the atomic charges for this sams_sampler state
        proton_count : int
            number of protons in this sams_sampler state

        Notes
        -----

        The relative free energy of a sams_sampler state is computed as

        relative_energy + kT * proton_count * ln (10^(pH - pKa))
        = relative_energy + kT * proton_count * (pH - pKa) * ln 10

        The number of charges specified must match the number (and order) of atoms in the defined sams_sampler group.

        """

        # Check input arguments.
        if titration_group_index not in range(self._get_num_titratable_groups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." %
                            (titration_group_index, self._get_num_titratable_groups()))
        if len(charges) != len(self.titrationGroups[titration_group_index]['atom_indices']):
            raise Exception('The number of charges must match the number (and order) of atoms in the defined sams_sampler group.')

        state = dict()
        state['pKref'] = pKref
        state['g_k'] = relative_energy * self.beta  # dimensionless quantity
        state['charges'] = copy.deepcopy(charges)
        state['proton_count'] = proton_count
        self.titrationGroups[titration_group_index]['titration_states'].append(state)

        # Increment count of sams_sampler states and set current state to last defined state.
        self.titrationStates[titration_group_index] = self.titrationGroups[titration_group_index]['nstates']
        self.titrationGroups[titration_group_index]['nstates'] += 1

        return

    def _get_titration_state(self, titration_group_index):
        """
        Return the current sams_sampler state for the specified titratable group.

        Parameters
        ----------

        titration_group_index : int
            the sams_sampler group to be queried

        Returns
        -------

        state : int
            the sams_sampler state for the specified sams_sampler group

        """
        if titration_group_index not in range(self._get_num_titratable_groups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." %
                            (titration_group_index, self._get_num_titratable_groups()))

        return self.titrationStates[titration_group_index]

    def _get_titration_states(self):
        """
        Return the current sams_sampler states for all titratable groups.

        Returns
        -------

        states : list of int
            the sams_sampler states for all titratable groups

        """
        return list(self.titrationStates)  # deep copy

    def _get_titration_state_total_charge(self, titration_group_index, titration_state_index):
        """
        Return the total charge for the specified sams_sampler state.

        Parameters
        ----------

        titration_group_index : int
            the sams_sampler group to be queried

        titration_state_index : int
            the sams_sampler state to be queried

        Returns
        -------

        charge : simtk.openmm.Quantity compatible with simtk.unit.elementary_charge
            total charge for the specified sams_sampler state

        """
        if titration_group_index not in range(self._get_num_titratable_groups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." %
                            (titration_group_index, self._get_num_titratable_groups()))
        if titration_state_index not in range(self._get_num_titration_states(titration_group_index)):
            raise Exception("Invalid sams_sampler state requested.  Requested %d, valid states are in range(%d)." %
                            (titration_state_index, self._get_num_titration_states(titration_group_index)))

        charges = self.titrationGroups[titration_group_index]['titration_states'][titration_state_index]['charges'][:]
        return simtk.unit.Quantity((charges / charges.unit).sum(), charges.unit)

    def _set_titration_state(self, titration_group_index, titration_state_index, context=None, debug=False):
        """
        Change the sams_sampler state of the designated group for the provided state.

        Parameters
        ----------

        titration_group_index : int
            the index of the titratable group whose sams_sampler state should be updated
        titration_state_index : int
            the sams_sampler state to set as active

        Other Parameters
        ----------------

        context : simtk.openmm.Context
            if provided, will update protonation state in the specified Context (default: None)
        debug : bool
            if True, will print debug information
        """

        # Check parameters for validity.
        if titration_group_index not in range(self._get_num_titratable_groups()):
            raise Exception("Invalid titratable group requested.  Requested %d, valid groups are in range(%d)." %
                            (titration_group_index, self._get_num_titratable_groups()))
        if titration_state_index not in range(self._get_num_titration_states(titration_group_index)):
            raise Exception("Invalid sams_sampler state requested.  Requested %d, valid states are in range(%d)." %
                            (titration_state_index, self._get_num_titration_states(titration_group_index)))

        self._update_forces(titration_group_index, titration_state_index, context=context)
        self.titrationStates[titration_group_index] = titration_state_index

        return

    def _update_forces(self, titration_group_index, final_titration_state_index, initial_titration_state_index=None, fractional_titration_state=1.0, context=None):
        """
        Update the force parameters to a new sams_sampler state by reading them from the cache

        Parameters
        ----------
        titration_group_index : int
            Index of the group that is changing state
        titration_state_index : int
            Index of the state of the chosen residue
        initial_titration_state_index : int, optional, default=None
            If blending two sams_sampler states, the initial sams_sampler state to blend.
            If `None`, set to `titration_state_index`
        fractional_titration_state : float, optional, default=1.0
            Fraction of `titration_state_index` to be blended with `initial_titration_state_index`.
            If 0.0, `initial_titration_state_index` is fully active; if 1.0, `titration_state_index` is fully active.
        context : simtk.openmm.Context, optional, default=None
            If provided, will update forces state in the specified Context

        Notes
        -----
        * Every sams_sampler state has a list called forces, which stores parameters for all forces that need updating.
        * Inside each list entry is a dictionary that always contains an entry called `atoms`, with single atom parameters by name.
        * NonbondedForces also have an entry called `exceptions`, containing exception parameters.

        """
        # `initial_titration_state_index` should have no effect if not specified, so set it identical to `final_titration_state_index` in that case
        if initial_titration_state_index is None:
            initial_titration_state_index = final_titration_state_index

        # Retrieve cached force parameters fro this sams_sampler state.
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
                    # TODO : if we ever change LJ parameters, we need to look into softcore potentials
                    # and separate out the changes in charge, and sigma/eps into different steps.
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

    def _cache_force(self, titration_group_index, titration_state_index):
        """
        Cache the force parameters for a single sams_sampler state.

        Parameters
        ----------
        titration_group_index : int
            Index of the group
        titration_state_index : int
            Index of the sams_sampler state of the group

        Notes
        -----

        Call this function to set up the 'forces' information for a single sams_sampler state.
        Every sams_sampler state has a list called forces, which stores parameters for all forces that need updating.
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

    def _ncmc_ghmc(self, context, titration_group_indices, initial_titration_states, final_titration_states):
        """
        Non equilibrium candidate Monte Carlo (NCMC) with a generalized Hamiltonian Monte Carlo (GHMC) propagator.

        Notes
        -----
        The GHMC integrator is an simtk.openmm.CustomIntegrator object that calculates the protocol work internally.
        Although it appears to only take one propagation step per perturbation step (i.e. ghmc.step(1)), multiple
        propagation steps can be made depending on how the GHMC integrator was initialized.

        To ensure the NCMC protocol is time symmetric, it has the form
            propagation --> perturbation --> propagation

        Parameters
        ----------
        context : simtk.openmm.Context
            The context in which NCMC will be performed.

        titration_group_indices :
            The indices of the titratable groups that will be perturbed

        initial_titration_states :
            The initial protonation state of the titration groups

        final_titration_states :
            The final protonation state of the titration groups

        Returns
        -------
        work : float
          the protocol work of the NCMC procedure in multiples of kT.
        """
        # Turn the barostat and center of mass remover off, otherwise they contribute to the work
        if self.barostat is not None:
            self.barostat.setFrequency(0)
        if self.cm_remover is not None:
            self.cm_remover.setFrequency(0)

        #TODO: remove explicit calculation of work with `getEnergy`
        ghmc = self.compound_integrator.getIntegrator(1)
        ghmc.setGlobalVariableByName("ntrials", 0)  # Reset the internally accumulated work
        ghmc.setGlobalVariableByName("naccept", 0)  # Reset the GHMC acceptance rate counter

        # The "work" in the acceptance test has a contribution from the titratable group weights.
        g_initial = 0
        for titration_group_index, (titration_group, titration_state_index) in enumerate(zip(self.titrationGroups, self.titrationStates)):
            titration_state = titration_group['titration_states'][titration_state_index]
            g_initial += titration_state['g_k']

        # PROPAGATION
        ghmc.step(1)
        work = 0.0
        for step in range(self.nsteps_per_trial):

            # Get the fractional stage of the the protocol
            titration_lambda = float(step + 1) / float(self.nsteps_per_trial)

            # The slow way to calculate the work
            nrg_initial = context.getState(getEnergy=True).getPotentialEnergy()

            # PERTURBATION
            for titration_group_index in titration_group_indices:
                self._update_forces(titration_group_index, final_titration_states[titration_group_index],
                                    initial_titration_state_index=initial_titration_states[titration_group_index],
                                    fractional_titration_state=titration_lambda, context=context)
            for force_index, force in enumerate(self.forces_to_update):
                force.updateParametersInContext(context)
            self.titrationStates[titration_group_index] = titration_state_index

            # The slow way to calculate the work
            nrg_final = context.getState(getEnergy=True).getPotentialEnergy()
            work += (nrg_final - nrg_initial) * self.beta

            # PROPAGATION
            ghmc.step(1)

        # Extract the internally calculated work from the integrator
        #work = ghmc.getGlobalVariableByName('work') * self.beta_unitless

        # Setting the titratable group to the final state so that the appropriate weight can be extracted
        for titration_group_index in titration_group_indices:
            self.titrationStates[titration_group_index] = final_titration_states[titration_group_index]

        # Extracting the final states weigth.
        g_final = 0
        for titration_group_index, (titration_group, titration_state_index) in enumerate(zip(self.titrationGroups, self.titrationStates)):
            titration_state = titration_group['titration_states'][titration_state_index]
            g_final += titration_state['g_k']

        # Extract the internally calculated work from the integrator
        work += (g_final - g_initial)

        # Turn the barostat on again
        if self.barostat is not None:
            self.barostat.setFrequency(self.barofrequency)
        if self.cm_remover is not None:
            self.cm_remover.setFrequency(self.cm_remover_freq)

        return work


    def _attempt_state_change(self, context, reject_on_nan=False):
        """
        Attempt a single Monte Carlo protonation state change.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update

        reject_on_nan: bool, (default=False)
            Reject proposal if NaN. Not recommended since NaN typically indicates issues with the simulation.

        Notes
        -----
        The sams_sampler state actually present in the given context is not checked; it is assumed the ProtonDrive internal state is correct.

        """

        # If using NCMC, store initial positions.
        if self.nsteps_per_trial > 0:
            initial_positions = context.getState(getPositions=True).getPositions(asNumpy=True)

        # Compute initial probability of this protonation state. Used in the acceptance test for instantaneous
        # attempts, and to record potential and kinetic energy.
        log_P_initial, pot1, kin1 = self._compute_log_probability(context)

        log.debug("   initial %s   %12.3f kcal/mol" % (str(self._get_titration_states()), pot1 / units.kilocalories_per_mole))

        # Store current sams_sampler state indices.
        initial_titration_states = copy.deepcopy(self.titrationStates)  # deep copy

        # Select new sams_sampler states.
        final_titration_states = copy.deepcopy(self.titrationStates)  # deep copy
        # Choose how many titratable groups to simultaneously attempt to update.
        # TODO: Refine how we select residues and groups of residues to titrate to increase efficiency.
        ndraw = 1
        if (self._get_num_titratable_groups() > 1) and (random.random() < self.simultaneous_proposal_probability):
            ndraw = 2
        # Select which titratible residues to update.
        titration_group_indices = random.sample(range(self._get_num_titratable_groups()), ndraw)
        # Select new sams_sampler states.
        for titration_group_index in titration_group_indices:
            # Choose a sams_sampler state with uniform probability (even if it is the same as the current state).
            titration_state_index = random.choice(range(self._get_num_titration_states(titration_group_index)))
            final_titration_states[titration_group_index] = titration_state_index
        # TODO: Always accept self transitions, or avoid them altogether.

        if self.maintainChargeNeutrality:
            # TODO: Designate waters/ions to switch to maintain charge neutrality
            raise Exception('maintainChargeNeutrality feature not yet supported')

        try:
            # Compute work for switching to new protonation states.
            if self.nsteps_per_trial == 0:
                # Use instantaneous switching.
                for titration_group_index in titration_group_indices:
                    self._set_titration_state(titration_group_index, final_titration_states[titration_group_index], context)
                for force_index, force in enumerate(self.forces_to_update):
                    force.updateParametersInContext(context)
                log_P_final, pot2, kin2 = self._compute_log_probability(context)
                work = - (log_P_final - log_P_initial)
            else:
                # Run NCMC integration.
                work = self._ncmc_ghmc(context, titration_group_indices, initial_titration_states, final_titration_states)
                # Update titrationStates so that log state penalties are accurately reflected. This is automatically
                # done for instantaneous switches.
                for titration_group_index in titration_group_indices:
                    self.titrationStates[titration_group_index] = titration_state_index

                # Save the kinetic and potential energy
                log_P, pot2, kin2 = self._compute_log_probability(context)
                #work = - (log_P - log_P_initial)
            #TODO: remove when certain velocity Verlet won't be used as the NCMC propagator
            # BEGIN old code
            ## Compute final probability of this protonation state.
            #log_P_final, pot2, kin2 = self._compute_log_probability(context)
            ## Compute work and store work history.
            #work = - (log_P_final - log_P_initial)
            #log_P_accept = -work
            #log.debug("LOGP" + str(log_P_accept))
            #log.debug("   proposed log probability change: %f -> %f | work %f\n" % (log_P_initial, log_P_final, work))
            # End old code

            # Store work history and potential and kinetic energy.
            self.work_history.append((initial_titration_states, final_titration_states, work))
            log_P_accept = -work

            # Accept or reject with Metropolis criteria.
            self.nattempted += 1
            if (log_P_accept > 0.0) or (random.random() < math.exp(log_P_accept)):
                # Accept.
                self.naccepted += 1
                self.pot_energies.append(pot2)
                self.kin_energies.append(kin2)
                # Update sams_sampler states.
                for titration_group_index in titration_group_indices:
                    self._set_titration_state(titration_group_index, final_titration_states[titration_group_index], context)
                # If using NCMC, flip velocities to satisfy super-detailed balance.
                if self.nsteps_per_trial > 0:
                    context.setVelocities(-context.getState(getVelocities=True).getVelocities(asNumpy=True))
            else:
                # Reject.
                self.nrejected += 1
                self.pot_energies.append(pot1)
                self.kin_energies.append(kin1)
                # Restore sams_sampler states.
                for titration_group_index in titration_group_indices:
                    self._set_titration_state(titration_group_index, initial_titration_states[titration_group_index], context)
                # If using NCMC, restore coordinates and flip velocities.
                if self.nsteps_per_trial > 0:
                    context.setPositions(initial_positions)

        except Exception as err:
            if str(err) == 'Particle coordinate is nan' and reject_on_nan:
                logging.warning("NaN during NCMC move, rejecting")
                # Reject.
                self.nrejected += 1
                self.pot_energies.append(pot1)
                self.kin_energies.append(kin1)
                # Restore sams_sampler states.
                for titration_group_index in titration_group_indices:
                    self._set_titration_state(titration_group_index, initial_titration_states[titration_group_index], context)
                # If using NCMC, restore coordinates and flip velocities.
                if self.nsteps_per_trial > 0:
                    context.setPositions(initial_positions)
            else:
                raise
        finally:
            # Restore user integrator
            self.compound_integrator.setCurrentIntegrator(0)

        return

    def update(self, context, nattempts=None):
        """
        Perform a number of Monte Carlo update trials for the system protonation/tautomer states of multiple residues.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update

        nattempts: int, optional
            Number of individual attempts per update.

        Notes
        -----
        The sams_sampler state actually present in the given context is not checked; it is assumed the ProtonDrive internal state is correct.

        """

        if nattempts is None:
            # Use default (picked at initialization)
            nattempts = self.nattempts_per_update

        # Perform a number of protonation state update trials.
        for attempt in range(nattempts):
            self._attempt_state_change(context)
        self.states_per_update.append(self._get_titration_states())

        return

    def calibrate(self, platform_name=None, g_k=None, **kwargs):
        """
        Calibrate all amino acids that are found in the structure.

        Parameters
        ----------
        platform_name : str, optional, default=None
            Use specified platform, or if None, use fastest platform.
        g_k : dict, optional
            dict of starting value g_k estimates in numpy arrays, with residue names as keys.

        kwargs : optional keyword arguments are passed to underlying calibration engine.
            Expert users: see `calibration.AmberCalibrationSystem#sams_till_converged` for details.

        Returns
        -------
        g_k - dict with residue names as keys. For each residue, a numpy array with the relative free energies is returned.

        TODO
        ----
        * Treating ligands
        * Further document the procedure

        """
        from .calibration import AmberCalibrationSystem
        resname_per_index, unique_residuenames = self._detect_residues(AmberCalibrationSystem.supported_aminoacids)

        if g_k is None:
            g_k = {key: None for (key) in unique_residuenames}
        else:
            g_k = dict(g_k)  # deepcopy

        for resn in unique_residuenames:
            if resn not in g_k:
                g_k[resn] = None

        calibration_settings = dict()
        calibration_settings["temperature"] = self.temperature
        # index 0 Should be the user integrator
        calibration_settings["timestep"] = self.compound_integrator.getIntegrator(0).getStepSize()
        calibration_settings["pressure"] = self.pressure
        calibration_settings["pH"] = self.pH
        calibration_settings["solvent"] = self.solvent
        calibration_settings["nsteps_per_trial"] = self.nsteps_per_trial
        calibration_settings["platform_name"] = platform_name

        # Only calibrate once for each unique residue type
        for residuename in unique_residuenames:
            # This sets up a system for calibration, with a SAMS sampler under the hood.
            calibration_system = AmberCalibrationSystem(residuename, calibration_settings, guess_free_energy=g_k[residuename])
            gk_values = None
            # sams_till_converged is a generator.
            # gk_values will contain the latest estimate when the loop ends
            for gk_values in calibration_system.sams_till_converged(**kwargs):
                pass
            g_k[residuename] = gk_values

        # Set the g_k values of the MCTitration to the calibrated values.
        for group_index, group in enumerate(self.titrationGroups):
            for state_index, state in enumerate(self.titrationGroups[group_index]['titration_states']):
                self.titrationGroups[group_index]['titration_states'][state_index]['g_k'] = g_k[resname_per_index[group_index]][state_index]

        log.debug("Calibration results %s", g_k)
        return g_k

    def import_gk_values(self, gk_dict):
        """Import precalibrated gk values. Only use this if your simulation settings are exactly the same.

        If you changed any details, rerun calibrate instead!

        Parameters
        ----------
        gk_dict : dict
            dict of starting value g_k estimates in numpy arrays, with residue names as keys.

        """
        from .calibration import AmberCalibrationSystem
        resname_per_index, unique_residuenames = self._detect_residues(AmberCalibrationSystem.supported_aminoacids)

        # Set the g_k values to the user supplied values.
        for group_index, group in enumerate(self.titrationGroups):
            for state_index, state in enumerate(self.titrationGroups[group_index]['titration_states']):
                self.titrationGroups[group_index]['titration_states'][state_index]['g_k'] = \
                    gk_dict[resname_per_index[group_index]][state_index]

    def _detect_residues(self, supported_residues=None):
        """
        Detect the residues in the system that can be calibrated.

        Parameters
        ----------
        supported_residues : set
            set of residue names to flag as titratable.

        Returns
        -------
        dict of resnames with group index as keys, set of unique titratable residue names found
        """
        if supported_residues is None:
            from .calibration import AmberCalibrationSystem
            supported_residues = AmberCalibrationSystem.supported_aminoacids


        unique_residuenames = set()
        resname_per_index = dict()
        for group_index, group in enumerate(self.titrationGroups):
            supported = False
            group_name = group['name'].lower()
            for resn in supported_residues:
                if resn in group_name:
                    supported = True
                    unique_residuenames.add(resn)
                    resname_per_index[group_index] = resn
                    break
            if not supported:
                raise ValueError("Unsupported residue/ligand found in sams_sampler groups: {}".format(group_name))

        return resname_per_index, unique_residuenames

    def _get_acceptance_probability(self):
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

        # Add energetic contribution to log probability.
        state = context.getState(getEnergy=True)
        pot_energy = state.getPotentialEnergy()
        kin_energy = state.getKineticEnergy()
        total_energy = pot_energy + kin_energy
        log_P = - self.beta * total_energy

        if self.pressure is not None:
            # Add pressure contribution for periodic simulations.
            volume = context.getState().getPeriodicBoxVolume()
            log.debug('beta = %s, pressure = %s, volume = %s, multiple = %s', str(self.beta), str(self.pressure), str(volume), str(-self.beta * self.pressure * volume * units.AVOGADRO_CONSTANT_NA))
            log_P -= self.beta * self.pressure * volume * units.AVOGADRO_CONSTANT_NA

        # Add reference free energy contributions.
        for titration_group_index, (titration_group, titration_state_index) in enumerate(zip(self.titrationGroups, self.titrationStates)):
            titration_state = titration_group['titration_states'][titration_state_index]
            g_k = titration_state['g_k']
            log.debug("g_k: %.2f", g_k)
            log_P -= g_k

        # Return the log probability.
        return log_P, pot_energy, kin_energy

    def _get_num_attempts_per_update(self):
        """
        Get the number of Monte Carlo sams_sampler state change attempts per call to update().

        Returns
        -------

        nattempts_per_iteration : int
            the number of attempts to be made per iteration

        """
        return self.nattempts_per_update

    def _set_num_attempts_per_update(self, nattempts=None):
        """
        Set the number of Monte Carlo sams_sampler state change attempts per call to update().

        Parameters
        ----------

        nattempts : int
            the number to attempts to make per iteration;
            if None, this value is computed automatically based on the number of titratable groups (default None)

        """
        self.nattempts_per_update = nattempts
        if nattempts is None:
            # TODO: Perform enough sams_sampler attempts to ensure thorough mixing without taking too long per update.
            # TODO: Cache already-visited states to avoid recomputing?
            self.nattempts_per_update = self._get_num_titratable_groups()

    def _get_reduced_potentials(self, context, group_index=0):
        """Retrieve the reduced potentials for all states of the system given a context.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        group_index : int, optional
            Index of the group that needs updating, defaults to 0.

        """
        # beta * U(x)_j

        ub_j = np.empty(len(self.titrationGroups[group_index]['titration_states']))
        for j in range(ub_j.size):
            ub_j[j] = self._reduced_potential(context, j)

        # Reset to current state
        return ub_j

    def _reduced_potential(self, context, state_index):
        """Retrieve the reduced potential for a given state (specified by index) in the given context.

        Parameters
        ----------
        context : simtk.openmm.Context
            The context to update
        state_index : int
            Index of the state for which the reduced potential needs to be calculated.

        """
        potential_energy = self._get_potential_energy(context, state_index)
        red_pot = self.beta * potential_energy

        # TODO is the below necessary?
        if self.solvent == "explicit":
            volume = context.getState().getPeriodicBoxVolume()
            red_pot -= self.beta * self.pressure * volume * units.AVOGADRO_CONSTANT_NA

        return red_pot

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

        TODO
        ----
         * NCMC version of this?
        """
        current_state = self._get_titration_state(group_index)
        self._set_titration_state(group_index, state_index, context)
        temp_state = context.getState(getEnergy=True)
        potential_energy = temp_state.getPotentialEnergy()
        self._set_titration_state(group_index, current_state, context)
        return potential_energy


class AmberProtonDrive(_BaseProtonDrive):
    """
    The AmberProtonDrive is a Monte Carlo driver for protonation state changes and tautomerism in OpenMM.
    It relies on Ambertools to set up a simulation system, and requires a ``.cpin`` input file with protonation states.

    Protonation state changes, and additionally, tautomers are treated using the constant-pH dynamics method of Mongan, Case and McCammon [Mongan2004]_, or Stern [Stern2007]_ and NCMC methods from Nilmeier [Nilmeier2011]_.

    References
    ----------

    .. [Mongan2004] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born implicit solvent. J Comput Chem 25:2038, 2004.
        http://dx.doi.org/10.1002/jcc.20139

    .. [Stern2007] Stern HA. Molecular simulation with variable protonation states at constant pH. JCP 126:164112, 2007.
        http://link.aip.org/link/doi/10.1063/1.2731781

    .. [Nilmeier2011] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. PNAS 108:E1009, 2011.
        http://dx.doi.org/10.1073/pnas.1106094108

    .. todo::

      * Add NCMC switching moves to allow this scheme to be efficient in explicit solvent.
      * Add alternative proposal types, including schemes that avoid proposing self-transitions (or always accept them):
        - Parallel Monte Carlo schemes: Compute N proposals at once, and pick using Gibbs sampling or Metropolized Gibbs?
      * Allow specification of probabilities for selecting N residues to change protonation state at once.
      * Add calibrate() method to automagically adjust relative energies of protonation states of titratable groups in molecule.
      * Add automatic tuning of switching times for optimal acceptance.
      * Extend to handle systems set up via OpenMM app Forcefield class.
      * Make integrator optional if not using NCMC

    """

    def __init__(self, system, temperature, pH, prmtop, cpin_filename, integrator, pressure=None, nattempts_per_update=None, simultaneous_proposal_probability=0.1, debug=False,
                 ncmc_steps_per_trial=0, ncmc_prop_per_step=1, ncmc_timestep=1.0 * units.femtoseconds,
                 maintainChargeNeutrality=False, cationName='Na+', anionName='Cl-', implicit=False):
        """
        Initialize a Monte Carlo sams_sampler driver for simulation of protonation states and tautomers.

        Parameters
        ----------
        system : simtk.openmm.System
            System to be titrated, containing all possible protonation sites.
        temperature : simtk.unit.Quantity compatible with kelvin
            Temperature at which the system is to be simulated.
        pH : float
            The pH at which the system is to be simulated.
        prmtop : simtk.openmm.app.Prmtop
            Parsed AMBER 'prmtop' file (necessary to provide information on exclusions)
        cpin_filename : string
            AMBER 'cpin' file defining protonation charge states and energies of amino acids
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
        ncmc_steps_per_trial : int, optional, default=0
            Number of perturbation steps per NCMC switching trial, or 0 if instantaneous Monte Carlo is to be used.
        ncmc_prop_per_step: int, optional, default=1
            Number of propagation steps for each NCMC perturbation step, unused if ncmc_steps_per_trial = 0
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
        * Allow constant-pH dynamics to be initialized in other ways than using the AMBER cpin file (e.g. from OpenMM app; automatically).
        * Generalize simultaneous_proposal_probability to allow probability of single, double, triple, etc. proposals to be specified?

        """

        # Set defaults.
        # probability of proposing two simultaneous protonation state changes
        self.simultaneous_proposal_probability = simultaneous_proposal_probability

        # Store parameters.
        self.system = system
        self.temperature = temperature
        kT = kB * temperature  # thermal energy
        self.beta = 1.0 / kT  # inverse temperature
        self.beta_unitless = strip_in_unit_system(self.beta)   # For more efficient calculation of the work (in multiples of KT) during NCMC
        self.pressure = pressure
        self.pH = pH
        self.cpin_filename = cpin_filename
        self.debug = debug
        self.nsteps_per_trial = ncmc_steps_per_trial
        self.ncmc_prop_per_step = ncmc_prop_per_step
        if implicit:
            self.solvent = "implicit"
        else:
            self.solvent = "explicit"
        # Create a GHMC integrator to handle NCMC integration
        self.compound_integrator = openmm.CompoundIntegrator()
        self.compound_integrator.addIntegrator(integrator)

        self.ncmc_propagation_integrator = GHMCIntegrator(temperature, 1/units.picosecond, ncmc_timestep, nsteps=self.ncmc_prop_per_step)
        self.compound_integrator.addIntegrator(self.ncmc_propagation_integrator)
        self.compound_integrator.setCurrentIntegrator(0)  # make user integrator active
        # Set constraint tolerance.
        self.ncmc_propagation_integrator.setConstraintTolerance(integrator.getConstraintTolerance())

        # Record the forces that need to be swicthed off for NCMC
        forces = {system.getForce(index).__class__.__name__: system.getForce(index) for index in
                  range(system.getNumForces())}
        # Control center mass remover
        if 'CMMotionRemover' in forces:
            self.cm_remover = forces['CMMotionRemover']
            self.cm_remover_freq = self.cm_remover.getFrequency()
        else:
            self.cm_remover = None
            self.cm_remover_freq = None
        # Check that system has MonteCarloBarostat if pressure is specified
        if pressure is not None:
            if 'MonteCarloBarostat' not in forces:
                raise Exception("`pressure` is specified, but `system` object lacks a `MonteCarloBarostat`")
            else:
                self.barostat = forces['MonteCarloBarostat']
                self.barofreq = self.barostat.getFrequency()
        else:
            self.barostat = None
            self.barofreq = None


        # Store options for maintaining charge neutrality by converting waters to/from monovalent ions.
        self.maintainChargeNeutrality = maintainChargeNeutrality
        if not implicit:
            self.water_residues = self._identify_water_residues(prmtop.topology) # water molecules that can be converted to ions
            self.anion_parameters = self._retrieve_ion_parameters(prmtop.topology, system, anionName) # dict of ['charge', 'sigma', 'epsilon'] for cation parameters
            self.cation_parameters = self._retrieve_ion_parameters(prmtop.topology, system, cationName) # dict of ['charge', 'sigma', 'epsilon'] for anion parameters
            self.anion_residues = list() # water molecules that have been converted to anions
            self.cation_residues = list() # water molecules that have been converted to cations

        if implicit and maintainChargeNeutrality:
            raise ValueError("Implicit solvent and charge neutrality are mutually exclusive.")

        # Initialize sams_sampler group records.
        self.titrationGroups = list()
        self.titrationStates = list()

        # Keep track of forces and whether they're cached.
        self.precached_forces = False

        # Track simulation state
        self.kin_energies = units.Quantity(list(), units.kilocalorie_per_mole)
        self.pot_energies = units.Quantity(list(), units.kilocalorie_per_mole)
        self.states_per_update = list()

        # Determine 14 Coulomb and Lennard-Jones scaling from system.
        # TODO: Get this from prmtop file?
        self.coulomb14scale = self._get14scaling(system)

        # Store list of exceptions that may need to be modified.
        self.atomExceptions = [list() for index in range(prmtop._prmtop.getNumAtoms())]
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
            if type(namelist['RESSTATE']) == int:
                namelist['RESSTATE'] = [namelist['RESSTATE']]

            # Make sure RESNAME is a list.
            if type(namelist['RESNAME']) == str:
                namelist['RESNAME'] = [namelist['RESNAME']]

            # Extract number of titratable groups.
            self.ngroups = len(namelist['RESSTATE'])

            # Define titratable groups and sams_sampler states.
            for group_index in range(self.ngroups):
                # Extract information about this sams_sampler group.
                name = namelist['RESNAME'][group_index + 1]
                first_atom = namelist['STATEINF(%d)%%FIRST_ATOM' % group_index] - 1
                first_charge = namelist['STATEINF(%d)%%FIRST_CHARGE' % group_index]
                first_state = namelist['STATEINF(%d)%%FIRST_STATE' % group_index]
                num_atoms = namelist['STATEINF(%d)%%NUM_ATOMS' % group_index]
                num_states = namelist['STATEINF(%d)%%NUM_STATES' % group_index]

                # Define titratable group.
                atom_indices = range(first_atom, first_atom + num_atoms)
                self._add_titratable_group(atom_indices, name=name)

                # Define sams_sampler states.
                for titration_state in range(num_states):
                    # Extract charges for this sams_sampler state.
                    # is defined in elementary_charge units
                    charges = namelist['CHRGDAT'][(first_charge+num_atoms*titration_state):(first_charge+num_atoms*(titration_state+1))]

                    # Extract relative energy for this sams_sampler state.
                    # relative_energy = namelist['STATENE'][first_state + titration_state] * units.kilocalories_per_mole
                    relative_energy = 0.0 * units.kilocalories_per_mole
                    # Don't use pKref for AMBER cpin files---reference pKa contribution is already included in relative_energy.
                    pKref = 0.0
                    # Get proton count.
                    proton_count = namelist['PROTCNT'][first_state + titration_state]
                    # Create sams_sampler state.
                    self._add_titration_state(group_index, pKref, relative_energy, charges, proton_count)
                    self._cache_force(group_index, titration_state)
                # Set default state for this group.

                self._set_titration_state(group_index, namelist['RESSTATE'][group_index])

        self._set_num_attempts_per_update(nattempts_per_update)

        # Reset statistics.
        self.reset_statistics()

        return


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