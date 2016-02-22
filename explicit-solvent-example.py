#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""
Constant pH dynamics test.

DESCRIPTION

This module tests the constant pH functionality in OpenMM.

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
from __future__ import print_function
from simtk import unit, openmm
import time

# ==============
# MAIN AND TESTS
# ==============

if __name__ == "__main__":
    import doctest
    doctest.testmod()

    #
    # Test with an example from the Amber 11 distribution.
    #

    # Parameters.
    niterations = 5000 # number of dynamics/titration cycles to run
    nsteps = 500  # number of timesteps of dynamics per iteration
    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    timestep = 1.0 * unit.femtoseconds
    collision_rate = 9.1 / unit.picoseconds
    platform_name = 'CPU'

    # Filenames.
    # prmtop_filename = 'amber-example/prmtop'
    # inpcrd_filename = 'amber-example/min.x'
    # cpin_filename = 'amber-example/cpin'
    # pH = 7.0

    solvent = 'implicit'

    if solvent == 'implicit':
        # Calibration on a terminally-blocked amino acid in implicit solvent
        prmtop_filename = 'calibration-implicit/tyr.prmtop'
        inpcrd_filename = 'calibration-implicit/tyr.inpcrd'
        cpin_filename =   'calibration-implicit/tyr.cpin'
        pH = 9.6
    elif solvent == 'explicit':
        # Calibration on a terminally-blocked amino acid in implicit solvent
        prmtop_filename = 'calibration-explicit/tyr.prmtop'
        inpcrd_filename = 'calibration-explicit/tyr.inpcrd'
        cpin_filename =   'calibration-explicit/tyr.cpin'
        pH = 9.6
    else:
        raise Exception("unknown solvent type '%s' (must be 'explicit' or 'implicit')" % solvent)

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

    print("Creating AMBER system...")
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    positions = inpcrd.getPositions()
    if solvent == 'implicit':
        system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        pressure = None
        running_implicit=True
    elif solvent == 'explicit':
        system = prmtop.createSystem(implicitSolvent=None, nonbondedMethod=app.CutoffPeriodic, constraints=app.HBonds)
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature))
        running_implicit=False

    # Minimize energy.
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    print("Minimizing energy...")
    print("Initial energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
    openmm.LocalEnergyMinimizer.minimize(context, 1.0, 1000)
    print("Final energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    del context, integrator

    # Create integrator.
    from openmmtools.integrators import VVVRIntegrator, GHMCIntegrator
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    #integrator = GHMCIntegrator(temperature, collision_rate, timestep)

    # Tighten bond constraint tolerance.
    CONSTRAINT_TOLERANCE = 1.0e-5 # can't go tighter than this or LocalEnergyMinimizer will fail.
    integrator.setConstraintTolerance(CONSTRAINT_TOLERANCE)

    # Initialize Monte Carlo titration.
    print("Initializing Monte Carlo titration...")
    from constph import MonteCarloTitration
    mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, integrator, debug=True, pressure=pressure, nsteps_per_trial=10, implicit=running_implicit)

    # Create Context (using compound integrator from MonteCarloTitration).
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, mc_titration.compound_integrator, platform)
    context.setPositions(positions)

    # Minimize energy.
    print("Minimizing energy...")
    print("Initial energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
    openmm.LocalEnergyMinimizer.minimize(context, 1.0, 1000)
    print("Final energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())

    # Run dynamics.
    state = context.getState(getEnergy=True)
    potential_energy = state.getPotentialEnergy()
    print("Initial protonation states: %s   %12.3f kcal/mol" % (str(mc_titration.getTitrationStates()), potential_energy / unit.kilocalories_per_mole))
    for iteration in range(niterations):
        # Run some dynamics.
        initial_time = time.time()
        integrator.step(nsteps)
        state = context.getState(getEnergy=True)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        print("  %.3f s elapsed for %d steps of dynamics" % (elapsed_time, nsteps))

        # Attempt protonation state changes.
        initial_time = time.time()
        mc_titration.update(context)
        state = context.getState(getEnergy=True)
        final_time = time.time()
        elapsed_time = final_time - initial_time
        print("  %.3f s elapsed for %d titration trials" % (elapsed_time, mc_titration.getNumAttemptsPerUpdate()))
        # Show titration states.
        state = context.getState(getEnergy=True)
        potential_energy = state.getPotentialEnergy()
        print("Iteration %5d / %5d:    %s   %12.3f kcal/mol (%d / %d accepted)" % (
        iteration, niterations, str(mc_titration.getTitrationStates()), potential_energy / unit.kilocalories_per_mole,
        mc_titration.naccepted, mc_titration.nattempted))
