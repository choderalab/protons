#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""
Constant pH dynamics benchmark.
"""

import sys
import time
import progressbar
sys.path.append('../../../../../') # only works if run from one of the benchmarks/system-solvent/results/timestamp/settings_or_something_stupid
from constph import *

def main(niterations,nsteps,integrator,mc_titration,context,titration_benchmark):
    # Runs dynamics.

    with progressbar.ProgressBar(max_value=niterations,redirect_stdout=True) as bar:
        for iteration in range(niterations):
            # Run some dynamics.
            initial_time = time.time()
            integrator.step(nsteps)
            final_time = time.time()
            elapsed_time = final_time - initial_time
            titration_benchmark[iteration][0] = elapsed_time / nsteps

            # Attempt protonation state changes.
            initial_time = time.time()
            mc_titration.update(context)
            final_time = time.time()
            elapsed_time = final_time - initial_time
            ntrials = mc_titration.getNumAttemptsPerUpdate()
            titration_benchmark[iteration][1] = elapsed_time / ntrials
            state = context.getState(getEnergy=True)
            titration_benchmark[iteration][2] = state.getPotentialEnergy().value_in_unit(units.kilocalorie_per_mole)
            bar.update(iteration)


if __name__ == "__main__":
    import simtk.unit as units
    import simtk.openmm.app as app
    from sys import argv
    from cProfile import run
    from os import getcwd
    import logging
    from pprint import pformat

    logging.getLogger().setLevel(0)
    # Here we go!
    logging.info("Running benchmark in {}".format(getcwd()))

    # retrieve solvent from directory name
    if 'explicit' in getcwd():
        solvent='explicit'
    else:
        solvent='implicit'

    # Filenames.
    prmtop_filename = '../../../complex.prmtop'
    pdb_file= '../../../min.pdb'
    inpcrd_filename = '../../../complex.inpcrd'
    cpin_filename = '../../../complex.cpin'

     # Integrator settings
    temperature = 300.0 * units.kelvin
    timestep = 1.0 * units.femtoseconds
    # Tighten bond constraint tolerance
    constraint_tolerance = 1.0e-5 # can't go tighter than this or LocalEnergyMinimizer will fail.
    collision_rate = 9.1 / units.picoseconds

    # Constph settings

    pH = 7.0

    # specify these options on the command line
    niterations = int(argv[1]) # number of dynamics/titration cycles to run
    nsteps = int(argv[2]) # number of timesteps of dynamics per iteration [500 is a good setting for now]
    nsteps_per_trial = int(argv[3])  # Number of steps per NCMC switching trial, or 0 if instantaneous Monte Carlo is to be used.
    nattempts_per_update = int(argv[4]) # Number of protonation state change attempts per update call;


    if solvent == 'explicit':
        system_settings = dict(implicitSolvent=None, nonbondedMethod=app.CutoffPeriodic, constraints=app.HBonds)
        pressure = 1.0 * units.atmospheres # for explicit simjulations

    else:
        system_settings = dict(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
        pressure = None

    logging.info("Benchmark settings:")
    logging.info(pformat(dict(temperature=temperature, timestep=timestep, constraint_tolerance=constraint_tolerance, collision_rate=collision_rate, niterations=niterations,nsteps=nsteps,nsteps_per_trial=nsteps_per_trial,nattempts_per_update=nattempts_per_update,pH=pH,pressure=pressure, solvent=solvent)))

    titration_benchmark = np.empty([niterations, 3])

    # Load the AMBER system.
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    pdb = app.PDBFile(pdb_file)
    logging.info("Creating new system")
    system = prmtop.createSystem(**system_settings)

    # Simulate NPT
    if solvent == 'explicit':
        system.addForce(openmm.MonteCarloBarostat(pressure, temperature))
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    integrator.setConstraintTolerance(constraint_tolerance)
    mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, integrator, debug=False, pressure=pressure, nsteps_per_trial=nsteps_per_trial, nattempts_per_update=nattempts_per_update)

    # Create Context (using compound integrator from MonteCarloTitration).
    platform_name = 'CUDA'
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, mc_titration.compound_integrator, platform)
    context.setPositions(inpcrd.getPositions())

    # Minimize energy.
    print("Minimizing energy...")
    print("Initial energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())
    openmm.LocalEnergyMinimizer.minimize(context, 10.0, 10)
    print("Final energy is %s" % context.getState(getEnergy=True).getPotentialEnergy())

    print("Starting benchmark.")

    run('main(niterations,nsteps,integrator,mc_titration,context,titration_benchmark)', filename='benchmark.prof')
    np.savetxt("states.txt", mc_titration.states_per_update, delimiter=', ', header=', '.join([x['name'] for x in mc_titration.titrationGroups]))
    np.savetxt("benchmark.txt", titration_benchmark, delimiter=", ", header="Time per timestep (sec), Time per titration attempt (sec), Potential energy (kcal/mole)")

