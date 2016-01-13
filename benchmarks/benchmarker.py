#!/usr/local/bin/env python
# -*- coding: utf-8 -*-


"""
Constant pH dynamics benchmark.
"""

import sys
import time
import progressbar
sys.path.append('../../') # only works if run from one of the benchmark system directories
from constph import *


def main(niterations,nsteps,integrator,mc_titration,titration_benchmark):
    # Runs dynamics.

    with progressbar.ProgressBar(max_value=niterations,redirect_stdout=True) as bar:
        for iteration in range(niterations):
            # Run some dynamics.
            initial_time = time.time()
            integrator.step(nsteps)
            final_time = time.time()
            elapsed_time = final_time - initial_time
            titration_benchmark[iteration][0] = elapsed_time / nsteps
            # print("%.3f s elapsed for %d steps of dynamics (%s)" % (elapsed_time, nsteps, nsteps * timestep))

            # Attempt protonation state changes.
            initial_time = time.time()
            mc_titration.update(context)
            final_time = time.time()
            elapsed_time = final_time - initial_time
            ntrials = mc_titration.getNumAttemptsPerUpdate()
            titration_benchmark[iteration][1] = elapsed_time / ntrials
            bar.update(iteration)
            # print("  %.3f s elapsed for %d titration trials" % (elapsed_time, ntrials))


if __name__ == "__main__":
    import doctest
    import simtk.unit as units
    from cProfile import run
    doctest.testmod()
    from os import getcwd
    niterations = 1000 # number of dynamics/titration cycles to run
    nsteps = 500  # number of timesteps of dynamics per iteration
    temperature = 300.0 * units.kelvin
    timestep = 1.0 * units.femtoseconds
    collision_rate = 9.1 / units.picoseconds

    # Filenames.
    prmtop_filename = 'complex.prmtop'
    pdb_file= 'min.pdb'
    inpcrd_filename = 'complex.inpcrd'
    cpin_filename = 'complex.cpin'
    pH = 7.0
    titration_benchmark = np.empty([niterations, 2])

    # Load the AMBER system.
    import simtk.openmm.app as app
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    pdb = app.PDBFile(pdb_file)

    if getcwd().split(sep='-')[-1] == 'explicit':
        print("explicit system, using PME")
        system = prmtop.createSystem(nonbondedMethod=app.PME, nonbondedCutoff=1 * units.nanometer,
                                     constraints=app.HBonds)
    else:
        print("implicit system, using NoCutoff")
        system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    mc_titration = MonteCarloTitration(system, temperature, pH, prmtop, cpin_filename, debug=False)
    platform_name = 'CUDA'
    platform = openmm.Platform.getPlatformByName(platform_name)
    integrator = openmm.LangevinIntegrator(temperature, collision_rate, timestep)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(pdb.getPositions())

    print("Starting benchmark.")
    run('main(niterations,nsteps,integrator,mc_titration,titration_benchmark)', filename='benchmark.prof')
    np.savetxt("states.txt", mc_titration.states_per_update, delimiter=', ', header=', '.join([x['name'] for x in mc_titration.titrationGroups]))
    np.savetxt("benchmark.txt", titration_benchmark, delimiter=", ", header="Time per timestep (sec), Time per titration attempt (sec)")
    np.savetxt("pot_energies.txt", mc_titration.pot_energies.value_in_unit(units.kilocalorie_per_mole), header="Potential energy (kcal/mole)")
