Running a Constant-pH simulation
================================
.. automodule:: constph.constph

The :py:mod:`constph.constph` module contains the main functionality of the package.
In order to run a simulation containing multiple protonation states and tautomers, you can use :py:class:`constph.constph.TitrationDriver`.
This driver object contains a dictionary of all possible protonation states and tautomers of each residue in the simulation system, in terms of their parameters.
When simulating, it is responsible for keeping track of the current system state, and updates the OpenMM context with the correct parameters.
It uses an instantaneous Monte Carlo sampling method [1]_ to update implicit solvent systems.
Explicit solvent systems are treated using NCMC [2]_, [3]_.


.. autoclass:: constph.constph.TitrationDriver

    :members:

    .. automethod:: constph.constph.TitrationDriver.__init__
    .. automethod:: constph.constph.TitrationDriver.update
    .. automethod:: constph.constph.TitrationDriver.calibrate
    .. automethod:: constph.constph.TitrationDriver.import_gk_values


Solvent reference state calibrations
====================================

In order to simulate a protein or small molecule with multiple protonation states and/or tautomers, it is necessary to calculate the free energy difference between a reference state.
This free energy can depend on the temperature, solvent model, pressure, pH and other factors. It is therefore imperative that this is run whenever you've changed simulation settings.

This state is taken to be the state of a single capped amino acids in solvents, or in the case of small molecules a single molecule.
We use self-adjusted mixture sampling (SAMS) to perform a free energy calculation of all single residues in solvent.
As part of the package, we provide a set of minimized amino acids in solvent that can be treated using constant-pH methods.


Residues
--------

The package supports the following residues by default, denoted by the residue name with the max number of protons added.
Refence pKa values

* Glutamic acid, ``GL4`` (pKa=4.4)
* Aspartic acid, ``AS4`` (pKa=4.0)
* Histidine, ``HIP``  (pKa delta=6.5, pKa epsilon = 7.1)
* Tyrosine, ``TYR`` (pKa=9.6)
* Cysteine, ``CYS`` (pKa=8.5)
* Lysine, ``LYS`` (pKa=10.4)

To automatically calibrate all amino acids available in a system, one can use the calibrate method.

.. automethod::  constph.constph.TitrationDriver.calibrate

Advanced settings
-----------------
This will start a series of SAMS free energy calculations for each unique amino acid found in the system.
The :py:class:`TitrationDriver` object will be updated automatically. The function will also return the calibrated gk values.
You can store these and load them in later using :py:meth:`TitrationDriver.import_gk_values`.

.. automethod:: constph.constph.TitrationDriver.import_gk_values




Basic example
=============

Below is a basic example of how to run a simulation using the TitrationDriver without using the calibration API.

.. code-block:: python

  """Run the complete API using precalibrated reference states"""
    from __future__ import print_function
    from simtk import unit, openmm
    from simtk.openmm import app
    from constph import get_data
    from constph.logger import logger
    from constph.constph import MonteCarloTitration
    import numpy as np
    import openmmtools
    import logging
    from sys import stdout


    # Import one of the standard systems.
    temperature = 300.0 * unit.kelvin
    timestep = 1.0 * unit.femtoseconds
    pH = 7.4

    platform = openmm.Platform.getPlatformByName('CUDA')

    prmtop = app.AmberPrmtopFile('complex.prmtop')
    inpcrd = app.AmberInpcrdFile('complex.inpcrd')
    positions = inpcrd.getPositions()
    topology = prmtop.topology
    cpin_filename = 'complex.cpin'
    integrator = openmmtools.integrators.VelocityVerletIntegrator(timestep)

    # Create a system from the AMBER prmtop file
    system = prmtop.createSystem(implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds)
    # Create the driver that will track the state of the simulation and provides the updating API
    driver = TitrationDriver(system, temperature, pH, prmtop, cpin_filename, integrator, pressure=None, ncmc_steps_per_trial=0, implicit=True)

    # Create an OpenMM simulation object as one normally would.
    simulation = app.Simulation(topology, system, integrator, platform)
    simulation.context.setPositions(positions)
    simulation.context.setVelocitiesToTemperature(temperature)

    # pre-equilibrated values.
    # temperature = 300.0 * unit.kelvin
    # pressure = None
    # timestep = 1.0 * unit.femtoseconds
    # pH = 7.4
    # Amber 99 constant ph residues, converged to threshold of 1.e-7

    calibration_results = {'as4': np.array([3.98027947e-04,  -3.61785292e+01,  -3.98046143e+01,
                                            -3.61467735e+01,  -3.97845096e+01]),
                           'cys': np.array([7.64357397e-02,   1.30386793e+02]),
                           'gl4': np.array([9.99500333e-04,  -5.88268681e+00,  -8.98650420e+00,
                                            -5.87149375e+00,  -8.94086390e+00]),
                           'hip': np.array([2.39229276,   5.38886021,  13.12895206]),
                           'lys': np.array([9.99500333e-04,  -1.70930870e+01]),
                           'tyr': np.array([6.28975142e-03,   1.12467299e+02])}

    mc_titration.import_gk_values(calibration_results)

    # 60 ns, 10000 state updates
    niter, mc_freq = 10000, 6000
    simulation.reporters.append(app.DCDReporter('trajectory.dcd', mc_freq))

    for iteration in range(1, niter):
        simulation.step(mc_freq) # MD
        mc_titration.update(simulation.context)  # protonation



References
----------

.. [1] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born implicit solvent. J Comput Chem 25:2038, 2004.
  http://dx.doi.org/10.1002/jcc.20139

.. [2] Stern HA. Molecular simulation with variable protonation states at constant pH. JCP 126:164112, 2007.
  http://link.aip.org/link/doi/10.1063/1.2731781

.. [3] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. PNAS 108:E1009, 2011.
  http://dx.doi.org/10.1073/pnas.1106094108
