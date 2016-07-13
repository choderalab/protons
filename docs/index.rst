.. Constant-pH for OpenMM documentation master file, created by
   sphinx-quickstart on Wed Jul 13 14:24:11 2016.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to Constant-pH for OpenMM's documentation!
==================================================

Constant pH dynamics in OpenMM.

Description
-----------

This module implements a pure python constant pH functionality in OpenMM.

Notes
-----

This is still in development.

References
----------

.. [1] Mongan J, Case DA, and McCammon JA. Constant pH molecular dynamics in generalized Born implicit solvent. J Comput Chem 25:2038, 2004.
  http://dx.doi.org/10.1002/jcc.20139

.. [2] Stern HA. Molecular simulation with variable protonation states at constant pH. JCP 126:164112, 2007.
  http://link.aip.org/link/doi/10.1063/1.2731781

.. [3] Nonequilibrium candidate Monte Carlo is an efficient tool for equilibrium simulation. PNAS 108:E1009, 2011.
  http://dx.doi.org/10.1073/pnas.1106094108

Examples
--------

Coming soon to an interpreter near you!

TODO
----

    * Add NCMC switching moves to allow this scheme to be efficient in explicit solvent.
    * Add alternative proposal types, including schemes that avoid proposing self-transitions (or always accept them):
      - Parallel Monte Carlo schemes: Compute N proposals at once, and pick using Gibbs sampling or Metropolized Gibbs?
    * Allow specification of probabilities for selecting N residues to change protonation state at once.
    * Add calibrate() method to automagically adjust relative energies of protonation states of titratable groups in molecule.
    * Add automatic tuning of switching times for optimal acceptance.
    * Extend to handle systems set up via OpenMM app Forcefield class.

Contents:

.. toctree::
   :maxdepth: 2


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`


Classes
=======

.. autoclass:: constph.constph.MonteCarloTitration

    :members:

    .. automethod:: constph.constph.MonteCarloTitration.__init__







