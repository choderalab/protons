Introduction
============

This python module implements a constant-pH MD scheme for sampling protonation states
and tautomers of amino acids and small molecules in OpenMM. The code is currently under development.
Here are some of the planned features.



Planned features
----------------

* Simulation of proteins and small molecules in multiple protonation states, and tautomers.
* Support for implicit and explicit solvent models.
* Support for instantaneous Monte Carlo, and Non-equilibrium (NCMC) state switches.
* Compatibility with the Yank_ free energy calculation framework.


.. _Yank: http://getyank.org/latest/


Development notes
-----------------

This code is currently under heavy development. Here is a status update on some of the features we're working on.

Protein simulation
~~~~~~~~~~~~~~~~~~

Our code is capable of performing instantaneous state switching for implicit solvent simulations.
For the time being, we only support changing the protonation state of amino acids.
Small molecule support is a feature we plan to implement soon.
At the current time, we are ironing out potential bugs, and extending the test suite.

Small molecule support
~~~~~~~~~~~~~~~~~~~~~~

Small molecule support is a much anticipated feature.
For our first official release, we plan to support output from Epik calculations [5]_.
For parameter generation, we will rely on antechamber [6]_.


Explicit solvent
~~~~~~~~~~~~~~~~

Explicit solvent is supported by the code.
However, the code in the current form only features a working implementation of instantaneous Monte Carlo.
This will likely lead to low acceptance rates for protonation state switching.
We are in the process of developing an approach that uses NCMC, but the current implementation is not finished.




