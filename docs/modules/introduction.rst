Introduction
============

Protons is a python module that implements a constant-pH molecular dynamics scheme for sampling protonation states
and tautomers of amino acids and small molecules in OpenMM.
The codebase is currently under development by the `Chodera lab`_ at Memorial Sloan Kettering Cancer center.
There is no official release of the code yet.

If you are interested in the development of this code, you can watch the `repository on Github`_.
Here are some of the planned features of the code

.. _Chodera lab: http://www.choderalab.org
.. _repository on Github: https://github.com/choderalab/openmm-constph

Planned features
----------------

* Simulation of proteins and small molecules in multiple protonation states, and tautomers.
* Support for implicit and explicit solvent models.
* Support for instantaneous Monte Carlo, and Non-equilibrium (NCMC) state switches.
* Compatibility with the Yank_ free energy calculation framework.


.. _Yank: http://getyank.org/latest/


Development notes
-----------------

This code is currently under heavy development.
At the current time, we are ironing out potential bugs, and extending the test suite.
Here is a status update on some of the features we're working on.

Protein simulation
~~~~~~~~~~~~~~~~~~

Our code is capable of performing instantaneous state switching for implicit solvent simulations.
At the moment, we only support updating the protonation states of the sidechains if the following amino acids:

* Glutamic acid, (pKa=4.4)
* Aspartic acid, (pKa=4.0)
* Histidine, (pKa delta=6.5, pKa epsilon = 7.1)
* Tyrosine, (pKa=9.6)
* Cysteine, (pKa=8.5)
* Lysine, (pKa=10.4)

The pKa values used originate from [Mongan2004]_.


Small molecule support
~~~~~~~~~~~~~~~~~~~~~~
Small molecule support is is a much anticipated feature that we plan to implement soon.
The code will be extended to use output from Epik calculations ([Shelley2007]_, [Greenwood2010]_, [Epik2016]_) to provide us with
the populations of different protonation states and tautomers in aqueous solvent conditions.
For parameter generation, we will rely on the GAFF forcefield [Wang2004]_, as provided with the antechamber program [Amber2016]_.


Explicit solvent
~~~~~~~~~~~~~~~~

Explicit solvent is supported by the code, but it is not feature complete.
The current version of the software only features a working implementation of instantaneous Monte Carlo.
This will likely lead to low acceptance rates for protonation state switching.
We are in the process of developing an approach that uses NCMC [Nilmeier2011]_, [Chen2015]_ but the implementation is not finished.


