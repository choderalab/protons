Protons: Protonation states and tautomers for OpenMM
****************************************************

.. Note::

    This is module is undergoing heavy development. None of the API calls are final.

Introduction
============

This python module implements a constant-pH MD scheme for sampling protonation states and tautomers of amino acids and small molecules in OpenMM.


Installation
------------

Use the command

.. code-block:: bash

    python setup.py install

to install the package. The installation does not automatically check for requirements.

To test the installation, run

.. code-block:: bash

    nosetests protons


Requirements
~~~~~~~~~~~~

The ``protons`` package has the following requirements:

* python 2.7, 3.4 or 3.5
* openmm 7.0rc1
* numpy >=1.10
* scipy >=0.17.0
* openmmtools 0.7.5
* pymbar
* openmoltools 0.7.0
* ambermini
* joblib
* lxml


Table of contents
-----------------

.. toctree::
   :name: mastertoc
   :maxdepth: 2

   modules/protons
   modules/calibration
   modules/ligutils
   api


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`







