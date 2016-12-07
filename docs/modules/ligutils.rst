The ligutils submodule
**********************

.. automodule:: protons.ligands

This submodule will contain the features that enable parametrizing ligands for constant-ph simulation.

Treating small molecules
========================

.. todo::
  * Small molecules are currently work in progress.
  * Will need to add support to determine populations and molecular data from other sources than Epik.
  * Adding openeye am1 bcc charge support as an optional feature for those that have openeye available

Several steps are needed in order to prepare a small molecule for simulation using the constant-pH code.

#. Protonation states and tautomers of the ligand need to be enumerated. This relies on Epik_ at the moment.

#. Molecule states need to be parametrized using antechamber.

#. All states need to be combined into a single residue template as an ffxml file.

#. Calibrate reference free energies of the molecule in solvent to return the expected state populations in the reference state.

This code provides an automated pipeline to sequentially perform all of these steps using this handy function!

.. autofunction:: protons.ligands.generate_protons_ffxml


.. _Epik:  Epik, version 3.3, Schrödinger, LLC, New York, NY, 2015.