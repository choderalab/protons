API documentation
*****************

ProtonDrives
============

.. py:currentmodule:: protons.app

The main engine of the protons code is the ``ProtonDrive``.


NCMCProtonDrive class
----------------------

.. autoclass:: protons.app.driver.NCMCProtonDrive

    :members:

    .. automethod:: NCMCProtonDrive.__init__
    .. automethod:: NCMCProtonDrive.attach_context
    .. automethod:: NCMCProtonDrive.attach_swapper
    .. automethod:: NCMCProtonDrive.update    
    .. automethod:: NCMCProtonDrive.define_pools
    .. automethod:: NCMCProtonDrive.import_gk_values
    .. automethod:: NCMCProtonDrive.adjust_to_ph

AmberProtonDrive class
----------------------

.. autoclass:: protons.app.driver.AmberProtonDrive

    :members:

    .. automethod:: AmberProtonDrive.__init__    

ForceFieldProtonDrive class
---------------------------

.. autoclass:: protons.app.driver.ForceFieldProtonDrive

    :members:
    .. automethod:: ForceFieldProtonDrive.__init__


Simulation runners
==================

To run a constant-pH simulation, some bookkeeping may need to be done.
The simulation context needs to be linked to the residues and states bookkeeping,
the frequency of updates needs to be set, data needs to be written out at regular intevals, et cetera.

To this end, two simulation runners are available in protons


ConstantPHSimulation
--------------------

.. autoclass:: protons.app.simulation.ConstantPHSimulation

    :members:

    .. automethod:: ConstantPHSimulation.__init__
    .. automethod:: ConstantPHSimulation.step
    .. automethod:: ConstantPHSimulation.updates    

ConstantPHCalibration
---------------------

.. autoclass:: protons.app.simulation.ConstantPHCalibration

   :members:

   .. automethod:: ConstantPHCalibration.__init__
   .. automethod:: ConstantPHCalibration.step
   .. automethod:: ConstantPHCalibration.update
   .. automethod:: ConstantPHCalibration.adapt

