# coding=utf-8
"""Pre-configured OpenMM Topology object for use with the default protons forcefield."""

from simtk.openmm.app import Topology
import os
PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))


# Patch topology to unload standard bond definitions
def unloadStandardBonds(cls):
    """
    Resets _standardBonds and _hasLoadedStandardBonds to original state.
    """

    cls._hasLoadedStandardBonds = False
    cls._standardBonds = dict()

Topology.unloadStandardBonds = classmethod(unloadStandardBonds)
Topology.unloadStandardBonds()
Topology.loadBondDefinitions(os.path.join(PACKAGE_ROOT, 'data', 'bonds-amber10-constph.xml'))


