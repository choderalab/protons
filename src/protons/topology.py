# coding=utf-8
"""Pre-configured OpenMM Topology object for use with the default protons forcefield."""

from simtk.openmm.app import Topology

from protons import _xmlshortcuts

Topology.loadBondDefinitions(_xmlshortcuts.bonds)

