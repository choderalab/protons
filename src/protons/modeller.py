# coding=utf-8
"""Pre-configured OpenMM Modeller object for use with the default protons forcefield."""

from simtk.openmm.app import Modeller

from protons import _xmlshortcuts

Modeller.loadHydrogenDefinitions(_xmlshortcuts.hydrogens)

