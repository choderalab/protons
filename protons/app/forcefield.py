# coding=utf-8
"""Augmented OpenMM.app.ForceField."""

from protons import app
from simtk.openmm.app.forcefield import _dataDirectories
from lxml import etree
import os

_dataDirectories.append(os.path.join(os.path.dirname(app.__file__), 'data'))