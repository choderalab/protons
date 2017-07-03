# coding=utf-8
"""Augmented OpenMM.app.ForceField."""

from protons import app
from simtk.openmm.app import forcefield
from lxml import etree
import os

forcefield._dataDirectories.append(os.path.join(os.path.dirname(app.__file__), 'data'))