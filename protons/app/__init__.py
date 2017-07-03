#!/usr/local/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from simtk.openmm.app import *
from .topology import Topology
from .calibration import SelfAdjustedMixtureSampling
from .simulation import ConstantPHCalibration, ConstantPHSimulation
from .driver import ForceFieldProtonDrive, AmberProtonDrive
from .proposals import UniformProposal, DoubleProposal, CategoricalProposal
from .integrators import GBAOABIntegrator
from .modeller import Modeller
from .logger import log
from .record import netcdf_file

import os


def get_datadir():
    """Returns the data directory of this package"""
    return os.path.join(os.path.dirname(__file__), 'data')
