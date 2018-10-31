#!/usr/local/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

from simtk.openmm.app import *
from .topology import Topology
from .calibration import SelfAdjustedMixtureSampler,  UpdateRule, Stage
from .simulation import ConstantPHCalibration, ConstantPHSimulation
from .driver import ForceFieldProtonDrive, AmberProtonDrive, NCMCProtonDrive
from .proposals import UniformProposal, DoubleProposal, CategoricalProposal
from .integrators import GBAOABIntegrator
from .modeller import Modeller
from .logger import log
from .samsreporter import SAMSReporter
from .ncmcreporter import NCMCReporter
from .metadatareporter import MetadataReporter
from .titrationreporter import TitrationReporter
from simtk.unit import Quantity
import numpy as np

from simtk.openmm import State

def err_on_nan(func):
    """This decorator causes a RuntimeError when a function returns NaN."""
    def nan_wrapper(self):
        val = func(self)
        if isinstance(val, Quantity):
            if np.isnan(val._value):
                raise RuntimeError("NaN value returned by {}.".format(func.__name__))
            else:
                return val
        else:
            if np.isnan(val):
                raise RuntimeError("NaN value returned by {}.".format(func.__name__))
            else:
                return val
    return nan_wrapper

State.getPotentialEnergy = err_on_nan(State.getPotentialEnergy)
State.getKineticEnergy = err_on_nan(State.getKineticEnergy)