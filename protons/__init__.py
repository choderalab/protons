#!/usr/local/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function
import os

from .driver import AmberProtonDrive
from .logger import log

# Module constants
PACKAGE_ROOT = os.path.abspath(os.path.dirname(__file__))


def get_data(path,folder):
    # Function to obtain data that is part of the package, such as the calibration systems
    return os.path.join(PACKAGE_ROOT, folder, path)


