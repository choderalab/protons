# -*- coding: utf-8 -*-
"""Constant-pH molecular dynamics for OpenMM.
"""

from __future__ import print_function, absolute_import

DOCLINES = __doc__.split("\n")

import os
import sys
from os.path import join as pjoin
from os.path import relpath
from setuptools import setup, Extension, find_packages
import versioneer

try:
    sys.dont_write_bytecode = True
    sys.path.insert(0, ".")
    from basesetup import write_version_py, CompilerDetection, check_dependencies
finally:
    sys.dont_write_bytecode = False


if "--debug" in sys.argv:
    sys.argv.remove("--debug")
    DEBUG = True
else:
    DEBUG = False


def find_package_data(data_root, package_root):
    files = []
    for root, dirnames, filenames in os.walk(data_root):
        for fn in filenames:
            files.append(relpath(pjoin(root, fn), package_root))
    return files


CLASSIFIERS = """\
Intended Audience :: Science/Research
Intended Audience :: Developers
Programming Language :: Python
Development Status :: 2 - Pre-Alpha
Topic :: Scientific/Engineering
Operating System :: POSIX
Operating System :: Unix
Operating System :: MacOS
Programming Language :: Python :: 3
Programming Language :: Python :: 3.6
"""

extensions = []

setup(
    name="protons",
    author="Bas Rustenburg",
    author_email="bas.rustenburg@choderalab.org",
    description=DOCLINES[0],
    long_description="\n".join(DOCLINES[2:]),
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    url="https://github.com/choderalab/protons",
    platforms=["any"],
    classifiers=CLASSIFIERS.splitlines(),
    packages=["protons", "protons.app", "protons.scripts", "protons.tests"],
    package_data={
        "protons": find_package_data("protons/examples", "protons")
        + find_package_data("protons/tests/testsystems", "protons")
        + find_package_data("protons/tests/cli-tests", "protons")
        + find_package_data("protons/app/data", "protons")
    },
    zip_safe=False,
    entry_points={
        "openmm.forcefielddir": "protons=protons.app.utils:get_datadir",
        "console_scripts": ["protons=protons.scripts.cli:cli"],
    },
    ext_modules=extensions,
)
