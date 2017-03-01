Protons
=======

[![Build Status](https://travis-ci.org/choderalab/protons.svg?branch=master)](https://travis-ci.org/choderalab/protons)
[![Code Issues](https://www.quantifiedcode.com/api/v1/project/b852925afabc4ff3aa41d94b4b2623dc/badge.svg)](https://www.quantifiedcode.com/app/project/b852925afabc4ff3aa41d94b4b2623dc)
[![Documentation Status](https://readthedocs.org/projects/protons/badge/?version=latest)](http://protons.readthedocs.io/en/latest/?badge=latest)
[![codecov](https://codecov.io/gh/choderalab/protons/branch/master/graph/badge.svg)](https://codecov.io/gh/choderalab/protons)


Testbed for constant-pH methodologies using OpenMM.

## Manifest ##

`protons/`  - Python module implementing constant-pH methodologies in Python

```
   calibration.py        - Calibration engines
   cnstphgbforces.py     - CustomGBForces that exclude contributions from discharged protons
   ligutils.py           - Work in progress code for ligand parametrization.
   tests/                - Unit tests
```

`protons/examples/`

```
   explicit-solvent-example.py - explicit solvent NCMC example
   amber-example/        - example system set up with AmberTools constant-pH tools
   calibration-implicit/ - terminally-blocked amino acids parameterized for implicit solvent relative free energy calculations
   calibration-explicit/ - terminally-blocked amino acids parameterized for explicit solvent relative free energy calculations
```

`references/`           - some relevant literature references

## Dependencies ##

### Runtime
    - python
    - openmm >=7.1
    - numpy >=1.10
    - scipy >=0.17.0
    - netcdf4 >=1.2.4
    - hdf4 >4.2.11
    - openmoltools >=0.7.0
    - ambermini >=15.0.4
    - joblib
    - lxml
    - parmed

### Testing  
    - pytest
    - pytest-cov
    - behave


## Contributors / coauthors ##
* Bas Rustenburg <bas.rustenburg@choderalab.org>
* Gregory Ross <gregory.ross@choderalab.org>
* John D. Chodera <choderaj@mskcc.org>
* Patrick Grinaway <grinawap@mskcc.org>
* Jason Swails <jason.swails@gmail.com>
* Jason Wagoner <jawagoner@gmail.com>
