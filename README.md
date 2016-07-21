Protons
=======

[![Build Status](https://travis-ci.org/choderalab/constph-openmm.svg?branch=master)](https://travis-ci.org/choderalab/constph-openmm)
[![Code Health](https://landscape.io/github/choderalab/constph-openmm/master/landscape.svg?style=flat)](https://landscape.io/github/choderalab/constph-openmm/master)
[![Coverage Status](https://coveralls.io/repos/github/choderalab/openmm-constph/badge.svg?branch=master)](https://coveralls.io/github/choderalab/openmm-constph?branch=master)
[![Documentation Status](https://readthedocs.org/projects/constph-openmm/badge/?version=latest)](http://constph-openmm.readthedocs.io/en/latest/?badge=latest)

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


## Contributors / coauthors ##

* John D. Chodera <choderaj@mskcc.org>
* Patrick Grinaway <grinawap@mskcc.org>
* Jason Swails <jason.swails@gmail.com>
* Jason Wagoner <jawagoner@gmail.com>
