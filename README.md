openmm-constph
==============

[![Build Status](https://travis-ci.org/choderalab/openmm-constph.svg?branch=master)](https://travis-ci.org/choderalab/openmm-constph)
[![Code Health](https://landscape.io/github/choderalab/openmm-constph/master/landscape.svg?style=flat)](https://landscape.io/github/choderalab/openmm-constph/master)
[![Coverage Status](https://coveralls.io/repos/github/choderalab/openmm-constph/badge.svg?branch=master)](https://coveralls.io/github/choderalab/openmm-constph?branch=master)

Testbed for constant-pH methodologies using OpenMM.

## Manifest ##

`constph/`

```
   constph.py            - Python module implementing constant-pH methodologies in Python
   cnstphgbforces.py     - CustomGBForces that exclude contributions from discharged protons
   ligutils.py           - Work in progress code for ligand parametrization.
   tests/                - Unit tests
```

`constph/examples/`

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
