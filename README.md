Protons
=======

[![Build Status](https://travis-ci.org/choderalab/protons.svg?branch=master)](https://travis-ci.org/choderalab/protons)
[![Code Issues](https://www.quantifiedcode.com/api/v1/project/b852925afabc4ff3aa41d94b4b2623dc/badge.svg)](https://www.quantifiedcode.com/app/project/b852925afabc4ff3aa41d94b4b2623dc)
[![Documentation Status](https://readthedocs.org/projects/protons/badge/?version=latest)](http://protons.readthedocs.io/en/latest/?badge=latest)


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
