Protons
=======

[![Build Status](https://travis-ci.org/choderalab/protons.svg?branch=master)](https://travis-ci.org/choderalab/protons)
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

`protons` will eventually be made conda installable. The list of dependencies can be found [here](devtools/conda-recipe/meta.yaml).  


## Contributors / coauthors ##
* Bas Rustenburg <bas.rustenburg@choderalab.org>
* Gregory Ross <gregory.ross@choderalab.org>
* John D. Chodera <choderaj@mskcc.org>
* Patrick Grinaway <grinawap@mskcc.org>
* Jason Swails <jason.swails@gmail.com>
* Jason Wagoner <jawagoner@gmail.com>
