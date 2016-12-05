# Initial application of `protons` to explicit water
**UNDER DEVELOPMENT**

This folder contains initial results of `protons` applied to constant-pH
simulations of amino acids in explicit water. Long range electrostatics are handled with
PME, and changes in the charge of thet system are handled automatically
with a uniform neutralizing field.

## Manifest

### Calibration of SAMS weights

```
run_calibration.py              Command line tool to calibrate amino acids
```

```
AminoAcid_calibration.py        Jupyter notebook that looks at the convergence and state weights for the latest runs
```

```
as4/, cys/, gl4/, hip/, lys/, tyr/     The folders containing the results from various repeats from run_calibration.py 
```

The latest SAMS weights to achieve uniform sampling between the states are,
in order:

|  Residue | Weight 1 | Weight 2  | Weight 3  | Weight 4  |  Weight 5 |
|:-:|---|---|---|---|---|
| AS4 | 0.0 | -63.2 | -65.1 | -63.1 | -69.5 |
| GL4 | 0.0 | -33.8 | -39.7 | -36.1 | -38.5 |
| HIP | 0.0 | 27.5 |29.6  | - | - |
| CYS | 0.0 | 154.4 | - | - | - |
| LYS | 0.0 | -6.8 | - | - | - |
| TYR | 0.0 | 126.7 | - | - | - |

The pH dependent state penalties have to deducted from the above weights to have
the final set of weights for a simulation. See `AminoAcid_calibration.py`.

```
previous_AminoAcid_calibration.py  Jupyter notebook that looks at the convergence and state weights for the latest runs
```

### NCMC acceptance rate study
`NCMC_Sweep/`
Different NCMC protocol lengths were applied to histidine and the effect on
acceptance rates was recorded.
```
Analysis.ipynb              The notebook summarizing the results of the NCMC study
```

```
Setup_main.sh                An example of a script to submit many NCMC jobs
```

```
submit_dummy                 A template submission script that is called by Setup_main.sh
```


