#!/bin/bash

# Bash script to submit multiple proton simulations with different numbers of NCMC perturbations.

npert=(0 100 1000 5000 10000 20000 30000)

for k in ${npert[*]}
do
    mkdir "${k}_ncmc_steps"
    cd "${k}_ncmc_steps"
    sed "s/REPLACE/--ncmc_steps $k/" ../submit_dummy > submit
    qsub submit -N "${k}_ncmc"
    cd ../
done
