#! /usr/bin/bash
set -e # Exit on error
source activate constph

time_of_benchmark=$(date +"%m-%d-%Y-%T")
for folder in dhfr-explicit dhfr-implicit
do
cd ${folder}
mkdir -p results && cd $_
mkdir ${time_of_benchmark} && cd $_
echo Running ${folder} benchmarks, timestamp ${time_of_benchmark}!
niterations=5000 #md 
nsteps=500    #md steps / md step
for nsteps_p_trial in 0 1 10 100
do
  for nattempts_p_update in 1 10
  do
    mkdir ${niterations}_${nsteps}_${nsteps_p_trial}_${nattempts_p_update} && cd $_
    python ../../../../benchmarker.py ${niterations} ${nsteps} ${nsteps_p_trial} ${nattempts_p_update}
    cd ..
  done
done
cd ../../..
done
