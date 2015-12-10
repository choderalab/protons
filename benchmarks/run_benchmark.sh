#! /usr/bin/bash
source activate constph

for folder in dhfr-implicit abl-implicit dhfr-explicit abl-explicit ldha-implicit ldha-explicit
do
cd ${folder}
echo Running ${folder} benchmark!
python ../benchmarker.py
cd ..
done
