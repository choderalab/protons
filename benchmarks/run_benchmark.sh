#! /usr/bin/bash
source activate constph

for folder in histidine-implicit dhfr-implicit dhfr-explicit abl-implicit abl-explicit
do
cd ${folder}
echo Running ${folder} benchmark!
python ../benchmarker.py
cd ..
done
