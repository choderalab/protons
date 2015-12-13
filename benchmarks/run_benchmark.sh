#! /usr/bin/bash
source activate constph

for folder in his-implicit dhfr-implicit abl-implicit ldha-implicit 
do
cd ${folder}
echo Running ${folder} benchmark!
python ../benchmarker.py
cd ..
done
