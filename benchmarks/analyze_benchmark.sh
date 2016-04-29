#! /usr/bin/bash
source activate constph

for folder in histidine-implicit dhfr-implicit dhfr-explicit abl-implicit abl-explicit
do
cd ${folder}
echo Analyzing ${folder} benchmark!
#python ../summary.py
python ../analyze_timeseries.py
cd ..
done
