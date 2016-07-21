#!/bin/tcsh

# Generate .cpin files for terminally-blocked amino acids
rm -f leap.log *.prmtop *.inpcrd
tleap -f generate-solvated-amino-acids.leap.in > leap.out

foreach name ( asp glu his lys tyr cys )
#     ambpdb -p ${name}.prmtop < ${name}.inpcrd | cpinutil.pl > ${name}.cpin
    python ../cpinutil.py -p ${name}.prmtop -o ${name}.cpin -op ${name}.new.prmtop
end
