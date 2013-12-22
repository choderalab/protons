#!/bin/tcsh

# Generate .cpin files for terminally-blocked amino acids

foreach name ( asp glu his lys tyr cys )
#     ambpdb -p ${name}.prmtop < ${name}.inpcrd | cpinutil.pl > ${name}.cpin
    cpinutil.py -p ${name}.prmtop -o ${name}.cpin
end
