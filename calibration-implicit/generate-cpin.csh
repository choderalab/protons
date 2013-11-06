#!/bin/tcsh

# Generate .cpin files for terminally-blocked amino acids

foreach name ( asp glu his lys tyr )
     ambpdb -p ${name}.prmtop < ${name}.inpcrd | cpinutil.pl > ${name}.cpin
end
