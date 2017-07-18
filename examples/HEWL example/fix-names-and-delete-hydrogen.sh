pdbfile="input.pdb"

# All cysteines in this file are involved in disulfide bonds
sed -i 's/CYS/CYX/g' ${pdbfile}

# Rename to constph residues
sed -i 's/HIE/HIP/g' ${pdbfile}
sed -i 's/HID/HIP/g' ${pdbfile}
sed -i 's/HIS/HIP/g' ${pdbfile}
sed -i 's/ASP/AS4/g' ${pdbfile}
sed -i 's/GLU/GL4/g' ${pdbfile}

# Remove hydrogens (print lines with atom names not starting with H)
awk '$3 !~ /^ *H/' ${pdbfile} > tmp && mv tmp ${pdbfile}
