## Preparation for constant pH simulation in explicit solvent

* Copied 2HYY-minimized.pdb from PDBFinder/pdbs/ 
* In 2HYY-minimized.pdb changed ASP-->AS4, GLU-->GL4, HIS-->HIP to create 2HYY-fixed.pdb
* Removed cap on N and C termi, HETATMS, hydrogens and CONECT records, but kept crystal waters
    * `awk '$3 !~ /^ *H/' 2HYY-fixed.pdb | grep -v HETATM | grep -v CONECT > 2HYY-clean.pdb`
    * `grep HOH 2HYY-fixed.pdb > xtal-waters.pdb`
    * Appended xtal-waters.pdb to 2HYY-clean.pdb (and taking care to ensure the END line doesn't occur before the waters
* Ran tleap: `tleap -f tleap_xtalwaters.in` to create `2HYY.inpcrd` and `2HYY.parmtop`
* Created a cpin: `cpinutil.py -p 2HYY.parmtop -o 2HYY.cpin`

The structure of the prepared system, before the simulation, is `leap_output.pdb`.