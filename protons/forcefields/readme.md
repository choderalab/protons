# Force field files for use with `protons`

This is a short guide, detailing how to compile ffxml files from the amber input files for constant-pH simulation using 
`protons`. All source files, and scripts generated to convert them can be found in the `Amber input files` directory. These were 
taken from the Ambertools16 distribution.

## Manifest

This procedure generates 5 files.

* `protons.xml`, which contains the amber ff10 forcefield with constant-pH residues 
* `hydrogens-protons.xml`, custom hydrogen definitions for protons residues

If you are including ions in your simulation, be sure to include one of these files, depending on your solvent model:

* `ions_spce.xml`, separate amber ion parameters for use with spce solvent model
* `ions_tip3p.xml`, separate amber ion parameters for use with tip3p solvent model
* `ions_tip4pew.xml`, separate amber ion parameters for use with tip4pew solvent model

Parameters and templates are included for Li+, Na+, K+, Rb+, Cs+, F-, Cl-, Br-, and I-. 

Also included are `gaff.xml` and `gaff2.xml`, necessary for running simulations using GAFF forcefields. These files
were obtained from openmoltools, and are included for convenience. They do not include residue templates.

## Compiling the files from scratch

Follow these steps to generate the xml files. The specific workings of the script are documented in the comments.

1. Run `1-compile_ffxml.py` inside of the `Amber input files` directory to generate raw xml files.
2. Then, run `2-process_raw_xml_file.py` to generate `protons.xml`, which contains `<Protons>` blocks for use in 
constant-pH simulation.
3. Create hydrogen definitions by running `3-create_hydrogen_definitions.py`.

### Ion files

To run simulations with ions, you'll need these. 

From within the `ions` subdirectory, follow these instructions:
 
1. Run `1-compile_ion_ffxml.py` to generate raw ffxml files. These include some residues with missing parameters,
which won't work with openmm. The next step fixes that.
2. Run `2-process_raw_ff10_ions.py` to remove residues without parameters. This should provide you with the three xml files
containing ion parameters and templates.