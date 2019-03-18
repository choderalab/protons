# coding=utf-8
"""
This script parametrizes a ligand.
"""

from sys import argv
from ..app.ligands import *
from ..app import logger
from ..app.template_patches import patch_cooh
from ..app.logger import log
import json
import sys
import os
from warnings import warn

log.setLevel(logger.logging.INFO)


def run_parametrize_main(inputfile):
    """
    Run the program
    Parameters
    ----------
    args - cmd line arguments

    """

    with open(inputfile.strip(), "r") as settingsfile:
        settings = json.load(settingsfile)

    # Check all available fields.
    # TODO use json schema for this

    # Parameter settings, for file names et cetera
    try:
        prms = settings["parameters"]
    except KeyError:
        warn(
            "No parameters were provided. Will proceed, but please make sure your documents are named adequately.",
            UserWarning,
        )
        prms = dict()

    try:
        format_vars = prms["format_vars"]
    except KeyError:
        format_vars = dict()

    # Input files block
    try:
        inp = settings["input"]
    except KeyError:
        raise KeyError("Input block is missing from JSON")

    # Output files block
    try:
        # retrieve output field
        out = settings["output"]
    except KeyError:
        raise KeyError("Output block is missing from JSON.")

    # Epik data block
    try:
        run_epik = True
        epik = settings["epik"]
    except KeyError:
        run_epik = False
        try:
            epik = inp["epik"]
        except KeyError:
            raise KeyError("No epik block was found, and no epik data was provided.")

    # pH as input for certain programs, and to denote the compatibility of the force field settings for certain pH values.
    pH = float(prms["pH"])
    # 3 letter residue name
    resname = prms["pdb_resname"]

    if "omega_max_confs" in prms:
        max_confs = int(prms["omega_max_confs"])
    else:
        max_confs = 200

    # retrieve input fields
    idir = inp["dir"].format(**format_vars)
    if "structure" in inp:
        istructure = inp["structure"].format(**format_vars)
        ical_path = os.path.abspath(os.path.join(idir, istructure))
        create_systems = True
        if not os.path.isfile(ical_path):
            raise FileNotFoundError(f"Could not find the structure file: {ical_path}.")
    else:
        log.warn(
            "Warning 🛂: No calibration systems will be created for this system, since no structure was provided."
        )
        create_systems = False

    # Hydrogen definitions
    odir = out["dir"].format(**format_vars)
    obase = out["basename"].format(**format_vars)

    offxml = f"{obase}.xml"
    ohxml = f"{obase}-h.xml"

    # residue extracted from source
    oextres = f"{obase}-extracted.pdb"

    # mol2 file with unique and matching atom names
    state_mol2 = f"{obase}-states.mol2"

    if not run_epik:
        # Previously generated mae file with the output from epik
        oepik = epik["input"]["epik"].format(**format_vars)
    else:
        if "smiles" in epik["input"]:
            # Converts smiles to maestro file and uses that maestro file as input
            iepik = smiles_to_mae(epik["input"]["smiles"].format(**format_vars))
            try:
                shutil.copy(iepik, os.path.join(idir, iepik))
            except shutil.SameFileError:
                pass
        elif "mae" in epik["input"]:
            # Uses the user-specified maestro file
            iepik = epik["input"]["mae"].format(**format_vars)

        oepik = epik["output"]["mae"].format(**format_vars)

    if not os.path.isdir(odir):
        os.makedirs(odir)

    lastdir = os.getcwd()

    # Begin processing
    # TODO copy files over to output dir?
    # run epik
    if run_epik:
        log.info("⚗ Running Epik to generate protonation states.")
        iepik_path = os.path.abspath(os.path.join(idir, iepik))
        if not os.path.isfile(iepik_path):
            raise FileNotFoundError(
                "💥: Could not find epik input at {}.".format(**locals())
            )

        max_penalty = float(epik["parameters"]["max_penalty"])
        tautomerize = bool(epik["parameters"]["tautomerize"])
        generate_epik_states(
            iepik_path,
            oepik,
            pH=pH,
            max_penalty=max_penalty,
            workdir=odir,
            tautomerize=tautomerize,
        )

    os.chdir(odir)

    # process into mol2
    log.info("🛠 Processing epik results.")
    epik_results_to_mol2(oepik, state_mol2)

    # Retrieve protonation state weights et cetera from epik output file
    isomer_info = retrieve_epik_info(oepik)

    # parametrize
    log.info("🔬 Attempting to parameterize protonation states (takes a while).")
    if max_confs < 0:
        log.info(
            "☢ Warning: Dense conformer selection. Parameterization will take longer than usual."
        )
    generate_protons_ffxml(
        state_mol2, isomer_info, offxml, pH, resname=resname, omega_max_confs=max_confs
    )
    # create hydrogens
    log.info("🛠 Creating hydrogen definitions for ligand.")
    create_hydrogen_definitions(offxml, ohxml)
    log.info("💊 Adding residue patches for carboxylic acid sampling (if applicable).")
    patch_cooh(offxml, resname)

    # set up calibration system
    if create_systems:
        log.info(
            "🏊 Creating solvated systems for performing calibration (takes a while)."
        )
        extract_residue(ical_path, oextres, resname=resname)

        # prepare solvated system
        prepare_calibration_systems(oextres, obase, offxml, ohxml)

        os.chdir(lastdir)
    else:
        log.info("🚱 Solvated system generation skipped.")

    log.info(f"🖖 Script finished. Find your results in {odir}")


if __name__ == "__main__":

    if len(argv) != 2:
        print("Please provide a single json input file.")
        sys.exit(1)
    run_parametrize_main(argv[1])
