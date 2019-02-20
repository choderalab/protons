# coding=utf-8
"""
This script parametrizes a ligand.
"""

from sys import argv
from protons.app.ligands import *
from protons.app import logger
from protons.app.template_patches import patch_cooh
from protons.app.logger import log
from lxml import etree
import json
import sys
import os
from warnings import warn


log.setLevel(logger.logging.DEBUG)


def main(args):
    """
    Run the program
    Parameters
    ----------
    args - cmd line arguments

    """

    if len(args) != 2:
        print("Please provide a single json input file.")
        sys.exit(1)

    with open(args[1].strip(), "r") as settingsfile:
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

    # retrieve input fields
    idir = inp["dir"].format(**format_vars)
    istructure = inp["structure"].format(**format_vars)
    ical_path = os.path.abspath(os.path.join(idir, istructure))

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
        # mae file with epik results
        oepik = inp["epik"].format(**format_vars)
    else:
        oepik = f"{obase}-epik-out.mae"

    if not os.path.isdir(odir):
        os.makedirs(odir)

    lastdir = os.getcwd()

    # Begin processing
    # TODO copy files over to output dir?
    # run epik
    if run_epik:
        iepik = epik["input"].format(**format_vars)
        iepik_path = os.path.abspath(os.path.join(idir, iepik))
        if not os.path.isfile(iepik_path):
            raise IOError("Could not find epik input at {}.".format(**locals()))

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

    shutil.copyfile(oepik, os.path.join(odir, oepik))

    os.chdir(odir)

    # process into mol2
    epik_results_to_mol2(oepik, state_mol2)

    # Retrieve protonation state weights et cetera from epik output file
    isomer_info = retrieve_epik_info(oepik)

    # parametrize
    generate_protons_ffxml(state_mol2, isomer_info, offxml, pH, resname=resname)
    # create hydrogens
    create_hydrogen_definitions(offxml, ohxml)
    patch_cooh(offxml, resname)

    # set up calibration system
    extract_residue(ical_path, oextres, resname=resname)

    # prepare solvated system
    prepare_calibration_systems(oextres, obase, offxml, ohxml)

    os.chdir(lastdir)


if __name__ == "__main__":
    main(argv)
