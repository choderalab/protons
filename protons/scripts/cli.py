"""This library contains a minimal command line interface to enable tracking the common use cases.

This should be more maintainable than having separate python scripts inside the examples directory.
"""


import os
import sys

from ..app.logger import log, logging

from typing import List
from .run_simulation import run_main
from .run_prep_ffxml import run_prep_ffxml_main
from .run_parametrize_ligand import run_parametrize_main

log.setLevel(logging.DEBUG)


# Define a main function that can read in a json file with simulation settings, sets up, and runs the simulation.


_logo = """
    ██████╗ ██████╗  ██████╗ ████████╗ ██████╗ ███╗   ██╗███████╗
    ██╔══██╗██╔══██╗██╔═══██╗╚══██╔══╝██╔═══██╗████╗  ██║██╔════╝
    ██████╔╝██████╔╝██║   ██║   ██║   ██║   ██║██╔██╗ ██║███████╗
    ██╔═══╝ ██╔══██╗██║   ██║   ██║   ██║   ██║██║╚██╗██║╚════██║
    ██║     ██║  ██║╚██████╔╝   ██║   ╚██████╔╝██║ ╚████║███████║
    ╚═╝     ╚═╝  ╚═╝ ╚═════╝    ╚═╝    ╚═════╝ ╚═╝  ╚═══╝╚══════╝
"""


def validate(args: List[str]) -> str:
    """Validate input or return appropriate help string for the given arguments"""

    usage = (
        _logo
        + """
   
    Protons minimal command line interface.

    Usage
    -----

    protons param <json>
        Parameterize a ligand based on a user provided json input file.
    protons prep <json>
        Produce an input file for a constant-pH simulation or calibration by specifying simulation settings in a json file.
        Note: Currently only ffxml supported.
    protons run <json>
        Run a constant-pH simulation or calibration from a json file.
    
    protons help <cmd>
        Provide a longer explanation for a command, including example json files.

    
    Copyright
    ---------
    The John D. Chodera lab, 2019
    
    See more at https://protons.readthedocs.io/en/latest/, 
    or check out our website: https://www.choderalab.org
    """
    )

    if len(args) != 3:
        return usage

    cmd = args[1].lower()
    arg = args[2]

    if cmd == "help":
        if arg.lower() == "param":
            raise NotImplementedError("This help feature is incomplete.")
        elif arg.lower() == "prep":
            raise NotImplementedError("This help feature is incomplete.")
        elif arg.lower() == "run":
            raise NotImplementedError("This help feature is incomplete.")
        else:
            return f"Unknown command: {arg}. \n" + usage

    elif cmd in ["param", "prep", "run"]:
        if os.path.splitext(arg)[1].lower() != ".json":
            return "Please provide a '.json' file as input."
        else:
            return ""


def cli() -> None:
    """Command line interface for protons run script."""

    args = sys.argv

    result = validate(args)

    if result:
        sys.stderr.write(result)
        sys.exit(1)
    else:
        cmd = args[1].lower()
        arg = args[2]

        if cmd == "run":
            run_main(arg)
        elif cmd == "prep":
            run_prep_ffxml_main(arg)
        elif cmd == "param":
            run_parametrize_main(arg)

        sys.exit(0)
