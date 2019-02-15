# This script instantiates a file for running constant-pH simulation
import json
import os
import signal
import sys
import netCDF4

from protons import app
from protons.app import NCMCProtonDrive
from protons.app.calibration import SAMSApproach
from protons.app.logger import log, logging

from lxml import etree
from tqdm import trange
from saltswap.wrappers import Salinator
from simtk import openmm as mm
from simtk import unit
from typing import List, Dict, Tuple, Callable, Any, AnyStr
from protons.scripts.utilities import (
    TimeOutError,
    timeout_handler,
    create_calibration_checkpoint_file,
    ExternalGBAOABIntegrator,
    xml_to_topology,
    deserialize_openmm_element,
    deserialize_state_vector,
)

log.setLevel(logging.DEBUG)


# Define a main function that can read in a json file with simulation settings, sets up, and runs the simulation.


def main(jsonfile):
    """Main simulation loop."""

    # TODO Validate json input with json schema?
    settings = json.load(open(jsonfile))

    try:
        format_vars: Dict[str, str] = settings["format_vars"]
    except KeyError:
        format_vars = dict()

    # Retrieve runtime settings
    run = settings["run"]

    # Start timeout to enable clean exit on uncompleted runs
    # Note, does not work on windows!
    if os.name != "nt":
        signal.signal(signal.SIGALRM, timeout_handler)
        script_timeout = int(run["timeout_sec"])
        signal.alarm(script_timeout)

    # Input files
    inp = settings["input"]
    idir = inp["dir"].format(**format_vars)
    input_checkpoint_file = os.path.abspath(
        os.path.join(idir, inp["checkpoint"].format(**format_vars))
    )
    # Load checkpoint file
    with open(input_checkpoint_file, "r") as checkpoint:
        checkpoint_tree = etree.fromstring(checkpoint.read())

    checkpoint_date = checkpoint_tree.attrib["date"]
    log.info(f"Reading checkpoint from '{checkpoint_date}'.")

    topology_element = checkpoint_tree.xpath("TopologyFile")[0]
    topology: app.Topology = xml_to_topology(topology_element)

    # Quick fix for histidines in topology
    # Openmm relabels them HIS, which leads to them not being detected as
    # titratable. Renaming them fixes this.

    for residue in topology.residues():
        if residue.name == "HIS":
            residue.name = "HIP"
        # TODO doublecheck if ASH GLH need to be renamed
        elif residue.name == "ASP":
            residue.name = "ASH"
        elif residue.name == "GLU":
            residue.name = "GLH"

    # Naming the output files
    out = settings["output"]
    odir = out["dir"].format(**format_vars)
    obasename = out["basename"].format(**format_vars)
    runid = format_vars["run_idx"]
    if not os.path.isdir(odir):
        os.makedirs(odir)
    lastdir = os.getcwd()
    os.chdir(odir)

    # File for resuming simulation
    output_checkpoint_file = f"{obasename}-checkpoint-{runid}.xml"

    # System Configuration
    system_element = checkpoint_tree.xpath("System")[0]
    system: mm.System = deserialize_openmm_element(system_element)

    # Deserialize the integrator
    integrator_element = checkpoint_tree.xpath("Integrator")[0]
    integrator: mm.CompoundIntegrator = deserialize_openmm_element(integrator_element)

    perturbations_per_trial = int(run["perturbations_per_ncmc_trial"])
    propagations_per_step = int(run["propagations_per_ncmc_step"])

    # Deserialize the proton drive
    drive_element = checkpoint_tree.xpath("NCMCProtonDrive")[0]
    temperature = float(drive_element.get("temperature_kelvin")) * unit.kelvin
    if "pressure_bar" in drive_element.attrib:
        pressure = float(drive_element.get("pressure_bar")) * unit.bar
    else:
        pressure = None

    driver = NCMCProtonDrive(
        temperature,
        topology,
        system,
        pressure=pressure,
        perturbations_per_trial=perturbations_per_trial,
        propagations_per_step=propagations_per_step,
    )
    driver.state_from_xml_tree(drive_element)

    if driver.calibration_state.approach == SAMSApproach.ONESITE:
        driver.define_pools({"calibration": driver.calibration_state.group_index})

    platform = mm.Platform.getPlatformByName("CUDA")
    properties = {
        "CudaPrecision": "mixed",
        "DeterministicForces": "true",
        "CudaDeviceIndex": os.environ["CUDA_VISIBLE_DEVICES"],
    }

    simulation = app.ConstantPHSimulation(
        topology,
        system,
        integrator,
        driver,
        platform=platform,
        platformProperties=properties,
    )

    # Set the simulation state
    state_element = checkpoint_tree.xpath("State")[0]
    state: mm.State = deserialize_openmm_element(state_element)
    boxvec = state.getPeriodicBoxVectors()
    pos = state.getPositions()
    vel = state.getVelocities()

    simulation.context.setPositions(pos)
    simulation.context.setPeriodicBoxVectors(*boxvec)
    simulation.context.setVelocities(vel)

    # Check if the system has an associated salinator

    saltswap_element = checkpoint_tree.xpath("Saltswap")
    if saltswap_element:
        saltswap_element = saltswap_element[0]
        salt_concentration = (
            float(saltswap_element.get("salt_concentration_molar")) * unit.molar
        )
        salinator = Salinator(
            context=simulation.context,
            system=system,
            topology=topology,
            ncmc_integrator=integrator.getIntegrator(1),
            salt_concentration=salt_concentration,
            pressure=pressure,
            temperature=temperature,
        )
        swapper = salinator.swapper
        deserialize_state_vector(saltswap_element, swapper)
        # If counterion is false, openmm automatically uses a neutralizing background charge
        if run["counter-ion"]:
            driver.attach_swapper(swapper)

    else:
        salinator = None

    # Add reporters
    ncfile = netCDF4.Dataset(f"{obasename}-{runid}.nc", "w")
    dcd_output_name = f"{obasename}-{runid}.dcd"
    reporters = settings["reporters"]
    if "metadata" in reporters:
        simulation.update_reporters.append(app.MetadataReporter(ncfile))

    if "coordinates" in reporters:
        freq = int(reporters["coordinates"]["frequency"])
        simulation.reporters.append(
            app.DCDReporter(dcd_output_name, freq, enforcePeriodicBox=True)
        )

    if "titration" in reporters:
        freq = int(reporters["titration"]["frequency"])
        simulation.update_reporters.append(app.TitrationReporter(ncfile, freq))

    if "sams" in reporters:
        freq = int(reporters["sams"]["frequency"])
        simulation.calibration_reporters.append(app.SAMSReporter(ncfile, freq))

    if "ncmc" in reporters:
        freq = int(reporters["ncmc"]["frequency"])
        if "work_interval" in reporters["ncmc"]:
            work_interval = int(reporters["ncmc"]["work_interval"])
        else:
            work_interval = 0
        simulation.update_reporters.append(
            app.NCMCReporter(ncfile, freq, work_interval)
        )

    total_iterations = int(run["total_update_attempts"])
    md_steps_between_updates = int(run["md_steps_between_updates"])

    # MAIN SIMULATION LOOP STARTS HERE

    try:

        for i in trange(total_iterations, desc="NCMC attempts"):
            if i == 2:
                log.info("Simulation seems to be working. Suppressing debugging info.")
                log.setLevel(logging.INFO)
            simulation.step(md_steps_between_updates)
            # Perform a few COOH updates in between)
            driver.update("COOH", nattempts=3)
            if driver.calibration_state.approach is SAMSApproach.ONESITE:
                simulation.update(1, pool="calibration")
            else:
                simulation.update(1)
            simulation.adapt()

    except TimeOutError:
        log.warn("Simulation ran out of time, saving current results.")

    finally:
        # export the context
        create_calibration_checkpoint_file(
            output_checkpoint_file,
            driver,
            simulation.context,
            simulation.system,
            simulation.integrator,
            topology_element.text,
            salinator=salinator,
        )
        ncfile.close()
        os.chdir(lastdir)


# Actual cmdline interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide a single json file as input.")
    else:
        # Provide the json file to main function
        main(sys.argv[1])
