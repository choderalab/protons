# This script instantiates a file for running constant-pH simulation
import json
import os
import signal
import sys
import netCDF4

from protons import app
from protons.app import ForceFieldProtonDrive

from protons.app.logger import log, logging

from saltswap.wrappers import Salinator
from simtk import openmm as mm
from simtk import unit
from typing import List, Dict, Tuple, Callable, Any, AnyStr
from warnings import warn
from protons.scripts.utilities import TimeOutError, timeout_handler, create_calibration_checkpoint_file, ExternalGBAOABIntegrator
from protons.app.driver import SAMSApproach, Stage, UpdateRule
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

    # Input files
    inp = settings["input"]
    idir = inp["dir"].format(**format_vars)

    # Make a ForceField object from user directed files

    # Look for block in input
    try:
        ff: Dict[str, List[str]] = settings['forcefield']
    except KeyError:
        raise KeyError("No forcefield block specified")

    # Any files included with openmm/protons by default are retrieved here
    try:
        default_ff: List[str] = ff["default"]
    except KeyError:
        # In typical use case I wouldn't expect purely user files to be used.
        warn("'default' list missing from 'forcefield' block.", UserWarning)
        default_ff = []

    # all user provided files here
    try:
        user_ff: List[str] = ff['user']
        user_ff_paths: List[str] = []
        for user_file in user_ff:
            rel_path = os.path.join(idir, user_file.format(**format_vars))
            user_ff_paths.append(os.path.abspath(rel_path))
    except KeyError:
        user_ff_paths = []

    if len(default_ff) + len(user_ff_paths) == 0:
        raise ValueError("No forcefield files provided.")
    forcefield = app.ForceField(*(user_ff_paths + default_ff))



    # Load structure
    # The input should be an mmcif/pdbx file
    input_pdbx_file = os.path.join(idir, inp["structure"].format(**format_vars))
    pdb_object = app.PDBxFile(input_pdbx_file)

    # Atoms , connectivity, residues
    topology = pdb_object.topology

    # XYZ positions for every atom
    positions = pdb_object.positions

    # Quick fix for histidines in topology
    # Openmm relabels them HIS, which leads to them not being detected as
    # titratable. Renaming them fixes this.

    for residue in topology.residues():
        if residue.name == 'HIS':
            residue.name = 'HIP'
        # TODO doublecheck if ASH GLH need to be renamed
        elif residue.name == 'ASP':
            residue.name = 'ASH'
        elif residue.name == 'GLU':
            residue.name = 'GLH'

    # Naming the output files
    out = settings["output"]
    odir = out["dir"].format(**format_vars)
    obasename = out["basename"].format(**format_vars)
    if not os.path.isdir(odir):
        os.makedirs(odir)
    lastdir = os.getcwd()
    os.chdir(odir)

    # File for resuming simulation
    output_checkpoint_file = f"{obasename}-checkpoint-0.xml"

    # Structure preprocessing settings
    preproc: Dict[str, Any] = settings["preprocessing"]

    # Steps of MD before starting the main loop
    num_thermalization_steps = int(preproc["num_thermalization_steps"])
    pre_run_minimization_tolerance: unit.Quantity = float(
        preproc["minimization_tolerance_kjmol"]) * unit.kilojoule / unit.mole
    minimization_max_iterations = int(preproc["minimization_max_iterations"])

    # System Configuration
    sysprops = settings["system"]
    temperature = float(sysprops["temperature_k"]) * unit.kelvin
    if "salt_concentration_molar" in sysprops:
        salt_concentration: unit.Quantity = float(sysprops["salt_concentration_molar"]) * unit.molar
    else:
        salt_concentration = None

    rigidWater = True
    constraints = app.HBonds

    if 'PME' in sysprops:
        pmeprops = sysprops['PME']
        nonbondedMethod = app.PME
        ewaldErrorTolerance = float(pmeprops["ewald_error_tolerance"])
        barostatInterval = int(pmeprops["barostat_interval"])
        switching_distance = float(pmeprops["switching_distance_nm"]) * unit.nanometers
        nonbondedCutoff = float(pmeprops["nonbonded_cutoff_nm"]) * unit.nanometers
        pressure = float(pmeprops["pressure_atm"]) * unit.atmosphere

        system = forcefield.createSystem(topology, nonbondedMethod=nonbondedMethod, constraints=constraints,
                                         rigidWater=rigidWater, ewaldErrorTolerance=ewaldErrorTolerance,
                                         nonbondedCutoff=nonbondedCutoff)
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setUseSwitchingFunction(True)

                force.setSwitchingDistance(switching_distance)

        # TODO disable in implicit solvent
        # NPT simulation
        system.addForce(
            mm.MonteCarloBarostat(
                pressure,
                temperature,
                barostatInterval))
    else:
        pressure = None
        system = forcefield.createSystem(topology, nonbondedMethod=app.NoCutoff,
                                         constraints=app.HBonds, rigidWater=True)

    # Integrator options
    integrator_opts = settings["integrator"]
    timestep = integrator_opts["timestep_fs"] * unit.femtosecond
    constraint_tolerance = integrator_opts["constraint_tolerance"]
    collision_rate = integrator_opts["collision_rate_per_ps"] / unit.picosecond
    number_R_steps = 1

    integrator = ExternalGBAOABIntegrator(number_R_steps=number_R_steps, temperature=temperature,
                                          collision_rate=collision_rate, timestep=timestep,
                                          constraint_tolerance=constraint_tolerance)
    ncmc_propagation_integrator = ExternalGBAOABIntegrator(number_R_steps=number_R_steps, temperature=temperature,
                                                           collision_rate=collision_rate, timestep=timestep,
                                                           constraint_tolerance=constraint_tolerance)

    # Define a compound integrator
    compound_integrator = mm.CompoundIntegrator()
    compound_integrator.addIntegrator(integrator)
    compound_integrator.addIntegrator(ncmc_propagation_integrator)
    compound_integrator.setCurrentIntegrator(0)



    # Script specific settings

    # Register the timeout handling 
    signal.signal(signal.SIGABRT, timeout_handler)
    script_timeout = 3600  # 1 h

    driver = ForceFieldProtonDrive(temperature, topology, system, forcefield, user_ff_paths + ["amber10-constph.xml"],
                                   pressure=pressure,
                                   perturbations_per_trial=10000, propagations_per_step=1)

    # # properties = {'CudaPrecision': 'mixed', 'DeterministicForces': 'true',
    # #               'CudaDeviceIndex': os.environ['CUDA_VISIBLE_DEVICES']}
    # properties = dict()

    platform = mm.Platform.getPlatformByName('OpenCL')
    properties = {'OpenCLPrecision': 'double'}

    # Set up calibration mode
    # SAMS settings
    sams = settings["SAMS"]

    beta_burnin = float(sams["beta"])
    min_burnin = int(sams["min_burn"])
    flatness_criterion = float(sams["flatness_criterion"])
    if sams["update_rule"] =='binary':
        update_rule = UpdateRule.BINARY
    elif sams["update_rule"] == 'global':
        update_rule = UpdateRule.GLOBAL
    else:
        update_rule = UpdateRule.BINARY

    # Assumes calibration residue is always the last titration group if onesite

    if sams["sites"] == "multi":
        driver.enable_calibration(approach=SAMSApproach.MULTISITE, update_rule=update_rule, flatness_criterion=flatness_criterion, min_burn=min_burnin, beta_sams=beta_burnin)
    elif sams["sites"] == "one":
        if "group_index" in sams:
            calibration_titration_group_index = int(sams['group_index'])
        else:
            calibration_titration_group_index = len(driver.titrationGroups) - 1

        driver.enable_calibration(approach=SAMSApproach.ONESITE, group_index=calibration_titration_group_index, update_rule=update_rule, flatness_criterion=flatness_criterion, min_burn= min_burnin)
        # Define residue pools
        pools = {'calibration': [calibration_titration_group_index]}
        driver.define_pools(pools)

    # Create SAMS sampler
    simulation = app.ConstantPHSimulation(topology, system, compound_integrator, driver, platform=platform, platformProperties=properties)
    simulation.context.setPositions(positions)

    # After the simulation system has been defined, we can add salt to the system using saltswap.
    if salt_concentration is not None and "PME" in sysprops:
        salinator = Salinator(context=simulation.context,
                              system=system,
                              topology=topology,
                              ncmc_integrator=compound_integrator.getIntegrator(1),
                              salt_concentration=salt_concentration,
                              pressure=pressure,
                              temperature=temperature)
        salinator.neutralize()
        salinator.initialize_concentration()
        swapper = salinator.swapper
    else:
        swapper = None
        # Dont attach swapper yet, we might not want to use counterions

    # Minimize the initial configuration to remove bad contacts
    simulation.minimizeEnergy(tolerance=pre_run_minimization_tolerance, maxIterations=minimization_max_iterations)
    # Slightly equilibrate the system, detect early issues.
    simulation.step(num_thermalization_steps)

    # MAIN SIMULATION LOOP STARTS HERE
    # export the context
    create_calibration_checkpoint_file(output_checkpoint_file, driver, simulation.context, simulation.system, simulation.integrator, swapper=swapper)
    os.chdir(lastdir)


# Actual cmdline interface
if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Please provide a single json file as input.")
    else:
        # Provide the json file to main function
        main(sys.argv[1])
