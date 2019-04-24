from protons.app.ligands import prepare_protein_simulation_systems, create_bond_definitions, prepare_calibration_systems, create_hydrogen_definitions, generate_protons_ffxml, prepare_mol2_for_parametrization
from protons.scripts.utilities import *
from protons.app.protein import prepare_protein
from protons.app import TautomerForceFieldProtonDrive, TautomerNCMCProtonDrive
from protons.app import MetadataReporter, TitrationReporter, NCMCReporter, SAMSReporter
from simtk import unit, openmm as mm
from saltswap.wrappers import Salinator
from protons.app.driver import SAMSApproach, Stage, UpdateRule
from tqdm import trange
import netCDF4
import shutil
import os


def run_simulation(simulation, driver, pdb_object, settings):
    """Main simulation loop."""

    # Add reporters
    ncfile = netCDF4.Dataset(settings['simulation']['netCDF4'], "w")
    if settings['simulation']['dcd_filename']:
        dcd_output_name = settings['simulation']['dcd_filename']
        simulation.reporters.append(app.DCDReporter(dcd_output_name, 100, enforcePeriodicBox=True))

    if int(settings['reporters']['metadata']) == 1:    
        simulation.update_reporters.append(app.MetadataReporter(ncfile))
    if settings['reporters']['titration']:
        simulation.update_reporters.append(app.TitrationReporter(ncfile, 1))
    if settings['reporters']['sams']:
        simulation.calibration_reporters.append(app.SAMSReporter(ncfile, 1))
    if settings['reporters']['ncmc']:
        simulation.update_reporters.append(app.NCMCReporter(ncfile, 1, 0))
    total_iterations = int(settings['simulation']["total_update_attempts"])
    md_steps_between_updates = int(settings['simulation']["md_steps_between_updates"])
    perturbations_per_trial = int(settings['simulation']["perturbations_per_trial"])

    # MAIN SIMULATION LOOP STARTS HERE
    pos = simulation.context.getState(getPositions=True).getPositions() 
    mm.app.PDBFile.writeFile(pdb_object.topology, pos, open(settings['output']['dir'] + '/tmp/start.pdb', 'w'))

    for i in trange(total_iterations, desc="NCMC attempts"):
        if i == 50:
            log.info("Simulation seems to be working. Suppressing debugging info.")
            #log.setLevel(log.INFO)
        simulation.step(md_steps_between_updates)
        # Perform a few COOH updates in between
        driver.update("COOH", nattempts=3)
        pos = simulation.context.getState(getPositions=True).getPositions() 
        mm.app.PDBFile.writeFile(pdb_object.topology, pos, open(settings['output']['dir'] + '/tmp/mcmc_'+str(i)+'.pdb', 'w'))
        log.info(driver.calibration_state.approach)
        if driver.calibration_state is not None:
            if driver.calibration_state.approach is SAMSApproach.ONESITE:
                simulation.update(1, pool="calibration")
            else:
                simulation.update(1)
            simulation.adapt()
        else:
            simulation.update(1)


def setting_up_tautomer(settings, isomer_dictionary):

    pH = settings['pH']
    resname = settings['ligand-resname']

    ofolder = settings['output']['dir']
    if os.path.isdir(ofolder):
        shutil.rmtree(ofolder)
        os.makedirs(ofolder)

    #retrieve input files
    tautomer_heavy_atoms = settings['input']['dir'] + '/' + resname + '.pdb'

    input_mol2 = settings['input']['dir'] + '/' + resname + '_tautomer_set.mol2'
    input_sdf = settings['input']['dir'] + '/' + resname + '_tautomer_set.sdf'
    
    #define location of set up files
    hydrogen_def = settings['input']['dir'] + '/' + resname + '.hxml'
    bond_def = settings['input']['dir'] + '/' + resname + '.bxml'
    offxml = settings['output']['dir'] + '/' + resname + '.ffxml'

    # Debugging/intermediate files
    dhydrogen_fix = settings['input']['dir'] + '/' + resname + '_hydrogen_fixed.mol2'  
    
    ofolder = settings['output']['dir']
    
    # Begin processing
    # process into mol2
    prepare_mol2_for_parametrization(input_mol2, input_sdf, dhydrogen_fix, patch_bonds=True, keep_intermediate=False)

    # parametrize
    generate_protons_ffxml(inputmol2=dhydrogen_fix, isomer_dicts=isomer_dictionary, outputffxml=offxml, pH=pH, resname=resname.upper(), tautomers=True, pdb_file_path=tautomer_heavy_atoms)
    # create hydrogen definitions for the superset of all tautomers
    create_hydrogen_definitions(offxml, hydrogen_def, tautomers=True)
    # create bond definitions for the superset of all tautomers
    create_bond_definitions(offxml, bond_def)
    from simtk.openmm.app import Topology
    Topology.loadBondDefinitions(bond_def)


def generate_calibration_system(settings):
    #define location of set up files
    resname = settings['ligand-resname']
    hydrogen_def = settings['input']['dir'] + '/' + resname + '.hxml'
    bond_def = settings['input']['dir'] + '/' + resname + '.bxml'
    offxml = settings['output']['dir'] + '/' + resname + '.ffxml'
    tautomer_heavy_atoms = settings['input']['dir'] + '/' + resname + '.pdb'
    # prepare solvated system
    prepare_calibration_systems(vacuum_file=tautomer_heavy_atoms, output_basename=settings['output']['dir'] + '/' + resname, ffxml=offxml, hxml=hydrogen_def, bxml=bond_def)

def generate_protein_systems(settings):
    #define location of set up files

    log.info('Prepare protein ligand simulations system.')
    resname = settings['ligand-resname']
    pdb_name = settings['protein-name']
    hydrogen_def = settings['input']['dir'] + '/' + resname + '.hxml'
    bond_def = settings['input']['dir'] + '/' + resname + '.bxml'
    offxml = settings['output']['dir'] + '/' + resname + '.ffxml'
    protein_tautomer_system_raw = settings['input']['dir'] + '/' + pdb_name + '.pdb'
    protein_tautomer_system = settings['input']['dir'] + '/' + pdb_name + '_prepared.pdb'
    # prepare solvated system
    log.info('Preprocess the protein ligand structure.')
    prepare_protein(protein_tautomer_system_raw, protein_tautomer_system)
    prepare_protein_simulation_systems(protein_name=pdb_name, protein_ligand=protein_tautomer_system, ligand_resname=resname, output_basename=settings['output']['dir'], ffxml=offxml, hxml=hydrogen_def, bxml=bond_def)


def generate_simulation_and_driver(settings):

    #change to output file
    ofolder = settings['output']['dir']
    os.chdir(ofolder)

    # Naming the output files
    
    input_pdbx_file = settings['input']['structure']
    custom_xml = settings['forcefield']['user-xml']
    forcefield = app.ForceField(*settings['forcefield']['default'] + custom_xml)

    # Load structure
    # The input should be an mmcif/pdbx file'
    pdb_object = app.PDBxFile(input_pdbx_file)
    modeller = app.Modeller(pdb_object.topology, pdb_object.positions)

    # Atoms , connectivity, residues
    topology = modeller.topology

    # XYZ positions for every atom
    positions = modeller.positions

    # Quick fix for histidines in topology
    # Openmm relabels them HIS, which leads to them not being detected as
    # titratable. Renaming them fixes this.

    for residue in topology.residues():
        if residue.name == "HIS":
            residue.name = "HIP"
        # TODO doublecheck if ASH GLH need to be renamed
        #elif residue.name == "ASP":
        #    residue.name = "ASH"
        #elif residue.name == "GLU":
        #    residue.name = "GLH"

    if not os.path.isdir('tmp'):
        os.makedirs('tmp')

    # Structure preprocessing settings
    if "preprocessing" in settings:
        preproc: Dict[str, Any] = settings["preprocessing"]

        # Steps of MD before starting the main loop
        num_thermalization_steps = int(preproc["num_thermalization_steps"])
        pre_run_minimization_tolerance: unit.Quantity = float(
            preproc["minimization_tolerance_kjmol"]
        ) * unit.kilojoule / unit.mole
        minimization_max_iterations = int(preproc["minimization_max_iterations"])

    # System Configuration
    sysprops = settings["system"]
    temperature = float(sysprops["temperature_k"]) * unit.kelvin
    if "salt_concentration_molar" in sysprops:
        salt_concentration: unit.Quantity = float(
            sysprops["salt_concentration_molar"]
        ) * unit.molar
    else:
        salt_concentration = None

    rigidWater = True
    constraints = app.HBonds

    if "PME" in sysprops:
        pmeprops = sysprops["PME"]
        nonbondedMethod = app.PME
        ewaldErrorTolerance = float(pmeprops["ewald_error_tolerance"])
        barostatInterval = int(pmeprops["barostat_interval"])
        switching_distance = float(pmeprops["switching_distance_nm"]) * unit.nanometers
        nonbondedCutoff = float(pmeprops["nonbonded_cutoff_nm"]) * unit.nanometers
        pressure = float(pmeprops["pressure_atm"]) * unit.atmosphere

        system = forcefield.createSystem(
            topology,
            nonbondedMethod=nonbondedMethod,
            constraints=constraints,
            rigidWater=rigidWater,
            ewaldErrorTolerance=ewaldErrorTolerance,
            nonbondedCutoff=nonbondedCutoff,
        )
        for force in system.getForces():
            if isinstance(force, mm.NonbondedForce):
                force.setUseSwitchingFunction(True)

                force.setSwitchingDistance(switching_distance)

        # TODO disable in implicit solvent
        # NPT simulation
        system.addForce(mm.MonteCarloBarostat(pressure, temperature, barostatInterval))
    else:
        pressure = None
        system = forcefield.createSystem(
            topology,
            nonbondedMethod=app.NoCutoff,
            constraints=app.HBonds,
            rigidWater=True,
        )

    # Integrator options
    integrator_opts = settings["integrator"]
    timestep = integrator_opts["timestep_fs"] * unit.femtosecond
    constraint_tolerance = integrator_opts["constraint_tolerance"]
    collision_rate = integrator_opts["collision_rate_per_ps"] / unit.picosecond
    number_R_steps = 1

    integrator = ExternalGBAOABIntegrator(
        number_R_steps=number_R_steps,
        temperature=temperature,
        collision_rate=collision_rate,
        timestep=timestep,
        constraint_tolerance=constraint_tolerance,
    )
    ncmc_propagation_integrator = ExternalGBAOABIntegrator(
        number_R_steps=number_R_steps,
        temperature=temperature,
        collision_rate=collision_rate,
        timestep=timestep,
        constraint_tolerance=constraint_tolerance,
    )

    # Define a compound integrator
    compound_integrator = mm.CompoundIntegrator()
    compound_integrator.addIntegrator(integrator)
    compound_integrator.addIntegrator(ncmc_propagation_integrator)
    compound_integrator.setCurrentIntegrator(0)

    # Script specific settings

    # Register the timeout handling
    driver = TautomerForceFieldProtonDrive(
        temperature,
        topology,
        system,
        forcefield,
        ffxml_files=custom_xml,
        pressure=pressure,
        perturbations_per_trial=perturbations_per_trial,
        propagations_per_step=1,
    )

    if "reference_free_energies" in settings:
        # calibrated free energy values can be provided here
        gk_dict = settings["reference_free_energies"]

        # Clean comments inside dictionary
        to_delete = list()
        for key in gk_dict.keys():
            if key.startswith("_"):
                to_delete.append(key)

        for key in to_delete:
            del (gk_dict[key])

        # Make arrays
        for key, value in gk_dict.items():
            # Convert to array of float
            val_array = np.asarray(value).astype(np.float64)
            if not val_array.ndim == 1:
                raise ValueError("Reference free energies should be simple lists.")
            gk_dict[key] = val_array

        driver.import_gk_values(gk_dict)

    # TODO allow platform specification for setup

    if settings['platform']['driver'] == 'CUDA':
        platform = mm.Platform.getPlatformByName("CUDA")
        properties = {"Precision": "mixed"}
    else:
        platform = mm.Platform.getPlatformByName("CPU")
        properties = {}
    # Set up calibration mode
    # SAMS settings

    if "SAMS" in settings:
        sams = settings["SAMS"]

        beta_burnin = float(sams["beta"])
        min_burnin = int(sams["min_burn"])
        min_slow = int(sams["min_slow"])
        min_fast = int(sams["min_fast"])

        flatness_criterion = float(sams["flatness_criterion"])
        if sams["update_rule"] == "binary":
            update_rule = UpdateRule.BINARY
        elif sams["update_rule"] == "global":
            update_rule = UpdateRule.GLOBAL
        else:
            update_rule = UpdateRule.BINARY

        # Assumes calibration residue is always the last titration group if onesite

        if sams["sites"] == "multi":
            driver.enable_calibration(
                approach=SAMSApproach.MULTISITE,
                update_rule=update_rule,
                flatness_criterion=flatness_criterion,
                min_burn=min_burnin,
                min_slow=min_slow,
                min_fast=min_fast,
                beta_sams=beta_burnin,
            )
        elif sams["sites"] == "one":
            if "group_index" in sams:
                calibration_titration_group_index = int(sams["group_index"])
            else:
                calibration_titration_group_index = len(driver.titrationGroups) - 1

            driver.enable_calibration(
                approach=SAMSApproach.ONESITE,
                group_index=calibration_titration_group_index,
                update_rule=update_rule,
                flatness_criterion=flatness_criterion,
                min_burn=min_burnin,
                min_slow=min_slow,
                min_fast=min_fast,
            )
            # Define residue pools
            pools = {"calibration": [calibration_titration_group_index]}
            # TODO the pooling feature could eventually be exposed in the json
            driver.define_pools(pools)
    properties = None
    # Create simulation object
    # If calibration is required, this class will automatically deal with it.
    simulation = app.ConstantPHSimulation(
        topology,
        system,
        compound_integrator,
        driver,
        platform=platform,
        platformProperties=properties,
    )
    simulation.context.setPositions(positions)

    # After the simulation system has been defined, we can add salt to the system using saltswap.
    if salt_concentration is not None and "PME" in sysprops:
        salinator = Salinator(
            context=simulation.context,
            system=system,
            topology=topology,
            ncmc_integrator=compound_integrator.getIntegrator(1),
            salt_concentration=salt_concentration,
            pressure=pressure,
            temperature=temperature,
        )
        salinator.neutralize()
        salinator.initialize_concentration()
    else:
        salinator = None

    # Minimize the initial configuration to remove bad contacts
    if "preprocessing" in settings:
        simulation.minimizeEnergy(
            tolerance=pre_run_minimization_tolerance,
            maxIterations=minimization_max_iterations,
        )
        # Slightly equilibrate the system, detect early issues.
        simulation.step(num_thermalization_steps)

    return simulation, driver, pdb_object
