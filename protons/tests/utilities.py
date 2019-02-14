from __future__ import print_function

import logging

from simtk import unit, openmm
from simtk.openmm import app

from protons.app.integrators import GHMCIntegrator, GBAOABIntegrator

try:
    openmm.Platform.getPlatformByName("CUDA")
    hasCUDA = True
except Exception:
    logging.info("CUDA unavailable on this system.")
    hasCUDA = False

try:
    from openeye import oechem

    if not oechem.OEChemIsLicensed():
        raise (ImportError("Need License for OEChem!"))
    from openeye import oequacpac

    if not oequacpac.OEQuacPacIsLicensed():
        raise (ImportError("Need License for oequacpac!"))
    from openeye import oeiupac

    if not oeiupac.OEIUPACIsLicensed():
        raise (ImportError("Need License for OEOmega!"))
    from openeye import oeomega

    if not oeomega.OEOmegaIsLicensed():
        raise (ImportError("Need License for OEOmega!"))
    hasOpenEye = True
    openeye_exception_message = str()
except Exception as e:
    hasOpenEye = False
    openeye_exception_message = str(e)


class SystemSetup:
    """Empty class for storing systems and relevant attributes"""

    pass


def make_method(func, input):
    # http://blog.kevinastone.com/generate-your-tests.html
    def test_input(self):
        func(self, input)

    test_input.__name__ = "test_{func}_{input}".format(func=func.__name__, input=input)
    return test_input


def generate(func, *inputs):
    """
    Take a TestCase and add a test method for each input
    """
    # http://blog.kevinastone.com/generate-your-tests.html
    def decorator(testcase):
        for input in inputs:
            test_input = make_method(func, input)
            setattr(testcase, test_input.__name__, test_input)
        return testcase

    return decorator


# TODO default paths are outdated
def make_xml_explicit_tyr(
    inpcrd_filename="constph/examples/calibration-explicit/tyr.inpcrd",
    prmtop_filename="constph/examples/calibration-explicit/tyr.prmtop",
    outfile="tyrosine_explicit",
):

    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    outfile1 = open("{}.sys.xml".format(outfile), "w")
    outfile2 = open("{}.state.xml".format(outfile), "w")
    platform_name = "CPU"
    pH = 9.6
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    positions = inpcrd.getPositions()
    system = prmtop.createSystem(
        implicitSolvent=None, nonbondedMethod=app.CutoffPeriodic, constraints=app.HBonds
    )
    system.addForce(openmm.MonteCarloBarostat(pressure, temperature))
    context = minimizer(platform_name, system, positions)
    outfile1.write(openmm.XmlSerializer.serialize(system))
    outfile2.write(openmm.XmlSerializer.serialize(context.getState(getPositions=True)))


def make_xml_implicit_tyr(
    inpcrd_filename="protons/examples/calibration-implicit/tyr.inpcrd",
    prmtop_filename="protons/examples/calibration-implicit/tyr.prmtop",
    outfile="tyrosine_implicit",
):

    temperature = 300.0 * unit.kelvin
    outfile1 = open("{}.sys.xml".format(outfile), "w")
    outfile2 = open("{}.state.xml".format(outfile), "w")
    platform_name = "CPU"
    pH = 9.6
    inpcrd = app.AmberInpcrdFile(inpcrd_filename)
    prmtop = app.AmberPrmtopFile(prmtop_filename)
    positions = inpcrd.getPositions()
    system = prmtop.createSystem(
        implicitSolvent=app.OBC2, nonbondedMethod=app.NoCutoff, constraints=app.HBonds
    )
    context = minimizer(platform_name, system, positions)
    outfile1.write(openmm.XmlSerializer.serialize(system))
    outfile2.write(openmm.XmlSerializer.serialize(context.getState(getPositions=True)))


def make_xml_explicit_imidazole(
    pdb_filename="protons/examples/Ligand example/imidazole.pdb",
    ffxml_filename="protons/examples/Ligand example/imidazole.xml",
    outfile="imidazole-explicit",
):
    """Solvate an imidazole pdb file and minimize the system"""

    temperature = 300.0 * unit.kelvin
    pressure = 1.0 * unit.atmospheres
    outfile1 = open("{}.sys.xml".format(outfile), "w")
    outfile2 = open("{}.state.xml".format(outfile), "w")
    gaff = get_data("gaff.xml", "forcefields")
    pdb = app.PDBFile(pdb_filename)
    forcefield = app.ForceField(gaff, ffxml_filename, "amber99sbildn.xml", "tip3p.xml")
    integrator = openmm.LangevinIntegrator(
        300 * unit.kelvin, 1.0 / unit.picoseconds, 2.0 * unit.femtoseconds
    )

    integrator.setConstraintTolerance(0.00001)

    modeller = app.Modeller(pdb.topology, pdb.positions)
    modeller.addSolvent(
        forcefield,
        boxSize=openmm.Vec3(3.5, 3.5, 3.5) * unit.nanometers,
        model="tip3p",
        ionicStrength=0.1 * unit.molar,
        positiveIon="Na+",
        negativeIon="Cl-",
    )
    system = forcefield.createSystem(
        modeller.topology,
        nonbondedMethod=app.PME,
        nonbondedCutoff=1.0 * unit.nanometers,
        constraints=app.HBonds,
        rigidWater=True,
        ewaldErrorTolerance=0.0005,
    )
    system.addForce(openmm.MonteCarloBarostat(pressure, temperature))

    simulation = app.Simulation(modeller.topology, system, integrator)
    simulation.context.setPositions(modeller.positions)

    simulation.minimizeEnergy()

    outfile1.write(openmm.XmlSerializer.serialize(simulation.system))
    positions = simulation.context.getState(getPositions=True)
    outfile2.write(openmm.XmlSerializer.serialize(positions))
    app.PDBFile.writeFile(
        simulation.topology,
        modeller.positions,
        open("imidazole-solvated-minimized.pdb", "w"),
    )


def compute_potential_components(context):
    """
    Compute potential energy, raising an exception if it is not finite.

    Parameters
    ----------
    context : simtk.openmm.Context
        The context from which to extract, System, parameters, and positions.

    """
    import copy

    system = context.getSystem()
    system = copy.deepcopy(system)
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    parameters = context.getParameters()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        force.setForceGroup(index)

    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    platform = openmm.Platform.getPlatformByName("Reference")
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    for (parameter, value) in parameters.items():
        context.setParameter(parameter, value)
    energy_components = list()
    for index in range(system.getNumForces()):
        force = system.getForce(index)
        forcename = force.__class__.__name__
        groups = 1 << index
        potential = context.getState(getEnergy=True, groups=groups).getPotentialEnergy()
        energy_components.append((forcename, potential))
    del context, integrator
    return energy_components


def minimizer(platform_name, system, positions, nsteps=1000):
    integrator = openmm.VerletIntegrator(1.0 * unit.femtoseconds)
    CONSTRAINT_TOLERANCE = 1.0e-5
    integrator.setConstraintTolerance(CONSTRAINT_TOLERANCE)
    platform = openmm.Platform.getPlatformByName(platform_name)
    context = openmm.Context(system, integrator, platform)
    context.setPositions(positions)
    logging.info(
        "Initial energy is %s" % context.getState(getEnergy=True).getPotentialEnergy()
    )
    openmm.LocalEnergyMinimizer.minimize(context, 1.0, nsteps)
    logging.info(
        "Final energy is %s" % context.getState(getEnergy=True).getPotentialEnergy()
    )
    positions = context.getState(getPositions=True).getPositions(asNumpy=True)
    return context, positions


def create_compound_ghmc_integrator(testsystem):
    """
    Sets up a compound integrator that uses GHMC.
    Parameters
    ----------
    testsystem - a SystemSetup object that contains details such as the temperature, timestep, et cetera.

    Returns
    -------
    simtk.openmm.openmm.CompoundIntegrator
    """
    integrator = GHMCIntegrator(
        temperature=testsystem.temperature,
        collision_rate=testsystem.collision_rate,
        timestep=testsystem.timestep,
        nsteps=testsystem.nsteps_per_ghmc,
    )
    ncmc_propagation_integrator = GHMCIntegrator(
        temperature=testsystem.temperature,
        collision_rate=testsystem.collision_rate,
        timestep=testsystem.timestep,
        nsteps=testsystem.nsteps_per_ghmc,
    )
    compound_integrator = openmm.CompoundIntegrator()
    compound_integrator.addIntegrator(integrator)
    compound_integrator.addIntegrator(ncmc_propagation_integrator)
    compound_integrator.setCurrentIntegrator(0)
    return compound_integrator


def create_compound_gbaoab_integrator(testsystem):
    """
    Sets up a compound integrator that uses gBAOAB.
    Parameters
    ----------
    testsystem - a SystemSetup object that contains details such as the temperature, timestep, et cetera.

    Returns
    -------
    simtk.openmm.openmm.CompoundIntegrator
    """
    integrator = GBAOABIntegrator(
        temperature=testsystem.temperature,
        collision_rate=testsystem.collision_rate,
        timestep=testsystem.timestep,
        constraint_tolerance=testsystem.constraint_tolerance,
    )
    ncmc_propagation_integrator = GBAOABIntegrator(
        temperature=testsystem.temperature,
        collision_rate=testsystem.collision_rate,
        timestep=testsystem.timestep,
        constraint_tolerance=testsystem.constraint_tolerance,
    )
    compound_integrator = openmm.CompoundIntegrator()
    compound_integrator.addIntegrator(integrator)
    compound_integrator.addIntegrator(ncmc_propagation_integrator)
    compound_integrator.setCurrentIntegrator(0)
    return compound_integrator
