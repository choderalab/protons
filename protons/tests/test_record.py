# coding=utf-8
"""
Tests the storing of specific protons objects in netcdf files.
"""
import shutil
import tempfile

from simtk import unit
from simtk.openmm import openmm, app

from protons import ForceFieldProtonDrive
from protons import record
from protons.integrators import GHMCIntegrator
from . import get_test_data
from .utilities import SystemSetup


def setup_forcefield_drive():
    """
    Set up a forcefield drive containing the imidazole system
    """
    testsystem = SystemSetup()
    testsystem.temperature = 300.0 * unit.kelvin
    testsystem.pressure = 1.0 * unit.atmospheres
    testsystem.timestep = 1.0 * unit.femtoseconds
    testsystem.collision_rate = 1.0 / unit.picoseconds
    testsystem.pH = 9.6
    testsystems = get_test_data('imidazole_explicit', 'testsystems')
    testsystem.positions = openmm.XmlSerializer.deserialize(
        open('{}/imidazole-explicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
    testsystem.system = openmm.XmlSerializer.deserialize(
        open('{}/imidazole-explicit.sys.xml'.format(testsystems)).read())
    testsystem.ffxml_filename = '{}/protons-imidazole.xml'.format(testsystems)
    testsystem.gaff = get_test_data("gaff.xml", "../forcefields/")
    testsystem.pdbfile = app.PDBFile(
        get_test_data("imidazole-solvated-minimized.pdb", "testsystems/imidazole_explicit"))
    testsystem.topology = testsystem.pdbfile.topology
    testsystem.nsteps_per_ghmc = 1
    integrator = GHMCIntegrator(temperature=testsystem.temperature, collision_rate=testsystem.collision_rate,
                                timestep=testsystem.timestep, nsteps=testsystem.nsteps_per_ghmc)

    drive = ForceFieldProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH,
                                  [testsystem.ffxml_filename],
                                  testsystem.topology, integrator, debug=False,
                                  pressure=testsystem.pressure, ncmc_steps_per_trial=2, implicit=False,
                                  residues_by_name=['LIG'], nattempts_per_update=1)
    platform = openmm.Platform.getPlatformByName('CPU')
    context = openmm.Context(testsystem.system, drive.compound_integrator, platform)
    context.setPositions(testsystem.positions)  # set to minimized positions
    context.setVelocitiesToTemperature(testsystem.temperature)
    integrator.step(1)
    drive.update(context)

    return drive, integrator, context, testsystem.system


def test_record_drive():
    """
    Record the variables of a ForceFieldProtonDrive
    """
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    drive, integrator, context, system = setup_forcefield_drive()
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), len(drive.titrationGroups), 2, 1)
    for iteration in range(10):
        record.record_drive_data(ncfile, drive, iteration=iteration)
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)


def test_record_ghmc_integrator():
    """
    Record the variables of a GHMCIntegrator
    """
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    drive, integrator, context, system = setup_forcefield_drive()
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), len(drive.titrationGroups), 2, 1)
    for iteration in range(10):
        record.record_ghmc_integrator_data(ncfile, integrator, iteration)
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)


def test_record_state():
    """
    Record the variables of a Context State
    """
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    drive, integrator, context, system = setup_forcefield_drive()
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), len(drive.titrationGroups), 2, 1)
    for iteration in range(10):
        record.record_state_data(ncfile, context, system, iteration)
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)


def test_record_all():
    """
    Record the variables of multiple objects using convenience function
    """
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    drive, integrator, context, system = setup_forcefield_drive()
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), len(drive.titrationGroups),2 , 1)
    for iteration in range(10):
        record.record_all(ncfile, iteration, drive, integrator, context, system)
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)
