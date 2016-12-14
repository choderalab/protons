# coding=utf-8
"""
Tests the storing of specific protons objects in netcdf files.
"""
from protons import ForceFieldProtonDrive
from protons import record
from protons.integrators import GHMCIntegrator
from simtk import unit
from simtk.openmm import openmm, app
import tempfile
import shutil
import netCDF4
from .utilities import SystemSetup
from . import get_test_data
import pytest


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
    testsystem.pdbfile = app.PDBFile(get_test_data("imidazole-solvated-minimized.pdb", "testsystems/imidazole_explicit"))
    testsystem.topology = testsystem.pdbfile.topology
    testsystem.nsteps_per_ghmc = 1
    integrator = GHMCIntegrator(temperature=testsystem.temperature, collision_rate=testsystem.collision_rate,
                                timestep=testsystem.timestep, nsteps=testsystem.nsteps_per_ghmc)

    drive = ForceFieldProtonDrive(testsystem.system, testsystem.temperature, testsystem.pH,
                                   [testsystem.ffxml_filename],
                                   testsystem.topology, integrator, debug=False,
                                   pressure=testsystem.pressure, ncmc_steps_per_trial=0, implicit=False,
                                   residues_by_name=['LIG'])
    platform = openmm.Platform.getPlatformByName('CPU')
    context = openmm.Context(testsystem.system, drive.compound_integrator, platform)
    context.setPositions(testsystem.positions)  # set to minimized positions
    context.setVelocitiesToTemperature(testsystem.temperature)

    return drive, integrator, context, testsystem.system


def test_save_drive():
    """
    Store the variables of a ForceFieldProtonDrive

    """
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    drive,integrator, context, system = setup_forcefield_drive()
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), len(drive.titrationGroups))
    for iteration in range(10):
        record.save_drive_data(ncfile, drive, iteration=iteration)
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)


def test_save_ghmc_integrator():
    """
    Store the variables of a GHMCIntegrator

    """
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    drive,integrator, context, system = setup_forcefield_drive()
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), len(drive.titrationGroups))
    for iteration in range(10):
        record.save_ghmc_integrator_data(ncfile, integrator, iteration)
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)


def test_save_state():
    """
    Store the variables of a Context State
    """
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    drive, integrator, context, system = setup_forcefield_drive()
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), len(drive.titrationGroups))
    for iteration in range(10):
        record.save_state_data(ncfile, context, system, iteration)
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)


def test_save_all():
    """
    Store the variables of multiple objects using convenience function
    """
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    drive, integrator, context, system = setup_forcefield_drive()
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), len(drive.titrationGroups))
    for iteration in range(10):
        record.save_all(ncfile, drive, integrator, context, system, iteration)
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)