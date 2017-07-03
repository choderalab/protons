# coding=utf-8
"""
Tests the storing of specific protons objects in netcdf files.
"""
import shutil
import tempfile

import numpy as np
import pytest
from simtk import unit
from simtk.openmm import openmm

from protons import ForceFieldProtonDrive
from protons.app import record
from protons import app
from . import get_test_data
from .utilities import SystemSetup, create_compound_gbaoab_integrator


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
    testsystem.forcefield = app.ForceField('gaff.xml', testsystem.ffxml_filename)
    testsystem.gaff = get_test_data("gaff.xml", "../forcefields/")
    testsystem.pdbfile = app.PDBFile(
        get_test_data("imidazole-solvated-minimized.pdb", "testsystems/imidazole_explicit"))
    testsystem.topology = testsystem.pdbfile.topology
    testsystem.nsteps_per_ghmc = 1
    testsystem.constraint_tolerance = 1.e-7
    integrator = create_compound_gbaoab_integrator(testsystem)

    drive = ForceFieldProtonDrive(ffxml_files=[testsystem.ffxml_filename], system=testsystem.system, forcefield=testsystem.forcefield, pressure=testsystem.pressure, topology=testsystem.topology, temperature=testsystem.temperature, perturbations_per_trial=2)
    platform = openmm.Platform.getPlatformByName('CPU')
    context = openmm.Context(testsystem.system, integrator, platform)
    context.setPositions(testsystem.positions)  # set to minimized positions
    context.setVelocitiesToTemperature(testsystem.temperature)
    drive.attach_context(context)
    integrator.step(1)
    drive.update(app.UniformProposal())

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


def test_record_sams():
    """Record the variables for a single interation of a sams algorithm"""
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    # Using required arguments
    # filename,
    # num_titratable_groups : 1
    # ncmc_steps_per_trial 2
    # num_attempts_per_update : 1 num_iterations=None
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), 1, 2, 1, calibration=True, nstates_calibration=4)

    # Arbitrary sequence of weights
    samples = np.random.multivariate_normal([0.000, 1.e2, 0.7e2, -3e1], np.matrix("3 0 0 0; 0 5 0 0; 0 0 7 0; 0 0 0 8"), 10)
    # Arbitrary sequence of deviations
    dev = np.asarray([np.exp(-0.05*(n+1)) for n in range(10)])
    for iteration in range(10):
        record.record_sams_data(ncfile, samples[iteration], dev[iteration], iteration)

    record.display_content_structure(ncfile)
    assert ncfile['SelfAdjustedMixtureSampling/g_k'][4,2] == samples[4,2], "The weights in the netCDF file should match the supplied samples."
    assert ncfile['SelfAdjustedMixtureSampling/deviation'][7] == dev[7], "The deviation recored in the netCDF file should match the supplied value."

    ncfile.close()
    shutil.rmtree(tmpdir)


def test_record_sams_with_metadata():
    """Record the variables for a single interation of a sams algorithm"""
    tmpdir = tempfile.mkdtemp(prefix="protons-test-")
    # Using required arguments
    # filename,
    # num_titratable_groups : 1
    # ncmc_steps_per_trial 2
    # num_attempts_per_update : 1 num_iterations=None
    ncfile = record.netcdf_file('{}/new.nc'.format(tmpdir), 1, 2, 1, calibration=True, nstates_calibration=4)

    # Arbitrary sequence of weights
    samples = np.random.multivariate_normal([0.000, 1.e2, 0.7e2, -3e1], np.matrix("3 0 0 0; 0 5 0 0; 0 0 7 0; 0 0 0 8"),
                                            10)
    # Arbitrary sequence of deviations
    dev = np.asarray([np.exp(-0.05 * (n + 1)) for n in range(10)])
    stage = "burn-in"
    end_of_burnin = 0
    beta = 0.5
    scheme = "binary"
    for iteration in range(10):

        # pretend switch to slow-gain phase
        if iteration == 7:
            stage = "slow-gain"
            end_of_burnin = 7

        record.record_sams_data(ncfile, samples[iteration], dev[iteration], iteration, stage=stage, end_of_burnin=end_of_burnin, beta=beta, scheme=scheme)

    record.display_content_structure(ncfile)
    assert ncfile['SelfAdjustedMixtureSampling/g_k'][4, 2] == samples[
        4, 2], "The weights in the netCDF file should match the supplied samples."
    assert ncfile['SelfAdjustedMixtureSampling/deviation'][7] == dev[
        7], "The deviation recored in the netCDF file should match the supplied value."
    assert ncfile['SelfAdjustedMixtureSampling/stage'][0] == "slow-gain", "The stage should be recorded as slow-gain"
    assert ncfile['SelfAdjustedMixtureSampling/beta'][0] == beta, "Beta should be recorded as {}".format(beta)
    assert ncfile['SelfAdjustedMixtureSampling/scheme'][0] == scheme, "The scheme should be recored as binary."
    assert ncfile['SelfAdjustedMixtureSampling/end_of_burnin'][0] == 7, "The end of burn-in should be recorded as iteration 7."
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)


@pytest.mark.skip("Needs revamping.")
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
    # TODO Disabled integrator writing for now!
    for iteration in range(10):
        record.record_all(ncfile, iteration, drive, integrator=None, context=context, system=system)
    record.display_content_structure(ncfile)
    ncfile.close()
    shutil.rmtree(tmpdir)


def test_read_ncfile():
    """
    Read a protons netcdf file
    """

    from netCDF4 import Dataset
    from protons.app.record import display_content_structure
    filename = get_test_data('sample.nc', 'testsystems/record')
    rootgrp = Dataset(filename, "r", format="NETCDF4")
    print(rootgrp['GHMCIntegrator/naccept'][:] / rootgrp['GHMCIntegrator/ntrials'][:])
    display_content_structure(rootgrp)
    rootgrp.close()