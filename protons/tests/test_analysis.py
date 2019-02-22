# coding=utf-8
"""Unit tests for the app.analysis module."""
import matplotlib as mpl

# For non-gui plotting
mpl.use("Agg")
from protons.app import analysis
from . import get_test_data
import netCDF4
import pytest
import os

travis = os.environ.get("TRAVIS", None)

# TODO write tests for dataframe and array generating functions


@pytest.mark.skipif(travis == "true", reason="Tests have segfaulting risk on Linux.")
class TestBARforSAMS:
    """These tests ensure that the BAR analysis functionality works for the protons format SAMS data."""

    # This file contains the last 50 samples from an imatinib calibration in solvent
    # NOTE These files are in an outdated format.
    sams_netcdf = get_test_data("imatinib-sams-solvent.nc", "testsystems/reporter_data")
    abl_imatinib_netcdf = get_test_data(
        "Abl-Imatinib-full.nc", "testsystems/reporter_data"
    )
    a1 = get_test_data("viologen-a1.nc", "testsystems/netcdf-test")

    def test_bar_on_pregenerated_netcdf(self):
        """Run calculations on a netCDF file from a simulation."""
        with netCDF4.Dataset(self.a1, "r") as dataset:
            output = analysis.bar_all_states(dataset, bootstrap=False)
            assert (
                len(output) == 3
            ), "There should be three free energy differences in the output."
            assert (
                len(output["0->1"]) == 2
            ), "There should be two values for each transition (estimate, error)"

    def test_nonparametric_bootstrap_on_bar_on_pregenerated_netcdf(self):
        """Run non-parametric bootstrap on transitions and then calculate BAR free energy estimates."""

        with netCDF4.Dataset(self.a1, "r") as dataset:
            output = analysis.bar_all_states(
                dataset, bootstrap=True, num_bootstrap_samples=5
            )
            assert (
                len(output) == 3
            ), "There should be three free energy differences in the output."
            assert (
                len(output["0->1"]) == 2
            ), "There should be two values for each transition (estimate, error)"

    def test_plotting_calibration_defaults(self):
        """
        Try plotting calibration data using the defaults
        """
        with netCDF4.Dataset(self.a1, "r") as dataset:
            ax = analysis.plot_calibration_weight_traces(dataset)

    def test_plotting_calibration_zerobased(self):
        """
        Try plotting calibration data using zerobased labels
        """
        with netCDF4.Dataset(self.a1, "r") as dataset:
            ax = analysis.plot_calibration_weight_traces(dataset, zerobased=True)

    def test_plotting_calibration_bootstrap(self):
        """
        Try plotting calibration data using bootstrap BAR estimates
        """
        with netCDF4.Dataset(self.a1, "r") as dataset:
            ax = analysis.plot_calibration_weight_traces(
                dataset, error="bootstrap", num_bootstrap_samples=5
            )

    def test_plotting_calibration_nobar(self):
        """
        Try plotting calibration data without bar
        """
        with netCDF4.Dataset(self.a1, "r") as dataset:
            ax = analysis.plot_calibration_weight_traces(dataset, bar=False)

    def test_calc_acceptance(self):
        """Calculate the NCMC acceptance rate for a netCDF file"""
        with netCDF4.Dataset(self.a1, "r") as dataset:
            assert (
                1.0 >= analysis.calculate_ncmc_acceptance_rate(dataset) >= 0.0
            ), "The acceptance rate is not between 0, 1"


@pytest.mark.skipif(travis == "true", reason="Tests have segfaulting risk on Linux.")
class TestPlots:
    """These tests plot data not specific to calibration."""

    # This file contains the last 50 samples from an imatinib calibration in solvent
    sams_netcdf = get_test_data("imatinib-sams-solvent.nc", "testsystems/reporter_data")
    abl_imatinib_netcdf = get_test_data(
        "Abl-Imatinib-full.nc", "testsystems/reporter_data"
    )
    a1 = get_test_data("viologen-a1.nc", "testsystems/netcdf-test")

    def test_states_trace(self):
        """
        Try plotting a trace of the states.
        """
        with netCDF4.Dataset(self.a1, "r") as dataset:
            ax = analysis.plot_residue_state_traces(dataset, 0)

    def test_state_heatmap(self):
        """Plot the heatmap of titration states"""
        with netCDF4.Dataset(self.a1, "r") as dataset:
            analysis.plot_heatmap(dataset, residues=None, color="state")

    def test_charge_heatmap(self):
        """Plot the heatmap of residue charges"""
        with netCDF4.Dataset(self.a1, "r") as dataset:
            analysis.plot_heatmap(dataset, residues=None, color="charge")

    def test_taut_heatmap(self):
        """Plot the heatmap of residue charges"""
        with netCDF4.Dataset(self.a1, "r") as dataset:
            analysis.plot_tautomer_heatmap(dataset, residues=None)

    def test_extracting_work(self):
        """Extract work distributions for a single residue"""
        with netCDF4.Dataset(self.a1, "r") as dataset:
            analysis.extract_work_distributions(dataset, 0, 1, -1)


class TestMultifileAnalysis:
    """Tests using new multisite type data."""

    a1 = get_test_data("viologen-a1.nc", "testsystems/netcdf-test")
    a2 = get_test_data("viologen-a2.nc", "testsystems/netcdf-test")
    a3 = get_test_data("viologen-a3.nc", "testsystems/netcdf-test")

    def test_multiple_files(self):
        """Test analysis from reading multiple files."""
        datasets = [
            netCDF4.Dataset(filename, "r") for filename in [self.a1, self.a2, self.a3]
        ]

        analysis.plot_calibration_weight_traces(datasets)
        analysis.plt.show()
        return
