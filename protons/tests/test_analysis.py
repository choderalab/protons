# coding=utf-8
"""Unit tests for the app.analysis module."""

from protons.app import analysis
from . import get_test_data
import netCDF4
import matplotlib as mpl
# For non-gui plotting
mpl.use('Agg')


class TestBARforSAMS:
    """These tests ensure that the BAR analysis functionality works for the protons format SAMS data."""

    # This file contains the last 50 samples from an imatinib calibration in solvent
    sams_netcdf = get_test_data('imatinib-sams-solvent.nc', 'testsystems/reporter_data')
    abl_imatinib_netcdf = get_test_data('Abl-Imatinib-full.nc', 'testsystems/reporter_data')

    def test_bar_on_pregenerated_netcdf(self):
        """Run calculations on a netCDF file from a simulation."""
        dataset = netCDF4.Dataset(self.sams_netcdf)
        output = analysis.bar_all_states(dataset, bootstrap=False)
        assert len(output) == 2, "There should be two free energy differences in the output."
        assert len(output["0->1"]) == 2, "There should be two values for each transition (estimate, error)"

    def test_nonparametric_bootstrap_on_bar_on_pregenerated_netcdf(self):
        """Run non-parametric bootstrap on transitions and then calculate BAR free energy estimates."""

        dataset = netCDF4.Dataset(self.sams_netcdf)
        output = analysis.bar_all_states(dataset, bootstrap=True, num_bootstrap_samples=5)
        assert len(output) == 2, "There should be two free energy differences in the output."
        assert len(output["0->1"]) == 2, "There should be two values for each transition (estimate, error)"

    def test_plotting_calibration_defaults(self):
        """
        Try plotting calibration data using the defaults
        """
        dataset = netCDF4.Dataset(self.sams_netcdf)

        ax = analysis.plot_calibration_weight_traces(dataset)
        analysis.plt.show()

    def test_plotting_calibration_zerobased(self):
        """
        Try plotting calibration data using zerobased labels
        """
        dataset = netCDF4.Dataset(self.sams_netcdf)
        ax = analysis.plot_calibration_weight_traces(dataset, zerobased=True)

    def test_plotting_calibration_bootstrap(self):
        """
        Try plotting calibration data using bootstrap BAR estimates
        """
        dataset = netCDF4.Dataset(self.sams_netcdf)
        ax = analysis.plot_calibration_weight_traces(dataset, error='bootstrap', num_bootstrap_samples=5)

    def test_plotting_calibration_nobar(self):
        """
        Try plotting calibration data without bar
        """
        dataset = netCDF4.Dataset(self.sams_netcdf)
        ax = analysis.plot_calibration_weight_traces(dataset, bar=False)

    def test_calc_acceptance(self):
        """Calculate the NCMC acceptance rate for a netCDF file"""
        dataset = netCDF4.Dataset(self.sams_netcdf)
        assert 1.0 >= analysis.calculate_ncmc_acceptance_rate(dataset) >= 0.0, "The acceptance rate is not between 0, 1"


class TestPlots:
    """These tests plot data not specific to calibration."""
    # This file contains the last 50 samples from an imatinib calibration in solvent
    sams_netcdf = get_test_data('imatinib-sams-solvent.nc', 'testsystems/reporter_data')
    abl_imatinib_netcdf = get_test_data('Abl-Imatinib-full.nc', 'testsystems/reporter_data')

    def test_states_trace(self):
        """
        Try plotting a trace of the states.
        """
        dataset = netCDF4.Dataset(self.sams_netcdf)
        ax = analysis.plot_residue_state_traces(dataset,0)

    def test_state_heatmap(self):
        """Plot the heatmap of titration states"""
        dataset = netCDF4.Dataset(self.abl_imatinib_netcdf)
        analysis.plot_heatmap(dataset, residues=None, color='state')


    def test_charge_heatmap(self):
        """Plot the heatmap of residue charges"""
        dataset = netCDF4.Dataset(self.abl_imatinib_netcdf)
        analysis.plot_heatmap(dataset, residues=None, color='charge')

    def test_taut_heatmap(self):
        """Plot the heatmap of residue charges"""
        dataset = netCDF4.Dataset(self.abl_imatinib_netcdf)
        analysis.plot_tautomer_heatmap(dataset, residues=None)
