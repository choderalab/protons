from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph.constph import MonteCarloTitration
from constph.calibration import CalibrationTitration, MBarCalibrationTitration, AminoAcidCalibrator
from unittest import TestCase, skip, skipIf
from . import get_data
from nose.plugins.skip import SkipTest
import os


class TyrosineExplicitTestCase(TestCase):
    
    def setUp(self):
        self.temperature = 300.0 * unit.kelvin
        self.pressure = 1.0 * unit.atmospheres
        self.timestep = 1.0 * unit.femtoseconds
        self.collision_rate = 9.1 / unit.picoseconds
        self.pH = 9.6
        self.platform_name = 'CPU'
        testsystems = get_data('tyr_explicit', 'testsystems')
        self.positions = openmm.XmlSerializer.deserialize(open('{}/tyr.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        self.system = openmm.XmlSerializer.deserialize(open('{}/tyr.sys.xml'.format(testsystems)).read())
        self.prmtop = app.AmberPrmtopFile('{}/tyr.prmtop'.format(testsystems))
        self.cpin_filename = '{}/tyr.cpin'.format(testsystems)
        
    def test_tyrosine_instantaneous(self):
        """
        Run tyrosine in explicit solvent with an instanteneous state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename, integrator, debug=False,
                                           pressure=self.pressure, nsteps_per_trial=0, implicit=False)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation


    def test_tyrosine_calibration_instantaneous_binary(self):
        """
        Calibrate (binary update) tyrosine in explicit solvent with an instanteneous state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = CalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                            integrator, debug=False,
                                            pressure=self.pressure, nsteps_per_trial=0, implicit=False)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context, 'binary')

    def test_tyrosine_calibration_instantaneous_global(self):
        """
        Calibrate (global update) tyrosine in explicit solvent with an instanteneous state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = CalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                            integrator, debug=False,
                                            pressure=self.pressure, nsteps_per_trial=0, implicit=False)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context, 'global')

    @skip("Current api incompatible, circular dependency on context")
    def test_tyrosine_calibration_instantaneous_mbar(self):
        """
        Calibrate (MBAR) tyrosine in explicit solvent with an instantaneous state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MBarCalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                                context, integrator, debug=False,
                                                pressure=self.pressure, nsteps_per_trial=0, implicit=False)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context)

    def test_tyrosine_ncmc(self):
        """
        Run tyrosine in explicit solvent with an ncmc state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename, integrator, debug=False,
                                           pressure=self.pressure, nsteps_per_trial=10, implicit=False)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_calibration_ncmc_binary(self):
        """
        Calibrate (binary update) tyrosine in explicit solvent with an ncmc state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = CalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                           integrator, debug=False,
                                           pressure=self.pressure, nsteps_per_trial=10, implicit=False)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context, 'binary')

    def test_tyrosine_calibration_ncmc_global(self):
        """
        Calibrate (global update) tyrosine in explicit solvent with an ncmc state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = CalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                            integrator, debug=False,
                                            pressure=self.pressure, nsteps_per_trial=10, implicit=False)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context, 'global')

    @skip("Current api incompatible, circular dependency on context")
    def test_tyrosine_calibration_ncmc_mbar(self):
        """
        Calibrate (MBAR) tyrosine in explicit solvent with an ncmc state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MBarCalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                            context, integrator, debug=False,
                                            pressure=self.pressure, nsteps_per_trial=10, implicit=False)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context)


class TestAminoAcidsExplicitCalibration(object):

    @classmethod
    def setup(cls):
        settings = dict()
        settings["temperature"] = 300.0 * unit.kelvin
        settings["timestep"] = 1.0 * unit.femtosecond
        settings["pressure"] = 1.0 * unit.hectopascal
        settings["collision_rate"] = 9.1 / unit.picoseconds
        settings["nsteps_per_trial"] = 5
        settings["pH"] = 7.4
        settings["solvent"] = "explicit"
        cls.settings = settings

    def test_calibration(self):
        """
        Calibrate a single amino acid in explicit solvent
        """

        for acid in ("lys", "cys", "tyr", "as4", "gl4", "hip"):
            yield self.calibrate, acid

    def calibrate(self, resname):
        if os.environ.get("TRAVIS", None) == 'true':
            # Only run a single amino acid in explicit due to expensiveness.
            if not resname == "hip":
                raise SkipTest

        print(resname)
        aac = AminoAcidCalibrator(resname, self.settings, minimize=False)
        print(aac.calibrate(iterations=5, mc_every=4, weights_every=1))


@skipIf(os.environ.get("TRAVIS", None) == 'true', "Skip expensive test on travis")
class PeptideExplicitTestCase(TestCase):

    def setUp(self):
        self.temperature = 300.0 * unit.kelvin
        self.pressure = 1.0 * unit.atmospheres
        self.timestep = 1.0 * unit.femtoseconds
        self.collision_rate = 9.1 / unit.picoseconds
        self.pH = 7.4
        self.platform_name = 'CPU'
        testsystems = get_data('edchky_explicit', 'testsystems')
        self.positions = openmm.XmlSerializer.deserialize(open('{}/edchky-explicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        self.system = openmm.XmlSerializer.deserialize(open('{}/edchky-explicit.sys.xml'.format(testsystems)).read())
        self.prmtop = app.AmberPrmtopFile('{}/edchky-explicit.prmtop'.format(testsystems))
        self.cpin_filename = '{}/edchky-explicit.cpin'.format(testsystems)
        calibration_settings = dict()
        calibration_settings["temperature"] = self.temperature
        calibration_settings["timestep"] = self.timestep
        calibration_settings["pressure"] = self.pressure
        calibration_settings["collision_rate"] = self.collision_rate
        calibration_settings["pH"] = self.pH
        calibration_settings["solvent"] = "explicit"
        calibration_settings["nsteps_per_trial"] = 5
        self.calibration_settings = calibration_settings

    def test_peptide_ncmc_calibrated(self):
        """
        Run edchky peptide in explicit solvent with an ncmc state switch and calibration
        """

        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                           integrator, debug=False,
                                           pressure=self.pressure, nsteps_per_trial=10, implicit=False)
        mc_titration.calibrate(self.calibration_settings, iterations=5, mc_every=4, weights_every=1)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation