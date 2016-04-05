from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph.constph import MonteCarloTitration
from constph.calibration import CalibrationTitration, MBarCalibrationTitration, AminoAcidCalibrator
from unittest import TestCase, skip, skipIf
from . import get_data
from nose.plugins.skip import SkipTest
from simtk import unit


class TyrosineImplicitTestCase(TestCase):
    
    def setUp(self):
        # Precalculate and set up a system that will be shared for all tests
        self.temperature = 300.0 * unit.kelvin
        self.pressure = 1.0 * unit.atmospheres
        self.timestep = 1.0 * unit.femtoseconds
        self.collision_rate = 9.1 / unit.picoseconds
        self.pH = 9.6
        self.platform_name = 'CPU'
        testsystems = get_data('tyr_implicit', 'testsystems')
        self.positions = openmm.XmlSerializer.deserialize(open('{}/tyr.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        self.system = openmm.XmlSerializer.deserialize(open('{}/tyr.sys.xml'.format(testsystems)).read())
        self.prmtop = app.AmberPrmtopFile('{}/tyr.prmtop'.format(testsystems))
        self.cpin_filename = '{}/tyr.cpin'.format(testsystems)
        
    def test_tyrosine_instantaneous(self):
        """
        Run tyrosine in implicit solvent with an instanteneous state switch.
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename, integrator, debug=False,
                                           pressure=None, nsteps_per_trial=0, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_instantaneous_calibrated(self):
        """
        Run tyrosine in implicit solvent with an instanteneous state switch.
        """
        calibration_settings = dict()
        calibration_settings["temperature"] = 300.0 * unit.kelvin
        calibration_settings["timestep"] = 1.8 * unit.femtosecond
        calibration_settings["pressure"] = 0.101325 * unit.megapascal
        calibration_settings["collision_rate"] = 9.1 / unit.picoseconds
        calibration_settings["pH"] = 7.4
        calibration_settings["solvent"] = "implicit"
        calibration_settings["nsteps_per_trial"] = 0

        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                           integrator, debug=False,
                                           pressure=None, nsteps_per_trial=0, implicit=True)

        mc_titration.calibrate(calibration_settings, iterations=20, mc_every=10, weights_every=1)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_calibration_instantaneous_eq9(self):
        """
        Calibrate (eq 9) tyrosine in implicit solvent with an instantaneous state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = CalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                            integrator, debug=False,
                                            pressure=None, nsteps_per_trial=0, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context, 'eq9')

    def test_tyrosine_calibration_instantaneous_eq12(self):
        """
        Calibrate (eq 12) tyrosine in implicit solvent with an instantaneous state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = CalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                            integrator, debug=False,
                                            pressure=None, nsteps_per_trial=0, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context, 'eq12')

    @skip("Current api incompatible, circular dependency on context")
    def test_tyrosine_calibration_instantaneous_mbar(self):
        """
        Calibrate (MBAR) tyrosine in implicit solvent with an instantaneous state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MBarCalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                                context, integrator, debug=False,
                                                pressure=None, nsteps_per_trial=0, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context)

    def test_tyrosine_ncmc(self):
        """
        Run tyrosine in implicit solvent with an ncmc state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename, integrator, debug=False,
                                           pressure=None, nsteps_per_trial=10, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_calibration_ncmc_eq9(self):
        """
        Calibrate (eq 9) tyrosine in implicit solvent with an ncmc state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = CalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                            integrator, debug=False,
                                            pressure=None, nsteps_per_trial=10, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context, 'eq9')

    def test_tyrosine_calibration_ncmc_eq12(self):
        """
        Calibrate (eq 12) tyrosine in implicit solvent with an ncmc state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = CalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                            integrator, debug=False,
                                            pressure=None, nsteps_per_trial=10, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context, 'eq12')

    @skip("Current api incompatible, circular dependency on context")
    def test_tyrosine_calibration_ncmc_mbar(self):
        """
        Calibrate (MBAR) tyrosine in implicit solvent with an ncmc state switch
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MBarCalibrationTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                                context, integrator, debug=False,
                                            pressure=None, nsteps_per_trial=10, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation
        mc_titration.adapt_weights(context)


class TestAminoAcidsImplicitCalibration(object):

    @classmethod
    def setup(cls):
        settings = dict()
        settings["temperature"] = 300.0 * unit.kelvin
        settings["timestep"] = 1.8 * unit.femtosecond
        settings["pressure"] = 1.0 * unit.hectopascal
        settings["collision_rate"] = 9.1 / unit.picoseconds
        settings["pH"] = 7.4
        settings["solvent"] = "implicit"
        settings["nsteps_per_trial"] = 0
        cls.settings = settings

    def test_calibration(self):
        """
        Calibrate a single amino acid in implicit solvent
        """

        for acid in ("lys", "cys", "tyr", "as4", "gl4", "hip"):
            yield self.calibrate, acid

    def calibrate(self, resname):
        print(resname)
        aac = AminoAcidCalibrator(resname, self.settings, platform_name="CPU", minimize=False)
        print(aac.calibrate(iterations=100, mc_every=26, weights_every=1))

class PeptideImplicitTestCase(TestCase):

    def setUp(self):
        self.temperature = 300.0 * unit.kelvin
        self.pressure = 1.0 * unit.atmospheres
        self.timestep = 1.0 * unit.femtoseconds
        self.collision_rate = 9.1 / unit.picoseconds
        self.pH = 7.4
        self.platform_name = 'CPU'
        testsystems = get_data('edchky_implicit', 'testsystems')
        self.positions = openmm.XmlSerializer.deserialize(
            open('{}/edchky-implicit.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        self.system = openmm.XmlSerializer.deserialize(open('{}/edchky-implicit.sys.xml'.format(testsystems)).read())
        self.prmtop = app.AmberPrmtopFile('{}/edchky-implicit.prmtop'.format(testsystems))
        self.cpin_filename = '{}/edchky-implicit.cpin'.format(testsystems)
        calibration_settings = dict()
        calibration_settings["temperature"] = self.temperature
        calibration_settings["timestep"] = self.timestep
        calibration_settings["pressure"] = self.pressure
        calibration_settings["collision_rate"] = self.collision_rate
        calibration_settings["pH"] = self.pH
        calibration_settings["solvent"] = "implicit"
        calibration_settings["nsteps_per_trial"] = 0
        self.calibration_settings = calibration_settings

    def test_peptide_instantaneous_calibrated(self):
        """
        Run edchky peptide in implicit solvent with an instanteneous state switch. with calibration
        """

        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                           integrator, debug=False,
                                           pressure=None, nsteps_per_trial=0, implicit=True)

        mc_titration.calibrate(self.calibration_settings, iterations=20, mc_every=10, weights_every=1)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation