from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph.constph import MonteCarloTitration, CalibrationTitration, MBarCalibrationTitration
from unittest import TestCase, skip
from . import get_data


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

    def test_tyrosine_ncmc_VV(self):
        """
        Run tyrosine in implicit solvent with an ncmc state switch with VelocityVerlet
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename,
                                           integrator, debug=False,
                                           pressure=None, nsteps_per_trial=10, implicit=True, vvvr=False)
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