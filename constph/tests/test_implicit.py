from __future__ import print_function
from simtk import unit, openmm
from simtk.openmm import app
from constph.constph import *
from .helper_func import *
from unittest import TestCase, skip

class TestImplicit(TestCase):
    
    def setUp(self):
        # Precalculate and set up a system that will be shared for all tests
        self.temperature = 300.0 * unit.kelvin
        self.pressure = 1.0 * unit.atmospheres
        self.timestep = 1.0 * unit.femtoseconds
        self.collision_rate = 9.1 / unit.picoseconds
        self.pH = 9.6
        self.platform_name = 'CPU'
        testsystems = 'constph/tests/testsystems/tyr_implicit'
        self.positions = openmm.XmlSerializer.deserialize(open('{}/tyr.state.xml'.format(testsystems)).read()).getPositions(asNumpy=True)
        self.system = openmm.XmlSerializer.deserialize(open('{}/tyr.sys.xml'.format(testsystems)).read())
        self.prmtop = app.AmberPrmtopFile('{}/tyr.prmtop'.format(testsystems))
        self.cpin_filename = '{}/tyr.cpin'.format(testsystems)
        
    def test_tyrosine_instantaneous(self):
        """
        Perform a single timestep and single instantenous titration attempt.
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename, integrator, debug=False,
                                           pressure=None, nsteps_per_trial=0, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation

    def test_tyrosine_ncmc(self):
        """
        Perform a single timestep and single instantenous titration attempt.
        """
        integrator = openmm.LangevinIntegrator(self.temperature, self.collision_rate, self.timestep)
        mc_titration = MonteCarloTitration(self.system, self.temperature, self.pH, self.prmtop, self.cpin_filename, integrator, debug=False,
                                           pressure=None, nsteps_per_trial=10, implicit=True)
        platform = openmm.Platform.getPlatformByName('CPU')
        context = openmm.Context(self.system, mc_titration.compound_integrator, platform)
        context.setPositions(self.positions)  # set to minimized positions
        integrator.step(10)  # MD
        mc_titration.update(context)  # protonation


        
