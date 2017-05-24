from openmmtools import testsystems
from simtk import unit
from simtk.openmm import openmm
from protons.integrators import ReferenceGBAOABIntegrator
import pytest
import os


class TestGBAOABIntegrator(object):
    """Tests the GBAOAB integrator work accumulation"""
    def test_protocol_work_accumulation_harmonic_oscillator(self):
        """Testing protocol work accumulation for ExternalPerturbationLangevinIntegrator with HarmonicOscillator
        """
        testsystem = testsystems.HarmonicOscillator()
        parameter_name = 'testsystems_HarmonicOscillator_x0'
        parameter_initial = 0.0 * unit.angstroms
        parameter_final = 10.0 * unit.angstroms
        for platform_name in ['Reference', 'CPU']:
            self.compare_external_protocol_work_accumulation(testsystem, parameter_name, parameter_initial, parameter_final, platform_name=platform_name)

    @pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
    def test_protocol_work_accumulation_waterbox(self):
        """Testing protocol work accumulation for ExternalPerturbationLangevinIntegrator with AlchemicalWaterBox
        """
        from simtk.openmm import app
        parameter_name = 'lambda_electrostatics'
        parameter_initial = 1.0
        parameter_final = 0.0
        platform_names = [ openmm.Platform.getPlatform(index).getName() for index in range(openmm.Platform.getNumPlatforms()) ]
        for nonbonded_method in ['CutoffPeriodic', 'PME']:
            testsystem = testsystems.AlchemicalWaterBox(nonbondedMethod=getattr(app, nonbonded_method))
            for platform_name in platform_names:
                name = '%s %s %s' % (testsystem.name, nonbonded_method, platform_name)
                self.compare_external_protocol_work_accumulation(testsystem, parameter_name, parameter_initial, parameter_final, platform_name=platform_name, name=name)

    @pytest.mark.skipif(os.environ.get("TRAVIS", None) == 'true', reason="Skip slow test on travis.")
    def test_protocol_work_accumulation_waterbox_barostat(self):
        """
        Testing protocol work accumulation for ExternalPerturbationLangevinIntegrator with AlchemicalWaterBox
        with an active barostat. For brevity, only using CutoffPeriodic as the non-bonded method.
        """
        from simtk.openmm import app
        parameter_name = 'lambda_electrostatics'
        parameter_initial = 1.0
        parameter_final = 0.0
        platform_names = [ openmm.Platform.getPlatform(index).getName() for index in range(openmm.Platform.getNumPlatforms()) ]
        nonbonded_method = 'CutoffPeriodic'
        testsystem = testsystems.AlchemicalWaterBox(nonbondedMethod=getattr(app, nonbonded_method))

        # Adding the barostat with a high frequency
        testsystem.system.addForce(openmm.MonteCarloBarostat(1*unit.atmospheres, 300*unit.kelvin, 2))

        for platform_name in platform_names:
            name = '%s %s %s' % (testsystem.name, nonbonded_method, platform_name)
            self.compare_external_protocol_work_accumulation(testsystem, parameter_name, parameter_initial, parameter_final, platform_name=platform_name, name=name)


    def compare_external_protocol_work_accumulation(self, testsystem, parameter_name, parameter_initial, parameter_final, platform_name='Reference', name=None):
        """Compare external work accumulation between Reference and CPU platforms.
        """

        if name is None:
            name = testsystem.name

        from openmmtools.constants import kB
        system, topology = testsystem.system, testsystem.topology
        temperature = 298.0 * unit.kelvin
        platform = openmm.Platform.getPlatformByName(platform_name)

        # TODO: Set precision and determinism if platform is ['OpenCL', 'CUDA']

        nsteps = 20
        kT = kB * temperature
        integrator = ReferenceGBAOABIntegrator(temperature=temperature)
        context = openmm.Context(system, integrator, platform)
        context.setParameter(parameter_name, parameter_initial)
        context.setPositions(testsystem.positions)
        context.setVelocitiesToTemperature(temperature)
        assert(integrator.getGlobalVariableByName('protocol_work') == 0), "Protocol work should be 0 initially"
        integrator.step(1)
        assert(integrator.getGlobalVariableByName('protocol_work') == 0), "There should be no protocol work."

        external_protocol_work = 0.0
        for step in range(nsteps):
            lambda_value = float(step+1) / float(nsteps)
            parameter_value = parameter_initial * (1-lambda_value) + parameter_final * lambda_value
            initial_energy = context.getState(getEnergy=True).getPotentialEnergy()
            context.setParameter(parameter_name, parameter_value)
            final_energy = context.getState(getEnergy=True).getPotentialEnergy()
            external_protocol_work += (final_energy - initial_energy) / kT

            integrator.step(1)
            integrator_protocol_work = integrator.getGlobalVariableByName('protocol_work') * unit.kilojoules_per_mole / kT

            message = '\n'
            message += 'protocol work discrepancy noted for %s on platform %s\n' % (name, platform_name)
            message += 'step %5d : external %16e kT | integrator %16e kT | difference %16e kT' % (step, external_protocol_work, integrator_protocol_work, external_protocol_work - integrator_protocol_work)
            # Test relative tolerance
            assert pytest.approx(external_protocol_work, 0.001) == integrator_protocol_work, message

        del context, integrator