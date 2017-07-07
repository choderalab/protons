# coding=utf-8
"""Test functionality of the NCMCReporter."""
from protons import app
from protons.app import ncmcreporter as ncr
from simtk import unit, openmm as mm
from protons.app import GBAOABIntegrator, ForceFieldProtonDrive
from . import get_test_data
import uuid
import os


class TestNCMCReporter(object):
    """Tests use cases for ConstantPHSimulation"""

    _default_platform = mm.Platform.getPlatformByName('Reference')

    def test_reports(self):
        """Instantiate a ConstantPHSimulation at 300K/1 atm for a small peptide."""

        pdb = app.PDBxFile(get_test_data('glu_ala_his-solvated-minimized-renamed.cif', 'testsystems/tripeptides'))
        forcefield = app.ForceField('amber10-constph.xml', 'ions_tip3p.xml', 'tip3p.xml')

        system = forcefield.createSystem(pdb.topology, nonbondedMethod=app.PME,
                                         nonbondedCutoff=1.0 * unit.nanometers, constraints=app.HBonds, rigidWater=True,
                                         ewaldErrorTolerance=0.0005)

        temperature = 300 * unit.kelvin
        integrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds, timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=False)
        ncmcintegrator = GBAOABIntegrator(temperature=temperature, collision_rate=1.0 / unit.picoseconds, timestep=2.0 * unit.femtoseconds, constraint_tolerance=1.e-7, external_work=True)

        compound_integrator = mm.CompoundIntegrator()
        compound_integrator.addIntegrator(integrator)
        compound_integrator.addIntegrator(ncmcintegrator)
        pressure = 1.0 * unit.atmosphere

        system.addForce(mm.MonteCarloBarostat(pressure, temperature))
        driver = ForceFieldProtonDrive(temperature, pdb.topology, system, forcefield, ['amber10-constph.xml'], pressure=pressure,
                                       perturbations_per_trial=3)

        simulation = app.ConstantPHSimulation(pdb.topology, system, compound_integrator, driver, platform=self._default_platform)
        simulation.context.setPositions(pdb.positions)
        simulation.context.setVelocitiesToTemperature(temperature)
        filename = uuid.uuid4().hex + ".nc"
        print("Temporary file: ",filename)
        newreporter = ncr.NCMCReporter(filename, 2, shared=False)
        simulation.update_reporters.append(newreporter)

        # Regular MD step
        simulation.step(1)
        # Update the titration states using the uniform proposal
        simulation.update(4)
        # Basic checks for dimension
        assert newreporter.ncfile['Protons/NCMC'].dimensions['update'].size == 2, "There should be 3 updates recorded."
        assert newreporter.ncfile['Protons/NCMC'].dimensions['residue'].size == 2, "There should be 2 residues recorded."
        assert newreporter.ncfile['Protons/NCMC'].dimensions['perturbation'].size == 3, "There should be max 3 perturbations recorded."
        # cleanup
        newreporter.ncfile.close()
        del(newreporter)

        os.remove(filename)



