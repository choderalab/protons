from __future__ import print_function
from constph.calibration import AminoAcidCalibrator
from unittest import TestCase, skipIf
from nose.plugins.skip import SkipTest
import simtk.unit as units
import os


class TestAminoAcidsImplicitCalibration(object):

    @classmethod
    def setup(cls):
        settings = dict()
        settings["temperature"] = 300.0 * units.kelvin
        settings["timestep"] = 1.8 * units.femtosecond
        settings["pressure"] = 1.0 * units.hectopascal
        settings["collision_rate"] = 9.1 / units.picoseconds
        settings["pH"] = 7.4
        settings["solvent"] = "implicit"
        settings["nsteps_per_trial"] = 0
        cls.settings = settings

    def test_calibration(self):
        """
        Calibrate a single amino acid in implicit solvent
        """

        for acid in ("cys", "lys", "glu", "his", "tyr", "asp"):
            yield self.calibrate, acid

    def calibrate(self, resname):
        print(resname)
        aac = AminoAcidCalibrator(resname, self.settings, platform_name="CPU", minimize=False)
        print(aac.calibrate(iterations=1000, mc_every=9, weights_every=1))



class TestAminoAcidsExplicitCalibration(object):

    @classmethod
    def setup(cls):
        settings = dict()
        settings["temperature"] = 300.0 * units.kelvin
        settings["timestep"] = 1.8 * units.femtosecond
        settings["pressure"] = 1.0 * units.hectopascal
        settings["collision_rate"] = 9.1 / units.picoseconds
        settings["nsteps_per_trial"] = 5
        settings["pH"] = 7.4
        settings["solvent"] = "explicit"
        cls.settings = settings

    def test_calibration(self):
        """
        Calibrate a single amino acid in explicit solvent
        """

        for acid in ("cys", "lys", "glu", "his", "tyr", "asp"):
            yield self.calibrate, acid

    def calibrate(self, resname):
        if os.environ.get("TRAVIS", None) == 'true':
            raise SkipTest

        print(resname)
        aac = AminoAcidCalibrator(resname, self.settings, platform_name="CPU", minimize=False)
        print(aac.calibrate(iterations=10, mc_every=9, weights_every=1))