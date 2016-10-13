"""Tests for pH curve calculation"""
from __future__ import print_function

from protons.calibration import Histidine, Lysine, Tyrosine, Aspartic4, Cysteine, Glutamic4
from numpy import linspace
from pytest import approx


class TestHistidine(object):

    def test_pKa_delta(self):
        """
        Weights for histidine at pH == pKa_delta
        """
        weights = Histidine(Histidine.pKa_d).populations()
        approx(weights[0], weights[1])

    def test_pKa_eps(self):
        """
        Weights for histidine at pH == pKa_eps
        """
        weights = Histidine(Histidine.pKa_e).populations()
        approx(weights[0], weights[2])

    def test_totals(self):
        """
        Fractional concentrations should always sum to 1.0
        """

        for pH in linspace(-1, 15, 50):
            approx(1.0, sum(Histidine(pH).populations()))


class TestLysine(object):

    def test_pH_eq_pKa(self):
        """
        Weights for lysine at pH == pKa
        """
        weights = Lysine(Lysine.pKa).populations()
        approx(weights[0], weights[1])

    def test_pH_greater_than_pKa(self):
        """
        Weights for lysine at pH == pKa + 1
        """

        weights = Lysine(Lysine.pKa + 1).populations()
        approx(10.0 * weights[0], weights[1])

    def test_pH_less_than_pKa(self):
        """
        Weights for lysine at pH == pKa - 1
        """

        weights = Lysine(Lysine.pKa - 1).populations()
        approx(weights[0], 10.0 * weights[1])

    def test_totals(self):
        """
        Fractional concentrations of lysine should always sum to 1.0
        """
        for pH in linspace(-1, 15, 50):
            approx(1.0, sum(Lysine(pH).populations()))


class TyrosineTestCase(object):

    def test_pKa(self):
        """
        Weights for tyrosine at pH == pKa
        """
        weights = Tyrosine(Tyrosine.pKa).populations()
        approx(weights[0], weights[1])

    def test_pH_greater_than_pKa(self):
        """
        Weights for tyrosine at pH == pKa + 1
        """

        weights = Tyrosine(Tyrosine.pKa + 1).populations()
        approx(10.0 * weights[0], weights[1])

    def test_pH_less_than_pKa(self):
        """
        Weights for tyrosine at pH == pKa - 1
        """

        weights = Tyrosine(Tyrosine.pKa - 1).populations()
        approx(weights[0], 10.0 * weights[1])

    def test_totals(self):
        """
        Fractional concentrations of tyrosine should always sum to 1.0
        """

        for pH in linspace(-1, 15, 50):
            approx(1.0, sum(Tyrosine(pH).populations()))


class CysteineTestCase(object):

    def test_pKa(self):
        """
        Weights for cysteine at pH == pKa
        """
        weights = Cysteine(Cysteine.pKa).populations()
        approx(weights[0], weights[1])

    def test_pH_greater_than_pKa(self):
        """
        Weights for cysteine at pH == pKa + 1
        """

        weights = Cysteine(Cysteine.pKa + 1).populations()
        approx(10.0 * weights[0], weights[1])

    def test_pH_less_than_pKa(self):
        """
        Weights for cysteine at pH == pKa - 1
        """

        weights = Cysteine(Cysteine.pKa - 1).populations()
        approx(weights[0], 10.0 * weights[1])

    def test_totals(self):
        """
        Fractional concentrations of cysteine should always sum to 1.0
        """

        for pH in linspace(-1, 15, 50):
            approx(1.0, sum(Cysteine(pH).populations()))


class GlutamicAcidTestCase(object):

    def test_pKa(self):
        """
        Weights for glutamic acid at pH == pKa
        """
        weights = Glutamic4(Glutamic4.pKa).populations()

        # 4 protonated forms of the same state
        approx(weights[0], weights[1]*4)
        approx(weights[0], weights[2]*4)
        approx(weights[0], weights[3]*4)
        approx(weights[0], weights[4]*4)

    def test_pH_greater_than_pKa(self):
        """
        Weights for glutamic acid at pH == pKa + 1
        """

        weights = Glutamic4(Glutamic4.pKa + 1).populations()

        # 4 protonated forms of the same state
        approx(weights[0], 10.0 * weights[1]*4)
        approx(weights[0], 10.0 * weights[2]*4)
        approx(weights[0], 10.0 * weights[3]*4)
        approx(weights[0], 10.0 * weights[4]*4)

    def test_pH_less_than_pKa(self):
        """
        Weights for glutamic acid at pH == pKa - 1
        """

        weights = Glutamic4(Glutamic4.pKa - 1).populations()

        # 4 protonated forms of the same state
        approx(10.0 * weights[0], 4.0 * weights[1])
        approx(10.0 * weights[0], 4.0 * weights[2])
        approx(10.0 * weights[0], 4.0 * weights[3])
        approx(10.0 * weights[0], 4.0 * weights[4])

    def test_totals(self):
        """
        Fractional concentrations of glutamic acid should always sum to 1.0
        """

        for pH in linspace(-1, 15, 50):
            approx(1.0, sum(Glutamic4(pH).populations()))


class AsparticAcidTestCase(object):

    def test_pKa(self):
        """
        Weights for aspartic acid at pH == pKa
        """
        weights = Aspartic4(Aspartic4.pKa).populations()

        # 4 protonated forms of the same state
        approx(weights[0], weights[1]*4)
        approx(weights[0], weights[2]*4)
        approx(weights[0], weights[3]*4)
        approx(weights[0], weights[4]*4)

    def test_pH_greater_than_pKa(self):
        """
        Weights for aspartic acid at pH == pKa + 1
        """

        weights = Aspartic4(Aspartic4.pKa + 1).populations()

        # 4 protonated forms of the same state
        approx(weights[0], 10.0 * weights[1]*4)
        approx(weights[0], 10.0 * weights[2]*4)
        approx(weights[0], 10.0 * weights[3]*4)
        approx(weights[0], 10.0 * weights[4]*4)

    def test_pH_less_than_pKa(self):
        """
        Weights for aspartic acid at pH == pKa - 1
        """

        weights = Aspartic4(Aspartic4.pKa - 1).populations()

        # 4 protonated forms of the same state
        approx(10.0 * weights[0], 4.0 * weights[1])
        approx(10.0 * weights[0], 4.0 * weights[2])
        approx(10.0 * weights[0], 4.0 * weights[3])
        approx(10.0 * weights[0], 4.0 * weights[4])

    def test_totals(self):
        """
        Fractional concentrations of aspartic acid should always sum to 1.0
        """

        for pH in linspace(-1, 15, 50):
            approx(1.0, sum(Aspartic4(pH).populations()))