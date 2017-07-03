"""Tests for pH curve calculation"""
from __future__ import print_function

from protons.app.calibration import Histidine, Lysine, Tyrosine, Aspartic4, Cysteine, Glutamic4
from numpy import linspace
from pytest import approx


class TestHistidine(object):
    """Test the titration curve of histidine"""

    def test_pka_delta(self):
        """
        Weights for histidine at pH == pKa_delta
        """
        weights = Histidine(Histidine.pka_d).populations()
        approx(weights[0], weights[1])

    def test_pka_eps(self):
        """
        Weights for histidine at pH == pKa_eps
        """
        weights = Histidine(Histidine.pka_e).populations()
        approx(weights[0], weights[2])

    def test_totals(self):
        """
        Fractional concentrations should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            approx(1.0, sum(Histidine(ph).populations()))


class TestLysine(object):
    """Test the titration curve of lysine"""

    def test_ph_eq_pka(self):
        """
        Weights for lysine at pH == pKa
        """
        weights = Lysine(Lysine.pka).populations()
        approx(weights[0], weights[1])

    def test_ph_greater_than_pka(self):
        """
        Weights for lysine at pH == pKa + 1
        """

        weights = Lysine(Lysine.pka + 1).populations()
        approx(10.0 * weights[0], weights[1])

    def test_ph_less_than_pka(self):
        """
        Weights for lysine at pH == pKa - 1
        """

        weights = Lysine(Lysine.pka - 1).populations()
        approx(weights[0], 10.0 * weights[1])

    def test_totals(self):
        """
        Fractional concentrations of lysine should always sum to 1.0
        """
        for ph in linspace(-1, 15, 50):
            approx(1.0, sum(Lysine(ph).populations()))


class TestTyrosine(object):
    """Test the titration curve of tyrosine"""

    def test_pka(self):
        """
        Weights for tyrosine at pH == pKa
        """
        weights = Tyrosine(Tyrosine.pka).populations()
        approx(weights[0], weights[1])

    def test_ph_greater_than_pka(self):
        """
        Weights for tyrosine at pH == pKa + 1
        """

        weights = Tyrosine(Tyrosine.pka + 1).populations()
        approx(10.0 * weights[0], weights[1])

    def test_ph_less_than_pka(self):
        """
        Weights for tyrosine at pH == pKa - 1
        """

        weights = Tyrosine(Tyrosine.pka - 1).populations()
        approx(weights[0], 10.0 * weights[1])

    def test_totals(self):
        """
        Fractional concentrations of tyrosine should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            approx(1.0, sum(Tyrosine(ph).populations()))


class TestCysteine(object):
    """Test the titration curve of cysteine"""

    def test_pka(self):
        """
        Weights for cysteine at pH == pka
        """
        weights = Cysteine(Cysteine.pka).populations()
        approx(weights[0], weights[1])

    def test_ph_greater_than_pka(self):
        """
        Weights for cysteine at pH == pKa + 1
        """

        weights = Cysteine(Cysteine.pka + 1).populations()
        approx(10.0 * weights[0], weights[1])

    def test_ph_less_than_pka(self):
        """
        Weights for cysteine at pH == pKa - 1
        """

        weights = Cysteine(Cysteine.pka - 1).populations()
        approx(weights[0], 10.0 * weights[1])

    def test_totals(self):
        """
        Fractional concentrations of cysteine should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            approx(1.0, sum(Cysteine(ph).populations()))


class TestGlutamicAcid(object):
    """Test the titration curve of glutamic acid"""

    def test_pka(self):
        """
        Weights for glutamic acid at pH == pKa
        """
        weights = Glutamic4(Glutamic4.pka).populations()

        # 4 protonated forms of the same state
        approx(weights[0], weights[1]*4)
        approx(weights[0], weights[2]*4)
        approx(weights[0], weights[3]*4)
        approx(weights[0], weights[4]*4)

    def test_ph_greater_than_pka(self):
        """
        Weights for glutamic acid at pH == pka + 1
        """

        weights = Glutamic4(Glutamic4.pka + 1).populations()

        # 4 protonated forms of the same state
        approx(weights[0], 10.0 * weights[1]*4)
        approx(weights[0], 10.0 * weights[2]*4)
        approx(weights[0], 10.0 * weights[3]*4)
        approx(weights[0], 10.0 * weights[4]*4)

    def test_ph_less_than_pka(self):
        """
        Weights for glutamic acid at pH == pKa - 1
        """

        weights = Glutamic4(Glutamic4.pka - 1).populations()

        # 4 protonated forms of the same state
        approx(10.0 * weights[0], 4.0 * weights[1])
        approx(10.0 * weights[0], 4.0 * weights[2])
        approx(10.0 * weights[0], 4.0 * weights[3])
        approx(10.0 * weights[0], 4.0 * weights[4])

    def test_totals(self):
        """
        Fractional concentrations of glutamic acid should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            approx(1.0, sum(Glutamic4(ph).populations()))


class TestAsparticAcid(object):
    """Test titration curve of aspartic acid"""
    def test_pka(self):
        """
        Weights for aspartic acid at pH == pKa
        """
        weights = Aspartic4(Aspartic4.pka).populations()

        # 4 protonated forms of the same state
        approx(weights[0], weights[1]*4)
        approx(weights[0], weights[2]*4)
        approx(weights[0], weights[3]*4)
        approx(weights[0], weights[4]*4)

    def test_ph_greater_than_pka(self):
        """
        Weights for aspartic acid at pH == pKa + 1
        """

        weights = Aspartic4(Aspartic4.pka + 1).populations()

        # 4 protonated forms of the same state
        approx(weights[0], 10.0 * weights[1]*4)
        approx(weights[0], 10.0 * weights[2]*4)
        approx(weights[0], 10.0 * weights[3]*4)
        approx(weights[0], 10.0 * weights[4]*4)

    def test_ph_less_than_pka(self):
        """
        Weights for aspartic acid at pH == pKa - 1
        """

        weights = Aspartic4(Aspartic4.pka - 1).populations()

        # 4 protonated forms of the same state
        approx(10.0 * weights[0], 4.0 * weights[1])
        approx(10.0 * weights[0], 4.0 * weights[2])
        approx(10.0 * weights[0], 4.0 * weights[3])
        approx(10.0 * weights[0], 4.0 * weights[4])

    def test_totals(self):
        """
        Fractional concentrations of aspartic acid should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            approx(1.0, sum(Aspartic4(ph).populations()))