"""Tests for pH curve calculation"""
from __future__ import print_function

from protons.app.pka import (
    HIP as Histidine,
    TYR as Tyrosine,
    AS4 as Aspartic4,
    CYS as Cysteine,
    GL4 as Glutamic4,
    LYS as Lysine,
    GLH as Glutamic,
    ASH as Aspartic,
)
from numpy import linspace
from pytest import approx


class TestHistidine(object):
    """Test the titration curve of histidine"""

    def test_pka_delta(self):
        """
        Weights for histidine at pH == pKa_delta
        """
        weights = Histidine(Histidine.pka_d).populations()
        assert approx(weights[0], weights[1])

    def test_pka_eps(self):
        """
        Weights for histidine at pH == pKa_eps
        """
        weights = Histidine(Histidine.pka_e).populations()
        assert approx(weights[0], weights[2])

    def test_totals(self):
        """
        Fractional concentrations should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            assert approx(1.0, sum(Histidine(ph).populations()))


class TestLysine(object):
    """Test the titration curve of lysine"""

    def test_ph_eq_pka(self):
        """
        Weights for lysine at pH == pKa
        """
        weights = Lysine(Lysine.pka).populations()
        assert approx(weights[0], weights[1])

    def test_ph_greater_than_pka(self):
        """
        Weights for lysine at pH == pKa + 1
        """

        weights = Lysine(Lysine.pka + 1).populations()
        assert approx(10.0 * weights[0], weights[1])

    def test_ph_less_than_pka(self):
        """
        Weights for lysine at pH == pKa - 1
        """

        weights = Lysine(Lysine.pka - 1).populations()
        assert approx(weights[0], 10.0 * weights[1])

    def test_totals(self):
        """
        Fractional concentrations of lysine should always sum to 1.0
        """
        for ph in linspace(-1, 15, 50):
            assert approx(1.0, sum(Lysine(ph).populations()))


class TestTyrosine(object):
    """Test the titration curve of tyrosine"""

    def test_pka(self):
        """
        Weights for tyrosine at pH == pKa
        """
        weights = Tyrosine(Tyrosine.pka).populations()
        assert approx(weights[0], weights[1])

    def test_ph_greater_than_pka(self):
        """
        Weights for tyrosine at pH == pKa + 1
        """

        weights = Tyrosine(Tyrosine.pka + 1).populations()
        assert approx(10.0 * weights[0], weights[1])

    def test_ph_less_than_pka(self):
        """
        Weights for tyrosine at pH == pKa - 1
        """

        weights = Tyrosine(Tyrosine.pka - 1).populations()
        assert approx(weights[0], 10.0 * weights[1])

    def test_totals(self):
        """
        Fractional concentrations of tyrosine should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            assert approx(1.0, sum(Tyrosine(ph).populations()))


class TestCysteine(object):
    """Test the titration curve of cysteine"""

    def test_pka(self):
        """
        Weights for cysteine at pH == pka
        """
        weights = Cysteine(Cysteine.pka).populations()
        assert approx(weights[0], weights[1])

    def test_ph_greater_than_pka(self):
        """
        Weights for cysteine at pH == pKa + 1
        """

        weights = Cysteine(Cysteine.pka + 1).populations()
        assert approx(10.0 * weights[0], weights[1])

    def test_ph_less_than_pka(self):
        """
        Weights for cysteine at pH == pKa - 1
        """

        weights = Cysteine(Cysteine.pka - 1).populations()
        assert approx(weights[0], 10.0 * weights[1])

    def test_totals(self):
        """
        Fractional concentrations of cysteine should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            assert approx(1.0, sum(Cysteine(ph).populations()))


class TestGlutamicAcid4(object):
    """Test the titration curve of glutamic acid (syn/anti restype)"""

    def test_pka(self):
        """
        Weights for glutamic acid at pH == pKa
        """
        weights = Glutamic4(Glutamic4.pka).populations()

        # 4 protonated forms of the same state
        assert approx(weights[0], weights[1] * 4)
        assert approx(weights[0], weights[2] * 4)
        assert approx(weights[0], weights[3] * 4)
        assert approx(weights[0], weights[4] * 4)

    def test_ph_greater_than_pka(self):
        """
        Weights for glutamic acid at pH == pka + 1
        """

        weights = Glutamic4(Glutamic4.pka + 1).populations()

        # 4 protonated forms of the same state
        assert approx(weights[0], 10.0 * weights[1] * 4)
        assert approx(weights[0], 10.0 * weights[2] * 4)
        assert approx(weights[0], 10.0 * weights[3] * 4)
        assert approx(weights[0], 10.0 * weights[4] * 4)

    def test_ph_less_than_pka(self):
        """
        Weights for glutamic acid at pH == pKa - 1
        """

        weights = Glutamic4(Glutamic4.pka - 1).populations()

        # 4 protonated forms of the same state
        assert approx(10.0 * weights[0], 4.0 * weights[1])
        assert approx(10.0 * weights[0], 4.0 * weights[2])
        assert approx(10.0 * weights[0], 4.0 * weights[3])
        assert approx(10.0 * weights[0], 4.0 * weights[4])

    def test_totals(self):
        """
        Fractional concentrations of glutamic acid should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            assert approx(1.0, sum(Glutamic4(ph).populations()))


class TestGlutamicAcid(object):
    """Test the titration curve of glutamic acid (syn/anti restype)"""

    def test_pka(self):
        """
        Weights for glutamic acid at pH == pKa
        """
        weights = Glutamic(Glutamic.pka).populations()

        # Equal weights
        assert approx(weights[0], weights[1])

    def test_ph_greater_than_pka(self):
        """
        Weights for glutamic acid at pH == pka + 1
        """

        weights = Glutamic(Glutamic.pka + 1).populations()

        # protonated forms of the same state should be 10X less
        assert approx(weights[0], 10.0 * weights[1])

    def test_ph_less_than_pka(self):
        """
        Weights for glutamic acid at pH == pKa - 1
        """

        weights = Glutamic(Glutamic.pka - 1).populations()

        # protonated forms of the same state should be 10X more
        assert approx(10.0 * weights[0], weights[1])

    def test_totals(self):
        """
        Fractional concentrations of glutamic acid should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            assert approx(1.0, sum(Glutamic(ph).populations()))


class TestAsparticAcid4(object):
    """Test titration curve of aspartic acid  (syn/anti restype)"""

    def test_pka(self):
        """
        Weights for aspartic acid at pH == pKa
        """
        weights = Aspartic4(Aspartic4.pka).populations()

        # 4 protonated forms of the same state
        assert approx(weights[0], weights[1] * 4)
        assert approx(weights[0], weights[2] * 4)
        assert approx(weights[0], weights[3] * 4)
        assert approx(weights[0], weights[4] * 4)

    def test_ph_greater_than_pka(self):
        """
        Weights for aspartic acid at pH == pKa + 1
        """

        weights = Aspartic4(Aspartic4.pka + 1).populations()

        # 4 protonated forms of the same state
        assert approx(weights[0], 10.0 * weights[1] * 4)
        assert approx(weights[0], 10.0 * weights[2] * 4)
        assert approx(weights[0], 10.0 * weights[3] * 4)
        assert approx(weights[0], 10.0 * weights[4] * 4)

    def test_ph_less_than_pka(self):
        """
        Weights for aspartic acid at pH == pKa - 1
        """

        weights = Aspartic4(Aspartic4.pka - 1).populations()

        # 4 protonated forms of the same state
        assert approx(10.0 * weights[0], 4.0 * weights[1])
        assert approx(10.0 * weights[0], 4.0 * weights[2])
        assert approx(10.0 * weights[0], 4.0 * weights[3])
        assert approx(10.0 * weights[0], 4.0 * weights[4])

    def test_totals(self):
        """
        Fractional concentrations of aspartic acid should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            assert approx(1.0, sum(Aspartic4(ph).populations()))


class TestAsparticAcid(object):
    """Test titration curve of aspartic acid  (syn/anti restype)"""

    def test_pka(self):
        """
        Weights for aspartic acid at pH == pKa
        """
        weights = Aspartic(Aspartic.pka).populations()

        # weights should be equal
        assert approx(weights[0], weights[1])

    def test_ph_greater_than_pka(self):
        """
        Weights for aspartic acid at pH == pKa + 1
        """

        weights = Aspartic(Aspartic.pka + 1).populations()

        # protonated forms of the same state should be 10X less
        assert approx(weights[0], 10.0 * weights[1])

    def test_ph_less_than_pka(self):
        """
        Weights for aspartic acid at pH == pKa - 1
        """

        weights = Aspartic(Aspartic.pka - 1).populations()

        # protonated forms of the same state should be 10X more
        assert approx(10.0 * weights[0], weights[1])

    def test_totals(self):
        """
        Fractional concentrations of aspartic acid should always sum to 1.0
        """

        for ph in linspace(-1, 15, 50):
            assert approx(1.0, sum(Aspartic(ph).populations()))
