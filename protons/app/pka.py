# coding=utf-8
"""Objects for calculation of state populations based on pKa. This library is specific to Amber 10 constant-pH force field amino acid residues,
though not all of those residues are included here.
"""
import abc


class PopulationCalculator(metaclass=abc.ABCMeta):
    """
    Abstract base class for determining state populations from a pH curve
    """

    @abc.abstractmethod
    def populations(self):
        """ Return population of each state of the amino acid.

        Returns
        -------
        list of float
        """
        raise NotImplemented("This is an abstract class.")


class HistidineType(PopulationCalculator):
    """
    Amber constant-pH HIP residue state weights at given pH
    """

    pka_d = 6.5
    pka_e = 7.1

    def __init__(self, pH):
        self.kd = pow(10.0, pH - HistidineType.pka_d)
        self.ke = pow(10.0, pH - HistidineType.pka_e)

    def hip_concentration(self):
        """
        Concentration of the doubly protonated form
        """
        return 1.0 / (self.ke + self.kd + 1.0)

    def hie_concentration(self):
        """
        Concentration of the epsilon protonated form
        """
        return self.ke / (self.ke + self.kd + 1.0)

    def hid_concentration(self):
        """
        Concentration of the delta pronated form
        """
        return self.kd / (self.ke + self.kd + 1.0)

    def populations(self):
        """
        Returns
        -------
        list of float : state weights in order of AMBER cpH residue
        """
        return [
            self.hip_concentration(),
            self.hid_concentration(),
            self.hie_concentration(),
        ]


HIP = HistidineType


class SynAntiAcidType(PopulationCalculator):
    """
    Amber constant-pH AS4/GL4 syn/anti residue state weights at given pH
    """

    pka = 0.0

    def __init__(self, pH):
        self.k = pow(10.0, pH - self.pka)

    def protonated_concentration(self):
        """
        Concentration of protonated form
        """
        return 1.0 / (self.k + 1.0)

    def deprotonated_concenration(self):
        """
        Concentration of deprotonated form
        """
        return self.k / (self.k + 1.0)

    def populations(self):
        """
        Returns
        -------
        list of float : state weights in order of AMBER cpH residue
        """
        acid = self.protonated_concentration() / 4.0
        return [self.deprotonated_concenration(), acid, acid, acid, acid]


class AS4(SynAntiAcidType):
    """Aspartic acid with syn/anti protons."""

    pka = 4.0


class GL4(SynAntiAcidType):
    """Glutamic acid with syn/anti protons"""

    pka = 4.4


class AcidType(PopulationCalculator):
    """
    Amber constant-pH acid residue residue state weights at given pH
    """

    pka = 0.0

    def __init__(self, pH):
        self.k = pow(10.0, pH - self.pka)

    def protonated_concentration(self):
        """
        Concentration of protonated form
        """
        return 1.0 / (self.k + 1.0)

    def deprotonated_concenration(self):
        """
        Concentration of deprotonated form
        """
        return self.k / (self.k + 1.0)

    def populations(self):
        """
        Returns
        -------
        list of float : state weights in order of AMBER cpH residue
        """
        return [self.deprotonated_concenration(), self.protonated_concentration()]


class GLH(AcidType):
    "Glutamic acid residue"
    pka = 4.4


class ASH(AcidType):
    "Aspartic acid residue"
    pka = 4.0


class BasicType(PopulationCalculator):
    """
    Amber constant-pH basic residue (e.g. LYS) state weights at given pH
    """

    pka = 10.4

    def __init__(self, pH):
        self.k = pow(10.0, pH - self.pka)

    def protonated_concentration(self):
        """
        Concentration of protonated form
        """
        return 1.0 / (self.k + 1.0)

    def deprotonated_concenration(self):
        """
        Concentration of deprotonated form
        """
        return self.k / (self.k + 1.0)

    def populations(self):
        """
            Returns
            -------
            list of float : state weights in order of AMBER cpH residue
        """
        return [self.protonated_concentration(), self.deprotonated_concenration()]


class LYS(BasicType):
    """Lysine residue"""

    pka = 10.4


class TYR(BasicType):
    """Tyrosine residue."""

    pka = 9.6


class CYS(BasicType):
    """
    Cysteine residue.
    """

    pka = 8.5


available_pkas = {
    "ASH": ASH,
    "GLH": GLH,
    "AS4": AS4,
    "GL4": GL4,
    "CYS": CYS,
    "HIP": HIP,
    "HIS": HIP,
    "TYR": TYR,
    "LYS": LYS,
}
