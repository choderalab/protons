# coding=utf-8
"""Objects for calculation of state populations based on pKa."""
import abc


class PopulationCalculator(metaclass=abc.ABCMeta):
    """
    Abstract base class for determining state populations from a pH curve
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def populations(self):
        """ Return population of each state of the amino acid.

        Returns
        -------
        list of float
        """
        return NotImplemented


class Histidine(PopulationCalculator):
    """
    Amber constant-pH HIP residue state weights at given pH
    """
    pka_d = 6.5
    pka_e = 7.1

    def __init__(self, pH):
        self.kd = pow(10.0, pH - Histidine.pka_d)
        self.ke = pow(10.0, pH - Histidine.pka_e)

    def hip_concentration(self):
        """
        Concentration of the doubly protonated form
        """
        return 1.0/(self.ke + self.kd + 1.0)

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
        return [self.hip_concentration(), self.hid_concentration(), self.hie_concentration()]


class Aspartic4(PopulationCalculator):
    """
    Amber constant-pH AS4 residue state weights at given pH
    """
    pka = 4.0

    def __init__(self, pH):
        self.k = pow(10.0, pH - self.pka)

    def protonated_concentration(self):
        """
        Concentration of protonated form
        """
        return 1.0/(self.k + 1.0)

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


class Glutamic4(Aspartic4):
    """
    Amber constant-pH GL4 residue state weights at given pH
    """
    pka = 4.4


class Lysine(Aspartic4):
    """
    Amber constant-pH LYS residue state weights at given pH
    """
    pka = 10.4

    def populations(self):
        """
            Returns
            -------
            list of float : state weights in order of AMBER cpH residue
        """
        return [self.protonated_concentration(), self.deprotonated_concenration()]


class Tyrosine(Lysine):
    """
    Amber constant-pH TYR residue state weights at given pH
    """
    pka = 9.6


class Cysteine(Lysine):
    """
    Amber constant-pH CYS residue state weights at given pH
    """
    pka = 8.5
