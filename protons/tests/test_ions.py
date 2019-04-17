from copy import deepcopy
from protons.app import ions
import numpy as np


class TestIonSwapping:
    """This class contains some simulation-independent testing features of schemes for selecting what ions need to be
    added/removed from a simulation to facilitate charge changes."""

    histidine = np.asarray(
        [0, 0, +1], dtype=int
    )  # Has two neutral states, and one positive state
    aspartate = np.asarray(
        [-1, 0, 0, 0, 0], dtype=int
    )  # Has 4 neutral syn/anti hydrogen positions, also covers glutamate
    lysine = np.asarray([0, +1], dtype=int)
    tyrosine = np.asarray([0, -1], dtype=int)
    ash = np.asarray([-1, 0], dtype=int)

    diprotic_acid = np.asarray([-2, -1, -1, 0], dtype=int)
    diprotic_base = np.asarray([0, 1, 1, 2], dtype=int)
    zwitter_one = np.asarray([1, 0, 0, -1], dtype=int)
    zwitter_two = np.asarray(
        [-2, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 2], dtype=int
    )
    random_order = np.asarray([3, -1, 2, 1, 0, 0, 2, 2, -1, 1], dtype=int)

    residues = {
        "His": histidine,
        "Ash": ash,
        "As4": aspartate,
        "Gl4": deepcopy(aspartate),
        "Cys": deepcopy(tyrosine),
        "Tyr": tyrosine,
        "Lys": lysine,
        "Diprotic acid": diprotic_acid,
        "Diprotic base": diprotic_base,
        "Zwitter ion": zwitter_one,
        "Double zwitter ion": zwitter_two,
        "Random": random_order,
    }

    def test_no_ions(self):
        """Ensure no ions are added when the NO_IONS rule is used"""
        for resname, rescharges in self.residues.items():
            for charge in rescharges:
                ncat, nani = ions.choose_neutralizing_ions_by_method(
                    rescharges[0], charge, ions.NeutralChargeRule.NO_IONS
                )
                assert (
                    ncat + nani + charge == charge
                ), "Ions were added when they should not have been."

    def test_counter_ions(self):
        """Ensure no charge is added when COUNTER_IONS rule is used"""
        for resname, rescharges in self.residues.items():
            for charge in rescharges:
                ncat, nani = ions.choose_neutralizing_ions_by_method(
                    rescharges[0], charge, ions.NeutralChargeRule.COUNTER_IONS
                )
                assert (
                    ncat - nani + charge == rescharges[0]
                ), "Charge was added when it should not have been."

    def test_replacement_ions(self):
        """Ensure no charge is added when COUNTER_IONS rule is used"""
        for resname, rescharges in self.residues.items():
            for charge in rescharges:
                ncat, nani = ions.choose_neutralizing_ions_by_method(
                    rescharges[0], charge, ions.NeutralChargeRule.REPLACEMENT_IONS
                )
                assert (
                    ncat - nani + charge == rescharges[0]
                ), "Charge was added when it should not have been."
