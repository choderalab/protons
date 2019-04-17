from typing import Tuple
from enum import Enum


class NeutralChargeRule(Enum):
    """
    Strict enumeration of different methods to maintain charge neutrality for single residues.
    """

    # Ions will not be used to neutralize system. This means OpenMM will add a neutralizing background charge.
    NO_IONS = 0

    # Inserted ions will be countered, and removed ions will be accompanied by removing a counter charge.
    COUNTER_IONS = 1

    # Removed charges will be replaced, and added charges will replace a charge in solvent.
    REPLACEMENT_IONS = 2


def choose_neutralizing_ions_by_method(
    reference_state_charge: int, to_state_charge: int, rule: NeutralChargeRule
) -> Tuple[int, int]:
    """Pick ions using specific rule for single residue changes.

    Notes
    -----
    To ensure consistency, for multiple changes, calculate result for individual residues separately and sum up the
    delta ion results.

    Parameters
    ----------
    reference_state_charge - The charge of a reference state (assumed to be charge neutral)
    to_state_charge - Charge of a state that needs to be neutralized according to rule.
    rule - The rule (see ``NeutralChargeRule`` used to determine ionic changes.

    Returns
    -------
    delta_cations, delta_anions
    """

    # No ions
    if rule is NeutralChargeRule.NO_IONS:
        return 0, 0

    # Add or remove a counter charge
    elif rule is NeutralChargeRule.COUNTER_IONS:
        return _delta_ions_by_counter_charge(reference_state_charge, to_state_charge)

    # Add or remove a replacement charge
    elif rule is NeutralChargeRule.REPLACEMENT_IONS:
        return _delta_ions_by_replacement_charge(
            reference_state_charge, to_state_charge
        )


def _delta_ions_by_counter_charge(
    reference_state_charge: int, to_state_charge: int
) -> Tuple[int, int]:
    """Calculate the change in ionic composition between titration states using counter ion.

    Parameters
    ----------
    reference_state_charge - the charge of a single reference state for

    """

    delta_cation = 0
    delta_anion = 0

    charge_to_counter = to_state_charge - reference_state_charge

    counter = 0
    while abs(charge_to_counter) > 0:
        # The protonation state change annihilates a positive charge
        if (reference_state_charge > 0 >= to_state_charge) or (
            0 < to_state_charge < reference_state_charge
        ):
            # annihilate a solvent anion
            delta_anion -= 1
            charge_to_counter += 1
            reference_state_charge -= (
                1
            )  # One part of the initial charge has been countered

        # The protonation state change annihilates a negative charge
        elif reference_state_charge < 0 <= to_state_charge or (
            0 > to_state_charge > reference_state_charge
        ):
            # annihilate a solvent cation
            delta_cation -= 1
            charge_to_counter -= 1
            reference_state_charge += 1

        # The protonation state change adds a negative charge
        elif reference_state_charge == 0 > to_state_charge or (
            0 > reference_state_charge > to_state_charge
        ):
            # add a positive charge
            delta_cation += 1
            charge_to_counter += 1
            reference_state_charge -= 1

        # The protonation state adds a positive charge
        elif (reference_state_charge == 0 < to_state_charge) or (
            0 < reference_state_charge < to_state_charge
        ):
            # add an anion
            delta_anion += 1
            charge_to_counter -= 1
            reference_state_charge += 1
        else:
            raise ValueError("Impossible scenario reached.")

        counter += 1
        if counter > 10000:
            raise RuntimeError(
                "Infinite while loop predicted for salt resolution. Halting."
            )

    return delta_cation, delta_anion


def _delta_ions_by_replacement_charge(
    reference_state_charge: int, to_state_charge: int
) -> Tuple[int, int]:
    """Calculate the change in ionic composition between titration states by adding replacement ion.

    N.B.: This is similar to the approach employed by Chen and Roux 2015. TODO add reference.
    """

    # Note that we don't allow for direct transitions between ions of different charge.

    delta_cation = 0
    delta_anion = 0

    charge_to_counter = to_state_charge - reference_state_charge

    counter = 0
    while abs(charge_to_counter) > 0:
        # The protonation state change annihilates a positive charge
        if (reference_state_charge > 0 >= to_state_charge) or (
            0 < to_state_charge < reference_state_charge
        ):
            delta_cation += 1
            charge_to_counter += 1
            reference_state_charge -= (
                1
            )  # One part of the initial charge has been countered

        # The protonation state change annihilates a negative charge
        elif reference_state_charge < 0 <= to_state_charge or (
            0 > to_state_charge > reference_state_charge
        ):
            delta_anion += 1
            charge_to_counter -= 1
            reference_state_charge += 1

        # The protonation state change adds a negative charge
        elif reference_state_charge == 0 > to_state_charge or (
            0 > reference_state_charge > to_state_charge
        ):
            delta_anion -= 1
            charge_to_counter += 1
            reference_state_charge -= 1

        # The protonation state adds a positive charge
        elif (reference_state_charge == 0 < to_state_charge) or (
            0 < reference_state_charge < to_state_charge
        ):
            # remove cation
            delta_cation -= 1
            charge_to_counter -= 1
            reference_state_charge += 1
        else:
            raise ValueError("Impossible scenario reached.")

        counter += 1
        if counter > 10000:
            raise RuntimeError(
                "Infinite while loop predicted for salt resolution. Halting."
            )

    return delta_cation, delta_anion
