from saltswap.wrappers import Swapper
import numpy as np


def update_fractional_stateVector(
    swapper: Swapper,
    new_vector: np.ndarray,
    fraction: float = 1.0,
    set_vector_indices: bool = True,
):
    """Given the old state vector and the new state vector, update ions accordingly.


    Parameters
    ----------
    swapper - saltswap Swapper class to update.
    new_vector - saltswap state vector for the new state.
    fraction - for fractional updates, use number between 0 and 1. By default, 1.0.
    set_vector_indices - set the stateVector indices to the new vector.

    Note
    ----
    requires call to updateParametersInContext

    """
    # No need for updates if this is true.
    if np.all(swapper.stateVector == new_vector):
        return

    ion_parameters = {
        0: swapper.water_parameters,
        1: swapper.cation_parameters,
        2: swapper.anion_parameters,
    }

    for i, (from_ion_state, to_ion_state) in enumerate(
        zip(swapper.stateVector, new_vector)
    ):
        # Save time if states aren't changing
        if from_ion_state == to_ion_state:
            continue

        from_parameter = ion_parameters[from_ion_state]
        to_parameter = ion_parameters[to_ion_state]
        swapper.update_fractional_ion(i, from_parameter, to_parameter, fraction)

    if set_vector_indices:
        swapper.stateVector = new_vector
