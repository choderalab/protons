"""Processing of netCDF files into structures data frames for further analysis and plotting."""
import pandas as pd
from netCDF4 import Dataset
from typing import List, Union
from tqdm import tqdm, trange
import numpy as np
from protons.app.driver import SAMSApproach


def dataset_to_dataframe(
    datasets: List[str], return_as_merged: bool = True
) -> List[pd.DataFrame]:
    """
    Convert a netCDF dataset with protons data to a pandas dataframe

    Parameters
    ----------
    datasets

    Returns
    -------

    Notes
    -----

    The cumulative_work and work_per_step columns contain lists of floats as entries, and may prove difficult to
    serialize to every format. CSV seems robust enough to do the trick though.
    """
    df_to_merge: List[pd.DataFrame] = list()
    for ncfile in tqdm(datasets, desc="file"):
        ds = Dataset(ncfile, "r")
        work = ds["Protons/NCMC/total_work"][:]
        init_states = ds["Protons/NCMC/initial_state"][:, :]
        prop_states = ds["Protons/NCMC/proposed_state"][:, :]
        final_states = ds["Protons/Titration/state"][:, :]
        # TODO add macro in case no sams
        res_names = ds["Protons/Metadata/residue_name"][:]
        state_charges = ds["Protons/Metadata/total_charge"][:, :]

        n_states_per_res: List[int] = _deduce_states_from_metadata(ds)
        # Define macroscopic states for the system from combinations of individual residue states
        # Multires macrostates indexed by the per residue state
        macrostate_labels = np.arange(np.product(n_states_per_res)).reshape(
            n_states_per_res
        )
        # Make tuples for indexing array
        init_macro = _micro_to_macro_labels(init_states, macrostate_labels)
        prop_macro = _micro_to_macro_labels(prop_states, macrostate_labels)
        final_macro = _micro_to_macro_labels(final_states, macrostate_labels)

        update = ds["Protons/NCMC/update"][:]
        logp = ds["Protons/NCMC/logp_accept"][:]
        logp[logp > 0.0] = 0.0
        p_accept = np.exp(logp)
        data_dict = {
            "update": update,
            "work": work,
            "source": [ncfile] * update.size,
            "logp_accept": logp,
            "p_accept": p_accept,
        }

        for r, residue in enumerate(res_names):
            data_dict[f"init_state_{residue}"] = init_states[:, r]
            data_dict[f"prop_state_{residue}"] = prop_states[:, r]
            data_dict[f"final_state_{residue}"] = final_states[:, r]
            data_dict[f"init_charge_{residue}"] = [
                int(round(state_charges[r, s])) for s in init_states[:, r]
            ]
            data_dict[f"prop_charge_{residue}"] = [
                int(round(state_charges[r, s])) for s in prop_states[:, r]
            ]
            data_dict[f"final_charge_{residue}"] = [
                int(round(state_charges[r, s])) for s in final_states[:, r]
            ]

        # Add work data if present
        try:
            cumulative_work = ds["Protons/NCMC/cumulative_work"][:, :]
            data_dict["cumulative_work"] = cumulative_work[:].tolist()
            data_dict["work_per_step"] = np.diff(cumulative_work[:, :], axis=1).tolist()
            protocol_length = np.asarray(
                [arr.size - np.ma.count_masked(arr) for arr in cumulative_work[:]]
            )
            data_dict["protocol_length"] = protocol_length
            data_dict["efficiency"] = p_accept / protocol_length
        except IndexError:
            pass

        # Add SAMS data if present
        try:
            approach = SAMSApproach(ds["Protons/SAMS/approach"][0])
            # Redo macrostate labels if one residue was specified
            if approach is SAMSApproach.ONE_RESIDUE:
                g_index = ds["Protons/SAMS/group_index"][0]
                init_macro = init_states[:, g_index]
                prop_macro = prop_states[:, g_index]
                final_macro = final_states[:, g_index]
            elif approach is SAMSApproach.MULTI_RESIDUE:
                # macrostate labels are correct
                pass
            else:
                raise NotImplementedError(
                    "Processing can't yet handle {}".format(approach)
                )

            stage = ds["Protons/SAMS/stage"][:]
            adaptation = ds["Protons/SAMS/adaptation"][:]
            flatness = ds["Protons/SAMS/flatness"][:]
            gks = ds["Protons/SAMS/g_k"][:, :]
            data_dict["stage"] = stage
            data_dict["adaptation"] = adaptation
            data_dict["flatness"] = flatness

            for i in range(gks.shape[-1]):
                data_dict[f"bias_state_{i}"] = gks[:, i]
        except KeyError:
            # Set constant g_k bias from metadata
            gk_per_macrostate = np.zeros(macrostate_labels.size)
            for m in range(macrostate_labels.size):
                microstates = np.argwhere(macrostate_labels == m).flatten()
                for res, state in enumerate(microstates):
                    gk_per_macrostate[m] += ds["Protons/Metadata/g_k"][res, state]

            for i, gk in enumerate(gk_per_macrostate):
                data_dict[f"bias_state_{i}"] = [gk] * update.size

        # Release file

        # After SAMS processing, macrostate labels are final
        data_dict["init_macrostate"] = init_macro
        data_dict["prop_macrostate"] = prop_macro
        data_dict["final_macrostate"] = final_macro

        ds.close()
        del (ds)

        df = pd.DataFrame(data_dict)
        df_to_merge.append(df)

    if return_as_merged:
        # Return one dataframe in list
        final_df: pd.DataFrame = pd.concat(
            df_to_merge, axis=0, ignore_index=True, sort=False
        )
        return [final_df]
        # Return list of individual data frames
    else:
        return df_to_merge


def _micro_to_macro_labels(microstates, macrostate_labels):
    joint_init_microstates = [tuple(x) for x in microstates]
    return [macrostate_labels[y] for y in joint_init_microstates]


def _deduce_states_from_metadata(dataset: Dataset) -> List[int]:
    """Deduce the numner of states by using metadata."""
    nres: int = dataset["Protons/Metadata"].dimensions["residue"].size
    nstate_max: int = dataset["Protons/Metadata"].dimensions["state"].size
    return [
        nstate_max
        - np.ma.count_masked(dataset["Protons/Metadata/proton_count"][res, :])
        for res in range(nres)
    ]
