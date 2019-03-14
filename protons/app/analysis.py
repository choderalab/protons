# coding=utf-8
"""Tools for the analysis of standard data structure files."""
import numpy as np
from pymbar import bar
from .logger import log
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import netCDF4
from typing import List, Tuple, Union, Dict, Optional
import pandas as pd
from protons.app.utils import OutdatedFileError
from protons.app.driver import SAMSApproach
from tqdm import tqdm

sns.set_style("ticks")


def calibration_dataset_to_arrays(
    dataset: netCDF4.Dataset
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Extracts the necessary arrays for BAR analysis on a protons netCDF DataSet.

    Parameters
    ----------
    dataset - netCDF4 Dataset with 'Protons/SAMS' and 'Protons/NCMC' data

    Returns
    -------
    This function returns a tuple of arrays

    initial_states, initial state of the calibrated residue, indexed by update
    proposed_states, proposed state of the calibrated residue, indexed by update
    proposal_work, the NCMC work including delta g_k between states
    n_states, int tne number of states of the calibrated residue
    gk, the weight, indexed by update, state
    """

    try:
        sams = dataset["Protons/SAMS"]
    except KeyError:
        raise ValueError("This data set does not appear to have calibration data.")

    try:
        ncmc = dataset["Protons/NCMC"]
    except KeyError:
        raise ValueError("This data set does not appear to have NCMC data.")

    try:
        approach = SAMSApproach(dataset["Protons/SAMS/approach"][0])
    except IndexError:
        raise OutdatedFileError(
            "This file was generated with an older version of Protons. "
            "Please try a protons version <=0.0.1a4 to analyze it."
        )

    group = sams["group_index"][0]
    # This if statement checks if the group index is an integer, or undefined/masked (multisite calibration)
    if approach is SAMSApproach.ONESITE:
        initial_states = ncmc["initial_state"][:, group]
        proposed_states = ncmc["proposed_state"][:, group]
    elif approach is SAMSApproach.MULTISITE:
        initial_states = ncmc["initial_state"][:, :]
        proposed_states = ncmc["proposed_state"][:, :]
    else:
        raise NotImplementedError(
            f"This method does not support the {str(approach)} approach."
        )

    proposal_work = ncmc["total_work"][:]
    # Sams weighs iteration, state
    gk = dataset["Protons/SAMS/g_k"]

    # Stage of SAMS run
    stage = dataset["Protons/SAMS/stage"]

    # The number of states should be equal to the shape of the last dimension of the weights array.
    n_states = gk.shape[-1]

    return initial_states, proposed_states, proposal_work, n_states, gk, stage


def equibrium_dataset_to_arrays(
    dataset: netCDF4.Dataset, group: int = -1
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Extracts the necessary arrays for BAR analysis on a protons netCDF DataSet.

    Parameters
    ----------
    dataset - netCDF4 Dataset with 'Protons/SAMS' and 'Protons/NCMC' data

    Returns
    -------
    This function returns a tuple of arrays

    initial_states, initial state of the calibrated residue, indexed by update
    proposed_states, proposed state of the calibrated residue, indexed by update
    proposal_work, the NCMC work including delta g_k between states
    n_states, int tne number of states of the calibrated residue
    gk, the weight, indexed by update, state
    """

    try:
        meta = dataset["Protons/Metadata"]
    except KeyError:
        raise ValueError("This data set does not appear to have Metadata.")

    try:
        ncmc = dataset["Protons/NCMC"]
    except KeyError:
        raise ValueError("This data set does not appear to have NCMC data.")

    # This if statement checks if the group index is an integer, or undefined/masked (multisite calibration)
    initial_states = ncmc["initial_state"][:, group]
    proposed_states = ncmc["proposed_state"][:, group]
    proposal_work = ncmc["total_work"][:]

    # The number of states should be equal to the shape of the last dimension of the weights array.
    gk = meta["g_k"][group, :]
    if np.isscalar(gk.mask):
        if gk.mask == False:
            n_states = gk.size
    else:
        n_states = np.count_nonzero(~gk.mask)

    # mimic calibration array shape
    n_iters = proposal_work.size
    gk_iter = np.asarray([gk[0:n_states] for x in range(n_iters)])

    return (
        initial_states,
        proposed_states,
        proposal_work,
        n_states,
        gk_iter,
        np.asarray([]),
    )


def stitch_data(
    datasets: List[netCDF4.Dataset], has_sams=True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int, np.ndarray, np.ndarray]:
    """Collect data from multiple netCDF datasets and return them as single arrays.
    
    Parameters
    ----------
    datasets - ordered list of netCDF data set
    has_sams - set to True if the data is calibration data, set to False if equilibrium data without a SAMS group in the netCDF file

    """
    initial_states_collection: List[np.ndarray] = list()
    proposed_states_collection: List[np.ndarray] = list()
    proposal_work_collection: List[np.ndarray] = list()
    n_states_reference: int = 0
    gk_collection: List[np.ndarray] = list()
    stage_collection: List[np.ndarray] = list()
    # Start gathering datasets in lists, and concatenate them in the end.

    for d, dataset in enumerate(datasets):
        if has_sams:
            initial_states, proposed_states, proposal_work, n_states, gk, stages = calibration_dataset_to_arrays(
                dataset
            )
        else:
            initial_states, proposed_states, proposal_work, n_states, gk, stages = equibrium_dataset_to_arrays(
                dataset
            )

        if d == 0:
            n_states_reference = n_states

        if n_states_reference != n_states:
            raise ValueError(
                f"Number of states in set {d + 1} ({n_states}) does not match the first set ({n_states_reference})."
            )

        initial_states_collection.append(initial_states)
        proposed_states_collection.append(proposed_states)
        proposal_work_collection.append(proposal_work)
        gk_collection.append(gk)
        stage_collection.append(stages)

    # Use ma concatenate to preserve masked (incomplete) data
    init_states = np.ma.concatenate(initial_states_collection)
    prop_states = np.ma.concatenate(proposed_states_collection)
    prop_work = np.ma.concatenate(proposal_work_collection)
    gks = np.ma.concatenate(gk_collection, axis=0)
    all_stages = np.ma.concatenate(stage_collection)

    return init_states, prop_states, prop_work, n_states_reference, gks, all_stages


def create_ncmc_benchmark_dataframe(
    labeled_sets: Dict[str, List[netCDF4.Dataset]],
    timestep_ps: float = 0.002,
    has_sams: bool = True,
    extract_equilibrium: bool = True,
) -> pd.DataFrame:
    """Create a dataframe for analyzing NCMC benchmark results.

    Parameters
    ----------
    labeled_sets - a collection of datasets that have unique labels, provided as a dict of lists.
    timestep_ps - the timestep the simulations used in picoseconds. Used to convert protocol length to time.
    has_sams - True if analyzing calibration data, set to False for equilbrium simulations without SAMS data present
    extract_equilibrium - If true, will only return samples that were from equilibrium SAMS stage. Requires SAMS data to be present
    """

    df = _datasets_to_dataframe(labeled_sets, has_sams=has_sams)
    df = _calculate_ncmc_properties(df, timestep_ps=timestep_ps)
    if extract_equilibrium and has_sams:
        df = df.loc[df["Stage"] == 2]

    return df


def _calculate_ncmc_properties(df: pd.DataFrame, timestep_ps: float = 0.002):
    """Private function. Perform a number of operations on a comparison dataframe to add extra columns.

    Parameters
    ----------
    df - A dataframe with data from NCMC proposals, generated by ``dataset_to_dataframe``.

    """
    df = df.loc[df["Initial_State"] != df["Proposed_State"]]
    df["Pair"] = df["Initial_State"] + "->" + df["Proposed_State"]
    df["P_accept"] = np.exp(-1 * df["Work"])
    df[df["P_accept"] > 1.0] = 1.0
    df["ln_P_accept"] = np.log(df.P_accept)
    df["log10_P_accept"] = np.log10(df.P_accept)
    df["picoseconds"] = df["Length"] * timestep_ps
    df["Efficiency"] = df["P_accept"] / df["picoseconds"]

    return df


def _datasets_to_dataframe(
    dataset_dict: Dict[str, List[netCDF4.Dataset]], has_sams: bool = True
):
    """Private function to prepare a dataframe for comparing multiple runs.

    Parameters
    ----------
    dataset_dict - A dictionary where the key is the label for the dataset and the value
        is a list of netCDF4 datasets.
    has_sams - set to True if SAMS block present (e.g. data is calibration data.) or False for equilibrium only simulations.
    
    """

    df = pd.DataFrame(columns=["Initial_State", "Proposed_State", "Work", "Length"])
    for data_label, datasets in tqdm(dataset_dict.items(), "data converted"):

        initial_states, proposed_states, proposal_work, n_states, gk, stages = stitch_data(
            datasets, has_sams=has_sams
        )

        # TODO see if we need to handle variable length NCMC protocols too
        length = datasets[0]["Protons/NCMC/cumulative_work"][0, :].size
        longdict = {
            "Initial_State": [str(state) for state in initial_states],
            "Proposed_State": [str(state) for state in proposed_states],
            "Work": list(proposal_work),
            "Length": [length] * proposal_work.size,
            "Label": [data_label] * proposal_work.size,
            "Stage": list(stages),
        }

        if not has_sams:
            del (longdict["Stage"])
        # Extract state dependent free energy values
        for i in range(n_states):
            longdict[f"G_{i}"] = gk[:, i]

        new_data = pd.DataFrame.from_dict(longdict)
        df = df.append(new_data, ignore_index=True, sort=False)

    return df


def bar_all_states(
    dataset: Union[netCDF4.Dataset, List[netCDF4.Dataset]],
    bootstrap: bool = False,
    num_bootstrap_samples: int = 1000,
    to_first_state_only: bool = True,
    equilibrium_only: bool = True,
):
    """
    Run BAR between the first state and all other states.

    Parameters
    ----------
    dataset - a Protons formatted netCDF4 file
    bootstrap - enable to perform nonparametric bootstrapping over transitions to get uncertainty estimates.
    num_bootstrap_samples - int, the number of samples for bootstrapping
    to_first_state_only - only calculate values for pairswith first state by default, set to false for bar estimate
    between every pair of states
    equilibrium_only - analyze only samples from equilibrium simulations

    Notes
    -----
    The default uncertainty estimates come from bar, but the bootstrap estimates are the standard error estimated using the
    standard deviation of the individual bar free energy estimates.

    Returns
    -------
    dict - BAR estimates + error (SD or the SEM estimated through nonparametric bootstrap)
    Keys are formated as 'i->j' where i is the initial state and j is the proposed state

    """

    if type(dataset) == netCDF4.Dataset:
        initial_states, proposed_states, proposal_work, n_states, gk, stages = calibration_dataset_to_arrays(
            dataset
        )
        group = dataset["Protons/SAMS/group_index"][0]
        approach = SAMSApproach(dataset["Protons/SAMS/approach"][0])
    elif type(dataset) == list:
        initial_states, proposed_states, proposal_work, n_states, gk, stages = stitch_data(
            dataset
        )
        group = dataset[0]["Protons/SAMS/group_index"][0]
        approach = SAMSApproach(dataset[0]["Protons/SAMS/approach"][0])
    else:
        raise TypeError(
            f"Not sure how to handle type ({str(type(dataset))}) of dataset. Please provide a netCDF4 data set or a list of data sets."
        )

    if equilibrium_only:
        valid_indices = stages == 2
        initial_states, proposed_states, proposal_work, n_states, gk, stages = (
            initial_states[valid_indices],
            proposed_states[valid_indices],
            proposal_work[valid_indices],
            n_states,
            gk[valid_indices],
            stages[valid_indices],
        )

    bars_per_transition = dict()
    outer = range(1) if to_first_state_only else range(n_states)
    for from_state in outer:
        for to_state in range(n_states):
            if from_state == to_state:
                continue

            transition = "{}->{}".format(from_state, to_state)
            log.debug(transition)

            # SAMS bias estimate has the opposite sign of the free energy difference

            if approach is SAMSApproach.ONESITE:
                sams_estimate = -gk[group, to_state] + gk[group, from_state]
            elif approach is SAMSApproach.MULTISITE:
                sams_estimate = -gk[to_state] + gk[from_state]
            else:
                raise NotImplementedError(
                    "This method has not been implemented yet for this approach."
                )

            forward, reverse = _gather_transitions(
                from_state, to_state, initial_states, proposed_states, proposal_work, gk
            )
            skip = False
            if forward.size == 0:
                log.warn(f"No forward trajectories for {transition}")
                skip = True

            if reverse.size == 0:
                log.warn(f"No reverse trajectories for {transition}")
                skip = True

            if skip:
                bars_per_transition[transition] = (np.nan, np.nan)
                continue

            if bootstrap:
                bar_estimate, bar_sd = _nonparametric_bootstrap_bar(
                    forward, reverse, num_bootstrap_samples, sams_estimate
                )
            else:

                bar_estimate, bar_sd = bar.BAR(forward, reverse, DeltaF=sams_estimate)

            bars_per_transition[transition] = (bar_estimate, bar_sd)

    return bars_per_transition


def extract_work_distributions(
    dataset: netCDF4.Dataset, state1_idx: int, state2_idx: int, res_idx: int
) -> tuple:
    """Extract the forward and reverse work distributions for ncmc protocols between two states, for a given residue.
           
    Parameters
    ----------
    dataset - a Dataset with a Protons/NCMC group to analyze
    state1_idx - the "from" state index, for defining forward protocol
    state2_idx - the "to" state index, for defining forward protocol
    res_idx - the titratable residue index
    
    Returns
    -------
    tuple(np.ndarray, np.ndarray) the forward, and reverse work distributions. 
    
    Note
    ----
    The distribution of the reverse proposals is returned as -W, to give it the same sign 
    as the forward distribution.    
    """

    ncmc = dataset["Protons/NCMC"]
    forward_work = []
    neg_reverse_work = []

    initial_states = ncmc["initial_state"][:, res_idx]
    proposed_states = ncmc["proposed_state"][:, res_idx]
    tot_work = ncmc["total_work"][:]
    for update in ncmc["update"]:
        update -= 1  # 1 indexed variable
        init = initial_states[update]
        prop = proposed_states[update]

        # Forward distribution
        if init == state1_idx:
            if prop == state2_idx:
                forward_work.append(tot_work[update])
        # Reverse distribution
        elif init == state2_idx:
            if prop == state1_idx:
                # Use negative value of the work
                # so that the two distributions have the same sign.
                neg_reverse_work.append(-tot_work[update])

    return np.asarray(forward_work), np.asarray(neg_reverse_work)


def _nonparametric_bootstrap_bar(
    forward: np.ndarray, reverse: np.ndarray, nbootstraps: int, sams_estimate: float
):
    """Perform sampling with replacement on forward and reverse trajectories and perform BAR.

    Parameters
    ----------
    forward - array of work values for forward proposals
    reverse - array of work values for reverse proposals
    nbootstraps - number of bootstrap samples to run
    sams_estimate - initial deltaF guess from SAMS

    Returns
    -------
    mean, standard error

    """
    num_forward = forward.size
    num_reverse = reverse.size

    strap_bars = list()
    for bootstrap in range(nbootstraps):
        # pick new set with the same length from forward trajectories using sampling with replacement
        bootstrap_forward = forward[np.random.choice(num_forward, num_forward)]
        # pick new set with the same length from reverse trajectories using sampling with replacement
        bootstrap_reverse = reverse[np.random.choice(num_reverse, num_reverse)]
        strap_bars.append(
            bar.BAR(bootstrap_forward, bootstrap_reverse, DeltaF=sams_estimate)[0]
        )

    # standard deviation of the bootstrap samples corresponds to the standard error.
    return np.mean(strap_bars), np.std(strap_bars)


def _gather_transitions(
    from_state: int,
    to_state: int,
    initial_states: np.ndarray,
    proposed_states: np.ndarray,
    proposal_work: np.ndarray,
    gk: np.ndarray,
):
    """Gather the total work for all forward and reverse proposals for a given pair of states.

    Parameters
    ----------
    from_state - int, the state from which transitions originate
    to_state - int, the final state of the transition
    initial_states - array, the sequence of the initial state of every proposal
    proposed_states - array, the sequence of the proposed state of every proposal
    proposal_work - array, the work value of the NCMC proposal, including the delta g_k

    Returns
    -------
    array[float], array[float]
    forward, reverse work values as arrays
    """
    forward = list()
    reverse = list()

    for i in range(0, gk.shape[0]):
        # subtract delta g_k forward
        if initial_states[i] == from_state and proposed_states[i] == to_state:
            forward.append(proposal_work[i] - gk[i][to_state] + gk[i][from_state])
        # subtract delta g_k reverse
        elif initial_states[i] == to_state and proposed_states[i] == from_state:
            reverse.append(proposal_work[i] - gk[i][from_state] + gk[i][to_state])
    forward = np.array(forward)
    reverse = np.array(reverse)
    num_forward = forward.size
    num_reverse = reverse.size
    log.debug("forward, reverse")
    log.debug("%d, %d", num_forward, num_reverse)

    return forward, reverse


def plot_calibration_weight_traces(
    dataset: Union[netCDF4.Dataset, List[netCDF4.Dataset]],
    ax: plt.Axes = None,
    bar: bool = True,
    error: str = "stdev",
    num_bootstrap_samples: int = 1000,
    zerobased: bool = False,
):
    """Plot the results of a calibration netCDF dataset.

    Parameters
    ----------
    dataset - netCDF4 dataset with `Protons` data
    ax - matplotlib axes object, if None creates a new figure and axes
    bar - bool, plot BAR estimates
    error - str, if bar is True 'stdev' will plot the BAR error as 2 times standard deviation returned from BAR, 'bootstrap' will plot standard error from bootstrap.
    num_bootstrap_samples - Used if 'bootstrap' is used as error method. Number of bootstrap samples to draw.
    zerobased - if True, label states in zero based fashion in the plots. Defaults to 1-based


    Returns
    -------
    plt.Axes containing the plot

    """
    if ax is None:
        ax = plt.gca()

    # Two methods of calculating the error
    if error == "stdev":
        # Use 2 * standard deviation, as returned by bar
        bootstrap = False
        err_multiply = 2
        err_label = r"BAR estimate $\pm$ $2\sigma$, State {}"

    elif error == "bootstrap":
        # Use the standard error from bootstrap sampling of reverse and forward trajectories
        bootstrap = True
        err_multiply = 1
        err_label = r"BAR estimate $\pm$ SEM, State {}"
    else:
        raise ValueError("Unsupported error method: {}.".format(error))

    if zerobased:
        label_offset = 0
    else:
        label_offset = 1

    # Make the y-axis label, ticks and tick labels match the line color.
    ax.set_xlabel("Update", fontsize=18)

    ax.set_ylabel(r"$g_k/RT$ (unitless) relative to state 1", fontsize=18)
    if type(dataset) == netCDF4.Dataset:
        n_states, gk = calibration_dataset_to_arrays(dataset)[-3:-1]
    elif type(dataset) == list:
        n_states, gk = stitch_data(dataset)[-3:-1]
    cp = sns.color_palette("husl", n_states)

    if bar:
        bar_data = bar_all_states(
            dataset, bootstrap=bootstrap, num_bootstrap_samples=num_bootstrap_samples
        )
    else:
        bar_data = None

    xaxis_points = list(range(gk.shape[0]))
    for state in range(1, n_states):
        state_color = cp[state]
        ax.plot(
            gk[:, state],
            label="State {}".format(state + label_offset),
            color=state_color,
        )

        if bar:
            estimate, err = bar_data["0->{}".format(state)]
            err *= err_multiply
            ax.fill_between(
                xaxis_points,
                -estimate - err,
                -estimate + err,
                alpha=0.3,
                color=state_color,
                label=err_label.format(state + label_offset),
            )
    sns.despine(ax=ax)
    return ax


def plot_residue_state_traces(
    dataset: netCDF4.Dataset,
    residue_index: int,
    ax: plt.Axes = None,
    zerobased_states: bool = False,
):
    """
    Plot the state of the given residue for individual updates.

    Parameters
    ----------
    dataset - netCDF4.Dataset, dataset containing Protons data
    residue_index - int, the index of the residue for which to plot a trace
    ax - matplotlib axes object, if None creates a new figure and axes
    zerobased_states - bool, default False. Label the states 0 or 1 based.

    Returns
    -------
    plt.Axes object containing the plot
    """

    if ax is None:
        ax = plt.gca()

    if zerobased_states:
        label_offset = 0
    else:
        label_offset = 1
    dataset.set_auto_mask(True)
    titration_states = dataset["Protons/Titration/state"][:, residue_index]
    meta = dataset["Protons/Metadata"]
    g_ks = meta["g_k"][residue_index, :]
    # all True values in the inverse mask are actual states
    try:
        nstates = np.count_nonzero(~g_ks.mask)
    except AttributeError:
        nstates = g_ks.shape[0]

    cp = sns.color_palette("husl", nstates)

    # Gather every update index per state
    state_sets = []
    for check_state in range(nstates):
        state_sets.append([])
        for index, state in enumerate(titration_states):
            if state == check_state:
                state_sets[check_state].append(index)

    encountered = np.unique(titration_states)
    for axis in [ax.xaxis, ax.yaxis]:
        axis.set_major_locator(
            mpl.ticker.MaxNLocator(integer=True, min_n_ticks=encountered.size)
        )

    xaxis = list(range(titration_states.shape[0]))

    # Line connecting states
    ax.plot(xaxis, label_offset + titration_states, c="k", alpha=0.02)

    # Plot individual points in the right color
    for s, state in enumerate(state_sets):
        ax.scatter(state, [s + 1] * len(state), c=cp[s])

    ax.set_xlabel("Iteration", fontsize=18)
    ax.set_ylabel("State", fontsize=18)

    return ax


def charge_taut_trace(dataset: netCDF4.Dataset):
    """Return a trace of the total charge for each residue, and the tautomer of the charge.

    Parameters
    ----------
    dataset - netCDF4.Dataset, a dataset containing Protons data.

    Returns
    -------
    charges - array of charges indexed as [iteration,residue]
    tautomers - array of tautomers per charge, indexed as iteration, residue
    """
    # charge by residue, state, rounded to nearest integer
    charge_data = np.rint(dataset["Protons/Metadata/total_charge"][:, :]).astype(int)

    # tautomer per residue, state, counted per charge
    # filled in below
    tautomer_data = np.empty_like(charge_data)

    for residue in range(tautomer_data.shape[0]):
        # Keep track of how many times a charge was observed in the states of this residue
        charge_counts = dict()
        for state in range(tautomer_data.shape[1]):
            charge = charge_data[residue, state]
            # for non-existent states
            if type(charge) is np.ma.core.MaskedConstant:
                tautomer_data[residue, state] = 0
                continue

            if charge not in charge_counts:
                charge_counts[charge] = 0
            tautomer_data[residue, state] = charge_counts[charge]
            charge_counts[charge] += 1

    # State per iteration, residue
    titration_states = dataset["Protons/Titration/state"][:, :]

    charges = np.empty_like(titration_states)
    tautomers = np.empty_like(titration_states)
    for iteration in range(titration_states.shape[0]):
        for residue in range(titration_states.shape[1]):
            state = titration_states[iteration, residue]
            charges[iteration, residue] = charge_data[residue, state]
            tautomers[iteration, residue] = tautomer_data[residue, state]

    return charges, tautomers


def plot_heatmap(
    dataset: netCDF4.Dataset,
    ax: plt.Axes = None,
    color: str = "charge",
    residues: list = None,
    zerobased: bool = False,
):
    """Plot the states, or the charges as colored blocks

    Parameters
    ----------
    dataset - netCDF$.Dataset containing Protons information.
    ax - matplotlib Axes object
    color - 'charge', 'state', 'taut' , color by charge, by state, or charge and shade by tautomer
    residues - list, residues to plot
    zerobased - bool default False - use zero based labeling for states.

    Returns
    -------
    ax - plt.Axes
    """
    # Convert to array, and make sure types are int
    if ax is None:
        ax = plt.gca()

    if zerobased:
        label_offset = 0
    else:
        label_offset = 1

    if color == "charge":
        vmin = -2
        vmax = 2
        center = 0
        cmap = sns.diverging_palette(
            25, 244, l=60, s=95, sep=80, center="light", as_cmap=True
        )
        ticks = np.arange(vmin, vmax + 1)
        boundaries = np.arange(vmin - 0.5, vmax + 1.5)
        cbar_kws = {"ticks": ticks, "boundaries": boundaries, "label": color.title()}

    elif color == "state":
        vmin = 0 + label_offset
        vmax = label_offset + np.amax(dataset["Protons/Titration/state"][:, :])
        ticks = np.arange(vmin, vmax + 1)
        boundaries = np.arange(vmin - 0.5, vmax + 1.5)
        cbar_kws = {"ticks": ticks, "boundaries": boundaries, "label": color.title()}
        center = None
        cmap = "Accent"

    else:
        raise ValueError("color argument should be 'charge', or 'state'.")

    to_plot = None
    if residues is None:
        if color == "charge":
            to_plot = charge_taut_trace(dataset)[0][:, :]
        elif color == "state":
            titration_states = dataset["Protons/Titration/state"][:, :]
            to_plot = titration_states + label_offset

    else:
        if isinstance(residues, int):
            residues = [residues]
        residues = np.asarray(residues).astype(np.int)
        if color == "charge":
            to_plot = charge_taut_trace(dataset)[0][:, residues]
        elif color == "state":
            to_plot = dataset["Protons/Titration/state"][:, residues] + label_offset

    ax = sns.heatmap(
        to_plot.T,
        ax=ax,
        vmin=vmin,
        vmax=vmax,
        center=center,
        xticklabels=int(np.floor(to_plot.shape[0] / 7)) - 1,
        yticklabels=int(np.floor(to_plot.shape[1] / 4)) - 1,
        cmap=cmap,
        cbar_kws=cbar_kws,
        edgecolor="None",
        snap=True,
    )

    for residue in range(to_plot.T.shape[1]):
        ax.axhline(residue, lw=0.4, c="w")

    ax.set_ylabel("Residue")
    ax.set_xlabel("Update")
    return ax


def plot_tautomer_heatmap(
    dataset: netCDF4.Dataset,
    ax: plt.Axes = None,
    residues: list = None,
    zerobased: bool = False,
):
    """Plot the charge of residues on a blue-red (negative-positive) scale, and add different shades for different tautomers.

    Parameters
    ----------
    dataset - netCDF4 dataset containing protons data
    ax - matplotlib Axes object
    residues - list, residues to plot
    zerobased - bool default False - use zero based labeling for states.

    Returns
    -------
    plt.Axes

    """
    # Convert to array, and make sure types are int
    if ax is None:
        ax = plt.gca()

    if zerobased:
        label_offset = 0
    else:
        label_offset = 1

    # color charges, and add shade for tautomers
    vmin = -2
    vmax = 2
    center = 0
    cmap = sns.diverging_palette(
        25, 244, l=60, s=95, sep=80, center="light", as_cmap=True
    )
    ticks = np.arange(vmin, vmax + 1)
    boundaries = np.arange(vmin - 0.5, vmax + 1.5)
    cbar_kws = {"ticks": ticks, "boundaries": boundaries, "label": "Charge"}

    taut_vmin = 0 + label_offset
    taut_vmax = label_offset + np.amax(dataset["Protons/Titration/state"][:, :])
    taut_ticks = np.arange(taut_vmin, taut_vmax + 1)
    taut_boundaries = np.arange(taut_vmin - 0.5, taut_vmax + 1.5)
    taut_cbar_kws = {"boundaries": taut_boundaries}

    taut_center = None
    taut_cmap = "Greys"

    to_plot = None
    if residues is None:
        to_plot, taut_to_plot = charge_taut_trace(dataset)

    else:
        if isinstance(residues, int):
            residues = [residues]
        residues = np.asarray(residues).astype(np.int)
        charges, tauts = charge_taut_trace(dataset)
        to_plot = charges[:, residues]
        taut_to_plot = tauts[:, residues]

    mesh = ax.pcolor(to_plot.T, cmap=cmap, vmin=vmin, vmax=vmax, snap=True, alpha=1.0)
    plt.colorbar(mesh, ax=ax, **cbar_kws)
    taut_mesh = ax.pcolor(
        taut_to_plot.T,
        cmap=taut_cmap,
        vmin=taut_vmin,
        vmax=taut_vmax,
        alpha=0.1,
        snap=True,
    )

    for residue in range(to_plot.T.shape[0]):
        ax.axhline(residue, lw=0.4, c="w")

    ax.set_ylabel("Residue")
    ax.set_xlabel("Update")

    return ax


def calculate_ncmc_acceptance_rate(dataset: netCDF4.Dataset):
    """Calculate the NCMC acceptance rate from a dataset containing Protons/NCMC data
    Parameters
    ----------
    dataset - netCDF4.Dataset, should contain Protons/NCMC data

    Returns
    -------
    float - acceptance rate

    """
    titration_states = dataset["Protons/Titration/state"][:, :]
    initial_states = dataset["Protons/NCMC/initial_state"][:, :]
    proposed_states = dataset["Protons/NCMC/proposed_state"][:, :]
    naccepted = 0
    ntotal = titration_states.shape[0]
    for i in range(ntotal):
        if titration_states[i] == proposed_states[i]:
            if proposed_states[i] != initial_states[i]:
                naccepted += 1

    return float(naccepted) / float(ntotal)


def gather_trajectories(
    datasets: List[netCDF4.Dataset],
    from_state: int,
    to_state: int,
    cumulative: bool = True,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """Gather lists of work trajectories from a list of netCDF datasets.

    Parameters
    ----------
    datasets - A list of different netCDF datasets to retrieve NCMC work trajectories from
    from_state - The state from which the protocol was initiated
    to_state - The state to which the protocol is changing the system
    cumulative - Set to false if incremental work is desired, rather than cumulative work.
    """
    forward_trajectories = list()
    reverse_trajectories = list()

    for d, dataset in enumerate(datasets):
        for update in dataset["Protons/NCMC/update"][:]:
            x = update - 1
            if (
                from_state == dataset["Protons/NCMC/initial_state"][x, :]
                and to_state == dataset["Protons/NCMC/proposed_state"][x, :]
            ):

                if cumulative:
                    trajectory = dataset["Protons/NCMC/cumulative_work"][x, :]
                else:
                    trajectory = np.diff(dataset["Protons/NCMC/cumulative_work"][x, :])

                forward_trajectories.append(trajectory)
            elif (
                to_state == dataset["Protons/NCMC/initial_state"][x, :]
                and from_state == dataset["Protons/NCMC/proposed_state"][x, :]
            ):
                if cumulative:
                    trajectory = dataset["Protons/NCMC/cumulative_work"][x, :]
                else:
                    trajectory = np.diff(dataset["Protons/NCMC/cumulative_work"][x, :])
                reverse_trajectories.append(trajectory)
    return forward_trajectories, reverse_trajectories


def plot_work_per_step(
    datasets: List[netCDF4.Dataset],
    from_state: int,
    to_state: int,
    color: str,
    cumulative=True,
    alpha: float = 0.1,
    label: str = "",
    which: int = 0,
):
    """Plot the work trajectories from a series of netCDF data sets.

     Parameters
    ----------
    datasets - A list of different netCDF datasets to retrieve NCMC work trajectories from
    from_state - The state from which the protocol was initiated
    to_state - The state to which the protocol is changing the system
    color - A color name for matplotlib
    cumulative - Set to false if incremental work is desired, rather than cumulative work.
    Alpha - float between 0 and 1 for transparency of lines
    label - label for the data in the legend.
    which - -1 for plotting reverse only, 1 for forward only, or 0 for both
    """
    if which not in [-1, 0, 1]:
        raise ValueError(
            "Please use -1 for plotting reverse only, 1 for forward only, or 0 for both."
        )

    forward, reverse = gather_trajectories(
        datasets, from_state, to_state, cumulative=cumulative
    )
    size = forward[0].shape[0]
    # Only label first line
    if which == 0 or which == 1:
        for traj in forward:
            plt.plot(
                np.linspace(0, 100, num=size),
                traj,
                color=color,
                alpha=alpha,
                label=label,
                linestyle="-",
            )
            label = ""
    if which == 0 or which == -1:
        for traj in reverse:
            plt.plot(
                np.linspace(0, 100, num=size),
                traj,
                color=color,
                alpha=alpha,
                label=label,
                linestyle=":",
            )
            label = ""
    return


def get_stage_data(
    dataset: Union[netCDF4.Dataset, List[netCDF4.Dataset]]
) -> np.ndarray:
    """Retrieve stage, and adaptation as numpy arrays of integers."""

    stages = list()
    adapts = list()

    if type(dataset) == netCDF4.Dataset:
        stages.append(dataset["Protons/SAMS/stage"][:])
        adapts.append(dataset["Protons/SAMS/adaptation"][:])

    elif type(dataset) == list:
        for ds in dataset:
            stages.append(ds["Protons/SAMS/stage"][:])
            adapts.append(ds["Protons/SAMS/adaptation"][:])

    return np.ma.concatenate(stages), np.ma.concatenate(adapts)


def get_flatness_data(
    dataset: Union[netCDF4.Dataset, List[netCDF4.Dataset]]
) -> np.ndarray:
    """Retrieve stage, and adaptation as numpy arrays of integers."""

    flatness = list()
    adapts = list()

    if type(dataset) == netCDF4.Dataset:
        flatness.append(dataset["Protons/SAMS/flatness"][:])
        adapts.append(dataset["Protons/SAMS/adaptation"][:])

    elif type(dataset) == list:
        for ds in dataset:
            flatness.append(ds["Protons/SAMS/flatness"][:])
            adapts.append(ds["Protons/SAMS/adaptation"][:])

    return np.ma.concatenate(flatness), np.ma.concatenate(adapts)


def plot_calibration_flatness(
    dataset: Union[netCDF4.Dataset, List[netCDF4.Dataset]],
    sams_only: bool = False,
    plot_burnin: bool = False,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = "Calibration flatness",
) -> Tuple[Optional[plt.Figure], plt.Axes]:
    """Plot the histogram flatness as a function of the iteration.

    Parameters
    ----------
    dataset - A netCDF calibration dataset, or list of data sets.
    sams_only - Set to True to not plot the equilibrium stage.
    plot_burnin - Set to True to plot the burn in.
    ax - optional, an axes object to plot in. If not provided a new one is generated.
    title - title of the plot.

    Note
    ----
    By default the burn in is not plotted since flatness is not evaluated and reported as 1.0/100% deviation.

    Returns
    -------
    fig, ax containing the plot

    """

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    colors = {-1: "red", 0: "blue", 1: "green", 2: "black"}
    styles = {-1: "-", 0: "--", 1: "-.", 2: ":"}
    stage_names = {-1: "Burn-in", 0: "Slow-decay", 1: "Fast-decay", 2: "Equilibrium"}

    if type(dataset) == netCDF4.Dataset:
        n_states, gk, stages = calibration_dataset_to_arrays(dataset)[-3:]
    elif type(dataset) == list:
        n_states, gk, stages = stitch_data(dataset, has_sams=True)[-3:]

    flatness, adapts = get_flatness_data(dataset)

    ax.margins(x=0)
    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Update", fontsize=18)
    ax.set_ylabel(r"Deviation from target", fontsize=18)
    flatness_state = flatness[:]
    iter_nums = np.arange(flatness_state.size)
    for stage in colors.keys():
        if not plot_burnin and stage == -1:
            continue
        indexes = np.where(stages == stage)
        if sams_only:
            # If sams only plotting desired, plot based on adaptation number (cancels out eq. parts)
            x_values = adapts[indexes]
        else:
            x_values = iter_nums[indexes]

        ax.plot(
            x_values,
            flatness_state[indexes],
            color=colors[stage],
            ls=styles[stage],
            label=stage_names[stage],
        )
    return fig, ax


def plot_calibration_staging_per_state(
    dataset: Union[netCDF4.Dataset, List[netCDF4.Dataset]],
    sams_only: bool = True,
    plot_burnin: bool = True,
    axes: Optional[List[plt.Axes]] = None,
):
    """Separately plot the calibration weights of each state and indicate their stage.

    Parameters
    ----------
    dataset - Dataset or list of Datasets containing calibration data.
    sams_only - set to True to not plot the equilibrium stage
    plot_burnin - set to False to not plot the burnin stage
    axes - list of axes for plotting the different states.
        Needs to be equal to number of states -1 (since first state is always 0 and not plotted).
    """

    colors = {-1: "red", 0: "blue", 1: "green", 2: "black"}
    styles = {-1: "-", 0: "--", 1: "-.", 2: ":"}
    stage_names = {-1: "Burn-in", 0: "Slow-decay", 1: "Fast-decay", 2: "Equilibrium"}

    if type(dataset) == netCDF4.Dataset:
        n_states, gk = calibration_dataset_to_arrays(dataset)[-3:-1]
    elif type(dataset) == list:
        n_states, gk = stitch_data(dataset, has_sams=True)[-3:-1]

    figures: List[plt.Figure] = list()
    if axes is None:
        axes: List[plt.Axes] = list()
        for x in range(1, n_states):
            fig, ax = plt.subplots()
            figures.append(fig)
            axes.append(ax)
    else:
        if len(axes) != n_states - 1:
            raise ValueError(
                "The number of axes provided should be equal to the number of states, minus one."
            )

    stages, adapts = get_stage_data(dataset)

    # Don't plot state 0 as it should be 0 by definition
    for state in range(1, n_states):
        ax = axes[state - 1]
        ax.margins(x=0)
        ax.set_title(f"State {state}", fontsize=18)
        # Make the y-axis label, ticks and tick labels match the line color.
        ax.set_xlabel("Update", fontsize=18)
        ax.set_ylabel(r"$g_k/RT$ (unitless) relative to state 1", fontsize=18)
        gk_state = gk[:, state]
        iter_nums = np.arange(gk_state.size)
        for stage in colors.keys():
            if not plot_burnin and stage == -1:
                continue
            indexes = np.where(stages == stage)
            if sams_only:
                # If sams only plotting desired, plot based on adaptation number (cancels out eq. parts)
                x_values = adapts[indexes]
            else:
                x_values = iter_nums[indexes]

            ax.plot(
                x_values,
                gk_state[indexes],
                color=colors[stage],
                ls=styles[stage],
                label=stage_names[stage],
            )
    return figures, axes


def plot_calibration_staging_joint(
    dataset: Union[netCDF4.Dataset, List[netCDF4.Dataset]],
    sams_only: bool = True,
    plot_burnin: bool = True,
    ax: Optional[plt.Axes] = None,
    title: Optional[str] = "Calibration stages",
):
    """Plot calibration traces for all states whilst highlighting the stages in the background."""

    fig = None
    if ax is None:
        fig, ax = plt.subplots()

    ax.margins(x=0)

    ax.set_title(title, fontsize=18)
    ax.set_xlabel("Update", fontsize=18)
    ax.set_ylabel(r"$g_k/RT$ (unitless) relative to state 1", fontsize=18)

    stage_hatches = {-1: "xxx", 0: "//", 1: "--", 2: ""}
    stage_names = {-1: "Burn-in", 0: "Slow-decay", 1: "Fast-decay", 2: "Equilibrium"}

    if type(dataset) == netCDF4.Dataset:
        n_states, gk, stages = calibration_dataset_to_arrays(dataset)[-3:]
    elif type(dataset) == list:
        n_states, gk, stages = stitch_data(dataset, has_sams=True)[-3:]

    colors = sns.color_palette("hls", n_states)
    stages, adapts = get_stage_data(dataset)
    # Every stage boundary is 1, everything else is 0
    stage_boundaries = np.argwhere(np.diff(stages)) + 1

    # Don't plot state 0 as it should be 0 by definition
    for state in range(1, n_states):
        gk_state = gk[:, state]
        iter_nums = np.arange(gk_state.size)
        labeled = True
        for stage in stage_hatches.keys():
            if not plot_burnin and stage == -1:
                continue
            indexes = np.where(stages == stage)
            if sams_only:
                # If sams only plotting desired, plot based on adaptation number (cancels out eq. parts)
                if stage == 2:
                    continue
                x_values = adapts[indexes]

            else:
                x_values = iter_nums[indexes]

            if labeled:
                labeled = False
                label = f"State {state}"
            else:
                label = ""
            ax.plot(x_values, gk_state[indexes], color=colors[state - 1], label=label)

    # Fix current x limits on plot
    ax.set_xlim(*ax.get_xlim())

    # plot the background
    prev_bound = 0

    for stage in stage_hatches.keys():
        try:
            current_bound = stage_boundaries[stage + 1]
        except IndexError:
            current_bound = -1

        if not plot_burnin and stage == -1:
            continue

        if sams_only:
            # If sams only plotting desired, plot based on adaptation number (cancels out eq. parts)
            lower = adapts[prev_bound]
            upper = adapts[current_bound]
            if stage == 2:
                continue
        else:
            lower = iter_nums[prev_bound]
            upper = iter_nums[current_bound]

        ax.axvspan(
            lower,
            upper,
            facecolor="none",
            edgecolor="black",
            alpha=0.3,
            hatch=stage_hatches[stage],
            label=stage_names[stage],
        )
        prev_bound = current_bound
    ax.legend(fontsize=12, prop={"size": 16})

    return fig, ax
