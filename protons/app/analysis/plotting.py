# coding=utf-8
"""Tools for the analysis of standard data structure files."""
import numpy as np
from pymbar import BAR
from ..logger import log
import seaborn as sns
from matplotlib import pyplot as plt
import matplotlib as mpl
import netCDF4
from typing import List, Tuple, Union, Dict, Optional
import pandas as pd
from .statistics import bar_calibration_data
from protons.app.utils import OutdatedFileError
from protons.app.driver import SAMSApproach
from tqdm import tqdm

sns.set_style("ticks")


def plot_sams_free_energy(
    dataframe: pd.DataFrame, ax: plt.Axes = None, as_bias: bool = False, colors=None
) -> plt.Axes:
    """Plot the SAMS weights from a calibration dataframe

    Parameters
    ----------
    dataframe - pandas dataframe with calibration data
    ax - matplotlib axes object, if None creates a new figure and axes
    as_bias - if True, plot the SAMS estimate as the bias (opposite sign of free energy)

    Notes
    -----
    Y - axis : the g_k (SAMS bias) value at iteration number

    Returns
    -------
    plt.Axes containing the plot

    """
    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = sns.color_palette()

    # Make the y-axis label, ticks and tick labels match the line color.
    ax.set_xlabel("Adaptation")

    # Obtain the columns containing SAMS data
    sams_bias_names = [col for col in dataframe.columns if "bias_state_" in col]
    state_names = [col.split("_")[-1] for col in sams_bias_names]
    n_states = len(sams_bias_names)
    x_axis_points = dataframe["adaptation"]

    for state_index, colname in enumerate(sams_bias_names):
        if not as_bias:
            data = -1 * dataframe[colname]
        else:
            data = dataframe[colname]

        ax.plot(
            x_axis_points,
            data,
            label="State {}".format(state_names[state_index]),
            color=colors[state_index],
        )

    return ax


def plot_horizontal_mean_with_error(
    mean, error, ax: plt.Axes = None, color=None
) -> plt.Axes:
    if ax is None:
        ax = plt.gca()

    ax.axhline(mean, linestyle=":", color=color)
    ax.axhspan(mean - error, mean + error, color=color, alpha=0.1)


def plot_bar_free_energy(dataframe: pd.DataFrame, ax=None, colors=None, as_bias=False):
    if ax is None:
        ax = plt.gca()
    if colors is None:
        colors = sns.color_palette()

    # Make the y-axis label, ticks and tick labels match the line color.
    ax.set_xlabel("Adaptation")

    # Obtain the columns containing SAMS data
    sams_bias_names = [col for col in dataframe.columns if "bias_state_" in col]
    state_names = [int(col.split("_")[-1]) for col in sams_bias_names]
    n_states = len(sams_bias_names)

    for state_index, state in enumerate(state_names):
        dF, std = bar_calibration_data(dataframe, 0, state, bootstrap_size=-1)
        if as_bias:
            dF *= -1
        plot_horizontal_mean_with_error(dF, std, ax=ax, color=colors[state_index])


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
