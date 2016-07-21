"""
Optional plotting tools, useful for diagnosing simulation and calibration.
"""

import matplotlib
import os
# Failsafe for non visual environments (linux only)
if os.environ.get("DISPLAY") is None:
    matplotlib.use('Agg')
import seaborn as sns


def _add_subplot_axes(ax, rect):
    """
    Add an embedded subplot to an axis

    ----------
    ax - axes object
    rect - relative dimension

    Returns
    -------
    subaxis

    """
    # Based on http://stackoverflow.com/a/17479417
    with sns.axes_style("white"):
        fig = sns.plt.gcf()
        box = ax.get_position()
        width = box.width
        height = box.height
        inax_position  = ax.transAxes.transform(rect[0:2])
        transFigure = fig.transFigure.inverted()
        infig_position = transFigure.transform(inax_position)
        x = infig_position[0]
        y = infig_position[1]
        width *= rect[2]
        height *= rect[3]  # <= Typo was here
        subax = fig.add_axes([x,y,width,height])
        x_labelsize = subax.get_xticklabels()[0].get_size()
        y_labelsize = subax.get_yticklabels()[0].get_size()
        x_labelsize *= rect[2]**0.25
        y_labelsize *= rect[3]**0.25
        subax.xaxis.set_tick_params(labelsize=x_labelsize)
        subax.yaxis.set_tick_params(labelsize=y_labelsize)
    return subax


def plot_sams_trace(ydata, xdata=None, ax=None, subax=None, window=0, rect=None, title="", ylabel="", y_zoom=0.5, filename=""):
    if rect is None:
        rect = [0.1,0.1,0.25,0.2]

    if ax is None:
        f, ax = sns.plt.subplots(1)

    if subax is None:
        subax = _add_subplot_axes(ax, rect=rect)

    if xdata is None:
        xdata = range(len(ydata))

    spread = max(ydata[-window:]) - min(ydata[-window:])
    if spread > y_zoom:
        y_zoom += spread


    ax.plot(xdata, ydata,)
    ax.scatter(xdata[-window:], ydata[-window:],  color='red')
    ax.set_title(title)
    ax.set_ylabel(ylabel, rotation=0)
    ax.yaxis.set_label_position('left')
    ax.set_ylim(ydata[-1]-y_zoom,ydata[-1]+y_zoom)
    ax.set_xlim(xdata[-window]-1, xdata[-1]+1)
    ax.set_xlabel("Updates")

    subax.plot(xdata, ydata, )
    subax.set_xlim(xdata[0]-1, xdata[-1]+1)
    subax.scatter(xdata[-window:], ydata[-window:], color='red')
    ax.set_title(title)
    sns.plt.legend()

    if filename:
        sns.plt.savefig(filename,dpi=150)

    return ax, subax
