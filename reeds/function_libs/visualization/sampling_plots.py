from typing import List

import numpy as np
from matplotlib import pyplot as plt

from reeds.function_libs.visualization import plots_style as ps
from reeds.function_libs.visualization.utils import nice_s_vals


def plot_t_statepres(data: dict,
                     out_path: str = None,
                     title: str = "test",
                     xlim: List[int] = False):
    """plot_t_statepres
    gives out a plot, showing the if a state is undersampling or the
    dominating (min state) of a system at given t.

    Parameters
    ----------
    data : dict
    out_path : str, optional
        path for output files (default None)
    title : str, optional
        title string (default "test")
    xlim: List[int], optional
        default False

    Returns
    -------
    None
    """

    # sort data:
    num_states = len(data["occurrence_t"])
    x_ax = data["dominating_state"].index  # time axis
    ymin = np.array(data["dominating_state"], dtype=float)  # to get correct state (counted with 0)

    yunders = []
    for state in range(num_states):
        y_state = {"x": data["occurrence_t"][state], "y": np.full(len(data["occurrence_t"][state]), 1 + state)}
        yunders.append(y_state)

    # plot:
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ##plotting
    first = True
    for xy in yunders:
        if first:
            ax.scatter(xy["x"], xy["y"], label='undersample', alpha=0.5, c="blue", s=2, lw=0, marker=".",
                       edgecolors=None)
            first = False
        else:
            ax.scatter(xy["x"], xy["y"], alpha=0.5, c="blue", s=2, lw=0, marker=".", edgecolors=None)

    ax.scatter(x_ax, ymin, label="minstate", alpha=0.7, c="red", lw=0.0, s=5, marker=".", edgecolors=None)

    ##define limits
    ax.set_ylim(0.25, num_states + 0.5)
    if (xlim):
        xlim = xlim
    else:
        xlim = [0, x_ax[len(x_ax) - 1]]
    ax.set_xlim(xlim)

    ##labels
    title = "$" + title + "$"
    ax.set_title("state occurence in " + title)
    ax.set_ylabel("states")
    ax.set_xlabel("time [ps]")
    ax.set_yticks(range(0, num_states))
    ax.set_yticks(range(1, num_states + 1))

    ##legends
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.85, chartBox.height])
    lgnd = ax.legend(title="states:", loc=2, borderaxespad=0, bbox_to_anchor=(1.05, 1), ncol=1, prop={"size": 10})
    for handle in lgnd.legendHandles:
        handle.set_sizes([28.0])

    ##savefigure
    if (not out_path is None):
        fig.savefig(out_path, bbox_extra_artists=(lgnd,), bbox_inches='tight')
        plt.close()


def plot_stateOccurence_hist(data: dict,
                             out_path: str = None,
                             title: str = "sampling histogramms",
                             verbose: bool = False):
    """plot_stateOccurence_hist
    plot histogram of state occurrence

    Parameters
    ----------
    data : dict
    out_path : str, optional
        output file path (default None)
    title : str, optional
        title (default "sampling histogramms")
    verbose : bool
        verbose output (default False)

    Returns
    -------
    None
    """

    def autolabel(rects, max_val=1, xpos='center'):
        """
        From WEB: https://matplotlib.org/gallery/api/barchart.html
        Attach a text label above each bar in *rects*, displaying its height.

        *xpos* indicates which side to place the text w.r.t. the center of
        the bar. It can be one of the following {'center', 'right', 'left'}.
        """

        xpos = xpos.lower()  # normalize the case of the parameter
        ha = {'center': 'center', 'right': 'left', 'left': 'right'}
        offset = {'center': 0.5, 'right': 0.75, 'left': 0.35}  # x_txt = x + w*off

        for rect in rects:
            height = rect.get_height() if (rect.get_height() <= 0.05) else 0.92
            label = round(rect.get_height() / max_val, str(rect.get_height() / max_val).count("0") + 1)

            ax.text(rect.get_x() + rect.get_width() * offset[xpos], 1.01 * height,
                    '{}'.format(label), ha=ha[xpos], va='bottom')

    # histogramm
    bins_dom = list(data['dominating_state'].values())
    bins_und = list(data['occurence_state'].values())
    labels = list(data['occurence_state'].keys())

    if verbose: print(str(data.keys()) + "\n" + str(labels) + "\n" + str(bins_und))
    width = 0.35

    fig = plt.figure()
    ax = fig.add_subplot(111)

    sampled = ax.bar(x=labels, height=bins_und, width=width, label="occurrence", color="C0")
    autolabel(sampled, xpos='left')

    sampled = ax.bar(x=labels, height=bins_dom, width=width, label="dominating", color="C3")
    autolabel(sampled, xpos='right')

    ax.set_xticks(range(0, len(labels) + 1))
    ax.set_xticklabels([""] + list(range(1, len(labels) + 1)))
    title = "$" + title + "$"
    ax.set_title(title)
    ax.set_xlabel("state")
    ax.set_ylabel("number of steps")
    ax.set_ylim([0, 1])
    ax.legend()

    if (not out_path is None):
        fig.savefig(out_path)
        plt.close()


def plot_stateOccurence_matrix(data: dict,
                               out_dir: str = None,
                               s_values: list = None,
                               place_undersampling_threshold: bool = False,
                               title_suffix: str = None):
    """plot_stateOccurence_matrix
    This function generates a plot of the state occurrence matrix

    Parameters
    ----------
    data: dict
    out_dir: str, optional
        output file path (default None)
    s_values: list, optional
        list of s-values (default None)
    place_undersampling_threshold: bool, optional
        (default False)
    title_suffix: str, optional
        suffix for title (default None)

    Returns
    -------
    None
    """
    states_num = len(data[list(data.keys())[0]]["occurence_state"])

    occurrence_sampling_matrix = np.array(
        [np.array([data[replica]["occurence_state"][key] for key in sorted(data[replica]["occurence_state"])])
         for replica in sorted(data)]).T
    domination_sampling_matrix = np.array(
        [np.array([data[replica]["dominating_state"][key] for key in sorted(data[replica]["dominating_state"])])
         for replica in sorted(data)]).T

    # Plot occurence:
    ##Title setting
    title = "$ state occurence"
    if title_suffix is not None:
        title += title_suffix
    title += "$"

    fig = plt.figure(figsize=ps.figsize_doubleColumn)
    ax = fig.add_subplot(111)

    mappable = ax.matshow(occurrence_sampling_matrix, cmap="Blues")

    ## set ticks
    ax.set_yticks(range(0, states_num))
    ax.set_yticklabels(range(1, states_num + 1))

    # nice s-value x-axis
    if (not s_values is None):
        ax.set_xticks(np.arange(0, len(s_values) - 0.25))
        ax.set_xticklabels(nice_s_vals(s_values), rotation=45)

    ##set colorbar
    cax = fig.add_axes([ax.get_position().x1 + 0.1, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(mappable, cax=cax)
    cbar.set_label('time fraction', rotation=90)

    if (place_undersampling_threshold):
        for undersampling_ind in data:
            if (data[undersampling_ind]["undersampling"]):
                break
        ax.vlines(x=undersampling_ind - 1.5, ymin=-0.5, ymax=states_num - 0.5, color="red", lw=3, label="undersampling")

    ##labelling
    ax.set_title(title)
    ax.set_xlabel("s-values")
    ax.set_ylabel("states")
    ax.xaxis.set_ticks_position("bottom")

    fig.tight_layout()

    if (not out_dir is None):
        fig.savefig(out_dir + '/sampling_undersample_matrix.png', bbox_inches='tight')
        plt.close()

    # Plot domination samp:
    ##Title setting
    title = "$ state domination "
    if title_suffix is not None:
        title += title_suffix
    title += "$"

    fig = plt.figure(figsize=ps.figsize_doubleColumn)
    ax = fig.add_subplot(111)

    mappable = ax.matshow(domination_sampling_matrix, cmap="Reds")

    ## set ticks
    ax.set_yticks(range(0, states_num))
    ax.set_yticklabels(range(1, states_num + 1))

    if (not s_values is None):
        ax.set_xticks(range(0, len(s_values)))
        ax.set_xticklabels(nice_s_vals(s_values), rotation=45)

    ##set colorbar
    cax = fig.add_axes([ax.get_position().x1 + 0.1, ax.get_position().y0, 0.02, ax.get_position().height])
    cbar = plt.colorbar(mappable, cax=cax)
    cbar.set_label('time fraction', rotation=90)

    if (place_undersampling_threshold):
        for undersampling_ind in data:
            if (data[undersampling_ind]["undersampling"]):
                break
        ax.vlines(x=undersampling_ind - 1.5, ymin=-0.5, ymax=states_num - 0.5, color="k", lw=3, label="undersampling")

    ##labelling
    ax.set_title(title)
    ax.set_xlabel("s-values")
    ax.set_ylabel("states")
    ax.xaxis.set_ticks_position("bottom")

    fig.tight_layout()

    if (not out_dir is None):
        fig.savefig(out_dir + '/sampling_minstate_matrix.png', bbox_inches='tight')
        plt.close()