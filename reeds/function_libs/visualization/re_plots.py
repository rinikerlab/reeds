from typing import List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from reeds.function_libs.visualization import plots_style as ps
from reeds.function_libs.visualization.utils import generate_trace_from_transition_dict, y_axis_for_s_plots, x_axis, \
    prepare_system_state_data


def plot_replica_transitions(transition_dict: pd.DataFrame,
                             out_path: str = None,
                             title_prefix: str = "test",
                             s_values=None,
                             cut_1_replicas=False,
                             xBond: tuple = None,
                             equilibration_border: int = None,
                             transparency=0.7,
                             use_gradient_colorMap=True,
                             show_repl_leg=False,
                             trace_line_width: float = 1) -> str:
    """plot_replica_transitions

    Parameters
    ----------
    transition_dict : pd.DataFrame
    out_path : str, optional
    title_prefix : str, optional
    s_values : List, optional
    cut_1_replicas : bool, optional
    xBond : tuple, optional
    equilibration_border : int, optional
    transparency : float, optional
    use_gradient_colorMap : bool, optional
    show_repl_leg : bool, optional
    trace_line_width : float, optional

    Returns
    -------
    out_path : str
        output file path

    """
    num_replicas = len(np.unique(transition_dict.replicaID))

    if use_gradient_colorMap:
        trace_color_dict = ps.active_qualitative_map_mligs(num_replicas)
        repnum = num_replicas
    else:
        trace_color_dict = ps.active_qualitative_map.colors[::-1]
        repnum = len(trace_color_dict)

    if (cut_1_replicas and s_values):
        count_1 = s_values.count(1.0)  # filter 1 replicas@!
        yBond = (count_1, len(s_values))
        s_values = s_values[count_1 - 1:] if (count_1 != 0) else s_values
    else:
        yBond = None

    # PREPRAE SETTINGS AND DATA:
    # init
    amount_of_x_labels = 5
    amount_of_y_labels = 21

    # replica_trace options:
    transition_range = 0.35
    trace_width = trace_line_width

    replica_traces = []
    # prepare transition dict
    traces, max_exch, max_y = generate_trace_from_transition_dict(transition_dataFrame=transition_dict,
                                                                  transition_range=transition_range)

    # DO PLOTTING
    fig = plt.figure(figsize=ps.figsize)
    ax = fig.add_subplot(1, 1, 1)

    for replica in range(1, num_replicas + 1):
        trace = traces[replica]
        label = str(replica)
        if (not isinstance(equilibration_border, type(None)) and equilibration_border > 0):
            equil_border_index = np.argmax(trace[0] > equilibration_border)
            replica_traces.append(ax.plot(trace[0][:equil_border_index], trace[1][:equil_border_index],
                                          label=label, lw=trace_width, alpha=transparency / 3,
                                          color=trace_color_dict[replica]))
            replica_traces.append(ax.plot(trace[0][equil_border_index:], trace[1][equil_border_index:],
                                          label=label, lw=trace_width, alpha=transparency,
                                          color=trace_color_dict[replica]))
        else:
            replica_traces.append(ax.plot(trace[0], trace[1], label=label, lw=trace_width, alpha=transparency,
                                          color=trace_color_dict[replica % repnum]))

    if (not isinstance(equilibration_border, type(None)) and equilibration_border > 0):
        ax.axvline(x=equilibration_border, linewidth=2, color='r', label="equilibrated")

    # axis
    if (yBond != None):
        y_axis_for_s_plots(ax=ax, s_values=s_values, yBond=yBond, amount_of_y_labels=amount_of_y_labels)
    else:
        y_axis_for_s_plots(ax=ax, s_values=s_values, amount_of_y_labels=amount_of_y_labels)
    if (xBond != None):
        x_axis(ax=ax, xBond=xBond, amount_of_x_labels=amount_of_x_labels)
    else:
        x_axis(ax=ax, max_x=max_exch, amount_of_x_labels=amount_of_x_labels)

    fig.suptitle(title_prefix + " - transitions/trial")
    ax.set_xlabel("exchange trials")
    ax.set_ylabel("replica position")

    # Legend positioning
    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.95, chartBox.height])

    if (len(transition_dict) <= 21) or show_repl_leg:
        leg = plt.legend()
        for line in leg.get_lines():
            line.set_linewidth(8.0)
        ncol = int(round(float(len(transition_dict)) / 14.0, 0)) if (
                round(float(len(transition_dict)) / 14.0, 0) > 0) else 1

        lgnd = ax.legend(title="replica:", loc=2, bbox_to_anchor=(1.05, 1), ncol=ncol, prop={"size": 10})
        if (out_path is None):
            return fig
        else:
            fig.savefig(out_path, bbox_extra_artists=(lgnd,), bbox_inches='tight')

    else:
        if (out_path is None):
            return fig
        else:
            fig.savefig(out_path, bbox_inches='tight', )
            plt.close(fig)

    return out_path


def plot_replica_transitions_min_states(transition_dict: dict,
                                        out_path: str,
                                        title_prefix: str = "test",
                                        s_values=None,
                                        show_only_states: list = None,
                                        cut_1_replicas=False,
                                        xBond: tuple = None,
                                        show_repl_leg: bool = False,
                                        cluster_size: int = 10,
                                        sub_cluster_threshold: float = 0.6):
    """plot_replica_transitions_min_states

    Parameters
    ----------
    transition_dict : dict
    out_path : str
    title_prefix : str, optional
    s_values : str, optional
    show_only_states : list, optional
    cut_1_replicas : bool, optional
    xBond : tuple, optional
    show_repl_leg : bool, optional
    cluster_size : int, optional
    sub_cluster_threshold : float, optional

    Returns
    -------
    None
    """
    if (cut_1_replicas and s_values):
        count_1 = s_values.count(1.0)  # filter 1 replicas@!
        yBond = (count_1 - 1, len(s_values))
        s_values = s_values[count_1 - 1:] if (not count_1 == 0) else s_values
    else:
        yBond = None

    # general_Settings:
    # init
    amount_of_x_labels = 5
    amount_of_y_labels = 21
    # replica_trace options:
    transition_range = 0.35
    trace_width = 1
    trace_transp = 0.1  # 0.99 #
    trace_color = "k"

    # marker settings for state vis:
    marker_shape = "|"
    marker_size = 5
    marker_transp = 0.80

    # do
    replica_traces = []
    # prepare transition dict
    traces, max_exch, max_y = generate_trace_from_transition_dict(transition_dataFrame=transition_dict,
                                                                  transition_range=transition_range)

    for replica in traces:
        trace = traces[replica]
        label = str(replica)
        replica_traces.append(
            plt.plot(trace[0], trace[1], label=label, lw=trace_width, alpha=trace_transp, color=trace_color))

    # prepare maker data:
    replica_bins, marker_color_dict, num_states = prepare_system_state_data(transition_dataFrame=transition_dict,
                                                                            cluster_size=cluster_size,
                                                                            sub_cluster_threshold=sub_cluster_threshold)
    for replica in sorted(replica_bins):
        ##plot markers
        labels = []
        fractions = []
        replica_bin = replica_bins[replica]
        for state in replica_bin:
            tmp_frac = cluster_size * (float(len(replica_bin[state][1])) / float(max_exch))
            fractions.append(tmp_frac)
            if (show_only_states == None or state + 1 in show_only_states):
                labels.append(plt.scatter(replica_bin[state][0], replica_bin[state][1], color=marker_color_dict[state],
                                          alpha=marker_transp, marker=marker_shape, s=marker_size))  #
            else:
                labels.append(plt.scatter([], [], color=marker_color_dict[state], alpha=marker_transp, s=marker_size))

    # axis
    ax = plt.gca()
    if (yBond != None):
        y_axis_for_s_plots(ax=ax, s_values=s_values, yBond=yBond, amount_of_y_labels=amount_of_y_labels)
    else:
        y_axis_for_s_plots(ax=ax, s_values=s_values, amount_of_y_labels=amount_of_y_labels)
    if (xBond != None):
        x_axis(ax=ax, xBond=xBond, amount_of_x_labels=amount_of_x_labels)
    else:
        x_axis(ax=ax, max_x=max_exch, amount_of_x_labels=amount_of_x_labels)

    plt.title(title_prefix + " - transitions/trial")
    plt.xlabel("exchange trials")
    plt.ylabel("replica position")

    # for checking implementation useful!
    # ax.yaxis.grid(True, which='major')
    # ax.xaxis.grid(True, which='major')

    chartBox = ax.get_position()
    ax.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.95, chartBox.height])
    if (len(transition_dict) == 1) or show_repl_leg:
        legend_text = ["state" + str(x + 1) + ": " + str(round(fractions[x] * 100, 3)) + "%" if (
                int(x) < num_states) else "undef: " + str(round(fractions[x] * 100, 3)) + "%" for x in
                       range(num_states + 1)]
        legend_text += ["replica " + str(x) for x in transition_dict]
        lgnd = ax.legend(labels + replica_traces[0], legend_text, loc=2, bbox_to_anchor=(1.05, 1), scatterpoints=3,
                         fontsize=8)
        plt.savefig(out_path, bbox_extra_artists=(lgnd,), bbox_inches='tight')

    else:
        legend_text = ["state" + str(x + 1) + ": " + str(round(fractions[x] * 100, 3)) + "%" if (
                int(x) < num_states) else "undef: " + str(round(fractions[x] * 100, 3)) + "%" for x in
                       range(num_states + 1)]
        legend_text += ["replica traces" for x in range(1)]
        lgnd = ax.legend(labels + replica_traces[0], legend_text, title="states:",
                         scatterpoints=3,
                         fontsize=8)

    if(out_path is None):
        plt.show()
    else:
        plt.savefig(out_path, bbox_inches='tight', bbox_extra_artists=(lgnd,))
        plt.close()


def plot_repPos_replica_histogramm(data: pd.DataFrame,
                                   replica_offset: int = 0,
                                   out_path: str = None,
                                   cut_front: int = 0,
                                   title: str = "test",
                                   s_values: List[float] = None):
    """plot_repPos_replica_histogramm

    This function is building a plot, that shows the position counts for each replica in the distribution (histgoram)


    Parameters
    ----------
    data: pd.DataFrame
        contains repdat information of the single replica traces
    replica_offset : int, optional
        cut off the first replicas with s=1 (default 0)
    out_path : str, optional
        write out the figures (default None)
    cut_front : int, optional
        cut first steps of traj (default 0)
    title : str, optional
        plot title (default "test")
    s_values : List[float], optional
        s_values for yaxis (default None)

    Returns
    -------
    None

    """

    # data preperation
    replicas = np.unique(data.replicaID)[replica_offset:]
    x = []
    y = []
    for replica in replicas:
        y_rep = np.array(data.loc[data.replicaID == replica].position)[cut_front:]

        for delNum in range(replica_offset + 1):
            y_rep = np.delete(y_rep, np.argwhere(y_rep == delNum))
        y_rep *= -1

        x_rep = np.array([replica for x in range(len(y_rep))])
        x.append(x_rep)
        y.append(y_rep)
    x = np.concatenate(x)
    y = np.concatenate(y)

    fig, ax = plt.subplots(ncols=1, figsize=ps.figsize_doubleColumn)
    hist = np.histogram2d(x=x, y=y, bins=(len(np.unique(x)), len(np.unique(y))))

    total_steps = np.sum(hist[0], axis=0)
    opt_vals = 1 / len(replicas)
    arr = hist[0] / total_steps
    surf = ax.imshow(arr.T[::-1], cmap=ps.active_gradient_centered, vmin=0, vmax=2.1 * opt_vals,
                     extent=[min(hist[1]), max(hist[1]), min(hist[2]), max(hist[2])])

    steps = len(s_values) // 10 if (len(s_values) // 10 > 0) else 10
    positionsY = [x + ((y - x) / 2) for x, y in zip(hist[2], hist[2][1:])][::steps]
    sval = list(reversed(s_values[::steps]))
    ax.set_yticks(positionsY)
    ax.set_yticklabels(sval)

    positionsX = [x + ((y - x) / 2) for x, y in zip(hist[1], hist[1][1:])]
    ax.set_xticks(positionsX)
    ax.set_xticklabels(range(1, len(replicas) + 1))

    ticks = np.round([0, opt_vals, opt_vals * 2], 2)
    cb = plt.colorbar(surf, ax=ax, ticks=ticks, )
    cb.set_label("residence time of replica [%]")
    # cbar.ax.set_yticklabels(ticks)

    ax.set_ylabel("position")
    ax.set_xlabel("replica")
    ax.set_title(title + " positions/replica")
    fig.tight_layout()

    if (not isinstance(out_path, type(None))):
        fig.savefig(out_path)
        plt.close()
        return hist
    else:
        return fig,  hist

def plot_exchange_freq(s_values:List[float], exchange_freq:List[float], outfile:str = None, title:str = None):
    """
    This function plots the frequency of exchange from the replica exchange data of a RE-EDS
    simulation. This allows to visualize the potential regions of bottleneck.
    
    Parameters
    ----------
    s_values: List [float]
        List of s-values for each of the replicas
    exchange_freq: List [float]
        Exchange frequencies obtained from calculate_exchange_freq(). Note the list has 
        (N-1) elements, where N is the number of s-values.
    outfile: str
        path to which the plot is made. If None is given, the plot is displayed (jupyter compatible)
    title: str
        title to give the plot
        
    Returns
    ----------
    None
    
    """
    x = np.arange(1, len(s_values)+1) + 0.5
    exchange_freq = np.append(exchange_freq, 0)

    # Find a nice automatic way to deal with the size issues here.
    size = 8 * len(s_values) / 13

    fig, ax = plt.subplots(figsize = [size,8])
    plt.grid(axis='y', lw = 1, ls = 'dashed')
    ax.set_axisbelow(True)

    ax.bar(x, exchange_freq, width=0.45, color='firebrick', edgecolor = 'black', alpha = 0.7)

    # Make the right labels
    labels = []
    for i in range(len(s_values)):
        labels.append(str(s_values[i]))

    ax.set_xticks(x-0.5)
    ax.set_xticklabels(labels, fontsize = 10)
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")

    # Set the x and y-limits properly
    ax.set_xlim(x[0]-1, x[len(x)-1])
    ax.set_ylim(0, 1.1)

    # Set the proper axis labels and title

    if title is None: title = 'Exchange frequency in the RE-EDS simulation'
    plt.title(title, fontsize = 14)

    plt.xlabel('s-value', fontsize = 14)
    plt.ylabel(r'$P_{exchange}$', fontsize = 14)

    if (outfile is None):
        plt.show()
    else:
        fig.savefig(outfile)
        plt.close()

    return None
