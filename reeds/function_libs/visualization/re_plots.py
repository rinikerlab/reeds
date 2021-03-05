from typing import Union, List

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from reeds.function_libs.utils import plots_style as ps
from reeds.function_libs.visualization.utils import generate_trace_from_transition_dict, y_axis_for_s_plots, x_axis, \
    prepare_system_state_data
from reeds.function_libs.visualization.pot_energy_plots import color_map_categorical, figsize, color_map_centered


def s_optimization_visualization(s_opt_data: dict, out_path: str = None,
                                 nRT_range=None, avRT_range=None, s_range=None):
    niterations = len(s_opt_data)

    y_nRT = []
    y_RTd = []
    x = []

    x_svalues = []
    y_svalues = []

    bar_heights = []
    bar_x = []
    for it in sorted(s_opt_data):
        opti = s_opt_data[it]
        x.append(it.replace('sopt', ""))
        y_nRT.append(opti['nRoundTrips'])

        # dirty helper. not needed in future! TODO: remove
        roundTripTimeavg = 3333333 if (np.nan_to_num(opti['avg_rountrip_durations']) == 0) else np.nan_to_num(
            opti['avg_rountrip_durations'])
        y_RTd.append(roundTripTimeavg)

        x_svalues.extend(opti["s_values"])
        y_svalues.extend([int(it.replace('sopt', "")) for x in range(len(opti["s_values"]))])

        bar_heights.append([opti["state_sampling"][state] for state in opti["state_sampling"]])
        bar_x.append(np.array(
            [int(state.replace("V", "").replace("r", "").replace("i", "")) for state in
             opti["state_domination_sampling"]]))

    y_RTd = np.array(y_RTd) * 20 * 0.002
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(ncols=2, nrows=2, figsize=ps.figsize_doubleColumn)

    ax1.bar(x=x, height=y_nRT, color="dimgray")
    ax1.set_title("Total Number Of Roundtrips")
    ax1.set_ylabel("n [1]")
    ax1.set_xlabel("s-opt iteration")

    if (not isinstance(nRT_range, type(None))):
        ax1.set_ylim(nRT_range)

    ax2.bar(x=x, height=y_RTd, color="dimgray")
    ax2.set_title("Average Roundtrip Time")
    ax2.set_ylabel("t [ps]")
    ax2.set_xlabel("s-opt iteration")

    if (not isinstance(avRT_range, type(None))):
        ax2.set_ylim(avRT_range)
    else:
        ax2.set_ylim([0, max(y_RTd) * 1.2])

    x_svalues = (-1 * np.log10(np.array(x_svalues)))[::-1]
    y_svalues = y_svalues[::-1]

    ax3.scatter(x=x_svalues, y=y_svalues, c="k", alpha=0.6)

    ax3.set_yticks(np.unique(y_svalues))
    ax3.set_yticklabels(np.unique(y_svalues))

    ax3.set_title("Replica Placement")
    ax3.set_ylabel("s-opt iteration")
    ax3.set_xlabel("-log(s)")

    # Making the bottom right corner plot

    num_sopts = len(bar_heights)
    num_states = (len(bar_heights[0]))

    labels = [str(i) for i in range(1, num_states + 1)]

    # Making the offsets between the different bars

    width = 1 / (niterations * 1.15)
    x = np.arange(num_states) + 0.5 * num_states * width  # the label locations

    num = num_sopts - 1

    # finding proper offsets

    if num % 2 == 0:
        offsets = np.arange(-num / 2, num / 2 + 0.001, step=1)
    else:
        offsets = np.arange(-num / 2, num / 2 + 0.001)

    for i in range(num_sopts):
        normalized_heights = bar_heights[i] / np.sum(bar_heights[i])
        percent_heights = [100 * j for j in normalized_heights]

        ax4.bar(x + offsets[i] * width, percent_heights, width=width,
                alpha=i / num_sopts * 0.8 + 0.2, color=["C" + str(k) for k in range(num_states)],
                label="iteration " + str(i + 1))

    xmin = x[0] + offsets[0] / 3
    xmax = x[num_states - 1] + offsets[num_sopts - 1] / 3

    ax4.hlines(y=100 / num_states, xmin=xmin, xmax=xmax, color="red")
    ax4.set_xlim([xmin, xmax])

    ax4.set_title("State Sampling For $s=1$")
    ax4.set_ylabel("fraction [%]")
    ax4.set_xlabel("states")

    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()

    fig.tight_layout()

    if (out_path is None):
        return fig
    else:
        fig.savefig(out_path)


def visualize_s_optimisation_sampling_optimization(s_opt_data: dict, out_path: str = None) -> Union[str, plt.figure]:
    """

    Parameters
    ----------
    s_opt_data
    out_path

    Returns
    -------

    """
    fig, ax = plt.subplots(ncols=1, figsize=ps.figsize_doubleColumn)
    mae_mean = []
    mae_std = []
    for iteration, data in s_opt_data.items():
        mae_mean.append(data['MAE_optimal_sampling'])
        mae_std.append(data['MAE_std_optimal_sampling'])

    # ax.errorbar(list(range(1,len(maes)+1)), maes, approach_MAE_optSamp_std[approach],
    ax.plot(list(range(1, len(mae_mean) + 1)), mae_mean, alpha=0.75, c="k")

    ax.set_title("Sampling distribution deviaton from optimal sampling distribution")
    ax.set_ylabel("MAE [%]")
    ax.set_xlabel("s-opt iterations")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, )

    if (out_path is None):
        fig.show()
    else:
        fig.savefig(out_path)
        plt.close()


def visualize_s_optimisation_convergence(s_opt_data: dict, out_path: str = None, convergens_radius: int = 100) -> \
        Union[str, plt.figure]:
    """
        This function visualizes the s-optimization round trip optimization time efficency convergence.
        Ideally the roundtrip time is  reduced by the s-optimization, if the average time converges towards 1ps it is assumed to be converged.

    Parameters
    ----------
    s_opt_data : dict
        contains statistics over the optimization is generated by RE_EDS_soptimizatoin_final
    out_path:str, optional
        if provided, the plot will be saved here. if not provided, the plot will be shown directly.

    Returns
    -------
    Union[str, plt.figure]
        the outpath is returned if one is given. Alternativley the plot direclty will be returned.
    """
    y_RTd_efficency = []
    for it in sorted(s_opt_data):
        if ("avg_rountrip_duration_optimization_efficiency" in s_opt_data[it]):
            y_RTd_efficency.append(s_opt_data[it]["avg_rountrip_duration_optimization_efficiency"])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=ps.figsize_doubleColumn, sharex=True, sharey=True)

    ax.plot(np.nan_to_num(np.log10(y_RTd_efficency)), label="efficiency", c="k")
    ax.hlines(y=np.log10(convergens_radius), xmin=0, xmax=8, label="$convergence criterium$", color="grey")

    ax.set_ylim([-2, 4])
    ax.set_xlim([0, 7])
    ax.set_xticks(range(len(s_opt_data) - 1))
    ax.set_xticklabels([str(x) + "_" + str(x + 1) for x in range(1, len(s_opt_data))])
    ax.set_yticks(range(-1, 4))
    ax.set_yticklabels(range(-1, 4))

    ax.set_ylabel("$log(\overline{\\tau_j} - \overline{\\tau_i})$ [ps]")
    ax.set_xlabel("iteration ij")
    ax.set_title("AvgRoundtriptime optimization efficiency")
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0)
    if (out_path is None):
        return fig
    else:
        fig.savefig(out_path)
        plt.close()
        return out_path


def plot_replica_transitions(transition_dict: pd.DataFrame, out_path: str = None, title_prefix: str = "test",
                             s_values=None, cut_1_replicas=False, xBond: tuple = None, equilibration_border: int = None,
                             transparency=0.7, use_gradient_colorMap=True, show_repl_leg=False,
                             trace_line_width: float = 1) -> str:
    num_replicas = len(np.unique(transition_dict.replicaID))

    if use_gradient_colorMap:
        trace_color_dict = ps.active_qualitative_map(np.linspace(1, 0, num_replicas))
        repnum = num_replicas
    else:
        trace_color_dict = color_map_categorical.colors[::-1]
        repnum = len(trace_color_dict)

    if (cut_1_replicas and s_values):
        count_1 = s_values.count(1.0)  # filter 1 replicas@!
        yBond = (count_1, len(s_values))
        s_values = s_values[count_1 - 1:] if (count_1 != 0) else s_values
    else:
        yBond = None

    # PREPRAE SETTINGS AND DATA:
    # init
    ammount_of_x_labels = 5
    ammount_of_y_labels = 21

    # replica_trace options:
    transition_range = 0.35
    trace_width = trace_line_width

    replica_traces = []
    # prepare transition dict
    traces, max_exch, max_y = generate_trace_from_transition_dict(transition_dataFrame=transition_dict,
                                                                  transition_range=transition_range)

    # DO PLOTTING
    fig = plt.figure(figsize=figsize)
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
        y_axis_for_s_plots(ax=ax, s_values=s_values, yBond=yBond, ammount_of_y_labels=ammount_of_y_labels)
    else:
        y_axis_for_s_plots(ax=ax, s_values=s_values, ammount_of_y_labels=ammount_of_y_labels)
    if (xBond != None):
        x_axis(ax=ax, xBond=xBond, ammount_of_x_labels=ammount_of_x_labels)
    else:
        x_axis(ax=ax, max_x=max_exch, ammount_of_x_labels=ammount_of_x_labels)

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


def plot_replica_transitions_min_states(transition_dict: dict, out_path: str, title_prefix: str = "test", s_values=None,
                                        show_only_states: list = None, cut_1_replicas=False, xBond: tuple = None,
                                        show_repl_leg=False,
                                        cluster_size=10, sub_cluster_threshold=0.6):
    """
    Args:
        transition_dict (dict):
        out_path (str):
        title_prefix (str):
        s_values:
        show_only_states (list):
        cut_1_replicas:
        xBond (tuple):
        show_repl_leg:
        cluster_size:
        sub_cluster_threshold:
    """
    if (cut_1_replicas and s_values):
        count_1 = s_values.count(1.0)  # filter 1 replicas@!
        yBond = (count_1 - 1, len(s_values))
        s_values = s_values[count_1 - 1:] if (not count_1 == 0) else s_values
    else:
        yBond = None

    # general_Settings:
    # init
    ammount_of_x_labels = 5
    ammount_of_y_labels = 21
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
        y_axis_for_s_plots(ax=ax, s_values=s_values, yBond=yBond, ammount_of_y_labels=ammount_of_y_labels)
    else:
        y_axis_for_s_plots(ax=ax, s_values=s_values, ammount_of_y_labels=ammount_of_y_labels)
    if (xBond != None):
        x_axis(ax=ax, xBond=xBond, ammount_of_x_labels=ammount_of_x_labels)
    else:
        x_axis(ax=ax, max_x=max_exch, ammount_of_x_labels=ammount_of_x_labels)

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

    if (out_path is None):
        plt.show()
    else:
        plt.savefig(out_path, bbox_inches='tight', bbox_extra_artists=(lgnd,))
        plt.close()


def plot_repPos_replica_histogramm(data: pd.DataFrame, replica_offset: int = 0, out_path: str = None,
                                   cut_front: int = 0, title: str = "test", s_values: List[float] = None):
    """plot_repPos_replica_histogramm

        This function is building a plot, that shows the position counts for each replica in the distribution (histgoram)


    Parameters
    ----------
    data: pd.DataFrame
        contains repdat information of the single replica traces

    replica_offset
        cut off the first replicas with s=1
    out_path
        write out the figures
    cut_front
        cut first steps of traj
    title
        plot title
    s_values
        s_values for yaxis

    Returns
    -------


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

    fig, ax = plt.subplots(ncols=1, figsize=figsize)
    hist = np.histogram2d(x=x, y=y, bins=(len(np.unique(x)), len(np.unique(y))))

    total_steps = np.sum(hist[0], axis=0)
    opt_vals = 1 / len(replicas)
    arr = hist[0] / total_steps
    surf = ax.imshow(arr.T[::-1], cmap=color_map_centered, vmin=0, vmax=2.1 * opt_vals,
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
        return fig, hist
