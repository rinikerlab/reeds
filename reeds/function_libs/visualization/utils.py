import numpy as np
import pandas as pd

from reeds.function_libs.visualization.pot_energy_plots import color_map_categorical


def nice_s_vals(svals: list, base10=False) -> list:
    """
    Args:
        svals (list):
        base10:
    """
    nicer_labels = []
    if (base10):
        for val in svals:
            if (float(np.log10(val)).is_integer() or val == min(svals)):
                nicer_labels.append(round(val, str(val).count("0") + 3))
            else:
                nicer_labels.append("")
    else:
        for val in svals:
            nicer_labels.append(round(val, str(val).count("0") + 2))
    return nicer_labels


def x_axis(ax, xBond: list = None, max_x=None, ammount_of_x_labels=4):
    """
    Args:
        ax:
        xBond (list):
        max_x:
        ammount_of_x_labels:
    """
    if (xBond):
        x_length = xBond[1] - xBond[0]
        steps = x_length // ammount_of_x_labels if (not x_length // ammount_of_x_labels == 0) else 1
        x_range = range(xBond[0], xBond[1] + 1, steps)

        ax.set_xlim(left=xBond[0], right=xBond[1] + 1)
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_range)
    elif (max_x):
        steps = int(max_x // ammount_of_x_labels) if (max_x // ammount_of_x_labels != 0) else 3

        x_range = range(0, max_x + 1, steps)
        ax.set_xlim(left=0, right=max_x + 1)
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_range)
    else:
        raise IOError("Please provide either max_x or s_values to X_axis_for_s_plots() in visualisation ")
    return ax


def y_axis_for_s_plots(ax, yBond: tuple = None, s_values: list = None, ammount_of_y_labels=10):
    """
        This function builds a nice inverse (so that s=1 is on top)  y-axis.
    Parameters
    ----------
    ax
    yBond
    s_values
    ammount_of_y_labels

    Returns
    -------

    """
    if (not isinstance(yBond, type(None))):
        y_range = range(min(-1 * np.array(yBond)), max(-1 * np.array(yBond) + 1))
        step_size = len(y_range) // ammount_of_y_labels if (len(y_range) // ammount_of_y_labels > 0) else 1
        ax.set_ylim(bottom=min(y_range), top=max(y_range))
        ax.set_yticks(y_range[::step_size])

    if (not isinstance(s_values, type(None))):

        if(not yBond is None):
            y_range = range(min(-1 * np.array(yBond)), max(-1 * np.array(yBond) + 1))
        else:
            y_range = list(reversed([-x - 1 for x in range(0, len(s_values))]))

        step_size = len(y_range) // ammount_of_y_labels if (len(y_range) // ammount_of_y_labels > 0) else 1

        nice_y = nice_s_vals(s_values) + [""]
        ax.set_ylim([min(y_range), max(y_range)])
        ax.set_yticks(y_range[::step_size])
        ax.set_yticklabels([nice_y[ind]  for ind in range(len(y_range[::step_size]))][::-1])

    else:
        raise IOError("Please provide either max_y or s_values to Y_axis_for_s_plots() in visualisation ")

    return ax


def generate_trace_from_transition_dict(transition_dataFrame: pd.DataFrame, transition_range: float = 0.35) -> (
        dict, float, float):
    """
    Args:
        transition_dataFrame:
        transition_range (float):
    """
    traces = {}
    max_x = 0
    max_y = 0
    for replica in sorted(set(transition_dataFrame.replicaID)):
        tmp_frame = transition_dataFrame.loc[transition_dataFrame.replicaID == replica]
        x = list(tmp_frame.trial)
        y = list(tmp_frame.position)

        max_x = max(x) if (max(x) > max_x) else max_x
        max_y = max(y) if (max(y) > max_y) else max_y

        #  transition_trace
        trace = [[], []]
        trace[0] = np.array(list(zip(x, x))) + np.array([-transition_range, transition_range])
        trace[0] = np.concatenate(trace[0])
        trace[1] = np.concatenate(np.array(list(zip(y, y)))) * -1
        traces.update({replica: trace})

    return traces, max_x, max_y


def prepare_system_state_data(transition_dataFrame: pd.DataFrame, cluster_size: int = 10,
                              sub_cluster_threshold: float = 0.6) -> dict:
    # prepare transition dict
    """
    Args:
        transition_dataFrame (dict):
        cluster_size (int):
        sub_cluster_threshold (float):
    """
    replica_bins = {}
    for replica in sorted(set(transition_dataFrame.replicaID)):
        tmp_replica_df = transition_dataFrame.loc[transition_dataFrame.replicaID == replica]
        x = list(tmp_replica_df.trial)
        y = list(tmp_replica_df.position)
        reversed_order_y = list(map(lambda x: -1 * x, y))  # block_order replicas inverse for nicer visualisation

        cur_replica = transition_dataFrame.state_pot.index[0][0] # get current replica index from data frame
        num_states = len(transition_dataFrame.state_pot[cur_replica][0])

        marker_color_dict = color_map_categorical(np.linspace(0, 1, num_states + 1))
        # marker plotting
        ##cluster_dtraj state data, to avoid to see only noise!
        cluster_counter = 0
        tmp_sub_bin = []
        tmp_sub_cluster_coords = ([], [])
        bin = {state: ([], []) for state in range(num_states + 1)}
        for ind2, z in enumerate(transition_dataFrame.loc[transition_dataFrame.replicaID == replica].state_pot):
            minE = min(z.values())
            cluster_counter += 1

            for ind, tmp in enumerate(sorted(z)):
                if (z[tmp] == minE):
                    tmp_sub_bin.append(ind)
                    tmp_sub_cluster_coords[0].append(x[ind2])
                    tmp_sub_cluster_coords[1].append(reversed_order_y[ind2])
                    break

            if (cluster_counter >= cluster_size):  # finished data gathering for one cluster_dtraj?
                # print(tmp_sub_bin)
                ratios = [tmp_sub_bin.count(x) / cluster_size for x in
                          range(0, num_states)]  # calculate presence of state
                above_treshold = {key: value for key, value in enumerate(ratios) if (value > sub_cluster_threshold)}
                major_presence = ratios.index(max(ratios)) if (max(
                    ratios) >= sub_cluster_threshold and len(
                    above_treshold) == 1) else num_states  # is one state dominating? - numstates is the index of undefined.
                # print(ratios)
                # print(above_treshold)
                # print(major_presence)
                # append date and reset vars
                bin[major_presence][0].append(tmp_sub_cluster_coords[0])
                bin[major_presence][1].append(tmp_sub_cluster_coords[1])
                cluster_counter = 0
                tmp_sub_bin = []
                tmp_sub_cluster_coords = ([], [])
        replica_bins.update({replica: bin})

    return replica_bins, marker_color_dict, num_states


def discard_high_energy_points(pot_i, threshold:int = 1000):
    """
    This is a helper function, which will allow us to filter out
    some of the data we dont want to plot. The default value for the threshold
    is 1000 kJ/mol

    Parameters
    ----------
    pot_i : List of float (e.x. potential energies)
    threshold: float, upper_threshold above which we discard data points
    Returns
    -------
    List of float (energies below the threshold)
    """
    return [e for e in pot_i if e < threshold]