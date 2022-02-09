from typing import List

import numpy as np
import pandas as pd

from reeds.function_libs.visualization import plots_style as ps

def nice_s_vals(svals: list,
                base10=False) -> list:
    """nice_s_vals
    this function creates nice s-value labels for the plots

    Parameters
    ----------
    svals : list
        list of s-values
    base10 : bool, optional
        use base 10 (default False)

    Returns
    -------
    list
        list of labels for the s-values
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


def x_axis(ax,
           xBond: list = None,
           max_x=None,
           amount_of_x_labels=4):
    """x_axis
    This function formats the x-axis of a subplot

    Parameters
    ----------
    ax : AxesSubplot
    xBond : list, optional
        minimum and maximum x-axis value. Takes priority over max_x (default None)
    max_x : int, optional
        maximum x-axis value (default None)
    amount_of_x_labels : int
        amount of x-axis labels (default 4)

    Returns
    -------
    None
    """
    if (xBond):
        x_length = xBond[1] - xBond[0]
        steps = x_length // amount_of_x_labels if (not x_length // amount_of_x_labels == 0) else 1
        x_range = range(xBond[0], xBond[1] + 1, steps)

        ax.set_xlim(left=xBond[0], right=xBond[1] + 1)
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_range)
    elif (max_x):
        steps = int(max_x // amount_of_x_labels) if (max_x // amount_of_x_labels != 0) else 3

        x_range = range(0, max_x + 1, steps)
        ax.set_xlim(left=0, right=max_x + 1)
        ax.set_xticks(x_range)
        ax.set_xticklabels(x_range)
    else:
        raise IOError("Please provide either max_x or s_values to X_axis_for_s_plots() in visualisation ")
    return ax


def y_axis_for_s_plots(ax,
                       yBond: tuple = None,
                       s_values: list = None,
                       amount_of_y_labels=10):
    """y_axis_for_s_plots
    This function builds a nice inverse (so that s=1 is on top) y-axis.
    Parameters
    ----------
    ax : AxesSubplot
    yBond : tuple, optional
        minimum and maximum y-axis value (default None)
    s_values : list, optional
        list of s-values (default None)
    amount_of_y_labels : int, optional
        amount of y-axis labels (default 10)

    Returns
    -------
    None
    """
    if (not isinstance(yBond, type(None))):
        y_range = range(min(-1 * np.array(yBond)), max(-1 * np.array(yBond) + 1))
        step_size = len(y_range) // amount_of_y_labels if (len(y_range) // amount_of_y_labels > 0) else 1
        ax.set_ylim(bottom=min(y_range), top=max(y_range))
        ax.set_yticks(y_range[::step_size])

    if (not isinstance(s_values, type(None))):

        if(not yBond is None):
            y_range = range(min(-1 * np.array(yBond)), max(-1 * np.array(yBond) + 1))
        else:
            y_range = list(reversed([-x - 1 for x in range(0, len(s_values))]))

        step_size = len(y_range) // amount_of_y_labels if (len(y_range) // amount_of_y_labels > 0) else 1

        nice_y = nice_s_vals(s_values) + [""]
        ax.set_ylim([min(y_range), max(y_range)])
        ax.set_yticks(y_range[::step_size])
        ax.set_yticklabels([nice_y[ind]  for ind in range(len(y_range[::step_size]))][::-1])

    else:
        raise IOError("Please provide either max_y or s_values to Y_axis_for_s_plots() in visualisation ")

    return ax


def generate_trace_from_transition_dict(transition_dataFrame: pd.DataFrame,
                                        transition_range: float = 0.35) -> (dict, float, float):
    """generate_trace_from_transition_dict
    This function creates a replica trace plot from a transition dict
    Parameters
    ----------
    transition_dataFrame : pd.DataFrame
        pandas dataframe containing the replica transitions
    transition_range : float, optional
        default 0.35

    Returns
    -------
    traces : dict
        dict containing replica traces
    max_x : float
    max_y : float
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


def prepare_system_state_data(transition_dataFrame: pd.DataFrame,
                              cluster_size: int = 10,
                              sub_cluster_threshold: float = 0.6) -> dict:
    """prepare_system_state_data
    This function prepares a transition dict

    Parameters
    ----------
    transition_dataFrame : pd.DataFrame
        pandas dataframe containing information on the transitions
    cluster_size : int, optional
        default 10
    sub_cluster_threshold : float, optional
        default 0.6

    Returns
    -------
    replica_bins : dict
    marker_color_dict : dict
    num_states : int
    """
    replica_bins = {}
    for replica in sorted(set(transition_dataFrame.replicaID)):
        tmp_replica_df = transition_dataFrame.loc[transition_dataFrame.replicaID == replica]
        x = list(tmp_replica_df.trial)
        y = list(tmp_replica_df.position)
        reversed_order_y = list(map(lambda x: -1 * x, y))  # block_order replicas inverse for nicer visualisation

        cur_replica = transition_dataFrame.state_pot.index[0][0] # get current replica index from data frame
        num_states = len(transition_dataFrame.state_pot[cur_replica][0])

        marker_color_dict =  ps.active_qualitative_map_mligs(num_states + 1)
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


def discard_high_energy_points(pot_i:List[float],
                               threshold:int = 1000):
    """discard_high_energy_points
    This is a helper function, which will allow us to filter out
    some of the data we dont want to plot. The default value for the threshold
    is 1000 kJ/mol

    Parameters
    ----------
    pot_i : List[float]
        list of potential energies
    threshold: float, optional
        upper_threshold above which we discard data points (default 1000)

    Returns
    -------
    List[float]
        list of energies below the threshold
    """
    return [e for e in pot_i if e < threshold]

def determine_vrange(traj_data, num_states, tight:bool = False):
    """
    This functions allows use to determine the range of potential energy 
    values we will use in our plots!

    It determines this by looking at the minimum for each state of each 
    of the siulations (either the RE-EDS replicas, or the independent parallel EDS
    simulations in the pipelines first steps).
    
    Parameters
    ----------
    traj_data : List[pd.DataFrame]
        list of potential energies dataframes for each state
    num_states: int 
        Number of different end states in the EDS simulation.
    tight: bool
        If true, will also set a maximum close to the minimum found

    Returns
    -------
    List[float]
        list of energies below the threshold
    """

    v_max = 1000 # Manually defined
    current_min = 1000

    # 1: Find minimum energy value 

    for sub_traj in traj_data:
        for i in range(1, num_states+1):
            tmp_min = np.min(sub_traj['e'+str(i)])
            if current_min > tmp_min:
                current_min = tmp_min

    # 2: To make it look nicer, compare this minimum 
    # with a set of predefined values. 
    # These can of course be changed if the minimum is below that value.
    # or if we want tighter possibilities. 
    
    increment = 500

    predetermined_mins = np.arange(-5000, 1, increment)
    use_lower_lim = predetermined_mins[0]

    for v in predetermined_mins:
        if current_min < v: break
        use_lower_lim = v

    # Return upper and lower values to use.
    if tight: return [use_lower_lim, use_lower_lim+increment]
   
    return [use_lower_lim, v_max]
