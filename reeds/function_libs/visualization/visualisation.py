"""
visualisation
-------------

visualisation.py
This File contains functions which are generating matplotlib plots for reeds sims.
"""

from typing import Iterable, List, Tuple, Union

# plotting
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000  # avoid chunksize error

from matplotlib import pyplot as plt

# PLOT STYLE
from reeds.function_libs.utils import plots_style as ps

# General Plottsettings
for key, value in ps.plot_layout_settings.items():
    matplotlib.rcParams[key] = value

color_gradient = ps.active_gradient_map
color_map_categorical = ps.active_qualitative_map
color_map_centered = ps.active_gradient_centered

figsize = ps.figsize
alpha = ps.alpha_val


##helper_Functions
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


# Plots

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

def plot_optimized_states_potential_energies(outfile:str, ene_trajs):
    """
    Plots the potential energy distributions of all states in a single plot,
    where the data is represented as a histogram.

    Parameters
    ----------
    outfile: str containing the path to the output png file
    ene_trajs: pandas DataFrame containing trajectory pot energies

    Returns
    -------
    None
    """

    nstates = len(ene_trajs)

    colors = ps.candide_colors

    # Split the different states in subplots:

    ncols = 4 if nstates > 11 else 3
    nrows = int(np.ceil(nstates/ncols))

    figsize = [5*ncols, 5*nrows]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    fig.suptitle('Potential Energy Distribution for optimized States', fontsize = 20)

    row_num = 0 # this starts at 0, and gets updated when col_num = ncols
    col_num = 0

    for i in range(nstates):
        if col_num == ncols-1: row_num +=1
        col_num = i % ncols
        state_num = i+1

        # data to plot
        energies = np.array(ene_trajs[i]["e" + str(state_num)])

        # discard any high energy points (i.e. points with V > 1000 kJ/mol)
        up_energies = discard_high_energy_points(energies, threshold = 1000)

        # Don't plot anything if we discarded all of the data
        if len(up_energies) < 1: continue

        # calculate average and std
        avg = (np.average(up_energies))
        std = (np.std(up_energies))

        legend = "State " + str(state_num) + "\n"
        legend += "n: "   + str(len(up_energies))  + "\n"
        legend += "avg: " + str(round(avg, 2))  + "\n"
        legend += "std: " + str(round(std, 2))

        # plot the actual data now:
        hist1 = axes[row_num, col_num].hist(up_energies, bins = 100, label = legend, color=colors[i%len(colors)])

        # formatting:
        axes[row_num, col_num].legend(loc='upper right', fontsize=12, edgecolor='black')

        axes[row_num, col_num].set_xlim([-1250, 0])
        axes[row_num, col_num].set_ylim([0, 1.35 * max(hist1[0])])

        axes[row_num, col_num].set_ylabel('Count')
        axes[row_num, col_num].set_xlabel(r'$V_{i}$ [kJ/mol]')

    # Done plotting everything, save the figure

    plt.tight_layout()
    plt.savefig(outfile, facecolor='white')
    plt.close()

    return None

def plot_energy_distribution_by_replica(traj_data, outfile, replica_num, s_value, manual_xlim = None, shared_xaxis = True):
    """
    Plots the potential energy distributions for the data generated
    in a RE-EDS simulation (or for N independant EDS simulations (i.e. lower bound))

    It generates the plot for a specific replica in which the
    data for each end states is represented in a different subplot.

    Parameters
    ----------
    traj_data : pandas DataFrame containing the data for the specific replica of interest
    outfile : str, name of the output png file
    replica_num : int, replica number (used an index)
    s_value : int,   s value associated to this replica
    manual_xlim: List of float, minimum and maximum value for the x-axes
    shared_xaxis: boolean, determines if each plot has the same axes limits

    Returns
    -------
    None
    """

    # find number of states and replicas:
    nstates = 0
    for elem in traj_data.keys():
        if (elem[0] == 'e' and elem != 'eR')  : nstates += 1

    n_replicas = len(traj_data)

    # Find the minimum energy value for the axes
    x_min = 0
    for j in range(1, nstates+1):
        sim_min_energy = min(np.array((traj_data["e" + str(j)])))
        if sim_min_energy < x_min:
            x_min = sim_min_energy

    x_min -= 200 # Decrement so data is more clear.
    upper_threshold = 1000 # kJ/mol, we discard data above this limit

    xlimits = [x_min, upper_threshold]

    # General plotting options:
    colors = ps.candide_colors

    ncols = 4 if nstates > 11 else 3
    nrows = int(np.ceil(nstates/ncols))

    figsize = [5*ncols, 5*nrows]

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    fig.suptitle('Average Potential Energy - Replica ' + str(replica_num)
                 + ' - s = ' + str(s_value), fontsize = 20, y=0.995)

    # Plot each histogram one by one:
    row_num = 0 # this starts at 0, and gets updated when col_num = ncols
    col_num = 0

    for i in range(nstates):
        if col_num == ncols-1: row_num +=1
        col_num = i % ncols
        state_num = i+1

        # get data, and discard high energies
        energies = np.array(traj_data["e" + str(state_num)])
        up_energies = discard_high_energy_points(energies, threshold = upper_threshold)

        # Don't plot anything if we discarded all of the data
        if len(up_energies) <= 1 : continue

        # calculate average and std
        avg = (np.average(up_energies))
        std = (np.std(up_energies))

        legend = "State " + str(state_num) + "\n"
        legend += "n: "   + str(len(up_energies))  + "\n"
        legend += "avg: " + str(round(avg, 2))  + "\n"
        legend += "std: " + str(round(std, 2))

        # Plot the data:
        hist1 = axes[row_num, col_num].hist(up_energies, bins = 250, label = legend, color=colors[i%len(colors)])

        axes[row_num, col_num].legend(loc='upper right', fontsize=12, edgecolor='black')

        # Use the correct set of x-limits
        if manual_xlim is None: xlimits = [-1250, upper_threshold]
        else: xlimits = manual_xlim

        if not shared_xaxis:
            high_lim = min(upper_threshold, max(up_energies))
            low_lim  = min(up_energies)
            low_lim  = low_lim - 0.1 * abs(low_lim)
            high_lim = high_lim + 0.1* abs(high_lim)
            xlimits  = [low_lim, high_lim]

        axes[row_num, col_num].set_xlim(xlimits)
        axes[row_num, col_num].set_ylabel('Count')
        axes[row_num, col_num].set_xlabel(r'$V_{i}$ [kJ/mol]')

        # Set the ylim so the legend can be read well / no overlap with the plot
        axes[row_num, col_num].set_ylim([0, 1.35 * max(hist1[0])])

    # Done plotting everything, save the figure

    plt.tight_layout()
    plt.savefig(outfile, facecolor='white')
    plt.close()

    return None

def plot_energy_distribution_by_state(energy_trajs, outfile, state_num, s_values, manual_xlim = None, shared_xaxis = True):
    """
    Plots the potential energy distributions for the data generated
    in a RE-EDS simulation (or for N independant EDS simulations (i.e. lower bound))

    The overall plot generated corresponds to a specific end state,
    each replica is represented in a different subplot.

    Parameters
    ----------
    energy_trajs : List of pandas DataFrame containing the data for all states/replicas
    outfile: str name of the output png file
    state_num: int number of the state to plot
    s_values : List of float s-values of all the replicas
    manual_xlim: List of float, minimum and maximum value for the x-axes
    shared_xaxis: boolean, determines if each plot has the same axes limits
                  (this includes all other plots, as this function is called from a loop)

    Returns
    -------
    None
    """

    # find number of states and replicas:
    nstates = 0
    for elem in energy_trajs[0].keys():
        if (elem[0] == 'e' and elem != 'eR')  : nstates += 1

    n_replicas = len(energy_trajs)

    # Find the minimum energy value for the axes
    x_min = 0

    for i in range(n_replicas):
        for j in range(1, nstates+1):
            sim_min_energy = min(np.array(energy_trajs[i]["e" + str(j)]))
            if sim_min_energy < x_min:
                x_min = sim_min_energy

    x_min -= 200 # Decrement so data is more clear.
    upper_threshold = 1000 # kJ/mol, we discard data above this limit

    xlimits = [x_min, upper_threshold]

    # General plotting options:

    colors = ps.candide_colors
    color = colors[state_num-1%len(colors)]

    n_replicas = len(energy_trajs)

    ncols = 4 if n_replicas > 11 else 3
    nrows = int(np.ceil(n_replicas/ncols))

    figsize = [5*ncols, 5*nrows]

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    fig.suptitle('Average Potential Energy - State ' + str(state_num) + '\n', fontsize = 20, y = 0.995)

    # Plot each histogram one by one:

    row_num = 0 # this starts at 0, and gets updated when col_num = ncols
    col_num = 0

    for i in range(n_replicas):
        if col_num == ncols-1: row_num +=1
        col_num = i % ncols
        replica_num = i+1

        # data to plot
        energies = np.array(energy_trajs[i]["e" + str(state_num)])

        # Updated energies, is to plot the histogram.
        up_energies = discard_high_energy_points(energies, threshold = upper_threshold)

        # Don't plot anything if we discarded all of the data
        if len(up_energies) <= 1 : continue

        # calculate average and std
        avg = (np.average(up_energies))
        std = (np.std(up_energies))

        legend = "State " + str(state_num) + "\n"
        legend += "n: "   + str(len(up_energies))  + "\n"
        legend += "avg: " + str(round(avg, 2))  + "\n"
        legend += "std: " + str(round(std, 2))

        # Plot the data:

        hist1 = axes[row_num, col_num].hist(up_energies, bins = 100, label = legend, color = color)
        axes[row_num, col_num].legend(loc='upper left', fontsize=12, edgecolor='black')

        # Use the correct set of x-limits
        if manual_xlim is None: xlimits = [-1250, upper_threshold]
        else: xlimits = manual_xlim

        if not shared_xaxis:
            high_lim = min(upper_threshold, max(up_energies))
            low_lim  = min(up_energies)
            low_lim  = low_lim - 0.1 * abs(low_lim)
            high_lim = high_lim + 0.1* abs(high_lim)
            xlimits  = [low_lim, high_lim]

        axes[row_num, col_num].set_xlim(xlimits)

        axes[row_num, col_num].set_ylabel('Count')
        axes[row_num, col_num].set_xlabel(r'$V_{i}$ [kJ/mol]')
        axes[row_num, col_num].set_title('Replica ' + str(replica_num) + ' - s = ' + str(s_values[i]))

        # Set the ylim so the legend can be read well / no overlap with the plot
        axes[row_num, col_num].set_ylim([0, 1.35 * max(hist1[0])])

    # Done plotting everything, save the figure

    plt.tight_layout()
    plt.savefig(outfile, facecolor='white')
    plt.close()

    return None

def plot_ref_pot_energy_distribution(energy_trajs, outfile, s_values, optimized_state:bool = False):
    """
    Plots the reference potential energy distribution
    for either the optimized states, or a set of EDS/RE-EDS
    simulation, for which data has been previously parsed by
    parse_csv_energy_trajectories()

    Parameters
    ----------
    energy_trajs: List of pandas DataFrame containing traj info
    outfile: str: path to the output png file
    s_values: List of s-values corresponding to the different replicas
               an empty list can be given when optimized_state = True
    optimized_state: boolean, True when we want to make the plot for the optimized states

    Returns
    -------
    None
    """

    upper_threshold = 1000 # kJ/mol, we discard data above this limit

    # General plotting options:

    n_replicas = len(energy_trajs)

    ncols = 4 if n_replicas > 11 else 3
    nrows = int(np.ceil(n_replicas/ncols))

    figsize = [5*ncols, 5*nrows]

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    fig.suptitle('Potential Energy Distribution of the Reference State', fontsize = 20, y =0.995)

    # Plot each histogram one by one:

    row_num = 0 # this starts at 0, and gets updated when col_num = ncols
    col_num = 0

    for i in range(n_replicas):
        if col_num == ncols-1: row_num +=1
        col_num = i % ncols
        replica_num = i+1

        # data to plot
        energies = np.array(energy_trajs[i]["eR"])
        up_energies = discard_high_energy_points(energies, threshold = upper_threshold)

        # Don't plot anything if we discarded all of the data
        if len(up_energies) < 1 : continue

        # calculate average and std
        avg = (np.average(up_energies))
        std = (np.std(up_energies))

        legend = "Reference State\n"
        legend += "n: "   + str(len(up_energies))  + "\n"
        legend += "avg: " + str(round(avg, 2))  + "\n"
        legend += "std: " + str(round(std, 2))

        # Plot the data:

        hist1 = axes[row_num, col_num].hist(up_energies, bins = 100, label = legend, color = 'firebrick')
        axes[row_num, col_num].legend(loc='upper right', fontsize=12, edgecolor='black')
        #
        axes[row_num, col_num].set_xlim([min(up_energies) - 100, upper_threshold])
        axes[row_num, col_num].set_ylim([0, 1.35 * max(hist1[0])])
        axes[row_num, col_num].set_ylabel('Count')
        axes[row_num, col_num].set_xlabel(r'$V_{R}$ [kJ/mol]')

        if optimized_state :
            axes[row_num, col_num].set_title('System biased to state ' + str(replica_num))
        else :
            axes[row_num, col_num].set_title('Replica ' + str(replica_num) +
                ' - s = ' + str(s_values[i]))

    # Done plotting everything, save the figure

    plt.tight_layout()
    plt.savefig(outfile, facecolor='white')
    plt.close()

    return None

def plot_ref_pot_ene_timeseries(ene_trajs, outfile, s_values, optimized_state:bool = False):
    """
    Plots the reference potential energy timeseries
    for either the optimized states, or a set of EDS/RE-EDS
    simulation, for which data has been previously parsed by
    parse_csv_energy_trajectories()

    Parameters
    ----------
    energy_trajs: List of pandas DataFrame containing traj info
    outfile: str, path to the output png file
    s_values: List of float, s-values corresponding to the different replicas
               an empty list can be given when optimized_state = True
    optimized_state: boolean, True when we want to make the plot for the optimized states

    Returns
    -------
    None
    """

    upper_threshold = 1000
    n_replicas = len(ene_trajs)

    # Split the different states in subplots:

    ncols = 4 if n_replicas > 11 else 3
    nrows = int(np.ceil(n_replicas/ncols))

    figsize = [5*ncols, 5*nrows]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    fig.suptitle('Potential Energy Timeseries of the Reference State', fontsize = 20, y=0.995)

    # Plot each histogram one by one:

    row_num = 0 # this starts at 0, and gets updated when col_num = ncols
    col_num = 0

    for i in range(n_replicas):
        if col_num == ncols-1: row_num +=1
        col_num = i % ncols
        replica_num = i+1

        # Plot the data
        e_ref = np.array(ene_trajs[i]['eR'])
        t = np.array(ene_trajs[i]['time'])

        axes[row_num, col_num].scatter(t, e_ref, color='firebrick', s = 4, marker = 'D')
        axes[row_num, col_num].set_xlabel('time [ps]')
        axes[row_num, col_num].set_ylabel(r'$V_{R}$ [kJ/mol]')

        axes[row_num, col_num].set_ylim(min(e_ref)-100,max(e_ref)+100 )

        if optimized_state :
            axes[row_num, col_num].set_title('System biased to state ' + str(replica_num))
        else :
            axes[row_num, col_num].set_title('Replica ' + str(replica_num) +
                ' - s = ' + str(s_values[i]))

    # Done plotting everything, save the figure

    plt.tight_layout()
    plt.savefig(outfile, facecolor='white')
    plt.close()

    return None


#
# This is the violin plot representation of the potential energy
# distributions, which might be removed.
#

def plot_potential_distribution(potentials: (pd.Series or pd.DataFrame), out_path: str,
                                y_label="V/[kj/mol]", y_range: Iterable[float] = [-1000, 1000],
                                title: str = "Energy Distribution in range", label_medians: bool = True) -> str:
    # get good suplot distributions
    nstates = sum([1 for x in potentials.columns if (x.startswith("e"))]) - 1  # -1 for reference state

    # build figure
    fig, ax = plt.subplots(ncols=1, nrows=1)

    boxes = []
    box_lables = []
    # plot states
    for state in range(nstates):
        boxes.append(list(filter(lambda x: not x > max(y_range), potentials["e" + str(state + 1)])))
        box_lables.append("e" + str(state + 1))

    nans = [float('nan'), float('nan')]

    boxen = ax.violinplot([box or nans for box in boxes], showmedians=True)
    ax.set_xticklabels([" "] + box_lables)
    ax.set_xlabel("states")
    ax.set_ylabel(y_label)

    # annotate
    if (label_medians):
        for median_line in boxen["cmedians"].get_segments():
            if(len(median_line) == 0):
                continue
            x = sum(np.array(median_line)[:, 0]) / 2
            y = sum(np.array(median_line)[:, 1]) / 2 + 2
            plt.text(x, y, '%.2f' % y, horizontalalignment='center')

    # final layout
    ax.set_title(title)
    fig.tight_layout()

    # save files
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_replicaEnsemble_property_2D(ene_trajs: List[pd.DataFrame], out_path: str,
                                     temperature_property: str = "solvtemp2") -> str:
    """
    ..autofunction: plot_replicaEnsemble_property_2D
    :author: Kay Schaller

    Plots the temperature time series for all replicas as heat map.

    :param ene_trajs: ene_ana_property object, that contains time and solv2temp property
    :type ene_trajs: List[ene_traj] - ene_ana_property_traj class from pygromos.files.energy
    :param temperature_property: is the name of the property, that is representing the Temperature
    :type temperature_property: str
    :param out_path: path to output directroy
    :type out_path:str
    :return: out_file path
    :rtype: str
    """

    # Params
    outfilename = out_path
    num_ticks = 5

    cmap = plt.cm.coolwarm
    withlines = False
    step = 2400  # NSTLIM*NRETRIAL
    nostep = 5  # NRETRIAL
    stepsize = 0.002
    writerate = 6  # NTWE
    s_val_num = len(ene_trajs)

    # build up plot data
    x = []
    for ene_traj in ene_trajs:
        x_temp = list(map(lambda z: float(z), ene_traj[temperature_property].tolist()))
        time = list(map(lambda z: float(z), ene_traj["time"].tolist()))

        if len(x) == 0:
            x = x_temp
        else:
            x = np.vstack([x, x_temp])

    # plotting
    plt.imshow(x, aspect='auto', cmap=cmap, interpolation='None')  # bilinear plots
    plt.ylabel("replica")
    plt.xlabel("time [ps]")
    plt.yticks(list(range(s_val_num)), list(range(1, s_val_num + 1)))
    plt.xticks([float(i) for i in np.arange(0.0, len(time) + 1, (len(time) + 1) / num_ticks)],
               [round(float(i), 3) for i in np.arange(0.0, max(time) + stepsize, (max(time) + stepsize) / num_ticks)])
    ##legend
    cbar = plt.colorbar()
    cbar.set_label('temperature [K]', rotation=90)

    if withlines:
        for i in range(nostep):
            if i > 0:
                plt.axvline(x=step * i / writerate, linewidth=1, color='k')
    plt.savefig(outfilename)
    plt.close()
    return outfilename

# comment from Candide:
# I suggest completely removing this function
# as it does the same as scatter_potential_timeseries (but not as well in my opinion)

def plot_potential_timeseries(time: pd.Series, potentials: (pd.DataFrame or pd.Series),
                              title: str,
                              y_range: Tuple[float, float] = None, x_range: Tuple[float, float] = None,
                              x_label: str = "t", y_label: str = "V/kJ",
                              alpha: float = 0.6,
                              out_path: str = None):
    fig = plt.figure(figsize=[20, 15])
    ax = fig.add_subplot(1, 1, 1)
    for y in potentials.columns:
        ax.plot(time, potentials[y], label=y, alpha=alpha)

    ax.legend()
    if (y_range):
        ax.set_ylim(y_range)
    if (x_range):
        ax.set_xlim(x_range)
    else:
        ax.set_xlim((time.min(), time.max()))

    if (y_label):
        ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)
    fig.suptitle(title + " timeseries")

    if (out_path == None):
        fig.show()
    else:
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def scatter_potential_timeseries(time: pd.Series, potentials: (pd.DataFrame or pd.Series),
                                 title: str,
                                 y_range: Tuple[float, float] = None, x_range: Tuple[float, float] = None,
                                 x_label: str = "t", y_label: str = "V [kJ]",
                                 marker_size=5, alpha: float = 0.5, show_legend=True,
                                 out_path: str = None):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    for prop in potentials.columns:
        ax.scatter(time, potentials[prop], label=prop, alpha=alpha, s=marker_size)

    if (show_legend):
        ax.legend()

    if (y_range):
        ax.set_ylim(y_range)
    if (x_range):
        ax.set_xlim(x_range)
    else:
        ax.set_xlim((time.min(), time.max()))

    if (y_label):
        ax.set_ylabel(y_label)

    ax.set_xlabel(x_label)

    fig.suptitle(title + " timeseries")
    if (out_path == None):
        fig.show()
    else:
        fig.savefig(out_path)
        plt.close(fig)
    return out_path


def plot_sampling_grid(traj_data, out_path= None, y_range = [-1000, 1000], title = None):
    """
    Plots the potential energy of each end state in a single plot (grid of subplots)
    where each subplot corresponds to one end state.

    Parameters
    ----------
    traj_data: pandas dataFrame containing timestep/potential energy info of the trajectory.
    out_path: string containing path to the output file
    y_range: List (float) containing lower and upper bound for the y-axes
    title: string Title to give to the plot.
    Returns
    -------
    out_path
    """
    nstates = 0
    for elem in traj_data.keys():
        if (elem[0] == 'e' and elem != 'eR')  : nstates += 1

    # get good suplot distributions
    ncols = int(np.ceil(nstates / 2)) if (int(np.ceil(nstates / 2)) < 10) else 7
    nrows = int(np.ceil(nstates / ncols))

    # build figure
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=[20, 15], sharex=True, sharey=True)
    axes = axes.flat

    colors = ps.candide_colors

    # set main title to plot
    if title is None: title = 'Potential Energy Timeseries'
    fig.suptitle(title, fontsize = 20, y =0.995)

    # plot states
    for (i, ax) in zip(range(nstates),axes):
        ax.scatter(traj_data["time"], traj_data['e' + str(i+1)], s=2, c = colors[i%len(colors)])
        ax.set_ylim(y_range)
        ax.set_title("State " + str(i+1), fontsize = 18)

        if (i % ncols == 0):
            ax.set_ylabel('V [kJ/mol]', fontsize = 16)
        if (nrows > 1 and ncols > 1 and i / ((nrows - 1) * (ncols - 1)) > 1):
            ax.set_xlabel('time [ps]', fontsize = 16)

    fig.tight_layout()

    if (out_path is not None):
        fig.savefig(out_path)
        plt.close(fig)

    return out_path


# STATE sampling plots

def plot_t_statepres(data: dict, out_path: str = None, title="test", xlim=False):
    """gives out a plot, showing the if a state is undersampling or the
    dominating (min state) of a system at given t.

    Args:
        data (dict):
        states (int):
        out_path (str):
        title:
        xlim:
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


def plot_stateOccurence_hist(data: dict, out_path: str = None,
                             title: str = "sampling histogramms", verbose=False):
    """

    Parameters
    ----------
    data
    out_path
    pottresh
    title
    verbose

    Returns
    -------

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


def plot_stateOccurence_matrix(data: dict, out_dir: str = None, s_values: list = None,
                               place_undersampling_threshold: bool = False, title_suffix: str = None):
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


## Eoff plots
def plot_peoe_eoff_vs_s(eoff: dict, energy_offsets, title: str, out_path: str):
    """gives out a plot, showing the development of Eoffsets along :param eoff:
    :param title: :param out_path: :return:

    Args:
        eoff (dict):
        energy_offsets: contains the energy offsets for each state (average over undersampling cases)
        title (str):
        out_path (str):
        color_gradient_flag:
    """

    s_vals = sorted(list(eoff.keys()))
    x = range(len(s_vals))

    eoffsets_per_s = []
    for i in sorted(eoff):
        eoffsets_per_s.append(eoff[i]["eoff"])
    min_eoff, max_eoff = np.min(eoffsets_per_s[-1]), np.max(eoffsets_per_s[-1])

    num_stats = len(eoffsets_per_s[0])
    y = [list(map(lambda x: x[i], eoffsets_per_s)) for i in range(num_stats)]

    # collect nice labels
    s_vals = list(map(lambda x: float(x), s_vals))

    number_of_labels = 5
    step_size = len(s_vals) // number_of_labels
    labels = s_vals[::step_size]  # list(map(lambda x: np.round(np.log(x),2), s_vals[::step_size]))

    colors = ps.active_qualitative_list_small.colors
    repnum = num_stats + 1

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, )

    ##plot1
    for i in range(num_stats):
        ax1.plot(x, y[i] - energy_offsets[i].mean, label=(i + 1),  # color=colors[i % repnum],
                 lw=2)

    ax1.set_ylabel("$(E_i^R(s)-\overline{E}_i^R)$/(kJ/mol)")
    ax1.set_xticks(x[::step_size])
    ax1.set_xticklabels(labels)

    ##plot2
    for i in range(num_stats):
        ax2.plot(x, y[i], label=(i + 1),  # color=colors[i % repnum],
                 lw=2)

    # plt.title(title)
    # ax2.set_ylim([-max_eoff - 10, max_eoff + 10])
    ax2.set_xlabel("s")
    ax2.set_ylabel("$E^R_i(s)$/(kJ/mol)")

    # position legend
    ax2.set_xticks(x[::step_size])
    ax2.set_xticklabels(labels)

    # legAX=fig.addsuplot(1,2, 1)
    chartBox = ax1.get_position()
    ax1.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.85, chartBox.height])

    ncol = len(labels) // 10 if (len(labels) // 10 > 0) else 1
    lgnd = ax1.legend(title="states:", loc=2, bbox_to_anchor=(1.05, 1), ncol=ncol, prop={"size": 15})

    fig.tight_layout()
    fig.suptitle(title + " $E^R_i$ per s-value", y=1.05)
    fig.savefig(out_path, bbox_extra_artists=(lgnd,), bbox_inches='tight')
    plt.close()


def plot_peoe_eoff_time_convergence(state_time_dict: dict, out_path: str):
    fig, ax = plt.subplots(ncols=1, figsize=(16, 9))
    c_steps = 20 // len(state_time_dict) if (20 // len(state_time_dict) != 0) else 1
    for (state, time_dict), c in zip(state_time_dict.items(), ps.active_qualitative_map.colors[::c_steps]):
        ax.plot(time_dict["time"], time_dict["mean"], label="state " + str(state), c=c, lw=3)
        ax.errorbar(x=time_dict["time"], y=time_dict["mean"], yerr=time_dict["std"], c=c)

    ax.set_xlabel("time [ps]")
    ax.set_ylabel("$E_i^R$ [kJ/mol]")
    ax.set_title("$E_i^R$ time convergence")

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.tight_layout()

    fig.savefig(out_path)
    plt.close()


# RE-Plots
def s_optimization_visualization(s_opt_data: dict, out_path: str=None,
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
            [int(state.replace("V", "").replace("r", "").replace("i", "")) for state in opti["state_domination_sampling"]]))

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

    labels = [str(i) for i in range(1, num_states+1)]

    # Making the offsets between the different bars

    width = 1 / (niterations * 1.15)
    x = np.arange(num_states) + 0.5*num_states*width # the label locations

    num = num_sopts-1

    # finding proper offsets

    if num%2==0:offsets = np.arange(-num/2, num/2+0.001, step =1)
    else:offsets = np.arange(-num/2, num/2+0.001)

    for i in range(num_sopts):
        normalized_heights = bar_heights[i]/np.sum(bar_heights[i])
        percent_heights = [100*j for j in normalized_heights]

        ax4.bar(x + offsets[i]*width, percent_heights, width= width,
              alpha= i/num_sopts * 0.8 + 0.2, color=["C" + str(k) for k in range(num_states)],
              label="iteration " + str(i+ 1))

    xmin = x[0] + offsets[0]/3
    xmax = x[num_states-1] + offsets[num_sopts-1]/3

    ax4.hlines(y= 100/num_states, xmin=xmin, xmax=xmax, color="red")
    ax4.set_xlim([xmin, xmax])

    ax4.set_title("State Sampling For $s=1$")
    ax4.set_ylabel("fraction [%]")
    ax4.set_xlabel("states")

    ax4.set_xticks(x)
    ax4.set_xticklabels(labels)
    ax4.legend()

    fig.tight_layout()

    if(out_path is None):
        return fig
    else:
        fig.savefig(out_path)


def visualize_s_optimisation_convergence(s_opt_data:dict, out_path:str=None)->Union[str, plt.figure]:
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
        if("avg_rountrip_duration_optimization_efficiency" in s_opt_data[it]):
            y_RTd_efficency.append(s_opt_data[it]["avg_rountrip_duration_optimization_efficiency" ])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=ps.figsize_doubleColumn, sharex=True, sharey=True)

    ax.plot(np.nan_to_num(np.log10(y_RTd_efficency)), label="efficiency", c="k")
    ax.hlines(y=0, xmin=0, xmax=8, label="$\Delta \\tau_{ij}=1 trial$", color="grey")

    ax.set_ylim([-2, 4])
    ax.set_xlim([0, 7])
    ax.set_xticks(range(8))
    ax.set_xticklabels([str(x) + "_" + str(x + 1) for x in range(1, 8)])
    ax.set_yticks(range(-1, 4))
    ax.set_yticklabels(range(-1, 4))

    ax.set_ylabel("$log(\overline{\\tau_j} - \overline{\\tau_i})$ [trials ]")
    ax.set_xlabel("iteration ij")
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0)
    if(out_path is None):
        return fig
    else:
        fig.savefig(out_path)
        plt.close()
        return out_path


def visualize_s_optimisation_sampling_optimization(s_opt_data:dict, out_path:str=None)->Union[str, plt.figure]:
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

    #ax.errorbar(list(range(1,len(maes)+1)), maes, approach_MAE_optSamp_std[approach],
    ax.plot(list(range(1, len(mae_mean) + 1)), mae_mean, alpha=0.75, c="k")

    ax.set_title("Sampling distribution deviaton from optimal sampling distribution")
    ax.set_ylabel("MAE [%]")
    ax.set_xlabel("s-opt iterations")

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, )

    if(out_path is None):
        fig.show()
    else:
        fig.savefig(out_path)
        plt.close()


def visualize_s_optimisation_convergence(s_opt_data:dict, out_path:str=None, convergens_radius:int=100)->Union[str, plt.figure]:
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
        if("avg_rountrip_duration_optimization_efficiency" in s_opt_data[it]):
            y_RTd_efficency.append(s_opt_data[it]["avg_rountrip_duration_optimization_efficiency" ])

    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=ps.figsize_doubleColumn, sharex=True, sharey=True)

    ax.plot(np.nan_to_num(np.log10(y_RTd_efficency)), label="efficiency", c="k")
    ax.hlines(y=np.log10(convergens_radius), xmin=0, xmax=8, label="$convergence criterium$", color="grey")


    ax.set_ylim([-2, 4])
    ax.set_xlim([0, 7])
    ax.set_xticks(range(len(s_opt_data)-1))
    ax.set_xticklabels([str(x) + "_" + str(x + 1) for x in range(1, len(s_opt_data))])
    ax.set_yticks(range(-1, 4))
    ax.set_yticklabels(range(-1, 4))

    ax.set_ylabel("$log(\overline{\\tau_j} - \overline{\\tau_i})$ [ps]")
    ax.set_xlabel("iteration ij")
    ax.set_title("AvgRoundtriptime optimization efficiency")
    ax.legend(fontsize=6)

    fig.tight_layout()
    fig.subplots_adjust(wspace=0.0, hspace=0)
    if(out_path is None):
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

    if(out_path is None):
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
        return fig,  hist


# FREE ENERGY CONVERGENCE
def plot_dF_conv(dF_timewise, title: str, out_path: str, dt: float = (1000),
                 verbose: bool = False, show_legend: bool = True):
    """

    :param dF_timewise: Dictionary containing the data.

    :param out_path: Path of the directory in which plot will be written

    :param dt: dt for x_axis. 1000 - converts ps to ns

    :param verbose: parmeter to print out some of the data we are working on.

    :returns: fig, ax
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 10])
    fig.tight_layout()

    last_dF = []

    for replica_ligands, data in dF_timewise.items():
        replica = replica_ligands.split("_")[:2]
        ligands = "_".join(replica_ligands.split("_")[2:])
        data = dF_timewise[replica_ligands]

        t = [float(x) / dt for x in list(data.keys())]
        dF = [x["mean"] for x in data.values()]
        err = [x["err"] for x in data.values()]

        # recenter to 0, from the average values of the last 5 elements.
        avg = np.average(dF[-5:])
        dF_recentered = [i - avg for i in dF]

        last_dF.append(dF[-1])

        if verbose: print("err:\n", err)
        if verbose: print("y:\n", dF)
        if verbose: print("ylen:\n", len(dF))

        # Now plot both plots.

        axes[0].errorbar(t, dF, err)
        axes[0].scatter(t, dF, label=ligands, marker='D')

        axes[1].errorbar(t, dF_recentered, err)
        axes[1].scatter(t, dF_recentered, label=ligands, marker='D')

    # Set the Correct y-limits for both plots, and write title/axis labels

    y_max = max(last_dF) + abs(0.2 * max(last_dF) - np.mean(last_dF))
    y_min = min(last_dF) - abs(0.2 * min(last_dF) - np.mean(last_dF))
    axes[0].set_ylim([y_min, y_max])
    axes[1].set_ylim([-15, 15])

    axes[0].set_title('Free Energy Convergence')
    axes[0].set_ylabel(r'$\Delta G_{AB}$ [kJ/mol]')
    axes[0].set_xlabel("time [ns]")

    axes[1].set_title('Free Energy Convergence (recentered)')
    axes[1].set_xlabel("time [ns]")

    if (show_legend): axes[0].legend(fontsize=8, loc='upper right', title='Pair A-B:', ncol=2, edgecolor='black')
    if (show_legend): axes[1].legend(fontsize=8, loc='upper right', title='Pair A-B:', ncol=2, edgecolor='black')

    fig.suptitle(title)
    fig.savefig(out_path + ".png")
    plt.close(fig)

    return fig, axes


def plot_thermcycle_dF_convergence(dF_time, out_path: str = None, title_prefix: str = "", verbose: bool = True):
    cols = 3
    print(dF_time)
    rows = len(dF_time) // cols if (len(dF_time) % cols == 0) else len(dF_time) // cols + 1

    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    colors = ["orange", "blue", "dark green", "purple"]
    max_y = 200
    min_y = -500
    for ind, (name, simulation) in enumerate(dF_time.items()):
        x = [x / (20 * 1000) for x in list(simulation.keys())]
        y = [x["mean"] for x in simulation.values()]
        std = [x["err"] for x in simulation.values()]

        if verbose: print("err:\n", std)
        if verbose: print("y:\n", y)
        if verbose: print("ylen:\n", len(y))

        axes[0].errorbar(x, y, std, label=name, c=colors[ind])
        axes[0].scatter(x, y, c=colors[ind])

        axes[ind + 1].set_title("dF " + name)
        axes[ind + 1].errorbar(x, y, std, label=name, c=colors[ind])
        axes[ind + 1].scatter(x, y, c=colors[ind])

        if (max(y) > max_y): max_y = max(y)
        if (min(y) < min_y): min_y = min(y)

        clean_y = [val for val in y if (val < 1000)]
        if (abs(3 * np.std(clean_y)) < 10):
            axes[ind + 1].set_ylim([np.mean(clean_y) - 10, np.mean(clean_y) + 10])
        else:
            axes[ind + 1].set_ylim([np.mean(clean_y) - 3 * np.std(clean_y), np.mean(clean_y) + 3 * np.std(clean_y)])

    if (max_y > 200):
        axes[0].set_ylim([min_y, 200])
    axes[0].set_title("dF all")

    for ax in axes:
        ax.set_ylabel("dF/kJ")
        ax.set_xlabel("t/ns")
        ax.legend()

    fig.set_size_inches([20, 10])
    fig.suptitle(title_prefix + " dF convergence")
    if (out_path != None): fig.savefig(out_path)

    return fig, axes
