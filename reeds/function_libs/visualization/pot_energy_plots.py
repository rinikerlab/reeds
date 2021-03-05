"""
visualisation
-------------

visualisation.py
This File contains functions which are generating matplotlib plots for reeds sims.
"""

from typing import Iterable, List, Tuple

# plotting
import matplotlib
import numpy as np
import pandas as pd

from reeds.function_libs.visualization.utils import discard_high_energy_points

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

# Plots
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

