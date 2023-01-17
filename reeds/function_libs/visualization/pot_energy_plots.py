from typing import List, Iterable, Tuple, Union

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from reeds.function_libs.visualization import plots_style as ps
from reeds.function_libs.visualization.utils import discard_high_energy_points
from reeds.function_libs.visualization.utils import determine_vrange

def plot_optimized_states_potential_energies(outfile:str,
                                             ene_trajs:pd.DataFrame,):
    """plot_optimized_states_potential_energies
    Plots the potential energy distributions of all states in a single plot,
    where the data is represented as a histogram.

    Parameters
    ----------
    outfile: str
        string containing the path to the output png file
    ene_trajs: pd.DataFrame
        pandas dataframe containing potential energy trajectories

    Returns
    -------
    None
    """

    nstates = len(ene_trajs)
    v_range = determine_vrange(ene_trajs, nstates)
    colors = ps.active_qualitative_map_mligs(nstates)

    # Split the different states in subplots:
    ncols = 4 if nstates > 11 else 3
    nrows = int(np.ceil(nstates/ncols))

    figsize = [5*ncols, 5*nrows]
    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    fig.suptitle('Potential Energy Distribution for optimized States', fontsize = 20)
    
    axes = axes.flatten()

    for i in range(nstates):
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
        hist1 = axes[i].hist(up_energies, bins = 100, label = legend, color=colors[i%len(colors)])

        # formatting:
        axes[i].legend(loc='upper right', fontsize=12, edgecolor='black')

        axes[i].set_xlim(v_range)
        axes[i].set_ylim([0, 1.35 * max(hist1[0])])

        axes[i].set_ylabel('Count')
        axes[i].set_xlabel(r'$V_{i}$ [kJ/mol]')

    # Done plotting everything, save the figure

    plt.tight_layout()
    plt.savefig(outfile, facecolor='white')
    plt.close()

    return None


def plot_energy_distribution_by_replica(traj_data : pd.DataFrame,
                                        replica_num : int,
                                        s_value : float,
                                        outfile_path: str=None,
                                        manual_xlim : List[float] = None,
                                        shared_xaxis : bool = True,
                                        sampling_thresholds = None,
                                        undersampling_thresholds = None)->Union[str, None]:
    """plot_energy_distribution_by_replica
    Plots the potential energy distributions for the data generated
    in a RE-EDS simulation (or for N independant EDS simulations (i.e. lower bound))

    It generates the plot for a specific replica in which the
    data for each end states is represented in a different subplot.

    Parameters
    ----------
    traj_data : pd.DataFrame
        pandas DataFrame containing the data for the specific replica of interest
    replica_num : int
        replica number (used an index)
    s_value : float
        s value associated to this replica
    outfile_path : str, optional
        name of the output png file
    manual_xlim: List[float], optional
        minimum and maximum value for the x-axes
    shared_xaxis: bool, optional
        determines if each plot has the same axes limits
    sampling_thresholds: List [float], optional
        list (length = number of end states) of the potential thresholds which 
        determines when a state is physically sampled. If given this is added to the plot.
    undersampling_thresholds: List [float], optional
        list (length = number of end states) of the undersampling potential thresholds which 
        determines when a state is is undersampling. If given this is added to the plot.

    Returns
    -------
    str, None
        outpath
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
    colors = ps.active_qualitative_map_mligs(nstates)

    ncols = 4 if nstates > 11 else 3
    nrows = int(np.ceil(nstates/ncols))

    figsize = [5*ncols, 5*nrows]

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)

    fig.suptitle('Average Potential Energy - Replica ' + str(replica_num)
                 + ' - s = ' + str(s_value), fontsize = 20, y=0.995)
    
    axes = axes.flatten()

    # Plot each histogram one by one:
    for i in range(nstates):
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
        hist1 = axes[i].hist(up_energies, bins = 250, label = legend, color=colors[i%len(colors)])
        axes[i].legend(loc='upper right', fontsize=12, edgecolor='black')

        # Use the correct set of x-limits
        if manual_xlim is None: xlimits = [-1250, upper_threshold]
        else: xlimits = manual_xlim

        if not shared_xaxis:
            high_lim = min(upper_threshold, max(up_energies))
            low_lim  = min(up_energies)
            low_lim  = low_lim - 0.1 * abs(low_lim)
            high_lim = high_lim + 0.1* abs(high_lim)
            xlimits  = [low_lim, high_lim]
        
        # Draw a vertical line to delimit physical and undersampling if requested. 
        if sampling_thresholds is not None:
            axes[i].axvline(x=sampling_thresholds[i], color = 'black', lw = 1, ls = '--')
        if undersampling_thresholds is not None:
            axes[i].axvline(x=undersampling_thresholds[i], color = 'grey', lw = 1, ls = '--')
        
        axes[i].set_xlim(xlimits)
        axes[i].set_ylabel('Count')
        axes[i].set_xlabel(r'$V_{i}$ [kJ/mol]')

        # Set the ylim so the legend can be read well / no overlap with the plot
        axes[i].set_ylim([0, 1.35 * max(hist1[0])])

    # Done plotting everything, save the figure

    fig.tight_layout()

    if(outfile_path is None):
        fig.show()
    else:
        fig.savefig(outfile_path, ) #facecolor='white')
        plt.close()

    return outfile_path


def plot_energy_distribution_by_state(energy_trajs : List[pd.DataFrame],
                                      outfile : str,
                                      state_num : int,
                                      s_values : List[float],
                                      manual_xlim : List[float] = None,
                                      shared_xaxis : bool = True):
    """plot_energy_distribution_by_state
    Plots the potential energy distributions for the data generated
    in a RE-EDS simulation (or for N independant EDS simulations (i.e. lower bound))

    The overall plot generated corresponds to a specific end state,
    each replica is represented in a different subplot.

    Parameters
    ----------
    energy_trajs : List[pd.DataFrame]
        List of pandas DataFrame containing the data for all states/replicas
    outfile: str
        name of the output png file
    state_num: int
        number of the state to plot
    s_values : List[float]
        list of float s-values of all the replicas
    manual_xlim: List[float], optional
        minimum and maximum value for the x-axes (default None)
    shared_xaxis: bool
        determines if each plot has the same axes limits (default True)
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

    colors = ps.active_qualitative_map_mligs(nstates)
    color = colors[state_num-1%len(colors)]

    n_replicas = len(energy_trajs)

    ncols = 4 if n_replicas > 11 else 3
    nrows = int(np.ceil(n_replicas/ncols))

    figsize = [5*ncols, 5*nrows]

    fig, axes = plt.subplots(ncols=ncols, nrows=nrows, figsize=figsize)
    fig.suptitle('Average Potential Energy - State ' + str(state_num) + '\n', fontsize = 20, y = 0.995)
    
    axes = axes.flatten()

    # Plot each histogram one by one:
    for i in range(n_replicas):
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

        hist1 = axes[i].hist(up_energies, bins = 100, label = legend, color = color)
        axes[i].legend(loc='upper left', fontsize=12, edgecolor='black')

        # Use the correct set of x-limits
        if manual_xlim is None: xlimits = [-1250, upper_threshold]
        else: xlimits = manual_xlim

        if not shared_xaxis:
            high_lim = min(upper_threshold, max(up_energies))
            low_lim  = min(up_energies)
            low_lim  = low_lim - 0.1 * abs(low_lim)
            high_lim = high_lim + 0.1* abs(high_lim)
            xlimits  = [low_lim, high_lim]

        axes[i].set_xlim(xlimits)

        axes[i].set_ylabel('Count')
        axes[i].set_xlabel(r'$V_{i}$ [kJ/mol]')
        axes[i].set_title('Replica ' + str(replica_num) + ' - s = ' + str(s_values[i]))

        # Set the ylim so the legend can be read well / no overlap with the plot
        axes[i].set_ylim([0, 1.35 * max(hist1[0])])

    # Done plotting everything, save the figure

    plt.tight_layout()
    plt.savefig(outfile, facecolor='white')
    plt.close()

    return None


def plot_ref_pot_energy_distribution(energy_trajs: List[pd.DataFrame],
                                     outfile: str,
                                     s_values: List[float],
                                     optimized_state: bool = False):
    """plot_ref_pot_energy_distribution
    Plots the reference potential energy distribution
    for either the optimized states, or a set of EDS/RE-EDS
    simulation, for which data has been previously parsed by
    parse_csv_energy_trajectories()

    Parameters
    ----------
    energy_trajs : List[pd.DataFrame]
        List of pandas DataFrame containing traj info
    outfile : str
        path to the output png file
    s_values : List[float]
        List of s-values corresponding to the different replicas
        an empty list can be given when optimized_state = True
    optimized_state: bool, optional
        True when we want to make the plot for the optimized states (default False)

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
    axes = axes.flatten()

    for i in range(n_replicas):
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

        hist1 = axes[i].hist(up_energies, bins = 100, label = legend, color = 'firebrick')
        axes[i].legend(loc='upper right', fontsize=12, edgecolor='black')
        #
        xmax = upper_threshold
        if max(up_energies)+100 < upper_threshold: xmax = max(up_energies)+100

        axes[i].set_xlim([min(up_energies) - 100, xmax])
        axes[i].set_ylim([0, 1.35 * max(hist1[0])])
        axes[i].set_ylabel('Count')
        axes[i].set_xlabel(r'$V_{R}$ [kJ/mol]')

        if optimized_state :
            axes[i].set_title('System biased to state ' + str(replica_num))
        else :
            axes[i].set_title('Replica ' + str(replica_num) + ' - s = ' + str(s_values[i]))

    # Done plotting everything, save the figure

    plt.tight_layout()
    plt.savefig(outfile, facecolor='white')
    plt.close()

    return None


def plot_ref_pot_ene_timeseries(ene_trajs: List[pd.DataFrame],
                                outfile: str,
                                s_values: List[float],
                                optimized_state: bool = False):
    """plot_ref_pot_ene_timeseries
    Plots the reference potential energy timeseries
    for either the optimized states, or a set of EDS/RE-EDS
    simulation, for which data has been previously parsed by
    parse_csv_energy_trajectories()

    Parameters
    ----------
    energy_trajs: List[pd.DataFrame]
        List of pandas DataFrame containing traj info
    outfile: str
        path to the output png file
    s_values: List[float]
        List of float, s-values corresponding to the different replicas
        an empty list can be given when optimized_state = True
    optimized_state: bool
        True when we want to make the plot for the optimized states (default False)

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
    
    axes = axes.flatten()

    # Plot each histogram one by one:
    for i in range(n_replicas):
        replica_num = i+1

        # Plot the data
        e_ref = np.array(ene_trajs[i]['eR'])
        t = np.array(ene_trajs[i]['time'])

        axes[i].scatter(t, e_ref, color='firebrick', s = 4, marker = 'D')
        axes[i].set_xlabel('time [ps]')
        axes[i].set_ylabel(r'$V_{R}$ [kJ/mol]')

        axes[i].set_ylim(min(e_ref)-100,max(e_ref)+100 )

        if optimized_state :
            axes[i].set_title('System biased to state ' + str(replica_num))
        else :
            axes[i].set_title('Replica ' + str(replica_num) + ' - s = ' + str(s_values[i]))

    # Done plotting everything, save the figure

    plt.tight_layout()
    plt.savefig(outfile, facecolor='white')
    plt.close()

    return None


def plot_potential_distribution(potentials: (pd.Series or pd.DataFrame), out_path: str,
                                y_label="V/[kj/mol]", y_range: Iterable[float] = [-1000, 1000],
                                title: str = "Energy Distribution in range", label_medians: bool = True) -> str:
    """
    @ WARNING MIGHT BE REMOVED - OLD Function@
    """
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


def plot_replicaEnsemble_property_2D(ene_trajs: List[pd.DataFrame],
                                     out_path: str,
                                     temperature_property: str = "solvtemp2") -> str:
    """plot_replicaEnsemble_property_2D
    author: Kay Schaller

    Plots the temperature time series for all replicas as heat map.

    Parameters
    ----------
    ene_trajs : List[pd.DataFrame]
        ene_ana_property object, that contains time and solv2temp property
    out_path : str
        output file path
    temperature_property : str, optional
       is the name of the property, that is representing the Temperature (default "solvtemp2")

    Returns
    -------
    str
        output file name

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


def plot_potential_timeseries(time: pd.Series, potentials: (pd.DataFrame or pd.Series),
                              title: str,
                              y_range: Tuple[float, float] = None, x_range: Tuple[float, float] = None,
                              x_label: str = "t", y_label: str = "V/kJ",
                              alpha: float = 0.6,
                              out_path: str = None):

    """
    @ WARNING MIGHT BE REMOVED - OLD Function@
    """

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


def scatter_potential_timeseries(time: pd.Series,
                                 potentials: (pd.DataFrame or pd.Series),
                                 title: str,
                                 y_range: Tuple[float, float] = None,
                                 x_range: Tuple[float, float] = None,
                                 x_label: str = "t",
                                 y_label: str = "V [kJ]",
                                 marker_size=5,
                                 alpha: float = 0.5,
                                 show_legend: bool = True,
                                 out_path: str = None):

    """scatter_potential_timeseries
    This function creates scatter plots of the potential energy timeseries

    Parameters
    ----------
    time : pd.Series
        time series for the x-axis
    potentials : pd.DataFrame or pd.Series
        potential energies
    title : str
        plot title
    y_range : Tuple[float, float], optional
        y-axis ranges (default None)
    x_range : Tuple[float, float], optional
        x-axis ranges (default None)
    x_label : str, optional
        label for x-axis (default "t")
    y_label : str, optional
        label for y-axis (default "V [kJ]")
    marker_size: int, optional
        size of markers (default 5)
    alpha: float
        default 0.5
    show_legend : bool, optional
        show legend (default True)
    out_path : str, optional
        path for output (default None)

    Returns
    -------
    None
    """

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


def plot_sampling_grid(traj_data: pd.DataFrame,
                       out_path: str= None,
                       y_range: List[float] = [-1000, 1000],
                       title: str = None, 
                       sampling_thresholds = None,
                       undersampling_thresholds = None) -> str:
    """plot_sampling_grid
    Plots the potential energy of each end state in a single plot (grid of subplots)
    where each subplot corresponds to one end state.

    Parameters
    ----------
    traj_data: pd.DataFrame
        pandas dataFrame containing timestep/potential energy info of the trajectory.
    out_path: str
        string containing path to the output file
    y_range: List[float], optional
        List (float) containing lower and upper bound for the y-axes (default [-1000, 1000])
    title: str, optional
        string Title to give to the plot (default None)
    sampling_thresholds: List [float], optional
        list (length = number of end states) of the potential thresholds which 
        determines when a state is physically sampled. If given this is added to the plot.
    undersampling_thresholds: List [float], optional
        list (length = number of end states) of the undersampling potential thresholds which 
        determines when a state is is undersampling. If given this is added to the plot.
    Returns
    -------
    out_path: str
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

    colors = ps.active_qualitative_map_mligs(nstates)

    # set main title to plot
    if title is None: title = 'Potential Energy Timeseries'
    fig.suptitle(title, fontsize = 20, y =0.995)

    # plot states
    for (i, ax) in zip(range(nstates),axes):
        ax.scatter(traj_data["time"], traj_data['e' + str(i+1)], s=2, color = colors[i%len(colors)])
        ax.set_ylim(y_range)
        ax.set_title("State " + str(i+1), fontsize = 18)

        if (i % ncols == 0):
            ax.set_ylabel('V [kJ/mol]', fontsize = 16)
        if (nrows > 1 and ncols > 1 and i / ((nrows - 1) * (ncols - 1)) > 1):
            ax.set_xlabel('time [ps]', fontsize = 16)
        
        # Add horizontal lines to show the thresholds
        if sampling_thresholds is not None:
          ax.axhline(y=sampling_thresholds[i], color = 'black', lw = 1, ls = '--')
        if undersampling_thresholds is not None:
          ax.axhline(y=undersampling_thresholds[i], color = 'grey', lw = 1, ls = '--')
        
    fig.tight_layout()

    if (out_path is not None):
        fig.savefig(out_path)
        plt.close(fig)
    else:
        fig.show()

    return out_path
