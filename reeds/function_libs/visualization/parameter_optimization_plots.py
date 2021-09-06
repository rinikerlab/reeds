from typing import List, Union

import numpy as np
from matplotlib import pyplot as plt

from reeds.function_libs.visualization import plots_style as ps

def plot_offsets_vs_s(energy_offsets, mean_undersampling_eoffs: List, s_values: List, out_path: str, title: str = None, s_increment:int = 2):
    """
    plot_offsets_vs_s creates a plot of the energy offsets predicted at each of the replicas.
    The plot contains two sub-plots, one of which shows the data recentered to the average
    value among all replicas, and the other simply showing the raw data.

    Parameters
    ----------
    energy_offsets : numpy 2-D array
        matrix of energy offsets where each row corresponds to a replica
        and each column to a specific end-state.
    mean_undersampling_eoffs: List[float]
        contains the mean energy offsets for each statefrom the undersampling replicas
        i.e. the energy offsets chosen for the simulation
    s_values: List[float]
        list of s-values
    out_path : str
        output file path
    title : str
        title of the plot
    s_increment: int
        determines how many s-values are displayed on the plot. default value of 
        2 means we take every other s-value.
    
    Returns
    -------
    None
    """

    num_states = len(energy_offsets.T)
    colors = ps.active_qualitative_cycler_mligs(num_states)

    fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(7,6))
    ax1.set_prop_cycle(colors)
    ax2.set_prop_cycle(colors)
        
    if title is None: title = 'Energy Offsets predicted at all s-values'
    fig.suptitle(title)

    x = range(len(s_values))

    # plot
    for i in range(num_states):
        eoffs = energy_offsets.T[i]
        eoffs_recentered = eoffs - mean_undersampling_eoffs[i]

        ax1.plot(x, np.flip(eoffs_recentered), lw = 1, ms = 3, marker = 'D', 
                 label = 'state ' + str(i+1))
        ax2.plot(x, np.flip(eoffs), lw = 1, ms = 3, marker = 'D')
    
    # Change box format to match legend 
    box1 = ax1.get_position()
    ax1.set_position([box1.x0, box1.y0, box1.width * 0.8, box1.height])
    
    box2 = ax2.get_position()
    ax2.set_position([box2.x0, box2.y0, box2.width * 0.8, box2.height])
    
    fig.legend(loc='upper center', bbox_to_anchor=(0.85, 0.65), fancybox=True,
              shadow=True, ncol=1, fontsize = 12, edgecolor='black')

    # labels for the y-axis
    ax1.set_ylabel("$(E_i^R(s)-\overline{E}_i^R)$ [kJ/mol]")
    ax2.set_ylabel("$E^R_i(s)$ [kJ/mol]")

    ax2.set_xlabel("s-value")
 
    # labels for the x-axis
    ax1.set_xticks(x[::s_increment])
    ax1.set_xticklabels(labels=[])
    
    ax2.set_xticks(x[::s_increment])
    
    ax2.set_xticklabels(np.flip(s_values[::s_increment]), fontsize = 8)
    plt.setp(ax2.get_xticklabels(), rotation=45, ha="right",
         rotation_mode="anchor")
    
    fig.savefig(out_path, facecolor='white')
    plt.close()
    return None


def plot_peoe_eoff_time_convergence(state_time_dict: dict,
                                    out_path: str):
    """plot_peoe_eoff_time_convergence

    Parameters
    ----------
    state_time_dict : dict
    out_path : str

    Returns
    -------
    None

    """
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


def visualization_s_optimization_summary(s_opt_data: dict,
                                         out_path: str = None,
                                         nRT_range: List[float] = None,
                                         avRT_range: List[float] = None) -> Union[str, plt.figure]:
    """visualization_s_optimization_summary

    Parameters
    ----------
    s_opt_data : dict
    out_path : str, optional
    nRT_range : List[float]
    avRT_range : List[float]

    Returns
    -------
    Union[str, plt.figure]
        the outpath is returned if one is given. Alternativley the plot direclty will be returned.
    """
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
        y_svalues.extend([int(it.replace('sopt', "").replace("eoffRB", "")) for x in range(len(opti["s_values"]))])
        
        bar_heights.append([opti["state_maxContributing_sampling"][state] for state in opti["state_maxContributing_sampling"]])
        bar_x.append(np.array(
            [int(state.replace("V", "").replace("r", "").replace("i", "")) for state in opti["state_maxContributing_sampling"]]))

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
        pass
        #ax2.set_ylim([0, max(y_RTd) * 1.2])

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

def visualize_s_optimisation_convergence(s_opt_data:dict,
                                         out_path:str=None,
                                         convergens_radius:int=100) -> Union[str, plt.figure]:
    """visualize_s_optimisation_convergence
    This function visualizes the s-optimization round trip optimization time efficency convergence.
    Ideally the roundtrip time is  reduced by the s-optimization, if the average time converges towards 1ps it is assumed to be converged.

    Parameters
    ----------
    s_opt_data : dict
        contains statistics over the optimization is generated by RE_EDS_soptimizatoin_final
    out_path : str, optional
        if provided, the plot will be saved here. if not provided, the plot will be shown directly.
    convergens_radius: int, optional
        default 100

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


def visualize_s_optimisation_sampling_optimization(s_opt_data:dict,
                                                   out_path:str=None) -> Union[str, plt.figure]:
    """visualize_s_optimisation_sampling_optimization

    Parameters
    ----------
    s_opt_data : dict
    out_path : str, optional

    Returns
    -------
    Union[str, plt.figure]
        the outpath is returned if one is given. Alternativley the plot direclty will be returned.
    """
    fig, ax = plt.subplots(ncols=1, figsize=ps.figsize_doubleColumn)
    mae_mean = []
    mae_std = []
    for iteration, data in s_opt_data.items():
        mae_mean.append(data['sampling_distribution_optimal_deviation_s1']['MAE'])
        mae_std.append(data['sampling_distribution_optimal_deviation_s1']['MAE_std'])

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
        return out_path
