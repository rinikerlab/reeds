import warnings
from typing import List, Dict

import numpy as np
import pandas as pd

import reeds.function_libs.visualization.sampling_plots


def undersampling_occurence_potential_threshold_densityClustering(ene_traj_csvs: List[pd.DataFrame],
                                                                  max_distance_kJ: float = 300,
                                                                  sampling_fraction_treshold: float = 0.9)->List[float]:
    """
    This function is estimating the pot_tresh for all states by using DBSCAN identifying the density region containing sampling_fraction_treshold of the data.
    The mean and std of the density region will result in pot_tresh = mean+3std
    This function is rather performance expensive.
    ! WARNING REQUIRES SKLEARN -Not Used by the pipeline. - Prototype!

    Parameters
    ----------
    ene_traj_csvs : List[pd.Dataframe]
        a list of pandas dataframes containing the energy data of each state eX
    max_distance_kJ : float, optional
        the maximum distance between two points for dbscan clustering (eps) (default:300kJ).
    sampling_fraction_treshold : float, optional
        the sampling threshold for detecting undersampling (default: 0.9).

    Returns
    -------
    List[float]
        list of individual identified potential thresholds
    """
    try:
        from sklearn.cluster import DBSCAN
    except:
        raise Exception("Please provide package sklearn!")

    pot_tresh_pre_rep = []

    state_names = [i for i in ene_traj_csvs[0].columns if (i.startswith("e") and not i == "eR")]
    num_s_values = len(ene_traj_csvs)
    num_states = len(state_names)

    total_step_num = ene_traj_csvs[0].shape[0]
    threshold_fulfilled = [True for j in range(num_s_values)]

    v_pot_sampling = [[0 for y in state_names] for x in range(num_s_values)]

    for i, replica_data in enumerate(ene_traj_csvs):
        # print("")
        # print("replica " + str(i + 1) + ":")
        pot_tresh = []

        # calculate for each state the pot thresh, depending on clustering.
        for k, state in enumerate(state_names):
            vec = np.array(replica_data[state]).reshape(-1, 1)
            db = DBSCAN(eps=max_distance_kJ, min_samples=total_step_num * sampling_fraction_treshold).fit(vec)
            v_pot_sampling[i][k] = len(db.components_) / total_step_num

            if (not len(db.components_)):
                threshold_fulfilled[i] = False
            else:
                # print(k)
                threshold = np.mean(db.components_) + 3 * np.std(db.components_)
                pot_tresh.append(threshold)
        if (len(pot_tresh) == len(state_names)):
            pot_tresh_pre_rep.append(pot_tresh)

    # final averaging
    pot_thresh_per_state_and_replica = np.array([t for t in pot_tresh_pre_rep if (len(t) == num_states)]).T
    pot_thresh_per_state = np.mean(pot_thresh_per_state_and_replica, axis=1) + np.std(pot_thresh_per_state_and_replica,
                                                                                      axis=1)
    return pot_thresh_per_state


def undersampling_occurence_potential_threshold_distribution_based(ene_traj_csvs: List[pd.DataFrame],
                                                                   sampling_fraction_treshold: float = 0.9,
                                                                    verbose:bool = False)->List[float]:
    """undersampling_occurence_potential_threshold_distribution_based
    This function is estimating the pot_tresh for all states by testing if around the minimal energy 90% of the data is located in a threshold of  max_distance_kJ.
    The mean and std of the density region will result in pot_tresh = min(V)+6*std(V
    This function has low demand in performance.

    Parameters
    ----------
    ene_traj_csvs : List[pd.Dataframe]
        a list of pandas dataframes containing the energy data of each state eX
    sampling_fraction_treshold : float, optional
        the sampling threshold for detecting undersampling (default: 0.9).

    Returns
    -------
    List[float]
        list of individual identified potential thresholds
    """
    pot_tresh_pre_rep = []
    state_names = [i for i in ene_traj_csvs[0].columns if (i.startswith("e") and not i == "eR")]
    num_states = len(state_names)
    total_step_num = ene_traj_csvs[0].shape[0]

    std = [6*np.std(ene_traj_csvs[-1][state]) for state in state_names]
    #mean_potential = [np.mean(ene_traj_csvs[-1][state]) for state in state_names]
    for i, replica_data in enumerate(ene_traj_csvs):
        if(verbose): print("replica " + str(i + 1) + ":")
        mean_potential = [np.min(replica_data[state]) for state in state_names]
        thresholds = [m+s for m, s in zip(mean_potential, std)]
        if(verbose): print("Checkout std: ", thresholds)
        # iterate over states, identify all undersampling replicas, calc fraction
        pot_tresh = []
        for k, state in enumerate(state_names):
            vec = np.array(replica_data[state]).reshape(-1, 1)
            treshold =  thresholds[k]

            below_thresh = vec[vec < treshold]
            below_thresh_fraction = len(below_thresh) / total_step_num
            if(verbose): print("\t belowTresh: ", state, sampling_fraction_treshold < below_thresh_fraction)
            if (sampling_fraction_treshold < below_thresh_fraction):
                threshold = np.mean(below_thresh) + 3 * np.std(below_thresh)
                pot_tresh.append(threshold)

        # only save undersampling pot_threshes
        if (len(pot_tresh) == num_states):
            pot_tresh_pre_rep.append(pot_tresh)

    # final averaging.
    pot_thresh_per_state_and_replica = np.array([t for t in pot_tresh_pre_rep if (len(t) == num_states)]).T
    
    print(pot_thresh_per_state_and_replica)
    if (len(pot_thresh_per_state_and_replica) > 0):
        pot_thresh_per_state = np.min(pot_thresh_per_state_and_replica, axis=1) + np.std(pot_thresh_per_state_and_replica, axis=1)
    else:
        pot_thresh_per_state = [0 for x in range(num_states)]
    return pot_thresh_per_state

def physical_occurence_potential_threshold_distribution_based(ene_traj_csv: pd.DataFrame, equilibrate_dominationState:float=0.01, verbose:bool=False)->List[float]:
    """physical_occurence_potential_threshold_distribution_based
    This function is estimating the pot_tresh for all states by testing if around the minimal energy 90% of the data is located in a threshold of  max_distance_kJ.
    The mean and std of the density region will result in pot_tresh = mean+3*std
    This function is cheap in performance.

    Parameters
    ----------
    ene_traj_csv : pd.Dataframe
        a pandas dataframe containing the energy data of each state eX
    equilibrate_dominationState : int, optional
        equilibrate the domination state for this fraction
    verbose: bool, optional
        print fun
    Returns
    -------
    List[float]
        list of individual identified potential thresholds
    """

    state_names = [i for i in ene_traj_csv.columns if (i.startswith("e") and not i == "eR")]
    total_step_num = ene_traj_csv.shape[0]
    data =ene_traj_csv

    # iterate over states, identify all undersampling replicas, calc fraction
    pot_tresh = []
    for k, state in enumerate(state_names):
        #get only the sampling, in which the target state is dominating: pre-filter noise!
        state_domination_sampling = data.where(data[state_names].min(axis=1) == data[state]).dropna()
        start_after_eq = int(np.round(equilibrate_dominationState*state_domination_sampling.shape[0]))
        state_domination_sampling = state_domination_sampling.iloc[start_after_eq:]

        if(verbose):
            below_thresh_fraction = state_domination_sampling.shape[0] / total_step_num
            print("State "+str(state)+" - Domination state sampling fraction ", below_thresh_fraction)

        #calculate threshold
        threshold = np.min(state_domination_sampling[state]) + 6 * np.std(state_domination_sampling[state])
        pot_tresh.append(threshold)

    return pot_tresh


def get_all_physical_occurence_potential_threshold_distribution_based(ene_trajs: List[pd.DataFrame])->List[float]:
    """
        This function is used in the state optimization approach and gives the physical potential energy threshold for occurrence sampling back, that is estimated from the optimized EDS-System.

    Parameters
    ----------
    ene_trajs: List[pd.DataFrame]
        energy trajectories from all end state optimizations

    Returns
    -------
    List[float]
        list of potential energy thresholds.
    """
    opt_pot_tresh = []
    for key, traj in enumerate(ene_trajs):
        print(key)
        pot_tresh_state = physical_occurence_potential_threshold_distribution_based(traj)
        print(pot_tresh_state)
        if (np.isnan(pot_tresh_state[key])):
            warnings.warn("A state potential threshold was NaN -> This hints on that you did not sample the state as a dominating one! Please check your simulatigons!")
        opt_pot_tresh.append(pot_tresh_state[key])
    return opt_pot_tresh


def calculate_sampling_distributions(ene_traj_csvs: List[pd.DataFrame],
                                     potential_treshold: List[float],
                                     undersampling_occurence_sampling_tresh: float = 0.75)-> Dict[int, Dict[str, Dict[int, float]]]:
    """calculate_sampling_distributions
    This function is using the dominating state sampling and occurrence state sampling definition, to calculate
    for both definitions the sampling distributions including each stat.
    Further each replica, having an occurrence sampling >= undersampling_occurence_sampling_tresh for each state is categorized as undersampling.

    Parameters
    ----------
    ene_traj_csvs: List[pd.Dataframe]
        a list of pandas dataframes containing the energy data of each state eX
    potential_treshold: List[float]
        a list of potential thresholds for undersampling
    undersampling_occurence_sampling_tresh: float, optional
        threshold for the fraction of the energies which has to be below the threshold (default 0.75)

    Returns
    -------
    Dict[int, Dict[str, Dict[int, float]]]
        the dictionary contains for all replicas, the occurrence/dominating sampling fractions of each state  and undersamplin classification
    """

    replica_sampling_dist = {}
    num_states = sum([1 for i in ene_traj_csvs[0].columns if (i.startswith("e") and not i == "eR")])
    states = ["e" + str(s) for s in range(1, 1 + num_states)]
    for replica in ene_traj_csvs:
        total_number_steps = replica.shape[0]

        # Domination sampling
        dominating_state_sampling = {x: 0 for x in range(1, 1 + num_states)}
        min_state, counts = np.unique(replica[states].idxmin(axis=1), return_counts=True)
        dominating_state_sampling.update(
            {int(key.replace("e", "")): val / total_number_steps for key, val in zip(min_state, counts)})

        # occurence samplng
        occurrence_state_sampling = {}
        for ind, state in enumerate(states):
            occurrence_sampling_frac = replica[replica[state] < potential_treshold[ind]].shape[0] / total_number_steps
            occurrence_state_sampling.update({int(state.replace("e", "")): occurrence_sampling_frac})

        # undersampling?
        undersampling_criterium = True if (
            all([occ >= undersampling_occurence_sampling_tresh for occ in
                 occurrence_state_sampling.values()])) else False

        # update results
        replica_sampling_dist.update({int(replica.s.replace("s", "")): {"dominating_state": dominating_state_sampling,
                                                                        "occurence_state": occurrence_state_sampling,
                                                                        "undersampling": undersampling_criterium}})

    return replica_sampling_dist


def sampling_analysis(ene_traj_csvs: List[pd.DataFrame], s_values: List[float], out_path: str = None,
                      xmax: bool = False, do_plot: bool = True, verbose: bool = False) -> (dict, str):
    """sampling_analysis
    This function is analysing the samplings

    Parameters
    ----------
    out_path :  str
        path out for the plotfiles
    ene_traj_csvs : List[pd.DataFrame]
        contains the energy data
    s_values :  List[float]
        list of s_values
    xmax :  bool, optional
        additionally output a plot only with a smaller x_range. (0->xmax) (default False)
    do_plot: bool, opptional
        create additional plots (default True)
    verbose: bool, optional
        story time :) (default False)

    Returns
    -------
    str
        out_path
    dict
      returns a dictionary containing the information of undersampling start, undersampling_potential thresholds and the different sampling type fractions. (physical sampling occurence not included!)
    """

    # read all Vy_sx_files!
    from reeds.function_libs.visualization.utils import nice_s_vals
    if (verbose): print("\n\n Potential Threshold\n\n")
    potential_treshold = undersampling_occurence_potential_threshold_distribution_based(ene_traj_csvs=ene_traj_csvs)

    ##glob vars
    num_states = len(potential_treshold)
    select_states = ["e" + str(x) for x in range(1, num_states + 1)]
    s_vals_nice = nice_s_vals(s_values)

    # show_presence of undersampling
    if (verbose): print("\n\n Sampling Timeseries\n\n")
    for ind, replica in enumerate(ene_traj_csvs):
        if (verbose): print("\t replica " + replica.s)
        dominating_state_sampling = replica[select_states].idxmin(axis=1).replace("e", "", regex=True)

        occurrence_sampling_replica = []
        for state in select_states:
            occurrence_sampling_state_replica = replica.index[
                replica[state] < potential_treshold[int(state.replace("e", "")) - 1]]
            occurrence_sampling_replica.append(occurrence_sampling_state_replica)

        data = {"occurrence_t": occurrence_sampling_replica, "dominating_state": dominating_state_sampling}

        if (do_plot):
            reeds.function_libs.visualization.sampling_plots.plot_t_statepres(data=data,
                                                                              title="s=" + str(s_vals_nice[ind]),
                                                                              out_path=out_path + "/sampling_timeseries_s" + str(ind + 1) + ".png")

        if (xmax):
            reeds.function_libs.visualization.sampling_plots.plot_t_statepres(data=data,
                                                                              title="s=" + str(s_vals_nice[ind]),
                                                                              out_path=out_path + "/sampling_timeseries_s" + str(ind + 1) + "_upto_" + str(
                                     xmax) + ".png",
                                                                              xlim=[0, xmax])

    # SamplingMatrix by kays
    if (verbose): print("\n\n Calculate Sampling Distributions\n\n")
    replica_sampling_distributions = calculate_sampling_distributions(ene_traj_csvs=ene_traj_csvs,
                                                                      potential_treshold=potential_treshold,
                                                                      undersampling_occurence_sampling_tresh=0.75)
    if (do_plot):
        if (verbose): print("\n\n Sampling Histograms\n\n")
        for ind, x in enumerate(replica_sampling_distributions):
            reeds.function_libs.visualization.sampling_plots.plot_stateOccurence_hist(data=replica_sampling_distributions[x],
                                                                                      title="s=" + str(s_vals_nice[ind]),
                                                                                      out_path=out_path + "/sampling_hist_" + str(x) + ".png")

    if (do_plot):
        if (verbose): print("\n\n Sampling Matrix\n\n")
        reeds.function_libs.visualization.sampling_plots.plot_stateOccurence_matrix(data=replica_sampling_distributions, out_dir=out_path, s_values=s_vals_nice,
                                                                                    place_undersampling_threshold=True, title_suffix="")

    ##get undersampling id:
    undersampling_idx = None
    for i in replica_sampling_distributions:
        if (replica_sampling_distributions[i]["undersampling"]):
            undersampling_idx = i
            break

    if (undersampling_idx is None):
        warnings.warn("Could not find undersampling!")

    final_results = {"undersamplingThreshold": undersampling_idx,
                     "potentialThreshold": potential_treshold,
                     "samplingDistributions": replica_sampling_distributions, 
                    }

    return final_results, out_path
