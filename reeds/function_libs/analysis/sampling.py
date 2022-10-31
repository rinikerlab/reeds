import warnings
from typing import List, Dict
from numbers import Number

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

    #remove first 10% as equilibration
    total_step_num = ene_traj_csvs[0].shape[0] - 0.1*ene_traj_csvs[0].shape[0]
    std = [6*np.std(ene_traj_csvs[-1][state][int(len(ene_traj_csvs[-1][state])/10):len(ene_traj_csvs[-1][state])]) for state in state_names]

    #mean_potential = [np.mean(ene_traj_csvs[-1][state]) for state in state_names]
    for i, replica_data in enumerate(ene_traj_csvs):
        if(verbose): print("replica " + str(i + 1) + ":")
        mean_potential = [np.min(replica_data[state][int(len(replica_data[state])/10):len(replica_data[state])]) for state in state_names]
        thresholds = [m+s for m, s in zip(mean_potential, std)]
        if(verbose): print("Checkout std: ", thresholds)
        # iterate over states, identify all undersampling replicas, calc fraction
        pot_tresh = []
        for k, state in enumerate(state_names):
            vec = np.array(replica_data[state]).reshape(-1, 1)
            treshold =  thresholds[k]
            #remove first 10% as equilibration
            vec = vec[int(len(vec)/10) : len(vec)]

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
    
    #print(pot_thresh_per_state_and_replica)
    if (len(pot_thresh_per_state_and_replica) > 0):
        pot_thresh_per_state = np.min(pot_thresh_per_state_and_replica, axis=1) + np.std(pot_thresh_per_state_and_replica, axis=1)
    else:
        pot_thresh_per_state = [0 for x in range(num_states)]
    return pot_thresh_per_state

def physical_occurence_potential_threshold_distribution_based(ene_traj_csv: pd.DataFrame, equilibrate_dominationState:float=0.01, _vacuum_simulation:bool=False,  verbose:bool=False)->List[float]:

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
    _vacuum_simulation=vacuum_simulation: bool, optional
        this flag switches the algorithm to use all potential energies of a state, instead of only the domination sampling.
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
        if(_vacuum_simulation):
            state_domination_sampling = data
        else:
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


def get_all_physical_occurence_potential_threshold_distribution_based(ene_trajs: List[pd.DataFrame], _vacuum_simulation:bool=False,)->List[float]:
    """
        This function is used in the state optimization approach and gives the physical potential energy threshold for occurrence sampling back, that is estimated from the optimized EDS-System.

    Parameters
    ----------
    ene_trajs: List[pd.DataFrame]
        energy trajectories from all end state optimizations
    _vacuum_simulation=vacuum_simulation: bool, optional
        this flag switches the algorithm to use all potential energies of a state, instead of only the domination sampling.
    Returns
    -------
    List[float]
        list of potential energy thresholds.
    """
    opt_pot_tresh = []
    for key, traj in enumerate(ene_trajs):
        #print(key)
        pot_tresh_state = physical_occurence_potential_threshold_distribution_based(traj, _vacuum_simulation=_vacuum_simulation)

        #print(pot_tresh_state)
        if (np.isnan(pot_tresh_state[key])):
            warnings.warn("A state potential threshold was NaN -> This hints on that you did not sample the state as a dominating one! Please check your simulatigons!")
        opt_pot_tresh.append(pot_tresh_state[key])
    return opt_pot_tresh

def calculate_sampling_distributions(ene_traj_csvs: List[pd.DataFrame], eoffs: List[List[float]],
                                     potential_treshold: List[float]) -> Dict[int, Dict[str, Dict[int, float]]]:
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
    eoffs : List[List[float]]
        energy offsets for each replica
    undersampling_occurence_sampling_tresh: float, optional
        threshold for the fraction of the energies which has to be below the threshold (default 0.75)

    Returns
    -------
    Dict[int, Dict[str, Dict[int, float]]]
        the dictionary contains for all replicas, the occurrence/minState/maxContribution sampling fractions of each state  and undersamplin classification
    """

    replica_sampling_dist = {}
    num_states = sum([1 for i in ene_traj_csvs[0].columns if (i.startswith("e") and not i == "eR")])
    states = ["e" + str(s) for s in range(1, 1 + num_states)]
    for ind, replica in enumerate(ene_traj_csvs):
        total_number_steps = replica.shape[0]

        # Domination sampling
        minV_state_sampling = {x: 0 for x in range(1, 1 + num_states)}
        min_state, counts = np.unique(replica[states].idxmin(axis=1), return_counts=True)
        minV_state_sampling.update(
            {int(key.replace("e", "")): val / total_number_steps for key, val in zip(min_state, counts)})

        # Corr
        max_contributing_state_sampling = {x: 0 for x in range(1, 1 + num_states)}
        contrib_corr = replica[states] - eoffs[ind]
        contrib_corr = contrib_corr.idxmin(axis=1).replace("e", "", regex=True)
        min_state, counts = np.unique(contrib_corr, return_counts=True)
        max_contributing_state_sampling.update(
            {int(key): val / total_number_steps for key, val in zip(min_state, counts)})

        # occurence samplng
        occurrence_state_sampling = {}
        for indState, state in enumerate(states):
            occurrence_sampling_frac = replica[replica[state] < potential_treshold[indState]].shape[0] / total_number_steps
            occurrence_state_sampling.update({int(state.replace("e", "")): occurrence_sampling_frac})

        # update results
        replica_sampling_dist.update({int(replica.s.replace("s", "")): {"minV_state": minV_state_sampling,
                                                                        "max_contributing_state": max_contributing_state_sampling,
                                                                        "occurence_state": occurrence_state_sampling}})

    return replica_sampling_dist

def sampling_analysis(ene_traj_csvs: List[pd.DataFrame],
                      state_potential_treshold: List[float], eoffs: List[List[float]],
                      s_values: List[float],
                      out_path: str = None,
                      xmax: bool = False,
                      _visualize: bool = True,
                      verbose: bool = False, _usample_run:bool=False) -> (dict, str):
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
    state_potential_treshold : List[float]
        potential energy thresholds, for considering a state as sampled
    eoffs : List[List[float]]
        energy offsets for each replica
    xmax :  bool, optional
        additionally output a plot only with a smaller x_range. (0->xmax) (default False)
    _visualize: bool, opptional
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

    ##glob vars
    num_states = len(state_potential_treshold)
    select_states = ["e" + str(x) for x in range(1, num_states + 1)]
    s_vals_nice = nice_s_vals(s_values)

    # show_presence of undersampling
    if (verbose): print("\n\n Sampling Timeseries\n\n")
    if(isinstance(eoffs[0], Number)): #only 1D eoff vector given!
        eoffs = [eoffs for x in range(len(ene_traj_csvs))]

    for ind, replica in enumerate(ene_traj_csvs):
        if (verbose): print("\t replica " + replica.s)

        minV_state_sampling = replica[select_states].idxmin(axis=1).replace("e", "", regex=True)

        # Corr
        eoffs_rep = np.array(eoffs[ind])
        max_contributing_sampling = (replica[select_states] - eoffs_rep).idxmin(axis=1).replace("e", "", regex=True)

        occurrence_sampling_replica = []
        for state in select_states:
            occurrence_sampling_state_replica = replica.index[
                replica[state] < state_potential_treshold[int(state.replace("e", "")) - 1]]
            occurrence_sampling_replica.append(occurrence_sampling_state_replica)

        data = {"time": replica.time, "occurrence_t": occurrence_sampling_replica, "minV_state": minV_state_sampling, "maxContrib_state":max_contributing_sampling}
        
        if (_visualize):
            reeds.function_libs.visualization.sampling_plots.plot_t_statepres(data=data,
                                                                              title="s=" + str(s_vals_nice[ind]),
                                                                              out_path=out_path + "/sampling_timeseries_s" + str(ind + 1) + ".png")

        if (xmax):
            reeds.function_libs.visualization.sampling_plots.plot_t_statepres(data=data,
                                                                              title="s=" + str(s_vals_nice[ind]),
                                                                              out_path=out_path + "/sampling_timeseries_s" + str(ind + 1) + "_upto_" + str(xmax) + ".png",
                                                                              xlim=[0, xmax])

    # SamplingMatrix by kays
    if (verbose): print("\n\n Calculate Sampling Distributions\n\n")
    replica_sampling_distributions = calculate_sampling_distributions(ene_traj_csvs=ene_traj_csvs, eoffs=eoffs,
                                                                      potential_treshold=state_potential_treshold)

    if (_visualize):
        if (verbose): print("\n\n Sampling Histograms\n\n")
        for ind, x in enumerate(replica_sampling_distributions):
            reeds.function_libs.visualization.sampling_plots.plot_stateOccurence_hist(data=replica_sampling_distributions[x],
                                                                                      title="s=" + str(s_vals_nice[ind]),
                                                                                      out_path=out_path + "/sampling_hist_" + str(x) + ".png")
    if (_visualize and not _usample_run):
        if (verbose): print("\n\n Sampling Matrix\n\n")
        reeds.function_libs.visualization.sampling_plots.plot_stateOccurence_matrix(data=replica_sampling_distributions, out_dir=out_path, s_values=s_vals_nice,
                                                                                    place_undersampling_threshold=False, title_suffix="")

    final_results = {"potentialThreshold": state_potential_treshold,
                     "samplingDistributions": replica_sampling_distributions,
                    }

    return final_results, out_path



def detect_undersampling(ene_traj_csvs: List[pd.DataFrame],
                         state_potential_treshold: List[float],
                         s_values: List[float], eoffs: List[List[float]],
                         out_path: str = None,
                         undersampling_occurence_sampling_tresh: float = 0.9,
                         xmax: bool = False,
                         _visualize: bool = True,
                         verbose: bool = False) -> (dict, str):
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
    state_potential_treshold : List[float]
        potential energy thresholds, for considering a state as sampled
    xmax :  bool, optional
        additionally output a plot only with a smaller x_range. (0->xmax) (default False)
    _visualize: bool, opptional
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

    ##glob vars
    sampling_stat, out_path = sampling_analysis(ene_traj_csvs=ene_traj_csvs,
                                                state_potential_treshold=state_potential_treshold,
                                                s_values=s_values, eoffs=eoffs,
                                                out_path=out_path,
                                                xmax= xmax,
                                                _visualize=_visualize,
                                                verbose = verbose, _usample_run=True)

    ##get undersampling id:
    found_undersampling = False
    undersampling_idx = None
    replica_sampling_distributions = sampling_stat["samplingDistributions"]
    for i in replica_sampling_distributions:
        # undersampling?
        undersampling_criterium = True if (all([occ >= undersampling_occurence_sampling_tresh for occ in replica_sampling_distributions[i]["occurence_state"].values()])) else False
        sampling_stat["samplingDistributions"][i].update({"undersampling":undersampling_criterium})

        if (undersampling_criterium):
            undersampling_idx = i
            found_undersampling = True
            break

    if (not found_undersampling):
        warnings.warn("Could not find undersampling!")

    if (_visualize):
        if (verbose): print("\n\n Sampling Matrix\n\n")
        s_vals_nice = nice_s_vals(s_values)
        if(_visualize):
            reeds.function_libs.visualization.sampling_plots.plot_stateOccurence_matrix(data=replica_sampling_distributions, out_dir=out_path, s_values=s_vals_nice,
                                                                                    place_undersampling_threshold=True, title_suffix="")

    sampling_stat.update({"undersamplingThreshold": undersampling_idx})
    sampling_stat.update({"state_undersampling_potTresh": state_potential_treshold,
                          "undersampling_occurence_sampling_tresh": undersampling_occurence_sampling_tresh})


    return sampling_stat, out_path
