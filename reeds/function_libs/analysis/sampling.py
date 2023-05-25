import warnings
from typing import List, Dict
from numbers import Number

import numpy as np
import pandas as pd

import reeds.function_libs.visualization.sampling_plots


def undersampling_occurence_potential_threshold_densityClustering(ene_trajs: List[pd.DataFrame],
                                                                  max_distance_kJ: float = 300,
                                                                  sampling_fraction_treshold: float = 0.9)->List[float]:
    """
    This function is estimating the pot_tresh for all states by using DBSCAN identifying the density region containing sampling_fraction_treshold of the data.
    The mean and std of the density region will result in pot_tresh = mean+3std
    This function is rather performance expensive.
    ! WARNING REQUIRES SKLEARN -Not Used by the pipeline. - Prototype!

    Parameters
    ----------
    ene_trajs : List[pd.Dataframe]
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

    state_names = [i for i in ene_trajs[0].columns if (i.startswith("e") and not i == "eR")]
    num_s_values = len(ene_trajs)
    num_states = len(state_names)

    total_step_num = ene_trajs[0].shape[0]
    threshold_fulfilled = [True for j in range(num_s_values)]

    v_pot_sampling = [[0 for y in state_names] for x in range(num_s_values)]

    for i, replica_data in enumerate(ene_trajs):
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


def findOccurenceSamplingThreshold(traj, eoffs, trim_equil=0.1, filter_maxContrib=False):

    """findOccurenceSamplingThreshold
    This function finds the potential energy threshold which can be used to determine if a state is sampled according 
    to the occurence sampling definition based on a set of simulation data. A threshold is determined for each state 
    and the value of 0 is returned if that state was never sampled (according the maximally contributing criteria) in the simulation.
    
    Parameters
    ----------
    ene_trajs : pd.Dataframe
        a pandas dataframe containing the energy data of each state eX
    eoffs : List[float]
        energy offsets for each state in this simulation
    filter_maxContrib: 
        pre-filter the data to remove non-sampled points (important for end-state)
    
    trim_equil: float
        fraction of the simulation to remove for "equilibration"

    Returns
    -------
    List[float]
        list of individual identified potential thresholds
    """

    state_names = [i for i in traj.columns if (i.startswith("e") and not i == "eR")]
    num_steps = traj.shape[0]
    
    v_min_eoff = np.array(traj[state_names]) - eoffs
    v_min_eoff = v_min_eoff[int(trim_equil*num_steps):]
    
    pot_tresh = np.zeros(len(state_names))
    for k, state in enumerate(state_names):
        if filter_maxContrib: # Extract data points where column k is the minimum
            idx = np.where(np.argmin(v_min_eoff, axis=1) ==k)
            pot_ene = (v_min_eoff[idx, k] + eoffs[k]).flatten()
        else:
            pot_ene = (v_min_eoff[:, k] + eoffs[k]).flatten()
        
        if len(pot_ene) == 0:
            pot_tresh[k] = 0
        else:
            pot_tresh[k] = np.min(pot_ene) + 6 * np.std(pot_ene)
    return pot_tresh

def findPhysicalSamplingThresholds(ene_trajs: List[pd.DataFrame], eoffs: List[List[float]])->List[float]:
    """
        This function is used in the end state generation stage of the pipeline and finds 
        the potential energy threshold under which each state state can be considered to be sampled "physically"
        according to the occurrence sampling definition.

    Parameters
    ----------
    ene_trajs: List[pd.DataFrame]
        energy trajectories from all end state generation.
    eoffs: List [List[float]]
        Energy offsets for each state in each of the independent end state generation simulations.
    _vacuum_simulation=vacuum_simulation: bool, optional
        this flag switches the algorithm to use all potential energies of a state, instead of only the domination sampling.
    Returns
    -------
    List[float]
        list of potential energy thresholds.
    """
    opt_pot_tresh = []
    for key, (traj, sub_eoffs) in enumerate(zip(ene_trajs, eoffs)):
        pot_tresh_state = findOccurenceSamplingThreshold(traj, sub_eoffs, filter_maxContrib=True)

        if (np.isnan(pot_tresh_state[key])):
            warnings.warn("A state potential threshold was NaN -> This hints on that you did not sample the state as a dominating one! Please check your simulatigons!")
        opt_pot_tresh.append(pot_tresh_state[key])
    return opt_pot_tresh

def findUnderSamplingPotentialEnergyThresholds(ene_trajs: List[pd.DataFrame], eoffs: List[List[float]],
                                               sampling_fraction: float = 0.95)->List[float]:
    """findUnderSamplingPotentialEnergyThresholds
    This function determines the potential energy values under which a state can be considered to be in undersampling. 
    There is one threshold per state, and a replica is considered to be in undersampling the potential energies of all
    states in the given configuration are below their respective thresholds.

    This is done in two steps, first by finding an occurence sampling threshold for each state (just as for the end state generation), 
    which assumes that all states are sampled with a "narrow" distribution of energies. Then very large thresholds which occur when we are not
    sampling certain states are simply replaced by an arbitary value (1000 kJ/mol).

    Parameters
    ----------
    ene_trajs : List[pd.Dataframe]
        potential energies of all of the end states
    eoffs: List [List[float]]
        Energy offsets for each state in each of the independent end state generation simulations.
    sampling_fraction : float, optional
        the sampling threshold for detecting undersampling (default: 0.95).

    Returns
    -------
    List[float]
        potential energy thresholds for each state
    """
    
    state_names = [i for i in ene_trajs[0].columns if (i.startswith("e") and not i == "eR")]
    num_states = len(state_names)

    undersampling_thresholds = - np.Inf * np.ones(num_states)

    fraction = sampling_fraction * len(ene_trajs[-1])

    for k, traj in enumerate(ene_trajs):
        thres = findOccurenceSamplingThreshold(traj, eoffs[k], filter_maxContrib=False)
        
        pot_ene = np.array(traj[state_names])
        thres[thres > 1000] = 0 
        
        n_below_thres = [np.sum(pot_ene[:, i] < thres[i]) for i in range(num_states)]
        if np.all(np.array(n_below_thres) > fraction):
            # Keep max as the overall threshold (to be a bit more conservative)         
            undersampling_thresholds = np.maximum(thres, undersampling_thresholds)
    
    return undersampling_thresholds

def calculate_sampling_distributions(ene_trajs: List[pd.DataFrame], eoffs: List[List[float]],
                                     potential_treshold: List[float]) -> Dict[int, Dict[str, Dict[int, float]]]:
    """calculate_sampling_distributions
    This function is using the dominating state sampling and occurrence state sampling definition, to calculate
    for both definitions the sampling distributions including each stat.
    Further each replica, having an occurrence sampling >= undersampling_occurence_sampling_tresh for each state is categorized as undersampling.

    Parameters
    ----------
    ene_trajs: List[pd.Dataframe]
        a list of pandas dataframes containing the energy data of each state eX
    potential_treshold: List[float]
        a list of potential thresholds for undersampling
    eoffs : List[List[float]]
        energy offsets for each replica

    Returns
    -------
    Dict[int, Dict[str, Dict[int, float]]]
        the dictionary contains for all replicas, the occurrence/minState/maxContribution sampling fractions of each state  and undersamplin classification
    """

    replica_sampling_dist = {}
    num_states = sum([1 for i in ene_trajs[0].columns if (i.startswith("e") and not i == "eR")])
    states = ["e" + str(s) for s in range(1, 1 + num_states)]
    for ind, replica in enumerate(ene_trajs):
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

def sampling_analysis(ene_trajs: List[pd.DataFrame],
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
    ene_trajs : List[pd.DataFrame]
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
        eoffs = [eoffs for x in range(len(ene_trajs))]

    for ind, replica in enumerate(ene_trajs):
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
    replica_sampling_distributions = calculate_sampling_distributions(ene_trajs=ene_trajs, eoffs=eoffs,
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



def detect_undersampling(ene_trajs: List[pd.DataFrame],
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
    ene_trajs : List[pd.DataFrame]
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
    sampling_stat, out_path = sampling_analysis(ene_trajs=ene_trajs,
                                                state_potential_treshold=state_potential_treshold,
                                                s_values=s_values, eoffs=eoffs,
                                                out_path=out_path,
                                                xmax= xmax,
                                                _visualize=_visualize,
                                                verbose = verbose, _usample_run=True)

    ##get undersampling id:
    undersampling_idx = None
    replica_sampling_distributions = sampling_stat["samplingDistributions"]
    for i in replica_sampling_distributions:
        # undersampling?
        undersampling_criterium = True if (all([occ >= undersampling_occurence_sampling_tresh for occ in replica_sampling_distributions[i]["occurence_state"].values()])) else False
        sampling_stat["samplingDistributions"][i].update({"undersampling":undersampling_criterium})

        if undersampling_criterium:
            undersampling_idx = i
            found_undersampling = True
            break

    if undersampling_idx is None:
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
