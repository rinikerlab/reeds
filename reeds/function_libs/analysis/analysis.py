"""
analysis
--------

the :mod:`analysis` contains multiple functions, which can be used:

 - concat Files
 - analyse a run
 - generate a result folder for following simulations

This script analyses a Reeds-simulation the two parameters: Eoff and sopt
"""

import json
import os
import warnings
from typing import List, Dict

import numpy as np
import pandas as pd

import reeds

from pygromos.files import repdat, imd
from pygromos.gromos import gromosPP
from pygromos.utils import bash

from reeds.function_libs.analysis import visualisation as vis
from reeds.function_libs.optimization import eds_energy_offsets as eoff, eds_s_values as sopt_wrap
from reeds.function_libs.optimization.src import sopt_Pathstatistic as parseS

np.set_printoptions(suppress=True)

# SORT this file!
def fetch_lig_atom(atom_id: int, atoms_lig: Dict) -> (int, int):
    """

    Parameters
    ----------
    atom_id
    atoms_lig

    Returns
    -------

    """
    start_sum = 0
    for x in sorted(atoms_lig):
        new_start_sum = start_sum + atoms_lig[x]["n"]
        if (atom_id <= new_start_sum):
            return x, atom_id - start_sum  # lig, atom
        else:
            start_sum = new_start_sum
    raise ValueError("Could not find the atom!")


def analyse_V_t(data_dict: dict, pot_tresh=0.0):  # make bins for minimal V and undersample V
    """
        move this function
    Parameters
    ----------
    data_dict
    pot_tresh

    Returns
    -------

    """
    time_axis = data_dict.pop("time")  # remove time column from dict and store

    stuff = {data_dict.pop(x) for x in data_dict if (not x.startswith("e"))}
    del stuff
    data = {}
    for index, t in enumerate(time_axis):  # go rowise along time

        Vy = []

        for x in sorted(data_dict, key=lambda x: int(x.replace("e", ""))):
            Vy.append(data_dict[x][index])  # extract column wise e1, e2 ...en

        Vmin = min(Vy)
        statemin = Vy.index(Vmin)  # find min Vy
        stateundersampling = [Vy.index(y) for y in Vy if (float(y) <= pot_tresh)]  # find undersampling Vy
        data.update({t: {"Vmin": statemin, "undersampling": stateundersampling}})
        # print(str(t)+"\t"+str(statemin)+"\t"+str(stateundersampling))

    return data


"""
    Sampling fractions
"""

## Calculate automatically the pot_thresh factor
def undersampling_occurence_potential_threshold_densityClustering(ene_traj_csvs: List[pd.DataFrame],
                                                                  max_distance_kJ: float = 300,
                                                                  sampling_fraction_treshold: float = 0.9):
    """
    This function is estimating the pot_tresh for all states by using DBSCAN identifying the density region containing sampling_fraction_treshold of the data.
    The mean and std of the density region will result in pot_tresh = mean+3std
    This function is rather performance expensive.
    ! WARNING REQUIRES SKLEARN

    Parameters
    ----------
    ene_traj_csvs: List[pd.Dataframe]
        a list of pandas dataframes containing the energy data of each state eX
    max_distance_kJ: float, optional
        the maximum distance between two points for dbscan clustering (eps) (default:300kJ).
    sampling_fraction_treshold: float, optional
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
                                                                   max_distance_kJ: float = 300,
                                                                   sampling_fraction_treshold: float = 0.9):
    """
    This function is estimating the pot_tresh for all states by testing if around the minimal energy 90% of the data is located in a threshold of  max_distance_kJ.
    The mean and std of the density region will result in pot_tresh = mean+3std
    This function is cheap in performance.
    ! WARNING REQUIRES SKLEARN

    Parameters
    ----------
    ene_traj_csvs: List[pd.Dataframe]
        a list of pandas dataframes containing the energy data of each state eX
    max_distance_kJ: float, optional
        the maximum deviation around minimal energy (default:300kJ).
    sampling_fraction_treshold: float, optional
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

    for i, replica_data in enumerate(ene_traj_csvs):
        # print("")
        # print("replica " + str(i + 1) + ":")

        # iterate over states, identify all undersampling replicas, calc fraction
        pot_tresh = []
        for k, state in enumerate(state_names):
            vec = np.array(replica_data[state]).reshape(-1, 1)
            min_pot = np.min(vec)
            below_thresh = vec[vec < min_pot + max_distance_kJ]
            below_thresh_fraction = len(below_thresh) / total_step_num

            if (sampling_fraction_treshold < below_thresh_fraction):
                threshold = np.mean(below_thresh) + 3 * np.std(below_thresh)
                pot_tresh.append(threshold)

        # only save undersapmling pot_threshes
        if (len(pot_tresh) == num_states):
            pot_tresh_pre_rep.append(pot_tresh)

    # final averaging.
    pot_thresh_per_state_and_replica = np.array([t for t in pot_tresh_pre_rep if (len(t) == num_states)]).T
    pot_thresh_per_state = np.mean(pot_thresh_per_state_and_replica, axis=1) + np.std(pot_thresh_per_state_and_replica,
                                                                                      axis=1)
    return pot_thresh_per_state

## Calculate the sampling of the states
def calculate_sampling_distributions(ene_traj_csvs: List[pd.DataFrame],
                                     potential_treshold: List[float],
                                     undersampling_occurence_sampling_tresh: float = 0.75) -> Dict[
    int, Dict[str, Dict[int, float]]]:
    """
        This function is using the dominating state sampling and occurrence state sampling definition, to calculate
        for both definitions the sampling distributions including each stat.
        Further each replica, having an occurrence sampling >= undersampling_occurence_sampling_tresh for each state is categorized as undersampling.

    Parameters
    ----------
    ene_traj_csvs
    potential_treshold
    undersampling_occurence_sampling_tresh

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


## Combining all samplnig functions
def sampling_analysis(ene_traj_csvs: List[pd.DataFrame], s_values: List[float], pot_tresh: List[float],
                      out_path: str = None, xmax: bool = False, verbose: bool = False, do_plot: bool = True) -> str:
    """
    This function, is analysing the samplings

    Parameters
    ----------
    out_path :  str
        path out for the plotfiles
    ene_traj_csvs : List[pd.DataFrame]
        contains the energy data
    s_values :  List[float]
        list of s_values
    pot_tresh : float
        potential energy threshold, for considering a state as sampled
    xmax :  bool
        additionally output a plot only with a smaller x_range. (0->xmax)
    verbose: bool
        story time :)

    Returns
    -------
    str
        out_path
    """

    # read all Vy_sx_files!
    from reeds.function_libs.analysis.visualisation import nice_s_vals
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
            vis.plot_t_statepres(data=data,
                                 title="s=" + str(s_vals_nice[ind]) + ", with V_{tresh}=" + str(pot_tresh),
                                 out_path=out_path + "/sampling_timeseries_s" + str(ind + 1) + ".png")

        if (xmax):
            vis.plot_t_statepres(data=data,
                                 title="s=" + str(s_vals_nice[ind]) + ", with V_{tresh}=" + str(pot_tresh),
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
            vis.plot_stateOccurence_hist(data=replica_sampling_distributions[x],
                                         title="s=" + str(s_vals_nice[ind]) + ", V_{tresh}=" + str(pot_tresh),
                                         out_path=out_path + "/sampling_hist_" + str(x) + ".png")

    if (do_plot):
        if (verbose): print("\n\n Sampling Matrix\n\n")
        vis.plot_stateOccurence_matrix(data=replica_sampling_distributions, out_dir=out_path, s_values=s_vals_nice,
                                       place_undersampling_threshold=True, title_suffix="V_{tresh}=" + str(pot_tresh))

    ##get undersampling id:
    found_undersampling = False
    for undersampling_ind in replica_sampling_distributions:
        if (replica_sampling_distributions[undersampling_ind]["undersampling"]):
            found_undersampling = True
            break

    if (not found_undersampling):
        warnings.warn("Could not find undersampling!")

    final_results = {"undersamlingThreshold": undersampling_ind,
                     "potentialThreshold": potential_treshold,
                     "samplingDistributions": replica_sampling_distributions}
    return final_results, out_path


"""
    Parameter optimization
"""


# Energy Offsets

def estimate_Eoff(ene_ana_trajs: List[pd.DataFrame], Eoff: List[float], s_values: List[float], out_path: str,
                  temp: float = 298,
                  kb: float = 0.00831451, pot_tresh: float = 0.0, frac_tresh: float = [0.9],
                  convergence_radius: float = 0.05, max_iter=12,
                  take_last_n: int = None, plot_title_prefix: str = "RE-EDS", visualize: bool = True) -> Dict:
    """estimate_Eoff
        estimate Energy Offsets of REEDS sim.

    Parameters
    ----------
    Eoff :  List[float]
    s_values :  List[float]
    ene_ana_trajs : List[pd.DataFrame]
    out_path : str
    plot_title_prefix : str, optional
    temp : float, optional
    kb: float, optional
    pot_tresh : float, optional
    frac_tresh : float, optional
    convergence_radius:float, optional
    max_iter:   float, optional
    visualize : bool, optional

    Returns
    -------
    Dict
        EOFF dictionary
    """

    # TODO: NEW AND BETTER?
    # calculate energy offsets:

    if take_last_n is not None:
        ene_ana_trajs = ene_ana_trajs[-take_last_n:]
        s_values = s_values[-take_last_n:]
        print("using Replicas: ", len(ene_ana_trajs), s_values)
    statistic = eoff.estEoff(ene_ana_trajs=ene_ana_trajs, out_path=out_path + "/Eoff_estimate.out", Temp=temp,
                             s_values=s_values, Eoff=Eoff, frac_tresh=frac_tresh, pot_tresh=pot_tresh,
                             convergenceradius=convergence_radius, kb=kb, max_iter=max_iter)

    # make plots and gather stuff!
    if (visualize):
        eoff_per_replica = statistic[3]
        energy_offsets = statistic[0]

        # plott mean eoffs vs. sf
        vis.plot_peoe_eoff_vs_s(eoff_per_replica, energy_offsets,  title=plot_title_prefix + " - Eoff/s",
                                out_path=out_path + "/" + plot_title_prefix + "_eoffs_vs_s.png")
    return statistic


def energyOffset_time_convergence(ene_ana_trajs, out_dir: str, Eoff: List[float], s_values: List[float],
                                  steps: int = 10,
                                  temp: float = 298, kb: float = 0.00831451, pot_tresh: float = 0.0,
                                  frac_tresh: float = [0.9], convergence_radius: float = 0.05, max_iter=12,
                                  plot_title_prefix: str = "RE-EDS", visualize: bool = True):
    ene_traj = ene_ana_trajs[0]

    start_index, end_index = (min(ene_traj.index), max(ene_traj.index))
    step_size = (end_index - start_index) // steps
    time_step = ene_traj.time[1] - ene_traj.time[0]
    steps_range = range(step_size, end_index, step_size)

    time_statistic = {}
    for last_step in steps_range:
        # shorten data
        shortened_trajs = []
        for ene_traj in ene_ana_trajs:
            tmp_ene_traj = ene_traj.iloc[ene_traj.index < last_step]
            tmp_ene_traj.replicaID = ene_traj.replicaID
            shortened_trajs.append(tmp_ene_traj)
            print(len(tmp_ene_traj))

        # estm eoff
        statistic = eoff.estEoff(ene_ana_trajs=shortened_trajs, out_path=out_dir + "/Eoff_estimate_new.out",
                                 Temp=temp, s_values=s_values, Eoff=Eoff,
                                 frac_tresh=frac_tresh, pot_tresh=pot_tresh, convergenceradius=convergence_radius,
                                 kb=kb,
                                 max_iter=max_iter)

        time_statistic.update({last_step: statistic})

    state_time_dict = {"time_step": time_step}
    sorted_keys = sorted(time_statistic)
    last_steps = list(steps_range)
    for stateInd, (mean, std) in enumerate(time_statistic[sorted_keys[0]].offsets):
        state_time_dict.update({stateInd + 1: {"mean": [mean], "std": [std], "time": [time_step * last_steps[0]]}})

    for ene_traj_key in sorted_keys[1:]:
        ene_traj = time_statistic[ene_traj_key]
        for stateInd, (mean, std) in enumerate(ene_traj.offsets):
            state_time_dict[stateInd + 1]["mean"].append(mean)
            state_time_dict[stateInd + 1]["std"].append(std)
            state_time_dict[stateInd + 1]["time"].append(ene_traj_key * time_step)

    if (visualize):
        vis.plot_peoe_eoff_time_convergence(state_time_dict,
                                            out_path=out_dir + "/" + plot_title_prefix + "_eoff_convergence.png")
    return state_time_dict


# S-optimization
def optimize_s(in_file: str, add_s_vals: int, out_dir: str, title_prefix: str = "sOpt",
               state_weights=None, trial_range: (int or tuple) = None,
               with_skipped: bool = True, verbose: bool = True,
               run_NLRTO: bool = True, run_NGRTO: bool = False, in_imd: str = None) -> Dict:
    """optimize s
        This function is doing the S-optimization for a RE-EDS simulation.

    Parameters
    ----------
    in_file :   str
        input path to repdat-file
    add_s_vals :    int
        number of s_values to be added
    out_dir :   str
        output_directory path
    title_prefix :  str, optional
        title prefix for plots
    state_weights : List[float], optional
        weights for each individual endstate of eds potential
    trial_range :   [int, tuple], optional
        range of trials (time dimension
    with_skipped :  bool , optional
        ???
    verbose :   bool , optional
    run_NLRTO : bool, optional
    run_NGRTO : bool, optional
    in_imd :    str, optional
        this path to an imd file is suggested to be used, as it increases the s_value accuracy.

    Returns
    -------
    Dict

    """

    # do soptimisation
    if verbose:
        print("RUN sopt")
        print("\tRead in repdat: " + in_file)
    stat = parseS.generate_PathStatistic_from_file(in_file, trial_range=trial_range)

    if (in_imd != None):
        imd_file = imd.Imd(in_imd)
        svals = list(map(float, imd_file.REPLICA_EDS.RES))
        setattr(stat, "s_values", sorted(list(set(svals)), reverse=True))
        setattr(stat, "raw_s_values", sorted(svals, reverse=True))

    else:
        warnings.warn("Careful no imd given, the accuracy of repdat s_vals is very low!")

    if verbose: print("\n\tOptimize S-Dist")
    # NLRTO
    if (run_NLRTO):
        if verbose: print("\tNLRTO")
        new_svals_NLRTO, NLRTO = sopt_wrap.calc_NLRTO(stat, add_n_s=add_s_vals, state_weights=state_weights,
                                                      verbose=verbose)
    else:
        new_svals_NLRTO = []
        NLRTO = None

    # NGRTO
    if (run_NGRTO):
        if verbose: print("\tNGRTO")
        smin = min(stat.s_values)
        ds = float("0.0" + "".join(["0" for x in range(str(smin).count("0") + 1)]) + "1")
        new_svals_NGRTO, NGRTO = sopt_wrap.calc_NGRTO(stat, add_n_s=add_s_vals, state_weights=state_weights, ds=ds,
                                                      verbose=verbose)
    else:
        new_svals_NGRTO = []
        NGRTO = None

    if (with_skipped):
        data = {0: stat.raw_s_values, 1: new_svals_NLRTO, 2: new_svals_NGRTO}
    else:
        data = {0: stat.s_values, 1: NLRTO.opt_replica_parameters, 2: NGRTO.opt_replica_parameters}

    # output of svals NLRTO
    if (run_NLRTO):
        output_file = open(out_dir + "/" + title_prefix + "_NLRTO.out", "w")
        output_file.write(str(NLRTO))
        output_file.close()

    # output of svals NGRTO
    if (run_NGRTO):
        output_file = open(out_dir + "/" + title_prefix + "_NGRTO.out", "w")
        output_file.write(str(NGRTO))
        output_file.close()

    if verbose: print("Done!\n\n")

    return data


def get_s_optimization_transitions(out_dir: str, rep_dat: str, title_prefix, verbose=True):
    # visualize s-distribution
    """
    Args:
        out_dir (str):
        rep_dat (str):
        title_prefix:
        verbose:
    """
    if verbose: print("Vizualise \n\t")
    if verbose: print("Plot output in: " + out_dir)

    # visualize transitions
    repdat_file = repdat.Repdat(rep_dat)
    old_svals = list(map(float, repdat_file.system.s))

    # print(list(repdat_file.keys()))
    if verbose: print("\t transition plots ")
    transitions = repdat_file.get_replica_traces()
    if verbose: print("\t\t draw replica traces ")
    vis.plot_replica_transitions(transitions, s_values=old_svals, out_path=out_dir + "/transitions.png",
                                 title_prefix=title_prefix)
    vis.plot_replica_transitions(transitions, s_values=old_svals, out_path=out_dir + "/transitions_cutted.png",
                                 title_prefix=title_prefix, cut_1_replicas=True)
    vis.plot_replica_transitions(transitions, s_values=old_svals, out_path=out_dir + "/transitions_cutted_250.png",
                                 title_prefix=title_prefix, cut_1_replicas=True, xBond=(0, 250))
    # single trace replica
    if verbose: print("\t\t draw single replica trace ")
    for replica in range(1, len(repdat_file.system.s) + 1):  # future: change to ->repdat_file.num_replicas
        single_transition_trace = transitions.loc[transitions.replicaID == replica]
        vis.plot_replica_transitions_min_states(single_transition_trace, s_values=old_svals,
                                                out_path=out_dir + "/transitions_trace_" + str(replica) + ".png",
                                                title_prefix=title_prefix,
                                                cut_1_replicas=True)
        # vis.plot_state_transitions_min_states(single_transition, s_values=old_svals,
        #                           out_path=out_path + "/transitions_trace_" + str(replica) + "_250t.png",
        #                           title_prefix=title_prefix,
        #                           cut_1_replicas=True, xBond=(0, 250))


def get_s_optimization_roundtrips_per_replica(data, max_pos: int, min_pos: int, repOffsets=0) -> dict:
    replicas = np.unique(data.replicaID)[repOffsets:]
    max_pos, min_pos = min_pos, max_pos

    replica_stats = {}
    for replica in replicas:
        df = data.loc[data.replicaID == replica]
        extreme_pos = df.loc[(df.position == min_pos) | (df.position == max_pos)]

        roundtrip_counter = 0
        durations = []
        if (len(extreme_pos) > 0):
            lastExtreme = extreme_pos.iloc[0].position
            lastTrial = extreme_pos.iloc[0].trial
            for row_ind, extr in extreme_pos.iterrows():
                if (extr.position != lastExtreme):
                    roundtrip_counter += 1
                    durations.append(extr.trial - lastTrial)

                    lastExtreme = extr.position
                    lastTrial = extr.trial
        replica_stats.update({replica: {"roundtrips": roundtrip_counter, "durations": durations}})
        
    return replica_stats


def get_s_optimization_roundtrip_averages(stats):
    nReplicasRoundtrips = np.sum([1 for stat in stats if (stats[stat]["roundtrips"] > 0)])
    numberOfRoundtrips = [stats[stat]["roundtrips"] for stat in stats]
    avg_numberOfRoundtrips = np.mean(numberOfRoundtrips)

    roundtrip_durations = [np.mean(stats[stat]["durations"]) for stat in stats if (len(stats[stat]["durations"]) > 0)]
    avg_roundtrips = np.inf if (np.mean(roundtrip_durations) == 0 or len(roundtrip_durations) == 0) else np.mean(
        roundtrip_durations)

    print("Number of roundtrips per replica: ", numberOfRoundtrips)
    print("Avg. Roundtrips per replica: ", roundtrip_durations)
    print()
    return nReplicasRoundtrips, avg_numberOfRoundtrips, avg_roundtrips


"""
    Production
"""


# Free Energy calculation
def multi_lig_dfmult(num_replicas: int, num_states: int, in_folder, out_file, eR_prefix: str = "eR_s",
                     eX_prefix: str = "eY_s", temp: int = 298, gromos_binary="dfmult", verbose=True) -> dict:
    """multi_lig_dfmult
        @ WARNING GOING TO BE REMOVED - OLD Function@
        This function is calculating dfmult with multiple ligands in a system

    Parameters
    ----------
    num_replicas :
    num_states :
    in_folder :
    out_file :
    eR_prefix :
    eX_prefix :
    temp :
    gromos_binary :
    verbose :

    Returns
    -------

    """

    def gen_results_string(results_dict: dict) -> str:
        # This func generates an readable output
        # generate out string:
        result_string = ""
        float_format = "{: > 13.3f}"
        str_format = "{:>13}"
        str_head_format = "{:^27}"
        state = ["lig" + str(state) for state in range(1, num_states + 1)]
        state.append("ligR")

        for s in results_dict:
            first = True
            s_results_dict = results_dict[s]
            print("result_keys: " + str(list(s_results_dict.keys())))
            result_string += "\n dF for s= " + str(s) + "\n"
            for l1 in sorted(s_results_dict):

                # Header
                if (first):
                    results = []
                    format_list = []
                    for l2 in sorted(s_results_dict["lig" + str(num_states)]):
                        results.append(l2)
                        format_list.append(str_head_format)
                    formatting = "{: <10} " + " | ".join(format_list) + "\n"
                    result_string += formatting.format("ligname", *results)

                    results = []
                    format_list = []
                    for l2 in state:
                        results.append("mean")
                        results.append("err")
                        format_list.extend([str_format + " " + str_format])
                    formatting = "{: <10} " + " | ".join(format_list) + "\n"
                    result_string += formatting.format("ligname", *results)
                    first = False

                results = []
                format_list = []
                for l2 in state:
                    if (l2 in s_results_dict[l1]):
                        results.append(float(s_results_dict[l1][l2]["mean"]))
                        results.append(float(s_results_dict[l1][l2]["err"]))
                        format_list.extend([float_format + " " + float_format])
                    else:
                        results.append(str(" - "))
                        results.append(str(" - "))
                        format_list.extend([str_format + " " + str_format])

                formatting = "{: <10} " + " | ".join(format_list) + "\n"
                result_string += formatting.format(l1, *results)

            result_string += "\n"
        result_string += "\n"
        return result_string

    # DO PER REPLICA
    grom = gromosPP.GromosPP()
    results_dict = {}
    dfmult_replica_results = []
    for s in range(1, num_replicas + 1):
        # reference Hamiltonian Potential file at s
        results_dict.update({s: {}})
        eR_file = in_folder + "/" + eR_prefix + str(s) + ".dat"

        # find Potentials of each state
        ei_files = []
        for i in range(1, num_states + 1):
            ei_files.append(in_folder + "/" + eX_prefix.replace("Y", str(i)) + str(s) + ".dat ")
        print("\n", str(s), "\n")

        tmp_file_path = grom.dfmult(ei_files, in_reference_state_file_path=eR_file, temperature=temp,
                                    _binary_name=gromos_binary,
                                    out_file_path=os.path.dirname(out_file) + "/tmp_" + str(s) + ".out")

        # parse again file for nicer
        tmp_file = open(tmp_file_path, "r")
        for line in tmp_file.readlines():

            fields = line.split()
            keyfields = fields[0].split("_")
            dF_mean = fields[1]
            dF_err = fields[2]
            if (len(fields) < 3 or len(keyfields) < 3):
                continue
            else:
                l1 = "lig" + str(keyfields[1])
                if ("lig" + str(keyfields[1]) in results_dict[s]):
                    results_dict[s][l1].update({"lig" + str(keyfields[2]): {"mean": dF_mean, "err": dF_err}})
                else:
                    results_dict[s].update({l1: {"lig" + str(keyfields[2]): {"mean": dF_mean, "err": dF_err}}})
        tmp_file.close()
        dfmult_replica_results.append(tmp_file_path)

    # FINAL write out
    result_file = open(out_file, "w")
    result_file.write(gen_results_string(results_dict))
    result_file.close()

    for file_path in dfmult_replica_results:
        bash.remove_file(file_path)
    return results_dict


def free_energy_convergence_analysis(ene_ana_trajs: List[pd.DataFrame], out_dir: str, in_prefix: str = "",
                                     out_prefix: str = "energy_convergence", dfmult_all_replicas: bool = True,
                                     time_blocks: int = 10, recalulate_all: bool = False, verbose: bool = False):
    """free_energy_convergence

    Parameters
    ----------
    ene_ana_trajs:  List[pd.DataFrame]
    out_dir :
    in_prefix :
    out_prefix :
    recalulate_all :
    verbose:    bool, optional
    time_blocks:    int,    optional
    Returns
    -------

    """

    if (verbose): print("start dfmult")
    ene_trajs = {int(ene_ana_traj.s.replace("s", "")): ene_ana_traj for ene_ana_traj in ene_ana_trajs}

    # DF Convergence
    if (verbose): print("Calc conf")
    dF_conv_all_replicas = {}

    # generate time conevergence data
    summary_file_path = out_dir + "/summary_allReplicas_dfmult_timeblocks_" + out_prefix + ".dat"
    if (False and os.path.exists(summary_file_path) and not recalulate_all):
        if (verbose): print("Found summary file\t LOAD: " + summary_file_path)
        dF_conv_all_replicas = json.load(open(summary_file_path, "rb"))
    else:

        if (verbose): print("CALCULATE")
        if dfmult_all_replicas:
            svals_for_analysis = sorted(ene_trajs)
        else:
            svals_for_analysis = [1]  # just do s = 1.

        for s_index in svals_for_analysis:
            replica_key = "replica_" + str(s_index)
            ene_traj = ene_trajs[s_index]
            if (verbose): print("\n\nREPLICA: ", s_index)

            # CALC_ Free Energy
            dF_time = eds_dF_time_convergence_dfmult(eds_eneTraj=ene_traj, out_dir=out_dir, time_blocks=time_blocks,
                                                     verbose=verbose)
            dF_conv_all_replicas.update({replica_key: dF_time})

            # Plotting
            if (verbose): print("Plotting")
            updated_keys = {str(replica_key) + "_" + str(key): value for key, value in
                            dF_conv_all_replicas[replica_key].items() if (key.endswith("1"))}
            vis.plot_dF_conv(updated_keys, title="Free energy convergence",
                             out_path=out_dir + "/" + out_prefix + "_" + str(replica_key), show_legend=True)
            json.dump(dF_conv_all_replicas, fp=open(out_dir + "/tmp_dF_" + str(s_index) + ".dat", "w"), indent=4)

        json.dump(dF_conv_all_replicas, fp=open(summary_file_path, "w"), indent=4)

    # nice_result file
    out_dfmult = open(out_dir + "/free_energy_result.dat", "w")
    out_string = gen_results_string(dF_conv_all_replicas)
    out_dfmult.write(out_string)
    out_dfmult.close()


## Helper functions
def eds_dF_time_convergence_dfmult(eds_eneTraj: pd.DataFrame, out_dir, time_blocks: int = 10, gromos_bindir: str = None,
                                   verbose: bool = False) -> Dict:
    """dF_time_convergence_dfmult

            This function generates a dictionary, which contains time-dependent dF values.

    Parameters
    ----------
    eds_eneTraj :
    out_dir :
    time_blocks :
    gromos_bindir :
    verbose :

    Returns
    -------

    """
    # Split the data timewise!

    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    traj_length = eds_eneTraj["time"].size
    if (verbose): print("traj length: " + str(traj_length))

    step_size = traj_length // time_blocks
    if (verbose): print("step size: " + str(step_size))
    data_chunk_indices = list(range(step_size, traj_length + step_size, step_size))
    data_chunk_indices.insert(0, 10)  # add a very small time interval at the beginning

    if (verbose): print("\nGATHER ", time_blocks, "time _blocks in steps of ", step_size, "\tleading to final: ",
                        eds_eneTraj["time"][traj_length - 1], "\n")
    if (verbose): print("data chunks: ")
    if (verbose): print(data_chunk_indices)

    # STEP1: generate files with blocked data
    properties_timesolv = {}
    for property in sorted(filter(lambda x: x.startswith("e"), eds_eneTraj.columns)):
        # chunk data
        if (verbose): print("\t", property)
        sub_ste = []
        for step, end_step in enumerate(data_chunk_indices):
            if (verbose): print(step, " ", end_step, end="\t")
            out_path = out_dir + "/tmp_" + property + "_step" + str(step) + ".dat"
            eds_eneTraj[["time", property]][:end_step].to_csv(out_path, sep="\t", header=["# time", property],
                                                              index=False)
            sub_ste.append(out_path)

        if (end_step != traj_length):
            if (verbose): print(step + 1, " ", end_step, end="\t")
            out_path = out_dir + "/tmp_" + property + "_step" + str(step + 1) + ".dat"
            eds_eneTraj[["time", property]].to_csv(out_path, sep="\t", header=["# time", property], index=False)
            sub_ste.append(out_path)
            bash.wait_for_fileSystem(out_path)
        if (verbose): print()
        properties_timesolv.update({property: sub_ste})

        # check if lsf is has the result files already.
        bash.wait_for_fileSystem(sub_ste)

    # STEP2: calc dF timedep - with generated files from above
    if (verbose): print("\n\ncalc dF with dfmult")
    gromPP = gromosPP.GromosPP(gromos_bindir)
    dF_timewise = {}
    for step, stop_index in enumerate(data_chunk_indices):
        Vr = properties_timesolv["eR"][step]
        Vi = list(
            sorted([value[step] for key, value in properties_timesolv.items() if (key != "eR" and key.startswith("e"))],
                   key=lambda x: int(os.path.basename(x).split("_")[1].replace("e", ""))))

        tmp_out = out_dir + "/tmp_dfmult_" + str(step) + ".dat"
        if (verbose): print("Dfmult: ", step)
        tmp_out = gromPP.dfmult(in_endstate_file_paths=Vi, in_reference_state_file_path=Vr, out_file_path=tmp_out,
                                verbose=verbose)

        out_tmp = open(tmp_out, "r")
        lines = out_tmp.readlines()

        for line in lines:
            if ("DF_" in line):
                print(line)
                ind_1, ind_2 = line.strip().split("  ")[0].replace("DF_", "").split("_")
                key_v = str(ind_1) + "_" + str(ind_2)
                if (not key_v in dF_timewise):
                    dF_timewise.update({key_v: {}})
                h, meanDF, errDF = line.split()
                dF_timewise[key_v].update(
                    {float(eds_eneTraj["time"][stop_index - 1]): {"mean": float(meanDF), "err": float(errDF)}})

        # STEP 3: CLEAN
        Vi.append(Vr)
        for tmp_path in Vi:
            os.remove(tmp_path)
        out_tmp.close()
        os.remove(tmp_out)

    return dF_timewise


def gen_results_string(results_dict: dict) -> str:
    # This func generates an readable output
    # generate out string:
    result_string = ""
    float_format = "{: > 13.3f}"
    str_format = "{:>13}"
    str_head_format = "{:^27}"
    states = []
    for s in sorted(results_dict, key=lambda x: int(x.split("_")[1])):
        first = True
        s_results_dict = results_dict[s]
        result_string += "\n dF for " + str(s) + "\n\n"

        # translate dict
        file_dict = {}
        for ligand_pair in s_results_dict:
            l1 = ligand_pair.split("_")[0]
            l2 = ligand_pair.split("_")[1]
            states.append(l1)
            states.append(l2)
            if (not l1 in file_dict):
                file_dict.update({l1: {l2: {"mean": "-", "err": "-"}}})
            else:
                file_dict[l1].update({l2: {"mean": "-", "err": "-"}})

            last_step = list(sorted(s_results_dict[ligand_pair].keys()))[-1]
            file_dict[l1][l2].update({"mean": float(s_results_dict[ligand_pair][last_step]["mean"]),
                                      "err": float(s_results_dict[ligand_pair][last_step]["err"])})

        # fill up missing states comparison
        states = sorted(list(set(states)))
        for l1 in states:
            for l2 in states:
                if (not l1 in file_dict):
                    file_dict.update({l1: {l2: {"mean": "-", "err": "-"}}})
                elif (l1 in file_dict and not l2 in file_dict[l1]):
                    file_dict[l1].update({l2: {"mean": "-", "err": "-"}})

        # write files

        l1_ordered_states = list(sorted(filter(lambda x: not "R" in x, list(file_dict.keys())), key=lambda x: int(x)))
        for l1 in l1_ordered_states:
            # Header
            l2_ordered_states = list(
                sorted(filter(lambda x: not "R" in x, list(file_dict[l1].keys())), key=lambda x: int(x)))
            l2_ordered_states.append("R")
            if (first):
                results = []
                format_list = []
                # ligandname line
                for l2 in l2_ordered_states:
                    results.append(l2)
                    format_list.append(str_head_format)
                formatting = "| {: <10} | " + " | | ".join(format_list) + "|\n"
                result_string += formatting.format("ligname", *results)

                # column description line
                results = []
                format_list = []
                for l2 in l2_ordered_states:
                    results.append("mean")
                    results.append("err")
                    format_list.extend([str_format + " " + str_format])
                formatting = "| {: <10} |" + " | ".join(format_list) + "|\n"
                result_string += formatting.format("ligname", *results)
                result_string += "|---" * (len(results) + 1) + "|\n"

                first = False

            results = []
            format_list = []
            for l2 in l2_ordered_states:
                if (file_dict[l1][l2]["mean"] != "-"):
                    results.append(float(file_dict[l1][l2]["mean"]))
                    results.append(float(file_dict[l1][l2]["err"]))
                    format_list.extend([float_format + " " + float_format])
                else:
                    results.append(str(" - "))
                    results.append(str(" - "))
                    format_list.extend([str_format + " " + str_format])
            formatting = "| {: <10} " + " | ".join(format_list) + "|\n"
            result_string += formatting.format(l1, *results)
        result_string += "\n"
    result_string += "\n"
    return result_string
