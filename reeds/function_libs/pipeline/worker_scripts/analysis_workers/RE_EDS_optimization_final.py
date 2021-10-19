import glob
import os
import pickle

import numpy as np
import pandas as pd
from typing import Union, List

from pygromos.files import repdat
from pygromos.utils import bash

import reeds.function_libs.visualization.re_plots
from reeds.function_libs.analysis.parameter_optimization import get_s_optimization_roundtrips_per_replica, \
    get_s_optimization_roundtrip_averages
from reeds.function_libs.visualization.parameter_optimization_plots import visualization_s_optimization_summary, \
    visualize_s_optimisation_convergence, visualize_s_optimisation_sampling_optimization
from reeds.function_libs.file_management.file_management import parse_csv_energy_trajectory



def analyse_optimization_iteration(repdat_path: str, out_dir: str, title: str, time: float, pot_tresh=0) -> dict:
    """
        analysis of a single optimization iteration.
        - analyse sampling, round trips, round trip time
        - generate replica transition plots, position histogram

    Parameters
    ----------
    repdat_path: str
        path to the repdat file - contains information about the replcia exchange
    out_dir : str
        output dir for the analysis
    title: str
        title of the iteration
    time : float
        total simulation time
    pot_tresh : Union[float, List[float]], optional
        potential threshold for observing
    

    Returns
    -------
    dict
        returns the results of the analaysis.

    """
    # repdat file

    repdat_file = repdat.Repdat(repdat_path)

    s_values = repdat_file.system.s
    trans_dict = repdat_file.get_replica_traces()
    repOff = s_values.count(1)-1

    # ouput_struct
    sopt_it = {}

    # state Sampling
    domination_counts, max_pos, min_pos, occurrence_counts = samplingAnalysisFromRepdat(pot_tresh, repOff, repdat_file)

    sopt_it.update({"state_occurence_sampling": occurrence_counts})
    sopt_it.update({"state_maxContributing_sampling": domination_counts})

    del repdat_file

    reeds.function_libs.visualization.re_plots.plot_repPos_replica_histogramm(out_path=out_dir + "/" + title.replace(" ", "_") + "replicaPositions_hist.png",
                                                                              data=trans_dict, title=title,
                                                                              s_values=s_values[repOff:], replica_offset=repOff)

    reeds.function_libs.visualization.re_plots.plot_replica_transitions(transition_dict=trans_dict,
                                                                        out_path=out_dir +   "/" + title.replace(" ", "_") + "_transitions.png",
                                                                        title_prefix=title,
                                                                        s_values=s_values, cut_1_replicas=True, equilibration_border=None)
    # calc roundtrips:
    stats = get_s_optimization_roundtrips_per_replica(data=trans_dict, repOffsets=repOff,
                                                      min_pos=min_pos, max_pos=max_pos,
                                                      time = time)
    sopt_it.update({"stats_per_replica": stats})
    sopt_it.update({"s_values": s_values[repOff:]})

    sorted_dominating_state_samping =   np.array([sopt_it["state_maxContributing_sampling"][x] for x in sorted( sopt_it["state_maxContributing_sampling"], key=lambda x: int(x.replace("Vr","")) )])
    print(sorted_dominating_state_samping)

    optimal_samp =1/len(sorted_dominating_state_samping)*100
    iteration_norm = np.round(sorted_dominating_state_samping / np.sum(sorted_dominating_state_samping) * 100, 2)
    optimal_samp_dev = np.abs(iteration_norm - optimal_samp)

    MAE = np.round(np.mean(optimal_samp_dev), 2)
    MAE_std = np.std(optimal_samp_dev)

    sopt_it.update({"sampling_distribution_optimal_deviation_s1": {"MAE": MAE, "MAE_std": MAE_std}})
    sopt_it.update({"s_values": s_values[repOff:]})



    nReplicasRoundtrips, avg_numberOfRoundtripsPerNs, avg_roundtrips = get_s_optimization_roundtrip_averages(stats)
    print("replicas doing at least one roundtrip:", nReplicasRoundtrips)
    print("avg. number of roundtrips per ns: ", avg_numberOfRoundtripsPerNs)
    print("avg. roundtrip durations:", avg_roundtrips)
    sopt_it.update({"nRoundTrips": nReplicasRoundtrips,
                    "avg_nRoundtripsPerNs": avg_numberOfRoundtripsPerNs,
                    "avg_rountrip_durations": avg_roundtrips})

    del trans_dict
    return sopt_it


def samplingAnalysisFromRepdat(pot_tresh, repOff, repdat_file):
    states = repdat_file.DATA.state_potentials.iloc[0].keys()
    eoffs = repdat_file.system.state_eir
    occurrence_counts = {state: 0 for state in states}
    maxContributing_counts = {state: 0 for state in states}

    print("cols: ", repdat_file.DATA.columns)
    all_pos = list(sorted(np.unique(repdat_file.DATA.ID)))
    min_pos, max_pos = (all_pos[repOff], all_pos[-1])
    print("extremePos: ", min_pos, max_pos)
    replica1 = repdat_file.DATA.loc[repdat_file.DATA.ID == 1]
    if (isinstance(pot_tresh, float)):
        pot_tresh = {x: pot_tresh for x in replica1.iloc[0].state_potentials}
    elif (isinstance(pot_tresh, float)):
        pot_tresh = {x: y for x, y in zip(sorted(replica1.iloc[0].state_potentials), pot_tresh)}
    print("potTresh", pot_tresh)
    for rowID, row in replica1.iterrows():
        state_pots = row.state_potentials

        for ind, state in enumerate(state_pots):
            if (state_pots[state] < pot_tresh[int(state.replace("Vr", "")) - 1]):
                occurrence_counts[state] += 1
        eoff = np.array(list(eoffs.values())).T
        eoff = eoff[0]
        id_ene = {val-eoff[int(key.replace("Vr", ""))-1]: key for key, val in state_pots.items()}
        min_state = id_ene[min(id_ene)]
        maxContributing_counts[min_state] += 1
    return maxContributing_counts, max_pos, min_pos, occurrence_counts


def do(project_dir: str, optimization_name:str,
       state_physical_occurrence_potential_threshold:Union[List, float]=0, title="", out_dir: str = None, rt_convergence=100):
    """
        This function does the final analysis of an s-optimization. It analyses the outcome of the full s-optimization iterations.
        Features:
            - generate s-optimization summary
            - average roundtrip time improvement efficiency
            - deviation of sampling occurence (normalized) from the optimal sampling distribution.

        The analysis of the individual s-optimization steps is stored and can be used at later executions. (if you want to analyse form scratch, keep in mind to delete the generated .npy files.)

    Parameters
    ----------
    project_dir : str
        directory containing all s-optimization iterations
    state_physical_occurrence_potential_threshold : Union[float, List[float]], optional
        potential energy threshold, determining undersampling (default: 0)
    title : str, optional
        title of run (default: "")
    out_dir : str, optional
        ananlysis output dir, default uses the same dir like root_dir (default: None)
    rt_convergence : int, optional
        roundtrip time convergence criterium  in ps(default: 10ps)

    """
    sopt_dirs = [x for x in os.listdir(project_dir) if (optimization_name in x and os.path.isdir(project_dir + "/" + x))]

    # sopt out_dir
    if (isinstance(out_dir, type(None))):
        out_dir = project_dir + "/analysis"

    out_dir = bash.make_folder(out_dir)

    # analyse the individual iterations
    sopt_data = {}
    repdat_files = {}
    converged = False
    print(sopt_dirs)
    for iteration_folder in sorted(sopt_dirs):
        if("sopt" in iteration_folder):
            iteration = int(iteration_folder.replace("sopt", ""))
        elif("eoffRB" in iteration_folder):
            iteration = int(iteration_folder.replace("eoffRB", ""))
        else:
            raise Exception("OHOH!")

        print( iteration, end="\t")
        repdat = glob.glob(project_dir + "/" + iteration_folder + "/analysis/data/*repdat*")
        out_iteration_file_path = out_dir + "/" + iteration_folder + "_ana_data.npy"

        if (os.path.exists(out_iteration_file_path)):
            print("\n\nLoading precomputed data for iteration: ", end="\t")
            opt_it_stats = pickle.load(open(out_iteration_file_path, "rb"))
            sopt_data.update({iteration_folder: opt_it_stats})

        elif (len(repdat) == 1):
            print("\nCalculate statistics for iteration: ", iteration)
            repdat_files.update({iteration: repdat[0]})
            energies_s1 = os.path.abspath(glob.glob(project_dir + "/" + iteration_folder + "/analysis/data/*energies_s1.dat")[0])
            energy_trajectory_s1 = parse_csv_energy_trajectory(in_ene_traj_path = energies_s1, verbose = False)
            time = energy_trajectory_s1.time[len(energy_trajectory_s1)-1]
            opt_it_stats = analyse_optimization_iteration(repdat_path=repdat_files[iteration], out_dir=out_dir,
                                                           title="s-opt " + str(iteration), pot_tresh=state_physical_occurrence_potential_threshold,
                                                           time = time)

            pickle.dump(obj=opt_it_stats, file=open(out_iteration_file_path, "wb"))
            sopt_data.update({iteration_folder: opt_it_stats})
        else:
            continue
        # round trip time efficiency
        if (iteration > 1):
            prefix = "sopt" if("sopt" in list(sopt_data.keys())[0]) else "eoffRB"
            opt_it_stats.update({"avg_rountrip_duration_optimization_efficiency": sopt_data[prefix+str(iteration- 1)]["avg_rountrip_durations"] -opt_it_stats["avg_rountrip_durations"]})

        #assign convergence in an conserative fasion.
        opt_it_stats.update({"converged": converged})
        if(iteration > 1 and opt_it_stats["avg_rountrip_duration_optimization_efficiency"] <  rt_convergence ):
            converged=True


    print("\n Do summary Plots:\n")

    #overview
    print("\tmetric quartet")
    visualization_s_optimization_summary(s_opt_data=sopt_data, out_path=out_dir + "/" + title + "_"+optimization_name+"_analysis.png")

    #Optimization  convergence:
    print("\tRT-convergence")
    visualize_s_optimisation_convergence(s_opt_data=sopt_data, out_path=out_dir + "/" + title + "_"+optimization_name+"_efficiency.png", convergens_radius=rt_convergence)

    #Optimization Distribution
    print("\tsamplingDist - convergence")
    visualize_s_optimisation_sampling_optimization(s_opt_data=sopt_data, out_path=out_dir + "/" + title + "_"+optimization_name+"_sampling_optimumDev.png")

    #Final storing
    ##serialization
    print("\tstore")
    final_analysis_out_path = out_dir + "/"+title+"_final_"+optimization_name+"_analysis.obj"
    pickle.dump(opt_it_stats, open(final_analysis_out_path, "wb"))

    ##Human readable:
    final_analysis_out_csv_path = out_dir + "/"+title+"_final_"+optimization_name+"_analysis.csv"
    df = pd.DataFrame(sopt_data).T
    df.to_csv(path_or_buf=final_analysis_out_csv_path, sep="\t")
    print("done!")
