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


def analyse_sopt_iteration(repdat_path: str, out_dir: str, title: str, pot_tresh=0) -> dict:
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
    states = repdat_file.DATA.state_potentials.iloc[0].keys()
    occurrence_counts = {state: 0 for state in states}
    domination_counts = {state: 0 for state in states}
    print("cols: ", repdat_file.DATA.columns)
    all_pos = list(sorted(np.unique(repdat_file.DATA.ID)))
    min_pos, max_pos = (all_pos[repOff], all_pos[-1])
    print("extremePos: ", min_pos, max_pos)
    replica1 = repdat_file.DATA.loc[repdat_file.DATA.ID == 1]


    if (isinstance(pot_tresh, float)):
        pot_tresh = {x:pot_tresh for x in replica1.iloc[0].state_potentials}
    print("potTresh", pot_tresh)

    for rowID, row in replica1.iterrows():
        state_pots = row.state_potentials

        for ind, state in enumerate(state_pots):
            if (state_pots[state] < pot_tresh[ind]):
                occurrence_counts[state] += 1

        id_ene = {val: key for key, val in state_pots.items()}
        min_state = id_ene[min(id_ene)]
        domination_counts[min_state] +=1

    sopt_it.update({"state_occurence_sampling": occurrence_counts})
    sopt_it.update({"state_domination_sampling": domination_counts})

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
                                                      min_pos=min_pos, max_pos=max_pos)
    sopt_it.update({"stats_per_replica": stats})
    sopt_it.update({"s_values": s_values[repOff:]})

    sopt_it.update({"state_domination_sampling": domination_counts})


    sorted_dominating_state_samping =   np.array([sopt_it["state_domination_sampling"][x] for x in sorted( sopt_it["state_domination_sampling"], key=lambda x: int(x.replace("Vr","")) )])
    print(sorted_dominating_state_samping)

    optimal_samp =1/len(sorted_dominating_state_samping)*100
    iteration_norm = np.round(sorted_dominating_state_samping / np.sum(sorted_dominating_state_samping) * 100, 2)
    optimal_samp_dev = np.abs(iteration_norm - optimal_samp)

    MAE = np.round(np.mean(optimal_samp_dev), 2)
    MAE_std = np.std(optimal_samp_dev)

    sopt_it.update({"sampling_distribution_optimal_deviation_s1": {"MAE": MAE, "MAE_std": MAE_std}})
    sopt_it.update({"s_values": s_values[repOff:]})



    nReplicasRoundtrips, avg_numberOfRoundtrips, avg_roundtrips = get_s_optimization_roundtrip_averages(stats)
    print("replicas doing at least one roundtrip:", nReplicasRoundtrips)
    print("avg. number of roundtrips: ", avg_numberOfRoundtrips)
    print("avg. roundtrip durations:", avg_roundtrips)
    sopt_it.update({"nRoundTrips": nReplicasRoundtrips,
                    "avg_nRoundtrips": avg_numberOfRoundtrips,
                    "avg_rountrip_durations": avg_roundtrips})

    del trans_dict
    return sopt_it


def do(sopt_root_dir: str, pot_tresh:Union[List, float]=0, title="", out_dir: str = None, rt_convergence=100):
    """
        This function does the final analysis of an s-optimization. It analyses the outcome of the full s-optimization iterations.
        Features:
            - generate s-optimization summary
            - average roundtrip time improvement efficiency
            - deviation of sampling occurence (normalized) from the optimal sampling distribution.

        The analysis of the individual s-optimization steps is stored and can be used at later executions. (if you want to analyse form scratch, keep in mind to delete the generated .npy files.)

    Parameters
    ----------
    sopt_root_dir : str
        directory containing all s-optimization iterations
    pot_tresh : Union[float, List[float]], optional
        potential energy threshold, determining undersampling (default: 0)
    title : str, optional
        title of run (default: "")
    out_dir : str, optional
        ananlysis output dir, default uses the same dir like root_dir (default: None)
    rt_convergence : int, optional
        roundtrip time convergence criterium  in ps(default: 10ps)

    """
    sopt_dirs = [x for x in os.listdir(sopt_root_dir) if ("sopt" in x and os.path.isdir(sopt_root_dir+"/"+x))]

    # sopt out_dir
    if (isinstance(out_dir, type(None))):
        out_dir = sopt_root_dir + "/analysis"

    out_dir = bash.make_folder(out_dir)

    # analyse the individual iterations
    sopt_data = {}
    repdat_files = {}
    converged = False
    print(sopt_dirs)
    for iteration_folder in sorted(sopt_dirs):
        iteration = int(iteration_folder.replace("sopt", ""))
        print( iteration, end="\t")
        repdat = glob.glob(sopt_root_dir + "/" + iteration_folder + "/analysis/data/*repdat*")
        out_iteration_file_path = out_dir + "/" + iteration_folder + "_ana_data.npy"

        if (os.path.exists(out_iteration_file_path)):
            print("\n\nLoading precomputed data for iteration: ", end="\t")
            sopt_it_stats = pickle.load(open(out_iteration_file_path, "rb"))
            sopt_data.update({iteration_folder: sopt_it_stats})

        elif (len(repdat) == 1):
            print("\nCalculate statistics for iteration: ", iteration)
            repdat_files.update({iteration: repdat[0]})
            sopt_it_stats = analyse_sopt_iteration(repdat_path=repdat_files[iteration], out_dir=out_dir,
                                                   title="s-opt " + str(iteration), pot_tresh=pot_tresh)

            pickle.dump(obj=sopt_it_stats, file=open(out_iteration_file_path, "wb"))
            sopt_data.update({iteration_folder: sopt_it_stats})
        else:
            continue
        # round trip time efficiency
        if (iteration > 1):
            sopt_it_stats.update({"avg_rountrip_duration_optimization_efficiency": sopt_data["sopt"+str(iteration- 1)]["avg_rountrip_durations"] -sopt_it_stats["avg_rountrip_durations"]})

        #assign convergence in an conserative fasion.
        sopt_it_stats.update({"converged": converged})
        if(iteration > 1 and sopt_it_stats["avg_rountrip_duration_optimization_efficiency"] <  rt_convergence ):
            converged=True


    print(sopt_data)

    #overview
    visualization_s_optimization_summary(s_opt_data=sopt_data, out_path=out_dir + "/" + title + "_sopt_analysis.png")

    #Optimization  convergence:
    visualize_s_optimisation_convergence(s_opt_data=sopt_data, out_path=out_dir + "/" + title + "_sopt_efficiency.png", convergens_radius=rt_convergence)

    #Optimization Distribution
    visualize_s_optimisation_sampling_optimization(s_opt_data=sopt_data, out_path=out_dir + "/" + title + "_sopt_sampling_optimumDev.png")

    #Final storing
    ##serialization
    final_analysis_out_path = out_dir + "/"+title+"_final_soptimization_analysis.obj"
    pickle.dump(sopt_it_stats, open(final_analysis_out_path, "wb"))

    ##Human readable:
    final_analysis_out_csv_path = out_dir + "/"+title+"_final_soptimization_analysis.csv"
    df = pd.DataFrame(sopt_data).T
    df.to_csv(path_or_buf=final_analysis_out_csv_path, sep="\t")

