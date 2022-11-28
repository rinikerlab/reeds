#!/usr/bin/python env
import glob
import os

import numpy as np

from pygromos.files import imd
from pygromos.utils import bash

from reeds.function_libs.analysis.sampling import undersampling_occurence_potential_threshold_distribution_based as find_undersampling_pot_tresh
from reeds.function_libs.analysis.sampling import detect_undersampling

from reeds.function_libs.visualization.utils import determine_vrange

import reeds.function_libs.analysis.sampling
import reeds.function_libs.visualization.pot_energy_plots
from reeds.function_libs.file_management import file_management as fM
import reeds.function_libs.utils.s_log_dist as s_log_dist

np.set_printoptions(suppress=True,formatter={'float_kind':'{:0.7f}'.format})
from reeds.data import ene_ana_libs

def do(out_analysis_dir: str, system_name: str,
       in_simulation_dir: str, in_topology_path: str, in_imd_path: str,
       undersampling_occurrence_fraction_threshold: float = 0.9,
       gromosPP_bin: str = None,
       in_ene_ana_lib: str = ene_ana_libs.ene_ana_lib_path,
       verbose: bool = True):
    """
        This analysis worker is used to analyse the exploration of the b_step: high s-undersampling boundary.

        Features:
            - test stability of simulations with starting coordinates
            - find turning point of transition-phase -> undersampling
            - determine potential thresholds for undersampling region.

    Parameters
    ----------
    out_analysis_dir : str
        output dir for analysis data-results
    system_name : str
        name of the system
    in_simulation_dir : str
        simulation dir path contains the gromos simulation data
    in_topology_path : str
        path to the gromos topology
    in_imd_path : str
        path to one gromos paramter file, that was used
    undersampling_pot_tresh : float, optional
        initial undersampling potential threshold (<-OLD)
    gromosPP_bin : str, optional
        path to the gromosPP binary folder
    in_ene_ana_lib : str, optional
        path to the ene_ana library
    verbose : bool, optional
        Sing, Sing, Sing, ....

    Returns
    -------

    """
    control_dict = {
        "cp_cnf": True,
        "cat_trc": False,
        "convert_trcs": False,
        "remove_gromosTRC": True,
        "cat_tre": False,
        "ene_ana": True,
        "cat_repdat": False,
        "pot_ene_by_replica": True,
        "pot_ene_by_state": True,
        "plot_pot_ene_timeseries": True,
        "plot_ref_timeseries": True,
        "plot_ref_distrib": True
        }

    if (verbose): print("out: ", out_analysis_dir)
    bash.make_folder(out_analysis_dir)

    # global vars
    out_prefix = system_name
    data_dir = out_analysis_dir + "/data"
    imd_file = imd.Imd(in_imd_path + "_1.imd")
    num_states = int(imd_file.EDS.NUMSTATES)
    
    # Read in all s-values that were used

    imd_files = sorted(glob.glob(in_imd_path + "*.imd"), key=lambda x: int(x.split("_")[-1].replace(".imd", "")))
    s_values = [float((imd.Imd(f)).EDS.S) for f in imd_files]
       
    eoffs = []
    # Get entries and convert to float
    for f in imd_files:
        eoff_state_values = imd.Imd(f).EDS.EIR
        for index, item in enumerate(eoff_state_values):
            eoff_state_values[index] = float(item)
        eoffs.append(eoff_state_values)

    # Count the number of simulations wich were succesful
    if (verbose): print("START file organization")
    if (os.path.exists(in_simulation_dir)):
        succsessful_sim_count = 0
        print("all_omds: ", glob.glob(in_simulation_dir + "/*.omd"))
        successfull_files = []
        for omd_file_path in sorted(glob.glob(in_simulation_dir + "/*.omd"),
                                    key=lambda x: int(x.split("_")[-1].replace(".omd", ""))):
            found_success = False
            for line in reversed(list(open(omd_file_path, "r"))):
                if "successfully" in line:
                    succsessful_sim_count += 1
                    found_success = True
                    successfull_files.append(omd_file_path)
                    break
            if (not found_success):
                print("Stop : ", succsessful_sim_count)
                break

        print("Successful sims: " + str(succsessful_sim_count), " out of ", len(s_values))
        print("Files: ", successfull_files)
        bash.make_folder(data_dir, additional_option="-p")

        # organize simulatoin Files:
        if (os.path.exists(in_simulation_dir)):
            fM.project_concatenation(in_folder=in_simulation_dir, in_topology_path=in_topology_path,
                                     additional_properties=["eR"] + ["e" + str(i) for i in range(1, num_states + 1)],
                                     in_imd=in_imd_path + "_1.imd", num_replicas=len(s_values[:succsessful_sim_count]),
                                     control_dict=control_dict, out_folder=data_dir, in_ene_ana_lib_path=in_ene_ana_lib,
                                     out_file_prefix=out_prefix, gromosPP_bin_dir=gromosPP_bin)

    elif (os.path.exists(data_dir) and os.path.exists(in_simulation_dir + ".tar.gz")):
        cnfs = glob.glob(data_dir + "/*.cnf")
        succsessful_sim_count = len(cnfs)

    else:
        raise IOError("could not find simulation dir or analysis dir!")

    if (verbose): print("START analysis")
    # do sampling_plot
    out_analysis_plot_dir = out_analysis_dir + "/plots"
    bash.make_folder(out_analysis_plot_dir, "-p")
    ene_trajs = fM.parse_csv_energy_trajectories(data_dir, out_prefix)  # gather potentials

    state_undersampling_pot_treshold = find_undersampling_pot_tresh(ene_traj_csvs=ene_trajs, sampling_fraction_treshold = undersampling_occurrence_fraction_threshold)

    sampling_analysis_results, out_plot_dirs = detect_undersampling(out_path = out_analysis_plot_dir,
                                                                    ene_trajs = ene_trajs,
                                                                    s_values = s_values[:succsessful_sim_count],
                                                                    state_potential_treshold=state_undersampling_pot_treshold, 
                                                                    eoffs = eoffs)
    
    v_range = determine_vrange(ene_trajs, num_states) 

    # Plotting the different potential energy distributions
    if control_dict["pot_ene_by_state"]:
        for i in range(num_states):
            outfile = out_analysis_plot_dir + '/' + system_name + '_pot_ene_state_' + str(i+1) + '.png'
            reeds.function_libs.visualization.pot_energy_plots.plot_energy_distribution_by_state(energy_trajs=ene_trajs, outfile=outfile, 
                                                                                                 state_num=i + 1, s_values=s_values, manual_xlim=v_range)
    
    if control_dict["pot_ene_by_replica"]:
        for i in range(len(ene_trajs)):
            outfile = out_analysis_plot_dir + '/' + system_name + '_pot_ene_replica_' + str(i+1) + '.png'
            reeds.function_libs.visualization.pot_energy_plots.plot_energy_distribution_by_replica(traj_data=ene_trajs[i], outfile_path=outfile, 
                                                                                                   replica_num=i + 1, s_value=s_values[i], manual_xlim=v_range)
    
    if control_dict["plot_ref_timeseries"]:
        outfile = out_analysis_plot_dir + '/' + system_name + '_ref_pot_ene_timeseries.png'
        reeds.function_libs.visualization.pot_energy_plots.plot_ref_pot_ene_timeseries(ene_trajs=ene_trajs, outfile=outfile, s_values=s_values)

    if control_dict["plot_ref_distrib"]:
        outfile = out_analysis_plot_dir + '/' + system_name + '_ref_pot_ene_distrib.png'
        reeds.function_libs.visualization.pot_energy_plots.plot_ref_pot_energy_distribution(energy_trajs=ene_trajs, outfile=outfile, s_values=s_values)

    # plot the potential energy timeseries as a grid:
    if control_dict["plot_pot_ene_timeseries"]:
        for i, ene_traj in enumerate(ene_trajs):
            out_path = out_analysis_plot_dir + '/' + system_name  + '_pot_ene_timeseries_' + str(i+1) + '.png'
            title = 'Lower Bound Analysis potential energy timeseries - s = ' + str(s_values[i])
            reeds.function_libs.visualization.pot_energy_plots.plot_sampling_grid(traj_data = ene_traj, y_range = v_range,
                                                                                  out_path = out_path, title = title)

    # Preparing input for the energy offset run
    if (verbose): print("Start next folder")
    out_analysis_next_dir = out_analysis_dir + "/next"
    bash.make_folder(out_analysis_next_dir, "-p")

    #print(sampling_analysis_results)
    u_idx = sampling_analysis_results["undersamplingThreshold"]
    
    # Make the new s-distribution based on this 
    print("undersampling found after replica: " + str(u_idx+1) + ' with s = ' + str(s_values[u_idx]))    
    print('New s distribution will place ' + str(num_states) + ' replicas between  s = ' + str(s_values[u_idx]) + ' and s = ' +str(s_values[u_idx+2]))
 
    new_sdist = s_values[:u_idx]
    lower_sdist = s_log_dist.get_log_s_distribution_between(s_values[u_idx], s_values[u_idx+2], num_states)
    new_sdist.extend(lower_sdist)  
    
    print ('new s-dist: ' + str(new_sdist))

    # Write the s-values to a csv file
    out_file = open(out_analysis_next_dir + "/s_vals.csv", "w")
    out_file.write("\t".join(list(map(str, new_sdist))))
    out_file.write("\n")
    out_file.close()

    # Write the potential energy thresholds to a csv file
    out_file = open(out_analysis_next_dir + "/state_occurence_pot_thresh.csv", "w")
    out_file.write("\t".join(map(str, np.round(sampling_analysis_results["potentialThreshold"], 2))))
    out_file.write("\n")
    out_file.close()

    # compress out_trc/out_tre Files & simulation dir
    trx_files = glob.glob(data_dir + "/*.tr?")
    for trx in trx_files:
        bash.compress_gzip(in_path=trx)

    if (not os.path.exists(in_simulation_dir + ".tar.gz") and os.path.exists(in_simulation_dir)):
        tar_sim_dir = bash.compress_tar(in_path=in_simulation_dir, gunzip_compression=True, )
        bash.wait_for_fileSystem(tar_sim_dir)
        bash.remove_file(in_simulation_dir, additional_options="-r")
