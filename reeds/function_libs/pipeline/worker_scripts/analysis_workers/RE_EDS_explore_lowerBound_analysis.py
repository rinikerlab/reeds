#!/usr/bin/python env
import glob
import os

import numpy as np

from pygromos.files import imd
from pygromos.utils import bash

import reeds.function_libs.analysis.sampling
import reeds.function_libs.visualization.pot_energy_plots
from reeds.function_libs.file_management import file_management as fM

np.set_printoptions(suppress=True)
from reeds.data import ene_ana_libs


def do(out_analysis_dir: str, system_name: str,
       in_simulation_dir: str, in_topology_path: str, in_imd_path: str,
       undersampling_pot_tresh: float = 200,
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
        "pot_ene_by_replica":False,
        "pot_ene_by_state":True,
        "plot_pot_ene_timeseries":True,
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

    # successful_sim_count sucessful Runs!:
    if (verbose): print("START file organization")
    if (os.path.exists(in_simulation_dir)):
        succsessful_sim_count = 0
        print("all_omds: ", glob.glob(in_simulation_dir + "/*.omd"))
        successfull_files = []
        for omd_file_path in sorted(glob.glob(in_simulation_dir + "/*.omd"),
                                    key=lambda x: int(x.split("_")[-1].replace(".omd", ""))):
            found_success = False
            print(omd_file_path)

            for line in open(omd_file_path, "r"):
                if "successfully" in line:
                    succsessful_sim_count += 1
                    found_success = True
                    successfull_files.append(omd_file_path)
                    break
            if (not found_success):
                print("Stop : ", succsessful_sim_count)
                break

        print("Successful Sims: " + str(succsessful_sim_count), " of ", len(s_values))
        print("Files: ", successfull_files)
        bash.make_folder(data_dir, additional_option="-p")

        # organize simulatoin Files:
        if (os.path.exists(in_simulation_dir)):
            fM.project_concatenation(in_folder=in_simulation_dir, in_topology_path=in_topology_path,
                                     additional_properties=["eR"] + ["e" + str(i) for i in range(1, num_states + 1)],
                                     in_imd=in_imd_path + "_1.imd", num_replicas=len(s_values[:succsessful_sim_count]),
                                     control_dict=control_dict, out_folder=data_dir, in_ene_ana_lib_path=in_ene_ana_lib,
                                     out_file_prefix=out_prefix, nofinal=True, gromosPP_bin_dir=gromosPP_bin)

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
    sampling_analysis_results, out_plot_dirs = reeds.function_libs.analysis.sampling.sampling_analysis(out_path = out_analysis_plot_dir,
                                                                                                       ene_traj_csvs = ene_trajs,
                                                                                                       s_values = s_values[:succsessful_sim_count],
                                                                                                       pot_tresh = undersampling_pot_tresh)

    # Plotting the different potential energy distributions
    if control_dict["pot_ene_by_state"]:
        for i in range(num_states):
            outfile = out_analysis_plot_dir + '/' + system_name + '_pot_ene_state_' + str(i+1) + '.png'
            reeds.function_libs.visualization.pot_energy_plots.plot_energy_distribution_by_state(ene_trajs, outfile, i + 1, s_values)
    
    if control_dict["pot_ene_by_replica"]:
        for i in range(len(ene_trajs)):
            outfile = out_analysis_plot_dir + '/' + system_name + '_pot_ene_replica_' + str(i+1) + '.png'
            reeds.function_libs.visualization.pot_energy_plots.plot_energy_distribution_by_replica(ene_trajs[i], outfile, i + 1, s_values[i])
    
    if control_dict["plot_ref_timeseries"]:
        outfile = out_analysis_plot_dir + '/' + system_name + '_ref_pot_ene_timeseries.png'
        reeds.function_libs.visualization.pot_energy_plots.plot_ref_pot_ene_timeseries(ene_trajs, outfile, s_values)

    if control_dict["plot_ref_distrib"]:
        outfile = out_analysis_plot_dir + '/' + system_name + '_ref_pot_ene_distrib.png'
        reeds.function_libs.visualization.pot_energy_plots.plot_ref_pot_energy_distribution(ene_trajs, outfile, s_values)

    # plot the potential energy timeseries as a grid:
    if control_dict["plot_pot_ene_timeseries"]:
        for i, ene_traj in enumerate(ene_trajs):
            out_path = out_analysis_plot_dir + '/' + system_name  + '_pot_ene_timeseries_' + str(i+1) + '.png'
            title = 'Lower Bound Analysis potential energy timeseries - s = ' + str(s_values[i])
            reeds.function_libs.visualization.pot_energy_plots.plot_sampling_grid(traj_data = ene_traj, y_range = (-1000, 1000),
                                                                                  out_path = out_path, title = title)



    # NExt Folder
    if (verbose): print("START next folder")
    out_analysis_next_dir = out_analysis_dir + "/next"
    bash.make_folder(out_analysis_next_dir, "-p")

    ##Coord Files
    cnfs = sorted(glob.glob(data_dir + "/*.cnf"), key=lambda x: int(x.split("_")[-1].replace(".cnf", "")))
    successfull_cnf = []
    for cnf in cnfs:
        check_cnf = cnf.split("_")
        if (int(check_cnf[len(check_cnf) - 1].replace(".cnf", "")) <= succsessful_sim_count):
            successfull_cnf.append(cnf)
    if (verbose): print("succesful_cnfs: \n" + "\n".join(successfull_cnf))

    undersampling_limit = sampling_analysis_results[
                              "undersamlingThreshold"] + 2  # conservative lower limit for undersampling +2

    print("successful sims found : ", len(successfull_cnf))
    print("undersampling found after replica: ", undersampling_limit)
    for cnf in successfull_cnf[:undersampling_limit + 1]:
        bash.copy_file(cnf, out_analysis_next_dir + "/" + os.path.basename(cnf))

    ##write_s
    out_file = open(out_analysis_next_dir + "/s_vals.csv", "w")
    out_file.write("\t".join(list(map(str, s_values))[:undersampling_limit + 1]))
    out_file.close()

    ##write_pot_tresh:
    out_file = open(out_analysis_next_dir + "/state_occurence_pot_thresh.csv", "w")
    out_file.write("\t".join(map(str, sampling_analysis_results["potentialThreshold"])))
    out_file.close()

    # compress out_trc/out_tre Files & simulation dir
    trx_files = glob.glob(data_dir + "/*.tr?")
    for trx in trx_files:
        bash.compress_gzip(in_path=trx)

    if (not os.path.exists(in_simulation_dir + ".tar.gz") and os.path.exists(in_simulation_dir)):
        tar_sim_dir = bash.compress_tar(in_path=in_simulation_dir, gunzip_compression=True, )
        bash.wait_for_fileSystem(tar_sim_dir)
        bash.remove_file(in_simulation_dir, additional_options="-r")
