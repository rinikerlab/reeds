#!/usr/bin/python env
import glob
import os
import numpy as np
import warnings

from pygromos.files import imd
from pygromos.utils import bash

import reeds.function_libs.visualization.pot_energy_plots
from reeds.data import ene_ana_libs
from reeds.function_libs.file_management import file_management as fM
from reeds.function_libs.analysis import sampling

from reeds.function_libs.visualization.utils import determine_vrange

from pygromos.gromos import gromosPP


def find_lowest_energy_frames(opt_trajs, f_ene, f_cnf, sim_steps):
    """
    This function will find the lowest energies in the 2nd half of the trajectory
    for which we have a conformation that was printed out. This will allow us 
    to have better SSM starting conformations which really match the minima of 
    the end states.

    -------
    Arguments
    -------
    opt_trajs: List pd.DataFrame
        optimized states pandas dataframe with pot energies
    f_ene: int
        number of steps between printout of the energies
    f_cnf: int 
        number of steps between printout of the conformations
    sim_steps: int
        total number of simulation steps 
    
    -------
    Returns
    -------
    lowest_frames: List [int]
        list of the lowest frames for each end state to frameout
    """
    # Start from second half of the optimized states simulation.
    num_states = len(opt_trajs)
    
    start =  int(sim_steps/ (2 * f_ene)) # starts at half the sim.
    end = int(sim_steps/f_ene)
    
    lowest_frames = []
    
    # indices to check = we have both a cnf and an energy here
    idx = np.arange(start, end, f_cnf/f_ene)    
    for i in range(num_states):
        idx_min = (opt_trajs[i]['e' + str(i+1)][idx].idxmin())
        print ('Found a mimimum for state ' + str(i+1) + ' at:')
        print (str(opt_trajs[i]['time'][idx_min]) + 'ps, with ' \
               +str(round(opt_trajs[i]['e' + str(i+1)][idx_min],2)) + ' kJ/mol \n')
        # +1 required
        lowest_frames.append(int(idx_min*f_ene/f_cnf +1))
        
    return lowest_frames


def do(in_simulation_dir: str, in_topology_path: str, in_imd_path: str,
       out_analysis_dir: str,
       vacuum_simulation:bool=False,
       gromosPP_bin: str = None,
       in_ene_ana_lib: str = ene_ana_libs.ene_ana_lib_path,
       verbose: bool = True):
    """
        This analysis_worker is analyzing step a_state optimization.

        Features:
        - concatenation an d compressoin of simulated data
        - analysis of potential energies

    Parameters
    ----------
    in_simulation_dir : str
        path to gromos simulation folder.
    in_topology_path
        path to gromos topology
    in_imd_path
        path to gromos parameter
    out_analysis_dir
        output analysis path
    pot_tresh
        potential threshold for undersampling
    gromosPP_bin
        path to gromos binary dir
    in_ene_ana_lib
        path to ene_ana library
    verbose
        DummDiDummDiDu

    Returns
    -------

    """
    control_dict = {
        "fileManagment": {
            "cp_cnf": True,
            "cat_trc": True, 
            "convert_trcs": False,
            "ene_ana": True,
            "cat_tre": False,
        },
        "plots": True,
        "compress": False,
        "plot_timeseries": True,
        "plot_grid_timeseries": True,
        "plot_ref_timeseries": True,
        "plot_ref_distrib": True
    }

    if (verbose): print("Start analysis of the optimized state generation:")
    # generate
    bash.make_folder(out_analysis_dir, "-p")

    # gen e dat
    print(in_imd_path)
    imd_file = imd.Imd(in_imd_path + "_1.imd")
    numstates = int(imd_file.EDS.NUMSTATES)
    s_values = [1.0 for x in range(1, numstates + 1)]
    
    # Read in all eoffs
    imd_files = sorted(glob.glob(in_imd_path + "*.imd"), key=lambda x: int(x.split("_")[-1].replace(".imd", "")))
    eoffs = []
    # Get entries and convert to float
    for f in imd_files:
        eoff_state_values = imd.Imd(f).EDS.EIR
        for index, item in enumerate(eoff_state_values):
            eoff_state_values[index] = float(item)
        eoffs.append(eoff_state_values)

    # organize siulation Files:
    coord_dir = out_analysis_dir + "/data"
    out_prefix = "REEDS_state_optimisation"
    bash.make_folder(coord_dir)

    # if we're using Stochastic Dynamics, use solutemp2 for ene_ana instead of solvtemp2
    if (imd_file.MULTIBATH is None) and (imd_file.STOCHDYN is not None):
        additional_properties = ("solutemp2", "totdisres")
        boundary_conditions = "v"
    else:
        additional_properties = ("solvtemp2", "totdisres")
        boundary_conditions = "r"

    fM.project_concatenation(in_folder=in_simulation_dir, in_topology_path=in_topology_path,
                             additional_properties=["eR"] + ["e" + str(i) for i in range(1, numstates + 1)],
                             in_imd=in_imd_path + "_1.imd", num_replicas=numstates,
                             control_dict=control_dict["fileManagment"], gromosPP_bin_dir=gromosPP_bin,
                             out_folder=coord_dir, in_ene_ana_lib_path=in_ene_ana_lib,
                             out_file_prefix=out_prefix, boundary_conditions=boundary_conditions)

    ene_trajs = fM.parse_csv_energy_trajectories(in_folder=coord_dir, ene_trajs_prefix=out_prefix + "_energies")

    # do sampling_plot
    out_analysis_plot_dir = out_analysis_dir + "/plots"
    bash.make_folder(out_analysis_plot_dir, "-p")

    ## write pot_treshholds to next
    physical_state_occurrence_treshold = sampling.get_all_physical_occurence_potential_threshold_distribution_based(ene_trajs, _vacuum_simulation=vacuum_simulation)

    sampling.sampling_analysis(out_path=out_analysis_plot_dir, ene_traj_csvs=ene_trajs, s_values=s_values,
                                state_potential_treshold=physical_state_occurrence_treshold, eoffs=eoffs)

    # Plot of all of the potential energy distributions in a single plot:
    reeds.function_libs.visualization.pot_energy_plots.plot_optimized_states_potential_energies(outfile=out_analysis_plot_dir + "/optimized_states_potential_energies.png",
                                                                                                ene_trajs=ene_trajs)

    if control_dict["plot_ref_timeseries"]:
        outfile = out_analysis_plot_dir + "/optimized_states_ref_potential_ene_timeseries.png"
        reeds.function_libs.visualization.pot_energy_plots.plot_ref_pot_ene_timeseries(ene_trajs, outfile, s_values=[], optimized_state = True)

    if control_dict["plot_ref_distrib"]:
        outfile = out_analysis_plot_dir + "/optimized_states_ref_potential_ene_distrib.png"
        reeds.function_libs.visualization.pot_energy_plots.plot_ref_pot_energy_distribution(ene_trajs, outfile, s_values=[], optimized_state = True)

    # do visualisation pot energies

    num_states = len(ene_trajs)
    
    v_range = determine_vrange(ene_trajs, num_states)

    for i, ene_traj in enumerate(ene_trajs):
        singleStates = ['e' + str(i) for i in range(1, num_states+1)]
        
        if control_dict["plot_timeseries"]:
            reeds.function_libs.visualization.pot_energy_plots.plot_potential_timeseries(time = ene_traj.time, potentials = ene_traj[singleStates],
                                                                                         y_range = v_range, title = "EDS_stateV_scatter",
                                                                                         out_path = out_analysis_plot_dir + "/edsState_potential_timeseries_"
                                                     + str(ene_traj.s) + ".png")
        if control_dict["plot_grid_timeseries"]:
            out_path = out_analysis_plot_dir + "/edsState_potential_timeseries_stageGrid_" + str(ene_traj.s) + ".png"
            title = 'Optimized State potential energy timeseries - System biased to state ' + str(i+1)
            reeds.function_libs.visualization.pot_energy_plots.plot_sampling_grid(traj_data = ene_traj, y_range = v_range,
                                                                                  out_path = out_path, title = title)


    # generate
    next_dir = bash.make_folder(out_analysis_dir+"/next",)

    # find the lowest energy conformation to use as SSM input 
    # for the rest of the pipeline
    
    lowest_frames = find_lowest_energy_frames(opt_trajs = ene_trajs, 
                                              f_ene = int(imd_file.WRITETRAJ.NTWE), 
                                              f_cnf = int(imd_file.WRITETRAJ.NTWX), 
                                              sim_steps = int(imd_file.STEP.NSTLIM))
    
    print ('lowest frames to print out: ' + str(lowest_frames))

    trcs = sorted(glob.glob(coord_dir + '/*.trc.gz'), key = lambda x: int(x.split('_')[-1].replace('.trc.gz', '')))
    gromPP = gromosPP.GromosPP()
    
    # frameout
    for i in range(num_states):
        gromPP.frameout(in_top_path = in_topology_path,
                        in_coord_path = trcs[i],
                        periodic_boundary_condition = " r cog",
                        out_file_path = coord_dir + "/REEDS_SSM_state_" + str(i+1) +'.cnf',
                        out_file_format = 'cnf',
                        include = 'ALL',
                        frames = lowest_frames[i]
                       )
     
    
    bash.copy_file(coord_dir + "/REEDS_SSM_state*.cnf", next_dir)


    ##write_pot_tresh:
    out_file = open(next_dir + "/state_occurence_physical_pot_thresh.csv", "w")
    out_file.write("\t".join(map(str, np.round(physical_state_occurrence_treshold, 2))))
    out_file.close()

    

    # compress out_trc/out_tre Files
    if (control_dict["compress"]):
        trx_files = glob.glob(coord_dir + "/*.tr?")
        for trx in trx_files:
            bash.compress_tar(in_path=trx, gunzip_compression=True)

        if (not os.path.exists(in_simulation_dir + ".tar.gz") and os.path.exists(in_simulation_dir)):
            tar_sim_dir = bash.compress_tar(in_path=in_simulation_dir, gunzip_compression=True)
            bash.wait_for_fileSystem(tar_sim_dir)
            bash.remove_file(in_simulation_dir, additional_options="-r")

    print ('\n\nAnalysis of the Optimized States completed successfully !')

if __name__ == "__main__":
    base_dir = "/home/bschroed/Projects/REEDS/systems/BRD4/water/2_ligands/test_ana_optstr"
    sim_dir = base_dir + "/simulation"
    gromos_path = "/home/bschroed/gromos/reeds_gromos_bschroed_AmberFF_CG/gromosPP/bin"
    ene_ana_lib = "/home/bschroed/code/reeds/gromos_files/ene_ana_libs/ene_ana_REEDS_7state.md++.lib"
    tre_prefix = "2_ligs_4MR3_3U5L_work"
    ana_dir = base_dir + "/analysis"

    do(out_analysis_dir=base_dir + "/analysis", in_simulation_dir=sim_dir, in_ene_ana_lib=ene_ana_lib,
       gromosPP_bin=gromos_path)
