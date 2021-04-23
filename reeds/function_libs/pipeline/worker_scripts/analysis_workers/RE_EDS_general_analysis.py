import os, glob, warnings
import numpy as np

from collections import OrderedDict
from typing import Union, Dict, List

from pygromos.files import imd, repdat
from pygromos.utils import bash

import reeds.function_libs.analysis.free_energy
import reeds.function_libs.analysis.parameter_optimization
import reeds.function_libs.analysis.sampling as sampling_ana
import reeds.function_libs.optimization.eds_energy_offsets as eds_energy_offsets

import reeds.function_libs.visualization.pot_energy_plots
import reeds.function_libs.visualization.re_plots

from reeds.function_libs.file_management import file_management
from reeds.function_libs.file_management.file_management import parse_csv_energy_trajectories
from reeds.function_libs.utils import s_log_dist as sdist
from reeds.function_libs.utils.structures import adding_Scheme_new_Replicas

template_control_dict = OrderedDict({  # this dictionary is controlling the post  Simulation analysis procedure!
    "concat": {"do": True,
               "sub": {
                   "cp_cnf": True,
                   "cat_trc": True,
                   "cat_tre": False,
                   "ene_ana": True,
                   "convert_trcs": False,
                   "cat_repdat": True, }
               },
    "plot_property_timeseries": {"do": True,
                                 "sub": {
                                    "pot_ene_by_state":True,
                                    "pot_ene_by_replica":False,
                                    "pot_ene_timeseries": False,
                                    "pot_ene_grid_timeseries": True,
                                    "ref_timeseries": True,
                                    "ref_distrib": False,
                                    "distance_restraints": False,
                                    "temperature_2d_plot": False
                                  }
                                 },
    "eoffset": {"do": True,
                "sub": {
                    "calc_eoff": True,
                    "sampling_plot": True, }
                },
    "sopt": {"do": True,
             "sub": {
                 "detect_flow_equilib": True,
                 "run_RTO": True,
                 "run_NLRTO": True,
                 "run_NGRTO": False,
                 "visualize_transitions": True,
                 "roundtrips": True,
                 "generate_replica trace": True}
             },
    "dfmult": {"do": False},
    "compress_simulation_folder": {"do": True},
    "prepare_input_folder": {"do": True,
                             "sub": {
                                 "eoff_to_sopt": False,
                                 "write_eoff": False,
                                 "write_s": True
                             },
                             }
})


def dict_to_nice_string(control_dict: Dict) -> str:
    """
        Converts a dictionary of options (like template_control_dict)
	  to a more human readable format. Which can then be printed to a text file,
	  which can be manually modified before submiting analysis jobs.

    Parameters
    ----------
    control_dict : Dict
        analysis control dictonary

    Returns
    -------
    str
        nice formatting of the control dictionary for printing.

    """
    script_text = "control_dict = {\n"
    for key, value in control_dict.items():
        script_text += "\t\"" + key + "\": "
        first = False
        if (type(value) == dict):
            if ("do" in value):  # do should always be first in this list
                script_text += "{\"do\":" + str(value["do"]) + ","
                if (len(value) > 1):
                    script_text += "\n"
                first = True
            for key2, value2 in value.items():  # alternative keys

                # prefix
                if (first):
                    prefix = " "
                    first = False
                else:
                    prefix = "\t\t"

                # key_val
                if (key2 == "do"):
                    continue
                elif (type(value2) == dict):
                    script_text += prefix + "\"" + str(key2) + "\": " + _inline_dict(value2, "\t\t\t") + ",\n"
                else:
                    script_text += prefix + "\"" + str(key2) + "\": " + str(value2) + ","
            script_text += prefix + " },\n"
        else:
            script_text += str(value) + ",\n"
    script_text += "}\n"
    return script_text


def _inline_dict(in_dict: Dict, prefix: str = "\t"):
    """
        translate dictionary to one code line. can be used for meta-scripting

    Parameters
    ----------
    in_dict: Dict
        analysis control dict
    prefix : str, optional
        prfix symbol to dict write out.

    Returns
    -------
    str
        code line.

    """
    msg = "{\n"
    for key, value in in_dict.items():
        if (type(value) == dict):
            msg += prefix + "\"" + str(key) + "\": " + _inline_dict(in_dict=value, prefix=prefix + "\t") + ","
        else:
            msg += prefix + "\"" + str(key) + "\": " + str(value) + ",\n"
    return msg + prefix + "}"


def check_script_control(control_dict: dict = None) -> dict:
    if isinstance(control_dict, type(None)):
        return template_control_dict
    else:
        for x in template_control_dict:
            if x not in control_dict:
                control_dict.update({x: template_control_dict[x]})
    return control_dict

def do_Reeds_analysis(in_folder: str, out_folder: str, gromos_path: str,
                      topology: str, in_ene_ana_lib: str, in_imd: str,
                      optimized_eds_state_folder: str = "../a_optimizedState/analysis/data",
                      state_undersampling_pot_tresh: List[float] = None,
                      state_physcial_pot_tresh: List[float] = None,
                      undersampling_frac_thresh: float = 0.9,
                      add_s_vals: int = 0, state_weights: List[float]=None, s_opt_trial_range:int=None,
                      adding_new_sReplicas_Scheme: adding_Scheme_new_Replicas = adding_Scheme_new_Replicas.from_below,
                      grom_file_prefix: str = "test", title_prefix: str = "test", ene_ana_prefix="ey_sx.dat",
                      repdat_prefix: str = "run_repdat.dat",
                      n_processors: int = 1, verbose=False, dfmult_all_replicas=False,
                      control_dict: Dict[str, Union[bool, Dict[str, bool]]] = None) -> (
        dict, dict, dict):
    """
         Master calling point from which all jobs can call the analysis functions for a RE-EDS simulation.
	  This function generates: plots, compress files, and/or calculate values of interest.


    Parameters
    ----------
    in_folder : str
        input folder for the simulation.
    out_folder : str
        output folder for the simulation
    gromos_path : str
        gromosPP binary path
    topology : str
        path to topology
    in_ene_ana_lib : str
        in path for ene_ana lib
    in_imd : str
        in path for imd_file
    optimized_eds_state_folder : str, optional
        path to optimized eds_state folders (default: "../a_optimizedState/analysis/data")
    pot_tresh : float, optional
        potential energy treshold (default: 0)
    undersampling_frac_thresh : int, optional
        fraction threshold (default: 0.6)
    take_last_n : int, optional
        this parameter can be used to force the energy offset estimation to use a certain amount of replicas.  (default: None)
    add_s_vals : int, optional
        this parameter can be used to add a number of s-values  during the s-optimization (default: 0)
    state_weights : List[float], optional
        allows to weight the different states in the s-optimization differently (default: None)
    s_opt_trial_range : int, optional
        give a range of trials, that define the start and end of the s-optimization run (default: adding_Scheme_new_Replicas.from_below)
    adding_new_sReplicas_Scheme : int, optional
        how shall the coordinates for new replicas be added to an exchange bottle-neck. (default: adding_Scheme_new_Replicas.from_below)
    grom_file_prefix : str, optional
         provide here a gromos_file prefix of this run (default: test)
    title_prefix : str, optional
        proivde here a output_prefx and plot prefix (default: test)
    ene_ana_prefix : str, optional
        prefix for the ene ana analysis @WARNING: NOT USED ANYMORE! - FUTURE REMOVE!.
    repdat_prefix : str, optional
        prefix for the repdat files. required to read in the repdats. (default:run_repdat.dat )
    n_processors : int, optional
        number of processors
    verbose : bool, optional
        verbosity level
    dfmult_all_replicas : bool, optional
        shall dfmult be calculated for all replicas
    control_dict : dict, optional
        control dict for analysis

    Returns
    -------
    (dict, dict, dict)
        eoff_statistic, svals, dFs - the function returns the eoff_statistics,
        the s-values of the s-optimization-results and the free energy calculation results,
        if calculated.

    """

    eoff_statistic = {}
    svals = {}
    dFs = {}

    print("Starting RE-EDS analysis:")

    # subfolder for clearer structure
    plot_folder_path = out_folder + "/plots"
    concat_file_folder = bash.make_folder(out_folder + "/data", "-p")

    if (not os.path.exists(out_folder)):
        print("Generating out_folder: ", out_folder)
        bash.make_folder(out_folder)
    if (not os.path.exists(concat_file_folder)):
        bash.make_folder(concat_file_folder)

    # out_files
    repdat_file_out_path = concat_file_folder + "/" + title_prefix + "_" + repdat_prefix
    ene_trajs_prefix = title_prefix + "_energies"

    # manual script control
    control_dict = check_script_control(control_dict)

    # parameter file: <-not needed!
    # if(verbose): print("Reading imd: "+in_imd)
    imd_file = imd.Imd(in_imd)
    s_values = list(map(float, imd_file.REPLICA_EDS.RES))
    Eoff = np.array(list(map(lambda vec: list(map(float, vec)), imd_file.REPLICA_EDS.EIR))).T
    num_states = int(imd_file.REPLICA_EDS.NUMSTATES)

    try:
        if (not isinstance(imd_file.MULTIBATH, type(None))):
            temp = float(imd_file.MULTIBATH.TEMP0[0])
        elif (not isinstance(imd_file.STOCHDYN, type(None))):
            temp = float(imd_file.STOCHDYN.TEMPSD)
        else:
            raise Exception("Either STOCHDYN or MULTIBATH block needs to be defined in imd.")

    except Exception as err:
        print("Failed during analysis\n\t" + "\n\t".join(map(str, err.args)))
        exit(1)

    if (control_dict["concat"]["do"]):
        print("STARTING CONCATENATION.")
        num_replicas = len(s_values)

        # if we're using Stochastic Dynamics, use solutemp2 for ene_ana instead of solvtemp2
        if (isinstance(imd_file.MULTIBATH, type(None)) and not isinstance(imd_file.STOCHDYN, type(None))):
            additional_properties = ("solutemp2", "totdisres")
            boundary_conditions = "v cog"

        # if there's only one bath, use solutemp2 for ene_ana instead of solvtemp2
        elif (not isinstance(imd_file.MULTIBATH, type(None)) and imd_file.MULTIBATH.NBATHS == "1"):
            additional_properties = ("solutemp2", "totdisres")
            boundary_conditions = "r cog"

        else:
            additional_properties = ("solvtemp2", "totdisres")
            boundary_conditions = "r cog"

        out_files = file_management.reeds_project_concatenation(in_folder=in_folder, in_topology_path=topology,
                                                                in_imd=in_imd, num_replicas=num_replicas,
                                                                control_dict=control_dict["concat"]["sub"],
                                                                out_folder=concat_file_folder,
                                                                in_ene_ana_lib_path=in_ene_ana_lib,
                                                                repdat_file_out_path=repdat_file_out_path,
                                                                out_file_prefix=grom_file_prefix, starting_time=0,
                                                                n_processes=n_processors, gromosPP_bin_dir=gromos_path,
                                                                verbose=False,
                                                                additional_properties=additional_properties,
                                                                boundary_conditions=boundary_conditions)
        if (verbose): print("Done\n")

    # intermezzo generating plots_folder
    if (not os.path.exists(plot_folder_path)):
        plot_folder_path = bash.make_folder(plot_folder_path)

    # Set this to None as a checker to avoid redundant parsing
    energy_trajectories = None

    if (control_dict["plot_property_timeseries"]["do"]):
        sub_control = control_dict["plot_property_timeseries"]["sub"]

        if (verbose): print("\tParse the data:\n")
        
        # No need to check if trajectories are parsed here, as it is the first access point. 
        energy_trajectories = parse_csv_energy_trajectories(concat_file_folder, ene_trajs_prefix)
        
        # Plots related to the potential energy distributions of the end states.

        if sub_control["pot_ene_by_state"]:
            if (verbose): print("\n\tPlotting end state potential energy distributions (by state)\n")        
            for state_num in range(1, num_states+1):
                outfile = plot_folder_path + '/' + title_prefix + '_pot_ene_state_' + str(state_num) + '.png'
                reeds.function_libs.visualization.pot_energy_plots.plot_energy_distribution_by_state(energy_trajectories, outfile, state_num, s_values,
                                                                                                     manual_xlim = None, shared_xaxis = True)
        
        if sub_control["pot_ene_by_replica"]:
            if (verbose): print("\n\tPlotting end state potential energy distributions (by replica)\n")
            for replica_num in range(1, len(energy_trajectories) + 1):
                outfile =  plot_folder_path + '/' + title_prefix + '_pot_ene_replica_' + str(replica_num) + '.png'
                reeds.function_libs.visualization.pot_energy_plots.plot_energy_distribution_by_replica(energy_trajectories[replica_num - 1], outfile,
                                                                                                       replica_num, s_values[replica_num-1],
                                                                                                       manual_xlim = None, shared_xaxis = True)
        
        # this variable allows to access particular elements in the pandas DataFrame
        singleStates = ['e' + str(i) for i in range(1, num_states+1)]
        
        # Timeseries of the potential energy of the end states.
        for i, ene_traj in enumerate(energy_trajectories):
            if sub_control["pot_ene_timeseries"]:
                out_path = plot_folder_path + "/edsState_potential_timeseries_" + str(ene_traj.s) + ".png"
                reeds.function_libs.visualization.pot_energy_plots.plot_potential_timeseries(time=ene_traj.time, potentials=ene_traj[singleStates],
                                                                                             y_range=(-1000, 1000), title="EDS_stateV_scatter",
                                                                                             out_path=out_path)
             
            if sub_control["pot_ene_grid_timeseries"]:
                out_path = plot_folder_path + '/' + title_prefix + '_pot_ene_timeseries_' + str(i+1) + '.png'
                title = title_prefix + ' potential energy timeseries - s = ' + str(s_values[i])
                reeds.function_libs.visualization.pot_energy_plots.plot_sampling_grid(traj_data = ene_traj, y_range=(-1000, 1000), out_path=out_path, title=title)

        # Plots related to the reference potential energy (V_R)

        if sub_control["ref_timeseries"]:
            outfile = plot_folder_path + '/' + title_prefix + '_ref_pot_ene_timeseries.png'
            reeds.function_libs.visualization.pot_energy_plots.plot_ref_pot_ene_timeseries(energy_trajectories, outfile, s_values)

        if sub_control["ref_distrib"]:
            outfile = plot_folder_path + '/' + title_prefix + '_ref_pot_ene_distrib.png'
            reeds.function_libs.visualization.pot_energy_plots.plot_ref_pot_energy_distribution(energy_trajectories, outfile, s_values)
            
        if (sub_control["distance_restraints"]):
            if (verbose): print("\tPLOT Disres_bias timeseries:\n")
            for ene_traj in energy_trajectories:
                # plot disres_contrib:
                out_path = plot_folder_path + "/distance_restraints_" + str(ene_traj.s) + ".png"
                singleStates = ["totdisres"]

                reeds.function_libs.visualization.pot_energy_plots.plot_potential_timeseries(time=ene_traj["time"], potentials=ene_traj[singleStates],
                                                                                             title="EDS disres Potential s" + str(ene_traj.s), y_label="E/[kj/mol]",
                                                                                             x_label="t/[ps]",
                                                                                             out_path=out_path)

        if (sub_control["temperature_2d_plot"]):
            print("\tPLOT temperature 2D histogram:\t")

            if (isinstance(imd_file.MULTIBATH, type(None)) and not isinstance(imd_file.STOCHDYN, type(None))):
                reeds.function_libs.visualization.pot_energy_plots.plot_replicaEnsemble_property_2D(ene_trajs=energy_trajectories,
                                                                                                    out_path=plot_folder_path + "/temperature_heatMap.png",
                                                                                                    temperature_property="solutemp2")

            elif (not isinstance(imd_file.MULTIBATH, type(None)) and imd_file.MULTIBATH.NBATHS == 1):
                reeds.function_libs.visualization.pot_energy_plots.plot_replicaEnsemble_property_2D(ene_trajs=energy_trajectories,
                                                                                                    out_path=plot_folder_path + "/temperature_heatMap.png",
                                                                                                    temperature_property="solutemp2")
                
            else:
                reeds.function_libs.visualization.pot_energy_plots.plot_replicaEnsemble_property_2D(ene_trajs=energy_trajectories,
                                                                                                    out_path=plot_folder_path + "/temperature_heatMap.png",
                                                                                                    temperature_property="solvtemp2")

            if (verbose): print("DONE\n")
        # del energy_trajectories -- remove if memory without this is fine

    if (control_dict["eoffset"]["do"]):
        print("Start Eoffset")
        sub_control = control_dict["eoffset"]["sub"]
        out_dir = bash.make_folder(out_folder + "/eoff")

        # parsing_ene_traj_csvs 
        if energy_trajectories is None:
            energy_trajectories = parse_csv_energy_trajectories(concat_file_folder, ene_trajs_prefix)

        if (not os.path.exists(concat_file_folder)):
            raise IOError("could not find needed energies (contains all ene ana .dats) folder in:\n " + out_folder)
        
        if (sub_control["sampling_plot"]):
            # plot if states are sampled and minimal state
            print("\tplot sampling: ")
            (sampling_results, out_dir) = sampling_ana.detect_undersampling(out_path = out_dir, ene_traj_csvs = energy_trajectories,
                                                                            s_values = s_values, state_potential_treshold= state_undersampling_pot_tresh, undersampling_occurence_sampling_tresh=undersampling_frac_thresh)

        if (sub_control["calc_eoff"]):
            print("calc Eoff: ")
            # WARNING ASSUMPTION THAT ALL EOFF VECTORS ARE THE SAME!
            print("\tEoffs(" + str(len(Eoff[0])) + "): ", Eoff[0])
            print("\tS_values(" + str(len(s_values)) + "): ", s_values)
            print("\tsytsemTemp: ", temp)
            # set trim_beg to 0.1 when analysing non equilibrated data
            new_eoffs, all_eoffs = eds_energy_offsets.estimate_energy_offsets(ene_trajs = energy_trajectories, initial_offsets = Eoff[0], sampling_stat=sampling_results, s_values = s_values,
                                                                              out_path = out_dir, temp = temp, trim_beg = 0., state_undersampling_potential_threshold=state_undersampling_pot_tresh,
                                                                              undersampling_idx = sampling_results['undersamplingThreshold'], 
                                                                              plot_results = True, calc_clara = False)
        
        if (verbose): print("Done\n")

    if (control_dict["sopt"]["do"]):
        sub_control = control_dict["sopt"]["sub"]
        out_dir = bash.make_folder(out_folder + "/s_optimization")

        # get repdat file
        print(repdat_file_out_path)
        in_file = glob.glob(repdat_file_out_path)[0]
        print("Found RepFILE: " + str(in_file))

        if (sub_control["run_RTO"]):
            print("Start Sopt\n")
            print("repdat_in_file: ", in_file, "\n")
            svals = reeds.function_libs.analysis.parameter_optimization.optimize_s(in_file=in_file, out_dir=out_dir,
                                                                                   title_prefix="s_opt", in_imd=in_imd,
                                                                                   add_s_vals=add_s_vals, trial_range=s_opt_trial_range,
                                                                                   state_weights=state_weights,
                                                                                   run_NLRTO=sub_control["run_NLRTO"], run_NGRTO=sub_control["run_NGRTO"],
                                                                                   verbose=verbose)

        if (sub_control["visualize_transitions"]):
            print("\t\tvisualize transitions")
            reeds.function_libs.analysis.parameter_optimization.get_s_optimization_transitions(out_dir=out_dir, rep_dat=in_file, title_prefix=title_prefix)

        if (sub_control["roundtrips"] and sub_control["run_RTO"]):
            print("\t\tshow roundtrips")
            in_repdat_file = repdat.Repdat(in_file)

            # retrieve data:
            if verbose: print("get replica transitions")
            s_values = in_repdat_file.system.s
            trans_dict = in_repdat_file.get_replica_traces()

            # plot
            if verbose: print("Plotting Histogramm")
            reeds.function_libs.visualization.re_plots.plot_repPos_replica_histogramm(out_path=out_dir + "/replica_repex_pos.png", data=trans_dict,
                                                                                      title=title_prefix,
                                                                                      s_values=s_values)

        if (verbose): print("Done\n")

    if (control_dict["dfmult"]["do"]):
        print("Start Dfmult")

        # check convergence:
        dfmult_convergence_folder = out_folder + "/free_energy"
        if (not os.path.isdir(dfmult_convergence_folder)):
            bash.make_folder(dfmult_convergence_folder, "-p")

        if energy_trajectories is None:
            energy_trajectories = parse_csv_energy_trajectories(concat_file_folder, ene_trajs_prefix)

        reeds.function_libs.analysis.free_energy.free_energy_convergence_analysis(ene_ana_trajs=energy_trajectories, out_dir=dfmult_convergence_folder,
                                                                                  out_prefix=title_prefix, in_prefix=ene_trajs_prefix, verbose=verbose,
                                                                                  dfmult_all_replicas=dfmult_all_replicas)

    # When we reach here, we no longer need the data in energy_trajectories, memory can be freed.
    del energy_trajectories

    if (control_dict["prepare_input_folder"]["do"]):
        sub_control = control_dict["prepare_input_folder"]["sub"]
        print("PREPARE NEXT FOLDER- for next run")

        next_dir = bash.make_folder(out_folder + "/next", "-p")
        next_imd = next_dir + "/next.imd"

        # add ne w cnf s for the new S-distribution
        print("generating new Cnfs for new s_dist")

        if (sub_control["eoff_to_sopt"]):  # if the s_dist should be converted from eoff to sopt
            new_sval = [s_values, []]
            new_sval[1] = [1.0 for x in range(num_states)] + list(
                sdist.get_log_s_distribution_between(start=1.0, end=min(s_values), num=len(s_values) - 4))[
                                                             1:]  # todo: hardcoded make clever and do automatic!
            svals = new_sval

        print('new_s(' + str(len(svals)) + ") ", svals)
        print("svalues var", s_values)
        input_cnfs = os.path.dirname(in_imd) + "/coord"
        print("svals", svals)
            
        # Put the proper cnfs in place        
 
        if (sub_control["eoff_to_sopt"]):
            if (not os.path.isdir(optimized_eds_state_folder)):
                raise IOError("Could not find optimized state output dir: " + optimized_eds_state_folder)
            
            opt_state_cnfs = sorted(glob.glob(optimized_eds_state_folder+'/*.cnf'), 
                                    key=lambda x: int(x.split("_")[-1].replace(".cnf", "")))
            for i in range(1, len(svals[1])+1):
                bash.copy_file(opt_state_cnfs[(i-1)%num_states], next_dir + '/sopt_run_' + str(i) + '.cnf')

        elif (len(list(set(svals[0]))) > len(list(set(svals[1])))):
            if verbose: print("reduce coordinate Files:")
            
            if (not os.path.isdir(optimized_eds_state_folder)):
                raise IOError("Could not find optimized state output dir: " + optimized_eds_state_folder)
            file_management.reduce_cnf_eoff(in_num_states=num_states, in_opt_struct_cnf_dir=optimized_eds_state_folder,
                                            in_current_sim_cnf_dir=input_cnfs,
                                            in_old_svals=s_values, in_new_svals=svals[1],
                                            out_next_cnfs_dir=next_dir)

        elif (len(svals[0]) < len(svals[1])):
            if verbose: print("reduce coordinate Files:")
            file_management.add_cnf_sopt_LRTOlike(in_dir=concat_file_folder, out_dir=next_dir, in_old_svals=s_values,
                                                  cnf_prefix=title_prefix,
                                                  in_new_svals=svals[1], replica_add_scheme=adding_new_sReplicas_Scheme,
                                                  verbose=verbose)

        else:
            if verbose: print("same ammount of s_vals -> simply copying output:")
            bash.copy_file(concat_file_folder + "/*cnf", next_dir)

        # write next_imd.
        print("write out imd file ")
        imd_file = imd.Imd(in_imd)

        ##New EnergyOffsets?
        if sub_control["write_eoff"] and control_dict["eoffset"]:
            imd_file.edit_REEDS(EIR=np.round(new_eoffs, 2))
        elif (sub_control["write_eoff"] and not control_dict["Eoff"]["sub"]["calc_eoff"]):
            warnings.warn("Could not set Eoffs to imd, as not calculated in this run!")

        ##New S-Values?=
        if (sub_control["write_s"] and control_dict["sopt"]["sub"]["run_RTO"]) or sub_control["eoff_to_sopt"]:
            imd_file.edit_REEDS(SVALS=svals[1])
        elif (sub_control["write_s"] and not control_dict["sopt"]["sub"]["run_RTO"]):
            warnings.warn("Could not set s-values to imd, as not calculated in this run!")

        imd_file.write(next_imd)
        if (verbose): print("Done\n")

    if (control_dict["compress_simulation_folder"]["do"]):
        print("Compress simulation folder")
        in_tre = sorted(glob.glob(concat_file_folder + "/*.tre"))
        in_trc = sorted(glob.glob(concat_file_folder + "/*.trc"))
        compress_files = in_tre + in_trc
        compress_list = [in_folder]

        file_management.compress_files(in_paths=compress_files)
        file_management.compress_folder(in_paths=compress_list)
        if (verbose): print("Done\n")

    return eoff_statistic, svals, dFs
