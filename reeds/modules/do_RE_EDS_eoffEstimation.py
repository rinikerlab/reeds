#!/usr/bin/env python3
"""
SCRIPT:            Do Estimation Energy Offsets
Description:
    This script Executes 3 simulation steps of reads (each 400ps). First one is used as equilibration.
    The rest is then used with the Energy Offset Estimator, to calculate the Eoffs. (Folder to continue with sopt is found in analysis/next)
Author: Benjamin Schroeder
"""

import copy
import glob
import os
import sys
import traceback
from collections import OrderedDict
from typing import Tuple

import reeds.function_libs.pipeline.module_functions
from pygromos.euler_submissions import FileManager as fM
from pygromos.utils import bash
from reeds.data import imd_templates
from reeds.function_libs.analysis import file_management as fileReeds
from reeds.function_libs.pipeline import generate_euler_job_files as gjs
from reeds.function_libs.pipeline.jobScheduling_scripts import RE_EDS_simulation_scheduler
from reeds.function_libs.pipeline.module_functions import adapt_imd_template_eoff
from reeds.function_libs.pipeline.worker_scripts.analysis_workers import RE_EDS_general_analysis
from reeds.function_libs.utils.structures import spacer


def do(out_root_dir: str, in_simSystem: fM.System, in_ene_ana_lib: str,
       in_template_imd_path: str = imd_templates.reeds_md_path,
       ssm_approach: bool = True, optimized_states: str =os.path.abspath("a_optimizedState/analysis/data"),
       gromosXX_bin_dir: str = None, gromosPP_bin_dir: str = None,
       exclude_residues: list = [], nmpi_per_replica: int = 1, num_simulation_runs: int = 2,
       num_equilibration_runs: int = 1, equilibration_trial_num: int = None,
       s_num: int = None, s_range: Tuple[float, float] = None, pot_tresh: float = 0.0,
       queueing_sys: object = None, work_dir: str = None, previous_job_ID: int = None,
       submit: bool = True, duration_per_job: str = "24:00", 
       initialize_first_run: bool = True, reinitialize: bool = False,
       do_not_doubly_submit_to_queue: bool = True,
       verbose: bool = True):
    """do Energy Offsets estimation

        This  function is triggering the work for the scheduling of Energy Offset estimation on Euler (currently).
        It autmatically builds folders into the out_root_dir, collecting input Files, a simulation folder,
        for which it triggers simulations and a analysis folder, in which the automatic analysis will be performed.


Parameters
----------
out_root_dir
    out_root_dir used for output of this script
simSystem
     this is the system obj. containing all paths to all system relevant Files.
in_ene_ana_lib
    path to in_ene_ana_lib,that can read the reeds system.
in_template_imd_path
    imd file
ssm_approach
    the Starting State Mixing Approach.
gromosXX_bin_dir
    path to gromosXX binary dir
gromosPP_bin_dir
    path to gromosPP binary dir
exclude_residues
    for cofactors etc.
nmpi_per_replica
     number of nmpis per replica @ at the moment, due to gromos, only 1 possible! @
num_simulation_runs
    how many simulation stepsd
num_equilibration_runs
    how many runs of equilibration?
equilibration_trial_num
    how many long shall the equil time be? equil?
s_num
    number of s_values( soptional)
s_range
    start end end of desired s_distribution (optional)
pot_tresh
    set the potential threshold for the analysis
queueing_sys
    Not implemented yet!
submit
    should the folder and Files just be builded or also the sopt_job be submitted to the queue?
work_dir
    define where the workdir is. Default is on the computing node.
previous_job_ID
    queue this job after job in  the queue
duration_per_job
    duration of one simulation step in the queue
initialize_first_run : bool, optional
    should the velocities of the first run be reinitialized?
reinitialize : bool, optional
    should the velocities be reinitialized for all runs?
do_not_doubly_submit_to_queue : bool, optional
    Check if there is already a job with this name, do not submit if true.
verbose
    I can be very talkative, you know...

Returns
-------
int
    returns the last queue submitted job ID or 0 if nothing was submitted or <0 if error occured

    """

    if (verbose): print(spacer + "START Eoff_estm.")
    #################
    # Prepare Jobs
    #################
    try:
        simSystem = copy.deepcopy(in_simSystem)

        # PATH DEFINITIONS
        input_dir = out_root_dir + "/input"
        coord_dir = input_dir + "/coord"
        out_dir_path = out_root_dir + "/simulation"
        in_imd_path = input_dir + "/repex_eoff.imd"
        old_result_folder = os.path.dirname(simSystem.coordinates[0])
        analysis_dir = out_root_dir + "/analysis"
        if (verbose): print("Generating Folders")

        # make folders:
        bash.make_folder(out_root_dir)
        bash.make_folder(input_dir)
        bash.make_folder(coord_dir)

        if (ssm_approach):
            os.chdir(out_root_dir)
            optimized_coordinates = glob.glob(optimized_states + "/*.cnf")
            print("found coords: ", optimized_coordinates)
            if (len(optimized_coordinates) == 0):
                raise IOError("Could not find any optimized coordinates. Did the simulation finish?")

            simSystem.coordinates = optimized_coordinates
            os.chdir("..")

        # IMD file
        ##from lower_bound
        s_vals_csv_path = old_result_folder + "/s_vals.csv"
        if (not os.path.exists(s_vals_csv_path) and not isinstance(s_range, type(None))):
            raise IOError("COULD NOT FIND S_VALS.CSV in : ", s_vals_csv_path, "\n TRY TO USE s_RANGE!")

        svals_file = open(s_vals_csv_path, "r")
        old_svals = list(map(float, " ".join(svals_file.readlines()).split()))
        # from reeds.function_libs.utils import s_log_dist
        # new_s_dist = s_log_dist.get_log_s_distribution_between(max_sval, min_sval, num_svals_desired)

        print(simSystem)

        ##adapt imd_templates
        if (verbose): print("Writing imd_templates")
        imd_file = adapt_imd_template_eoff(system=simSystem, imd_out_path=in_imd_path, imd_path=in_template_imd_path,
                                           old_svals=old_svals, s_num=s_num, s_range=s_range,
                                           non_ligand_residues=exclude_residues)
        in_imd_path = imd_file.path
        svals = imd_file.REPLICA_EDS.RES
        numstates = imd_file.REPLICA_EDS.NUMSTATES

        # COORDINATES:
        ## copy coordinates to input:
        if (verbose): print("Copy the coordinates in place")

        cnf_prefix = "REEDS_eoff_run"
        if (ssm_approach):
            coordinate_files_initial = simSystem.coordinates
            coordinate_dir = coord_dir
            s_value_coord = {sval: coordinate_files_initial[i % numstates] for i, sval in enumerate(svals)}

            print(s_value_coord)
            print(coordinate_dir + "/" + cnf_prefix)

            out_cnfs = [
                bash.copy_file(s_value_coord[s], coordinate_dir + "/" + cnf_prefix + "_ssm_s" + str(ind + 1) + ".cnf")
                for ind, s in enumerate(s_value_coord)]
            simSystem.coordinates = out_cnfs

        else:
            ## map coordinates
            if (len(old_svals) > 0 and len(old_svals) != len(svals)):
                if (verbose): print("Add replicas to match svalues")
                out_cnfs = fileReeds.adapt_cnfs_to_new_sDistribution(in_old_svals=old_svals, in_new_svals=svals,
                                                                     in_cnf_files=simSystem.coordinates,
                                                                     out_cnf_dir=coord_dir,
                                                                     cnf_prefix=cnf_prefix)
                setattr(simSystem, "coordinates", out_cnfs)
            else:
                simSystem.move_input_coordinates(coord_dir)

        # GENERATE array scripts
        
        # This dictionary only contains control specific        
        # to the energy offsets, the rest is written automatically        

        control_dict = {
            "eoffset": {"do": True,
                "sub": {
                    "calc_eoff": True,
                    "sampling_plot": True, }
                 },
            "prepare_input_folder": {"do": True,
                 "sub": {
                     "eoff_to_sopt": True,
                     "write_eoff": True,
                     "write_s": False
                 },
            }
        }

        nmpi = len(svals) * int(nmpi_per_replica)  # How many MPIcores needed?
        workdir = None
        jobname = simSystem.name

        # Generate execution Scripts
        if (verbose): print("generating Scripts in output dir")
        if (verbose): print("SVALS: ", len(svals), " nmpi_per_rep: ", nmpi_per_replica, "   nmpi", nmpi)

        ##Build analysis_script
        if (verbose): print("Analysis Script")
        analysis_vars = OrderedDict({
            "in_folder": out_dir_path,
            "in_imd": in_imd_path,
            "topology": simSystem.top.top_path,
            "optimized_eds_state_folder": optimized_states,
            "out_folder": analysis_dir,
            "gromos_path": gromosPP_bin_dir,
            "in_ene_ana_lib": in_ene_ana_lib,
            "n_processors": 5,
            "pot_tresh": pot_tresh,
            "frac_tresh": [0.1],
            "verbose": True,
            "grom_file_prefix": simSystem.name,
            "title_prefix": simSystem.name,
            "control_dict": control_dict,
        })
        in_analysis_script_path = reeds.function_libs.pipeline.module_functions.write_job_script(out_script_path=out_root_dir + "/job_analysis.py",
                                                                                                 target_function=RE_EDS_general_analysis.do_Reeds_analysis,
                                                                                                 variable_dict=analysis_vars)

        ##Build Job Script
        if (verbose): print("Scheduling Script")
        schedule_jobs_script_path = reeds.function_libs.pipeline.module_functions.write_job_script(out_script_path=out_root_dir + "/schedule_eoff_jobs.py",
                                                                                                   target_function=RE_EDS_simulation_scheduler.do,
                                                                                                   variable_dict=locals())

        ##set access
        bash.execute("chmod +x " + schedule_jobs_script_path + " " + in_analysis_script_path)  # make executables

    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR in file preperations")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1

    #################
    # Submit Jobs
    #################
    try:
        # JOB SUBMISSION
        if (submit):
            if (verbose): print("\n\nSUBMITTING: " + schedule_jobs_script_path)
            last_submitted_jobID = RE_EDS_simulation_scheduler.do(in_simSystem=simSystem, in_imd_path=in_imd_path,
                                                                  gromosXX_bin_dir=gromosXX_bin_dir,
                                                                  out_dir_path=out_dir_path, jobname=jobname, nmpi=nmpi,
                                                                  duration_per_job=duration_per_job,
                                                                  num_simulation_runs=num_simulation_runs,
                                                                  work_dir=workdir,
                                                                  num_equilibration_runs=num_equilibration_runs,
                                                                  in_analysis_script_path=in_analysis_script_path,
                                                                  do_not_doubly_submit_to_queue=do_not_doubly_submit_to_queue,
                                                                  initialize_first_run=initialize_first_run,
                                                                  reinitialize=reinitialize,
                                                                  verbose=verbose)

        else:
            if (verbose): print("\n\nSKIP submitting!")
            last_submitted_jobID = 0

        return last_submitted_jobID

    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR during job-submissoin")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1


# MAIN Execution from BASH
if __name__ == "__main__":
    from reeds.function_libs.utils.argument_parser import execute_module_via_bash

    print(spacer + "\t\tRE-EDS ENERGY OFFSET ESTIMATION \n" + spacer + "\n")
    requiers_gromos_files = [("in_top_path", "input topology .top file."),
                             ("in_coord_path", "input coordinate .cn file."),
                             ("in_perttop_path", "input pertubation topology .ptp file."),
                             ("in_disres_path", "input distance restraint .dat file.")]

    execute_module_via_bash(__doc__, do, requiers_gromos_files)