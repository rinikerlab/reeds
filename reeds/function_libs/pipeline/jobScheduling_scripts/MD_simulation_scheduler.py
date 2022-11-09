"""
Wrapper for Long simulations -  similar to repex_EDS_long_production_run.
This script schedules Simulations ons euler into the queue.

TODO: THIS IS OLD!
"""

import os
import sys
import traceback

from pygromos.euler_submissions import FileManager as fM
from pygromos.euler_submissions.Submission_Systems import LSF
from pygromos.utils import bash
from reeds.function_libs.pipeline.jobScheduling_scripts.scheduler_functions import do_skip_job
from reeds.function_libs.pipeline.worker_scripts.simulation_workers import MD_simulation_run_worker as workerScript
from reeds.function_libs.utils.structures import spacer


def do(in_simSystem: fM.System, in_imd_path: str,
       programm_path: str, out_dir_path: str,
       nmpi: int, nomp: int,
       duration_per_job: str,
       simulation_num: int,
       work_dir: str = None, equilibration_num: int = 0,
       write_out_free_energy_traj: bool = False,
       analysis_script: str = None, previous_job_ID: int = None, verbose: bool = False)->int:
    """
        schedule an MD-Simulation approach.
        This script will schedule a multi-step simulation job-chain to the job-queue.
        If provided it will attach as well an analysis-script to the queue.

        @Warning: This function will be moved to pyGromos


    Parameters
    ----------
    in_simSystem : fM.System
        system object, containing all required paths
    in_imd_path : str
        gromos imd path required.
    programm_path : str
        path to the binary folder in gromos
    out_dir_path : str
        output path for the simulation
    nmpi : int
        number of mpi cores
    nomp : int
        number of omp cores
    duration_per_job : str
        duration per job in the queue
    simulation_num : int
        number of simulations queued
    work_dir : str, optional
        directory in which the work is executed
    equilibration_num : int, optional
        number of equilibrations
    write_out_free_energy_traj : bool, optional
        write out free enegy traj
    analysis_script : str, optional
        path to the analysis script, will be queued as well.
    previous_job_ID : int, optional
        ID of the previous job
    verbose : bool, optional
        verbosity level.

    Returns
    -------
    int
        returns previous job id

    """
    # prepare
    try:
        if (verbose): print("Script: ", __file__)
        if (verbose): print("prepare sim")

        if (not isinstance(in_simSystem.coordinates, str)):
            raise ValueError("Expecting string for system.coordinates!")

        # Outdir
        bash.make_folder(out_dir_path)  # final output_folder

        # workdir:
        if (not isinstance(work_dir, type(None)) and work_dir != "None"):
            if (verbose): print("\t -> Generating given workdir: " + work_dir)
            bash.make_folder(work_dir, "-p")
            os.chdir(work_dir)
            prepared_imd = work_dir + "/" + os.path.basename(in_imd_path)

        else:
            if (verbose): print("\t -> Using on node workdir")
            prepared_imd = out_dir_path + "/" + os.path.basename(in_imd_path)

        prefix_command = " cp " + in_imd_path + " " + prepared_imd + " \n"

        # sim vars logs
        out_prefix = in_simSystem.name
        error_log = out_dir_path + "/" + out_prefix + ".err"
        std_log = out_dir_path + "/" + out_prefix + ".out"
        slave_script = workerScript.__file__

        # all Files present?
        if (verbose): print("in: " + str(prepared_imd))

        ##needed variables
        check_path_dependencies_paths = [slave_script, in_simSystem.top.top_path, out_dir_path,
                                         programm_path, ]  # Coord file is used by repex in_imd_path prepared_im
        ##variable paths
        if (not isinstance(in_simSystem.top.perturbation_path,
                           type(None)) and not in_simSystem.top.perturbation_path == "None"):
            check_path_dependencies_paths.append(in_simSystem.top.perturbation_path)
        if (not isinstance(in_simSystem.top.disres_path, type(None)) and not in_simSystem.top.disres_path == "None"):
            check_path_dependencies_paths.append(in_simSystem.top.disres_path)
        if (not isinstance(work_dir, type(None)) and work_dir != "None"):
            check_path_dependencies_paths.append(work_dir)
        if (isinstance(previous_job_ID, type(None))):
            check_path_dependencies_paths.append(in_simSystem.coordinates)
            check_path_dependencies_paths.append(in_imd_path)
            prepared_imd = bash.copy_file(in_imd_path, prepared_imd)

        bash.check_path_dependencies(check_path_dependencies_paths, verbose=verbose)

    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR in Preperations")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1

    # RUN Job
    try:

        if (equilibration_num > 0):  # calc the needed iterations
            equilibration_runs = equilibration_num
            if (verbose): print("equis: ", equilibration_runs)
        else:
            equilibration_runs = 0
            if (verbose): print("equis: ", equilibration_runs)

        # add optional script commands
        add_options = " "
        if (write_out_free_energy_traj):
            add_options += " -out_trg"
        lsf = LSF()

        for i in range(1, equilibration_num + simulation_num + 1):
            # Equil or normal run?
            if (i <= equilibration_runs):
                if (verbose): print(spacer + "\n submit EQ part " + str(i) + "\n")
                tmp_outprefix = "eq_" + out_prefix + "_" + str(i)
                tmp_jobname = in_simSystem.name + "_" + "_eq_" + str(i)

            else:
                if (verbose): print(spacer + "\n submit MD part " + str(i) + "\n")
                tmp_outprefix = out_prefix + "_" + str(i)
                tmp_jobname = in_simSystem.name + "_" + str(i)
            tmp_outdir = out_dir_path + "/" + tmp_outprefix
            tmp_in_imd = tmp_outdir + "/" + tmp_outprefix + ".imd"
            tmp_out_cnf = tmp_outdir + "/" + tmp_outprefix + ".cnf"

            skip_job, previous_job_ID = do_skip_job(tmp_out_cnf=tmp_out_cnf, simSystem=in_simSystem,
                                                    tmp_jobname=tmp_jobname, job_submission_system=lsf,
                                                    previous_job=previous_job_ID, verbose=verbose)
            if (not skip_job):
                prefix_command += "sleep 2s && cp " + in_imd_path + " " + tmp_in_imd  # initial
                bash.make_folder(tmp_outdir)
                # formulate commands
                md_script_command = prefix_command + " && python " + slave_script + " -imd " + tmp_in_imd + " -top " + in_simSystem.top.top_path + " -disres " + str(
                    in_simSystem.top.disres_path) + " -perttop " + str(
                    in_simSystem.top.perturbation_path) + " -coord " + in_simSystem.coordinates + " -nmpi " + str(
                    nmpi) + " -nomp " + str(nomp) \
                                    + " -bin " + programm_path + " -outdir " + tmp_outdir + " -workdir " + str(
                    work_dir) + add_options
                if (verbose): print("COMMAND: ", md_script_command)

                try:
                    # post_execution_command= "gzip "+tmp_outdir+"/*tr?",
                    previous_job_ID = lsf.submit_to_queue(md_script_command, jobName=tmp_jobname,
                                                      outLog=std_log, errLog=error_log,
                                                      queue_after_jobID=previous_job_ID,
                                                      nmpi=nmpi, nomp=nomp, duration=duration_per_job,
                                                      verbose=verbose)

                    if (verbose): print("process returned id: " + str(previous_job_ID))
                except:
                    raise Exception("could not submit this command: \n" + md_script_command)
            else:
                if (verbose): print("\t\t NOT SUBMITTED!")
            prefix_command = ""
            setattr(in_simSystem, "coordinates", tmp_out_cnf)

        if (analysis_script != None):
            tmp_jobname = in_simSystem.name + "_ana"
            ana_log = os.path.dirname(analysis_script) + "/ana_out.log"
            previous_job_ID = lsf.submit_to_queue(analysis_script, jobName=tmp_jobname,
                                                  outLog=ana_log, queue_after_jobID=previous_job_ID,
                                                  nmpi=nmpi, nomp=nomp, verbose=verbose)

            if (verbose): print(spacer + "\n submit ANA part " + str(i) + "\n")
            if (verbose): print(analysis_script)
            if (verbose): print("ANA jobID: " + str(previous_job_ID))

    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR in Submission")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1

    return previous_job_ID
