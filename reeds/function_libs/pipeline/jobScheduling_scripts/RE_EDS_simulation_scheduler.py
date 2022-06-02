"""
Wrapper for Long simulations -  similar to repex_EDS_run just chops the running time into different small simulations.
"""

import copy
import os
import sys
import traceback
from typing import Iterable

from pygromos.euler_submissions import FileManager as fM
from pygromos.euler_submissions.FileManager import Simulation_System
from pygromos.euler_submissions.Submission_Systems import LSF
from pygromos.euler_submissions.Submission_Systems import _SubmissionSystem
from pygromos.files.coord import cnf
from pygromos.files.imd import Imd
from pygromos.utils import bash
from reeds.function_libs.pipeline.jobScheduling_scripts.scheduler_functions import chain_submission
from reeds.function_libs.pipeline.worker_scripts.simulation_workers import RE_EDS_simulation_run_worker as workerScript
from reeds.function_libs.utils.structures import spacer, additional_argparse_argument


def do(in_simSystem: Simulation_System.System, in_imd_path: str, out_dir_path: str, gromosXX_bin_dir: str,
       jobname: str, nmpi: int, duration_per_job: str,
       num_simulation_runs: int = 2,
       num_equilibration_runs: int = 1, equilibration_trial_num: int = None,
       work_dir: str = None, force_queue_start_for_analysis: bool = False,
       in_analysis_script_path: str = None, run_analysis_script_every_x_runs: int = 0, initial_command: str = "",
       do_not_doubly_submit_to_queue: bool = True, previous_job_ID: int = None, write_free_energy_traj: bool = False,
       type_job_submission_system: _SubmissionSystem = LSF, initialize_first_run:bool = True, reinitialize:bool = False, 
       memory: int= None,
       verbose: bool = False):
    """ RE_EDS_simulation_scheduler

        The RE-EDS scheduler function is scheduling a re-eds simulation in multi steps in the queue.

        If an analysis_script path is provided, it will be chained after the simulation.

Parameters
----------
in_simSystem : system
    This is a long fun description .
in_imd_path : str
    simulation parameters .
out_dir_path : str
    path to the out directory .
gromosXX_bin_dir : str, optional
    path to the gromosXX binary dir.
jobname : str, optional
    name of the jobs in the queue.
nmpi : int, optional
    number of MPI cores.
duration_per_job : str, optional
    define length of the one job.
num_simulation_runs : int, optional
    number of repeated simulatoin runs.
num_equilibration_runs : int, optional
    number of equilibration runs before the production runs.
equilibration_trial_num : int
    number of trials for each equilibration run
work_dir : str, optional
    alternative work dir location. good for debbuging
in_analysis_script_path : str, optional
    path to analysis script.
run_analysis_script_every_x_runs : int, optional
    shall there be analysis runs during simulation?
do_not_doubly_submit_to_queue : bool, optional
    Check if there is already a job with this name, do not submit if true.
previous_job_ID : int, optional
    Queue this job chain after this job ID.
write_free_energy_traj : bool, optional
    writting a free energy traj?
type_job_submission_system : _SubmissionSystem, optional
    type of submission system to use.
initialize_first_run : bool, optional
    should the velocities of the first run be reinitialized?
reinitialize : bool, optional
    should the velocities be reinitialized for all runs?
memory : int, optional
    how much memory to reserve for submission
verbose : bool, optional
    I can be loud and noisy!

Returns
-------
int
    returns the jobID of the last submitted job or -1 on failure

    """

    # Prepare
    try:
        if (verbose): print("Script: ", __file__)
        simSystem = copy.deepcopy(in_simSystem)
        if (verbose): print("prepare sim")

        # Outdir
        bash.make_folder(out_dir_path)  # final output_folder

        # workdir:
        if (not isinstance(work_dir, type(None)) and work_dir != "None"):
            if (verbose): print("\t -> Generating given workdir: " + work_dir)
            bash.make_folder(work_dir, "-p")
            os.chdir(work_dir)
            prepared_imd = work_dir + "/" + os.path.basename(in_imd_path) if (
                not "*" in in_imd_path) else work_dir + "/repex_sim.imd"

        else:
            if (verbose): print("\t -> Using on node workdir ")
            prepared_imd = out_dir_path + "/" + os.path.basename(in_imd_path) if (
                not "*" in in_imd_path) else out_dir_path + "/repex_sim.imd"

        if (verbose): print(in_imd_path)
        prefix_command = ""
        if (initial_command != ""):
            prefix_command += initial_command + "\n\n"
        
        # sim vars logs
        out_prefix = jobname
        slave_script = workerScript.__file__

        # all Files present?
        if (verbose): print("in: " + str(prepared_imd))
        check_path_warn_paths = []
        check_path_dependencies_paths = [slave_script, simSystem.top.top_path, 
                                         simSystem.top.perturbation_path,
                                         out_dir_path, ]  # Coord file is used by repex in_imd_path prepared_imd
              
        # accounting for the different types of restraint used:
        if not hasattr(simSystem.top, 'refpos_path'):
            simSystem.top.refpos_path = None
        if not hasattr(simSystem.top, 'posres_path'):
            simSystem.top.posres_path = None
        if(not ((simSystem.top.disres_path is None) or (simSystem.top.disres_path == "None"))):
                            check_path_dependencies_paths.append(simSystem.top.disres_path)
        
        if(not (simSystem.top.refpos_path is None and simSystem.top.posres_path is None)):
            check_path_dependencies_paths.append(simSystem.top.posres_path)
            check_path_dependencies_paths.append(simSystem.top.refpos_path)
        if(not simSystem.top.disres_path is None):
              check_path_dependencies_paths.append(simSystem.top.disres_path)

        # optional paths
        if (not isinstance(work_dir, type(None)) and work_dir != "None"):
            check_path_dependencies_paths.append(work_dir)

        if (isinstance(gromosXX_bin_dir, type(None)) or gromosXX_bin_dir == "None"):
            check_path_warn_paths.append("Using gromosXX from shell environment")
            gromosXX_bin_dir = "None"
        else:
            check_path_dependencies_paths.append(gromosXX_bin_dir)

        if (isinstance(simSystem.coordinates, str)):
            check_path_dependencies_paths.append(simSystem.coordinates)
            if ("*" in simSystem.coordinates):
                setattr(simSystem, "coordinates", simSystem.coordinates.replace("*", ""))

            else:
                setattr(simSystem, "coordinates",
                        "_".join(simSystem.coordinates.split("_")[:-1]).replace("?", "").replace("*", "") + ".cnf")
        elif (isinstance(simSystem.coordinates, Iterable)):
            if (all([isinstance(x, str) for x in simSystem.coordinates])):
                check_path_dependencies_paths.extend(simSystem.coordinates)
            elif (all([isinstance(x, cnf.Cnf) for x in simSystem.coordinates])):
                check_path_dependencies_paths.extend(map(lambda x: x._orig_file_path, simSystem.coordinates))
            else:
                raise IOError(
                    "I could not understand the in_system coordinate file type! please give str or Cnf - Class" + str(
                        list(map(type, simSystem.coordinates))))
            # gromos repex_mpi wants to get only one CNF file!
            setattr(simSystem, "coordinates",
                    "_".join(simSystem.coordinates[0].split("_")[:-1]).replace("?", "").replace("*", "") + ".cnf")

        elif (isinstance(simSystem.coordinates, cnf.Cnf)):
            check_path_dependencies_paths.append(simSystem.coordinates._orig_file_path)
            setattr(simSystem, "coordinates",
                    "_".join(simSystem.coordinates._orig_file_path[0].split("_")[:-1]).replace("?", "").replace("*",
                                                                                                                "") + ".cnf")
        else:
            raise IOError("Could not find any coord path in system!\n" + str(simSystem.coordinates))

        check_path_dependencies_paths.append(in_imd_path)
        bash.check_path_dependencies(check_required_paths=check_path_dependencies_paths,
                                     check_warn_paths=check_path_warn_paths, verbose=verbose)

    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR in Preperations")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1

    # RUN Job
    original_path = os.getcwd()
    os.chdir(out_dir_path)
    job_submission_system = type_job_submission_system()

    # Submitting part
    try:
        # System Equilibration runs
        if (num_equilibration_runs > 0):
            eq_num = num_equilibration_runs
            if (verbose): print("equis: ", eq_num)

            # for shorter equils than production runs
            if (isinstance(equilibration_trial_num, int)):
                # maybe adapt trials?
                eq_imd_path = os.path.dirname(in_imd_path) + "/eq_" + os.path.basename(prepared_imd)
                eq_imd = Imd(in_imd_path)
                eq_imd.REPLICA_EDS.NRETRIAL = equilibration_trial_num
                eq_imd_path = eq_imd.write(eq_imd_path)
                if (verbose): print("EQ-trials, ", equilibration_trial_num)
            else:
                eq_imd_path = in_imd_path
                if (verbose): print("EQ-trials same as normal run.")

            ##submit equils with a hell of a function! sorry! :(
            previous_job_ID, tmp_jobname, simSystem = chain_submission(in_imd_path=eq_imd_path, simSystem=simSystem,
                                                                       gromosXX_bin_dir=gromosXX_bin_dir,
                                                                       out_dir_path=out_dir_path,
                                                                       out_prefix="eq_" + out_prefix,
                                                                       chain_job_repetitions=num_equilibration_runs,
                                                                       slave_script=slave_script,
                                                                       job_submission_system=job_submission_system,
                                                                       job_queue_duration=duration_per_job,
                                                                       do_not_doubly_submit_to_queue=do_not_doubly_submit_to_queue,
                                                                       jobname=jobname + "_eq", nmpi=nmpi,
                                                                       prefix_command=prefix_command,
                                                                       previous_job_ID=previous_job_ID,
                                                                       work_dir=work_dir,
                                                                       write_free_energy_traj=write_free_energy_traj,
                                                                       initialize_first_run=initialize_first_run,
                                                                       reinitialize=reinitialize,
                                                                       memory = memory,
                                                                       verbose=verbose)
            prefix_command = ""

        # Production Simulatoins
        ##submit Simulatoin with a hell of a function! sorry! :(
        previous_job_ID, tmp_jobname, simSystem = chain_submission(in_imd_path=in_imd_path, simSystem=simSystem,
                                                                   gromosXX_bin_dir=gromosXX_bin_dir,
                                                                   out_dir_path=out_dir_path, out_prefix=out_prefix,
                                                                   chain_job_repetitions=num_simulation_runs + num_equilibration_runs,
                                                                   start_run_index=num_equilibration_runs + 1,
                                                                   slave_script=slave_script,
                                                                   run_analysis_script_every_x_runs=run_analysis_script_every_x_runs,
                                                                   in_analysis_script_path=in_analysis_script_path,
                                                                   job_submission_system=job_submission_system,
                                                                   job_queue_duration=duration_per_job,
                                                                   do_not_doubly_submit_to_queue=do_not_doubly_submit_to_queue,
                                                                   jobname=jobname, nmpi=nmpi,
                                                                   prefix_command=prefix_command,
                                                                   previous_job_ID=previous_job_ID,
                                                                   work_dir=work_dir,
                                                                   write_free_energy_traj=write_free_energy_traj,
                                                                   initialize_first_run=initialize_first_run,
                                                                   reinitialize=reinitialize,
                                                                   memory = memory,
                                                                   verbose=verbose)

        # schedule - final analysis
        if (in_analysis_script_path != None):
            os.chdir(out_dir_path)
            tmp_ana_jobname = jobname + "_final_ana"

            ## check if already submitted
            queued_job_ids = job_submission_system.get_jobs_from_queue(job_name=tmp_ana_jobname)
            if (do_not_doubly_submit_to_queue and len(queued_job_ids) > 0):  # check if job is already submitted:
                if (verbose): print(
                    "\t\t\tSKIP submission of final Analysis: " + tmp_jobname + " was already submitted to the queue! \n\t\t\t\tSKIP\n"
                    + "\t\t\tsubmitted IDS: " + "\n\t\t\t".join(map(str, queued_job_ids)) + "\n")
                if (verbose): print("\nSkipped submission of final analysis.\n")
                previous_job_ID = queued_job_ids[0]
            else:

                if (verbose): print(
                    spacer + "\n submit ANA part " + str(num_simulation_runs + num_equilibration_runs) + "\n")
                try:
                    if (verbose): print("\tFINAL ANALYSIS")
                    outLog = out_dir_path + "/../" + jobname + "_Ana.out"
                    errLog = out_dir_path + "/../" + jobname + "_Ana.err"
                    previous_job_ID = job_submission_system.submit_to_queue(command=in_analysis_script_path,
                                                                            jobName=tmp_ana_jobname,
                                                                            submit_from_dir=out_dir_path,
                                                                            outLog=outLog, errLog=errLog,
                                                                            force_queue_start_after=force_queue_start_for_analysis,
                                                                            maxStorage=5000,
                                                                            queue_after_jobID=previous_job_ID, nmpi=10,
                                                                            verbose=verbose)

                except ValueError as err:  # job already in the queue
                    print("\n".join(err.args))
                    pass
                if (verbose): print("\n")
                if (verbose): print("\t\tANA jobID: \n\t\t" + str(previous_job_ID))
        if (verbose): print("\n\n")

        os.chdir(original_path)
        if (verbose): print("Done")
        return previous_job_ID

    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR in Submission")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1


if __name__ == "__main__":
    # Dynamic argparse
    import argparse
    from numpydoc.docscrape import NumpyDocString
    from reeds.function_libs.utils.argument_parser import make_parser_dynamic

    module_doc = NumpyDocString(str(__doc__))
    module_desc = "\n".join(module_doc['Summary']) + "\n" + "\n".join(module_doc['Extended Summary'])
    parser = argparse.ArgumentParser(description=module_desc)
    function_parameter_exceptions = ["in_SimSystem"]
    additional_argparse_argument = [
        additional_argparse_argument(name='in_system_name', type=str, required=True, desc="give your system a name"),
        additional_argparse_argument(name='in_coord_path', type=str, required=True, desc="input coordinate .cn file."),
        additional_argparse_argument(name='in_top_path', type=str, required=True, desc="input topology .top file."),
        additional_argparse_argument(name='in_perttop_path', type=str, required=True,
                                     desc="input perturbation topology .ptp file."),
        additional_argparse_argument(name='in_disres_path', type=str, required=True,
                                     desc="input distance restraint .dat file.")
    ]

    args = make_parser_dynamic(parser=parser, target_function=do,
                               additional_argparse_argument=additional_argparse_argument,
                               function_parameter_exceptions=function_parameter_exceptions, verbose=True)

    if (isinstance(args, int)):
        exit(1)

    # Build System:
    in_system_name = args.in_system_name
    in_topo_path = args.top
    in_coord_path = args.coord
    in_disres_path = args.disres
    in_perttopo_path = args.perttop

    top = fM.Topology(top_path=in_topo_path, disres_path=in_disres_path, perturbation_path=in_perttopo_path)
    system = Simulation_System.System(top=top, coordinates=in_coord_path, name=in_system_name)

    # do everything in here :)
    ret = do(in_simSystem=system, **args)

    # if error ocurred ret is smaller 0
    if (ret < 0):
        exit(1)
    else:
        exit(0)
