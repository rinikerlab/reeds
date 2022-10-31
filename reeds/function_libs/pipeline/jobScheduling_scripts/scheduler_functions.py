import glob
import os

from pygromos.euler_submissions.FileManager import Simulation_System
from pygromos.euler_submissions.Submission_Systems import _SubmissionSystem
from pygromos.utils import bash
from pygromos.files import imd

from reeds.function_libs.pipeline.worker_scripts.simulation_workers import clean_up_simulation_files as clean_up
from reeds.function_libs.pipeline.worker_scripts.simulation_workers import prepare_imd_initialisation
from reeds.function_libs.utils.structures import spacer


def do_skip_job(tmp_out_cnf: str, simSystem: Simulation_System, tmp_jobname: str,
                job_submission_system: _SubmissionSystem, previous_job: int, do_not_doubly_submit_to_queue: bool = True,
                verbose: bool = True):
    """
        This function is detecting if a simulation was already carried out or if the simulation is already queued in the submission system.

    Parameters
    ----------
    tmp_out_cnf : str
        temporary output coordinate file
    simSystem : Simulation_System
        this obj. contains paths to the different simulation files.
    tmp_jobname : str
        temporary job name
    job_submission_system : _SubmissionSystem
        the submission system of choice
    previous_job : str
        the id of the previous job.
    do_not_doubly_submit_to_queue : bool, optional
        if a job already exists in the submission queue, shall this job not be submitted?
    verbose: bool, optional
        Monster noise!

    Returns
    -------
    bool, int
        return a bool, deciding wether to skip the job and a int to get the correct jobid, if should be skipped.

    """
    # Check if job with same name is already in the queue!
    if (do_not_doubly_submit_to_queue):  # can we find an job with this name in the queue?
        if (verbose): print("Checking for jobs with name: " + tmp_jobname)
        queued_job_ids = job_submission_system.get_jobs_from_queue(job_name=tmp_jobname)

        ## check if already submitted
        if (len(queued_job_ids) > 0):  # check if job is already submitted:
            if (verbose): print(
                "\t\t\tSKIP job Submission: " + tmp_jobname + " was already submitted to the queue! \n\t\t\t\tSKIP\n"
                + "\t\t\tsubmitted IDS: " + "\n\t\t\t".join(map(str, queued_job_ids)) + "\n")
            setattr(simSystem, "coord_seeds", tmp_out_cnf)  # set next coord Files
            prefix_command = ""

            if (len(queued_job_ids) == 1):
                previous_job = queued_job_ids[0]
                if (verbose): print("\nTRY to attach next job to ", previous_job, "\n")
            else:
                raise ValueError("\nthere are multiple jobs, that could be the precessor. " + " ".join(queued_job_ids))
            return True, previous_job

    # Check if run was already finished:
    tmp_out_cnfs_regex = "_".join(tmp_out_cnf.split("_")[:-1]) + "*.cnf"
    if (verbose): print("Checking for resulting files: " + tmp_out_cnfs_regex)

    if (len(glob.glob(tmp_out_cnfs_regex)) > 0):  # was this job already run and finished?
        if (verbose): print(
            "\t\t NOT SUBMITTED!(inScript) as these Files were found: \n\t" + tmp_out_cnfs_regex)
        setattr(simSystem, "coord_seeds", tmp_out_cnf)  # set next coord Files
        prefix_command = ""
        if (verbose): print(simSystem.coordinates)
        return True, None

    return False, previous_job


def chain_submission(gromosXX_bin_dir: str, in_imd_path: str, simSystem:Simulation_System,
                     out_dir_path: str, out_prefix: str,
                     chain_job_repetitions: int, slave_script: str,
                     job_submission_system: _SubmissionSystem, jobname: str, nmpi: int,
                     job_queue_duration: str = "24:00",
                     run_analysis_script_every_x_runs: int = 0, in_analysis_script_path: str = "",
                     do_not_doubly_submit_to_queue: bool = True, start_run_index: int = 1,
                     prefix_command: str = "", previous_job_ID: int = None, work_dir: str = None,
                     write_free_energy_traj: bool = False, 
                     initialize_first_run: bool = True, reinitialize: bool = False,
                     memory: int = None, 
                     verbose: bool = False)->(int, str, Simulation_System):
    """
        This function takes care of submiting a given chain of jobs to the job queue.
        if an analysis script is provided, it will be added last or multiple times according to run_analysis_script_every_x_runs

    Parameters
    ----------
    gromosXX_bin_dir : str
        gromos binary dir
    in_imd_path : str
        path to the input imd-file
    simSystem : Simulation_System
        paths to the system files
    out_dir_path :str
        out directory for the simulations
    out_prefix : str
        out prefix
    chain_job_repetitions : int
        number of job iterations
    slave_script : str
        script, that will be submitted as worker
    job_submission_system : _SubmissionSystem
        job submission system for the job queue
    jobname : str
        name of the job
    nmpi : int
        number of MPI cores
    job_queue_duration : str
        job duration of each iteration.
    run_analysis_script_every_x_runs : int
        run intermediate analysis
    in_analysis_script_path : str
        path to the analysis script
    do_not_doubly_submit_to_queue : bool
        shall the script submit jobs doubly to the queue
    start_run_index : int
        start the chain from iteration x
    prefix_command : str
        command before the actual command
    previous_job_ID : int
        chain the job submission to this previous job - ID
    work_dir : str
        tmp workdir for the simulation
    write_free_energy_traj : bool
        shall the free energy traj be written out?
    initialize_first_run : bool
        shall the first run, be initialized with gromos parameters
    reinitialize : bool
        shall all runs, be initialized with gromos parameters (@Warning: not recommended)
    memory : int, optional
        how much memory to reserve for submission
    verbose : bool
        verbosity level

    Returns
    -------
    (int, str, Simulation_System)
        return the jobID of the last job, return a temporary job name and the simSystem path - obj
    """

    print("start_run_index " + str(start_run_index))
    print("job rep " + str(chain_job_repetitions))

    for run in range(start_run_index, chain_job_repetitions + 1):

        if (verbose): print(spacer + "\n submit  " + jobname + "_" + str(run) + "\n")

        tmp_outprefix = out_prefix + "_" + str(run)
        tmp_jobname = jobname + "_" + str(run)
        tmp_outdir = out_dir_path + "/" + tmp_outprefix
        tmp_in_imd = tmp_outdir + "/" + tmp_outprefix + ".imd"
        tmp_out_cnf = tmp_outdir + "/" + tmp_outprefix + ".cnf"

        # Checks if run should be skipped (useful for rerun in case of crash)!
        do_skip, previous_job_ID = do_skip_job(tmp_out_cnf=tmp_out_cnf, simSystem=simSystem, tmp_jobname=tmp_jobname,
                                               job_submission_system=job_submission_system,
                                               previous_job=previous_job_ID,
                                               do_not_doubly_submit_to_queue=do_not_doubly_submit_to_queue,
                                               verbose=verbose)
            
        # Note: Prefix command sometimes contains important previous commands to run
        #       in the sopt for example. 
        if (not do_skip):
            bash.make_folder(tmp_outdir)
            
            # We will write the arguments to the python script in a bash array 
            # to make it simpler to read in our input files. 
            prefix_command += "\n#Command to initiliaze imd file properly.\n" 
            prefix_command += "init_args=(\n"
            prefix_command += "-run " + str(run) + " \n"
            prefix_command += "-in_imd_path " + in_imd_path + " \n"
            prefix_command += "-tmp_in_imd " + tmp_in_imd + " \n"
            prefix_command += "-initialize_first_run " + str(initialize_first_run) + " \n"
            prefix_command += "-reinitialize " + str(reinitialize) + " \n"
            prefix_command += ")\n"

            # build command
            prefix_command += "python " + prepare_imd_initialisation.__file__ + " \"${init_args[@]}\" \n"
            prefix_command += "sleep 2s\n" 
            
            # optionals:
            add_option = ""
            if (write_free_energy_traj):
                add_option += "-write_free_energy_traj"
            
            # As above: write the arguments into a bash array:
            
            md_script_command = prefix_command + "\n\n#Command to submit job\n"
            
            md_script_command += "\nmd_args=(\n"
            md_script_command += "-in_imd " + tmp_in_imd + "\n"
            md_script_command += "-in_top " + simSystem.top.top_path + "\n"
            
            is_restrained = False
            if(hasattr(simSystem.top, "disres_path") and simSystem.top.disres_path is not None):
                md_script_command += "-in_disres " + simSystem.top.disres_path + "\n"
                is_restrained = True
            if(hasattr(simSystem.top, "posres_path") and simSystem.top.posres_path is not None):
                md_script_command += "-in_posres " + simSystem.top.posres_path + "\n"
                md_script_command += "-in_refpos " + simSystem.top.refpos_path + "\n"
                is_restrained = True
            # system is not going to be restrained
            if (is_restrained):
                if(not hasattr(simSystem.top, "refpos_path")):
                    print("Warning - No restraint file given.")
            else:
                print("No restraint file, suuuure?")
            
            md_script_command += "-in_perttop " + simSystem.top.perturbation_path + "\n"
            md_script_command += "-in_coord " + simSystem.coordinates + "\n"
            md_script_command += "-nmpi " + str(nmpi) + "\n"
            md_script_command += "-gromosXX_bin_dir " + gromosXX_bin_dir + "\n"
            md_script_command += "-out_dir " + tmp_outdir + "\n"
            md_script_command += "-work_dir " + str(work_dir) + "\n"
            md_script_command += add_option
            md_script_command += ")\n"
            
            # Write the line which will call the script            
                
            md_script_command += "python " + slave_script + "  \"${md_args[@]}\" \n"

            # cleanup from the same job!
            
            clean_up_command = "python " + str(clean_up.__file__) + "  -in_simulation_dir " + \
                                str(tmp_outdir) + " -n_processes " + str(nmpi)
            
            md_script_command += "\n\n" + clean_up_command + "\n"
            
            print("COMMAND: \n", md_script_command)

            try:
                if (verbose): print("\tSubmitting simulation")
                os.chdir(tmp_outdir)
                outLog = tmp_outdir + "/" + out_prefix + "_md.out"
                errLog = tmp_outdir + "/" + out_prefix + "_md.err"
                previous_job_ID = job_submission_system.submit_to_queue(command=md_script_command, jobName=tmp_jobname,
                                                                        duration=job_queue_duration,
                                                                        submit_from_dir=tmp_outdir,
                                                                        queue_after_jobID=previous_job_ID,
                                                                        outLog=outLog, errLog=errLog,
                                                                        nmpi=nmpi,
                                                                        end_mail=False,
                                                                        maxStorage = memory,
                                                                        verbose=verbose)

                # OPTIONAL schedule - analysis inbetween.
                if (run > 1 and run_analysis_script_every_x_runs != 0 and
                        run % run_analysis_script_every_x_runs == 0
                        and run < chain_job_repetitions):
                    if (verbose): print("\tINBETWEEN ANALYSIS")
                    tmp_ana_jobname = jobname + "_intermediate_ana_run_" + str(run)
                    outLog = tmp_outdir + "/" + out_prefix + "_inbetweenAna.out"
                    errLog = tmp_outdir + "/" + out_prefix + "_inbetweenAna.err"
                    ana_id = job_submission_system.submit_to_queue(command=in_analysis_script_path,
                                                                   jobName=tmp_ana_jobname,
                                                                   outLog=outLog, errLog=errLog,
                                                                   maxStorage=20000, queue_after_jobID=previous_job_ID, nmpi=5,
                                                                   verbose=verbose)
                if (verbose): print("\n")
            except ValueError as err:  # job already in the queue
                print("ERROR during submission:\n")
                print("\n".join(err.args))
        else:
            previous_job_ID = None
        
        if (verbose): print("\n")
        if (verbose): print("job_postprocess ")
        prefix_command = ""
        setattr(simSystem, "coordinates", tmp_out_cnf)

    return previous_job_ID, tmp_jobname, simSystem
