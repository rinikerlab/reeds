#!/usr/bin/env python3
"""

SCRIPT:            Do Eoff Re-Balancing
Description:
    This script executes multiple simuatlions. The scripts runs mutliple iterations of EoffRebalancing runs.
    In each iteration x simulations a 400 ps * iteration are exectued and afterwards the Eoffs get rebalanced.

Author: bschroed

"""

import copy
import os
import sys
import traceback
from collections import OrderedDict
from typing import Iterable, List

import reeds
import reeds.function_libs.pipeline.module_functions
from pygromos.euler_submissions import FileManager as fM
from pygromos.euler_submissions.Submission_Systems import LSF
# todo! make the queueing system exchangeable
from pygromos.euler_submissions.Submission_Systems import _SubmissionSystem
from pygromos.files import imd
from pygromos.files.coord import cnf as cnf_cls
from pygromos.utils import bash
from reeds.data import ene_ana_libs
from reeds.function_libs.pipeline import generate_euler_job_files as gjs
from reeds.function_libs.pipeline.module_functions import submit_job_sopt, build_sopt_step_dir
from reeds.function_libs.pipeline.worker_scripts.analysis_workers import RE_EDS_soptimization_final
from reeds.function_libs.utils.structures import adding_Scheme_new_Replicas, soptimization_params
from reeds.function_libs.utils.structures import spacer


def do(out_root_dir: str, in_simSystem: fM.System, in_template_imd: str = None,
       iterations: int = 4,
       noncontinous: bool = False,
       optimized_states_dir: str = os.path.abspath("a_optimizedState/analysis/next"),
       lower_bound_dir: str = os.path.abspath("b_lowerBound/analysis/next"),
       state_physical_occurrence_potential_threshold:List[float]=None,
       state_undersampling_occurrence_potential_threshold: List[float]=None,
       undersampling_fraction_threshold:float=0.9,
       equil_runs: int = None, steps_between_trials: int = 20, trials_per_run: int = 12500,
       non_ligand_residues: list = [],
       in_gromosXX_bin_dir: str = None, in_gromosPP_bin_dir: str = None,
       in_ene_ana_lib_path: str = ene_ana_libs.ene_ana_lib_path,
       nmpi_per_replica: int = 1, submit: bool = True, duration_per_job: str = "24:00",
       queueing_system: _SubmissionSystem = LSF,
       do_not_doubly_submit_to_queue: bool = True,
       initialize_first_run: bool = True, reinitialize: bool = False,
       verbose: bool = True):
    """
    SCRIPT:            Do Eoff rebalancing
    Description:
        This script executes multiple simuatlions. The scripts runs mutliple iterations of EoffRebalancing runs.
        In each iteration x simulations a 400 ps * iteration are exectued and afterwards the Eoffs get rebalanced.

    Author: bschroed

Parameters
----------
out_root_dir : str
    out_root_dir used for output of this script
in_simSystem : pygromos.PipelineManager.Simulation_System
     this is the system obj. containing all paths to all system relevant Files.
in_template_imd : str
    path to the imd file to simulate

iterations : int, optional
    How many optimization iterations do you want to perform?
add_replicas : int, optional
    How many replicas do you want to add per otpimization run?
adding_new_sReplicas_Scheme : reeds.function_libs.utils.structures.adding_Scheme_new_Replicas
    How shall new coordinate files be added to the system?
noncontinous : bool, optional
    Shall I use the output coordinates of the last sopt for the next sopt(False), or shall I always use the same starting coordinates per s-Optimization Iteration? (True)
state_physical_occurrence_potential_threshold : List[float], optional
    potential thresholds for physical sampling (default: read in from step a)
state_undersampling_occurrence_potential_threshold : List[float], optional
    potential thresholds for occurrence sampling (default: read in from step b)
undersampling_fraction_threshold : float, optional
    fraction threshold for physical/occurrence sampling (default: 0.9)
equil_runs : int, optional
    How often do you want to run prequilibration, before each run ? give int times 50ps
steps_between_trials : int, optional
    How many steps shall be executed between the trials?
trials_per_run : int, optional
    How many exchange trials shall be exectued per run?
non_ligand_residues : List[str], optional
    list of molecules (except solv and protein) that should not be considered as EDS-State.
in_gromosXX_bin_dir : str, optional
     path to gromosXX_bin binary dir
in_gromosPP_bin_dir : str, optional
    path to gromosPP_bin binary dir
in_ene_ana_lib_path : str, optional
     path to in_ene_ana_lib,that can read the reeds system.
nmpi_per_replica : int, optional
    number of nmpis per replica @ at the moment, due to gromos, only 1 possible! @
submit : bool, optional
    should the folder and Files just be builded or also the sopt_job be submitted to the queue?
queueing_system : Submission_System, optional
    @Development -  not Implemented yet but this shall allow different queueing systems or even a dummy queueing sys.
duration_per_job : str, optional
    duration of each job in the queue
initialize_first_run : bool, optional
    should the velocities of the first run be reinitialized?
reinitialize : bool, optional
    should the velocities be reinitialized for all runs?
do_not_doubly_submit_to_queue : bool, optional
    Check if there is already a job with this name, do not submit if true.
verbose : bool, optional
    I can be very talkative! :)

Returns
-------
int
    last submitted job ID; is -1 if Error, 0 if not submitted

    """

    if (verbose): print(spacer + "\n\tSTART sopt_process.")
    #################
    # Prepare general stuff
    #################
    try:
        simSystem = copy.deepcopy(in_simSystem)
        sopt_input = out_root_dir + "/input"

        if (not os.path.exists(sopt_input)):
            os.mkdir(sopt_input)

        # retrieve old Information:a
        last_data_folder = os.path.dirname(
            simSystem.coordinates[0])  # the input coord file should also contain the initial .imd file!

        if (isinstance(in_template_imd, type(None))):
            imd_path_last = last_data_folder + "/next.imd"
            if not (os.path.exists(imd_path_last)):
                raise IOError("could not find initial IMD with path: \n\t" + last_data_folder + "/next.imd")
        else:
            imd_path_last = in_template_imd

        if (isinstance(simSystem.coordinates, Iterable)):
            new_coords = []
            for coordinate_file_path in simSystem.coordinates:
                new_file_path = bash.copy_file(coordinate_file_path,
                                               sopt_input + "/" + os.path.basename(coordinate_file_path))
                new_coords.append(new_file_path)
            setattr(simSystem, "coordinates", new_coords)

        elif (isinstance(simSystem.coordinates, str)):
            new_coord = bash.copy_file(simSystem.coordinates,
                                       sopt_input + "/" + os.path.basename(simSystem.coordinates))
            setattr(simSystem, "coordinates", new_coord)
        else:
            raise IOError(
                "Could not copy system coordinates into input folder! please give them as str or List[str]. \n GOT: " + str(
                    simSystem.coordinates))

        last_data_folder = sopt_input
        cnf = cnf_cls.Cnf(simSystem.coordinates[0])

        ##get first coord file
        raw_residues = cnf.get_residues()
        residues, ligands, protein, non_ligands = imd.Imd.clean_residue_list_for_imd(raw_residues, non_ligand_residues)
        # get system information of system
        if (verbose): print("\tLIGSNum", ligands.number)
        if (verbose): print("\tResidues: ", residues)
        if (verbose): print()

        imd_file = imd.Imd(imd_path_last)
        # get number of s-vals from imd:
        num_svals = int(imd_file.REPLICA_EDS.NRES)
        num_states = int(imd_file.REPLICA_EDS.NUMSTATES)
        # set new step number between trials and new number of trials if necessary
        imd_file.STEP.NSTLIM = steps_between_trials
        imd_file.edit_REEDS(NRETRIAL=trials_per_run)
        imd_path_last = imd_file.write(sopt_input + "/repex_sopt_template.imd")

        # Setup s-optimization
        ## global vars
        imd_name_prefix = simSystem.name + "_"
        iteration_folder_prefix = out_root_dir + "/sopt"
        bash.make_folder(out_root_dir)

        state_undersampling_pot_tresh_path = optimized_states_dir + "/state_occurence_physical_pot_thresh.csv"
        if(state_physical_occurrence_potential_threshold is None and os.path.exists(state_undersampling_pot_tresh_path)):
            if not os.path.exists(state_undersampling_pot_tresh_path) :
                raise IOError("COULD NOT FIND state_occurence_pot_thresh.CSV in : ", state_undersampling_pot_tresh_path, "\n")
            else:
                tmp = open(state_undersampling_pot_tresh_path, "r")
                state_physical_occurrence_potential_threshold =  list(map(float, " ".join(tmp.readlines()).split()))
        elif(state_physical_occurrence_potential_threshold is None):
            state_physical_occurrence_potential_threshold = [0 for x in range(num_states)]

        state_undersampling_pot_tresh_path = lower_bound_dir + "/state_occurence_pot_thresh.csv"
        if(state_undersampling_occurrence_potential_threshold is None and os.path.exists(state_undersampling_pot_tresh_path)):
            if not os.path.exists(state_undersampling_pot_tresh_path) :
                raise IOError("COULD NOT FIND state_occurence_pot_thresh.CSV in : ", state_undersampling_pot_tresh_path, "\n")
            else:
                tmp = open(state_undersampling_pot_tresh_path, "r")
                state_undersampling_occurrence_potential_threshold =  list(map(float, " ".join(tmp.readlines()).split()))
        elif(state_undersampling_occurrence_potential_threshold is None):
            state_undersampling_occurrence_potential_threshold = [0 for x in range(num_states)]


    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR in file preperations of iteration()")
        print("#####################################################################################")
        print("\n".join(map(str, err.args)))
        print()
        traceback.print_exception(*sys.exc_info())
        return -1

    #################
    # Prepare each sopt iteration
    #################
    ## Loop vars
    job_id = None  # id for chaining
    repetitions = 1  # needed to elongate simulation length

    standard_name = simSystem.name
    iteration_sopt_job = None
    cur_svals = num_svals

    ## Prepare final analysis:

    ana_out_dir = out_root_dir + "/analysis"
    job_name = in_simSystem.name + "_final_sOptimization"
    analysis_vars = OrderedDict({
        "sopt_root_dir": out_root_dir,
        "title": in_simSystem.name,
        "state_physical_occurrence_potential_threshold": state_physical_occurrence_potential_threshold,
        "out_dir": ana_out_dir
    })

    in_final_analysis_script_path = reeds.function_libs.pipeline.module_functions.write_job_script(out_script_path=out_root_dir + "/job_final_analysis.py",
                                                                                                   target_function=RE_EDS_soptimization_final.do,
                                                                                                   variable_dict=analysis_vars)
    bash.execute("chmod +x " + in_final_analysis_script_path)  # make executables
    # generate each iteration folder && submission
    for iteration in range(1, iterations + 1):
        print("\n\nITERATION: " + str(iteration))

        # note: current_num_svals is the number of svalues used for the analysis, i.e. the
        # number of svalues *before* the optimization
        #todo: Remove
        soptimization_options = soptimization_params(add_replicas=0,
                                                     adding_new_sReplicas_Scheme=None,
                                                     current_num_svals=cur_svals)

        ##default equilibration scheme for equilibrations run
        if (not equil_runs):
            equil_runs = 1

        simSystem.name = standard_name + "_" + str(iteration)
        try:  # JOB preperation
            iteration_sopt_job = build_sopt_step_dir(iteration=iteration,
                                                     iteration_folder_prefix=iteration_folder_prefix,
                                                     soptimization_options=soptimization_options,
                                                     num_equilibration_runs=equil_runs,
                                                     imd_name_prefix=imd_name_prefix, in_simSystem=simSystem,
                                                     in_ene_ana_lib_path=in_ene_ana_lib_path,
                                                     state_undersampling_pot_tresh=state_undersampling_occurrence_potential_threshold,
                                                     state_physical_pot_tresh=state_physical_occurrence_potential_threshold,
                                                     undersampling_frac_thresh=undersampling_fraction_threshold,
                                                     in_gromosPP_bin_dir=in_gromosPP_bin_dir,
                                                     in_gromosXX_bin_dir=in_gromosXX_bin_dir,
                                                     ligands=ligands, old_sopt_job=iteration_sopt_job,
                                                     last_data_folder=last_data_folder,
                                                     nmpi_per_replica=nmpi_per_replica,
                                                     pot_tresh=state_physical_occurrence_potential_threshold, duration_per_job=duration_per_job,
                                                     num_simulation_runs=repetitions)

        except Exception as err:
            print("#####################################################################################")
            print("\t\tERROR in file preperations of iteration()")
            print("#####################################################################################")
            print("\n".join(map(str, err.args)))
            print()
            traceback.print_exception(*sys.exc_info())
            return -1

        try:  # JOB SUBMISSION
            # print(iteration_sopt_job.job_file_path, iteration_sopt_job.job_analysis_path)
            # print("COORD IN SUB: ",  iteration_sopt_job.sim_system.coordinates)
            bash.execute(
                "chmod +x " + iteration_sopt_job.job_file_path + " " + iteration_sopt_job.job_analysis_path)  # make executables

            job_id = submit_job_sopt(iteration_sopt_job, gromosXX_bin_dir=in_gromosXX_bin_dir,
                                     duration_per_job=duration_per_job, submit=submit, previous_job_id=job_id,
                                     do_not_doubly_submit_to_queue = do_not_doubly_submit_to_queue,
                                     initialize_first_run=initialize_first_run, reinitialize=reinitialize,
                                     verbose=verbose)

            # UPDATE vars for next run
            if (noncontinous):
                last_data_folder = sopt_input
            else:
                last_data_folder = os.path.dirname(iteration_sopt_job.check_analysis_files)
        except Exception as err:
            print("#####################################################################################")
            print("\t\tERROR during job-submissoin")
            print("#####################################################################################")
            print("\n".join(map(str, err.args)))
            print()
            traceback.print_exception(*sys.exc_info())
            return -1

        if(iteration>1):
            try:  # JOB SUBMISSION
                if (verbose): print("Final Analysis Script")

                job_submission_system = queueing_system()
                root_dir = os.getcwd()
                os.chdir(os.path.dirname(ana_out_dir))
                job_id_final_ana = job_submission_system.submit_to_queue(command=in_final_analysis_script_path,
                                                               jobName=job_name+"_sopt"+str(iteration),
                                                               outLog=ana_out_dir + "/" + job_name + ".out",
                                                               errLog=ana_out_dir + "/" + job_name + ".err",
                                                               maxStorage=5000, queue_after_jobID=job_id, nmpi=1,
                                                               verbose=verbose, sumbit_from_file=False)
                os.chdir(root_dir)

            except Exception as err:
                print("#####################################################################################")
                print("\t\tERROR during job-final Analysis-submissoin")
                print("#####################################################################################")
                print("\n".join(map(str, err.args)))
                print()
                traceback.print_exception(*sys.exc_info())
                return -1

    return job_id

    # MAIN Execution from BASH if __name__ == "__main__":
    from reeds.function_libs.utils.argument_parser import execute_module_via_bash
    print(spacer + "\t\tRE-EDS S-OPTIMIZATION \n" + spacer + "\n")
    execute_module_via_bash(__doc__, do)
