#!/usr/bin/env python3
"""

SCRIPT:            Do S-optimisation
Description:
    This script executes multiple simuatlions. The scripts runs mutliple iterations of s-distribution runs.
    In each iteration x simulations a 400 ps are exectued an afterwards the s-distribution get optimised by NLRTO.
    here it runs continous
    TODOS: make adapt imd file more genearl
Author: bschroed

"""

import os
from typing import List

import reeds.function_libs.pipeline.module_functions
from pygromos.euler_submissions import FileManager as fM
from pygromos.euler_submissions.Submission_Systems import LSF, SLURM
# todo! make the queueing system exchangeable
from pygromos.euler_submissions.Submission_Systems import _SubmissionSystem
from reeds.data import ene_ana_libs
from reeds.function_libs.utils.structures import adding_Scheme_new_Replicas
from reeds.function_libs.utils.structures import spacer
from reeds.modules._do_RE_EDS_Optimization import do_optimization


def do(out_root_dir: str, in_simSystem: fM.System, in_template_imd: str = None,
       iterations: int = 4, add_replicas: int = 4,
       adding_new_sReplicas_Scheme: adding_Scheme_new_Replicas = adding_Scheme_new_Replicas.from_below,
       noncontinous: bool = False,
       optimized_states_dir: str = os.path.abspath("a_optimizedState/analysis/next"),
       lower_bound_dir: str = os.path.abspath("b_lowerBound/analysis/next"),
       state_physical_occurrence_potential_threshold:List[float]=None,
       state_undersampling_occurrence_potential_threshold: List[float]=None,
       undersampling_fraction_threshold:float=0.9,
       equil_runs: int = 0, prod_runs = 1, steps_between_trials: int = 50,
       trials_per_run: int = 5000, non_ligand_residues: list = [],
       in_gromosXX_bin_dir: str = None, in_gromosPP_bin_dir: str = None,
       in_ene_ana_lib_path: str = ene_ana_libs.ene_ana_lib_path,
       nmpi_per_replica: int = 1, submit: bool = True, duration_per_job: str = "24:00:00",
       queueing_system: _SubmissionSystem = SLURM,
       run_NLRTO:bool=True, run_NGRTO:bool=False,
       do_not_doubly_submit_to_queue: bool = True,
       initialize_first_run: bool = True, reinitialize: bool = False, randomize: bool=False,
       memory: str = None,
       verbose: bool = True):
    """
    SCRIPT:            Do S-optimisation
    Description:
        This script executes multiple simuatlions. The scripts runs mutliple iterations of s-distribution runs.
        In each iteration x simulations a 400 ps are exectued an afterwards the s-distribution get optimised by NLRTO.
        here it runs continous
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
equil_runs : int, optional (default: 1)
    How often do you want to run prequilibration, before each run ? give int times 50ps
prod_runs : int, optional (default: 1)
    How many production runs after equilibration ?
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
randomize : bool, optional
    randomize the simulation seed
memory : str, optional
    how much memory to reserve for submission
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
    if(run_NGRTO and run_NLRTO or (not run_NGRTO and not run_NLRTO)):
        raise Exception("Please specify either NLRTO or NGRTO!")
    optimization_name = "sopt"
    learningFactors = [1 for _ in range(iterations)]
    pseudocount = None
    eoffRB_correctionPerReplica = False

    job_id = do_optimization(out_root_dir=out_root_dir, in_simSystem=in_simSystem, optimization_name=optimization_name, in_template_imd=in_template_imd,
                             iterations=iterations,
                             eoffEstimation_undersampling_fraction_threshold=undersampling_fraction_threshold,
                             sOpt_add_replicas= add_replicas, sOpt_adding_new_sReplicas_Scheme= adding_new_sReplicas_Scheme,
                             run_NLRTO = run_NLRTO, run_NGRTO = run_NGRTO,
                             eoffRB_learningFactors = learningFactors, eoffRB_pseudocount = pseudocount,
                             eoffRB_correctionPerReplica=eoffRB_correctionPerReplica,
                             non_ligand_residues = non_ligand_residues,
                             state_physical_occurrence_potential_threshold=state_physical_occurrence_potential_threshold,
                             state_undersampling_occurrence_potential_threshold=state_undersampling_occurrence_potential_threshold,
                             equil_runs=equil_runs, prod_runs=prod_runs, steps_between_trials=steps_between_trials, trials_per_run=trials_per_run,
                             optimized_states_dir=optimized_states_dir,
                             lower_bound_dir=lower_bound_dir,
                             in_gromosXX_bin_dir=in_gromosXX_bin_dir, in_gromosPP_bin_dir=in_gromosPP_bin_dir,
                             in_ene_ana_lib_path=in_ene_ana_lib_path,
                             nmpi_per_replica=nmpi_per_replica, submit=submit, duration_per_job=duration_per_job,
                             queueing_system=queueing_system,
                             do_not_doubly_submit_to_queue=do_not_doubly_submit_to_queue,
                             initialize_first_run=initialize_first_run, reinitialize=reinitialize, randomize=randomize, noncontinous=noncontinous,
                             memory = memory,
                             verbose=verbose)


    return job_id

# MAIN Execution from BASH 
if __name__ == "__main__":
    from reeds.function_libs.utils.argument_parser import execute_module_via_bash
    print(spacer + "\t\tRE-EDS S-OPTIMIZATION \n" + spacer + "\n")
    requiers_gromos_files = [("in_top_path", "input topology .top file."),
                             ("in_coord_path", "input coordinate .cn file."),
                             ("in_perttop_path", "input perturbation topology .ptp file."),
                             ("in_disres_path", "input distance restraint .dat file.")]
    execute_module_via_bash(__doc__, do, requiers_gromos_files)


