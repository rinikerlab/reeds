import os, sys, copy, traceback, warnings
from collections import OrderedDict
from typing import List, Iterable

from pygromos.euler_submissions import FileManager as fM
from pygromos.euler_submissions.Submission_Systems import _SubmissionSystem, LSF, SLURM
from pygromos.files import imd
from pygromos.files.coord import cnf as cnf_cls
from pygromos.utils import bash
from reeds.data import ene_ana_libs
from reeds.function_libs.pipeline.module_functions import build_optimization_step_dir, submit_iteration_job, write_job_script
from reeds.function_libs.pipeline.worker_scripts.analysis_workers import RE_EDS_optimization_final
from reeds.function_libs.utils.structures import adding_Scheme_new_Replicas, optimization_params


def do_optimization(out_root_dir: str, in_simSystem: fM.System, optimization_name:str, in_template_imd: str = None,
                    iterations: int = 4,
                    eoffEstimation_undersampling_fraction_threshold: float = 0.9,
                    sOpt_add_replicas: int = 4, sOpt_adding_new_sReplicas_Scheme: adding_Scheme_new_Replicas = adding_Scheme_new_Replicas.from_below,
                    run_NLRTO: bool = False, run_NGRTO: bool = True, run_eoffRB: bool=False,
                    eoffRB_learningFactors:List[float]=None, eoffRB_pseudocount:float=None,
                    eoffRB_correctionPerReplica: bool=False,
                    non_ligand_residues: list = [],
                    state_physical_occurrence_potential_threshold:List[float]=None,
                    state_undersampling_occurrence_potential_threshold: List[float]=None,
                    equil_runs: int = 1, prod_runs: int = 1, steps_between_trials: int = 50, trials_per_run: int = 12500,
                    optimized_states_dir: str = os.path.abspath("a_optimizedState/analysis/next"),
                    lower_bound_dir: str = os.path.abspath("b_lowerBound/analysis/next"),
                    in_gromosXX_bin_dir: str = None, in_gromosPP_bin_dir: str = None,
                    in_ene_ana_lib_path: str = ene_ana_libs.ene_ana_lib_path,
                    nmpi_per_replica: int = 1, submit: bool = True, duration_per_job: str = "24:00:00",
                    queueing_system: _SubmissionSystem = SLURM,
                    do_not_doubly_submit_to_queue: bool = True,
                    initialize_first_run: bool = True, reinitialize: bool = False, randomize: bool=False, noncontinous: bool = False,
                    memory: int = None,
                    ssm_next_cnf: bool = False,
                    verbose: bool = True):


    if run_NGRTO and run_NLRTO:
        warnings.warn("You have asked for optimization with both LRTO and GRTO algorithms.\nPlease note that priority is given to the GRTO algorithm (preparation of output for future iterations).")
        print ("Output of the LRTO will still be printed in path/to/analysis/s_optimization/ for comparison.")
    try:
        simSystem = copy.deepcopy(in_simSystem)
        sopt_input = out_root_dir + "/input"

        if (not os.path.exists(sopt_input)):
            os.mkdir(sopt_input)

        if(eoffRB_learningFactors is None):
            eoffRB_learningFactors = [1 for _ in range(iterations)]

        # retrieve old Information:a
        last_data_folder = os.path.dirname(
            simSystem.coordinates[0])  # the input coord file should also contain the initial .imd file!

        if (isinstance(in_template_imd, type(None))):
            imd_path_last = last_data_folder + "/next.imd"
            if not (os.path.exists(imd_path_last)):
                raise IOError("could not find initial IMD with path: \n\t" + last_data_folder + "/next.imd")
        else:
            imd_path_last = in_template_imd
        imd_file = imd.Imd(imd_path_last)

        if (isinstance(simSystem.coordinates, Iterable)):
            new_coords = []
            # if single cnf file, make copies to match number of replicas
            if len(simSystem.coordinates)==1:
                svals = imd_file.REPLICA_EDS.RES
                for i in range(len(svals)):
                    new_file_path = bash.copy_file(simSystem.coordinates[0],
                                               sopt_input + "/" + "REEDS_coordinate_replica_" + str(i+1) + ".cnf")
                    new_coords.append(new_file_path)
            else:
                for coordinate_file_path in simSystem.coordinates:
                    new_file_path = bash.copy_file(coordinate_file_path,
                                               sopt_input + "/" + os.path.basename(coordinate_file_path))
                    new_coords.append(new_file_path)
            setattr(simSystem, "coordinates", new_coords)        

        else:
            raise IOError(
                "Could not copy system coordinates into input folder! please give them as List[str]. \n GOT: " + str(
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

        # get number of s-vals from imd:
        num_svals = int(imd_file.REPLICA_EDS.NRES)
        num_states = int(imd_file.REPLICA_EDS.NUMSTATES)
        # set new step number between trials and new number of trials if necessary
        imd_file.STEP.NSTLIM = steps_between_trials
        imd_file.edit_REEDS(NRETRIAL=trials_per_run)
        if (randomize): imd_file.randomize_seed()
        imd_path_last = imd_file.write(sopt_input + "/repex_template.imd")

        # Setup s-optimization
        ## global vars
        imd_name_prefix = simSystem.name + "_"
        iteration_folder_prefix = out_root_dir + "/" + optimization_name
        bash.make_folder(out_root_dir)

        state_undersampling_pot_tresh_path = optimized_states_dir + "/state_occurence_physical_pot_thresh.csv"
        if (state_physical_occurrence_potential_threshold is None and os.path.exists(
                state_undersampling_pot_tresh_path)):
            if not os.path.exists(state_undersampling_pot_tresh_path):
                raise IOError("COULD NOT FIND state_occurence_pot_thresh.CSV in : ", state_undersampling_pot_tresh_path,
                              "\n")
            else:
                tmp = open(state_undersampling_pot_tresh_path, "r")
                state_physical_occurrence_potential_threshold = list(map(float, " ".join(tmp.readlines()).split()))
        elif (state_physical_occurrence_potential_threshold is None):
            state_physical_occurrence_potential_threshold = [0 for x in range(num_states)]

        state_undersampling_pot_tresh_path = lower_bound_dir + "/state_occurence_pot_thresh.csv"
        if (state_undersampling_occurrence_potential_threshold is None and os.path.exists(
                state_undersampling_pot_tresh_path)):
            if not os.path.exists(state_undersampling_pot_tresh_path):
                raise IOError("COULD NOT FIND state_occurence_pot_thresh.CSV in : ", state_undersampling_pot_tresh_path,
                              "\n")
            else:
                tmp = open(state_undersampling_pot_tresh_path, "r")
                state_undersampling_occurrence_potential_threshold = list(map(float, " ".join(tmp.readlines()).split()))
        elif (state_undersampling_occurrence_potential_threshold is None):
            state_undersampling_occurrence_potential_threshold = [0 for x in range(num_states)]


    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR in file preperations of iteration()")
        print("#####################################################################################")
        print("\n".join(map(str, err.args)))
        print()
        traceback.print_exception(*sys.exc_info())
        raise Exception("ERROR in file preperations of iteration()")
    #################
    # Prepare each sopt iteration
    #################
    ## Loop vars
    job_id = None  # id for chaining
    repetitions = prod_runs  # needed to elongate simulation length
    standard_name = simSystem.name
    iteration_sopt_job = None
    add_replicas_mode = sOpt_add_replicas
    cur_svals = num_svals
    
    ## Prepare final analysis:
    ana_out_dir = out_root_dir + "/analysis"
    job_name = in_simSystem.name + "_final_" + optimization_name
    analysis_vars = OrderedDict({
        "project_dir": out_root_dir,
        "title": in_simSystem.name,
        "state_physical_occurrence_potential_threshold": state_physical_occurrence_potential_threshold,
        "optimization_name": optimization_name,
        "out_dir": ana_out_dir
    })
    
    in_final_analysis_script_path = write_job_script(
        out_script_path=out_root_dir + "/job_final_analysis.py",
        target_function=RE_EDS_optimization_final.do,
        variable_dict=analysis_vars)
    
    bash.execute("chmod +x " + in_final_analysis_script_path)  # make executables
    
    os.mkdir(ana_out_dir)

    # generate each iteration folder && submission
    for iteration in range(1, iterations + 1):
        print("\n\nITERATION: " + str(iteration))

        if (add_replicas_mode == -1):  # doubleing case
            sOpt_add_replicas = cur_svals - ligands.number

        # note: current_num_svals is the number of svalues used for the analysis, i.e. the
        # number of svalues *before* the optimization
        optimization_options = optimization_params(learningFactor=eoffRB_learningFactors[iteration - 1],
                                                   pseudocount=eoffRB_pseudocount,
                                                   eoffRB_correctionPerReplica=eoffRB_correctionPerReplica,
                                                   add_replicas=sOpt_add_replicas, adding_new_sReplicas_Scheme=sOpt_adding_new_sReplicas_Scheme,
                                                   current_num_svals=cur_svals)
        # increase cur_svals by add_replicas so it can be used to define soptimization_options in the next iteration
        cur_svals = cur_svals + sOpt_add_replicas

        simSystem.name = standard_name + "_" + str(iteration)
        try:  # JOB preperation
            iteration_sopt_job = build_optimization_step_dir(iteration=iteration,
                                                             iteration_folder_prefix=iteration_folder_prefix,
                                                             optimization_options=optimization_options,
                                                             num_equilibration_runs=equil_runs,
                                                             imd_name_prefix=imd_name_prefix, in_simSystem=simSystem,
                                                             in_ene_ana_lib_path=in_ene_ana_lib_path,
                                                             state_undersampling_pot_tresh=state_undersampling_occurrence_potential_threshold,
                                                             state_physical_pot_tresh=state_physical_occurrence_potential_threshold,
                                                             undersampling_frac_thresh=eoffEstimation_undersampling_fraction_threshold,
                                                             in_gromosPP_bin_dir=in_gromosPP_bin_dir,
                                                             in_gromosXX_bin_dir=in_gromosXX_bin_dir,
                                                             ligands=ligands, old_sopt_job=iteration_sopt_job,
                                                             last_data_folder=last_data_folder,
                                                             nmpi_per_replica=nmpi_per_replica,
                                                             run_NLRTO=run_NLRTO, run_NGRTO=run_NGRTO, run_eoffRB=run_eoffRB,
                                                             pot_tresh=state_physical_occurrence_potential_threshold,
                                                             duration_per_job=duration_per_job,
                                                             num_simulation_runs=repetitions,
                                                             memory = memory,
                                                             ssm_next_cnf = ssm_next_cnf, 
                                                             optimized_states_dir = optimized_states_dir)

        except Exception as err:
            print("#####################################################################################")
            print("\t\tERROR in file preperations of iteration()")
            print("#####################################################################################")
            print("\n".join(map(str, err.args)))
            print()
            traceback.print_exception(*sys.exc_info())
            raise Exception("ERROR in file preperations of iteration()")

        try:  # JOB SUBMISSION
            # print(iteration_sopt_job.job_file_path, iteration_sopt_job.job_analysis_path)
            # print("COORD IN SUB: ",  iteration_sopt_job.sim_system.coordinates)
            bash.execute(
                "chmod +x " + iteration_sopt_job.job_file_path + " " + iteration_sopt_job.job_analysis_path)  # make executables

            job_id = submit_iteration_job(iteration_sopt_job, gromosXX_bin_dir=in_gromosXX_bin_dir,
                                          duration_per_job=duration_per_job, submit=submit, previous_job_id=job_id,
                                          do_not_doubly_submit_to_queue=do_not_doubly_submit_to_queue,
                                          initialize_first_run=initialize_first_run, reinitialize=reinitialize,
                                          memory = memory,
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
            raise Exception("ERROR during job job-submissoin")
        
        if (iteration > 1):
            try:  # JOB SUBMISSION
                if (verbose): print("Final Analysis Script")
                
                qsys = queueing_system() # inittialize object
                
                _ = qsys.submit_to_queue(command=in_final_analysis_script_path,
                                         jobName=job_name + "_opt" + str(iteration),
                                         outLog=ana_out_dir + "/" + job_name + ".out",
                                         errLog=ana_out_dir + "/" + job_name + ".err",
                                         maxStorage=5000, queue_after_jobID=job_id,
                                         nmpi=1,
                                         verbose=verbose, 
                                         sumbit_from_file=True, 
                                         submit_from_dir = out_root_dir) 

            except Exception as err:
                print("#####################################################################################")
                print("\t\tERROR during job-final Analysis-submissoin")
                print("#####################################################################################")
                print("\n".join(map(str, err.args)))
                print()
                traceback.print_exception(*sys.exc_info())
                raise Exception("ERROR during job-final Analysis-submissoin")
    return job_id
