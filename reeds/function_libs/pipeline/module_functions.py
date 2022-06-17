import copy
import glob
import os
import typing as t
from inspect import signature, _empty
from numbers import Number
from typing import Tuple, List, Union, Dict

from pygromos.euler_submissions import FileManager as fM
from pygromos.files.imd import Imd
from pygromos.files.blocks import imd_blocks
from pygromos.files.coord.cnf import Cnf
from pygromos.utils import bash
from pygromos.files._basics.parser import read_ptp
from pygromos.data.imd_templates import template_md
from reeds.function_libs.pipeline.jobScheduling_scripts import RE_EDS_simulation_scheduler
from reeds.function_libs.pipeline.worker_scripts.analysis_workers import RE_EDS_general_analysis, \
    RE_EDS_general_analysis as reeds_analysis
from reeds.function_libs.utils import s_log_dist
from reeds.function_libs.utils.structures import optimization_job, spacer
from reeds.function_libs.utils.structures import optimization_params

"""
IMD template adaptations
"""

def initialize_imd(system: fM.System, imd_out_path: str = "template.imd", single_bath: bool = False,
                   non_ligand_residues: list = [], verbose: bool = True):
    """
            This function initialises a template imd for the very beginning of the pipeline

    Parameters
    ----------
    system : pygromos.PipelineManager.Simulation_System
        this is the system obj. containing all paths to all system relevant Files.
    imd_out_path : str
        path where the new .imd should be written
    non_ligand_residues: List[str], optional
        non ligand residues are not considered as possible states for the eds simulation, will not have their own force group
    single_bath : bool, optional
        use single bath for all atoms (default False)

    verbose : bool, optional
        I wanna screaaaam

    Returns
    -------
    Imd
        returns imd_class obj

    """

    cnf = Cnf(system.coordinates, verbose=False)
    raw_residues = cnf.get_residues()
    # TODO: Get ligands (or rather, perturbed residues) vs non-ligands from ptp file instead of deducing them from cnf?
    residues, ligands, protein, non_ligands = Imd.clean_residue_list_for_imd(raw_residues, non_ligand_residues)
    if (verbose): print("counted residues, ligands, proteins etc.", )
    if (verbose): print("\n all_resis:\n ", residues,
                        "\n\n ligands:\n ", ligands,
                        "\n\n protein:\n ", protein,
                        "\n\n non-ligands:\n ", non_ligands, "\n\n")
    if(single_bath):
        solvent_bath = (ligands.number_of_atoms + protein.number_of_atoms + residues["SOLV"])
        temp_baths = {solvent_bath: 1}

    ##Define temperature baths
    elif (protein.number_of_atoms > 0):
        if(len(non_ligand_residues) > 0):
            solvent_bath = (ligands.number_of_atoms + protein.number_of_atoms + non_ligands.number_of_atoms)
        elif (isinstance(non_ligands, type(None))):  # TODO: DIRTY HACK: in PNMT is Cofactor at end of the file.
            solvent_bath = (ligands.number_of_atoms + protein.number_of_atoms + residues["SOLV"])
        else:
            solvent_bath = (ligands.number_of_atoms + protein.number_of_atoms + non_ligands.number_of_atoms + residues[
                "SOLV"])

        if (max(ligands.positions) < protein.position):
            temp_baths = {ligands.number_of_atoms: 1, (ligands.number_of_atoms + protein.number_of_atoms): 2,
                          solvent_bath: 3}
        else:
            temp_baths = {protein.number_of_atoms: 1, (ligands.number_of_atoms + protein.number_of_atoms): 2,
                          solvent_bath: 3}
    else:
        temp_baths = {ligands.number_of_atoms: 1, (ligands.number_of_atoms + residues["SOLV"]): 2} if (
                    "SOLV" in residues) else {ligands.number_of_atoms: 1}

    all_atoms = ligands.number_of_atoms + protein.number_of_atoms + non_ligands.number_of_atoms
    if ("SOLV" in residues):
        all_atoms += residues["SOLV"]
    
    imd = Imd(template_md)

    imd.TITLE.content = system.name + "\nGeneral imd template for the RE-EDS pipeline"
    atoms_per_solv = 3 # Assumes SOLV is water
    imd.SYSTEM.NSM = int(residues["SOLV"] / atoms_per_solv) if ("SOLV" in residues) else 0

    # Deal with restraints
    if system.top.posres_path:
        imd.add_block(block=imd_blocks.POSITIONRES(1,1,0,500))
    if system.top.disres_path:
        imd.add_block(block=imd_blocks.DISTANCERES(1,0,5000,1,1,0,0,0))
    
    if (not isinstance(imd.MULTIBATH, type(None))):
        imd.MULTIBATH.adapt_multibath(last_atoms_bath=temp_baths)
    if (verbose): print("TEMPBATHS: ",temp_baths)
    imd.FORCE.adapt_energy_groups(residues)

    if (verbose):
        print("Imd has been initialized. Please make any additional changes to "+imd_out_path+" before the next steps of the pipeline")
    imd.write(imd_out_path)

def adapt_imd_template_emin(system: fM.System, in_template_imd_path: str, out_imd_dir: str)->(str, list):
    """
      modify_imds for an energy minmization for each of the end states
        This function prepares the imd. protocol file for gromos simulation.

    Parameters
    ----------
    system : pygromos.PipelineManager.Simulation_System
        this is the system obj. containing all paths to all system relevant Files.
    in_template_imd_path : str
        path to the template imd, prepared with initialize_imd
    out_imd_dir : str
        output path for the imds

    Returns
    -------
    str, list
        imd_template_path, states_num
    """

    states_num = int(read_ptp(system.top.perturbation_path)['MPERTATOM']['NPTB'])
    imd = Imd(in_template_imd_path)
    
    # Remove blocks that shouldn't be there, add blocks that should be there
    del imd.COMTRANSROT
    del imd.PRESSURESCALE 
    
    emin_options = {'NTEM':1, 'NCYC':0, 'DELE':0.1, 'DX0':0.01, 'DXM':0.05, 'NMIN':10000, 'FLIM':0.0}
    imd.add_block(blocktitle="ENERGYMIN", content=emin_options) 
 
    # edit EDS part
    imd_template_path = out_imd_dir + "/emin_state"
    s_values = [1.0 for x in range(states_num)]
    for state, s_values in zip(range(1, states_num + 1), s_values):
        imd.edit_EDS(NUMSTATES=states_num, S=s_values,
                          EIR=[500 if (x == state) else -500 for x in range(1, states_num + 1)])
        imd.write(imd_template_path + "_" + str(state) + ".imd")

    return imd_template_path, states_num

def adapt_imd_template_optimizedState(system: fM.System, in_template_imd_path: str, out_imd_dir: str,
                                      randomize: bool = False, simulation_steps: int = 100000)->(str, list):
    """
      modify_imds - for generateOptimizedStates
        This function prepares the imd. protocol file for gromos simulation.

    Parameters
    ----------
    system : pygromos.PipelineManager.Simulation_System
        this is the system obj. containing all paths to all system relevant Files.
    in_template_imd_path : str
        path to the template imd, prepared with initialize_imd
    out_imd_dir : str
        output path for the imds
    randomize: bool, optional
        randomize initial velocities
    simulation_steps : int, optional
        number of steps to simulate

    Returns
    -------
    str, list
        imd_template_path, states_num
    """

    states_num = int(read_ptp(system.top.perturbation_path)['MPERTATOM']['NPTB'])
    imd = Imd(in_template_imd_path)
    imd.STEP.NSTLIM = simulation_steps
    if(randomize): imd.randomize_seed()
    # edit EDS part
    imd_template_path = out_imd_dir + "/opt_structs_state"
    s_values = [1.0 for x in range(states_num)]
    for state, s_values in zip(range(1, states_num + 1), s_values):
        imd.edit_EDS(NUMSTATES=states_num, S=s_values,
                          EIR=[500 if (x == state) else -500 for x in range(1, states_num + 1)])
        imd.write(imd_template_path + "_" + str(state) + ".imd")

    return imd_template_path, states_num


def adapt_imd_template_lowerBound(system: fM.System, in_template_imd_path: str, out_imd_dir: str, randomize: bool = False,
                                  s_values: t.List[float] = None, simulation_steps: int = 1000000)->(str, list, int):
    """
        This function prepares the imd. protocol file for a gromos simulation.

    Parameters
    ----------
    system : pygromos.PipelineManager.Simulation_System
        this is the system obj. containing all paths to all system relevant Files.
    in_template_imd_path : str
        path to the template imd, prepared with initialize_imd
    out_imd_dir : str
        output path for the imds
    randomize : bool, optional
        randomize initial velocities
    s_values : List[float], optional
        range of values to use instead of the automatically generated log distribution
    simulation_steps : int, optional
        number of steps to simulate

    Returns
    -------
    str, list
        imd_template_path, s_values

    """

    states_num = int(read_ptp(system.top.perturbation_path)['MPERTATOM']['NPTB'])
    imd = Imd(in_template_imd_path)
    imd.STEP.NSTLIM = simulation_steps
    if(randomize): imd.randomize_seed()

    imd_template_path = out_imd_dir + "/lower_bound_sval"

    if (type(s_values) == type(None)):
        s_values = s_log_dist.get_log_s_distribution_between(start=1.0, end=0.00001)

    for ind, sval in enumerate(s_values):
        imd.edit_EDS(NUMSTATES=states_num, S=sval, EIR=[0.0 for x in range(states_num)])
        imd.write(imd_template_path + "_" + str(ind+1) + ".imd")

    return imd_template_path, s_values


def adapt_imd_template_eoff(system: fM.System, imd_out_path: str, in_template_imd_path: str,
                            input_svals: List[float] = None, randomize: bool = False, verbose: bool = True) -> Imd:
    """
            This function is preparing the imd_template in gromos_files for the REEDS SYSTEM>

    Parameters
    ----------
    system : pygromos.PipelineManager.Simulation_System
        this is the system obj. containing all paths to all system relevant Files.
    imd_out_path : str
        path, where the new .imd should be written to.
    imd_path : str
        imd template path that shall be modified
    input_svals : List[float], optional
        list containing previously generated s_values
    randomize : bool, optional
        randomize initial velocities

    verbose : bool, optional
        I can talk :)

    Returns
    -------
    Imd
        returns imd_class obj

    """

    svals=input_svals
    if verbose: print(str(len(svals)) + " SVALUES: " + str(svals))
    
    # BUILD IMD
    imd = Imd(in_template_imd_path)  # reads IMD
    states_num = int(read_ptp(system.top.perturbation_path)['MPERTATOM']['NPTB'])
    if(randomize): imd.randomize_seed()
    # build REEDS Block
    imd.add_block(block=imd_blocks.NEW_REPLICA_EDS(REEDS=1, NRES=len(svals), NUMSTATES=states_num, NEOFF=len(svals),
                                                    RES=svals, EIR=0, NRETRIAL=10, NREQUIL=0, EDS_STAT_OUT=1,
                                                    CONT=1, PERIODIC=0))
    imd.edit_REEDS(EIR=0.0)
    imd_out_path = imd.write(imd_out_path)
    if (verbose): print(imd.REPLICA_EDS)
    return imd


"""
    OPTIMIZATION - Scheduling
"""
def build_optimization_step_dir(iteration: int,  iteration_folder_prefix: str, pot_tresh: List[float],
                                optimization_options:optimization_params, in_simSystem: fM.System,
                                state_undersampling_pot_tresh:List[float], state_physical_pot_tresh:List[float], undersampling_frac_thresh:float,
                                num_equilibration_runs: int, imd_name_prefix: str,
                                in_ene_ana_lib_path: str, in_gromosPP_bin_dir: str,
                                in_gromosXX_bin_dir: str, ligands, last_data_folder: str, nmpi_per_replica: int,
                                duration_per_job: str, num_simulation_runs: int,
                                run_NLRTO:bool=True, run_NGRTO: bool=False, run_eoffRB:bool = False,
                                optimized_states_dir:str = "../../a_optimizedState/analysis/next",
                                memory: int = None,
                                old_sopt_job: optimization_job = False, verbose: bool = False) -> optimization_job:
    """
        This function is setting up the folder structure of an s-optimization iteration, copies some files and builds an settings object of the sopt-iteration.

    Parameters
    ----------
    iteration : int
        iteration of the s-optimization
    iteration_folder_prefix : str
        prefix name of the folders
    optimization_options : ?
        settings for the s-optimization
    in_simSystem : fM.System
        system obj, containing all required paths
    num_equilibration_runs : int
        number of equilibrations before the used s-optimizatoin simulation
    imd_name_prefix : str
        prefix name for the imd file output, final name will have an additional iteration number.
    in_ene_ana_lib_path : str
        path to an ene_ana lib
    in_gromosPP_bin_dir : str
        path to gromosPP binary dir
    in_gromosXX_bin_dir : str
        path to gromosXX binary dir
    optimized_states_dir : str, optional
        path to the optimized state results
    last_data_folder : str
        data folder from previous rund
    nmpi_per_replica : int
        mpi cores per simulatoin
    duration_per_job : str
        queue job duration (format "HHH:MM")
    num_simulation_runs : int
        number of simulation runs
    pot_tresh : float, optional
        potential threshold for occurrence sampling
    old_sopt_job : sopt_job, optional
        last soptimization job namedtuple, contains all settings of previous run
    memory : int, optional
        how much memory to use for job submission
    verbose : bool, optional
        TADAAAAAAAAAAa Here you go!

    Returns
    -------
    sopt_job
        returns the sopt settings of this runs
    """
    # PATHS
    iteration_folder = iteration_folder_prefix + str(iteration)

    input_dir = iteration_folder + "/input"
    coord_dir = input_dir + "/coord"
    sim_dir = iteration_folder + "/simulation"
    analysis_dir = iteration_folder + "/analysis"
    pre_in_imd_path = input_dir + "/" + imd_name_prefix + str(iteration) + ".imd"
    final_in_imd_path = input_dir + "/" + imd_name_prefix + str(iteration) + ".imd"

    ##make folders:
    if (verbose): print("\tGenerating Folders")
    bash.make_folder(iteration_folder)
    bash.make_folder(input_dir)
    bash.make_folder(coord_dir)

    ##copy coordinates & imd:
    if (verbose): print("preparing files: imd & cnf\t", iteration)
    if (iteration == 1):
        print("FIRST ITERATION: ")
        need_to_be_created = False
        pre_in_imd_path = bash.copy_file(glob.glob(last_data_folder + "/*imd")[0],
                                         input_dir + "/" + in_simSystem.name + ".imd")
        in_simSystem.move_input_coordinates(coord_dir)
        last_coord_in = in_simSystem.coordinates[0]
    else:
        # the * helps for later, so no error is thrown
        need_to_be_created = True
        pre_in_imd_path = last_data_folder + "/next*.imd"
        last_coord_in = old_sopt_job.check_analysis_files
        last_coord_file_path = input_dir + "/coord/" + old_sopt_job.sim_system.name + "*.cnf"  # "_"+str(old_sopt_job.num_simulation_runs+old_sopt_job.num_equilibration_runs)+
        setattr(in_simSystem, "coordinates", last_coord_file_path)
        print(last_coord_in)

    if (verbose): print("COORD: ", in_simSystem.coordinates)
    if (verbose): print("IMD: ", pre_in_imd_path)

    # PARAMS: 
    ## fix for euler! - write out to workdir not on node. - so no data is lost in transfer

    workdir = None

    nmpi = int(optimization_options.current_num_svals) * int(nmpi_per_replica)  # How many MPIcores needed?s

    ##which analysis functions to execute
    control_dict = {  # this dictionary is controlling the post  Simulation analysis procedure!
        "sopt": {"do": run_NLRTO or run_NGRTO,
                 "sub": {
                     "run_RTO": True,
                     "run_NLRTO": run_NLRTO,
                     "run_NGRTO": run_NGRTO,
                     "visualize_transitions": True,
                     "roundtrips": True,
                     "generate_replica trace": True}
                 },
        "eoffset": {"do": True if run_eoffRB else False,
                    "sub": {
                        "eoff_estimation": not run_eoffRB,
                        "sampling_plot": False,
                        "eoffset_rebalancing": run_eoffRB,
                        }
                    },
        "prepare_input_folder": {"do": True,
                                 "sub": {
                                     "eoff_to_sopt": False,
                                     "write_eoffRB": run_eoffRB,
                                     "write_eoffEstm": False,
                                     "write_s": run_NLRTO or run_NGRTO
                                 },
                                 }
    }

    # BUILD FILES
    ##Build analysis_script
    from collections import OrderedDict
    if (verbose): print("Analysis Script")
    analysis_vars = OrderedDict({
        "in_folder": sim_dir,
        "in_imd": final_in_imd_path,
        "topology": in_simSystem.top.top_path,
        "optimized_eds_state_folder": optimized_states_dir,
        "out_folder": analysis_dir,
        "gromos_path": in_gromosPP_bin_dir,
        "in_ene_ana_lib": in_ene_ana_lib_path,
        "n_processors": 5,
        "state_physical_occurrence_potential_threshold": state_physical_pot_tresh,
        'eoffRebalancing_learningFactor': optimization_options.learningFactor,
        'eoffRebalancing_pseudocount': optimization_options.pseudocount,
        'eoffRebalancing_correctionPerReplica': optimization_options.eoffRB_correctionPerReplica,
        "state_undersampling_occurrence_potential_threshold": state_undersampling_pot_tresh,
        "undersampling_frac_thresh": undersampling_frac_thresh,
        "verbose": True,
        "add_s_vals": optimization_options.add_replicas,
        "control_dict": control_dict,
        "title_prefix": in_simSystem.name,
        "grom_file_prefix": in_simSystem.name,
    })
    in_analysis_script_path = write_job_script(out_script_path=iteration_folder + "/job_analysis.py",
                                               target_function=RE_EDS_general_analysis.do_Reeds_analysis,
                                               variable_dict=analysis_vars)

    ##Build Job Script
    if (verbose): print("Scheduling Script")
    # for job_scheduling needed in local variables
    out_dir_path = iteration_folder
    gromosXX_bin_dir = in_gromosXX_bin_dir
    jobname = in_simSystem.name
    in_imd_path = pre_in_imd_path
    out_dir_path = sim_dir

    schedule_jobs_script_path = write_job_script(out_script_path=iteration_folder + "/schedule_optimization_jobs.py",
                                                 target_function=RE_EDS_simulation_scheduler.do,
                                                 variable_dict=locals())

    ##output folders
    check_simulation_files = sim_dir + "/*" + str(num_simulation_runs + num_equilibration_runs) + "/*cnf"
    check_analysis_files = analysis_dir + "/next/*.cnf"

    last_iteration_job = optimization_job(iteration=iteration, job_file_path=schedule_jobs_script_path,
                                          job_analysis_path=in_analysis_script_path,
                                          check_simulation_files=check_simulation_files,
                                          check_analysis_files=check_analysis_files,
                                          sim_system=copy.deepcopy(in_simSystem), nmpi=nmpi,
                                          num_simulation_runs=num_simulation_runs,
                                          num_equilibration_runs=num_equilibration_runs,
                                          workdir=workdir, in_imd_path=pre_in_imd_path, out_folder=sim_dir,
                                          last_coord_in=last_coord_in)

    return last_iteration_job


def submit_iteration_job(last_iteration_job: optimization_job, duration_per_job: str, submit: bool = True,
                         stdout_prefix: str = "\n\t\t", memory: int = None,
                         gromosXX_bin_dir: str = None, do_not_doubly_submit_to_queue: bool = True,
                         previous_job_id: int = None, initialize_first_run: bool = True,
                         reinitialize: bool = False, verbose: bool = False) -> Union[int, None]:
    """

        This function is doing the little bit more sophisticated part of submitting safely the sopt scripts.


    Parameters
    ----------
    last_iteration_job : sopt_job - NamedTuple("sopt_Job", [("iteration", int),("job_file_path", str), ("job_analysis_path",str), ("check_simulation_files",str), ("check_analysis_files",str)])
        this namedtuple contains both executable script paths and the regular expressions for their checkFiles.
    duration_per_job : str
        job duration for a single simulation step in format "HHH:MM"
    submit : bool, optional
        shall this simulation be submitted to LSF?
    stdout_prefix: str, optional
        std-out log file prefix
    gromosXX_bin_dir: str, optional
        path to the gromos binary dir.
    do_not_doubly_submit_to_queue : bool , optional
        check if there is a queued job with the same job-name
    previous_job_id : int , optional
        id from the previous job onto which the job is queued to.
    initialize_first_run : bool, optional
        initialize the velocities of the first run?
    reinitialize : bool, optional
        reinitialize velocities in all runs?
    memory : int, optional
        how much memory to reserve for submission
    verbose : bool, optional
        let me tell you a looong story

    Returns
    -------
    int
        the jobID or 0 is returned.

    Raises
    ------
    ValueError
        if system call fails
    """
    # job_file_path Submission:
    check_simulation_files = glob.glob(last_iteration_job.check_simulation_files)
    check_analysis_files = glob.glob(last_iteration_job.check_analysis_files)

    if (len(check_simulation_files) > 0 or len(
            check_analysis_files) > 0):  # check if control Files are present(sim already done?)
        if not (len(check_analysis_files) > 0) and submit:  # if analysis is meassing do>:
            if (verbose): print(stdout_prefix + "SKIP JOB (inModule), as I found: \n\t",
                                last_iteration_job.check_simulation_files)
            if (verbose): print(stdout_prefix + "RUNNING MISSING ANALYSIS NOW:")
            bash.execute(last_iteration_job.job_analysis_path)
        else:
            if (verbose): print(stdout_prefix + "SKIP SUBMITTING JOB AND ANALYSIS!(inModule) Found Check Files!\n\t",
                                last_iteration_job.check_simulation_files,
                                "\n\t", last_iteration_job.check_analysis_files)
            return None

    elif submit:  # if no checkfiles present, submit job to the queue
        print(spacer + stdout_prefix + "\n\tSUBMITTING:  Iteration " + str(last_iteration_job.iteration) + "\n" + spacer)
        if verbose: print(stdout_prefix + "JOB:")

        orig_path = os.getcwd()
        job_dir = os.path.dirname(last_iteration_job.job_file_path)
        sim_system = last_iteration_job.sim_system

        if (verbose): print("EQ: ", last_iteration_job.num_equilibration_runs)

        # INITIAL COMMAND
        cmd = ""
        print("SETTING INITIAL CMD")
        in_imd_path = job_dir + "/input/" + sim_system.name + "*.imd"
        if (not os.path.exists(in_imd_path) and last_iteration_job.in_imd_path != in_imd_path.replace("*", "")):
            cmd = "cp " + last_iteration_job.in_imd_path + " " + in_imd_path.replace("*",
                                                                                     "") + " && cp " + os.path.dirname(
                last_iteration_job.in_imd_path) + "/*cnf " + job_dir + "/input/coord"

        os.chdir(job_dir)
        job_id = RE_EDS_simulation_scheduler.do(in_simSystem=sim_system, in_imd_path=in_imd_path,
                                                previous_job_ID=previous_job_id,
                                                gromosXX_bin_dir=gromosXX_bin_dir,
                                                out_dir_path=last_iteration_job.out_folder, jobname=sim_system.name,
                                                nmpi=last_iteration_job.nmpi,
                                                duration_per_job=duration_per_job, initial_command=cmd,
                                                num_simulation_runs=last_iteration_job.num_simulation_runs,
                                                work_dir=last_iteration_job.workdir,
                                                num_equilibration_runs=last_iteration_job.num_equilibration_runs,
                                                in_analysis_script_path=last_iteration_job.job_analysis_path,
                                                do_not_doubly_submit_to_queue=do_not_doubly_submit_to_queue,
                                                initialize_first_run=initialize_first_run,
                                                reinitialize=reinitialize,
                                                memory = memory,
                                                verbose=True)

        if verbose: print("process returned id: " + str(job_id))
        if (job_id < 0):
            raise ChildProcessError("Got an error from scheduler function!")

        if (job_id == "" and job_id.isalnum()):
            raise ValueError("Did not get at job ID!")

        os.chdir(orig_path)

        return job_id
    else:
        print(stdout_prefix + "SKIP SUBMITTING! \n\tDUE to --noSubmit option")
        return 0

"""
    General Functions
"""
def write_job_script(out_script_path: str, target_function: callable, variable_dict: dict, python_cmd: str = "python3",
                      no_reeds_control_dict: bool = False, verbose: bool = False) -> str:
    """
        this function writes submission commands into a file. The command will be started from a bash env into python.


    Parameters
    ----------
    out_script_path: str
        path of the output script.
    target_function : callable
        the function, that shall be submitted
    variable_dict : dict
        variables for this function
    python_cmd : str, optional
        which python command shall be supplied
    no_reeds_control_dict : bool, optional
        there is no controldict in the vars.
    verbose : bool, optional
        c'est la vie

    Returns
    -------
    str
        returns an out script path.

    Raises
    ------
    IOERROR
        if outpath is not possible
    ValueError
        if required variable from the var-dict for the function is missing
    """

    if (not os.path.exists(os.path.dirname(out_script_path))):
        raise IOError(
            "Could not find path of dir, that should contain the schedule script!\n\t Got Path: " + out_script_path)

    # Build str:
    s = signature(target_function)
    import_string = "#IMPORTS\n"
    import_string += "from " + str(target_function.__module__) + " import " + target_function.__name__
    vars_string = "#VARIABLES: \n"
    cmd_options = ""

    missed_keys = []
    for key in s.parameters:
        if (key in variable_dict):
            value = variable_dict[key]
            if (key == "in_simSystem"):  # this is a nasty way! ... tends to fail! MAKE SURE CALL POINT HAS a variable like this!
                sys = value
                vars_string += sys.get_script_generation_command(var_name=key, var_prefixes="system")
            elif (isinstance(value, Dict)):
                if (key == "control_dict"):
                    if (no_reeds_control_dict):
                        vars_string += reeds_analysis.dict_to_nice_string(value)
                    else:
                        vars_string += reeds_analysis.dict_to_nice_string(reeds_analysis.check_script_control(value))
                else:
                    vars_string += reeds_analysis.dict_to_nice_string(value)
            elif (isinstance(value, List)):
                vars_string += key + "= [ " + ", ".join(map(str, value)) + "]\n"
            elif (isinstance(value, str)):
                vars_string += key + " = \"" + str(value) + "\"\n"
            else:
                vars_string += key + " = " + str(value) + "\n"
            cmd_options += key + "=" + key + ", "
        elif (s.parameters[key].default == _empty):
            missed_keys.append(key)

    if (len(missed_keys) > 0):
        raise ValueError(
            "Found some variables missing in variable dict,that are required!\n\t" + "\n\t".join(missed_keys))

    cmd_string = "\n#DO\n"
    cmd_string += target_function.__name__ + "(" + cmd_options + ")"

    script_text = "#!/usr/bin/env " + python_cmd + "\n\n" + import_string + "\n\n" + vars_string + "\n\n" + cmd_string + "\n"
    if (verbose): print(script_text)

    # write out file
    out_script_file = open(out_script_path, "w")
    out_script_file.write(script_text)
    out_script_file.close()

    return out_script_path
