#!/usr/bin/env python3
"""
    Generate Optimized structures with EDS
  Description:
    This script executes simulations, that optimize the input EDS system in each state seperatley.
    The number of simulations is dependent on the ligands (endstates) of the system.
    Each simulation is biased towards one state via the Energy Offsets.
    After simulating the potentials energies get analysed and plotted.
    You should see, that in each simulation only one state is dominating the system. (e.g.: Check root_path_to_out_dir/analysis/sampling_undersampling.png)\n
    This output will be automatically used by do_RE_EDS_eoffEstimation.py(provide: "root_path_to_out_dir/simulation/")

    As input template_RE_EDS_project file the file: .../reeds/gromos_files/imd_templates/eds_md.imd

    Further optimisations suggestions, if you can not generate the optimized states:
        * use coordinates of this System in an a undersampling -> allows easier transitions
        * increase the absolute values of the EOFFs in .imd file -> ""
        * increase simulation duration, to get the desired transition

    Todo: enable protein useage in system (Problem with parameter file adaption. - protein should be one energygroup)

  Useage:
    * via commandline
       use: python do_RE_EDS_generateOptimizedStates.py -h to get input help.
    * via python
       for getting the input right see main part under the function.
   Author: Benjamin Schroeder

"""

import copy
import os
import sys
import traceback

import reeds.function_libs.pipeline.module_functions
from pygromos.euler_submissions import gen_Euler_LSF_jobarray, FileManager as fM
from pygromos.files.coord import cnf as cnf_cls
from pygromos.utils import bash
from reeds.data import imd_templates, ene_ana_libs
from reeds.function_libs.pipeline import generate_euler_job_files as gjs
from reeds.function_libs.pipeline.module_functions import adapt_imd_template_optimizedState
from reeds.function_libs.pipeline.worker_scripts.analysis_workers import RE_EDS_state_optimization_analysis as ana
from reeds.function_libs.utils.structures import spacer


def do(out_root_dir: str, in_simSystem: fM.System,
       in_imd_template_path: str = imd_templates.eds_md_path,
       in_gromosXX_bin_dir: str = None,
       in_gromosPP_bin_dir: str = None,
       ene_ana_lib: str = ene_ana_libs.ene_ana_lib_path,
       simulation_steps: int = 1000000, exclude_residues: list = [],
       nmpi_per_replica: int = 4, submit: bool = True, verbose: bool = True,
       solvent_keyword: str = "SOLV",
       vacuum_simulation:bool =False,
       single_bath: bool = False,
       memory: int = None, job_duration: str = "8:00") -> int:
    """      Generate Optimized structures with EDS

            This script executes simulations, that optimize the input EDS system in each state seperatley.
            The number of simulations is dependent on the ligands (endstates) of the system.
            Each simulation is biased towards one state via the Energy Offsets.
            After simulating the potentials energies get analysed and plotted.
            You should see, that in each simulation only one state is dominating the system. (e.g.: Check root_path_to_out_dir/analysis/sampling_undersampling.png)\n
            This output will be automatically used by do_RE_EDS_eoffEstimation.py(provide: "root_path_to_out_dir/simulation/")

            As input template_RE_EDS_project file the file: .../reeds/gromos_files/imd_templates/eds_md.imd

            Further optimisations suggestions, if you can not generate the optimized states:
                * use coordinates of this System in an a undersampling -> allows easier transitions
                * increase the absolute values of the EOFFs in .imd file -> ""
                * increase simulation duration, to get the desired transition


Parameters
----------
out_root_dir : str
    path to the root output dir, in which the script directories wild be build into
system :  pygromos.PipelineManager.Simulation_System
    System obj. that contains information about the System-Files (cnf, top,...0)
in_imd_template_path : str, optional
    gives the path to the template_RE_EDS_project parameter file (.imd), that is adapted to fit the system.
in_gromosXX_bin_dir : str, optional
    Flag, if the generated sopt_job should be submitted to lsf queue.
in_gromosPP_bin_dir : str, optional
    How many cpus shall be used per simulation?
ene_ana_lib : str, optional
    ene ana lib for the analysis
simulation_steps : int, optional
    how many simulation steps, shall be carried out?
exclude_residues : List[str], optional
    Which non protein residues shall be ignored and not treated as eds-state(cofactors)
job_duration : str, optional
    Duration of each submitted job in the queue (depends on imd setting)
nmpi_per_replica : int, optional
    how many mpi cores per job, shall be used?
vacuum_simulation : bool, optional
    this flag can be used for vacuum simulations
single_bath : bool, optional
    use a single bath for all atoms? (i.e. NBATHS = 1) (default False)
submit : bool, optional
    should the prepared scripts be executed? (or even queued)
verbose : bool, optional
    verbosity of the code


Returns
-------
int
    returnCode

    """
    #################
    # Prepare Jobs
    #################
    try:
        if (verbose): print(spacer + "START structure Optimization.")

        # paths
        input_dir = out_root_dir + "/input"
        coord_dir = input_dir + "/coord"
        sim_dir = out_root_dir + "/simulation"

        # make folders:
        if (verbose): print("Generating Folders")
        bash.make_folder(out_root_dir)
        bash.make_folder(input_dir)
        bash.make_folder(coord_dir)

        system = copy.deepcopy(in_simSystem)

        # generate parameter Files (.imd) and adapt them to the system.
        if (verbose): print("Writing parameter Files")
        cnf = cnf_cls.Cnf(system.coordinates,
                          clean_resiNumbers_by_Name=True)  # TODO: Careful with cleaning flag! protein is not correctly described.

        # prepare imd_templates
        imd_template_path, s_values, lig_num = adapt_imd_template_optimizedState(in_template_imd_path=in_imd_template_path,
                                                                                 out_imd_dir=input_dir, cnf=cnf,
                                                                                 non_ligand_residues=exclude_residues,
                                                                                 simulation_steps=simulation_steps,
                                                                                 solvent_keyword=solvent_keyword,
                                                                                 single_bath = single_bath)

        # copy and prepare cnfs:
        for state in range(1, lig_num + 1):
            bash.copy_file(system.coordinates, coord_dir + "/" + os.path.basename(system.coordinates).replace(".cnf",
                                                                                                              "_" + str(
                                                                                                                  state) + ".cnf"))

        cnf_array = coord_dir + "/" + os.path.basename(system.coordinates).replace(".cnf", "_${RUNID}.cnf")
        setattr(system, "coordinates", cnf_array)

        # GENERATE array scripts
        if (verbose): print("generating LSF-Bashscripts")

        ## build: worker_scripts-script - Old... but works :)
        worker_script = gen_Euler_LSF_jobarray.build_worker_script_multImds(
            out_script_path=input_dir + "/worker_scripts.sh", out_dir=sim_dir,
            job_name=system.name + "_work", in_system=system,
            in_imd_prefix=imd_template_path, cores=nmpi_per_replica, gromosXX_bin=in_gromosXX_bin_dir)

        ## build: analysis_script
        out_script_path = out_root_dir + "/analysis.py"
        if (verbose): print("\tgenerating analysis script: ", out_script_path)
        analysis_vars = {"out_analysis_dir": out_root_dir + "/analysis",
                         "in_topology_path": system.top.top_path,
                         "in_simulation_dir": sim_dir,
                         "in_imd_path": imd_template_path,
                         "numstates": lig_num,
                         "gromosPP_bin": in_gromosPP_bin_dir,
                         "vacuum_simulation": vacuum_simulation,
                         "in_ene_ana_lib": ene_ana_lib,
                         "in_topology": system.top.top_path
                         }
        print(out_script_path)
        in_analysis_script_path = reeds.function_libs.pipeline.module_functions.write_job_script(out_script_path=out_script_path, target_function=ana.do,
                                                                                                 variable_dict=analysis_vars)

        ## build: sopt_job array_schedule script - Old... but works :)
        job_array_script = gen_Euler_LSF_jobarray.build_jobarray(script_out_path=out_root_dir + "/job_array.sh",
                                                                 output_dir=sim_dir, duration=job_duration,
                                                                 run_script=worker_script,
                                                                 analysis_script=in_analysis_script_path,
                                                                 array_length=lig_num,
                                                                 array_name=system.name, cpu_per_job=nmpi_per_replica,
                                                                 memory=memory)
        ###bash make job_array script executable
        bash.execute("chmod +x " + job_array_script + " " + in_analysis_script_path)
    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR during Preperations")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1

    #################
    # Job submission.
    #################
    try:
        if (submit):
            if (verbose): print("SUBMITTING jobs: ", job_array_script)
            bash.execute("bash " + job_array_script)
        else:
            if (verbose): print("SKIP submitting!")

        if (verbose): print("DONE")

        return 0
    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR during Submission")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1


# MAIN Execution from BASH
if __name__ == "__main__":
    from reeds.function_libs.utils.argument_parser import execute_module_via_bash

    print(spacer + "\t\tRE-EDS ENERGY OFFSET ESTIMATION \n" + spacer + "\n")
    requiers_gromos_files = [("in_top_path", "input topology .top file."),
                             ("in_coord_path", "input coordinate .cn file."),
                             ("in_perttop_path", "input perturbation topology .ptp file."),
                             ("in_disres_path", "input distance restraint .dat file.")]
    execute_module_via_bash(__doc__, do, requiers_gromos_files)
