#!/usr/bin/env python3
"""
.. automodule:          Find lower S-bound for EDS System
 - Description:\n
    This script tries to find the lower s-bound of an EDS system, that is possible to be simulated ab initio and where undersampling occurs.
    Therefore a job_array will executed, that runs seperate EDS simulations with different s_values.

    In the output folder ("root_path_to_out_dir/analysis/"), you should find an s_values.csv and multiple coordinate (.cnf) Files.
    These Files were generated as  result of successfull finished simulations from this script.
    This output will be automatically used by do_RE_EDS_eoffEstimation.py(provide: "root_path_to_out_dir/analysis/")

 - Useage:
    - via commandline
       use: python do_RE_EDS_generateOptimizedStates.py -h to get input help.
    - via python
       for getting the input right see main part under the function.
 - Author: Benjamin Schroeder

"""

import copy
import os
import sys
import traceback

import reeds.function_libs.pipeline.module_functions
from pygromos.euler_submissions import gen_Euler_LSF_jobarray, gen_Euler_slurm_jobarray, FileManager as fM
from pygromos.files import coord
from pygromos.utils import bash
from reeds.data import ene_ana_libs
from reeds.function_libs.pipeline import generate_euler_job_files as gjs
from reeds.function_libs.pipeline.module_functions import adapt_imd_template_lowerBound
from reeds.function_libs.pipeline.worker_scripts.analysis_workers import RE_EDS_explore_lowerBound_analysis as ana
from reeds.function_libs.utils.structures import spacer
from scipy import rand


def do(out_root_dir: str, in_simSystem: fM.System, undersampling_occurrence_fraction: float = 0.95,
       template_imd: str = None,
       gromosXX_bin: str = None, gromosPP_bin: str = None,
       ene_ana_lib: str = ene_ana_libs.ene_ana_lib_path,
       submit: bool = True, s_values = None, randomize_seed: bool = False,
       simulation_steps: int = 100000, job_duration: str = "24:00:00", memory: int = None, nmpi_per_replica: int = 4,
       verbose: bool = True) -> int:
    """ Find lower S-bound for EDS System
         - Description:\n
            This script tries to find the lower s-bound of an EDS system, that is possible to be simulated ab initio and where undersampling occurs.
            Therefore a job_array will executed, that runs seperate EDS simulations with different s_values.

            In the output folder ("root_path_to_out_dir/analysis/"), you should find an s_values.csv and multiple coordinate (.cnf) Files.
            These Files were generated as  result of successfull finished simulations from this script.
            This output will be automatically used by do_RE_EDS_eoffEstimation.py(provide: "root_path_to_out_dir/analysis/")

         - Useage:
            - via commandline
               use: python do_RE_EDS_generateOptimizedStates.py -h to get input help.
            - via python
               for getting the input right see main part under the function.
         - Author: Benjamin Schroeder

Parameters
----------
out_root_dir
    path to the root output dir, in which the script directories wild be build into
in_simSystem
    System obj. that contains information about the System-Files (cnf, top,...)
template_imd
    gives the path to the template_RE_EDS_project parameter file (.imd), that is adapted to fit the system.
undersampling_occurrence_fraction : float, optional
    potential threshold deciding if state was sampled or not - used in analysis
gromosXX_bin : str, optional
    path to gromosXX binary dir
gromosPP_bin : str, optional
    path to gromosPP binary dir
ene_ana_lib : str, optional
    path to in_ene_ana_lib,that can read the reeds system.
submit : bool, optional
    Flag, if the generated sopt_job should be submitted to lsf queue.
s_values : list[float], optional
    manually specify s_values
randomize_seed : bool, optional
    randomize initial velocities
exclude_residues : str, optional
    for cofactors, so that they are not considered as eds states
simulation_steps : int, optional
    how many steps per simulations?
single_bath : bool, optional
    only use a single MULTIBATH (i.e. NBATHS = 1)
job_duration : str, optional
    duration of one simulation step in the queue
verbose : bool, optional
    Let me tell you a story....

Returns
-------
int
    returns 0 on successful script execution.

    """
    #################
    # Prepare Jobs
    #################
    try:
        if (verbose): print(spacer + "START Param_exploration.")
        system = copy.deepcopy(in_simSystem)
        # make folders:
        input_dir = out_root_dir + "/input"
        coord_dir = input_dir + "/coord"
        sim_dir = out_root_dir + "/simulation"

        if (verbose): print("Generating Folders")
        bash.make_folder(out_root_dir)
        bash.make_folder(input_dir)
        bash.make_folder(coord_dir)

        # generate adapted parameter file from template_RE_EDS_project file. (check ligand num, adapt atom counts")
        if (verbose): print("Writing imd_templates")
        cnf = coord.Cnf(system.coordinates, clean_resiNumbers_by_Name=True)

        # build imd_templates
        imd_template_path, s_values = adapt_imd_template_lowerBound(system=system,
                                                                           in_template_imd_path=template_imd,
                                                                           out_imd_dir=input_dir,
                                                                           s_values=s_values,
                                                                           simulation_steps=simulation_steps,
                                                                           randomize=randomize_seed)

        # copy cnfs:
        for ind in range(len(s_values)):
            bash.copy_file(system.coordinates, coord_dir + "/" + \
                    os.path.basename(system.coordinates).replace(".cnf", "_" + str(ind+1) + ".cnf"))

        cnf_array = coord_dir + "/" + os.path.basename(system.coordinates).replace(".cnf", "_${RUNID}.cnf")
        setattr(system, "coord_seeds", cnf_array)

        # GENERATE array scripts
        if (verbose): print("generating job array scripts")

        # build: worker_scripts-script
        worker_script = gen_Euler_slurm_jobarray.build_worker_script_multImds(
            out_script_path=input_dir + "/worker_scripts.sh", gromosXX_bin=gromosXX_bin,
            out_dir=sim_dir, job_name=system.name + "_work", in_system=system,
            in_imd_prefix=imd_template_path, cores=nmpi_per_replica)

        ## build: analysis_script
        out_script_path = out_root_dir + "/analysis.py"
        if (verbose): print("\tgenerating analysis script: ", out_script_path)
        analysis_vars = {"out_analysis_dir": out_root_dir + "/analysis",
                         "in_topology_path": system.top.top_path,
                         "in_simulation_dir": sim_dir,
                         "undersampling_occurrence_fraction_threshold": undersampling_occurrence_fraction,
                         "in_imd_path": imd_template_path,
                         "gromosPP_bin": gromosPP_bin,
                         "in_ene_ana_lib": ene_ana_lib,
                         "system_name": system.name,
                         }
        in_analysis_script_path = reeds.function_libs.pipeline.module_functions.write_job_script(out_script_path=out_script_path, target_function=ana.do,
                                                                                                 variable_dict=analysis_vars)

        # build: sopt_job array_schedule script
        job_array_script = gen_Euler_slurm_jobarray.build_jobarray(script_out_path=out_root_dir + "/job_array.sh",
                                                                 output_dir=sim_dir, run_script=worker_script,
                                                                 array_length=len(s_values), array_name=system.name,
                                                                 cpu_per_job=nmpi_per_replica,
                                                                 analysis_script=in_analysis_script_path,
                                                                 memory=memory,
                                                                 duration=job_duration)

        # bash make job_array script executable
        bash.execute("chmod +x " + job_array_script + " " + in_analysis_script_path)

    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR during Preperations")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1

    #################
    # Submit Jobs
    #################
    try:
        # Job submission.
        if (submit):
            if (verbose): print("SUBMITTING: " + job_array_script)
            bash.execute("bash " + job_array_script)
        else:
            if (verbose): print("SKIP Submitting!")

        if (verbose): print("Done")
        return 0
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
                             ("in_perttop_path", "input perturbation topology .ptp file."),
                             ("in_disres_path", "input distance restraint .dat file.")]
    execute_module_via_bash(__doc__, do, requiers_gromos_files)
