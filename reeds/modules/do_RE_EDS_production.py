#!/usr/bin/env python3
import copy
import os
import sys
import traceback
from typing import List
from collections import OrderedDict

import reeds.function_libs.pipeline.module_functions
from pygromos.euler_submissions import FileManager as fM
from pygromos.files import imd
from pygromos.files.coord import cnf as cnf_cls
from pygromos.utils import bash
from reeds.data import ene_ana_libs
from reeds.function_libs.pipeline import generate_euler_job_files as gjs
from reeds.function_libs.pipeline.jobScheduling_scripts import RE_EDS_simulation_scheduler
from reeds.function_libs.pipeline.worker_scripts.analysis_workers import RE_EDS_general_analysis
from reeds.function_libs.utils.structures import spacer


def do(out_root_dir: str, in_simSystem: fM.System, in_template_imd: str,
       gromosXX_bin_dir: str = None, gromosPP_bin_dir: str = None,
       in_ene_ana_lib_path: str = ene_ana_libs.ene_ana_lib_path,
       nmpi_per_replica: int = 1,
       submit: bool = True,
       optimized_states_dir: str = os.path.abspath("a_optimizedState/analysis/next"),
       lower_bound_dir: str = os.path.abspath("b_lowerBound/analysis/next"),
       state_physical_occurrence_potential_threshold: List[float] = None,
       state_undersampling_occurrence_potential_threshold: List[float] = None,
       undersampling_fraction_threshold: float = 0.9,
       num_simulation_runs: int = 10, duration_per_job: str = "24:00",
       num_equilibration_runs: int = 0,
       do_not_doubly_submit_to_queue: bool = True, randomize : bool = False,
       initialize_first_run: bool = True, reinitialize: bool = False,
       verbose: bool = False):
    """RE-EDS Production run

    SCRIPT:            Do REEDS Production
    This function starts the production phase of the RE-EDS approach. It holds all parameters constant and samples the system.
    Finally it calculates the relative Free energies.
    Author: Benjamin Ries


Parameters
----------
out_root_dir : str
    this is the output directory
in_simSystem : fm.System
    input system
in_template_imd : str
    input parameter file
gromosXX_bin_dir : str, optional
    binaries for gromosXX(default: from shell)
gromosPP_bin_dir : str, optional
    binaries for gromosPP(default: from shell)
in_ene_ana_lib_path : str, optional
    path to ene_ana_lib(default: somelib)
nmpi_per_replica : int, optional
    how many cores per replica?
submit : bool, optional
    shall the jobs be submitted to the queue?
num_simulation_runs : int, optional
    number of repetitions of the imd.
num_equilibration_runs : int, optional
    number of equilibrations.
state_physical_occurrence_potential_threshold : List[float], optional
    potential thresholds for physical sampling (default: read in from step a)
state_undersampling_occurrence_potential_threshold : List[float], optional
    potential thresholds for occurrence sampling (default: read in from step b)
undersampling_fraction_threshold : float, optional
    fraction threshold for physical/occurrence sampling (default: 0.9)
duration_per_job : str, optional
    how long shall each job take?
do_not_doubly_submit_to_queue : bool, optional
    Check if there is already a job with this name, do not submit if true.
initialize_first_run : bool, optional
    should the velocities of the first run be reinitialized?
reinitialize : bool, optional
    should the velocities be reinitialized for all runs?
verbose : bool, optional
    Scream!

Returns
-------
int
    last submitted

    """

    print(spacer + "START production.")

    simSystem = copy.deepcopy(in_simSystem)

    input_dir = out_root_dir + "/input"
    coord_dir = input_dir + "/coord"
    out_dir_path = out_root_dir + "/simulation"
    in_imd_path = input_dir + "/repex_prod.imd"

    old_result_folder = os.path.dirname(in_simSystem.coordinates[0])
    analysis_dir = out_root_dir + "/analysis"
    print("Generating Folders")

    # make folders:
    bash.make_folder(out_root_dir)
    bash.make_folder(input_dir)
    bash.make_folder(coord_dir)

    try:

        # Get System Information
        # Generate appropriate imd file from the template
        
        print("Writing imd_templates")
        cnf = cnf_cls.Cnf(in_simSystem.coordinates[0])
        residues = cnf.get_residues()

        lig_atoms = sum([sum(list(residues[res].values())) for res in residues if res != "SOLV"])
        lig_num = sum([1 for res in residues if res != "SOLV"])

        #in_imd_path = bash.copy_file(in_template_imd, in_imd_path)

        # Remove multi s=1 (so there is only 1)
        imd_file = imd.Imd(in_template_imd)
        s_1_ammount = list(map(float, imd_file.REPLICA_EDS.RES)).count(1.0)
        imd_file.edit_REEDS(SVALS=imd_file.REPLICA_EDS.RES[s_1_ammount-1:])
        num_states = int(imd_file.REPLICA_EDS.NUMSTATES)
        num_svals = int(imd_file.REPLICA_EDS.NRES)
        if(randomize):
            imd_file.randomize()
        imd_file.write(in_imd_path)

        # Copy coordinates to path/to/input/coord

        new_cnfs = list(sorted(in_simSystem.coordinates, key=lambda x: int(x.split("_")[-1].replace(".cnf", ""))))[s_1_ammount-1:]
        new_coords = []
        for ind, new_cnf in enumerate(new_cnfs):
            new_coord = bash.copy_file(new_cnf, coord_dir+"/"+in_simSystem.name+"_"+str(ind+1)+".cnf" )
            new_coords.append(new_coord)
        
        # Set the gromos System coordinates to these new coords. 
        # This variable is accessed when giving locals() to write_job_script
        # I will set it for both in case!
        setattr(in_simSystem, "coordinates", new_coords) # this was the previous code
        setattr(simSystem, "coordinates", new_coords)


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


        # fix for euler!
        if (num_svals > 15):
            workdir = out_root_dir + "/local_scratch"
        else:
            workdir = None
        nmpi = nmpi_per_replica * num_svals

        # GENERATE array scripts
        control_dict = {
            "dfmult": {"do": True},
            "compress_simulation_folder": {"do": False},
            "prepare_input_folder": {"do": False,
                                     "sub": {
                                         "eoff_to_sopt": False,
                                         "write_eoff": False,
                                         "write_s": False,
                                     }
                                     }
        }

        # Generate execution Scripts
        if (verbose): print("generating Scripts in output dir")
        if (verbose): print("SVALS: ", num_svals, " nmpi_per_rep: ", nmpi_per_replica, "   nmpi", nmpi)

        ##Build analysis_script
        if (verbose): print("Analysis Script")
        analysis_vars = OrderedDict({
            "in_folder": out_dir_path,
            "in_imd": in_imd_path,
            "topology": in_simSystem.top.top_path,
            "out_folder": analysis_dir,
            "gromos_path": gromosPP_bin_dir,
            "in_ene_ana_lib": in_ene_ana_lib_path,
            "n_processors": 5,
            "state_undersampling_occurrence_potential_threshold": state_undersampling_occurrence_potential_threshold,
            "state_physical_occurrence_potential_threshold": state_physical_occurrence_potential_threshold,
            "undersampling_frac_thresh": undersampling_fraction_threshold,
            "dfmult_all_replicas": False,
            "verbose": True,
            "grom_file_prefix": simSystem.name,
            "title_prefix": simSystem.name,
            "control_dict": control_dict,
        })

        in_analysis_script_path = reeds.function_libs.pipeline.module_functions.write_job_script(out_script_path=out_root_dir + "/job_analysis.py",
                                                                                                 target_function=RE_EDS_general_analysis.do_Reeds_analysis,
                                                                                                 variable_dict=analysis_vars)

        # Build Job Script 
        if (verbose): print("Scheduling Script")
        
        jobname = simSystem.name
         
        schedule_jobs_script_path = reeds.function_libs.pipeline.module_functions.write_job_script(
                                        out_script_path=out_root_dir + "/schedule_production_jobs.py",
                                        target_function=RE_EDS_simulation_scheduler.do,
                                        variable_dict = locals())

        # Make simulation and analysis jobs executable
        bash.execute("chmod +x " + schedule_jobs_script_path + " " + in_analysis_script_path)

    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR in file preperations")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1

    try:
        if (submit):
            if (verbose): print("\n\nSUBMITTING: " + schedule_jobs_script_path)
            last_submitted_jobID = RE_EDS_simulation_scheduler.do(in_simSystem=simSystem, in_imd_path=in_imd_path,
                                                                  gromosXX_bin_dir=gromosXX_bin_dir,
                                                                  out_dir_path=out_dir_path, jobname=jobname, nmpi=nmpi,
                                                                  duration_per_job=duration_per_job,
                                                                  num_simulation_runs=num_simulation_runs,
                                                                  work_dir=workdir,
                                                                  num_equilibration_runs=num_equilibration_runs,
                                                                  do_not_doubly_submit_to_queue=do_not_doubly_submit_to_queue,
                                                                  initialize_first_run=initialize_first_run,
                                                                  reinitialize=reinitialize,
                                                                  in_analysis_script_path=in_analysis_script_path)
        else:
            if (verbose): print("\n\nSKIP submitting!")
            last_submitted_jobID = 0

    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR during job-submissoin")
        print("#####################################################################################")

        traceback.print_exception(*sys.exc_info())
        return -1

    return last_submitted_jobID


# MAIN Execution from BASH
if __name__ == "__main__":
    from reeds.function_libs.utils.argument_parser import execute_module_via_bash

    print(spacer + "\n\t\tRE_EDS PRODUCTION\n" + spacer + "\n")
    execute_module_via_bash(__doc__, do)
