#!/usr/bin/env python3
"""
SCRIPT:            Do single Ligand MD
Description:
    This script is exectuing an gromos simulation for a single ligand in a water box
    TODO: very old, test!
Author: Benjamin Schroeder
"""

import copy

import reeds.function_libs.pipeline.module_functions
from pygromos.euler_submissions import FileManager as fM
from pygromos.files import imd, coord
from pygromos.utils import bash
from reeds.function_libs.pipeline import generate_euler_job_files as gjs
from reeds.function_libs.pipeline.jobScheduling_scripts import MD_simulation_scheduler as md

spacer = "////////////////////////////////////////////////////////////////////////////////////////////////////////////"


def do(out_root_dir: str, in_simSystem: fM.System, in_template_imd: str, in_gromosXX_bin_dir: str,
       in_gromosPP_bin_dir: str, in_ene_ana_lib_path: str,
       nmpi: int = 1, nomp: int = 1,
       submit: bool = True,
       simulation_runs: int = 3, duration: str = "04:00", write_free_energy_traj: bool = False,
       equilibration_runs: int = 0, verbose: bool = False):
    """
This function is building a gromos MD-simulation on Euler

Parameters
----------
out_root_dir : str
    out_root_dir used for output of this script
in_simSystem : fM.System
    this is the system obj. containing all paths to all system relevant Files.
in_template_imd : str
    imd file path
in_gromosXX_bin_dir : str
    path to gromosXX binary dir
in_gromosPP_bin_dir : str
    path to gromosPP binary dir
in_ene_ana_lib_path : str
    path to in_ene_ana_lib,that can read the reeds system.
nmpi : int, optional
     number of nmpi threads for replicas
nomp : int, optional
     number of nomp threads for replicas
submit : bool, optional
    should the folder and Files just be builded or also the sopt_job be submitted to the queue?
simulation_runs : int, optional
    how many simulation runs shall be queued?
duration : str, optional
    job duration in the queue
write_free_energy_traj : bool, optional
    output_free energy traj
equilibration_runs : int, optional
    how many runs are equilibration?
verbose : bool, optional
    I can be very talkative, you know...

Returns
-------
int
    returns last job_id

    """

    simSystem = copy.deepcopy(in_simSystem)
    print(spacer + "START production.")
    input_dir = out_root_dir + "/input"
    sim_dir = out_root_dir + "/simulation"
    imd_out_path = input_dir + "/md.imd"
    analysis_dir = out_root_dir + "/analysis"

    print("Generating Folders")
    # make folders:
    bash.make_folder(out_root_dir)
    bash.make_folder(input_dir)

    # copy coordinates:
    simSystem.move_input_coordinates(input_dir)

    # generate imd_templates
    print("Writing imd_templates")
    cnf = coord.Cnf(simSystem.coordinates)
    residues = cnf.get_residues()
    imd_out_path = bash.copy_file(in_template_imd, imd_out_path)
    imd_file = imd.Imd(imd_out_path)

    clean_residues, ligands, protein, non_ligands = imd_file.clean_residue_list_for_imd(residues)

    ##Define temperature baths
    if (protein.number_of_atoms > 0):
        temp_baths = {}
        if (isinstance(non_ligands, type(None))):  # TODO: DIRTY HACK: in PNMT is Cofactor at end of the file.
            solvent_bath = (ligands.number_of_atoms + protein.number_of_atoms + clean_residues["SOLV"])
        else:
            solvent_bath = (ligands.number_of_atoms + protein.number_of_atoms + non_ligands.number_of_atoms +
                            clean_residues[
                                "SOLV"])

        if (max(ligands.positions) < protein.position):
            temp_baths = {ligands.number_of_atoms: 1, (ligands.number_of_atoms + protein.number_of_atoms): 2,
                          solvent_bath: 3}
        else:
            temp_baths = {protein.number_of_atoms: 1, (ligands.number_of_atoms + protein.number_of_atoms): 2,
                          solvent_bath: 3}
    else:
        temp_baths = {ligands.number_of_atoms: 1, (ligands.number_of_atoms + clean_residues["SOLV"]): 2}

    if (non_ligands):
        non_lig_atoms = non_ligands.number_of_atoms
    else:
        non_lig_atoms = 0

    if (protein):
        protein_number_of_atoms = protein.number_of_atoms
    else:
        protein_number_of_atoms = 0

    imd_file.SYSTEM.NSM = int(clean_residues["SOLV"] / 3)  # FOR WATERSOLVENT GIVE RESIDUES
    imd_file.FORCE.adapt_energy_groups(clean_residues)
    imd_file.MULTIBATH.adapt_multibath(last_atoms_bath=temp_baths)
    imd_file.write(imd_out_path)

    # GENERATE array scripts
    from reeds.function_libs.pipeline.worker_scripts.analysis_workers import MD_simulation_analysis as md_ana
    analysis_vars = {"out_analysis_dir": analysis_dir, "in_simulation_dir": sim_dir,
                     "in_topology_path": simSystem.top.top_path, "in_simulation_name": simSystem.name,
                     "in_imd_path": imd_out_path,
                     "control_dict": md_ana.template_control_dict,
                     "in_ene_ana_lib_path": in_ene_ana_lib_path, "gromosPP_bin_dir": in_gromosPP_bin_dir, }

    in_analysis_script_path = reeds.function_libs.pipeline.module_functions.write_job_script(out_script_path=out_root_dir + "/job_analysis.py",
                                                                                             target_function=md_ana.do, variable_dict=analysis_vars,
                                                                                             no_reeds_control_dict=True)

    print("generating LSF-Bashscripts")
    print(" nmpi: ", nmpi, "   nomp: ", nomp)

    args_dict = {"in_simSystem": simSystem, "in_imd_path": imd_out_path, "programm_path": in_gromosXX_bin_dir,
                 "out_dir_path": sim_dir,
                 "equilibration_num": equilibration_runs, "simulation_num": simulation_runs,
                 "write_out_free_energy_traj": write_free_energy_traj,
                 "nmpi": nmpi, "nomp": nomp, "duration_per_job": duration, "analysis_script": in_analysis_script_path,
                 "verbose": verbose}

    schedule_jobs_script_path = reeds.function_libs.pipeline.module_functions.write_job_script(out_script_path=out_root_dir + "/production_job.py",
                                                                                               target_function=md.do, variable_dict=args_dict)
    bash.execute("chmod +x " + schedule_jobs_script_path + " " + in_analysis_script_path)

    if (submit):
        print("SUBMITTING: ")
        job_id = md.do(**args_dict)
    else:
        print("SKIP submitting!")

    print("DONE")
    if (not submit):
        job_id = -1
    return job_id


if __name__ == "__main__":
    from reeds.function_libs.utils.argument_parser import execute_module_via_bash

    print(spacer + "\n\t\tMD PRODUCTION\n" + spacer + "\n")
    requiers_gromos_files = [("in_top_path", "input topology .top file."),
                             ("in_coord_path", "input coordinate .cn file.")]
    execute_module_via_bash(__doc__, do, requiers_gromos_files)
