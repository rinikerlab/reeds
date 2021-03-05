#!/usr/bin/env python3
"""
.. automodule:          Find lower S-bound for each endstate of a system. for EDS System
 - Description:\n
    This script tries to find the lower s-bound of an EDS system, that is possible to be simulated ab initio and where undersampling occurs.
    Therefore a job_array will executed, that runs seperate EDS simulations with different s_values.

    In the output folder ("root_path_to_out_dir/analysis/"), you should find an s_values.csv and multiple coordinate (.cnf) Files.
    These Files were generated as  result of successfull finished simulations from this script.
    This output will be automatically used by do_RE_EDS_eoffEstimation.py(provide: "root_path_to_out_dir/analysis/")

    Todo: automatically adapt the s-distribution to the information recieved in the simulation step.
    Todo: thoughts: How many under sampling replicas needed, How to cleverly distribute replicas between the s=1 and undersampling.
    Todo: enable protein useage in system (Problem with parameter file adaption. - protein should be one energygroup)

 - Useage:
    - via commandline
       use: python do_RE_EDS_generateOptimizedStates.py -h to get input help.
    - via python
       for getting the input right see main part under the function.
 - Author: Benjamin Schroeder

"""

import os

from pygromos.euler_submissions import gen_Euler_LSF_jobarray, FileManager as fM
from pygromos.files import imd
from pygromos.files.coord import cnf as cnf_cls
from pygromos.utils import bash
from reeds.data import imd_templates, ene_ana_libs
from reeds.function_libs.pipeline.worker_scripts.analysis_workers import RE_EDS_explore_lowerBound_analysis as ana
from reeds.function_libs.utils import s_log_dist as slog

spacer = "//////////////////////////////////////\n"


def do(root_dir: str, system: fM.System,
       template_imd: str = imd_templates.__path__._path[0] + "/eds_md.imd",
       gromosXX_bin: str = None, gromosPP_bin: str = None,
       in_ene_ana_lib: str = ene_ana_libs.__path__._path[0] + "/new_ene_ana_REEDS_9state.md++.lib",
       pot_tresh: float = 200, non_ligand_residues: list = [], min_s_limit: float = 0.00036, final_num_s: int = None,
       submit: bool = True, simulation_steps: int = 4000, memory: int = None) -> int:
    """
    __ UNDER CONSTRUCTION!__
    .. autofunction:

        This function is the core of this script. It does all, that was described in the module description.

    :param out_root_dir: path to the root_dir output dir, in which the script directories wild be build into
    :param system:  System obj. that contains information about the System-Files (cnf, top,...0
    :param submit:  Flag, if the generated sopt_job should be submitted to lsf queue.
    :param template_imd: gives the path to the template_RE_EDS_project parameter file (.imd), that is adapted to fit the system.
    :return: returns 0 on successful script execution.

    :type out_root_dir: str
    :type template_imd: str
    :type system: pygromos.PipelineManager.Simulation_System
    :type submit: bool
    :rtype: int

    """
    print(spacer + spacer + "\tSTART Param_exploration.\n" + spacer + spacer)
    # make folders:
    general_input_dir = root_dir + "/general_input"
    state_dirs = root_dir + "/states"

    print("\nGenerating Folders\n" + spacer)
    bash.make_folder(root_dir)
    bash.make_folder(general_input_dir)
    bash.make_folder(state_dirs)

    print("\nGet Clean Residue list\n" + spacer)
    # BUILD UP a clean residue list
    ##for making changes in parameter file and so on
    ##read in coord file and get a clean residue list
    cnf = cnf_cls.Cnf(system.coordinates[0]) if (isinstance(system.coordinates, list)) else cnf_cls.Cnf(
        system.coordinates)
    raw_residues = cnf.get_residues()
    residues, ligands, protein, non_ligands = imd.Imd.clean_residue_list_for_imd(raw_residues, non_ligand_residues)

    print("SYSTEM Residues: ", residues)
    print("SYSTEM nLigands: ", ligands.number)
    print("SYSTEM Ligands: ", ligands.names)

    print("\nWriting ParameterFiles\n" + spacer)
    # Modify IMD - Parameter File
    imd_path = template_imd
    imd_file = imd.Imd(imd_path)
    imd_file.SYSTEM.NSM = int(round(residues["SOLV"] / 3)) if ("SOLV" in residues) else 0
    imd_file.FORCE.adapt_energy_groups(residues)
    imd_file.STEP.NSTLIM = simulation_steps

    ##Define temperature baths
    if (protein.number_of_atoms > 0):
        print("PROTEIN PART")
        temp_baths = {}
        if (len(non_ligand_residues) < 1):  # TODO: DIRTY HACK: in PNMT is Cofactor at end of the file.
            solvent_bath = (ligands.number_of_atoms + protein.number_of_atoms + residues["SOLV"])
        else:
            solvent_bath = (ligands.number_of_atoms + protein.number_of_atoms + non_ligands.number_of_atoms + residues[
                "SOLV"])
        print(protein)
        if (max(ligands.positions) < protein.position):
            temp_baths = {ligands.number_of_atoms: 1, (ligands.number_of_atoms + protein.number_of_atoms): 2,
                          solvent_bath: 3}
        else:
            temp_baths = {protein.number_of_atoms: 1, (ligands.number_of_atoms + protein.number_of_atoms): 2,
                          solvent_bath: 3}
    else:
        temp_baths = {ligands.number_of_atoms: 1, (ligands.number_of_atoms + residues["SOLV"]): 2} if (
                    "SOLV" in residues) else {ligands.number_of_atoms: 1}

    imd_file.MULTIBATH.adapt_multibath(last_atoms_bath=temp_baths)

    ##Define EDS Block - and generate variants.
    imd_template_path = general_input_dir + "/opt_structs_" + "_".join(ligands.names)
    s_log_dist = slog.get_log_s_distribution_between(start=1.0,
                                                     end=min_s_limit)  # Todo: this parameters are hardcoded (not very bad, as up to now they were never changed)
    print("SlogDIS: \n", s_log_dist)

    for ind, sval in enumerate(s_log_dist):
        imd_file.edit_EDS(NUMSTATES=ligands.number, S=sval, EIR=[0.0 for x in range(ligands.number)])
        imd_file.write(imd_template_path + "_" + str(ind) + ".imd")

    # GEnerate lower Bound jobs for each state!
    new_location_seeds = []
    exec_commands = []
    print("LIGANDS: ", ligands)
    for lig_idx, ligand in enumerate(ligands.names):
        print("\n\tBUILD structure for " + str(lig_idx) + "_Ligand_" + str(ligand) + "\n" + spacer)

        # subfolder
        out_state_dir = state_dirs + "/" + str(lig_idx) + "_" + str(ligand)
        input_dir = out_state_dir + "/input"
        coord_dir = input_dir + "/coord"
        sim_dir = out_state_dir + "/simulation"

        bash.make_folder(out_state_dir)
        bash.make_folder(input_dir)
        bash.make_folder(coord_dir)
        bash.make_folder(sim_dir)

        # copy cnfs:
        coord_path = coord_dir + "/" + os.path.basename(system.coordinates[lig_idx])
        new_location_seeds.append(bash.copy_file(system.coordinates[lig_idx], coord_path))

        # copy imd_templates:
        bash.copy_file(general_input_dir + "/*.imd", input_dir, "-r")

        # GENERATE LSF array scripts
        print("\n\t\tgenerating LSF-Bashscripts\n")
        cores_per_job = 4

        # silly juggeling with coord files
        tmp_coord_list = getattr(system, "coord_seeds")
        setattr(system, "coord_seeds", coord_path)

        # build: worker_scripts-script
        worker_script = gen_Euler_LSF_jobarray.build_worker_script_multImds(
            out_script_path=input_dir + "/worker_scripts.sh",
            job_name=system.name + "_" + str(lig_idx) + "_Ligand_" + str(ligand),
            in_system=system,
            gromosXX_bin=gromosXX_bin,
            out_dir=sim_dir,
            in_imd_prefix=imd_template_path, cores=cores_per_job)

        # build: analysis_script for each step.
        analysis_script = ana.write_controlling_metascript(out_script_path=input_dir + "/analysis.py",
                                                           out_analysis_dir=out_state_dir + "/analysis",
                                                           in_simulation_dir=sim_dir, in_top=system.top.top_path,
                                                           s_vals=s_log_dist, numstates=ligands.number,
                                                           undersampling_pot_tresh=pot_tresh,
                                                           ene_ana_lib=in_ene_ana_lib, gromosPP_bin=gromosPP_bin)

        # build: sopt_job array_schedule script
        job_array_script = gen_Euler_LSF_jobarray.build_jobarray(script_out_path=out_state_dir + "/job_array.sh",
                                                                 output_dir=sim_dir, run_script=worker_script,
                                                                 array_length=len(s_log_dist),
                                                                 array_name=system.name + "_" + str(
                                                                     lig_idx) + "_Ligand_" + str(ligand),
                                                                 cpu_per_job=cores_per_job,
                                                                 analysis_script=analysis_script, noFailInChain=False,
                                                                 memory=memory)

        # bash make job_array script executable
        bash.execute("chmod +x " + job_array_script + " " + analysis_script)
        print("\tscript: " + job_array_script + "\n")
        exec_commands.append(job_array_script)

        # Job submission.
        if (submit):
            print("\tSUBMITTING: ")
            bash.execute(job_array_script)
        else:
            print("\tSKIP Submitting!")
        setattr(system, "coord_seeds", tmp_coord_list)  # silly juggeling with coord files back

    # write execution script for all states!:
    master_execution_file = open(root_dir + "/execute_all_jobs.sh", "w")
    print(exec_commands)
    master_execution_file.write("# !/bin/bash\n"
                                "#Execute all sub scripts\n\n")
    master_execution_file.write("\n".join(exec_commands))
    master_execution_file.close()

    # build: final analysis_script considering each state results.
    shuffle_script = ana.write_controlling_metascript_shuffle(out_script_path=root_dir + "/shuffle.py",
                                                              out_shuffled_next_dir=root_dir + "/next",
                                                              in_states_dir=state_dirs,
                                                              min_s_limit=min_s_limit, num_s=final_num_s)

    setattr(system, "coord_seeds", new_location_seeds)


# ana:
"grep \"successfully\" 2_param_exp/simulation/*.omd"
"write s dist"

# MAIN Execution from BASH
if __name__ == "__main__":
    from reeds.function_libs.utils.argument_parser import execute_module_via_bash

    print(spacer + "\t\tRE-EDS FIND LOWER BOUND FOR EACH STATE \n" + spacer + "\n")
    execute_module_via_bash(__doc__, do)
