"""
#TODO: Remove as not used

from pygromos.euler_submissions.FileManager import Simulation_System as sys
from reeds.function_libs.pipeline.jobScheduling_scripts import RE_EDS_simulation_scheduler

def build_md_job_script(script_out_path: str, job_name: str, system: sys.System, in_protocol: str, out_dir: str,
                        gromosXX_bin: str,
                        repetitions: int, equil_runs: int = 0, offset: int = 1, nmpi: int = 1, nomp: int = 0,
                        work_dir: str = None,
                        analysis_script: str = None,
                        run_script: str = RE_EDS_simulation_scheduler.__file__, duration: str = "23:59",
                        write_free_energy_traj: bool = False,
                        last_sim_dir: str = None, previous_runID: int = None):
    \"""

    Parameters
    ----------
    script_out_path :
    job_name :
    system :
    in_protocol :
    out_dir :
    gromosXX_bin :
    repetitions :
    equil_runs :
    offset :
    nmpi :
    nomp :
    work_dir :
    analysis_script :
    run_script :
    duration :
    write_free_energy_traj :
    last_sim_dir :
    previous_runID :

    Returns
    -------

    \"""
    import os
    in_imd = in_protocol
    in_cnf = system.coordinates
    in_top = system.top.top_path
    in_ptp = system.top.perturbation_path
    in_disres = system.top.disres_path

    # for copying old Files
    old_imd_command = ""
    copyCommand = False
    if (last_sim_dir != None and not os.path.exists(in_imd)):
        copyCommand = True
        if (previous_runID != None):
            old_imd_command = "jobID=$(bsub -W 0:20 -J \"" + job_name + "_copy\" -w \"done(" + str(
                previous_runID) + ")\" \"cp " + str(last_sim_dir) + "/next.imd " + in_imd + " && "
        else:
            old_imd_command = "jobID=$(bsub -W 0:20 -J \"" + job_name + "_copy\" \"cp " + str(
                last_sim_dir) + "/next.imd " + in_imd + " && "

        old_imd_command += " cp " + str(last_sim_dir) + "/*cnf " + os.path.dirname(
            in_cnf) + "\" | cut -d \"<\" -f2 | cut -d \">\" -f1)\n echo \"Submitted copying last Files: ${jobID}\"\n"

    in_files = ("#FILES\n"
                "TOP=\"" + in_top + "\"\n"
                                    "COORD=\"" + in_cnf + "\"\n"
                                                          "IMDRUN=" + in_imd + "\n"
                )

    variable_args = ""
    if (in_ptp != None):
        variable_args += " -perttop ${PERTTOP} "
        in_files += "PERTTOP=\"" + str(in_ptp) + "\"\n"

    if (in_disres != None):
        variable_args += " -disres ${DISRES} "
        in_files += "DISRES=\"" + str(in_disres) + "\"\n"
    in_files += "\n"

    # final python command - submitting sopt_job chain
    pythoncommand = "\"python ${RUNSCRIPT} -outdir ${OUTDIR} -imd \"${IMDRUN}\" -top ${TOP} " + variable_args + " -offsetstep ${OFFSET} -coord ${COORD} -nmpi ${MPI} -nomp ${OMP} -bin ${BIN} -simulation_runs ${PRODUCTIONTIME} -jobname ${JOBNAME} -equilibration_runs " + str(
        equil_runs) + " -duration_per_job " + duration + " \""

    if (work_dir != None):
        pythoncommand += " -workdir " + work_dir

    if (previous_runID != None or copyCommand):
        pythoncommand += " -waitFor ${jobID}"

    if analysis_script != None:
        pythoncommand += " -analysisScript " + analysis_script

    if (write_free_energy_traj):
        pythoncommand += " -write_free_energy_traj"

    pythoncommand += "\""

    script_text = ("#!/bin/bash"
                   "\n"
                   "OMP=" + str(nomp) + "\n"
                                        "MPI=" + str(nmpi) + "\n"
                                                             "\n"
                                                             "JOBNAME=\"" + job_name + "\"\n"
                                                                                       "OUTDIR=" + out_dir + "\n"
                                                                                                             "\n"
                                                                                                             "#LSF\n"
                                                                                                             "BIN=\"" + gromosXX_bin + "\"\n"
                                                                                                                                       "RUNSCRIPT=" + run_script + "\n"
                                                                                                                                                                   "jobID=" + str(
        previous_runID) + "  #Wait for this sopt_job\n"
                          "PRODUCTIONTIME=" + str(repetitions) + "\n"
                                                                 "OFFSET=" + str(offset) + "\n"
                                                                                           "\n"
                   + in_files +
                   "\n"
                   "#do\n"
                   "echo \"JOB: ${JOBNAME}\"\n"
                   "echo \"reserve MPI *OMP : ${MPI} * ${OMP} CPUS\"\n"
                   "mkdir -p ${OUTDIR}\n"
                   "cd $BASEDIR\n"
                   "echo \"got Initial JobID: ${jobID}\"\n"
                   "" + old_imd_command + "\n"
                                          "\nPYTHONCOMMANDRUN=" + str(pythoncommand) + "\n"
                                                                                       "\n"
                                                                                       "##Run Production\n"
                                                                                       "eval \"$PYTHONCOMMANDRUN\"\n"
                                                                                       "\n"
                                                                                       "cd ..")

    script = open(script_out_path, "w")
    script.write(script_text)
    script.close()
    return script_out_path


def build_MD_analysis_script(in_simulation_name: str, in_folder: str, in_topology_path: str,
                             out_dir: str, out_script_path: str,
                             in_ene_ana_lib: str, gromosPP_path: str):
    import_text = "#!/usr/bin/env python\n\n"
    import_text += "#################################\n"
    import_text += "# This is a small automatic generated ana_script wrapper.\n"
    import_text += "#################################\n"
    import_text += "from reeds.function_libs.jobScheduling_scripts import MD_simulation_analysis as ana\n\n"

    dependencies_text = "# INPUT DIRS/PATHS\n"
    dependencies_text += "##binary or file paths\n"
    dependencies_text += "in_ene_ana_lib = \"" + in_ene_ana_lib + "\"\n"
    dependencies_text += "gromosPP_path = \"" + gromosPP_path + "\"\n"
    dependencies_text += "\n"
    dependencies_text += "##dependencies\n"
    dependencies_text += "in_simulation_name = \"" + in_simulation_name + "\"\n"
    dependencies_text += "in_folder = \"" + in_folder + "\"\n"
    dependencies_text += "in_topology_path = \"" + in_topology_path + "\"\n"
    dependencies_text += "out_dir = \"" + out_dir + "\"\n"
    dependencies_text += "\n"
    dependencies_text += "##controls\n"
    dependencies_text += "control_dict = {\n"
    dependencies_text += "\"concat\": {\"do\": True,\n"
    dependencies_text += "\t\"sub\": {\n"
    dependencies_text += "\t\t\"tre\": True,\n"
    dependencies_text += "\t\t\"cnf\": True,\n"
    dependencies_text += "\t\t\"trc\": True,\n"
    dependencies_text += "\t\t\"convert_trc_xtc\": True\n"
    dependencies_text += "\t\t}\n"
    dependencies_text += "\t},\n"
    dependencies_text += "\"ene_ana\":{\"do\":True}\n"
    dependencies_text += "}\n"
    dependencies_text += "trc_take_every_step = 1\n"
    dependencies_text += "\n\n"

    command_text = "ana.do(in_simulation_name=in_simulation_name, in_folder=in_folder, " \
                   "in_topology_path=in_topology_path, out_dir=out_dir," \
                   "in_ene_ana_lib=in_ene_ana_lib, gromosPP_path=gromosPP_path)\n\n"

    script_text = import_text + dependencies_text + command_text

    ana_file = open(out_script_path, "w")
    ana_file.write(script_text)
    ana_file.close()

    return out_script_path
"""