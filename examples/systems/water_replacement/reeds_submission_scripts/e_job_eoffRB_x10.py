#!/usr/bin/env python3
import os, sys, glob
from reeds.modules import do_RE_EDS_eoffRebalancing as eoffRB
sys.path.append(os.getcwd())
from global_definitions import *

#paths
name = "system_inProtein_intensityFactorx10"
in_name = name+"_eoffRB"
out_sopt_dir = root_dir+"/e_"+in_name
next_sopt_dir = root_dir+"/d_system_sOpt/sopt1/analysis/next/"
out_sopt_dir = bash.make_folder(out_sopt_dir)

#Input Files
topology = fM.Topology(top_path=in_top_file, disres_path=None, posres_path=in_posres_file, refpos_path=in_refpos_file, perturbation_path=in_pert_file)
system =fM.System(coordinates=glob.glob(next_sopt_dir+"/*cnf"), name=in_name, top=topology)

#Additional Options
job_duration="24:00:00"
nmpi_per_replica = 8
iterations= 4
learningFactors = None
individualCorrection = True
num_states = 8
intensity_factor = 10
pseudocount = (1/num_states)/intensity_factor
n_equilibrations = 1
trials_per_run = 10000
steps_between_trials = 20

last_jobID = eoffRB.do(out_root_dir=out_sopt_dir,
                                    in_simSystem=system,
                                    in_gromosXX_bin_dir=gromosXX_bin,
                                    in_gromosPP_bin_dir=gromosPP_bin,
                                    in_ene_ana_lib_path=ene_ana_lib,
                                    nmpi_per_replica=nmpi_per_replica,
                                    duration_per_job = job_duration,
                                    iterations=iterations,
                                    learningFactors=learningFactors,
                                    individualCorrection=individualCorrection,
                                    pseudocount=pseudocount,
                                    equil_runs=n_equilibrations,
                                    steps_between_trials=steps_between_trials,
                                    trials_per_run=trials_per_run,
                                    verbose= True)
