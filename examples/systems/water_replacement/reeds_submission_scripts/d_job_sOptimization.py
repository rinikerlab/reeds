#!/usr/bin/env python3
import os, sys, glob
from reeds.modules import do_RE_EDS_sOptimisation as sOptimisation
sys.path.append(os.getcwd())
from global_definitions import *

#paths
name = "system_inProtein"
in_name = name+"_sOpt"
out_sopt_dir = root_dir+"/d_"+in_name
input_dir = root_dir+"/input_files"
out_sopt_dir = bash.make_folder(out_sopt_dir)

#Input Files
topology = fM.Topology(top_path=in_top_file, disres_path=None, posres_path=in_posres_file, refpos_path=in_refpos_file, perturbation_path=in_pert_file)
system =fM.System(coordinates=glob.glob(input_dir+"/*cnf"), name=in_name, top=topology)

#Additional Options
job_duration="24:00:00"
nmpi_per_replica = 8
iterations= 1
add_replicas = 0
steps_between_trials = 20
trials_per_run = 10000
n_equilibrations = 1

last_jobID = sOptimisation.do(out_root_dir=out_sopt_dir,
                              in_simSystem=system,
                              in_template_imd=input_dir+"/template_reeds.imd",
                              in_gromosXX_bin_dir=gromosXX_bin,
                              in_gromosPP_bin_dir=gromosPP_bin,
                              in_ene_ana_lib_path=ene_ana_lib,
                              nmpi_per_replica=nmpi_per_replica,
                              duration_per_job=job_duration, 
                              iterations=iterations,
                              add_replicas=add_replicas,
                              run_NLRTO=False,
                              run_NGRTO=True,
                              equil_runs=n_equilibrations,
                              steps_between_trials=steps_between_trials,
                              trials_per_run=trials_per_run,
                              verbose= True)
