#!/usr/bin/env python3
import os, sys, glob
sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_pert_file, in_disres_file
from global_definitions import undersampling_frac_thresh
from global_definitions import job_duration, nmpi_per_replica
from global_definitions import sopt_iterations, run_NLRTO, run_NGRTO, add_replicas

from reeds.modules import do_RE_EDS_sOptimisation as sOptimization


#Paths
in_name = name+"_sopt"
next_sopt_dir = root_dir+"/c_eoff/analysis/next"
out_sopt_dir = root_dir+"/d_"+in_name

##make folder
out_sopt_dir = bash.make_folder(out_sopt_dir)

#In-Files
topology = fM.Topology(top_path=in_top_file,    disres_path=in_disres_file, pertubation_path=in_pert_file)
coords = glob.glob(next_sopt_dir+"/*cnf")
system =fM.System(coordinates=coords, name=in_name, top=topology)
print(system)

#DO:
last_jobID = sOptimization.do(out_root_dir=out_sopt_dir, in_simSystem=system,
                              in_gromosXX_bin_dir= gromosXX_bin, in_gromosPP_bin_dir= gromosPP_bin,
                              in_ene_ana_lib_path=ene_ana_lib,
                              undersampling_fraction_threshold=undersampling_frac_thresh,
                              iterations= sopt_iterations,
                              add_replicas = add_replicas,
                              run_NLRTO=run_NLRTO, run_NGRTO=run_NGRTO,
                              duration_per_job=job_duration,
                              nmpi_per_replica=nmpi_per_replica,
                              )


