#!/usr/bin/env python3
import os, sys, glob
sys.path.append(os.getcwd())
from global_definitions import *
from reeds.modules import do_RE_EDS_production as production

#Paths
name = "system_inProtein_intensityFactorx10"
in_name = name+"_fromRb2_production"
next_production_dir = root_dir+"/e_"+name+"_eoffRB/eoffRB2/analysis/next" #CHANGE HERE
out_production_dir = root_dir+"/f_"+in_name

if(not os.path.exists(out_production_dir)):
    os.mkdir(out_production_dir)

# Input Files
topology = fM.Topology(top_path=in_top_file, disres_path=None, posres_path=in_posres_file, refpos_path=in_refpos_file, perturbation_path=in_pert_file)
system = fM.System(coordinates=glob.glob(next_production_dir+"/*cnf"), name=in_name, top=topology)
in_template_reeds_imd = glob.glob(next_production_dir+"/*imd")[0]

#Additional Options
num_simulation_runs=15
job_duration="24:00:00"
nmpi_per_replica = 8

last_jobID = production.do(out_root_dir=out_production_dir,
                           in_simSystem=system,
                           in_template_imd=in_template_reeds_imd,
                           gromosXX_bin_dir=gromosXX_bin,
                           gromosPP_bin_dir=gromosPP_bin,
                           in_ene_ana_lib_path=ene_ana_lib,
                           verbose=True,
                           duration_per_job=job_duration,
                           nmpi_per_replica=nmpi_per_replica,
                           num_simulation_runs=num_simulation_runs,
                           submit=True)
