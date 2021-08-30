#!/usr/bin/env python3
import os, sys, glob
sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_pert_file, in_disres_file
from global_definitions import undersampling_frac_thresh

from reeds.modules import do_RE_EDS_production as production

        
#Paths
in_name = name+"_production"
next_production_dir = root_dir+"/d_sopt/soptX/analysis/next" #CHANGE HERE
out_production_dir = root_dir+"/f_"+in_name
 
##make folder
out_production_dir = bash.make_folder(out_production_dir)

#In-Files
topology = fM.Topology(top_path=in_top_file, disres_path=in_disres_file, pertubation_path=in_pert_file)
coords = glob.glob(next_production_dir+"/*cnf") 
in_template_reeds_imd = glob.glob(next_production_dir+"/*imd")[0]
system = fM.System(coordinates=coords, name=in_name, top=topology)
print(system)


#Additional Options
## Simulation Params
num_simulation_runs=25
job_duration="24:00"
nmpi_per_replica = 6


#Do:
last_jobID = production.do(out_root_dir=out_production_dir, in_simSystem=system, in_template_imd=in_template_reeds_imd,
                           gromosXX_bin_dir = gromosXX_bin, gromosPP_bin_dir = gromosPP_bin,
                           in_ene_ana_lib_path=ene_ana_lib,
                           undersampling_fraction_threshold=undersampling_frac_thresh,
                           num_simulation_runs=num_simulation_runs,
                           duration_per_job=job_duration, nmpi_per_replica=nmpi_per_replica)

