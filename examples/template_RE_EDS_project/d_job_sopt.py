#!/usr/bin/env python3
import os, sys, glob
from reeds.modules import do_RE_EDS_sOptimisation as sOptimization

sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_pert_file, in_disres_file
from global_definitions import undersampling_frac_thresh


#spefici parts
in_name = name+"_sopt"
next_sopt_dir = root_dir+"/c_"+name+"_energy_offsets/analysis/next"
optimized_states_dir = root_dir + "/a_"+name+"_optimize_single_state/analysis/next"
lower_bound_dir = root_dir + "/b_"+name+"_find_lower_bound/analysis/next"
out_sopt_dir = root_dir+"/d_"+in_name

##make folder
out_sopt_dir = bash.make_folder(out_sopt_dir)

#In-Files
topology = fM.Topology(top_path=in_top_file,    disres_path=in_disres_file, pertubation_path=in_pert_file)
coords = glob.glob(next_sopt_dir+"/*cnf")
system =fM.System(coordinates=coords, name=in_name, top=topology)

# Additional options 

soptIterations = 4 
add_replicas = 4

nmpi_per_replica = 6
memory = 10

last_jobID = sOptimization.do(out_root_dir=out_sopt_dir,in_simSystem=system,
                              in_gromosXX_bin_dir= gromosXX_bin, in_gromosPP_bin_dir= gromosPP_bin,
                              in_ene_ana_lib_path=ene_ana_lib,
                              undersampling_fraction_threshold=undersampling_frac_thresh,
                              soptIterations = soptIterations,
                              add_replicas = add_replicas,
                              nmpi_per_replica = nmpi_per_replica,
                              memory = memory,
                              trials_per_run = 1000,
                              optimized_states_dir = optimized_states_dir,
                              lower_bound_dir = lower_bound_dir
                              )


