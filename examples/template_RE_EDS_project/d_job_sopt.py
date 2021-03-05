#!/usr/bin/env python3
import os, sys, glob
from reeds.modules import do_RE_EDS_sOptimisation as sOptimization

sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_pert_file, in_disres_file, in_template_reeds_imd


#spefici parts
out_sopt_dir = root_dir+"/d_sopt"
next_sopt_dir = root_dir+"/c_eoff/analysis/next"
in_name = name+"_sopt"

##make folder
out_sopt_dir = bash.make_folder(out_sopt_dir)

#In-Files
topology = fM.Topology(top_path=in_top_file,    disres_path=in_disres_file, pertubation_path=in_pert_file)
coords = glob.glob(next_sopt_dir+"/*cnf")
system =fM.System(coordinates=coords, name=in_name, top=topology)

# Additional options 

soptIterations = 4 
add_replicas = 4

last_jobID = sOptimization.do(out_root_dir=out_sopt_dir,in_simSystem=system,
                              gromosXX_bin_dir = gromosXX_bin, gromosPP_bin_dir = gromosPP_bin,
                              in_ene_ana_lib_path=ene_ana_lib,
                              soptIterations = soptIterations,
                              add_replicas = add_replicas
                              )


