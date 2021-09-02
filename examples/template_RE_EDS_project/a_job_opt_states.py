#!/usr/bin/env python3
import os, sys, glob
sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_cnf_file, in_pert_file, in_disres_file, in_template_md_imd

from reeds.modules import do_RE_EDS_generateOptimizedStates as optimizeStates


#Paths
in_name = name+"_optimize_single_state"
out_gOptStates_dir = root_dir+"/a_"+in_name

##make folder
out_gOptStates_dir = bash.make_folder(out_gOptStates_dir)

#In-Files
topology_state_opt = fM.Topology(top_path=in_top_file, disres_path=in_disres_file, pertubation_path=in_pert_file)
system = fM.System(coordinates=in_cnf_file, name=in_name, top=topology_state_opt)
print(system)


# Additional options
## Simulation Params
job_duration="24:00"
nmpi_per_replica = 6

#DO:

optimizeStates.do(in_simSystem=system,
  in_imd_template_path=in_template_md_imd, 
  out_root_dir=out_gOptStates_dir,
  in_gromosXX_bin_dir=gromosXX_bin, 
  in_gromosPP_bin_dir=gromosPP_bin,
  simulation_steps = 10000,
  ene_ana_lib = ene_ana_lib
  )

