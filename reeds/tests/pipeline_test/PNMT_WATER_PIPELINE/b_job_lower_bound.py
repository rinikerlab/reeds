#!/usr/bin/env python3
import os, sys, glob
from reeds.modules import do_RE_EDS_findLowerBound as findLowerBound

sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_cnf_file, in_pert_file, in_disres_file, in_template_md_imd

#Step specifics
out_lowerBound_dir = root_dir+"/TEST_b_lowerBound_TEST"
in_name = name+"_find_lower_bound"

##make folder
out_lowerBound_dir = bash.make_folder(out_lowerBound_dir)

#In- Files
topology_state_opt = fM.Topology(top_path=in_top_file, disres_path=in_disres_file, perturbation_path=in_pert_file)
system = fM.System(coordinates=in_cnf_file, name=in_name, top=topology_state_opt)
print(system)

#DO:
findLowerBound.do(in_simSystem=system,template_imd=in_template_md_imd, out_root_dir=out_lowerBound_dir,gromosXX_bin=gromosXX_bin, simulation_steps=10)
