#!/usr/bin/env python3
import os, sys, glob
import reeds
import pygromos

from reeds.modules import do_RE_EDS_eoffEstimation as eoffEstm

sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_pert_file, in_disres_file, in_template_reeds_imd
from global_definitions import undersampling_frac_thresh

#STEP specifics
in_name = name + "_energy_offsets"
sval_file = root_dir + "/b_"+name+"_find_lower_bound/analysis/next/s_vals.csv"
out_eoff_dir = root_dir + "/c_"+in_name

##make folder
out_eoff_dir = bash.make_folder(out_eoff_dir)

opt_states = root_dir + "/a_"+name+"_optimize_single_state/analysis/next/"

#In- Files
topology = fM.Topology(top_path=in_top_file, disres_path=in_disres_file, pertubation_path=in_pert_file)
coords = glob.glob(opt_states + "/*.cnf")
system = fM.System(coordinates=coords, name=in_name, top=topology)

nmpi_per_replica = 6
memory = 10

last_jobID = eoffEstm.do(out_root_dir=out_eoff_dir, in_simSystem=system,
                         sval_file = sval_file,
                         in_template_imd_path = in_template_reeds_imd,
                         in_ene_ana_lib=ene_ana_lib,
                         gromosXX_bin_dir=gromosXX_bin, gromosPP_bin_dir=gromosPP_bin,
                         undersampling_fraction_threshold=undersampling_frac_thresh,
                         optimized_states = opt_states,
                         submit=True,
                         nmpi_per_replica = nmpi_per_replica,
                         memory = memory
                         )

