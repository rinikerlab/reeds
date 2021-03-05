#!/usr/bin/env python3
import os, sys, glob
from reeds.modules import do_RE_EDS_eoffEstimation as eoffEstm


sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_pert_file, in_disres_file, in_template_reeds_imd



#STEP specifics
out_eoff_dir = root_dir+"/c_eoff"
next_lowerBound_dir = root_dir+"/b_lowerBound/analysis/next"
in_name = name+"_energy_offsets"


##make folder
out_eoff_dir = bash.make_folder(out_eoff_dir)

#In- Files
topology = fM.Topology(top_path=in_top_file, disres_path=in_disres_file, pertubation_path=in_pert_file)
coords  =glob.glob(next_lowerBound_dir+"/*.cnf")
system = fM.System(coordinates=coords, name=in_name, top=topology)
print(system)

last_jobID = eoffEstm.do(out_root_dir=out_eoff_dir, in_simSystem=system,
                         gromosXX_bin_dir = gromosXX_bin, gromosPP_bin_dir = gromosPP_bin,
                         in_template_imd_path=in_template_reeds_imd, in_ene_ana_lib=ene_ana_lib)
