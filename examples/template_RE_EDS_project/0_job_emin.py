#!/usr/bin/env python3
import os, sys, glob
from reeds.modules import do_EDS_energy_minimization as energy_minimization

sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_cnf_file, in_pert_file, in_disres_file, in_template_eds_imd

#Step Specifics
out_dir = root_dir+"/0_emin"
in_name = name+"_minimization"

##make folder
out_dir = bash.make_folder(out_dir)

#In- Files
topology_state_opt = fM.Topology(top_path=in_top_file, disres_path=in_disres_file, perturbation_path=in_pert_file)
system = fM.System(coordinates=in_cnf_file, name=in_name, top=topology_state_opt)
print(system)

#DO:

energy_minimization.do(in_simSystem = system,
                       in_imd_template_path = in_template_eds_imd,
                       out_root_dir = out_dir,
                       in_gromosXX_bin_dir = gromosXX_bin,
                       in_gromosPP_bin_dir = gromosPP_bin,
                       submit=True,
                       nmpi_per_replica=1
                     )

