#!/usr/bin/env python3
import os, sys, glob
from reeds.modules import do_RE_EDS_production as production

sys.path.append(os.getcwd())

from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_pert_file, in_disres_file, in_template_reeds_imd

        
#STEP Specifics
out_production_dir = root_dir+"/TEST_e_production_TEST"
next_production_dir = root_dir+"/input/3_next_prod"
in_name = name+"_production"
 
##make folder
out_production_dir = bash.make_folder(out_production_dir)

#In-Files
topology = fM.Topology(top_path=in_top_file, disres_path=in_disres_file, pertubation_path=in_pert_file)
coords = glob.glob(next_production_dir+"/*cnf") 
system = fM.System(coordinates=coords, name=in_name, top=topology)
in_template_reeds_imd = glob.glob(next_production_dir+"/*imd")[0]

print(system)

#Do:
last_jobID = production.do(out_root_dir=out_production_dir,  in_simSystem=system, in_template_imd=in_template_reeds_imd, in_ene_ana_lib_path=ene_ana_lib, verbose=True, submit=True)

