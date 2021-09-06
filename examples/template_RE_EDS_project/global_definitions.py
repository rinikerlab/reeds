import os
import reeds
from pygromos.utils import bash
from pygromos.euler_submissions import FileManager as fM

#needs to be provided via gromos compiling
gromosXX_bin = None
gromosPP_bin = None
ene_ana_lib = os.path.dirname(reeds.__file__) + "/data/ene_ana_libs/new_ene_ana_REEDS_9state.md++.lib"

#System Dependent settings:
name = "example_PNMT_water" #adapt for your system
root_dir = os.getcwd()
input_folder =    root_dir+"/input/"

print(root_dir)

##input Files
###general Templates
in_template_md_imd = root_dir+"/input/template_md.imd"
in_template_reeds_imd = root_dir+"/input/template_reeds.imd"

###System dependent Files
in_cnf_file =     input_folder+"/PNMT_9lig_water.cnf"
in_top_file =     input_folder+"/PNMT_9lig_water.top"
in_pert_file =    input_folder+"/PNMT_9lig_water.ptp"
in_disres_file =  input_folder+"/PNMT_9lig_water_disres.dat"

##global Parameters
undersampling_frac_thresh= 0.9



