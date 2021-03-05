import os
from pygromos.utils import bash
from pygromos.euler_submissions import FileManager as fM

#needs to be provided via gromos compiling
gromosXX_bin = "/cluster/home/bschroed/gromos/gromos_native/gromosXX_mpi/bin"
gromosPP_bin = "/cluster/home/bschroed/gromos/gromos_native/gromos++/bin"
ene_ana_lib ="/cluster/home/bschroed/gromos/ene_ana_libs/ene_ana.md.lib"

#System Dependent settings:
name = "PNMT_PIPELINE_TEST"
root_dir = os.getcwd()
input_folder =    root_dir+"/input/0_input_start/"

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





