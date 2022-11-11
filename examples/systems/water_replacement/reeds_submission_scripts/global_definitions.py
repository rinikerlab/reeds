import os
from pygromos.euler_submissions import FileManager as fM
from pygromos.utils import bash

#System Dependent settings
name = "inProtein_BPTI"
root_dir = os.getcwd()
input_folder =    root_dir+"/path_to_input_files"
in_cnf_file =     input_folder+"/BPTI_6Cl_3alchemicalWaters.cnf"
in_top_file =     input_folder+"/BPTI_6Cl_3alchemicalWaters.top"
in_pert_file =    input_folder+"/BPTI_6Cl_3waters_ch3Probes.ptp"
in_posres_file =  input_folder+"/BPTI_6Cl_3alchemicalWaters.por"
in_refpos_file =  input_folder+"/BPTI_6Cl_3alchemicalWaters.rpr"

gromosXX_bin = "/path/to/gromosXX/bin"
gromosPP_bin = "/path/to/gromos++/bin"
ene_ana_lib = "/new_ene_ana_REEDS_8state.md++.lib"
