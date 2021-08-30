import os, json
import reeds
from reeds.function_libs.file_management import file_management as fM
from reeds.submodules.pygromos.pygromos.utils import bash

#PATHS
## Dependencies
gromosXX_bin = None
gromosPP_bin = None
ene_ana_lib = None

##System Dependent settings:
name = "MY_NAME"
root_dir = os.getcwd()

##Input Files
input_folder = root_dir+"/input/"

###general Templates
in_template_md_imd = root_dir+"/input/template_md.imd"
in_template_reeds_imd = root_dir+"/input/template_reeds.imd"

###System dependent Files
in_cnf_file =     input_folder+"/PNMT_9lig_water.cnf"
in_top_file =     input_folder+"/PNMT_9lig_water.top"
in_pert_file =    input_folder+"/PNMT_9lig_water.ptp"
in_disres_file =  input_folder+"/PNMT_9lig_water_disres.dat"

## Out Files
analysis_results_file_path = root_dir+"/analysis_results.json"
#if(not os.path.exists(root_dir+"/analysis_results.json")):
from reeds.function_libs.file_management.analysis_file import get_standard_pipeline
json.dump(get_standard_pipeline(), open(analysis_results_file_path, "w"), indent=4, sort_keys=True)


##global Parameters

## General
### Simulation Params
job_duration="24:00"
nmpi_per_replica = 6

### RE-EDS
undersampling_frac_thresh= 0.9

## Eoff Estimation

## S-Optimization
sopt_iterations = 4
run_NLRTO = True
run_NGRTO = False
add_replicas = 4

## Eoff Rebalancing
eoffRB_iterations = 4
learningFactors = None
individualCorrection = False
### Pseudocount
num_states = 5
intensity_factor = 10
pseudocount = (1/num_states)/intensity_factor


## Production
num_production_runs=25


