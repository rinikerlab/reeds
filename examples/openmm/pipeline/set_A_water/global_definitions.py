import os
from openmm import unit as u

num_endstates = 6

#system dependent settings
system_name = "hyd_set_wat"
root_dir = os.getcwd()
input_folder = root_dir+"/0_input"

#input files
param_file = os.path.abspath("0_input/all_ligands_solv.leap.prm")
crd_files = os.path.abspath("0_input/all_ligands_solv.leap.crd")

#distance restraints
# Careful: taken from Restraintmaker, so indices start at one, and will be decreased by one during the assignment of the distance restraints!!
restraint_pairs = [[31,78], [34,75], [35,80], [32,77], [16,52], [19,49], [18,54], [15,51], [49,78], [52,75], [53,80], [50,77], [1,32], [4,35], [5,34], [2,31], [16,64], [19,67], [18,66], [15,63], [2,67], [5,64], [4,65], [1,68]]

state_optimization_dir = root_dir + "/a_state_optimization"
lower_bound_dir = root_dir + "/b_lower_bound"
eoff_estimation_dir = root_dir + "/c_eoff"
s_opt_dir = root_dir + "/d_sopt"
eoff_rebalancing_dir = root_dir + "/e_eoff_rebal"
production_dir = root_dir + "/f_production"

eps_reaction_field = 78.5
pressure = 1.01325 * u.bar

sopt_iterations = 2
rebal_iterations = 2
production_iterations = 1

num_added_svalues_sopt = 4
rebalancing_intensity_factor = 10
rebalancing_learning_factors = 1

vacuum_simulation = False
environment = "wat"
dataset_name = "A"

n_gpus = 4
time_eds = "1:00:00"
time_parameter_search = "4:00:00"
time_production = "4:00:00"
