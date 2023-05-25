import os
from openmm import unit as u

num_endstates = 5

#system dependent settings
system_name = "CHK1_complex"
root_dir = os.getcwd()
input_folder = root_dir+"/0_input"

#input files
param_file = os.path.abspath("0_input/complex.prm")
crd_files = os.path.abspath("0_input/complex.crd")

#distance restraints
# Careful: taken from Restraintmaker, so indices start at one, and will be decreased by one during the assignment of the distance restraints!!
restraint_pairs = [[76, 128], [57, 109], [68, 120], [64, 116], [72, 124], [60, 112], [74, 126], [62, 114], [76, 178], [57, 159], [64, 166], [68, 170], [72, 174], [60, 162], [78, 182], [74, 176], [4, 213], [20, 231], [26, 237], [12, 223], [7, 216], [23, 234], [17, 228], [9, 220], [110, 231], [126, 213], [116, 237], [120, 223], [132, 216], [113, 234], [107, 228], [122, 220], [4, 176], [20, 160], [26, 166], [12, 170], [7, 184], [23, 163], [17, 157], [9, 172]]

state_optimization_dir = root_dir + "/a_state_optimization"
lower_bound_dir = root_dir + "/b_lower_bound"
eoff_estimation_dir = root_dir + "/c_eoff"
s_opt_dir = root_dir + "/d_sopt"
eoff_rebalancing_dir = root_dir + "/e_eoff_rebal"
production_dir = root_dir + "/f_production"

eps_reaction_field = 78.5
pressure = 1.01325 * u.bar

sopt_iterations = 5
rebal_iterations = 5
production_iterations = 5

num_added_svalues_sopt = 5
rebalancing_intensity_factor = 10
rebalancing_learning_factors = 1

vacuum_simulation = False
environment = "complex"
dataset_name = "CHK1"

n_gpus = 8
time_eds = "4:00:00"
time_parameter_search = "24:00:00"
time_production = "24:00:00"
