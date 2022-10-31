import os
import sys
import glob

from global_definitions import *
from reeds.openmm.reeds_openmm_parallel import *

# arguments are: s_value file, state_file_path_prefix, simulation directory
if(len(sys.argv) != 4):
  print("not enough arguments")
  exit()
  
s_val_file = sys.argv[1]
file = open(s_val_file)
line = file.readline()
s_values = np.array([float(s) for s in line.split()])

num_replicas = len(s_values)
state_file_path_prefix = sys.argv[2]

simulation_dir = sys.argv[3]

energy_offsets = np.array([0] * num_endstates)

os.makedirs(eoff_estimation_dir, exist_ok = True)
os.makedirs(simulation_dir, exist_ok = True)
os.chdir(simulation_dir)

state_files = sorted(glob.glob(state_file_path_prefix + "*[0-9]"), key=lambda x: int(x.split("_")[-1]))

print("state_files", state_files)

while (len(state_files) < num_replicas):
  state_files.extend(state_files)  

reeds_input_files = REEDSInputFiles(param_file, crd_files, state_files)

if("equilibration" in simulation_dir):
  total_steps = 50000
else:
  total_steps = 200000
  
reeds_simulation_variables = REEDSSimulationVariables(s_values, energy_offsets, restraint_pairs, total_steps = total_steps, eps_reaction_field = eps_reaction_field, pressure = pressure)
reeds_simulation = REEDS(system_name, reeds_simulation_variables, reeds_input_files)
reeds_simulation.run()
