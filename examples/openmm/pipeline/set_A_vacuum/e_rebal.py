import os
import sys
import glob 

from global_definitions import *
from reeds.openmm.reeds_openmm_parallel import *

if(len(sys.argv) != 5):
  print("not enough arguments")
  exit()
  
eoff_file = sys.argv[4]

file = open(eoff_file)
line = file.readline()
energy_offsets = np.array([float(s) for s in line.split()])
print(energy_offsets)

num_endstates = len(energy_offsets)
  
s_val_file = sys.argv[1]
file = open(s_val_file)
line = file.readline()
s_values = np.array([float(s) for s in line.split()])
print(s_values)

num_replicas = len(s_values)
print(s_values)

simulation_dir = sys.argv[3]
os.makedirs(simulation_dir, exist_ok = True)
os.chdir(simulation_dir)

file = open("s_vals.csv", "w")
for s in s_values:
  file.write(str(s) + " ")
file.close()

file = open("energy_offsets.csv", "w")
for e in energy_offsets:
  file.write(str(e) + " ")
file.close()

state_file_path = sys.argv[2]

state_files = sorted(glob.glob(state_file_path + "*[0-9]"), key=lambda x: float(x.split("_")[-1]))

while (len(state_files) < num_replicas):
  state_files.append(state_files[-1])  
  
print(state_files)

reeds_input_files = REEDSInputFiles(param_file, crd_files, state_files)

if("equilibration" in simulation_dir):
  total_steps = 50000
else:
  total_steps = 200000
  
reeds_simulation_variables = REEDSSimulationVariables(s_values, energy_offsets, restraint_pairs, total_steps = total_steps, eps_reaction_field = eps_reaction_field, pressure = pressure)
reeds_simulation = REEDS(system_name, reeds_simulation_variables, reeds_input_files)
reeds_simulation.run()

