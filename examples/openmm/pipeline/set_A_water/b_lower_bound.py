import os
import sys

from global_definitions import *
from reeds.openmm.reeds_openmm_parallel import *

#sys.argv should be s-value
if(len(sys.argv) != 2):
  print("not enough arguments")
  exit()
  
s_val = float(sys.argv[1])
s_values = np.array([s_val])

energy_offsets = np.array([0] * num_endstates)

simulation_dir = lower_bound_dir + "/simulation"
os.makedirs(lower_bound_dir, exist_ok = True)
os.makedirs(simulation_dir, exist_ok = True)
os.chdir(simulation_dir)

reeds_input_files = REEDSInputFiles(param_file, crd_files)

reeds_simulation_variables = REEDSSimulationVariables(s_values, energy_offsets, restraint_pairs, total_steps = 100000, eps_reaction_field = eps_reaction_field, pressure = pressure)
reeds_simulation = REEDS(f"{system_name}_{s_val}", reeds_simulation_variables, reeds_input_files)
reeds_simulation.run()
