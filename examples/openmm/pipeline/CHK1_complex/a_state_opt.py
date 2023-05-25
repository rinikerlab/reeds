import os
import sys

from global_definitions import *
from reeds.openmm.reeds_openmm_parallel import *

#sys.argv[1] should be end-state index (starting at 0)
if(len(sys.argv) != 2):
  print("not enough arguments")
  exit()
  
state = int(sys.argv[1])

# do EDS simulation at s = 1 with energy offsets biased towards one end-state
s_values = np.array([1.0])
energy_offsets = np.array([-500] * num_endstates)
energy_offsets[state] = 500

simulation_dir = state_optimization_dir + "/simulation"

os.makedirs(state_optimization_dir, exist_ok = True)
os.makedirs(simulation_dir, exist_ok = True)
os.chdir(simulation_dir)

reeds_input_files = REEDSInputFiles(param_file, crd_files)

reeds_simulation_variables = REEDSSimulationVariables(s_values, energy_offsets, restraint_pairs, total_steps = 500000, eps_reaction_field = eps_reaction_field, pressure = pressure)
reeds_simulation = REEDS(f"{system_name}_{state}", reeds_simulation_variables, reeds_input_files)
reeds_simulation.run()
