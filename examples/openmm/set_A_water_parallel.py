"""
Example script for parallel execution of RE-EDS simulation with OpenMM
Each replica EDS simulation is assigned its own MPI process. If one or more GPU is available, the EDS simulations are distributed evenly among the GPUs.
The number of MPI processes needs to be euqal to the number of replicas, so the script should be called as 'mpiexec -np 16 python set_A_water_parallel.py'
"""

import os
from reeds.openmm.reeds_openmm_parallel import *

os.makedirs("water_parallel", exist_ok = True)
os.chdir("water_parallel")

s_values = np.array([1.0,0.518,0.393,0.3305,0.268,0.225,0.182,0.1605,0.139,0.1055,0.0887,0.072,0.0546,0.0373,0.0193,0.01])
energy_offsets = np.array([0.0, -25.13, -94.15, 105.73, 122.62, 139.76])

# Careful: taken from Restraintmaker, so indices start at one, and will be decreased by one during the assignment of the distance restraints!!
restraint_pairs = [[31,78], [34,75], [35,80], [32,77], [16,52], [19,49], [18,54], [15,51], [49,78], [52,75], [53,80], [50,77], [1,32], [4,35], [5,34], [2,31], [16,64], [19,67], [18,66], [15,63], [2,67], [5,64], [4,65], [1,68]]

reeds_simulation_variables = REEDSSimulationVariables(s_values, energy_offsets, restraint_pairs, total_steps = 250000, distance_restraints_start_at_1 = True)

param_file = "../input/all_ligands_solv.leap.prm"
crd_files = "../input/all_ligands_solv.leap.crd"
#state_files = ["0/hyd_set_water_state_replica_" + str(i) for i in range(len(s_values))]
reeds_input_files = REEDSInputFiles(param_file, crd_files)

reeds_simulation = REEDS("set_A_water", reeds_simulation_variables, reeds_input_files)
reeds_simulation.run()

df = reeds_simulation.calculate_free_energy_differences(0) # calculate free-energy differences at s=1
print("free energies: ", df)
