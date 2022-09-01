import os
from reeds.openmm.reeds_openmm import *

os.makedirs("vacuum", exist_ok = True)
os.chdir("vacuum")

s_values = np.array([1.0,0.511,0.261,0.197,0.133,0.112,0.090,0.068,0.052,0.035,0.018])
energy_offsets = np.array([0, -26.97, -87.77, 91.96, 99.9, 118.63])

# Careful: taken from Restraintmaker, so indices start at one, and will be decreased by one during the assignment of the distance restraints!!
restraint_pairs = [[31,78], [34,75], [35,80], [32,77], [16,52], [19,49], [18,54], [15,51], [49,78], [52,75], [53,80], [50,77], [1,32], [4,35], [5,34], [2,31], [16,64], [19,67], [18,66], [15,63], [2,67], [5,64], [4,65], [1,68]]

reeds_simulation_variables = ReedsSimulationVariables(s_values, energy_offsets, restraint_pairs, total_steps = 250000, distance_restraints_start_at_1 = True, eps_reaction_field = 1, pressure = None)

param_file = "../input/all_ligands_vac.leap.prm"
crd_files = "../input/all_ligands_vac.leap.crd"
#state_files = ["0/hyd_set_water_state_replica_" + str(i) for i in range(len(s_values))]
reeds_input_files = ReedsInputFiles(param_file, crd_files)

reeds_simulation = Reeds("set_A_vacuum", reeds_simulation_variables, reeds_input_files)
reeds_simulation.run()

df = reeds_simulation.calculate_free_energy_differences(0) # calculate free-energy differences at s=1
print("free energies: ", df)
