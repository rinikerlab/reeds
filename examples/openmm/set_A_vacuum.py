from reeds.openmm.reeds_openmm import *

os.makedirs("vacuum", exist_ok = True)
os.chdir("vacuum")

s_values = np.array([1.0,0.518,0.393,0.3305,0.268,0.225,0.182,0.1605,0.139,0.1055,0.0887,0.072,0.0546,0.0373,0.0193,0.01])
energy_offsets = np.array([0.0, -25.13, -94.15, 105.73, 122.62, 139.76])

# Careful: taken from Restraintmaker, so indices start at one, and will be decreased by one during the assignment of the distance restraints!!
restraint_pairs = [[31,78], [34,75], [35,80], [32,77], [16,52], [19,49], [18,54], [15,51], [49,78], [52,75], [53,80], [50,77], [1,32], [4,35], [5,34], [2,31], [16,64], [19,67], [18,66], [15,63], [2,67], [5,64], [4,65], [1,68]]

reeds_simulation_variables = ReedsSimulationVariables(s_values, energy_offsets, restraint_pairs, total_steps = 20, distance_restraints_start_at_1 = True, eps_reaction_field = 1, pressure = None)

param_file = "input/all_ligands_vac.leap.prm"
crd_files = "input/all_ligands_vac.leap.crd"
#state_files = ["0/hyd_set_water_state_replica_" + str(i) for i in range(len(s_values))]
reeds_input_files = ReedsInputFiles(param_file, crd_files)

reeds_simulation = Reeds("set_A_vacuum", reeds_simulation_variables, reeds_input_files)
reeds_simulation.run()

df = reeds_simulation.calculate_free_energy_differences(0) # calculate free-energy differences at s=1
print("free energies: ", df)
