import os
import sys
import numpy as np

from global_definitions import *
from reeds.function_libs.utils import s_log_dist

def get_previous_job_id(sub_file):
  submission = open(sub_file)
  line = submission.readline()
  jobid = line.split()[-1]
  return jobid
  

#sys.argv[1] should be step name
if(len(sys.argv) < 2):
  print("not enough arguments")
  exit()

step = sys.argv[1]

if(step == "a"):
  g = min(n_gpus, 1)
  for state in range(num_endstates):
    if g:
      command = f"sbatch -J state_optimization_{system_name} -n 1 --gpus={g} --time={time_eds} --wrap 'mpirun -np 1 python a_state_opt.py {state}'"
    else:
      command = f"sbatch -J state_optimization_{system_name} -n 1 --time={time_eds} --wrap 'mpirun -np 1 python a_state_opt.py {state}'"
    os.system(command)
    
elif(step == "b"):
  g = min(n_gpus, 1)
  s_values = s_log_dist.get_log_s_distribution_between(start=1.0, end=0.00001)
  for s_val in s_values:
    if g:
      command = f"sbatch -J lower_bound_{s_val} -n 1 --time={time_eds} --gpus={g} --wrap 'mpirun -np 1 python b_lower_bound.py {s_val}'"
    else:
      command = f"sbatch -J lower_bound_{s_val} -n 1 --time={time_eds} --wrap 'mpirun -np 1 python b_lower_bound.py {s_val}'"
    os.system(command)
    
elif(step == "c"):
  # optimized state coordinates
  state_file_path_prefix = state_optimization_dir + "/simulation/" + system_name
  
  # s-values
  s_val_file = lower_bound_dir + "/analysis/s_vals.csv"
  file = open(s_val_file)
  line = file.readline()
  s_values = np.array([float(s) for s in line.split()])
  num_s = len(s_values)
  
  # equilibration
  dir = eoff_estimation_dir + "/equilibration"
  command = f"echo $(sbatch -J eoff_eq -n {num_s} --time={time_parameter_search} --gpus={n_gpus} --wrap 'mpirun python c_eoff.py {s_val_file} {state_file_path_prefix} {dir}') > submission"
  print(command)
  os.system(command)

  jobid = get_previous_job_id("submission")
    
  # simulation
  state_file_path_prefix = eoff_estimation_dir + "/equilibration/" + system_name
  dir =  eoff_estimation_dir + "/simulation"
  command = f"echo $(sbatch -J eoff_sim -n {num_s} --time={time_parameter_search} --gpus={n_gpus} -d afterany:{jobid} --wrap 'mpirun python c_eoff.py {s_val_file} {state_file_path_prefix} {dir}') > submission"
  print(command)
  os.system(command)
  
  jobid = get_previous_job_id("submission")
  
  command = f"sbatch -J eoff_ana -d afterany:{jobid} --wrap 'python analysis.py c simulation'"
  print(command)
  os.system(command)
  
elif(step == "d"):
  # optimized state coordinates
  state_file_path_prefix = state_optimization_dir + "/simulation/" + system_name
  
  # s-values
  s_val_file = lower_bound_dir + "/analysis/s_vals.csv"
  file = open(s_val_file)
  line = file.readline()
  s_values = np.array([float(s) for s in line.split()])
  # generate new distribution
  s_values = [1.0 for x in range(num_endstates)] + list(
                  s_log_dist.get_log_s_distribution_between(start=1.0, end=min(s_values), num=len(s_values) - 4))[1:]
  os.makedirs(s_opt_dir, exist_ok = True)
  file = open(s_opt_dir + "/s_vals_init.csv", "w")
  for s in s_values:
    file.write(str(s) + " ")
  file.close()
  s_val_file = s_opt_dir + "/s_vals_init.csv"
  file = open(s_val_file)
  line = file.readline()
  s_values = np.array([float(s) for s in line.split()])
  num_s = len(s_values)
  jobid = None
  
  # energy offsets
  eoff_file = eoff_estimation_dir + "/analysis/energy_offsets.csv"
  
  for i in range(sopt_iterations): # increase for more iterations
    cur_dir = s_opt_dir + f"/s_opt_{i+1}"
    os.makedirs(cur_dir, exist_ok = True)
    # equilibration
    dir = cur_dir + "/equilibration"
    
    if(jobid is None):
      command = f"echo $(sbatch -J sopt_{i}_eq -n {num_s} --time={time_parameter_search} --gpus={n_gpus} --wrap 'mpirun python d_sopt.py {s_val_file} {state_file_path_prefix} {dir} {eoff_file}') > submission"
    else:
      command = f"echo $(sbatch -J sopt_{i}_eq -n {num_s} --time={time_parameter_search} --gpus={n_gpus} -d afterany:{jobid} --wrap 'mpirun python d_sopt.py {s_val_file} {state_file_path_prefix} {dir} {eoff_file}') > submission"
    print(command)
    os.system(command)
     
    jobid = get_previous_job_id("submission")
    state_file_path_prefix = cur_dir + "/equilibration/" + system_name
    
    #simulation
    dir = cur_dir + "/simulation"
    command = f"echo $(sbatch -J sopt_{i}_sim -n {num_s} --time={time_parameter_search} --gpus={n_gpus} -d afterany:{jobid} --wrap 'mpirun python d_sopt.py {s_val_file} {state_file_path_prefix} {dir} {eoff_file}') > submission"
    print(command)
    os.system(command)
     
    jobid = get_previous_job_id("submission")
    
    command = f"echo $(sbatch -J sopt_{i}_ana -d afterany:{jobid} --wrap 'python analysis.py d {cur_dir}') > submission"
    print(command)
    os.system(command)
    
    jobid = get_previous_job_id("submission")
    
    # new s-values and state files
    s_val_file = cur_dir + "/analysis/s_vals_new.csv"
    num_s += num_added_svalues_sopt
    state_file_path_prefix = cur_dir + "/simulation/" + system_name
    
elif(step == "e"):
  # optimized state coordinates
  state_file_path_prefix = s_opt_dir + "/s_opt_" + str(sopt_iterations) + "/simulation/" + system_name
  
  # s-values
  s_val_file = s_opt_dir + "/s_opt_" + str(sopt_iterations) + "/analysis/s_vals_new.csv"
  file = open(s_val_file)
  line = file.readline()
  s_values = np.array([float(s) for s in line.split()])
  num_s = len(s_values)
  
  jobid = None
  
  # energy offsets
  eoff_file = eoff_estimation_dir + "/analysis/energy_offsets.csv"
  
  for i in range(rebal_iterations): # increase for more iterations
    cur_dir = eoff_rebalancing_dir + f"/rebal_{i+1}"
    os.makedirs(cur_dir, exist_ok = True)
    
    # equilibration
    dir = cur_dir + "/equilibration"
    
    if(jobid is None):
      command = f"echo $(sbatch -J rebal_{i}_sim -n {num_s} --time={time_parameter_search} --gpus={n_gpus} --wrap 'mpirun python e_rebal.py {s_val_file} {state_file_path_prefix} {dir} {eoff_file}') > submission"
    else:
      command = f"echo $(sbatch -J rebal_{i}_sim -n {num_s} --time={time_parameter_search} --gpus={n_gpus} -d afterany:{jobid} --wrap 'mpirun python e_rebal.py {s_val_file} {state_file_path_prefix} {dir} {eoff_file}') > submission"
      
    print(command)
    os.system(command)
     
    jobid = get_previous_job_id("submission")
    state_file_path_prefix = cur_dir + "/equilibration/" + system_name
    
    # simulation
    dir = cur_dir + "/simulation"
    
    command = f"echo $(sbatch -J rebal_{i}_sim -n {num_s} --time={time_parameter_search} --gpus={n_gpus} -d afterany:{jobid} --wrap 'mpirun python e_rebal.py {s_val_file} {state_file_path_prefix} {dir} {eoff_file}') > submission"
    print(command)
    os.system(command)
     
    jobid = get_previous_job_id("submission")
    
    command = f"echo $(sbatch -J rebal_{i}_ana -d afterany:{jobid} --wrap 'python analysis.py e {cur_dir}') > submission"
    print(command)
    os.system(command)
    
    jobid = get_previous_job_id("submission")
    
    # new s-values and state files
    state_file_path_prefix = cur_dir + "/simulation/" + system_name
    eoff_file = cur_dir + "/analysis/eoffs_new.csv"
  
elif(step == "f"):
  # optimized state coordinates
  state_file_path_prefix = state_optimization_dir + "/simulation/" + system_name
  
  #s-values
  s_val_file = s_opt_dir + "/s_opt_" + str(sopt_iterations) + "/analysis/s_vals_new.csv"
  file = open(s_val_file)
  line = file.readline()
  s_values = np.array([float(s) for s in line.split()])[num_endstates - 1:] # remove SSM replicas with s = 1 
  num_s = len(s_values)
  
  os.makedirs(production_dir, exist_ok = True)
  s_val_file = production_dir + "/s_vals.csv"
  file = open(s_val_file, 'w')
  for s in s_values:
    file.write(str(s) + " ")
  file.close()
  
  #energy offsets
  if rebal_iterations:
    eoff_file = eoff_rebalancing_dir + "/rebal_" + str(rebal_iterations) + "/analysis/eoffs_new.csv"
  else:
    eoff_file = eoff_estimation_dir + "/analysis/energy_offsets.csv"
  
  jobid = None
  
  for i in range(production_iterations): # increase for more iterations
  
    dir = production_dir + "/simulation_" + str(i)

    if jobid is None:
      command = f"echo $(sbatch -J prod_sim -n {num_s} --time={time_production} --gpus={n_gpus} --wrap 'mpirun python f_prod.py {s_val_file} {state_file_path_prefix} {dir} {eoff_file}') > submission"
    else:
      command = f"echo $(sbatch -J prod_sim -n {num_s} --time={time_production} --gpus={n_gpus} -d afterany:{jobid} --wrap 'mpirun python f_prod.py {s_val_file} {state_file_path_prefix} {dir} {eoff_file}') > submission"
    print(command)
    os.system(command) 
    
    jobid = get_previous_job_id("submission")
    state_file_path_prefix = dir + "/" + system_name
    
  command = f"echo $(sbatch -J prod_ana -d afterany:{jobid} --wrap 'python analysis.py f') > submission"
  print(command)
  os.system(command)
  
  jobid = get_previous_job_id("submission")
  state_file_path_prefix = dir + "/" + system_name
