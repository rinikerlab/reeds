import sys, os, glob
from global_definitions import *

from openmm import unit as u
from parmed import load_file

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import ticker
from matplotlib.ticker import FormatStrFormatter
import numpy as np
import seaborn as sns
from typing import List
from collections.abc import Iterable  

from scipy.special import logsumexp
from scipy.stats import pearsonr
from scipy.stats import spearmanr

from reeds.function_libs.analysis.sampling import undersampling_occurence_potential_threshold_distribution_based as find_undersampling_pot_tresh
from reeds.function_libs.analysis.sampling import get_all_physical_occurence_potential_threshold_distribution_based as find_physical_sampling_pot_tresh
from reeds.function_libs.analysis.sampling import sampling_analysis
from reeds.function_libs.analysis import sampling as sampling_ana
import reeds.function_libs.utils.s_log_dist as s_log_dist
from reeds.function_libs.analysis import parameter_optimization
from reeds.function_libs.visualization import re_plots as re_plots
from reeds.function_libs.analysis import replica_exchanges as repex
from reeds.function_libs.optimization import eds_energy_offsets as eds_energy_offsets
from reeds.function_libs.optimization.eds_eoff_rebalancing import rebalance_eoffs_directCounting

from pygromos.files import repdat

## Helper Functions

def calculate_free_energies(ene_traj, T, step = None):
  """
  calculate free energy differences given a RE-EDS energy trajectory and a temperature
  """
  kb = (u.BOLTZMANN_CONSTANT_kB*u.AVOGADRO_CONSTANT_NA).value_in_unit(u.kilojoule_per_mole/u.kelvin)
  beta = 1/(kb*T)

  df = [- 1/beta * np.log(np.mean(np.exp(-beta * (ene_traj["e" + str(j+1)][:step] - ene_traj["eR"][:step])))/np.mean(np.exp(-beta * (ene_traj["e" + str(i+1)][:step] - ene_traj["eR"][:step])))) for i in range(num_endstates) for j in range(i+1, num_endstates)]
        
  return df
    
def plot_fe_convergence(num_ligs, ene_traj, file_out, env, subset, file2 = None):
  """
  plot the convergence of the free energy differences given a RE-EDS energy trajectory
  """
  min_ = 1000
  max_ = -1000
  
  leng = 0
  for i in range(num_ligs-1):
    for j in range(i+1, num_ligs):
      leng += 1
  pal = sns.color_palette("turbo", leng)
  print("leng ", leng)
  sys.stdout.flush()
  
  fig, ax = plt.subplots(nrows=1,ncols=2, dpi=300, figsize=(10,8))
  ax = ax.ravel()
  
  steps = (np.linspace(10, len(ene_traj[0])-1,20))
  steps = [int(s) for s in steps]
  df = []
  
  T = 298.15
  for step in steps:
    df.append(calculate_free_energies(ene_traj[0], T, step))
  
  df_recentered = [[d[i] - df[-1][i] for d in df ] for i in range(len(df[-1]))]
  t = ene_traj[0]["time"]/1000

  print(df_recentered)
  ii = 0
  for i in range(num_ligs):

    for idx in range(i+1, num_ligs):
      print(i, idx)
      ax[1].plot(t[steps], df_recentered[ii], color = pal[ii], label = subset + str(i+1) + "_" + subset + str(idx+1),alpha=0.8)
    
      d = [d[ii] for d in df]
      ax[0].plot(t[steps], d, color = pal[ii])
      min_ = min(min_, d[-1] - abs(d[-1])*0.1-20)
      max_ = max(max_, d[-1] + abs(d[-1])*0.1+20)
      
      ii += 1
  
  ax[0].set_ylim(min_, max_)
  ax[1].set_ylim(-20, 20)
  l = ax[1].legend(ncol=3, prop={'size': 10}, markerscale=1, title = "molecule pairs")
  l.get_title().set_fontsize('13')
  
  ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  
  ax[0].set_xlabel('t / ns')
  ax[1].set_xlabel('t / ns')
  ax[0].set_ylabel('$\Delta G_{' + env + '}^{RE-EDS}$ / kJ mol$^{-1}$')
  ax[1].set_ylabel('$\left(\Delta G_{' + env + '}^{RE-EDS} - \Delta G_{' + env + ',t=' + "{:.1f}".format(t.tolist()[-1]) + '\mathrm{ns}}^{RE-EDS}\\right)$ / kJ mol$^{-1}$')
   
  plt.tight_layout()
  plt.savefig(file_out + ".png", dpi= 300)
  ax[0].cla()
  ax[1].cla()
  plt.cla()
  
def plot_sampling_distribution(num_endstates, ene_traj, ER, file_name, undersampling_thresh_file = None, physical_thresh_file = None):
  """
  plot sampling distribution given a RE-EDS energy trajectory
  """
  if not undersampling_thresh_file is None:

    f = open(undersampling_thresh_file)
    thresh = [float(t) for t in f.readline().split('\t')]
    occur = [0]*num_endstates

  if not physical_thresh_file is None:

    f = open(physical_thresh_file)
    phys_thresh = [float(t) for t in f.readline().split('\t')]
    phys = [0]*num_endstates
  
  max_contrib = [0]*num_endstates
  
  for i in range(len(ene_traj["eR"])):
    corrected_energies = [ene_traj["e" + str(state+1)][i] - ER[state] for state in range(num_endstates)]
    idx = corrected_energies.index(min(corrected_energies))
    max_contrib[idx] += 1
    if not undersampling_thresh_file is None:
      for state in range(num_endstates):
        if(ene_traj["e" + str(state+1)][i] < thresh[state]):
          occur[state] += 1
    if not physical_thresh_file is None:
      for state in range(num_endstates):
        if(ene_traj["e" + str(state+1)][i] < phys_thresh[state]):
          phys[state] += 1
    
  print(max_contrib)
  max_contrib = np.array(max_contrib) / len(ene_traj["eR"]) * 100
  print(max_contrib)
  
  if not undersampling_thresh_file is None:
    print(occur)
    occur = np.array(occur) / len(ene_traj["eR"]) * 100
    plt.bar(range(1, len(occur) + 1), occur, label = "undersampling")
    
  if not physical_thresh_file is None:
    print(phys)
    phys = np.array(phys) / len(ene_traj["eR"]) * 100
    plt.bar(range(1, len(phys) + 1), phys, label = "physical sampling")
    
  plt.bar(range(1, len(max_contrib) + 1), max_contrib, label = "maximum contributing")
  plt.xlabel("end-state")
  plt.ylabel("sampling [%]")
  plt.legend()
  plt.savefig(file_name, dpi = 300)
  
  plt.close()
  
def plot_replica_trace(repdat_file, num_replicas, svals, file_name, subdirs = None, sim_time_per_dir = None):
  """
  plot the replica traces given a repdat file
  """
  if(subdirs is not None):
    repdat = pd.read_csv(repdat_file[0], header = [0], delim_whitespace = True)
    for idx, r in enumerate(repdat_file[1:]):
      repdat_ = pd.read_csv(r, header = [0], delim_whitespace = True)
      repdat_["time"] += sim_time_per_dir * (idx+1)
      repdat = pd.concat([repdat, repdat_], ignore_index = True)
      
  else:
    repdat = pd.read_csv(repdat_file, header = [0], delim_whitespace = True)

  pal = sns.color_palette("turbo", num_replicas)
  for i in [0]:
    trace = []
    time = []
    for index, row in repdat.iterrows():
      if row["position_i"] == i:
        trace.append(row["s_i"])
        time.append(row["time"])
      elif row["position_j"] == i:
        time.append(row["time"])
        trace.append(row["s_j"])
     
    plt.plot(time, trace, label = "replica " + str(i+1), color = pal[i])
  
  plt.yticks(svals)
  plt.yscale("log")
  plt.legend()
  plt.ylabel("s-value")
  plt.yticks(svals)
  ax = plt.gca()
  
  ax.yaxis.set_minor_locator(ticker.NullLocator())  # no minor ticks
  ax.yaxis.set_major_formatter(ticker.ScalarFormatter())  # set regular formatting
  
  plt.xlabel("time / ps")
  plt.savefig(file_name + "_replica_1.png", dpi = 300)
  plt.close()
  for i in range(num_replicas):
    trace = []
    time = []
    for index, row in repdat.iterrows():
      if row["position_i"] == i:
        trace.append(row["s_i"])
        time.append(row["time"])
      elif row["position_j"] == i:
        time.append(row["time"])
        trace.append(row["s_j"])
     
    plt.plot(time, trace, label = "replica " + str(i+1), color = pal[i])
    
  plt.yticks(svals)
  plt.yscale("log")
  plt.legend()
  plt.ylabel("s-value")
  plt.yticks(svals)
  ax = plt.gca()
  
  ax.yaxis.set_minor_locator(ticker.NullLocator())  # no minor ticks
  ax.yaxis.set_major_formatter(ticker.ScalarFormatter())  # set regular formatting
  plt.savefig(file_name + ".png", dpi = 300)
  plt.close()
  
  for i in range(num_replicas):
    trace = []
    time = []
    for index, row in repdat.iterrows():
      if(index > 1000):
        break
      if row["position_i"] == i:
        trace.append(row["s_i"])
        time.append(row["time"])
      elif row["position_j"] == i:
        time.append(row["time"])
        trace.append(row["s_j"])
     
    plt.plot(time, trace, label = "replica " + str(i+1), color = pal[i])
  
  plt.yticks(svals)
  plt.yscale("log")
  plt.legend()
  plt.ylabel("s-value")
  plt.yticks(svals)
  ax = plt.gca()
  
  ax.yaxis.set_minor_locator(ticker.NullLocator())  # no minor ticks
  ax.yaxis.set_major_formatter(ticker.ScalarFormatter())  # set regular formatting
  plt.xlabel("time / ps") 
  plt.savefig(file_name + "_cut.png", dpi = 300)
  plt.close()
  
def read_ene_trajs(ene_traj_files, s_value = None):  
  ene_trajs: List[pd.DataFrame] = []
  
  for idx, in_ene_traj_path in enumerate(ene_traj_files):
    ene_traj = pd.read_csv(in_ene_traj_path, header= [0], delim_whitespace=True)
    ene_traj.columns = ene_traj.columns.str.replace('V_', 'e') # gromos writes e1 instead of V_1
    ene_traj.columns = ene_traj.columns.str.replace('t', 'time') # gromos writes time instead of t
    setattr(ene_traj, "in_path", in_ene_traj_path)
    if s_value is None:
      setattr(ene_traj, "s", f"s{idx}")
    else:
      setattr(ene_traj, "s", s_value)
    setattr(ene_traj, "replicaID", idx)
    ene_trajs.append(ene_traj)
    
  return ene_trajs
  
def plot_ene_trajs(ene_trajs, svalues):
  fig, ax = plt.subplots(nrows=int(np.ceil(len(svalues)/3)),ncols=3, figsize=(12, 12))
  ax = ax.ravel()
  for i in range(len(svalues)):
      ene_traj_ = ene_trajs[i]
      ax[i].set_title("s = " + str(svalues[i]))
      ax[i].plot(ene_traj_["time"], ene_traj_["eR"], label = "$V_R$", marker='o', linewidth=0)

      for j in range(1,num_endstates+1):
          #traj = pd.Series(ene_traj[i], time, name="V1")
          #traj.plot()
          ax[i].plot(ene_traj_["time"], ene_traj_["e" + str(j)], label = "$V_" + str(j) + "$", marker='.', linewidth=0, markersize=1)

  for i in range(len(svalues)):
      ax[i].set_ylim([-1000,1000])
      #ax[i].set_xlim([0,500])
      
  ax[0].legend(fontsize='xx-small', ncol = 2)
  plt.tight_layout()
  plt.savefig("ene_trajs", dpi = 300)
  plt.cla()
  plt.clf()
  plt.close()

## Analysis given a step

#sys.argv[1] should be step name
if(len(sys.argv) < 2):
  print("not enough arguments")
  exit()
  
step = sys.argv[1]

# state optimization analysis
if(step == "a"):
  os.chdir(state_optimization_dir)
  os.makedirs("analysis", exist_ok = True)
  os.chdir("analysis")
  
  ene_traj_files = glob.glob("../simulation/ene_traj_*")
  ene_traj_files = sorted(ene_traj_files, key=lambda x: float(x.split("_")[-1]), reverse = False)
  states = [str(ene_traj_files[i].split('_')[-1]) for i in range(len(ene_traj_files))]
  
  ene_trajs = read_ene_trajs(ene_traj_files, "1")
  
  for i in range(len(ene_traj_files)):
    ER = np.array([-500] * num_endstates)
    ER[i] = 500
    plot_sampling_distribution(num_endstates, ene_trajs[i], ER, "sampling_distribution_state_" + str(int(states[i])+1) + ".png")
  
  plot_ene_trajs(ene_trajs, svalues)
  
  physical_state_occurrence_treshold = find_physical_sampling_pot_tresh(ene_trajs, _vacuum_simulation=vacuum_simulation)

  sampling_analysis(out_path=".", ene_traj_csvs=ene_trajs, s_values=[1] * num_endstates,
                                state_potential_treshold=physical_state_occurrence_treshold, eoffs=[0 for _ in range(num_endstates)])
                                
  next_dir = "."                                
  out_file = open(next_dir + "/state_occurence_physical_pot_thresh.csv", "w")
  out_file.write("\t".join(map(str, physical_state_occurrence_treshold)))
  out_file.close()    

# lower bound analysis
elif(step == "b"):
  os.chdir(lower_bound_dir)
  os.makedirs("analysis", exist_ok = True)
  os.chdir("analysis")
  
  ER = np.array([0] * num_endstates)
  
  ene_traj_files = glob.glob("../simulation/ene_traj_*")
  ene_traj_files = sorted(ene_traj_files, key=lambda x: float(x.split("_")[-1]), reverse = True)
  svals = [str(ene_traj_files[i].split('_')[-1]) for i in range(len(ene_traj_files))]
  
  ene_trajs = read_ene_trajs(ene_traj_files)
  
  for i in range(len(ene_traj_files)):
    plot_sampling_distribution(num_endstates, ene_trajs[i], ER, "sampling_distribution_s_" + svals[i] + ".png")
  
  plot_ene_trajs(ene_trajs, svalues)
  
  state_undersampling_pot_treshold = find_undersampling_pot_tresh(ene_traj_csvs=ene_trajs, verbose = True, sampling_fraction_treshold = 0.95)

  sampling_analysis_results, out_plot_dirs = reeds.function_libs.analysis.sampling.detect_undersampling(out_path = ".",
                                                                                                       ene_traj_csvs = ene_trajs, eoffs= [0 for _ in 
                                                                                                       range(num_endstates)],
                                                                                                       s_values = [float(s) for s in svals],
                                                                                                       state_potential_treshold=state_undersampling_pot_treshold, verbose = True)
                                                                                                       

  out_analysis_next_dir = "."
  #bash.make_folder(out_analysis_next_dir, "-p")

  print(sampling_analysis_results)
  u_idx = sampling_analysis_results["undersamplingThreshold"]
  print(u_idx)
  # Make the new s-distribution based on this 
  print("undersampling found after replica: " + str(u_idx) + ' with s = ' + str(svals[u_idx]))    
  print('New s distribution will place ' + str(num_endstates) + ' replicas between  s = ' + str(svals[u_idx]) + ' and s = ' +str(svals[u_idx+3]))

  new_sdist = svals[:u_idx-2]
  print(svals[u_idx-1], svals[u_idx])
  lower_sdist = s_log_dist.get_log_s_distribution_between(float(svals[u_idx-1]), float(svals[u_idx]), num_endstates)
  new_sdist.extend(lower_sdist)

  # Write the s-values to a csv file
  out_file = open(out_analysis_next_dir + "/s_vals.csv", "w")
  out_file.write("\t".join(list(map(str, new_sdist))))
  out_file.write("\n")
  out_file.close()

  # Write the potential energy thresholds to a csv file
  out_file = open(out_analysis_next_dir + "/state_occurence_pot_thresh.csv", "w")
  out_file.write("\t".join(map(str, sampling_analysis_results["potentialThreshold"])))
  out_file.write("\n")
  out_file.close()
  
# energy offset estimation analysis
elif(step == "c"):
  os.chdir(eoff_estimation_dir)
  dir = os.path.abspath(sys.argv[2])
  os.makedirs("analysis", exist_ok = True)
  os.chdir("analysis")
  
  ene_traj_files = glob.glob(dir + "/ene_traj_*")
  ene_traj_files = sorted(ene_traj_files, key=lambda x: float(x.split("_")[-1]))
  s_val_file = lower_bound_dir + "/analysis/s_vals.csv"
  file = open(s_val_file)
  line = file.readline()
  svals = [float(s) for s in line.split("\t")]
  num_replicas = len(svals)
  ER = np.array(np.array([([0]*num_endstates)]*num_replicas))
  print(ER)
  
  ene_trajs = read_ene_trajs(ene_traj_files)
  
  df = calculate_free_energies(ene_trajs[0], 298.15)
  for f in df:
    print(f)
  
  undersampling_file = lower_bound_dir + "/analysis/state_occurence_pot_thresh.csv"
  file = open(undersampling_file)
  line = file.readline()
  undersampling_thresh = np.array([float(t) for t in line.split("\t")])
  
  physical_sampling_file = state_optimization_dir + "/analysis/state_occurence_physical_pot_thresh.csv"
  file = open(physical_sampling_file)
  line = file.readline()
  physical_sampling_thresh = np.array([float(t) for t in line.split("\t")])
  
  repdat_file = f"{dir}/repdat_{system_name}"
  #plot_replica_trace(repdat_file, num_replicas, svals, "replica_trace")
  
  for i in range(len(ene_traj_files)):
    plot_sampling_distribution(num_endstates, ene_trajs[i], ER[0], "sampling_distribution_s_" + str(svals[i]) + ".png", undersampling_thresh_file = undersampling_file, physical_thresh_file = physical_sampling_file)
  
  plot_ene_trajs(ene_trajs, svalues)
  
  (sampling_results, out_dir) = sampling_ana.detect_undersampling(out_path = ".", ene_traj_csvs = ene_trajs,_visualize=True, 
                                                                            s_values = svals, eoffs=ER, 
                                                                            state_potential_treshold= undersampling_thresh, 
                                                                            undersampling_occurence_sampling_tresh=0.9)
  temp = 298.15
  print("calc Eoff: ")
  # WARNING ASSUMPTION THAT ALL EOFF VECTORS ARE THE SAME!
  print("\tEoffs(" + str(len(ER[0])) + "): ", ER[0])
  print("\tS_values(" + str(len(svals)) + "): ", svals)
  print("\tsytsemTemp: ", temp)
  # set trim_beg to 0.1 when analysing non equilibrated data

  # Decrement the value of undersampling_idx by 1. As indexing followed a different convention. 
  new_eoffs_estm, all_eoffs = eds_energy_offsets.estimate_energy_offsets(ene_trajs = ene_trajs, initial_offsets = ER[0], sampling_stat=sampling_results, s_values = svals ,out_path = ".", temp = temp, trim_beg = 0., undersampling_idx = sampling_results['undersamplingThreshold'],plot_results = True, calc_clara = False)
  print("ENERGY OFF: ", new_eoffs_estm, all_eoffs)
  
  file = open("energy_offsets.csv", "w")

  for e in new_eoffs_estm:
    file.write(str(e) + " ")
 
# s-optimization analysis  
elif(step == "d"):
  dir = sys.argv[2]
  os.chdir(dir)
  os.makedirs("analysis", exist_ok = True)
  os.chdir("analysis")
  
  ene_traj_files = glob.glob("../simulation/ene_traj_*")
  ene_traj_files = sorted(ene_traj_files, key=lambda x: float(x.split("_")[-1]))
  s_val_file = "../simulation/s_vals.csv"
  file = open(s_val_file)
  line = file.readline()
  svalues = np.array([float(s) for s in line.split(" ") if len(s)])
  num_replicas = len(svalues)
  energy_offset_file = eoff_estimation_dir + "/analysis/energy_offsets.csv"

  file = open(energy_offset_file)
  line = file.readline()
  ER = np.array([float(s) for s in line.split()])
  num_endstates = len(ER)

  print(svalues)
  print(ER)
  
  ene_trajs = read_ene_trajs(ene_traj_files)
  
  undersampling_file = lower_bound_dir + "/analysis/state_occurence_pot_thresh.csv"
  file = open(undersampling_file)
  line = file.readline()
  undersampling_thresh = np.array([float(t) for t in line.split("\t")])
  
  physical_sampling_file = state_optimization_dir + "/analysis/state_occurence_physical_pot_thresh.csv"
  file = open(physical_sampling_file)
  line = file.readline()
  physical_sampling_thresh = np.array([float(t) for t in line.split("\t")])
  
  repdat_file = f"repdat_{system_name}"
  
  #plot_replica_trace(repdat_file, num_replicas, svalues, "replica_trace_wat")
  
  df = calculate_free_energies(ene_trajs[0], 298.15)
  for d in df:
    print(d)
  
  for i in range(len(ene_traj_files)):
    plot_sampling_distribution(num_endstates, ene_trajs[i], ER, "sampling_distribution_s_" + str(svalues[i]) + ".png", undersampling_thresh_file = undersampling_file, physical_thresh_file = physical_sampling_file)
  
  plot_ene_trajs(ene_trajs, svalues)
  
  exchange_data = repdat.Repdat(f"../simulation/repdat_gromos_{system_name}")
  exchange_freq = repex.calculate_exchange_freq(exchange_data)
  transitions = exchange_data.get_replica_traces() 
  print(exchange_freq)
  
  svals = parameter_optimization.optimize_s(in_file=f"../simulation/repdat_gromos_{system_name}", out_dir=".",
                                                                                   title_prefix="s_opt",
                                                                                   add_s_vals=4, 
                                                                                   run_NLRTO=True, run_NGRTO=False,
                                                                                   verbose=True)
   
  print(svals)                                                                                   
  file = open("s_vals_new.csv", "w")
  #ss = [1.] * (num_endstates-1)
  #ss.extend(svals[1])
  for s in svals[1]:
    file.write(str(s) + " ") 
                                                                                   
  print("\t\tvisualize transitions")
  parameter_optimization.get_s_optimization_transitions(transitions = transitions, out_dir=".", repdat=exchange_data, title_prefix="s_opt", undersampling_thresholds = undersampling_thresh)
  re_plots.plot_exchange_freq(svalues, exchange_freq, outfile = 'exchange_frequencies.png')   
  
  print("\t\tshow roundtrips")
  in_repdat_file = exchange_data

  # retrieve data:
  print("get replica transitions")
  s_values = in_repdat_file.system.s
  trans_dict = in_repdat_file.get_replica_traces()

  # plot
  print("Plotting Histogramm")
  re_plots.plot_repPos_replica_histogramm(out_path="replica_repex_pos.png", 
                                          data=trans_dict, title="s_opt", s_values=svalues) 

# energy offset rebalancing analysis                                                                                  
elif(step == "e"):
  dir = sys.argv[2]
  os.chdir(dir)
  os.makedirs("analysis", exist_ok = True)
  os.chdir("analysis")
  
  iteration = int(dir.split('_')[-1])
    
  ene_traj_files = glob.glob("../simulation/ene_traj_*")
  ene_traj_files = sorted(ene_traj_files, key=lambda x: float(x.split("_")[-1]))
  s_val_file = "../simulation/s_vals.csv"
  file = open(s_val_file)
  line = file.readline()
  svalues = np.array([float(s) for s in line.split(" ") if len(s)])
  num_replicas = len(svalues)
  energy_offset_file = "../simulation/energy_offsets.csv"
  
  

  file = open(energy_offset_file)
  line = file.readline()
  ER = np.array(np.array([[float(s) for s in line.split()] for i in range(num_replicas)]))
  num_endstates = len(ER[0])
  
  #svalues = svalues[num_endstates-1:]

  print(svalues)
  print(ER)
  
  ene_trajs = read_ene_trajs(ene_traj_files)
  
  df = calculate_free_energies(ene_trajs[0], 298.15)
  for d in df:
    print(d)
  
  undersampling_file = lower_bound_dir + "/analysis/state_occurence_pot_thresh.csv"
  file = open(undersampling_file)
  line = file.readline()
  undersampling_thresh = np.array([float(t) for t in line.split("\t")])
  
  physical_sampling_file = state_optimization_dir + "/analysis/state_occurence_physical_pot_thresh.csv"
  file = open(physical_sampling_file)
  line = file.readline()
  physical_sampling_thresh = np.array([float(t) for t in line.split("\t")])
  
  repdat_file = f"../simulation/repdat_{system_name}"
  
  #plot_replica_trace(repdat_file, num_replicas, svalues, "replica_trace_wat")
  
  for i in range(len(ene_traj_files)):
    plot_sampling_distribution(num_endstates, ene_trajs[i], ER[i], "sampling_distribution_s_" + str(i) + ".png", undersampling_thresh_file = undersampling_file, physical_thresh_file = physical_sampling_file)
  
  plot_ene_trajs(ene_trajs, svalues)
  
  print(svalues)
  
  (sampling_results, out_dir) = sampling_ana.detect_undersampling(out_path = ".", ene_traj_csvs = ene_trajs,_visualize=True, 
                                                                            s_values = svalues, eoffs=ER, 
                                                                            state_potential_treshold= undersampling_thresh, 
                                                                            undersampling_occurence_sampling_tresh=0.9)
                                                                            
  pseudocount = (1/num_endstates)/rebalancing_intensity_factor 
  if(isinstance(rebalancing_learning_factors, Iterable)):
    learning_factor = rebalancing_learning_factors[iteration-1] 
  else:
    learning_factor = rebalancing_learning_factors   
    
  print("learning factor:", learning_factor)                                                                   
  
  new_eoffs_rb = rebalance_eoffs_directCounting(sampling_stat=sampling_results['samplingDistributions'], old_eoffs=ER,
                                                       learningFactor=learning_factor, pseudo_count=pseudocount,
                                                       correct_for_s1_only=True, temperature = 298.15)
  new_eoffs_rb = new_eoffs_rb.T
  print(new_eoffs_rb)
  
  file = open("eoffs_new.csv", "w")
  for e in new_eoffs_rb:
    file.write(str(e[0]) + " ")

# production analysis  
elif(step == "f"):
  os.chdir(production_dir)
  os.makedirs("analysis", exist_ok = True)
  os.chdir("analysis")
  
  ene_trajs: List[pd.DataFrame] = []
  
  subdirs = sorted(glob.glob("../simulation_*"), key = lambda x: float(x.split("_")[-1]))
  ene_traj_files = glob.glob(subdirs[0] + "/ene_traj_*")
  ene_traj_files = sorted(ene_traj_files, key=lambda x: float(x.split("_")[-1]))
  
  print(subdirs)
  
  for idx, in_ene_traj_path in enumerate(ene_traj_files):
    ene_traj = pd.read_csv(in_ene_traj_path, header= [0], delim_whitespace=True)
    ene_traj.columns = ene_traj.columns.str.replace('V_', 'e') # gromos writes e1 instead of V_1
    ene_traj.columns = ene_traj.columns.str.replace('t', 'time') # gromos writes time instead of t
    setattr(ene_traj, "in_path", in_ene_traj_path)
    setattr(ene_traj, "s", f"s{idx}")
    setattr(ene_traj, "replicaID", idx)
    ene_trajs.append(ene_traj)
    sim_time_per_dir = ene_traj["time"][ene_traj.index[-1]]
          
  for d in subdirs[1:]:
    print(d)
    ene_traj_files = glob.glob(d + "/ene_traj_*")
    ene_traj_files = sorted(ene_traj_files, key=lambda x: float(x.split("_")[-1]))
    for idx, in_ene_traj_path in enumerate(ene_traj_files):
      ene_traj = pd.read_csv(in_ene_traj_path, header= [0], delim_whitespace=True)
      ene_traj.columns = ene_traj.columns.str.replace('V_', 'e') # gromos writes e1 instead of V_1
      ene_traj.columns = ene_traj.columns.str.replace('t', 'time') # gromos writes time instead of t
      ene_traj["time"] += sim_time_per_dir*(int(d.split('_')[-1]))
      
      setattr(ene_traj, "in_path", in_ene_traj_path)
      setattr(ene_traj, "s", f"s{idx}")
      setattr(ene_traj, "replicaID", idx)

      ene_trajs[idx] = pd.concat([ene_trajs[idx], ene_traj], ignore_index = True)  
      
  s_val_file = "../simulation_0/s_vals.csv"
  file = open(s_val_file)
  line = file.readline()
  svalues = np.array([float(s) for s in line.split(" ") if len(s)])
  num_replicas = len(svalues)
  energy_offset_file = "../simulation_0/energy_offsets.csv"
  
  file = open(energy_offset_file)
  line = file.readline()
  ER = np.array([float(s) for s in line.split()])
  num_endstates = len(ER)
  
  #svalues = svalues[num_endstates-1:]

  print(svalues)
  print(ER)
  
  plot_fe_convergence(6, ene_trajs, "convergence", environment, dataset_name)
  df = calculate_free_energies(ene_trajs[0], 298.15)
  for d in df:
    print(d)
    
  df_file = open(f"df_{system_name}.csv", "w")
  df_file.write("df\n")
  for d in df:
    df_file.write(str(d))
    df_file.write("\n")
  df_file.close()
  
  undersampling_file = lower_bound_dir + "/analysis/state_occurence_pot_thresh.csv"
  file = open(undersampling_file)
  line = file.readline()
  undersampling_thresh = np.array([float(t) for t in line.split("\t")])
  
  physical_sampling_file = state_optimization_dir + "/analysis/state_occurence_physical_pot_thresh.csv"
  file = open(physical_sampling_file)
  line = file.readline()
  physical_sampling_thresh = np.array([float(t) for t in line.split("\t")])
  
  repdat_files = [d + f"/repdat_{system_name}" for d in subdirs]
  
  plot_replica_trace(repdat_files, num_replicas, svalues, "replica_trace_wat", subdirs, sim_time_per_dir)
  
  for i in range(len(ene_traj_files)):
    plot_sampling_distribution(num_endstates, ene_trajs[i], ER, "sampling_distribution_s_" + str(svalues[i]) + ".png", undersampling_thresh_file = undersampling_file, physical_thresh_file = physical_sampling_file)
  
  plot_ene_trajs(ene_trajs, svalues)
  
else:
  print("unknown step argument")
