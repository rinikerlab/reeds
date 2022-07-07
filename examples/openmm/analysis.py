import os
import pandas as pd
import glob
from openmm import unit as u
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
import sys


"""
This script performs a very general analysis of the RE-EDS calculations for a set of six small molecules in vacuum and water to calculate relative hydration free energies (see Rieder et al. J. Chem. Inf. Model. 2022, 62 for description of the dataset). Execute as 'python analysis.py' after executing the two scripts set_A_water.py and set_A_vacuum.py.
"""

def calculate_free_energies(ene_traj, T, step = None):
    """
    calculates the free-energy differences between all end-states for a given energy trajectory

    Parameters
    ----------
    ene_traj: pandas Dataframe
      energy trajectory from a RE-EDS simulation
    T: float
      temperature
    step: int
      number of values of the energy trajectory to take into account (used for convergence plotting)
    Returns
    -------
    df: list
      list of all pairwise free-energy differences
    """

    if step is None:
      step = -1
    kb = (u.BOLTZMANN_CONSTANT_kB*u.AVOGADRO_CONSTANT_NA).value_in_unit(u.kilojoule_per_mole/u.kelvin)
    beta = 1/(kb*T)
    df = [- 1/beta * np.log(np.mean(np.exp(-beta * (ene_traj["V_" + str(j+1)][:step] - ene_traj["V_R"][:step])))/np.mean(np.exp(-beta * (ene_traj["V_" + str(i+1)][:step] - ene_traj["V_R"][:step])))) for i in range(num_endstates) for j in range(i+1, num_endstates)]    
    return df
    
def plot_fe_convergence(num_ligs, ene_traj, file_out, environment, subset, T):
  """
  plots the convergence of the RE-EDS free-energy calculation

  Parameters
  ----------
  num_ligs: int
    number of end-states
  ene_traj: pandas Dataframe
    energy trajectory from a RE-EDS simulation
  file_out: string
    name of output png file
  environment: string
    name of environment for plot legends (e.g. "water", "vacuum", ...)
  subset: string
    name of dataset (e.g. "A", "PNMT", ...)
  T: float
    temperature
  """
  
  min_ = 1000
  max_ = -1000
  
  pal = sns.color_palette("turbo", num_ligs)
  
  fig, ax = plt.subplots(nrows=1,ncols=2, dpi=300, figsize=(10,8))
  ax = ax.ravel()
  
  steps = (np.linspace(10, len(ene_traj[0])-1,20))
  steps = [int(s) for s in steps]
  df = []
  
  for step in steps:
    df.append(calculate_free_energies(ene_traj[0], T, step))
  
  df_recentered = [[d[i] - df[-1][i] for d in df ] for i in range(len(df[-1]))]
  t = ene_traj[0]["t"]/1000
  ref = 0
  for i in range(num_ligs-1):

    lab = subset + str(i+1)
    for idx in range(ref,ref+num_ligs-i-1):
      ax[1].plot(t[steps], df_recentered[idx], color = pal[i], label = lab,alpha=0.8)
      lab = '_nolegend_'
    
    for idx in range(ref,ref+num_ligs-i-1):
      d = [d[idx] for d in df]
      ax[0].plot(t[steps], d, color = pal[i])
      min_ = min(min_, d[-1] - abs(d[-1])*0.1-20)
      max_ = max(max_, d[-1] + abs(d[-1])*0.1+20)
    
    ref += num_ligs - i -1
  
  ax[0].set_ylim(min_, max_)
  ax[1].set_ylim(-20, 20)
  l = ax[1].legend(ncol=3, prop={'size': 12}, markerscale=1, title = "reference molecule")
  l.get_title().set_fontsize('15')
  
  ax[0].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  ax[1].yaxis.set_major_formatter(FormatStrFormatter('%.1f'))
  
  ax[0].set_xlabel('t / ns')
  ax[1].set_xlabel('t / ns')
  ax[0].set_ylabel('$\Delta G_{' + environment + '}^{RE-EDS}$ / kJ mol$^{-1}$')
  ax[1].set_ylabel('$\left(\Delta G_{' + environment + '}^{RE-EDS} - \Delta G_{' + environment + ',t=' + "{:.1f}".format(t.tolist()[-1]) + '\mathrm{ns}}^{RE-EDS}\\right)$ / kJ mol$^{-1}$')
   
  plt.tight_layout()
  plt.savefig(file_out + ".png", dpi= 300)
  ax[0].cla()
  ax[1].cla()
  plt.cla() 
  
def plot_ene_traj(ene_traj, s_values):
  """
  plots the energy trajectories of all replicas 
  
  Parameters
  ----------
  ene_traj: pandas Dataframe
    energy trajectory from a RE-EDS simulation
  s_values: np.array
    list of s values
  
  """
  nrows = int(np.ceil(len(ene_traj)/3))
  ncols=3
  
  fig, ax = plt.subplots(nrows=nrows, ncols=3, figsize=(12, 12), sharex = True, sharey = True)
  ax = ax.ravel()
  for i in range(len(ene_traj)):
      ene_traj_ = ene_traj[i]
      ax[i].plot(ene_traj_["t"], ene_traj_["V_R"], label = "$V_R$", marker='o', linewidth=0)

      for j in range(1,num_endstates+1):
          ax[i].plot(ene_traj_["t"], ene_traj_["V_" + str(j)], label = "$V_" + str(j) + "$", marker='.', linewidth=0, markersize=1)

  for i in range(len(ene_traj)):
      ax[i].set_ylim([-1000,1000])
      ax[i].set_ylabel("V / kJ mol$^{-1}$")
      ax[i].set_xlabel("t / ps")
      ax[i].set_title("s = " + str(s_values[i]))
      
      
  ax[0].legend(fontsize='xx-small', ncol = 2)
  plt.tight_layout()
  plt.savefig("ene_trajs", dpi = 300)
  plt.cla()
  plt.clf()
  plt.close()


num_endstates = 6
T = 298.15
s_values_vac = np.array([1.0,0.518,0.393,0.3305,0.268,0.225,0.182,0.1605,0.139,0.1055,0.0887,0.072,0.0546,0.0373,0.0193,0.01])
s_values_wat = np.array([1.0,0.518,0.393,0.3305,0.268,0.225,0.182,0.1605,0.139,0.1055,0.0887,0.072,0.0546,0.0373,0.0193,0.01])

for dir, sval in zip(["vacuum", "water"], [s_values_vac, s_values_wat]):
  print(dir)
  os.chdir(dir)

  ene_traj_files = glob.glob("ene_traj_*")
  ene_traj_files = sorted(ene_traj_files, key=lambda x: float(x.split("_")[-1]), reverse = False)
  ene_traj = [pd.read_csv(name, header = [0], delim_whitespace = True) for name in ene_traj_files]

  df = calculate_free_energies(ene_traj[0], T)
  print("free energies:")
  for d in df:
    print(d)
  
  print("plotting convergence in " + dir + "/convergence.png")
  plot_fe_convergence(num_endstates, ene_traj, "convergence.png", dir[:3], "A", T)
  print("plotting energy trajectories in " + dir + "/ene_traj.png")
  plot_ene_traj(ene_traj, sval)
  os.chdir('..')
  print()

