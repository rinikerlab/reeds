from reeds.function_libs.file_management import file_management as fM

import copy
import pandas as pd
import numpy as np
from scipy import stats
from scipy import constants as const
from scipy import special

from mpmath import * # This is for floating point arithmetic ! 
mp.dps = 15

from typing import List

#
# Main call point
#

def estimate_energy_offsets(ene_trajs: List[pd.DataFrame], initial_offsets: List[float],
                            s_values: List[float], out_path: str, temp: float = 298.0,
                            pot_tresh: float = 0.0, frac_tresh: float = [0.9],
                            trim_beg: float = 0.1):

    """
    This function will estimate the energy offsets one should use to    
    obtain equal sampling of all end states in a RE-EDS simulation. 
    This estimation is based on the potential energies of a prior 
    RE-EDS simulation, in which energy offsets were all set to 0. 
    
    This function calculates the energy offsets of all replicas given, 
    determines which of these replicas can be considered to be undersampling
    replicas, and then returns the average value of the offsets from these 
    undersampling replicas.     
    
    Parameters
    ----------
    ene_trajs: List of pandas DataFrames 
        contains all replicas potential energies 
    initial_offsets: List[float]
        the set of energy offsets used in the simulation which generated this data
    s_values: List[float]
        the set of s-values used in the simulation which generated this data    
    out_path: str
        path to the file in which the results will be printed out.    
    temp: float
        temperature of the simulation
    pot_tresh: float
        energy threshold to consider a state as undersampling
    frac_tresh: List [float]    
        fraction of energies which need to be below the threshold to be considered as
        undersampling
    trim_beg: float
        fraction of the values to remove at the begining for "equilibration"
    Returns
    -------
        all_eoffs: np.array
            Array containing the energy offsets of each replica
        means: np.array
            Array containing the energy offsets to use in the next simulations        
    """

    num_states = len(initial_offsets)
    num_replicas = len(ene_trajs)

    f = open(out_path, "w")

    all_eoffs = np.zeros(num_states*num_replicas).reshape(num_replicas, num_states)
    all_eoffs_2 = np.zeros(num_states*num_replicas).reshape(num_replicas, num_states)    
    
    # Calculate the Energy Offsets for reach replica
    for i in range(num_replicas):
        all_eoffs[i] = calc_offsets(ene_trajs[i], temp, trim_beg, num_states)
        (all_eoffs_2[i], converged, steps) = calc_offsets_clara_eqn(ene_trajs[i], temp, num_states, initial_offsets, trim_beg)

    f.writelines(format_as_jnb_table("Energy offsets predicted for each replica", s_values, all_eoffs, 2))
    f.writelines(format_as_jnb_table("Energy offsets predicted for each replica - Clara's eqn", s_values, all_eoffs_2, 2))

    # Analyse the data in the replica, and find when we can consider undersampling to start

    (tables, undersampling_idx) = analyse_replicas(ene_trajs, num_states, s_values, pot_tresh = pot_tresh, frac_tresh = frac_tresh)
    f.writelines(tables)
    
    # Take the average from the undersampling replicas    
    means = None
    stdevs = None 
    
    if undersampling_idx is None: 
            f.writelines('\nDid not find any undersampling replicas.')
    else: 
        undersampling_eoffs = all_eoffs[undersampling_idx:]
        results = 'Undersampling found at replica ' + str(undersampling_idx+1) \
                  + ' with s = ' + str(s_values[undersampling_idx]) + '\n\n' 
        results += 'New energy offset for each state:\n'
        means   = undersampling_eoffs.mean(axis=0)
        stdevs  = undersampling_eoffs.std(axis=0)
        for i in range(num_states):
            results += 'state ' + str(i+1) + ' : ' + str(round(means[i], 2)) \
                    + ' +-  ' + str(round(stdevs[i], 2)) + '\n'
        f.writelines(results)
    f.close()
    return (all_eoffs, means)


#
# Functions calculating the energy offsets by aplying the equations
#

def calc_offsets(energy_trajectory: pd.DataFrame, temp:float, trim_beg:float, num_states:int):

    """
    This function applies eqn. 6  of Sidler et al., J. Chem. Phys. 2016, 145, 154114 
    to estimate the energy offsets for a specific replica.
    Note that the original offset do not go into this equation!
    Note: We use special.logsumexp, (logsum and not log average) as 
          all states have the name number of data points, and log(1/N) cancels out.

    Parameters
    ----------
        energy_trajectory: pandas DataFrame 
            contains the potential energies of the end state and the ref. state
        temp: float
            temperature in Kelvin
        trim_beg: float
            fraction of the values to remove at the begining for "equilibration"
        num_states: int
            number of end states in our RE-EDS simulation
                
    Returns
    -------
        new_eoffs: np.array
            Energy offsets estimated for this replica scaled such that
            ligand 1 has an offset of 0. 
    """

    beta =  1000 * 1 / (temp * const.k * const.Avogadro)
    new_eoffs = np.zeros(num_states)
    
    initial_offsets = np.zeros(num_states)
    
    # note exp_term is a vector (each element is a timestep) 
    # containing the term to be exponentiated

    for i in range(num_states):
            trim_beg = int(trim_beg*len(energy_trajectory['e' + str(i+1)]))
        
            v_i = np.array(energy_trajectory['e' + str(i+1)])[trim_beg:]
            v_r = np.array(energy_trajectory['eR'])[trim_beg:]
            exp_term = - beta * (v_i - v_r - initial_offsets[i])
            new_eoffs[i] =  -(1/beta) * special.logsumexp(exp_term)

    return (new_eoffs - new_eoffs[0])

def calc_offsets_clara_eqn(energy_trajectory, temp:float, num_states:int, initial_offsets:List[float], trim_beg:float):

    """
    This function applies eqn. 5 of Sidler et al., J. Chem. Phys. 2016, 145, 154114 
    to estimate the energy offsets for a specific replica.
    
    Note: This function doesn't apply the 'reweighting mentionned in C. Christ's
          original EDS publications.
    
    Note: The function is numerically stable.

    Parameters
    ----------
        energy_trajectory: pandas DataFrame 
            contains the potential energies of the end state and the ref. state
        temp: float
            temperature in Kelvin
        trim_beg: float
            fraction of the values to remove at the begining for "equilibration"
        num_states: int
            number of end states in our RE-EDS simulation
        initial_offsets: List [float]        
            energy offset values used in the simulation which generated this data.
                
    Returns
    -------
        new_eoffs: np.array
            Energy offsets estimated for this replica scaled such that
            ligand 1 has an offset of 0.
        converged: bool
            True if calculation converged
        steps: int
            Number of steps till convergence 
    """

    beta =  1000 * 1 / (temp * const.k * const.Avogadro)
    new_eoffs = np.array(initial_offsets)
    converged = False
    steps = 0 
    
    trim_beg = int(trim_beg*len(energy_trajectory['e1']))

    while not converged: 
        tmp_eoffs = copy.copy(new_eoffs)        
        for i in range(num_states):
            v_i = np.array(energy_trajectory['e' + str(i+1)])[trim_beg:]
            e_i = tmp_eoffs[i]
            deltaV = np.zeros(len(v_i))
            for j in range(num_states):
                if i == j: continue           
                v_j = np.array(energy_trajectory['e' + str(j+1)])[trim_beg:]
                e_j = tmp_eoffs[j]
                    
                deltaV += (v_j - v_i) - (e_j - e_i)
        
            # here we replace all very low values of deltaV by a minimum cap
            # to avoid overflow. (Overflow as sign is reversed below)
            deltaV[deltaV < -1000.0] = -1000.0
        
            # now we scale and average:
            new_eoffs[i] = -  (1/beta) * np.log(np.average(1/(1+(np.exp(-beta*deltaV)))))
    
        # Note: we need to add the initial offsets to this too !!
        new_eoffs += tmp_eoffs
        new_eoffs -= new_eoffs[0]
            
        if np.sum(np.abs(new_eoffs - tmp_eoffs)) < 0.1*num_states : converged = True
        steps += 1
        if converged or steps > 300: break
    
    # If the calculation did not converge, we just set them all to 0.
    if not converged: new_eoffs = np.zeros(num_states)         
    return (new_eoffs, converged, steps)
 
#
# Functions analysing the replicas
#

def undersampling_detection(ene_trajs: pd.DataFrame, num_states:int, s_values:List[float], bandwidth:float = 300.0):
    """
    This function determines which of the replicas can be considered to be undersampling
    replicas. It will ensure all data points are below a particular energy, which will
    be determined by a maximum bandwidth (or the standard deviation of the distribution).
    
    Parameters
    ----------
    ene_trajs: List of pd.DataFrame
        contains all replicas potential energies 
    num_states: int
        number of end states
    s_values: List[float]
        the set of s-values used in the simulation which generated this data    
    bandwidth: float
        bandwidth added to the min pot. energy to consider a state as undersampling (kJ/mol)
    
    Returns
    -------
    undersampling_idx: int
        index of the first replica which can be considered an undersampling replica
    """

    # number of timesteps
    num_timesteps = len(np.array(ene_trajs[0]['e1']))

    for i, traj in enumerate(ene_trajs):
        limit_reached = True
        for j in range(1, num_states+1):
            pot_ene = np.array(ene_trajs[i]['e' + str(j)])
            threshold_ene = max(min(3*np.std(pot_ene), bandwidth), 30)
            
            if ((np.sum(pot_ene < threshold_ene)) / num_timesteps) < 0.999:
                limit_reached = False
        if limit_reached:
            print ('we found undersampling for replica ' + str(i+1) + ' with '
                   's = ' + str(s_values[i]))
            return i

    return None

def analyse_replicas(ene_trajs: pd.DataFrame, num_states:int, s_values:List[float], pot_tresh, frac_tresh: float):
    """
    This function determines which of the replicas can be considered to be undersampling
    replicas, and a few other properties of the potential distributions.     
    
    Parameters
    ----------
    ene_trajs: List of pandas DataFrames 
        contains all replicas potential energies 
    num_states: int
        number of end states
    s_values: List[float]
        the set of s-values used in the simulation which generated this data    
    pot_tresh: float
        energy threshold to consider a state as undersampling
    frac_tresh: List [float]    
        fraction of energies which need to be below the threshold to be considered as
        undersampling
    
    Returns
    -------
    table: str
        Data formatted as a table to be printed to a file
    undersampling_idx: int
        index of the first undersampling replica        
    """

    select_states = ['e' + str(i) for i in range(1, num_states+1)]
    num_replicas = len(ene_trajs)

    # Calculate the minimum energy counts. 
    min_counts = np.zeros([num_replicas, num_states])

    for i, traj in enumerate(ene_trajs):
        for lowest_state in traj[select_states].idxmin(axis=1).replace("e", "", regex=True):
            min_counts[i][(int(lowest_state)-1)] += 1

    title = "Minimum potential energy count per replica"
    min_table = format_as_jnb_table(title, s_values, min_counts, 0)

    # Find the negative energy counts:
    neg_counts = np.zeros([num_replicas, num_states])

    for i, traj in enumerate(ene_trajs):
        for j, state in enumerate(select_states):
            v = np.array(traj[state])
            neg_counts[i][j] = np.size(v[v<0])

    title = "Negative potential energy count per replica"
    neg_table = format_as_jnb_table(title, s_values, neg_counts, 0)

    # Find the index of the first undersampling replica:
    # for now we will keep it hardcoded at 7 for my CHK1 runs
    undersampling_idx = undersampling_detection(ene_trajs, num_states, s_values, bandwidth = 300)

    return (min_table + '\n' + neg_table + '\n', undersampling_idx)


#
# Formatting functions for the output 
#

def format_as_jnb_table(title:str, s_values:List[float], data:np.array, num_decimal:int):
    """
    This function converts data from a np.array format into 
    a jupyter notebook compatible format for a table.     
    
    Parameters
    ----------
    title: str
        title to write above the table
    s_values: List[float]
        the set of s-values used in the simulation which generated this data    
    data: np.array
        2-D array containing the data to print out
    num_decimal: int
        number of decimals points to print the numbers as    
    
    Returns
    -------
    table: str
        data formatted as a table to be printed to a file
    """
    table = '\n' + title + '\n'
    sep ='\t|\t'
    header = [ 'state ' + str(i+1) for i in range(len(data[0]))]

    table += '| s\t' + sep + '\t|'.join(header) + '|\n'
    table += '|---\t' * (len(header)+1)+ '|\n'

    for i, s_val in enumerate(s_values):
        line = '| ' + '%.5f' % s_val + sep
        for d in data[i]: line += ('%.' + str(num_decimal) + 'f') % d + '\t|'
        line += '\n'
        table += line
    return table
