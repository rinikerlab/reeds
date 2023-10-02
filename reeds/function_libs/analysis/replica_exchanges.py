from typing import List, Dict

import numpy as np

from reeds.submodules.pygromos.pygromos.files.repdat import ExpandedRepdat

def calculate_exchange_freq(exchange_data):
    """
    This function calculates the exchange frequency for each of the replicas.
    
    Parameters
    ----------
    exchange_data: Repdat object
        instance of the Repdat class which contains all of the information 
        related to the exchanges during a RE-EDS simulation.
    Returns
    ----------
    exchanges: np.array() 
        numpy array containing the exchange frequencies at each s-value.
        the array has length (N-1), with N being the number of s-values.
    """
    n_replicas = len(exchange_data.system.s)
    exchange_trials = 0.5 * len(exchange_data.DATA.run)/n_replicas

    exchanges = np.zeros(n_replicas-1)
    for i, row in exchange_data.DATA.iterrows():
        if row.s == 1 and row.ID <= row.partner:
            exchanges[row.ID-1] += 1

    # Normalize by the number of exchange trials to get probability
    exchanges /=  exchange_trials
    return exchanges

def calculate_exchange_probability_per_endstate(expanded_repdat: ExpandedRepdat):
    """
    This function calculates the average exchange probability between each pair of s-values for each end-state.
    
    Parameters
    ----------
    expanded_repdat: ExpandedRepdat
        ExpandedRepdat object which contains all the exchange information of a 
        RE-EDS simulation plus the potential energies of the end-states
    Returns
    ----------
    exchanges: Dict[List[float]]
       dictionary containing the exchange frequencies for each state
    """
    states = expanded_repdat.system.state_eir.keys()
    s_values = sorted(expanded_repdat.DATA["ID"].unique())
    state_exchanges = {}
    
    for state in states:
        # Get all exchanges involving the state and skip cases where no exchange is attempted
        state_repdat = expanded_repdat.DATA.query(f"Vmin == 'Vr{state}' & Epoti != 0")
        exchange_probabilities = []
        for s_val in s_values[:-1]:
            # Get exchanges from s to s+1 and from s+1 to s
            s_val_exchanges = state_repdat.query(f"(ID == {s_val} & partner == {s_val+1}) \
            | (ID == {s_val+1} & partner == {s_val})")
            exchange_probabilities.append(np.mean(s_val_exchanges["p"]))
        state_exchanges[state] = exchange_probabilities
        
    return state_exchanges