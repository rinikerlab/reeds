import warnings
from typing import List, Dict

import numpy as np

import reeds.function_libs.visualization.re_plots

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
    exchange_trials = int(len(exchange_data.DATA.run)/len(exchange_data.system.s))

    exchanges = np.zeros(n_replicas-1)
    for i, row in exchange_data.DATA.iterrows():
        if row.s == 1 and row.ID <= row.partner:
            exchanges[row.ID-1] += 1

    # Normalize by the number of exchange trials to get probability
    exchanges /=  exchange_trials
    return exchanges

