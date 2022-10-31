"""
    s_log_dist
       This module is  a small wrapper for generating s_log_distributed vectors.
"""
from numbers import Number
import numpy as np

np.set_printoptions(suppress=True)


def get_log_s_distribution_between(start: Number, end: Number, num: Number = 21, verbose: bool = False) -> np.array:
    """
        This function returns a log distributed vector between two extreme values

    Parameters
    ----------
    start: Number
        start number
    end: Number
        end number
    num: Number, optional
        how many numbers between start and end? (default: 21)
    verbose:bool, optional
        Brrrrr (default: false)

    Returns
    -------
    np.array
        logdistributed vector
    """

    if (verbose):
        print("start:\t" + str(start) + "\tend:\t" + str(end))
        print("EXPS: start:\t" + str(np.log10(start)) + "\tend:\t" + str(np.log10(end)))
        print("dist: " + str(np.logspace(start=start, stop=end, num=num)))
    return np.array([np.round(x, decimals=int(round(abs(np.log10(min(start, end))) + 2))) for x in
                     np.logspace(start=np.log10(start), stop=np.log10(end), num=num)])


def get_log_s_distribution_between_exp(start: Number, end: Number, num: Number = 21) -> np.array:
    """
        This function returns a log distributed vector between two extreme exponents

    Parameters
    ----------
    start: Number
        start exponent number
    end: Number
        end exponent number
    num: Number, optional
        how many numbers between start and end? (default: 21)

    Returns
    -------
    np.array
        logdistributed vector
    """
    dist = np.logspace(start=start, stop=end, num=num)
    return dist



# Candide: I also added here functions which generate the input for the s-optimization, such that just a function is called

def default_eoff_to_sopt(eoff_s_values, num_states):
    """
    This function makes a new s-distribution for the s-opt iteration #1
    It is the default behavior we previously had in the pipeline    
    
    Parameters
    ----------
    eoff_s_values: List [float]
        s-value distribution used in the energy offset estimation
    num_states: int
        number of end states in the RE-EDS simulation
    Returns
    ----------
    new_sval: List[List[float]] 
        A list containing the list of the previous s_values (from the energy offset run)
        and the newly distributed s-values for the 1st iteration of s-optmization.   
    """    
    new_sval = [eoff_s_values, []]
    new_sval[1] = [1.0 for x in range(num_states-1)] + list(
    get_log_s_distribution_between(start=1.0, end=min(eoff_s_values), num=len(eoff_s_values) - (num_states-1)))

    return new_sval

def generate_preoptimized_sdist(eoff_s_values, num_states, exchange_freq, undersampling_s, num_svals:int  = 32):
    """
    This function makes a new s-distribution for the s-opt iteration.
    This distribution will keep exactly the same values in the upper s-range
    It will place many in the intermediate range, and a few in the undersampling 
    to ensure some exchanges in both upper parts of the distribution.
    
    Parameters
    ----------
    eoff_s_values: List [float]
        s-value distribution used in the energy offset estimation
    num_states: int
        number of end states in the RE-EDS simulation
    exchange_freq: List [float]
        list of exchange frequencies in the energy offset run
    undersampling_s: float
        s-value for the replica that is 3 replicas below undersampling
    num_svals:int
        total number of s-values which make up the prooptimized distribution
    Returns
    ----------
    new_sval: List[List[float]] 
        A list containing the list of the previous s_values (from the energy offset run)
        and the newly distributed s-values for the 1st iteration of s-optmization.   
    """    
    new_sval = [eoff_s_values, []]
    
    # 1: Find the upper and lower s-values of the gap region from the exchange frequencies

    upper = []
    lower = []
    
    upper_gap_s = 0
    lower_gap_s = 0

    for i, f in enumerate(exchange_freq):
        if f < 0.8:
            upper_gap_s = eoff_s_values[i]
            break
        upper.append(eoff_s_values[i])
    
    for i, f in reversed(list(enumerate(exchange_freq))):
        # do not add s-value that is lower than the limit found to be appropriate
        if eoff_s_values[i+1] < undersampling_s: continue
        
        if f < 0.8: 
            lower_gap_s = eoff_s_values[i+1]
            break
        lower.insert(0, eoff_s_values[i+1]) 

    # Check that at least some values were placed in the lower region.
    if (len(lower) == 0):
        print ('\n\nWarning: There are no s-values in the lower part of the distribution')
        print ('    this may be because the undersamping detection failed, or that somehow')
        print ('    the particular system was very low exchanges in the undersampling region.')

    # 2: Now that the extrema have been defined, we can build our new distribution.
    # This distribution will keep exactly the same values in the upper and lower s-ranges
    # and place many in the gap region

    # Make it automatic so we always have 32 s-values
    num_svals_gap = num_svals - len(upper) - len(lower) 

    new_s_distrib = np.zeros(0)
    gap = get_log_s_distribution_between(start = upper_gap_s, end = lower_gap_s, num= num_svals_gap)

    new_s_distrib = np.append(new_s_distrib, upper)
    new_s_distrib = np.append(new_s_distrib, gap)
    new_s_distrib = np.append(new_s_distrib, lower)

    new_sval[1] = new_s_distrib.tolist()
    return new_sval
