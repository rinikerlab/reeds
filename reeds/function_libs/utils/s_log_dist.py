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
