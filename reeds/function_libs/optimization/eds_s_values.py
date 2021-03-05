"""
    eds_s_optimization wrappers

"""

import argparse
import sys
from typing import List
from reeds.function_libs.optimization.src import s_optimizer as optimizer
from reeds.function_libs.optimization.src.sopt_Pathstatistic import generate_PathStatistic_from_file

def calc_NLRTO(stat, add_n_s: int, state_weights:List[float]=None, verbose:bool=False) -> (list, optimizer.N_LRTO):
    """ calculate N-Local Round Trip Optimizer
        This function is optimizing the replica / s-distribution according to the provided exchange-statistcs.
        The used algorithm is the N-Local Roundtrip Optimizer by Sidler et al. 2017.
        A feature of this algorithm is, that it consideres multiple states in a system and optimizes the exchange rate for all of them.
        The algorithm will not change the already present replica positions, but only adds new replicas into the bottle-necks

    Parameters
    ----------
    stat :
        exchange statistics
    add_n_s : int
        add n-replica to the s-distribution.
    state_weights : List[float]
        weight the individual states of  the system accordingly
    verbose : bool
        Shalalalala

    Returns
    -------
    (list, optimizer.N_LRTO)
        returns a list of new optimized s and the Optimizer obj.
    """
    if (verbose):
        print('\n===========================================\n')
        print('Starting N-LRTO algorithm.\n')

    # build LRTO optimizer
    NLRTO_optimizer = optimizer.N_LRTO(replica_exchange_statistics=stat, state_weights=state_weights)
    # optimize
    NLRTO_optimizer.optimize(add_n_s, verbose=verbose)
    if (verbose):
        print(NLRTO_optimizer)
    # get new s_dist
    s_new = NLRTO_optimizer.get_new_replica_dist()
    return s_new, NLRTO_optimizer


def calc_oneLRTO(stat, add_n_s, verbose=False) -> (list, optimizer.One_LRTO):
    """
        This function is optimizing the replica / s-distribution according to the provided exchange-statistcs.
        The used algorithm is the Local Roundtrip Optimizer described in Sidler et al. 2017 or Katzgraber .
        The algorithm will not change the already present replica positions, but only adds new replicas into the bottle-necks

    Parameters
    ----------
    stat :
        exchange statistics
    add_n_s : int
        add n-replica to the s-distribution.
    verbose : bool
        Shalalalala

    Returns
    -------
    (list, optimizer.One_LRTO)
        returns a list of new optimized s and the Optimizer obj.
    """
    if (verbose):
        print('\n===========================================\n')
        print('Starting 1-LRTO algorithm.\n')

    # build LRTO optimizer
    oneLRTO_optimizer = optimizer.One_LRTO(replica_exchange_statistics=stat)
    # optimize
    oneLRTO_optimizer.optimize(add_n_s, verbose=verbose)
    if (verbose):
        print(oneLRTO_optimizer)
    # get new s_dist
    s_new = oneLRTO_optimizer.get_new_replica_dist()
    return s_new, oneLRTO_optimizer


def calc_oneGRTO(stat, add_n_s, ds=0.0001, verbose=False, detail_verbose=0) -> (list, optimizer.One_GRTO):
    """
        This function is optimizing the replica / s-distribution according to the provided exchange-statistcs.
        The used algorithm is the Global Roundtrip Optimizer described in Katzgraber et al. .
        The algorithm will change the replica positioning by distributing the replicas equally over the area of the calculated flow curve for all replicas.


    Parameters
    ----------
    stat :
        exchange statistics
    add_n_s : int
        add n-replica to the s-distribution.
    ds : float
        integration step
    verbose : bool
        Shalalalala
    detail_verbose : int
        there are shades of verbosity (default: 0)

    Returns
    -------
    (list, optimizer.One_GRTO)
        returns a list of new optimized s and the Optimizer obj.

    """
    if (verbose):
        print('\n===========================================\n')
        print('Starting 1-GRTO algorithm.\n')

        # build LRTO optimizer
    one_GRTO_optimizer = optimizer.One_GRTO(replica_exchange_statistics=stat)
    # optimize
    one_GRTO_optimizer.optimize(add_n_s, verbose=verbose, detail_verbose=detail_verbose, ds=ds)
    # get new s_dist
    if (verbose):
        print(one_GRTO_optimizer)
    s_new = one_GRTO_optimizer.get_new_replica_dist()
    return s_new, one_GRTO_optimizer


def calc_NGRTO(stat, add_n_s, state_weights=None, ds=0.0001, verbose=False, detail_verbose=0) -> (
list, optimizer.N_GRTO):
    """
        This function is optimizing the replica / s-distribution according to the provided exchange-statistcs.
        The used algorithm is the Global Roundtrip Optimizer described in Sidler et al. 2017.
        A feature of this algorithm is, that it consideres multiple states in a system and optimizes the exchange rate for all of them.
        The algorithm will change the replica positioning by distributing the replicas equally over the area of the calculated flow curve for all replicas.


    Parameters
    ----------
    stat :
        exchange statistics
    add_n_s : int
        add n-replica to the s-distribution.
    ds : float
        integration step
    verbose : bool
        Shalalalala
    detail_verbose : int
        there are shades of verbosity (default: 0)

    Returns
    -------
    (list, optimizer.One_GRTO)
        returns a list of new optimized s and the Optimizer obj.
    """
    if (verbose):
        print('\n===========================================\n')
        print('Starting N-GRTO algorithm. \n')
    # parse
    N_GRTO_optimizer = optimizer.N_GRTO(replica_exchange_statistics=stat, state_weights=state_weights)
    # optimize
    N_GRTO_optimizer.optimize(add_replicas=add_n_s, ds=ds, verbose=verbose, detail_verbose=detail_verbose)
    # get new s_dist
    if (verbose):
        print(N_GRTO_optimizer)
    s_new = N_GRTO_optimizer.get_new_replica_dist()
    return s_new, N_GRTO_optimizer


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="S-Optimization for RE-EDS")
    parser.add_argument('-i', type=str, nargs='+', required=True, help='gromos repdat Files', dest='infile')
    parser.add_argument('-n', type=int, required=True, help='Number of s-values to be added', dest='n')
    parser.add_argument('-o', type=str, required=True, help='define Outputfile', dest='out_path')
    args = parser.parse_args()

    # if(os.path.exists(os.path.dirname(args.out_path))):
    #    raise IOError("Cannot find path to "+os.path.dirname(args.out_path))

    try:
        stat = generate_PathStatistic_from_file(args.infile)
    except IOError as e:
        print('ERROR while reading file:')
        print('    ' + str(e))
        sys.exit(1)

    out_file = open(args.out_path, "w")
    out_file.write('Finished reading file.')

    # add Svalues
    add_s = args.n
    s_new, NLRTO = calc_NLRTO(stat, add_s, True)

    out_file.write(NLRTO)
    out_file.close()
    print(NLRTO)
