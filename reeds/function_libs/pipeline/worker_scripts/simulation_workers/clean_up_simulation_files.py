#!/usr/bin/env python3

""" After simulation cleanup
This module is thought to be used as a clean up script after each run, reducing the need of storage for RE-EDS simulations on Euler or any other cluster :)
it mainly removes not needed temporary files and compresses long trajectory files.
It should be hanged in after each simulation step.
"""

import argparse
import glob

from reeds.function_libs.file_management import file_management as fM


def do(in_simulation_dir: str, n_processes: int = 1, verbose: bool = True) -> None:
    """do clean_up_simulation_files
        this function is compressing resulting files form an simulation approach.
    Parameters
    ----------
    in_simulation_dir : str
        path to folder, which contains the raw gromos output
    n_processes :   int, optional
        how many processes should be used?
    verbose : bool, optional
        loud and noisy?
    Returns
    -------
    None
    """
    import time
    time.sleep(3)
    # compress files:
    if (verbose): print("Search Files START", "\n")
    tre_files = glob.glob(in_simulation_dir + "/*.tre")
    trc_files = glob.glob(in_simulation_dir + "/*.trc")

    if (len(tre_files + trc_files) == 0):
        raise IOError("Could not find any file in : " + in_simulation_dir)

    if (verbose): print("Found trcs: ", trc_files, "\n")
    if (verbose): print("Found tres: ", tre_files, "\n")

    if (verbose): print("COMPRESS START\n")
    fM.compress_files(tre_files + trc_files, n_processes=n_processes)

    if (verbose): print("COMPRESS DONE\n")

    # do more?


if __name__ == "__main__":
    # INPUT JUGGELING
    parser = argparse.ArgumentParser(description="Run Gromos after simulation cleanup\n\n" + str(__doc__))
    parser.add_argument('-in_simulation_dir', type=str, required=True,
                        help="give the simulation directory path of a gromos simulation.")
    parser.add_argument('-n_processes', type=int, required=False, default=1,
                        help="give the number of process for parallelisation.")

    args, unkown_args = parser.parse_known_args()

    if (len(unkown_args) > 0):
        print("ERROR FOUND UNKNOWN ARG")
        parser.print_help()
        raise IOError("Parser does not know following arguments:\n\t" + "\n\t".join(unkown_args))
    else:
        print("got: \n\t", args.in_simulation_dir, "\n\t", args.n_processes)
        do(in_simulation_dir=args.in_simulation_dir, n_processes=args.n_processes)
