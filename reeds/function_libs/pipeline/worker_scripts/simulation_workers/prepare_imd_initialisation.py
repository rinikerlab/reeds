import argparse
import time

import reeds
from pygromos.files import imd
from pygromos.utils import bash


def prepare_imd_initialisation(run: int, in_imd_path: str, tmp_in_imd: str, initialize_first_run: bool = True, reinitialize: bool = False):
    """
        this function is taking care of adapting the imd file in an consecutive md-run in multiple steps.

    Parameters
    ----------
    run : int
        ID number of the run
    in_imd_path : str
        template imd file, that is used for the simulation run
    tmp_in_imd : str
        tmp imd file, used for the final md-run step
    initialize_first_run : bool, optional
        use the run initialization options for the first run.  (default: True)
    reinitialize : bool, optional
        use the run initialization options of gromos in each step (not recommended!) (default: False)

    Returns
    -------
    none

    Raises
    ------
    Exception
        something went definetly wrong.

    """
    try:

        print("PREPARE IMD INITIALISATION")
        print(in_imd_path + "\n" + tmp_in_imd)

        time.sleep(5)

        # os.system("cp " + in_imd_path + " " + tmp_in_imd)

        bash.copy_file(in_file_path=in_imd_path, out_file_path=tmp_in_imd)

        time.sleep(5)

        imd_file = imd.Imd(tmp_in_imd)
        is_stochastic_dynamics_sim = False

        if (hasattr(imd_file, 'STOCHDYN')):
            if (imd_file.STOCHDYN.NTSD):
                is_stochastic_dynamics_sim = True

        if (reinitialize or (initialize_first_run and run == 1)):
            imd_file.INITIALISE.NTIVEL = 0
            imd_file.INITIALISE.NTISHK = 0
            imd_file.INITIALISE.NTINHT = 0
            imd_file.INITIALISE.NTINHB = 0
            imd_file.INITIALISE.NTISHI = 0
            imd_file.INITIALISE.NTIRTC = 0
            imd_file.INITIALISE.NTICOM = 0
            imd_file.INITIALISE.NTISTI = 0
            if (is_stochastic_dynamics_sim):
                imd_file.INITIALISE.NTISHI = 1
                imd_file.INITIALISE.NTISTI = 1
            else:
                imd_file.INITIALISE.NTIVEL = 1
                imd_file.INITIALISE.NTISHK = 3
                imd_file.INITIALISE.NTISHI = 1

        else:
            imd_file.INITIALISE.NTIVEL = 0
            imd_file.INITIALISE.NTISHK = 0
            imd_file.INITIALISE.NTINHT = 0
            imd_file.INITIALISE.NTINHB = 0
            imd_file.INITIALISE.NTISHI = 0
            imd_file.INITIALISE.NTIRTC = 0
            imd_file.INITIALISE.NTICOM = 0
            imd_file.INITIALISE.NTISTI = 0
            if (is_stochastic_dynamics_sim):
                imd_file.INITIALISE.NTISHI = 1

        imd_file.write(tmp_in_imd)

        time.sleep(5)

    except Exception as err:
        print("ERROR during imd preparation:\n")
        print("\n", "\n\t".join(err.args))


if __name__ == "__main__":
    # INPUT JUGGELING
    parser = argparse.ArgumentParser(
        description="RE_EDS_worker Script\n\t for setting initialisation block in imd for continues runs.")
    parser.add_argument('-run', type=int, required=True, help="interation in chain submission")
    parser.add_argument('-in_imd_path', type=str, required=True, help="path of input imd file")
    parser.add_argument('-tmp_in_imd', type=str, required=True, help="path of output imd file")
    parser.add_argument('-initialize_first_run', type=str, required=True, help="should the velocities of the first run be reinitialized?")
    parser.add_argument('-reinitialize', type=str, required=True, help="should the velocities be reinitialized in all runs?")

    # user defined
    args, unkown_args = parser.parse_known_args()

    run = int(args.run)
    in_imd_path = args.in_imd_path
    tmp_in_imd = args.tmp_in_imd
    initialize_first_run = True
    reinitialize = False
    
    # note from salomé: bools are not parsed as expected, this is an ugly fix for now
    if(args.initialize_first_run == "True"):
      initialize_first_run = True
    else:
      initialize_first_run = False

    if(args.reinitialize == "True"):
      reinitialize = True
    else:
      reinitialize = False

    prepare_imd_initialisation(run = run, in_imd_path = in_imd_path, tmp_in_imd = tmp_in_imd, initialize_first_run = initialize_first_run, reinitialize = reinitialize)
