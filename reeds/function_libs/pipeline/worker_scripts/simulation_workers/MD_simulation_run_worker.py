import argparse
import os
import sys

from pygromos.gromos import gromosXX as mdGromosXX
from pygromos.utils import bash as bash

spacer = "================================================================================="


def work(out_dir: str, in_coord: str, in_imd_path: str, in_topo_path: str, in_perttopo_path: str, in_disres_path: str,
         nmpi: int = 1, nomp: int = 1, out_trg: bool = False,
         gromos_bin: str = None, work_dir: str = None):
    """
            Executed by repex_EDS_long_production_run as worker_scripts
            #TODO: This was used to do TIs, it will go in future to the pygromos package

    Parameters
    ----------
    out_dir : str
        final output dir
    in_coord : str
        input coordinates
    in_imd_path : str
         input imd-parameter file
    in_topo_path : str
        input topology
    in_perttopo_path : str
        input pertubation
    in_disres_path : str
        input disres
    nmpi : int, optional
        number of mpi cores (default: 1)
    nomp : int, optional
        number of omp cores (default: 1)
    out_trg  : str, optional

    gromos_bin : str, optional
        path to gromos binary (default: None)
    work_dir : str, optional
        work directory (default: None)

    Returns
    -------
    int
        0 if code was passed through.
    """


    # WORKDIR SetUP
    if ((isinstance(work_dir, type(None)) or work_dir == "None") and "TMPDIR" in os.environ):
        work_dir = os.environ["TMPDIR"]
        print("using TmpDir")
    elif (isinstance(work_dir, type(None)) and work_dir == "None"):
        print("Could not find TMPDIR!\n Switched to outdir for work")
        work_dir = out_dir

    print("workDIR: " + work_dir)
    if (not os.path.isdir(work_dir)):
        bash.make_folder(work_dir)

    os.chdir(work_dir)
    print("workDIR: " + work_dir)

    md = mdGromosXX.GromosXX(bin=gromos_bin)
    # RUN
    try:
        print(spacer + "\n start MD " + str(os.path.basename(imd_path)) + "\n")
        out_prefix = os.path.splitext(os.path.basename(imd_path))[0]
        md_failed = False

        try:
            md_run = md.md_mpi_run(in_topo_path=in_topo_path, in_coord_path=in_coord, in_imd_path=in_imd_path,
                                   nmpi=nmpi,
                                   in_pert_topo_path=in_perttopo_path, out_trg=out_trg,
                                   in_disres_path=in_disres_path, out_prefix=out_prefix, nomp=nomp, verbose=True)

        except Exception as err:
            print("Failed! process returned: \n Err: \n" + "\n".join(err.args))
            md_failed = True

        if (out_dir != work_dir):
            os.system("mv " + work_dir + "/*  " + out_dir)
        # post simulation cleanup
        if not (isinstance(work_dir, type(None)) and work_dir == "None" and "TMPDIR" in os.environ):
            bash.remove_folder(work_dir, verbose=True)
        # bash.move_file(work_dir + "/*", out_dir)
        # bash.remove_file(out_dir + "/slave*.out")
        # os.system("rmdir "+work_dir)

    except Exception as err:
        print("\nFailed during simulations: ", file=sys.stderr)
        print(type(err), file=sys.stderr)
        print(err.args, file=sys.stderr)
        exit(1)

    return 0


if __name__ == "__main__":
    # INPUT JUGGELING
    parser = argparse.ArgumentParser(description="Run EDS-parameter Exploration")
    parser.add_argument('-imd', type=str, required=True, help="give .imd file which should be used.")
    parser.add_argument('-top', type=str, required=True, help="give input .top - file.")
    parser.add_argument('-coord', type=str, required=True, help="give input coordinates .cnf file.")
    parser.add_argument('-perttop', type=str, required=True, help="give input pertubation topology.")
    parser.add_argument('-disres', type=str, required=False, default=False, help="give input distance restraints.")

    parser.add_argument('-nmpi', type=int, required=False, default=1, help="number of MPI threads for the sopt_job.")
    parser.add_argument('-nomp', type=int, required=False, default=1, help="number of OMP threads for the sopt_job.")
    parser.add_argument('-outdir', required=True, default=None,
                        help="where to final store the Files?.")
    parser.add_argument('-workdir', required=False, default=None,
                        help="where to work? Default on node.")
    parser.add_argument('-bin', required=False, default="md_mpi", help="where to work?.")
    parser.add_argument("-out_trg", action='store_true', default=False, required=False,
                        help="shall gromos write out free energy trajs?")

    # user defined
    args, unkown_args = parser.parse_known_args()

    # svalues = args.svalues # ive s vals!
    nmpi = args.nmpi
    nomp = args.nomp

    # LSF interatctions
    if "CORENUM" in os.environ:
        cores = os.environ['CORENUM']
        if (nmpi * nomp > cores):
            raise IOError("Please define the nmpi and nomp, so that nmpi + nomp = cores.")

    # inputFiles
    in_topo_path = args.top
    in_coord = args.coord
    in_disres_path = args.disres
    in_perttopo_path = args.perttop
    imd_path = args.imd
    out_trg = args.out_trg

    work_dir = args.workdir
    out_dir = args.outdir
    programm_path = args.bin

    # WORK Command
    work(out_dir=out_dir, work_dir=work_dir, in_topo_path=in_topo_path, in_coord=in_coord, in_imd_path=imd_path,
         gromos_bin=programm_path, in_perttopo_path=in_perttopo_path, in_disres_path=in_disres_path, nmpi=nmpi,
         nomp=nomp, out_trg=out_trg)
