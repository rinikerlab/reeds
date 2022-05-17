import argparse
import os
import sys
import traceback

import reeds 
from pygromos.gromos import gromosXX as mdGromosXX
from pygromos.utils import bash

spacer = "================================================================================="


def work(out_dir: str, in_coord: str, in_imd_path: str, in_topo_path: str, in_perttopo_path: str,
         in_disres_path: str=None, in_posres_path: str=None, in_refpos_path: str=None,
         nmpi: int = 1,
         gromosXX_bin_dir: str = None, work_dir: str = None, write_free_energy_traj: bool = False):
    """
        This worker script is scheduled by the RE_EDS_simulation scheduler to a job-queue.
        It conductes a gromos md-simulation for RE-EDS.

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
        input pertubation topology
    in_disres_path : str, optional
        input distance restraints (default: None)
    in_posres_path : str, optional
        input path to position restraints file (.por; default: None)
    in_refpos_path : str, optional
            input path to reference position file (.rpf; default: None)
    nmpi : int, optional
        number of mpi cores (default: 1)
    gromosXX_bin_dir : str, optional
        path to gromosXX binary folder (default: None)
    work_dir : str, optional
        work directory (default: None)
         if None, the storage on the local node is used.
    write_free_energy_traj : bool, optional
        shall trgs be written? (default: False)

    Returns
    -------
    int
        return 0 if successfully the code was walked through 1 if error was found (@warning does not necessarily find all errors!)
    """

    try:
        # WORKDIR SetUP
        if (work_dir is None or work_dir == "None") and "TMPDIR" in os.environ:
            work_dir = os.environ["TMPDIR"]
            print("using TmpDir")
        elif (work_dir is None and work_dir == "None"):
            print("Could not find TMPDIR!\n Switched to outdir for work")
            work_dir = out_dir
        
        # Check if the calculation is running on multiple nodes:
        hosts = os.environ['LSB_HOSTS'].split()
        multi_node = True if len(hosts) > 1 else False        
        
        # run a euler script to create tmpdir on all nodes
        if multi_node:
            os.system('remote_tmpdir create')
        elif not os.path.isdir(work_dir): # when we specify a local directory, ensure it exists
            bash.make_folder(work_dir)

        os.chdir(work_dir)

        # MD RUN
        md = mdGromosXX.GromosXX(bin=gromosXX_bin_dir)

        print(spacer + "\n start MD " + str(os.path.basename(imd_path)) + "\n")
        # get file outprefix
        out_prefix = os.path.splitext(os.path.basename(imd_path))[0]

        #TODO: This is a stupid workaround as Euler tends to place nans in the euler angles, that should not be there!
        from pygromos.files.coord import cnf
        import math
        import glob
        for in_cnf_tmp in glob.glob(os.path.dirname(in_coord)+"/*.cnf"):
            cnf_file = cnf.Cnf(in_cnf_tmp)
            if(hasattr(cnf_file, "GENBOX") and any([math.isnan(x) for x in cnf_file.GENBOX._euler])):
                cnf_file.GENBOX._euler = [0.0, 0.0, 0.0]
                cnf_file.write(in_cnf_tmp)

        try:
            key_args = {"in_topo_path": in_topo_path,
                        "in_coord_path" : in_coord,
                        "in_imd_path" : in_imd_path,
                        "nmpi" : nmpi,
                        "in_pert_topo_path" : in_perttopo_path,
                        "out_prefix" : out_prefix,
                        "out_trg" : write_free_energy_traj}
            
            restrained = False
            if in_disres_path is not None and in_disres_path != "None":
                key_args.update({"in_disres_path": in_disres_path})
                restrained = True

            if(not (in_posres_path is None or in_refpos_path is None)):
                key_args.update({"in_posresspec_path": in_posres_path,
                                 "in_refpos_path": in_refpos_path})
                restrained = True

            if not restrained:
                print ("Are you really sure you want to run without restraints?\n")

            md_run_log_path = md.repex_mpi_run(**key_args)
            
        except Exception as err:
            print("\n#####################################################################################")
            print("\t\tERROR in Reeds_simulationWorker - Gromos Execution")
            print("Copying files back to a local directory and exiting.")
            print("#####################################################################################\n")
        
        # This part of the code (which copies all files back)
        # must be reached after succesful and unsuccesful runs.
        if (out_dir != work_dir):
            if not multi_node: 
                os.system("mv " + work_dir + "/*  " + out_dir)
            else: 
                # when copying the data back from multiple nodes, data has to be copied back manually from all nodes.
                for host in hosts:
                    command = 'ssh ' + host + '  \"cp ${TMPDIR}/* ' + out_dir + '\"'
                    os.system(command)
            os.system('remote_tmpdir delete')
    
    except Exception as err:
        print("#####################################################################################")
        print("\t\tERROR in Reeds_simulationWorker")
        print("#####################################################################################")
        traceback.print_exception(*sys.exc_info(), file=sys.stderr)
        return 1

    return 0


if __name__ == "__main__":

    """
    parser = argparse.ArgumentParser(description=work.__doc__)
    """

    # INPUT JUGGELING
    parser = argparse.ArgumentParser(description="RE_EDS_worker Script\n\t for job submission to queueing system.")
    parser.add_argument('-in_imd', type=str, required=True, help="give .imd file which should be used.")
    parser.add_argument('-in_top', type=str, required=True, help="give input .top - file.")
    parser.add_argument('-in_coord', type=str, required=True, help="give input coordinates .cnf file.")
    parser.add_argument('-in_perttop', type=str, required=True, help="give input pertubation topology.")
    parser.add_argument('-in_disres', type=str, required=False, default=None, help="give input distance restraints.")
    parser.add_argument('-in_posres', type=str, required=False, default=None, help="give input for position restraints.")
    parser.add_argument('-in_refpos', type=str, required=False, default=None, help="give input for reference positoins.")

    parser.add_argument('-out_dir', required=True, default=None, help="where to final store the Files?.")
    parser.add_argument('-work_dir', required=False, default=None, help="where to work? Default on node.")

    parser.add_argument('-nmpi', type=int, required=False, default=1, help="number of MPI threads for the sopt_job.")
    parser.add_argument('-gromosXX_bin_dir', required=False, default=None,
                        help="where are your gromosXXbinaries, default(None - uses shell env)")

    # user defined
    args, unkown_args = parser.parse_known_args()

    # svalues = args.svalues # ive s vals!
    nmpi = args.nmpi
    gromosXX_bin_dir = args.gromosXX_bin_dir

    # inputFiles
    in_topo_path = args.in_top
    in_coord = args.in_coord
    in_disres_path = args.in_disres
    in_posres_path = args.in_posres
    in_refpos_path = args.in_refpos
    in_perttopo_path = args.in_perttop
    imd_path = args.in_imd

    work_dir = args.work_dir
    out_dir = args.out_dir

    # LSF interatctions
    if "CORENUM" in os.environ:
        cores = os.environ['CORENUM']
        if (nmpi > cores):
            raise IOError("Please define the nmpi and nomp, so that nmpi + nomp = cores.")

    # WORK Command
    work(out_dir=out_dir, work_dir=work_dir, in_topo_path=in_topo_path, in_coord=in_coord, in_imd_path=imd_path,
         gromosXX_bin_dir=gromosXX_bin_dir, in_perttopo_path=in_perttopo_path, in_disres_path=in_disres_path, in_posres_path=in_posres_path,
         in_refpos_path=in_refpos_path, nmpi=nmpi)
