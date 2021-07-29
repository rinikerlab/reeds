"""
    This module is doing all the post simulation file juggling needed for gromos.
"""
import glob, os, tempfile, warnings
import multiprocessing as mult

from typing import List, Dict, Union, Tuple, Iterable
from numbers import Number

import numpy as np
import pandas as pd

import reeds
from pygromos.files import repdat, imd
from pygromos.files.energy import tre
from pygromos.gromos import gromosPP
from pygromos.utils import bash

from reeds.function_libs.utils.structures import adding_Scheme_new_Replicas as add_scheme

"""
    PARALLEL WORKER - These functions are required for parallelized code Execution
"""


def _thread_worker_cat_trc(job: int,
                           replicaID_range: (Iterable, List[int]),
                           trc_files: Dict[int, List[str]],
                           out_prefix: str,
                           topology_path: str,
                           out_trcs: dict,
                           dt: float,
                           time: float = 0,
                           verbose: bool = False,
                           boundary_conditions: str = "r cog",
                           include_all: bool = False):
    """_thread_worker_cat_trc
        This thread worker_scripts concatenates all .trc files of one replica into one file.

    Parameters
    ----------
    job : int
        rank of this thread
    replicaID_range : (Iterable, List[int])
        x_range - list of all
    trc_files : Dict[int, List[str]]
        Dictionary containing all replicas, with list of all trc files concerning one trc.
    out_prefix : str
        output prefix
    topology_path : str
        path to the topology file
    out_trcs : dict
        output trajectories
    dt : float
        timestep
    time : float, optional
        start time (default 0)
    boundary_conditions : str, optional
        boundary conditions (default "r cog")
    include_all : bool, optional
        include SOLVENT? (default: False)
    verbose : bool
        verbosity?

    Returns
    -------
    None
    """

    gromPP = gromosPP.GromosPP()
    start_dir = os.getcwd()
    if (verbose): print("JOB " + str(job) + ": range " + str(list(replicaID_range)))
    for replicaID in replicaID_range:
        out_path = out_prefix + str(replicaID) + ".trc"
        compress_out_path = out_path + ".gz"

        out_trcs.update({replicaID: compress_out_path})

        if (os.path.exists(compress_out_path)):  # found perfect compressed trc file:)
            warnings.warn("Skipped generating file as I found: " + compress_out_path)
            if (os.path.exists(out_path)):
                bash.remove_file(out_path)
            continue
        elif (os.path.exists(out_path)):  # did not find compressed file. will compress
            warnings.warn("Skipped generating file as I found: " + out_path)
            continue
        else:  # concat files
            if (verbose): print("JOB " + str(job) + ": " + "write out " + out_path + "\n")
            out_dir = os.path.dirname(out_path)
            tmp_dir = bash.make_folder(out_dir + "/TMP_replica_" + str(replicaID), additional_option="-p")
            os.chdir(tmp_dir)
            if (include_all):
                out_path = gromPP.frameout(in_top_path=topology_path, in_coord_path=" ".join(trc_files[replicaID]),
                                           periodic_boundary_condition=boundary_conditions, single_file=True,
                                           out_file_format="trc",
                                           out_file_path=out_path, time=time, dt=dt, include="ALL")
            else:
                out_path = gromPP.frameout(in_top_path=topology_path, in_coord_path=" ".join(trc_files[replicaID]),
                                           periodic_boundary_condition=boundary_conditions, single_file=True,
                                           out_file_format="trc",
                                           out_file_path=out_path, time=time, dt=dt)
            os.chdir(start_dir)
            bash.wait_for_fileSystem(out_path)
            bash.remove_folder(tmp_dir)
            if (verbose): print("JOB " + str(job) + ": " + "write out " + out_path + "\t DONE\n")

        if (verbose): print("JOB " + str(job) + ": " + "compress " + compress_out_path + "\n")
        compressed_trc = bash.compress_gzip(out_path, out_path=compress_out_path)

        if (verbose): print("JOB " + str(job) + ": " + "compress " + compressed_trc + "\t DONE\n")


def _thread_worker_cat_tre(job: int,
                           replicaID_range: (Iterable, List[int]),
                           tre_files: Dict[int, List[str]],
                           out_prefix: str,
                           out_tres: dict,
                           verbose: bool = False):
    """_thread_worker_cat_tre
    This functions concatenates energy trajectories

    Parameters
    ----------
    job : int
        rank of this thread
    replicaID_range : (Iterable, List[int])
        x_range - list of all
    tre_files : Dict[int, List[str]]
        energy trajectory files
    out_prefix : str
        prefix for output files
    out_tres : dict
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    None

    """
    if (verbose): print("JOB " + str(job) + ": range " + str(list(replicaID_range)))

    for replicaID in replicaID_range:
        use_tre_file_paths, unarchived_files = find_and_unarchive_tar_files(tre_files[replicaID], verbose=verbose)
        if (verbose): print("FILES: ", use_tre_file_paths)
        if (verbose): print("Archs:", unarchived_files)

        out_path = out_prefix + str(replicaID) + ".tre"
        compressed_tre = out_path + ".gz"
        if (os.path.exists(compressed_tre)):
            warnings.warn("Skipped generating .tre.gz file as I found: " + out_path)
        else:
            if (os.path.exists(out_path)):
                warnings.warn("Skipped generating .tre file as I found: " + out_path + "\n\t Continue Compressing.")
            else:
                tre_file = tre.Tre(use_tre_file_paths[0])
                if verbose: print("JOB " + str(job) + ": parsing " + os.path.basename(use_tre_file_paths[0]))
                if (len(use_tre_file_paths) > 1):
                    for tre_file_path in use_tre_file_paths[1:]:
                        if verbose: print("JOB " + str(job) + ": append " + os.path.basename(tre_file_path))
                        tre_file.append_tre(tre.Tre(tre_file_path))
                if verbose: print("JOB " + str(job) + ": write out " + os.path.basename(out_path))
                tre_file.write(out_path)
                bash.wait_for_fileSystem(out_path)
                if verbose: print("JOB " + str(job) + ": done " + os.path.basename(out_path))
                del tre_file

                if verbose: print("JOB " + str(job) + ":  compress " + os.path.basename(out_path))
                compressed_tre = bash.compress_gzip(out_path, out_path=compressed_tre)
                if (verbose): print("JOB " + str(job) + ": " + "compress " + compressed_tre + "\t DONE\n")

        # file cleaning:
        for file_path in use_tre_file_paths:
            bash.compress_gzip(file_path)
        out_tres.update({replicaID: compressed_tre})


def thread_worker_concat_repdat(job: int,
                                repdat_file_out_path: str,
                                repdat_file_paths: (str, List[str]),
                                verbose: bool = False):
    """thread_worker_concat_repdat
    This function concatenates repdat files

    Parameters
    ----------
    job : int
        rank of this thread
    repdat_file_out_path : str
        output path for concatenated repdat files
    repdat_file_paths : (str, List[str])
        path to repdat files
    verbose : bool, optional  
        verbose output (default False)

    Returns
    -------
    None
    """
    if (os.path.exists(repdat_file_out_path)):
        warnings.warn("Skipped repdat creation as already existed!: " + repdat_file_out_path)
    else:
        if verbose: print("JOB " + str(job) + ": Found repdats: " + str(repdat_file_paths))  # verbose_repdat
        if (isinstance(repdat_file_paths, str)):
            repdat_file_paths = [repdat_file_paths]

        if verbose: print("JOB " + str(job) + ": Concatenate repdats: \tSTART")  # verbose_repdat
        repdat_file = repdat.Repdat(repdat_file_paths.pop(0))  # repdat Class
        for repdat_path in repdat_file_paths:
            if verbose: print("JOB " + str(job) + ": concat:\t", repdat_path)
            tmp_repdat_file = repdat.Repdat(repdat_path)
            repdat_file.append(tmp_repdat_file)
            del tmp_repdat_file

        if verbose: print("JOB " + str(job) + ": Concatenate repdats: \tDONE")  # verbose_repdat
        if verbose: print("JOB " + str(job) + ": Write out repdat: " + str(repdat_file_out_path))  # verbose_repdat
        repdat_file.write(repdat_file_out_path)
        del repdat_file


def _thread_worker_cnfs(job : int,
                        out_cnfs : dict,
                        in_cnfs : (str, List[str]),
                        replica_range : List[int],
                        out_folder : str,
                        verbose: bool = False):
    """_thread_worker_cnfs

    Parameters
    ----------
    job : int
        rank of this thread
    out_cnfs : dict
    in_cnfs : (str, List[str])
        list of input cnf files
    replica_range : List[int]
        list of replica IDs
    out_folder : str
        path to output directory
    verbose : bool
        verbose output (default False)

    Returns
    -------
    None
    """
    if (verbose): print("JOB: " + str(job) + " copy to " + out_folder)
    for replicaID in replica_range:
        out_cnfs.update({replicaID: bash.copy_file(in_cnfs[replicaID][-1],
                                                   out_folder + "/" + os.path.basename(in_cnfs[replicaID][-1]))})


def _thread_worker_conv_trc(job: int,
                            replica_range: Iterable[int],
                            trc_files: List[str],
                            in_topology_path: str,
                            gromos_path: str,
                            out_traj: dict,
                            fit_traj_to_mol: int = 1,
                            boundary_conditions: str = "r",
                            verbose: bool = False):
    """_thread_worker_conv_trc

    Parameters
    ----------
    job : int
        rank of this thread
    replica_range : Iterable[int]
        list of replica IDs
    trc_files : List[str]
        list of trc files
    in_topology_path : str
        path to input topology file
    gromos_path : str
        path to gromos binaries
    out_traj : dict
    fit_traj_to_mol : int, optional
        default 1
    boundary_conditions : str, optional
        default "r"
    verbose: bool, optional
        verbose output (default False)

    Returns
    -------
    None
    """
    if (verbose): print("JOB: " + str(job) + " RANGE\t" + str(replica_range))
    gromPP = gromosPP.GromosPP(gromos_path)
    first = True
    import mdtraj
    start_dir = os.getcwd()
    for replicaID in replica_range:
        trc_path = trc_files[replicaID]
        if (first):
            temp = tempfile.mkdtemp(suffix="_job" + str(job), prefix="convert_", dir=os.path.dirname(trc_path))
            os.chdir(temp)
            first = False

        unarchived = False
        use_trc = trc_path

        if (verbose): print("using trc:", use_trc)
        out_top_path = use_trc.replace(".trc", "_last.pdb").replace(".gz", "")
        out_traj_path = use_trc.replace(".trc", ".dcd").replace(".gz", "")

        out_traj.update({replicaID: {"top": out_top_path, "traj": out_traj_path}})

        if (os.path.exists(out_top_path) and os.path.exists(out_traj_path)):
            warnings.warn("Found already the traj and its top!:" + out_traj_path)
            continue

        if verbose: print(
            "JOB " + str(job) + ": Converting trc_path to -> " + use_trc.replace("trc", "pdb").replace(".tar.gz", ""))
        pdb = gromPP.frameout(in_top_path=in_topology_path, in_coord_path=use_trc,
                              periodic_boundary_condition=boundary_conditions,
                              gather="cog", out_file_format="pdb",
                              atomsfit=str(fit_traj_to_mol) + ":a", single_file=True,
                              out_file_path=use_trc.replace("trc", "pdb").replace("tar", "").replace(".gz", ""))

        if verbose: print("JOB " + str(job) + ": loading pdb : " + pdb)
        traj = mdtraj.load_pdb(pdb)
        if verbose: print("JOB " + str(job) + ": write out data to " + use_trc.replace(".trc", ".dcd/.pdb"))
        traj.save(out_traj_path)
        traj[0].save(out_top_path)

        del traj
        if (verbose): print("Done writing out")
        bash.remove_file(pdb)

        if (unarchived):
            if (verbose): print("Clean unarchiving")
            bash.remove_file(use_trc)
    bash.remove_folder(temp)
    os.chdir(start_dir)


def thread_worker_isolate_energies(in_en_file_paths: str,
                                   out_folder: str,
                                   properties: List[str],
                                   replicas: List[int],
                                   in_ene_ana_lib: str,
                                   gromosPP_path: str,
                                   out_prefix: str = "",
                                   tre_prefix: str = "",
                                   time=None, dt=None,
                                   job: int = -1,
                                   verbose=True) -> List[str]:
    """isolate_properties_from_tre
        This func uses Ene Ana from gromos to isolate potentials from out_tre Files
        in in_folder generated by reeds.

    Parameters
    ----------
    in_en_file_paths : str
        path, in which the input tre_folders are situated.
    out_folder : str
         output folder, where to write the energy .csvs
    properties : List[str]
        potentials to isolate from the .out_tre Files
    replicas : List[int]
        list of replicas, that should be found
    in_ene_ana_lib : str
         path to the ene_ana lib, encoding the out_tre Files
    gromosPP_path : str
        path to the ene_ana lib, encoding the out_tre Files
    out_prefix : str, optional
    tre_prefix : str, optional
    time : float, optional
        start time (default None)
    dt : float, optional
        timestep (default None)
    job : int, optional
        rank of current job
    verbose : bool, optional
        verbose output (default True)

    Returns
    -------
    List[str]
        return list of result Files.

    """

    gromos = gromosPP.GromosPP(gromosPP_path)
    result_files = []
    temp = tempfile.mkdtemp(suffix="_job" + str(job), prefix="ene_ana_", dir=out_folder)

    start_dir = os.getcwd()
    os.chdir(temp)
    # get the potentials from each replica.tre*
    if (verbose): print("JOB" + str(job) + ": working with job: " + str(list(replicas)))
    for replicaID in replicas:
        in_en_file_path = in_en_file_paths[replicaID]

        out_suffix = "energies_s" + str(replicaID)
        out_path = out_folder + "/" + out_prefix + "_" + out_suffix + ".dat"

        if (verbose): print("CHECKING: " + out_path)
        if (not os.path.exists(out_path)):
            verbose = True
            if verbose: print("JOB" + str(job) + ":\t" + str(replicaID))
            if (verbose): print("JOB" + str(job) + ":", in_en_file_path)
            tmp_out = gromos.ene_ana(in_ene_ana_library_path=in_ene_ana_lib, in_en_file_paths=in_en_file_path,
                                     out_energy_folder_path=out_folder,
                                     out_files_suffix=out_suffix, out_files_prefix=out_prefix,
                                     time=str(time) + " " + str(dt),
                                     in_properties=properties, verbose=verbose, single_file=True, workdir=True)
            result_files.append(tmp_out)
            bash.remove_file(temp + "/*")  # remove logs if succesfull

    os.chdir(start_dir)
    bash.remove_folder(temp)
    if (verbose): print("JOB" + str(job) + ": DONE")
    return result_files


def _thread_worker_delete(job: int,
                          file_paths: List[str],
                          verbose: bool = False) -> int:
    """_thread_worker_delete
    delete files in list

    Parameters
    ----------
    job : int
        rank of current job
    file_paths : List[str]
        paths to files which should be deleted
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    int
    """
    for file_path in file_paths:
        if (verbose): print("JOB" + str(job) + " - rm: " + file_path + "")
        bash.remove_file(file_path)
    return 0


def _thread_worker_compress(job: int,
                            in_file_paths: List[str], 
                            verbose: bool = False) -> int:
    """_thread_worker_compress
    compress files in list

    Parameters
    ----------
    job : int
        rank of current job
    in_file_paths : List[str]
        paths to files which should be deleted
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    int
    """
    for file_path in in_file_paths:
        if (verbose): print("JOB" + str(job) + " - gz: " + file_path)
        bash.compress_gzip(in_path=file_path, verbose=verbose)
    return 0


"""
    FILE FINDER
"""


def find_and_unarchive_tar_files(trc_files: List[str], verbose: bool = False):
    """find_and_unarchive_tar_files
    find and untar/unzip archived files

    Parameters
    ----------
    trc_files : List[str]
        trajectory files to be untared
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    None
    """
    # archive handling
    archived_files = list(filter(lambda x: (".tar" in x or ".gz" in x or ".tar.gz" in x), trc_files))
    not_archived_files = list(filter(lambda x: not ("tar" in x or ".gz" in x or ".tar.gz" in x), trc_files))
    unarchived_files = []
    if (verbose): print("archives: ", archived_files)
    if (verbose): print("narchives: ", not_archived_files)

    # untar files:
    for tared_file in archived_files:
        if (len(not_archived_files) == 0 or not any([noAfile in tared_file for noAfile in not_archived_files])):
            try:
                # print("Ungzip ->\t", tared_file)
                out_path = bash.compress_gzip(in_path=tared_file,
                                              out_path=tared_file.replace(".tar", "").replace(".gz", ""), extract=True)
            except:
                # print("Failed gzip, trying tar")
                out_path = bash.extract_tar(in_path=tared_file,
                                            out_path=tared_file.replace(".tar", "").replace(".gz", ""),
                                            gunzip_compression=True)

            # fix for stupid taring!    #todo: remove part
            if (any(["cluster" == xfile for xfile in os.listdir(os.path.dirname(tared_file))]) and not os.path.exists(
                    out_path)):
                nfound = True
                for cpath, tdir, files in os.walk(os.path.dirname(tared_file) + "/cluster"):
                    if (os.path.basename(tared_file).replace(".tar", "").replace(".gz", "") in files):
                        if (verbose): print("FOUND PATH: ",
                                            cpath + "/" + os.path.basename(tared_file).replace(".tar", "").replace(
                                                ".gz", ""))
                        wrong_path = cpath + "/" + os.path.basename(tared_file).replace(".tar", "").replace(".gz", "")
                        out_file = bash.move_file(wrong_path, tared_file.replace(".tar", "").replace(".gz", ""))
                        unarchived_files.append(out_file)
                        nfound = False
                        break
                if (nfound):
                    raise IOError("could not find untarred file!")
            else:
                unarchived_files.append(out_path)

                # raise Exception("this tar needs special treatment as it is in cluster/yadda/yadda/fun.trc")
        else:
            if (verbose): print([noAfile for noAfile in not_archived_files if (noAfile in tared_file)])
    new_files = not_archived_files
    new_files.extend(unarchived_files)

    use_tre_file_paths = sorted(new_files, key=lambda x: int(x.split("_")[-1].split(".")[0]))
    return use_tre_file_paths, unarchived_files


def gather_simulation_replica_file_paths(in_folder: str,
                                         replicas: int,
                                         filePrefix: str = "",
                                         fileSuffixes: Union[str, List[str]] = [".tre", ".tre.tar.gz", ".tre.gz"],
                                         verbose: bool = False,
                                         finalNumberingSort=False) -> Dict[int, List[str]]:
    """gather_simulation_replica_file_paths

    Finds all trajectory paths in a simulation folder and sorts them by replica.


    Parameters
    ----------
    in_folder : str
        folder, containing the files
    replicas : int
        Number of replicas
    filePrefix : str, optional
        str prefix the desired files (default "")
    fileSuffixes : str, optional
        str suffix of the desired files (default [".tre", ".tre.tar.gz"])
    finalNumberingSort : bool, optional
        default False
    verbose :   bool
        toggle verbosity

    Returns
    -------
    Dict[int, List[str]]
    """

    if (isinstance(fileSuffixes, str)):
        fileSuffixes = [fileSuffixes]

    # browse folders
    ##init statewise dic
    files = {}
    for replica in range(1, replicas + 1):
        files.update({replica: []})

    if (verbose): print("SEARCH PATTERN: " + filePrefix + " + * +" + str(fileSuffixes))

    if(finalNumberingSort):
        files[1]=[]
        for dirname, dirnames, filenames in os.walk(in_folder):
            if (str(dirname[-1]).isdigit() and os.path.basename(dirname).startswith("eq")):
                continue

            # check actual in_dir for fle pattern 
            tmp_files = [file for file in filenames if (filePrefix in file and any([suffix in file for suffix in fileSuffixes]))]
            if(len(tmp_files)>0):
                print(tmp_files)
                file_paths = sorted([dirname + "/" + filep for filep in tmp_files ])
                files[1].extend(file_paths)

        # final_cleanup
        for x in files:
            files[x].sort(key=lambda x: int(x.split("_")[-1].split(".")[0]))


    else:
       for dirname, dirnames, filenames in os.walk(in_folder):
            if (str(dirname[-1]).isdigit() and os.path.basename(dirname).startswith("eq")):
               continue
            # check actual in_dir for fle pattern
            tmp_files = [file for file in filenames if
                        (filePrefix in file and any([suffix in file for suffix in fileSuffixes]))]
            
            if len(tmp_files) != 0:
                for x in range(1, replicas + 1):
                    grom_file_prefix = sorted([dirname + "/" + file for file in tmp_files if
                                           (any(["_" + str(x) + suffix in file for suffix in fileSuffixes]))])
                    files[x] += grom_file_prefix
            if verbose: print("walking to in_dir: ", os.path.basename(dirname), "found: ", len(tmp_files))

        # final_cleanup
       for x in files:
           files[x].sort(key=lambda x: int(x.split("_")[-2]))

    if (verbose):
        print("\nfoundFiles:\n")
        for x in sorted(files):
            print("\n" + str(x))
            print("\t" + "\t".join([y + "\n" for y in files[x]]))

    if (len(files[1]) == 0):
        raise ValueError("could not find any file with the prefix: " + filePrefix + " in folder : \n" + in_folder)

    return files


def gather_simulation_file_paths(in_folder: str, filePrefix: str = "",
                                 fileSuffixes: Union[str, List[str]] = [".tre", ".tre.tar.gz"],
                                 files_per_folder: int = 1,
                                 verbose: bool = False) -> List[str]:
    """gather_simulation_file_paths
    find energy trajectory files in a folder

    Parameters
    ----------
    in_folder : str
        directory where the files should be searched
    filePrefix : str, optional
        prefix of the file name pattern (default "")
    fileSuffixes : Union[str, List[str]]
        suffixes of the file name pattern (default [".tre", ".tre.tar.gz"])
    files_per_folder : int, optional
        number of files per folder (default 1)
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    List[str]  
        list of sorted files

    """

    files = []
    if (isinstance(fileSuffixes, str)):
        fileSuffixes = [fileSuffixes]

    if (verbose): print("SEARCH PATTERN: " + filePrefix + " + * +" + str(fileSuffixes))

    for dirname, dirnames, filenames in os.walk(in_folder):
        if (str(dirname[-1]).isdigit() and os.path.basename(dirname).startswith("eq")):
            continue
        # check actual in_dir for fle pattern
        tmp_files = [file for file in filenames if
                     (filePrefix in file and any([suffix in file for suffix in fileSuffixes]))]

        if (len(tmp_files) == files_per_folder):
            files.extend(list(map(lambda x: dirname + "/" + x, tmp_files)))

        if verbose: print("walking to in_dir: ", os.path.basename(dirname), "found: ", len(tmp_files))

    try:
        keys = [[int(y) for y in x.split("_") if (y.isdecimal())][-1] for x in files]
        sorted_files = list(map(lambda y: y[1], sorted(zip(keys, files), key=lambda x: x[0])))
    except:
        warnings.warn("Files are not all enumerated! no file sorting.")
        sorted_files = files

    if (verbose):
        print("\nfoundFiles:\n")
        print("\t" + "\n\t".join(sorted_files))

    if (len(sorted_files) == 0):
        raise ValueError("could not find any file with the prefix: " + filePrefix + " in folder : \n" + in_folder)

    return sorted_files


"""
    ENERGY FILE FUNCTIONS
"""


def extract_eds_energies_from_tre(in_dir: str, out_dir: str,
                                  in_ene_ana_lib_path: str,
                                  num_replicas: int, num_states: int,
                                  in_gromosPP_bin_dir: str = None,
                                  out_file_prefix: str = "eds_energies",
                                  additional_properties: Union[Tuple[str], List[str]] = ("solvtemp2", "totdisres"),
                                  n_processes: int = 1,
                                  verbose: bool = False) -> Iterable[str]:
    """extract_eds_energies_from_tre
    This function extracts the EDS energies from a Gromos energy trajectory file

    Parameters
    ----------
    in_dir : str
        directory where the tre files are located
    out_dir : str
        directory where the concatenated energy file should be stored
    in_ene_ana_lib_path : str
        path to the ene_ana library
    num_replicas : int
        number of replicas in the system
    num_states : int
        number of states in the system
    in_gromosPP_bin_dir : str, optional
        path to directory with gromosPP executable (default None)
    out_file_prefix : str, optional
        file name prefix for output file (default "eds_energies")
    additional_properties : Union[Tuple[str], List[str]], optional
        additional properties given to ene_ana (default ("solvtemp2", "totdisres"))
    n_processes : int, optional
        number of parallel processes to use (default 1)
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    Iterable[str]
        list of energy output files
    """


    # gather potentials
    properties = list(additional_properties) + ["eR"] + ["e" + str(state) for state in range(1, num_states + 1)]

    # isolate potentials
    if verbose: print("Isolate ene_ana:")
    if (n_processes > 1):
        p = mult.Pool(n_processes)
        distribute_jobs = [(in_dir, out_dir, properties, range(n, num_replicas, n_processes), in_ene_ana_lib_path,
                            in_gromosPP_bin_dir, out_file_prefix, "", n, verbose) for n in range(n_processes)]
        p_ene_ana = p.starmap_async(thread_worker_isolate_energies, distribute_jobs)
        p.close()
        p.join()

    else:
        out_files = thread_worker_isolate_energies(in_en_file_paths=in_dir,
                                                   out_folder=out_dir,
                                                   properties=properties,
                                                   out_prefix=out_file_prefix,
                                                   in_ene_ana_lib=in_ene_ana_lib_path,
                                                   gromosPP_path=in_gromosPP_bin_dir,
                                                   replicas=[num_replicas])

    return out_files


def find_header(path: str) -> int:
    """find_header
    this function counts the lines of the header (i.e. how many lines there are
    until the first line with doesn't start with a '#' is encountered)

    Parameters
    ----------
    path : str
        file path

    Returns
    -------
    int
        number of lines belonging to the header
    """
    comment_lines = -1
    with open(path, "r") as file:
        for line in file.readlines():
            if (line.strip().startswith("#")):
                comment_lines += 1
                continue
            else:
                break
        if (comment_lines == -1):
            return 0
    return comment_lines


def parse_csv_energy_trajectory(in_ene_traj_path: str, verbose: bool = False) -> pd.DataFrame:
    """parse_csv_energy_trajectory

    Parameters
    ----------
    in_ene_traj_path : str
        path to input file

    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    pd.DataFrame
        return a pandas data frame containing all energies
    """
    if (verbose): print("deal with: ", in_ene_traj_path)
    ene_traj = pd.read_csv(in_ene_traj_path, header=find_header(in_ene_traj_path), delim_whitespace=True)
    ene_traj.columns = [x.replace("#", "").strip() for x in ene_traj.columns]
    setattr(ene_traj, "in_path", in_ene_traj_path)
    return ene_traj


def parse_csv_energy_trajectories(in_folder: str,
                                  ene_trajs_prefix: str,
                                  verbose: bool = False) -> List[pd.DataFrame]:
    """parse_csv_energy_trajectories
    searches a directory and loads energy eds csvs as pandas dataframes.

    Parameters
    ----------
    in_folder : str
        folder with energy_traj - csvs
    ene_trajs_prefix : str
        prefix name
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    List[pd.DataFrame]
        return a list with pandas data frames containing all energy infos.
    """

    if (verbose): print("SEARCH: ", in_folder + "/" + ene_trajs_prefix + "*.dat")
    in_ene_traj_paths = sorted(glob.glob(in_folder + "/" + ene_trajs_prefix + "*.dat"),
                               key=lambda x: int(x.split("_")[-1].split(".")[0].replace("s", "")))
    ene_trajs: List[pd.DataFrame] = []
    if (verbose): print("FOUND: ", "\n".join(in_ene_traj_paths))

    for in_ene_traj_path in in_ene_traj_paths:
        ene_traj = parse_csv_energy_trajectory(in_ene_traj_path, verbose=verbose)
        if (verbose): print("csv columns: \t", ene_traj.columns)
        # note: the previous version created problems for filenames which contained periods
        #setattr(ene_traj, "s", ((ene_traj.in_path.split("."))[0]).split("_")[-1])
        #setattr(ene_traj, "replicaID", int(((ene_traj.in_path.split("."))[0]).split("_")[-1].replace("s", "")))
        setattr(ene_traj, "s", ((ene_traj.in_path.split("."))[-2]).split("_")[-1])
        setattr(ene_traj, "replicaID", int(((ene_traj.in_path.split("."))[-2]).split("_")[-1].replace("s", "")))
        ene_trajs.append(ene_traj)

    if (len(ene_trajs) == 0):
        raise ValueError("could not find any energy_trajectory in: ", in_folder + "/" + ene_trajs_prefix + "*.dat")

    ene_trajs = list(sorted(ene_trajs, key=lambda x: int(x.s.replace("s", ""))))
    return ene_trajs


"""
    concatenation wrapper
"""


def project_concatenation(in_folder: str,
                          in_topology_path: str,
                          in_imd: str, num_replicas: int,
                          control_dict: Dict[str, bool],
                          out_folder: str,
                          in_ene_ana_lib_path: str,
                          out_file_prefix: str = "test",
                          fit_traj_to_mol: int = 1,
                          starting_time: float = 0,
                          include_water_in_trc=True,
                          additional_properties: Union[Tuple[str],List[str]] = ("solvtemp2", "totdisres"),
                          n_processes: int = 1,
                          gromosPP_bin_dir: str = None,
                          nofinal=False,
                          boundary_conditions: str = "r cog",
                          verbose: bool = False) -> dict:
    """project_concatenation
    wrapper for the concatenation

    Parameters
    ----------
    in_folder : str
        input directory
    in_topology_path : str
        path to the topology file
    in_imd : str
        path to the imd file
    num_replicas : int
        number of replicas
    control_dict : Dict[str, bool]
        control dict to specify what should be executed or skipped
    out_folder : str
        output directory
    in_ene_ana_lib_path : str
        path to ene_ana lib file
    out_file_prefix : str, optional
        prefix for the output file names (default "test")
    fit_traj_to_mol : int, optional
        parameter for gromos++ frameout @atomsfit (default 1)
    starting_time : float, optional
        start time of trajectory (default 0)
    include_water_in_trc : bool, optional
        should water molecules be included in trajectory file? (default True)
    additional_properties : Union[Tuple[str],List[str]], optional
        additional properties to give to gromos++ ene_ana (default ("solvtemp2", "totdisres"))
    n_processes : int, optional
        number of parallel processes to use (default 1)
    gromosPP_bin_dir : str, optional
        path to gromos++ executable (default None)
    nofinal : bool, optional
        (default False)
    boundary_condition : str, optional
        boundary condition for gromos++ frameout (default "r cog")
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    dict
        dict containg the out_folder, cnfs, repdat, tres, trcs and dcds
    """
    if (verbose): print("reading imd file: " + in_imd)

    imd_file = imd.Imd(in_imd)
    dt = float(imd_file.STEP.DT)
    dt_trc = int(imd_file.WRITETRAJ.NTWX) * dt
    dt_tre = int(imd_file.WRITETRAJ.NTWE) * dt

    tmp_dir = out_folder + "/tmp_file"
    if (os.path.isdir(tmp_dir)):
        bash.make_folder(tmp_dir)

    out_cnfs = out_tres = out_trcs = out_dcd = out_repdat = None
    p_conv = p_cnf = p_repdat = p_trc = p_tre = p_ene_ana = False
    submitted_trc_job = submitted_tre_job = False

    if (n_processes > 1):
        p = mult.Pool(n_processes)
        manager = mult.Manager()

    if (control_dict["cp_cnf"]):
        if (verbose): print("\tStart cnfs")
        # find all cnf files in this project
        sim_dir_cnfs = gather_simulation_replica_file_paths(in_folder, num_replicas, filePrefix="", fileSuffixes=".cnf",
                                                            verbose=verbose, finalNumberingSort=nofinal)

        # do parallel
        if (n_processes > 1):
            out_cnfs = manager.dict()
            distribute = [(n, out_cnfs, sim_dir_cnfs, range(n, len(sim_dir_cnfs) + 1, n_processes), out_folder, verbose)
                          for n in range(1, n_processes + 1)]
            # _async
            p_cnf = p.starmap(_thread_worker_cnfs, distribute)
        else:
            out_cnfs = {}
            _thread_worker_cnfs(job=-1, out_cnfs=out_cnfs, in_cnfs=sim_dir_cnfs,
                                replica_range=list(sim_dir_cnfs.keys()), out_folder=out_folder, verbose=verbose)
        if (verbose): print("Out cnfs: ", out_cnfs)

    if (control_dict["cat_trc"]):
        print("\tStart Trc Cat")
        # find all trc files in this project
        trc_files = gather_simulation_replica_file_paths(in_folder, num_replicas, filePrefix="",
                                                         fileSuffixes=[".trc", ".trc.gz", ".trc.tar.gz"],
                                                         verbose=verbose,
                                                         finalNumberingSort=nofinal)

        out_prefix = out_folder + "/" + out_file_prefix + "_"

        # concat all files to a single .trc
        if (n_processes > 1):  # parallel
            submitted_trc_job = True
            if (verbose): print("going parallel: n_processes - " + str(n_processes))
            out_trcs = manager.dict()
            distributed_jobs = [
                (n, range(n, len(trc_files) + 1, n_processes), trc_files, out_prefix, in_topology_path, out_trcs,
                 dt_trc, starting_time, verbose, include_water_in_trc) for n in range(1, n_processes + 1)]
            # _async
            p_trc = p.starmap(_thread_worker_cat_trc, distributed_jobs)
            p.close()
            p.join()

        else:
            out_trcs = {}
            _thread_worker_cat_trc(job=-1, topology_path=in_topology_path, replicaID_range=list(trc_files.keys()),
                                   trc_files=trc_files, out_prefix=out_prefix, dt=dt_trc, time=starting_time,
                                   out_trcs=out_trcs,
                                   verbose=verbose, boundary_conditions=boundary_conditions,
                                   include_all=include_water_in_trc)

    if (control_dict["cat_tre"]):
        print("\tStart Tre Cat")

        # find all trc files in this project
        tre_files = gather_simulation_replica_file_paths(in_folder, num_replicas, filePrefix="",
                                                         fileSuffixes=[".tre", ".tre.tar.gz"], verbose=verbose,
                                                         finalNumberingSort=nofinal)

        out_prefix = out_folder + "/" + out_file_prefix + "_"
        # concat all files to a single .trc
        if (n_processes > 1):
            if (verbose): print("going parallel: n_processes - " + str(n_processes), " for ", len(tre_files))
            submitted_tre_job = True
            out_tres = manager.dict()
            distributed_jobs = [(n, range(n, len(tre_files) + 1, n_processes), tre_files, out_prefix, out_tres, verbose)
                                for n in range(1, n_processes + 1)]
            p = mult.Pool(n_processes)
            p_tre = p.starmap(_thread_worker_cat_tre, distributed_jobs)
            p.close()
            p.join()
        else:
            out_tres = {}
            _thread_worker_cat_tre(job=-1, replicaID_range=tre_files.keys(), tre_files=tre_files, out_prefix=out_prefix,
                                   out_tres=out_tres, verbose=verbose)

    if (control_dict["ene_ana"]):
        print("\tStart ene ana")

        # wait for async job creating the trcs.
        if (submitted_tre_job):
            p_tre.wait()

        # gather potentials
        properties = list(additional_properties)
        # find all trc files in this project
        tre_files = gather_simulation_replica_file_paths(in_folder, num_replicas, filePrefix="",
                                                         fileSuffixes=[".tre", ".tre.gz"], verbose=verbose,
                                                         finalNumberingSort=nofinal)  # ".tre.tar.gz"
        # isolate potentials
        if verbose: print("Isolate ene_ana:")
        if (n_processes > 1):
            p = mult.Pool(n_processes)
            distribute_jobs = [
                (out_folder, out_folder, properties, list(tre_files.keys())[n::n_processes], in_ene_ana_lib_path,
                 gromosPP_bin_dir, out_file_prefix, "", dt_tre, n, verbose) for n in range(n_processes)]
            p_ene_ana = p.starmap_async(thread_worker_isolate_energies, distribute_jobs)
        else:
            thread_worker_isolate_energies(in_en_file_paths=tre_files, out_folder=out_folder, properties=properties,
                                           out_prefix=out_file_prefix,
                                           in_ene_ana_lib=in_ene_ana_lib_path,
                                           gromosPP_path=gromosPP_bin_dir, dt=dt_tre,
                                           replicas=list(tre_files.keys()), verbose=verbose)

    if (control_dict["convert_trcs"]):
        print("\tStart Trc Conversion")
        # wait for async job creating the trcs.
        if (submitted_trc_job):
            p_trc.wait()

        # get files:
        final_trc_files = list(
            sorted(glob.glob(out_folder + "/*.trc*"), key=lambda x: int(x.split("_")[-1].split(".")[0])))

        if (n_processes > 1):
            out_dcd = manager.dict()
            distributed_jobs = [
                (n, range(n, num_replicas, n_processes), final_trc_files, in_topology_path, gromosPP_bin_dir, out_dcd,
                 fit_traj_to_mol, verbose) for
                n in range(n_processes)]
            p_conv = p.starmap_async(_thread_worker_conv_trc, distributed_jobs)
        else:
            out_dcd = {}
            _thread_worker_conv_trc(job=-1, replica_range=range(num_replicas), trc_files=final_trc_files,
                                    in_topology_path=in_topology_path,
                                    gromos_path=gromosPP_bin_dir, out_traj=out_dcd, fit_traj_to_mol=1, verbose=verbose,
                                    boundary_conditions=boundary_conditions)

    if (n_processes > 1):
        # wait for the jobs to finish
        if ((not p_conv or p_conv.wait()) and (not p_cnf or p_cnf.wait()) and
                (not p_repdat or p_repdat.wait()) and (not p_trc or p_trc.wait()) and
                (not p_tre or p_tre.wait()) and (not p_ene_ana or p_ene_ana.wait())):
            raise ChildProcessError("A process failed! ")

        p.close()
        p.join()

        out_dict = {"out_folder": out_folder, "cnfs": dict(out_cnfs), "repdat": out_repdat, "tres": dict(out_tres),
                    "trcs": dict(out_trcs), "dcds": dict(out_dcd)}
        manager.shutdown()

    else:
        out_dict = {"out_folder": out_folder, "cnfs": out_cnfs, "repdat": out_repdat, "tres": out_tres,
                    "trcs": out_trcs, "dcds": out_dcd}
    if (verbose): print("all jobs finished")
    return out_dict


def reeds_project_concatenation(in_folder: str,
                                in_topology_path: str,
                                in_imd: str,
                                num_replicas: int,
                                control_dict: Dict[str, bool],
                                out_folder: str,
                                in_ene_ana_lib_path: str,
                                repdat_file_out_path: str or None,
                                out_file_prefix: str = "test",
                                fit_traj_to_mol: int = 1,
                                starting_time: float = 0,
                                include_water_in_trc=True,
                                additional_properties: Union[Tuple[str], List[str]] = ("solvtemp2", "totdisres"),
                                n_processes: int = 1,
                                gromosPP_bin_dir: str = None,
                                nofinal=False,
                                boundary_conditions: str = "r cog",
                                verbose: bool = False) -> dict:
    """reeds_project_concatenation 
    wrapper for the concatenation for REEDS

    Parameters
    ----------
    in_folder : str
        input directory
    in_topology_path : str
        path to the topology file
    in_imd : str
        path to the imd file
    num_replicas : int
        number of replicas
    control_dict : Dict[str, bool]
        control dict to specify what should be executed or skipped
    out_folder : str
        output directory
    in_ene_ana_lib_path : str
        path to ene_ana lib file
    out_file_prefix : str, optional
        prefix for the output file names (default "test")
    fit_traj_to_mol : int, optional
        parameter for gromos++ frameout @atomsfit (default 1)
    starting_time : float, optional
        start time of trajectory (default 0)
    include_water_in_trc : bool, optional
        should water molecules be included in trajectory file? (default True)
    additional_properties : Union[Tuple[str],List[str]], optional
        additional properties to give to gromos++ ene_ana (default ("solvtemp2", "totdisres"))
    n_processes : int, optional
        number of parallel processes to use (default 1)
    gromosPP_bin_dir : str, optional
        path to gromos++ executable (default None)
    nofinal : bool, optional
        (default False)
    boundary_condition : str, optional
        boundary condition for gromos++ frameout (default "r cog")
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    dict
        dict containg the out_folder, cnfs, repdat, tres, trcs and dcds
    """
    if (verbose): print("reading imd file: " + in_imd)
    imd_file = imd.Imd(in_imd)
    dt = float(imd_file.STEP.DT)
    dt_trc = int(imd_file.WRITETRAJ.NTWX) * dt
    dt_tre = int(imd_file.WRITETRAJ.NTWE) * dt
    num_states = int(imd_file.REPLICA_EDS.NUMSTATES)

    tmp_dir = out_folder + "/tmp_file"
    if (os.path.isdir(tmp_dir)):
        bash.make_folder(tmp_dir)

    out_cnfs = out_tres = out_trcs = out_dcd = out_repdat = None
    p_conv = p_cnf = p_repdat = p_trc = p_tre = p_ene_ana = False
    submitted_trc_job = submitted_tre_job = False

    if (n_processes > 1):
        p = mult.Pool(n_processes)
        manager = mult.Manager()

    if (control_dict["cp_cnf"]):
        if (verbose): print("\tStart cnfs")
        # find all cnf files in this project
        sim_dir_cnfs = gather_simulation_replica_file_paths(in_folder, num_replicas, filePrefix="", fileSuffixes=".cnf",
                                                            verbose=verbose, finalNumberingSort=nofinal)

        # do parallel
        if (n_processes > 1):
            out_cnfs = manager.dict()
            distribute = [(n, out_cnfs, sim_dir_cnfs, range(n, len(sim_dir_cnfs) + 1, n_processes), out_folder, verbose)
                          for n in range(1, n_processes + 1)]
            # _async
            p = mult.Pool(n_processes)
            p_cnf = p.starmap(_thread_worker_cnfs, distribute)
            p.close()
            p.join()
        else:
            out_cnfs = {}
            _thread_worker_cnfs(job=-1, out_cnfs=out_cnfs, in_cnfs=sim_dir_cnfs,
                                replica_range=list(sim_dir_cnfs.keys()), out_folder=out_folder, verbose=verbose)
        if (verbose): print("Out cnfs: ", out_cnfs)

    if (control_dict["cat_trc"]):
        print("\tStart Trc Cat")
        # find all trc files in this project
        trc_files = gather_simulation_replica_file_paths(in_folder, num_replicas, filePrefix="",
                                                         fileSuffixes=[".trc", ".trc.gz"], verbose=verbose,
                                                         finalNumberingSort=nofinal)  # ,".trc.tar.gz"

        # trc_files=helper_function_change_compression_safe_incomplete_files(trc_files, in_topology_path=in_topology_path, tmp_dir=tmp_dir)

        out_prefix = out_folder + "/" + out_file_prefix + "_"

        # concat all files to a single .trc
        if (n_processes > 1):  # parallel
            submitted_trc_job = True
            if (verbose): print("going parallel: n_processes - " + str(n_processes))
            out_trcs = manager.dict()
            distributed_jobs = [(
                n, range(n, len(trc_files) + 1, n_processes), trc_files, out_prefix, in_topology_path, out_trcs,
                dt_trc, starting_time, verbose, boundary_conditions, include_water_in_trc) for n in
                range(1, n_processes + 1)]
            p = mult.Pool(n_processes)
            p_trc = p.starmap(_thread_worker_cat_trc, distributed_jobs)
            p.close()
            p.join()
        else:
            out_trcs = {}
            _thread_worker_cat_trc(job=-1, topology_path=in_topology_path, replicaID_range=list(trc_files.keys()),
                                   trc_files=trc_files, out_prefix=out_prefix, dt=dt_trc, time=starting_time,
                                   out_trcs=out_trcs,
                                   verbose=verbose, boundary_conditions=boundary_conditions,
                                   include_all=include_water_in_trc)

    if (control_dict["cat_tre"]):
        print("\tStart Tre Cat")

        # find all trc files in this project
        tre_files = gather_simulation_replica_file_paths(in_folder, num_replicas, filePrefix="",
                                                         fileSuffixes=[".tre", ".tre.gz"], verbose=verbose,
                                                         finalNumberingSort=nofinal)  # ".tre.tar.gz"

        # tre_files=helper_function_change_compression_safe_incomplete_files(tre_files, in_topology_path=in_topology_path, tmp_dir=tmp_dir)

        out_prefix = out_folder + "/" + out_file_prefix + "_"
        # concat all files to a single .trc
        if (n_processes > 1):
            if (verbose): print("going parallel: n_processes - " + str(n_processes), " for ", len(tre_files))
            submitted_tre_job = True
            out_tres = manager.dict()
            distributed_jobs = [(n, range(n, len(tre_files) + 1, n_processes), tre_files, out_prefix, out_tres, verbose)
                                for n in range(1, n_processes + 1)]
            p = mult.Pool(n_processes)
            p_tre = p.starmap(_thread_worker_cat_tre, distributed_jobs)
            p.close()
            p.join()
        else:
            out_tres = {}
            _thread_worker_cat_tre(job=-1, replicaID_range=tre_files.keys(), tre_files=tre_files, out_prefix=out_prefix,
                                   out_tres=out_tres, verbose=verbose)

    if (control_dict["cat_repdat"]):
        print("\tStart Cat_repdat")

        # browse folders
        repdat_file_paths = gather_simulation_file_paths(in_folder, filePrefix="", fileSuffixes=["repdat.dat"],
                                                         verbose=verbose)
        if (n_processes > 1 and False):  # submit to pool for async execution
            distributed_jobs = [(0, repdat_file_out_path, repdat_file_paths, verbose)]
            p = mult.Pool(n_processes)
            p_repdat = p.apply(thread_worker_concat_repdat, distributed_jobs)
            p.close()
            p.join()
        else:
            thread_worker_concat_repdat(job=-1, repdat_file_out_path=repdat_file_out_path,
                                        repdat_file_paths=repdat_file_paths, verbose=verbose)
        out_repdat = repdat_file_out_path

    if (control_dict["ene_ana"]):
        print("\tStart ene ana")

        # wait for async job creating the trcs.
        # if (submitted_tre_job):
        # p_tre.wait()

        # gather potentials
        # find all trc files in this project
        tre_files = gather_simulation_replica_file_paths(in_folder, num_replicas, filePrefix="",
                                                         fileSuffixes=[".tre", ".tre.gz"], verbose=verbose,
                                                         finalNumberingSort=nofinal)  # ".tre.tar.gz"

        if (verbose): print(tre_files)
        properties = list(additional_properties) + ["eR"] + ["e" + str(state) for state in range(1, num_states + 1)]

        # isolate potentials
        if verbose: print("Isolate ene_ana:")
        if (n_processes > 1):
            p = mult.Pool(n_processes)
            distribute_jobs = [
                (tre_files, out_folder, properties, list(tre_files.keys())[n::n_processes], in_ene_ana_lib_path,
                 gromosPP_bin_dir, out_file_prefix, "", 0, dt_tre, n, verbose) for n in range(n_processes)]
            p_ene_ana = p.starmap(thread_worker_isolate_energies, distribute_jobs)
            p.close()
            p.join()
        else:
            out_files = thread_worker_isolate_energies(in_en_file_paths=tre_files, out_folder=out_folder,
                                                       properties=properties,
                                                       out_prefix=out_file_prefix,
                                                       in_ene_ana_lib=in_ene_ana_lib_path,
                                                       gromosPP_path=gromosPP_bin_dir, time=0, dt=dt_tre,
                                                       replicas=list(tre_files.keys()),
                                                       verbose=verbose)

    if (control_dict["convert_trcs"]):
        print("\tStart Trc Conversion")
        # wait for async job creating the trcs.
        # if (submitted_trc_job):
        #    p_trc.wait()

        # get files:
        final_trc_files = list(
            sorted(glob.glob(out_folder + "/*.trc*"), key=lambda x: int(x.split("_")[-1].split(".")[0])))

        if (n_processes > 1):
            out_dcd = manager.dict()
            p = mult.Pool(n_processes)

            distributed_jobs = [
                (n, range(n, num_replicas, n_processes), final_trc_files, in_topology_path, gromosPP_bin_dir, out_dcd,
                 fit_traj_to_mol, verbose) for
                n in range(n_processes)]
            p_conv = p.starmap(_thread_worker_conv_trc, distributed_jobs)
            p.close()
            p.join()
        else:
            out_dcd = {}
            _thread_worker_conv_trc(job=-1, replica_range=range(num_replicas), trc_files=final_trc_files,
                                    in_topology_path=in_topology_path,
                                    gromos_path=gromosPP_bin_dir, out_traj=out_dcd, fit_traj_to_mol=1, verbose=verbose)

    if (n_processes > 1):
        """
        #wait for the jobs to finish
        if ((not p_conv or p_conv.wait()) and (not p_cnf or p_cnf.wait()) and
                (not p_repdat or p_repdat.wait()) and (not p_trc or p_trc.wait()) and
                (not p_tre or p_tre.wait()) and (not p_ene_ana or p_ene_ana.wait())):
            raise ChildProcessError("A process failed! ")

        p.close()
        p.join()
        """
        out_cnfs = None if (isinstance(out_cnfs, type(None))) else dict(out_cnfs)
        out_tres = None if (isinstance(out_tres, type(None))) else dict(out_tres)
        out_trcs = None if (isinstance(out_trcs, type(None))) else dict(out_trcs)
        out_dcd = None if (isinstance(out_dcd, type(None))) else dict(out_dcd)

        manager.shutdown()

    out_dict = {"out_folder": out_folder, "cnfs": out_cnfs, "repdat": out_repdat, "tres": out_tres,
                "trcs": out_trcs, "dcds": out_dcd}
    if (verbose): print("all jobs finished")
    return out_dict


"""
    COMPRESS FUNCTION
"""


def clean_up_dirs(root_dirs: List[str],
                  run_dry: bool = False,
                  delete_stuff: bool = False,
                  n_processors: int = 1,
                  verbose: bool = True):
    """clean_up_dirs
    clean up a directory by removing and compressing files/subdirectories

    Parameters
    ----------
    root_dirs : List[str]
        directories to be cleaned
    run_dry : bool, optional
        test run? (default False)
    delete_stuff : bool, optional
        should simulation.tar.gz be deleted? (default False)
    n_processors : int, optional
        number of parallel processes to use (default 1)
    verbose : bool, optional
        verbose output (default True)

    Returns
    -------
    int
        None    
    """
    if (delete_stuff == True):
        warnings.warn(
            "DELETE_STUFF is true -> I will delete simulation.tar.gz folders if there is an ana folder and all .logs")

    tasks = {"rm": [], "tar": [], }
    if (type(root_dirs) == str):
        root_dirs = [root_dirs]
    for root_dir in root_dirs:
        for path, dirs, files in os.walk(root_dir):
            trx_files = [path + "/" + file for file in files if (file.endswith(".trc") or file.endswith(".tre"))]
            delete_files = [path + "/" + file for file in files if
                            (file == "simulation.tar.gz" or file.endswith(".log") or
                             ("_repout_" in file and ".dat" in file) or
                             ("elm4" in path and "lsf.o" in file) or
                             ("elm4" in path and "emin_box_" in file) or
                             ("elm4" in path and ".sh.err" in file) or
                             ("elm4" in path and ".sh.out" in file) or
                             ("elm4" in path and ".omd" in file) or
                             ("elm2" in path and ".trs" in file) or
                             ("slave" in file and ".out" in file)
                             )]

            tasks["rm"].extend(delete_files)
            tasks["tar"].extend(trx_files)

    if verbose: print("FOUND FOLLOWING TASKS")
    if verbose: print("Remove: \n", "\n".join(tasks["rm"]))
    if verbose: print("Tar: \n", "\n".join(tasks["tar"]))
    if (not run_dry):
        if (n_processors > 1):
            p = mult.Pool(n_processors)
            if (delete_stuff and len(tasks["rm"]) > 0):
                if verbose: print("Do rm: \n")
                distribute = [(job, tasks["rm"][job::n_processors], True) for job in range(n_processors)]
                p.starmap(_thread_worker_delete, distribute)
            if verbose: print("Do Tar: \n")
            distribute = [(job, tasks["tar"][job::n_processors], True, True) for job in range(n_processors)]
            p.starmap(_thread_worker_compress, distribute)
            p.close()
            p.join()
        else:
            if (delete_stuff):
                if verbose: print("Do rm: \n")
                for remove_file in tasks["rm"]:
                    bash.remove_file(remove_file)

            if verbose: print("Do Tar: \n")
            for tar_file in tasks["tar"]:
                bash.compress_gzip(in_path=tar_file)
        if verbose: print("DONE!")
    else:
        if verbose: print("This was a dry run! so I did not do a thing. :)")


def clean_up_euler_dirs(root_dirs: List[str],
                        run_dry: bool = False,
                        delete_stuff: bool = False,
                        n_processors: int = 1,
                        verbose: bool = True):
    """clean_up_euler_dirs

    Parameters
    ----------
    root_dirs : List[str]
        directories to be cleaned
    run_dry : bool, optional
        test run? (default False)
    delete_stuff : bool, optional
        should simulation.tar.gz be deleted? (default False)
    n_processors : int, optional
        number of parallel processes to use (default 1)
    verbose : bool, optional
        verbose output (default True)

    Returns
    -------
    int
        None  
    """

    if (delete_stuff == True):
        warnings.warn(
            "DELETE_STUFF is true -> I will delete simulation.tar.gz folders if there is an ana folder and all .logs")

    tasks = {"rm": [], "tar": [], }
    if (type(root_dirs) == str):
        root_dirs = [root_dirs]
    for root_dir in root_dirs:
        for path, dirs, files in os.walk(root_dir):
            if ("scratch" in path):
                continue
            elif ("simulation" in path):
                trx_files = [path + "/" + file for file in files if (file.endswith(".trc") or file.endswith(".tre"))]
            else:
                trx_files = [path + "/" + file for file in files if (file.endswith(".trc") or file.endswith(".tre"))]
                delete_files = [path + "/" + file for file in files if
                                (file == "simulation.tar.gz" or file.endswith(".log") or
                                 ("_repout_" in file and ".dat" in file) or
                                 (".sh.err" in file) or (".sh.out" in file) or
                                 (".trs" in file) or ("stat.out" in file) or
                                 (file.startswith("e") and not file.startswith("eds_energies") and file.endswith(
                                     ".dat"))
                                 )]
            #                                                                    ("elm4" in path and ".omd" in file) or
            tasks["rm"].extend(delete_files)
            tasks["tar"].extend(trx_files)

    if verbose: print("FOUND FOLLOWING TASKS")
    if verbose: print("Remove: \n", "\n".join(tasks["rm"]))
    if verbose: print("Tar: \n", "\n".join(tasks["tar"]))
    if (not run_dry):
        if (n_processors > 1):
            p = mult.Pool(n_processors)
            if (delete_stuff and len(tasks["rm"]) > 0):
                if verbose: print("Do rm: \n")
                distribute = [(job, tasks["rm"][job::n_processors], True) for job in range(n_processors)]
                p.starmap(_thread_worker_delete, distribute)
            if verbose: print("Do Tar: \n")
            distribute = [(job, tasks["tar"][job::n_processors], True, True) for job in range(n_processors)]
            p.starmap(_thread_worker_compress, distribute)
            p.close()
            p.join()
        else:
            if (delete_stuff):
                if verbose: print("Do rm: \n")
                for remove_file in tasks["rm"]:
                    bash.remove_file(remove_file)

            if verbose: print("Do Tar: \n")
            for tar_file in tasks["tar"]:
                bash.compress_tar(in_path=tar_file, gunzip_compression=True, remove_in_file_afterwards=True)
        if verbose: print("DONE!")
    else:
        if verbose: print("This was a dry run! so I did not do a thing. :)")


def compress_folder(in_paths: list) -> str:
    """compress_folder
    compress a folder
    
    Parameters
    ----------
    in_paths : list
        list of file paths

    Returns
    -------
    list
        list of output file paths
    """
    # compress data folder?
    out_paths = []
    if (type(in_paths) == str):
        in_paths = [in_paths]
    for path in in_paths:
        if (os.path.exists(path)):
            archive_path = bash.compress_tar(in_path=path, out_path=path + ".tar", gunzip_compression=False,
                                             remove_in_dir_afterwards=True)
            out_paths.append(archive_path)
        else:
            warnings.warn("File Path: " + path + " was not found!")
    # add
    return out_paths


def compress_files(in_paths: List[str],
                   n_processes: int = 1) -> List[str]:
    """compress_files
    compress a list of files

    Parameters
    ----------
    in_paths : List[str]
        input file paths
    n_processes: int, optional
        number of parallel processes to use (default 1)
        
    Returns
    -------
    List[str]
        outpaths
    """

    if (type(in_paths) == str):
        in_paths = [in_paths]
    out_paths = []

    # check:
    for path in in_paths:
        if (os.path.exists(path)):
            archive_path = path + ".gz"
            out_paths.append(archive_path)
        else:
            warnings.warn("File Path: " + path + " was not found!")

    # do:
    print("Gen Gzips:")
    if (n_processes == 1):
        for path in in_paths:
            bash.compress_gzip(in_path=path, out_path=path + ".gz")
    else:  # do parallel
        p = mult.Pool(n_processes)
        distribute = [(job, in_paths[job::n_processes], True) for job in range(n_processes)]
        p.starmap(_thread_worker_compress, distribute)
        p.close()
        p.join()
    return out_paths


"""
    Coordinate File Mappings 
    These functions can be used to map the cnf coordinates most smoothly to a new s-distribution.
"""


def adapt_cnfs_to_new_sDistribution(in_old_svals: list,
                                    in_new_svals: list,
                                    in_cnf_files: List[str],
                                    out_cnf_dir: str,
                                    cnf_prefix: str = "REEDS_EOFF_run",
                                    verbose: bool = True) -> list:
    """adapt_cnfs_to_new_sDistribution
    this function should map coordinate files, belonging to an old s-distribution in an optimal way to a new s-distribution.
    Therefore it removes or adds coordinate files.

    Parameters
    ----------
    in_old_svals : list
        list of old s-values
    in_new_svals : list
        list of new s-values
    in_cnf_files : List[str]
        list of old cnf files
    out_cnf_dir : str
        path to output directory for cnf files
    cnf_prefix : str, optional
        prefix for output cnf files (default "REEDS_EOFF_run")
    verbose : bool, optional
        verbose output (default True)

    Returns
    -------
    list
        list of output cnf files
    """
    # INPUT PARSING
    if (isinstance(in_cnf_files, List)):
        in_cnfs = in_cnf_files
        if (len(in_cnfs) == 0):
            raise IOError("Got empty list!: " + in_cnf_files)
    else:
        raise ValueError("Please give either correct in_cnf_files!")

    if (len(in_cnfs) != len(in_old_svals)):
        raise IOError("got different number of old svals and cnfs - no mapping possible!:\n\t svals: " + str(
            len(in_old_svals)) + ""
                                 "\n\t in_cnfs: " + str(
            len(in_cnfs)) + "")
    in_cnfs = list(sorted(in_cnfs, key=lambda x: int(x.split("_")[-1].replace(".cnf", ""))))

    if (verbose): print("Input CNFs")
    if (verbose): print(in_cnfs)
    if (verbose): print()

    s1_repetitions = in_old_svals.count(1.0)

    # DECIDE ON HOW TO MAP
    if (len(in_old_svals) > len(in_new_svals)):
        print("too less ")
        out_cnfs = reduce_cnf_eoff(in_opt_struct_cnf_dir=None, in_num_states=5,
                                   in_current_sim_cnf_dir=in_cnfs, in_old_svals=in_old_svals, in_new_svals=in_new_svals,
                                   out_next_cnfs_dir=out_cnf_dir, run_prefix=cnf_prefix, s1_repetitions=s1_repetitions)
    elif (len(in_old_svals) < len(in_new_svals)):
        print("too many")
        out_cnfs = map_cnfs_to_svalues(in_old_cnf_files=in_cnfs, out_dir=out_cnf_dir, in_old_svals=in_old_svals,
                                       in_new_svals=in_new_svals,
                                       cnf_prefix=cnf_prefix)

    else:
        print("Number of CNF is fine!")
        out_cnfs = in_cnfs

    return out_cnfs


def adapt_cnfs_Eoff_to_Sopt(in_old_svals: list,
                            in_new_svals: list,
                            in_cnf_files: List[str],
                            out_cnf_dir: str,
                            in_opt_struct_cnf_dir: str,
                            in_num_states: int,
                            out_cnf_prefix: str = "REEDS_run",
                            LRTO_mode: bool = False) -> List[str]:
    """adapt_cnfs_Eoff_to_Sopt

    Parameters
    ----------
    in_old_svals : list
        list of old s-values
    in_new_svals : list
        list of new s-values
    in_cnf_files : List[str]
        list of old cnf files
    out_cnf_dir : str
        path to output directory for cnf files
    in_opt_struct_cnf_dir : str
        input path to where cnf files are located
    in_num_states : int
        number of states
    out_cnf_prefix : str, optional
        prefix for output cnf files (default "REEDS_run")
    LRTO_mode : bool, optional
        add cnfs in LRTO-like fashion (default False)

    Returns
    -------
    List[str]
        list of output cnf files
    """

    # INPUT PARSING
    if (isinstance(in_cnf_files, List)):
        in_cnfs = in_cnf_files
        if (len(in_cnfs) == 0):
            raise IOError("Got empty list!: " + in_cnf_files)
    elif (isinstance(in_opt_struct_cnf_dir, str)):
        in_cnfs = glob.glob(in_opt_struct_cnf_dir + "/*cnf")
        if (len(in_cnfs) == 0):
            raise IOError("Could not find any cnf in: " + in_opt_struct_cnf_dir)
    else:
        raise ValueError("Please give either in_cnf_dir or in_cnf_files!")

    if (len(in_cnfs) != len(in_old_svals)):
        raise IOError("got different number of old svals and cnfs - no mapping possible!:\n\t svals: " + str(
            len(in_old_svals)) + ""
                                 "\n\t in_cnfs: " + str(
            len(in_cnfs)) + "")
    in_cnfs = list(sorted(in_cnfs, key=lambda x: int(x.split("_")[-1].replace(".cnf", ""))))

    # DECIDE ON HOW TO MAP
    if (len(in_old_svals) > len(in_new_svals)):
        print("too less ")
        out_cnfs = reduce_cnf_eoff(in_opt_struct_cnf_dir=in_opt_struct_cnf_dir, in_num_states=5,
                                   in_current_sim_cnf_dir=os.path.dirname(in_cnf_files[0]), in_old_svals=in_old_svals,
                                   in_new_svals=in_new_svals,
                                   out_next_cnfs_dir=out_cnf_dir, run_prefix=out_cnf_prefix,
                                   s1_repetitions=in_num_states)
    elif (len(in_old_svals) < len(in_new_svals)):
        if (LRTO_mode):
            out_cnfs = add_cnf_sopt_LRTOlike(in_dir=in_opt_struct_cnf_dir, out_dir=out_cnf_dir,
                                             in_old_svals=in_old_svals, in_new_svals=in_new_svals,
                                             cnf_prefix=out_cnf_prefix)
        else:
            out_cnfs = map_cnfs_to_svalues(in_old_cnf_files=in_cnfs, out_dir=out_cnf_dir, in_old_svals=in_old_svals,
                                           in_new_svals=in_new_svals,
                                           cnf_prefix=out_cnf_prefix)

    else:
        print("Number of CNF is fine!")
        out_cnfs = in_cnfs

    return out_cnfs


def map_cnfs_to_svalues(in_old_cnf_files: List[str],
                        out_dir: str,
                        in_old_svals: list,
                        in_new_svals: list,
                        cnf_prefix: str = "run",
                        replica_add_scheme: add_scheme = add_scheme.from_bothSides,
                        verbose: bool = True) -> List[str]:
    """map_cnfs_to_svalues
    
    Parameters
    ----------
    in_old_cnf_files : List[str]
        path to old cnf files
    out_dir : str
        path to output directory
    in_old_svals : list
        list of old s-values
    in_new_svals : list
        list of new s-values
    cnf_prefix : str, optional
        prefix for output cnf files (default "run")
    replica_add_scheme : add_scheme, optional
        enum for addition scheme from class adding_Scheme_new_Replicas (default from_bothSides)
    verbose : bool, optional
        verbose output (Default True)

    Returns
    -------
    List[str]
        list of new cnf files
    """
    if (type(replica_add_scheme) == type(None)):
        raise IOError("Please provide only one adding scheme for adding the new replicas.")

    new_cnf_files = []
    in_old_svals = np.array(list(map(float, in_old_svals)))
    in_new_svals = list(map(float, in_new_svals))[::-1]
    print("new(" + str(len(in_new_svals)) + "): ", in_new_svals)
    print("old(" + str(len(in_old_svals)) + "): ", list(in_old_svals))
    if (verbose): print("s value number difference (offfset): ", len(in_new_svals) - len(in_old_svals))

    out_prefix = out_dir + "/" + cnf_prefix + "_"
    if verbose: print("\nin_cnfs: " + str(in_old_cnf_files))
    if verbose: print("out_folder: " + out_dir)

    added_svals, offset = identify_closest_svalue(in_new_svals, in_old_svals, verbose, relocate_all=True)
    print("\n ==>add replicas at positions :", added_svals)

    # add cnfs if needed?
    if (verbose): print("\nAdding cnf files?")

    # mapping cnf to svals
    if (len(in_old_cnf_files) > len(in_old_svals)):  # error check
        raise Exception(
            "There were more cnf files found than old_svalues! - This means that the coordinate files may not sequentially belong to the svals!\n old_svals: " + str(
                in_old_svals) + "\n cnfs: " + str(in_old_cnf_files))
    else:
        fill_inds = map_old_cnf_files_to_new_svalues(added_svals, in_old_svals, offset, replica_add_scheme, verbose)
    print(" \n ==>Final cnf Mapping: ", fill_inds)

    # COPY
    print("\nCOPY CNFs in place: ")
    for old_cnf_indx, new_cnf_indx in fill_inds:
        new_cnf_name = out_prefix + str(new_cnf_indx + 1) + ".cnf"
        new_cnf_files.append(bash.copy_file(in_old_cnf_files[old_cnf_indx], new_cnf_name))
        print("\t{}\t -> \t{}".format(os.path.basename(in_old_cnf_files[old_cnf_indx]), os.path.basename(new_cnf_name)))

    if verbose: print("Done!\n\n")
    return new_cnf_files


def add_cnf_sopt_LRTOlike(in_dir: str,
                          out_dir: str,
                          in_old_svals: list,
                          in_new_svals: list,
                          cnf_prefix: str = "sopt_run",
                          replica_add_scheme: add_scheme = add_scheme.from_bothSides,
                          verbose: bool = True) -> List[str]:
    """add_cnf_sopt_LRTOlike
    
    Parameters
    ----------
    in_dir : str
        path where input cnf files are located
    out_dir : str
        path to output directory
    in_old_svals : list
        list of old s-values
    in_new_svals : list
        list of new s-values
    cnf_prefix : str, optional
        prefix for output cnf files (default "sopt_run")
    replica_add_scheme : add_scheme, optional
        enum for addition scheme from class adding_Scheme_new_Replicas (default from_bothSides)
    verbose : bool, optional
        verbose output (default True)

    Returns
    -------
    List[str]
        list of new cnf files
    """
    if (type(replica_add_scheme) == type(None)):
        raise IOError("Please provide only one adding scheme for adding the new replicas.")

    new_cnf_files = []
    in_old_svals = np.array(list(map(float, in_old_svals)))
    in_new_svals = list(map(float, in_new_svals))[::-1]
    print("new(" + str(len(in_new_svals)) + "): ", in_new_svals)
    print("old(" + str(len(in_old_svals)) + "): ", list(in_old_svals))
    if (verbose): print("s value number difference (offfset): ", len(in_new_svals) - len(in_old_svals))

    in_prefix = in_dir + "/" + cnf_prefix + "_"
    out_prefix = out_dir + "/" + cnf_prefix + "_"
    if verbose: print("\nin_folder: " + in_dir)
    if verbose: print("out_folder: " + out_dir)

    # find added Svals:
    if (len(in_new_svals) > len(in_old_svals)):
        added_svals, offset = identify_closest_svalue(in_new_svals, in_old_svals, verbose)
        print("\n ==>add replicas at positions :", added_svals)

        if (offset != len(in_new_svals) - len(in_old_svals)):
            raise Exception(
                "could not map new_svals to old_sval! probably an rounding error - checkout accuracy.\n Found " + str(
                    offset) + " insertion sites. Should be: " + str(len(in_new_svals) - len(in_old_svals)))

        # add cnfs if needed?
        if (verbose): print("\nAdding cnf files?")
        old_cnfs = list(sorted(glob.glob(in_dir + "/*cnf"), key=lambda x: int(x.split("_")[-1].replace(".cnf", ""))))
        num_old_cnf_files = len(old_cnfs)

        if (num_old_cnf_files > len(in_old_svals)):  # error check
            raise Exception(
                "There were more cnf files found than old_svalues! - This means that the coordinate files may not sequentially belong to the svals!\n old_svals: " + str(
                    in_old_svals) + "\n cnfs: " + str(old_cnfs))
        else:
            # map cnfs to new svals
            fill_inds = add_map_old_cnf_files_to_new_svalues(added_svals, in_old_svals, offset, replica_add_scheme,
                                                             verbose)

            print(" \n ==>Final cnf Mapping: ", fill_inds)

            # COPY
            print("\nCOPY CNFs in place: ")
            for old_cnf_indx, new_cnf_indx in fill_inds:
                new_cnf_name = out_prefix + str(new_cnf_indx + 1) + ".cnf"
                new_cnf_files.append(bash.copy_file(old_cnfs[old_cnf_indx], new_cnf_name))
                print("\t{}\t -> \t{}".format(os.path.basename(old_cnfs[old_cnf_indx]), os.path.basename(new_cnf_name)))

    if verbose: print("Done!\n\n")
    return new_cnf_files


def map_old_cnf_files_to_new_svalues(added_svals: list,
                                     in_old_svals: list,
                                     total_number_of_final_cnfs: int,
                                     replica_add_scheme: add_scheme,
                                     verbose: bool = True):

    """map_old_cnf_files_to_new_svalues
    
    Parameters
    ----------
    added_svals : list
        list of new s-values
    in_old_svals : list
        list of old s-values
    total_number_of_final_cnfs: int
        number of output cnf files
    replica_add_scheme : add_scheme
        enum for addition scheme from class adding_Scheme_new_Replicas
    verbose : bool, optional
        verbose output (default True)

    Returns
    -------
    List
    """
    # do insert into coordinate Files
    print("final_total_number_of_cnfs: ", total_number_of_final_cnfs)
    number_of_cnfs_left = total_number_of_final_cnfs

    if (verbose): print("\ts\tadd\t Sval\t\toffset\t addingScheme")
    # determine cnf coordinate mapping from old s values to new s value structures
    fill_inds = []
    for s in sorted(added_svals)[::-1]:
        number_of_cnfs_left -= added_svals[s]
        if (verbose): print(
            "\t{}\t{}\t{:>5.5f}\t\t{}\t".format(s, added_svals[s], in_old_svals[s], number_of_cnfs_left), end="")

        # fill gap:
        if (added_svals[s] > 0):
            # Assign filling direction
            if (sorted(added_svals)[-1] != s and (replica_add_scheme == add_scheme.from_below or
                                                  (replica_add_scheme == add_scheme.from_bothSides and added_svals[
                                                      s] == 1))):  # check your not at bottom and this is the adding scheme from bottom or  both sides with only one new replica
                if (verbose): print("\tadd from below", end=" ")
                fill_inds.extend(
                    [(s + 1, number_of_cnfs_left + (added_svals[s] - x)) for x in range(1, added_svals[s] + 1)])

            elif (sorted(added_svals)[
                      -1] != s and replica_add_scheme == add_scheme.from_bothSides):  # check your not at bottom and this is the adding scheme
                half_range = int(np.round(0.5 * added_svals[s]))

                add_svals_coords_from_below = [(s + 1, number_of_cnfs_left - 1 + (added_svals[s] - x)) for x in
                                               range(half_range)]  # add half coordinates from below

                add_svals_coords_from_top = [(s, number_of_cnfs_left - 1 + (added_svals[s] - x)) for x in
                                             range(half_range, added_svals[s])]  # add half coordinates from top
                fill_inds.extend(add_svals_coords_from_below + add_svals_coords_from_top)

                if (verbose):
                    print("\tadd from bothSides: \tHalf", half_range, end="\t")
                    print("\tfrom bot: " + str(len(add_svals_coords_from_below)), add_svals_coords_from_below, end="\t")
                    print("\ttop: " + str(len(add_svals_coords_from_top)), add_svals_coords_from_top, end="")
            else:
                if (verbose): print("\tadd from top", end=" ")
                fill_inds.extend(
                    [(s, number_of_cnfs_left + (added_svals[s] - x)) for x in range(1, added_svals[s] + 1)])

        if (verbose): print()
    return fill_inds


def add_map_old_cnf_files_to_new_svalues(added_svals: list,
                                         in_old_svals: list,
                                         offset: int,
                                         replica_add_scheme : add_scheme,
                                         verbose: bool = True):
    """add_map_old_cnf_files_to_new_svalues

    Parameters
    ----------
    added_svals : list
        list of new s-values
    in_old_svals : list
        list of old s-values
    offset: int
        offset to total amount of cnfs
    replica_add_scheme : add_scheme
        enum for addition scheme from class adding_Scheme_new_Replicas
    verbose : bool, optional
        verbose output (default True)

    Returns
    -------
    List
    """
    # do insert into coordinate Files
    if (verbose): print("\ts\tadd\t Sval\t\toffset\t addingScheme")
    # determine cnf coordinate mapping from old s values to new s value structures
    fill_inds = []
    for s in sorted(added_svals)[::-1]:
        offset -= added_svals[s]
        if (verbose): print("\t{}\t{}\t{:>5.5f}\t\t{}\t".format(s, added_svals[s], in_old_svals[s], offset), end="")

        # fill gap:
        if (added_svals[s] > 0):
            # Assign filling direction
            if (sorted(added_svals)[-1] != s and (replica_add_scheme == add_scheme.from_below or
                                                  (replica_add_scheme == add_scheme.from_bothSides and added_svals[
                                                      s] == 1))):  # check your not at bottom and this is the adding scheme from bottom or  both sides with only one new replica
                if (verbose): print("\tadd from below", end=" ")
                fill_inds.extend([(s + 1, s + 1 + offset + x) for x in range(added_svals[s])])

            elif (sorted(added_svals)[
                      -1] != s and replica_add_scheme == add_scheme.from_bothSides):  # check your not at bottom and this is the adding scheme
                half_range = int(np.round(0.5 * added_svals[s]))

                add_svals_coords_from_below = [(s + 1, s + offset + (added_svals[s] - x)) for x in
                                               range(half_range)]  # add half coordinates from below
                add_svals_coords_from_top = [(s, s + offset + (added_svals[s] - x)) for x in
                                             range(half_range, added_svals[s])]  # add half coordinates from top
                fill_inds.extend(add_svals_coords_from_below + add_svals_coords_from_top)

                if (verbose):
                    print("\tadd from bothSides: \tHalf", half_range, end="\t")
                    print("\tfrom bot: " + str(len(add_svals_coords_from_below)), add_svals_coords_from_below, end="\t")
                    print("\ttop: " + str(len(add_svals_coords_from_top)), add_svals_coords_from_top, end="")
            else:
                if (verbose): print("\tadd from top", end=" ")
                fill_inds.extend([(s, s + 1 + offset) for x in range(1, added_svals[s] + 1)])

        fill_inds.append((s, s + offset))

        if (verbose): print()
    return fill_inds


def identify_closest_svalue(in_new_svals: list,
                            in_old_svals: list,
                            verbose: bool = True,
                            relocate_all: bool = False):
    """identify_closest_svalue

    Parameters
    ----------
    in_new_svals : list
        list of new s-values
    in_old_svals : list
        list of old s-values
    relocate_all : bool, optional
        default False
    verbose : bool, optional
        verbose output (default True)

    Returns
    -------
    dict
        added s-values
    int
        offset

    """
    added_svals = {key: 0 for key in range(len(in_old_svals))}
    if verbose: print("\n\tCheckout, where new Values were added: ")
    if (verbose):     print("\tnew_s\t\t\tnearestNum\tnearNumIDX")
    for ind, s_new in enumerate(in_new_svals):
        # determine needed accuracy

        nearest_s_idx = len(in_old_svals) - np.searchsorted(in_old_svals[::-1], s_new) - 1
        nearest_s = in_old_svals[nearest_s_idx]

        significant_decimals = 10 ** -(
                int(round(max([abs(np.log10(nearest_s)), abs(np.log10(s_new))]))) + 3)  # dynamic accuracy
        if (np.isclose(nearest_s, s_new, rtol=significant_decimals)):
            if (verbose):     print(
                "\t{:8.5f}\t\t{:8.5f}\t{:8>}\t".format(round(s_new, 5), round(nearest_s, 5), nearest_s_idx))
            if (relocate_all):
                added_svals.update({nearest_s_idx: added_svals[nearest_s_idx] + 1})
            continue
        else:
            if (verbose):     print(
                "NEW\t{:8.5f}\t\t{:8.5f}\t{:8>}\t".format(round(s_new, 5), round(nearest_s, 5), nearest_s_idx))
            added_svals.update({nearest_s_idx: added_svals[nearest_s_idx] + 1})

    offset = sum([added_svals[s] for s in added_svals])  # offset to total amount of cnfs
    return added_svals, offset


def reduce_cnf_eoff(in_num_states: int, in_opt_struct_cnf_dir: str, in_current_sim_cnf_dir: str,
                    in_old_svals: List[Number],
                    in_new_svals: List[Number], out_next_cnfs_dir: str, run_prefix="sopt_run",
                    s1_repetitions: int = None,
                    s_repeating_value: float = 1.0, verbose=False) -> list:
    """reduce_cnf_eoff
    This function should facilitate the transition from Eoff -> sopt.
    Todo: think about adding replicas and also s_values dependent cnf
    selection. bschroed

    Parameters
    ----------
    in_num_states : int
        int which gives the number of states
    in_opt_struct_cnf_dir : str
        path to the optimized eds state .cnfs(each state
        needs to be present - depends on in_num_states)
    in_current_sim_cnf_dir : str
        the initial .cnf coordinates for the rest of the
        replicas
    in_old_svals : List[Number]
        list of old s-values
    in_new_svals : List[number]
        list of new s-values
    out_next_cnfs_dir : str
        where to store the resulting cnfs
    run_prefix : str, optional
        prefix_name for the var (default "sopt_run")
    s1_repetitions : int, optional
        how many repetitions of s1? (default None)
    s_repeating_value : float, optional
        value of s1 (default 1.0)
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    List[str]
        returns the paths of the list of reduced cnfs
    """
    collected_files = []
    out_num_svals = len(in_new_svals)
    in_num_svals = len(in_old_svals)
    print(s1_repetitions)
    if (not s1_repetitions and in_new_svals.count(s_repeating_value) == in_num_states):
        s1_repetitions = in_num_states
    elif (not s1_repetitions):
        raise IOError("please specify any state repetitions")
    if (in_num_svals <= out_num_svals - s1_repetitions):
        raise IOError("There are more new s values than old ones! This method only reduces the range of s-values.\n"
                      "\tOld = " + str(in_num_svals) + "\t New= " + str(out_num_svals) + " (" + str(
            s1_repetitions) + " s1Repetitions + " + str(
            out_num_svals - in_num_states) + " samplingSvals)\n"
                                             "\tOLD S-vals: " + str(in_old_svals) + "\n"
                                                                                    "\tNew S-vals: " + str(
            in_new_svals) + "\n")

    # general information
    print("\n\tGENERAL INFO")
    print("\t\tnum states: ", in_num_states, "out_num svals: ", out_num_svals, "in_num_svals: ", in_num_svals)
    print("\t\tpicks :", out_num_svals - in_num_states)

    # get optimized structs cnfs:
    print("\n\tGET in_opt_struct_cnf_dir cnfs")
    print("\t\topt, PATH: ", in_opt_struct_cnf_dir + "/*cnf")
    print("\t\tcur_path = ", os.getcwd())

    found_cnfs = glob.glob(in_opt_struct_cnf_dir + "/*cnf")
    if (verbose): print(found_cnfs)
    print()
    if (len(found_cnfs) == 0):
        raise IOError("Could not find any cnf file for next dir in: " + in_opt_struct_cnf_dir + "/*cnf")

    cnf_prefix = os.path.basename(found_cnfs[0]).replace(".cnf", "")
    cnf_prefix = cnf_prefix[:-1]
    if verbose: print("\told Opt Prefix:\t", cnf_prefix)

    if (verbose): print("NUmber of optimized states: ", in_num_states)
    for state in range(1, in_num_states + 1):
        print("\t", cnf_prefix + str(state) + ".cnf\t-->\t" + out_next_cnfs_dir + "/" + run_prefix + "_" + str(
            state) + ".cnf")
        collected_files.append(bash.copy_file(in_opt_struct_cnf_dir + "/" + cnf_prefix + str(state) + ".cnf",
                                              out_next_cnfs_dir + "/" + run_prefix + "_" + str(state) + ".cnf"))

    # get s_equil states cnfs:
    print("\n\tGET in_current_sim_cnf_dir cnfs")
    ##coherent sorting:
    simulation_result_cnf = list(
        sorted(glob.glob(in_current_sim_cnf_dir + "/*cnf"), key=lambda x: int(x.split("_")[-1].replace(".cnf", ""))))
    in_new_svals = np.array(list(sorted(filter(lambda x: x != s_repeating_value, map(float, in_new_svals)))))
    in_old_svals = np.array(list(sorted(map(float, in_old_svals))))
    if verbose: print(simulation_result_cnf)

    ##neW
    if (len(in_new_svals) > len(simulation_result_cnf)):
        raise IOError("Found less simulated result cnf files than there were old s values?\n"
                      "\tsvals(" + str(len(in_old_svals)) + "): " + str(in_old_svals) + "\n"
                                                                                        "\tcnfs(" + str(
            len(simulation_result_cnf)) + "): " + str(
            simulation_result_cnf) + "\n")

    print("\ts\t\t\tnearestNum\tnearNumIDX\tused_cnf -> copied to")

    # copy cnf with the closest old s_value (lleft oriented) for a new s_value
    collected_files = []

    for new_index, new_s in enumerate(in_new_svals):
        closest_s_idx = len(in_old_svals) - np.searchsorted(in_old_svals, new_s)
        closest_s = in_old_svals[-closest_s_idx]
        tmp_cnf_path = simulation_result_cnf[closest_s_idx - 1]
        tmp_new_cnf_path = out_next_cnfs_dir + "/" + run_prefix + "_" + str(
            len(in_new_svals) - new_index + in_num_states) + ".cnf"
        collected_files.append(bash.copy_file(tmp_cnf_path, tmp_new_cnf_path))

        print("\t{:8.5f}\t\t{:8.5f}\t{:8>}\t\t{}".format(round(new_s, 5), round(closest_s, 5), closest_s_idx,
                                                         tmp_cnf_path + "  ->  " + tmp_new_cnf_path))

    return collected_files
