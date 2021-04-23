"""
Nasty script ;)

"""
# !/usr/bin/env python
import os

from pygromos.utils import bash
from reeds.function_libs.file_management import file_management

template_control_dict = {
    "concat": {"do": True,
               "sub": {
                   "cp_cnf": True,
                   "cat_trc": False,
                   "cat_tre": False,
                   "ene_ana": True,
                   "convert_trcs": False,
                   "cat_repdat": False, }
               }
}

def do(in_simulation_name: str, in_simulation_dir: str, in_topology_path: str, in_imd_path: str,
       out_analysis_dir: str,
       in_ene_ana_lib_path: str, gromosPP_bin_dir: str = None, n_processes: int = 1,
       trc_take_every_step: int = 1, control_dict: dict = None, verbose: bool = True):
    """
            this function is concatenating simulatoin files of normal MD runs
        #Todo: move to PyGromos

    Parameters
    ----------
    in_simulation_name : str
        name of the simulation approach
    in_simulation_dir : str
        path to the simulation dir, containing the gromos simulation files
    in_topology_path : str
        path to the topology
    in_imd_path : str
        path to the gromos parameter file
    out_analysis_dir : str
        path to the out dir
    in_ene_ana_lib_path : str
        path to the ene_ana library
    gromosPP_bin_dir : str
        path to the gromos PP binary
    n_processes : int
        number of processors
    trc_take_every_step : int
        use every n step
    control_dict : dict
        which analysis steps shall be performed
    verbose : bool
        Kaching!


    """

    if (not os.path.exists(out_analysis_dir)):
        bash.make_folder(out_analysis_dir)
    if (not isinstance(control_dict, dict)):
        control_dict = template_control_dict

    ##################################################
    #   ORGANIZE FILES
    ##################################################
    # DO
    if (not os.path.isdir(out_analysis_dir)):
        raise IOError("Could not find: \n\t", out_analysis_dir)

    out_data_dir = out_analysis_dir + "/data"
    if (not os.path.isdir(out_data_dir)):
        os.mkdir(out_data_dir)

    properties = ["totpot", "ligangle", "ligtors", "ligimptors", "lignonbself", "lignonbenv"]
    file_management.project_concatenation(in_folder=in_simulation_dir, in_topology_path=in_topology_path,
                                          in_imd=in_imd_path,
                                          num_replicas=1, control_dict=control_dict["concat"]["sub"],
                                          out_folder=out_data_dir, out_file_prefix=in_simulation_name,
                                          in_ene_ana_lib_path=in_ene_ana_lib_path, n_processes=n_processes,
                                          gromosPP_bin_dir=gromosPP_bin_dir,
                                          nofinal=False, verbose=verbose, additional_properties=properties)
