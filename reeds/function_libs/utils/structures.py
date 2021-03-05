"""
In this file some general structures are defined, that are used throughout the pipeline.

named tuples for jo generation of program description.

"""
from collections import namedtuple
from enum import Enum

spacer = "////////////////////////////////////////////////////////////////////////////////////////////////////////////"


class adding_Scheme_new_Replicas(Enum):
    from_above = 1
    from_below = 2
    from_bothSides = 3

sopt_job = namedtuple("sopt_Job", ["iteration", "job_file_path", "job_analysis_path", "check_simulation_files",
                                   "check_analysis_files", "sim_system",
                                   "nmpi", "num_simulation_runs", "num_equilibration_runs", "workdir", "in_imd_path",
                                   "out_folder", "last_coord_in"])
soptimization_params = namedtuple("soptimization_params",
                                  ["add_replicas", "adding_new_sReplicas_Scheme", "current_num_svals"])
additional_argparse_argument: namedtuple = namedtuple("specialArgument", ['name', 'type', 'required', 'desc'])
