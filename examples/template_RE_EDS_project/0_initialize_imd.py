#!/usr/bin/env python3
from reeds.function_libs.pipeline.module_functions import initialize_imd
from global_definitions import fM
from global_definitions import in_top_file, in_cnf_file, in_pert_file, in_posres_file, in_refpos_file, in_disres_file, name

topology_state_opt = fM.Topology(top_path=in_top_file, perturbation_path=in_pert_file, disres_path=in_disres_file, posres_path=in_posres_file, refpos_path=in_refpos_file)
system = fM.System(coordinates=in_cnf_file, name=name, top=topology_state_opt)

initialize_imd(system=system, imd_out_path="input/template.imd")