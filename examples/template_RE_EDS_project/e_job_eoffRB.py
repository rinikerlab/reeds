#!/usr/bin/env python3
import os, sys, glob
from reeds.modules import do_RE_EDS_eoffRebalancing as eoffRB

sys.path.append(os.getcwd())


from global_definitions import fM, bash
from global_definitions import name, root_dir
from global_definitions import gromosXX_bin, gromosPP_bin, ene_ana_lib
from global_definitions import in_top_file, in_pert_file, in_disres_file
from global_definitions import undersampling_frac_thresh

#paths
in_name = name+"_eoffRB"
out_sopt_dir = root_dir+"/e_"+in_name
next_sopt_dir = root_dir+"/input/2_next_sopt"

out_sopt_dir = bash.make_folder(out_sopt_dir)


#In-Files
topology = fM.Topology(top_path=in_top_file, disres_path=in_disres_file, pertubation_path=in_pert_file)
system =fM.System(coordinates=glob.glob(next_sopt_dir+"/*cnf"),name=in_name,    top=topology)

#Additional Options

job_duration="120:00"
nmpi_per_replica = 6
iterations=4
learningFactors = None
individualCorrection = False
pseudocount = (1/5)/10


last_jobID = eoffRB.do(out_root_dir=out_sopt_dir,in_simSystem=system,
    in_ene_ana_lib_path=ene_ana_lib, nmpi_per_replica=nmpi_per_replica,
    duration_per_job = job_duration, iterations=4, learningFactors=learningFactors, individualCorrection=individualCorrection,
    verbose= True, )


