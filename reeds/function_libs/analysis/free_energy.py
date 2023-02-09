import json, os, copy
from typing import List, Dict

import numpy as np
import pandas as pd
from pygromos.gromos import gromosPP
from pygromos.utils import bash

from reeds.function_libs.visualization.free_energy_plots import plot_mbar_convergence, plot_dF_conv

from scipy import constants as const
from scipy.special import logsumexp


def multi_lig_dfmult(num_replicas: int,
                     num_states: int,
                     in_folder,
                     out_file,
                     eR_prefix: str = "eR_s",
                     eX_prefix: str = "eY_s",
                     temp: int = 298,
                     gromos_binary="dfmult",
                     verbose=True) -> dict:
    """multi_lig_dfmult
        @ WARNING GOING TO BE REMOVED - OLD Function@
        This function is calculating dfmult with multiple ligands in a system

    Parameters
    ----------
    num_replicas :
    num_states :
    in_folder :
    out_file :
    eR_prefix :
    eX_prefix :
    temp :
    gromos_binary :
    verbose :

    Returns
    -------
    None
    """

    def gen_results_string(results_dict: dict) -> str:
        # This func generates an readable output
        # generate out string:
        result_string = ""
        float_format = "{: > 13.3f}"
        str_format = "{:>13}"
        str_head_format = "{:^27}"
        state = ["lig" + str(state) for state in range(1, num_states + 1)]
        state.append("ligR")

        for s in results_dict:
            first = True
            s_results_dict = results_dict[s]
            print("result_keys: " + str(list(s_results_dict.keys())))
            result_string += "\n dF for s= " + str(s) + "\n"
            for l1 in sorted(s_results_dict):

                # Header
                if (first):
                    results = []
                    format_list = []
                    for l2 in sorted(s_results_dict["lig" + str(num_states)]):
                        results.append(l2)
                        format_list.append(str_head_format)
                    formatting = "{: <10} " + " | ".join(format_list) + "\n"
                    result_string += formatting.format("ligname", *results)

                    results = []
                    format_list = []
                    for _ in state:
                        results.append("mean")
                        results.append("err")
                        format_list.extend([str_format + " " + str_format])
                    formatting = "{: <10} " + " | ".join(format_list) + "\n"
                    result_string += formatting.format("ligname", *results)
                    first = False

                results = []
                format_list = []
                for l2 in state:
                    if (l2 in s_results_dict[l1]):
                        results.append(float(s_results_dict[l1][l2]["mean"]))
                        results.append(float(s_results_dict[l1][l2]["err"]))
                        format_list.extend([float_format + " " + float_format])
                    else:
                        results.append(str(" - "))
                        results.append(str(" - "))
                        format_list.extend([str_format + " " + str_format])

                formatting = "{: <10} " + " | ".join(format_list) + "\n"
                result_string += formatting.format(l1, *results)

            result_string += "\n"
        result_string += "\n"
        return result_string

    # DO PER REPLICA
    grom = gromosPP.GromosPP()
    results_dict = {}
    dfmult_replica_results = []
    for s in range(1, num_replicas + 1):
        # reference Hamiltonian Potential file at s
        results_dict.update({s: {}})
        eR_file = in_folder + "/" + eR_prefix + str(s) + ".dat"

        # find Potentials of each state
        ei_files = []
        for i in range(1, num_states + 1):
            ei_files.append(in_folder + "/" + eX_prefix.replace("Y", str(i)) + str(s) + ".dat ")
        print("\n", str(s), "\n")

        tmp_file_path = grom.dfmult(ei_files, in_reference_state_file_path=eR_file, temperature=temp,
                                    _binary_name=gromos_binary,
                                    out_file_path=os.path.dirname(out_file) + "/tmp_" + str(s) + ".out")

        # parse again file for nicer
        tmp_file = open(tmp_file_path, "r")
        for line in tmp_file.readlines():

            fields = line.split()
            keyfields = fields[0].split("_")
            dF_mean = fields[1]
            dF_err = fields[2]
            if (len(fields) < 3 or len(keyfields) < 3):
                continue
            else:
                l1 = "lig" + str(keyfields[1])
                if ("lig" + str(keyfields[1]) in results_dict[s]):
                    results_dict[s][l1].update({"lig" + str(keyfields[2]): {"mean": dF_mean, "err": dF_err}})
                else:
                    results_dict[s].update({l1: {"lig" + str(keyfields[2]): {"mean": dF_mean, "err": dF_err}}})
        tmp_file.close()
        dfmult_replica_results.append(tmp_file_path)

    # FINAL write out
    result_file = open(out_file, "w")
    result_file.write(gen_results_string(results_dict))
    result_file.close()

    for file_path in dfmult_replica_results:
        bash.remove_file(file_path)
    return results_dict


def free_energy_convergence_analysis(ene_trajs: List[pd.DataFrame],
                                     out_dir: str,
                                     in_prefix: str = "",
                                     out_prefix: str = "energy_convergence",
                                     dfmult_all_replicas: bool = True,
                                     time_blocks: int = 10,
                                     recalculate_all: bool = False,
                                     verbose: bool = False):
    """free_energy_convergence
    This function calculates and visualizes the convergence of the free energy calculation

    Parameters
    ----------
    ene_trajs:  List[pd.DataFrame]
        a list of pandas dataframes containing the energy data of each state eX
    out_dir : str
        path where the plots and data should be stored
    in_prefix : str, optional
        prefix for the input files (default "")
    out_prefix : str, optional
        prefix for the output files (default "energy_convergence")
    dfmult_all_replicas : bool, optional
        calculate free energy differences for all replicas or only s = 1 (default False)
    time_blocks : int, optional
        number of time blocks for the convergence visualization (default 10)
    recalculate_all : bool, optional
        recalculate or use old summary file (default False)
    verbose: bool, optional
        verbose output (default False)

    Returns
    -------
    None
    """

    if (verbose): print("start dfmult")
    e_trajs = {int(ene_ana_traj.s.replace("s", "")): ene_ana_traj for ene_ana_traj in ene_trajs}

    # DF Convergence
    if (verbose): print("Calc conf")
    dF_conv_all_replicas = {}

    # generate time conevergence data
    summary_file_path = out_dir + "/summary_allReplicas_dfmult_timeblocks_" + out_prefix + ".dat"
    if (False and os.path.exists(summary_file_path) and not recalculate_all):
        if (verbose): print("Found summary file\t LOAD: " + summary_file_path)
        dF_conv_all_replicas = json.load(open(summary_file_path, "rb"))
    else:

        if (verbose): print("CALCULATE")
        if dfmult_all_replicas:
            svals_for_analysis = sorted(e_trajs)
        else:
            svals_for_analysis = [1]  # just do s = 1.

        for s_index in svals_for_analysis:
            replica_key = "replica_" + str(s_index)
            ene_traj = e_trajs[s_index]
            if (verbose): print("\n\nREPLICA: ", s_index)

            # CALC_ Free Energy
            dF_time = eds_dF_time_convergence_dfmult(ene_traj=ene_traj, out_dir=out_dir, time_blocks=time_blocks,
                                                     verbose=verbose)
            dF_conv_all_replicas.update({replica_key: dF_time})

            # Plotting
            if (verbose): print("Plotting")
            updated_keys = {str(replica_key) + "_" + str(key): value for key, value in
                            dF_conv_all_replicas[replica_key].items()}
            plot_dF_conv(updated_keys, title="Free energy convergence", out_path=out_dir + "/" + out_prefix + "_" + str(replica_key), show_legend=True)
            json.dump(dF_conv_all_replicas, fp=open(out_dir + "/tmp_dF_" + str(s_index) + ".dat", "w"), indent=4)

        json.dump(dF_conv_all_replicas, fp=open(summary_file_path, "w"), indent=4)

    # nice_result file
    out_dfmult = open(out_dir + "/free_energy_result.dat", "w")
    out_string = gen_results_string(dF_conv_all_replicas)
    out_dfmult.write(out_string)
    out_dfmult.close()


def eds_dF_time_convergence_dfmult(ene_traj: pd.DataFrame,
                                   out_dir: str,
                                   time_blocks: int = 10,
                                   gromos_bindir: str = None,
                                   verbose: bool = False) -> Dict:
    """eds_dF_time_convergence_dfmult
    This function generates a dictionary, which contains time-dependent dF values.

    Parameters
    ----------
    ene_traj : pd.DataFrame
        pandas dataframe containing energy trajectories
    out_dir : str
        path for the output directory
    time_blocks : int, optional
        number of time blocks for the free energy convergence calculation (default 10)
    gromos_bindir : str, optional
        path to gromos binaries (default None)
    verbose : bool, optional
        verbose output (default False)

    Returns
    -------
    dF_timewise : Dict
        dict containing free energies after different time lengths
    """
    # Split the data timewise!

    if (not os.path.exists(out_dir)):
        os.mkdir(out_dir)
    traj_length = ene_traj["time"].size
    if (verbose): print("traj length: " + str(traj_length))

    step_size = traj_length // time_blocks
    if (verbose): print("step size: " + str(step_size))
    data_chunk_indices = list(range(step_size, traj_length + step_size, step_size))
    data_chunk_indices.insert(0, 10)  # add a very small time interval at the beginning

    if (verbose): print("\nGATHER ", time_blocks, "time _blocks in steps of ", step_size, "\tleading to final: ",
                        ene_traj["time"][traj_length - 1], "\n")
    if (verbose): print("data chunks: ")
    if (verbose): print(data_chunk_indices)

    # STEP1: generate files with blocked data
    properties_timesolv = {}
    for property in sorted(filter(lambda x: x.startswith("e"), ene_traj.columns)):
        # chunk data
        if (verbose): print("\t", property)
        sub_ste = []
        for step, end_step in enumerate(data_chunk_indices):
            if (verbose): print(step, " ", end_step, end="\t")
            out_path = out_dir + "/tmp_" + property + "_step" + str(step) + ".dat"
            ene_traj[["time", property]][:end_step].to_csv(out_path, sep="\t", header=["# time", property],
                                                              index=False)
            sub_ste.append(out_path)

        if (end_step != traj_length):
            if (verbose): print(step + 1, " ", end_step, end="\t")
            out_path = out_dir + "/tmp_" + property + "_step" + str(step + 1) + ".dat"
            ene_traj[["time", property]].to_csv(out_path, sep="\t", header=["# time", property], index=False)
            sub_ste.append(out_path)
            bash.wait_for_fileSystem(out_path)
        if (verbose): print()
        properties_timesolv.update({property: sub_ste})

        # check if lsf is has the result files already.
        bash.wait_for_fileSystem(sub_ste)

    # STEP2: calc dF timedep - with generated files from above
    if (verbose): print("\n\ncalc dF with dfmult")
    gromPP = gromosPP.GromosPP(gromos_bindir)
    dF_timewise = {}
    for step, stop_index in enumerate(data_chunk_indices):
        Vr = properties_timesolv["eR"][step]
        Vi = list(
            sorted([value[step] for key, value in properties_timesolv.items() if (key != "eR" and key.startswith("e"))],
                   key=lambda x: int(os.path.basename(x).split("_")[1].replace("e", ""))))

        tmp_out = out_dir + "/tmp_dfmult_" + str(step) + ".dat"
        if (verbose): print("Dfmult: ", step)
        tmp_out = gromPP.dfmult(in_endstate_file_paths=Vi, in_reference_state_file_path=Vr, out_file_path=tmp_out,
                                verbose=verbose)

        out_tmp = open(tmp_out, "r")
        lines = out_tmp.readlines()

        for line in lines:
            if ("DF_" in line):
                print(line)
                ind_1, ind_2 = line.strip().split("  ")[0].replace("DF_", "").split("_")
                key_v = str(ind_1) + "_" + str(ind_2)
                if (not key_v in dF_timewise):
                    dF_timewise.update({key_v: {}})
                h, meanDF, errDF = line.split()
                dF_timewise[key_v].update(
                    {float(ene_traj["time"][stop_index - 1]): {"mean": float(meanDF), "err": float(errDF)}})

        # STEP 3: CLEAN
        Vi.append(Vr)
        for tmp_path in Vi:
            os.remove(tmp_path)
        out_tmp.close()
        os.remove(tmp_out)

    return dF_timewise


def gen_results_string(results_dict: dict) -> str:
    """gen_results_string
    This function generates a string for the output of the free energy calculations

    Parameters
    ----------
    results_dict : dict
        dict containing the free energy calculations

    Returns
    -------
    str
        nicely formated string for the output of the free energy calculations
    """
    result_string = ""
    float_format = "{: > 13.3f}"
    str_format = "{:>13}"
    str_head_format = "{:^27}"
    states = []
    for s in sorted(results_dict, key=lambda x: int(x.split("_")[1])):
        first = True
        s_results_dict = results_dict[s]
        result_string += "\n dF for " + str(s) + "\n\n"

        # translate dict
        file_dict = {}
        for ligand_pair in s_results_dict:
            l1 = ligand_pair.split("_")[0]
            l2 = ligand_pair.split("_")[1]
            states.append(l1)
            states.append(l2)
            if (not l1 in file_dict):
                file_dict.update({l1: {l2: {"mean": "-", "err": "-"}}})
            else:
                file_dict[l1].update({l2: {"mean": "-", "err": "-"}})

            last_step = list(sorted(s_results_dict[ligand_pair].keys()))[-1]
            file_dict[l1][l2].update({"mean": float(s_results_dict[ligand_pair][last_step]["mean"]),
                                      "err": float(s_results_dict[ligand_pair][last_step]["err"])})

        # fill up missing states comparison
        states = sorted(list(set(states)))
        for l1 in states:
            for l2 in states:
                if (not l1 in file_dict):
                    file_dict.update({l1: {l2: {"mean": "-", "err": "-"}}})
                elif (l1 in file_dict and not l2 in file_dict[l1]):
                    file_dict[l1].update({l2: {"mean": "-", "err": "-"}})

        # write files

        l1_ordered_states = list(sorted(filter(lambda x: not "R" in x, list(file_dict.keys())), key=lambda x: int(x)))
        for l1 in l1_ordered_states:
            # Header
            l2_ordered_states = list(
                sorted(filter(lambda x: not "R" in x, list(file_dict[l1].keys())), key=lambda x: int(x)))
            l2_ordered_states.append("R")
            if (first):
                results = []
                format_list = []
                # ligandname line
                for l2 in l2_ordered_states:
                    results.append(l2)
                    format_list.append(str_head_format)
                formatting = "| {: <10} | " + " | | ".join(format_list) + "|\n"
                result_string += formatting.format("ligname", *results)

                # column description line
                results = []
                format_list = []
                for _ in l2_ordered_states:
                    results.append("mean")
                    results.append("err")
                    format_list.extend([str_format + " " + str_format])
                formatting = "| {: <10} |" + " | ".join(format_list) + "|\n"
                result_string += formatting.format("ligname", *results)
                result_string += "|---" * (len(results) + 1) + "|\n"

                first = False

            results = []
            format_list = []
            for l2 in l2_ordered_states:
                if (file_dict[l1][l2]["mean"] != "-"):
                    results.append(float(file_dict[l1][l2]["mean"]))
                    results.append(float(file_dict[l1][l2]["err"]))
                    format_list.extend([float_format + " " + float_format])
                else:
                    results.append(str(" - "))
                    results.append(str(" - "))
                    format_list.extend([str_format + " " + str_format])
            formatting = "| {: <10} " + " | ".join(format_list) + "|\n"
            result_string += formatting.format(l1, *results)
        result_string += "\n"
    result_string += "\n"
    return result_string

def reformat_trajs_for_mbar(ene_trajs, s_values, eoffs, temp, l = 1, with_decorrelation=True):
    """
    Reformats a gromos energy trajectory to fit the expected input of pymbar (u_kln)
    
    This function re-evaluates the reference potential using the reference potential 
    parameters of every other replica (i.e. using different values for s and eoffs)

    Note: If l = 1, this applying M-BAR in this way is equivalent to using the Zwanzig equation (information from other replicas are not used).
    # we could potentially remove completely dependence on dfmult and calculate all free energies with MBAR (faster) in future if code is stable

    Parameters
    ----------
    ene_trajs: List [pd.DataFrame] 
        contains all replicas potential energies 
    s_values: List[float]
        the set of s-values used in the simulation which generated this data
    eoffs: List[List[float]]
        the set of energy offsets used in the simulation which generated this data (for all s values)
    temp: float
        temperature of the simulation
    l: int
        number of replicas to include in the MBAR calculation, if 1 equivalent to Zwanzig eqn.
    with_decorrelation: bool
        determines if the input timeseries are subsampled to remove correlated data-points.
        with M-BAR input timeseries should be decorellated. 

    Returns
    -------
        u_kn, N_k 
        reduced potential energies for all thermodynamic states
        number of samples for each state 
        this output can be given directly to pymbar
    
    """
    
    from pymbar.timeseries import subsample_correlated_data

    kt = (temp * const.k * const.Avogadro) / 1000
    beta =  1 / kt
    
    num_states = len(eoffs[0])
    end_states = [f'e{i+1}' for i in range(num_states)]
    
    # a little bit ugly we do work twice here
    if with_decorrelation:
        n = [len(subsample_correlated_data(traj['eR'])) for traj in ene_trajs[:l]]
        idx_ns = np.append([0], np.cumsum(n))
    else:
        n = [len(ene_trajs[0]['eR'])] * l
        idx_ns = np.append([0], np.cumsum(n))

    k_tot = num_states + l # we will always have l additional states (all different Vrs)
    
    u_kn = np.zeros([k_tot, np.sum(n)]) # energies evaluated at all states k for all n samples
    N_k = np.zeros(k_tot) # number of samples from states k
    
    for i, traj in enumerate(ene_trajs):
        if i == l:
            break

        beg = idx_ns[i]
        end = idx_ns[i+1]
        
        # Reformat the data
        vr = np.array(traj['eR'])

        if with_decorrelation:
            idx_subsample = subsample_correlated_data(vr)
            
            vr = vr[idx_subsample]
            vis = np.array(traj[end_states])[idx_subsample]
        else:
            vis = np.array(traj[end_states])      

        # Add the potential energies of the end states
        for k, vk in enumerate(vis.T): 
            u_kn[k, beg:end] = vk
        
        # Add the potential energies of the reference states at simulated 
        for k in range(num_states, num_states+l):
            idx_params = (k-num_states)
            if idx_params == i:
                u_kn[k, beg:end] = vr
            else: #recalc ref potential at all other values of s-values/eoffs
                s = s_values[idx_params]
                _eoffs = eoffs[idx_params]                
                expterm =  - (beta*s) * np.subtract(vis,  _eoffs).T
                u_kn[k, beg:end] = -1/(beta*s) * logsumexp(expterm, axis=0)
            
        N_k[i+num_states] = len(vr)
    
    # Convert to reduced potential energies
    u_kn *= beta
    
    return u_kn, N_k 

def calc_free_energies_with_mbar(ene_trajs, s_values, eoffs, out_dir, temp=298, num_replicas=1, ) -> None:
    """
    Calculate the free energies between all end states and the uppermost reference state (i.e. s = 1) by using 
    information from more than 1 replica. The amount of replicas to use can be determined with the parameter num_replicas
    and using a single replica leads to identical results as using the Zwanzig formula. 
    
    All returned free energies are dG ( i-> R ) in numpy array format. This is in my opinion the best way to return results 
    as MBAR solves the system of equation up to an additive constant. Additionaly, this allows to calculate more meaningful 
    errors when running simulations with different replicates (with different starting velocities).

    Additionally, I print out in text format the full MBAR matrix as well as errors. Those can be re-opened in python with:
    np.loadtxt('/path/to/file.txt')

    Note: statistical errors from within the simulation are printed for the full matrix 
    they could also be accessed for all free energies i>R with: 
        results['dDelta_f'][num_states][0:num_states] * kt # in kJ/mol


    Parameters
    ----------
    ene_trajs: List [pd.DataFrame] 
        contains all replicas potential energies 
    s_values: List[float]
        the set of s-values used in the simulation which generated this data
    initial_offsets: List[List[float]]
        the set of energy offsets used in the simulation which generated this data (for all s values)
    out_dir: str
        path to which the results will be printed out.    
    temp: float
        temperature of the simulation
    num_replicas:
        number of replicas to include in the MBAR calculation, if 1 equivalent to Zwanzig eqn.

    Returns
    -------
    None

    """

    try:
        import pymbar
        from pymbar import MBAR
        if int(pymbar.__version__.split('.')[0]) < 4:
            print('\nThe version of pymbar you have installed is < 4.0.1. Please update your pymbar to version 4.0.1 or higher.')
            raise Exception()
    except:
        print ('\nCould not find pymbar module, free energies will only be calculated with dfmult (Zwanzig formula).\n')
        return None

    kt = (temp * const.k * const.Avogadro) / 1000 # in kJ/mol
    num_states = len(eoffs[0])

    # Doing the actual work (will also do a convergence analysis)
    percents = np.arange(10, 101, 10)

    num_points = len(percents)
    mbar_convergence = np.zeros((num_points, num_states))

    size_ene = len(ene_trajs[0])

    for i, percent in enumerate(percents):
        imax = int(size_ene * percent/100)
        tmp =  [ t[0:imax] for t in copy.deepcopy(ene_trajs)]

        try:
            u_kn, N_k = reformat_trajs_for_mbar(tmp, s_values, eoffs, temp, l=num_replicas, with_decorrelation=True)
            
            mbar = MBAR(u_kn, N_k)
            results = mbar.compute_free_energy_differences()
            mbar_convergence[i] = results['Delta_f'][num_states][0:num_states] * kt # convert back to kJ/mol
        except:
            print ('Got an error during the calculation of the free energies with M-BAR.')


    # Print the free energies (using 100% of the simulation and at all intermediate points to evaluate convergence)
    np.save(f'{out_dir}/deltaGs_mbar.npy', mbar_convergence[-1])
    np.save(f'{out_dir}/deltaGs_mbar_convergence.npy', mbar_convergence)

    # Also print the full MBAR matrix including errors. 
    header =  '\t'.join(([f'state{i}' for i in range(1, num_states+1)] + [f'ref{i}' for i in range(1, num_replicas+1)]))

    np.savetxt(f'{out_dir}/mbar_full_matrix.txt', results['Delta_f']*kt, header=header, fmt='%.2f', delimiter ='\t')
    np.savetxt(f'{out_dir}/mbar_errors_full_matrix.txt', results['dDelta_f']*kt, header=header, fmt='%.2f', delimiter = '\t')


    # Make a convergence plot 
    tmax = (ene_trajs[0]['time'].iloc[-1] + ene_trajs[0]['time'].iloc[1]) / 1000
    time = percents /100 * tmax

    plot_mbar_convergence(time, mbar_convergence, num_states, f'{out_dir}/mbar_convergence.png')

    return None 
