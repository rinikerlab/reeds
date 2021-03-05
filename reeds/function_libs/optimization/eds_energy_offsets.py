"""
    eds_energy_offsets  - A program for optimization of energy offsets from eds simulations
    @author: Simon Frasch, Benjamin Ries

    You can use the program from your shell (not tested) or python environment.


"""

import argparse
import pandas as pd

from typing import List

from reeds.function_libs.optimization.src.energy_offset_optimization import optimize_energy_offsets


def parse_args(ene_ana_trajs: List[pd.DataFrame], Temp: float, s_values: List[float], Eoff: List[float],
               frac_tresh: List[float] = [0.9],
               pot_tresh=0.0, convergenceradius=0.0, kb=0.00831451, max_iter=20) -> dict:
    """
    This wrapper converts the input into a format which is readable by the
    eoff script.

    Parameters
    ----------
    ene_ana_trajs
        concatenated energy trajs from ene ana step.
    Temp
        temp of simulation
    s_values
        list of svalues used
    Eoff
        list containing Eoff values (give 0.0 if non)
    frac_tresh
        Threshold on fraction of negative potentials energies for each state. Either one value or a value for each state.
    pot_tresh
        Potential threshold defines maximal energyvalue to be considered in Fraction building
    convergenceradius : float
        radius of convergence
    kb : float
        boltzmann constant
    max_iter : int
        maximal_iteration

    Returns
    -------

    """

    # preparing in_arguments:
    num_s_values = len(s_values)
    num_states = len(Eoff)
    time_steps = len(ene_ana_trajs[0]["time"])
    rounding_values = 12

    # sort energies for algorithm
    vvr = []
    vvy = []
    reference_potential_key = "eR"
    state_potential_keys = list(
        sorted(["e" + str(state) for state in range(1, num_states + 1)], key=lambda x: int(x.replace("e", ""))))

    for ene_traj in sorted(ene_ana_trajs, key=lambda x: int(x.replicaID)):
        vvr.append(ene_traj[reference_potential_key].round(rounding_values).tolist())
        vvy.append([ene_traj[key].round(rounding_values).tolist() for key in state_potential_keys])

    ##Eoff as eir
    eir_old_init = Eoff
    if len(eir_old_init) != num_states:
        raise IOError("Number of EiR values does not match number of states.")

    ## thresholds
    ## fraction threshold how many values below the pottresh needed?
    if len(frac_tresh) == 1:
        frac_tresh = [frac_tresh[0] for i in range(num_states)]
    elif len(frac_tresh) != num_states:
        raise IOError("frac_tresh option requires one or n values, where n is the number of states.")

    ## potentials threshold: when is undersampling reached? (below pottresh)
    pot_tresh = pot_tresh

    ##Thermodynamic constants
    beta = 1.0 / (Temp * kb)

    rho = convergenceradius
    in_dict = {"eir_old_init": eir_old_init, "vvr": vvr, "vvy": vvy, "s_values": s_values, "beta": beta,
               "frac_tresh": frac_tresh, "max_iter": max_iter, "rho": rho, "pot_tresh": pot_tresh,
               "time_steps": time_steps}

    return in_dict


def standard_out(s_values: List[float], statistic, time_steps, pot_tresh, frac_tresh):
    """

    Parameters
    ----------
    s_values
    statistic
    time_steps
    pot_tresh
    frac_tresh

    Returns
    -------

    """
    num_s_values = len(s_values)
    num_states = len(statistic.minCount[0])

    print_lines = ["\n\nREEDS - Eoff Estimation:\n========================\n\n", ]

    print_lines.append("\n\n\tEnergy Offsets for each Replica\n\n")
    print_lines.append("| S\t| " + " | ".join(["e" + str(i) for i in range(1, num_states + 1)]) + " | iterations |\n")
    print_lines.append("|---\t|---" + " |--- ".join(["" for i in range(1, num_states + 1)]) + "|--- |\n")
    eoff_per_rep = statistic[3]
    for i in sorted(eoff_per_rep)[::-1]:
        print_lines.append('|{:2.4f}\t|'.format(float(i)) + " |\t".join(
            list(map(lambda x: str(round(x, 4)), eoff_per_rep[i]["eoff"]))) + " \t|\t" + str(
            eoff_per_rep[i]["iterations"]) + "|\n")

    print_lines.append("\n\n\tMinimum energy count per state\n")
    print_lines.append("| S\t| " + " | ".join(["e" + str(i) for i in range(1, num_states + 1)]) + " |\n")
    print_lines.append("|---\t|---" + " |--- ".join(["" for i in range(1, num_states + 1)]) + "|\n")
    for i in range(num_s_values):
        print_lines.append('| {:2.4f}\t|'.format(float(s_values[i])) + ' |\t'.join(
            ' {:8d}'.format(val) for val in statistic.minCount[i]) + " |\n")

    print_lines.append("\n\tNegative energy count per state(pottresh=" + str(pot_tresh) + ")\n\n")
    print_lines.append("| S\t| " + " | ".join(["e" + str(i) for i in range(1, num_states + 1)]) + "|\n")
    print_lines.append("|---\t|---" + " |--- ".join(["" for i in range(1, num_states + 1)]) + "|\n")
    for i in range(num_s_values):
        print_lines.append('{:2.4f}\t|'.format(float(s_values[i])) + ' |\t'.join(
            ' {:8d}'.format(val) for val in statistic.negCount[i]) + " |\n")

    print_lines.append("\n\tFraction of undersampling energies per state (fractresh=" + str(frac_tresh) + ")\n\n")
    print_lines.append("| S\t| " + " | ".join(["e" + str(i) for i in range(1, num_states + 1)]) + " |\n")
    print_lines.append("|---	\t|---	" + " |--- ".join(["" for i in range(1, num_states + 1)]) + " |\n")
    for i in range(num_s_values):
        print_lines.append('| {:2.4f}\t|'.format(float(s_values[i])) + ' |\t'.join(
            ' {:8.4f}'.format(float(val) / float(time_steps)) for val in statistic.negCount[i]) + "\n")

    print_lines.append("\n\nNew energy offset for each state:\n\n")
    for ind, offset in enumerate(statistic.offsets):
        print_lines.append(
            str(ind + 1) + '. {:10.4f}\t+-\t{:8.4f}'.format(float(offset.mean), float(offset.std)) + "\n")

    print("".join(print_lines))
    return print_lines
    # give output


def estEoff(ene_ana_trajs: List[pd.DataFrame],
            Temp: float, s_values: List[float], Eoff: List[float], out_path: str,
            frac_tresh=[0.9], pot_tresh=0.0, convergenceradius=0.0, kb=0.00831451, max_iter=20):
    """
    This function should provide access via an python script to eds energy
    estimations

    Parameters
    ----------
    ene_ana_trajs
    Temp
    s_values
    Eoff
    out_path
    frac_tresh
    pot_tresh
    convergenceradius
    kb
    max_iter

    Returns
    -------

    """

    # parsing input
    in_dict = parse_args(ene_ana_trajs=ene_ana_trajs, Temp=Temp, s_values=s_values, Eoff=Eoff, kb=kb,
                         frac_tresh=frac_tresh, max_iter=max_iter, convergenceradius=convergenceradius,
                         pot_tresh=pot_tresh)
    # print(in_dict)
    # do calclulation
    statistic = optimize_energy_offsets(eir_old_init=in_dict["eir_old_init"], vvr=in_dict["vvr"], vvy=in_dict["vvy"],
                                        s_values=in_dict["s_values"], beta=in_dict["beta"],
                                        frac_tresh=in_dict["frac_tresh"],
                                        max_iter=in_dict["max_iter"], rho=in_dict["rho"],
                                        pot_tresh=in_dict["pot_tresh"])  # calculate new energy offset
    # output:
    stdout = standard_out(s_values=in_dict["s_values"], statistic=statistic, time_steps=in_dict["time_steps"],
                          pot_tresh=in_dict["pot_tresh"], frac_tresh=frac_tresh)
    file = open(out_path, "w")
    file.writelines(stdout)
    file.close()

    return statistic


# MAIN
if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Energy offseta optimization for RE-EDS")
    parser.add_argument('-vr', type=str, nargs='+', required=True, help='vr Files', dest='vr_files')
    parser.add_argument('-temp', type=float, required=True, help='temperature', dest='temp')
    parser.add_argument('-s', type=float, nargs='+', required=True, help='s values', dest='s_values')
    parser.add_argument('-rho', type=float, required=False, default=0.0, help='convergence radius', dest='rho')
    parser.add_argument('-kb', type=float, required=False, default=0.00831451, help='boltzman constant', dest='kb')
    parser.add_argument('-maxiter', type=int, required=False, default=20, help='maximum iterations', dest='maxiter')
    parser.add_argument('-eir', type=float, nargs='+', required=True, help='EiR', dest='eir')
    parser.add_argument('-fractresh', type=float, nargs='+', required=True,
                        help='Threshold on fraction of negative potentials energies for each state. Either one value or a value for each state.',
                        dest='frac_tresh')
    parser.add_argument('-vy_s1', type=str, nargs='+', required=True,
                        help='vy Files of all states per s value, e.g. -vy_s1, -vy_s2 ...', dest='vy_s1')
    parser.add_argument('-pottresh', type=float, nargs='+', required=False,
                        help='Potential threshold defines maximal energyvalue to be considered in Fraction building',
                        dest='pottresh')
    args, unkown_args = parser.parse_known_args()

    for i in range(1, len(args.s_values)):
        parser.add_argument('-vy_s' + str(i + 1), type=str, nargs='+', required=True,
                            help='vy Files for s' + str(i + 1), dest='vy_s' + str(i + 1))
    args = parser.parse_args()
    argsDic = vars(args)
    vy_sx = []
    for x in argsDic:
        if ("vy_s" in x):
            vy_sx += argsDic[x]

    # prepare input
    in_dict = parse_args(Vr_files=args.vr_files, Temp=args.temp, s_values=args.s_values, Eoff=args.eir, Vy_sx=vy_sx,
                         frac_tresh=args.frac_tresh, pot_tresh=args.pottresh[0], convergenceradius=args.rho, kb=args.kb,
                         max_iter=args.maxiter)

    # Do calculation
    statistic = optimize_energy_offsets(eir_old_init=in_dict["eir_old_init"], vvr=in_dict["vvr"], vvy=in_dict["vvy"],
                                        s_values=in_dict["s_values"], beta=in_dict["beta"],
                                        frac_tresh=in_dict["frac_tresh"],
                                        max_iter=in_dict["max_iter"], rho=in_dict["rho"],
                                        pot_tresh=in_dict["pot_tresh"])  # calculate new energy offset

    # output:
    standard_out(s_values=in_dict["s_values"], statistic=statistic, time_steps=in_dict["time_steps"],
                 pot_tresh=in_dict["pot_tresh"], frac_tresh=in_dict["frac_tresh"])
