# eds_optimization  - A library for optimization of input values for eds simulations
# Copyright (c) 2017 ETH Zurich
#
# This library is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# This library is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with this library; if not, write to the Free Software
# Foundation, Inc., 51 Franklin St, Fifth Floor, Boston, MA  02110-1301  USA

import copy
import math
from collections import namedtuple
from typing import List


def ln_exp_avg(values: List[float]):
    """Log average.

    Args:
        values: List of input values.

    Returns:
        Log average
    """
    if len(values) == 0:
        return 0.
    avg = values[0]
    for i in range(len(values)):
        avg = max(avg, values[i]) + math.log(1. + math.exp(min(avg, values[i]) - max(avg, values[i])))
    return avg - math.log(float(len(values)))


def calculate_f_ir(vr: List[float], vy: List[List[float]], eir_old: List[float],
                   s_ref: float, s_stat: float, beta: float, p_i: int, num_states: int) -> float:
    """Calculate new energy offset

    Args:
        vr: Reference energies per state
        vy: Energies per state
        eir_old: Energy offsets of previous iteration
        s_ref (float): Reference s value
        s_stat (float): s value
        beta (float): kb*T
        p_i (int): index of state considered
        num_states (int): number of states

    Returns:
        new offset for state
    """
    times_steps = len(vr)
    state_sum = [0 for i in range(times_steps)]
    weights = [0 for i in range(times_steps)]
    for k in range(len(vr)):
        # BUILD VRnew
        vr_new = 0.
        for j in range(num_states):
            beta_s = beta * s_stat
            diff_ln = -beta_s * (vy[j][k] - eir_old[j])
            vr_new = max(vr_new, diff_ln) + math.log(1.0 + math.exp(min(vr_new, diff_ln) - max(vr_new, diff_ln))) / \
                     s_stat
        vr_new = - vr_new / beta

        # Build VRstat
        vr_stat = 0.
        for j in range(num_states):
            diff_ln = - beta * s_ref * (vy[j][k] - eir_old[j])
            vr_stat = max(vr_stat, diff_ln) + math.log(1.0 + math.exp(min(vr_stat, diff_ln) - max(vr_stat, diff_ln))) / \
                      s_ref
        vr_stat = -vr_stat / beta
        sum_ln = -beta * (vy[p_i][k] - vr_stat)

        diff_wei = -beta * (vr_new - vr[k])

        state_sum[k] = diff_wei + sum_ln
        weights[k] = diff_wei

    ln_xy = ln_exp_avg(state_sum) - ln_exp_avg(weights)

    # now calculate the new EiR
    return -ln_xy / beta


def calc_form_5(eir_old_init: List[float], vr: List[float], vy: List[List[float]],
                s_in: List[float], beta: float, num_states: int, max_iter: int, rho: float = 0.0) -> (List[float], int):
    """Calculate "form 5" of original cc program to estimate new offsets for a
    single s value

    Args:
        eir_old_init: Energy offsets used in simulation.
        vr: Reference energies
        vy: Energies for each state
        s_in: s values
        beta (float): kb*T
        num_states (int): number of states
        max_iter (int): maximum number of iterations
        rho (float): convergence criteria

    Returns:
        New energy offsets
    """
    eir_old = copy.deepcopy(eir_old_init)
    eir_new = [0.0 for i in range(num_states)]
    for i in range(max_iter):
        diff_eir = 0.
        for j in range(num_states):
            # note: s_vlaues now hardcoded to 1.0, was s_in[0] for s_stat
            eir_new[j] = calculate_f_ir(vr=vr, vy=vy, eir_old=eir_old, s_ref=1.0, s_stat=1.0, beta=beta, p_i=j,
                                        num_states=num_states)  # Calc effectivley new EIR

        e0_r = eir_new[0]
        for j in range(num_states):
            eir_new[j] -= e0_r
            diff_eir += math.fabs(eir_new[j] - eir_old[j])
            eir_old[j] = eir_new[j]
            # print(eir_old[j], ", ", end='')

        # print('{:3d} / {:3d}'.format(i + 1, max_iter))

        if diff_eir <= rho:
            print("converged after " + str(i) + " steps")
            break
    return eir_old, i


def optimize_energy_offsets(eir_old_init: List[float], vvr: List[List[float]], vvy: List[List[List[float]]],
                            s_values: List[float], beta: float, frac_tresh: List[float], max_iter: int, rho: float,
                            pot_tresh: float = 0.0):
    """Optimizes the energy offsets based on simulation results.

    Args:
        eir_old_init: Energy offsets used in simulation.
        vvr: Reference.
        vvy: Energies for each s value and state. Outer index: s value. Inner
            index: state
        s_values: s values
        beta (float): k_b * T
        frac_tresh: Threshold on fraction of negative energies
        max_iter (int): Maximum iterations
        rho (float): convergence threshold
        pot_tresh (float): b

    Returns:
        List of new energy offsets for each s value.
    """
    num_states = len(vvy[0])

    ##Make it more
    if(isinstance(pot_tresh, (int, float))):
        pot_tresh = [pot_tresh for x in range(num_states)]

    num_s_values = len(vvr)
    time_steps = len(vvr[0])
    new_energies = []
    eoff_per_replica = {}
    # calc new EIR
    for i in range(num_s_values):
        # print('\nS = {:.4f}:'.format(s_values[i]))
        eir_out, iterations = calc_form_5(eir_old_init=eir_old_init, vr=vvr[i], vy=vvy[i], s_in=s_values, beta=beta,
                                          num_states=num_states, max_iter=max_iter,
                                          rho=rho)  # BUG? vvr[i] was there before
        new_energies.append(eir_out)
        eoff_per_replica.update({s_values[i]: {"eoff": eir_out, "iterations": iterations}})

    # do statistics ?
    ## calc fraction of minimal counts and undersampling counts
    zero_sate_list = [0 for j in range(num_states)]
    v_pot_num_negative = [copy.copy(zero_sate_list) for j in range(num_s_values)]
    v_pot_min_count = [copy.copy(zero_sate_list) for j in range(num_s_values)]
    for si in range(num_s_values):
        for t in range(time_steps):
            min_stat_index = 0
            for k in range(num_states):
                if vvy[si][k][t] < vvy[si][min_stat_index][t]:
                    min_stat_index = k
                if vvy[si][k][t] < pot_tresh[k]:
                    v_pot_num_negative[si][k] += 1

            v_pot_min_count[si][min_stat_index] += 1

    ## check if frac tresh is fullfiled
    threshold_fulfilled = [True for j in range(num_s_values)]
    for i in range(num_states):
        for j in range(num_s_values):
            if (v_pot_num_negative[j][i] / float(time_steps) < frac_tresh[i]):
                threshold_fulfilled[j] = False

    ##get mean and std for Energy offsets
    new_offsets = []
    EnergyOffset = namedtuple('EnergyOffset', ['mean', 'std'])
    for i in range(num_states):
        state_energy_sum = 0.0
        state_energy_squared_sum = 0.0
        state_energy_counter = 0
        for j in range(num_s_values):
            if threshold_fulfilled[j]:
                state_energy_sum += new_energies[j][i]
                state_energy_squared_sum += new_energies[j][i] * new_energies[j][i]
                state_energy_counter += 1
        if state_energy_counter == 0:
            standard_deviation = 0
            mean = eir_old_init[i]
            print('WARNING: Threshold never fulfilled for state {:d} (-frac_tresh parameter)!'.format(i))
        else:
            mean = state_energy_sum / float(state_energy_counter)
            standard_deviation = abs(math.sqrt(
                round((float(state_energy_squared_sum) / float(state_energy_counter)) - mean ** 2,
                      10)))  # rounding for numerical robustness!

        new_offsets.append(EnergyOffset(mean, standard_deviation))

    EnergyStatistic = namedtuple('EnergyStatistic', ['offsets', 'minCount', 'negCount', "eoff_per_replica"])
    return EnergyStatistic(new_offsets, v_pot_min_count, v_pot_num_negative, eoff_per_replica)
