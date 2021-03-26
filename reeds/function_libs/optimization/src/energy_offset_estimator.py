import copy
import math
from collections import namedtuple
from typing import List

# struct
EnergyOffset = namedtuple('EnergyOffset', ['mean', 'std'])
EnergyStatistic = namedtuple('EnergyStatistic', ['offsets', 'minCount', 'negCount', "eoff_per_replica"])


class Eoff_estimator():
    def __init__(self, eir_old_init: List[float], vvr: List[List[float]], vvy: List[List[List[float]]],
                 s_values: List[float]):
        """
        Args:
            eir_old_init:
            vvr:
            vvy:
            s_values:
        """

        # Input
        self.s_values = s_values
        self.eir_old_init = eir_old_init
        self.vvr = vvr
        self.vvy = vvy

        # Results
        self._fractiontresh: float = None
        self._pot_tresh: float = None
        self._times_steps: int = None
        self.result: EnergyStatistic = None
        self.eoff_per_replica: List[EnergyOffset] = None
        self.new_Eoffs: List[EnergyOffset] = None

    def __str__(self) -> str:
        if (self.result != None):
            num_s_values = len(self.s_values)

            print_lines = ["\n\nREEDS - Eoff Estimation:\n\n", ]
            print_lines.append("\n\nS\t|\tEoff_per_s\t|\titerations\n")
            eoff_per_rep = self.result.eoff_per_replica
            for i in sorted(eoff_per_rep)[::-1]:
                print_lines.append('{:2.4f}\t|'.format(float(i)) + "\t".join(
                    list(map(lambda x: str(round(x, 4)), eoff_per_rep[i]["eoff"]))) + " \t|\t" + str(
                    eoff_per_rep[i]["iterations"]) + "\n")

            print_lines.append("\n\nS\t|\tMinimum energy count per state\n")
            for i in range(num_s_values):
                print_lines.append('{:2.4f}\t|'.format(float(self.s_values[i])) + '\t'.join(
                    ' {:8d}'.format(val) for val in self.result.minCount[i]) + "\n")

            print_lines.append("\nS\t|\tNegative energy count per state(pottresh=" + str(self._pot_tresh) + ")\n")
            for i in range(num_s_values):
                print_lines.append('{:2.4f}\t|'.format(float(self.s_values[i])) + '\t'.join(
                    ' {:8d}'.format(val) for val in self.result.negCount[i]) + "\n")

            print_lines.append(
                "\nS\t|\tFraction of undersampling energies per state (fractresh=" + str(self._frac_tresh) + ")\n")
            for i in range(num_s_values):
                print_lines.append('{:2.4f}\t|'.format(float(self.s_values[i])) + '\t'.join(
                    ' {:8.4f}'.format(float(val) / float(self._time_steps)) for val in self.result.negCount[i]) + "\n")

            print_lines.append("\n\nNew energy offset for each state:\n")
            for offset in self.result.offsets:
                print_lines.append('{:10.4f}   +- {:8.4f}'.format(float(offset.mean), float(offset.std)) + "\n")

            return print_lines
        else:
            raise ValueError("Energy Offset estimation was not executed yet! use optimize_energy_offsets")

    def _calculate_E_i_r(self, vr: List[float], vy: List[List[float]], eir_old: List[float],
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
        self._times_steps = len(vr)
        state_sum = [0 for i in range(self._times_steps)]
        weights = [0 for i in range(self._times_steps)]
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
                vr_stat = max(vr_stat, diff_ln) + math.log(
                    1.0 + math.exp(min(vr_stat, diff_ln) - max(vr_stat, diff_ln))) / \
                          s_ref
            vr_stat = -vr_stat / beta
            sum_ln = -beta * (vy[p_i][k] - vr_stat)

            diff_wei = -beta * (vr_new - vr[k])

            state_sum[k] = diff_wei + sum_ln
            weights[k] = diff_wei

        ln_xy = self._ln_exp_avg(state_sum) - self._ln_exp_avg(weights)

        # now calculate the new EiR
        return -ln_xy / beta

    def _calc_form_5(self, vr: List[float], vy: List[List[float]],
                     s_in: List[float], beta: float, num_states: int, max_iter: int, rho: float = 0.0) -> (
    List[float], int):
        """Calculate "form 5" of original cc program to estimate new offsets for
        a single s value :param self.eir_old_init: Energy offsets used in
        simulation. :param vr: Reference energies :param vy: Energies for each
        state :param s_in: s values :param beta: kb*T :param num_states: number
        of states :param max_iter: maximum number of iterations :param rho:
        convergence criteria :return: New energy offsets

        Args:
            vr:
            vy:
            s_in:
            beta (float):
            num_states (int):
            max_iter (int):
            rho (float):
        """
        eir_old = copy.deepcopy(self.eir_old_init)
        eir_new = [0.0 for i in range(num_states)]
        for i in range(max_iter):
            diff_eir = 0.
            for j in range(num_states):
                # note: s_vaues now hardcoded to 1.0, was s_in[0] for s_stat
                eir_new[j] = self._calculate_f_ir(vr=vr, vy=vy, eir_old=eir_old, s_ref=1.0, s_stat=1.0, beta=beta,
                                                  p_i=j, num_states=num_states)  # Calc effectivley new EIR

            e0_r = eir_new[0]
            for j in range(num_states):
                eir_new[j] -= e0_r
                diff_eir += math.fabs(eir_new[j] - eir_old[j])
                eir_old[j] = eir_new[j]
                # print(eir_old[j], ", ", end='')

            # print('{:3d} / {:3d}'.format(i + 1, max_iter))

            if diff_eir <= rho:
                print("State Eoff converged after " + str(i) + " steps")
                break
        return eir_old, i

    def _calculate_f_ir(self, vr: List[float], vy: List[List[float]], eir_old: List[float],
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
                vr_stat = max(vr_stat, diff_ln) + math.log(
                    1.0 + math.exp(min(vr_stat, diff_ln) - max(vr_stat, diff_ln))) / \
                          s_ref
            vr_stat = -vr_stat / beta
            sum_ln = -beta * (vy[p_i][k] - vr_stat)

            diff_wei = -beta * (vr_new - vr[k])

            state_sum[k] = diff_wei + sum_ln
            weights[k] = diff_wei

        ln_xy = self._ln_exp_avg(state_sum) - self._ln_exp_avg(weights)

        # now calculate the new EiR
        return -ln_xy / beta

    def _ln_exp_avg(self, values: List[float]):
        """Log average.
            implementation for Potential energies, avoiding overflows errors

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
