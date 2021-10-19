"""
    eds_optimization  - A library for optimization of input values for eds simulations
        This module is containing all the replica exchange round trip alogrithms, described by Sidler et al. and Katzgraber et al. 2006.

    References:
        - Sidler et al. 2017
        - Katzgraber et al. 2006

    @author bries

"""

from typing import List

import numpy as np
import pandas as pd

from pygromos.files import repdat
from reeds.function_libs.optimization.src import sopt_Pathstatistic
from reeds.function_libs.optimization.src.util import get_str_from_list


# Fields
class Replica_Flow_Position:
    """Pair of Replica and corresponding replica visit fraction f from Eq. 11.

    Stores additional S values between S_i and S_i+1 with a counter,
    for equidisftant distribution of new S values in interval.
    """

    def __init__(self, s: float, f: float) -> None:
        """Constructor of SFPair.

        :param s: S value of replica
        :param f: f value calculated from path statistic
        """
        self.s = s
        self.f = f
        self.num_s_in_interval = 1  # [s_i, s_i+1)



class _RTOptimizer():
    """
    Base class for RTO optimizers
    """

    def __init__(self, replica_exchange_statistics: (sopt_Pathstatistic.PathStatistic or repdat.Repdat),
                 state_weights:List[float]=None):
        """
            This baseclass is the common substructure of all optimizer approaches.

        Parameters
        ----------
        replica_exchange_statistics: (sopt_Pathstatistic.PathStatistic or repdat.Repdat)
            input for the replica exchange statistics
        state_weights : List[float]
            weigthing of the different states.

        """
        self.statistic = replica_exchange_statistics  # statistic containing repdat

        if (type(replica_exchange_statistics) == sopt_Pathstatistic.PathStatistic):
            self._replica_position_flow_list = self._calculate_replica_visit_fraction(n_up_list=self.statistic.n_up,
                                                                                      n_down_list=self.statistic.n_down,
                                                                                      s_in=self.statistic.s_values,
                                                                                      state_weights=state_weights)
        elif (type(replica_exchange_statistics) == repdat.Repdat):
            replica_position_counts_per_state = self.statistic.get_replicaPosition_dependend_nup_ndown_for_each_state()

            print("count replicated S")
            orig = self.statistic.system.s
            replicated_s_values = [x for x in set(orig) if (orig.count(x) > 1)]
            s_without_skipped_s = []
            replica_index_without_skipped_s = []

            for x in orig:
                if (x not in replica_index_without_skipped_s):
                    replica_index_without_skipped_s.append(orig.index(x))
                    s_without_skipped_s.append(x)
            setattr(self.statistic, "skipped_s_values", replicated_s_values)
            print("left: ", s_without_skipped_s)

            # reformat the file structurer for getting flows
            ##up can not destinguish states at the moment!
            print(replica_position_counts_per_state.keys())
            replicas_without_skipped_s = [key for ind, key in enumerate(replica_position_counts_per_state.keys()) if
                                          (ind in replica_index_without_skipped_s)]
            n_up = [sum(replica_position_counts_per_state[replica_position]["tot_nup"]) for replica_position in
                    replicas_without_skipped_s]
            n_down = [replica_position_counts_per_state[replica_position]["tot_nup"] for replica_position in
                      replicas_without_skipped_s]
            self._replica_position_flow_list = self._calculate_replica_visit_fraction(n_up_list=n_up,
                                                                                      n_down_list=n_down,
                                                                                      s_in=self.statistic.system.s[
                                                                                     self.statistic.skipped_s_values:],
                                                                                      state_weights=state_weights)

        # visit fractions
        self._replica_position_flow_list.sort(key=lambda sf: sf.s, reverse=False)
        self._replica_position_flow_list_opt = self._replica_position_flow_list

        # svalues:
        self.orig_replica_parameters = [f.s for f in self._replica_position_flow_list[::-1]]
        self.opt_replica_parameters = None

    def __str__(self):
        col_format_string = "|{0:>10}|\t{1:>10}|\t{2:>10}|\t{3:>10}|\n"
        offset = len(self.statistic.skipped_s_values) + 1
        message = "\nSOPT - " + str(self.__name__) + "\n=================\n"
        warn_msg = ""
        # print statistic
        if (self.statistic.n_up != None):
            if (hasattr(self, "state_weights")):
                message += "\n\t> State-WEIGHTS: " + "\t".join(map(str, self.state_weights)) + "\n\n"
            message += "\n## FLOW STATISTICS\n\n"
            message += "|s        | n_up    |  " + "|\t".join(
                ["n_down_state_" + str(i) for i in range(1, 1 + len(self.statistic.n_down[0]))]) + "|\n"
            message += "|---\t|---\t" + "|---\t" * len(self.statistic.n_down[0]) + "|\n"
            for i in range(len(self.statistic.s_values)):
                n_dwon_text = ['{:8d}'.format(val) for val in self.statistic.n_down[i]]
                message += '|\t{:f} | {:7d} |'.format(self.statistic.s_values[i], self.statistic.n_up[i]) + '|\t'.join(
                    n_dwon_text) + "|\n"
        else:
            warn_msg += "WARNING!!!: statistic is not available!"

        # added S vals
        if (self._replica_position_flow_list_opt != None):
            message += "\n## ADDED-S\n\n"
            if (hasattr(self, "c_prime") and hasattr(self, "ds")):
                message += "\t> ds = " + str(self.ds) + "\n"
                message += "\t> C_PRIME = " + str(self.c_prime) + "\n" + "\n"

            message += col_format_string.format("repID", "s", "f(s)", "svals_in_interval")
            message += col_format_string.format("---", "---", "---", "---")
            message += "".join(
                [col_format_string.format(str(i + offset), str(x.s), str(round(x.f, 4)), str(x.num_s_in_interval - 1))
                 for i, x in enumerate(self._replica_position_flow_list_opt[::-1])])
        else:
            warn_msg += "WARNING!!!: did not add s_vals yet!"

        # calculate new s values
        message += "\n## S_VALUES\n\n"
        if (self.statistic.s_values != None):
            message += "\n### Input s_values: used for statistics: #" + str(len(self.statistic.s_values))
            message += "\n > " + get_str_from_list(self.statistic.s_values, 'float') + "\n\n"
        else:
            warn_msg += "WARNING!!!: RTO has no s_values available!"

        if (self.opt_replica_parameters != None):
            message += "\n### New s_values: #" + str(len(self.opt_replica_parameters))
            message += "\n > " + (get_str_from_list(self.opt_replica_parameters, 'float')) + "\n\n"

            if hasattr(self.statistic, "skipped_s_values") and len(self.statistic.skipped_s_values) > 0:
                message += "\n### New s values including skipped values: #" + str(
                    len(self.statistic.skipped_s_values) + len(self.opt_replica_parameters))
                message += "\n > " + get_str_from_list(
                    sorted(self.opt_replica_parameters + self.statistic.skipped_s_values)[::-1], 'float') + "\n\n"
        else:
            warn_msg += "WARNING!!!: optimized s_vals are not available!\n"
        return message + "\n" + warn_msg

    # Basic Functions
    def optimize(self, add_replicas: int):
        raise NotImplementedError("The function is not implemented yet: " + str(self.optimize))

    def _calculate_replica_visit_fraction(self, s_in: List[float], n_up_list: List[float],
                                          n_down_list: List[List[float]],
                                          state_weights: List[float] = None) -> List[float]:
        raise NotImplementedError("The function is not implemented yet: " + str(self._calculate_replica_visit_fraction))


    @property
    def replica_position_flow_list(self)->pd.DataFrame:
        return self.get_replica_position_flows()

    @property
    def replica_position_flow_list_opt(self)->pd.DataFrame:
        return self.get_replica_position_flows_optimization()

    def get_replica_position_flows(self)->pd.DataFrame:
        """
            Returns the replica-position  flow distribution and the number of replicas in a bin

        Returns
        -------
        pd.Dataframe
            contains the flow distribution

        """
        return pd.DataFrame({x :vars(self._replica_position_flow_list[x]) for x in range(len(self._replica_position_flow_list))}).T

    def get_replica_position_flows_optimization(self)->pd.DataFrame:
        """
            Returns the optimized replica-position flow distribution and the number of added replicas in a bin

        Returns
        -------
        pd.Dataframe
            contains the flow distribution

        """
        return pd.DataFrame({x :vars(self._replica_position_flow_list_opt[x]) for x in range(len(self._replica_position_flow_list_opt))}).T


    def get_new_replica_dist(self) -> list:
        """
        Get the current replica parameters.

        Returns
        -------
             List of current parameters values
        """

        result_dist = []
        if (self._replica_position_flow_list_opt == None or self.opt_replica_parameters == None):
            print(self.opt_replica_parameters)
            raise Exception(
                "Can not give back new_replica distribution, first run optimization! (get_new_replica_dist)")

        elif hasattr(self.statistic, "skipped_s_values") and len(self.statistic.skipped_s_values) > 0:
            return self.statistic.skipped_s_values + self.opt_replica_parameters
        else:
            return self.opt_replica_parameters

    # Optimizer
    def _optimize_GRTO(self, add_replicas: int, ds: float = 0.0001, verbose: bool = True,
                       detail_verbose: int = 0) -> None:
        """
            optimization in GRTO style

        Parameters
        ----------
        add_replicas: int
            add number of replicas
        ds : int
            integration step size
        verbose : bool
            verbosity
        detail_verbose : int
            level of verbosity

        """
        f_n_list = self._replica_position_flow_list[::-1]
        old_s_dist = [f.s for f in f_n_list]
        smin = min(old_s_dist)
        smax = max(old_s_dist)
        old_num_replicas = len(old_s_dist)
        self.ds = ds

        assert (
                           smin / 10) >= ds, "ds (integration step) in _optimize_GRTO is not small enough. Please ensure that ds is a tenth of smin. ds= " + str(
            ds) + "\n should be at least: " + str(smin / 10)

        self._replica_position_flow_list_opt = self._replica_position_flow_list
        new_replica_num = add_replicas + old_num_replicas

        if verbose:
            print("smin ", smin)
            print("smax ", smax)
            print("list:flow,svals ", [(f.f, f.s) for f in f_n_list])
            print("old_num_replicas: ", old_num_replicas)
            print("add_replicas: ", add_replicas)

        # 1: calculate normalisation factor c_prime from troyer 2006
        if (detail_verbose == 1):
            norm_verbose = True
        else:
            norm_verbose = False

        self.c_prime = self._calculate_normalisation_c_prime(old_s_dist=old_s_dist, f_n_list=f_n_list, ds=ds,
                                                             verbose=norm_verbose)

        if verbose:
            print("\n\n")
            print("1/C_prime", 1 / self.c_prime)
            print("C_prime", self.c_prime)

        # 2: set new S_distribution with area cals
        if (detail_verbose == 2):
            add_verbose = True
        else:
            add_verbose = False

        new_s_dist = self._add_s_values_accord_to_flow_area(old_s_dist=old_s_dist, f_n_list=f_n_list,
                                                            c_prime=self.c_prime, ds=ds,
                                                            new_replica_num=new_replica_num, verbose=add_verbose)
        new_s_dist_round = self._nice_sval_list(new_s_dist)
        self.opt_replica_parameters = new_s_dist_round

        if verbose:
            print("\nold_s_vals", old_s_dist)
            print("new_s dist ", new_s_dist)
            print("new_s dist round ", new_s_dist_round)
            print("len old, new", len(old_s_dist), len(new_s_dist))

    def _optimize_LRTO(self, add_replicas: int, verbose=False) -> None:
        """
            optimization in LRTO style (Katzgraber et al. 2006)

        Parameters
        ----------
        add_replicas : int
            number of replicas to add.
        verbose : bool
            not used here

        Returns
        -------

        """

        # if verbose:
        #    raise IOError("verbosity was not yet implemented! for _optimize_LRTO")

        # new distribution init
        self._replica_position_flow_list_opt = self._replica_position_flow_list

        # fill in replicas to replica flow gaps
        for k in range(add_replicas):
            # find_maximal flow difference in distribution
            index = self._find_max_diff_index()
            # add a replica to max diff
            self._add_dummy_replica_to_intervall(index)

        # fill up linearly gaps with new  replica parametrs (s_values)
        self.opt_replica_parameters = []
        # add initial val 1
        self.opt_replica_parameters.append(self._replica_position_flow_list_opt[-1].s)

        # interpolate between existing replica parameters, for taking dummy replicas into account.
        for i in range(len(self._replica_position_flow_list_opt) - 1)[::-1]:
            sf = self._replica_position_flow_list_opt[i]
            self.opt_replica_parameters.append(sf.s)
            if (sf.num_s_in_interval > 1):
                # calculate the range, which needed to be distributed over additional replicas
                ds = (self._replica_position_flow_list_opt[i + 1].s - sf.s) / float(sf.num_s_in_interval)
                # actual adding s_vals
                for j in range(1, sf.num_s_in_interval):
                    self.opt_replica_parameters.append(sf.s + j * ds)

        self.opt_replica_parameters.sort(reverse=True)
        self.opt_replica_parameters = self._nice_sval_list(svals_list=self.opt_replica_parameters)

        for index in range(len(self._replica_position_flow_list_opt)):
            self._add_dummy_replica_to_intervall(index)




    # Flow calculations calculate f_n for optimizers - preparation
    def _calculate_replica_visit_fraction_one_state(self, s_in: List[float], n_up_list: List[float],
                                                    n_down_list: List[List[float]]) -> List[float]:
        """
            Sidler et al. 2017 - Eq. 5 - calculate the fraction of replicas, that have visited this replica_position position in a certain state, for 1 state.

        Parameters
        ----------
        s_in : List[float]
            input s-distribution
        n_up_list : List[float]
             replicas moving up (coming from down)
        n_down_list : List[List[float]]
            replicas moving down (coming from up)

        Returns
        -------
        List[float]
             gives the fraction of how many replicas visited a replica_position position.
        """

        # get ammounts of replicas and states
        num_states = len(n_down_list[0])
        num_replicas = len(s_in)

        # Calculate f(s) for input s values and states
        replica_visits_fraction = list()

        for replica_position in range(num_replicas):
            # init list with Replica_Flow_position objects
            replica_visits_fraction.append(Replica_Flow_Position(s_in[replica_position], 0))
            f_n = 0.0

            # calc
            n_up = np.float64(n_up_list[replica_position])
            n_down = np.float64(0.0)
            for state in range(num_states):
                n_down += np.float64(n_down_list[replica_position][state])

            if (n_up + n_down > 0):
                f_n = n_down / (n_up + n_down)

            replica_visits_fraction[-1].f = f_n

        return replica_visits_fraction

    def _calculate_replica_visit_fraction_n_states(self, s_in: List[float], n_up_list: List[float],
                                                   n_down_list: List[List[float]],
                                                   state_weights: List[float] = None) -> List[float]:
        """
            Sidler et al. 2017 - Eq. 11 - calculate the fraction of replicas, that have visited this replica position in a certain state. for N-states!

        Parameters
        ----------
        s_in : List[float]
            list of input s-values
        n_up_list : List[float]
            replicas moving up (coming from down)
        n_down_list : List[float]
            replicas moving down (coming from up)
        state_weights : List[float]
            weigth factors for the different states in the replicas

        Returns
        -------
        List[float]
            gives the fraction of how many replicas visited a replica position.
        """

        # get ammounts of replicas and states
        num_states = len(n_down_list[0])
        num_replicas = len(s_in)

        # w - weights for each state
        if (state_weights == None):
            state_weights = [1.0 / num_states for x in range(num_states)]
        self.state_weights = state_weights

        # Calculate f(s) for input s values and states
        replica_visits_fraction = list()

        for replica in range(num_replicas):
            # init list with Replica_Flow_position objects
            replica_visits_fraction.append(Replica_Flow_Position(s_in[replica], 0))
            f_n = 0.0

            # calc
            n_up = float(n_up_list[replica])
            for state in range(num_states):
                weigth = state_weights[state]
                n_down = float(n_down_list[replica][state])

                if (n_up + n_down > 0):
                    f_n += (weigth * n_down * num_states) / (n_up + n_down * num_states)  # Eq. 11

            replica_visits_fraction[-1].f = f_n

        return replica_visits_fraction

    def _calculate_replica_visit_fraction_n_states_equalized(self, s_in: List[float], n_up_list: List[float],
                                                             n_down_list: List[List[float]],
                                                             state_weights: List[float] = None) -> List[float]:
        """
        @DEVELOP! - try and thing about it!
        Modified Sidler et al. 2017 - Eq. 11 - calculate the fraction of replicas, that have visited this replica position in a certain state. for N-states!
        Difference is factor N for the states comming down, try to evaluate down and up as equals.
        :param n_up_in: replicas moving up (coming from down)
        :param n_down_in: replicas moving down (coming from up)
        :param num_replicas: total number of replicas
        :param state_weights:
        :return: replica_visits_fraction: gives the fraction of how many replicas visited a replica position.
        """

        # get ammounts of replicas and states
        num_states = len(n_down_list[0])
        num_replicas = len(s_in)

        # w - weights for each state
        if (state_weights == None):
            state_weights = [1.0 for x in range(num_states)]
        self.state_weights = state_weights

        # Calculate f(s) for input s values and states
        replica_visits_fraction = list()

        for replica in range(num_replicas):
            # init list with Replica_Flow_position objects
            replica_visits_fraction.append(Replica_Flow_Position(s_in[replica], 0))
            f_n = 0.0

            # calc
            n_up = float(n_up_list[replica])
            n_down = 0.0
            for state in range(num_states):
                state_weight = state_weights[state]
                state_n_down = float(n_down_list[replica][state])
                n_down += state_weight * state_n_down

            if (n_up + n_down > 0):  # if the denominator is not zero, else 0.0
                f_n += n_down / (n_up + n_down)  # modified version of eq. 11 (putted sums inside the division.)

            replica_visits_fraction[-1].f = f_n

        return replica_visits_fraction

    # FUNCTIONs for GRTO
    def _add_s_values_accord_to_flow_area_fast(self, old_s_dist: list, f_n_list: list, c_prime: np.float64, ds: float,
                                               new_replica_num: int, verbose=False):

        """
        @DEVELOPING!
        Think more thorrow about it! not working at this time

        :param old_s_dist:
        :param f_n_list:
        :param c_prime:
        :param ds:
        :param new_replica_num:
        :param verbose:
        :return:
        """

        raise NotImplementedError("fast version is not implemented and tested. ")
        # add initial s_value (smin comes later)
        replicas_to_add = new_replica_num - 2  # because min and max s_values are again in s_distribution
        smax = max(old_s_dist)
        new_s_dist = [smax]
        self._add_dummy_replica_to_intervall(len(self._replica_position_flow_list_opt) - 1)

        # define relative area content of each replica
        add_if = (np.float64(1.0) / np.float64(new_replica_num - 2))  # Todo: rename sensefull!
        if verbose:
            print()
            print("Add Replicas")
            print("\tadd_if", add_if)

        area = np.float64(0)
        for index in range(0, len(old_s_dist) - 1):
            # get svals
            s_i = np.float64(old_s_dist[index])
            s_j = np.float64(old_s_dist[index + 1])

            if (index == len(old_s_dist) - 1):
                ammount_of_ds = round(abs(s_i - s_j) / ds) - 1
            else:
                ammount_of_ds = round(abs(s_i - s_j) / ds)

            # get precalculated flows
            f_n_i = np.float64(f_n_list[index].f)
            f_n_j = np.float64(f_n_list[index + 1].f)

            # calc c_prime contribution
            nominator = np.float64(f_n_i - f_n_j)
            denominator = np.float64((s_i - s_j) ** 2)
            area += np.float64(c_prime * (np.sqrt(abs(nominator / denominator)) * ds * ammount_of_ds))

            if verbose:
                print()
                print("STEP STATISTIC")
                print("\tindex", index)
                print("\tf_n i, i+1: ", f_n_i, f_n_j)
                print("\ts i, i+1: ", s_i, s_j)
                print("\tammount_ds:", ammount_of_ds)
                print("\tdenominator: ", denominator)
                print("\tnominator: ", nominator)
                print("\ttmp_area: ", area)

            # add new s_val if area is big enough
            if (area > add_if):
                # adding area:
                tmp_add_replicas = int(area // add_if)
                tmp_rest_area = area % add_if
                area_percentage = (add_if * tmp_add_replicas) / area
                s_dist = (area_percentage * abs(s_j - s_i)) / tmp_add_replicas

                s_values = [s_i - s_dist * x for x in range(1, tmp_add_replicas + 1)]

                if verbose:
                    print("ADDING SVALS <--")
                    print("\treplicas to add: ", replicas_to_add)
                    print("\tadd replicas to interval", tmp_add_replicas)
                    print("\tarea percentage for replicas of s_intervall: ", area_percentage)
                    print("\tadd s_distance from s_i*x: ", s_dist)
                    print("\tadded svals: ", s_values)
                    print("\trest area of interval", tmp_rest_area)

                new_s_dist += s_values
                replicas_to_add -= tmp_add_replicas
                bucket_index = (len(self._replica_position_flow_list_opt) - 1) - 1 - index
                for _ in range(int(tmp_add_replicas)):
                    self._add_dummy_replica_to_intervall(bucket_index)
                area = tmp_rest_area

        smin = min(old_s_dist)
        new_s_dist.append(smin)
        self._add_dummy_replica_to_intervall(0)

        return new_s_dist

    def _add_s_values_accord_to_flow_area(self, old_s_dist: list, f_n_list: list, c_prime: np.float64, ds: float,
                                          new_replica_num: int, verbose=False):
        """
            This function is required for GRTO algorithm.
            it calculates the new position for the replicas accordingt to the integral of the flow curve over all replicas

        Parameters
        ----------
        old_s_dist : List[float]
            old s-values
        f_n_list : List[List[float]]
            state dependendt flow lsit
        c_prime : float
            coefficient for flow//replicas
        ds : float
            integral step size
        new_replica_num : int
            adding replicas
        verbose : bool
            verbosity level

        Returns
        -------
        List[float]
            new_s_dist
        """
        smin = min(old_s_dist)
        smax = max(old_s_dist)
        replicas_to_add = new_replica_num - 2

        new_s_dist = [smax]
        self._add_dummy_replica_to_intervall(len(self._replica_position_flow_list_opt) - 1)

        add_if = (np.float64(1.0) / np.float64(new_replica_num - 1))  # Todo: rename sensefull!
        if verbose:
            print()
            print("Add Replicas")
            print("add_if", add_if)

        area = np.float64(0)
        current_replica_index = 0
        for s in reversed(np.arange(smin + ds, smax, ds)):
            # get precalculated flows
            f_n_i = np.float64(f_n_list[current_replica_index].f)
            f_n_j = np.float64(f_n_list[current_replica_index + 1].f)
            # get svals
            s_i = np.float64(old_s_dist[current_replica_index])
            s_j = np.float64(old_s_dist[current_replica_index + 1])

            # calc c_prime contribution
            nominator = np.float64(f_n_j - f_n_i)
            denominator = np.float64((s_j - s_i) ** 2)

            area += np.float64(c_prime * (np.sqrt(abs(nominator / denominator)) * ds))

            # add new s_val if area is big enough
            if (area > add_if):
                if verbose:
                    print()
                    print("adding replica: ", replicas_to_add)
                    print("s: ", s)
                    print("area: ", area)
                    print("current_rep: ", current_replica_index)

                replicas_to_add -= 1
                new_s_dist.append(s)
                self._add_dummy_replica_to_intervall(
                    (len(self._replica_position_flow_list_opt) - 1) - 1 - current_replica_index)
                area = 0

            if (s < s_j):
                current_replica_index += 1
            """
            if verbose:
                print()
                print("s_value: ", s)
                print("current_replica: ", current_replica_index)
                print("f_n i, i+1: ", f_n_i, f_n_j)
                print("s i, i+1: ", s_i, s_j)
                print("nominator: ", nominator)
                print("denominator: ", denominator)
                print("area", area)
            """

        new_s_dist.append(smin)
        self._add_dummy_replica_to_intervall(0)

        return new_s_dist

    def _calculate_normalisation_c_prime(self, old_s_dist: list, f_n_list: list, ds: float,
                                         verbose=False) -> np.float64:
        """

        Parameters
        ----------
        old_s_dist : List[float]
            s-list
        f_n_list : List[List[float]]
            flows of the replicas state dependend
        ds : float
            stepsize in the integral
        verbose : bool, optional
            verbosity on?

        Returns
        -------
        float
            c_prime
        """

        c_prime = 0
        for index in range(1, len(old_s_dist)):
            # get svals
            s_i = np.float64(old_s_dist[index])
            s_j = np.float64(old_s_dist[index - 1])
            if (index == len(old_s_dist) - 1):
                ammount_of_ds = round(abs(s_j - s_i) / ds) - 1
            else:
                ammount_of_ds = round(abs(s_j - s_i) / ds)

            # get precalculated flows
            f_n_i = np.float64(f_n_list[index].f)
            f_n_j = np.float64(f_n_list[index - 1].f)

            # calc c_prime contribution
            nominator = np.float64(f_n_j - f_n_i)
            denominator = np.float64((s_j - s_i) ** 2)
            c_prime += np.float64(np.sqrt(abs(nominator / denominator)) * ds * ammount_of_ds)

            if verbose:
                print()
                print("index", index)
                print("f_n i-1, i: ", f_n_j, f_n_i)
                print("s i-1, i: ", s_j, s_i)
                print("ammount_ds:", ammount_of_ds)
                print("denominator: ", denominator)
                print("nominator: ", nominator)
                print("tmp_cprime: ", c_prime)

        c_prime = 1.0 / np.float64(c_prime)
        return c_prime

    def _calculate_normalisation_c_prime_integral(self, old_s_dist: list, f_n_list: list, ds: float,
                                                  verbose=False) -> np.float64:
        """
                Deapriceated: Old implementation, doing a real integral. Not needed for a linear propagation. - slow

        Parameters
        ----------
        old_s_dist : List[float]
            s-list
        f_n_list : List[List[float]]
            flows of the replicas state dependend
        ds : float
            stepsize in the integral
        verbose : bool, optional
            verbosity on?

        Returns
        -------
        float
            c_prime
        """

        smin = min(old_s_dist)
        smax = max(old_s_dist)

        current_replica_index = 0
        c_prime = 0
        tmp_c_p = 0
        ds_count = 0
        # Old real integral slow! for a linear case here.
        for s in reversed(np.arange(smin + ds, smax, ds)):

            # get svals
            s_i = np.float64(old_s_dist[current_replica_index])
            s_j = np.float64(old_s_dist[current_replica_index + 1])

            if (s < s_j):
                print()
                print(ds)
                print("f_n i, i+1: ", f_n_i, f_n_j)
                print("s i, i+1: ", s_i, s_j)
                print("local c: ", c_prime - tmp_c_p)
                print("ammount of ds: ", ds_count)
                tmp_c_p = c_prime
                ds_count = 0
                current_replica_index += 1
                # get new svals
                s_i = s_j
                s_j = np.float64(old_s_dist[current_replica_index + 1])
                print("new: s i, i+1: ", s_i, s_j)

            # get precalculated flows
            f_n_i = np.float64(f_n_list[current_replica_index].f)
            f_n_j = np.float64(f_n_list[current_replica_index + 1].f)

            # calc c_prime contribution
            nominator = np.float64(f_n_i - f_n_j)
            denominator = np.float64((s_i - s_j) ** 2)

            c_prime += np.float64(np.sqrt(abs(nominator / denominator)) * ds)
            ds_count += 1
            if verbose:
                print()
                print("s: ", s)
                print("current_replica: ", current_replica_index)
                print("f_n i, i+1: ", f_n_i, f_n_j)
                print("s i, i+1: ", s_i, s_j)
                print("denominator: ", denominator)
                print("nominator: ", nominator)
                print("tmp_cprime: ", c_prime)
        c_prime = 1.0 / np.float64(c_prime)
        return c_prime

    # Functions for LRTO
    def _find_max_diff_index(self) -> int:
        """
            Find maximal flow loss off replicas

        Returns
        -------
        int
            index of upper replica position of the bottle neck
        """
        index = 0
        diff_max = 0

        # get biggest flow loss in the replica set.
        for i in range(len(self._replica_position_flow_list) - 1):
            diff = abs(self._replica_position_flow_list[i + 1].f - self._replica_position_flow_list[i].f) / float(
                self._replica_position_flow_list[i].num_s_in_interval)
            if diff > diff_max:
                diff_max = diff
                index = i

        return index

    # general useable fuctions
    def _add_dummy_replica_to_intervall(self, index:int) -> None:
        """
            index of lower replica, of the gap where a new replica should be added

        Parameters
        ----------
        index : int
            index of the replica bucket


        """

        # add replica to bucket
        self._replica_position_flow_list_opt[index].num_s_in_interval += 1

    def _nice_sval_list(self, svals_list: List[float], sig_digits: int = 6) -> List[float]:
        round_list = []
        for s in svals_list:
            number = str(s).split(".")
            count_digits = 0
            if (len(number) == 1):
                round_list.append(float(s))
                continue
            else:
                for x in number[1]:
                    if (x != "." and x != "0"):
                        count_digits += 3
                        break
                    else:
                        count_digits += 1
                round_list.append(np.round(s, count_digits + sig_digits))
        return round_list


class N_LRTO(_RTOptimizer):

    def __init__(self, replica_exchange_statistics, state_weights=None):
        """
            Performs a Local roundtrip optimization
            Sidler et al. 2017

        Parameters
        ----------
        replica_exchange_statistics:  (sopt_Pathstatistic.PathStatistic or repdat.Repdat)
            the exchange statistics, that should be analysed
        state_weights : List[float]
            weights for the individual states in a replica, used to optimize the s-distribution state dependingly
        """
        super().__init__(replica_exchange_statistics, state_weights)
        self.__name__ = "N-LRTO"

    def _calculate_replica_visit_fraction(self, s_in: List[float], n_up_list: List[float],
                                          n_down_list: List[List[float]],
                                          state_weights: List[float] = None) -> List[float]:
        """
            see _RTOptimizer._calculate_replica_visit_fraction_n_states
        """
        return self._calculate_replica_visit_fraction_n_states(s_in=s_in, n_up_list=n_up_list, n_down_list=n_down_list,
                                                               state_weights=state_weights)

    def optimize(self, add_replicas: int, verbose=False) -> None:
        """
            see _RTOptimizer._optimize_LRTO
        """
        self._optimize_LRTO(add_replicas=add_replicas, verbose=verbose)


class Equalized_N_LRTO(_RTOptimizer):
    """
   DEAPRECIATED
    """

    def __init__(self, replica_exchange_statistics, state_weights=None):

        super().__init__(replica_exchange_statistics, state_weights)
        self.__name__ = "EqN-LRTO"

    def _calculate_replica_visit_fraction(self, s_in: List[float], n_up_list: List[float],
                                          n_down_list: List[List[float]],
                                          state_weights: List[float] = None) -> List[float]:
        """
            see _RTOptimizer._calculate_replica_visit_fraction_n_states_equalized
        """
        return self._calculate_replica_visit_fraction_n_states_equalized(s_in=s_in, n_up_list=n_up_list,
                                                                         n_down_list=n_down_list,
                                                                         state_weights=state_weights)

    def optimize(self, add_replicas: int, verbose=False) -> None:
        """
            see _RTOptimizer._optimize_LRTO
        """
        self._optimize_LRTO(add_replicas=add_replicas, verbose=verbose)


class N_GRTO(_RTOptimizer):
    def __init__(self, replica_exchange_statistics, state_weights=None):
        """
            Performs a global roundtrip optimization
            Sidler et al. 2017

        Parameters
        ----------
        replica_exchange_statistics:  (sopt_Pathstatistic.PathStatistic or repdat.Repdat)
            the exchange statistics, that should be analysed
        state_weights : List[float]
            weights for the individual states in a replica, used to optimize the s-distribution state dependingly
        """

        super().__init__(replica_exchange_statistics, state_weights)
        self.__name__ = "N-GRTO"

    def _calculate_replica_visit_fraction(self, s_in: List[float], n_up_list: List[float],
                                          n_down_list: List[List[float]],
                                          state_weights: List[float] = None) -> List[float]:
        """
            see _RTOptimizer._calculate_replica_visit_fraction_n_states
        """
        return self._calculate_replica_visit_fraction_n_states(s_in=s_in, n_up_list=n_up_list, n_down_list=n_down_list,
                                                               state_weights=state_weights)

    def optimize(self, add_replicas: int, ds: float = 0.0001, verbose: bool = True, detail_verbose: int = 0):
        """
            see _RTOptimizer._optimize_GRTO
        """
        self._optimize_GRTO(add_replicas=add_replicas, ds=ds, verbose=verbose, detail_verbose=detail_verbose)


class One_GRTO(_RTOptimizer):
    """
    Performs a global roundtrip optimization
    """

    def __init__(self, replica_exchange_statistics):
        """
            Performs a global roundtrip optimization
            Katgraber et al. 2006

        Parameters
        ----------
        replica_exchange_statistics:  (sopt_Pathstatistic.PathStatistic or repdat.Repdat)
            the exchange statistics, that should be analysed
        """
        super().__init__(replica_exchange_statistics, None)
        self.__name__ = "1-GRTO"

    def optimize(self, add_replicas: int, ds: float = 0.0001, verbose: bool = True, detail_verbose: int = 0):
        """
            see _RTOptimizer._optimize_GRTO
        """
        self._optimize_GRTO(add_replicas=add_replicas, ds=ds, verbose=verbose, detail_verbose=detail_verbose)

    def _calculate_replica_visit_fraction(self, s_in: List[float], n_up_list: List[float],
                                          n_down_list: List[List[float]],
                                          state_weights: List[float] = None) -> List[float]:
        """
            see _RTOptimizer._calculate_replica_visit_fraction_one_state
        """
        return self._calculate_replica_visit_fraction_one_state(n_up_list=n_up_list, n_down_list=n_down_list, s_in=s_in)


class One_LRTO(_RTOptimizer):

    def __init__(self, replica_exchange_statistics):
        """
            Performs a Local roundtrip optimization
            Katzgraber et al. 2006

        Parameters
        ----------
        replica_exchange_statistics:  (sopt_Pathstatistic.PathStatistic or repdat.Repdat)
            the exchange statistics, that should be analysed
        """
        super().__init__(replica_exchange_statistics, None)
        self.__name__ = "1-LRTO"

    def optimize(self, add_replicas: int, ds: float = 0.001, verbose: bool = True,
                 detail_verbose: int = 0):
        """
            see _RTOptimizer._optimize_LRTO
        """
        self._optimize_LRTO(add_replicas=add_replicas, verbose=verbose)

    def _calculate_replica_visit_fraction(self, s_in: List[float], n_up_list: List[float],
                                          n_down_list: List[List[float]],
                                          state_weights: List[float] = None) -> List[float]:
        """
            see _RTOptimizer._calculate_replica_visit_fraction_one_state
        """
        return self._calculate_replica_visit_fraction_one_state(n_up_list=n_up_list, n_down_list=n_down_list, s_in=s_in)
