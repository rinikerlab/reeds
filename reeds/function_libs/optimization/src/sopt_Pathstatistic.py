"""
    This is an old way of using the optimizers. In general I would like to try to avoid this.
"""
import copy
from numbers import Number
from typing import List

import numpy as np

from pygromos.files.repdat import Repdat


class ReplicaPath:
    """A path of a specific configuration, tracking its current state, assigned
    replica level. Tracks current state, assigned replica level and direction
    (up or down).
    """

    def __init__(self, level: int, state: int) -> None:
        """ReplicaPath

        Parameters
        ----------
        level :
             Level / Replica ID
        state :
            State at current level
        """

        self._down = True
        self._state = state
        self._level = level

    @property
    def down(self):
        return self._down

    @down.setter
    def down(self, x: bool):
        self._down = x

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, x: int):
        self._state = x

    @property
    def level(self):
        return self._level

    @level.setter
    def level(self, x: int):
        self._level = x


class Replica:
    """A replica with its parameters at a time step.

    Stores its id and partner id, as well as the current state, determined
    from the minimum of energies.
    """

    def __init__(self, id=None, partner_id=None, swap=None, state=None, num_skip_replicas: int = 0) -> None:
        """Constructor of Replica

        Parameters
        ----------

        num_skip_replicas : int, optional
            Number of replicas skipped due to duplication/

        Raises
        ------
        IOError

        """
        if (isinstance(id, Number) and isinstance(partner_id, Number) and isinstance(swap, Number) and isinstance(state,
                                                                                                                  list)):
            self._id = int(id)
            self._partner_id = int(partner_id)
            self._swap = int(swap)
            self._state = state.index(min(state))
        else:
            raise IOError("Please give me good arguments!")

    @property
    def id(self):
        return self._id

    @id.setter
    def id(self, x: int):
        self._id = x

    @property
    def partner_id(self):
        return self._partner_id

    @partner_id.setter
    def partner_id(self, x: int):
        self._partner_id = x

    @property
    def swap(self):
        return self._swap

    @swap.setter
    def swap(self, x: int):
        self.swap = x

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, x: list):
        self._state = x


class PathStatistic:
    """Statistic of all paths of initial configurations between different
    replica levels. Stores the number of "up" and "down" movements. For "down",
    the counter is incremented only for the corresponding state at the current
    replica level.
    """

    def __init__(self, init_block: List[Replica], n_states: int, s_values: List[float],
                 skipped_s_values: List[float]) -> None:
        """Constructor of PathStatistic.


        Parameters
        ----------
        init_block :    List[Replica]
            First block of replicas in gromos_file
        n_states :  int
             Number of states
        s_values :   List[float]
            Initial S values
        skipped_s_values :  List[float]
            s_vals to be skipped

        Raises
        ------
        IOError

        """
        self.n_replicas = len(init_block)
        self.n_up = [0] * self.n_replicas
        self.n_down = [[0] * n_states for i in range(self.n_replicas)]
        self.paths = []

        for i in range(self.n_replicas):
            self.paths.append(ReplicaPath(i + 1, init_block[i].state))
        #  self.paths[-1].down = False

        self.s_values = copy.deepcopy(s_values)
        self.skipped_s_values = copy.deepcopy(skipped_s_values)
        self.append_block(init_block)

    def append_block(self, block: List[Replica]) -> None:
        """Adds a block of replicas from a single time step to the statistic.

        Parameters
        ----------
        block :  List[Replica]
             Block of replicas at time step

        Returns
        -------
        None
        """
        # update state
        self.paths[0].state = block[0].state

        # update paths and statistic
        for rep in block:
            path = self.paths[rep.id - 1]
            # going down or up
            if path.down:
                self.n_down[rep.id - 1][path.state] += 1
            elif not path.down:
                self.n_up[rep.id - 1] += 1
            # set new path level, ignore swaps with other s=1.0 levels
            if rep.swap:
                path.level = max(rep.partner_id, 0)

        # set path direction
        self.paths[0].down = True
        self.paths[-1].down = False

        # sort paths to match new levels
        self.paths.sort(key=lambda sl: sl.level)


def generate_PathStatistic_from_file(file_name: str, trial_range: tuple = None, verbose: bool = False) -> PathStatistic:
    """Reads GROMOS repdat file and generates the path statistic.

    Parameters
    ----------
    file_names :    List[str]
        input Files
    trial_range :   tuple
        give a range of trials to be evaluated (time dimension)
    verbose :   bool
        Loud and noisy?

    Returns
    -------
    PathStatistic
        Statistic of exchanges between replicas generated from paths taken by initial configurations
    """
    start_at_trial = end_at_trial = False
    if (trial_range == None):
        pass
    elif (type(trial_range) == int):
        start_at_trial = trial_range
    elif (type(trial_range) == tuple and len(trial_range) == 2):
        start_at_trial = trial_range[0]
        end_at_trial = trial_range[1]
    else:
        raise IOError("could not translate the trial range option in read_gromos_file.")

    # read first file
    if (verbose): print("Parse Repdat: " + file_name)
    repdat = Repdat(file_name)
    s_values = repdat.system.s
    num_states = len(repdat.system.state_eir)
    num_replicas = len(s_values)

    num_skip_replicas = 0
    clipped_s_values = s_values
    skipped_s_values = []

    # only use the last s=1.0 replica, ignore others
    num_skip_replicas = 0
    for i in range(num_replicas):
        if s_values[i] < 1.0:
            break
        num_skip_replicas = i
    skipped_s_values = s_values[:num_skip_replicas]
    clipped_s_values = s_values[num_skip_replicas:]

    if num_skip_replicas > 0:
        print("\tReading %i replicas with %i states, skipping the first %i replicas." % (num_replicas, num_states,
                                                                                         num_skip_replicas), "\n")
    else:
        print("\tReading %i replicas with %i states." % (num_replicas, num_states), "\n")

    if (verbose): print("\treadData: ")
    data = repdat.DATA
    runs = np.unique(data.run)
    if (start_at_trial):
        runs = runs[runs > start_at_trial]
    if (end_at_trial):
        runs = runs[runs < end_at_trial]
    ids = np.unique(data.ID)[num_skip_replicas:]
    first = True
    stat = None

    # Here "run" corresponds to the timestep index.

    for cRun in runs:
        run_block = data.loc[data.run == cRun]
        clean_run_block = run_block.loc[run_block.ID.isin(ids)]
        block = []

        # Here we are doing the analysis for each replica (at the specific timestep)

        for row, replica in clean_run_block.iterrows():
            energies = [replica.state_potentials[key] for key in
                        sorted(replica.state_potentials, key=lambda x: int(x.replace("Vr", "")))]
            block.append(
                Replica(id=replica.ID - num_skip_replicas, partner_id=replica.partner, swap=replica.s, state=energies))

        # build up PathStatistic
        if (first):
            stat = PathStatistic(block, num_states, clipped_s_values, skipped_s_values)
            first = False
        else:
            stat.append_block(block)
    del repdat
    return stat
