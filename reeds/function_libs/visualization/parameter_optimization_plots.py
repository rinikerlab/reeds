import numpy as np
from matplotlib import pyplot as plt

from reeds.function_libs.utils import plots_style as ps

"""
  Eoff plots
"""

def plot_peoe_eoff_vs_s(eoff: dict, energy_offsets, title: str, out_path: str):
    """gives out a plot, showing the development of Eoffsets along :param eoff:
    :param title: :param out_path: :return:

    Args:
        eoff (dict):
        energy_offsets: contains the energy offsets for each state (average over undersampling cases)
        title (str):
        out_path (str):
        color_gradient_flag:
    """

    s_vals = sorted(list(eoff.keys()))
    x = range(len(s_vals))

    eoffsets_per_s = []
    for i in sorted(eoff):
        eoffsets_per_s.append(eoff[i]["eoff"])
    min_eoff, max_eoff = np.min(eoffsets_per_s[-1]), np.max(eoffsets_per_s[-1])

    num_stats = len(eoffsets_per_s[0])
    y = [list(map(lambda x: x[i], eoffsets_per_s)) for i in range(num_stats)]

    # collect nice labels
    s_vals = list(map(lambda x: float(x), s_vals))

    number_of_labels = 5
    step_size = len(s_vals) // number_of_labels
    labels = s_vals[::step_size]  # list(map(lambda x: np.round(np.log(x),2), s_vals[::step_size]))

    colors = ps.active_qualitative_list_small.colors
    repnum = num_stats + 1

    # plot
    fig, (ax1, ax2) = plt.subplots(nrows=2, )

    ##plot1
    for i in range(num_stats):
        ax1.plot(x, y[i] - energy_offsets[i].mean, label=(i + 1),  # color=colors[i % repnum],
                 lw=2)

    ax1.set_ylabel("$(E_i^R(s)-\overline{E}_i^R)$/(kJ/mol)")
    ax1.set_xticks(x[::step_size])
    ax1.set_xticklabels(labels)

    ##plot2
    for i in range(num_stats):
        ax2.plot(x, y[i], label=(i + 1),  # color=colors[i % repnum],
                 lw=2)

    # plt.title(title)
    # ax2.set_ylim([-max_eoff - 10, max_eoff + 10])
    ax2.set_xlabel("s")
    ax2.set_ylabel("$E^R_i(s)$/(kJ/mol)")

    # position legend
    ax2.set_xticks(x[::step_size])
    ax2.set_xticklabels(labels)

    # legAX=fig.addsuplot(1,2, 1)
    chartBox = ax1.get_position()
    ax1.set_position([chartBox.x0, chartBox.y0, chartBox.width * 0.85, chartBox.height])

    ncol = len(labels) // 10 if (len(labels) // 10 > 0) else 1
    lgnd = ax1.legend(title="states:", loc=2, bbox_to_anchor=(1.05, 1), ncol=ncol, prop={"size": 15})

    fig.tight_layout()
    fig.suptitle(title + " $E^R_i$ per s-value", y=1.05)
    fig.savefig(out_path, bbox_extra_artists=(lgnd,), bbox_inches='tight')
    plt.close()


def plot_peoe_eoff_time_convergence(state_time_dict: dict, out_path: str):
    fig, ax = plt.subplots(ncols=1, figsize=(16, 9))
    c_steps = 20 // len(state_time_dict) if (20 // len(state_time_dict) != 0) else 1
    for (state, time_dict), c in zip(state_time_dict.items(), ps.active_qualitative_map.colors[::c_steps]):
        ax.plot(time_dict["time"], time_dict["mean"], label="state " + str(state), c=c, lw=3)
        ax.errorbar(x=time_dict["time"], y=time_dict["mean"], yerr=time_dict["std"], c=c)

    ax.set_xlabel("time [ps]")
    ax.set_ylabel("$E_i^R$ [kJ/mol]")
    ax.set_title("$E_i^R$ time convergence")

    plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left')
    fig.tight_layout()

    fig.savefig(out_path)
    plt.close()