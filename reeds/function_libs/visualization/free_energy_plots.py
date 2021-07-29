from typing import Dict

import numpy as np
from matplotlib import pyplot as plt
from reeds.function_libs.visualization import plots_style as ps


def plot_dF_conv(dF_timewise : dict,
                 title: str,
                 out_path: str,
                 dt: float = (1000),
                 verbose: bool = False,
                 show_legend: bool = True):
    """plot_dF_conv

    Parameters
    ----------
    dF_timewise : Dict
        Dictionary containing the data.
    title : str
        plot title
    out_path : str
        Path of the directory in which plot will be written
    dt : float, optional
        dt for x_axis. (default 1000 - converts ps to ns)
    verbose : bool, optional
        verbose output (default False)
    show_legend : bool, optional
        (default True)

    Returns
    -------
    None
    """
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=[10, 10])
    fig.tight_layout()

    last_dF = []
    num_pairs = len(dF_timewise.items())
    colors = ps.active_qualitative_cycler_mligs(num_pairs)
    axes[0].set_prop_cycle(colors)
    axes[1].set_prop_cycle(colors)

    for replica_ligands, data in dF_timewise.items():
        replica = replica_ligands.split("_")[:2]
        ligands = "_".join(replica_ligands.split("_")[2:])
        data = dF_timewise[replica_ligands]

        t = [float(x) / dt for x in list(data.keys())]
        dF = [x["mean"] for x in data.values()]
        err = [x["err"] for x in data.values()]

        # recenter to 0, from the values of the last 5 elements.
        dF_recentered = [i - dF[-1] for i in dF]

        last_dF.append(dF[-1])

        if verbose: print("err:\n", err)
        if verbose: print("y:\n", dF)
        if verbose: print("ylen:\n", len(dF))

        # Now plot both plots.

        axes[0].errorbar(t, dF, err)
        axes[0].scatter(t, dF, label=ligands, marker='D')

        axes[1].errorbar(t, dF_recentered, err)
        axes[1].scatter(t, dF_recentered, label=ligands, marker='D')

    # Set the Correct y-limits for both plots, and write title/axis labels

    y_max = max(last_dF) + abs(0.2 * max(last_dF) - np.mean(last_dF))
    y_min = min(last_dF) - abs(0.2 * min(last_dF) - np.mean(last_dF))
    axes[0].set_ylim([y_min, y_max])
    axes[1].set_ylim([-15, 15])

    axes[0].set_title('Free Energy Convergence')
    axes[0].set_ylabel(r'$\Delta G_{AB}$ [kJ/mol]')
    axes[0].set_xlabel("time [ns]")

    axes[1].set_title('Free Energy Convergence (recentered)')
    axes[1].set_xlabel("time [ns]")

    if (show_legend): axes[0].legend(fontsize=8, loc='upper right', title='Pair A-B:', ncol=2, edgecolor='black')
    if (show_legend): axes[1].legend(fontsize=8, loc='upper right', title='Pair A-B:', ncol=2, edgecolor='black')

    fig.suptitle(title)
    fig.savefig(out_path + ".png")
    plt.close(fig)

    return fig, axes


def plot_thermcycle_dF_convergence(dF_time : Dict,
                                   out_path: str = None,
                                   title_prefix: str = "",
                                   verbose: bool = True):
    """plot_thermcycle_dF_convergence

    Parameters
    ----------
    dF_time : Dict
        Dictionary containing the data.
    out_path : str, optional
        output file path (default None)
    title_prefix : str, optional
        title prefix (default "")
    verbose : bool, optional
        verbose output (default True)

    Returns
    -------
    None
    """

    cols = 3
    print(dF_time)
    rows = len(dF_time) // cols if (len(dF_time) % cols == 0) else len(dF_time) // cols + 1

    fig, axes = plt.subplots(nrows=rows, ncols=cols)
    colors = ps.thermcycle_dF_convergence
    max_y = 200
    min_y = -500
    for ind, (name, simulation) in enumerate(dF_time.items()):
        x = [x / (20 * 1000) for x in list(simulation.keys())]
        y = [x["mean"] for x in simulation.values()]
        std = [x["err"] for x in simulation.values()]

        if verbose: print("err:\n", std)
        if verbose: print("y:\n", y)
        if verbose: print("ylen:\n", len(y))

        axes[0].errorbar(x, y, std, label=name, c=colors[ind])
        axes[0].scatter(x, y, c=colors[ind])

        axes[ind + 1].set_title("dF " + name)
        axes[ind + 1].errorbar(x, y, std, label=name, c=colors[ind])
        axes[ind + 1].scatter(x, y, c=colors[ind])

        if (max(y) > max_y): max_y = max(y)
        if (min(y) < min_y): min_y = min(y)

        clean_y = [val for val in y if (val < 1000)]
        if (abs(3 * np.std(clean_y)) < 10):
            axes[ind + 1].set_ylim([np.mean(clean_y) - 10, np.mean(clean_y) + 10])
        else:
            axes[ind + 1].set_ylim([np.mean(clean_y) - 3 * np.std(clean_y), np.mean(clean_y) + 3 * np.std(clean_y)])

    if (max_y > 200):
        axes[0].set_ylim([min_y, 200])
    axes[0].set_title("dF all")

    for ax in axes:
        ax.set_ylabel("dF/kJ")
        ax.set_xlabel("t/ns")
        ax.legend()

    fig.set_size_inches([20, 10])
    fig.suptitle(title_prefix + " dF convergence")
    if (out_path != None): fig.savefig(out_path)

    return fig, axes