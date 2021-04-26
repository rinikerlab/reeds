"""
PlotStyleLibrary!
This file can be used to set the global matplotlib styles.
for example in reeds.function_libs.analysis.visualization


"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from scipy import constants
from cycler import cycler

#is used for example in reeds.function_libs.analysis.visualization

def cm2inch(value: float):
    return value / 2.54


# settings:
figsize = [cm2inch(8.6), cm2inch(8.6 / constants.golden)]
figsize_doubleColumn = [cm2inch(2 * 8.6), cm2inch(2 * 8.6 / constants.golden)]

plot_layout_settings = {'font.family': 'sans-serif',
                        "font.serif": 'Times',
                        "font.size": 11,
                        'xtick.labelsize': 12,
                        'ytick.labelsize': 12,
                        'axes.labelsize': 12,
                        'axes.titlesize': 14,
                        'legend.fontsize': 3,
                        'savefig.dpi': 200,
                        }

# resolution
alpha_val = 1

# coloring
qualitative_tab_list = ["brown", "orange", "olive", "green", "blue", "cyan", "purple", "pink", "red"]
qualitative_90s_list = ["navy", "blue", "royalblue", "darkgreen", "forestgreen", "firebrick", "salmon"]
candide_colors = ['#082F6D', '#EF9F26', '#006347', '#AC0123', '#56187D',
                  '#BBDAF6', '#E6CD69', '#97D0A7', 'lightcoral']

gradient_kays_list = ['gold', 'orange', 'darkorange', 'tomato', 'orangered', 'red', 'crimson']
gradient_blue_list = ["deepskyblue", "skyblue", "steelblue", "cornflowerblue", "royalblue", "mediumblue",
                      "midgnightblue"]
gradient_green_list = ["chartreuse", "lawngreen", "limegreen", "forestgreen", "seagreen", "green", "darkgreen"]



# maps:
gradient_centered = cm.get_cmap("gist_earth_r")
gradient_green_map = cm.get_cmap("Greens")
gradient_red_map = cm.get_cmap("Reds")
gradient_blue_map = cm.get_cmap("Blues")

gradient_blueGreenYellow_map = cm.get_cmap("viridis")
gradient_kays_map = cm.get_cmap("inferno")
qualitative_tab_map_small = cm.get_cmap("Dark2")
qualitative_tab_map = cm.get_cmap("tab20b")
qualitative_dark_map = cm.get_cmap("Dark2")

thermcycle_dF_convergence = ["orange", "blue", "dark green", "purple"]

#### ACTIVE STYLE:  ###
active_gradient_map = gradient_green_map
active_qualitative_map = candide_colors
active_qualitative_map_mligs = lambda num_ligs: plt.cm.jet(np.linspace(0,1,num_ligs))
active_qualitative_cycler_mligs = lambda num_ligs: cycler('color', plt.cm.jet(np.linspace(0,1,num_ligs)))
active_gradient_list = gradient_green_list
active_qualitative_list_small = qualitative_tab_map_small
active_qualitative_list = qualitative_tab_list
active_gradient_centered = gradient_centered
