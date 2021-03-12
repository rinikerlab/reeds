"""
visualisation
-------------

This File contains functions which generate matplotlib plots for RE-EDS simulations.
"""

# plotting
import matplotlib

matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000  # avoid chunksize error

# PLOT STYLE
from reeds.function_libs.visualization import plots_style as ps

# General Plottsettings
for key, value in ps.plot_layout_settings.items():
    matplotlib.rcParams[key] = value

color_gradient = ps.active_gradient_map
color_map_categorical = ps.active_qualitative_map
color_map_centered = ps.active_gradient_centered

figsize = ps.figsize
alpha = ps.alpha_val