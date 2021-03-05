"""
visualisation
-------------

This File contains functions which generate matplotlib plots for RE-EDS simulations.
"""

from typing import Iterable, List, Tuple, Union, Dict

# plotting
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use('Agg')
matplotlib.rcParams['agg.path.chunksize'] = 10000  # avoid chunksize error

from matplotlib import pyplot as plt

# PLOT STYLE
from reeds.function_libs.utils import plots_style as ps

# General Plottsettings
for key, value in ps.plot_layout_settings.items():
    matplotlib.rcParams[key] = value

color_gradient = ps.active_gradient_map
color_map_categorical = ps.active_qualitative_map
color_map_centered = ps.active_gradient_centered

figsize = ps.figsize
alpha = ps.alpha_val