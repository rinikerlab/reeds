from typing import Union, List
import numpy as np

from matplotlib.colors import Colormap, to_rgba
import plotly.graph_objects as go
from plotly.colors import convert_to_RGB_255

from reeds.function_libs.visualization import plots_style as ps


def plot_state_transitions(state_transitions: np.ndarray, title: str = None, colors: Union[List[str], Colormap] =  ps.qualitative_tab_map, out_path: str = None):
    """
    Make a Sankey plot showing the flows between states.
     
    Parameters
    ----------
    state_transitions : np.ndarray
        num_states * num_states 2D array containing the number of transitions between states
    title: str, optional
        printed title of the plot
    colors: Union[List[str], Colormap], optional
        if you don't like the default colors
    out_path: str, optional
        path to save the image to. if none, the image is returned as a plotly figure
    Returns
    -------
    None or fig
        plotly figure if if was not saved
    """
    num_states = len(state_transitions)
    
    if isinstance(colors, Colormap):
        colors = [colors(i) for i in np.linspace(0, 1, num_states)]
    elif len(colors) < num_states:
        raise Exception("Insufficient colors to plot all states")

    def v_distribute(total_transitions):
        # Vertically distribute nodes in plot based on total number of transitions per state
        box_sizes = total_transitions / total_transitions.sum()
        box_vplace = [np.sum(box_sizes[:i]) + box_sizes[i]/2 for i in range(len(box_sizes))]
        return box_vplace

    y_placements = v_distribute(np.sum(state_transitions, axis=1)) + v_distribute(np.sum(state_transitions, axis=0))

    # Convert colors to plotly format and make them transparent
    rgba_colors = []
    for color in colors:
        rgba = to_rgba(color)
        rgba_plotly = convert_to_RGB_255(rgba[:-1])
        # Add opacity
        rgba_plotly = rgba_plotly + (0.8,)
        # Make string
        rgba_colors.append("rgba" + str(rgba_plotly))

    # Indices 0..n-1 are the source and n..2n-1 are the target.
    fig = go.Figure(data=[go.Sankey(
        node = dict(
          pad = 5,
          thickness = 20,
          line = dict(color = "black", width = 2),
          label = [f"state {i+1}" for i in range(num_states)]*2,
          color = rgba_colors[:num_states]*2,
          x = [0.1]*num_states + [1]*num_states,
          y = y_placements
        ),
        link = dict(
          arrowlen = 30,
          source = np.array([[i]*num_states for i in range(num_states)]).flatten(),
          target = np.array([[i for i in range(num_states, 2*num_states)] for _ in range(num_states)]).flatten(),
          value = state_transitions.flatten(),
          color = np.array([[c]*num_states for c in rgba_colors[:num_states]]).flatten()
      ),
        arrangement="fixed",
    )])
    fig.update_layout(title_text=title, font_size=20, title_x=0.5, height=max(600, num_states*100))

    if out_path:
        fig.write_image(out_path)
        return None
    else:
        return fig