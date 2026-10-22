# Visualization fspace: shows the slice of an explanation in the 2D feature space through the sample (see slice),
# together with the diameter (see diam) and the sample
# Only 2D slices are supported, assumes that the domains of the features are normalized to [0,1]
# Based on explviz/plot/panel.py of the branch explviz
#+ also show the decision regions of the classifier as the background (requires the model and torch)

import numpy as np
import matplotlib.pyplot as plt

from .. import Error
from ..interpret import slice as slice_

# The styles of the explanations, cycled (from explviz/style.py)
COLORS = ('red', 'darkorange', 'green', 'blue', 'purple', 'cyan')
FILL_ALPHAS = (0.20, 0.25, 0.35, 0.15, 0.30, 0.70)
# The line widths are decreasing to the minimal one, so that coinciding outlines stay visible
MIN_LINE_WIDTH = 2
SAMPLE_COLOR = 'black'
MARGIN = 0.05


class Config:
    """The configuration of the visualization given by its parameters."""

    def __init__(self, params, n_vars):
        self.var_indices = None

        names = set()
        for param in params:
            name, *args = param.split()
            if name in names:
                raise Error(f'Duplicate parameter of fspace: {name}')
            names.add(name)
            if name == 'vars':
                self.var_indices = slice_.parse_vars(args, n_vars, 'fspace')
            else:
                raise Error(f'Unknown parameter of fspace: {param}')


def save(series, sample, config, file, show=False):
    """Save the explanations and the sample.

    Each of the series of the explanations is (label, vertices of the region of the slice, diameter or None,
    whether the explanation is invalid); the diameter is given by a pair of points, all points with all features.
    """
    a, b = config.var_indices
    single = len(series) == 1
    fig, axes = plt.subplots()

    for k, (label, vertices, diameter, invalid) in enumerate(series):
        color, alpha = (style[k % len(style)] for style in (COLORS, FILL_ALPHAS))
        width = MIN_LINE_WIDTH + len(series) - 1 - k
        if invalid:
            # Keep the entry in the legend
            axes.plot([], [], color=color, linewidth=width, label=f'{label} (<null>)')
            continue

        if len(vertices) >= 3:
            ring = np.vstack([vertices, vertices[:1]])
            axes.fill(ring[:, 0], ring[:, 1], color=color, alpha=alpha, zorder=1)
            axes.plot(ring[:, 0], ring[:, 1], color=color, linewidth=width, zorder=2, label=label)
        elif len(vertices) > 0:
            # A degenerate region: a segment or a point
            axes.plot(vertices[:, 0], vertices[:, 1], color=color, linewidth=width, marker='o', zorder=2,
                      label=label)
        else:
            axes.plot([], [], color=color, linewidth=width, label=label)

        if diameter is not None:
            p, q = diameter
            axes.plot([p[a], q[a]], [p[b], q[b]], color=color, linewidth=width, linestyle='--', marker='o', zorder=3,
                      label='Diameter' if single else f'{label} (diameter)')

    axes.scatter([sample[a]], [sample[b]], marker='d', color=SAMPLE_COLOR, s=40, zorder=4, edgecolors='white',
                 linewidths=0.8, label='Sample')

    axes.set_xlim(-MARGIN, 1 + MARGIN)
    axes.set_ylim(-MARGIN, 1 + MARGIN)
    axes.set_aspect('equal')
    axes.set_xlabel(f'$x_{{{a + 1}}}$')
    axes.set_ylabel(f'$x_{{{b + 1}}}$')
    axes.grid(alpha=0.3)
    axes.legend(loc='best')

    fig.savefig(file)
    if show:
        plt.show()
    plt.close(fig)
