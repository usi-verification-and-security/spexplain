# Visualization fspace: shows the slice of an explanation in the 2D feature space through the sample (see slice),
# together with the diameter (see diam) and the sample
# Only 2D slices are supported, assumes that the domains of the features are normalized to [0,1]
# Based on explviz/plot/panel.py of the branch explviz
#+ also show the decision regions of the classifier as the background (requires the model and torch)

import numpy as np
import matplotlib.pyplot as plt

from .. import Error
from ..interpret import slice as slice_

REGION_COLOR = 'red'
DIAMETER_COLOR = 'blue'
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


def save(vertices, diameter, sample, config, file, show=False):
    """Save the region of the slice given by its vertices (or None), the diameter (or None) and the sample.

    The diameter is given by a pair of points, all points with all features.
    """
    a, b = config.var_indices
    fig, axes = plt.subplots()

    if vertices is not None and len(vertices) > 0:
        if len(vertices) >= 3:
            ring = np.vstack([vertices, vertices[:1]])
            axes.fill(ring[:, 0], ring[:, 1], color=REGION_COLOR, alpha=0.2, zorder=1)
            axes.plot(ring[:, 0], ring[:, 1], color=REGION_COLOR, linewidth=2, zorder=2, label='Explanation')
        else:
            # A degenerate region: a segment or a point
            axes.plot(vertices[:, 0], vertices[:, 1], color=REGION_COLOR, linewidth=2, marker='o', zorder=2,
                      label='Explanation')

    if diameter is not None:
        p, q = diameter
        axes.plot([p[a], q[a]], [p[b], q[b]], color=DIAMETER_COLOR, linewidth=2, marker='o', zorder=3,
                  label='Diameter')

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
