# Visualization fspace: shows the slice of an explanation in the 2D feature space through the sample (see slice),
# together with the diameter (see diam), the sample, and the decision regions of the classifier if given
# Only 2D slices are supported, assumes that the domains of the features are normalized to [0,1]
# Based on explviz/plot/panel.py of the branch explviz
# The samples are numbered from 1 as in the engine

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from .. import Error
from ..interpret import slice as slice_

# The styles of the explanations, cycled (from explviz/style.py)
COLORS = ('red', 'darkorange', 'green', 'blue', 'purple', 'cyan')
FILL_ALPHAS = (0.20, 0.25, 0.35, 0.15, 0.30, 0.70)
# The line widths are decreasing to the minimal one, so that coinciding outlines stay visible
MIN_LINE_WIDTH = 2
SAMPLE_COLOR = 'black'
MARGIN = 0.05
# The decision regions: light colors different from those of the explanations, cycled
CLASS_COLORS = ('navajowhite', 'lightskyblue', 'palegreen', 'thistle', 'lightgray', 'lightpink', 'paleturquoise',
                'wheat', 'lavender', 'peachpuff')
CLASS_ALPHA = 0.5
BOUNDARY_COLOR = 'black'
# The number of the points of the grid in each dimension
RESOLUTION = 300


class Config:
    """The configuration of the visualization given by its parameters."""

    def __init__(self, params, n_vars):
        self.var_indices = None
        # The domains of the features in the real units, otherwise normalized
        self.domains = None
        # The names of the features, or None
        self.names = None
        # The classifier for the decision regions, or None, and the names of the classes
        self.model = None
        self.class_names = None

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


def _axis_label(i, names):
    name = names[i] if names else None
    var = f'$x_{{{i + 1}}}$'
    return f'{name} ({var})' if name else var


def _draw_decision_regions(axes, sample, config):
    """The classes of the grid of points on the plane of the features through the sample."""
    a, b = config.var_indices
    grid = np.linspace(0, 1, RESOLUTION)
    xx, yy = np.meshgrid(grid, grid)
    X = np.tile(sample, (xx.size, 1))
    X[:, a], X[:, b] = xx.ravel(), yy.ravel()
    labels = config.model.classify(X).reshape(xx.shape)

    n_classes = config.model.n_classes()
    colors = [CLASS_COLORS[c % len(CLASS_COLORS)] for c in range(n_classes)]
    axes.imshow(labels, extent=(0, 1, 0, 1), origin='lower', interpolation='nearest', cmap=ListedColormap(colors),
                vmin=-0.5, vmax=n_classes - 0.5, alpha=CLASS_ALPHA, zorder=0)
    axes.contour(xx, yy, labels, levels=np.arange(n_classes - 1) + 0.5, colors=BOUNDARY_COLOR, linewidths=1.2,
                 zorder=0.5)

    names = config.class_names or [f'Class {c}' for c in range(n_classes)]
    return [Patch(facecolor=colors[c], alpha=CLASS_ALPHA, label=names[c]) for c in np.unique(labels)]


def _class_name(c, config):
    return config.class_names[c] if config.class_names else str(c)


def caption(idx, expected, sample, config):
    """The number of the sample and its computed class if the classifier is given, with the expected class if different,
    otherwise only the expected class."""
    if not config.model:
        return f'Sample {idx} (expected class: {_class_name(expected, config)})'
    computed = int(config.model.classify(sample[None, :])[0])
    title = f'Sample {idx} class {_class_name(computed, config)}'
    if computed != expected:
        title += f' (expected: {_class_name(expected, config)})'
    return title


def save(series, sample, config, file, title=None, show=False):
    """Save the explanations and the sample.

    Each of the series of the explanations is (label, vertices of the region of the slice, diameter or None,
    whether the explanation is invalid); the diameter is given by a pair of points, all points with all features.
    """
    a, b = config.var_indices
    single = len(series) == 1
    fig, axes = plt.subplots()

    class_handles = _draw_decision_regions(axes, sample, config) if config.model else []

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
    axes.set_xlabel(_axis_label(a, config.names))
    axes.set_ylabel(_axis_label(b, config.names))
    if config.domains:
        # Only the ticks are in the real units, with round values
        for axis, i in ((axes.xaxis, a), (axes.yaxis, b)):
            lo, hi = config.domains.lower[i], config.domains.upper[i]
            ticks = [v for v in MaxNLocator().tick_values(lo, hi) if lo <= v <= hi]
            axis.set_ticks([(v - lo) / (hi - lo) for v in ticks], [f'{v:g}' for v in ticks])
    axes.grid(alpha=0.3)
    if title:
        axes.set_title(title)
    handles, _ = axes.get_legend_handles_labels()
    axes.legend(handles=handles + class_handles, loc='best')

    fig.savefig(file)
    if show:
        plt.show()
    plt.close(fig)
