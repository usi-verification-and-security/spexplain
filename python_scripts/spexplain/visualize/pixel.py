# Visualization pixel: shows a point as an image
# Assumes square images with a single channel (gray scale, as in MNIST), with values in [0,1]

import math

import matplotlib.pyplot as plt

from .. import Error

# The size of a panel in inches, and the space between the panels relative to their width
PANEL_SIZE = 2.4
PANEL_SPACE = 0.05


class Config:
    """The configuration of the visualization given by its parameters."""

    def __init__(self, params):
        # The sample in between the points, otherwise first
        self.center = False

        names = set()
        for param in params:
            name, *args = param.split()
            if name in names:
                raise Error(f'Duplicate parameter of pixel: {name}')
            names.add(name)
            if name == 'center' and not args:
                self.center = True
            else:
                raise Error(f'Unknown parameter of pixel: {param}')


def image_shape(n_features):
    side = math.isqrt(n_features)
    if side * side != n_features:
        raise Error(f'Visualization pixel requires square images with a single channel, got {n_features} features')
    return side, side


def save(panels, file, show=False):
    """Save the points side by side as an image into the file, and optionally also show it.

    Each of the panels is (title, point), the point is None if the explanation is invalid (an empty panel).
    """
    shape = next(image_shape(point.size) for _, point in panels if point is not None)
    fig, axes_row = plt.subplots(1, len(panels), figsize=(PANEL_SIZE * len(panels), PANEL_SIZE), squeeze=False,
                                 gridspec_kw={'wspace': PANEL_SPACE})
    for axes, (title, point) in zip(axes_row[0], panels):
        if point is not None:
            axes.imshow(point.reshape(shape), aspect='equal', interpolation='nearest', cmap='gray', vmin=0.0, vmax=1.0)
        else:
            # The same extent as of the images
            axes.set_xlim(-0.5, shape[1] - 0.5)
            axes.set_ylim(shape[0] - 0.5, -0.5)
            axes.set_aspect('equal')
        axes.set_axis_off()
        axes.set_title(title)

    # Without the margins
    fig.savefig(file, bbox_inches='tight', pad_inches=0.05)
    if show:
        plt.show()
    plt.close(fig)
