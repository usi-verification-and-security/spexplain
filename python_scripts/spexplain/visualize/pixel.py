# Visualization pixel: shows a point as an image
# Assumes square images with a single channel (gray scale, as in MNIST), with values in [0,1]

import math

import matplotlib.pyplot as plt

from .. import Error

# The size of a panel in inches, and the space between the panels relative to their width
PANEL_SIZE = 2.4
PANEL_SPACE = 0.05


def image_shape(n_features):
    side = math.isqrt(n_features)
    if side * side != n_features:
        raise Error(f'Visualization pixel requires square images with a single channel, got {n_features} features')
    return side, side


def save(panels, file, show=False):
    """Save the points side by side as an image into the file, and optionally also show it.

    Each of the panels is (title, point).
    """
    fig, axes_row = plt.subplots(1, len(panels), figsize=(PANEL_SIZE * len(panels), PANEL_SIZE), squeeze=False,
                                 gridspec_kw={'wspace': PANEL_SPACE})
    for axes, (title, point) in zip(axes_row[0], panels):
        axes.imshow(point.reshape(image_shape(point.size)), aspect='equal', interpolation='nearest', cmap='gray',
                    vmin=0.0, vmax=1.0)
        axes.set_axis_off()
        axes.set_title(title)

    # Without the margins
    fig.savefig(file, bbox_inches='tight', pad_inches=0.05)
    if show:
        plt.show()
    plt.close(fig)
