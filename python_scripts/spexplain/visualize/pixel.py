# Visualization pixel: shows a point as an image
# Assumes square images with a single channel (gray scale, as in MNIST), with values in [0,1]

import math

import matplotlib.pyplot as plt

from .. import Error


def image_shape(n_features):
    side = math.isqrt(n_features)
    if side * side != n_features:
        raise Error(f'Visualization pixel requires square images with a single channel, got {n_features} features')
    return side, side


def save(point, file, show=False):
    """Save the point as an image into the file, and optionally also show it."""
    fig, axes = plt.subplots()
    axes.imshow(point.reshape(image_shape(point.size)), aspect='equal', interpolation='nearest', cmap='gray',
                vmin=0.0, vmax=1.0)
    axes.set_axis_off()

    fig.savefig(file)
    if show:
        plt.show()
    plt.close(fig)
