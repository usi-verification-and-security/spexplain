"""The class regions behind an explanation: where the model predicts what.

The old scripts drew this as a cloud of a few thousand uniformly random points
classified one at a time in a Python loop, and left the decision boundary to be
inferred from the gap between two colours.  Here the default is a grid
evaluated in a single batched forward pass, which is both faster and yields an
actual boundary contour.  The scatter is kept for reproducing older figures.
"""

from __future__ import annotations

from collections import namedtuple

import numpy as np

GridBackground = namedtuple("GridBackground", "xx yy proba")
SampleBackground = namedtuple("SampleBackground", "class0 class1")


def _tile_point(point, axes, XA, XB):
    """Replicate ``point`` over a set of (x_a, x_b) values."""
    a, b = axes
    n = len(point)
    X = np.tile(np.asarray(point, dtype=float).ravel()[:n], (XA.size, 1))
    X[:, a] = XA.ravel()
    X[:, b] = XB.ravel()
    return X


def class_region_grid(clf, point, axes, bounds, resolution: int = 300) -> GridBackground:
    """Evaluate P(class 1) on a grid over the two free axes.

    Every other coordinate stays at the sample point's value, so this is the
    model restricted to exactly the plane the snapshot is drawn on.
    """
    a, b = axes
    xs = np.linspace(bounds[a][0], bounds[a][1], resolution)
    ys = np.linspace(bounds[b][0], bounds[b][1], resolution)
    xx, yy = np.meshgrid(xs, ys)
    proba = clf.predict_proba(_tile_point(point, axes, xx, yy)).reshape(xx.shape)
    return GridBackground(xx, yy, proba)


def class_region_samples(clf, point, axes, bounds, num_samples: int = 5000, seed=None
                         ) -> SampleBackground:
    """Uniformly sample the 2D slice and split the points by predicted class.

    Same construction as the original ``generate_samples``, but the forward
    pass is batched rather than run once per point.
    """
    a, b = axes
    rng = np.random.default_rng(seed)
    sa = rng.uniform(bounds[a][0], bounds[a][1], num_samples)
    sb = rng.uniform(bounds[b][0], bounds[b][1], num_samples)
    pred = clf.predict(_tile_point(point, axes, sa, sb))
    pts = np.column_stack([sa, sb])
    return SampleBackground(class0=pts[pred == 0], class1=pts[pred == 1])
