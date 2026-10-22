# Technique diam: the diameter of the slice of an explanation (see slice) within the domains [0,1]^2,
# i.e. the segment between the two farthest points of the region
# Only 2D slices and the maximal diameter are supported

import numpy as np

from .. import Error
from . import region2d
from . import slice as slice_


class Config:
    """The configuration of the technique given by its parameters."""

    def __init__(self, params, n_vars):
        self.var_indices = None
        # Only the maximal diameter is implemented
        self.maximal = True

        names = set()
        for param in params:
            name, *args = param.split()
            if name in names:
                raise Error(f'Duplicate parameter of diam: {name}')
            names.add(name)
            if name == 'vars':
                self.var_indices = slice_.parse_vars(args, n_vars, 'diam')
            elif name == 'max' and not args:
                self.maximal = True
            else:
                raise Error(f'Unknown parameter of diam: {param}')

        if self.var_indices is None:
            raise Error('Technique diam requires the parameter vars')


def interpret(explanation, config, sample):
    """The endpoints of the diameter of the slice; the features outside of the slice are those of the sample."""
    A, B = slice_.interpret(explanation, config, sample)
    pts = region2d.vertices(A[:, config.var_indices], B)
    if len(pts) == 0:
        raise Error('The slice of the explanation is empty')

    p, q = sample.copy(), sample.copy()
    p[config.var_indices], q[config.var_indices] = region2d.farthest_pair(pts)
    # Deterministic orientation: the first varying feature increases
    diff = q - p
    varying = np.flatnonzero(diff)
    if varying.size and diff[varying[0]] < 0:
        p, q = q, p
    return p, q
