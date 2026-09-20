"""Reader for the Stanford/Reluplex ``.nnet`` text format."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def read_nnet(path, with_norm: bool = False):
    """Read a ``.nnet`` file into dense weight matrices and bias vectors.

    Returns ``(weights, biases)``, or additionally
    ``(input_mins, input_maxes, means, ranges)`` when ``with_norm`` is set.
    """
    with open(Path(path), "r") as f:
        line = f.readline()
        while line[:2] == "//":
            line = f.readline()

        num_layers, _input_size = (int(x) for x in line.split(",")[:2])
        layer_sizes = [int(x) for x in f.readline().split(",")[: num_layers + 1]]

        f.readline()  # obsolete parameter line

        input_mins = [float(x) for x in f.readline().strip().split(",") if x]
        input_maxes = [float(x) for x in f.readline().strip().split(",") if x]
        means = [float(x) for x in f.readline().strip().split(",") if x]
        ranges = [float(x) for x in f.readline().strip().split(",") if x]

        weights, biases = [], []
        for k in range(num_layers):
            n_in, n_out = layer_sizes[k], layer_sizes[k + 1]
            W = np.zeros((n_out, n_in))
            for i in range(n_out):
                row = [float(x) for x in f.readline().strip().split(",")[:-1]]
                W[i, :] = row[:n_in]
            bvec = np.zeros(n_out)
            for i in range(n_out):
                bvec[i] = float(f.readline().strip().split(",")[0])
            weights.append(W)
            biases.append(bvec)

    if with_norm:
        return weights, biases, input_mins, input_maxes, means, ranges
    return weights, biases
