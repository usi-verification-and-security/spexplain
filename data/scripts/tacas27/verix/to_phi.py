#!/usr/bin/env python3
"""
The explanations of VeriX-plus (verix1.py) as formulas over the variables x1, ..., xN as in the engine,
one line per sample of the dataset, in its order (to stdout).

Usage
-----
  python to_phi.py <dataset.csv> <epsilon> <results dir> [<channels>]

The results of the sample <idx> (from 0) are in <results dir>/<idx>/:
  sat.txt and unsat.txt   the explanation (the pixels fixed to the sample) and the irrelevant pixels
                          (the pixels perturbed by epsilon, within [0,1]), as indices from 0;
                          a pixel <i> is the features <c>*<pixels>+<i> of its channels <c> (default: 1 channel),
                          as the features are channel after channel
  robust.txt              robust already with all features perturbed, i.e. no feature is fixed
  otherwise               no explanation (e.g. the timeout), written as '<null>'
"""

import csv
import os
import sys

INVALID_EXPLANATION = '<null>'


def read_indices(path):
    with open(path) as f:
        return [int(line) for line in f if line.strip()]


def read_samples(dataset):
    """The features of the samples, without the label, which is the last column."""
    with open(dataset, newline='') as f:
        rows = list(csv.reader(f))
    return [[float(x) for x in row[:-1]] for row in rows[1:]]


def var(i):
    return f'x{i + 1}'


def formula(sample, epsilon, sat, unsat):
    atoms = [f'(= {var(i)} {sample[i]})' for i in sat]
    for i in unsat:
        atoms.append(f'(>= {var(i)} {max(sample[i] - epsilon, 0.0)})')
        atoms.append(f'(<= {var(i)} {min(sample[i] + epsilon, 1.0)})')
    return '(and' + ''.join(atoms) + ')'


def features(pixels, n_pixels, channels):
    return [c * n_pixels + i for i in pixels for c in range(channels)]


def explanation(sample, epsilon, sample_dir, channels):
    n_pixels = len(sample) // channels
    sat_file = os.path.join(sample_dir, 'sat.txt')
    unsat_file = os.path.join(sample_dir, 'unsat.txt')
    if os.path.exists(sat_file) and os.path.exists(unsat_file):
        sat = read_indices(sat_file)
        unsat = read_indices(unsat_file)
    elif os.path.exists(os.path.join(sample_dir, 'robust.txt')):
        sat = []
        unsat = list(range(n_pixels))
    else:
        return INVALID_EXPLANATION

    # Each pixel is either fixed or perturbed
    if sorted(sat + unsat) != list(range(n_pixels)):
        raise ValueError(f'{sample_dir}: sat.txt and unsat.txt do not partition the {n_pixels} pixels')

    return formula(sample, epsilon, features(sat, n_pixels, channels), features(unsat, n_pixels, channels))


def main():
    if len(sys.argv) not in (4, 5):
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    dataset, epsilon, results_dir = sys.argv[1], float(sys.argv[2]), sys.argv[3]
    channels = int(sys.argv[4]) if len(sys.argv) == 5 else 1

    for idx, sample in enumerate(read_samples(dataset)):
        print(explanation(sample, epsilon, os.path.join(results_dir, str(idx)), channels))


if __name__ == '__main__':
    main()
