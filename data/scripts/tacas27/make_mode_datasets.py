#!/usr/bin/env python3
"""Create the smaller datasets of the runme.sh modes from the <name>_s100_scaled.csv datasets.

For every benchmark and mode, writes data/datasets/<name>/<name>_s100_scaled_<mode>.csv with
the important samples of the benchmark plus random ones (fixed seed), in their original order.
The full mode uses <name>_s100_scaled.csv itself.

All maps from the sample indices of the mode datasets to the original indices (both 1-based)
are written to data/datasets/mode_sample_maps.csv: benchmark,mode,index,original_index

The neuron activations of the samples (used by the hybrid variant) are indexed by the samples
of the dataset, so they are renumbered for every mode dataset as well:
data/neuron_activations/<name>/s100_scaled_<mode>/<name>_<width>x<depth>.txt
All blocks of an original sample are kept, also when a file lists it more than once,
and a sample without a block has none, so the activations behave as with the original dataset.

USAGE: make_mode_datasets.py
"""

import csv
import random
import re
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2]
DATASETS_DIR = DATA_DIR / 'datasets'
ACTIVATIONS_DIR = DATA_DIR / 'neuron_activations'
MAP_FILE = DATASETS_DIR / 'mode_sample_maps.csv'

SEED = 2027

BENCHMARKS = ['heart_attack', 'obesity', 'loan', 'mnist', 'cifar', 'gtsrb']

## Samples (1-based) that every mode dataset of the benchmark must contain
IMPORTANT = {
    'heart_attack': [98, 63, 84, 55, 21, 90],
    'loan': [8, 70],
    'mnist': [1, 7],
}
## The smoke test needs only these of them
IMPORTANT_SMOKE = {
    'heart_attack': IMPORTANT['heart_attack'],
    'loan': IMPORTANT['loan'],
    'mnist': [1],
}

SHORT_SIZE = 8
## Benchmarks without important samples still get this many random samples in smoke
SMOKE_MIN_SIZE = 1
## The smoke test skips these benchmarks
SMOKE_SKIPPED = {'cifar', 'gtsrb'}


def read_dataset(name):
    lines = (DATASETS_DIR / name / f'{name}_s100_scaled.csv').read_text().splitlines()
    return lines[0], lines[1:]


def pick(important, size, n_samples, rng):
    """The important samples plus random others up to <size>, sorted."""
    assert all(1 <= idx <= n_samples for idx in important), important
    others = [idx for idx in range(1, n_samples + 1) if idx not in important]
    n_random = max(0, size - len(important))
    return sorted(important + rng.sample(others, n_random))


def read_activation_blocks(path, depth):
    """[(sample index, [one line of nodes per hidden layer])], in the order of the file."""
    tokens = path.read_text().split()
    blocks = []
    pos = 0
    while pos < len(tokens):
        idx = int(tokens[pos])
        pos += 1
        layers = []
        for _ in range(depth):
            end = tokens.index('0', pos)
            layers.append(' '.join(tokens[pos:end + 1]))
            pos = end + 1
        blocks.append((idx, layers))
    return blocks


def write_mode_activations(name, mode, selection):
    """Renumbered activations of all <name>_<width>x<depth> models for the mode dataset."""
    out_dir = ACTIVATIONS_DIR / name / f's100_scaled_{mode}'
    out_dir.mkdir(exist_ok=True)
    n_files = 0
    for path in sorted((ACTIVATIONS_DIR / name).glob(f'{name}_*x*.txt')):
        match = re.fullmatch(rf'{name}_\d+x(\d+)\.txt', path.name)
        if not match:
            continue
        blocks = read_activation_blocks(path, int(match.group(1)))
        lines = []
        for idx, orig in enumerate(selection, start=1):
            for block_idx, layers in blocks:
                if block_idx == orig:
                    lines += [str(idx)] + layers + ['']
        (out_dir / path.name).write_text('\n'.join(lines) + ('\n' if lines else ''))
        n_files += 1
    return n_files


def main():
    rng = random.Random(SEED)
    map_rows = []

    for name in BENCHMARKS:
        header, samples = read_dataset(name)
        n_samples = len(samples)

        short = pick(IMPORTANT.get(name, []), SHORT_SIZE, n_samples, rng)
        selections = {
            'short': short,
            ## same samples as short, so that the two modes differ only in the timeouts
            'quick': short,
            'full': list(range(1, n_samples + 1)),
        }
        if name not in SMOKE_SKIPPED:
            selections['smoke'] = pick(IMPORTANT_SMOKE.get(name, []), SMOKE_MIN_SIZE, n_samples, rng)

        for mode in ['smoke', 'quick', 'short', 'full']:
            if mode not in selections:
                continue
            selection = selections[mode]
            if mode != 'full':
                out = DATASETS_DIR / name / f'{name}_s100_scaled_{mode}.csv'
                out.write_text('\n'.join([header] + [samples[idx - 1] for idx in selection]) + '\n')
                n_activations = write_mode_activations(name, mode, selection)
            map_rows += [(name, mode, idx, orig) for idx, orig in enumerate(selection, start=1)]
            if mode == 'full':
                print(f'{name:13} {mode:6} {len(selection):3} samples: all (the original dataset)')
            else:
                print(f'{name:13} {mode:6} {len(selection):3} samples: {selection}, '
                      f'activations of {n_activations} models')

    with MAP_FILE.open('w', newline='') as f:
        writer = csv.writer(f, lineterminator='\n')
        writer.writerow(['benchmark', 'mode', 'index', 'original_index'])
        writer.writerows(map_rows)
    print(f'Map: {MAP_FILE}')


if __name__ == '__main__':
    main()
