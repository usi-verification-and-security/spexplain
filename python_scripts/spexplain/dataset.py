# Datasets in the CSV format: an optional header, then one sample per line
# with the values of the features followed by the expected classification

import csv

import numpy as np

from . import Error


class Dataset:
    def __init__(self, path):
        samples = []
        labels = []
        with open(path, newline='') as f:
            for i, row in enumerate(csv.reader(f)):
                try:
                    samples.append([float(x) for x in row[:-1]])
                    labels.append(int(float(row[-1])))
                except ValueError:
                    # The header
                    if i == 0:
                        continue
                    raise Error(f'Invalid sample at line {i + 1} of the dataset: {path}')

        self.path = path
        self.samples = np.array(samples)
        self.labels = np.array(labels)

    def __len__(self):
        return len(self.samples)

    def n_features(self):
        return self.samples.shape[1]

    def check_sample_idx(self, idx):
        """Samples are indexed from 1, same as in the engine."""
        if not 1 <= idx <= len(self):
            raise Error(f'Sample {idx} out of range of the dataset with {len(self)} samples: {self.path}')

    def get_sample(self, idx):
        self.check_sample_idx(idx)
        return self.samples[idx - 1]
