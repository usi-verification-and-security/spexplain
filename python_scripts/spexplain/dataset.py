# Datasets in the CSV format: an optional header with the names of the features, then one sample per line
# with the values of the features followed by the expected classification

import csv

import numpy as np

from . import Error

CLASS_NAMES_SEPARATOR = '|'


def default_class_name(c):
    """The name of the class c (from 0) if the names are not given."""
    return f'Class {c}'


class Dataset:
    def __init__(self, path):
        samples = []
        labels = []
        header = None
        with open(path, newline='') as f:
            for i, row in enumerate(csv.reader(f)):
                #+ do not require the expected class, as in the TODO in the engine
                try:
                    samples.append([float(x) for x in row[:-1]])
                    labels.append(int(float(row[-1])))
                except ValueError:
                    if i == 0:
                        header = row
                        continue
                    raise Error(f'Invalid sample at line {i + 1} of the dataset: {path}')

        self.path = path
        self.samples = np.array(samples)
        self.labels = np.array(labels)
        # The names of the features from the header, without the expected classification
        self.feature_names = [name.strip() or None for name in header[:-1]] if header else [None] * self.n_features()
        # The names of the classes, if given in the header of the expected classification separated by '|',
        # e.g. 'Low risk|High risk'
        self.class_names = ([name.strip() for name in header[-1].split(CLASS_NAMES_SEPARATOR)]
                            if header and CLASS_NAMES_SEPARATOR in header[-1] else None)

    def __len__(self):
        return len(self.samples)

    def n_features(self):
        return self.samples.shape[1]

    def n_classes(self):
        """The number of the names of the classes in the header, otherwise the largest expected class + 1."""
        n_expected = int(self.labels.max()) + 1 if len(self) else 0
        if self.class_names is None:
            return n_expected
        if len(self.class_names) < n_expected:
            raise Error(f'The number of the names of the classes in the header of the dataset '
                        f'({len(self.class_names)}) does not cover all the classes ({n_expected})')
        return len(self.class_names)

    def check_sample_idx(self, idx):
        """Samples are indexed from 1, same as in the engine."""
        if not 1 <= idx <= len(self):
            raise Error(f'Sample {idx} out of range of the dataset with {len(self)} samples: {self.path}')

    def get_sample(self, idx):
        self.check_sample_idx(idx)
        return self.samples[idx - 1]
