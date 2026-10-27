#!/usr/bin/env python3
"""
One explanation of VeriX-plus (in VERIX_DIR) of a sample of a dataset, as in the original experiments
(their scripts mnist2.py, gtsrb2.py and heart_attack50.py, merged), for the fully connected networks of SpEXplAIn.

Usage
-----
  python verix1.py <name> <model.nnet> <dataset.csv> <index> <epsilon> <results dir>

<name> is the benchmark (see BENCHMARKS) and <index> the sample of the dataset (from 0).
The model is read from <model>.onnx (see nnet_to_onnx.py) and <model>.nnet.
The results go to <results dir>/<index>/: sat.txt and unsat.txt of VeriX, or robust.txt
if the network is robust already with all features perturbed (see to_phi.py).
The indices of VeriX are of the pixels, each with all its channels, or of the features of tabular data.
The label is the last column of the dataset.
"""

import importlib.abc
import importlib.util
import os
import sys
import time
import types

import numpy as np
import pandas as pd
import torch

VERIX_DIR = os.environ.get('VERIX_DIR', os.path.expanduser('~/VeriX-plus'))
# VeriX.py imports maraboupy from 'Marabou' relative to the working directory
sys.path[:0] = [VERIX_DIR, os.path.join(VERIX_DIR, 'Marabou')]

# <name>: the dataset of VeriX-plus and the number of channels (see input_shape)
BENCHMARKS = {
    'heart_attack': ('tabular', 1),
    'obesity': ('tabular', 1),
    'loan': ('tabular', 1),
    'mnist': ('mnist', 1),
    'cifar': ('cifar10', 3),
    'gtsrb': ('gtsrb', 3),
}

# The original experiments traversed the features in the random order (with the seed 0 of VeriX)
ORDER = os.environ.get('VERIX_ORDER', 'random')
PROCEDURE = 'binary-search'


def input_shape(verix_dataset, n_features, channels):
    """The shape of a sample for VeriX, without the batch dimension: the images are square,
    the tabular samples are (<features>, 1, 1), as in the original experiments."""
    if verix_dataset == 'tabular':
        return (n_features, 1, 1)
    size = int(round(np.sqrt(n_features // channels)))
    return (size, size, channels)


def interleaved_order(n_features, channels):
    """The features of the datasets and the inputs of the networks are channel after channel (C, H, W),
    but VeriX takes the channels of each pixel together (H, W, C): the indices of the features in its order."""
    return np.arange(n_features).reshape(channels, -1).T.reshape(-1)


class _Unavailable:
    def __init__(self, name):
        self._name = name

    def __call__(self, *args, **kwargs):
        raise RuntimeError(f'{self._name} is not available: auto_LiRPA is not installed (it is only for CNNs)')


class _StubModule(types.ModuleType):
    def __getattr__(self, name):
        if name.startswith('__'):
            raise AttributeError(name)
        return _Unavailable(f'{self.__name__}.{name}')


class _StubFinder(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    def find_spec(self, fullname, path, target=None):
        if fullname == 'auto_LiRPA' or fullname.startswith('auto_LiRPA.'):
            return importlib.util.spec_from_loader(fullname, self, is_package=True)
        return None

    def create_module(self, spec):
        return _StubModule(spec.name)

    def exec_module(self, module):
        pass


def stub_auto_lirpa():
    """VeriX.py imports auto_LiRPA, which it uses only for CNNs, and which is not installed:
    import a stand-in, whose names fail only when used."""
    try:
        import auto_LiRPA  # noqa: F401
    except ImportError:
        sys.meta_path.insert(0, _StubFinder())


def fail_on_marabou_errors():
    """VeriX treats only 'sat' and 'TIMEOUT' as not robust, it would take an error of Marabou as robust;
    make it fail instead."""
    from maraboupy import MarabouNetwork

    solve = MarabouNetwork.MarabouNetwork.solve

    def checked_solve(self, *args, **kwargs):
        result = solve(self, *args, **kwargs)
        if result[0] not in ('sat', 'unsat', 'TIMEOUT'):
            raise RuntimeError(f'Marabou failed: {result[0]}')
        return result

    MarabouNetwork.MarabouNetwork.solve = checked_solve


def make_model(model, model_dir, shape):
    """VeriX reads <model_path>.onnx, and <model_path>.pt into model_class() (a torch.nn.Sequential for the bounds
    of its traversal order), and it recognizes fully connected networks by 'fc' in the name of the model.
    Both are made from the .nnet (some .pth differ from it, e.g. mnist_200x9), with the inputs in the order
    of VeriX (see interleaved_order) and of the shape of its samples (see input_shape),
    as the models of the original experiments (e.g. mnist_50x1_marabou-fc)."""
    import onnx
    from onnx import numpy_helper
    from nnet_to_onnx import parse_nnet

    base = os.path.splitext(model)[0]
    model_path = os.path.join(model_dir, os.path.basename(base) + '-fc')

    net = parse_nnet(model)
    # The torch model is without the normalization of the .nnet
    if np.any(net['means'] != 0) or np.any(net['ranges'] != 1):
        raise ValueError(f'{model}: the normalization is not the identity')
    sizes = net['layer_sizes']
    order = interleaved_order(sizes[0], shape[2])

    class FC(torch.nn.Sequential):
        def __init__(self):
            layers = []
            for i in range(len(sizes) - 1):
                layers.append(torch.nn.Linear(sizes[i], sizes[i + 1]))
                if i < len(sizes) - 2:
                    layers.append(torch.nn.ReLU())
            super().__init__(*layers)

    # The .pt last, its existence means that both are made
    if not os.path.exists(model_path + '.pt'):
        os.makedirs(model_dir, exist_ok=True)

        onnx_model = onnx.load(base + '.onnx')
        graph = onnx_model.graph
        # The weights of the first layer are transposed (see nnet_to_onnx.py)
        w0 = next(t for t in graph.initializer if t.name == 'W_0')
        w0.CopyFrom(numpy_helper.from_array(numpy_helper.to_array(w0)[order], name='W_0'))
        # VeriX runs the model on the sample with a batch dimension, e.g. (1, 28, 28, 1)
        for node in graph.node:
            node.input[:] = ['input_flat' if i == 'input' else i for i in node.input]
        graph.node.insert(0, onnx.helper.make_node('Flatten', ['input'], ['input_flat'], axis=1))
        graph.input[0].CopyFrom(onnx.helper.make_tensor_value_info(
            'input', onnx.TensorProto.FLOAT, [1, *shape]))
        onnx.checker.check_model(onnx_model)
        onnx.save(onnx_model, model_path + '.onnx')

        state = {}
        for i, (w, b) in enumerate(zip(net['weights'], net['biases'])):
            if i == 0:
                w = w[:, order]
            state[f'{2 * i}.weight'] = torch.from_numpy(np.ascontiguousarray(w))
            state[f'{2 * i}.bias'] = torch.from_numpy(b)
        torch.save(state, model_path + '.pt')

    return model_path, FC


def main():
    if len(sys.argv) != 7:
        print(__doc__, file=sys.stderr)
        sys.exit(1)

    name, model, dataset, index, epsilon, results_dir = sys.argv[1:]
    index = int(index)
    epsilon = float(epsilon)

    verix_dataset, channels = BENCHMARKS[name]

    df = pd.read_csv(dataset)
    # The datasets are already scaled to [0,1], the label is the last column
    features = df.iloc[:, :-1].values
    n_features = features.shape[1]
    shape = input_shape(verix_dataset, n_features, channels)
    x_test = features[:, interleaved_order(n_features, channels)]
    x_test = x_test.reshape(-1, *shape).astype('float32')

    result_dir = os.path.join(results_dir, str(index)) + '/'
    os.makedirs(result_dir, exist_ok=True)

    stub_auto_lirpa()
    fail_on_marabou_errors()
    model_path, model_class = make_model(model, os.path.join(results_dir, 'models'), shape)

    from VeriX import VeriX

    verix_tick = time.time()

    verix = VeriX(
        dataset=verix_dataset,
        image=x_test[index],
        model_path=model_path,
        directory=result_dir,
        model_class=model_class,
        plot_original=False,
    )

    if verix.check_robust(epsilon=epsilon):
        print(f"Model is robust to {epsilon}-perturbation.")
        with open(result_dir + 'robust.txt', 'w') as f:
            f.write(str(index) + '\n')
    else:
        verix.traversal_order(
            traverse=ORDER,
            epsilon=epsilon,
        )

        verix.compute_explanation(
            procedure=PROCEDURE,
            use_confidence_ranking=True,
            epsilon=epsilon,
        )

    verix_time = time.time() - verix_tick
    with open(result_dir + 'time.txt', 'w') as f:
        f.write(str(verix_time) + '\n')


if __name__ == '__main__':
    main()
