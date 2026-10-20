# Files with sample neuron activations, used by the engine via
# --input-fix-sample-neuron-activations and --input-guide-sample-neuron-activations

import contextlib
import os
import sys
from pathlib import Path

from . import ROOT_DIR, Error

ADVERSARIAL_DIR = ROOT_DIR / 'python_scripts' / 'adversarial'


def default_file(model):
    """The permanent activation file that belongs to the model.

    Same as in data/scripts/run1.sh: 'models/' -> 'neuron_activations/' and the extension -> '.txt',
    e.g. data/models/mnist/m.nnet -> data/neuron_activations/mnist/m.txt
    """
    model = str(model)
    if 'models/' not in model:
        raise Error(f"Cannot determine the neuron activations file: the model path does not contain 'models/': {model}")
    return Path(model.replace('models/', 'neuron_activations/', 1)).with_suffix('.txt')


def has_scaled_inputs(model):
    """Whether all input bounds of the NNet model are [0,1]."""
    with open(model) as f:
        lines = (line for line in f if not line.startswith('//'))
        # The input bounds follow the header lines with sizes and the unused flag
        for _ in range(3):
            next(lines)
        mins, maxes = ([float(v) for v in next(lines).strip().strip(',').split(',')] for _ in range(2))
    return all(v == 0 for v in mins) and all(v == 1 for v in maxes)


def generate_adversarial(model, dataset, output_file):
    """Compute the hybrid configuration of neuron activations to be fixed, based on adversarial counterexamples.

    Writes the activations of neurons that do not change in the counterexamples into the output file.
    Requires the PyTorch model (.pth) next to the NNet model, training is not part of spexplain.
    """
    model = Path(model)
    pytorch_model = model.with_suffix('.pth')
    if not pytorch_model.exists():
        raise Error(f'Generating adversarial neuron activations requires the PyTorch model: {pytorch_model}\n'
                    f'Alternatively, provide the neuron activations file: {output_file}')

    # Loading PyTorch and the model takes a while
    print(f'Writing adversarial neuron activations to: {output_file}', flush=True)

    # The scripts in python_scripts/adversarial are also used standalone
    for path in (ROOT_DIR, ADVERSARIAL_DIR):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    try:
        import activation_diff
    except ImportError as e:
        raise Error(f'Generating adversarial neuron activations requires PyTorch: {e}')

    # The task is the directory of the model, e.g. data/models/mnist/m.nnet -> mnist
    model_task = model.parent.name
    scaled = has_scaled_inputs(model)

    try:
        # The outputs of the script are not relevant here
        with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull), contextlib.redirect_stderr(devnull):
            activation_diff.compute_activation_changes(str(pytorch_model), str(dataset), str(output_file), model_task,
                                                       scaled)
    except BaseException:
        # Do not leave an incomplete file behind, it would be used next time
        Path(output_file).unlink(missing_ok=True)
        raise
