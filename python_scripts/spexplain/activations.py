# Files with sample neuron activations, used by the engine via
# --input-fix-sample-neuron-activations and --input-guide-sample-neuron-activations

from pathlib import Path

from . import Error


def default_file(model):
    """The permanent activation file that belongs to the model.

    Same as in data/scripts/run1.sh: 'models/' -> 'neuron_activations/' and the extension -> '.txt',
    e.g. data/models/mnist/m.nnet -> data/neuron_activations/mnist/m.txt
    """
    model = str(model)
    if 'models/' not in model:
        raise Error(f"Cannot determine the neuron activations file: the model path does not contain 'models/': {model}")
    return Path(model.replace('models/', 'neuron_activations/', 1)).with_suffix('.txt')


def generate_adversarial(model, dataset, output_file):
    """Compute the hybrid configuration of neuron activations to be fixed, based on adversarial counterexamples.

    Writes the activations of neurons that do not change in the counterexamples into the output file.
    """
    #+ integrate python_scripts/contrastive-guide/activation_diff.py
    raise Error(f'Generating adversarial neuron activations is not implemented yet: {output_file} does not exist')
