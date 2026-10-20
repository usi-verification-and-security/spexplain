# Action explain: runs the Explanation Computation Engine
# Options are forwarded to the engine, except of those that are processed here first

from . import Error, activations, engine

FIX_DEFAULT_OPT = '--fix-default-sample-neuron-activations'
INPUT_FIX_OPT = '--input-fix-sample-neuron-activations'

ADVERSARIAL = 'adversarial'


def pop_long_opt(args, opt):
    """Remove all occurrences of the long option with a value from args (in place), return the last value."""
    value = None
    i = 0
    while i < len(args):
        arg = args[i]
        if arg == opt and i + 1 < len(args):
            value = args[i + 1]
            del args[i:i + 2]
        elif arg.startswith(opt + '='):
            value = arg[len(opt) + 1:]
            del args[i]
        else:
            i += 1
    return value


def has_long_opt(args, opt):
    return any(arg == opt or arg.startswith(opt + '=') for arg in args)


def rewrite_args(args):
    """Translate the arguments of the user to the arguments of the engine."""
    # The engine expects the positional arguments first
    if len(args) < 2:
        return args
    model, dataset = args[0], args[1]

    new_args = list(args)
    fix_default = pop_long_opt(new_args, FIX_DEFAULT_OPT)
    if fix_default != ADVERSARIAL:
        return args
    args = new_args

    # The hybrid configuration: fix only the neuron activations that are not affected by adversarial counterexamples,
    # regardless of the default fixing of the engine; guiding uses the defaults of the engine
    if has_long_opt(args, INPUT_FIX_OPT):
        raise Error(f"Option '{FIX_DEFAULT_OPT} {ADVERSARIAL}' cannot be combined with '{INPUT_FIX_OPT}'")
    activations_file = activations.default_file(model)
    if not activations_file.exists():
        activations.generate_adversarial(model, dataset, activations_file)
    args += [FIX_DEFAULT_OPT, 'none', INPUT_FIX_OPT, str(activations_file)]
    return args


def main(args, engine_path=None):
    if not args or args[0] in ('-h', '--help'):
        print_usage(engine_path)
        return 0

    return engine.run(['explain', *rewrite_args(args)], engine_path)


def print_usage(engine_path=None):
    print('USAGE: spexplain [explain] <nn_model_fn> <dataset_fn> [<exp_strategies_spec>] [<options>]')
    print('Options processed by spexplain itself:')
    # Same layout as the options of the engine
    print(f'    {FIX_DEFAULT_OPT} {ADVERSARIAL}')
    print('              Fix the activations of neurons not affected by adversarial counterexamples (the hybrid configuration),')
    print('              stored in data/neuron_activations/... (computed if it does not exist yet,')
    print('              which requires PyTorch and the PyTorch model <nn_model_fn> with the extension .pth)')
    print('Other options are forwarded to the engine:\n')
    engine.print_usage(engine_path)
