# Action list: lists the variables or the classes of a dataset, one row per item
# Named listing, not list, which would shadow the built-in list where it is imported

import argparse
import re
import sys

from . import Error
from .cli import print_rows
from .interpret import warn

# The items that can be listed, with their arguments and descriptions
TARGETS = {
    'vars': ('[<var>...] <dataset>',
             'the variables x<i> of the dataset with the names and the domains of their features'),
    'classes': ('<dataset>', 'the classes of the dataset (from 0) with their names'),
}

# The options of all targets: (flags, keyword arguments, the targets where they are used, whether it is specific);
# the other targets accept them but do not show them in their help: a specific option selects or limits the listed
# items, so it is an error there, the others only add information, so they are ignored with a warning
OPTIONS = [
    (('--domains',), dict(metavar='<file>',
                          help='CSV file with the domains of the features in the real units (columns Min, Max, '
                               'optionally Var, Feature), otherwise normalized to [0,1]'),
     ('vars',), False),
    (('--vars',), dict(metavar='<var>[,<var>]...',
                       help="select the variables, each '<var>' is 'x<i>' or 'x<i>-x<j>' (from 1)"),
     ('vars',), True),
    (('--model',), dict(metavar='<nn_model_fn>',
                        help='the classifier in the NNet format, for the number of the classes'),
     ('classes',), False),
]


def info(msg):
    print(f'spexplain: info: {msg}', file=sys.stderr)


def print_usage(file=sys.stdout):
    print('usage: spexplain list <target> <args> [<options>]', file=file)
    print('TARGETS:', file=file)
    print_rows([(f'{target}:', args, desc) for target, (args, desc) in TARGETS.items()], file=file)
    print("Use 'spexplain list <target> --help' to print the usage of a particular target.", file=file)


def make_parser(target):
    args, desc = TARGETS[target]
    parser = argparse.ArgumentParser(prog=f'spexplain list {target}', usage=f'%(prog)s {args} [<options>]',
                                     description=f'List {desc}.')
    if target == 'vars':
        parser.add_argument('args', nargs='+', metavar='[<var>...] <dataset>',
                            help="the selected variables (default: all), each '<var>' is 'x<i>' or 'x<i>-x<j>' "
                                 "(from 1), as in the parameter vars of the techniques; the dataset in the CSV format")
    else:
        parser.add_argument('dataset', metavar='<dataset>', help='the dataset in the CSV format')
    for flags, kwargs, targets, _ in OPTIONS:
        if target not in targets:
            kwargs = dict(kwargs, help=argparse.SUPPRESS)
        parser.add_argument(*flags, **kwargs)
    return parser


def check_options(target, opts):
    """The options not used by the target: an error if specific, otherwise ignored with a warning."""
    for flags, _, targets, specific in OPTIONS:
        dest = flags[-1].lstrip('-').replace('-', '_')
        if target in targets or getattr(opts, dest) is None:
            continue
        if specific:
            raise Error(f'Option {flags[-1]} is not supported by list {target}')
        warn(f'Option {flags[-1]} is ignored by list {target}')


def split_selectors(values):
    """The selectors given by the arguments separated by spaces or commas."""
    return [sel for value in values for sel in re.split(r'[\s,]+', value) if sel]


def selectors_and_dataset(opts):
    """The positional selectors before the dataset, and the dataset."""
    *selectors, dataset_file = opts.args
    return split_selectors(selectors), dataset_file


def domain_str(lower, upper):
    return f'[{lower:g}, {upper:g}]'


def list_vars(opts):
    from .dataset import Dataset
    from .interpret import formula

    selectors, dataset_file = selectors_and_dataset(opts)
    if opts.vars is not None:
        if selectors:
            raise Error('Option --vars conflicts with the variables given as the arguments')
        selectors = split_selectors([opts.vars])
    dataset = Dataset(dataset_file)
    n_vars = dataset.n_features()
    indices = formula.parse_vars(selectors, n_vars, 'list vars') if selectors else range(n_vars)

    names = dataset.feature_names
    domains = None
    #+ the option --domains also for the other actions (interpret), instead of --visualization-domains
    if opts.domains is not None:
        from .domains import Domains
        domains = Domains(opts.domains, dataset)
        names = domains.names

    rows = [('Var', 'Feature', 'Domain')]
    for i in indices:
        domain = domain_str(domains.lower[i], domains.upper[i]) if domains else domain_str(0, 1)
        rows.append((f'x{i + 1}', names[i] or '-', domain))
    print_rows(rows)


def load_model(path, dataset):
    """The classifier in the NNet format, checked against the dataset."""
    from .model import Model

    model = Model(path)
    if model.n_inputs() != dataset.n_features():
        raise Error(f'The number of the inputs of the model ({model.n_inputs()}) does not match the number '
                    f'of the features of the dataset ({dataset.n_features()})')
    names = dataset.class_names
    if names is not None and len(names) != model.n_classes():
        raise Error(f'The number of the names of the classes in the header of the dataset ({len(names)}) '
                    f'does not match the number of the classes of the model ({model.n_classes()})')
    if dataset.n_classes() > model.n_classes():
        raise Error(f'The expected classes of the dataset ({dataset.n_classes()}) are not covered by the classes '
                    f'of the model ({model.n_classes()})')
    return model


def list_classes(opts):
    from .dataset import Dataset, default_class_name

    dataset = Dataset(opts.dataset)
    names = dataset.class_names
    # The number of the classes of the model, otherwise of the names in the header, otherwise estimated
    if opts.model is not None:
        n_classes = load_model(opts.model, dataset).n_classes()
    else:
        n_classes = dataset.n_classes()
        if names is None:
            info(f'The number of the classes ({n_classes}) is estimated by the largest expected class of the dataset, '
                 'give the model by --model to know it')
    rows = [('Class', 'Name')]
    for c in range(n_classes):
        rows.append((str(c), names[c] if names else default_class_name(c)))
    print_rows(rows)


def main(args):
    if not args or args[0] in ('-h', '--help'):
        print_usage()
        return 0
    target, args = args[0], args[1:]
    if target not in TARGETS:
        raise Error(f'Unknown target of list: {target}\nAvailable: {", ".join(TARGETS)}')
    opts = make_parser(target).parse_intermixed_args(args)
    check_options(target, opts)

    if target == 'vars':
        list_vars(opts)
    elif target == 'classes':
        list_classes(opts)

    return 0
