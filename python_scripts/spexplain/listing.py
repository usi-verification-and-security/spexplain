# Action list: lists the variables of a dataset, one row per item
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
}

# The options of all targets: (flags, keyword arguments, the targets where they are used);
# the others accept them but do not show them in their help
OPTIONS = [
    (('--domains',), dict(metavar='<file>',
                          help='CSV file with the domains of the features in the real units (columns Min, Max, '
                               'optionally Var, Feature), otherwise normalized to [0,1]'),
     ('vars',)),
    (('--vars',), dict(metavar='<var>[,<var>]...',
                       help="select the variables, each '<var>' is 'x<i>' or 'x<i>-x<j>' (from 1)"),
     ('vars',)),
]


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
    for flags, kwargs, targets in OPTIONS:
        if target not in targets:
            kwargs = dict(kwargs, help=argparse.SUPPRESS)
        parser.add_argument(*flags, **kwargs)
    return parser


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


def main(args):
    if not args or args[0] in ('-h', '--help'):
        print_usage()
        return 0
    target, args = args[0], args[1:]
    if target not in TARGETS:
        raise Error(f'Unknown target of list: {target}\nAvailable: {", ".join(TARGETS)}')
    opts = make_parser(target).parse_intermixed_args(args)

    if target == 'vars':
        list_vars(opts)

    return 0
