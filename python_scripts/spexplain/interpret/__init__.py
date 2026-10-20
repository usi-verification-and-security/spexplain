# Action interpret: runs the Explanation Interpreter on a stored explanation

import argparse

from .. import Error

# Extraction techniques, each determines its default visualization
TECHNIQUES = {
    'slice': 'fspace',
    'diam': 'pixel',
    'norm': 'pixel',
}

# Visualization techniques that can also be applied directly
VISUALIZATIONS = ('fspace', 'pixel')


def parse_technique(spec):
    """Parse the technique spec '<name>[ <param>]...', e.g. 'slice x1 x5' or 'norm inv'."""
    name, *params = spec.split()
    if name not in TECHNIQUES and name not in VISUALIZATIONS:
        raise Error(f'Unknown technique: {name}\nAvailable: {", ".join([*TECHNIQUES, *VISUALIZATIONS])}')
    return name, params


def make_parser():
    parser = argparse.ArgumentParser(
        prog='spexplain interpret',
        description='Interpret stored explanations using an extraction or a visualization technique.',
    )
    parser.add_argument('explanations', metavar='<explanation>',
                        help='file with explanations in the SMT-LIB2 format (e.g. phi.txt)')
    parser.add_argument('technique', metavar='<technique>',
                        help="'<name>[ <param>]...', e.g. 'slice x1 x5', 'diam', 'norm [inv]', 'fspace' or 'pixel'")
    parser.add_argument('dataset', metavar='<dataset>', nargs='?', help='CSV file with the explained samples')
    parser.add_argument('--visualize', nargs='?', const=True, choices=VISUALIZATIONS,
                        help='visualize the result (default: determined by the technique)')
    return parser


def main(args):
    opts = make_parser().parse_args(args)

    name, params = parse_technique(opts.technique)
    visualization = opts.visualize
    if visualization is True:
        visualization = TECHNIQUES.get(name, name)

    #+ integrate the existing scripts: the explanation parser and the particular techniques
    raise Error(f"Interpretation technique '{name}' is not implemented yet")
