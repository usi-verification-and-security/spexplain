# Action interpret: runs the Explanation Interpreter on stored explanations of samples

import argparse
from pathlib import Path

from .. import Error

# Extraction techniques, each determines its default visualization
EXTRACTIONS = {
    'slice': 'fspace',
    'diam': 'pixel',
    'norm': 'pixel',
}

# Visualization techniques, which can also be applied directly
VISUALIZATIONS = ('fspace', 'pixel')

DEFAULT_OUTPUT_EXPLANATIONS = 'tphi.txt'
DEFAULT_OUTPUT_VISUALIZATIONS = {
    'pixel': 'timg.png',
}

# Same as in the engine, e.g. if a timeout was reached
INVALID_EXPLANATION = '<null>'


def parse_technique_spec(spec):
    """Parse the technique spec '<name>[ <param>[, <param>]...]' as in the engine, e.g. 'norm inv'."""
    name, _, rest = spec.strip().partition(' ')
    params = [param.strip() for param in rest.split(',')] if rest.strip() else []
    if name not in EXTRACTIONS and name not in VISUALIZATIONS:
        raise Error(f'Unknown technique: {name}\nAvailable: {", ".join([*EXTRACTIONS, *VISUALIZATIONS])}')
    if not all(params):
        raise Error(f'Empty parameter of the technique: {spec}')
    return name, params


def parse_samples_spec(spec, n_samples):
    """Parse '<sel>[,<sel>]...' where each '<sel>' is '<idx>', '<idx>-<idx2>', '<idx>-' or '-<idx2>' (from 1)."""
    indices = set()
    for sel in spec.split(','):
        first, dash, last = sel.partition('-')
        try:
            first = int(first) if first else 1
            last = (int(last) if last else n_samples) if dash else first
        except ValueError:
            raise Error(f'Invalid samples specification: {spec}')
        if not sel.strip('-') or not 1 <= first <= last <= n_samples:
            raise Error(f'Invalid samples specification for {n_samples} samples: {spec}')
        indices.update(range(first, last + 1))
    return sorted(indices)


def count_lines(path):
    with open(path) as f:
        return sum(1 for _ in f)


def read_explanations(path, indices):
    """The explanations of the samples, assuming that the file contains one explanation per sample."""
    wanted = set(indices)
    with open(path) as f:
        return {idx: line.strip() for idx, line in enumerate(f, start=1) if idx in wanted}


def output_file(base, idx):
    base = Path(base)
    return base.with_name(f'{base.stem}_s{idx}{base.suffix}')


def make_parser():
    parser = argparse.ArgumentParser(
        prog='spexplain interpret',
        usage='%(prog)s [<explanation>] <technique_spec> [<dataset>] [<options>]',
        description='Interpret stored explanations of samples using an extraction or a visualization technique.',
    )
    parser.add_argument('args', nargs='+', metavar='[<explanation>] <technique_spec> [<dataset>]',
                        help='file with explanations in the SMT-LIB2 format, one per sample (e.g. phi.txt); '
                             "technique spec '<name>[ <param>[, <param>]...]', e.g. 'norm', 'norm inv', 'pixel', "
                             "'pixel sample'; "
                             'CSV file with the samples')
    parser.add_argument('-i', '--samples', metavar='<sel>[,<sel>]...',
                        help="only process the given samples, each '<sel>' is '<idx>' or '[<idx>]-[<idx2>]' (from 1)")
    parser.add_argument('-n', '--max-samples', type=int, metavar='<int>', help='maximum no. samples to be processed')
    parser.add_argument('-e', '--output-explanations', metavar='<file>',
                        help=f'output the extracted explanations into the file (default: {DEFAULT_OUTPUT_EXPLANATIONS} '
                             'unless visualizing)')
    parser.add_argument('--visualize', nargs='?', const=True, metavar='<visualization_spec>',
                        help='visualize the extracted explanations (default: determined by the technique)')
    parser.add_argument('--output-visualization', metavar='<file>',
                        help="output the visualizations into the files with '_s<idx>' added to the name "
                             '(default: ' + ', '.join(f'{file} for {name}'
                                                      for name, file in DEFAULT_OUTPUT_VISUALIZATIONS.items()) + ')')
    parser.add_argument('--show', action='store_true', help='also show the visualization (only for one sample)')
    return parser


def main(args):
    opts = make_parser().parse_intermixed_args(args)

    # <explanation> is optional, then the first argument is the technique
    positional = opts.args
    first_name = positional[0].split()[0] if positional[0].split() else ''
    explanations_file = None if first_name in EXTRACTIONS or first_name in VISUALIZATIONS else positional.pop(0)
    if not positional:
        raise Error('Missing the technique spec')
    technique_spec, *rest = positional
    if len(rest) > 1:
        raise Error(f'Unexpected arguments: {" ".join(rest[1:])}')
    dataset_file = rest[0] if rest else None

    name, params = parse_technique_spec(technique_spec)
    if name in EXTRACTIONS:
        extraction = (name, params)
        if opts.visualize is None:
            visualization = None
        elif opts.visualize is True:
            visualization = (EXTRACTIONS[name], [])
        else:
            visualization = parse_technique_spec(opts.visualize)
            if visualization[0] not in VISUALIZATIONS:
                raise Error(f'Not a visualization technique: {visualization[0]}')
            if visualization == ('pixel', ['sample']):
                raise Error("Visualization 'pixel sample' cannot follow an extraction technique")
    else:
        extraction = None
        visualization = (name, params)
        if opts.visualize is not None:
            raise Error('Option --visualize can only follow an extraction technique')
        if opts.output_explanations is not None:
            raise Error('Option --output-explanations can only be used with an extraction technique')

    if extraction and extraction[0] != 'norm':
        raise Error(f"Interpretation technique '{extraction[0]}' is not implemented yet")
    if extraction:
        from . import norm
        extraction_config = norm.Config(extraction[1])
    if visualization and visualization[0] != 'pixel':
        raise Error(f"Visualization technique '{visualization[0]}' is not implemented yet")
    if visualization and visualization[1] not in ([], ['sample']):
        raise Error(f'Unknown parameters of pixel: {", ".join(visualization[1])}')

    # Only the samples are visualized
    use_samples_only = not extraction and (visualization[1] == ['sample'] or explanations_file is None)
    if explanations_file is None and not use_samples_only:
        raise Error('Missing the explanations')
    if use_samples_only:
        explanations_file = None

    # All implemented techniques require the dataset
    #+ also allow a CSV with the domains of the features in case they are not normalized to [0,1]
    #+ then pixel visualizing an explanation will not require the dataset, only the number of features from the CSV
    if dataset_file is None:
        raise Error(f"Interpretation technique '{name}' requires the dataset")

    from ..dataset import Dataset
    dataset = Dataset(dataset_file) if dataset_file else None

    n_samples = len(dataset) if dataset else count_lines(explanations_file)
    if dataset and explanations_file:
        n_explanations = count_lines(explanations_file)
        if n_explanations != n_samples:
            raise Error(f'The number of explanations ({n_explanations}) does not match the size of the dataset '
                        f'({n_samples})')

    indices = parse_samples_spec(opts.samples, n_samples) if opts.samples else list(range(1, n_samples + 1))
    limiting_samples = opts.max_samples is not None and opts.max_samples < len(indices)
    if limiting_samples:
        indices = indices[:opts.max_samples]
    if opts.show and len(indices) > 1:
        raise Error('Option --show requires processing only one sample')

    explanations = read_explanations(explanations_file, indices) if explanations_file else None

    output_explanations = None
    if extraction and (opts.output_explanations or not visualization):
        output_explanations = opts.output_explanations or DEFAULT_OUTPUT_EXPLANATIONS
        print(f'Writing explanations to: {output_explanations}')
    output_visualization = None
    if visualization:
        output_visualization = opts.output_visualization or DEFAULT_OUTPUT_VISUALIZATIONS[visualization[0]]
        print(f'Writing visualizations to: {output_file(output_visualization, "<idx>")}')
    print()

    # Similar to the engine
    if dataset:
        print(f'Dataset size: {n_samples}')
        print(f'Number of variables: {dataset.n_features()}')
    else:
        print(f'Number of explanations: {n_samples}')
    if opts.samples:
        print(f'Selected samples: {opts.samples}')
    if limiting_samples:
        print(f'Limited number of samples: {opts.max_samples}')
    print('-' * 60)

    if visualization:
        import matplotlib
        if not opts.show:
            matplotlib.use('Agg')
        from ..visualize import pixel
    from . import formula

    selected = set(indices)
    out = open(output_explanations, 'w') if output_explanations else None
    try:
        # The output contains an explanation for each sample, the samples not processed are invalid
        for idx in range(1, n_samples + 1):
            if idx not in selected:
                if out:
                    out.write(INVALID_EXPLANATION + '\n')
                continue

            print(f'sample [{idx}/{n_samples}] ... ', end='', flush=True)

            if explanations and explanations[idx] == INVALID_EXPLANATION:
                if out:
                    out.write(INVALID_EXPLANATION + '\n')
                print(f'skipped ({INVALID_EXPLANATION})')
                continue

            sample = dataset.get_sample(idx) if dataset else None
            if extraction:
                point = norm.interpret(explanations[idx], extraction_config, sample)
                if out:
                    out.write(formula.from_point(point) + '\n')
            elif use_samples_only:
                point = sample
            else:
                point = formula.to_point(explanations[idx], dataset.n_features())

            # print(f'{idx}: norm {np.linalg.norm(point):.6g}')

            if visualization:
                pixel.save(point, output_file(output_visualization, idx), show=opts.show)

            print('done')
    finally:
        if out:
            out.close()

    print('\nDone.')

    return 0
