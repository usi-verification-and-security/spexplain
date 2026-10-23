# Action interpret: runs the Explanation Interpreter on stored explanations of samples

import argparse
import re
from pathlib import Path

from .. import Error

# Extraction techniques, each determines its default visualization
EXTRACTIONS = {
    'slice': 'fspace',
    'diam': 'fspace',
    'norm': 'pixel',
}

# Visualization techniques, which can also be applied directly
VISUALIZATIONS = ('fspace', 'pixel', 'mathexpr')

DEFAULT_OUTPUT_EXPLANATIONS = 'tphi.txt'
DEFAULT_OUTPUT_VISUALIZATIONS = {
    'fspace': 'timg.png',
    'pixel': 'timg.png',
    'mathexpr': 'tphi.tex',
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


def make_labels(paths):
    """Labels of the explanations from their paths without the common prefixes and suffixes of all of them.

    The paths are split at '/' and '_', e.g. 'guide-all/itp_aweak_bstrong.phi.txt'
    and 'fix-all/itp_aweak_bstrong.phi.txt' result in 'guide-all' and 'fix-all'.
    """
    names = [re.sub(r'(\.phi)?\.txt$', '', str(path)) for path in paths]
    tokens = [re.split(r'([/_])', name) for name in names]
    min_len = min(map(len, tokens))
    prefix = 0
    while prefix < min_len and all(t[prefix] == tokens[0][prefix] for t in tokens):
        prefix += 1
    suffix = 0
    while suffix < min_len - prefix and all(t[-1 - suffix] == tokens[0][-1 - suffix] for t in tokens):
        suffix += 1
    labels = [''.join(t[prefix:len(t) - suffix]).strip('/_') for t in tokens]
    return [label or Path(name).name for label, name in zip(labels, names)]


def output_file(base, idx):
    base = Path(base)
    return base.with_name(f'{base.stem}_s{idx}{base.suffix}')


def make_parser():
    parser = argparse.ArgumentParser(
        prog='spexplain interpret',
        usage='%(prog)s [<explanation>...] <technique_spec> [<dataset>] [<options>]',
        description='Interpret stored explanations of samples using an extraction or a visualization technique.',
    )
    parser.add_argument('args', nargs='+', metavar='[<explanation>...] <technique_spec> [<dataset>]',
                        help='files with explanations in the SMT-LIB2 format, one per sample (e.g. phi.txt), '
                             'more of them only for fspace; '
                             "technique spec '<name>[ <param>[, <param>]...]', e.g. 'norm', 'norm inv', 'pixel', "
                             "'mathexpr vars x1-x9'; "
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
    parser.add_argument('--visualization-domains', metavar='<file>',
                        help='CSV file with the domains of the features in the real units (columns Min, Max, '
                             'optionally Var, Feature), only for the outputs of fspace and mathexpr')
    parser.add_argument('--model', metavar='<nn_model_fn>',
                        help='the classifier in the NNet format, for the decision regions of fspace')
    #+ extend or replace by a CSV file with the classes, as for --visualization-domains
    parser.add_argument('--class-labels', metavar='<label>[,<label>]...',
                        help="labels of all the classes (default: from the header of the dataset separated by '|', "
                             "e.g. 'Low risk|High risk', otherwise the numbers of the classes)")
    parser.add_argument('--labels', metavar='<label>[,<label>]...',
                        help='labels of the explanations (default: from their paths without the common parts)')
    parser.add_argument('--samples-only', action='store_true',
                        help='visualize the samples and ignore the explanations (only for visualization techniques)')
    return parser


def main(args):
    opts = make_parser().parse_intermixed_args(args)

    # The explanations are the arguments before the technique, they are optional
    #+ give the dataset explicitly by --dataset <dataset> instead of as the positional argument after the technique,
    # the positional arguments would then be just the explanations and the technique
    positional = opts.args
    n_explanations = next((k for k, arg in enumerate(positional)
                           if arg.split() and (arg.split()[0] in EXTRACTIONS or arg.split()[0] in VISUALIZATIONS)),
                          len(positional))
    explanations_files, positional = positional[:n_explanations], positional[n_explanations:]
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
        if opts.samples_only:
            raise Error('Option --samples-only can only be used with a visualization technique')
    else:
        extraction = None
        visualization = (name, params)
        if opts.visualize is not None:
            raise Error('Option --visualize can only follow an extraction technique')
        if opts.output_explanations is not None:
            raise Error('Option --output-explanations can only be used with an extraction technique')

    if extraction and extraction[0] not in ('norm', 'slice', 'diam'):
        raise Error(f"Interpretation technique '{extraction[0]}' is not implemented yet")
    fspace_visualization = visualization and visualization[0] == 'fspace'
    pixel_visualization = visualization and visualization[0] == 'pixel'
    mathexpr_visualization = visualization and visualization[0] == 'mathexpr'
    if pixel_visualization and visualization[1]:
        raise Error(f'Unknown parameters of pixel: {", ".join(visualization[1])}')
    if pixel_visualization and extraction and extraction[0] != 'norm':
        raise Error(f"Visualization pixel requires a point, not the result of '{extraction[0]}'")
    if fspace_visualization and extraction and extraction[0] == 'norm':
        raise Error("Visualization fspace requires the result of 'slice' or 'diam', not of 'norm'")

    if len({Path(file).resolve() for file in explanations_files}) < len(explanations_files):
        raise Error('An explanations file is given more than once')
    multiple = len(explanations_files) > 1
    if multiple and not fspace_visualization:
        raise Error('Multiple explanations are supported only with the visualization fspace')
    if multiple and opts.output_explanations is not None:
        raise Error('Option --output-explanations requires a single explanations file')
    labels = opts.labels.split(',') if opts.labels else None
    if labels is not None and len(labels) != len(explanations_files):
        raise Error(f'The number of labels ({len(labels)}) does not match the number of explanations files '
                    f'({len(explanations_files)})')
    if labels is None:
        labels = make_labels(explanations_files) if multiple else ['Explanation'] * len(explanations_files)

    # Only the samples are visualized, also if the explanations are omitted
    use_samples_only = not extraction and (opts.samples_only or not explanations_files)
    if not explanations_files and not use_samples_only:
        raise Error('Missing the explanations')
    if use_samples_only:
        explanations_files = []

    # All implemented techniques require the dataset
    #+ also allow a CSV with the domains of the features by --domains in case they are not normalized to [0,1]
    #+ then pixel visualizing an explanation will not require the dataset, only the number of features from the CSV
    if dataset_file is None:
        raise Error(f"Interpretation technique '{name}' requires the dataset")

    from ..dataset import Dataset
    dataset = Dataset(dataset_file) if dataset_file else None

    n_samples = len(dataset) if dataset else count_lines(explanations_files[0])
    for file in explanations_files:
        if dataset and count_lines(file) != n_samples:
            raise Error(f'The number of explanations ({count_lines(file)}) in {file} does not match the size '
                        f'of the dataset ({n_samples})')

    indices = parse_samples_spec(opts.samples, n_samples) if opts.samples else list(range(1, n_samples + 1))
    limiting_samples = opts.max_samples is not None and opts.max_samples < len(indices)
    if limiting_samples:
        indices = indices[:opts.max_samples]
    if opts.show and len(indices) > 1:
        raise Error('Option --show requires processing only one sample')

    if extraction:
        if extraction[0] == 'norm':
            from . import norm as extraction_module
            extraction_config = extraction_module.Config(extraction[1])
        else:
            if extraction[0] == 'slice':
                from . import slice as extraction_module
            else:
                from . import diam as extraction_module
            extraction_config = extraction_module.Config(extraction[1], dataset.n_features())

    if fspace_visualization:
        from ..visualize import fspace
        visualization_config = fspace.Config(visualization[1], dataset.n_features())
        if extraction:
            if visualization_config.var_indices is None:
                visualization_config.var_indices = extraction_config.var_indices
            elif visualization_config.var_indices != extraction_config.var_indices:
                raise Error(f"Visualization fspace must use the features of '{extraction[0]}'")
        elif visualization_config.var_indices is None:
            raise Error('Visualization fspace requires the parameter vars')

    if mathexpr_visualization:
        from ..visualize import mathexpr
        visualization_config = mathexpr.Config(visualization[1], dataset.n_features())
        # The features outside of a slice are removed
        if extraction and extraction[0] != 'norm' and visualization_config.var_indices is None:
            visualization_config.var_indices = extraction_config.var_indices

    domains = None
    if opts.visualization_domains is not None:
        if not (fspace_visualization or mathexpr_visualization):
            raise Error('Option --visualization-domains can only be used with the visualizations fspace and mathexpr')
        from ..domains import Domains
        domains = Domains(opts.visualization_domains, dataset)
        visualization_config.domains = domains
    if fspace_visualization:
        # The names from the dataset, unless given by the domains
        visualization_config.names = domains.names if domains else dataset.feature_names

    if (opts.model is not None or opts.class_labels is not None) and not fspace_visualization:
        raise Error('Options --model and --class-labels can only be used with the visualization fspace')
    model = None
    if opts.model is not None:
        from ..model import Model
        model = Model(opts.model)
        if model.n_inputs() != dataset.n_features():
            raise Error(f'The number of the inputs of the model ({model.n_inputs()}) does not match the number '
                        f'of the features of the dataset ({dataset.n_features()})')
    if fspace_visualization:
        # All the classes: of the model, otherwise given by the header of the dataset, otherwise the expected ones
        n_expected_classes = int(dataset.labels.max()) + 1 if len(dataset) else 0
        if model:
            n_classes, of = model.n_classes(), 'the classes of the model'
        elif dataset.class_names is not None:
            n_classes, of = len(dataset.class_names), 'the names of the classes in the header of the dataset'
        else:
            n_classes, of = n_expected_classes, 'the expected classes of the dataset'
        n_all_classes = max(n_classes, n_expected_classes)
        if dataset.class_names is not None and len(dataset.class_names) < n_all_classes:
            raise Error(f'The number of the names of the classes in the header of the dataset '
                        f'({len(dataset.class_names)}) does not cover all the classes ({n_all_classes})')
        class_names = opts.class_labels.split(',') if opts.class_labels else dataset.class_names
        if class_names is not None and len(class_names) != n_classes:
            raise Error(f'The number of the labels of the classes ({len(class_names)}) does not match the number '
                        f'of {of} ({n_classes})')
        visualization_config.model = model
        visualization_config.class_names = class_names

    explanations_list = [read_explanations(file, indices) for file in explanations_files]
    # The single explanations file of the techniques other than fspace
    explanations = explanations_list[0] if explanations_list else None

    output_explanations = None
    if extraction and (opts.output_explanations or not visualization):
        output_explanations = opts.output_explanations or DEFAULT_OUTPUT_EXPLANATIONS
        print(f'Writing explanations to: {output_explanations}')
    output_visualization = None
    if visualization:
        output_visualization = opts.output_visualization or DEFAULT_OUTPUT_VISUALIZATIONS[visualization[0]]
        if mathexpr_visualization:
            mathexpr.check_output(output_visualization)
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

    if pixel_visualization or fspace_visualization:
        import matplotlib
        if not opts.show:
            matplotlib.use('Agg')
    if pixel_visualization:
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

            invalid = [exps[idx] == INVALID_EXPLANATION for exps in explanations_list]
            if invalid and all(invalid):
                if out:
                    out.write(INVALID_EXPLANATION + '\n')
                print(f'skipped ({INVALID_EXPLANATION})')
                continue

            sample = dataset.get_sample(idx) if dataset else None
            # An extracted point or the sample, a segment, otherwise the constraints of an explanation
            point = None
            segment = None
            if extraction and not multiple:
                result = extraction_module.interpret(explanations[idx], extraction_config, sample)
                if extraction[0] == 'norm':
                    point = result
                    if out:
                        out.write(formula.from_point(point) + '\n')
                elif extraction[0] == 'slice':
                    # The features outside of the slice are fixed to the sample
                    A, B = formula.fix(*result, extraction_module.fixed_values(extraction_config, sample))
                    if out:
                        out.write(formula.from_constraints(A, B) + '\n')
                else:
                    segment = result
                    A, B = formula.segment_constraints(*segment)
                    if out:
                        out.write(formula.from_constraints(A, B) + '\n')
            elif use_samples_only:
                point = sample
            elif pixel_visualization:
                point = formula.to_point(explanations[idx], dataset.n_features())
            elif mathexpr_visualization:
                A, B = formula.parse(explanations[idx], dataset.n_features())

            # print(f'{idx}: norm {np.linalg.norm(point):.6g}')

            if pixel_visualization:
                pixel.save(point, output_file(output_visualization, idx), show=opts.show)
            elif fspace_visualization:
                from . import region2d
                from . import slice as slice_
                # The label, the vertices of the slice of the explanation, the diameter, and whether it is invalid
                series = []
                for label, exps, is_invalid in zip(labels, explanations_list, invalid):
                    if is_invalid:
                        series.append((label, None, None, True))
                        continue
                    A2, B2 = slice_.interpret(exps[idx], visualization_config, sample)
                    vertices = region2d.vertices(A2[:, visualization_config.var_indices], B2)
                    diameter = None
                    if extraction and extraction[0] == 'diam':
                        diameter = segment if not multiple else extraction_module.interpret(
                            exps[idx], extraction_config, sample)
                    series.append((label, vertices, diameter, False))
                fspace.save(series, sample, visualization_config, output_file(output_visualization, idx),
                            title=fspace.caption(dataset.labels[idx - 1], sample, visualization_config),
                            show=opts.show)
            elif mathexpr_visualization:
                if point is not None:
                    expr = mathexpr.expression(point, point, visualization_config)
                elif visualization_config.param:
                    if segment is None:
                        segment = mathexpr.segment_endpoints(A, B)
                    expr = mathexpr.parametric_expression(*segment, visualization_config)
                else:
                    expr = mathexpr.explicit_expression(A, B, visualization_config)
                mathexpr.save(expr, output_file(output_visualization, idx), show=opts.show)

            invalid_labels = [label for label, is_invalid in zip(labels, invalid) if is_invalid]
            print(f'done ({INVALID_EXPLANATION}: {", ".join(invalid_labels)})' if invalid_labels else 'done')
    finally:
        if out:
            out.close()

    print('\nDone.')

    return 0
