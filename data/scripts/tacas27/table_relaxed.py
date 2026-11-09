#!/usr/bin/env python3
## The LaTeX tables of the relaxed features (runme.sh relaxed-features): for each table-relaxed_<name>[200].csv
## in the output directory, table-relaxed_<name>[200].tex, with a row per model and a column per tool and its
## configuration (VeriX+ and CORA per epsilon, ours per variant), each of four cells: the convex explanations of
## all of them without the timeouts [%], and over the convex ones: the relaxed features [%], the mean Linf diameter,
## and the interval [mean inner width, mean outer width] of the features [%] (diameter <= inner <= outer); '--' if
## none, the outer width 'n/a' if not computed (see OUTER_FILTER in runme.sh).
##
## USAGE: table_relaxed.py <output dir>

import csv
import os
import sys
import tempfile
from pathlib import Path

COLUMNS = ['Model Size', 'Tool', 'Configuration', 'Convex', 'Non-convex', 'Relaxed [%]', 'Linf diameter [%]',
           'Inner [%]', 'Outer [%]']


class MalformedError(Exception):
    pass


def check_row(row):
    """Fail unless the row of the CSV has all the columns, with the numbers where expected."""
    if any(row.get(c) in (None, '') for c in COLUMNS) or None in row:
        raise MalformedError('missing or extra fields')
    try:
        int(row['Model Size']), int(row['Convex']), int(row['Non-convex'])
        for c in COLUMNS[5:]:
            if row[c] != '-':
                float(row[c])
    except ValueError as e:
        raise MalformedError(str(e))


SUBCOLUMNS = ['Conv.', 'Rel.', 'Diam.', '[Inner, Outer]']


def percent(value):
    """A percentage of the CSV, with one decimal, or two significant digits below 1."""
    x = float(value)
    return f'{x:.2g}' if 0 < x < 1 else f'{x:.1f}'


def cells(row):
    """The cells of the configuration in the row: SUBCOLUMNS."""
    convex, nonconvex = int(row['Convex']), int(row['Non-convex'])
    conv = f'{100 * convex / (convex + nonconvex):.1f}' if convex + nonconvex else '--'
    if row['Relaxed [%]'] == '-':
        return [conv, '--', '--', '--']
    outer = 'n/a' if row['Outer [%]'] == '-' else row['Outer [%]']
    return [conv, row['Relaxed [%]'], percent(row['Linf diameter [%]']), f"[{row['Inner [%]']}, {outer}]"]


def write_table(csv_path):
    """-> (the path of the table or None, the number of the skipped malformed rows)"""
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != COLUMNS:
            print(f'{csv_path}: unexpected columns (rerun runme.sh relaxed-features): {reader.fieldnames}',
                  file=sys.stderr)
            return None, 1
        rows = []
        n_malformed = 0
        for row in reader:
            try:
                check_row(row)
            except MalformedError as e:
                print(f'{csv_path}:{reader.line_num}: malformed row ({e}): {",".join(map(str, row.values()))}',
                      file=sys.stderr)
                n_malformed += 1
                continue
            rows.append(row)
    if not rows:
        return None, n_malformed

    # The columns in the order of their first occurrence, grouped by the tool
    tools = []
    configs = {}
    for row in rows:
        tool = row['Tool']
        if tool not in configs:
            tools.append(tool)
            configs[tool] = []
        if row['Configuration'] not in configs[tool]:
            configs[tool].append(row['Configuration'])
    columns = [(t, c) for t in tools for c in configs[t]]

    table = {}
    sizes = []
    for row in rows:
        size = row['Model Size']
        if size not in sizes:
            sizes.append(size)
        table[(size, row['Tool'], row['Configuration'])] = cells(row)
    sizes.sort(key=int)
    n_sub = len(SUBCOLUMNS)

    def escape(s):
        return s.replace('%', r'\%').replace('_', r'\_')

    # The rules under the tools and under the configurations
    tool_rules, config_rules = [], []
    first = 2
    for t in tools:
        span = n_sub * len(configs[t])
        tool_rules.append(rf'\cmidrule(lr){{{first}-{first + span - 1}}}')
        for i in range(len(configs[t])):
            config_rules.append(rf'\cmidrule(lr){{{first + i * n_sub}-{first + (i + 1) * n_sub - 1}}}')
        first += span

    lines = [
        '% Per configuration: the convex explanations of all (without the timeouts) [%], and of the convex ones: the'
        ' relaxed features [%], the mean Linf diameter and [the mean inner width, outer width] of the features [%]'
        ' (diameter <= inner <= outer); by runme.sh relaxed-features from ' + csv_path.name,
        r'\begin{tabular}{r' + 'r' * n_sub * len(columns) + '}',
        r'\toprule',
        ' & '.join(['Layers'] + [rf'\multicolumn{{{n_sub * len(configs[t])}}}{{c}}{{{escape(t)}}}' for t in tools])
        + r' \\',
        ' '.join(tool_rules),
        ' & '.join([''] + [rf'\multicolumn{{{n_sub}}}{{c}}{{{escape(c)}}}' for _, c in columns]) + r' \\',
        ' '.join(config_rules),
        ' & '.join([''] + SUBCOLUMNS * len(columns)) + r' \\',
        r'\midrule',
    ]
    for size in sizes:
        lines.append(' & '.join([size] + [x for t, c in columns for x in table.get((size, t, c), [''] * n_sub)])
                     + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', '']

    # Via a new file, not rewriting the old one in place
    tex_path = csv_path.with_suffix('.tex')
    fd, tmp = tempfile.mkstemp(prefix=tex_path.name + '.', dir=tex_path.parent)
    umask = os.umask(0)
    os.umask(umask)
    os.chmod(tmp, 0o666 & ~umask)
    with os.fdopen(fd, 'w') as f:
        f.write('\n'.join(lines))
    os.replace(tmp, tex_path)
    return tex_path, n_malformed


def main(args):
    if len(args) != 1:
        print('USAGE: table_relaxed.py <output dir>', file=sys.stderr)
        return 1
    n_malformed = 0
    for csv_path in sorted(Path(args[0]).glob('table-relaxed_*.csv')):
        tex_path, n = write_table(csv_path)
        n_malformed += n
        if tex_path:
            print(tex_path)
    if n_malformed:
        print(f'Skipped {n_malformed} malformed rows or files', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main(sys.argv[1:]))
