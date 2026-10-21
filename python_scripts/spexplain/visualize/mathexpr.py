# Visualization mathexpr: shows an interval explanation as a human-readable math expression in the LaTeX syntax,
# which is also usable in Markdown math, optionally rendered to PDF
# The presentation matters more than exactness: the features may be restricted and the numbers are rounded
# Assumes that the domains of all features are normalized to [0,1]
#+ also output the Typst syntax ('.typ'; 'and', '<=', 'x_1', 'dot 10^(-5)', 'top'),
# then also choose the intermediate format of '.pdf' (e.g. '.typ.pdf')

import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from .. import Error

DEFAULT_DIGITS = 2

OUTPUT_SUFFIXES = ('.tex', '.pdf')

# Long expressions are wrapped
DOCUMENT = r'''\documentclass[border=2pt,varwidth=20cm]{standalone}
\begin{document}
$%s$
\end{document}
'''


class Config:
    """The configuration of the visualization given by its parameters."""

    def __init__(self, params, n_vars):
        self.var_indices = None
        self.digits = DEFAULT_DIGITS

        names = set()
        for param in params:
            name, *args = param.split()
            if name in names:
                raise Error(f'Duplicate parameter of mathexpr: {name}')
            names.add(name)
            if name == 'vars':
                self.var_indices = parse_vars(args, n_vars)
            elif name == 'digits' and len(args) == 1 and args[0].isdigit():
                self.digits = int(args[0])
            else:
                raise Error(f'Unknown parameter of mathexpr: {param}')


def parse_vars(args, n_vars):
    """Parse the features 'x<i>' or ranges 'x<i>-x<j>' (from 1) into sorted indices (from 0)."""
    if not args:
        raise Error('Missing the features of parameter vars of mathexpr')
    indices = set()
    for arg in args:
        m = re.fullmatch(r'x(\d+)(?:-x(\d+))?', arg)
        first, last = (int(m[1]), int(m[2] or m[1])) if m else (0, 0)
        if not 1 <= first <= last <= n_vars:
            raise Error(f'Invalid features for {n_vars} features: {arg}')
        indices.update(range(first - 1, last))
    return sorted(indices)


def check_output(file):
    if Path(file).suffix not in OUTPUT_SUFFIXES:
        raise Error(f'Visualization mathexpr requires the output with one of the extensions: {", ".join(OUTPUT_SUFFIXES)}')


def _strip_zeros(s):
    return s.rstrip('0').rstrip('.') if '.' in s else s


def format_number(v, digits):
    """Round to at most the digits of decimals, use the semi-logarithmic form for very small numbers."""
    if v != 0 and abs(v) < 10**-digits:
        mantissa, exponent = f'{v:.{digits}e}'.split('e')
        return rf'{_strip_zeros(mantissa)} \cdot 10^{{{int(exponent)}}}'
    s = _strip_zeros(f'{v:.{digits}f}')
    return '0' if s == '-0' else s


def expression(lower, upper, config):
    """The conjunction of the bounds of the features, omitting the free ones."""
    indices = config.var_indices if config.var_indices is not None else range(lower.size)
    terms = []
    for i in indices:
        lo, hi = lower[i], upper[i]
        if lo == 0 and hi == 1:
            continue
        var = 'x_{%d}' % (i + 1)
        lo_str, hi_str = format_number(lo, config.digits), format_number(hi, config.digits)
        if lo == hi:
            terms.append(f'{var} = {lo_str}')
        elif lo == 0:
            terms.append(rf'{var} \le {hi_str}')
        elif hi == 1:
            terms.append(rf'{lo_str} \le {var}')
        else:
            terms.append(rf'{lo_str} \le {var} \le {hi_str}')
    return r' \land '.join(terms) if terms else r'\top'


def _render(expr, pdf):
    if not shutil.which('pdflatex'):
        raise Error('Rendering LaTeX to PDF requires pdflatex')
    with tempfile.TemporaryDirectory(prefix='spexplain-') as tmp:
        tex = Path(tmp) / 'expression.tex'
        tex.write_text(DOCUMENT % expr)
        res = subprocess.run(['pdflatex', '-interaction=nonstopmode', '-halt-on-error', tex.name], cwd=tmp,
                             stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
        if res.returncode != 0:
            errors = [line for line in res.stdout.splitlines() if line.startswith('!')]
            raise Error(f'Rendering LaTeX to PDF failed: {" ".join(errors) or res.stdout[-500:]}')
        shutil.copyfile(tex.with_suffix('.pdf'), pdf)


def save(lower, upper, config, file, show=False):
    """Save the expression into the '.tex' file, and render it into the '.pdf' file if requested or shown."""
    file = Path(file)
    expr = expression(lower, upper, config)
    file.with_suffix('.tex').write_text(expr + '\n')

    if file.suffix == '.pdf':
        pdf = file
    elif show:
        # Only shown, not stored next to the '.tex' file; kept until the viewer opens it
        pdf = Path(tempfile.mkdtemp(prefix='spexplain-')) / file.with_suffix('.pdf').name
    else:
        return
    _render(expr, pdf)

    if show:
        if not shutil.which('xdg-open'):
            raise Error('Showing the PDF requires xdg-open')
        subprocess.Popen(['xdg-open', str(pdf)], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
