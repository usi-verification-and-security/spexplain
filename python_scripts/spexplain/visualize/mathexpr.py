# Visualization mathexpr: shows an explanation that is a conjunction of linear constraints (e.g. an interval,
# a segment or a bounded plane) as a human-readable math expression in the LaTeX syntax,
# which is also usable in Markdown math, optionally rendered to PDF
# The presentation matters more than exactness: the features may be restricted and the numbers are rounded
# Assumes that the domains of all features are normalized to [0,1]
#+ also output the Typst syntax ('.typ'; 'and', '<=', 'x_1', 'dot 10^(-5)', 'top'),
# then also choose the intermediate format of '.pdf' (e.g. '.typ.pdf')

import shutil
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from .. import Error
from ..interpret import formula

DEFAULT_DIGITS = 2

OUTPUT_SUFFIXES = ('.tex', '.pdf')

# Tolerance of the computations of the parametric form
TOLERANCE = 1e-9

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
        # The parametric form, otherwise the explicit one
        self.param = False
        # The domains of the features in the real units, otherwise normalized
        self.domains = None

        names = set()
        for param in params:
            name, *args = param.split()
            if name in names:
                raise Error(f'Duplicate parameter of mathexpr: {name}')
            names.add(name)
            if name == 'vars':
                self.var_indices = formula.parse_vars(args, n_vars, 'mathexpr')
            elif name == 'digits' and len(args) == 1 and args[0].isdigit():
                self.digits = int(args[0])
            elif name == 'param' and not args:
                self.param = True
            else:
                raise Error(f'Unknown parameter of mathexpr: {param}')


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


def _var(i):
    return 'x_{%d}' % (i + 1)


def _selected(indices, config):
    return config.var_indices is None or not set(config.var_indices).isdisjoint(indices)


def _bound_terms(lower, upper, config):
    """The bounds of the features, omitting the free ones."""
    indices = config.var_indices if config.var_indices is not None else range(lower.size)
    terms = []
    for i in indices:
        lo, hi = lower[i], upper[i]
        if lo == 0 and hi == 1:
            continue
        var = _var(i)
        if config.domains:
            lo, hi = config.domains.scale(lo, i), config.domains.scale(hi, i)
        lo_str, hi_str = format_number(lo, config.digits), format_number(hi, config.digits)
        if lower[i] == upper[i]:
            terms.append(f'{var} = {lo_str}')
        elif lower[i] == 0:
            terms.append(rf'{var} \le {hi_str}')
        elif upper[i] == 1:
            terms.append(rf'{lo_str} \le {var}')
        else:
            terms.append(rf'{lo_str} \le {var} \le {hi_str}')
    return terms


def _conjunction(terms):
    return r' \land '.join(terms) if terms else r'\top'


def expression(lower, upper, config):
    """The conjunction of the bounds of the features, omitting the free ones."""
    return _conjunction(_bound_terms(lower, upper, config))


def _linear_term(row, b, relation, config):
    """The linear constraint 'row * x <relation> b', with the first coefficient positive."""
    nonzero = np.flatnonzero(row)
    if row[nonzero[0]] < 0:
        row, b = -row, -b
        relation = {r'\le': r'\ge', '=': '='}[relation]
    lhs = ''
    for j in nonzero:
        c_str = format_number(abs(row[j]), config.digits)
        if c_str == '0':
            continue
        coef = '' if c_str == '1' else c_str + ' '
        sign = ('-' if row[j] < 0 else '') if not lhs else (' - ' if row[j] < 0 else ' + ')
        lhs += f'{sign}{coef}{_var(j)}'
    return f'{lhs or "0"} {relation} {format_number(b, config.digits)}'


def explicit_expression(A, B, config):
    """The conjunction of the bounds of the features and of the other linear constraints, constraints A x <= B."""
    single = np.count_nonzero(A, axis=1) == 1
    terms = _bound_terms(*formula.interval_bounds(A[single], B[single]), config)

    multi = np.flatnonzero(~single)
    pairs = formula.equalities(A[multi], B[multi])
    seconds = set(pairs.values())
    A_scaled, B_scaled = config.domains.scale_constraints(A, B) if config.domains else (A, B)
    for k, i in enumerate(multi):
        if k in seconds or not _selected(np.flatnonzero(A[i]), config):
            continue
        terms.append(_linear_term(A_scaled[i], B_scaled[i], '=' if k in pairs else r'\le', config))

    return _conjunction(terms)


def segment_endpoints(A, B):
    """The endpoints of the segment given by the constraints A x <= B within the domains [0,1]."""
    n_vars = A.shape[1]
    pairs = formula.equalities(A, B)
    eq = sorted(pairs)

    # The affine hull of the equalities: a point and the directions
    if eq:
        E, e = A[eq], B[eq]
        point = np.linalg.lstsq(E, e, rcond=None)[0]
        if not np.allclose(E @ point, e, atol=TOLERANCE):
            raise Error('The explanation is empty')
        _, sv, Vt = np.linalg.svd(E)
        rank = int(np.sum(sv > TOLERANCE * max(1, sv[0])))
        directions = Vt[rank:]
    else:
        point = np.zeros(n_vars)
        directions = np.eye(n_vars)

    if len(directions) > 1:
        raise Error('Parametric form of mathexpr supports only segments')
    if len(directions) == 0:
        return point, point

    # The range of the parameter given by all constraints including the domains
    d = directions[0]
    A_all = np.vstack([A, np.eye(n_vars), -np.eye(n_vars)])
    B_all = np.concatenate([B, np.ones(n_vars), np.zeros(n_vars)])
    ad, rest = A_all @ d, B_all - A_all @ point
    if np.any(rest[np.abs(ad) <= TOLERANCE] < -TOLERANCE):
        raise Error('The explanation is empty')
    t_min = np.max(rest[ad < -TOLERANCE] / ad[ad < -TOLERANCE])
    t_max = np.min(rest[ad > TOLERANCE] / ad[ad > TOLERANCE])
    if t_min > t_max + TOLERANCE:
        raise Error('The explanation is empty')

    p, q = point + t_min * d, point + t_max * d
    # Deterministic orientation: the first varying feature increases
    diff = q - p
    varying = np.flatnonzero(np.abs(diff) > TOLERANCE)
    if varying.size and diff[varying[0]] < 0:
        p, q = q, p
    return p, q


def parametric_expression(p, q, config):
    """The segment between the points as x = p + t (q - p) with t in [0,1]."""
    indices = config.var_indices if config.var_indices is not None else range(p.size)
    if config.domains:
        p, q = config.domains.scale(p), config.domains.scale(q)
    terms = []
    varying = False
    for i in indices:
        a, b = p[i], q[i] - p[i]
        a_str = format_number(a, config.digits)
        if abs(b) <= TOLERANCE:
            terms.append(f'{_var(i)} = {a_str}')
            continue
        varying = True
        b_str = format_number(abs(b), config.digits)
        coef = '' if b_str == '1' else b_str + r'\,'
        if a_str == '0':
            rhs = f'{"-" if b < 0 else ""}{coef}t'
        else:
            rhs = f'{a_str} {"-" if b < 0 else "+"} {coef}t'
        terms.append(f'{_var(i)} = {rhs}')
    if varying:
        terms.append(r't \in [0, 1]')
    return _conjunction(terms)


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


def save(expr, file, show=False):
    """Save the expression into the '.tex' file, and render it into the '.pdf' file if requested or shown."""
    file = Path(file)
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
