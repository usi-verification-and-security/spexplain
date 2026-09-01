#!/usr/bin/env python3
"""Semantic comparison of two SMT-LIB2 `psi` files produced by spexplain.

A plain textual `diff` of two psi files is close to useless: almost the whole
formula lives on a couple of very long `(assert (and ...))` lines, so any
difference -- however tiny -- reports the entire line as changed.

This tool instead parses both files, splits every assertion into its top-level
conjuncts, and matches the conjuncts of the two files against each other by
*structure*, ignoring the concrete numeric literals.  For every matched pair it
then compares the numeric literals one by one, so a coefficient that differs
only because of float32-vs-decimal rounding can be told apart from a genuine
encoding difference.

Reported per assertion:

  identical    conjuncts equal term by term, including all numbers
  close        same structure, every number within --tol relative difference
               (i.e. a rounding artefact)
  differing    same structure, but at least one number is really different
  only in A/B  conjuncts whose structure has no counterpart in the other file

Exit status: 0 when the files are identical or differ only within --tol,
1 when there are real differences, 2 on a usage/parse error.
"""

from __future__ import annotations

import argparse
import re
import sys
from fractions import Fraction
from typing import Iterator


# --------------------------------------------------------------------------
# S-expression parsing
# --------------------------------------------------------------------------

TOKEN_RE = re.compile(r'\(|\)|"[^"]*"|\|[^|]*\||[^\s()]+')


def tokenize(text: str) -> Iterator[str]:
    for line in text.splitlines():
        line = line.split(";", 1)[0]
        yield from TOKEN_RE.findall(line)


def parse_sexprs(text: str) -> list:
    """Parse `text` into a list of s-expressions (nested lists of str)."""
    stack: list[list] = [[]]
    for tok in tokenize(text):
        if tok == "(":
            stack.append([])
        elif tok == ")":
            if len(stack) == 1:
                raise ValueError("unbalanced ')' in input")
            done = stack.pop()
            stack[-1].append(done)
        else:
            stack[-1].append(tok)
    if len(stack) != 1:
        raise ValueError("unbalanced '(' in input")
    return stack[0]


def unparse(node) -> str:
    if isinstance(node, str):
        return node
    return "(" + " ".join(unparse(c) for c in node) + ")"


# --------------------------------------------------------------------------
# Numeric handling
# --------------------------------------------------------------------------

NUMERAL_RE = re.compile(r"^\d+(\.\d+)?$")


def as_number(node):
    """Return the exact value of `node` if it is a purely numeric term.

    Handles numerals, decimals, and the `(- x)`, `(/ x y)`, `(* ...)`, `(+ ...)`
    and `(- x y)` forms that OpenSMT emits.  Returns None for anything that
    mentions a variable.
    """
    if isinstance(node, str):
        if NUMERAL_RE.match(node):
            return Fraction(node)
        return None
    if not node:
        return None
    op, args = node[0], node[1:]
    if not isinstance(op, str):
        return None
    vals = [as_number(a) for a in args]
    if not vals or any(v is None for v in vals):
        return None
    if op == "-":
        return -vals[0] if len(vals) == 1 else vals[0] - sum(vals[1:])
    if op == "+":
        return sum(vals)
    if op == "*":
        acc = Fraction(1)
        for v in vals:
            acc *= v
        return acc
    if op == "/":
        acc = vals[0]
        for v in vals[1:]:
            if v == 0:
                return None
            acc /= v
        return acc
    return None


def skeleton(node, numbers: list) -> str:
    """Structure of `node` with every numeric term replaced by `#`.

    The numeric values encountered are appended to `numbers` in traversal
    order, so two terms with the same skeleton can be compared position by
    position.
    """
    value = as_number(node)
    if value is not None:
        numbers.append(value)
        return "#"
    if isinstance(node, str):
        return node
    return "(" + " ".join(skeleton(c, numbers) for c in node) + ")"


def rel_diff(a: Fraction, b: Fraction) -> float:
    if a == b:
        return 0.0
    scale = max(abs(a), abs(b), Fraction(1))
    return float(abs(a - b) / scale)


# --------------------------------------------------------------------------
# Assertion extraction
# --------------------------------------------------------------------------

ITE_RE = re.compile(r"\.ite\d+_\d+")


def canonicalise_aux_names(text: str) -> str:
    """Renumber solver-internal `.iteNNN_M` names by order of first appearance.

    OpenSMT numbers the auxiliary variables it introduces for ReLU encodings
    arbitrarily, and the numbers differ between two runs even when the encodings
    are equivalent.
    """
    mapping: dict[str, str] = {}

    def repl(match: re.Match) -> str:
        name = match.group(0)
        if name not in mapping:
            mapping[name] = ".aux%d" % len(mapping)
        return mapping[name]

    return ITE_RE.sub(repl, text)


def conjuncts_of(assertion) -> list:
    """Top-level conjuncts of an assert body, flattening nested `and`s."""
    if isinstance(assertion, list) and assertion and assertion[0] == "and":
        out = []
        for child in assertion[1:]:
            out.extend(conjuncts_of(child))
        return out
    return [assertion]


def extract_assertions(text: str, rename: bool) -> list[list]:
    if rename:
        text = canonicalise_aux_names(text)
    out = []
    for expr in parse_sexprs(text):
        if isinstance(expr, list) and len(expr) == 2 and expr[0] == "assert":
            out.append(conjuncts_of(expr[1]))
    return out


# --------------------------------------------------------------------------
# Matching
# --------------------------------------------------------------------------

class Conjunct:
    __slots__ = ("node", "skel", "numbers", "index")

    def __init__(self, node, index: int):
        self.node = node
        self.index = index
        self.numbers: list[Fraction] = []
        self.skel = skeleton(node, self.numbers)


def distance(a: Conjunct, b: Conjunct) -> float:
    return max((rel_diff(x, y) for x, y in zip(a.numbers, b.numbers)), default=0.0)


def match_group(group_a: list[Conjunct], group_b: list[Conjunct]):
    """Greedily pair conjuncts that share a skeleton, closest numbers first."""
    pairs = sorted(
        ((distance(a, b), a.index, b.index, a, b) for a in group_a for b in group_b),
        key=lambda t: (t[0], t[1], t[2]),
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matched = []
    for dist, _, _, a, b in pairs:
        if a.index in used_a or b.index in used_b:
            continue
        used_a.add(a.index)
        used_b.add(b.index)
        matched.append((a, b, dist))
    left_a = [a for a in group_a if a.index not in used_a]
    left_b = [b for b in group_b if b.index not in used_b]
    return matched, left_a, left_b


def group_by_skeleton(conjuncts: list[Conjunct]) -> dict[str, list[Conjunct]]:
    groups: dict[str, list[Conjunct]] = {}
    for c in conjuncts:
        groups.setdefault(c.skel, []).append(c)
    return groups


class AssertionReport:
    def __init__(self, idx: int, n_a: int, n_b: int):
        self.index = idx
        self.n_a = n_a
        self.n_b = n_b
        self.identical = 0
        self.close: list = []
        self.differing: list = []
        self.only_a: list[Conjunct] = []
        self.only_b: list[Conjunct] = []
        self.ignored = False

    @property
    def ok(self) -> bool:
        if self.ignored:
            return True
        return not self.differing and not self.only_a and not self.only_b

    @property
    def worst_close(self) -> float:
        return max((d for _, _, d in self.close), default=0.0)


def compare_assertion(idx: int, conj_a: list, conj_b: list, tol: float) -> AssertionReport:
    a_list = [Conjunct(n, i) for i, n in enumerate(conj_a)]
    b_list = [Conjunct(n, i) for i, n in enumerate(conj_b)]
    report = AssertionReport(idx, len(a_list), len(b_list))

    groups_a = group_by_skeleton(a_list)
    groups_b = group_by_skeleton(b_list)

    for skel, ga in groups_a.items():
        gb = groups_b.get(skel)
        if not gb:
            report.only_a.extend(ga)
            continue
        matched, left_a, left_b = match_group(ga, gb)
        for a, b, dist in matched:
            if dist == 0.0:
                report.identical += 1
            elif dist <= tol:
                report.close.append((a, b, dist))
            else:
                report.differing.append((a, b, dist))
        report.only_a.extend(left_a)
        report.only_b.extend(left_b)

    for skel, gb in groups_b.items():
        if skel not in groups_a:
            report.only_b.extend(gb)

    return report


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def abbreviate(text: str, limit: int) -> str:
    if limit <= 0 or len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def worst_number_pair(a: Conjunct, b: Conjunct):
    worst = None
    for x, y in zip(a.numbers, b.numbers):
        d = rel_diff(x, y)
        if worst is None or d > worst[0]:
            worst = (d, x, y)
    return worst


def describe_pair(a: Conjunct, b: Conjunct, dist: float, width: int) -> list[str]:
    worst = worst_number_pair(a, b)
    lines = ["      max relative difference: %.6g" % dist]
    if worst is not None:
        _, x, y = worst
        lines.append(
            "      worst literal: A = %s (%.12g)   B = %s (%.12g)"
            % (x, float(x), y, float(y))
        )
    n_diff = sum(1 for p, q in zip(a.numbers, b.numbers) if p != q)
    lines.append(
        "      literals differing: %d of %d" % (n_diff, len(a.numbers))
    )
    lines.append("      A: " + abbreviate(unparse(a.node), width))
    lines.append("      B: " + abbreviate(unparse(b.node), width))
    return lines


def print_report(reports: list[AssertionReport], name_a: str, name_b: str,
                 tol: float, width: int, max_show: int) -> None:
    print("A = %s" % name_a)
    print("B = %s" % name_b)
    print("tolerance: %g relative" % tol)
    print()

    header = "%-10s %9s %9s %10s %8s %10s %9s %9s" % (
        "ASSERT", "CONJ-A", "CONJ-B", "IDENTICAL", "CLOSE", "DIFFERING",
        "ONLY-A", "ONLY-B")
    print(header)
    print("-" * len(header))
    for r in reports:
        print("%-10s %9d %9d %10d %8d %10d %9d %9d%s" % (
            "#%d" % r.index, r.n_a, r.n_b, r.identical, len(r.close),
            len(r.differing), len(r.only_a), len(r.only_b),
            "   (ignored)" if r.ignored else ""))
    print()

    for r in reports:
        if r.ok and not r.close:
            continue
        print("--- assertion #%d%s ---"
              % (r.index, " (ignored for the verdict)" if r.ignored else ""))
        if r.close:
            worst = max(r.close, key=lambda t: t[2])
            print("  %d conjunct(s) match within tolerance "
                  "(rounding only, worst relative difference %.3g)"
                  % (len(r.close), worst[2]))
            for a, b, dist in sorted(r.close, key=lambda t: -t[2])[:max_show]:
                print("    conjunct A[%d] / B[%d]:" % (a.index, b.index))
                for line in describe_pair(a, b, dist, width):
                    print(line)
            if len(r.close) > max_show:
                print("    ... and %d more" % (len(r.close) - max_show))
        if r.differing:
            print("  %d conjunct(s) differ beyond tolerance:" % len(r.differing))
            for a, b, dist in sorted(r.differing, key=lambda t: -t[2])[:max_show]:
                print("    conjunct A[%d] / B[%d]:" % (a.index, b.index))
                for line in describe_pair(a, b, dist, width):
                    print(line)
            if len(r.differing) > max_show:
                print("    ... and %d more" % (len(r.differing) - max_show))
        for label, items in (("only in A", r.only_a), ("only in B", r.only_b)):
            if not items:
                continue
            print("  %d conjunct(s) %s (no structural counterpart):"
                  % (len(items), label))
            for c in items[:max_show]:
                print("      [%d] %s" % (c.index, abbreviate(unparse(c.node), width)))
            if len(items) > max_show:
                print("      ... and %d more" % (len(items) - max_show))
        print()


# --------------------------------------------------------------------------

def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(
        description="Structural comparison of two SMT-LIB2 psi files.")
    ap.add_argument("file_a")
    ap.add_argument("file_b")
    ap.add_argument("--tol", type=float, default=1e-3,
                    help="relative tolerance below which a numeric difference "
                         "counts as rounding noise (default: 1e-3)")
    ap.add_argument("--width", type=int, default=160,
                    help="truncate printed terms to this many characters, "
                         "0 to disable (default: 160)")
    ap.add_argument("--max-show", type=int, default=5,
                    help="how many example conjuncts to print per category "
                         "(default: 5)")
    ap.add_argument("--no-rename-aux", action="store_true",
                    help="do not canonicalise solver-internal .iteNNN_M names")
    ap.add_argument("--ignore-assert", type=int, action="append", default=[],
                    metavar="N",
                    help="do not let assertion N affect the verdict; use it to "
                         "excuse the known input-domain-bounds assertion "
                         "(repeatable)")
    ap.add_argument("--quiet", action="store_true",
                    help="print only the summary table")
    ap.add_argument("--stats", action="store_true",
                    help="print a single machine-readable line: "
                         "conj-a conj-b identical close differing only-a only-b "
                         "worst-close-reldiff")
    args = ap.parse_args(argv)

    try:
        with open(args.file_a) as fh:
            text_a = fh.read()
        with open(args.file_b) as fh:
            text_b = fh.read()
    except OSError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2

    rename = not args.no_rename_aux
    try:
        asserts_a = extract_assertions(text_a, rename)
        asserts_b = extract_assertions(text_b, rename)
    except ValueError as exc:
        print("error: parse failure: %s" % exc, file=sys.stderr)
        return 2

    if len(asserts_a) != len(asserts_b):
        print("error: different number of assertions: %d in A, %d in B"
              % (len(asserts_a), len(asserts_b)), file=sys.stderr)
        return 2

    reports = [compare_assertion(i, a, b, args.tol)
               for i, (a, b) in enumerate(zip(asserts_a, asserts_b))]
    for idx in args.ignore_assert:
        if 0 <= idx < len(reports):
            reports[idx].ignored = True

    if args.stats:
        counted = [r for r in reports if not r.ignored]
        print("%d %d %d %d %d %d %d %.6g" % (
            sum(r.n_a for r in reports),
            sum(r.n_b for r in reports),
            sum(r.identical for r in counted),
            sum(len(r.close) for r in counted),
            sum(len(r.differing) for r in counted),
            sum(len(r.only_a) for r in counted),
            sum(len(r.only_b) for r in counted),
            max((r.worst_close for r in counted), default=0.0)))
        return 0 if all(r.ok for r in reports) else 1

    if args.quiet:
        for r in reports:
            print("assert #%d: %d identical, %d close, %d differing, "
                  "%d only-A, %d only-B"
                  % (r.index, r.identical, len(r.close), len(r.differing),
                     len(r.only_a), len(r.only_b)))
    else:
        print_report(reports, args.file_a, args.file_b, args.tol,
                     args.width, args.max_show)

    if all(r.ok for r in reports):
        print("MATCH: formulas agree structurally; all numeric literals within "
              "%g relative tolerance." % args.tol)
        return 0
    print("MISMATCH: see the differing / only-in-A / only-in-B conjuncts above.")
    return 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
