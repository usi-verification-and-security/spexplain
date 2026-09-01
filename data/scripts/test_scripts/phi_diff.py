#!/usr/bin/env python3
"""Compare two `spexplain` .phi.txt explanation files line by line (one sample per line).

Each line is expected to be a `(and (= xI v) ...)` conjunction, as produced by the
`abductive` strategy: every conjunct fixes one input feature to its sample value. The
tool reports, per sample, which features are fixed on each side, and flags samples where
the two sides disagree on which/how many features are needed to explain the classification.

This is intentionally simpler than psi_diff.py: `itp`-based explanations are generic
linear formulas (not per-feature), so a feature-level comparison does not apply to them.
"""
from __future__ import annotations

import re
import sys

CONJUNCT_RE = re.compile(r"\(=\s*(x\d+)\s+([^()]+?)\)")


def parse_line(line: str) -> dict[str, str]:
    return {var: val.strip() for var, val in CONJUNCT_RE.findall(line)}


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print("usage: phi_diff.py <a.phi.txt> <b.phi.txt>", file=sys.stderr)
        return 2

    with open(argv[0]) as fh:
        lines_a = [l for l in fh.read().splitlines() if l.strip()]
    with open(argv[1]) as fh:
        lines_b = [l for l in fh.read().splitlines() if l.strip()]

    if len(lines_a) != len(lines_b):
        print(f"WARNING: different number of samples: A={len(lines_a)} B={len(lines_b)}")

    n = min(len(lines_a), len(lines_b))
    header = "%-8s %8s %8s %8s %8s  %s" % ("SAMPLE", "A-FIXED", "B-FIXED", "COMMON", "ONLY-A/B", "STATUS")
    print(header)
    print("-" * len(header))

    mismatches = 0
    for i in range(n):
        fixed_a = parse_line(lines_a[i])
        fixed_b = parse_line(lines_b[i])
        common = set(fixed_a) & set(fixed_b)
        only_a = set(fixed_a) - set(fixed_b)
        only_b = set(fixed_b) - set(fixed_a)
        # A value mismatch on a variable fixed by both sides is also a real difference.
        value_mismatch = {v for v in common if fixed_a[v] != fixed_b[v]}

        if not only_a and not only_b and not value_mismatch:
            status = "MATCH"
        else:
            status = "DIFFERS"
            mismatches += 1

        print("%-8d %8d %8d %8d %4d/%-3d  %s" % (
            i + 1, len(fixed_a), len(fixed_b), len(common) - len(value_mismatch),
            len(only_a), len(only_b), status))
        if status == "DIFFERS":
            if only_a:
                print("    only in A (nnet): %s" % ", ".join(sorted(only_a)))
            if only_b:
                print("    only in B (onnx): %s" % ", ".join(sorted(only_b)))
            if value_mismatch:
                print("    value mismatch:   %s" % ", ".join(
                    "%s (A=%s, B=%s)" % (v, fixed_a[v], fixed_b[v]) for v in sorted(value_mismatch)))

    print()
    print("--- Summary ---")
    print(f"samples compared: {n}")
    print(f"matching:         {n - mismatches}")
    print(f"differing:        {mismatches}")

    return 0 if mismatches == 0 and len(lines_a) == len(lines_b) else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
