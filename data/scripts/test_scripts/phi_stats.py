#!/usr/bin/env python3
"""Aggregate `spexplain` statistics files (`-s`) and time splits (`--output-times`).

The stats file produced by `spexplain -s <file>` contains one block per sample, e.g.

    sample [1/10]: 64,1,2,125,309,0,1,131,1,1.8,1,0,3
    expected output: 0
    computed output: 0
    #checks: 13
    #features: 10/13
    #fixed features: 0/13
    #terms: 10

This tool parses one or more such files and prints a side-by-side table of the averages,
so two runs (e.g. the same model as `.nnet` and as `.onnx`) can be compared directly.
"""
from __future__ import annotations

import argparse
import re
import sys

FIELD_RE = re.compile(r"^#(checks|features|fixed features|terms):\s*(\d+)(?:/(\d+))?\s*$")
SAMPLE_RE = re.compile(r"^sample \[(\d+)/(\d+)\]:")
OUTPUT_RE = re.compile(r"^(expected|computed) output:\s*(\S+)\s*$")


def parse_stats(path: str) -> list[dict]:
    samples: list[dict] = []
    current: dict | None = None
    with open(path) as fh:
        for line in fh:
            line = line.rstrip("\n")
            if SAMPLE_RE.match(line):
                current = {}
                samples.append(current)
                continue
            if current is None:
                continue
            m = OUTPUT_RE.match(line)
            if m:
                current[m.group(1)] = m.group(2)
                continue
            m = FIELD_RE.match(line)
            if m:
                key, value, total = m.group(1), int(m.group(2)), m.group(3)
                current[key] = value
                if total is not None:
                    current[key + " total"] = int(total)
    return samples


def parse_times(path: str) -> list[float]:
    times: list[float] = []
    try:
        with open(path) as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    times.append(float(line))
                except ValueError:
                    # `spexplain` writes a marker instead of a number on timeout.
                    pass
    except OSError:
        pass
    return times


def mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description="Aggregate spexplain stats/times files.")
    ap.add_argument("--label", action="append", default=[], required=True)
    ap.add_argument("--stats", action="append", default=[], required=True)
    ap.add_argument("--times", action="append", default=[])
    args = ap.parse_args(argv)

    if len(args.label) != len(args.stats):
        print("error: --label and --stats must be given the same number of times", file=sys.stderr)
        return 2
    times_files = args.times + [None] * (len(args.stats) - len(args.times))

    rows = []
    for label, stats_path, times_path in zip(args.label, args.stats, times_files):
        samples = parse_stats(stats_path)
        times = parse_times(times_path) if times_path else []
        if not samples:
            print("error: no samples parsed from %s" % stats_path, file=sys.stderr)
            return 2
        n_vars = samples[0].get("features total", 0)
        correct = sum(1 for s in samples
                      if s.get("expected") is not None and s.get("expected") == s.get("computed"))
        rows.append({
            "label": label,
            "n": len(samples),
            "vars": n_vars,
            "features": mean(s.get("features", 0) for s in samples),
            "fixed": mean(s.get("fixed features", 0) for s in samples),
            "terms": mean(s.get("terms", 0) for s in samples),
            "checks": mean(s.get("checks", 0) for s in samples),
            "time": mean(times),
            "total_time": sum(times),
            "correct": correct,
        })

    header = "%-8s %5s %6s %10s %10s %9s %9s %11s %11s %9s" % (
        "RUN", "N", "VARS", "AVG-FEAT", "AVG-FIXED", "AVG-TERMS", "AVG-CHK",
        "AVG-TIME(s)", "TOT-TIME(s)", "CORRECT")
    print(header)
    print("-" * len(header))
    for r in rows:
        print("%-8s %5d %6d %10.2f %10.2f %9.2f %9.2f %11.4f %11.3f %5d/%-3d" % (
            r["label"], r["n"], r["vars"], r["features"], r["fixed"], r["terms"],
            r["checks"], r["time"], r["total_time"], r["correct"], r["n"]))

    if len(rows) == 2:
        a, b = rows
        print()
        print("--- %s vs %s (relative to %s) ---" % (a["label"], b["label"], a["label"]))
        for key, name in (("features", "avg #features"), ("fixed", "avg #fixed"),
                          ("terms", "avg #terms"), ("checks", "avg #checks"),
                          ("time", "avg time")):
            av, bv = a[key], b[key]
            if av == 0:
                delta = "n/a" if bv == 0 else "+inf"
            else:
                delta = "%+.1f%%" % ((bv - av) / av * 100.0)
            print("  %-14s %10.4f -> %10.4f   %s" % (name, av, bv, delta))

    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
