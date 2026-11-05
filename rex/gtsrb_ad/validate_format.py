#!/usr/bin/env python3
"""Validate an activation file by re-implementing spexplain's parser.

This mirrors ``parseSampleNeuronActivationsImpl`` in
``src/spexplain/framework/Config.cpp`` token for token::

    while (ifs >> parsedSampleIdx)
        sampleIdx = parsedSampleIdx - 1            # must be >= 1
        for layerIdx in 1 .. nHiddenLayers:
            loop:
                ifs >> parsedNode                  # must succeed
                if parsedNode == 0: break
                node = abs(parsedNode) - 1, negated = parsedNode < 0

Reading is whitespace-delimited, so line breaks carry no meaning to the parser
-- but the file is written one layer per line, with a blank line between
samples, for human readability, and this validator checks that convention too.
Blank lines are skipped by ``operator>>`` and so are invisible to spexplain.

Also cross-checks every node index against the layer's true size, which the C++
side would only catch later (or not at all).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def parse(path: Path, n_hidden_layers: int, layer_sizes=None):
    text = path.read_text()
    toks = text.split()
    pos = 0
    samples = {}
    errors = []

    def nxt():
        nonlocal pos
        if pos >= len(toks):
            return None
        t = toks[pos]
        pos += 1
        try:
            return int(t)
        except ValueError:
            raise ValueError(f"non-numeric token {t!r} at position {pos - 1}")

    while pos < len(toks):
        s = nxt()
        if s is None:
            break
        if s < 1:
            errors.append(f"sample index {s} must be >= 1")
            break
        per_layer = []
        for layer in range(1, n_hidden_layers + 1):
            idxs = []
            while True:
                n = nxt()
                if n is None:
                    errors.append(f"sample {s}, layer {layer}: unexpected end of file")
                    return samples, errors
                if n == 0:
                    break
                idxs.append(n)
            if layer_sizes is not None:
                size = layer_sizes[layer - 1]
                bad = [i for i in idxs if not (1 <= abs(i) <= size)]
                if bad:
                    errors.append(
                        f"sample {s}, layer {layer}: {len(bad)} index(es) out of range "
                        f"1..{size}, e.g. {bad[:3]}"
                    )
                if len(set(map(abs, idxs))) != len(idxs):
                    errors.append(f"sample {s}, layer {layer}: duplicate indices")
            per_layer.append(idxs)
        samples[s] = per_layer
    return samples, errors


def check_line_layout(path: Path, n_hidden_layers: int):
    """Each record should be 1 index line + n_hidden_layers terminated lines."""
    lines = [l for l in path.read_text().splitlines() if l.strip()]
    if len(lines) % (n_hidden_layers + 1) != 0:
        return False, f"{len(lines)} non-empty lines is not a multiple of {n_hidden_layers + 1}"
    for i in range(0, len(lines), n_hidden_layers + 1):
        for j in range(1, n_hidden_layers + 1):
            if not lines[i + j].rstrip().endswith("0"):
                return False, f"line {i + j + 1} does not end with the 0 terminator"
    return True, f"{len(lines) // (n_hidden_layers + 1)} records, {n_hidden_layers + 1} lines each"


def check_complement(actives: Path, deactives: Path, n_hidden_layers: int, sizes):
    """actives and deactives must partition 1..layer_size for every sample/layer."""
    A, ea = parse(actives, n_hidden_layers, sizes)
    D, ed = parse(deactives, n_hidden_layers, sizes)
    if ea or ed:
        return False, "one of the files failed to parse"
    if set(A) != set(D):
        return False, f"sample sets differ ({len(A)} vs {len(D)})"
    bad = 0
    for s in A:
        for li, (a, d) in enumerate(zip(A[s], D[s])):
            full = set(range(1, sizes[li] + 1))
            if (set(a) & set(d)) or (set(a) | set(d)) != full:
                bad += 1
    n = len(A) * n_hidden_layers
    return bad == 0, f"{n - bad}/{n} sample-layers partition exactly"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+", type=Path)
    ap.add_argument("--complement", nargs=2, type=Path, metavar=("ACTIVES", "DEACTIVES"),
                    default=None, help="check the two files partition every layer")
    ap.add_argument("--meta", type=Path, default=None,
                    help="*_ad_meta.json giving n_hidden_layers and layer sizes")
    ap.add_argument("--layers", type=int, default=None)
    args = ap.parse_args()

    rc = 0
    for f in args.files:
        suffix = "_all" if f.stem.endswith("_all") else ""
        meta_path = args.meta or f.parent / (f.stem.rsplit("_ad_", 1)[0] + f"_ad_meta{suffix}.json")
        nL, sizes = args.layers, None
        if meta_path.exists():
            meta = json.loads(meta_path.read_text())
            nL = meta["n_hidden_layers"]
            sizes = meta["hidden_sizes"]
        if nL is None:
            print(f"{f.name}: cannot determine n_hidden_layers"); rc = 1; continue

        try:
            samples, errors = parse(f, nL, sizes)
        except ValueError as e:
            print(f"{f.name}: PARSE ERROR {e}"); rc = 1; continue

        ok_layout, layout_msg = check_line_layout(f, nL)
        counts = [sum(len(l) for l in v) for v in samples.values()]
        status = "OK" if (not errors and ok_layout) else "PROBLEM"
        print(f"{f.name}: {status}")
        print(f"  parsed {len(samples)} samples, {nL} layers each; "
              f"indices/sample min={min(counts) if counts else 0} "
              f"max={max(counts) if counts else 0} "
              f"mean={sum(counts)/len(counts):.1f}" if counts else "  no samples")
        print(f"  layout: {layout_msg}")
        ids = sorted(samples)
        if ids and ids != list(range(1, len(ids) + 1)):
            print(f"  note: sample ids are not 1..N contiguous (min {ids[0]}, max {ids[-1]})")
        for e in errors[:5]:
            print(f"  ERROR {e}")
        if errors:
            rc = 1

    if args.complement:
        a, d = args.complement
        sfx = "_all" if a.stem.endswith("_all") else ""
        meta_path = a.parent / (a.stem.rsplit("_ad_", 1)[0] + f"_ad_meta{sfx}.json")
        meta = json.loads(meta_path.read_text())
        ok, msg = check_complement(a, d, meta["n_hidden_layers"], meta["hidden_sizes"])
        print(f"complement {a.name} vs {d.name}: {'OK' if ok else 'PROBLEM'} -- {msg}")
        if not ok:
            rc = 1
    sys.exit(rc)


if __name__ == "__main__":
    main()
