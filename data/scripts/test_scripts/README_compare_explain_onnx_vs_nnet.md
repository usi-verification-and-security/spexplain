# Comparing ONNX and NNet explanations

`compare_explain_onnx_vs_nnet.sh` runs the two explanation entry points of `spexplain` on two
representations of the *same* network, over the *same* dataset, and compares the resulting
explanations feature by feature.

| Side | CLI action     | `main.cpp` function  | Network class | Verifier            |
|------|----------------|-----------------------|---------------|----------------------|
| ONNX | `explain-onnx` | `mainExplainOnnx()`  | `Network2`    | `OpenSMTVerifier2`  |
| NNet | `explain`      | `mainExplain()`      | `Network`     | `OpenSMTVerifier`   |

Unlike `compare_psi_onnx_vs_nnet.sh` (which diffs the SMT-LIB2 *encoding*), this script diffs the
*output* of the explanation search itself, so it is sensitive to behavioral differences between
`OpenSMTVerifier` and `OpenSMTVerifier2` (e.g. missing input domain bounds for ONNX), not just
encoding artefacts.

## Usage

```bash
cd <repo-root>
./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
  --model ./data/models/heart_attack/heart_attack-50 \
  --dataset ./data/datasets/heart_attack/heart_attack_quick.csv
```

`--model` is a path *prefix*: it expands to `<prefix>.onnx` and `<prefix>.nnet`. To point at the
two files explicitly:

```bash
./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
  --onnx ./data/models/heart_attack/heart_attack-50.onnx \
  --nnet ./data/models/heart_attack/heart_attack-50.nnet \
  --dataset ./data/datasets/heart_attack/heart_attack_quick.csv
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--model <path>` | – | Path prefix expanding to `<path>.onnx` and `<path>.nnet` |
| `--onnx <path>` | – | Explicit ONNX model |
| `--nnet <path>` | – | Explicit NNet model |
| `--dataset <csv>` | – | Dataset CSV shared by both runs (required) |
| `--strategy <spec>` | `abductive` | Explanation strategy spec forwarded to both runs |
| `--input-min <v1,v2,...>` | – | Per-feature input domain minimums, forwarded to the ONNX side only |
| `--input-max <v1,v2,...>` | – | Per-feature input domain maximums, forwarded to the ONNX side only |
| `--outdir <dir>` | `build-debug/explain-compare` | Where the generated files/logs are written |
| `--keep` | off | Do not wipe an existing output directory |
| `--` | – | Forward all remaining arguments to both `spexplain` invocations |

## Why `abductive` by default, and why input bounds matter

`abductive` produces one `(= xI v)` conjunct per feature it could not eliminate, which is trivial
to compare quantitatively (how many/which features are fixed). `itp`'s result is a generic linear
formula, not tied one-to-one to input features, so it is not directly comparable this way.

ONNX models carry **no per-feature input domain metadata**, unlike `.nnet` files. `Network2`
silently falls back to a `[0,1]` bound for every feature unless told otherwise
(`Network2::setInputMinimums`/`setInputMaximums`). Left at the `[0,1]` default, the
`abductive`/`itp` search can free almost any input feature without ever finding a genuine
counterexample (since the real feature range, e.g. `heart_attack`'s `x5` cholesterol range of
`126..594`, is squashed into `[0,1]`), producing drastically weaker explanations than the `.nnet`
side even though the two networks compute the same function. Passing the real bounds via
`--input-min`/`--input-max` (forwarded to `spexplain explain-onnx` as-is) fixes this.

If neither flag is given and the resolved ONNX path contains `heart_attack`, the script
automatically supplies the known heart_attack feature bounds:

```
min: 29,0,0,94,126,0,0,71,0,0,0,0,0
max: 77,1,3,200,594,1,2,202,1,6.2,2,4,3
```

## Output

```
SAMPLE    A-FIXED  B-FIXED   COMMON ONLY-A/B  STATUS
----------------------------------------------------
1               9       12        9    0/3    DIFFERS
    only in B (onnx): x1, x12, x6
...
6               8        8        8    0/0    MATCH
...
--- Summary ---
samples compared: 10
matching:         6
differing:        4
```

`A` is always the `.nnet` side, `B` the ONNX side. `MATCH` means both sides fixed exactly the same
features to exactly the same values. `DIFFERS` with only `only-B` entries (never `only-A`) means
the ONNX side found a valid but non-minimal superset of the `.nnet` explanation — expected, since
`abductive`'s greedy elimination order is not guaranteed to find a globally minimal explanation,
and tiny float32-vs-decimal weight differences between the two model formats can change which
features the greedy search happens to keep. An `only-A` entry (a feature the ONNX side dropped but
the `.nnet` side needed) would indicate a real soundness problem and should be investigated.

Exit code: `0` if every sample matches exactly, `1` otherwise.

## Using `phi_diff.py` directly

```bash
python3 data/scripts/test_scripts/phi_diff.py <a.phi.txt> <b.phi.txt>
```

Compares two `spexplain` `.phi.txt` files line by line (one sample per line), assuming each line is
an `abductive`-style `(and (= xI v) ...)` conjunction. Reports, per sample, the fixed-feature
overlap between the two files and any value mismatches on features fixed by both sides.
