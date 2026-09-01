# Comparing the ONNX and NNet SMT encodings

`compare_psi_onnx_vs_nnet.sh` runs the two encoding entry points of `spexplain` on two
representations of the *same* network and diffs the generated SMT-LIB2 queries.

| Side | CLI action    | `main.cpp` function | Network class | Verifier            |
|------|---------------|---------------------|---------------|---------------------|
| ONNX | `encode-onnx` | `readEncodeONNX()`  | `Network2`    | `OpenSMTVerifier2`  |
| NNet | `dump-psi`    | `mainDumpPsi()`     | `Network`     | `OpenSMTVerifier`   |

Both actions write `psi_d.smt2` and `psi_c<N>.smt2` into the **current working directory**, so the
script executes each one inside its own output folder and then compares the results file by file.

> **Note on `.h5`**: `dump-psi` loads models via `Network::fromNNetFile`, so it cannot read a Keras
> `.h5` file. Use the matching `.nnet` model. (`.h5` → `.onnx` conversion is handled separately by
> `convert_heart_attack_h5_to_onnx.sh`.)

## Usage

```bash
cd <repo-root>
./data/scripts/test_scripts/compare_psi_onnx_vs_nnet.sh --model ./data/models/heart_attack/heart_attack-50
```

`--model` is a path *prefix*: it expands to `<prefix>.onnx` and `<prefix>.nnet`. To point at the two
files explicitly:

```bash
./data/scripts/test_scripts/compare_psi_onnx_vs_nnet.sh \
  --onnx ./data/models/heart_attack/heart_attack-50.onnx \
  --nnet ./data/models/heart_attack/heart_attack-50.nnet
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--model <path>` | – | Path prefix expanding to `<path>.onnx` and `<path>.nnet` |
| `--onnx <path>` | – | Explicit ONNX model |
| `--nnet <path>` | – | Explicit NNet model |
| `--outdir <dir>` | `build-debug/psi-compare` | Where the generated files and reports are written |
| `--tol <t>` | `1e-3` | Relative tolerance below which a numeric difference counts as rounding noise |
| `--ignore-bounds` | off | Exclude the input-domain-bounds assertion (#0) from the verdict |
| `--textual-diff` | off | Additionally produce plain `diff -u` output per file |
| `--normalize` | off | Rewrite `.iteNNN_1` names in the files themselves (only affects `--textual-diff`) |
| `--polish` | off | Apply `data/scripts/polish_psi.sed` to the outputs first |
| `--keep` | off | Do not wipe an existing output directory |
| `--` | – | Forward all remaining arguments to both `spexplain` invocations |

### Recommended invocation

```bash
./data/scripts/test_scripts/compare_psi_onnx_vs_nnet.sh \
  --model ./data/models/heart_attack/heart_attack-50 --ignore-bounds
```

## Why not a plain `diff`?

Almost the whole formula lives on two very long `(assert (and ...))` lines — for
`heart_attack-50` the classification assertion is a single 56 KB line. A textual `diff` therefore
reports "this line changed" and tells you nothing about *how much* changed. Worse, OpenSMT numbers
the auxiliary `.iteNNN_1` variables it introduces for ReLU encodings arbitrarily, so the numbering
differs between the two runs even when the encodings are equivalent, producing ~100 lines of pure
naming noise.

The script therefore delegates to **`psi_diff.py`**, which:

1. parses both files as s-expressions;
2. renumbers `.iteNNN_M` names by order of first appearance, so the naming noise disappears;
3. splits each `(assert (and ...))` into its top-level conjuncts;
4. computes a *skeleton* for every conjunct — its structure with all purely numeric subterms
   (`5`, `(- 7)`, `(/ (- 659) 5475)`, ...) replaced by `#` — and matches the conjuncts of the two
   files by skeleton, pairing ambiguous cases greedily by smallest numeric distance;
5. compares the numeric literals of every matched pair one at a time, as exact `Fraction`s.

Each conjunct then lands in one of five buckets:

| Bucket | Meaning |
|---|---|
| `identical` | Equal term by term, including every number |
| `close` | Same structure, every number within `--tol` relative difference — i.e. rounding |
| `differing` | Same structure, but at least one number is genuinely different |
| `only-A` | Conjunct in the ONNX encoding with no structural counterpart in the NNet one |
| `only-B` | Conjunct in the NNet encoding with no structural counterpart in the ONNX one |

`only-A`/`only-B` are the interesting ones: they mean the two encodings are *structurally*
different (a missing constraint, a different layer, a flipped comparison).

## Output

```
FILE             A-CONJ B-CONJ  IDENTICAL   CLOSE  DIFFERING  ONLY-A  ONLY-B    WORST-REL  STATUS
psi_c0.smt2         127    127         78      23          0       0       0  2.21867e-05  EQUIVALENT
psi_c1.smt2         127    127         78      23          0       0       0  2.21867e-05  EQUIVALENT
psi_d.smt2           26     26          0       0          0       0       0            0  IDENTICAL
```

`IDENTICAL` means literally equal; `EQUIVALENT` means equal once rounding within `--tol` is
allowed. `WORST-REL` is the largest relative difference among the `close` conjuncts, so it tells you
how big the rounding error actually is.

A full report per file is written to `<outdir>/<file>.report`; it names the offending conjuncts and,
for each, prints the worst-differing literal on both sides:

```
    conjunct A[9] / B[9]:
      max relative difference: 1.26073e-05
      worst literal: A = -79318/61125 (-1.29763599182)   B = -79319/61125 (-1.29765235174)
      literals differing: 1 of 15
```

Exit codes: `0` agreement, `1` real differences, `≥2` an error.

## Using `psi_diff.py` directly

```bash
python3 data/scripts/test_scripts/psi_diff.py <a.smt2> <b.smt2> [options]
```

| Flag | Description |
|---|---|
| `--tol <t>` | Relative tolerance (default `1e-3`) |
| `--ignore-assert N` | Exclude assertion `N` from the verdict (repeatable) |
| `--max-show N` | How many example conjuncts to print per category (default 5) |
| `--width N` | Truncate printed terms (default 160, `0` disables) |
| `--stats` | One machine-readable line: `a b identical close differing only-a only-b worst-rel` |
| `--quiet` | Summary table only |
| `--no-rename-aux` | Keep the original `.iteNNN_M` names |

It is not tied to this script — it works on any pair of SMT-LIB2 files built from
`(assert (and ...))` conjunctions.

## Interpreting the expected differences

For the `heart_attack` models the two encodings agree except for two things:

1. **Input domain bounds in every file.** The `.nnet` format stores per-input minimum/maximum values,
   which are emitted as hard bounds. ONNX carries no such information, so `Network2` falls back to the
   default bounds `0`/`1` (`Network2::setInputShape`). Use `Network2::setInputMinimums` /
   `setInputMaximums` to supply real bounds.
2. **Occasional last-digit differences in weight coefficients.** `.nnet` stores decimal text while
   ONNX stores float32, so a few weights convert to slightly different rationals
   (e.g. `-79318/61125` vs `-79319/61125`). These land in the `close` bucket.

With `--ignore-bounds`, all four `heart_attack` models report **zero** `differing`, `only-A` and
`only-B` conjuncts:

| Model | Conjuncts | identical | close | differing | only-A | only-B | worst relative |
|---|---|---|---|---|---|---|---|
| `heart_attack_1hidden`   |  67 | 37 |  4 | 0 | 0 | 0 | 3.6e-06 |
| `heart_attack-20-10`     |  87 | 43 | 18 | 0 | 0 | 0 | 1.5e-05 |
| `heart_attack-10-20-10`  | 107 | 67 | 14 | 0 | 0 | 0 | 2.3e-05 |
| `heart_attack-50`        | 127 | 78 | 23 | 0 | 0 | 0 | 2.2e-05 |

Layer structure, ReLU encodings, biases and the classification constraints match exactly, which is
the property this script is meant to verify.

## Supported layers

`OpenSMTVerifier2` currently encodes `fc` and `relu` layers. It also:

- folds an `add` layer that immediately follows an `fc` layer into that layer's bias (ONNX exporters
  frequently emit `MatMul` + `Add` instead of a single `Gemm`);
- drops a trailing `sigmoid`, so the encoded output is the pre-sigmoid logit, matching the semantics
  of the `.nnet` networks.

Any other layer type raises `Unimplemented! Unsupported layer type: <type>`.
