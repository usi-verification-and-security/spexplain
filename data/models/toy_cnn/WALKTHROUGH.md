# toy_cnn: a hand-verifiable ONNX/Network2 test case

This mirrors `data/models/toy.nnet` / `data/datasets/toy.csv`, but exercises the ONNX/`Network2`/
`OpenSMTVerifier2` path instead of the legacy `.nnet`/`Network`/`OpenSMTVerifier` path. Rebuild with
`python python_scripts/build_toy_cnn.py`.

## Architecture

```
input [1,1,4,4] (16 pixels, row-major x0..x15, domain [0,3])
  -> Conv(1,1, kernel=2, stride=2)   4 non-overlapping 2x2 receptive fields
  -> ReLU
  -> MaxPool(kernel=2, stride=2)     collapses the 2x2 conv output to one scalar p
  -> Flatten
  -> Gemm (Linear(1,1))              output = -p + 2   (label = output<0 ? 0 : 1)
```

Conv kernel `[[1,-1],[1,0]]`, applied identically at each of the 4 positions. Because
stride == kernel size, the receptive fields don't overlap, so each conv output is a sum of exactly
3 distinct pixels (the 4th has weight 0), and no pixel is shared between two conv outputs:

```
y00 = x0 - x1 + x4     (top-left quadrant:     row0-1, col0-1)
y01 = x2 - x3 + x6     (top-right quadrant:    row0-1, col2-3)
y10 = x8 - x9 + x12    (bottom-left quadrant:  row2-3, col0-1)
y11 = x10 - x11 + x14  (bottom-right quadrant: row2-3, col2-3)

p = max(relu(y00), relu(y01), relu(y10), relu(y11))
output = -p + 2                       (label 1 iff p <= 2, label 0 iff p > 2)
```

spexplain declares input variables 1-indexed (`x1..x16` = pixel `x0..x15`), so e.g. `y00` appears
in the SMT-LIB2 output as `x1 - x2 + x5`.

## Samples (`data/datasets/toy_cnn.csv`)

| sample | pixels (nonzero only) | p | output | label |
|---|---|---|---|---|
| A | all zero | 0 | +2 | 1 |
| B | x0=3, x4=3 | 6 | -4 | 0 |
| C | x0=3 | 3 | -1 | 0 |
| D | x0=2 | 2 | 0 | 1 (exact boundary: `output=0` is not `<0`) |

Verified three independent ways: PyTorch's own forward pass (`build_toy_cnn.py`, asserted),
`spexplain read-onnx` + `onnx-eval` (Network2's evaluator), and hand arithmetic above -- all agree.

## `psi_d.smt2` / `psi_c0.smt2` / `psi_c1.smt2` (`encode-onnx`, the ONNX analog of `dump-psi`)

`psi_d.smt2` declares `x1..x16` bounded to `[0,3]` -- exactly the declared input domain.

`psi_c<label>.smt2` encodes: domain AND network computation AND "classification is *not* `label`
(with a small margin)". This is the query `check_phi_soundness.sh`-style tooling would conjoin with
a candidate explanation's bounds and check UNSAT, to prove the classification never leaves `label`
within that box. Concretely (see `Framework::Expand::assertClassification`,
`src/spexplain/framework/expand/Expand.cpp:499`, `threshold = 0.015625` = `1/64`):

* `psi_c1.smt2` asserts `p >= 129/64` (i.e. `output <= -1/64`, robustly *not* label 1)
* `psi_c0.smt2` asserts `p <= 127/64` (i.e. `output >= 1/64`, robustly *not* label 0)

Both files' `p` term (`ite574_1` in the raw output) is built from a chain of `ite`-style clause
pairs -- first 4 ReLUs (one per conv output, e.g. `ite466_1 = relu(x1-x2+x5) = relu(y00)`), then 3
pairwise max nodes chaining them into `max(y00,y01,y10,y11)` = the MaxPool. Every coefficient in
both files matches the hand-derived formulas above exactly; run `polish_psi.sed` on the raw files to
follow along, or read the `*.polished.smt2` copies here.

## Explanations (`explain-onnx`)

**`abductive.phi.txt`** (naive feature elimination, produces intervals -- readable directly):

```
(and (= x1 0) (= x3 0) (= x5 0) (= x7 0) (= x9 0) (= x11 0) (= x13 0) (= x15 0))   # sample A
(and (= x2 0) (= x5 3))                                                            # sample B
(and (= x1 3) (= x2 0))                                                            # sample C
(and (= x1 2) (= x3 0) (= x5 0) (= x7 0) (= x9 0) (= x11 0) (= x13 0) (= x15 0))    # sample D
```

Each is sound and hand-checkable (worked examples: sample A fixes exactly the 8 "weight +1"
pixels to 0, so every conv sum reduces to `-(a "weight -1" pixel) <= 0`, forcing `p=0` regardless of
the other 8 pixels' values -- proving label 1 for *any* value of those 8 free pixels. Sample D fixes
`x1=2` and the same 7 zeros, leaving only `x2` free in `[0,3]`; `y00 = 2 - x2` then ranges over
`[-1, 2]`, so `p` ranges over `[0, 2]` -- reaching the boundary `p=2` at `x2=0` but never exceeding
it, so `output` stays `>= 0` and label 1 holds throughout, including at the boundary itself).

**`itp.phi.txt`** (Craig interpolation, formula-style, needs `-S`): weaker (larger) regions derived
from the interpolation proof rather than direct feature elimination, e.g. sample A's is:

```
(and (not (<= 129/64 (+ x1 x5))) (not (<= 129/64 (+ x3 x7)))
     (not (<= 129/64 (+ x11 x15))) (not (<= 129/64 (+ x9 x13))))
```

i.e. each conv sum's two "weight +1" pixels must sum to under `129/64 ~ 2.0156` (the "weight -1"
and "weight 0" pixels of every quadrant are entirely unconstrained).

### A soundness caveat found while checking this (applies beyond this toy model)

The `129/64` bound above is *not* the exact classification threshold (`2`) -- it is `2 + 1/64`, the
same epsilon from `assertClassification`. That 1/64 leaks into the final explanation: a point can
satisfy sample A's itp region (e.g. `x1=2.01, x5=0`, all else `0`: `x1+x5=2.01 < 129/64`) while
`Network2`'s *exact* classifier disagrees with the claimed label:

```
$ ./build/onnx-eval data/models/toy_cnn/toy_cnn.onnx 2.01,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0
Output: [-0.01]
classification (label): 0        # sample A's own label is 1
```

So itp's guarantee is "output `> -1/64`", not "output `>= 0`" -- a soundness gap of up to `1/64` in
output-space, present for *any* single-output classifier (verified: the identical `1/64` constant
also appears in `toy.nnet`'s own `psi_c0.smt2`/`psi_c1.smt2`, via
`grep -o '(/ 1 64)' psi_c0.smt2` after `dump-psi`). It comes from one shared, hardcoded threshold in
`assertClassification` (`Expand.cpp:499`) used for every strategy on any single-output network
(nnet or onnx) -- it is not specific to Network2, itp, or this toy model. Multi-class networks
(argmax over >2 outputs) use threshold `0`, i.e. no such margin
(`Expand.cpp`, `verifierPtr->addClassificationConstraint(label, 0)`).

In practice the gap only matters for points landing within `1/64 ~ 0.0156` of the exact `output=0`
boundary, in *output* value (not input-pixel) units -- rare for real datasets, but a genuine
precision caveat worth knowing before treating an explanation region as exact.
