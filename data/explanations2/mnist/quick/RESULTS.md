# MNIST CNN space explanations — `itp` on `mnist_quick.csv`

Model `data/models/mnist/mnist_cnn.onnx`, strategy `itp` (default `aweak, bstrong`),
release build, per-pixel input domain `[0,255]`.

    Conv(1->4, k4, s2) -> ReLU -> Conv(4->8, k3, s2) -> ReLU -> MaxPool(2,2) -> Flatten -> Gemm(72->10)
    964 hidden ReLU neurons, 72 maxpool windows, 93.3% test accuracy

## Runtime per explanation (seconds)

**Read the caveat below before using these numbers.**

Batch run (`-i6,10` in one process, no per-sample time limit), in run order:

| sample | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| s | 201 | 320 | 243 | 299 | 303 | 234 | 265 | 7610 | 12200 | 14300 |

### The last three numbers are an artifact, not sample difficulty

Re-running **sample 10 alone** takes **256 s**, versus 14300 s in the batch -- a 56x
difference for a *byte-identical* explanation (1194084 bytes, 1139 terms, 1 solver check
in both cases). The cost is not a property of the sample.

Corroborating signs, all pointing at position-in-run rather than sample identity:

* Every sample makes exactly **1 solver check** and yields ~1100 terms, fast or slow.
* The three slow ones are the last three of a single process, in monotonically
  increasing order.
* RSS grows steadily *within* one sample (159 MB -> 356 MB over 4 min), so memory
  accumulation across a batch is a plausible mechanism.

So the honest per-explanation cost for this model is **~250-320 s (~4-5 min)**, uniformly,
and something degrades over a long batch. Root cause not yet identified.

Use `TIMEOUT_PER` for batch runs regardless -- it bounds the damage. `run1-2.sh` sets
`--time-limit-per`; the samples 6-10 rerun was launched by hand without it.

Each explanation is a single-line SMT-LIB interpolant of ~1 MB constraining all 784 pixel
variables -- dense, not a readable pixel region. `ucore min` after `itp` would sparsify it.

## Files

* `itp_aweak_bstrong.phi.txt` / `.times.txt` — all 10 samples (parts 1 and 2 concatenated)
* `part1_1-5.*` — samples 1-5, produced via `scripts/run1-2.sh`
* `part2_6-10.*` — samples 6-10, from a direct `spexplain` call (no time limit)
* `solo10.*` — sample 10 re-run alone: 256 s, identical explanation
* `seq810.*` — samples 8-10 in one process, 900 s cap, to test batch degradation

## Reproducing

    # train + export (writes data/models/mnist/mnist_cnn.{pth,onnx})
    python python_scripts/train_mnist_cnn.py

    # explanations, bounded per sample; run from data/
    CMD=../build/spexplain TIMEOUT_PER=600s bash ./scripts/run1-2.sh \
        explanations2/mnist/quick 'itp' itp_aweak_bstrong

Needs bash 4+ (`/opt/homebrew/bin/bash`); `run-experiments2.sh` additionally needs GNU `parallel`.
