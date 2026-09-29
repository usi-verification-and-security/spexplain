# CNN-bench: layer-by-layer architecture reference

Node-level structure of all **144 models** in the CNN-bench family — 36 architectures trained
identically on each of four image datasets: **MNIST**, **GTSRB**, **CIFAR-10** and
**Imagenette-64**.

Every table is generated directly from the exported `.onnx` graphs by walking the ONNX node list
with shape inference, so the shapes, parameter counts and ReLU counts are the ones spexplain's
`OnnxParser` actually sees — not a reconstruction from the training code. All 432 per-model
totals (144 models × params / ReLUs / pool neurons) were cross-checked against each dataset's
`manifest.csv` and agree exactly.

**Related documents.** [`CNN_BENCH_REPORT.md`](CNN_BENCH_REPORT.md) is the full experimental
report — training recipe, accuracies, validation, average-pool support. Its §11 carries a second
set of tables for the same 144 models in a *layer-level* format (one row per layer, with an
Activation column). This file is the *node-level* view: `Conv` and `ReLU` are separate rows, and
parameters, ReLU neurons and pool neurons are broken out per node.

---

## Contents

- [How to read the tables](#how-to-read-the-tables)
- [The architecture family](#the-architecture-family)
- [Why ReLU count is the axis](#why-relu-count-is-the-axis)
- [Size at a glance](#size-at-a-glance)
- [MNIST](#mnist)
- [GTSRB](#gtsrb)
- [CIFAR-10](#cifar-10)
- [Imagenette-64](#imagenette-64)
- [How the same architecture scales across datasets](#how-the-same-architecture-scales-across-datasets)
- [Provenance](#provenance)

---

## How to read the tables

Each row is **one ONNX node**. A "convolutional layer" in the usual sense spans two or three
rows: the `Conv` that holds the parameters, an optional pooling node, then the `ReLU`.

| Column | Meaning |
|---|---|
| **#** | Position in the ONNX node list (execution order). |
| **Layer** | Node type, with its output width where meaningful (`Conv16`, `FC32`). |
| **Config** | Kernel, stride and padding for `Conv`/pooling; in → out widths for `FC`. |
| **Input shape** | Shape entering the node, as `C×H×W` (or `N` for flat vectors). |
| **Output shape** | Shape leaving the node. |
| **Params** | Learnable values in this node: weights + bias. Non-zero only for `Conv` and `FC`. |
| **ReLUs** | ReLU neurons — the element count of the node's output. Each is a case split for the SMT solver. |
| **Pool** | Pooling output neurons. For `MaxPool` each is a 4-way maximum: a fresh variable plus a disjunction. For `AvgPool` each is a mean — **a plain linear term, no variable and no disjunction**. |

**The batch axis is dropped.** The graphs declare a `[1,C,H,W]` input; `Network2` strips the
leading axis, so shapes are written `C×H×W`. A flat CSV row maps onto that in **CHW row-major**
order — all of channel 0, then channel 1, then channel 2.

**Parameter arithmetic.** For a convolution, `params = out_ch × in_ch × kH × kW + out_ch`; for a
fully-connected layer, `params = in × out + out`. BatchNorm is used during training but folded
exactly into the preceding convolution at export, so it costs no parameters and appears in no
graph.

---

## The architecture family

A **6 × 6 grid**: six depths (1..6 convolutions) × six variants.

| id | width | downsampling | pool branching |
|---|---|---|---|
| `S<d>` | wide 8/16/32 | stride-2 convolutions | none |
| `P<d>` | wide 8/16/32 | 2×2 **max** pool | one 4-way max per pooled neuron |
| `A<d>` | wide 8/16/32 | 2×2 **average** pool | **none** — averaging is affine |
| `NS<d>` | narrow 4/8/16 | stride-2 convolutions | none |
| `NP<d>` | narrow 4/8/16 | 2×2 **max** pool | one 4-way max per pooled neuron |
| `NA<d>` | narrow 4/8/16 | 2×2 **average** pool | **none** — averaging is affine |

Everything is derived from one table of strided token lists at import time, so the grid cannot
drift: `P<d>` replaces every non-stem `c<N>s2` with `p, c<N>` and appends a final `p`; `A<d>` is
`P<d>` with `a` for `p`; `NS`/`NP`/`NA` are the 4/8/16 twins. `archs.py` asserts all of it.

Two consequences the tables make visible:

- **`P<d>`/`A<d>` and `NP<d>`/`NA<d>` are controlled pairs.** Within a pair the shapes, ReLU
  counts and parameter counts are *identical*; only the pooling operator differs. That isolates
  the cost of max-pool branching, once at each width.
- **`S<d>` has more ReLUs than its pooled siblings**, because its feature maps are not pooled
  before the activation. See the ReLU placement note below.

### Token grammar

| Token | Layer | Effect on spatial size |
|---|---|---|
| `c<N>` | Conv, N channels, 3×3, stride 1, pad 1 | unchanged |
| `c<N>s2` | Conv, N channels, 4×4, stride 2, pad 1 | exactly halved |
| `p` | Max pool, 2×2, stride 2 | halved (floor) |
| `a` | Average pool, 2×2, stride 2 | halved (floor) |

Every convolution is followed by a ReLU. Every network ends with **Flatten → FC(32) + ReLU →
FC(classes)**, emitting raw logits — softmax lives only in the training loss.

### ReLU placement

A convolution immediately followed by a pool emits `Conv → Pool → ReLU`, **not**
`Conv → ReLU → Pool`. In the tables below that is why a pooled model's `ReLU` row always comes
*after* its pooling row, and why its ReLU count is a quarter of the unpooled feature map.

For **max** pooling the two orders compute exactly the same function — ReLU is monotone
non-decreasing, so `max(relu(a), relu(b)) == relu(max(a, b))` — and applying it after the pool
puts it on the 4× smaller map for free. For **average** pooling the orders are *not* equal
(`avg(relu([-1,1])) = 0.5` while `relu(avg([-1,1])) = 0`); pooling first is a deliberate choice
that keeps the `A`/`NA` ReLU counts equal to `P`/`NP`, which is what makes the pairs comparable.

The ReLU is never dropped: it is the only nonlinearity between convolutions.

### The 36 architectures

*Tables are split by pooling kind -- **no pooling**, **average pooling**, **max pooling** -- and within each group the models are ordered **smallest first**, by ReLU count ascending. That order happens to be the same on all four datasets, so a row position means the same model everywhere.*

#### No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

| id | convs | pools | token sequence |
|---|---:|---:|---|
| `NS1` | 1 | 0 | `c4s2` |
| `NS2` | 2 | 0 | `c4s2 c8s2` |
| `S1` | 1 | 0 | `c8s2` |
| `NS3` | 3 | 0 | `c4s2 c8s2 c8` |
| `S2` | 2 | 0 | `c8s2 c16s2` |
| `NS4` | 4 | 0 | `c4s2 c4 c8s2 c8` |
| `NS5` | 5 | 0 | `c4s2 c4 c8s2 c8 c16s2` |
| `NS6` | 6 | 0 | `c4s2 c4 c8s2 c8 c16s2 c16` |
| `S3` | 3 | 0 | `c8s2 c16s2 c16` |
| `S4` | 4 | 0 | `c8s2 c8 c16s2 c16` |
| `S5` | 5 | 0 | `c8s2 c8 c16s2 c16 c32s2` |
| `S6` | 6 | 0 | `c8s2 c8 c16s2 c16 c32s2 c32` |

#### Average pooling

`A<d>` / `NA<d>` -- 2x2 average pooling. Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

| id | convs | pools | token sequence |
|---|---:|---:|---|
| `NA1` | 1 | 1 | `c4s2 a` |
| `NA2` | 2 | 2 | `c4s2 a c8 a` |
| `A1` | 1 | 1 | `c8s2 a` |
| `A2` | 2 | 2 | `c8s2 a c16 a` |
| `NA3` | 3 | 2 | `c4s2 a c8 c8 a` |
| `A3` | 3 | 2 | `c8s2 a c16 c16 a` |
| `NA4` | 4 | 2 | `c4s2 c4 a c8 c8 a` |
| `NA5` | 5 | 3 | `c4s2 c4 a c8 c8 a c16 a` |
| `NA6` | 6 | 3 | `c4s2 c4 a c8 c8 a c16 c16 a` |
| `A4` | 4 | 2 | `c8s2 c8 a c16 c16 a` |
| `A5` | 5 | 3 | `c8s2 c8 a c16 c16 a c32 a` |
| `A6` | 6 | 3 | `c8s2 c8 a c16 c16 a c32 c32 a` |

#### Max pooling

`P<d>` / `NP<d>` -- 2x2 max pooling. Each output needs a fresh variable with `m >= a_i` plus a 4-way disjunction `\/ m = a_i`.

| id | convs | pools | token sequence |
|---|---:|---:|---|
| `NP1` | 1 | 1 | `c4s2 p` |
| `NP2` | 2 | 2 | `c4s2 p c8 p` |
| `P1` | 1 | 1 | `c8s2 p` |
| `P2` | 2 | 2 | `c8s2 p c16 p` |
| `NP3` | 3 | 2 | `c4s2 p c8 c8 p` |
| `P3` | 3 | 2 | `c8s2 p c16 c16 p` |
| `NP4` | 4 | 2 | `c4s2 c4 p c8 c8 p` |
| `NP5` | 5 | 3 | `c4s2 c4 p c8 c8 p c16 p` |
| `NP6` | 6 | 3 | `c4s2 c4 p c8 c8 p c16 c16 p` |
| `P4` | 4 | 2 | `c8s2 c8 p c16 c16 p` |
| `P5` | 5 | 3 | `c8s2 c8 p c16 c16 p c32 p` |
| `P6` | 6 | 3 | `c8s2 c8 p c16 c16 p c32 c32 p` |

---

## Why ReLU count is the axis

Models are ordered by ReLU count rather than parameter count, because that is what an SMT solver
pays for: every ReLU is a case split, while a wide `Gemm` is just more linear arithmetic in
`QF_LRA`. The two measures disagree sharply, and often invert.

| MNIST | `S1` | `P6` |
|---|---:|---:|
| Convolutions | 1 | 6 |
| Parameters | 50,674 | 19,482 |
| ReLUs | 1,600 | 3,240 |
| Pool neurons | 0 | 568 |

`S1` carries **2.6× the parameters** of `P6` and is far the
easier model to verify: almost all its weight sits in one fully-connected layer reading a large
un-pooled feature map, which is cheap linear arithmetic.

---

## Size at a glance

ReLU neurons per model. `P<d>`/`A<d>` share a value, as do `NP<d>`/`NA<d>`, by construction.

**No pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NS1` | 816 | 1,056 | 1,056 | 4,128 |
| `NS2` | 1,208 | 1,568 | 1,568 | 6,176 |
| `S1` | 1,600 | 2,080 | 2,080 | 8,224 |
| `NS3` | 1,600 | 2,080 | 2,080 | 8,224 |
| `S2` | 2,384 | 3,104 | 3,104 | 12,320 |
| `NS4` | 2,384 | 3,104 | 3,104 | 12,320 |
| `NS5` | 2,528 | 3,360 | 3,360 | 13,344 |
| `NS6` | 2,672 | 3,616 | 3,616 | 14,368 |
| `S3` | 3,168 | 4,128 | 4,128 | 16,416 |
| `S4` | 4,736 | 6,176 | 6,176 | 24,608 |
| `S5` | 5,024 | 6,688 | 6,688 | 26,656 |
| `S6` | 5,312 | 7,200 | 7,200 | 28,704 |

**Average pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NA1` | 228 | 288 | 288 | 1,056 |
| `NA2` | 300 | 416 | 416 | 1,568 |
| `A1` | 424 | 544 | 544 | 2,080 |
| `A2` | 568 | 800 | 800 | 3,104 |
| `NA3` | 692 | 928 | 928 | 3,616 |
| `A3` | 1,352 | 1,824 | 1,824 | 7,200 |
| `NA4` | 1,476 | 1,952 | 1,952 | 7,712 |
| `NA5` | 1,492 | 2,016 | 2,016 | 7,968 |
| `NA6` | 1,636 | 2,272 | 2,272 | 8,992 |
| `A4` | 2,920 | 3,872 | 3,872 | 15,392 |
| `A5` | 2,952 | 4,000 | 4,000 | 15,904 |
| `A6` | 3,240 | 4,512 | 4,512 | 17,952 |

**Max pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NP1` | 228 | 288 | 288 | 1,056 |
| `NP2` | 300 | 416 | 416 | 1,568 |
| `P1` | 424 | 544 | 544 | 2,080 |
| `P2` | 568 | 800 | 800 | 3,104 |
| `NP3` | 692 | 928 | 928 | 3,616 |
| `P3` | 1,352 | 1,824 | 1,824 | 7,200 |
| `NP4` | 1,476 | 1,952 | 1,952 | 7,712 |
| `NP5` | 1,492 | 2,016 | 2,016 | 7,968 |
| `NP6` | 1,636 | 2,272 | 2,272 | 8,992 |
| `P4` | 2,920 | 3,872 | 3,872 | 15,392 |
| `P5` | 2,952 | 4,000 | 4,000 | 15,904 |
| `P6` | 3,240 | 4,512 | 4,512 | 17,952 |

Pooling neurons (zero for `S`/`NS`; a disjunction each for `P`/`NP`, free for `A`/`NA`):

**Average pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NA1` | 196 | 256 | 256 | 1,024 |
| `NA2` | 268 | 384 | 384 | 1,536 |
| `A1` | 392 | 512 | 512 | 2,048 |
| `A2` | 536 | 768 | 768 | 3,072 |
| `NA3` | 268 | 384 | 384 | 1,536 |
| `A3` | 536 | 768 | 768 | 3,072 |
| `NA4` | 268 | 384 | 384 | 1,536 |
| `NA5` | 284 | 448 | 448 | 1,792 |
| `NA6` | 284 | 448 | 448 | 1,792 |
| `A4` | 536 | 768 | 768 | 3,072 |
| `A5` | 568 | 896 | 896 | 3,584 |
| `A6` | 568 | 896 | 896 | 3,584 |

**Max pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NP1` | 196 | 256 | 256 | 1,024 |
| `NP2` | 268 | 384 | 384 | 1,536 |
| `P1` | 392 | 512 | 512 | 2,048 |
| `P2` | 536 | 768 | 768 | 3,072 |
| `NP3` | 268 | 384 | 384 | 1,536 |
| `P3` | 536 | 768 | 768 | 3,072 |
| `NP4` | 268 | 384 | 384 | 1,536 |
| `NP5` | 284 | 448 | 448 | 1,792 |
| `NP6` | 284 | 448 | 448 | 1,792 |
| `P4` | 536 | 768 | 768 | 3,072 |
| `P5` | 568 | 896 | 896 | 3,584 |
| `P6` | 568 | 896 | 896 | 3,584 |

Parameters:

**No pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NS1` | 25,518 | 34,415 | 33,326 | 131,630 |
| `NS2` | 13,494 | 18,551 | 17,462 | 66,614 |
| `S1` | 50,674 | 67,379 | 66,290 | 262,898 |
| `NS3` | 14,078 | 19,135 | 18,046 | 67,198 |
| `S2` | 27,650 | 36,675 | 35,586 | 133,890 |
| `NS4` | 14,226 | 19,283 | 18,194 | 67,346 |
| `NS5` | 8,354 | 13,155 | 12,066 | 36,642 |
| `NS6` | 10,674 | 15,475 | 14,386 | 38,962 |
| `S3` | 29,970 | 38,995 | 37,906 | 136,210 |
| `S4` | 30,554 | 39,579 | 38,490 | 136,794 |
| `S5` | 22,906 | 31,419 | 30,330 | 79,482 |
| `S6` | 32,154 | 40,667 | 39,578 | 88,730 |

**Average pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NA1` | 6,702 | 9,839 | 8,750 | 33,326 |
| `NA2` | 3,030 | 6,039 | 4,950 | 17,238 |
| `A1` | 13,042 | 18,227 | 17,138 | 66,290 |
| `A2` | 6,274 | 11,203 | 10,114 | 34,690 |
| `NA3` | 3,614 | 6,623 | 5,534 | 17,822 |
| `A3` | 8,594 | 13,523 | 12,434 | 37,010 |
| `NA4` | 3,762 | 6,771 | 5,682 | 17,970 |
| `NA5` | 3,138 | 5,891 | 4,802 | 10,946 |
| `NA6` | 5,458 | 8,211 | 7,122 | 13,266 |
| `A4` | 9,178 | 14,107 | 13,018 | 37,594 |
| `A5` | 10,234 | 14,651 | 13,562 | 25,850 |
| `A6` | 19,482 | 23,899 | 22,810 | 35,098 |

**Max pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NP1` | 6,702 | 9,839 | 8,750 | 33,326 |
| `NP2` | 3,030 | 6,039 | 4,950 | 17,238 |
| `P1` | 13,042 | 18,227 | 17,138 | 66,290 |
| `P2` | 6,274 | 11,203 | 10,114 | 34,690 |
| `NP3` | 3,614 | 6,623 | 5,534 | 17,822 |
| `P3` | 8,594 | 13,523 | 12,434 | 37,010 |
| `NP4` | 3,762 | 6,771 | 5,682 | 17,970 |
| `NP5` | 3,138 | 5,891 | 4,802 | 10,946 |
| `NP6` | 5,458 | 8,211 | 7,122 | 13,266 |
| `P4` | 9,178 | 14,107 | 13,018 | 37,594 |
| `P5` | 10,234 | 14,651 | 13,562 | 25,850 |
| `P6` | 19,482 | 23,899 | 22,810 | 35,098 |

---

## MNIST

Handwritten digits, greyscale.

- **Input:** `1×28×28`, pixel values in `[0,1]`
- **Classes:** 10
- **Files:** `data/models/mnist/cnn-bench/<id>.onnx` and `<id>.pth`

### No pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NS1`](#mnist-ns1) | 1 | 0 | 816 | 0 | 25,518 | 98.21% |
| [`NS2`](#mnist-ns2) | 2 | 0 | 1,208 | 0 | 13,494 | 98.57% |
| [`S1`](#mnist-s1) | 1 | 0 | 1,600 | 0 | 50,674 | 98.64% |
| [`NS3`](#mnist-ns3) | 3 | 0 | 1,600 | 0 | 14,078 | 98.84% |
| [`S2`](#mnist-s2) | 2 | 0 | 2,384 | 0 | 27,650 | 99.06% |
| [`NS4`](#mnist-ns4) | 4 | 0 | 2,384 | 0 | 14,226 | 99.00% |
| [`NS5`](#mnist-ns5) | 5 | 0 | 2,528 | 0 | 8,354 | 98.98% |
| [`NS6`](#mnist-ns6) | 6 | 0 | 2,672 | 0 | 10,674 | 99.15% |
| [`S3`](#mnist-s3) | 3 | 0 | 3,168 | 0 | 29,970 | 99.36% |
| [`S4`](#mnist-s4) | 4 | 0 | 4,736 | 0 | 30,554 | 99.35% |
| [`S5`](#mnist-s5) | 5 | 0 | 5,024 | 0 | 22,906 | 99.52% |
| [`S6`](#mnist-s6) | 6 | 0 | 5,312 | 0 | 32,154 | 99.51% |

### Average pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NA1`](#mnist-na1) | 1 | 1 | 228 | 196 | 6,702 | 96.43% |
| [`NA2`](#mnist-na2) | 2 | 2 | 300 | 268 | 3,030 | 95.43% |
| [`A1`](#mnist-a1) | 1 | 1 | 424 | 392 | 13,042 | 97.22% |
| [`A2`](#mnist-a2) | 2 | 2 | 568 | 536 | 6,274 | 97.53% |
| [`NA3`](#mnist-na3) | 3 | 2 | 692 | 268 | 3,614 | 97.78% |
| [`A3`](#mnist-a3) | 3 | 2 | 1,352 | 536 | 8,594 | 98.74% |
| [`NA4`](#mnist-na4) | 4 | 2 | 1,476 | 268 | 3,762 | 98.27% |
| [`NA5`](#mnist-na5) | 5 | 3 | 1,492 | 284 | 3,138 | 97.99% |
| [`NA6`](#mnist-na6) | 6 | 3 | 1,636 | 284 | 5,458 | 98.53% |
| [`A4`](#mnist-a4) | 4 | 2 | 2,920 | 536 | 9,178 | 99.10% |
| [`A5`](#mnist-a5) | 5 | 3 | 2,952 | 568 | 10,234 | 99.17% |
| [`A6`](#mnist-a6) | 6 | 3 | 3,240 | 568 | 19,482 | 99.21% |

### Max pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NP1`](#mnist-np1) | 1 | 1 | 228 | 196 | 6,702 | 96.84% |
| [`NP2`](#mnist-np2) | 2 | 2 | 300 | 268 | 3,030 | 97.36% |
| [`P1`](#mnist-p1) | 1 | 1 | 424 | 392 | 13,042 | 98.10% |
| [`P2`](#mnist-p2) | 2 | 2 | 568 | 536 | 6,274 | 98.56% |
| [`NP3`](#mnist-np3) | 3 | 2 | 692 | 268 | 3,614 | 97.86% |
| [`P3`](#mnist-p3) | 3 | 2 | 1,352 | 536 | 8,594 | 99.00% |
| [`NP4`](#mnist-np4) | 4 | 2 | 1,476 | 268 | 3,762 | 98.41% |
| [`NP5`](#mnist-np5) | 5 | 3 | 1,492 | 284 | 3,138 | 98.56% |
| [`NP6`](#mnist-np6) | 6 | 3 | 1,636 | 284 | 5,458 | 98.77% |
| [`P4`](#mnist-p4) | 4 | 2 | 2,920 | 536 | 9,178 | 99.32% |
| [`P5`](#mnist-p5) | 5 | 3 | 2,952 | 568 | 10,234 | 99.30% |
| [`P6`](#mnist-p6) | 6 | 3 | 3,240 | 568 | 19,482 | 99.35% |

#### MNIST · `NS1` (no pooling)

<a id="mnist-ns1"></a>

`c4s2` — narrow, strided; 1 conv, 0 pools. Test accuracy **98.21%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Flatten |  | `4×14×14` | `784` | — | — | — |
| 4 | FC32 | 784 → 32 | `784` | `32` | 25,120 | — | — |
| 5 | ReLU |  | `32` | `32` | — | 32 | — |
| 6 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **25,518** | **816** | **0** |

#### MNIST · `NS2` (no pooling)

<a id="mnist-ns2"></a>

`c4s2 c8s2` — narrow, strided; 2 convs, 0 pools. Test accuracy **98.57%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv8 | 4×4, stride 2, pad 1 | `4×14×14` | `8×7×7` | 520 | — | — |
| 4 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 5 | Flatten |  | `8×7×7` | `392` | — | — | — |
| 6 | FC32 | 392 → 32 | `392` | `32` | 12,576 | — | — |
| 7 | ReLU |  | `32` | `32` | — | 32 | — |
| 8 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,494** | **1,208** | **0** |

#### MNIST · `S1` (no pooling)

<a id="mnist-s1"></a>

`c8s2` — wide, strided; 1 conv, 0 pools. Test accuracy **98.64%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Flatten |  | `8×14×14` | `1568` | — | — | — |
| 4 | FC32 | 1568 → 32 | `1568` | `32` | 50,208 | — | — |
| 5 | ReLU |  | `32` | `32` | — | 32 | — |
| 6 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **50,674** | **1,600** | **0** |

#### MNIST · `NS3` (no pooling)

<a id="mnist-ns3"></a>

`c4s2 c8s2 c8` — narrow, strided; 3 convs, 0 pools. Test accuracy **98.84%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv8 | 4×4, stride 2, pad 1 | `4×14×14` | `8×7×7` | 520 | — | — |
| 4 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 5 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 6 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 7 | Flatten |  | `8×7×7` | `392` | — | — | — |
| 8 | FC32 | 392 → 32 | `392` | `32` | 12,576 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **14,078** | **1,600** | **0** |

#### MNIST · `S2` (no pooling)

<a id="mnist-s2"></a>

`c8s2 c16s2` — wide, strided; 2 convs, 0 pools. Test accuracy **99.06%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv16 | 4×4, stride 2, pad 1 | `8×14×14` | `16×7×7` | 2,064 | — | — |
| 4 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 5 | Flatten |  | `16×7×7` | `784` | — | — | — |
| 6 | FC32 | 784 → 32 | `784` | `32` | 25,120 | — | — |
| 7 | ReLU |  | `32` | `32` | — | 32 | — |
| 8 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **27,650** | **2,384** | **0** |

#### MNIST · `NS4` (no pooling)

<a id="mnist-ns4"></a>

`c4s2 c4 c8s2 c8` — narrow, strided; 4 convs, 0 pools. Test accuracy **99.00%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×14×14` | `8×7×7` | 520 | — | — |
| 6 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 8 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 9 | Flatten |  | `8×7×7` | `392` | — | — | — |
| 10 | FC32 | 392 → 32 | `392` | `32` | 12,576 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **14,226** | **2,384** | **0** |

#### MNIST · `NS5` (no pooling)

<a id="mnist-ns5"></a>

`c4s2 c4 c8s2 c8 c16s2` — narrow, strided; 5 convs, 0 pools. Test accuracy **98.98%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×14×14` | `8×7×7` | 520 | — | — |
| 6 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 8 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 9 | Conv16 | 4×4, stride 2, pad 1 | `8×7×7` | `16×3×3` | 2,064 | — | — |
| 10 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 11 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 12 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **8,354** | **2,528** | **0** |

#### MNIST · `NS6` (no pooling)

<a id="mnist-ns6"></a>

`c4s2 c4 c8s2 c8 c16s2 c16` — narrow, strided; 6 convs, 0 pools. Test accuracy **99.15%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×14×14` | `8×7×7` | 520 | — | — |
| 6 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 8 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 9 | Conv16 | 4×4, stride 2, pad 1 | `8×7×7` | `16×3×3` | 2,064 | — | — |
| 10 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `16×3×3` | `16×3×3` | 2,320 | — | — |
| 12 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 13 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 14 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 15 | ReLU |  | `32` | `32` | — | 32 | — |
| 16 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **10,674** | **2,672** | **0** |

#### MNIST · `S3` (no pooling)

<a id="mnist-s3"></a>

`c8s2 c16s2 c16` — wide, strided; 3 convs, 0 pools. Test accuracy **99.36%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv16 | 4×4, stride 2, pad 1 | `8×14×14` | `16×7×7` | 2,064 | — | — |
| 4 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 5 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 6 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 7 | Flatten |  | `16×7×7` | `784` | — | — | — |
| 8 | FC32 | 784 → 32 | `784` | `32` | 25,120 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **29,970** | **3,168** | **0** |

#### MNIST · `S4` (no pooling)

<a id="mnist-s4"></a>

`c8s2 c8 c16s2 c16` — wide, strided; 4 convs, 0 pools. Test accuracy **99.35%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×14×14` | `16×7×7` | 2,064 | — | — |
| 6 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 8 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 9 | Flatten |  | `16×7×7` | `784` | — | — | — |
| 10 | FC32 | 784 → 32 | `784` | `32` | 25,120 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **30,554** | **4,736** | **0** |

#### MNIST · `S5` (no pooling)

<a id="mnist-s5"></a>

`c8s2 c8 c16s2 c16 c32s2` — wide, strided; 5 convs, 0 pools. Test accuracy **99.52%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×14×14` | `16×7×7` | 2,064 | — | — |
| 6 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 8 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 9 | Conv32 | 4×4, stride 2, pad 1 | `16×7×7` | `32×3×3` | 8,224 | — | — |
| 10 | ReLU |  | `32×3×3` | `32×3×3` | — | 288 | — |
| 11 | Flatten |  | `32×3×3` | `288` | — | — | — |
| 12 | FC32 | 288 → 32 | `288` | `32` | 9,248 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **22,906** | **5,024** | **0** |

#### MNIST · `S6` (no pooling)

<a id="mnist-s6"></a>

`c8s2 c8 c16s2 c16 c32s2 c32` — wide, strided; 6 convs, 0 pools. Test accuracy **99.51%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×14×14` | `16×7×7` | 2,064 | — | — |
| 6 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 8 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 9 | Conv32 | 4×4, stride 2, pad 1 | `16×7×7` | `32×3×3` | 8,224 | — | — |
| 10 | ReLU |  | `32×3×3` | `32×3×3` | — | 288 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `32×3×3` | `32×3×3` | 9,248 | — | — |
| 12 | ReLU |  | `32×3×3` | `32×3×3` | — | 288 | — |
| 13 | Flatten |  | `32×3×3` | `288` | — | — | — |
| 14 | FC32 | 288 → 32 | `288` | `32` | 9,248 | — | — |
| 15 | ReLU |  | `32` | `32` | — | 32 | — |
| 16 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **32,154** | **5,312** | **0** |

#### MNIST · `NA1` (average pooling)

<a id="mnist-na1"></a>

`c4s2 a` — narrow, average pool; 1 conv, 1 pool. Test accuracy **96.43%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 3 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 4 | Flatten |  | `4×7×7` | `196` | — | — | — |
| 5 | FC32 | 196 → 32 | `196` | `32` | 6,304 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **6,702** | **228** | **196** |

#### MNIST · `NA2` (average pooling)

<a id="mnist-na2"></a>

`c4s2 a c8 a` — narrow, average pool; 2 convs, 2 pools. Test accuracy **95.43%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 3 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 5 | AvgPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 6 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 7 | Flatten |  | `8×3×3` | `72` | — | — | — |
| 8 | FC32 | 72 → 32 | `72` | `32` | 2,336 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,030** | **300** | **268** |

#### MNIST · `A1` (average pooling)

<a id="mnist-a1"></a>

`c8s2 a` — wide, average pool; 1 conv, 1 pool. Test accuracy **97.22%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 3 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 4 | Flatten |  | `8×7×7` | `392` | — | — | — |
| 5 | FC32 | 392 → 32 | `392` | `32` | 12,576 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,042** | **424** | **392** |

#### MNIST · `A2` (average pooling)

<a id="mnist-a2"></a>

`c8s2 a c16 a` — wide, average pool; 2 convs, 2 pools. Test accuracy **97.53%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 3 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 5 | AvgPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 6 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 7 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 8 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **6,274** | **568** | **536** |

#### MNIST · `NA3` (average pooling)

<a id="mnist-na3"></a>

`c4s2 a c8 c8 a` — narrow, average pool; 3 convs, 2 pools. Test accuracy **97.78%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 3 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 7 | AvgPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 8 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 9 | Flatten |  | `8×3×3` | `72` | — | — | — |
| 10 | FC32 | 72 → 32 | `72` | `32` | 2,336 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,614** | **692** | **268** |

#### MNIST · `A3` (average pooling)

<a id="mnist-a3"></a>

`c8s2 a c16 c16 a` — wide, average pool; 3 convs, 2 pools. Test accuracy **98.74%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 3 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 5 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 7 | AvgPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 8 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 9 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 10 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **8,594** | **1,352** | **536** |

#### MNIST · `NA4` (average pooling)

<a id="mnist-na4"></a>

`c4s2 c4 a c8 c8 a` — narrow, average pool; 4 convs, 2 pools. Test accuracy **98.27%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 5 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 10 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 11 | Flatten |  | `8×3×3` | `72` | — | — | — |
| 12 | FC32 | 72 → 32 | `72` | `32` | 2,336 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,762** | **1,476** | **268** |

#### MNIST · `NA5` (average pooling)

<a id="mnist-na5"></a>

`c4s2 c4 a c8 c8 a c16 a` — narrow, average pool; 5 convs, 3 pools. Test accuracy **97.99%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 5 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 10 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×3×3` | `16×3×3` | 1,168 | — | — |
| 12 | AvgPool | 2×2, stride 2 | `16×3×3` | `16×1×1` | — | — | 16 |
| 13 | ReLU |  | `16×1×1` | `16×1×1` | — | 16 | — |
| 14 | Flatten |  | `16×1×1` | `16` | — | — | — |
| 15 | FC32 | 16 → 32 | `16` | `32` | 544 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,138** | **1,492** | **284** |

#### MNIST · `NA6` (average pooling)

<a id="mnist-na6"></a>

`c4s2 c4 a c8 c8 a c16 c16 a` — narrow, average pool; 6 convs, 3 pools. Test accuracy **98.53%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 5 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 10 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×3×3` | `16×3×3` | 1,168 | — | — |
| 12 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×3×3` | `16×3×3` | 2,320 | — | — |
| 14 | AvgPool | 2×2, stride 2 | `16×3×3` | `16×1×1` | — | — | 16 |
| 15 | ReLU |  | `16×1×1` | `16×1×1` | — | 16 | — |
| 16 | Flatten |  | `16×1×1` | `16` | — | — | — |
| 17 | FC32 | 16 → 32 | `16` | `32` | 544 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **5,458** | **1,636** | **284** |

#### MNIST · `A4` (average pooling)

<a id="mnist-a4"></a>

`c8s2 c8 a c16 c16 a` — wide, average pool; 4 convs, 2 pools. Test accuracy **99.10%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 10 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 11 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 12 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **9,178** | **2,920** | **536** |

#### MNIST · `A5` (average pooling)

<a id="mnist-a5"></a>

`c8s2 c8 a c16 c16 a c32 a` — wide, average pool; 5 convs, 3 pools. Test accuracy **99.17%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 10 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×3×3` | `32×3×3` | 4,640 | — | — |
| 12 | AvgPool | 2×2, stride 2 | `32×3×3` | `32×1×1` | — | — | 32 |
| 13 | ReLU |  | `32×1×1` | `32×1×1` | — | 32 | — |
| 14 | Flatten |  | `32×1×1` | `32` | — | — | — |
| 15 | FC32 | 32 → 32 | `32` | `32` | 1,056 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **10,234** | **2,952** | **568** |

#### MNIST · `A6` (average pooling)

<a id="mnist-a6"></a>

`c8s2 c8 a c16 c16 a c32 c32 a` — wide, average pool; 6 convs, 3 pools. Test accuracy **99.21%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 10 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×3×3` | `32×3×3` | 4,640 | — | — |
| 12 | ReLU |  | `32×3×3` | `32×3×3` | — | 288 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×3×3` | `32×3×3` | 9,248 | — | — |
| 14 | AvgPool | 2×2, stride 2 | `32×3×3` | `32×1×1` | — | — | 32 |
| 15 | ReLU |  | `32×1×1` | `32×1×1` | — | 32 | — |
| 16 | Flatten |  | `32×1×1` | `32` | — | — | — |
| 17 | FC32 | 32 → 32 | `32` | `32` | 1,056 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **19,482** | **3,240** | **568** |

#### MNIST · `NP1` (max pooling)

<a id="mnist-np1"></a>

`c4s2 p` — narrow, max pool; 1 conv, 1 pool. Test accuracy **96.84%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 3 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 4 | Flatten |  | `4×7×7` | `196` | — | — | — |
| 5 | FC32 | 196 → 32 | `196` | `32` | 6,304 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **6,702** | **228** | **196** |

#### MNIST · `NP2` (max pooling)

<a id="mnist-np2"></a>

`c4s2 p c8 p` — narrow, max pool; 2 convs, 2 pools. Test accuracy **97.36%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 3 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 5 | MaxPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 6 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 7 | Flatten |  | `8×3×3` | `72` | — | — | — |
| 8 | FC32 | 72 → 32 | `72` | `32` | 2,336 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,030** | **300** | **268** |

#### MNIST · `P1` (max pooling)

<a id="mnist-p1"></a>

`c8s2 p` — wide, max pool; 1 conv, 1 pool. Test accuracy **98.10%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 3 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 4 | Flatten |  | `8×7×7` | `392` | — | — | — |
| 5 | FC32 | 392 → 32 | `392` | `32` | 12,576 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,042** | **424** | **392** |

#### MNIST · `P2` (max pooling)

<a id="mnist-p2"></a>

`c8s2 p c16 p` — wide, max pool; 2 convs, 2 pools. Test accuracy **98.56%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 3 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 5 | MaxPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 6 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 7 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 8 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **6,274** | **568** | **536** |

#### MNIST · `NP3` (max pooling)

<a id="mnist-np3"></a>

`c4s2 p c8 c8 p` — narrow, max pool; 3 convs, 2 pools. Test accuracy **97.86%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 3 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 7 | MaxPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 8 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 9 | Flatten |  | `8×3×3` | `72` | — | — | — |
| 10 | FC32 | 72 → 32 | `72` | `32` | 2,336 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,614** | **692** | **268** |

#### MNIST · `P3` (max pooling)

<a id="mnist-p3"></a>

`c8s2 p c16 c16 p` — wide, max pool; 3 convs, 2 pools. Test accuracy **99.00%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 3 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 5 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 7 | MaxPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 8 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 9 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 10 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **8,594** | **1,352** | **536** |

#### MNIST · `NP4` (max pooling)

<a id="mnist-np4"></a>

`c4s2 c4 p c8 c8 p` — narrow, max pool; 4 convs, 2 pools. Test accuracy **98.41%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 5 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 10 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 11 | Flatten |  | `8×3×3` | `72` | — | — | — |
| 12 | FC32 | 72 → 32 | `72` | `32` | 2,336 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,762** | **1,476** | **268** |

#### MNIST · `NP5` (max pooling)

<a id="mnist-np5"></a>

`c4s2 c4 p c8 c8 p c16 p` — narrow, max pool; 5 convs, 3 pools. Test accuracy **98.56%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 5 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 10 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×3×3` | `16×3×3` | 1,168 | — | — |
| 12 | MaxPool | 2×2, stride 2 | `16×3×3` | `16×1×1` | — | — | 16 |
| 13 | ReLU |  | `16×1×1` | `16×1×1` | — | 16 | — |
| 14 | Flatten |  | `16×1×1` | `16` | — | — | — |
| 15 | FC32 | 16 → 32 | `16` | `32` | 544 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,138** | **1,492** | **284** |

#### MNIST · `NP6` (max pooling)

<a id="mnist-np6"></a>

`c4s2 c4 p c8 c8 p c16 c16 p` — narrow, max pool; 6 convs, 3 pools. Test accuracy **98.77%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 5 | ReLU |  | `4×7×7` | `4×7×7` | — | 196 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 10 | ReLU |  | `8×3×3` | `8×3×3` | — | 72 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×3×3` | `16×3×3` | 1,168 | — | — |
| 12 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×3×3` | `16×3×3` | 2,320 | — | — |
| 14 | MaxPool | 2×2, stride 2 | `16×3×3` | `16×1×1` | — | — | 16 |
| 15 | ReLU |  | `16×1×1` | `16×1×1` | — | 16 | — |
| 16 | Flatten |  | `16×1×1` | `16` | — | — | — |
| 17 | FC32 | 16 → 32 | `16` | `32` | 544 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **5,458** | **1,636** | **284** |

#### MNIST · `P4` (max pooling)

<a id="mnist-p4"></a>

`c8s2 c8 p c16 c16 p` — wide, max pool; 4 convs, 2 pools. Test accuracy **99.32%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 10 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 11 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 12 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **9,178** | **2,920** | **536** |

#### MNIST · `P5` (max pooling)

<a id="mnist-p5"></a>

`c8s2 c8 p c16 c16 p c32 p` — wide, max pool; 5 convs, 3 pools. Test accuracy **99.30%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 10 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×3×3` | `32×3×3` | 4,640 | — | — |
| 12 | MaxPool | 2×2, stride 2 | `32×3×3` | `32×1×1` | — | — | 32 |
| 13 | ReLU |  | `32×1×1` | `32×1×1` | — | 32 | — |
| 14 | Flatten |  | `32×1×1` | `32` | — | — | — |
| 15 | FC32 | 32 → 32 | `32` | `32` | 1,056 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **10,234** | **2,952** | **568** |

#### MNIST · `P6` (max pooling)

<a id="mnist-p6"></a>

`c8s2 c8 p c16 c16 p c32 c32 p` — wide, max pool; 6 convs, 3 pools. Test accuracy **99.35%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 10 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×3×3` | `32×3×3` | 4,640 | — | — |
| 12 | ReLU |  | `32×3×3` | `32×3×3` | — | 288 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×3×3` | `32×3×3` | 9,248 | — | — |
| 14 | MaxPool | 2×2, stride 2 | `32×3×3` | `32×1×1` | — | — | 32 |
| 15 | ReLU |  | `32×1×1` | `32×1×1` | — | 32 | — |
| 16 | Flatten |  | `32×1×1` | `32` | — | — | — |
| 17 | FC32 | 32 → 32 | `32` | `32` | 1,056 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **19,482** | **3,240** | **568** |

---

## GTSRB

German traffic signs, resized to 32×32. The only 43-class family.

- **Input:** `3×32×32`, pixel values in `[0,1]`
- **Classes:** 43
- **Files:** `data/models/gtsrb/cnn-bench/<id>.onnx` and `<id>.pth`

### No pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NS1`](#gtsrb-ns1) | 1 | 0 | 1,056 | 0 | 34,415 | 84.68% |
| [`NS2`](#gtsrb-ns2) | 2 | 0 | 1,568 | 0 | 18,551 | 86.44% |
| [`S1`](#gtsrb-s1) | 1 | 0 | 2,080 | 0 | 67,379 | 86.20% |
| [`NS3`](#gtsrb-ns3) | 3 | 0 | 2,080 | 0 | 19,135 | 87.97% |
| [`S2`](#gtsrb-s2) | 2 | 0 | 3,104 | 0 | 36,675 | 90.78% |
| [`NS4`](#gtsrb-ns4) | 4 | 0 | 3,104 | 0 | 19,283 | 90.37% |
| [`NS5`](#gtsrb-ns5) | 5 | 0 | 3,360 | 0 | 13,155 | 90.04% |
| [`NS6`](#gtsrb-ns6) | 6 | 0 | 3,616 | 0 | 15,475 | 89.72% |
| [`S3`](#gtsrb-s3) | 3 | 0 | 4,128 | 0 | 38,995 | 92.61% |
| [`S4`](#gtsrb-s4) | 4 | 0 | 6,176 | 0 | 39,579 | 94.45% |
| [`S5`](#gtsrb-s5) | 5 | 0 | 6,688 | 0 | 31,419 | 94.64% |
| [`S6`](#gtsrb-s6) | 6 | 0 | 7,200 | 0 | 40,667 | 95.62% |

### Average pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NA1`](#gtsrb-na1) | 1 | 1 | 288 | 256 | 9,839 | 71.70% |
| [`NA2`](#gtsrb-na2) | 2 | 2 | 416 | 384 | 6,039 | 64.90% |
| [`A1`](#gtsrb-a1) | 1 | 1 | 544 | 512 | 18,227 | 77.65% |
| [`A2`](#gtsrb-a2) | 2 | 2 | 800 | 768 | 11,203 | 71.76% |
| [`NA3`](#gtsrb-na3) | 3 | 2 | 928 | 384 | 6,623 | 69.39% |
| [`A3`](#gtsrb-a3) | 3 | 2 | 1,824 | 768 | 13,523 | 83.40% |
| [`NA4`](#gtsrb-na4) | 4 | 2 | 1,952 | 384 | 6,771 | 78.67% |
| [`NA5`](#gtsrb-na5) | 5 | 3 | 2,016 | 448 | 5,891 | 78.42% |
| [`NA6`](#gtsrb-na6) | 6 | 3 | 2,272 | 448 | 8,211 | 78.05% |
| [`A4`](#gtsrb-a4) | 4 | 2 | 3,872 | 768 | 14,107 | 90.93% |
| [`A5`](#gtsrb-a5) | 5 | 3 | 4,000 | 896 | 14,651 | 89.93% |
| [`A6`](#gtsrb-a6) | 6 | 3 | 4,512 | 896 | 23,899 | 90.77% |

### Max pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NP1`](#gtsrb-np1) | 1 | 1 | 288 | 256 | 9,839 | 80.08% |
| [`NP2`](#gtsrb-np2) | 2 | 2 | 416 | 384 | 6,039 | 70.40% |
| [`P1`](#gtsrb-p1) | 1 | 1 | 544 | 512 | 18,227 | 85.52% |
| [`P2`](#gtsrb-p2) | 2 | 2 | 800 | 768 | 11,203 | 84.77% |
| [`NP3`](#gtsrb-np3) | 3 | 2 | 928 | 384 | 6,623 | 76.25% |
| [`P3`](#gtsrb-p3) | 3 | 2 | 1,824 | 768 | 13,523 | 88.96% |
| [`NP4`](#gtsrb-np4) | 4 | 2 | 1,952 | 384 | 6,771 | 81.10% |
| [`NP5`](#gtsrb-np5) | 5 | 3 | 2,016 | 448 | 5,891 | 81.37% |
| [`NP6`](#gtsrb-np6) | 6 | 3 | 2,272 | 448 | 8,211 | 81.39% |
| [`P4`](#gtsrb-p4) | 4 | 2 | 3,872 | 768 | 14,107 | 92.31% |
| [`P5`](#gtsrb-p5) | 5 | 3 | 4,000 | 896 | 14,651 | 92.30% |
| [`P6`](#gtsrb-p6) | 6 | 3 | 4,512 | 896 | 23,899 | 92.37% |

#### GTSRB · `NS1` (no pooling)

<a id="gtsrb-ns1"></a>

`c4s2` — narrow, strided; 1 conv, 0 pools. Test accuracy **84.68%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Flatten |  | `4×16×16` | `1024` | — | — | — |
| 4 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 5 | ReLU |  | `32` | `32` | — | 32 | — |
| 6 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **34,415** | **1,056** | **0** |

#### GTSRB · `NS2` (no pooling)

<a id="gtsrb-ns2"></a>

`c4s2 c8s2` — narrow, strided; 2 convs, 0 pools. Test accuracy **86.44%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 4 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 5 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 6 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 7 | ReLU |  | `32` | `32` | — | 32 | — |
| 8 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **18,551** | **1,568** | **0** |

#### GTSRB · `S1` (no pooling)

<a id="gtsrb-s1"></a>

`c8s2` — wide, strided; 1 conv, 0 pools. Test accuracy **86.20%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Flatten |  | `8×16×16` | `2048` | — | — | — |
| 4 | FC32 | 2048 → 32 | `2048` | `32` | 65,568 | — | — |
| 5 | ReLU |  | `32` | `32` | — | 32 | — |
| 6 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **67,379** | **2,080** | **0** |

#### GTSRB · `NS3` (no pooling)

<a id="gtsrb-ns3"></a>

`c4s2 c8s2 c8` — narrow, strided; 3 convs, 0 pools. Test accuracy **87.97%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 4 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 5 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 8 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **19,135** | **2,080** | **0** |

#### GTSRB · `S2` (no pooling)

<a id="gtsrb-s2"></a>

`c8s2 c16s2` — wide, strided; 2 convs, 0 pools. Test accuracy **90.78%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 4 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 5 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 6 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 7 | ReLU |  | `32` | `32` | — | 32 | — |
| 8 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **36,675** | **3,104** | **0** |

#### GTSRB · `NS4` (no pooling)

<a id="gtsrb-ns4"></a>

`c4s2 c4 c8s2 c8` — narrow, strided; 4 convs, 0 pools. Test accuracy **90.37%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 8 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 9 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 10 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **19,283** | **3,104** | **0** |

#### GTSRB · `NS5` (no pooling)

<a id="gtsrb-ns5"></a>

`c4s2 c4 c8s2 c8 c16s2` — narrow, strided; 5 convs, 0 pools. Test accuracy **90.04%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 8 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 9 | Conv16 | 4×4, stride 2, pad 1 | `8×8×8` | `16×4×4` | 2,064 | — | — |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 12 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **13,155** | **3,360** | **0** |

#### GTSRB · `NS6` (no pooling)

<a id="gtsrb-ns6"></a>

`c4s2 c4 c8s2 c8 c16s2 c16` — narrow, strided; 6 convs, 0 pools. Test accuracy **89.72%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 8 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 9 | Conv16 | 4×4, stride 2, pad 1 | `8×8×8` | `16×4×4` | 2,064 | — | — |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `16×4×4` | `16×4×4` | 2,320 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 14 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 15 | ReLU |  | `32` | `32` | — | 32 | — |
| 16 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **15,475** | **3,616** | **0** |

#### GTSRB · `S3` (no pooling)

<a id="gtsrb-s3"></a>

`c8s2 c16s2 c16` — wide, strided; 3 convs, 0 pools. Test accuracy **92.61%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 4 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 5 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 8 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **38,995** | **4,128** | **0** |

#### GTSRB · `S4` (no pooling)

<a id="gtsrb-s4"></a>

`c8s2 c8 c16s2 c16` — wide, strided; 4 convs, 0 pools. Test accuracy **94.45%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 8 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 9 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 10 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **39,579** | **6,176** | **0** |

#### GTSRB · `S5` (no pooling)

<a id="gtsrb-s5"></a>

`c8s2 c8 c16s2 c16 c32s2` — wide, strided; 5 convs, 0 pools. Test accuracy **94.64%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 8 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 9 | Conv32 | 4×4, stride 2, pad 1 | `16×8×8` | `32×4×4` | 8,224 | — | — |
| 10 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 11 | Flatten |  | `32×4×4` | `512` | — | — | — |
| 12 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **31,419** | **6,688** | **0** |

#### GTSRB · `S6` (no pooling)

<a id="gtsrb-s6"></a>

`c8s2 c8 c16s2 c16 c32s2 c32` — wide, strided; 6 convs, 0 pools. Test accuracy **95.62%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 8 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 9 | Conv32 | 4×4, stride 2, pad 1 | `16×8×8` | `32×4×4` | 8,224 | — | — |
| 10 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `32×4×4` | `32×4×4` | 9,248 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | Flatten |  | `32×4×4` | `512` | — | — | — |
| 14 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 15 | ReLU |  | `32` | `32` | — | 32 | — |
| 16 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **40,667** | **7,200** | **0** |

#### GTSRB · `NA1` (average pooling)

<a id="gtsrb-na1"></a>

`c4s2 a` — narrow, average pool; 1 conv, 1 pool. Test accuracy **71.70%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Flatten |  | `4×8×8` | `256` | — | — | — |
| 5 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **9,839** | **288** | **256** |

#### GTSRB · `NA2` (average pooling)

<a id="gtsrb-na2"></a>

`c4s2 a c8 a` — narrow, average pool; 2 convs, 2 pools. Test accuracy **64.90%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 6 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 7 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 8 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **6,039** | **416** | **384** |

#### GTSRB · `A1` (average pooling)

<a id="gtsrb-a1"></a>

`c8s2 a` — wide, average pool; 1 conv, 1 pool. Test accuracy **77.65%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 5 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **18,227** | **544** | **512** |

#### GTSRB · `A2` (average pooling)

<a id="gtsrb-a2"></a>

`c8s2 a c16 a` — wide, average pool; 2 convs, 2 pools. Test accuracy **71.76%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 6 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 7 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 8 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **11,203** | **800** | **768** |

#### GTSRB · `NA3` (average pooling)

<a id="gtsrb-na3"></a>

`c4s2 a c8 c8 a` — narrow, average pool; 3 convs, 2 pools. Test accuracy **69.39%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 7 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 8 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 9 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 10 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **6,623** | **928** | **384** |

#### GTSRB · `A3` (average pooling)

<a id="gtsrb-a3"></a>

`c8s2 a c16 c16 a` — wide, average pool; 3 convs, 2 pools. Test accuracy **83.40%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 7 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 8 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 9 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 10 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **13,523** | **1,824** | **768** |

#### GTSRB · `NA4` (average pooling)

<a id="gtsrb-na4"></a>

`c4s2 c4 a c8 c8 a` — narrow, average pool; 4 convs, 2 pools. Test accuracy **78.67%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 12 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **6,771** | **1,952** | **384** |

#### GTSRB · `NA5` (average pooling)

<a id="gtsrb-na5"></a>

`c4s2 c4 a c8 c8 a c16 a` — narrow, average pool; 5 convs, 3 pools. Test accuracy **78.42%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | AvgPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 13 | ReLU |  | `16×2×2` | `16×2×2` | — | 64 | — |
| 14 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 15 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **5,891** | **2,016** | **448** |

#### GTSRB · `NA6` (average pooling)

<a id="gtsrb-na6"></a>

`c4s2 c4 a c8 c8 a c16 c16 a` — narrow, average pool; 6 convs, 3 pools. Test accuracy **78.05%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×4×4` | `16×4×4` | 2,320 | — | — |
| 14 | AvgPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 15 | ReLU |  | `16×2×2` | `16×2×2` | — | 64 | — |
| 16 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 17 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **8,211** | **2,272** | **448** |

#### GTSRB · `A4` (average pooling)

<a id="gtsrb-a4"></a>

`c8s2 c8 a c16 c16 a` — wide, average pool; 4 convs, 2 pools. Test accuracy **90.93%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 12 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **14,107** | **3,872** | **768** |

#### GTSRB · `A5` (average pooling)

<a id="gtsrb-a5"></a>

`c8s2 c8 a c16 c16 a c32 a` — wide, average pool; 5 convs, 3 pools. Test accuracy **89.93%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | AvgPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 13 | ReLU |  | `32×2×2` | `32×2×2` | — | 128 | — |
| 14 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 15 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **14,651** | **4,000** | **896** |

#### GTSRB · `A6` (average pooling)

<a id="gtsrb-a6"></a>

`c8s2 c8 a c16 c16 a c32 c32 a` — wide, average pool; 6 convs, 3 pools. Test accuracy **90.77%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×4×4` | `32×4×4` | 9,248 | — | — |
| 14 | AvgPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 15 | ReLU |  | `32×2×2` | `32×2×2` | — | 128 | — |
| 16 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 17 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **23,899** | **4,512** | **896** |

#### GTSRB · `NP1` (max pooling)

<a id="gtsrb-np1"></a>

`c4s2 p` — narrow, max pool; 1 conv, 1 pool. Test accuracy **80.08%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Flatten |  | `4×8×8` | `256` | — | — | — |
| 5 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **9,839** | **288** | **256** |

#### GTSRB · `NP2` (max pooling)

<a id="gtsrb-np2"></a>

`c4s2 p c8 p` — narrow, max pool; 2 convs, 2 pools. Test accuracy **70.40%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 6 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 7 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 8 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **6,039** | **416** | **384** |

#### GTSRB · `P1` (max pooling)

<a id="gtsrb-p1"></a>

`c8s2 p` — wide, max pool; 1 conv, 1 pool. Test accuracy **85.52%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 5 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **18,227** | **544** | **512** |

#### GTSRB · `P2` (max pooling)

<a id="gtsrb-p2"></a>

`c8s2 p c16 p` — wide, max pool; 2 convs, 2 pools. Test accuracy **84.77%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 6 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 7 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 8 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **11,203** | **800** | **768** |

#### GTSRB · `NP3` (max pooling)

<a id="gtsrb-np3"></a>

`c4s2 p c8 c8 p` — narrow, max pool; 3 convs, 2 pools. Test accuracy **76.25%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 7 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 8 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 9 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 10 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **6,623** | **928** | **384** |

#### GTSRB · `P3` (max pooling)

<a id="gtsrb-p3"></a>

`c8s2 p c16 c16 p` — wide, max pool; 3 convs, 2 pools. Test accuracy **88.96%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 7 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 8 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 9 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 10 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **13,523** | **1,824** | **768** |

#### GTSRB · `NP4` (max pooling)

<a id="gtsrb-np4"></a>

`c4s2 c4 p c8 c8 p` — narrow, max pool; 4 convs, 2 pools. Test accuracy **81.10%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 12 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **6,771** | **1,952** | **384** |

#### GTSRB · `NP5` (max pooling)

<a id="gtsrb-np5"></a>

`c4s2 c4 p c8 c8 p c16 p` — narrow, max pool; 5 convs, 3 pools. Test accuracy **81.37%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | MaxPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 13 | ReLU |  | `16×2×2` | `16×2×2` | — | 64 | — |
| 14 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 15 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **5,891** | **2,016** | **448** |

#### GTSRB · `NP6` (max pooling)

<a id="gtsrb-np6"></a>

`c4s2 c4 p c8 c8 p c16 c16 p` — narrow, max pool; 6 convs, 3 pools. Test accuracy **81.39%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×4×4` | `16×4×4` | 2,320 | — | — |
| 14 | MaxPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 15 | ReLU |  | `16×2×2` | `16×2×2` | — | 64 | — |
| 16 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 17 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **8,211** | **2,272** | **448** |

#### GTSRB · `P4` (max pooling)

<a id="gtsrb-p4"></a>

`c8s2 c8 p c16 c16 p` — wide, max pool; 4 convs, 2 pools. Test accuracy **92.31%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 12 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **14,107** | **3,872** | **768** |

#### GTSRB · `P5` (max pooling)

<a id="gtsrb-p5"></a>

`c8s2 c8 p c16 c16 p c32 p` — wide, max pool; 5 convs, 3 pools. Test accuracy **92.30%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | MaxPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 13 | ReLU |  | `32×2×2` | `32×2×2` | — | 128 | — |
| 14 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 15 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **14,651** | **4,000** | **896** |

#### GTSRB · `P6` (max pooling)

<a id="gtsrb-p6"></a>

`c8s2 c8 p c16 c16 p c32 c32 p` — wide, max pool; 6 convs, 3 pools. Test accuracy **92.37%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×4×4` | `32×4×4` | 9,248 | — | — |
| 14 | MaxPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 15 | ReLU |  | `32×2×2` | `32×2×2` | — | 128 | — |
| 16 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 17 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **23,899** | **4,512** | **896** |

---

## CIFAR-10

Natural images across 10 object classes.

- **Input:** `3×32×32`, pixel values in `[0,1]`
- **Classes:** 10
- **Files:** `data/models/cifar10/cnn-bench/<id>.onnx` and `<id>.pth`

### No pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NS1`](#cifar10-ns1) | 1 | 0 | 1,056 | 0 | 33,326 | 55.11% |
| [`NS2`](#cifar10-ns2) | 2 | 0 | 1,568 | 0 | 17,462 | 58.63% |
| [`S1`](#cifar10-s1) | 1 | 0 | 2,080 | 0 | 66,290 | 61.36% |
| [`NS3`](#cifar10-ns3) | 3 | 0 | 2,080 | 0 | 18,046 | 61.88% |
| [`S2`](#cifar10-s2) | 2 | 0 | 3,104 | 0 | 35,586 | 67.78% |
| [`NS4`](#cifar10-ns4) | 4 | 0 | 3,104 | 0 | 18,194 | 62.54% |
| [`NS5`](#cifar10-ns5) | 5 | 0 | 3,360 | 0 | 12,066 | 65.47% |
| [`NS6`](#cifar10-ns6) | 6 | 0 | 3,616 | 0 | 14,386 | 66.27% |
| [`S3`](#cifar10-s3) | 3 | 0 | 4,128 | 0 | 37,906 | 70.54% |
| [`S4`](#cifar10-s4) | 4 | 0 | 6,176 | 0 | 38,490 | 73.39% |
| [`S5`](#cifar10-s5) | 5 | 0 | 6,688 | 0 | 30,330 | 74.79% |
| [`S6`](#cifar10-s6) | 6 | 0 | 7,200 | 0 | 39,578 | 76.30% |

### Average pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NA1`](#cifar10-na1) | 1 | 1 | 288 | 256 | 8,750 | 50.14% |
| [`NA2`](#cifar10-na2) | 2 | 2 | 416 | 384 | 4,950 | 51.34% |
| [`A1`](#cifar10-a1) | 1 | 1 | 544 | 512 | 17,138 | 54.44% |
| [`A2`](#cifar10-a2) | 2 | 2 | 800 | 768 | 10,114 | 56.45% |
| [`NA3`](#cifar10-na3) | 3 | 2 | 928 | 384 | 5,534 | 55.16% |
| [`A3`](#cifar10-a3) | 3 | 2 | 1,824 | 768 | 12,434 | 62.34% |
| [`NA4`](#cifar10-na4) | 4 | 2 | 1,952 | 384 | 5,682 | 57.78% |
| [`NA5`](#cifar10-na5) | 5 | 3 | 2,016 | 448 | 4,802 | 57.74% |
| [`NA6`](#cifar10-na6) | 6 | 3 | 2,272 | 448 | 7,122 | 60.87% |
| [`A4`](#cifar10-a4) | 4 | 2 | 3,872 | 768 | 13,018 | 70.02% |
| [`A5`](#cifar10-a5) | 5 | 3 | 4,000 | 896 | 13,562 | 69.72% |
| [`A6`](#cifar10-a6) | 6 | 3 | 4,512 | 896 | 22,810 | 73.41% |

### Max pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NP1`](#cifar10-np1) | 1 | 1 | 288 | 256 | 8,750 | 54.19% |
| [`NP2`](#cifar10-np2) | 2 | 2 | 416 | 384 | 4,950 | 55.50% |
| [`P1`](#cifar10-p1) | 1 | 1 | 544 | 512 | 17,138 | 61.29% |
| [`P2`](#cifar10-p2) | 2 | 2 | 800 | 768 | 10,114 | 65.55% |
| [`NP3`](#cifar10-np3) | 3 | 2 | 928 | 384 | 5,534 | 57.62% |
| [`P3`](#cifar10-p3) | 3 | 2 | 1,824 | 768 | 12,434 | 69.02% |
| [`NP4`](#cifar10-np4) | 4 | 2 | 1,952 | 384 | 5,682 | 58.90% |
| [`NP5`](#cifar10-np5) | 5 | 3 | 2,016 | 448 | 4,802 | 59.17% |
| [`NP6`](#cifar10-np6) | 6 | 3 | 2,272 | 448 | 7,122 | 61.04% |
| [`P4`](#cifar10-p4) | 4 | 2 | 3,872 | 768 | 13,018 | 71.07% |
| [`P5`](#cifar10-p5) | 5 | 3 | 4,000 | 896 | 13,562 | 71.77% |
| [`P6`](#cifar10-p6) | 6 | 3 | 4,512 | 896 | 22,810 | 74.26% |

#### CIFAR-10 · `NS1` (no pooling)

<a id="cifar10-ns1"></a>

`c4s2` — narrow, strided; 1 conv, 0 pools. Test accuracy **55.11%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Flatten |  | `4×16×16` | `1024` | — | — | — |
| 4 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 5 | ReLU |  | `32` | `32` | — | 32 | — |
| 6 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **33,326** | **1,056** | **0** |

#### CIFAR-10 · `NS2` (no pooling)

<a id="cifar10-ns2"></a>

`c4s2 c8s2` — narrow, strided; 2 convs, 0 pools. Test accuracy **58.63%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 4 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 5 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 6 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 7 | ReLU |  | `32` | `32` | — | 32 | — |
| 8 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,462** | **1,568** | **0** |

#### CIFAR-10 · `S1` (no pooling)

<a id="cifar10-s1"></a>

`c8s2` — wide, strided; 1 conv, 0 pools. Test accuracy **61.36%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Flatten |  | `8×16×16` | `2048` | — | — | — |
| 4 | FC32 | 2048 → 32 | `2048` | `32` | 65,568 | — | — |
| 5 | ReLU |  | `32` | `32` | — | 32 | — |
| 6 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **66,290** | **2,080** | **0** |

#### CIFAR-10 · `NS3` (no pooling)

<a id="cifar10-ns3"></a>

`c4s2 c8s2 c8` — narrow, strided; 3 convs, 0 pools. Test accuracy **61.88%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 4 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 5 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 8 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **18,046** | **2,080** | **0** |

#### CIFAR-10 · `S2` (no pooling)

<a id="cifar10-s2"></a>

`c8s2 c16s2` — wide, strided; 2 convs, 0 pools. Test accuracy **67.78%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 4 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 5 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 6 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 7 | ReLU |  | `32` | `32` | — | 32 | — |
| 8 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **35,586** | **3,104** | **0** |

#### CIFAR-10 · `NS4` (no pooling)

<a id="cifar10-ns4"></a>

`c4s2 c4 c8s2 c8` — narrow, strided; 4 convs, 0 pools. Test accuracy **62.54%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 8 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 9 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 10 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **18,194** | **3,104** | **0** |

#### CIFAR-10 · `NS5` (no pooling)

<a id="cifar10-ns5"></a>

`c4s2 c4 c8s2 c8 c16s2` — narrow, strided; 5 convs, 0 pools. Test accuracy **65.47%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 8 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 9 | Conv16 | 4×4, stride 2, pad 1 | `8×8×8` | `16×4×4` | 2,064 | — | — |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 12 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **12,066** | **3,360** | **0** |

#### CIFAR-10 · `NS6` (no pooling)

<a id="cifar10-ns6"></a>

`c4s2 c4 c8s2 c8 c16s2 c16` — narrow, strided; 6 convs, 0 pools. Test accuracy **66.27%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×16×16` | `8×8×8` | 520 | — | — |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 8 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 9 | Conv16 | 4×4, stride 2, pad 1 | `8×8×8` | `16×4×4` | 2,064 | — | — |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `16×4×4` | `16×4×4` | 2,320 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 14 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 15 | ReLU |  | `32` | `32` | — | 32 | — |
| 16 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **14,386** | **3,616** | **0** |

#### CIFAR-10 · `S3` (no pooling)

<a id="cifar10-s3"></a>

`c8s2 c16s2 c16` — wide, strided; 3 convs, 0 pools. Test accuracy **70.54%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 4 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 5 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 8 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **37,906** | **4,128** | **0** |

#### CIFAR-10 · `S4` (no pooling)

<a id="cifar10-s4"></a>

`c8s2 c8 c16s2 c16` — wide, strided; 4 convs, 0 pools. Test accuracy **73.39%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 8 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 9 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 10 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **38,490** | **6,176** | **0** |

#### CIFAR-10 · `S5` (no pooling)

<a id="cifar10-s5"></a>

`c8s2 c8 c16s2 c16 c32s2` — wide, strided; 5 convs, 0 pools. Test accuracy **74.79%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 8 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 9 | Conv32 | 4×4, stride 2, pad 1 | `16×8×8` | `32×4×4` | 8,224 | — | — |
| 10 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 11 | Flatten |  | `32×4×4` | `512` | — | — | — |
| 12 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **30,330** | **6,688** | **0** |

#### CIFAR-10 · `S6` (no pooling)

<a id="cifar10-s6"></a>

`c8s2 c8 c16s2 c16 c32s2 c32` — wide, strided; 6 convs, 0 pools. Test accuracy **76.30%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 8 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 9 | Conv32 | 4×4, stride 2, pad 1 | `16×8×8` | `32×4×4` | 8,224 | — | — |
| 10 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `32×4×4` | `32×4×4` | 9,248 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | Flatten |  | `32×4×4` | `512` | — | — | — |
| 14 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 15 | ReLU |  | `32` | `32` | — | 32 | — |
| 16 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **39,578** | **7,200** | **0** |

#### CIFAR-10 · `NA1` (average pooling)

<a id="cifar10-na1"></a>

`c4s2 a` — narrow, average pool; 1 conv, 1 pool. Test accuracy **50.14%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Flatten |  | `4×8×8` | `256` | — | — | — |
| 5 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **8,750** | **288** | **256** |

#### CIFAR-10 · `NA2` (average pooling)

<a id="cifar10-na2"></a>

`c4s2 a c8 a` — narrow, average pool; 2 convs, 2 pools. Test accuracy **51.34%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 6 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 7 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 8 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **4,950** | **416** | **384** |

#### CIFAR-10 · `A1` (average pooling)

<a id="cifar10-a1"></a>

`c8s2 a` — wide, average pool; 1 conv, 1 pool. Test accuracy **54.44%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 5 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,138** | **544** | **512** |

#### CIFAR-10 · `A2` (average pooling)

<a id="cifar10-a2"></a>

`c8s2 a c16 a` — wide, average pool; 2 convs, 2 pools. Test accuracy **56.45%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 6 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 7 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 8 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **10,114** | **800** | **768** |

#### CIFAR-10 · `NA3` (average pooling)

<a id="cifar10-na3"></a>

`c4s2 a c8 c8 a` — narrow, average pool; 3 convs, 2 pools. Test accuracy **55.16%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 7 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 8 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 9 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 10 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **5,534** | **928** | **384** |

#### CIFAR-10 · `A3` (average pooling)

<a id="cifar10-a3"></a>

`c8s2 a c16 c16 a` — wide, average pool; 3 convs, 2 pools. Test accuracy **62.34%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 7 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 8 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 9 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 10 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **12,434** | **1,824** | **768** |

#### CIFAR-10 · `NA4` (average pooling)

<a id="cifar10-na4"></a>

`c4s2 c4 a c8 c8 a` — narrow, average pool; 4 convs, 2 pools. Test accuracy **57.78%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 12 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **5,682** | **1,952** | **384** |

#### CIFAR-10 · `NA5` (average pooling)

<a id="cifar10-na5"></a>

`c4s2 c4 a c8 c8 a c16 a` — narrow, average pool; 5 convs, 3 pools. Test accuracy **57.74%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | AvgPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 13 | ReLU |  | `16×2×2` | `16×2×2` | — | 64 | — |
| 14 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 15 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **4,802** | **2,016** | **448** |

#### CIFAR-10 · `NA6` (average pooling)

<a id="cifar10-na6"></a>

`c4s2 c4 a c8 c8 a c16 c16 a` — narrow, average pool; 6 convs, 3 pools. Test accuracy **60.87%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×4×4` | `16×4×4` | 2,320 | — | — |
| 14 | AvgPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 15 | ReLU |  | `16×2×2` | `16×2×2` | — | 64 | — |
| 16 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 17 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **7,122** | **2,272** | **448** |

#### CIFAR-10 · `A4` (average pooling)

<a id="cifar10-a4"></a>

`c8s2 c8 a c16 c16 a` — wide, average pool; 4 convs, 2 pools. Test accuracy **70.02%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 12 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,018** | **3,872** | **768** |

#### CIFAR-10 · `A5` (average pooling)

<a id="cifar10-a5"></a>

`c8s2 c8 a c16 c16 a c32 a` — wide, average pool; 5 convs, 3 pools. Test accuracy **69.72%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | AvgPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 13 | ReLU |  | `32×2×2` | `32×2×2` | — | 128 | — |
| 14 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 15 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,562** | **4,000** | **896** |

#### CIFAR-10 · `A6` (average pooling)

<a id="cifar10-a6"></a>

`c8s2 c8 a c16 c16 a c32 c32 a` — wide, average pool; 6 convs, 3 pools. Test accuracy **73.41%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×4×4` | `32×4×4` | 9,248 | — | — |
| 14 | AvgPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 15 | ReLU |  | `32×2×2` | `32×2×2` | — | 128 | — |
| 16 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 17 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **22,810** | **4,512** | **896** |

#### CIFAR-10 · `NP1` (max pooling)

<a id="cifar10-np1"></a>

`c4s2 p` — narrow, max pool; 1 conv, 1 pool. Test accuracy **54.19%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Flatten |  | `4×8×8` | `256` | — | — | — |
| 5 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **8,750** | **288** | **256** |

#### CIFAR-10 · `NP2` (max pooling)

<a id="cifar10-np2"></a>

`c4s2 p c8 p` — narrow, max pool; 2 convs, 2 pools. Test accuracy **55.50%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 6 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 7 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 8 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **4,950** | **416** | **384** |

#### CIFAR-10 · `P1` (max pooling)

<a id="cifar10-p1"></a>

`c8s2 p` — wide, max pool; 1 conv, 1 pool. Test accuracy **61.29%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 5 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,138** | **544** | **512** |

#### CIFAR-10 · `P2` (max pooling)

<a id="cifar10-p2"></a>

`c8s2 p c16 p` — wide, max pool; 2 convs, 2 pools. Test accuracy **65.55%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 6 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 7 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 8 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **10,114** | **800** | **768** |

#### CIFAR-10 · `NP3` (max pooling)

<a id="cifar10-np3"></a>

`c4s2 p c8 c8 p` — narrow, max pool; 3 convs, 2 pools. Test accuracy **57.62%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 3 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 7 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 8 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 9 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 10 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **5,534** | **928** | **384** |

#### CIFAR-10 · `P3` (max pooling)

<a id="cifar10-p3"></a>

`c8s2 p c16 c16 p` — wide, max pool; 3 convs, 2 pools. Test accuracy **69.02%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 3 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 7 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 8 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 9 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 10 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **12,434** | **1,824** | **768** |

#### CIFAR-10 · `NP4` (max pooling)

<a id="cifar10-np4"></a>

`c4s2 c4 p c8 c8 p` — narrow, max pool; 4 convs, 2 pools. Test accuracy **58.90%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 12 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **5,682** | **1,952** | **384** |

#### CIFAR-10 · `NP5` (max pooling)

<a id="cifar10-np5"></a>

`c4s2 c4 p c8 c8 p c16 p` — narrow, max pool; 5 convs, 3 pools. Test accuracy **59.17%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | MaxPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 13 | ReLU |  | `16×2×2` | `16×2×2` | — | 64 | — |
| 14 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 15 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **4,802** | **2,016** | **448** |

#### CIFAR-10 · `NP6` (max pooling)

<a id="cifar10-np6"></a>

`c4s2 c4 p c8 c8 p c16 c16 p` — narrow, max pool; 6 convs, 3 pools. Test accuracy **61.04%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 5 | ReLU |  | `4×8×8` | `4×8×8` | — | 256 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 10 | ReLU |  | `8×4×4` | `8×4×4` | — | 128 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×4×4` | `16×4×4` | 2,320 | — | — |
| 14 | MaxPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 15 | ReLU |  | `16×2×2` | `16×2×2` | — | 64 | — |
| 16 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 17 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **7,122** | **2,272** | **448** |

#### CIFAR-10 · `P4` (max pooling)

<a id="cifar10-p4"></a>

`c8s2 c8 p c16 c16 p` — wide, max pool; 4 convs, 2 pools. Test accuracy **71.07%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 12 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,018** | **3,872** | **768** |

#### CIFAR-10 · `P5` (max pooling)

<a id="cifar10-p5"></a>

`c8s2 c8 p c16 c16 p c32 p` — wide, max pool; 5 convs, 3 pools. Test accuracy **71.77%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | MaxPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 13 | ReLU |  | `32×2×2` | `32×2×2` | — | 128 | — |
| 14 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 15 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,562** | **4,000** | **896** |

#### CIFAR-10 · `P6` (max pooling)

<a id="cifar10-p6"></a>

`c8s2 c8 p c16 c16 p c32 c32 p` — wide, max pool; 6 convs, 3 pools. Test accuracy **74.26%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 10 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×4×4` | `32×4×4` | 9,248 | — | — |
| 14 | MaxPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 15 | ReLU |  | `32×2×2` | `32×2×2` | — | 128 | — |
| 16 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 17 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **22,810** | **4,512** | **896** |

---

## Imagenette-64

The 10-class ImageNet subset from fast.ai, resized so the shorter side is 72 then centre-cropped to 64. Four times the pixels of CIFAR/GTSRB.

- **Input:** `3×64×64`, pixel values in `[0,1]`
- **Classes:** 10
- **Files:** `data/models/imagenette/cnn-bench/<id>.onnx` and `<id>.pth`

### No pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NS1`](#imagenette-ns1) | 1 | 0 | 4,128 | 0 | 131,630 | 55.59% |
| [`NS2`](#imagenette-ns2) | 2 | 0 | 6,176 | 0 | 66,614 | 64.03% |
| [`S1`](#imagenette-s1) | 1 | 0 | 8,224 | 0 | 262,898 | 62.39% |
| [`NS3`](#imagenette-ns3) | 3 | 0 | 8,224 | 0 | 67,198 | 65.66% |
| [`S2`](#imagenette-s2) | 2 | 0 | 12,320 | 0 | 133,890 | 67.62% |
| [`NS4`](#imagenette-ns4) | 4 | 0 | 12,320 | 0 | 67,346 | 67.31% |
| [`NS5`](#imagenette-ns5) | 5 | 0 | 13,344 | 0 | 36,642 | 68.61% |
| [`NS6`](#imagenette-ns6) | 6 | 0 | 14,368 | 0 | 38,962 | 71.26% |
| [`S3`](#imagenette-s3) | 3 | 0 | 16,416 | 0 | 136,210 | 71.95% |
| [`S4`](#imagenette-s4) | 4 | 0 | 24,608 | 0 | 136,794 | 72.87% |
| [`S5`](#imagenette-s5) | 5 | 0 | 26,656 | 0 | 79,482 | 74.55% |
| [`S6`](#imagenette-s6) | 6 | 0 | 28,704 | 0 | 88,730 | 75.90% |

### Average pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NA1`](#imagenette-na1) | 1 | 1 | 1,056 | 1,024 | 33,326 | 50.01% |
| [`NA2`](#imagenette-na2) | 2 | 2 | 1,568 | 1,536 | 17,238 | 55.97% |
| [`A1`](#imagenette-a1) | 1 | 1 | 2,080 | 2,048 | 66,290 | 57.86% |
| [`A2`](#imagenette-a2) | 2 | 2 | 3,104 | 3,072 | 34,690 | 63.75% |
| [`NA3`](#imagenette-na3) | 3 | 2 | 3,616 | 1,536 | 17,822 | 63.29% |
| [`A3`](#imagenette-a3) | 3 | 2 | 7,200 | 3,072 | 37,010 | 67.31% |
| [`NA4`](#imagenette-na4) | 4 | 2 | 7,712 | 1,536 | 17,970 | 62.80% |
| [`NA5`](#imagenette-na5) | 5 | 3 | 7,968 | 1,792 | 10,946 | 65.20% |
| [`NA6`](#imagenette-na6) | 6 | 3 | 8,992 | 1,792 | 13,266 | 66.90% |
| [`A4`](#imagenette-a4) | 4 | 2 | 15,392 | 3,072 | 37,594 | 70.22% |
| [`A5`](#imagenette-a5) | 5 | 3 | 15,904 | 3,584 | 25,850 | 73.35% |
| [`A6`](#imagenette-a6) | 6 | 3 | 17,952 | 3,584 | 35,098 | 74.06% |

### Max pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NP1`](#imagenette-np1) | 1 | 1 | 1,056 | 1,024 | 33,326 | 57.73% |
| [`NP2`](#imagenette-np2) | 2 | 2 | 1,568 | 1,536 | 17,238 | 64.10% |
| [`P1`](#imagenette-p1) | 1 | 1 | 2,080 | 2,048 | 66,290 | 64.89% |
| [`P2`](#imagenette-p2) | 2 | 2 | 3,104 | 3,072 | 34,690 | 70.19% |
| [`NP3`](#imagenette-np3) | 3 | 2 | 3,616 | 1,536 | 17,822 | 65.76% |
| [`P3`](#imagenette-p3) | 3 | 2 | 7,200 | 3,072 | 37,010 | 71.39% |
| [`NP4`](#imagenette-np4) | 4 | 2 | 7,712 | 1,536 | 17,970 | 66.93% |
| [`NP5`](#imagenette-np5) | 5 | 3 | 7,968 | 1,792 | 10,946 | 67.87% |
| [`NP6`](#imagenette-np6) | 6 | 3 | 8,992 | 1,792 | 13,266 | 68.71% |
| [`P4`](#imagenette-p4) | 4 | 2 | 15,392 | 3,072 | 37,594 | 73.45% |
| [`P5`](#imagenette-p5) | 5 | 3 | 15,904 | 3,584 | 25,850 | 74.50% |
| [`P6`](#imagenette-p6) | 6 | 3 | 17,952 | 3,584 | 35,098 | 76.25% |

#### Imagenette-64 · `NS1` (no pooling)

<a id="imagenette-ns1"></a>

`c4s2` — narrow, strided; 1 conv, 0 pools. Test accuracy **55.59%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Flatten |  | `4×32×32` | `4096` | — | — | — |
| 4 | FC32 | 4096 → 32 | `4096` | `32` | 131,104 | — | — |
| 5 | ReLU |  | `32` | `32` | — | 32 | — |
| 6 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **131,630** | **4,128** | **0** |

#### Imagenette-64 · `NS2` (no pooling)

<a id="imagenette-ns2"></a>

`c4s2 c8s2` — narrow, strided; 2 convs, 0 pools. Test accuracy **64.03%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv8 | 4×4, stride 2, pad 1 | `4×32×32` | `8×16×16` | 520 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | Flatten |  | `8×16×16` | `2048` | — | — | — |
| 6 | FC32 | 2048 → 32 | `2048` | `32` | 65,568 | — | — |
| 7 | ReLU |  | `32` | `32` | — | 32 | — |
| 8 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **66,614** | **6,176** | **0** |

#### Imagenette-64 · `S1` (no pooling)

<a id="imagenette-s1"></a>

`c8s2` — wide, strided; 1 conv, 0 pools. Test accuracy **62.39%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Flatten |  | `8×32×32` | `8192` | — | — | — |
| 4 | FC32 | 8192 → 32 | `8192` | `32` | 262,176 | — | — |
| 5 | ReLU |  | `32` | `32` | — | 32 | — |
| 6 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **262,898** | **8,224** | **0** |

#### Imagenette-64 · `NS3` (no pooling)

<a id="imagenette-ns3"></a>

`c4s2 c8s2 c8` — narrow, strided; 3 convs, 0 pools. Test accuracy **65.66%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv8 | 4×4, stride 2, pad 1 | `4×32×32` | `8×16×16` | 520 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 6 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 7 | Flatten |  | `8×16×16` | `2048` | — | — | — |
| 8 | FC32 | 2048 → 32 | `2048` | `32` | 65,568 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **67,198** | **8,224** | **0** |

#### Imagenette-64 · `S2` (no pooling)

<a id="imagenette-s2"></a>

`c8s2 c16s2` — wide, strided; 2 convs, 0 pools. Test accuracy **67.62%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv16 | 4×4, stride 2, pad 1 | `8×32×32` | `16×16×16` | 2,064 | — | — |
| 4 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 5 | Flatten |  | `16×16×16` | `4096` | — | — | — |
| 6 | FC32 | 4096 → 32 | `4096` | `32` | 131,104 | — | — |
| 7 | ReLU |  | `32` | `32` | — | 32 | — |
| 8 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **133,890** | **12,320** | **0** |

#### Imagenette-64 · `NS4` (no pooling)

<a id="imagenette-ns4"></a>

`c4s2 c4 c8s2 c8` — narrow, strided; 4 convs, 0 pools. Test accuracy **67.31%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×32×32` | `4×32×32` | 148 | — | — |
| 4 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×32×32` | `8×16×16` | 520 | — | — |
| 6 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 8 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 9 | Flatten |  | `8×16×16` | `2048` | — | — | — |
| 10 | FC32 | 2048 → 32 | `2048` | `32` | 65,568 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **67,346** | **12,320** | **0** |

#### Imagenette-64 · `NS5` (no pooling)

<a id="imagenette-ns5"></a>

`c4s2 c4 c8s2 c8 c16s2` — narrow, strided; 5 convs, 0 pools. Test accuracy **68.61%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×32×32` | `4×32×32` | 148 | — | — |
| 4 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×32×32` | `8×16×16` | 520 | — | — |
| 6 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 8 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 9 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 10 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 11 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 12 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **36,642** | **13,344** | **0** |

#### Imagenette-64 · `NS6` (no pooling)

<a id="imagenette-ns6"></a>

`c4s2 c4 c8s2 c8 c16s2 c16` — narrow, strided; 6 convs, 0 pools. Test accuracy **71.26%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×32×32` | `4×32×32` | 148 | — | — |
| 4 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 5 | Conv8 | 4×4, stride 2, pad 1 | `4×32×32` | `8×16×16` | 520 | — | — |
| 6 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 7 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 8 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 9 | Conv16 | 4×4, stride 2, pad 1 | `8×16×16` | `16×8×8` | 2,064 | — | — |
| 10 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 12 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 13 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 14 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 15 | ReLU |  | `32` | `32` | — | 32 | — |
| 16 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **38,962** | **14,368** | **0** |

#### Imagenette-64 · `S3` (no pooling)

<a id="imagenette-s3"></a>

`c8s2 c16s2 c16` — wide, strided; 3 convs, 0 pools. Test accuracy **71.95%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv16 | 4×4, stride 2, pad 1 | `8×32×32` | `16×16×16` | 2,064 | — | — |
| 4 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 5 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 6 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 7 | Flatten |  | `16×16×16` | `4096` | — | — | — |
| 8 | FC32 | 4096 → 32 | `4096` | `32` | 131,104 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **136,210** | **16,416** | **0** |

#### Imagenette-64 · `S4` (no pooling)

<a id="imagenette-s4"></a>

`c8s2 c8 c16s2 c16` — wide, strided; 4 convs, 0 pools. Test accuracy **72.87%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×32×32` | `8×32×32` | 584 | — | — |
| 4 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×32×32` | `16×16×16` | 2,064 | — | — |
| 6 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 8 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 9 | Flatten |  | `16×16×16` | `4096` | — | — | — |
| 10 | FC32 | 4096 → 32 | `4096` | `32` | 131,104 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **136,794** | **24,608** | **0** |

#### Imagenette-64 · `S5` (no pooling)

<a id="imagenette-s5"></a>

`c8s2 c8 c16s2 c16 c32s2` — wide, strided; 5 convs, 0 pools. Test accuracy **74.55%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×32×32` | `8×32×32` | 584 | — | — |
| 4 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×32×32` | `16×16×16` | 2,064 | — | — |
| 6 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 8 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 9 | Conv32 | 4×4, stride 2, pad 1 | `16×16×16` | `32×8×8` | 8,224 | — | — |
| 10 | ReLU |  | `32×8×8` | `32×8×8` | — | 2,048 | — |
| 11 | Flatten |  | `32×8×8` | `2048` | — | — | — |
| 12 | FC32 | 2048 → 32 | `2048` | `32` | 65,568 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **79,482** | **26,656** | **0** |

#### Imagenette-64 · `S6` (no pooling)

<a id="imagenette-s6"></a>

`c8s2 c8 c16s2 c16 c32s2 c32` — wide, strided; 6 convs, 0 pools. Test accuracy **75.90%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×32×32` | `8×32×32` | 584 | — | — |
| 4 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 5 | Conv16 | 4×4, stride 2, pad 1 | `8×32×32` | `16×16×16` | 2,064 | — | — |
| 6 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 7 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 8 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 9 | Conv32 | 4×4, stride 2, pad 1 | `16×16×16` | `32×8×8` | 8,224 | — | — |
| 10 | ReLU |  | `32×8×8` | `32×8×8` | — | 2,048 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `32×8×8` | `32×8×8` | 9,248 | — | — |
| 12 | ReLU |  | `32×8×8` | `32×8×8` | — | 2,048 | — |
| 13 | Flatten |  | `32×8×8` | `2048` | — | — | — |
| 14 | FC32 | 2048 → 32 | `2048` | `32` | 65,568 | — | — |
| 15 | ReLU |  | `32` | `32` | — | 32 | — |
| 16 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **88,730** | **28,704** | **0** |

#### Imagenette-64 · `NA1` (average pooling)

<a id="imagenette-na1"></a>

`c4s2 a` — narrow, average pool; 1 conv, 1 pool. Test accuracy **50.01%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 3 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 4 | Flatten |  | `4×16×16` | `1024` | — | — | — |
| 5 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **33,326** | **1,056** | **1,024** |

#### Imagenette-64 · `NA2` (average pooling)

<a id="imagenette-na2"></a>

`c4s2 a c8 a` — narrow, average pool; 2 convs, 2 pools. Test accuracy **55.97%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 3 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 5 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 8 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,238** | **1,568** | **1,536** |

#### Imagenette-64 · `A1` (average pooling)

<a id="imagenette-a1"></a>

`c8s2 a` — wide, average pool; 1 conv, 1 pool. Test accuracy **57.86%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 3 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 4 | Flatten |  | `8×16×16` | `2048` | — | — | — |
| 5 | FC32 | 2048 → 32 | `2048` | `32` | 65,568 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **66,290** | **2,080** | **2,048** |

#### Imagenette-64 · `A2` (average pooling)

<a id="imagenette-a2"></a>

`c8s2 a c16 a` — wide, average pool; 2 convs, 2 pools. Test accuracy **63.75%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 3 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 5 | AvgPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 8 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **34,690** | **3,104** | **3,072** |

#### Imagenette-64 · `NA3` (average pooling)

<a id="imagenette-na3"></a>

`c4s2 a c8 c8 a` — narrow, average pool; 3 convs, 2 pools. Test accuracy **63.29%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 3 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 5 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 7 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 8 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 9 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 10 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,822** | **3,616** | **1,536** |

#### Imagenette-64 · `A3` (average pooling)

<a id="imagenette-a3"></a>

`c8s2 a c16 c16 a` — wide, average pool; 3 convs, 2 pools. Test accuracy **67.31%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | AvgPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 3 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 5 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 7 | AvgPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 8 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 9 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 10 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **37,010** | **7,200** | **3,072** |

#### Imagenette-64 · `NA4` (average pooling)

<a id="imagenette-na4"></a>

`c4s2 c4 a c8 c8 a` — narrow, average pool; 4 convs, 2 pools. Test accuracy **62.80%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×32×32` | `4×32×32` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 5 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 7 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 10 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 11 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 12 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,970** | **7,712** | **1,536** |

#### Imagenette-64 · `NA5` (average pooling)

<a id="imagenette-na5"></a>

`c4s2 c4 a c8 c8 a c16 a` — narrow, average pool; 5 convs, 3 pools. Test accuracy **65.20%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×32×32` | `4×32×32` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 5 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 7 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 10 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 12 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 13 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 14 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 15 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **10,946** | **7,968** | **1,792** |

#### Imagenette-64 · `NA6` (average pooling)

<a id="imagenette-na6"></a>

`c4s2 c4 a c8 c8 a c16 c16 a` — narrow, average pool; 6 convs, 3 pools. Test accuracy **66.90%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×32×32` | `4×32×32` | 148 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 5 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 7 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 10 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 12 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 14 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 15 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 16 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 17 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,266** | **8,992** | **1,792** |

#### Imagenette-64 · `A4` (average pooling)

<a id="imagenette-a4"></a>

`c8s2 c8 a c16 c16 a` — wide, average pool; 4 convs, 2 pools. Test accuracy **70.22%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×32×32` | `8×32×32` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 5 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 7 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 10 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 11 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 12 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **37,594** | **15,392** | **3,072** |

#### Imagenette-64 · `A5` (average pooling)

<a id="imagenette-a5"></a>

`c8s2 c8 a c16 c16 a c32 a` — wide, average pool; 5 convs, 3 pools. Test accuracy **73.35%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×32×32` | `8×32×32` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 5 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 7 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 10 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×8×8` | `32×8×8` | 4,640 | — | — |
| 12 | AvgPool | 2×2, stride 2 | `32×8×8` | `32×4×4` | — | — | 512 |
| 13 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 14 | Flatten |  | `32×4×4` | `512` | — | — | — |
| 15 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **25,850** | **15,904** | **3,584** |

#### Imagenette-64 · `A6` (average pooling)

<a id="imagenette-a6"></a>

`c8s2 c8 a c16 c16 a c32 c32 a` — wide, average pool; 6 convs, 3 pools. Test accuracy **74.06%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×32×32` | `8×32×32` | 584 | — | — |
| 4 | AvgPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 5 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 7 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 9 | AvgPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 10 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×8×8` | `32×8×8` | 4,640 | — | — |
| 12 | ReLU |  | `32×8×8` | `32×8×8` | — | 2,048 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×8×8` | `32×8×8` | 9,248 | — | — |
| 14 | AvgPool | 2×2, stride 2 | `32×8×8` | `32×4×4` | — | — | 512 |
| 15 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 16 | Flatten |  | `32×4×4` | `512` | — | — | — |
| 17 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **35,098** | **17,952** | **3,584** |

#### Imagenette-64 · `NP1` (max pooling)

<a id="imagenette-np1"></a>

`c4s2 p` — narrow, max pool; 1 conv, 1 pool. Test accuracy **57.73%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 3 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 4 | Flatten |  | `4×16×16` | `1024` | — | — | — |
| 5 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **33,326** | **1,056** | **1,024** |

#### Imagenette-64 · `NP2` (max pooling)

<a id="imagenette-np2"></a>

`c4s2 p c8 p` — narrow, max pool; 2 convs, 2 pools. Test accuracy **64.10%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 3 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 5 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 6 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 7 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 8 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,238** | **1,568** | **1,536** |

#### Imagenette-64 · `P1` (max pooling)

<a id="imagenette-p1"></a>

`c8s2 p` — wide, max pool; 1 conv, 1 pool. Test accuracy **64.89%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 3 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 4 | Flatten |  | `8×16×16` | `2048` | — | — | — |
| 5 | FC32 | 2048 → 32 | `2048` | `32` | 65,568 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **66,290** | **2,080** | **2,048** |

#### Imagenette-64 · `P2` (max pooling)

<a id="imagenette-p2"></a>

`c8s2 p c16 p` — wide, max pool; 2 convs, 2 pools. Test accuracy **70.19%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 3 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 5 | MaxPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 6 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 7 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 8 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 9 | ReLU |  | `32` | `32` | — | 32 | — |
| 10 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **34,690** | **3,104** | **3,072** |

#### Imagenette-64 · `NP3` (max pooling)

<a id="imagenette-np3"></a>

`c4s2 p c8 c8 p` — narrow, max pool; 3 convs, 2 pools. Test accuracy **65.76%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 3 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 5 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 7 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 8 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 9 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 10 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,822** | **3,616** | **1,536** |

#### Imagenette-64 · `P3` (max pooling)

<a id="imagenette-p3"></a>

`c8s2 p c16 c16 p` — wide, max pool; 3 convs, 2 pools. Test accuracy **71.39%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | MaxPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 3 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 5 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 7 | MaxPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 8 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 9 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 10 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **37,010** | **7,200** | **3,072** |

#### Imagenette-64 · `NP4` (max pooling)

<a id="imagenette-np4"></a>

`c4s2 c4 p c8 c8 p` — narrow, max pool; 4 convs, 2 pools. Test accuracy **66.93%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×32×32` | `4×32×32` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 5 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 7 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 10 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 11 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 12 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,970** | **7,712** | **1,536** |

#### Imagenette-64 · `NP5` (max pooling)

<a id="imagenette-np5"></a>

`c4s2 c4 p c8 c8 p c16 p` — narrow, max pool; 5 convs, 3 pools. Test accuracy **67.87%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×32×32` | `4×32×32` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 5 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 7 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 10 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 12 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 13 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 14 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 15 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **10,946** | **7,968** | **1,792** |

#### Imagenette-64 · `NP6` (max pooling)

<a id="imagenette-np6"></a>

`c4s2 c4 p c8 c8 p c16 c16 p` — narrow, max pool; 6 convs, 3 pools. Test accuracy **68.71%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×64×64` | `4×32×32` | 196 | — | — |
| 2 | ReLU |  | `4×32×32` | `4×32×32` | — | 4,096 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×32×32` | `4×32×32` | 148 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `4×32×32` | `4×16×16` | — | — | 1,024 |
| 5 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×16×16` | `8×16×16` | 296 | — | — |
| 7 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 10 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 12 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 14 | MaxPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 15 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 16 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 17 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,266** | **8,992** | **1,792** |

#### Imagenette-64 · `P4` (max pooling)

<a id="imagenette-p4"></a>

`c8s2 c8 p c16 c16 p` — wide, max pool; 4 convs, 2 pools. Test accuracy **73.45%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×32×32` | `8×32×32` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 5 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 7 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 10 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 11 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 12 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **37,594** | **15,392** | **3,072** |

#### Imagenette-64 · `P5` (max pooling)

<a id="imagenette-p5"></a>

`c8s2 c8 p c16 c16 p c32 p` — wide, max pool; 5 convs, 3 pools. Test accuracy **74.50%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×32×32` | `8×32×32` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 5 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 7 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 10 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×8×8` | `32×8×8` | 4,640 | — | — |
| 12 | MaxPool | 2×2, stride 2 | `32×8×8` | `32×4×4` | — | — | 512 |
| 13 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 14 | Flatten |  | `32×4×4` | `512` | — | — | — |
| 15 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **25,850** | **15,904** | **3,584** |

#### Imagenette-64 · `P6` (max pooling)

<a id="imagenette-p6"></a>

`c8s2 c8 p c16 c16 p c32 c32 p` — wide, max pool; 6 convs, 3 pools. Test accuracy **76.25%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×64×64` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×64×64` | `8×32×32` | 392 | — | — |
| 2 | ReLU |  | `8×32×32` | `8×32×32` | — | 8,192 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×32×32` | `8×32×32` | 584 | — | — |
| 4 | MaxPool | 2×2, stride 2 | `8×32×32` | `8×16×16` | — | — | 2,048 |
| 5 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×16×16` | `16×16×16` | 1,168 | — | — |
| 7 | ReLU |  | `16×16×16` | `16×16×16` | — | 4,096 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×16×16` | `16×16×16` | 2,320 | — | — |
| 9 | MaxPool | 2×2, stride 2 | `16×16×16` | `16×8×8` | — | — | 1,024 |
| 10 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×8×8` | `32×8×8` | 4,640 | — | — |
| 12 | ReLU |  | `32×8×8` | `32×8×8` | — | 2,048 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×8×8` | `32×8×8` | 9,248 | — | — |
| 14 | MaxPool | 2×2, stride 2 | `32×8×8` | `32×4×4` | — | — | 512 |
| 15 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 16 | Flatten |  | `32×4×4` | `512` | — | — | — |
| 17 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **35,098** | **17,952** | **3,584** |

---

## How the same architecture scales across datasets

The 36 architectures are identical everywhere; only the **input resolution**, the **input
channel count** and the **class count** change. They propagate differently:

**Resolution drives ReLU count.** Feature maps scale with input area, so Imagenette at 64×64 has
about 4× the ReLUs of the same architecture at 32×32. `S6` runs 5,312 ReLUs on MNIST,
7,200 on GTSRB/CIFAR-10 and 28,704 on Imagenette — the same network, a
5.4× spread in verification cost.

**Input channels barely matter.** Going from 1 to 3 channels changes only the *stem*
convolution: on `S1`, 392 parameters instead of 136. Everything downstream is
unaffected, since the stem's output width is 8 either way.

**Class count only touches the last layer.** GTSRB's 43 classes versus CIFAR-10's 10 changes the
final `FC` from `32 × 10 + 10 = 330` parameters to `32 × 43 + 43 = 1,419`, a difference
of 1,089 — and nothing else. That is why the GTSRB and CIFAR-10 tables are otherwise
identical, row for row, down to every shape and every ReLU count.

**Resolution also drives the flatten width**, and that is where parameter counts jump. The
shallow un-pooled models feed a large feature map straight into `FC32`:

| model | dataset | flatten width | FC32 params | share of model |
|---|---|---:|---:|---:|
| `S1` | MNIST | 1,568 | 50,208 | 99% |
| `S1` | CIFAR-10 | 2,048 | 65,568 | 99% |
| `S1` | Imagenette-64 | 8,192 | 262,176 | 100% |
| `P1` | Imagenette-64 | 2,048 | 65,568 | 99% |
| `NA1` | Imagenette-64 | 1,024 | 32,800 | 98% |
| `S6` | Imagenette-64 | 2,048 | 65,568 | 74% |

This is the clearest structural reason the pooled and narrow variants are cheaper at shallow
depth: a single 2×2 pool before the head cuts the flatten width by 4× and the head's parameters
with it, and halving the channel width cuts it again.

---

## Provenance

- **Generated from:** `data/models/{mnist,gtsrb,cifar10,imagenette}/cnn-bench/*.onnx`, read with
  `onnx.shape_inference` and walked node by node.
- **Cross-checked against:** each dataset's `cnn-bench/manifest.csv`. All 144 models agree on
  parameters, ReLU neurons and pool neurons.
- **Architecture source:** [`python_scripts/cnn_bench/archs.py`](../../python_scripts/cnn_bench/README.md);
  the id list is read from `archs.FAMILY`, so this file follows the grid automatically.
- **Ops in every exported graph:** `Conv`, `Relu`, `MaxPool`, `AveragePool`, `Flatten`, `Gemm` —
  the subset `OnnxParser` supports. No BatchNorm, no Dropout, no Softmax, no `Reshape`.
- **Input domain:** `[0,1]` for all four datasets, which is already `Network2`'s default, so
  `--input-min` / `--input-max` are not needed.
