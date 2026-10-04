# CNN-bench: layer-by-layer architecture reference

Node-level structure of all **84 models** in the CNN-bench family — 24 architectures trained
identically on up to four image datasets: all 24 on **MNIST**, **GTSRB** and **CIFAR-10**, and the
12 un-pooled ones (`S<d>`, `NS<d>`) on **Imagenette-64**.

Every table is generated directly from the exported `.onnx` graphs by walking the ONNX node list
with shape inference, so the shapes, parameter counts and ReLU counts are the ones spexplain's
`OnnxParser` actually sees — not a reconstruction from the training code. All 252 per-model
totals (84 models × params / ReLUs / pool neurons) were cross-checked against each dataset's
`manifest.csv` and agree exactly.

**Related documents.** [`CNN_BENCH_REPORT.md`](CNN_BENCH_REPORT.md) is the full experimental
report — training recipe, accuracies, validation, average-pool support. Its §12 carries a second
set of tables for the same 84 models in a *layer-level* format (one row per layer, with an
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
rows: the `Conv` that holds the parameters, the `ReLU`, then an optional `AvgPool`.

| Column | Meaning |
|---|---|
| **#** | Position in the ONNX node list (execution order). |
| **Layer** | Node type, with its output width where meaningful (`Conv16`, `FC32`). |
| **Config** | Kernel, stride and padding for `Conv`/pooling; in → out widths for `FC`. |
| **Input shape** | Shape entering the node, as `C×H×W` (or `N` for flat vectors). |
| **Output shape** | Shape leaving the node. |
| **Params** | Learnable values in this node: weights + bias. Non-zero only for `Conv` and `FC`. |
| **ReLUs** | ReLU neurons — the element count of the node's output. Each is a case split for the SMT solver. |
| **Pool** | Pooling output neurons. Every pool is an `AvgPool`, so each is a 2×2 mean — **a plain linear term, no variable and no disjunction**. |

**The batch axis is dropped.** The graphs declare a `[1,C,H,W]` input; `Network2` strips the
leading axis, so shapes are written `C×H×W`. A flat CSV row maps onto that in **CHW row-major**
order — all of channel 0, then channel 1, then channel 2.

**Parameter arithmetic.** For a convolution, `params = out_ch × in_ch × kH × kW + out_ch`; for a
fully-connected layer, `params = in × out + out`. BatchNorm is used during training but folded
exactly into the preceding convolution at export, so it costs no parameters and appears in no
graph.

---

## The architecture family

A **6 × 4 grid**: six depths (1..6 convolutions) × four variants.

| id | width | downsampling | pool branching | datasets |
|---|---|---|---|---|
| `S<d>` | wide 8/16/32 | stride-2 convolutions | none | all four |
| `AB<d>` | wide 8/16/32 | 2×2 **average** pool, ReLU before the pool | **none** — averaging is affine | MNIST, GTSRB, CIFAR-10 |
| `NS<d>` | narrow 4/8/16 | stride-2 convolutions | none | all four |
| `NAB<d>` | narrow 4/8/16 | 2×2 **average** pool, ReLU before the pool | **none** — averaging is affine | MNIST, GTSRB, CIFAR-10 |

Everything is derived from one table of strided token lists at import time, so the grid cannot
drift: `AB<d>` replaces every non-stem `c<N>s2` with `a, c<N>` and appends a final `a` — except at
depth 2, where that final pool is left out (`AB2` = `c8s2 a c16`), so the last conv feeds the head
at full resolution; `NS`/`NAB` are the 4/8/16 twins. `archs.py` asserts all of it.

Two consequences the tables make visible:

- **`S<d>`/`AB<d>` and `NS<d>`/`NAB<d>` are controlled pairs.** Because the ReLU sits *before*
  the pool, a pooled model has exactly the ReLU count of its strided twin; only the way the map
  is downsampled differs (stride-2 conv vs 3×3 conv + 2×2 average pool), and with it the
  parameter count. That isolates the cost of the pooling layer, once at each width.
- **Average pooling adds no case splits.** Its pool neurons are affine means, so an `AB`/`NAB`
  model branches on exactly the same number of neurons as its `S`/`NS` twin.

### Token grammar

| Token | Layer | Effect on spatial size |
|---|---|---|
| `c<N>` | Conv, N channels, 3×3, stride 1, pad 1 | unchanged |
| `c<N>s2` | Conv, N channels, 4×4, stride 2, pad 1 | exactly halved |
| `a` | Average pool, 2×2, stride 2 | halved (floor) |

Every convolution is followed by a ReLU. Every network ends with **Flatten → FC(32) + ReLU →
FC(classes)**, emitting raw logits — softmax lives only in the training loss.

### ReLU placement

A convolution followed by a pool emits `Conv → ReLU → AvgPool`: the ReLU sits on the
full-resolution feature map, *before* the pool. In the tables below that is why a pooled model's
`ReLU` row always comes *before* its pooling row, and why its ReLU count equals that of its
strided twin.

For average pooling the order is not cosmetic — the two orders compute different functions
(`avg(relu([-1,1])) = 0.5` while `relu(avg([-1,1])) = 0`) — so `AB`/`NAB` are the conventional
`Conv → ReLU → Pool` networks, paying ~4× the ReLUs of a pool-first ordering on every pooled conv.

The ReLU is never dropped: it is the only nonlinearity between convolutions.

### The 24 architectures

*Tables are split first by **channel width** (wide 8/16/32 vs narrow 4/8/16), then by pooling kind -- **no pooling**, **average pooling (ReLU before the pool)**. Within each group the models are ordered **smallest first**, by ReLU count ascending.*

#### Wide (8/16/32 channels)

`S<d>` / `AB<d>` -- 8/16/32 channels.

*No pooling.* `S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

| id | convs | pools | token sequence |
|---|---:|---:|---|
| `S1` | 1 | 0 | `c8s2` |
| `S2` | 2 | 0 | `c8s2 c16s2` |
| `S3` | 3 | 0 | `c8s2 c16s2 c16` |
| `S4` | 4 | 0 | `c8s2 c8 c16s2 c16` |
| `S5` | 5 | 0 | `c8s2 c8 c16s2 c16 c32s2` |
| `S6` | 6 | 0 | `c8s2 c8 c16s2 c16 c32s2 c32` |

*Average pooling, ReLU before the pool.* `AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each pool output is the mean of its window, an affine map: no fresh variable and **no disjunction**. The ReLUs sit on the full-resolution map, so the ReLU count equals the strided twin's.

| id | convs | pools | token sequence |
|---|---:|---:|---|
| `AB1` | 1 | 1 | `c8s2 a` |
| `AB2` | 2 | 1 | `c8s2 a c16` |
| `AB3` | 3 | 2 | `c8s2 a c16 c16 a` |
| `AB4` | 4 | 2 | `c8s2 c8 a c16 c16 a` |
| `AB5` | 5 | 3 | `c8s2 c8 a c16 c16 a c32 a` |
| `AB6` | 6 | 3 | `c8s2 c8 a c16 c16 a c32 c32 a` |

#### Narrow (4/8/16 channels)

`NS<d>` / `NAB<d>` -- the half-width twins of the wide models, 4/8/16 channels. Same depth, same token sequence, same downsampling; roughly half the ReLUs and 2-6x fewer parameters.

*No pooling.* `S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

| id | convs | pools | token sequence |
|---|---:|---:|---|
| `NS1` | 1 | 0 | `c4s2` |
| `NS2` | 2 | 0 | `c4s2 c8s2` |
| `NS3` | 3 | 0 | `c4s2 c8s2 c8` |
| `NS4` | 4 | 0 | `c4s2 c4 c8s2 c8` |
| `NS5` | 5 | 0 | `c4s2 c4 c8s2 c8 c16s2` |
| `NS6` | 6 | 0 | `c4s2 c4 c8s2 c8 c16s2 c16` |

*Average pooling, ReLU before the pool.* `AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each pool output is the mean of its window, an affine map: no fresh variable and **no disjunction**. The ReLUs sit on the full-resolution map, so the ReLU count equals the strided twin's.

| id | convs | pools | token sequence |
|---|---:|---:|---|
| `NAB1` | 1 | 1 | `c4s2 a` |
| `NAB2` | 2 | 1 | `c4s2 a c8` |
| `NAB3` | 3 | 2 | `c4s2 a c8 c8 a` |
| `NAB4` | 4 | 2 | `c4s2 c4 a c8 c8 a` |
| `NAB5` | 5 | 3 | `c4s2 c4 a c8 c8 a c16 a` |
| `NAB6` | 6 | 3 | `c4s2 c4 a c8 c8 a c16 c16 a` |

---

## Why ReLU count is the axis

Models are ordered by ReLU count rather than parameter count, because that is what an SMT solver
pays for: every ReLU is a case split, while a wide `Gemm` is just more linear arithmetic in
`QF_LRA`. The two measures disagree sharply, and often invert.

| MNIST | `S1` | `AB6` |
|---|---:|---:|
| Convolutions | 1 | 6 |
| Parameters | 50,674 | 19,482 |
| ReLUs | 1,600 | 5,312 |
| Pool neurons | 0 | 568 |

`S1` carries **2.6× the parameters** of `AB6` and is far the
easier model to verify: almost all its weight sits in one fully-connected layer reading a large
un-pooled feature map, which is cheap linear arithmetic.

---

## Size at a glance

ReLU neurons per model. `S<d>`/`AB<d>` share a value, as do `NS<d>`/`NAB<d>`, by construction
(the ReLU sits before the pool). `AB`/`NAB` are not trained on Imagenette-64 (`--`).

**Wide -- no pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `S1` | 1,600 | 2,080 | 2,080 | 8,224 |
| `S2` | 2,384 | 3,104 | 3,104 | 12,320 |
| `S3` | 3,168 | 4,128 | 4,128 | 16,416 |
| `S4` | 4,736 | 6,176 | 6,176 | 24,608 |
| `S5` | 5,024 | 6,688 | 6,688 | 26,656 |
| `S6` | 5,312 | 7,200 | 7,200 | 28,704 |

**Wide -- average pooling, ReLU before the pool**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `AB1` | 1,600 | 2,080 | 2,080 | -- |
| `AB2` | 2,384 | 3,104 | 3,104 | -- |
| `AB3` | 3,168 | 4,128 | 4,128 | -- |
| `AB4` | 4,736 | 6,176 | 6,176 | -- |
| `AB5` | 5,024 | 6,688 | 6,688 | -- |
| `AB6` | 5,312 | 7,200 | 7,200 | -- |

**Narrow -- no pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NS1` | 816 | 1,056 | 1,056 | 4,128 |
| `NS2` | 1,208 | 1,568 | 1,568 | 6,176 |
| `NS3` | 1,600 | 2,080 | 2,080 | 8,224 |
| `NS4` | 2,384 | 3,104 | 3,104 | 12,320 |
| `NS5` | 2,528 | 3,360 | 3,360 | 13,344 |
| `NS6` | 2,672 | 3,616 | 3,616 | 14,368 |

**Narrow -- average pooling, ReLU before the pool**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NAB1` | 816 | 1,056 | 1,056 | -- |
| `NAB2` | 1,208 | 1,568 | 1,568 | -- |
| `NAB3` | 1,600 | 2,080 | 2,080 | -- |
| `NAB4` | 2,384 | 3,104 | 3,104 | -- |
| `NAB5` | 2,528 | 3,360 | 3,360 | -- |
| `NAB6` | 2,672 | 3,616 | 3,616 | -- |

Pooling neurons (zero for `S`/`NS`; each is an affine mean, so none of them is a case split):

**Wide -- average pooling, ReLU before the pool**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `AB1` | 392 | 512 | 512 | -- |
| `AB2` | 392 | 512 | 512 | -- |
| `AB3` | 536 | 768 | 768 | -- |
| `AB4` | 536 | 768 | 768 | -- |
| `AB5` | 568 | 896 | 896 | -- |
| `AB6` | 568 | 896 | 896 | -- |

**Narrow -- average pooling, ReLU before the pool**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NAB1` | 196 | 256 | 256 | -- |
| `NAB2` | 196 | 256 | 256 | -- |
| `NAB3` | 268 | 384 | 384 | -- |
| `NAB4` | 268 | 384 | 384 | -- |
| `NAB5` | 284 | 448 | 448 | -- |
| `NAB6` | 284 | 448 | 448 | -- |

Parameters:

**Wide -- no pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `S1` | 50,674 | 67,379 | 66,290 | 262,898 |
| `S2` | 27,650 | 36,675 | 35,586 | 133,890 |
| `S3` | 29,970 | 38,995 | 37,906 | 136,210 |
| `S4` | 30,554 | 39,579 | 38,490 | 136,794 |
| `S5` | 22,906 | 31,419 | 30,330 | 79,482 |
| `S6` | 32,154 | 40,667 | 39,578 | 88,730 |

**Wide -- average pooling, ReLU before the pool**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `AB1` | 13,042 | 18,227 | 17,138 | -- |
| `AB2` | 26,754 | 35,779 | 34,690 | -- |
| `AB3` | 8,594 | 13,523 | 12,434 | -- |
| `AB4` | 9,178 | 14,107 | 13,018 | -- |
| `AB5` | 10,234 | 14,651 | 13,562 | -- |
| `AB6` | 19,482 | 23,899 | 22,810 | -- |

**Narrow -- no pooling**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NS1` | 25,518 | 34,415 | 33,326 | 131,630 |
| `NS2` | 13,494 | 18,551 | 17,462 | 66,614 |
| `NS3` | 14,078 | 19,135 | 18,046 | 67,198 |
| `NS4` | 14,226 | 19,283 | 18,194 | 67,346 |
| `NS5` | 8,354 | 13,155 | 12,066 | 36,642 |
| `NS6` | 10,674 | 15,475 | 14,386 | 38,962 |

**Narrow -- average pooling, ReLU before the pool**

| id | MNIST | GTSRB | CIFAR-10 | Imagenette-64 |
|---|---:|---:|---:|---:|
| `NAB1` | 6,702 | 9,839 | 8,750 | -- |
| `NAB2` | 13,270 | 18,327 | 17,238 | -- |
| `NAB3` | 3,614 | 6,623 | 5,534 | -- |
| `NAB4` | 3,762 | 6,771 | 5,682 | -- |
| `NAB5` | 3,138 | 5,891 | 4,802 | -- |
| `NAB6` | 5,458 | 8,211 | 7,122 | -- |

---

## MNIST

Handwritten digits, greyscale.

- **Input:** `1×28×28`, pixel values in `[0,1]`
- **Classes:** 10
- **Files:** `data/models/mnist/cnn-bench/<id>.onnx` and `<id>.pth`

### Wide -- no pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`S1`](#mnist-s1) | 1 | 0 | 1,600 | 0 | 50,674 | 98.64% |
| [`S2`](#mnist-s2) | 2 | 0 | 2,384 | 0 | 27,650 | 99.06% |
| [`S3`](#mnist-s3) | 3 | 0 | 3,168 | 0 | 29,970 | 99.36% |
| [`S4`](#mnist-s4) | 4 | 0 | 4,736 | 0 | 30,554 | 99.35% |
| [`S5`](#mnist-s5) | 5 | 0 | 5,024 | 0 | 22,906 | 99.52% |
| [`S6`](#mnist-s6) | 6 | 0 | 5,312 | 0 | 32,154 | 99.51% |

### Wide -- average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`AB1`](#mnist-ab1) | 1 | 1 | 1,600 | 392 | 13,042 | 98.35% |
| [`AB2`](#mnist-ab2) | 2 | 1 | 2,384 | 392 | 26,754 | 99.11% |
| [`AB3`](#mnist-ab3) | 3 | 2 | 3,168 | 536 | 8,594 | 99.03% |
| [`AB4`](#mnist-ab4) | 4 | 2 | 4,736 | 536 | 9,178 | 99.33% |
| [`AB5`](#mnist-ab5) | 5 | 3 | 5,024 | 568 | 10,234 | 99.34% |
| [`AB6`](#mnist-ab6) | 6 | 3 | 5,312 | 568 | 19,482 | 99.38% |

### Narrow -- no pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NS1`](#mnist-ns1) | 1 | 0 | 816 | 0 | 25,518 | 98.21% |
| [`NS2`](#mnist-ns2) | 2 | 0 | 1,208 | 0 | 13,494 | 98.57% |
| [`NS3`](#mnist-ns3) | 3 | 0 | 1,600 | 0 | 14,078 | 98.84% |
| [`NS4`](#mnist-ns4) | 4 | 0 | 2,384 | 0 | 14,226 | 99.00% |
| [`NS5`](#mnist-ns5) | 5 | 0 | 2,528 | 0 | 8,354 | 98.98% |
| [`NS6`](#mnist-ns6) | 6 | 0 | 2,672 | 0 | 10,674 | 99.15% |

### Narrow -- average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NAB1`](#mnist-nab1) | 1 | 1 | 816 | 196 | 6,702 | 97.40% |
| [`NAB2`](#mnist-nab2) | 2 | 1 | 1,208 | 196 | 13,270 | 98.68% |
| [`NAB3`](#mnist-nab3) | 3 | 2 | 1,600 | 268 | 3,614 | 98.24% |
| [`NAB4`](#mnist-nab4) | 4 | 2 | 2,384 | 268 | 3,762 | 98.66% |
| [`NAB5`](#mnist-nab5) | 5 | 3 | 2,528 | 284 | 3,138 | 98.53% |
| [`NAB6`](#mnist-nab6) | 6 | 3 | 2,672 | 284 | 5,458 | 98.84% |

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

#### MNIST · `AB1` (average pooling, ReLU before the pool)

<a id="mnist-ab1"></a>

`c8s2 a` — wide, average pool, ReLU before the pool; 1 conv, 1 pool. Test accuracy **98.35%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 4 | Flatten |  | `8×7×7` | `392` | — | — | — |
| 5 | FC32 | 392 → 32 | `392` | `32` | 12,576 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,042** | **1,600** | **392** |

#### MNIST · `AB2` (average pooling, ReLU before the pool)

<a id="mnist-ab2"></a>

`c8s2 a c16` — wide, average pool, ReLU before the pool; 2 convs, 1 pool. Test accuracy **99.11%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 5 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 6 | Flatten |  | `16×7×7` | `784` | — | — | — |
| 7 | FC32 | 784 → 32 | `784` | `32` | 25,120 | — | — |
| 8 | ReLU |  | `32` | `32` | — | 32 | — |
| 9 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **26,754** | **2,384** | **392** |

#### MNIST · `AB3` (average pooling, ReLU before the pool)

<a id="mnist-ab3"></a>

`c8s2 a c16 c16 a` — wide, average pool, ReLU before the pool; 3 convs, 2 pools. Test accuracy **99.03%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 5 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | AvgPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 9 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 10 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **8,594** | **3,168** | **536** |

#### MNIST · `AB4` (average pooling, ReLU before the pool)

<a id="mnist-ab4"></a>

`c8s2 c8 a c16 c16 a` — wide, average pool, ReLU before the pool; 4 convs, 2 pools. Test accuracy **99.33%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 5 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 9 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 10 | AvgPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 11 | Flatten |  | `16×3×3` | `144` | — | — | — |
| 12 | FC32 | 144 → 32 | `144` | `32` | 4,640 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **9,178** | **4,736** | **536** |

#### MNIST · `AB5` (average pooling, ReLU before the pool)

<a id="mnist-ab5"></a>

`c8s2 c8 a c16 c16 a c32 a` — wide, average pool, ReLU before the pool; 5 convs, 3 pools. Test accuracy **99.34%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 5 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 9 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 10 | AvgPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×3×3` | `32×3×3` | 4,640 | — | — |
| 12 | ReLU |  | `32×3×3` | `32×3×3` | — | 288 | — |
| 13 | AvgPool | 2×2, stride 2 | `32×3×3` | `32×1×1` | — | — | 32 |
| 14 | Flatten |  | `32×1×1` | `32` | — | — | — |
| 15 | FC32 | 32 → 32 | `32` | `32` | 1,056 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **10,234** | **5,024** | **568** |

#### MNIST · `AB6` (average pooling, ReLU before the pool)

<a id="mnist-ab6"></a>

`c8s2 c8 a c16 c16 a c32 c32 a` — wide, average pool, ReLU before the pool; 6 convs, 3 pools. Test accuracy **99.38%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `1×28×28` | `8×14×14` | 136 | — | — |
| 2 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×14×14` | `8×14×14` | 584 | — | — |
| 4 | ReLU |  | `8×14×14` | `8×14×14` | — | 1,568 | — |
| 5 | AvgPool | 2×2, stride 2 | `8×14×14` | `8×7×7` | — | — | 392 |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×7×7` | `16×7×7` | 1,168 | — | — |
| 7 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×7×7` | `16×7×7` | 2,320 | — | — |
| 9 | ReLU |  | `16×7×7` | `16×7×7` | — | 784 | — |
| 10 | AvgPool | 2×2, stride 2 | `16×7×7` | `16×3×3` | — | — | 144 |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×3×3` | `32×3×3` | 4,640 | — | — |
| 12 | ReLU |  | `32×3×3` | `32×3×3` | — | 288 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×3×3` | `32×3×3` | 9,248 | — | — |
| 14 | ReLU |  | `32×3×3` | `32×3×3` | — | 288 | — |
| 15 | AvgPool | 2×2, stride 2 | `32×3×3` | `32×1×1` | — | — | 32 |
| 16 | Flatten |  | `32×1×1` | `32` | — | — | — |
| 17 | FC32 | 32 → 32 | `32` | `32` | 1,056 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **19,482** | **5,312** | **568** |

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

#### MNIST · `NAB1` (average pooling, ReLU before the pool)

<a id="mnist-nab1"></a>

`c4s2 a` — narrow, average pool, ReLU before the pool; 1 conv, 1 pool. Test accuracy **97.40%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 4 | Flatten |  | `4×7×7` | `196` | — | — | — |
| 5 | FC32 | 196 → 32 | `196` | `32` | 6,304 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **6,702** | **816** | **196** |

#### MNIST · `NAB2` (average pooling, ReLU before the pool)

<a id="mnist-nab2"></a>

`c4s2 a c8` — narrow, average pool, ReLU before the pool; 2 convs, 1 pool. Test accuracy **98.68%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Flatten |  | `8×7×7` | `392` | — | — | — |
| 7 | FC32 | 392 → 32 | `392` | `32` | 12,576 | — | — |
| 8 | ReLU |  | `32` | `32` | — | 32 | — |
| 9 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,270** | **1,208** | **196** |

#### MNIST · `NAB3` (average pooling, ReLU before the pool)

<a id="mnist-nab3"></a>

`c4s2 a c8 c8 a` — narrow, average pool, ReLU before the pool; 3 convs, 2 pools. Test accuracy **98.24%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 5 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | AvgPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 9 | Flatten |  | `8×3×3` | `72` | — | — | — |
| 10 | FC32 | 72 → 32 | `72` | `32` | 2,336 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,614** | **1,600** | **268** |

#### MNIST · `NAB4` (average pooling, ReLU before the pool)

<a id="mnist-nab4"></a>

`c4s2 c4 a c8 c8 a` — narrow, average pool, ReLU before the pool; 4 convs, 2 pools. Test accuracy **98.66%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 5 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 9 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 10 | AvgPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 11 | Flatten |  | `8×3×3` | `72` | — | — | — |
| 12 | FC32 | 72 → 32 | `72` | `32` | 2,336 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,762** | **2,384** | **268** |

#### MNIST · `NAB5` (average pooling, ReLU before the pool)

<a id="mnist-nab5"></a>

`c4s2 c4 a c8 c8 a c16 a` — narrow, average pool, ReLU before the pool; 5 convs, 3 pools. Test accuracy **98.53%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 5 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 9 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 10 | AvgPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×3×3` | `16×3×3` | 1,168 | — | — |
| 12 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 13 | AvgPool | 2×2, stride 2 | `16×3×3` | `16×1×1` | — | — | 16 |
| 14 | Flatten |  | `16×1×1` | `16` | — | — | — |
| 15 | FC32 | 16 → 32 | `16` | `32` | 544 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **3,138** | **2,528** | **284** |

#### MNIST · `NAB6` (average pooling, ReLU before the pool)

<a id="mnist-nab6"></a>

`c4s2 c4 a c8 c8 a c16 c16 a` — narrow, average pool, ReLU before the pool; 6 convs, 3 pools. Test accuracy **98.84%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `1×28×28` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `1×28×28` | `4×14×14` | 68 | — | — |
| 2 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×14×14` | `4×14×14` | 148 | — | — |
| 4 | ReLU |  | `4×14×14` | `4×14×14` | — | 784 | — |
| 5 | AvgPool | 2×2, stride 2 | `4×14×14` | `4×7×7` | — | — | 196 |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×7×7` | `8×7×7` | 296 | — | — |
| 7 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×7×7` | `8×7×7` | 584 | — | — |
| 9 | ReLU |  | `8×7×7` | `8×7×7` | — | 392 | — |
| 10 | AvgPool | 2×2, stride 2 | `8×7×7` | `8×3×3` | — | — | 72 |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×3×3` | `16×3×3` | 1,168 | — | — |
| 12 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×3×3` | `16×3×3` | 2,320 | — | — |
| 14 | ReLU |  | `16×3×3` | `16×3×3` | — | 144 | — |
| 15 | AvgPool | 2×2, stride 2 | `16×3×3` | `16×1×1` | — | — | 16 |
| 16 | Flatten |  | `16×1×1` | `16` | — | — | — |
| 17 | FC32 | 16 → 32 | `16` | `32` | 544 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **5,458** | **2,672** | **284** |

---

## GTSRB

German traffic signs, resized to 32×32. The only 43-class family.

- **Input:** `3×32×32`, pixel values in `[0,1]`
- **Classes:** 43
- **Files:** `data/models/gtsrb/cnn-bench/<id>.onnx` and `<id>.pth`

### Wide -- no pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`S1`](#gtsrb-s1) | 1 | 0 | 2,080 | 0 | 67,379 | 86.20% |
| [`S2`](#gtsrb-s2) | 2 | 0 | 3,104 | 0 | 36,675 | 90.78% |
| [`S3`](#gtsrb-s3) | 3 | 0 | 4,128 | 0 | 38,995 | 92.61% |
| [`S4`](#gtsrb-s4) | 4 | 0 | 6,176 | 0 | 39,579 | 94.45% |
| [`S5`](#gtsrb-s5) | 5 | 0 | 6,688 | 0 | 31,419 | 94.64% |
| [`S6`](#gtsrb-s6) | 6 | 0 | 7,200 | 0 | 40,667 | 95.62% |

### Wide -- average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`AB1`](#gtsrb-ab1) | 1 | 1 | 2,080 | 512 | 18,227 | 83.06% |
| [`AB2`](#gtsrb-ab2) | 2 | 1 | 3,104 | 512 | 35,779 | 87.74% |
| [`AB3`](#gtsrb-ab3) | 3 | 2 | 4,128 | 768 | 13,523 | 89.02% |
| [`AB4`](#gtsrb-ab4) | 4 | 2 | 6,176 | 768 | 14,107 | 92.69% |
| [`AB5`](#gtsrb-ab5) | 5 | 3 | 6,688 | 896 | 14,651 | 91.73% |
| [`AB6`](#gtsrb-ab6) | 6 | 3 | 7,200 | 896 | 23,899 | 93.05% |

### Narrow -- no pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NS1`](#gtsrb-ns1) | 1 | 0 | 1,056 | 0 | 34,415 | 84.68% |
| [`NS2`](#gtsrb-ns2) | 2 | 0 | 1,568 | 0 | 18,551 | 86.44% |
| [`NS3`](#gtsrb-ns3) | 3 | 0 | 2,080 | 0 | 19,135 | 87.97% |
| [`NS4`](#gtsrb-ns4) | 4 | 0 | 3,104 | 0 | 19,283 | 90.37% |
| [`NS5`](#gtsrb-ns5) | 5 | 0 | 3,360 | 0 | 13,155 | 90.04% |
| [`NS6`](#gtsrb-ns6) | 6 | 0 | 3,616 | 0 | 15,475 | 89.72% |

### Narrow -- average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NAB1`](#gtsrb-nab1) | 1 | 1 | 1,056 | 256 | 9,839 | 77.59% |
| [`NAB2`](#gtsrb-nab2) | 2 | 1 | 1,568 | 256 | 18,327 | 83.20% |
| [`NAB3`](#gtsrb-nab3) | 3 | 2 | 2,080 | 384 | 6,623 | 77.72% |
| [`NAB4`](#gtsrb-nab4) | 4 | 2 | 3,104 | 384 | 6,771 | 84.73% |
| [`NAB5`](#gtsrb-nab5) | 5 | 3 | 3,360 | 448 | 5,891 | 81.30% |
| [`NAB6`](#gtsrb-nab6) | 6 | 3 | 3,616 | 448 | 8,211 | 83.25% |

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

#### GTSRB · `AB1` (average pooling, ReLU before the pool)

<a id="gtsrb-ab1"></a>

`c8s2 a` — wide, average pool, ReLU before the pool; 1 conv, 1 pool. Test accuracy **83.06%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 4 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 5 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **18,227** | **2,080** | **512** |

#### GTSRB · `AB2` (average pooling, ReLU before the pool)

<a id="gtsrb-ab2"></a>

`c8s2 a c16` — wide, average pool, ReLU before the pool; 2 convs, 1 pool. Test accuracy **87.74%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 6 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 7 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 8 | ReLU |  | `32` | `32` | — | 32 | — |
| 9 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **35,779** | **3,104** | **512** |

#### GTSRB · `AB3` (average pooling, ReLU before the pool)

<a id="gtsrb-ab3"></a>

`c8s2 a c16 c16 a` — wide, average pool, ReLU before the pool; 3 convs, 2 pools. Test accuracy **89.02%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 9 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 10 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **13,523** | **4,128** | **768** |

#### GTSRB · `AB4` (average pooling, ReLU before the pool)

<a id="gtsrb-ab4"></a>

`c8s2 c8 a c16 c16 a` — wide, average pool, ReLU before the pool; 4 convs, 2 pools. Test accuracy **92.69%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 10 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 11 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 12 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **14,107** | **6,176** | **768** |

#### GTSRB · `AB5` (average pooling, ReLU before the pool)

<a id="gtsrb-ab5"></a>

`c8s2 c8 a c16 c16 a c32 a` — wide, average pool, ReLU before the pool; 5 convs, 3 pools. Test accuracy **91.73%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 10 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | AvgPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 14 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 15 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **14,651** | **6,688** | **896** |

#### GTSRB · `AB6` (average pooling, ReLU before the pool)

<a id="gtsrb-ab6"></a>

`c8s2 c8 a c16 c16 a c32 c32 a` — wide, average pool, ReLU before the pool; 6 convs, 3 pools. Test accuracy **93.05%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 10 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×4×4` | `32×4×4` | 9,248 | — | — |
| 14 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 15 | AvgPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 16 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 17 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **23,899** | **7,200** | **896** |

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

#### GTSRB · `NAB1` (average pooling, ReLU before the pool)

<a id="gtsrb-nab1"></a>

`c4s2 a` — narrow, average pool, ReLU before the pool; 1 conv, 1 pool. Test accuracy **77.59%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 4 | Flatten |  | `4×8×8` | `256` | — | — | — |
| 5 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **9,839** | **1,056** | **256** |

#### GTSRB · `NAB2` (average pooling, ReLU before the pool)

<a id="gtsrb-nab2"></a>

`c4s2 a c8` — narrow, average pool, ReLU before the pool; 2 convs, 1 pool. Test accuracy **83.20%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 7 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 8 | ReLU |  | `32` | `32` | — | 32 | — |
| 9 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **18,327** | **1,568** | **256** |

#### GTSRB · `NAB3` (average pooling, ReLU before the pool)

<a id="gtsrb-nab3"></a>

`c4s2 a c8 c8 a` — narrow, average pool, ReLU before the pool; 3 convs, 2 pools. Test accuracy **77.72%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 9 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 10 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **6,623** | **2,080** | **384** |

#### GTSRB · `NAB4` (average pooling, ReLU before the pool)

<a id="gtsrb-nab4"></a>

`c4s2 c4 a c8 c8 a` — narrow, average pool, ReLU before the pool; 4 convs, 2 pools. Test accuracy **84.73%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 10 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 11 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 12 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **6,771** | **3,104** | **384** |

#### GTSRB · `NAB5` (average pooling, ReLU before the pool)

<a id="gtsrb-nab5"></a>

`c4s2 c4 a c8 c8 a c16 a` — narrow, average pool, ReLU before the pool; 5 convs, 3 pools. Test accuracy **81.30%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 10 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | AvgPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 14 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 15 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **5,891** | **3,360** | **448** |

#### GTSRB · `NAB6` (average pooling, ReLU before the pool)

<a id="gtsrb-nab6"></a>

`c4s2 c4 a c8 c8 a c16 c16 a` — narrow, average pool, ReLU before the pool; 6 convs, 3 pools. Test accuracy **83.25%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 10 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×4×4` | `16×4×4` | 2,320 | — | — |
| 14 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 15 | AvgPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 16 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 17 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC43 | 32 → 43 | `32` | `43` | 1,419 | — | — |
| | **Total** | | | | **8,211** | **3,616** | **448** |

---

## CIFAR-10

Natural images across 10 object classes.

- **Input:** `3×32×32`, pixel values in `[0,1]`
- **Classes:** 10
- **Files:** `data/models/cifar/cnn-bench/<id>.onnx` and `<id>.pth`

### Wide -- no pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`S1`](#cifar-s1) | 1 | 0 | 2,080 | 0 | 66,290 | 61.36% |
| [`S2`](#cifar-s2) | 2 | 0 | 3,104 | 0 | 35,586 | 67.78% |
| [`S3`](#cifar-s3) | 3 | 0 | 4,128 | 0 | 37,906 | 70.54% |
| [`S4`](#cifar-s4) | 4 | 0 | 6,176 | 0 | 38,490 | 73.39% |
| [`S5`](#cifar-s5) | 5 | 0 | 6,688 | 0 | 30,330 | 74.79% |
| [`S6`](#cifar-s6) | 6 | 0 | 7,200 | 0 | 39,578 | 76.30% |

### Wide -- average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`AB1`](#cifar-ab1) | 1 | 1 | 2,080 | 512 | 17,138 | 61.20% |
| [`AB2`](#cifar-ab2) | 2 | 1 | 3,104 | 512 | 34,690 | 67.99% |
| [`AB3`](#cifar-ab3) | 3 | 2 | 4,128 | 768 | 12,434 | 70.33% |
| [`AB4`](#cifar-ab4) | 4 | 2 | 6,176 | 768 | 13,018 | 71.49% |
| [`AB5`](#cifar-ab5) | 5 | 3 | 6,688 | 896 | 13,562 | 73.68% |
| [`AB6`](#cifar-ab6) | 6 | 3 | 7,200 | 896 | 22,810 | 75.35% |

### Narrow -- no pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NS1`](#cifar-ns1) | 1 | 0 | 1,056 | 0 | 33,326 | 55.11% |
| [`NS2`](#cifar-ns2) | 2 | 0 | 1,568 | 0 | 17,462 | 58.63% |
| [`NS3`](#cifar-ns3) | 3 | 0 | 2,080 | 0 | 18,046 | 61.88% |
| [`NS4`](#cifar-ns4) | 4 | 0 | 3,104 | 0 | 18,194 | 62.54% |
| [`NS5`](#cifar-ns5) | 5 | 0 | 3,360 | 0 | 12,066 | 65.47% |
| [`NS6`](#cifar-ns6) | 6 | 0 | 3,616 | 0 | 14,386 | 66.27% |

### Narrow -- average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NAB1`](#cifar-nab1) | 1 | 1 | 1,056 | 256 | 8,750 | 54.31% |
| [`NAB2`](#cifar-nab2) | 2 | 1 | 1,568 | 256 | 17,238 | 58.15% |
| [`NAB3`](#cifar-nab3) | 3 | 2 | 2,080 | 384 | 5,534 | 57.09% |
| [`NAB4`](#cifar-nab4) | 4 | 2 | 3,104 | 384 | 5,682 | 61.23% |
| [`NAB5`](#cifar-nab5) | 5 | 3 | 3,360 | 448 | 4,802 | 59.88% |
| [`NAB6`](#cifar-nab6) | 6 | 3 | 3,616 | 448 | 7,122 | 62.60% |

#### CIFAR-10 · `S1` (no pooling)

<a id="cifar-s1"></a>

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

#### CIFAR-10 · `S2` (no pooling)

<a id="cifar-s2"></a>

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

#### CIFAR-10 · `S3` (no pooling)

<a id="cifar-s3"></a>

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

<a id="cifar-s4"></a>

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

<a id="cifar-s5"></a>

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

<a id="cifar-s6"></a>

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

#### CIFAR-10 · `AB1` (average pooling, ReLU before the pool)

<a id="cifar-ab1"></a>

`c8s2 a` — wide, average pool, ReLU before the pool; 1 conv, 1 pool. Test accuracy **61.20%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 4 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 5 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,138** | **2,080** | **512** |

#### CIFAR-10 · `AB2` (average pooling, ReLU before the pool)

<a id="cifar-ab2"></a>

`c8s2 a c16` — wide, average pool, ReLU before the pool; 2 convs, 1 pool. Test accuracy **67.99%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 6 | Flatten |  | `16×8×8` | `1024` | — | — | — |
| 7 | FC32 | 1024 → 32 | `1024` | `32` | 32,800 | — | — |
| 8 | ReLU |  | `32` | `32` | — | 32 | — |
| 9 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **34,690** | **3,104** | **512** |

#### CIFAR-10 · `AB3` (average pooling, ReLU before the pool)

<a id="cifar-ab3"></a>

`c8s2 a c16 c16 a` — wide, average pool, ReLU before the pool; 3 convs, 2 pools. Test accuracy **70.33%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 4 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 5 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 9 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 10 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **12,434** | **4,128** | **768** |

#### CIFAR-10 · `AB4` (average pooling, ReLU before the pool)

<a id="cifar-ab4"></a>

`c8s2 c8 a c16 c16 a` — wide, average pool, ReLU before the pool; 4 convs, 2 pools. Test accuracy **71.49%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 10 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 11 | Flatten |  | `16×4×4` | `256` | — | — | — |
| 12 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,018** | **6,176** | **768** |

#### CIFAR-10 · `AB5` (average pooling, ReLU before the pool)

<a id="cifar-ab5"></a>

`c8s2 c8 a c16 c16 a c32 a` — wide, average pool, ReLU before the pool; 5 convs, 3 pools. Test accuracy **73.68%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 10 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | AvgPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 14 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 15 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **13,562** | **6,688** | **896** |

#### CIFAR-10 · `AB6` (average pooling, ReLU before the pool)

<a id="cifar-ab6"></a>

`c8s2 c8 a c16 c16 a c32 c32 a` — wide, average pool, ReLU before the pool; 6 convs, 3 pools. Test accuracy **75.35%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv8 | 4×4, stride 2, pad 1 | `3×32×32` | `8×16×16` | 392 | — | — |
| 2 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 3 | Conv8 | 3×3, stride 1, pad 1 | `8×16×16` | `8×16×16` | 584 | — | — |
| 4 | ReLU |  | `8×16×16` | `8×16×16` | — | 2,048 | — |
| 5 | AvgPool | 2×2, stride 2 | `8×16×16` | `8×8×8` | — | — | 512 |
| 6 | Conv16 | 3×3, stride 1, pad 1 | `8×8×8` | `16×8×8` | 1,168 | — | — |
| 7 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 8 | Conv16 | 3×3, stride 1, pad 1 | `16×8×8` | `16×8×8` | 2,320 | — | — |
| 9 | ReLU |  | `16×8×8` | `16×8×8` | — | 1,024 | — |
| 10 | AvgPool | 2×2, stride 2 | `16×8×8` | `16×4×4` | — | — | 256 |
| 11 | Conv32 | 3×3, stride 1, pad 1 | `16×4×4` | `32×4×4` | 4,640 | — | — |
| 12 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 13 | Conv32 | 3×3, stride 1, pad 1 | `32×4×4` | `32×4×4` | 9,248 | — | — |
| 14 | ReLU |  | `32×4×4` | `32×4×4` | — | 512 | — |
| 15 | AvgPool | 2×2, stride 2 | `32×4×4` | `32×2×2` | — | — | 128 |
| 16 | Flatten |  | `32×2×2` | `128` | — | — | — |
| 17 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **22,810** | **7,200** | **896** |

#### CIFAR-10 · `NS1` (no pooling)

<a id="cifar-ns1"></a>

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

<a id="cifar-ns2"></a>

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

#### CIFAR-10 · `NS3` (no pooling)

<a id="cifar-ns3"></a>

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

#### CIFAR-10 · `NS4` (no pooling)

<a id="cifar-ns4"></a>

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

<a id="cifar-ns5"></a>

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

<a id="cifar-ns6"></a>

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

#### CIFAR-10 · `NAB1` (average pooling, ReLU before the pool)

<a id="cifar-nab1"></a>

`c4s2 a` — narrow, average pool, ReLU before the pool; 1 conv, 1 pool. Test accuracy **54.31%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 4 | Flatten |  | `4×8×8` | `256` | — | — | — |
| 5 | FC32 | 256 → 32 | `256` | `32` | 8,224 | — | — |
| 6 | ReLU |  | `32` | `32` | — | 32 | — |
| 7 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **8,750** | **1,056** | **256** |

#### CIFAR-10 · `NAB2` (average pooling, ReLU before the pool)

<a id="cifar-nab2"></a>

`c4s2 a c8` — narrow, average pool, ReLU before the pool; 2 convs, 1 pool. Test accuracy **58.15%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Flatten |  | `8×8×8` | `512` | — | — | — |
| 7 | FC32 | 512 → 32 | `512` | `32` | 16,416 | — | — |
| 8 | ReLU |  | `32` | `32` | — | 32 | — |
| 9 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **17,238** | **1,568** | **256** |

#### CIFAR-10 · `NAB3` (average pooling, ReLU before the pool)

<a id="cifar-nab3"></a>

`c4s2 a c8 c8 a` — narrow, average pool, ReLU before the pool; 3 convs, 2 pools. Test accuracy **57.09%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 4 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 5 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 9 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 10 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 11 | ReLU |  | `32` | `32` | — | 32 | — |
| 12 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **5,534** | **2,080** | **384** |

#### CIFAR-10 · `NAB4` (average pooling, ReLU before the pool)

<a id="cifar-nab4"></a>

`c4s2 c4 a c8 c8 a` — narrow, average pool, ReLU before the pool; 4 convs, 2 pools. Test accuracy **61.23%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 10 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 11 | Flatten |  | `8×4×4` | `128` | — | — | — |
| 12 | FC32 | 128 → 32 | `128` | `32` | 4,128 | — | — |
| 13 | ReLU |  | `32` | `32` | — | 32 | — |
| 14 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **5,682** | **3,104** | **384** |

#### CIFAR-10 · `NAB5` (average pooling, ReLU before the pool)

<a id="cifar-nab5"></a>

`c4s2 c4 a c8 c8 a c16 a` — narrow, average pool, ReLU before the pool; 5 convs, 3 pools. Test accuracy **59.88%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 10 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | AvgPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 14 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 15 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 16 | ReLU |  | `32` | `32` | — | 32 | — |
| 17 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **4,802** | **3,360** | **448** |

#### CIFAR-10 · `NAB6` (average pooling, ReLU before the pool)

<a id="cifar-nab6"></a>

`c4s2 c4 a c8 c8 a c16 c16 a` — narrow, average pool, ReLU before the pool; 6 convs, 3 pools. Test accuracy **62.60%**.

| # | Layer | Config | Input shape | Output shape | Params | ReLUs | Pool |
|---:|---|---|---|---|---:|---:|---:|
| — | *input* | | | `3×32×32` | — | — | — |
| 1 | Conv4 | 4×4, stride 2, pad 1 | `3×32×32` | `4×16×16` | 196 | — | — |
| 2 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 3 | Conv4 | 3×3, stride 1, pad 1 | `4×16×16` | `4×16×16` | 148 | — | — |
| 4 | ReLU |  | `4×16×16` | `4×16×16` | — | 1,024 | — |
| 5 | AvgPool | 2×2, stride 2 | `4×16×16` | `4×8×8` | — | — | 256 |
| 6 | Conv8 | 3×3, stride 1, pad 1 | `4×8×8` | `8×8×8` | 296 | — | — |
| 7 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 8 | Conv8 | 3×3, stride 1, pad 1 | `8×8×8` | `8×8×8` | 584 | — | — |
| 9 | ReLU |  | `8×8×8` | `8×8×8` | — | 512 | — |
| 10 | AvgPool | 2×2, stride 2 | `8×8×8` | `8×4×4` | — | — | 128 |
| 11 | Conv16 | 3×3, stride 1, pad 1 | `8×4×4` | `16×4×4` | 1,168 | — | — |
| 12 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 13 | Conv16 | 3×3, stride 1, pad 1 | `16×4×4` | `16×4×4` | 2,320 | — | — |
| 14 | ReLU |  | `16×4×4` | `16×4×4` | — | 256 | — |
| 15 | AvgPool | 2×2, stride 2 | `16×4×4` | `16×2×2` | — | — | 64 |
| 16 | Flatten |  | `16×2×2` | `64` | — | — | — |
| 17 | FC32 | 64 → 32 | `64` | `32` | 2,080 | — | — |
| 18 | ReLU |  | `32` | `32` | — | 32 | — |
| 19 | FC10 | 32 → 10 | `32` | `10` | 330 | — | — |
| | **Total** | | | | **7,122** | **3,616** | **448** |

---

## Imagenette-64

The 10-class ImageNet subset from fast.ai, resized so the shorter side is 72 then centre-cropped to 64. Four times the pixels of CIFAR/GTSRB.

- **Input:** `3×64×64`, pixel values in `[0,1]`
- **Classes:** 10
- **Files:** `data/models/imagenette/cnn-bench/<id>.onnx` and `<id>.pth`

### Wide -- no pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`S1`](#imagenette-s1) | 1 | 0 | 8,224 | 0 | 262,898 | 62.39% |
| [`S2`](#imagenette-s2) | 2 | 0 | 12,320 | 0 | 133,890 | 67.62% |
| [`S3`](#imagenette-s3) | 3 | 0 | 16,416 | 0 | 136,210 | 71.95% |
| [`S4`](#imagenette-s4) | 4 | 0 | 24,608 | 0 | 136,794 | 72.87% |
| [`S5`](#imagenette-s5) | 5 | 0 | 26,656 | 0 | 79,482 | 74.55% |
| [`S6`](#imagenette-s6) | 6 | 0 | 28,704 | 0 | 88,730 | 75.90% |

### Narrow -- no pooling

| id | convs | pools | ReLUs | pool | params | test acc |
|---|---:|---:|---:|---:|---:|---:|
| [`NS1`](#imagenette-ns1) | 1 | 0 | 4,128 | 0 | 131,630 | 55.59% |
| [`NS2`](#imagenette-ns2) | 2 | 0 | 6,176 | 0 | 66,614 | 64.03% |
| [`NS3`](#imagenette-ns3) | 3 | 0 | 8,224 | 0 | 67,198 | 65.66% |
| [`NS4`](#imagenette-ns4) | 4 | 0 | 12,320 | 0 | 67,346 | 67.31% |
| [`NS5`](#imagenette-ns5) | 5 | 0 | 13,344 | 0 | 36,642 | 68.61% |
| [`NS6`](#imagenette-ns6) | 6 | 0 | 14,368 | 0 | 38,962 | 71.26% |

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

---

## How the same architecture scales across datasets

The 24 architectures are identical on every dataset they are trained on (`AB`/`NAB` skip
Imagenette-64); only the **input resolution**, the **input channel count** and the **class count**
change. They propagate differently:

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
| `AB1` | CIFAR-10 | 512 | 16,416 | 96% |
| `NAB1` | CIFAR-10 | 256 | 8,224 | 94% |
| `S6` | Imagenette-64 | 2,048 | 65,568 | 74% |

This is the clearest structural reason the pooled and narrow variants are smaller at shallow
depth: a single 2×2 pool before the head cuts the flatten width by 4× and the head's parameters
with it, and halving the channel width cuts it again. For `AB`/`NAB` the saving is in parameters
only — their ReLU count equals the strided twin's, so the solver's case splits are unchanged.

---

## Provenance

- **Generated from:** `data/models/{mnist,gtsrb,cifar,imagenette}/cnn-bench/*.onnx`, read with
  `onnx.shape_inference` and walked node by node.
- **Cross-checked against:** each dataset's `cnn-bench/manifest.csv`. All 84 models agree on
  parameters, ReLU neurons and pool neurons.
- **Architecture source:** [`python_scripts/cnn_bench/archs.py`](../../python_scripts/cnn_bench/README.md);
  the id list is read from `archs.FAMILY`, so this file follows the grid automatically.
- **Ops across the exported graphs:** `Conv`, `Relu`, `Flatten`, `Gemm`, plus `AveragePool` in `AB`/`NAB` —
  the subset `OnnxParser` supports. No BatchNorm, no Dropout, no Softmax, no `Reshape`.
- **Input domain:** `[0,1]` for all four datasets, which is already `Network2`'s default, so
  `--input-min` / `--input-max` are not needed.
