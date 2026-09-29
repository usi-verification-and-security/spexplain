# CNN bench: 144 CNNs for spexplain scaling experiments

This is the report on the CNN benchmark family built for the paper's scaling study:

- **36 architectures**, identical across 4 image datasets: MNIST, GTSRB, CIFAR-10 and Imagenette at 64x64.
- All **144** models are trained, exported to ONNX and validated against spexplain's own ONNX front end.
- Everything is reproducible from the scripts in [`python_scripts/cnn_bench/`](../../python_scripts/cnn_bench/README.md).

Date: 2026-09-26. Hardware: Apple M3, 16 GB, trained on MPS.

Per-layer architecture tables for every model are in [section 11](#11-architecture-tables).

---

## 1. Summary

- **One family of 36 architectures for every dataset:** 6 depths (1..6 convolutions) x 6 variants.

  | variant | width | downsampling | pool branching |
  |---|---|---|---|
  | `S<d>` | wide 8/16/32 | stride-2 convolutions | none |
  | `P<d>` | wide 8/16/32 | 2x2 **max** pool | one 4-way max per pooled neuron |
  | `A<d>` | wide 8/16/32 | 2x2 **average** pool | **none** -- averaging is affine |
  | `NS<d>` | narrow 4/8/16 | stride-2 convolutions | none |
  | `NA<d>` | narrow 4/8/16 | 2x2 **average** pool | **none** |

- **`P<d>`/`A<d>` and `NP<d>`/`NA<d>` are the controlled pairs.** Each pair has identical shapes, identical ReLU
  counts and identical parameter counts, and differ *only* in the pooling operator. Max pooling
  makes every pooled neuron a fresh variable plus a disjunction; average pooling is a plain
  linear combination that adds **no case splits at all**. That pair isolates the cost of
  max-pool branching with everything else held fixed -- once at the wide 8/16/32 width and
  once at the narrow 4/8/16 width.
- **Average pooling is new C++ support.** `AveragePool` was not in `OnnxParser`; this run added
  `AvgPoolLayer` (eval), the parser case, and an encoder in `OpenSMTVerifier2` that reuses the
  affine encoder. See section 5.
- **ReLU now sits after pooling.** A conv immediately followed by a pool emits
  `Conv -> BN -> Pool -> ReLU`. For max pooling this is *exactly the same function* (ReLU is
  monotone, so it commutes with max) but puts the activation on the 4x smaller map, cutting the
  ReLU count ~4x at no cost in accuracy. See section 3.
- **Allowed non-linearities only.** The exported graphs contain only `Conv`, `Relu`, `MaxPool`,
  `AveragePool`, `Flatten` and `Gemm`, and output raw logits. The script checks this for every file.
  - BatchNorm is used in training and folded exactly into the convolutions at export.
  - Dropout exists only in training. Softmax exists only in the loss.
- **Complexity range:**

  | dataset | ReLUs | pool neurons |
  |---|---|---|
  | MNIST | 228 -> 5,312 | 0 -> 568 |
  | GTSRB | 288 -> 7,200 | 0 -> 896 |
  | CIFAR-10 | 288 -> 7,200 | 0 -> 896 |
  | Imagenette-64 | 1,056 -> 28,704 | 0 -> 3,584 |

- **Accuracy (top-1 on the full test split):**

  | dataset | range |
  |---|---|
  | MNIST | 95.4-99.5% |
  | GTSRB | 64.9-95.6% |
  | CIFAR-10 | 50.1-76.3% |
  | Imagenette-64 | 50.0-76.2% |

- **Validation:** for all 144 models, spexplain's `onnx-eval` (OnnxParser + Network2) reproduces
  onnxruntime's logits to <= 1.1e-5 on the 100 experiment rows, with **0 argmax disagreements**.
- **Neuron activations** were regenerated for all 144 models.

---

## 2. Decisions made for this round

| question | decision |
|---|---|
| Average pooling | **Implemented in C++.** `AveragePool` had no parser case and would throw. Rather than fake it with a fixed-weight convolution, `AvgPoolLayer` was added to `Network2`, a case to `OnnxParser`, and an encoder to `OpenSMTVerifier2`. Because averaging is affine it reuses the existing affine encoder, so it needs no fresh variables and adds no disjunctions. |
| ReLU vs. pooling order | **Pool first, then ReLU.** The ReLU is *not* removed -- it is the only nonlinearity between convolutions, and dropping it would collapse adjacent convs into a single linear map. Instead it moves after the pool. For max pooling that is an exact identity; for average pooling it is a deliberate choice that keeps the `A<d>` ReLU counts equal to `P<d>`. |
| Depth range | **1..6 convolutions.** Depth 5 was added; the old 8-deep `S8`/`P8` were dropped. |
| Narrow max pool | **Added as `NP<d>`**, completing the grid: the narrow row now mirrors the wide row exactly (strided / max pool / average pool), so the max-vs-average comparison exists at both widths. |
| Narrow width | **4/8/16.** A second width class, half the channels of the original family, to give genuinely small models. |
| Min pool | **Still not used.** spexplain has no min-pool support and ONNX has no MinPool op. |
| Padding | Symmetric `pad=1` convolutions, as before. |

---

## 3. Architecture family

Token grammar, from `archs.py`:

- `cN`: conv with N channels, 3x3, stride 1, pad 1. It keeps the size.
- `cNs2`: conv, 4x4, stride 2, pad 1. It exactly halves the size.
- `p`: 2x2 **max** pool, stride 2.
- `a`: 2x2 **average** pool, stride 2.

Every conv is followed by ReLU. Every network ends with **Flatten -> FC(32) + ReLU -> FC(classes)**.

The whole grid is derived from one table of strided token lists, so it cannot drift out of sync:
`P<d>` replaces every non-stem `cNs2` with `p, cN` and appends a final `p`; `A<d>` is `P<d>` with
`a` for `p`; `NS<d>`/`NA<d>` are the 4/8/16 twins of `S<d>`/`A<d>`. `archs.py` asserts all of this,
including that `P<d>` and `A<d>` agree on ReLU and pool counts for every dataset.

### ReLU placement

A convolution immediately followed by a pool emits

```
Conv -> BN -> Pool -> ReLU        (not Conv -> BN -> ReLU -> Pool)
```

For **max** pooling the two orders compute exactly the same function, because ReLU is monotone
non-decreasing and therefore commutes with max:

```
max(relu(a), relu(b)) == relu(max(a, b))
```

Applying the ReLU after the pool puts it on the 4x smaller pooled map, which cuts the ReLU count
by 4 with no change in behaviour at all. On MNIST, `P2` went from 2,384 ReLUs under the old
ordering to **568** under this one.

For **average** pooling the two orders are *not* equal (`avg(relu([-1,1])) = 0.5` while
`relu(avg([-1,1])) = 0`), so pooling first is a deliberate architectural choice here. It is what
keeps the `A<d>` ReLU counts identical to `P<d>`, which is what makes the pair comparable.

The ReLU is never dropped.

### The 36 architectures

*Tables are split by pooling kind -- **no pooling**, **average pooling**, **max pooling** -- and within each group the models are ordered **smallest first**, by ReLU count ascending. That order happens to be the same on all four datasets, so a row position means the same model everywhere.*

#### No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

| id | convs | pools | layers |
|---|---:|---:|---|
| `NS1` | 1 | 0 | Conv4/2 - FC32 - FC |
| `NS2` | 2 | 0 | Conv4/2 - Conv8/2 - FC32 - FC |
| `S1` | 1 | 0 | Conv8/2 - FC32 - FC |
| `NS3` | 3 | 0 | Conv4/2 - Conv8/2 - Conv8 - FC32 - FC |
| `S2` | 2 | 0 | Conv8/2 - Conv16/2 - FC32 - FC |
| `NS4` | 4 | 0 | Conv4/2 - Conv4 - Conv8/2 - Conv8 - FC32 - FC |
| `NS5` | 5 | 0 | Conv4/2 - Conv4 - Conv8/2 - Conv8 - Conv16/2 - FC32 - FC |
| `NS6` | 6 | 0 | Conv4/2 - Conv4 - Conv8/2 - Conv8 - Conv16/2 - Conv16 - FC32 - FC |
| `S3` | 3 | 0 | Conv8/2 - Conv16/2 - Conv16 - FC32 - FC |
| `S4` | 4 | 0 | Conv8/2 - Conv8 - Conv16/2 - Conv16 - FC32 - FC |
| `S5` | 5 | 0 | Conv8/2 - Conv8 - Conv16/2 - Conv16 - Conv32/2 - FC32 - FC |
| `S6` | 6 | 0 | Conv8/2 - Conv8 - Conv16/2 - Conv16 - Conv32/2 - Conv32 - FC32 - FC |

#### Average pooling

`A<d>` / `NA<d>` -- 2x2 average pooling. Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

| id | convs | pools | layers |
|---|---:|---:|---|
| `NA1` | 1 | 1 | Conv4/2 - AvgPool - FC32 - FC |
| `NA2` | 2 | 2 | Conv4/2 - AvgPool - Conv8 - AvgPool - FC32 - FC |
| `A1` | 1 | 1 | Conv8/2 - AvgPool - FC32 - FC |
| `A2` | 2 | 2 | Conv8/2 - AvgPool - Conv16 - AvgPool - FC32 - FC |
| `NA3` | 3 | 2 | Conv4/2 - AvgPool - Conv8 - Conv8 - AvgPool - FC32 - FC |
| `A3` | 3 | 2 | Conv8/2 - AvgPool - Conv16 - Conv16 - AvgPool - FC32 - FC |
| `NA4` | 4 | 2 | Conv4/2 - Conv4 - AvgPool - Conv8 - Conv8 - AvgPool - FC32 - FC |
| `NA5` | 5 | 3 | Conv4/2 - Conv4 - AvgPool - Conv8 - Conv8 - AvgPool - Conv16 - AvgPool - FC32 - FC |
| `NA6` | 6 | 3 | Conv4/2 - Conv4 - AvgPool - Conv8 - Conv8 - AvgPool - Conv16 - Conv16 - AvgPool - FC32 - FC |
| `A4` | 4 | 2 | Conv8/2 - Conv8 - AvgPool - Conv16 - Conv16 - AvgPool - FC32 - FC |
| `A5` | 5 | 3 | Conv8/2 - Conv8 - AvgPool - Conv16 - Conv16 - AvgPool - Conv32 - AvgPool - FC32 - FC |
| `A6` | 6 | 3 | Conv8/2 - Conv8 - AvgPool - Conv16 - Conv16 - AvgPool - Conv32 - Conv32 - AvgPool - FC32 - FC |

#### Max pooling

`P<d>` / `NP<d>` -- 2x2 max pooling. Each output needs a fresh variable with `m >= a_i` plus a 4-way disjunction `\/ m = a_i`.

| id | convs | pools | layers |
|---|---:|---:|---|
| `NP1` | 1 | 1 | Conv4/2 - MaxPool - FC32 - FC |
| `NP2` | 2 | 2 | Conv4/2 - MaxPool - Conv8 - MaxPool - FC32 - FC |
| `P1` | 1 | 1 | Conv8/2 - MaxPool - FC32 - FC |
| `P2` | 2 | 2 | Conv8/2 - MaxPool - Conv16 - MaxPool - FC32 - FC |
| `NP3` | 3 | 2 | Conv4/2 - MaxPool - Conv8 - Conv8 - MaxPool - FC32 - FC |
| `P3` | 3 | 2 | Conv8/2 - MaxPool - Conv16 - Conv16 - MaxPool - FC32 - FC |
| `NP4` | 4 | 2 | Conv4/2 - Conv4 - MaxPool - Conv8 - Conv8 - MaxPool - FC32 - FC |
| `NP5` | 5 | 3 | Conv4/2 - Conv4 - MaxPool - Conv8 - Conv8 - MaxPool - Conv16 - MaxPool - FC32 - FC |
| `NP6` | 6 | 3 | Conv4/2 - Conv4 - MaxPool - Conv8 - Conv8 - MaxPool - Conv16 - Conv16 - MaxPool - FC32 - FC |
| `P4` | 4 | 2 | Conv8/2 - Conv8 - MaxPool - Conv16 - Conv16 - MaxPool - FC32 - FC |
| `P5` | 5 | 3 | Conv8/2 - Conv8 - MaxPool - Conv16 - Conv16 - MaxPool - Conv32 - MaxPool - FC32 - FC |
| `P6` | 6 | 3 | Conv8/2 - Conv8 - MaxPool - Conv16 - Conv16 - MaxPool - Conv32 - Conv32 - MaxPool - FC32 - FC |

**Branching counts.**
- "ReLUs" counts every ReLU neuron: conv feature maps (after pooling, where there is pooling)
  plus the 32 head units.
- "Pool win." counts pooling output neurons. For `P<d>` each one is a 4-way max, which spexplain
  encodes as one fresh variable with `m >= a_i` and `\/ m = a_i`. For `A<d>`/`NA<d>` each one is a
  mean, encoded as a single linear term -- **no variable and no disjunction**.

---

## 4. Results

*Tables are split by pooling kind -- **no pooling**, **average pooling**, **max pooling** -- and within each group the models are ordered **smallest first**, by ReLU count ascending. That order happens to be the same on all four datasets, so a row position means the same model everywhere.*

Test accuracy is top-1 of the **exported ONNX model** under onnxruntime on the full test split.

- *CSV acc* is measured on the 100 rows used for explanation experiments, so it is noisy (+-5%).
- *N2* is the maximum absolute logit difference between Network2 (`onnx-eval`) and onnxruntime on those rows.
- The training time includes per-epoch evaluation.

### MNIST (1x28x28, 10 classes, 15 epochs)

#### MNIST -- No pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NS1` | 1 | 0 | 816 | 0 | 25,518 | 98.11 | **98.21** | 99.97 | 96 | 3e-06 | 18 |
| `NS2` | 2 | 0 | 1,208 | 0 | 13,494 | 98.47 | **98.57** | 100.00 | 97 | 4e-06 | 22 |
| `S1` | 1 | 0 | 1,600 | 0 | 50,674 | 98.62 | **98.64** | 99.98 | 99 | 2e-06 | 16 |
| `NS3` | 3 | 0 | 1,600 | 0 | 14,078 | 98.81 | **98.84** | 100.00 | 97 | 2e-06 | 26 |
| `S2` | 2 | 0 | 2,384 | 0 | 27,650 | 99.11 | **99.06** | 100.00 | 99 | 2e-06 | 17 |
| `NS4` | 4 | 0 | 2,384 | 0 | 14,226 | 99.03 | **99.00** | 99.99 | 98 | 2e-06 | 27 |
| `NS5` | 5 | 0 | 2,528 | 0 | 8,354 | 98.97 | **98.98** | 99.99 | 100 | 2e-06 | 30 |
| `NS6` | 6 | 0 | 2,672 | 0 | 10,674 | 99.23 | **99.15** | 100.00 | 99 | 2e-06 | 32 |
| `S3` | 3 | 0 | 3,168 | 0 | 29,970 | 99.36 | **99.36** | 99.99 | 99 | 2e-06 | 20 |
| `S4` | 4 | 0 | 4,736 | 0 | 30,554 | 99.43 | **99.35** | 99.99 | 99 | 1e-06 | 27 |
| `S5` | 5 | 0 | 5,024 | 0 | 22,906 | 99.54 | **99.52** | 99.99 | 99 | 1e-06 | 28 |
| `S6` | 6 | 0 | 5,312 | 0 | 32,154 | 99.62 | **99.51** | 100.00 | 99 | 1e-06 | 32 |

#### MNIST -- Average pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NA1` | 1 | 1 | 228 | 196 | 6,702 | 96.10 | **96.43** | 99.93 | 96 | 2e-06 | 17 |
| `NA2` | 2 | 2 | 300 | 268 | 3,030 | 94.91 | **95.43** | 99.93 | 95 | 2e-06 | 21 |
| `A1` | 1 | 1 | 424 | 392 | 13,042 | 97.12 | **97.22** | 99.93 | 96 | 2e-06 | 15 |
| `A2` | 2 | 2 | 568 | 536 | 6,274 | 97.34 | **97.53** | 99.97 | 97 | 2e-06 | 22 |
| `NA3` | 3 | 2 | 692 | 268 | 3,614 | 97.40 | **97.78** | 99.97 | 97 | 2e-06 | 23 |
| `A3` | 3 | 2 | 1,352 | 536 | 8,594 | 98.68 | **98.74** | 100.00 | 99 | 1e-06 | 23 |
| `NA4` | 4 | 2 | 1,476 | 268 | 3,762 | 98.30 | **98.27** | 100.00 | 97 | 1e-06 | 27 |
| `NA5` | 5 | 3 | 1,492 | 284 | 3,138 | 97.85 | **97.99** | 99.97 | 97 | 1e-06 | 32 |
| `NA6` | 6 | 3 | 1,636 | 284 | 5,458 | 98.57 | **98.53** | 99.98 | 99 | 2e-06 | 34 |
| `A4` | 4 | 2 | 2,920 | 536 | 9,178 | 99.23 | **99.10** | 100.00 | 100 | 2e-06 | 28 |
| `A5` | 5 | 3 | 2,952 | 568 | 10,234 | 99.32 | **99.17** | 100.00 | 100 | 2e-06 | 31 |
| `A6` | 6 | 3 | 3,240 | 568 | 19,482 | 99.45 | **99.21** | 99.99 | 100 | 1e-06 | 33 |

#### MNIST -- Max pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NP1` | 1 | 1 | 228 | 196 | 6,702 | 96.78 | **96.84** | 99.95 | 96 | 2e-06 | 17 |
| `NP2` | 2 | 2 | 300 | 268 | 3,030 | 96.76 | **97.36** | 99.97 | 96 | 2e-06 | 20 |
| `P1` | 1 | 1 | 424 | 392 | 13,042 | 98.12 | **98.10** | 99.99 | 97 | 2e-06 | 17 |
| `P2` | 2 | 2 | 568 | 536 | 6,274 | 98.49 | **98.56** | 99.99 | 99 | 1e-06 | 20 |
| `NP3` | 3 | 2 | 692 | 268 | 3,614 | 97.62 | **97.86** | 99.97 | 98 | 2e-06 | 25 |
| `P3` | 3 | 2 | 1,352 | 536 | 8,594 | 98.99 | **99.00** | 99.98 | 99 | 1e-06 | 24 |
| `NP4` | 4 | 2 | 1,476 | 268 | 3,762 | 98.35 | **98.41** | 99.98 | 99 | 2e-06 | 28 |
| `NP5` | 5 | 3 | 1,492 | 284 | 3,138 | 98.42 | **98.56** | 99.99 | 98 | 2e-06 | 32 |
| `NP6` | 6 | 3 | 1,636 | 284 | 5,458 | 98.83 | **98.77** | 99.97 | 97 | 2e-06 | 34 |
| `P4` | 4 | 2 | 2,920 | 536 | 9,178 | 99.29 | **99.32** | 99.99 | 100 | 2e-06 | 27 |
| `P5` | 5 | 3 | 2,952 | 568 | 10,234 | 99.40 | **99.30** | 99.97 | 99 | 1e-06 | 29 |
| `P6` | 6 | 3 | 3,240 | 568 | 19,482 | 99.54 | **99.35** | 100.00 | 100 | 2e-06 | 33 |

### GTSRB (3x32x32, 43 classes, 50 epochs)

#### GTSRB -- No pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NS1` | 1 | 0 | 1,056 | 0 | 34,415 | 95.53 | **84.68** | 97.18 | 85 | 8e-06 | 24 |
| `NS2` | 2 | 0 | 1,568 | 0 | 18,551 | 96.78 | **86.44** | 98.15 | 86 | 6e-06 | 29 |
| `S1` | 1 | 0 | 2,080 | 0 | 67,379 | 97.77 | **86.20** | 97.70 | 89 | 6e-06 | 25 |
| `NS3` | 3 | 0 | 2,080 | 0 | 19,135 | 97.65 | **87.97** | 98.23 | 84 | 7e-06 | 38 |
| `S2` | 2 | 0 | 3,104 | 0 | 36,675 | 99.12 | **90.78** | 98.35 | 90 | 4e-06 | 29 |
| `NS4` | 4 | 0 | 3,104 | 0 | 19,283 | 98.19 | **90.37** | 98.71 | 90 | 5e-06 | 43 |
| `NS5` | 5 | 0 | 3,360 | 0 | 13,155 | 97.90 | **90.04** | 97.99 | 90 | 5e-06 | 47 |
| `NS6` | 6 | 0 | 3,616 | 0 | 15,475 | 98.27 | **89.72** | 98.09 | 94 | 6e-06 | 54 |
| `S3` | 3 | 0 | 4,128 | 0 | 38,995 | 99.58 | **92.61** | 98.57 | 92 | 4e-06 | 35 |
| `S4` | 4 | 0 | 6,176 | 0 | 39,579 | 99.80 | **94.45** | 99.24 | 95 | 4e-06 | 43 |
| `S5` | 5 | 0 | 6,688 | 0 | 31,419 | 99.77 | **94.64** | 98.81 | 95 | 4e-06 | 47 |
| `S6` | 6 | 0 | 7,200 | 0 | 40,667 | 99.88 | **95.62** | 99.26 | 95 | 3e-06 | 53 |

#### GTSRB -- Average pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NA1` | 1 | 1 | 288 | 256 | 9,839 | 86.42 | **71.70** | 93.94 | 75 | 7e-06 | 23 |
| `NA2` | 2 | 2 | 416 | 384 | 6,039 | 80.15 | **64.90** | 91.97 | 61 | 6e-06 | 34 |
| `A1` | 1 | 1 | 544 | 512 | 18,227 | 93.42 | **77.65** | 96.02 | 79 | 6e-06 | 25 |
| `A2` | 2 | 2 | 800 | 768 | 11,203 | 90.20 | **71.76** | 94.43 | 70 | 5e-06 | 30 |
| `NA3` | 3 | 2 | 928 | 384 | 6,623 | 83.91 | **69.39** | 93.44 | 74 | 7e-06 | 36 |
| `A3` | 3 | 2 | 1,824 | 768 | 13,523 | 96.20 | **83.40** | 97.02 | 85 | 5e-06 | 37 |
| `NA4` | 4 | 2 | 1,952 | 384 | 6,771 | 89.14 | **78.67** | 96.49 | 78 | 6e-06 | 42 |
| `NA5` | 5 | 3 | 2,016 | 448 | 5,891 | 87.63 | **78.42** | 96.03 | 80 | 5e-06 | 47 |
| `NA6` | 6 | 3 | 2,272 | 448 | 8,211 | 90.28 | **78.05** | 96.14 | 85 | 9e-06 | 53 |
| `A4` | 4 | 2 | 3,872 | 768 | 14,107 | 98.59 | **90.93** | 98.38 | 95 | 5e-06 | 44 |
| `A5` | 5 | 3 | 4,000 | 896 | 14,651 | 98.07 | **89.93** | 98.27 | 90 | 5e-06 | 47 |
| `A6` | 6 | 3 | 4,512 | 896 | 23,899 | 99.18 | **90.77** | 98.23 | 93 | 4e-06 | 56 |

#### GTSRB -- Max pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NP1` | 1 | 1 | 288 | 256 | 9,839 | 89.76 | **80.08** | 97.21 | 80 | 1e-05 | 25 |
| `NP2` | 2 | 2 | 416 | 384 | 6,039 | 82.71 | **70.40** | 94.20 | 74 | 1e-05 | 33 |
| `P1` | 1 | 1 | 544 | 512 | 18,227 | 94.92 | **85.52** | 98.14 | 83 | 8e-06 | 25 |
| `P2` | 2 | 2 | 800 | 768 | 11,203 | 94.26 | **84.77** | 97.92 | 88 | 8e-06 | 30 |
| `NP3` | 3 | 2 | 928 | 384 | 6,623 | 86.63 | **76.25** | 95.37 | 77 | 7e-06 | 44 |
| `P3` | 3 | 2 | 1,824 | 768 | 13,523 | 97.18 | **88.96** | 98.55 | 89 | 6e-06 | 37 |
| `NP4` | 4 | 2 | 1,952 | 384 | 6,771 | 89.59 | **81.10** | 97.23 | 83 | 7e-06 | 49 |
| `NP5` | 5 | 3 | 2,016 | 448 | 5,891 | 89.14 | **81.37** | 96.77 | 77 | 6e-06 | 49 |
| `NP6` | 6 | 3 | 2,272 | 448 | 8,211 | 91.83 | **81.39** | 96.71 | 85 | 7e-06 | 60 |
| `P4` | 4 | 2 | 3,872 | 768 | 14,107 | 98.64 | **92.31** | 98.54 | 90 | 6e-06 | 43 |
| `P5` | 5 | 3 | 4,000 | 896 | 14,651 | 98.73 | **92.30** | 98.90 | 93 | 7e-06 | 47 |
| `P6` | 6 | 3 | 4,512 | 896 | 23,899 | 99.31 | **92.37** | 98.28 | 92 | 5e-06 | 55 |

### CIFAR-10 (3x32x32, 10 classes, 80 epochs)

#### CIFAR-10 -- No pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NS1` | 1 | 0 | 1,056 | 0 | 33,326 | 55.94 | **55.11** | 94.64 | 60 | 3e-06 | 71 |
| `NS2` | 2 | 0 | 1,568 | 0 | 17,462 | 59.22 | **58.63** | 95.46 | 54 | 3e-06 | 91 |
| `S1` | 1 | 0 | 2,080 | 0 | 66,290 | 62.02 | **61.36** | 95.91 | 64 | 2e-06 | 62 |
| `NS3` | 3 | 0 | 2,080 | 0 | 18,046 | 62.75 | **61.88** | 96.16 | 68 | 3e-06 | 112 |
| `S2` | 2 | 0 | 3,104 | 0 | 35,586 | 68.79 | **67.78** | 97.16 | 66 | 2e-06 | 83 |
| `NS4` | 4 | 0 | 3,104 | 0 | 18,194 | 63.17 | **62.54** | 96.03 | 62 | 3e-06 | 121 |
| `NS5` | 5 | 0 | 3,360 | 0 | 12,066 | 66.12 | **65.47** | 96.84 | 64 | 3e-06 | 133 |
| `NS6` | 6 | 0 | 3,616 | 0 | 14,386 | 67.18 | **66.27** | 96.86 | 69 | 4e-06 | 150 |
| `S3` | 3 | 0 | 4,128 | 0 | 37,906 | 72.69 | **70.54** | 97.72 | 78 | 3e-06 | 96 |
| `S4` | 4 | 0 | 6,176 | 0 | 38,490 | 74.73 | **73.39** | 98.21 | 76 | 2e-06 | 122 |
| `S5` | 5 | 0 | 6,688 | 0 | 30,330 | 76.87 | **74.79** | 98.28 | 78 | 2e-06 | 130 |
| `S6` | 6 | 0 | 7,200 | 0 | 39,578 | 78.73 | **76.30** | 98.40 | 82 | 3e-06 | 150 |

#### CIFAR-10 -- Average pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NA1` | 1 | 1 | 288 | 256 | 8,750 | 50.15 | **50.14** | 93.13 | 54 | 3e-06 | 65 |
| `NA2` | 2 | 2 | 416 | 384 | 4,950 | 51.63 | **51.34** | 93.59 | 50 | 2e-06 | 90 |
| `A1` | 1 | 1 | 544 | 512 | 17,138 | 55.29 | **54.44** | 93.92 | 59 | 2e-06 | 70 |
| `A2` | 2 | 2 | 800 | 768 | 10,114 | 57.83 | **56.45** | 95.21 | 54 | 2e-06 | 87 |
| `NA3` | 3 | 2 | 928 | 384 | 5,534 | 55.79 | **55.16** | 94.81 | 64 | 2e-06 | 104 |
| `A3` | 3 | 2 | 1,824 | 768 | 12,434 | 63.53 | **62.34** | 96.55 | 70 | 2e-06 | 105 |
| `NA4` | 4 | 2 | 1,952 | 384 | 5,682 | 57.86 | **57.78** | 95.42 | 66 | 3e-06 | 129 |
| `NA5` | 5 | 3 | 2,016 | 448 | 4,802 | 58.57 | **57.74** | 95.33 | 61 | 3e-06 | 137 |
| `NA6` | 6 | 3 | 2,272 | 448 | 7,122 | 61.40 | **60.87** | 95.66 | 63 | 3e-06 | 153 |
| `A4` | 4 | 2 | 3,872 | 768 | 13,018 | 70.98 | **70.02** | 97.35 | 79 | 2e-06 | 124 |
| `A5` | 5 | 3 | 4,000 | 896 | 13,562 | 71.07 | **69.72** | 97.60 | 74 | 2e-06 | 138 |
| `A6` | 6 | 3 | 4,512 | 896 | 22,810 | 74.97 | **73.41** | 97.99 | 70 | 2e-06 | 155 |

#### CIFAR-10 -- Max pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NP1` | 1 | 1 | 288 | 256 | 8,750 | 53.99 | **54.19** | 94.36 | 61 | 2e-06 | 75 |
| `NP2` | 2 | 2 | 416 | 384 | 4,950 | 56.15 | **55.50** | 94.73 | 51 | 4e-06 | 98 |
| `P1` | 1 | 1 | 544 | 512 | 17,138 | 61.90 | **61.29** | 95.97 | 62 | 3e-06 | 67 |
| `P2` | 2 | 2 | 800 | 768 | 10,114 | 66.58 | **65.55** | 96.98 | 70 | 3e-06 | 91 |
| `NP3` | 3 | 2 | 928 | 384 | 5,534 | 58.05 | **57.62** | 95.05 | 55 | 3e-06 | 108 |
| `P3` | 3 | 2 | 1,824 | 768 | 12,434 | 69.53 | **69.02** | 97.44 | 65 | 3e-06 | 105 |
| `NP4` | 4 | 2 | 1,952 | 384 | 5,682 | 58.94 | **58.90** | 95.63 | 60 | 4e-06 | 128 |
| `NP5` | 5 | 3 | 2,016 | 448 | 4,802 | 60.06 | **59.17** | 95.78 | 60 | 3e-06 | 144 |
| `NP6` | 6 | 3 | 2,272 | 448 | 7,122 | 61.62 | **61.04** | 95.82 | 61 | 4e-06 | 163 |
| `P4` | 4 | 2 | 3,872 | 768 | 13,018 | 71.41 | **71.07** | 97.83 | 72 | 3e-06 | 123 |
| `P5` | 5 | 3 | 4,000 | 896 | 13,562 | 73.71 | **71.77** | 98.10 | 80 | 4e-06 | 136 |
| `P6` | 6 | 3 | 4,512 | 896 | 22,810 | 75.87 | **74.26** | 98.10 | 77 | 4e-06 | 153 |

### Imagenette-64 (3x64x64, 10 classes, 60 epochs)

#### Imagenette-64 -- No pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NS1` | 1 | 0 | 4,128 | 0 | 131,630 | 62.41 | **55.59** | 91.16 | 54 | 3e-06 | 24 |
| `NS2` | 2 | 0 | 6,176 | 0 | 66,614 | 70.94 | **64.03** | 93.50 | 73 | 2e-06 | 27 |
| `S1` | 1 | 0 | 8,224 | 0 | 262,898 | 70.98 | **62.39** | 93.40 | 60 | 2e-06 | 28 |
| `NS3` | 3 | 0 | 8,224 | 0 | 67,198 | 73.10 | **65.66** | 94.34 | 66 | 2e-06 | 29 |
| `S2` | 2 | 0 | 12,320 | 0 | 133,890 | 79.90 | **67.62** | 95.31 | 75 | 2e-06 | 32 |
| `NS4` | 4 | 0 | 12,320 | 0 | 67,346 | 75.06 | **67.31** | 95.36 | 74 | 3e-06 | 34 |
| `NS5` | 5 | 0 | 13,344 | 0 | 36,642 | 76.24 | **68.61** | 95.34 | 69 | 4e-06 | 248 |
| `NS6` | 6 | 0 | 14,368 | 0 | 38,962 | 78.46 | **71.26** | 95.49 | 73 | 3e-06 | 282 |
| `S3` | 3 | 0 | 16,416 | 0 | 136,210 | 83.80 | **71.95** | 95.82 | 76 | 2e-06 | 35 |
| `S4` | 4 | 0 | 24,608 | 0 | 136,794 | 83.42 | **72.87** | 95.57 | 75 | 3e-06 | 43 |
| `S5` | 5 | 0 | 26,656 | 0 | 79,482 | 86.57 | **74.55** | 96.99 | 73 | 2e-06 | 354 |
| `S6` | 6 | 0 | 28,704 | 0 | 88,730 | 89.05 | **75.90** | 96.76 | 81 | 3e-06 | 386 |

#### Imagenette-64 -- Average pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NA1` | 1 | 1 | 1,056 | 1,024 | 33,326 | 56.47 | **50.01** | 88.84 | 54 | 2e-06 | 24 |
| `NA2` | 2 | 2 | 1,568 | 1,536 | 17,238 | 63.49 | **55.97** | 90.70 | 51 | 2e-06 | 26 |
| `A1` | 1 | 1 | 2,080 | 2,048 | 66,290 | 68.15 | **57.86** | 91.16 | 60 | 2e-06 | 25 |
| `A2` | 2 | 2 | 3,104 | 3,072 | 34,690 | 75.25 | **63.75** | 93.96 | 66 | 2e-06 | 29 |
| `NA3` | 3 | 2 | 3,616 | 1,536 | 17,822 | 69.48 | **63.29** | 93.68 | 64 | 3e-06 | 28 |
| `A3` | 3 | 2 | 7,200 | 3,072 | 37,010 | 78.47 | **67.31** | 94.39 | 64 | 2e-06 | 33 |
| `NA4` | 4 | 2 | 7,712 | 1,536 | 17,970 | 68.91 | **62.80** | 93.35 | 70 | 3e-06 | 224 |
| `NA5` | 5 | 3 | 7,968 | 1,792 | 10,946 | 70.20 | **65.20** | 94.70 | 57 | 3e-06 | 244 |
| `NA6` | 6 | 3 | 8,992 | 1,792 | 13,266 | 72.46 | **66.90** | 95.01 | 66 | 2e-06 | 254 |
| `A4` | 4 | 2 | 15,392 | 3,072 | 37,594 | 80.79 | **70.22** | 95.57 | 64 | 2e-06 | 42 |
| `A5` | 5 | 3 | 15,904 | 3,584 | 25,850 | 81.61 | **73.35** | 96.10 | 72 | 2e-06 | 313 |
| `A6` | 6 | 3 | 17,952 | 3,584 | 35,098 | 84.41 | **74.06** | 96.10 | 76 | 3e-06 | 353 |

#### Imagenette-64 -- Max pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NP1` | 1 | 1 | 1,056 | 1,024 | 33,326 | 64.68 | **57.73** | 91.92 | 63 | 4e-06 | 26 |
| `NP2` | 2 | 2 | 1,568 | 1,536 | 17,238 | 70.40 | **64.10** | 94.19 | 68 | 3e-06 | 28 |
| `P1` | 1 | 1 | 2,080 | 2,048 | 66,290 | 71.56 | **64.89** | 93.50 | 65 | 7e-06 | 26 |
| `P2` | 2 | 2 | 3,104 | 3,072 | 34,690 | 77.91 | **70.19** | 94.93 | 78 | 2e-06 | 29 |
| `NP3` | 3 | 2 | 3,616 | 1,536 | 17,822 | 71.42 | **65.76** | 94.75 | 75 | 4e-06 | 30 |
| `P3` | 3 | 2 | 7,200 | 3,072 | 37,010 | 80.45 | **71.39** | 95.36 | 77 | 3e-06 | 35 |
| `NP4` | 4 | 2 | 7,712 | 1,536 | 17,970 | 71.21 | **66.93** | 94.93 | 72 | 4e-06 | 35 |
| `NP5` | 5 | 3 | 7,968 | 1,792 | 10,946 | 72.47 | **67.87** | 94.60 | 70 | 3e-06 | 37 |
| `NP6` | 6 | 3 | 8,992 | 1,792 | 13,266 | 74.32 | **68.71** | 95.16 | 76 | 4e-06 | 38 |
| `P4` | 4 | 2 | 15,392 | 3,072 | 37,594 | 80.85 | **73.45** | 96.00 | 72 | 4e-06 | 40 |
| `P5` | 5 | 3 | 15,904 | 3,584 | 25,850 | 82.29 | **74.50** | 97.02 | 72 | 5e-06 | 346 |
| `P6` | 6 | 3 | 17,952 | 3,584 | 35,098 | 85.12 | **76.25** | 96.99 | 76 | 3e-06 | 335 |

### Observations

- **Depth pays off on every dataset.** Accuracy grows almost monotonically from d=1 to d=6 in
  every variant, so the family is meaningful at every size.
- **Max pool beats average pool, by a margin that depends on the dataset.** At identical shape,
  ReLU count and parameter count, for both the wide and the narrow pair:

  | dataset | `P<d>` - `A<d>` test accuracy |
  |---|---|
  | MNIST | +0.13 to +1.03 points |
  | GTSRB | +1.38 to +13.01 points |
  | CIFAR-10 | +0.85 to +9.10 points |
  | Imagenette-64 | +1.15 to +7.03 points |

  The gap is smallest on MNIST and largest on GTSRB, where sharp
  pictogram edges are exactly what averaging destroys. This is the accuracy price of removing
  max-pool branching, and it is the number to weigh against the verification saving.
- **The narrow family is genuinely cheaper.** `NS`/`NP`/`NA` roughly halve the ReLU count of
  their wide twins and cut parameters by 2-6x, at a few points of accuracy.
- **Why parameter counts fall with depth.** The flattened feature size before FC(32) shrinks as
  depth grows, so the shallow models carry most of their weight in one large FC layer.

---

## 5. Average pooling support in spexplain

`AveragePool` was absent from `OnnxParser`'s dispatch and would raise
`Unimplemented! Unsupported layer type`. Three pieces were added:

| file | change |
|---|---|
| `src/spexplain/network/AvgPoolLayer.{h,cpp}` | new `NetworkLayer` subclass; `windowIndices()` and `windowDivisors()` mirror `MaxPoolLayer`, `computeLayerOutput` averages each window. Honours `count_include_pad`. |
| `src/spexplain/network/OnnxParser.cpp` | `AveragePool` dispatch case + `avgPoolEquations`, mirroring `maxPoolEquations` (auto_pad, ceil_mode, dilations, kernel_shape, pads, strides, count_include_pad). |
| `src/verifiers/opensmt/OpenSMTVerifier2.cpp` | `avgPoolRows()` builds one affine row per output (`weight = 1/n` over the window) and the encoder returns an `AffineEncoder`. |

Because averaging is affine it reuses the same encoder path as `fc`/`cnn`/`add`: **no fresh
variables, no disjunctions, no case splits.**

The implementation was checked end to end before any model was trained:

1. **Parse + evaluate.** A hand-built Conv/AveragePool/ReLU/Gemm ONNX parses, and Network2
   reproduces onnxruntime's logits to **6e-09**.
2. **Encode.** `encode-onnx` succeeds. The dumped query contains exactly **one** `or` -- the
   classification constraint -- confirming avg pooling contributes zero disjunctions. (A max-pool
   model of the same shape contains one per pooled neuron.)
3. **Explain.** `explain-onnx` produces explanations for every sample.
4. **Soundness.** `check_phi_soundness.sh` reports **6/6 sound, 0 unsound**.
5. **Encoding vs. network.** Pinning every input to a sample's values and asking each `psi_c`
   query reproduces the network's argmax, for samples of all three classes.

`python_scripts/contrastive-guide/onnx_torch_model.py` also needed an `AveragePool` case, since
the neuron-activation guide interprets the ONNX graph directly. It matches onnxruntime to
**<= 2.2e-06** on avg-pool, narrow and max-pool models.

---

## 6. Validation

1. **Op whitelist.** Each ONNX graph contains only {Conv, Relu, MaxPool, AveragePool, Flatten,
   Gemm}. Passed for 120/120.
2. **PyTorch vs. onnxruntime.** Maximum relative logit difference on 32 random inputs is
   **5.2e-05**, against a tolerance of 1e-4. This includes the BatchNorm folding.
   - Full-test-set accuracy of the ONNX model equals the PyTorch model's exactly for 119 of 120
     models. The exception is `cifar10/A3`, where **one image out of 10,000** flips on a float32
     tie-break (62.34% vs 62.35%). The logits agree to 1.3e-05; this is rounding, not a bug.
3. **spexplain Network2 vs. onnxruntime.** `build/onnx-eval` was run on the 100 experiment rows
   per model. The max absolute logit difference is **1.09e-05**, with **0 argmax mismatches**,
   for **144/144** models. Results are in `data/models/<ds>/cnn-bench/validation.csv`.
   `validate_spexplain.py` now *merges* into `validation.csv` instead of overwriting it, so a
   filtered run (`--only`) refreshes just its own rows rather than dropping the others.
4. **Average-pool encoding.** See section 5.

---

## 7. Data notes

The experiment CSVs are now frozen and there is exactly one per dataset:

| dataset | CSV |
|---|---|
| MNIST | `data/datasets/mnist/mnist_s100_scaled.csv` |
| GTSRB | `data/datasets/gtsrb/gtsrb_s100_scaled.csv` |
| CIFAR-10 | `data/datasets/cifar/cifar_s100_scaled.csv` |
| Imagenette | `data/datasets/imagenette/imagenette_test100.csv` |

- `make_sample_csv.py` now holds this mapping in `SAMPLE_CSV` and **refuses to regenerate** the
  three legacy files, so the samples stay identical to the earlier FC experiments.
- **The `data/datasets/cifar10/` directory was removed.** CIFAR-10 uses `data/datasets/cifar/`
  with the `cifar_` prefix everywhere, including `tacas27/CIF-CNN.sh` and
  `make_neuron_activations.sh`.
- The unused `mnist_test100.csv` and `gtsrb_test100.csv` were deleted.
- **`mnist_s100_scaled.csv` and `cifar_s100_scaled.csv` hold training images** (#0-#99).
  Explanations computed on them are on images the models were trained on.
  `gtsrb_s100_scaled.csv` does come from the test split.
- `mnist_quick.csv` and `mnist_short.csv` were kept: the legacy
  `train_mnist_cnn_family.py` and `data/scripts/README.md` still reference them.
- **Input domain:** all models take `[0,1]` pixels, the default in spexplain's ONNX path.

---

## 8. Training details

**Recipe** (identical for every model of a dataset):
- AdamW, lr 2e-3, weight decay 5e-4, OneCycle schedule, batch 128.
- Cross-entropy with label smoothing 0.05.
- BatchNorm after every conv; dropout 0.2 before the classifier.
- Seed 0. The **final-epoch model is kept**: the test split is never used for model selection.
- Data is held on the MPS device as uint8, with augmentation vectorised on the device:

  | dataset | augmentation |
  |---|---|
  | MNIST | shift +-2 px |
  | GTSRB | shift +-2 px + per-image brightness/contrast x[0.5, 1.5], no flip |
  | CIFAR-10 | shift +-4 px + horizontal flip |
  | Imagenette | random 64-crop of a 72x72 image + flip |

**Cost of the full run** (36 models per dataset, single pass, no retries):

| dataset | training time |
|---|---|
| MNIST | 15 min |
| GTSRB | 23 min |
| CIFAR-10 | 67 min |
| Imagenette-64 | 68 min |

The first 30 variants per dataset took about 2h36m of wall clock; the `NP` row added roughly
another 20 minutes. Network2 validation and the neuron-activation generation each take about
half an hour for the full family.

---

## 9. Files

| path | content |
|---|---|
| `python_scripts/cnn_bench/` | all scripts and their README |
| `python_scripts/cnn_bench/arch_tables.py` | **new**: generates the per-layer tables in section 11 |
| `data/models/<ds>/cnn-bench/<id>.onnx` | the 144 spexplain-ready models |
| `data/models/<ds>/cnn-bench/<id>.pth` | training checkpoints (with BN) and metadata |
| `data/models/<ds>/cnn-bench/manifest.csv` | architecture, counts, accuracies and checks per model |
| `data/models/<ds>/cnn-bench/validation.csv` | Network2 vs. onnxruntime results |
| `data/models/CNN_BENCH_TABLES.md`, `cnn_bench_table.tex` | generated per-dataset tables |
| `data/neuron_activations/<ds>/cnn-bench/<id>.txt` | contrastive-guide activation files (144 files) |
| `src/spexplain/network/AvgPoolLayer.{h,cpp}` | **new**: average-pool layer |
| `data/scripts/tacas27/common-cnn`, `{MN,GTS,CIF,IMN}-CNN.sh` | cluster run scripts, `MODEL_IDS` updated to the 36-model family |

---

## 10. Reproduce

```bash
PY=~/miniconda3/envs/rex/bin/python
make                                          # AvgPool support must be compiled in
$PY python_scripts/cnn_bench/archs.py --list  # ReLU/pool/param counts, and the family assertions
python_scripts/cnn_bench/run_all.sh           # train -> CSVs -> Network2 validation -> tables
OVERWRITE=1 python_scripts/cnn_bench/make_neuron_activations.sh
$PY python_scripts/cnn_bench/arch_tables.py -o /tmp/arch_tables.md
```

MPS kernels are not bit-deterministic, so retraining reproduces the accuracies to within a few
tenths of a point, not exactly. The ONNX files in the repo are the ones this report describes.

---

## 11. Architecture tables

One table per model, read directly from the exported `.onnx` graphs with shape inference, so the
shapes are the ones `OnnxParser` actually sees. Shapes are `H x W x C` for feature maps and a
plain width for flat vectors; the batch axis is dropped, because `Network2` strips it and a flat
CSV row maps onto `C*H*W` features in CHW row-major order.

Each `Relu` node of the graph is folded into the **Activation** column of the layer that feeds
it. So a convolution whose activation cell reads `-` is one whose ReLU sits *after* the following
pooling layer -- the placement described in section 3.

### MNIST

Input `28 x 28 x 1`, 10 classes.

#### No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 1: Architecture of the MNIST `NS1` model.** *narrow, strided; tokens `c4s2`; 816 ReLUs, 0 pool neurons, 25,518 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Flatten | 14 x 14 x 4 | 784 | - |
| Fully Connected | 784 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 2: Architecture of the MNIST `NS2` model.** *narrow, strided; tokens `c4s2 c8s2`; 1,208 ReLUs, 0 pool neurons, 13,494 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU |
| Flatten | 7 x 7 x 8 | 392 | - |
| Fully Connected | 392 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 3: Architecture of the MNIST `S1` model.** *wide, strided; tokens `c8s2`; 1,600 ReLUs, 0 pool neurons, 50,674 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Flatten | 14 x 14 x 8 | 1568 | - |
| Fully Connected | 1568 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 4: Architecture of the MNIST `NS3` model.** *narrow, strided; tokens `c4s2 c8s2 c8`; 1,600 ReLUs, 0 pool neurons, 14,078 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU |
| Flatten | 7 x 7 x 8 | 392 | - |
| Fully Connected | 392 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 5: Architecture of the MNIST `S2` model.** *wide, strided; tokens `c8s2 c16s2`; 2,384 ReLUs, 0 pool neurons, 27,650 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU |
| Flatten | 7 x 7 x 16 | 784 | - |
| Fully Connected | 784 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 6: Architecture of the MNIST `NS4` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8`; 2,384 ReLUs, 0 pool neurons, 14,226 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU |
| Flatten | 7 x 7 x 8 | 392 | - |
| Fully Connected | 392 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 7: Architecture of the MNIST `NS5` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2`; 2,528 ReLUs, 0 pool neurons, 8,354 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 3 x 3 x 16 | ReLU |
| Flatten | 3 x 3 x 16 | 144 | - |
| Fully Connected | 144 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 8: Architecture of the MNIST `NS6` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2 c16`; 2,672 ReLUs, 0 pool neurons, 10,674 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 3 x 3 x 16 | ReLU |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 16 | ReLU |
| Flatten | 3 x 3 x 16 | 144 | - |
| Fully Connected | 144 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 9: Architecture of the MNIST `S3` model.** *wide, strided; tokens `c8s2 c16s2 c16`; 3,168 ReLUs, 0 pool neurons, 29,970 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU |
| Flatten | 7 x 7 x 16 | 784 | - |
| Fully Connected | 784 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 10: Architecture of the MNIST `S4` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16`; 4,736 ReLUs, 0 pool neurons, 30,554 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU |
| Flatten | 7 x 7 x 16 | 784 | - |
| Fully Connected | 784 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 11: Architecture of the MNIST `S5` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2`; 5,024 ReLUs, 0 pool neurons, 22,906 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 3 x 3 x 32 | ReLU |
| Flatten | 3 x 3 x 32 | 288 | - |
| Fully Connected | 288 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 12: Architecture of the MNIST `S6` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2 c32`; 5,312 ReLUs, 0 pool neurons, 32,154 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 3 x 3 x 32 | ReLU |
| Convolution 2D | 3 x 3 x 32 | 3 x 3 x 32 | ReLU |
| Flatten | 3 x 3 x 32 | 288 | - |
| Fully Connected | 288 | 32 | ReLU |
| Output | 32 | 10 | - |

#### Average pooling

`A<d>` / `NA<d>` -- 2x2 average pooling. Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 13: Architecture of the MNIST `NA1` model.** *narrow, average pool; tokens `c4s2 a`; 228 ReLUs, 196 pool neurons, 6,702 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | - |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Flatten | 7 x 7 x 4 | 196 | - |
| Fully Connected | 196 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 14: Architecture of the MNIST `NA2` model.** *narrow, average pool; tokens `c4s2 a c8 a`; 300 ReLUs, 268 pool neurons, 3,030 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | - |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | - |
| Average Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Flatten | 3 x 3 x 8 | 72 | - |
| Fully Connected | 72 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 15: Architecture of the MNIST `A1` model.** *wide, average pool; tokens `c8s2 a`; 424 ReLUs, 392 pool neurons, 13,042 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | - |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Flatten | 7 x 7 x 8 | 392 | - |
| Fully Connected | 392 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 16: Architecture of the MNIST `A2` model.** *wide, average pool; tokens `c8s2 a c16 a`; 568 ReLUs, 536 pool neurons, 6,274 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | - |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | - |
| Average Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Flatten | 3 x 3 x 16 | 144 | - |
| Fully Connected | 144 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 17: Architecture of the MNIST `NA3` model.** *narrow, average pool; tokens `c4s2 a c8 c8 a`; 692 ReLUs, 268 pool neurons, 3,614 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | - |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | - |
| Average Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Flatten | 3 x 3 x 8 | 72 | - |
| Fully Connected | 72 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 18: Architecture of the MNIST `A3` model.** *wide, average pool; tokens `c8s2 a c16 c16 a`; 1,352 ReLUs, 536 pool neurons, 8,594 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | - |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | - |
| Average Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Flatten | 3 x 3 x 16 | 144 | - |
| Fully Connected | 144 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 19: Architecture of the MNIST `NA4` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a`; 1,476 ReLUs, 268 pool neurons, 3,762 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | - |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | - |
| Average Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Flatten | 3 x 3 x 8 | 72 | - |
| Fully Connected | 72 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 20: Architecture of the MNIST `NA5` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a c16 a`; 1,492 ReLUs, 284 pool neurons, 3,138 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | - |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | - |
| Average Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Convolution 2D | 3 x 3 x 8 | 3 x 3 x 16 | - |
| Average Pooling 2D | 3 x 3 x 16 | 1 x 1 x 16 | ReLU |
| Flatten | 1 x 1 x 16 | 16 | - |
| Fully Connected | 16 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 21: Architecture of the MNIST `NA6` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a c16 c16 a`; 1,636 ReLUs, 284 pool neurons, 5,458 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | - |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | - |
| Average Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Convolution 2D | 3 x 3 x 8 | 3 x 3 x 16 | ReLU |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 16 | - |
| Average Pooling 2D | 3 x 3 x 16 | 1 x 1 x 16 | ReLU |
| Flatten | 1 x 1 x 16 | 16 | - |
| Fully Connected | 16 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 22: Architecture of the MNIST `A4` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a`; 2,920 ReLUs, 536 pool neurons, 9,178 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | - |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | - |
| Average Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Flatten | 3 x 3 x 16 | 144 | - |
| Fully Connected | 144 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 23: Architecture of the MNIST `A5` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a c32 a`; 2,952 ReLUs, 568 pool neurons, 10,234 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | - |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | - |
| Average Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 32 | - |
| Average Pooling 2D | 3 x 3 x 32 | 1 x 1 x 32 | ReLU |
| Flatten | 1 x 1 x 32 | 32 | - |
| Fully Connected | 32 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 24: Architecture of the MNIST `A6` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a c32 c32 a`; 3,240 ReLUs, 568 pool neurons, 19,482 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | - |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | - |
| Average Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 32 | ReLU |
| Convolution 2D | 3 x 3 x 32 | 3 x 3 x 32 | - |
| Average Pooling 2D | 3 x 3 x 32 | 1 x 1 x 32 | ReLU |
| Flatten | 1 x 1 x 32 | 32 | - |
| Fully Connected | 32 | 32 | ReLU |
| Output | 32 | 10 | - |

#### Max pooling

`P<d>` / `NP<d>` -- 2x2 max pooling. Each output needs a fresh variable with `m >= a_i` plus a 4-way disjunction `\/ m = a_i`.

**Table 25: Architecture of the MNIST `NP1` model.** *narrow, max pool; tokens `c4s2 p`; 228 ReLUs, 196 pool neurons, 6,702 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | - |
| Max Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Flatten | 7 x 7 x 4 | 196 | - |
| Fully Connected | 196 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 26: Architecture of the MNIST `NP2` model.** *narrow, max pool; tokens `c4s2 p c8 p`; 300 ReLUs, 268 pool neurons, 3,030 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | - |
| Max Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | - |
| Max Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Flatten | 3 x 3 x 8 | 72 | - |
| Fully Connected | 72 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 27: Architecture of the MNIST `P1` model.** *wide, max pool; tokens `c8s2 p`; 424 ReLUs, 392 pool neurons, 13,042 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | - |
| Max Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Flatten | 7 x 7 x 8 | 392 | - |
| Fully Connected | 392 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 28: Architecture of the MNIST `P2` model.** *wide, max pool; tokens `c8s2 p c16 p`; 568 ReLUs, 536 pool neurons, 6,274 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | - |
| Max Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | - |
| Max Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Flatten | 3 x 3 x 16 | 144 | - |
| Fully Connected | 144 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 29: Architecture of the MNIST `NP3` model.** *narrow, max pool; tokens `c4s2 p c8 c8 p`; 692 ReLUs, 268 pool neurons, 3,614 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | - |
| Max Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | - |
| Max Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Flatten | 3 x 3 x 8 | 72 | - |
| Fully Connected | 72 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 30: Architecture of the MNIST `P3` model.** *wide, max pool; tokens `c8s2 p c16 c16 p`; 1,352 ReLUs, 536 pool neurons, 8,594 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | - |
| Max Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | - |
| Max Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Flatten | 3 x 3 x 16 | 144 | - |
| Fully Connected | 144 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 31: Architecture of the MNIST `NP4` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p`; 1,476 ReLUs, 268 pool neurons, 3,762 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | - |
| Max Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | - |
| Max Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Flatten | 3 x 3 x 8 | 72 | - |
| Fully Connected | 72 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 32: Architecture of the MNIST `NP5` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p c16 p`; 1,492 ReLUs, 284 pool neurons, 3,138 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | - |
| Max Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | - |
| Max Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Convolution 2D | 3 x 3 x 8 | 3 x 3 x 16 | - |
| Max Pooling 2D | 3 x 3 x 16 | 1 x 1 x 16 | ReLU |
| Flatten | 1 x 1 x 16 | 16 | - |
| Fully Connected | 16 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 33: Architecture of the MNIST `NP6` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p c16 c16 p`; 1,636 ReLUs, 284 pool neurons, 5,458 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | - |
| Max Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | ReLU |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | - |
| Max Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | ReLU |
| Convolution 2D | 3 x 3 x 8 | 3 x 3 x 16 | ReLU |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 16 | - |
| Max Pooling 2D | 3 x 3 x 16 | 1 x 1 x 16 | ReLU |
| Flatten | 1 x 1 x 16 | 16 | - |
| Fully Connected | 16 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 34: Architecture of the MNIST `P4` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p`; 2,920 ReLUs, 536 pool neurons, 9,178 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | - |
| Max Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | - |
| Max Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Flatten | 3 x 3 x 16 | 144 | - |
| Fully Connected | 144 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 35: Architecture of the MNIST `P5` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p c32 p`; 2,952 ReLUs, 568 pool neurons, 10,234 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | - |
| Max Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | - |
| Max Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 32 | - |
| Max Pooling 2D | 3 x 3 x 32 | 1 x 1 x 32 | ReLU |
| Flatten | 1 x 1 x 32 | 32 | - |
| Fully Connected | 32 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 36: Architecture of the MNIST `P6` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p c32 c32 p`; 3,240 ReLUs, 568 pool neurons, 19,482 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | - |
| Max Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | ReLU |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | - |
| Max Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | ReLU |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 32 | ReLU |
| Convolution 2D | 3 x 3 x 32 | 3 x 3 x 32 | - |
| Max Pooling 2D | 3 x 3 x 32 | 1 x 1 x 32 | ReLU |
| Flatten | 1 x 1 x 32 | 32 | - |
| Fully Connected | 32 | 32 | ReLU |
| Output | 32 | 10 | - |

### GTSRB

Input `32 x 32 x 3`, 43 classes.

#### No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 37: Architecture of the GTSRB `NS1` model.** *narrow, strided; tokens `c4s2`; 1,056 ReLUs, 0 pool neurons, 34,415 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Flatten | 16 x 16 x 4 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 38: Architecture of the GTSRB `NS2` model.** *narrow, strided; tokens `c4s2 c8s2`; 1,568 ReLUs, 0 pool neurons, 18,551 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 39: Architecture of the GTSRB `S1` model.** *wide, strided; tokens `c8s2`; 2,080 ReLUs, 0 pool neurons, 67,379 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Flatten | 16 x 16 x 8 | 2048 | - |
| Fully Connected | 2048 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 40: Architecture of the GTSRB `NS3` model.** *narrow, strided; tokens `c4s2 c8s2 c8`; 2,080 ReLUs, 0 pool neurons, 19,135 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 41: Architecture of the GTSRB `S2` model.** *wide, strided; tokens `c8s2 c16s2`; 3,104 ReLUs, 0 pool neurons, 36,675 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 42: Architecture of the GTSRB `NS4` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8`; 3,104 ReLUs, 0 pool neurons, 19,283 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 43: Architecture of the GTSRB `NS5` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2`; 3,360 ReLUs, 0 pool neurons, 13,155 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 44: Architecture of the GTSRB `NS6` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2 c16`; 3,616 ReLUs, 0 pool neurons, 15,475 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 45: Architecture of the GTSRB `S3` model.** *wide, strided; tokens `c8s2 c16s2 c16`; 4,128 ReLUs, 0 pool neurons, 38,995 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 46: Architecture of the GTSRB `S4` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16`; 6,176 ReLUs, 0 pool neurons, 39,579 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 47: Architecture of the GTSRB `S5` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2`; 6,688 ReLUs, 0 pool neurons, 31,419 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 4 x 4 x 32 | ReLU |
| Flatten | 4 x 4 x 32 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 48: Architecture of the GTSRB `S6` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2 c32`; 7,200 ReLUs, 0 pool neurons, 40,667 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 4 x 4 x 32 | ReLU |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | ReLU |
| Flatten | 4 x 4 x 32 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 43 | - |

#### Average pooling

`A<d>` / `NA<d>` -- 2x2 average pooling. Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 49: Architecture of the GTSRB `NA1` model.** *narrow, average pool; tokens `c4s2 a`; 288 ReLUs, 256 pool neurons, 9,839 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Flatten | 8 x 8 x 4 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 50: Architecture of the GTSRB `NA2` model.** *narrow, average pool; tokens `c4s2 a c8 a`; 416 ReLUs, 384 pool neurons, 6,039 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 51: Architecture of the GTSRB `A1` model.** *wide, average pool; tokens `c8s2 a`; 544 ReLUs, 512 pool neurons, 18,227 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 52: Architecture of the GTSRB `A2` model.** *wide, average pool; tokens `c8s2 a c16 a`; 800 ReLUs, 768 pool neurons, 11,203 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 53: Architecture of the GTSRB `NA3` model.** *narrow, average pool; tokens `c4s2 a c8 c8 a`; 928 ReLUs, 384 pool neurons, 6,623 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 54: Architecture of the GTSRB `A3` model.** *wide, average pool; tokens `c8s2 a c16 c16 a`; 1,824 ReLUs, 768 pool neurons, 13,523 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 55: Architecture of the GTSRB `NA4` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a`; 1,952 ReLUs, 384 pool neurons, 6,771 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 56: Architecture of the GTSRB `NA5` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a c16 a`; 2,016 ReLUs, 448 pool neurons, 5,891 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | - |
| Average Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | ReLU |
| Flatten | 2 x 2 x 16 | 64 | - |
| Fully Connected | 64 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 57: Architecture of the GTSRB `NA6` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a c16 c16 a`; 2,272 ReLUs, 448 pool neurons, 8,211 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | - |
| Average Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | ReLU |
| Flatten | 2 x 2 x 16 | 64 | - |
| Fully Connected | 64 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 58: Architecture of the GTSRB `A4` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a`; 3,872 ReLUs, 768 pool neurons, 14,107 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 59: Architecture of the GTSRB `A5` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a c32 a`; 4,000 ReLUs, 896 pool neurons, 14,651 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | - |
| Average Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | ReLU |
| Flatten | 2 x 2 x 32 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 60: Architecture of the GTSRB `A6` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a c32 c32 a`; 4,512 ReLUs, 896 pool neurons, 23,899 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | ReLU |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | - |
| Average Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | ReLU |
| Flatten | 2 x 2 x 32 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

#### Max pooling

`P<d>` / `NP<d>` -- 2x2 max pooling. Each output needs a fresh variable with `m >= a_i` plus a 4-way disjunction `\/ m = a_i`.

**Table 61: Architecture of the GTSRB `NP1` model.** *narrow, max pool; tokens `c4s2 p`; 288 ReLUs, 256 pool neurons, 9,839 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Flatten | 8 x 8 x 4 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 62: Architecture of the GTSRB `NP2` model.** *narrow, max pool; tokens `c4s2 p c8 p`; 416 ReLUs, 384 pool neurons, 6,039 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 63: Architecture of the GTSRB `P1` model.** *wide, max pool; tokens `c8s2 p`; 544 ReLUs, 512 pool neurons, 18,227 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 64: Architecture of the GTSRB `P2` model.** *wide, max pool; tokens `c8s2 p c16 p`; 800 ReLUs, 768 pool neurons, 11,203 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 65: Architecture of the GTSRB `NP3` model.** *narrow, max pool; tokens `c4s2 p c8 c8 p`; 928 ReLUs, 384 pool neurons, 6,623 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 66: Architecture of the GTSRB `P3` model.** *wide, max pool; tokens `c8s2 p c16 c16 p`; 1,824 ReLUs, 768 pool neurons, 13,523 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 67: Architecture of the GTSRB `NP4` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p`; 1,952 ReLUs, 384 pool neurons, 6,771 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 68: Architecture of the GTSRB `NP5` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p c16 p`; 2,016 ReLUs, 448 pool neurons, 5,891 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | - |
| Max Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | ReLU |
| Flatten | 2 x 2 x 16 | 64 | - |
| Fully Connected | 64 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 69: Architecture of the GTSRB `NP6` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p c16 c16 p`; 2,272 ReLUs, 448 pool neurons, 8,211 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | - |
| Max Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | ReLU |
| Flatten | 2 x 2 x 16 | 64 | - |
| Fully Connected | 64 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 70: Architecture of the GTSRB `P4` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p`; 3,872 ReLUs, 768 pool neurons, 14,107 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 71: Architecture of the GTSRB `P5` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p c32 p`; 4,000 ReLUs, 896 pool neurons, 14,651 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | - |
| Max Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | ReLU |
| Flatten | 2 x 2 x 32 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

**Table 72: Architecture of the GTSRB `P6` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p c32 c32 p`; 4,512 ReLUs, 896 pool neurons, 23,899 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | ReLU |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | - |
| Max Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | ReLU |
| Flatten | 2 x 2 x 32 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 43 | - |

### CIFAR-10

Input `32 x 32 x 3`, 10 classes.

#### No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 73: Architecture of the CIFAR-10 `NS1` model.** *narrow, strided; tokens `c4s2`; 1,056 ReLUs, 0 pool neurons, 33,326 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Flatten | 16 x 16 x 4 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 74: Architecture of the CIFAR-10 `NS2` model.** *narrow, strided; tokens `c4s2 c8s2`; 1,568 ReLUs, 0 pool neurons, 17,462 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 75: Architecture of the CIFAR-10 `S1` model.** *wide, strided; tokens `c8s2`; 2,080 ReLUs, 0 pool neurons, 66,290 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Flatten | 16 x 16 x 8 | 2048 | - |
| Fully Connected | 2048 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 76: Architecture of the CIFAR-10 `NS3` model.** *narrow, strided; tokens `c4s2 c8s2 c8`; 2,080 ReLUs, 0 pool neurons, 18,046 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 77: Architecture of the CIFAR-10 `S2` model.** *wide, strided; tokens `c8s2 c16s2`; 3,104 ReLUs, 0 pool neurons, 35,586 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 78: Architecture of the CIFAR-10 `NS4` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8`; 3,104 ReLUs, 0 pool neurons, 18,194 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 79: Architecture of the CIFAR-10 `NS5` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2`; 3,360 ReLUs, 0 pool neurons, 12,066 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 80: Architecture of the CIFAR-10 `NS6` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2 c16`; 3,616 ReLUs, 0 pool neurons, 14,386 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 81: Architecture of the CIFAR-10 `S3` model.** *wide, strided; tokens `c8s2 c16s2 c16`; 4,128 ReLUs, 0 pool neurons, 37,906 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 82: Architecture of the CIFAR-10 `S4` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16`; 6,176 ReLUs, 0 pool neurons, 38,490 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 83: Architecture of the CIFAR-10 `S5` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2`; 6,688 ReLUs, 0 pool neurons, 30,330 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 4 x 4 x 32 | ReLU |
| Flatten | 4 x 4 x 32 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 84: Architecture of the CIFAR-10 `S6` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2 c32`; 7,200 ReLUs, 0 pool neurons, 39,578 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 4 x 4 x 32 | ReLU |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | ReLU |
| Flatten | 4 x 4 x 32 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

#### Average pooling

`A<d>` / `NA<d>` -- 2x2 average pooling. Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 85: Architecture of the CIFAR-10 `NA1` model.** *narrow, average pool; tokens `c4s2 a`; 288 ReLUs, 256 pool neurons, 8,750 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Flatten | 8 x 8 x 4 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 86: Architecture of the CIFAR-10 `NA2` model.** *narrow, average pool; tokens `c4s2 a c8 a`; 416 ReLUs, 384 pool neurons, 4,950 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 87: Architecture of the CIFAR-10 `A1` model.** *wide, average pool; tokens `c8s2 a`; 544 ReLUs, 512 pool neurons, 17,138 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 88: Architecture of the CIFAR-10 `A2` model.** *wide, average pool; tokens `c8s2 a c16 a`; 800 ReLUs, 768 pool neurons, 10,114 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 89: Architecture of the CIFAR-10 `NA3` model.** *narrow, average pool; tokens `c4s2 a c8 c8 a`; 928 ReLUs, 384 pool neurons, 5,534 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 90: Architecture of the CIFAR-10 `A3` model.** *wide, average pool; tokens `c8s2 a c16 c16 a`; 1,824 ReLUs, 768 pool neurons, 12,434 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 91: Architecture of the CIFAR-10 `NA4` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a`; 1,952 ReLUs, 384 pool neurons, 5,682 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 92: Architecture of the CIFAR-10 `NA5` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a c16 a`; 2,016 ReLUs, 448 pool neurons, 4,802 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | - |
| Average Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | ReLU |
| Flatten | 2 x 2 x 16 | 64 | - |
| Fully Connected | 64 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 93: Architecture of the CIFAR-10 `NA6` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a c16 c16 a`; 2,272 ReLUs, 448 pool neurons, 7,122 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | - |
| Average Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | ReLU |
| Flatten | 2 x 2 x 16 | 64 | - |
| Fully Connected | 64 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 94: Architecture of the CIFAR-10 `A4` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a`; 3,872 ReLUs, 768 pool neurons, 13,018 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 95: Architecture of the CIFAR-10 `A5` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a c32 a`; 4,000 ReLUs, 896 pool neurons, 13,562 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | - |
| Average Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | ReLU |
| Flatten | 2 x 2 x 32 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 96: Architecture of the CIFAR-10 `A6` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a c32 c32 a`; 4,512 ReLUs, 896 pool neurons, 22,810 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | ReLU |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | - |
| Average Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | ReLU |
| Flatten | 2 x 2 x 32 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

#### Max pooling

`P<d>` / `NP<d>` -- 2x2 max pooling. Each output needs a fresh variable with `m >= a_i` plus a 4-way disjunction `\/ m = a_i`.

**Table 97: Architecture of the CIFAR-10 `NP1` model.** *narrow, max pool; tokens `c4s2 p`; 288 ReLUs, 256 pool neurons, 8,750 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Flatten | 8 x 8 x 4 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 98: Architecture of the CIFAR-10 `NP2` model.** *narrow, max pool; tokens `c4s2 p c8 p`; 416 ReLUs, 384 pool neurons, 4,950 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 99: Architecture of the CIFAR-10 `P1` model.** *wide, max pool; tokens `c8s2 p`; 544 ReLUs, 512 pool neurons, 17,138 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 100: Architecture of the CIFAR-10 `P2` model.** *wide, max pool; tokens `c8s2 p c16 p`; 800 ReLUs, 768 pool neurons, 10,114 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 101: Architecture of the CIFAR-10 `NP3` model.** *narrow, max pool; tokens `c4s2 p c8 c8 p`; 928 ReLUs, 384 pool neurons, 5,534 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 102: Architecture of the CIFAR-10 `P3` model.** *wide, max pool; tokens `c8s2 p c16 c16 p`; 1,824 ReLUs, 768 pool neurons, 12,434 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 103: Architecture of the CIFAR-10 `NP4` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p`; 1,952 ReLUs, 384 pool neurons, 5,682 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Flatten | 4 x 4 x 8 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 104: Architecture of the CIFAR-10 `NP5` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p c16 p`; 2,016 ReLUs, 448 pool neurons, 4,802 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | - |
| Max Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | ReLU |
| Flatten | 2 x 2 x 16 | 64 | - |
| Fully Connected | 64 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 105: Architecture of the CIFAR-10 `NP6` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p c16 c16 p`; 2,272 ReLUs, 448 pool neurons, 7,122 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | - |
| Max Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | ReLU |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | - |
| Max Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | ReLU |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | - |
| Max Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | ReLU |
| Flatten | 2 x 2 x 16 | 64 | - |
| Fully Connected | 64 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 106: Architecture of the CIFAR-10 `P4` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p`; 3,872 ReLUs, 768 pool neurons, 13,018 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 107: Architecture of the CIFAR-10 `P5` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p c32 p`; 4,000 ReLUs, 896 pool neurons, 13,562 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | - |
| Max Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | ReLU |
| Flatten | 2 x 2 x 32 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 108: Architecture of the CIFAR-10 `P6` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p c32 c32 p`; 4,512 ReLUs, 896 pool neurons, 22,810 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | ReLU |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | - |
| Max Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | ReLU |
| Flatten | 2 x 2 x 32 | 128 | - |
| Fully Connected | 128 | 32 | ReLU |
| Output | 32 | 10 | - |

### Imagenette-64

Input `64 x 64 x 3`, 10 classes.

#### No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 109: Architecture of the Imagenette-64 `NS1` model.** *narrow, strided; tokens `c4s2`; 4,128 ReLUs, 0 pool neurons, 131,630 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Flatten | 32 x 32 x 4 | 4096 | - |
| Fully Connected | 4096 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 110: Architecture of the Imagenette-64 `NS2` model.** *narrow, strided; tokens `c4s2 c8s2`; 6,176 ReLUs, 0 pool neurons, 66,614 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU |
| Flatten | 16 x 16 x 8 | 2048 | - |
| Fully Connected | 2048 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 111: Architecture of the Imagenette-64 `S1` model.** *wide, strided; tokens `c8s2`; 8,224 ReLUs, 0 pool neurons, 262,898 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Flatten | 32 x 32 x 8 | 8192 | - |
| Fully Connected | 8192 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 112: Architecture of the Imagenette-64 `NS3` model.** *narrow, strided; tokens `c4s2 c8s2 c8`; 8,224 ReLUs, 0 pool neurons, 67,198 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Flatten | 16 x 16 x 8 | 2048 | - |
| Fully Connected | 2048 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 113: Architecture of the Imagenette-64 `S2` model.** *wide, strided; tokens `c8s2 c16s2`; 12,320 ReLUs, 0 pool neurons, 133,890 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU |
| Flatten | 16 x 16 x 16 | 4096 | - |
| Fully Connected | 4096 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 114: Architecture of the Imagenette-64 `NS4` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8`; 12,320 ReLUs, 0 pool neurons, 67,346 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Flatten | 16 x 16 x 8 | 2048 | - |
| Fully Connected | 2048 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 115: Architecture of the Imagenette-64 `NS5` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2`; 13,344 ReLUs, 0 pool neurons, 36,642 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 116: Architecture of the Imagenette-64 `NS6` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2 c16`; 14,368 ReLUs, 0 pool neurons, 38,962 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 117: Architecture of the Imagenette-64 `S3` model.** *wide, strided; tokens `c8s2 c16s2 c16`; 16,416 ReLUs, 0 pool neurons, 136,210 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | ReLU |
| Flatten | 16 x 16 x 16 | 4096 | - |
| Fully Connected | 4096 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 118: Architecture of the Imagenette-64 `S4` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16`; 24,608 ReLUs, 0 pool neurons, 136,794 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | ReLU |
| Flatten | 16 x 16 x 16 | 4096 | - |
| Fully Connected | 4096 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 119: Architecture of the Imagenette-64 `S5` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2`; 26,656 ReLUs, 0 pool neurons, 79,482 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 8 x 8 x 32 | ReLU |
| Flatten | 8 x 8 x 32 | 2048 | - |
| Fully Connected | 2048 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 120: Architecture of the Imagenette-64 `S6` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2 c32`; 28,704 ReLUs, 0 pool neurons, 88,730 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 8 x 8 x 32 | ReLU |
| Convolution 2D | 8 x 8 x 32 | 8 x 8 x 32 | ReLU |
| Flatten | 8 x 8 x 32 | 2048 | - |
| Fully Connected | 2048 | 32 | ReLU |
| Output | 32 | 10 | - |

#### Average pooling

`A<d>` / `NA<d>` -- 2x2 average pooling. Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 121: Architecture of the Imagenette-64 `NA1` model.** *narrow, average pool; tokens `c4s2 a`; 1,056 ReLUs, 1,024 pool neurons, 33,326 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | - |
| Average Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Flatten | 16 x 16 x 4 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 122: Architecture of the Imagenette-64 `NA2` model.** *narrow, average pool; tokens `c4s2 a c8 a`; 1,568 ReLUs, 1,536 pool neurons, 17,238 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | - |
| Average Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 123: Architecture of the Imagenette-64 `A1` model.** *wide, average pool; tokens `c8s2 a`; 2,080 ReLUs, 2,048 pool neurons, 66,290 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | - |
| Average Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Flatten | 16 x 16 x 8 | 2048 | - |
| Fully Connected | 2048 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 124: Architecture of the Imagenette-64 `A2` model.** *wide, average pool; tokens `c8s2 a c16 a`; 3,104 ReLUs, 3,072 pool neurons, 34,690 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | - |
| Average Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | - |
| Average Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 125: Architecture of the Imagenette-64 `NA3` model.** *narrow, average pool; tokens `c4s2 a c8 c8 a`; 3,616 ReLUs, 1,536 pool neurons, 17,822 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | - |
| Average Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 126: Architecture of the Imagenette-64 `A3` model.** *wide, average pool; tokens `c8s2 a c16 c16 a`; 7,200 ReLUs, 3,072 pool neurons, 37,010 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | - |
| Average Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | - |
| Average Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 127: Architecture of the Imagenette-64 `NA4` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a`; 7,712 ReLUs, 1,536 pool neurons, 17,970 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | - |
| Average Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 128: Architecture of the Imagenette-64 `NA5` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a c16 a`; 7,968 ReLUs, 1,792 pool neurons, 10,946 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | - |
| Average Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 129: Architecture of the Imagenette-64 `NA6` model.** *narrow, average pool; tokens `c4s2 c4 a c8 c8 a c16 c16 a`; 8,992 ReLUs, 1,792 pool neurons, 13,266 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | - |
| Average Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 130: Architecture of the Imagenette-64 `A4` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a`; 15,392 ReLUs, 3,072 pool neurons, 37,594 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | - |
| Average Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | - |
| Average Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 131: Architecture of the Imagenette-64 `A5` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a c32 a`; 15,904 ReLUs, 3,584 pool neurons, 25,850 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | - |
| Average Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | - |
| Average Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 32 | - |
| Average Pooling 2D | 8 x 8 x 32 | 4 x 4 x 32 | ReLU |
| Flatten | 4 x 4 x 32 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 132: Architecture of the Imagenette-64 `A6` model.** *wide, average pool; tokens `c8s2 c8 a c16 c16 a c32 c32 a`; 17,952 ReLUs, 3,584 pool neurons, 35,098 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | - |
| Average Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | - |
| Average Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 32 | ReLU |
| Convolution 2D | 8 x 8 x 32 | 8 x 8 x 32 | - |
| Average Pooling 2D | 8 x 8 x 32 | 4 x 4 x 32 | ReLU |
| Flatten | 4 x 4 x 32 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

#### Max pooling

`P<d>` / `NP<d>` -- 2x2 max pooling. Each output needs a fresh variable with `m >= a_i` plus a 4-way disjunction `\/ m = a_i`.

**Table 133: Architecture of the Imagenette-64 `NP1` model.** *narrow, max pool; tokens `c4s2 p`; 1,056 ReLUs, 1,024 pool neurons, 33,326 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | - |
| Max Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Flatten | 16 x 16 x 4 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 134: Architecture of the Imagenette-64 `NP2` model.** *narrow, max pool; tokens `c4s2 p c8 p`; 1,568 ReLUs, 1,536 pool neurons, 17,238 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | - |
| Max Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 135: Architecture of the Imagenette-64 `P1` model.** *wide, max pool; tokens `c8s2 p`; 2,080 ReLUs, 2,048 pool neurons, 66,290 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | - |
| Max Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Flatten | 16 x 16 x 8 | 2048 | - |
| Fully Connected | 2048 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 136: Architecture of the Imagenette-64 `P2` model.** *wide, max pool; tokens `c8s2 p c16 p`; 3,104 ReLUs, 3,072 pool neurons, 34,690 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | - |
| Max Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | - |
| Max Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 137: Architecture of the Imagenette-64 `NP3` model.** *narrow, max pool; tokens `c4s2 p c8 c8 p`; 3,616 ReLUs, 1,536 pool neurons, 17,822 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | - |
| Max Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 138: Architecture of the Imagenette-64 `P3` model.** *wide, max pool; tokens `c8s2 p c16 c16 p`; 7,200 ReLUs, 3,072 pool neurons, 37,010 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | - |
| Max Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | - |
| Max Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 139: Architecture of the Imagenette-64 `NP4` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p`; 7,712 ReLUs, 1,536 pool neurons, 17,970 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | - |
| Max Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Flatten | 8 x 8 x 8 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 140: Architecture of the Imagenette-64 `NP5` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p c16 p`; 7,968 ReLUs, 1,792 pool neurons, 10,946 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | - |
| Max Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 141: Architecture of the Imagenette-64 `NP6` model.** *narrow, max pool; tokens `c4s2 c4 p c8 c8 p c16 c16 p`; 8,992 ReLUs, 1,792 pool neurons, 13,266 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | - |
| Max Pooling 2D | 32 x 32 x 4 | 16 x 16 x 4 | ReLU |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | - |
| Max Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | ReLU |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | - |
| Max Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | ReLU |
| Flatten | 4 x 4 x 16 | 256 | - |
| Fully Connected | 256 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 142: Architecture of the Imagenette-64 `P4` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p`; 15,392 ReLUs, 3,072 pool neurons, 37,594 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | - |
| Max Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | - |
| Max Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Flatten | 8 x 8 x 16 | 1024 | - |
| Fully Connected | 1024 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 143: Architecture of the Imagenette-64 `P5` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p c32 p`; 15,904 ReLUs, 3,584 pool neurons, 25,850 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | - |
| Max Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | - |
| Max Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 32 | - |
| Max Pooling 2D | 8 x 8 x 32 | 4 x 4 x 32 | ReLU |
| Flatten | 4 x 4 x 32 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

**Table 144: Architecture of the Imagenette-64 `P6` model.** *wide, max pool; tokens `c8s2 c8 p c16 c16 p c32 c32 p`; 17,952 ReLUs, 3,584 pool neurons, 35,098 parameters.*

| Layer Type | Input Shape | Output Shape | Activation |
|---|---|---|---|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | - |
| Max Pooling 2D | 32 x 32 x 8 | 16 x 16 x 8 | ReLU |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 16 | ReLU |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | - |
| Max Pooling 2D | 16 x 16 x 16 | 8 x 8 x 16 | ReLU |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 32 | ReLU |
| Convolution 2D | 8 x 8 x 32 | 8 x 8 x 32 | - |
| Max Pooling 2D | 8 x 8 x 32 | 4 x 4 x 32 | ReLU |
| Flatten | 4 x 4 x 32 | 512 | - |
| Fully Connected | 512 | 32 | ReLU |
| Output | 32 | 10 | - |

