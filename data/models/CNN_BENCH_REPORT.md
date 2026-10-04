# CNN bench: 84 CNNs for spexplain scaling experiments

This is the report on the CNN benchmark family built for the paper's scaling study:

- **24 architectures** over 4 image datasets: MNIST, GTSRB, CIFAR-10 and Imagenette at 64x64.
  `S<d>`/`NS<d>` are trained on all four; `AB<d>`/`NAB<d>` on MNIST, GTSRB and CIFAR-10 only.
- All **84** models (24 + 24 + 24 + 12) are trained, exported to ONNX and validated against spexplain's own ONNX front end.
- Everything is reproducible from the scripts in [`python_scripts/cnn_bench/`](../../python_scripts/cnn_bench/README.md).

Date: 2026-09-26. Hardware: Apple M3, 16 GB, trained on MPS.

Per-layer architecture tables for every model are in [section 12](#12-architecture-tables).

---

## 1. Summary

- **One family of 24 architectures:** 6 depths (1..6 convolutions) x 4 variants.

  | variant | width | downsampling | pool branching | datasets |
  |---|---|---|---|---|
  | `S<d>` | wide 8/16/32 | stride-2 convolutions | none | all 4 |
  | `NS<d>` | narrow 4/8/16 | stride-2 convolutions | none | all 4 |
  | `AB<d>` | wide 8/16/32 | 2x2 **average** pool, ReLU before the pool | **none** -- averaging is affine | MNIST, GTSRB, CIFAR-10 |
  | `NAB<d>` | narrow 4/8/16 | 2x2 **average** pool, ReLU before the pool | **none** | MNIST, GTSRB, CIFAR-10 |

- **`AB<d>`/`NAB<d>` have exactly the ReLU counts of `S<d>`/`NS<d>`.** With the ReLU before the
  pool it sits on the full-resolution map, so the pooled twin activates as many neurons as the
  strided one; the average pool itself adds no case splits. `archs.py` asserts this for every
  dataset. See section 5.
- **Average pooling is new C++ support.** `AveragePool` was not in `OnnxParser`; this run added
  `AvgPoolLayer` (eval), the parser case, and an encoder in `OpenSMTVerifier2` that reuses the
  affine encoder. See section 6.
- **Allowed non-linearities only.** The exported graphs contain only `Conv`, `Relu`,
  `AveragePool`, `Flatten` and `Gemm`, and output raw logits. The script checks this for every file.
  - BatchNorm is used in training and folded exactly into the convolutions at export.
  - Dropout exists only in training. Softmax exists only in the loss.
- **Complexity range:**

  | dataset | ReLUs | pool neurons |
  |---|---|---|
  | MNIST | 816 -> 5,312 | 0 -> 568 |
  | GTSRB | 1,056 -> 7,200 | 0 -> 896 |
  | CIFAR-10 | 1,056 -> 7,200 | 0 -> 896 |
  | Imagenette-64 | 4,128 -> 28,704 | 0 -> 0 |

- **Accuracy (top-1 on the full test split):**

  | dataset | range |
  |---|---|
  | MNIST | 97.4-99.5% |
  | GTSRB | 77.6-95.6% |
  | CIFAR-10 | 54.3-76.3% |
  | Imagenette-64 | 55.6-75.9% |

- **Validation:** for all 84 models, spexplain's `onnx-eval` (OnnxParser + Network2) reproduces
  onnxruntime's logits to <= 8.7e-06 on the 100 experiment rows, with **0 argmax disagreements**.
- **Neuron activations** were generated for all 84 models.

---

## 2. Decisions made for this round

| question | decision |
|---|---|
| Average pooling | **Implemented in C++.** `AveragePool` had no parser case and would throw. Rather than fake it with a fixed-weight convolution, `AvgPoolLayer` was added to `Network2`, a case to `OnnxParser`, and an encoder to `OpenSMTVerifier2`. Because averaging is affine it reuses the existing affine encoder, so it needs no fresh variables and adds no disjunctions. |
| Pooled variants | **Average pooling with the ReLU before the pool** (`AB<d>`/`NAB<d>`), on MNIST, GTSRB and CIFAR-10. Imagenette-64 has only the strided `S<d>`/`NS<d>`. |
| ReLU vs. pooling order | **ReLU first, then pool** (`Conv -> BN -> ReLU -> Pool`). The ReLU is *not* removed -- it is the only nonlinearity between convolutions, and dropping it would collapse adjacent convs into a single linear map. With the ReLU before the pool, `AB<d>`/`NAB<d>` land on exactly the `S<d>`/`NS<d>` ReLU counts. |
| Depth range | **1..6 convolutions.** Depth 5 was added; the old 8-deep models were dropped. |
| Narrow width | **4/8/16.** A second width class, half the channels of the original family, to give genuinely small models. |
| Min pool | **Still not used.** spexplain has no min-pool support and ONNX has no MinPool op. |
| Padding | Symmetric `pad=1` convolutions, as before. |

---

## 3. Architecture family

Token grammar, from `archs.py`:

- `cN`: conv with N channels, 3x3, stride 1, pad 1. It keeps the size.
- `cNs2`: conv, 4x4, stride 2, pad 1. It exactly halves the size.
- `a`: 2x2 **average** pool, stride 2.

Every conv is followed by ReLU. Every network ends with **Flatten -> FC(32) + ReLU -> FC(classes)**.

The whole grid is derived from one table of strided token lists, so it cannot drift out of sync:
`AB<d>` replaces every non-stem `cNs2` of `S<d>` with `a, cN` and appends a final `a` -- except at
depth 2, where the final `a` is left out so `AB2` has a single pool (`c8s2 a c16`);
`NS<d>`/`NAB<d>` are the 4/8/16 twins of `S<d>`/`AB<d>`. `archs.py` asserts all of this,
including that `AB<d>` and `S<d>` (and `NAB<d>` and `NS<d>`) agree on ReLU counts for every dataset.

### ReLU placement

In `AB<d>`/`NAB<d>` a convolution immediately followed by a pool emits

```
Conv -> BN -> ReLU -> Pool
```

so the ReLU sits on the full-resolution map. For **average** pooling the order matters: the
opposite order is a different function, not a re-ordering (`avg(relu([-1,1])) = 0.5` while
`relu(avg([-1,1])) = 0`). With the ReLU first, `AB<d>`/`NAB<d>` land on exactly the ReLU
counts of `S<d>`/`NS<d>` (asserted by `archs.py` for every dataset).

The ReLU is never dropped.

### The 24 architectures

*Tables are split first by **channel width** (wide 8/16/32 vs narrow 4/8/16), then by pooling kind -- **no pooling**, **average pooling, ReLU before the pool**. Within each group the models are ordered **smallest first**, by ReLU count ascending.*

#### Wide (8/16/32 channels)

`S<d>` / `AB<d>` -- the original family, 8/16/32 channels.

*No pooling.* `S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

| id | convs | pools | layers |
|---|---:|---:|---|
| `S1` | 1 | 0 | Conv8/2 - FC32 - FC |
| `S2` | 2 | 0 | Conv8/2 - Conv16/2 - FC32 - FC |
| `S3` | 3 | 0 | Conv8/2 - Conv16/2 - Conv16 - FC32 - FC |
| `S4` | 4 | 0 | Conv8/2 - Conv8 - Conv16/2 - Conv16 - FC32 - FC |
| `S5` | 5 | 0 | Conv8/2 - Conv8 - Conv16/2 - Conv16 - Conv32/2 - FC32 - FC |
| `S6` | 6 | 0 | Conv8/2 - Conv8 - Conv16/2 - Conv16 - Conv32/2 - Conv32 - FC32 - FC |

*Average pooling, ReLU before the pool.* `AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

| id | convs | pools | layers |
|---|---:|---:|---|
| `AB1` | 1 | 1 | Conv8/2 - AvgPool - FC32 - FC |
| `AB2` | 2 | 1 | Conv8/2 - AvgPool - Conv16 - FC32 - FC |
| `AB3` | 3 | 2 | Conv8/2 - AvgPool - Conv16 - Conv16 - AvgPool - FC32 - FC |
| `AB4` | 4 | 2 | Conv8/2 - Conv8 - AvgPool - Conv16 - Conv16 - AvgPool - FC32 - FC |
| `AB5` | 5 | 3 | Conv8/2 - Conv8 - AvgPool - Conv16 - Conv16 - AvgPool - Conv32 - AvgPool - FC32 - FC |
| `AB6` | 6 | 3 | Conv8/2 - Conv8 - AvgPool - Conv16 - Conv16 - AvgPool - Conv32 - Conv32 - AvgPool - FC32 - FC |

#### Narrow (4/8/16 channels)

`NS<d>` / `NAB<d>` -- the half-width twins of the wide models, 4/8/16 channels. Same depth, same token sequence, same downsampling; roughly half the ReLUs and 1.8-3.6x fewer parameters.

*No pooling.* `S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

| id | convs | pools | layers |
|---|---:|---:|---|
| `NS1` | 1 | 0 | Conv4/2 - FC32 - FC |
| `NS2` | 2 | 0 | Conv4/2 - Conv8/2 - FC32 - FC |
| `NS3` | 3 | 0 | Conv4/2 - Conv8/2 - Conv8 - FC32 - FC |
| `NS4` | 4 | 0 | Conv4/2 - Conv4 - Conv8/2 - Conv8 - FC32 - FC |
| `NS5` | 5 | 0 | Conv4/2 - Conv4 - Conv8/2 - Conv8 - Conv16/2 - FC32 - FC |
| `NS6` | 6 | 0 | Conv4/2 - Conv4 - Conv8/2 - Conv8 - Conv16/2 - Conv16 - FC32 - FC |

*Average pooling, ReLU before the pool.* `AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

| id | convs | pools | layers |
|---|---:|---:|---|
| `NAB1` | 1 | 1 | Conv4/2 - AvgPool - FC32 - FC |
| `NAB2` | 2 | 1 | Conv4/2 - AvgPool - Conv8 - FC32 - FC |
| `NAB3` | 3 | 2 | Conv4/2 - AvgPool - Conv8 - Conv8 - AvgPool - FC32 - FC |
| `NAB4` | 4 | 2 | Conv4/2 - Conv4 - AvgPool - Conv8 - Conv8 - AvgPool - FC32 - FC |
| `NAB5` | 5 | 3 | Conv4/2 - Conv4 - AvgPool - Conv8 - Conv8 - AvgPool - Conv16 - AvgPool - FC32 - FC |
| `NAB6` | 6 | 3 | Conv4/2 - Conv4 - AvgPool - Conv8 - Conv8 - AvgPool - Conv16 - Conv16 - AvgPool - FC32 - FC |

**Branching counts.**
- "ReLUs" counts every ReLU neuron: conv feature maps plus the 32 head units.
- "Pool win." counts pooling output neurons. For `AB<d>`/`NAB<d>` each one is a mean, encoded as
  a single linear term -- **no variable and no disjunction**.

---

## 4. Results

*Tables are split first by **channel width** (wide 8/16/32 vs narrow 4/8/16), then by pooling kind -- **no pooling**, **average pooling, ReLU before the pool**. Within each group the models are ordered **smallest first**, by ReLU count ascending. `AB<d>`/`NAB<d>` are not trained on Imagenette-64.*

Test accuracy is top-1 of the **exported ONNX model** under onnxruntime on the full test split.

- *CSV acc* is measured on the 100 rows used for explanation experiments, so it is noisy (+-5%).
- *N2* is the maximum absolute logit difference between Network2 (`onnx-eval`) and onnxruntime on those rows.
- The training time includes per-epoch evaluation.

### MNIST (1x28x28, 10 classes, 15 epochs)

#### MNIST -- Wide, no pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `S1` | 1 | 0 | 1,600 | 0 | 50,674 | 98.62 | **98.64** | 99.98 | 99 | 2e-06 | 16 |
| `S2` | 2 | 0 | 2,384 | 0 | 27,650 | 99.11 | **99.06** | 100.00 | 99 | 2e-06 | 17 |
| `S3` | 3 | 0 | 3,168 | 0 | 29,970 | 99.36 | **99.36** | 99.99 | 99 | 2e-06 | 20 |
| `S4` | 4 | 0 | 4,736 | 0 | 30,554 | 99.43 | **99.35** | 99.99 | 99 | 1e-06 | 27 |
| `S5` | 5 | 0 | 5,024 | 0 | 22,906 | 99.54 | **99.52** | 99.99 | 99 | 1e-06 | 28 |
| `S6` | 6 | 0 | 5,312 | 0 | 32,154 | 99.62 | **99.51** | 100.00 | 99 | 1e-06 | 32 |

#### MNIST -- Wide, average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `AB1` | 1 | 1 | 1,600 | 392 | 13,042 | 98.16 | **98.35** | 99.99 | 96 | 2e-06 | 17 |
| `AB2` | 2 | 1 | 2,384 | 392 | 26,754 | 99.15 | **99.11** | 99.99 | 99 | 2e-06 | 20 |
| `AB3` | 3 | 2 | 3,168 | 536 | 8,594 | 99.12 | **99.03** | 99.98 | 99 | 2e-06 | 26 |
| `AB4` | 4 | 2 | 4,736 | 536 | 9,178 | 99.33 | **99.33** | 99.99 | 99 | 2e-06 | 29 |
| `AB5` | 5 | 3 | 5,024 | 568 | 10,234 | 99.50 | **99.34** | 100.00 | 100 | 1e-06 | 31 |
| `AB6` | 6 | 3 | 5,312 | 568 | 19,482 | 99.60 | **99.38** | 100.00 | 100 | 1e-06 | 34 |

#### MNIST -- Narrow, no pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NS1` | 1 | 0 | 816 | 0 | 25,518 | 98.11 | **98.21** | 99.97 | 96 | 3e-06 | 18 |
| `NS2` | 2 | 0 | 1,208 | 0 | 13,494 | 98.47 | **98.57** | 100.00 | 97 | 4e-06 | 22 |
| `NS3` | 3 | 0 | 1,600 | 0 | 14,078 | 98.81 | **98.84** | 100.00 | 97 | 2e-06 | 26 |
| `NS4` | 4 | 0 | 2,384 | 0 | 14,226 | 99.03 | **99.00** | 99.99 | 98 | 2e-06 | 27 |
| `NS5` | 5 | 0 | 2,528 | 0 | 8,354 | 98.97 | **98.98** | 99.99 | 100 | 2e-06 | 30 |
| `NS6` | 6 | 0 | 2,672 | 0 | 10,674 | 99.23 | **99.15** | 100.00 | 99 | 2e-06 | 32 |

#### MNIST -- Narrow, average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NAB1` | 1 | 1 | 816 | 196 | 6,702 | 97.16 | **97.40** | 99.96 | 97 | 3e-06 | 18 |
| `NAB2` | 2 | 1 | 1,208 | 196 | 13,270 | 98.48 | **98.68** | 99.98 | 98 | 2e-06 | 24 |
| `NAB3` | 3 | 2 | 1,600 | 268 | 3,614 | 98.08 | **98.24** | 100.00 | 98 | 2e-06 | 25 |
| `NAB4` | 4 | 2 | 2,384 | 268 | 3,762 | 98.51 | **98.66** | 100.00 | 100 | 2e-06 | 37 |
| `NAB5` | 5 | 3 | 2,528 | 284 | 3,138 | 98.50 | **98.53** | 99.98 | 98 | 2e-06 | 32 |
| `NAB6` | 6 | 3 | 2,672 | 284 | 5,458 | 98.96 | **98.84** | 99.99 | 99 | 1e-06 | 36 |

### GTSRB (3x32x32, 43 classes, 50 epochs)

#### GTSRB -- Wide, no pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `S1` | 1 | 0 | 2,080 | 0 | 67,379 | 97.77 | **86.20** | 97.70 | 89 | 6e-06 | 25 |
| `S2` | 2 | 0 | 3,104 | 0 | 36,675 | 99.12 | **90.78** | 98.35 | 90 | 4e-06 | 29 |
| `S3` | 3 | 0 | 4,128 | 0 | 38,995 | 99.58 | **92.61** | 98.57 | 92 | 4e-06 | 35 |
| `S4` | 4 | 0 | 6,176 | 0 | 39,579 | 99.80 | **94.45** | 99.24 | 95 | 4e-06 | 43 |
| `S5` | 5 | 0 | 6,688 | 0 | 31,419 | 99.77 | **94.64** | 98.81 | 95 | 4e-06 | 47 |
| `S6` | 6 | 0 | 7,200 | 0 | 40,667 | 99.88 | **95.62** | 99.26 | 95 | 3e-06 | 53 |

#### GTSRB -- Wide, average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `AB1` | 1 | 1 | 2,080 | 512 | 18,227 | 95.28 | **83.06** | 97.60 | 82 | 8e-06 | 26 |
| `AB2` | 2 | 1 | 3,104 | 512 | 35,779 | 98.67 | **87.74** | 98.54 | 86 | 6e-06 | 31 |
| `AB3` | 3 | 2 | 4,128 | 768 | 13,523 | 97.89 | **89.02** | 97.91 | 90 | 8e-06 | 37 |
| `AB4` | 4 | 2 | 6,176 | 768 | 14,107 | 99.28 | **92.69** | 99.00 | 93 | 5e-06 | 44 |
| `AB5` | 5 | 3 | 6,688 | 896 | 14,651 | 99.17 | **91.73** | 98.29 | 92 | 5e-06 | 49 |
| `AB6` | 6 | 3 | 7,200 | 896 | 23,899 | 99.66 | **93.05** | 98.62 | 92 | 4e-06 | 59 |

#### GTSRB -- Narrow, no pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NS1` | 1 | 0 | 1,056 | 0 | 34,415 | 95.53 | **84.68** | 97.18 | 85 | 8e-06 | 24 |
| `NS2` | 2 | 0 | 1,568 | 0 | 18,551 | 96.78 | **86.44** | 98.15 | 86 | 6e-06 | 29 |
| `NS3` | 3 | 0 | 2,080 | 0 | 19,135 | 97.65 | **87.97** | 98.23 | 84 | 7e-06 | 38 |
| `NS4` | 4 | 0 | 3,104 | 0 | 19,283 | 98.19 | **90.37** | 98.71 | 90 | 5e-06 | 43 |
| `NS5` | 5 | 0 | 3,360 | 0 | 13,155 | 97.90 | **90.04** | 97.99 | 90 | 5e-06 | 47 |
| `NS6` | 6 | 0 | 3,616 | 0 | 15,475 | 98.27 | **89.72** | 98.09 | 94 | 6e-06 | 54 |

#### GTSRB -- Narrow, average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NAB1` | 1 | 1 | 1,056 | 256 | 9,839 | 89.76 | **77.59** | 95.50 | 74 | 9e-06 | 27 |
| `NAB2` | 2 | 1 | 1,568 | 256 | 18,327 | 95.20 | **83.20** | 97.08 | 87 | 7e-06 | 36 |
| `NAB3` | 3 | 2 | 2,080 | 384 | 6,623 | 87.94 | **77.72** | 95.87 | 76 | 7e-06 | 42 |
| `NAB4` | 4 | 2 | 3,104 | 384 | 6,771 | 93.48 | **84.73** | 97.46 | 89 | 5e-06 | 51 |
| `NAB5` | 5 | 3 | 3,360 | 448 | 5,891 | 91.02 | **81.30** | 96.34 | 82 | 6e-06 | 54 |
| `NAB6` | 6 | 3 | 3,616 | 448 | 8,211 | 93.32 | **83.25** | 97.47 | 86 | 6e-06 | 60 |

### CIFAR-10 (3x32x32, 10 classes, 80 epochs)

#### CIFAR-10 -- Wide, no pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `S1` | 1 | 0 | 2,080 | 0 | 66,290 | 62.02 | **61.36** | 95.91 | 64 | 2e-06 | 62 |
| `S2` | 2 | 0 | 3,104 | 0 | 35,586 | 68.79 | **67.78** | 97.16 | 66 | 2e-06 | 83 |
| `S3` | 3 | 0 | 4,128 | 0 | 37,906 | 72.69 | **70.54** | 97.72 | 78 | 3e-06 | 96 |
| `S4` | 4 | 0 | 6,176 | 0 | 38,490 | 74.73 | **73.39** | 98.21 | 76 | 2e-06 | 122 |
| `S5` | 5 | 0 | 6,688 | 0 | 30,330 | 76.87 | **74.79** | 98.28 | 78 | 2e-06 | 130 |
| `S6` | 6 | 0 | 7,200 | 0 | 39,578 | 78.73 | **76.30** | 98.40 | 82 | 3e-06 | 150 |

#### CIFAR-10 -- Wide, average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `AB1` | 1 | 1 | 2,080 | 512 | 17,138 | 61.78 | **61.20** | 95.79 | 66 | 3e-06 | 78 |
| `AB2` | 2 | 1 | 3,104 | 512 | 34,690 | 69.65 | **67.99** | 97.18 | 71 | 2e-06 | 89 |
| `AB3` | 3 | 2 | 4,128 | 768 | 12,434 | 71.17 | **70.33** | 97.69 | 77 | 2e-06 | 112 |
| `AB4` | 4 | 2 | 6,176 | 768 | 13,018 | 72.48 | **71.49** | 98.00 | 74 | 2e-06 | 133 |
| `AB5` | 5 | 3 | 6,688 | 896 | 13,562 | 75.10 | **73.68** | 98.37 | 77 | 2e-06 | 161 |
| `AB6` | 6 | 3 | 7,200 | 896 | 22,810 | 77.76 | **75.35** | 98.36 | 76 | 2e-06 | 169 |

#### CIFAR-10 -- Narrow, no pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NS1` | 1 | 0 | 1,056 | 0 | 33,326 | 55.94 | **55.11** | 94.64 | 60 | 3e-06 | 71 |
| `NS2` | 2 | 0 | 1,568 | 0 | 17,462 | 59.22 | **58.63** | 95.46 | 54 | 3e-06 | 91 |
| `NS3` | 3 | 0 | 2,080 | 0 | 18,046 | 62.75 | **61.88** | 96.16 | 68 | 3e-06 | 112 |
| `NS4` | 4 | 0 | 3,104 | 0 | 18,194 | 63.17 | **62.54** | 96.03 | 62 | 3e-06 | 121 |
| `NS5` | 5 | 0 | 3,360 | 0 | 12,066 | 66.12 | **65.47** | 96.84 | 64 | 3e-06 | 133 |
| `NS6` | 6 | 0 | 3,616 | 0 | 14,386 | 67.18 | **66.27** | 96.86 | 69 | 4e-06 | 150 |

#### CIFAR-10 -- Narrow, average pooling, ReLU before the pool

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NAB1` | 1 | 1 | 1,056 | 256 | 8,750 | 54.48 | **54.31** | 94.48 | 50 | 3e-06 | 78 |
| `NAB2` | 2 | 1 | 1,568 | 256 | 17,238 | 58.70 | **58.15** | 95.37 | 60 | 2e-06 | 99 |
| `NAB3` | 3 | 2 | 2,080 | 384 | 5,534 | 58.39 | **57.09** | 95.40 | 63 | 2e-06 | 111 |
| `NAB4` | 4 | 2 | 3,104 | 384 | 5,682 | 61.46 | **61.23** | 95.82 | 60 | 2e-06 | 132 |
| `NAB5` | 5 | 3 | 3,360 | 448 | 4,802 | 60.28 | **59.88** | 95.91 | 59 | 2e-06 | 157 |
| `NAB6` | 6 | 3 | 3,616 | 448 | 7,122 | 63.41 | **62.60** | 96.17 | 63 | 2e-06 | 174 |

### Imagenette-64 (3x64x64, 10 classes, 60 epochs)

#### Imagenette-64 -- Wide, no pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `S1` | 1 | 0 | 8,224 | 0 | 262,898 | 70.98 | **62.39** | 93.40 | 60 | 2e-06 | 28 |
| `S2` | 2 | 0 | 12,320 | 0 | 133,890 | 79.90 | **67.62** | 95.31 | 75 | 2e-06 | 32 |
| `S3` | 3 | 0 | 16,416 | 0 | 136,210 | 83.80 | **71.95** | 95.82 | 76 | 2e-06 | 35 |
| `S4` | 4 | 0 | 24,608 | 0 | 136,794 | 83.42 | **72.87** | 95.57 | 75 | 3e-06 | 43 |
| `S5` | 5 | 0 | 26,656 | 0 | 79,482 | 86.57 | **74.55** | 96.99 | 73 | 2e-06 | 354 |
| `S6` | 6 | 0 | 28,704 | 0 | 88,730 | 89.05 | **75.90** | 96.76 | 81 | 3e-06 | 386 |

#### Imagenette-64 -- Narrow, no pooling

| id | convs | pools | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `NS1` | 1 | 0 | 4,128 | 0 | 131,630 | 62.41 | **55.59** | 91.16 | 54 | 3e-06 | 24 |
| `NS2` | 2 | 0 | 6,176 | 0 | 66,614 | 70.94 | **64.03** | 93.50 | 73 | 2e-06 | 27 |
| `NS3` | 3 | 0 | 8,224 | 0 | 67,198 | 73.10 | **65.66** | 94.34 | 66 | 2e-06 | 29 |
| `NS4` | 4 | 0 | 12,320 | 0 | 67,346 | 75.06 | **67.31** | 95.36 | 74 | 3e-06 | 34 |
| `NS5` | 5 | 0 | 13,344 | 0 | 36,642 | 76.24 | **68.61** | 95.34 | 69 | 4e-06 | 248 |
| `NS6` | 6 | 0 | 14,368 | 0 | 38,962 | 78.46 | **71.26** | 95.49 | 73 | 3e-06 | 282 |

### Observations

- **Depth pays off on every dataset.** In every variant on every dataset the d=6 model beats the
  d=1 model, so the family is meaningful at every size.
- **The narrow family is genuinely cheaper.** `NS`/`NAB` roughly halve the ReLU count of
  their wide twins and cut parameters by 1.8-3.6x, at a cost of up to 13.8 points of test
  accuracy (largest on CIFAR-10).
- **Why parameter counts fall with depth.** The flattened feature size before FC(32) shrinks as
  depth grows, so the shallow models carry most of their weight in one large FC layer.

---

## 5. ReLU before the pool: `AB<d>`/`NAB<d>` vs. `S<d>`/`NS<d>`

`AB<d>` downsamples with 2x2 average pooling where `S<d>` uses stride-2 convolutions, and emits
the ReLU **before** the pool:

```
S<d>    Conv(stride 2) -> ReLU               (ReLU on the downsampled map)
AB<d>   Conv -> ReLU -> AveragePool          (ReLU on the full-resolution map)
```

For average pooling the ReLU order matters: `avg(relu([-1,1])) = 0.5` while
`relu(avg([-1,1])) = 0`. With the ReLU first, `AB<d>` lands on exactly the `S<d>` ReLU count
(and `NAB<d>` on the `NS<d>` count), while the pool adds no case splits. The pair therefore has identical
branching and differs in the downsampling operator and the parameter count.

| dataset | pair | ReLUs | params strided | params pooled | test acc strided | test acc pooled | pooled - strided |
|---|---|---:|---:|---:|---:|---:|---:|
| MNIST | `S1` -> `AB1` | 1,600 | 50,674 | 13,042 | 98.64 | 98.35 | -0.29 |
| MNIST | `S2` -> `AB2` | 2,384 | 27,650 | 26,754 | 99.06 | 99.11 | +0.05 |
| MNIST | `S3` -> `AB3` | 3,168 | 29,970 | 8,594 | 99.36 | 99.03 | -0.33 |
| MNIST | `S4` -> `AB4` | 4,736 | 30,554 | 9,178 | 99.35 | 99.33 | -0.02 |
| MNIST | `S5` -> `AB5` | 5,024 | 22,906 | 10,234 | 99.52 | 99.34 | -0.18 |
| MNIST | `S6` -> `AB6` | 5,312 | 32,154 | 19,482 | 99.51 | 99.38 | -0.13 |
| MNIST | `NS1` -> `NAB1` | 816 | 25,518 | 6,702 | 98.21 | 97.40 | -0.81 |
| MNIST | `NS2` -> `NAB2` | 1,208 | 13,494 | 13,270 | 98.57 | 98.68 | +0.11 |
| MNIST | `NS3` -> `NAB3` | 1,600 | 14,078 | 3,614 | 98.84 | 98.24 | -0.60 |
| MNIST | `NS4` -> `NAB4` | 2,384 | 14,226 | 3,762 | 99.00 | 98.66 | -0.34 |
| MNIST | `NS5` -> `NAB5` | 2,528 | 8,354 | 3,138 | 98.98 | 98.53 | -0.45 |
| MNIST | `NS6` -> `NAB6` | 2,672 | 10,674 | 5,458 | 99.15 | 98.84 | -0.31 |
| GTSRB | `S1` -> `AB1` | 2,080 | 67,379 | 18,227 | 86.20 | 83.06 | -3.14 |
| GTSRB | `S2` -> `AB2` | 3,104 | 36,675 | 35,779 | 90.78 | 87.74 | -3.04 |
| GTSRB | `S3` -> `AB3` | 4,128 | 38,995 | 13,523 | 92.61 | 89.02 | -3.59 |
| GTSRB | `S4` -> `AB4` | 6,176 | 39,579 | 14,107 | 94.45 | 92.69 | -1.76 |
| GTSRB | `S5` -> `AB5` | 6,688 | 31,419 | 14,651 | 94.64 | 91.73 | -2.91 |
| GTSRB | `S6` -> `AB6` | 7,200 | 40,667 | 23,899 | 95.62 | 93.05 | -2.57 |
| GTSRB | `NS1` -> `NAB1` | 1,056 | 34,415 | 9,839 | 84.68 | 77.59 | -7.09 |
| GTSRB | `NS2` -> `NAB2` | 1,568 | 18,551 | 18,327 | 86.44 | 83.20 | -3.24 |
| GTSRB | `NS3` -> `NAB3` | 2,080 | 19,135 | 6,623 | 87.97 | 77.72 | -10.25 |
| GTSRB | `NS4` -> `NAB4` | 3,104 | 19,283 | 6,771 | 90.37 | 84.73 | -5.64 |
| GTSRB | `NS5` -> `NAB5` | 3,360 | 13,155 | 5,891 | 90.04 | 81.30 | -8.74 |
| GTSRB | `NS6` -> `NAB6` | 3,616 | 15,475 | 8,211 | 89.72 | 83.25 | -6.47 |
| CIFAR-10 | `S1` -> `AB1` | 2,080 | 66,290 | 17,138 | 61.36 | 61.20 | -0.16 |
| CIFAR-10 | `S2` -> `AB2` | 3,104 | 35,586 | 34,690 | 67.78 | 67.99 | +0.21 |
| CIFAR-10 | `S3` -> `AB3` | 4,128 | 37,906 | 12,434 | 70.54 | 70.33 | -0.21 |
| CIFAR-10 | `S4` -> `AB4` | 6,176 | 38,490 | 13,018 | 73.39 | 71.49 | -1.90 |
| CIFAR-10 | `S5` -> `AB5` | 6,688 | 30,330 | 13,562 | 74.79 | 73.68 | -1.11 |
| CIFAR-10 | `S6` -> `AB6` | 7,200 | 39,578 | 22,810 | 76.30 | 75.35 | -0.95 |
| CIFAR-10 | `NS1` -> `NAB1` | 1,056 | 33,326 | 8,750 | 55.11 | 54.31 | -0.80 |
| CIFAR-10 | `NS2` -> `NAB2` | 1,568 | 17,462 | 17,238 | 58.63 | 58.15 | -0.48 |
| CIFAR-10 | `NS3` -> `NAB3` | 2,080 | 18,046 | 5,534 | 61.88 | 57.09 | -4.79 |
| CIFAR-10 | `NS4` -> `NAB4` | 3,104 | 18,194 | 5,682 | 62.54 | 61.23 | -1.31 |
| CIFAR-10 | `NS5` -> `NAB5` | 3,360 | 12,066 | 4,802 | 65.47 | 59.88 | -5.59 |
| CIFAR-10 | `NS6` -> `NAB6` | 3,616 | 14,386 | 7,122 | 66.27 | 62.60 | -3.67 |

**Reading.** MNIST: `AB` - `S` -0.33 to +0.05, `NAB` - `NS` -0.81 to +0.11; GTSRB: `AB` - `S` -3.59 to -1.76, `NAB` - `NS` -10.25 to -3.24; CIFAR-10: `AB` - `S` -1.90 to +0.21, `NAB` - `NS` -5.59 to -0.48 points.

`AB<d>`/`NAB<d>` are trained on MNIST, GTSRB and CIFAR-10 only, so Imagenette-64 keeps the
12-model strided family. Everything that enumerates models (`archs.ids_for`, the run scripts,
the activation generator) skips ids a dataset does not have.

---

## 6. Average pooling support in spexplain

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
   classification constraint -- confirming avg pooling contributes zero disjunctions.
3. **Explain.** `explain-onnx` produces explanations for every sample.
4. **Soundness.** `check_phi_soundness.sh` reports **6/6 sound, 0 unsound**.
5. **Encoding vs. network.** Pinning every input to a sample's values and asking each `psi_c`
   query reproduces the network's argmax, for samples of all three classes.

`python_scripts/contrastive-guide/onnx_torch_model.py` also needed an `AveragePool` case, since
the neuron-activation guide interprets the ONNX graph directly.

---

## 7. Validation

1. **Op whitelist.** Each ONNX graph contains only {Conv, Relu, AveragePool, Flatten, Gemm}.
   Passed for 84/84.
2. **PyTorch vs. onnxruntime.** Maximum relative logit difference on 32 random inputs is
   **3.5e-05**, against a tolerance of 1e-4. This includes the BatchNorm folding.
   - Full-test-set accuracy of the ONNX model equals the PyTorch model's exactly for all
     84 models.
3. **spexplain Network2 vs. onnxruntime.** `build/onnx-eval` was run on the 100 experiment rows
   per model. The max absolute logit difference is **8.74e-06**, with **0 argmax mismatches**,
   for **84/84** models. Results are in `data/models/<ds>/cnn-bench/validation.csv`.
   `validate_spexplain.py` now *merges* into `validation.csv` instead of overwriting it, so a
   filtered run (`--only`) refreshes just its own rows rather than dropping the others.
4. **Average-pool encoding.** See section 6.

---

## 8. Data notes

The experiment CSVs are now frozen and there is exactly one per dataset:

| dataset | CSV |
|---|---|
| MNIST | `data/datasets/mnist/mnist_s100_scaled.csv` |
| GTSRB | `data/datasets/gtsrb/gtsrb_s100_scaled.csv` |
| CIFAR-10 | `data/datasets/cifar/cifar_s100_scaled.csv` |
| Imagenette | `data/datasets/imagenette/imagenette_test100.csv` |

- `make_sample_csv.py` now holds this mapping in `SAMPLE_CSV` and **refuses to regenerate** the
  three legacy files, so the samples stay identical to the earlier FC experiments.
- CIFAR-10 uses `data/datasets/cifar/`
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

## 9. Training details

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

**Cost of the full run** (24 models per dataset, 12 on Imagenette-64; sum of the per-model
`train_seconds` in `manifest.csv`):

| dataset | training time |
|---|---|
| MNIST | 10 min |
| GTSRB | 16 min |
| CIFAR-10 | 47 min |
| Imagenette-64 | 25 min |

---

## 10. Files

| path | content |
|---|---|
| `python_scripts/cnn_bench/` | all scripts and their README |
| `python_scripts/cnn_bench/arch_tables.py` | **new**: generates the per-layer tables in section 12 |
| `data/models/<ds>/cnn-bench/<id>.onnx` | the 84 spexplain-ready models |
| `data/models/<ds>/cnn-bench/<id>.pth` | training checkpoints (with BN) and metadata |
| `data/models/<ds>/cnn-bench/manifest.csv` | architecture, counts, accuracies and checks per model |
| `data/models/<ds>/cnn-bench/validation.csv` | Network2 vs. onnxruntime results |
| `data/models/CNN_BENCH_TABLES.md`, `cnn_bench_table.tex` | generated per-dataset tables |
| `data/neuron_activations/<ds>/cnn-bench/<id>.txt` | contrastive-guide activation files (84 files) |
| `src/spexplain/network/AvgPoolLayer.{h,cpp}` | **new**: average-pool layer |
| `data/scripts/tacas27/common-cnn`, `{MN,GTS,CIF,IMN}-CNN.sh` | cluster run scripts, `MODEL_IDS` updated to the 24-model family |

---

## 11. Reproduce

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

## 12. Architecture tables

One table per model, read directly from the exported `.onnx` graphs with shape inference, so the
shapes are the ones `OnnxParser` actually sees. Shapes are `H x W x C` for feature maps and a
plain width for flat vectors; the batch axis is dropped, because `Network2` strips it and a flat
CSV row maps onto `C*H*W` features in CHW row-major order.

Each `Relu` node of the graph is folded into the **Activation** column of the layer that feeds
it. So in an `AB`/`NAB` model the ReLU shows on the convolution, and the following pooling layer
reads `-` -- the placement described in section 3.

**ReLUs** is the number of ReLU neurons that layer contributes. **Branchings** is what the SMT
encoding actually pays for: one case split per ReLU neuron. Average pooling is affine and adds
none, so for every model in this family the two columns are equal.

### MNIST

Input `28 x 28 x 1`, 10 classes.

#### Wide (8/16/32 channels)

`S<d>` / `AB<d>` -- the original family, 8/16/32 channels.

##### Wide / No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 1: Architecture of the MNIST `S1` model.** *wide, strided; tokens `c8s2`; 1,600 ReLUs, 0 pool neurons, 50,674 parameters; 1,600 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Flatten | 14 x 14 x 8 | 1568 | - | 0 | 0 |
| Fully Connected | 1568 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,600** | **1,600** |

**Table 2: Architecture of the MNIST `S2` model.** *wide, strided; tokens `c8s2 c16s2`; 2,384 ReLUs, 0 pool neurons, 27,650 parameters; 2,384 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Flatten | 7 x 7 x 16 | 784 | - | 0 | 0 |
| Fully Connected | 784 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,384** | **2,384** |

**Table 3: Architecture of the MNIST `S3` model.** *wide, strided; tokens `c8s2 c16s2 c16`; 3,168 ReLUs, 0 pool neurons, 29,970 parameters; 3,168 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Flatten | 7 x 7 x 16 | 784 | - | 0 | 0 |
| Fully Connected | 784 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,168** | **3,168** |

**Table 4: Architecture of the MNIST `S4` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16`; 4,736 ReLUs, 0 pool neurons, 30,554 parameters; 4,736 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Flatten | 7 x 7 x 16 | 784 | - | 0 | 0 |
| Fully Connected | 784 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **4,736** | **4,736** |

**Table 5: Architecture of the MNIST `S5` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2`; 5,024 ReLUs, 0 pool neurons, 22,906 parameters; 5,024 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 3 x 3 x 32 | ReLU | 288 | 288 |
| Flatten | 3 x 3 x 32 | 288 | - | 0 | 0 |
| Fully Connected | 288 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **5,024** | **5,024** |

**Table 6: Architecture of the MNIST `S6` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2 c32`; 5,312 ReLUs, 0 pool neurons, 32,154 parameters; 5,312 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 3 x 3 x 32 | ReLU | 288 | 288 |
| Convolution 2D | 3 x 3 x 32 | 3 x 3 x 32 | ReLU | 288 | 288 |
| Flatten | 3 x 3 x 32 | 288 | - | 0 | 0 |
| Fully Connected | 288 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **5,312** | **5,312** |

##### Wide / Average pooling, ReLU before the pool

`AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 7: Architecture of the MNIST `AB1` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 a`; 1,600 ReLUs, 392 pool neurons, 13,042 parameters; 1,600 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | - | 0 | 0 |
| Flatten | 7 x 7 x 8 | 392 | - | 0 | 0 |
| Fully Connected | 392 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,600** | **1,600** |

**Table 8: Architecture of the MNIST `AB2` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 a c16`; 2,384 ReLUs, 392 pool neurons, 26,754 parameters; 2,384 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Flatten | 7 x 7 x 16 | 784 | - | 0 | 0 |
| Fully Connected | 784 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,384** | **2,384** |

**Table 9: Architecture of the MNIST `AB3` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 a c16 c16 a`; 3,168 ReLUs, 536 pool neurons, 8,594 parameters; 3,168 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Average Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | - | 0 | 0 |
| Flatten | 3 x 3 x 16 | 144 | - | 0 | 0 |
| Fully Connected | 144 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,168** | **3,168** |

**Table 10: Architecture of the MNIST `AB4` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 c8 a c16 c16 a`; 4,736 ReLUs, 536 pool neurons, 9,178 parameters; 4,736 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Average Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | - | 0 | 0 |
| Flatten | 3 x 3 x 16 | 144 | - | 0 | 0 |
| Fully Connected | 144 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **4,736** | **4,736** |

**Table 11: Architecture of the MNIST `AB5` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 c8 a c16 c16 a c32 a`; 5,024 ReLUs, 568 pool neurons, 10,234 parameters; 5,024 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Average Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | - | 0 | 0 |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 32 | ReLU | 288 | 288 |
| Average Pooling 2D | 3 x 3 x 32 | 1 x 1 x 32 | - | 0 | 0 |
| Flatten | 1 x 1 x 32 | 32 | - | 0 | 0 |
| Fully Connected | 32 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **5,024** | **5,024** |

**Table 12: Architecture of the MNIST `AB6` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 c8 a c16 c16 a c32 c32 a`; 5,312 ReLUs, 568 pool neurons, 19,482 parameters; 5,312 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Convolution 2D | 14 x 14 x 8 | 14 x 14 x 8 | ReLU | 1,568 | 1,568 |
| Average Pooling 2D | 14 x 14 x 8 | 7 x 7 x 8 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Convolution 2D | 7 x 7 x 16 | 7 x 7 x 16 | ReLU | 784 | 784 |
| Average Pooling 2D | 7 x 7 x 16 | 3 x 3 x 16 | - | 0 | 0 |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 32 | ReLU | 288 | 288 |
| Convolution 2D | 3 x 3 x 32 | 3 x 3 x 32 | ReLU | 288 | 288 |
| Average Pooling 2D | 3 x 3 x 32 | 1 x 1 x 32 | - | 0 | 0 |
| Flatten | 1 x 1 x 32 | 32 | - | 0 | 0 |
| Fully Connected | 32 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **5,312** | **5,312** |

#### Narrow (4/8/16 channels)

`NS<d>` / `NAB<d>` -- the half-width twins of the wide models, 4/8/16 channels. Same depth, same token sequence, same downsampling; roughly half the ReLUs and 2-6x fewer parameters.

##### Narrow / No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 13: Architecture of the MNIST `NS1` model.** *narrow, strided; tokens `c4s2`; 816 ReLUs, 0 pool neurons, 25,518 parameters; 816 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Flatten | 14 x 14 x 4 | 784 | - | 0 | 0 |
| Fully Connected | 784 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **816** | **816** |

**Table 14: Architecture of the MNIST `NS2` model.** *narrow, strided; tokens `c4s2 c8s2`; 1,208 ReLUs, 0 pool neurons, 13,494 parameters; 1,208 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Flatten | 7 x 7 x 8 | 392 | - | 0 | 0 |
| Fully Connected | 392 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,208** | **1,208** |

**Table 15: Architecture of the MNIST `NS3` model.** *narrow, strided; tokens `c4s2 c8s2 c8`; 1,600 ReLUs, 0 pool neurons, 14,078 parameters; 1,600 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Flatten | 7 x 7 x 8 | 392 | - | 0 | 0 |
| Fully Connected | 392 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,600** | **1,600** |

**Table 16: Architecture of the MNIST `NS4` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8`; 2,384 ReLUs, 0 pool neurons, 14,226 parameters; 2,384 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Flatten | 7 x 7 x 8 | 392 | - | 0 | 0 |
| Fully Connected | 392 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,384** | **2,384** |

**Table 17: Architecture of the MNIST `NS5` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2`; 2,528 ReLUs, 0 pool neurons, 8,354 parameters; 2,528 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 3 x 3 x 16 | ReLU | 144 | 144 |
| Flatten | 3 x 3 x 16 | 144 | - | 0 | 0 |
| Fully Connected | 144 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,528** | **2,528** |

**Table 18: Architecture of the MNIST `NS6` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2 c16`; 2,672 ReLUs, 0 pool neurons, 10,674 parameters; 2,672 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 3 x 3 x 16 | ReLU | 144 | 144 |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 16 | ReLU | 144 | 144 |
| Flatten | 3 x 3 x 16 | 144 | - | 0 | 0 |
| Fully Connected | 144 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,672** | **2,672** |

##### Narrow / Average pooling, ReLU before the pool

`AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 19: Architecture of the MNIST `NAB1` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 a`; 816 ReLUs, 196 pool neurons, 6,702 parameters; 816 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | - | 0 | 0 |
| Flatten | 7 x 7 x 4 | 196 | - | 0 | 0 |
| Fully Connected | 196 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **816** | **816** |

**Table 20: Architecture of the MNIST `NAB2` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 a c8`; 1,208 ReLUs, 196 pool neurons, 13,270 parameters; 1,208 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Flatten | 7 x 7 x 8 | 392 | - | 0 | 0 |
| Fully Connected | 392 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,208** | **1,208** |

**Table 21: Architecture of the MNIST `NAB3` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 a c8 c8 a`; 1,600 ReLUs, 268 pool neurons, 3,614 parameters; 1,600 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Average Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | - | 0 | 0 |
| Flatten | 3 x 3 x 8 | 72 | - | 0 | 0 |
| Fully Connected | 72 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,600** | **1,600** |

**Table 22: Architecture of the MNIST `NAB4` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 c4 a c8 c8 a`; 2,384 ReLUs, 268 pool neurons, 3,762 parameters; 2,384 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Average Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | - | 0 | 0 |
| Flatten | 3 x 3 x 8 | 72 | - | 0 | 0 |
| Fully Connected | 72 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,384** | **2,384** |

**Table 23: Architecture of the MNIST `NAB5` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 c4 a c8 c8 a c16 a`; 2,528 ReLUs, 284 pool neurons, 3,138 parameters; 2,528 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Average Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | - | 0 | 0 |
| Convolution 2D | 3 x 3 x 8 | 3 x 3 x 16 | ReLU | 144 | 144 |
| Average Pooling 2D | 3 x 3 x 16 | 1 x 1 x 16 | - | 0 | 0 |
| Flatten | 1 x 1 x 16 | 16 | - | 0 | 0 |
| Fully Connected | 16 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,528** | **2,528** |

**Table 24: Architecture of the MNIST `NAB6` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 c4 a c8 c8 a c16 c16 a`; 2,672 ReLUs, 284 pool neurons, 5,458 parameters; 2,672 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 28 x 28 x 1 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Convolution 2D | 14 x 14 x 4 | 14 x 14 x 4 | ReLU | 784 | 784 |
| Average Pooling 2D | 14 x 14 x 4 | 7 x 7 x 4 | - | 0 | 0 |
| Convolution 2D | 7 x 7 x 4 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Convolution 2D | 7 x 7 x 8 | 7 x 7 x 8 | ReLU | 392 | 392 |
| Average Pooling 2D | 7 x 7 x 8 | 3 x 3 x 8 | - | 0 | 0 |
| Convolution 2D | 3 x 3 x 8 | 3 x 3 x 16 | ReLU | 144 | 144 |
| Convolution 2D | 3 x 3 x 16 | 3 x 3 x 16 | ReLU | 144 | 144 |
| Average Pooling 2D | 3 x 3 x 16 | 1 x 1 x 16 | - | 0 | 0 |
| Flatten | 1 x 1 x 16 | 16 | - | 0 | 0 |
| Fully Connected | 16 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,672** | **2,672** |

### GTSRB

Input `32 x 32 x 3`, 43 classes.

#### Wide (8/16/32 channels)

`S<d>` / `AB<d>` -- the original family, 8/16/32 channels.

##### Wide / No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 25: Architecture of the GTSRB `S1` model.** *wide, strided; tokens `c8s2`; 2,080 ReLUs, 0 pool neurons, 67,379 parameters; 2,080 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Flatten | 16 x 16 x 8 | 2048 | - | 0 | 0 |
| Fully Connected | 2048 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **2,080** | **2,080** |

**Table 26: Architecture of the GTSRB `S2` model.** *wide, strided; tokens `c8s2 c16s2`; 3,104 ReLUs, 0 pool neurons, 36,675 parameters; 3,104 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **3,104** | **3,104** |

**Table 27: Architecture of the GTSRB `S3` model.** *wide, strided; tokens `c8s2 c16s2 c16`; 4,128 ReLUs, 0 pool neurons, 38,995 parameters; 4,128 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **4,128** | **4,128** |

**Table 28: Architecture of the GTSRB `S4` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16`; 6,176 ReLUs, 0 pool neurons, 39,579 parameters; 6,176 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **6,176** | **6,176** |

**Table 29: Architecture of the GTSRB `S5` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2`; 6,688 ReLUs, 0 pool neurons, 31,419 parameters; 6,688 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Flatten | 4 x 4 x 32 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **6,688** | **6,688** |

**Table 30: Architecture of the GTSRB `S6` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2 c32`; 7,200 ReLUs, 0 pool neurons, 40,667 parameters; 7,200 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Flatten | 4 x 4 x 32 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **7,200** | **7,200** |

##### Wide / Average pooling, ReLU before the pool

`AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 31: Architecture of the GTSRB `AB1` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 a`; 2,080 ReLUs, 512 pool neurons, 18,227 parameters; 2,080 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **2,080** | **2,080** |

**Table 32: Architecture of the GTSRB `AB2` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 a c16`; 3,104 ReLUs, 512 pool neurons, 35,779 parameters; 3,104 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **3,104** | **3,104** |

**Table 33: Architecture of the GTSRB `AB3` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 a c16 c16 a`; 4,128 ReLUs, 768 pool neurons, 13,523 parameters; 4,128 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | - | 0 | 0 |
| Flatten | 4 x 4 x 16 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **4,128** | **4,128** |

**Table 34: Architecture of the GTSRB `AB4` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 c8 a c16 c16 a`; 6,176 ReLUs, 768 pool neurons, 14,107 parameters; 6,176 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | - | 0 | 0 |
| Flatten | 4 x 4 x 16 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **6,176** | **6,176** |

**Table 35: Architecture of the GTSRB `AB5` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 c8 a c16 c16 a c32 a`; 6,688 ReLUs, 896 pool neurons, 14,651 parameters; 6,688 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | - | 0 | 0 |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Average Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | - | 0 | 0 |
| Flatten | 2 x 2 x 32 | 128 | - | 0 | 0 |
| Fully Connected | 128 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **6,688** | **6,688** |

**Table 36: Architecture of the GTSRB `AB6` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 c8 a c16 c16 a c32 c32 a`; 7,200 ReLUs, 896 pool neurons, 23,899 parameters; 7,200 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | - | 0 | 0 |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Average Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | - | 0 | 0 |
| Flatten | 2 x 2 x 32 | 128 | - | 0 | 0 |
| Fully Connected | 128 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **7,200** | **7,200** |

#### Narrow (4/8/16 channels)

`NS<d>` / `NAB<d>` -- the half-width twins of the wide models, 4/8/16 channels. Same depth, same token sequence, same downsampling; roughly half the ReLUs and 2-6x fewer parameters.

##### Narrow / No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 37: Architecture of the GTSRB `NS1` model.** *narrow, strided; tokens `c4s2`; 1,056 ReLUs, 0 pool neurons, 34,415 parameters; 1,056 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Flatten | 16 x 16 x 4 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **1,056** | **1,056** |

**Table 38: Architecture of the GTSRB `NS2` model.** *narrow, strided; tokens `c4s2 c8s2`; 1,568 ReLUs, 0 pool neurons, 18,551 parameters; 1,568 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **1,568** | **1,568** |

**Table 39: Architecture of the GTSRB `NS3` model.** *narrow, strided; tokens `c4s2 c8s2 c8`; 2,080 ReLUs, 0 pool neurons, 19,135 parameters; 2,080 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **2,080** | **2,080** |

**Table 40: Architecture of the GTSRB `NS4` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8`; 3,104 ReLUs, 0 pool neurons, 19,283 parameters; 3,104 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **3,104** | **3,104** |

**Table 41: Architecture of the GTSRB `NS5` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2`; 3,360 ReLUs, 0 pool neurons, 13,155 parameters; 3,360 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Flatten | 4 x 4 x 16 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **3,360** | **3,360** |

**Table 42: Architecture of the GTSRB `NS6` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2 c16`; 3,616 ReLUs, 0 pool neurons, 15,475 parameters; 3,616 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Flatten | 4 x 4 x 16 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **3,616** | **3,616** |

##### Narrow / Average pooling, ReLU before the pool

`AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 43: Architecture of the GTSRB `NAB1` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 a`; 1,056 ReLUs, 256 pool neurons, 9,839 parameters; 1,056 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Flatten | 8 x 8 x 4 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **1,056** | **1,056** |

**Table 44: Architecture of the GTSRB `NAB2` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 a c8`; 1,568 ReLUs, 256 pool neurons, 18,327 parameters; 1,568 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **1,568** | **1,568** |

**Table 45: Architecture of the GTSRB `NAB3` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 a c8 c8 a`; 2,080 ReLUs, 384 pool neurons, 6,623 parameters; 2,080 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | - | 0 | 0 |
| Flatten | 4 x 4 x 8 | 128 | - | 0 | 0 |
| Fully Connected | 128 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **2,080** | **2,080** |

**Table 46: Architecture of the GTSRB `NAB4` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 c4 a c8 c8 a`; 3,104 ReLUs, 384 pool neurons, 6,771 parameters; 3,104 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | - | 0 | 0 |
| Flatten | 4 x 4 x 8 | 128 | - | 0 | 0 |
| Fully Connected | 128 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **3,104** | **3,104** |

**Table 47: Architecture of the GTSRB `NAB5` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 c4 a c8 c8 a c16 a`; 3,360 ReLUs, 448 pool neurons, 5,891 parameters; 3,360 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | - | 0 | 0 |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Average Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | - | 0 | 0 |
| Flatten | 2 x 2 x 16 | 64 | - | 0 | 0 |
| Fully Connected | 64 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **3,360** | **3,360** |

**Table 48: Architecture of the GTSRB `NAB6` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 c4 a c8 c8 a c16 c16 a`; 3,616 ReLUs, 448 pool neurons, 8,211 parameters; 3,616 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | - | 0 | 0 |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Average Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | - | 0 | 0 |
| Flatten | 2 x 2 x 16 | 64 | - | 0 | 0 |
| Fully Connected | 64 | 32 | ReLU | 32 | 32 |
| Output | 32 | 43 | - | 0 | 0 |
| **Total** | | | | **3,616** | **3,616** |

### CIFAR-10

Input `32 x 32 x 3`, 10 classes.

#### Wide (8/16/32 channels)

`S<d>` / `AB<d>` -- the original family, 8/16/32 channels.

##### Wide / No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 49: Architecture of the CIFAR-10 `S1` model.** *wide, strided; tokens `c8s2`; 2,080 ReLUs, 0 pool neurons, 66,290 parameters; 2,080 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Flatten | 16 x 16 x 8 | 2048 | - | 0 | 0 |
| Fully Connected | 2048 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,080** | **2,080** |

**Table 50: Architecture of the CIFAR-10 `S2` model.** *wide, strided; tokens `c8s2 c16s2`; 3,104 ReLUs, 0 pool neurons, 35,586 parameters; 3,104 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,104** | **3,104** |

**Table 51: Architecture of the CIFAR-10 `S3` model.** *wide, strided; tokens `c8s2 c16s2 c16`; 4,128 ReLUs, 0 pool neurons, 37,906 parameters; 4,128 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **4,128** | **4,128** |

**Table 52: Architecture of the CIFAR-10 `S4` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16`; 6,176 ReLUs, 0 pool neurons, 38,490 parameters; 6,176 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **6,176** | **6,176** |

**Table 53: Architecture of the CIFAR-10 `S5` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2`; 6,688 ReLUs, 0 pool neurons, 30,330 parameters; 6,688 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Flatten | 4 x 4 x 32 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **6,688** | **6,688** |

**Table 54: Architecture of the CIFAR-10 `S6` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2 c32`; 7,200 ReLUs, 0 pool neurons, 39,578 parameters; 7,200 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Flatten | 4 x 4 x 32 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **7,200** | **7,200** |

##### Wide / Average pooling, ReLU before the pool

`AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 55: Architecture of the CIFAR-10 `AB1` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 a`; 2,080 ReLUs, 512 pool neurons, 17,138 parameters; 2,080 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,080** | **2,080** |

**Table 56: Architecture of the CIFAR-10 `AB2` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 a c16`; 3,104 ReLUs, 512 pool neurons, 34,690 parameters; 3,104 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,104** | **3,104** |

**Table 57: Architecture of the CIFAR-10 `AB3` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 a c16 c16 a`; 4,128 ReLUs, 768 pool neurons, 12,434 parameters; 4,128 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | - | 0 | 0 |
| Flatten | 4 x 4 x 16 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **4,128** | **4,128** |

**Table 58: Architecture of the CIFAR-10 `AB4` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 c8 a c16 c16 a`; 6,176 ReLUs, 768 pool neurons, 13,018 parameters; 6,176 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | - | 0 | 0 |
| Flatten | 4 x 4 x 16 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **6,176** | **6,176** |

**Table 59: Architecture of the CIFAR-10 `AB5` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 c8 a c16 c16 a c32 a`; 6,688 ReLUs, 896 pool neurons, 13,562 parameters; 6,688 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | - | 0 | 0 |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Average Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | - | 0 | 0 |
| Flatten | 2 x 2 x 32 | 128 | - | 0 | 0 |
| Fully Connected | 128 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **6,688** | **6,688** |

**Table 60: Architecture of the CIFAR-10 `AB6` model.** *wide, average pool, ReLU before the pool; tokens `c8s2 c8 a c16 c16 a c32 c32 a`; 7,200 ReLUs, 896 pool neurons, 22,810 parameters; 7,200 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Average Pooling 2D | 16 x 16 x 8 | 8 x 8 x 8 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 8 x 8 x 16 | 4 x 4 x 16 | - | 0 | 0 |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Convolution 2D | 4 x 4 x 32 | 4 x 4 x 32 | ReLU | 512 | 512 |
| Average Pooling 2D | 4 x 4 x 32 | 2 x 2 x 32 | - | 0 | 0 |
| Flatten | 2 x 2 x 32 | 128 | - | 0 | 0 |
| Fully Connected | 128 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **7,200** | **7,200** |

#### Narrow (4/8/16 channels)

`NS<d>` / `NAB<d>` -- the half-width twins of the wide models, 4/8/16 channels. Same depth, same token sequence, same downsampling; roughly half the ReLUs and 2-6x fewer parameters.

##### Narrow / No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 61: Architecture of the CIFAR-10 `NS1` model.** *narrow, strided; tokens `c4s2`; 1,056 ReLUs, 0 pool neurons, 33,326 parameters; 1,056 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Flatten | 16 x 16 x 4 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,056** | **1,056** |

**Table 62: Architecture of the CIFAR-10 `NS2` model.** *narrow, strided; tokens `c4s2 c8s2`; 1,568 ReLUs, 0 pool neurons, 17,462 parameters; 1,568 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,568** | **1,568** |

**Table 63: Architecture of the CIFAR-10 `NS3` model.** *narrow, strided; tokens `c4s2 c8s2 c8`; 2,080 ReLUs, 0 pool neurons, 18,046 parameters; 2,080 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,080** | **2,080** |

**Table 64: Architecture of the CIFAR-10 `NS4` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8`; 3,104 ReLUs, 0 pool neurons, 18,194 parameters; 3,104 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,104** | **3,104** |

**Table 65: Architecture of the CIFAR-10 `NS5` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2`; 3,360 ReLUs, 0 pool neurons, 12,066 parameters; 3,360 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Flatten | 4 x 4 x 16 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,360** | **3,360** |

**Table 66: Architecture of the CIFAR-10 `NS6` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2 c16`; 3,616 ReLUs, 0 pool neurons, 14,386 parameters; 3,616 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Flatten | 4 x 4 x 16 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,616** | **3,616** |

##### Narrow / Average pooling, ReLU before the pool

`AB<d>` / `NAB<d>` -- 2x2 average pooling with the ReLU *before* the pool (mnist/gtsrb/cifar only). Each output is the mean of its window, an affine map: no fresh variable and **no disjunction**.

**Table 67: Architecture of the CIFAR-10 `NAB1` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 a`; 1,056 ReLUs, 256 pool neurons, 8,750 parameters; 1,056 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Flatten | 8 x 8 x 4 | 256 | - | 0 | 0 |
| Fully Connected | 256 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,056** | **1,056** |

**Table 68: Architecture of the CIFAR-10 `NAB2` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 a c8`; 1,568 ReLUs, 256 pool neurons, 17,238 parameters; 1,568 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Flatten | 8 x 8 x 8 | 512 | - | 0 | 0 |
| Fully Connected | 512 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **1,568** | **1,568** |

**Table 69: Architecture of the CIFAR-10 `NAB3` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 a c8 c8 a`; 2,080 ReLUs, 384 pool neurons, 5,534 parameters; 2,080 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | - | 0 | 0 |
| Flatten | 4 x 4 x 8 | 128 | - | 0 | 0 |
| Fully Connected | 128 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **2,080** | **2,080** |

**Table 70: Architecture of the CIFAR-10 `NAB4` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 c4 a c8 c8 a`; 3,104 ReLUs, 384 pool neurons, 5,682 parameters; 3,104 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | - | 0 | 0 |
| Flatten | 4 x 4 x 8 | 128 | - | 0 | 0 |
| Fully Connected | 128 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,104** | **3,104** |

**Table 71: Architecture of the CIFAR-10 `NAB5` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 c4 a c8 c8 a c16 a`; 3,360 ReLUs, 448 pool neurons, 4,802 parameters; 3,360 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | - | 0 | 0 |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Average Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | - | 0 | 0 |
| Flatten | 2 x 2 x 16 | 64 | - | 0 | 0 |
| Fully Connected | 64 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,360** | **3,360** |

**Table 72: Architecture of the CIFAR-10 `NAB6` model.** *narrow, average pool, ReLU before the pool; tokens `c4s2 c4 a c8 c8 a c16 c16 a`; 3,616 ReLUs, 448 pool neurons, 7,122 parameters; 3,616 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 32 x 32 x 3 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 16 x 16 x 4 | 16 x 16 x 4 | ReLU | 1,024 | 1,024 |
| Average Pooling 2D | 16 x 16 x 4 | 8 x 8 x 4 | - | 0 | 0 |
| Convolution 2D | 8 x 8 x 4 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Convolution 2D | 8 x 8 x 8 | 8 x 8 x 8 | ReLU | 512 | 512 |
| Average Pooling 2D | 8 x 8 x 8 | 4 x 4 x 8 | - | 0 | 0 |
| Convolution 2D | 4 x 4 x 8 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Convolution 2D | 4 x 4 x 16 | 4 x 4 x 16 | ReLU | 256 | 256 |
| Average Pooling 2D | 4 x 4 x 16 | 2 x 2 x 16 | - | 0 | 0 |
| Flatten | 2 x 2 x 16 | 64 | - | 0 | 0 |
| Fully Connected | 64 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **3,616** | **3,616** |

### Imagenette-64

Input `64 x 64 x 3`, 10 classes.

#### Wide (8/16/32 channels)

`S<d>` / `AB<d>` -- the original family, 8/16/32 channels.

##### Wide / No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 73: Architecture of the Imagenette-64 `S1` model.** *wide, strided; tokens `c8s2`; 8,224 ReLUs, 0 pool neurons, 262,898 parameters; 8,224 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU | 8,192 | 8,192 |
| Flatten | 32 x 32 x 8 | 8192 | - | 0 | 0 |
| Fully Connected | 8192 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **8,224** | **8,224** |

**Table 74: Architecture of the Imagenette-64 `S2` model.** *wide, strided; tokens `c8s2 c16s2`; 12,320 ReLUs, 0 pool neurons, 133,890 parameters; 12,320 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU | 8,192 | 8,192 |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU | 4,096 | 4,096 |
| Flatten | 16 x 16 x 16 | 4096 | - | 0 | 0 |
| Fully Connected | 4096 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **12,320** | **12,320** |

**Table 75: Architecture of the Imagenette-64 `S3` model.** *wide, strided; tokens `c8s2 c16s2 c16`; 16,416 ReLUs, 0 pool neurons, 136,210 parameters; 16,416 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU | 8,192 | 8,192 |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | ReLU | 4,096 | 4,096 |
| Flatten | 16 x 16 x 16 | 4096 | - | 0 | 0 |
| Fully Connected | 4096 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **16,416** | **16,416** |

**Table 76: Architecture of the Imagenette-64 `S4` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16`; 24,608 ReLUs, 0 pool neurons, 136,794 parameters; 24,608 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU | 8,192 | 8,192 |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | ReLU | 8,192 | 8,192 |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | ReLU | 4,096 | 4,096 |
| Flatten | 16 x 16 x 16 | 4096 | - | 0 | 0 |
| Fully Connected | 4096 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **24,608** | **24,608** |

**Table 77: Architecture of the Imagenette-64 `S5` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2`; 26,656 ReLUs, 0 pool neurons, 79,482 parameters; 26,656 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU | 8,192 | 8,192 |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | ReLU | 8,192 | 8,192 |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 16 x 16 x 16 | 8 x 8 x 32 | ReLU | 2,048 | 2,048 |
| Flatten | 8 x 8 x 32 | 2048 | - | 0 | 0 |
| Fully Connected | 2048 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **26,656** | **26,656** |

**Table 78: Architecture of the Imagenette-64 `S6` model.** *wide, strided; tokens `c8s2 c8 c16s2 c16 c32s2 c32`; 28,704 ReLUs, 0 pool neurons, 88,730 parameters; 28,704 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 8 | ReLU | 8,192 | 8,192 |
| Convolution 2D | 32 x 32 x 8 | 32 x 32 x 8 | ReLU | 8,192 | 8,192 |
| Convolution 2D | 32 x 32 x 8 | 16 x 16 x 16 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 16 x 16 x 16 | 16 x 16 x 16 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 16 x 16 x 16 | 8 x 8 x 32 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 8 x 8 x 32 | 8 x 8 x 32 | ReLU | 2,048 | 2,048 |
| Flatten | 8 x 8 x 32 | 2048 | - | 0 | 0 |
| Fully Connected | 2048 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **28,704** | **28,704** |

#### Narrow (4/8/16 channels)

`NS<d>` / `NAB<d>` -- the half-width twins of the wide models, 4/8/16 channels. Same depth, same token sequence, same downsampling; roughly half the ReLUs and 2-6x fewer parameters.

##### Narrow / No pooling

`S<d>` / `NS<d>` -- downsampling is done by stride-2 convolutions, so there is no pooling layer at all and nothing beyond the ReLUs to encode.

**Table 79: Architecture of the Imagenette-64 `NS1` model.** *narrow, strided; tokens `c4s2`; 4,128 ReLUs, 0 pool neurons, 131,630 parameters; 4,128 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU | 4,096 | 4,096 |
| Flatten | 32 x 32 x 4 | 4096 | - | 0 | 0 |
| Fully Connected | 4096 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **4,128** | **4,128** |

**Table 80: Architecture of the Imagenette-64 `NS2` model.** *narrow, strided; tokens `c4s2 c8s2`; 6,176 ReLUs, 0 pool neurons, 66,614 parameters; 6,176 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Flatten | 16 x 16 x 8 | 2048 | - | 0 | 0 |
| Fully Connected | 2048 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **6,176** | **6,176** |

**Table 81: Architecture of the Imagenette-64 `NS3` model.** *narrow, strided; tokens `c4s2 c8s2 c8`; 8,224 ReLUs, 0 pool neurons, 67,198 parameters; 8,224 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Flatten | 16 x 16 x 8 | 2048 | - | 0 | 0 |
| Fully Connected | 2048 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **8,224** | **8,224** |

**Table 82: Architecture of the Imagenette-64 `NS4` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8`; 12,320 ReLUs, 0 pool neurons, 67,346 parameters; 12,320 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Flatten | 16 x 16 x 8 | 2048 | - | 0 | 0 |
| Fully Connected | 2048 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **12,320** | **12,320** |

**Table 83: Architecture of the Imagenette-64 `NS5` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2`; 13,344 ReLUs, 0 pool neurons, 36,642 parameters; 13,344 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **13,344** | **13,344** |

**Table 84: Architecture of the Imagenette-64 `NS6` model.** *narrow, strided; tokens `c4s2 c4 c8s2 c8 c16s2 c16`; 14,368 ReLUs, 0 pool neurons, 38,962 parameters; 14,368 branchings.*

| Layer Type | Input Shape | Output Shape | Activation | ReLUs | Branchings |
|---|---|---|---|---:|---:|
| Convolution 2D | 64 x 64 x 3 | 32 x 32 x 4 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 32 x 32 x 4 | 32 x 32 x 4 | ReLU | 4,096 | 4,096 |
| Convolution 2D | 32 x 32 x 4 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 16 x 16 x 8 | ReLU | 2,048 | 2,048 |
| Convolution 2D | 16 x 16 x 8 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Convolution 2D | 8 x 8 x 16 | 8 x 8 x 16 | ReLU | 1,024 | 1,024 |
| Flatten | 8 x 8 x 16 | 1024 | - | 0 | 0 |
| Fully Connected | 1024 | 32 | ReLU | 32 | 32 |
| Output | 32 | 10 | - | 0 | 0 |
| **Total** | | | | **14,368** | **14,368** |
