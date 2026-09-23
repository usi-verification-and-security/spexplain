# CNN bench: 48 CNNs for spexplain scaling experiments

This is the report on the CNN benchmark family built for the paper's scaling study:

- 12 architectures, identical across 4 image datasets: MNIST, GTSRB, CIFAR-10 and Imagenette at 64×64.
- All 48 models are trained, exported to ONNX and validated against spexplain's own ONNX front end.
- Everything is reproducible from the scripts in [`python_scripts/cnn_bench/`](../../python_scripts/cnn_bench/README.md).

Date: 2026-09-22. Hardware: Apple M3, 16 GB, trained on MPS.

---

## 1. Summary

- **One family of 12 architectures for every dataset:**
  - 6 depths (1, 2, 3, 4, 6 and 8 convolutions), each in 2 variants:
    - `S<d>`: downsamples with stride-2 convolutions, so ReLU is the only branching.
    - `P<d>`: downsamples with 2×2 max pooling, so there is ReLU and max-pool branching.
  - `S<d>` and `P<d>` have the **same depth and exactly the same number of ReLUs**. Each pair isolates the cost of max-pool branching.
- **Allowed non-linearities only.** The exported graphs contain only `Conv`, `Relu`, `MaxPool`, `Flatten` and `Gemm`, and output raw logits. The script checks this for every file.
  - BatchNorm is used in training and folded exactly into the convolutions at export.
  - Dropout exists only in training.
  - Softmax exists only in the loss.
- **Complexity range:**

  | dataset | ReLUs | max-pool neurons |
  |---|---|---|
  | MNIST | 1.6k → 7.7k | 0 → 568 |
  | GTSRB / CIFAR-10 | 2.1k → 10.3k | 0 → 896 |
  | Imagenette-64 | 8.2k → 41k | 0 → 3.6k |

  This sits between the older MNIST family (576 ReLUs, "too easy", up to 32.6k, "too complex") and goes deeper: up to 8 convolutions instead of 3.
- **Accuracy (top-1 on the full test split):**

  | dataset | range | notes |
  |---|---|---|
  | MNIST | 98.1–99.5% | |
  | GTSRB | 84.8–96.4% | 8 of 12 models ≥ 90% |
  | CIFAR-10 | 61.3–77.8% | |
  | Imagenette-64 | 62.4–76.3% | |

  Accuracy rises with depth on every dataset, so the family is meaningful at every size, not just a set of random networks.
- **Validation:**
  - For all 48 models, spexplain's `onnx-eval` (OnnxParser + Network2) reproduces onnxruntime's logits to ≤ 1.2e-5 on 100 held-out test images, with 0 argmax disagreements.
  - `encode-onnx` succeeds on the deepest model of every dataset.
  - `explain-onnx` produces explanations end to end (see §6).
- **New held-out sample CSVs** are at `data/datasets/<dir>/<dataset>_test100.csv`. The old `mnist_s100_scaled.csv` and `cifar_s100_scaled.csv` turned out to contain the first 100 **training** images (see §7).
- **Running experiments:** `data/scripts/tacas27/{MN,GTS,CIF,IMN}-CNN.sh` run the family through `run-experiments2.sh` (the `explain-onnx` action), with selectable models (`MODEL_IDS`) and variants (`VARIANTS`). See `tacas27/common-cnn` and `tacas27/commands_list_cnn`.

## 2. Decisions made before the run

| question | decision |
|---|---|
| Min pool | **Not used.** spexplain has no min-pool support: ONNX has no MinPool op and the parser rejects `Neg`/`Mul`. The C++ was left unchanged, so max pool is the only pooling. |
| 4th, larger benchmark | **Imagenette at 64×64.** It is the 10-class ImageNet subset from fast.ai, with 9,469 train and 3,925 val images, resized so the shorter side is 72, then cropped to 64. Its images are 4× the pixels of GTSRB/CIFAR at 32×32, but it keeps 10 classes, so the accuracy is meaningful for small ReLU-only CNNs. Tiny ImageNet (64×64, 200 classes) was the alternative. Small pure-ReLU CNNs would reach only about 20–40% top-1 there, and every classification query would carry 199 disjuncts. |
| Where models go | New directories `data/models/<dataset>/cnn-bench/`. The existing `cnn/` families are untouched. |
| Padding | Symmetric `pad=1` convolutions are allowed, which deeper nets on 28×28 need. Validated per model (§5). Note: the parser reads ONNX `pads` in the wrong order for *asymmetric* padding. That is harmless here because every pad is symmetric. |

## 3. Architecture family

Token grammar, from `archs.py`:

- `cN`: conv with N channels, 3×3, stride 1, pad 1. It keeps the size.
- `cNs2`: conv, 4×4, stride 2, pad 1. It exactly halves the size.
- `p`: 2×2 max pool, stride 2.

Every conv is followed by ReLU. Every network ends with **Flatten → FC(32) + ReLU → FC(classes)**.

`P<d>` is derived mechanically from `S<d>`: every non-stem `cNs2` becomes `p cN` (pool, then a stride-1 conv at the reduced resolution), and a final `p` is added. The script asserts this. Conv outputs therefore have the same spatial sizes in both variants, so the ReLU counts are identical.

| id | convs | pools | layers |
|---|---:|---:|---|
| S1 | 1 | 0 | Conv8/2 – FC32 – FC |
| P1 | 1 | 1 | Conv8/2 – MaxPool – FC32 – FC |
| S2 | 2 | 0 | Conv8/2 – Conv16/2 – FC32 – FC |
| P2 | 2 | 2 | Conv8/2 – MaxPool – Conv16 – MaxPool – FC32 – FC |
| S3 | 3 | 0 | Conv8/2 – Conv16/2 – Conv16 – FC32 – FC |
| P3 | 3 | 2 | Conv8/2 – MaxPool – Conv16 – Conv16 – MaxPool – FC32 – FC |
| S4 | 4 | 0 | Conv8/2 – Conv8 – Conv16/2 – Conv16 – FC32 – FC |
| P4 | 4 | 2 | Conv8/2 – Conv8 – MaxPool – Conv16 – Conv16 – MaxPool – FC32 – FC |
| S6 | 6 | 0 | Conv8/2 – Conv8 – Conv16/2 – Conv16 – Conv32/2 – Conv32 – FC32 – FC |
| P6 | 6 | 3 | Conv8/2 – Conv8 – MaxPool – Conv16 – Conv16 – MaxPool – Conv32 – Conv32 – MaxPool – FC32 – FC |
| S8 | 8 | 0 | Conv8/2 – Conv8 – Conv8 – Conv16/2 – Conv16 – Conv16 – Conv32/2 – Conv32 – FC32 – FC |
| P8 | 8 | 3 | Conv8/2 – Conv8 – Conv8 – MaxPool – Conv16 – Conv16 – Conv16 – MaxPool – Conv32 – Conv32 – MaxPool – FC32 – FC |

(`ConvN/2` is a stride-2 conv.)

**Branching counts.**
- "ReLUs" counts every ReLU neuron: conv feature maps plus the 32 head units.
- "Pool windows" counts max-pool output neurons. Each one is a 4-way max, which spexplain encodes as one fresh variable with `m ≥ aᵢ` and `∨ m = aᵢ`.
- Channel widths are kept small (8/16/32). Depth grows while the ReLU count grows only moderately, which gives the "in-between and deeper" range you asked for.

## 4. Results

Test accuracy is top-1 of the **exported ONNX model** under onnxruntime on the full test split. In all 48 cases it equals the accuracy of the un-fused PyTorch model to the last test image.

- *CSV acc* is measured on the 100 held-out rows used for explanation experiments, so it is noisy (±5%).
- *N2* is the maximum absolute logit difference between Network2 (`onnx-eval`) and onnxruntime on those rows.
- The training time includes per-epoch evaluation.

### MNIST (1×28×28, 10 classes, 15 epochs)

| id | ReLUs | pool win. | params | train acc | **test acc** | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 1600 | 0 | 50,674 | 98.62 | **98.64** | 98 | 2e-6 | 24 |
| P1 | 1600 | 392 | 13,042 | 98.12 | **98.10** | 97 | 2e-6 | 23 |
| S2 | 2384 | 0 | 27,650 | 99.11 | **99.06** | 97 | 2e-6 | 29 |
| P2 | 2384 | 536 | 6,274 | 98.49 | **98.56** | 97 | 2e-6 | 22 |
| S3 | 3168 | 0 | 29,970 | 99.36 | **99.36** | 97 | 2e-6 | 42 |
| P3 | 3168 | 536 | 8,594 | 98.99 | **99.00** | 98 | 2e-6 | 26 |
| S4 | 4736 | 0 | 30,554 | 99.43 | **99.35** | 98 | 2e-6 | 30 |
| P4 | 4736 | 536 | 9,178 | 99.29 | **99.32** | 98 | 2e-6 | 30 |
| S6 | 5312 | 0 | 32,154 | 99.62 | **99.51** | 99 | 2e-6 | 36 |
| P6 | 5312 | 568 | 19,482 | 99.54 | **99.35** | 98 | 1e-6 | 39 |
| S8 | 7664 | 0 | 35,058 | 99.64 | **99.51** | 98 | 1e-6 | 45 |
| P8 | 7664 | 568 | 22,386 | 99.63 | **99.42** | 97 | 2e-6 | 53 |

### GTSRB (3×32×32, 43 classes, 50 epochs)

| id | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 2080 | 0 | 67,379 | 97.77 | **86.20** | 97.70 | 84 | 1e-5 | 32 |
| P1 | 2080 | 512 | 18,227 | 94.92 | **85.52** | 98.14 | 88 | 1e-5 | 32 |
| S2 | 3104 | 0 | 36,675 | 99.12 | **90.78** | 98.35 | 93 | 5e-6 | 36 |
| P2 | 3104 | 768 | 11,203 | 94.26 | **84.77** | 97.92 | 87 | 8e-6 | 36 |
| S3 | 4128 | 0 | 38,995 | 99.58 | **92.61** | 98.57 | 92 | 6e-6 | 41 |
| P3 | 4128 | 768 | 13,523 | 97.18 | **88.96** | 98.55 | 95 | 6e-6 | 44 |
| S4 | 6176 | 0 | 39,579 | 99.80 | **94.45** | 99.24 | 97 | 6e-6 | 50 |
| P4 | 6176 | 768 | 14,107 | 98.64 | **92.31** | 98.54 | 91 | 7e-6 | 52 |
| S6 | 7200 | 0 | 40,667 | 99.88 | **95.62** | 99.26 | 95 | 4e-6 | 65 |
| P6 | 7200 | 896 | 23,899 | 99.31 | **92.37** | 98.28 | 97 | 7e-6 | 65 |
| S8 | 10272 | 0 | 43,571 | 99.92 | **96.42** | 99.39 | 97 | 4e-6 | 79 |
| P8 | 10272 | 896 | 26,803 | 99.64 | **94.76** | 99.15 | 95 | 5e-6 | 82 |

### CIFAR-10 (3×32×32, 10 classes, 80 epochs)

| id | ReLUs | pool win. | params | train acc | **test acc** | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 2080 | 0 | 66,290 | 62.02 | **61.36** | 50 | 2e-6 | 75 |
| P1 | 2080 | 512 | 17,138 | 61.90 | **61.29** | 49 | 2e-6 | 84 |
| S2 | 3104 | 0 | 35,586 | 68.79 | **67.78** | 56 | 2e-6 | 96 |
| P2 | 3104 | 768 | 10,114 | 66.58 | **65.55** | 60 | 3e-6 | 108 |
| S3 | 4128 | 0 | 37,906 | 72.69 | **70.54** | 63 | 2e-6 | 117 |
| P3 | 4128 | 768 | 12,434 | 69.53 | **69.02** | 65 | 3e-6 | 123 |
| S4 | 6176 | 0 | 38,490 | 74.73 | **73.39** | 65 | 3e-6 | 370* |
| P4 | 6176 | 768 | 13,018 | 71.41 | **71.07** | 69 | 2e-6 | 128 |
| S6 | 7200 | 0 | 39,578 | 78.73 | **76.30** | 66 | 3e-6 | 155 |
| P6 | 7200 | 896 | 22,810 | 75.87 | **74.26** | 70 | 3e-6 | 157 |
| S8 | 10272 | 0 | 42,482 | 80.32 | **77.81** | 76 | 4e-6 | 189 |
| P8 | 10272 | 896 | 25,714 | 76.91 | **74.05** | 71 | 3e-6 | 189 |

\*S4 shared the machine with a concurrent CPU-bound validation job.

### Imagenette-64 (3×64×64, 10 classes, 60 epochs)

| id | ReLUs | pool win. | params | train acc | **test acc** | top-5 | CSV acc | N2 | train s |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| S1 | 8224 | 0 | 262,898 | 70.98 | **62.39** | 93.40 | 60 | 2e-6 | 33 |
| P1 | 8224 | 2048 | 66,290 | 71.56 | **64.89** | 93.50 | 65 | 7e-6 | 31 |
| S2 | 12320 | 0 | 133,890 | 79.90 | **67.62** | 95.31 | 75 | 2e-6 | 37 |
| P2 | 12320 | 3072 | 34,690 | 77.91 | **70.19** | 94.93 | 78 | 2e-6 | 36 |
| S3 | 16416 | 0 | 136,210 | 83.80 | **71.95** | 95.82 | 76 | 2e-6 | 41 |
| P3 | 16416 | 3072 | 37,010 | 80.45 | **71.39** | 95.36 | 77 | 3e-6 | 39 |
| S4 | 24608 | 0 | 136,794 | 83.42 | **72.87** | 95.57 | 75 | 3e-6 | 48 |
| P4 | 24608 | 3072 | 37,594 | 80.85 | **73.45** | 96.00 | 72 | 4e-6 | 47 |
| S6 | 28704 | 0 | 88,730 | 89.05 | **75.90** | 96.76 | 81 | 3e-6 | 57 |
| P6 | 28704 | 3584 | 35,098 | 85.12 | **76.25** | 96.99 | 76 | 3e-6 | 53 |
| S8 | 40992 | 0 | 91,634 | 88.02 | **75.95** | 96.66 | 81 | 4e-6 | 67 |
| P8 | 40992 | 3584 | 38,002 | 84.99 | **75.90** | 97.04 | 75 | 4e-6 | 65 |

### Observations
- **Depth pays off on every dataset.** Accuracy grows almost monotonically from d=1 to d=6 or 8. The flattening at d=6→8 on Imagenette and MNIST is expected at this width.
- **P vs. S.**
  - On 32×32 inputs, the pooled twin is 1–6 points below the strided one, because pooling discards information and the P nets have far fewer parameters. The two are equal or better on Imagenette.
  - For the tool, a P model is strictly harder than its S twin: the same ReLUs plus max nodes. The accuracy gap is small enough that both are meaningful classifiers.
- **GTSRB's small models** (S1, P1, P2, P3) stay below 90% despite reaching 95–98% on the training set. GTSRB's training images come in tracks of near-duplicate frames, and the test set has much wider lighting variation. With 8–16 channels and no input normalisation allowed, a single conv layer cannot close that gap. The older `cnn/` family's c8s2k5 model got 84.5% for the same reason.
- **CIFAR-10** is the hardest dataset for small ReLU-only CNNs without normalisation, but 61–78% is the usual range for networks of this size.
- **Why the parameter counts fall with depth.** The flattened feature size before FC(32) shrinks as depth grows. This is why S1 on Imagenette has the most parameters (263k); almost all of them are in the 8192→32 FC layer.

## 5. Validation

Validation is done per model by `train.py` / `export_onnx.py` / `validate_spexplain.py`, and by `explain_smoke.py`.

1. **Op whitelist.** Each ONNX graph contains only {Conv, Relu, MaxPool, Flatten, Gemm}, all node types that `OnnxParser` and `OpenSMTVerifier2` encode exactly. Passed for 48/48.
2. **PyTorch vs. onnxruntime.**
   - Maximum relative logit difference on 32 random inputs is ≤ 5.2e-5, against a tolerance of 1e-4. This includes the BatchNorm folding.
   - Full-test-set accuracy of the ONNX model equals the PyTorch model's exactly (same number of correct predictions) for all 48 models.
3. **spexplain Network2 vs. onnxruntime.** `build/onnx-eval` was run on 100 held-out rows per model. The max absolute logit difference is ≤ 1.2e-5 (float32 noise), with 0 argmax mismatches, for 48/48 models. This is the check that covers the symmetric-padded convolutions, which the older families never used. Results are in `data/models/<ds>/cnn-bench/validation.csv`.
4. **SMT encoding.** `spexplain encode-onnx` succeeds on the deepest models (`data/models/cnn-bench-explain-smoke/…` and `data/models/cnn-bench-encode-check.csv`):

   | model | ReLUs | encode time | size of one class query |
   |---|---:|---:|---:|
   | mnist/S8 | 7,664 | 97 s | 81 MB |
   | mnist/P8 | 7,664 | 108 s | 113 MB |
   | gtsrb/P8 | 10,272 | 791 s (44 files) | 177 MB |
   | cifar10/S8 | 10,272 | 149 s | 113 MB |
   | cifar10/P8 | 10,272 | 186 s | 177 MB |
   | imagenette/S8 | 40,992 | 702 s | 433 MB |
   | imagenette/P8 | 40,992 | 859 s | 785 MB |

## 6. explain-onnx smoke test

The smoke test ran `spexplain explain-onnx <model> <dataset>_test100.csv itp -n 1`, i.e. the default strategy (`itp aweak, bstrong`) on the first held-out sample, with a wall-clock timeout. It ran on the release build of the current tree.

SMOKE_TABLE

These are single runs on a laptop with other jobs running at the same time, so treat them as orders of magnitude, not benchmark numbers. They already show the effect this family is meant to expose: at an **equal ReLU count**, adding a single max-pool layer (mnist S1 → P1) turns a run of about 18 minutes into a run that does not finish within an hour.

## 7. Data notes

- **New sample CSVs:** `data/datasets/{mnist,gtsrb,cifar10,imagenette}/<dataset>_test100.csv`.
  - Each holds 100 random **test** images (seed 0), CHW-flattened `k/255` values plus the label.
  - Preprocessing is identical to the evaluation pipeline.
  - They are used for the validation in §5 and the smoke tests in §6. The `tacas27/*-CNN.sh` run scripts instead default to the old `*_s100_scaled.csv` files (copied to `datasets/cifar10/cifar10_s100_scaled.csv` for CIFAR-10), which keeps the samples identical to the FC experiments. Imagenette has no old file and uses `imagenette_test100.csv`.
- **Existing CSVs are training images.** `mnist_s100_scaled.csv` and `cifar_s100_scaled.csv` hold exactly training images #0–#99. They match the torchvision train split to 3e-8, and the labels of MNIST train[0..] are 5, 0, 4, 1, 9, …. So explanations computed on them are on images the models were trained on. `gtsrb_s100_scaled.csv` does come from the test split. I left all three untouched.
- **Input domain:** all models take `[0,1]` pixels, the default in spexplain's ONNX path. The spec file still passes explicit bounds.

## 8. Training details and what happened during the run

**Recipe** (identical for every model of a dataset):
- AdamW, lr 2e-3, weight decay 5e-4, OneCycle schedule, batch 128.
- Cross-entropy with label smoothing 0.05.
- BatchNorm after every conv; dropout 0.2 before the classifier.
- Seed 0.
- The **final-epoch model is kept**: the test split is never used for model selection.
- Data is held on the MPS device as uint8, and augmentation is vectorised on the device:

  | dataset | augmentation |
  |---|---|
  | MNIST | shift ±2 px |
  | GTSRB | shift ±2 px + per-image brightness/contrast ×[0.5, 1.5], no flip |
  | CIFAR-10 | shift ±4 px + horizontal flip |
  | Imagenette | random 64-crop of a 72×72 image + flip |

**Timeline:**
1. **Smoke run.** MNIST S1/P2/S8 were trained for 1 epoch, exported and checked with Network2. This caught one bug before the real runs. The vectorised crop produced a tensor whose strides MPS rejected in conv backward (a "view size is not compatible" error). It happened only for 1-channel inputs, where `.contiguous()` is a no-op, and was fixed by gathering directly in NCHW layout.
2. **Round 1.**
   - Epochs: MNIST 15, GTSRB 25, CIFAR-10 40, Imagenette 60.
   - MNIST and Imagenette met their accuracy floors (≥ 97% and ≥ 55%) and were kept.
   - CIFAR-10 S1 reached 58.7%, below the 60% floor.
   - Five GTSRB models were below 90%: S1 82.3, P1 82.4, S2 86.6, P2 83.4, P3 85.7.
3. **Tuning (scratch runs on S1/P2).**
   - GTSRB brightness/contrast jitter: +1.6 points at 25 epochs.
   - Jitter plus 50 epochs: S1 86.2%, P2 84.8%.
   - Weight decay of 5e-3 or 2e-2 made no difference (±0.1 points).
   - CIFAR-10 S1 at 80 epochs: 61.4%.
4. **Round 2.** All 12 GTSRB models were retrained (jitter, 50 epochs) and all 12 CIFAR-10 models were retrained (80 epochs), so each dataset keeps a single recipe. Every model improved:
   - GTSRB by +1.4 to +4.2 points.
   - CIFAR-10 by +0.9 to +2.7 points.

   GTSRB S1, P1, P2 and P3 stay below 90%, for the reason given in §4. They had their one retry, as planned, and are reported as they are.

**Round-1 → round-2 test accuracy (%):**

| id | GTSRB r1 → r2 | CIFAR-10 r1 → r2 |
|---|---|---|
| S1 | 82.27 → 86.20 | 58.67 → 61.36 |
| P1 | 82.41 → 85.52 | 60.42 → 61.29 |
| S2 | 86.60 → 90.78 | 66.36 → 67.78 |
| P2 | 83.38 → 84.77 | 64.62 → 65.55 |
| S3 | 90.04 → 92.61 | 68.90 → 70.54 |
| P3 | 85.66 → 88.96 | 67.44 → 69.02 |
| S4 | 91.77 → 94.45 | 70.85 → 73.39 |
| P4 | 90.55 → 92.31 | 68.69 → 71.07 |
| S6 | 93.44 → 95.62 | 74.16 → 76.30 |
| P6 | 90.89 → 92.37 | 72.50 → 74.26 |
| S8 | 94.05 → 96.42 | 75.62 → 77.81 |
| P8 | 93.24 → 94.76 | 72.14 → 74.05 |

Total training compute was small: about 7 min for MNIST, 10 min for GTSRB, 30 min for CIFAR-10 and 9 min for Imagenette per full 12-model family (final rounds).

## 9. Files

| path | content |
|---|---|
| `python_scripts/cnn_bench/` | all scripts and their README (training, ONNX conversion, CSVs, validation, smoke test, report, `run_all.sh`) |
| `python_scripts/requirements-cnn-bench.txt` | Python dependencies (numpy, torch, torchvision, onnx, onnxruntime) |
| `data/models/<ds>/cnn-bench/{S1..P8}.onnx` | the 48 spexplain-ready models |
| `data/models/<ds>/cnn-bench/{S1..P8}.pth` | training checkpoints (with BN) and metadata |
| `data/models/<ds>/cnn-bench/manifest.csv` | architecture, counts, accuracies and checks per model |
| `data/models/<ds>/cnn-bench/validation.csv` | Network2 vs. onnxruntime results |
| `data/models/<ds>/cnn-bench/logs/`, `train.out` | per-epoch training logs |
| `data/datasets/*/<dataset>_test100.csv` | held-out sample CSVs |
| `data/models/CNN_BENCH_TABLES.md` | generated per-dataset tables |
| `data/models/cnn_bench_table.tex` | generated LaTeX table for the paper (arch × dataset: #ReLU, #Max, Acc) |
| `data/models/cnn-bench-explain-smoke(.csv)/` | smoke-test explanations, stats and logs |
| `data/models/cnn-bench-encode-check.csv` | smoke-test encoding checks |
| `data/scripts/tacas27/common-cnn`, `{MN,GTS,CIF,IMN}-CNN.sh`, `commands_list_cnn` | cluster run scripts for the family |
| `data/scripts/run-experiments2.sh`, `run1-2.sh`, `lib/run2` | ONNX run scripts, updated to the current `run-experiments.sh`/`run1.sh` interface |

Nothing was committed.

## 10. Reproduce

```bash
PY=~/miniconda3/envs/rex/bin/python          # torch 2.12 (MPS), torchvision 0.27, onnx 1.22, onnxruntime 1.26
python_scripts/cnn_bench/run_all.sh          # train (skips existing) -> CSVs -> Network2 validation -> tables
OVERWRITE=1 python_scripts/cnn_bench/run_all.sh gtsrb    # retrain one dataset from scratch
$PY python_scripts/cnn_bench/explain_smoke.py --models mnist/S1 mnist/P1 --timeout 3600 --soundness
```

MPS kernels are not bit-deterministic, so retraining reproduces the accuracies to within a few tenths of a point, not exactly. The ONNX files in the repo are the ones this report describes.
