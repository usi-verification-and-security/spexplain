# ReX deactivations vs. adversarial-search neuron files — GTSRB cnn-bench, NS and NAB

**Research plan.** spexplain computes formal explanations by encoding the network in SMT. Its two current weaknesses are speed and explanations that are too complex. This line of work asks whether ReX, a fast black-box causal explainer, and its Activation–Deactivation (A/D) idea can serve as cheap guidance for spexplain without weakening its guarantees.

The plan has three steps:

1. **Neuron fixing in hybrid mode: compare the two heuristics.** Hybrid mode fixes a chosen set of neurons, currently picked by adversarial search. We compare that heuristic with the A/D-based one in several respects:
   - **Which neurons they select:** overlap, set sizes and distribution over depth. This is done in this report.
   - **Effect on spexplain:** runtime, number of solver checks, and the share of samples explained within the time limit.
   - **Explanation quality:** size and complexity (number of terms), and how much of the input space the explanation covers.
   - **Robustness:** how results change across model families, depths and settings (γ for ReX, ε for PGD).
2. **Feature ordering.** Use ReX's responsibility ranking to decide the order in which features are relaxed. Use it also to choose which features interpolation focuses on, aiming for simpler explanations.
3. **Guess and check.** Use the ReX explanation as a first candidate, check it with a single solver call, and repair it from counterexamples if needed.

In every step the solver still certifies the result, so ReX can make spexplain faster or its explanations simpler, but never wrong.

**Question.** What is the overlap between the two heuristics for choosing which ReLU neurons spexplain's hybrid mode fixes?

- **ADV**: the adversarial-search heuristic (`activation_diff.py`). It writes the files in `data/neuron_activations/gtsrb/cnn-bench/`. For each sample, it lists the ReLUs whose phase does **not** flip between the sample and a PGD counterfactual (ε = 0.1·range, 50 steps).
- **AD**: the ReX + ConvAD heuristic (this folder). For each sample, it lists the ReLUs that ConvAD **deactivates** for the ReX explanation, i.e. units whose whole receptive field lies outside the explanation.

**Answer.** The overlap is exactly what chance predicts. On every model and every layer, the two heuristics are close to statistically independent:

- **Sharing no more than chance.** Pooled over the conv layers, the share of neurons both heuristics select is within 0.1–1.7 percentage points of what independence predicts (`both` vs `both if independent` in Table 6).
- **Deactivation says nothing about stability.** Being deactivated by ConvAD barely changes the chance that the adversarial search finds a neuron stable. Per layer, P(fixed | deact) and P(fixed | not deact) differ by 0–3 points (Table 7). The only exceptions are two cells where AD deactivates just 0–2% of a layer. The pooled gap of 1–7 points in Table 6 comes mostly from layer mixing: layer 1 has both the most deactivated and the most stable neurons.
- **Jaccard is 38–62%** only because both sets are large. It is not evidence of agreement.
- **The Oct 5 hypothesis is not supported.** It predicted that deactivated neurons, which see only free pixels, would be the *least* stable. They are as stable as the rest: about 0 points different within a layer, about 2 points *more* stable when layers are pooled.
- **The "inside" set leans slightly stable.** These are neurons whose whole receptive field lies inside the explanation. They are fixed by ADV a little more often than average (75–84% vs. 72–79%). The set is small, though: 3–9% of neurons.

So ReX/AD and adversarial search pick out **different, unrelated** structure. ReX marks neurons by *where they look* (spatial position relative to the explanation). PGD marks them by *how close they are to their switching point* (margin to 0). PGD perturbs every pixel by up to ε. Whether a neuron flips depends on its pre-activation margin, not on whether its receptive field overlaps the explanation.

Practical consequence: swapping the ADV file for the AD deactives file in hybrid mode is **not** "a cheaper version of the same heuristic". It fixes a different set of about the same size. Its effect on runtime and explanation quality has to be measured with spexplain; it cannot be inferred from ADV.

---

## Setup

- **Models.** `data/models/gtsrb/cnn-bench/NS1–6` and `NAB1–6`. These are 3×32×32 inputs and 43 classes. Every conv layer is followed by a ReLU, and NAB adds AvgPool after the ReLU.
- **Samples.** The 100 rows of `data/datasets/gtsrb/gtsrb_s100_scaled.csv`, the same samples the ADV files use. ReX reads the exact CSV values (`rex_script_cnnbench.py`), not PNG-rounded ones.
- **ReX.** The patched checkout in `rex/ReX` (v0.4.1 + 5 fixes), with ReX defaults (no `rex.toml`), confidence γ = 0.5, `--mask_value -1` and ConvAD τ = 0.
- **ConvAD on the ONNX graph** (`ad_onnx.py`). It extends the `OnnxTorchModel` interpreter that `activation_diff.py` uses. As a result, hidden layer k and neuron i mean the same thing in both files, and both follow spexplain's numbering.

### Checks

- **Theorem 4.1** (N(x) = N′(x, 1)): max |diff| = 0.0 exactly on all 12 models (`python ad_onnx.py …`).
- **Receptive-field ground truth** (`verify_rf.py`): set |W|, drop biases and ReLUs, and push the mask itself through the network. A unit is 0 iff its whole receptive field is masked. This matches ConvAD's dead set exactly on all 12 models and 32 random masks.
- **Format.** Every written file is parsed with a re-implementation of spexplain's reader (`parseSampleNeuronActivationsImpl`), with index ranges checked. `actives` and `deactives` partition every layer. The ADV files parse with the same layer sizes, which confirms the layer alignment.
- **Not done.** The files have not yet been fed to `explain-onnx --input-fix-sample-neuron-activations`.

---

## 1. Models

| model | architecture | ReLU | params | test acc | ONNX diff |
|---|---|---|---|---|---|
| NS1 | `Conv4/2 - FC32 - FC` | 1,056 | 34,415 | 0.847 | 6.65e-06 |
| NS2 | `Conv4/2 - Conv8/2 - FC32 - FC` | 1,568 | 18,551 | 0.864 | 5.26e-06 |
| NS3 | `Conv4/2 - Conv8/2 - Conv8 - FC32 - FC` | 2,080 | 19,135 | 0.880 | 1.32e-05 |
| NS4 | `Conv4/2 - Conv4 - Conv4/2 - Conv4 - FC32 - FC` | 2,592 | 10,395 | 0.868 | 2.54e-05 |
| NS5 | `Conv4/2 - Conv4 - Conv4/2 - Conv8 - Conv16/2 - FC32 - FC` | 3,104 | 12,607 | 0.875 | 4.28e-05 |
| NS6 | `Conv4/2 - Conv4 - Conv8/2 - Conv8 - Conv16/2 - Conv16 - FC32 - FC` | 3,616 | 15,475 | 0.897 | 2.93e-05 |
| NAB1 | `Conv4/2 - AvgPool - FC32 - FC` | 1,056 | 9,839 | 0.776 | 8.13e-06 |
| NAB2 | `Conv4/2 - AvgPool - Conv8 - FC32 - FC` | 1,568 | 18,327 | 0.832 | 1.65e-05 |
| NAB3 | `Conv4/2 - AvgPool - Conv8 - Conv8 - AvgPool - FC32 - FC` | 2,080 | 6,623 | 0.777 | 2.67e-05 |
| NAB4 | `Conv4/2 - Conv4 - AvgPool - Conv4 - Conv4 - AvgPool - FC32 - FC` | 2,592 | 4,139 | 0.771 | 2.82e-05 |
| NAB5 | `Conv4/2 - Conv4 - AvgPool - Conv4 - Conv8 - AvgPool - Conv16 - AvgPool - FC32 - FC` | 3,104 | 5,455 | 0.815 | 4.43e-05 |
| NAB6 | `Conv4/2 - Conv4 - AvgPool - Conv8 - Conv8 - AvgPool - Conv16 - Conv16 - AvgPool - FC32 - FC` | 3,616 | 8,211 | 0.833 | 3.35e-05 |

## 2. Explanations — ReX + ConvAD, γ = 0.5

100 samples per model, **every one explained, no failures**. Sizes are out of 1,024 pixels.

| model | explained | mean size | range | mean conf | time |
|---|---|---|---|---|---|
| NS1 | 100/100 | 169.0px (16.5%) | 10–880 | 0.515 | 0.6 min |
| NS2 | 100/100 | 124.3px (12.1%) | 10–410 | 0.502 | 0.8 min |
| NS3 | 100/100 | 174.9px (17.1%) | 10–690 | 0.508 | 0.9 min |
| NS4 | 100/100 | 198.6px (19.4%) | 10–710 | 0.498 | 1.0 min |
| NS5 | 100/100 | 150.4px (14.7%) | 10–510 | 0.545 | 1.2 min |
| NS6 | 100/100 | 176.1px (17.2%) | 10–470 | 0.551 | 1.3 min |
| NAB1 | 100/100 | 132.9px (13.0%) | 10–490 | 0.473 | 0.7 min |
| NAB2 | 100/100 | 128.9px (12.6%) | 10–740 | 0.504 | 0.8 min |
| NAB3 | 100/100 | 176.3px (17.2%) | 10–710 | 0.462 | 1.1 min |
| NAB4 | 100/100 | 248.5px (24.3%) | 10–850 | 0.455 | 1.2 min |
| NAB5 | 100/100 | 218.7px (21.4%) | 10–700 | 0.498 | 1.4 min |
| NAB6 | 100/100 | 180.8px (17.7%) | 10–710 | 0.543 | 1.5 min |

Explanations are 2–3× larger than on `gtsrb_a`–`e` (5–9%). These models are much smaller (4–16 channels), so they need more of the image to reach half their confidence.

## 3. Neuron files — 36 files + 12 meta

`% deactive` is the mean share of all ReLUs that ConvAD deactivates. `% inside` is the mean share whose receptive field lies entirely inside the explanation.

| model | layers | sizes | % deactive | % inside | spexplain-compatible |
|---|---|---|---|---|---|
| NS1 | 2 | [1024, 32] | 71.4% | 8.8% | yes |
| NS2 | 3 | [1024, 512, 32] | 69.7% | 3.6% | yes |
| NS3 | 4 | [1024, 512, 512, 32] | 51.3% | 4.3% | yes |
| NS4 | 5 | [1024, 1024, 256, 256, 32] | 52.8% | 5.8% | yes |
| NS5 | 6 | [1024, 1024, 256, 512, 256, 32] | 50.3% | 3.0% | yes |
| NS6 | 7 | [1024, 1024, 512, 512, 256, 256, 32] | 42.8% | 3.3% | yes |
| NAB1 | 2 | [1024, 32] | 74.4% | 5.6% | yes |
| NAB2 | 3 | [1024, 512, 32] | 63.5% | 3.7% | yes |
| NAB3 | 4 | [1024, 512, 512, 32] | 46.7% | 4.3% | yes |
| NAB4 | 5 | [1024, 1024, 256, 256, 32] | 45.8% | 7.8% | yes |
| NAB5 | 6 | [1024, 1024, 256, 512, 256, 32] | 42.8% | 5.6% | yes |
| NAB6 | 7 | [1024, 1024, 512, 512, 256, 256, 32] | 40.5% | 3.7% | yes |

## 4. Deactivation per depth — ReX + ConvAD

| model | layer 1 | layer 2 | layer 3 | layer 4 | layer 5 | layer 6 | layer 7 |
|---|---|---|---|---|---|---|---|
| NS1 | 74% | 0% (FC) | – | – | – | – | – |
| NS2 | 78% | 57% | 0% (FC) | – | – | – | – |
| NS3 | 71% | 47% | 20% | 0% (FC) | – | – | – |
| NS4 | 69% | 54% | 32% | 12% | 0% (FC) | – | – |
| NS5 | 75% | 61% | 38% | 14% | 2% | 0% (FC) | – |
| NS6 | 72% | 57% | 33% | 11% | 1% | 0% | 0% (FC) |
| NAB1 | 77% | 0% (FC) | – | – | – | – | – |
| NAB2 | 77% | 40% | 0% (FC) | – | – | – | – |
| NAB3 | 71% | 34% | 13% | 0% (FC) | – | – | – |
| NAB4 | 63% | 48% | 15% | 4% | 0% (FC) | – | – |
| NAB5 | 67% | 53% | 21% | 9% | 1% | 0% (FC) | – |
| NAB6 | 71% | 56% | 22% | 8% | 0% | 0% | 0% (FC) |

This is the same pattern as on `gtsrb_a`–`e`. Deactivation falls with depth as receptive fields grow. It reaches 0% before the FC layer in the deepest models (NS6, NAB6 layer 6: the receptive field already covers the whole image) and is always 0% at the FC layer.

## 5. Fixed (phase-stable) per depth — adversarial search, for comparison

| model | layer 1 | layer 2 | layer 3 | layer 4 | layer 5 | layer 6 | layer 7 |
|---|---|---|---|---|---|---|---|
| NS1 | 79% | 60% (FC) | – | – | – | – | – |
| NS2 | 81% | 72% | 55% (FC) | – | – | – | – |
| NS3 | 77% | 73% | 69% | 54% (FC) | – | – | – |
| NS4 | 81% | 75% | 71% | 69% | 52% (FC) | – | – |
| NS5 | 82% | 74% | 72% | 70% | 66% | 51% (FC) | – |
| NS6 | 83% | 76% | 74% | 72% | 69% | 64% | 54% (FC) |
| NAB1 | 76% | 54% (FC) | – | – | – | – | – |
| NAB2 | 76% | 67% | 53% (FC) | – | – | – | – |
| NAB3 | 77% | 70% | 66% | 49% (FC) | – | – | – |
| NAB4 | 83% | 74% | 68% | 65% | 48% (FC) | – | – |
| NAB5 | 79% | 73% | 70% | 69% | 63% | 47% (FC) | – |
| NAB6 | 82% | 73% | 72% | 69% | 66% | 58% | 47% (FC) |

ADV also declines with depth, but slowly, and stays near 50% at the FC layer. That is where AD contributes nothing.

## 6. Overlap — conv layers pooled

The table pools over all (sample, neuron) pairs in conv layers. The FC layer is left out because AD is 0% there by construction. NS1 and NAB1 have 99 samples: PGD found no counterfactual for one sample, so the ADV file skips it.

Columns:
- **ADV fixed**: P(S), the share of neurons the adversarial search marks stable.
- **AD deact**: P(D), the share ConvAD deactivates.
- **both**: P(S ∩ D).
- **both if independent**: P(S)·P(D).
- **ADV covered by deact**: |S ∩ D| / |S|.
- **flipped that are deact**: the share of the neurons that *did* flip under PGD that AD deactivates.

| model | samples | ADV fixed | AD deact | both | both if independent | Jaccard | P(fixed \| deact) | P(fixed \| not deact) | ADV covered by deact | P(fixed \| inside) | flipped that are deact |
|---|---|---|---|---|---|---|---|---|---|---|---|
| NS1 | 99 | 78.9% | 73.7% | 58.6% | 58.2% | 62.3% | 79.5% | 77.3% | 74.2% | 76.7% | 71.7% |
| NS2 | 100 | 78.3% | 71.1% | 56.1% | 55.7% | 60.1% | 78.8% | 77.0% | 71.6% | 81.6% | 69.4% |
| NS3 | 100 | 73.7% | 52.1% | 39.1% | 38.4% | 45.1% | 75.0% | 72.3% | 53.0% | 76.6% | 49.6% |
| NS4 | 100 | 76.4% | 53.5% | 41.7% | 40.9% | 47.3% | 78.0% | 74.6% | 54.6% | 78.6% | 49.8% |
| NS5 | 100 | 74.8% | 50.8% | 39.0% | 38.0% | 45.0% | 76.8% | 72.8% | 52.1% | 80.9% | 46.9% |
| NS6 | 100 | 75.9% | 43.2% | 34.0% | 32.8% | 40.0% | 78.7% | 73.8% | 44.8% | 84.4% | 38.2% |
| NAB1 | 99 | 75.9% | 76.7% | 58.3% | 58.2% | 61.8% | 76.1% | 75.6% | 76.8% | 75.2% | 76.4% |
| NAB2 | 100 | 73.0% | 64.9% | 48.0% | 47.3% | 53.4% | 74.0% | 71.2% | 65.7% | 79.4% | 62.5% |
| NAB3 | 100 | 72.5% | 47.4% | 35.5% | 34.4% | 42.1% | 74.9% | 70.2% | 49.1% | 77.5% | 43.2% |
| NAB4 | 100 | 76.1% | 46.4% | 36.8% | 35.3% | 42.9% | 79.3% | 73.4% | 48.3% | 79.0% | 40.3% |
| NAB5 | 100 | 73.4% | 43.2% | 33.0% | 31.7% | 39.5% | 76.3% | 71.2% | 45.0% | 78.4% | 38.5% |
| NAB6 | 100 | 73.2% | 40.8% | 31.6% | 29.9% | 38.3% | 77.4% | 70.4% | 43.1% | 78.9% | 34.5% |

How to read it:

- **`both` ≈ `both if independent` everywhere.** The excess is at most 1.7 points.
- **`ADV covered by deact` ≈ `AD deact`.** AD catches the stable neurons in the same proportion as it catches neurons overall.
- **`flipped that are deact` ≈ `AD deact` too, just slightly lower.** AD contains neither more nor fewer of the unstable neurons than chance.

## 7. Overlap per depth — P(fixed) / P(fixed | deact) / P(fixed | not deact)

| model | layer 1 | layer 2 | layer 3 | layer 4 | layer 5 | layer 6 | layer 7 |
|---|---|---|---|---|---|---|---|
| NS1 | 79% / 80% / 77% | 60% / – / 60% (FC) | – | – | – | – | – |
| NS2 | 81% / 81% / 81% | 72% / 72% / 72% | 55% / – / 55% (FC) | – | – | – | – |
| NS3 | 77% / 76% / 77% | 73% / 73% / 73% | 69% / 69% / 69% | 54% / – / 54% (FC) | – | – | – |
| NS4 | 81% / 81% / 80% | 75% / 76% / 75% | 71% / 73% / 70% | 69% / 71% / 69% | 52% / – / 52% (FC) | – | – |
| NS5 | 82% / 81% / 83% | 74% / 73% / 75% | 72% / 71% / 72% | 70% / 70% / 70% | 66% / 58% / 66% | 51% / – / 51% (FC) | – |
| NS6 | 83% / 83% / 85% | 76% / 75% / 76% | 74% / 74% / 74% | 72% / 73% / 72% | 69% / 69% / 69% | 64% / – / 64% | 54% / – / 54% (FC) |
| NAB1 | 76% / 76% / 76% | 54% / – / 54% (FC) | – | – | – | – | – |
| NAB2 | 76% / 76% / 77% | 67% / 68% / 67% | 53% / – / 53% (FC) | – | – | – | – |
| NAB3 | 77% / 77% / 78% | 70% / 70% / 69% | 66% / 67% / 66% | 49% / – / 49% (FC) | – | – | – |
| NAB4 | 83% / 83% / 82% | 74% / 76% / 73% | 68% / 69% / 68% | 65% / 67% / 65% | 48% / – / 48% (FC) | – | – |
| NAB5 | 79% / 80% / 79% | 73% / 73% / 73% | 70% / 72% / 70% | 69% / 70% / 69% | 63% / 64% / 63% | 47% / – / 47% (FC) | – |
| NAB6 | 82% / 82% / 82% | 73% / 73% / 73% | 72% / 73% / 72% | 69% / 71% / 69% | 66% / 78% / 66% | 58% / – / 58% | 47% / – / 47% (FC) |

Within a layer, the three numbers agree to within 0–3 points in almost every cell. Two cells differ more:
- NS5 layer 5: 58% vs 66%.
- NAB6 layer 5: 78% vs 66%.

Both are in layers where AD deactivates only 0–2% of neurons, so those estimates rest on very few neurons. The pooled differences in Table 6 come mostly from mixing layers. This is a Simpson effect: layer 1 has both the highest deactivation rate and the highest stability.

---

## Caveats

- **ReX ran with defaults** (no `rex.toml`) and γ = 0.5. Mean confidence sits right at that threshold (0.46–0.55), so the explanations are the smallest ones ReX accepts. A different γ changes the mask sizes, and with them the AD deactivation rates. It is unlikely to create a dependence that is absent at every depth here.
- **ADV depends on PGD settings** (ε = 0.1, 50 steps). A smaller ε would mark more neurons stable but would not make the selection spatial.
- **Overlap is not usefulness.** Independence means the two sets are different. It does not say which one helps hybrid mode more. That needs `explain-onnx` runs: hybrid with the ADV file vs the AD deactives file vs the AD inside file, comparing time, `#checks` and `#terms`.

## Files (all in `rex/cnn_bench_ad/`)

| file | what it does |
|---|---|
| `onnx_torch_model.py` | copied verbatim from `contrastive-guide/`, the interpreter `activation_diff.py` uses |
| `ad_onnx.py` | ConvAD forward pass on the ONNX graph; `python ad_onnx.py <onnx>…` runs the Thm 4.1 self-test |
| `verify_rf.py` | independent receptive-field check of the dead sets |
| `rex_script_cnnbench.py` | ReX `--script` glue; reads exact inputs from the CSV |
| `run_rex.py` | runs ReX over N samples; writes masks `.pt`, sizes CSV, explanation PNGs to `out/` |
| `neuron_files.py` | writes `neuron_activations/<model>_ad_{deactives,actives,inside}.txt` + `_ad_meta.json` |
| `compare.py` | validates all files, writes `out/tables.md` and `out/overlap_per_layer.csv` (all metrics per model × layer) |
| `run_all.sh` | the whole pipeline (≈ 13 min on CPU for 12 models × 100 samples) |


# Questions
- How many of the fixed neurons are  fixed to active, and how many to inactive?
- Try random hybrid (active/inactive/mix)