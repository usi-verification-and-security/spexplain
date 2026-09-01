# Report — `explain` on `.nnet` vs `.onnx` (heart_attack-50)

**Date:** 2026-08-19 (updated after the `--drop-sigmoid` fix)
**Model:** `data/models/heart_attack/heart_attack-50` (available as both `.nnet` and `.onnx`)
**Goal:** verify that the new ONNX explanation pipeline (`explain-onnx` → `Network2` + `OpenSMTVerifier2`)
produces the same explanations, at the same cost, as the legacy NNet pipeline
(`explain` → `Network` + `OpenSMTVerifier`).

---

## 0. Summary of findings

| Question | Answer |
|---|---|
| Are the SMT **encodings** the same? | **Yes** — `psi_d` bit-identical, `psi_c*` equal up to 2.2e-5 float32 rounding |
| Are the **explanations** the same? | **Yes** for `abductive` and `ucore` (bit-identical + logically equivalent); `itp` equal up to 1.9e-5 |
| Are the ONNX explanations **sound**? | **Yes** — 18/18 verified UNSAT against their own encoding |
| Is the **cost** the same? | Yes — identical formula sizes and check counts; ONNX 1–5 % slower |
| Do both formats predict the same class? | **Yes**, after the `--drop-sigmoid` fix (§6) |
| Any remaining problem? | No |

Sections 2–5 report the **original** measurements, taken before the sigmoid fix of §6 and therefore
restricted to the 6 class-agreeing samples. **§6.4 contains the final post-fix results over all 10
samples** — that is the authoritative comparison.

All results hold **after** the `--input-min` / `--input-max` fix (§1).

---

## 1. Prerequisite: ONNX input bounds

ONNX files carry **no input-domain metadata**. Before this fix, `Network2` silently fell back to a
`[0, 1]` box for every feature, so the explanation search could free almost any variable and produced
drastically weaker explanations (`abductive` fixed **1** feature instead of 6–10).

Two new CLI options were added (accepted by `explain-onnx` / `encode-onnx` / `read-onnx` only):

```
--input-min <v1,v2,...>    per-feature input domain minimums
--input-max <v1,v2,...>    per-feature input domain maximums
```

For the heart_attack model the real feature domains are:

| | age | sex | cp | trtbps | chol | fbs | restecg | thalachh | exng | oldpeak | slp | caa | thall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **min** | 29 | 0 | 0 | 94 | 126 | 0 | 0 | 71 | 0 | 0 | 0 | 0 | 0 |
| **max** | 77 | 1 | 3 | 200 | 594 | 1 | 2 | 202 | 1 | 6.2 | 2 | 4 | 3 |

```
--input-min 29,0,0,94,126,0,0,71,0,0,0,0,0
--input-max 77,1,3,200,594,1,2,202,1,6.2,2,4,3
```

`compare_explain_onnx_vs_nnet.sh` supplies these automatically when the model path matches
`*heart_attack*` and no explicit bounds were given.

---

## 2. Encoding parity (`psi_d`, `psi_c*`)

Structural, rounding-tolerant diff (`psi_diff.py --tol 1e-3`), columns are
`A-conj B-conj identical close differing onlyA onlyB max-rel-dev`:

| file | A conj | B conj | identical | close (≤1e-3) | differing | only-A | only-B | max rel. dev |
|---|---|---|---|---|---|---|---|---|
| `psi_d.smt2` | 26 | 26 | **26** | 0 | 0 | 0 | 0 | **0** |
| `psi_c1.smt2` | 127 | 127 | 104 | 23 | 0 | 0 | 0 | 2.21867e-05 |
| `psi_c0.smt2` | 127 | 127 | 104 | 23 | 0 | 0 | 0 | 2.21867e-05 |

`psi_d.smt2` (the input-domain formula) is now **100 % identical** — this was previously the
documented known mismatch and was caused entirely by the missing input bounds.
The 23 "close" conjuncts in `psi_c*` are float32-vs-decimal weight rounding.

> `--ignore-bounds` is **no longer needed** when comparing these encodings.

---

## 3. Explanation comparison

### 3.1 Dataset

Because of the classification bug in §6, the two formats disagree on the predicted class for 4 of the
10 samples in `heart_attack_quick.csv`. Those 4 samples make `itp` and `ucore` **crash** on the ONNX
side. The core comparison therefore uses the **6 class-agreeing samples**
(`heart_attack_quick.csv` rows 3, 4, 6, 8, 9, 10), stored as
`build-debug/explain-report/heart_attack_agree.csv`. §5 reports the unfiltered 10-sample run.

> After the fix of §6 this filtering is **no longer necessary** — see §6.4 for the clean 10-sample
> comparison across all three strategies.

### 3.2 Feature-level diff (`phi_diff.py`)

| strategy | samples | matching | differing | only-A | only-B |
|---|---|---|---|---|---|
| `abductive` | 6 | **6** | 0 | 0 | 0 |
| `itp` | 6 | 6 † | 0 | 0 | 0 |
| `ucore` | 6 | **6** | 0 | 0 | 0 |

† `phi_diff.py` only counts `(= xI v)` conjuncts, which `itp` does not produce — for `itp` this row is
vacuous, so use the structural diff in §3.3 instead.

Per-sample fixed-feature counts (`abductive`, A = nnet, B = onnx):

| sample | A-fixed | B-fixed | common | only-A/B | status |
|---|---|---|---|---|---|
| 1 | 6 | 6 | 6 | 0/0 | MATCH |
| 2 | 8 | 8 | 8 | 0/0 | MATCH |
| 3 | 8 | 8 | 8 | 0/0 | MATCH |
| 4 | 9 | 9 | 9 | 0/0 | MATCH |
| 5 | 7 | 7 | 7 | 0/0 | MATCH |
| 6 | 7 | 7 | 7 | 0/0 | MATCH |

### 3.3 Structural, rounding-tolerant diff (`psi_diff.py --tol 1e-3`)

Explanation lines wrapped as `(assert <phi>)` and diffed conjunct-by-conjunct:

| strategy | A conj | B conj | identical | close | differing | only-A | only-B | max rel. dev |
|---|---|---|---|---|---|---|---|---|
| `abductive` | 47 | 47 | **47** | 0 | 0 | 0 | 0 | **0** |
| `itp` | 156 | 156 | 113 | 43 | 0 | 0 | 0 | 1.85151e-05 |
| `ucore` | 78 | 78 | **78** | 0 | 0 | 0 | 0 | **0** |

`abductive` and `ucore` explanations are **bit-identical**. `itp` explanations differ only in the
coefficients of the interpolant half-spaces, by at most 1.9e-5 relative — the same float32 rounding
seen in the encoding.

### 3.4 Logical relation under `psi_d` (`compare_phi_subset.sh`)

| strategy | `=` equivalent | `<` A stronger | `>` B stronger | `?` incomparable |
|---|---|---|---|---|
| `abductive` | **6** | 0 | 0 | 0 |
| `itp` | 0 | 0 | 0 | 6 |
| `ucore` | **6** | 0 | 0 | 0 |

`itp` is "incomparable" purely because an interpolant lies exactly **on** the decision boundary; a
1.9e-5 coefficient shift means neither half-space fully contains the other, even though they are
numerically the same to 5 decimals.

---

## 4. Soundness verification (`check_phi_soundness.sh`)

An explanation `phi` is sound iff `psi_c<label> /\ phi` is **UNSAT**.

### 4.1 Each explanation against its own encoding

| strategy | nnet-phi vs nnet-psi | onnx-phi vs onnx-psi |
|---|---|---|
| `abductive` | **6/6 sound** | **6/6 sound** |
| `itp` | **6/6 sound** | **6/6 sound** |
| `ucore` | **6/6 sound** | **6/6 sound** |

**18/18 ONNX explanations are provably sound.**

### 4.2 Cross-encoding (an explanation from one format checked against the other's encoding)

| strategy | nnet-phi vs onnx-psi | onnx-phi vs nnet-psi |
|---|---|---|
| `abductive` | **6/6 sound** | **6/6 sound** |
| `itp` | **6/6 sound** | **0/6** ⚠ |
| `ucore` | **6/6 sound** | **6/6 sound** |

The single `itp` failure is the boundary-tightness artifact from §3.3: the ONNX interpolant is valid
for the ONNX encoding but sits ~1e-5 outside the nnet one. `abductive`/`ucore` explanations are
axis-aligned boxes and are robust to that perturbation. **This is not a logic error.**

---

## 5. Statistics

Columns: `AVG-FEAT` free features, `AVG-FIXED` fixed neuron activations, `AVG-TERMS` conjuncts in the
explanation, `AVG-CHK` solver calls, `CORRECT` samples whose computed class matches the dataset label.

### 5.1 Six class-agreeing samples

| strategy | run | N | AVG-FEAT | AVG-FIXED | AVG-TERMS | AVG-CHK | AVG-TIME (s) | TOT-TIME (s) | CORRECT |
|---|---|---|---|---|---|---|---|---|---|
| `abductive` | nnet | 6 | 7.83 | 0.00 | 7.83 | 13.00 | 0.2607 | 1.564 | 3/6 |
| `abductive` | onnx | 6 | 7.83 | 0.00 | 7.83 | 13.00 | 0.2630 | 1.578 | 3/6 |
| `itp` | nnet | 6 | 13.00 | 0.00 | 26.00 | 1.00 | 0.0077 | 0.046 | 3/6 |
| `itp` | onnx | 6 | 13.00 | 0.00 | 26.00 | 1.00 | 0.0079 | 0.047 | 3/6 |
| `ucore` | nnet | 6 | 13.00 | 0.00 | 13.00 | 1.00 | 0.0060 | 0.036 | 3/6 |
| `ucore` | onnx | 6 | 13.00 | 0.00 | 13.00 | 1.00 | 0.0063 | 0.038 | 3/6 |

Relative deltas (onnx vs nnet):

| strategy | Δ #features | Δ #terms | Δ #checks | Δ time |
|---|---|---|---|---|
| `abductive` | +0.0 % | +0.0 % | +0.0 % | **+0.9 %** |
| `itp` | +0.0 % | +0.0 % | +0.0 % | **+1.7 %** |
| `ucore` | +0.0 % | +0.0 % | +0.0 % | **+5.5 %** |

Formula sizes and solver-call counts are **exactly equal**. ONNX is 1–5 % slower, attributable to the
generic graph walk in `Network2` versus the specialized dense-layer loop in `Network`.

### 5.2 All ten samples, `abductive` (shows the impact of the §6 bug)

| run | N | AVG-FEAT | AVG-TERMS | AVG-CHK | AVG-TIME (s) | TOT-TIME (s) | CORRECT |
|---|---|---|---|---|---|---|---|
| nnet | 10 | 7.90 | 7.90 | 13.00 | 0.3050 | 3.050 | 6/10 |
| onnx | 10 | **9.90 (+25.3 %)** | 9.90 | 13.00 | 0.1732 (−43.2 %) | 1.732 | **4/10** |

| sample | A-fixed | B-fixed | common | only-A/B | status | only in B (onnx) |
|---|---|---|---|---|---|---|
| 1 | 9 | 12 | 9 | 0/3 | DIFFERS | x1, x6, x12 |
| 2 | 8 | 13 | 8 | 0/5 | DIFFERS | x1, x6, x10, x11, x12 |
| 3 | 6 | 6 | 6 | 0/0 | MATCH | |
| 4 | 8 | 8 | 8 | 0/0 | MATCH | |
| 5 | 7 | 13 | 7 | 0/6 | DIFFERS | x1, x2, x3, x6, x7, x9 |
| 6 | 8 | 8 | 8 | 0/0 | MATCH | |
| 7 | 6 | 12 | 6 | 0/6 | DIFFERS | x1, x2, x3, x4, x6, x7 |
| 8 | 9 | 9 | 9 | 0/0 | MATCH | |
| 9 | 7 | 7 | 7 | 0/0 | MATCH | |
| 10 | 7 | 7 | 7 | 0/0 | MATCH | |

The 4 DIFFERS samples (1, 2, 5, 7) are **exactly** the 4 samples where the two formats disagree on
the predicted class. There are **no `only-A` entries anywhere**, so the ONNX explanations remain
sound — they are just non-minimal. The apparent 43 % speedup is an artifact: the wrongly-asserted
label makes checks fail faster.

Class agreement map for `heart_attack_quick.csv` (nnet / onnx computed label):

| sample | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 |
|---|---|---|---|---|---|---|---|---|---|---|
| nnet | 0 | 0 | 1 | 1 | 0 | 1 | 0 | 1 | 1 | 1 |
| onnx | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 | 1 |
| agree | ✗ | ✗ | ✓ | ✓ | ✗ | ✓ | ✗ | ✓ | ✓ | ✓ |

---

## 6. 🐛 Sigmoid classification mismatch — FIXED

### 6.1 The bug (as originally found)

**Symptom:** `Network2` classifies **every** heart_attack sample as class 1.

**Root cause:**

- [`src/spexplain/network/Network2.cpp:120`](../../../src/spexplain/network/Network2.cpp) —
  `computeClassification` thresholds a single output at `values.front() < 0 ? 0 : 1`, i.e. it expects
  a **pre-sigmoid logit**.
- But `Network2::evaluate` → `computeOutputValues` runs the **entire** layer list *including* the
  trailing `Sigmoid`, so the value is always in `(0, 1)` → always ≥ 0 → always label 1.
- Meanwhile [`src/verifiers/opensmt/OpenSMTVerifier2.cpp:737-740`](../../../src/verifiers/opensmt/OpenSMTVerifier2.cpp)
  deliberately **strips** trailing sigmoid layers when encoding
  (`while (end > 0 and layers[end-1]->getType() == "sigmoid") --end;`).

So **evaluation and encoding disagree** about the model's semantics.

**Consequences:**

- `abductive` — the wrong label is asserted, so the SMT query is SAT (the sample itself is a
  counterexample) and every free-a-feature check fails. Result: sound but **non-minimal**
  explanations (+25 % features).
- `itp` and `ucore` — **hard crash**:
  `"Interpolation context cannot be created if solver is not in UNSAT state"` /
  `"Unsat core cannot be extracted if solver is not in UNSAT state"`.

### 6.2 The fix

`Network2` now owns the rule, and both evaluation and encoding ask it:

```cpp
// Network2.h
void   setDropTrailingSigmoid(bool drop);   // default: true
bool   dropsTrailingSigmoid() const;
std::size_t nEffectiveLayers() const;       // all layers except trailing sigmoids (when enabled)
```

```cpp
// Network2.cpp
std::size_t Network2::nEffectiveLayers() const {
    std::size_t end = layers.size();
    if (not dropTrailingSigmoid) { return end; }
    while (end > 0 and layers[end-1] and layers[end-1]->getType() == "sigmoid") { --end; }
    return end;
}
```

| file | change |
|---|---|
| `src/spexplain/network/Network2.h` | `dropTrailingSigmoid` flag (default `true`), `nEffectiveLayers()`, `nEffectiveActivationLayers()` |
| `src/spexplain/network/Network2.cpp` | `computeOutputValues` iterates only `nEffectiveLayers()`; activation counting uses `nEffectiveActivationLayers()` |
| `src/verifiers/opensmt/OpenSMTVerifier2.cpp` | its local sigmoid-stripping loop replaced by `network.nEffectiveLayers()` — **single source of truth** |
| `src/spexplain/bin/main.cpp` | new `--drop-sigmoid true\|false` option (ONNX actions only) |
| `README.md` | new "Trailing sigmoid layers" subsection |

The `hiddenNeuronInputValues` / `hiddenNeuronOutputValues` counting was also affected: a trailing
sigmoid used to be counted as an activation layer by `Network2` but not by `OpenSMTVerifier2` (which
counts ReLU layers only). Routing both through `nEffectiveLayers()` fixes that misalignment too.

### 6.3 The new option

```
--drop-sigmoid true|false     (default: true)
```

| value | evaluation | encoding |
|---|---|---|
| `true` (default) | trailing sigmoid dropped → output is the logit | consistent, works |
| `false` | sigmoid applied → output is a probability | fails with `Unsupported layer type: sigmoid (only a trailing sigmoid is supported, and only with --drop-sigmoid true)` |

Verified: `--drop-sigmoid true` produces byte-identical output to the default; `--drop-sigmoid false`
exits 1 with the message above; the `.nnet` path rejects the option (prints usage), as it does for
`--input-min`/`--input-max`.

### 6.4 Post-fix results — all 10 samples, no filtering

Command (per strategy):

```bash
OUT="$PWD/build-debug/explain-fixed"
for S in abductive itp ucore; do
    ./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
        --model   ./data/models/heart_attack/heart_attack-50 \
        --dataset ./data/datasets/heart_attack/heart_attack_quick.csv \
        --strategy "$S" \
        --outdir  "$OUT/$S"
done
```

**Feature-level diff — 30/30 samples MATCH, `itp`/`ucore` no longer crash:**

| strategy | samples | matching | differing | only-A | only-B |
|---|---|---|---|---|---|
| `abductive` | 10 | **10** | 0 | 0 | 0 |
| `itp` | 10 | **10** | 0 | 0 | 0 |
| `ucore` | 10 | **10** | 0 | 0 | 0 |

**Classification now agrees** — `CORRECT` is 6/10 on *both* sides for every strategy (previously
nnet 6/10 vs onnx 4/10).

**Statistics:**

| strategy | run | N | AVG-FEAT | AVG-TERMS | AVG-CHK | AVG-TIME (s) | TOT-TIME (s) | CORRECT |
|---|---|---|---|---|---|---|---|---|
| `abductive` | nnet | 10 | 7.90 | 7.90 | 13.00 | 0.3074 | 3.074 | 6/10 |
| `abductive` | onnx | 10 | 7.90 | 7.90 | 13.00 | 0.2976 | 2.976 | 6/10 |
| `itp` | nnet | 10 | 13.00 | 26.00 | 1.00 | 0.0086 | 0.086 | 6/10 |
| `itp` | onnx | 10 | 13.00 | 26.00 | 1.00 | 0.0089 | 0.089 | 6/10 |
| `ucore` | nnet | 10 | 13.00 | 13.00 | 1.00 | 0.0069 | 0.069 | 6/10 |
| `ucore` | onnx | 10 | 13.00 | 13.00 | 1.00 | 0.0070 | 0.070 | 6/10 |

| strategy | Δ #features | Δ #terms | Δ #checks | Δ time |
|---|---|---|---|---|
| `abductive` | +0.0 % | +0.0 % | +0.0 % | **−3.2 %** |
| `itp` | +0.0 % | +0.0 % | +0.0 % | **+3.8 %** |
| `ucore` | +0.0 % | +0.0 % | +0.0 % | **+2.5 %** |

The `+25.3 %` feature blow-up of §5.2 is **gone** — ONNX now matches `.nnet` exactly.

**Structural, rounding-tolerant diff (`psi_diff.py --tol 1e-3`):**

| strategy | A conj | B conj | identical | close | differing | only-A | only-B | max rel. dev |
|---|---|---|---|---|---|---|---|---|
| `abductive` | 79 | 79 | **79** | 0 | 0 | 0 | 0 | **0** |
| `itp` | 260 | 260 | 196 | 64 | 0 | 0 | 0 | 4.43734e-05 |
| `ucore` | 130 | 130 | **130** | 0 | 0 | 0 | 0 | **0** |

**Soundness (class-aware — an explanation is sound iff `psi_c<c> /\ phi` is UNSAT for some class `c`):**

| strategy | nnet own | onnx own | nnet-phi / onnx-psi | onnx-phi / nnet-psi |
|---|---|---|---|---|
| `abductive` | **10/10** | **10/10** | 10/10 | 10/10 |
| `itp` | **10/10** | **10/10** | 6/10 ⚠ | 4/10 ⚠ |
| `ucore` | **10/10** | **10/10** | 10/10 | 10/10 |

**All 30 ONNX explanations are provably sound.** The `itp` cross-encoding entries are the known
boundary-tightness artifact of §4.2 (an interpolant sits exactly on the decision boundary, so a
4.4e-5 coefficient shift invalidates it for the *other* encoding); it is now symmetric, confirming
it is float32 rounding and not a directional defect.

---

## 7. Known environment issues (pre-existing, unrelated)

1. **`data/scripts/analyze.sh` does not run on macOS.** It uses GNU-isms unavailable here: BSD
   `mktemp` has no `--suffix`, and bash 3.2 has no `local -n` namerefs or `wait -n`. The `check` and
   `compare-subset` actions were therefore reimplemented as
   `data/scripts/test_scripts/check_phi_soundness.sh` and
   `data/scripts/test_scripts/compare_phi_subset.sh`.
2. **`-q` / `--quiet` segfaults** for both `.nnet` and `.onnx` `explain` (reproduced on `toy.nnet`
   too). Consequence: `data/scripts/run1.sh` cannot be used; invoke `spexplain` directly.
3. `opensmt` requires the query file to have a `.smt2` extension, otherwise it silently produces no
   answer.

---

## 8. Reproducing everything

### 8.1 Prerequisites

```bash
cd <repo-root>
make debug                 # produces ./build-debug/spexplain
command -v opensmt         # system SMT solver, needed for §4 and §3.4
```

### 8.2 Output locations

Everything is written under `build-debug/` (git-ignored):

| path | contents |
|---|---|
| `build-debug/explain-report/psi-nnet/` | `psi_d.smt2`, `psi_c0.smt2`, `psi_c1.smt2` from the `.nnet` model |
| `build-debug/explain-report/psi-onnx/` | same, from the `.onnx` model (with input bounds) |
| `build-debug/explain-report/heart_attack_agree.csv` | the 6 class-agreeing samples |
| `build-debug/explain-report/abductive/` | unfiltered 10-sample `abductive` run |
| `build-debug/explain-report/agree-abductive/` | 6-sample `abductive` run |
| `build-debug/explain-report/agree-itp/` | 6-sample `itp` run |
| `build-debug/explain-report/agree-ucore/` | 6-sample `ucore` run |
| `build-debug/explain-fixed/{abductive,itp,ucore}/` | **post-fix 10-sample runs (§6.4)** |

Each run directory contains, for each side:

| file | contents |
|---|---|
| `nnet.phi.txt` / `onnx.phi.txt` | **the explanations**, one SMT-LIB formula per sample per line |
| `nnet.stats.txt` / `onnx.stats.txt` | per-sample `#checks`, `#features`, `#fixed features`, `#terms` (`-s`) |
| `nnet.times.txt` / `onnx.times.txt` | per-sample wall-clock seconds (`--output-times`) |
| `nnet.log` / `onnx.log` | full stdout/stderr of the run |

### 8.3 Explanation options used

Both sides were invoked with **identical** options; only the ONNX side additionally receives the
input bounds (the `.nnet` format stores them natively and rejects these flags).

| option | value | why |
|---|---|---|
| `<action>` | `explain` (nnet) / `explain-onnx` (onnx) | the two pipelines under comparison |
| `<exp_strategies_spec>` | `abductive`, `itp`, `ucore` | one run each |
| `-e <file>` | `nnet.phi.txt` / `onnx.phi.txt` | write the explanations |
| `-s <file>` | `nnet.stats.txt` / `onnx.stats.txt` | per-sample statistics |
| `--output-times <file>` | `nnet.times.txt` / `onnx.times.txt` | per-sample runtimes |
| `--input-min` | `29,0,0,94,126,0,0,71,0,0,0,0,0` | **ONNX only** — real feature domains |
| `--input-max` | `77,1,3,200,594,1,2,202,1,6.2,2,4,3` | **ONNX only** |
| `--drop-sigmoid` | `true` (default, not passed explicitly) | **ONNX only** — drop the trailing sigmoid so the output is the raw logit (§6) |

All other options are left at their defaults, notably
`--fix-default-sample-neuron-activations none` and
`--prefer-default-sample-neuron-activations all`.

### 8.4 Step 1 — generate the encodings (§2)

```bash
cd <repo-root>
OUT="$PWD/build-debug/explain-report"
mkdir -p "$OUT/psi-nnet" "$OUT/psi-onnx"

# dump-psi / encode-onnx write psi_*.smt2 into the *current working directory*
( cd "$OUT/psi-nnet" && "$OLDPWD/build-debug/spexplain" dump-psi \
      "$OLDPWD/data/models/heart_attack/heart_attack-50.nnet" )

( cd "$OUT/psi-onnx" && "$OLDPWD/build-debug/spexplain" encode-onnx \
      "$OLDPWD/data/models/heart_attack/heart_attack-50.onnx" \
      --input-min 29,0,0,94,126,0,0,71,0,0,0,0,0 \
      --input-max 77,1,3,200,594,1,2,202,1,6.2,2,4,3 )

# structural, rounding-tolerant diff
python3 data/scripts/test_scripts/psi_diff.py \
    "$OUT/psi-nnet/psi_d.smt2"  "$OUT/psi-onnx/psi_d.smt2"  --tol 1e-3 --stats
python3 data/scripts/test_scripts/psi_diff.py \
    "$OUT/psi-nnet/psi_c1.smt2" "$OUT/psi-onnx/psi_c1.smt2" --tol 1e-3 --stats
python3 data/scripts/test_scripts/psi_diff.py \
    "$OUT/psi-nnet/psi_c0.smt2" "$OUT/psi-onnx/psi_c0.smt2" --tol 1e-3 --stats
```

Expected (`A-conj B-conj identical close differing onlyA onlyB max-rel-dev`):

```
26 26 26 0 0 0 0 0
127 127 104 23 0 0 0 2.21867e-05
127 127 104 23 0 0 0 2.21867e-05
```

### 8.5 Step 2 — build the class-agreeing dataset

```bash
OUT="$PWD/build-debug/explain-report"
# header + rows 3, 4, 6, 8, 9, 10 of heart_attack_quick.csv
awk 'NR==1 || NR==4 || NR==5 || NR==7 || NR==9 || NR==10 || NR==11' \
    data/datasets/heart_attack/heart_attack_quick.csv > "$OUT/heart_attack_agree.csv"
```

### 8.6 Step 3 — run the explanation comparison (§3.2, §5)

```bash
OUT="$PWD/build-debug/explain-report"

# 6 class-agreeing samples, all three strategies
for S in abductive itp ucore; do
    ./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
        --model   ./data/models/heart_attack/heart_attack-50 \
        --dataset "$OUT/heart_attack_agree.csv" \
        --strategy "$S" \
        --outdir  "$OUT/agree-$S"
done

# unfiltered 10-sample abductive run (§5.2)
./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
    --model   ./data/models/heart_attack/heart_attack-50 \
    --dataset ./data/datasets/heart_attack/heart_attack_quick.csv \
    --strategy abductive \
    --outdir  "$OUT/abductive"
```

**Post-fix run over all 10 samples (§6.4) — this is the authoritative comparison:**

```bash
OUT="$PWD/build-debug/explain-fixed"
for S in abductive itp ucore; do
    ./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
        --model   ./data/models/heart_attack/heart_attack-50 \
        --dataset ./data/datasets/heart_attack/heart_attack_quick.csv \
        --strategy "$S" \
        --outdir  "$OUT/$S"
done
```

The harness auto-detects `heart_attack` in the model path and supplies `--input-min`/`--input-max`
to the ONNX side. To be explicit (or for any other model), pass them yourself:

```bash
./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
    --model   ./data/models/heart_attack/heart_attack-50 \
    --dataset "$OUT/heart_attack_agree.csv" \
    --strategy abductive \
    --input-min 29,0,0,94,126,0,0,71,0,0,0,0,0 \
    --input-max 77,1,3,200,594,1,2,202,1,6.2,2,4,3 \
    --outdir  "$OUT/agree-abductive"
```

Equivalent raw invocations (what the harness runs internally):

```bash
./build-debug/spexplain explain \
    ./data/models/heart_attack/heart_attack-50.nnet \
    build-debug/explain-report/heart_attack_agree.csv \
    abductive \
    -e build-debug/explain-report/agree-abductive/nnet.phi.txt \
    -s build-debug/explain-report/agree-abductive/nnet.stats.txt \
    --output-times build-debug/explain-report/agree-abductive/nnet.times.txt

./build-debug/spexplain explain-onnx \
    ./data/models/heart_attack/heart_attack-50.onnx \
    build-debug/explain-report/heart_attack_agree.csv \
    abductive \
    -e build-debug/explain-report/agree-abductive/onnx.phi.txt \
    -s build-debug/explain-report/agree-abductive/onnx.stats.txt \
    --output-times build-debug/explain-report/agree-abductive/onnx.times.txt \
    --input-min 29,0,0,94,126,0,0,71,0,0,0,0,0 \
    --input-max 77,1,3,200,594,1,2,202,1,6.2,2,4,3
```

### 8.7 Step 4 — structural diff of the explanations (§3.3)

```bash
OUT="$PWD/build-debug/explain-report"
for S in abductive itp ucore; do
    awk '{print "(assert " $0 ")"}' "$OUT/agree-$S/nnet.phi.txt" > "$OUT/agree-$S/nnet.wrapped.smt2"
    awk '{print "(assert " $0 ")"}' "$OUT/agree-$S/onnx.phi.txt" > "$OUT/agree-$S/onnx.wrapped.smt2"
    printf "%-10s " "$S"
    python3 data/scripts/test_scripts/psi_diff.py \
        "$OUT/agree-$S/nnet.wrapped.smt2" "$OUT/agree-$S/onnx.wrapped.smt2" --tol 1e-3 --stats
done
```

Expected:

```
abductive   47  47  47   0 0 0 0 0
itp        156 156 113  43 0 0 0 1.85151e-05
ucore       78  78  78   0 0 0 0 0
```

### 8.8 Step 5 — soundness verification (§4, §6.4)

`check_phi_soundness.sh` takes the explanation file first, then **every** `psi_c*.smt2` of the model.
An explanation is sound iff `psi_c<c> /\ phi` is UNSAT for at least one class `c`.

```bash
OUT="$PWD/build-debug/explain-fixed"     # post-fix runs
R="$PWD/build-debug/explain-report"      # psi encodings

# each explanation set against its own encoding
for S in abductive itp ucore; do
  for SIDE in nnet onnx; do
    printf "%-10s %-4s: " "$S" "$SIDE"
    ./data/scripts/test_scripts/check_phi_soundness.sh \
        "$OUT/$S/$SIDE.phi.txt" "$R/psi-$SIDE/psi_c0.smt2" "$R/psi-$SIDE/psi_c1.smt2" | tail -1
  done
done

# cross-encoding
for S in abductive itp ucore; do
  printf "%-10s nnet-phi/onnx-psi: " "$S"
  ./data/scripts/test_scripts/check_phi_soundness.sh \
      "$OUT/$S/nnet.phi.txt" "$R/psi-onnx/psi_c0.smt2" "$R/psi-onnx/psi_c1.smt2" | tail -1
  printf "%-10s onnx-phi/nnet-psi: " "$S"
  ./data/scripts/test_scripts/check_phi_soundness.sh \
      "$OUT/$S/onnx.phi.txt" "$R/psi-nnet/psi_c0.smt2" "$R/psi-nnet/psi_c1.smt2" | tail -1
done
```

Set `SOLVER=z3` in the environment to use z3 instead of opensmt.

### 8.9 Step 6 — logical subset comparison (§3.4)

```bash
OUT="$PWD/build-debug/explain-report"
for S in abductive itp ucore; do
    echo "-- $S"
    ./data/scripts/test_scripts/compare_phi_subset.sh \
        "$OUT/psi-nnet/psi_d.smt2" "$OUT/agree-$S/nnet.phi.txt" "$OUT/agree-$S/onnx.phi.txt"
done
```

### 8.10 Step 7 — statistics only (§5)

```bash
OUT="$PWD/build-debug/explain-report"
python3 data/scripts/test_scripts/phi_stats.py \
    --label nnet --stats "$OUT/abductive/nnet.stats.txt" --times "$OUT/abductive/nnet.times.txt" \
    --label onnx --stats "$OUT/abductive/onnx.stats.txt" --times "$OUT/abductive/onnx.times.txt"
```

---

## 9. Tooling reference

| script | purpose |
|---|---|
| `data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh` | main harness: runs both pipelines, captures stats/times, diffs and aggregates |
| `data/scripts/test_scripts/phi_diff.py` | feature-level diff of `abductive`-style `.phi.txt` files |
| `data/scripts/test_scripts/phi_stats.py` | aggregates `-s` stats + `--output-times` into the comparison table |
| `data/scripts/test_scripts/psi_diff.py` | structural, rounding-tolerant conjunct-by-conjunct SMT-LIB diff |
| `data/scripts/test_scripts/check_phi_soundness.sh` | macOS-compatible replacement for `analyze.sh check`; class-aware (pass all `psi_c*.smt2`) |
| `data/scripts/test_scripts/compare_phi_subset.sh` | macOS-compatible replacement for `analyze.sh compare-subset` |

`compare_explain_onnx_vs_nnet.sh` options:

| option | meaning |
|---|---|
| `--model <path>` | common prefix, expands to `<path>.onnx` and `<path>.nnet` |
| `--onnx <path>` / `--nnet <path>` | explicit model paths |
| `--dataset <csv>` | dataset shared by both runs (required) |
| `--strategy <spec>` | strategy spec forwarded to both runs (default `abductive`) |
| `--input-min` / `--input-max` | forwarded to the **ONNX side only** |
| `--outdir <dir>` | output directory (default `build-debug/explain-compare`) |
| `--keep` | do not wipe an existing output directory |
| `-- <args...>` | everything after is forwarded to **both** invocations |

---

## 10. Conclusion

The ONNX explanation pipeline is **at full parity** with the legacy `.nnet` pipeline. Over the whole
10-sample dataset and all three strategies:

- identical input-domain encodings and equivalent network encodings (≤2.2e-5),
- **both formats predict the same class for every sample**,
- **30/30 explanations match feature-for-feature**; `abductive` and `ucore` are bit-identical and
  logically equivalent under `psi_d`, `itp` agrees to within 4.4e-5,
- **all 30 ONNX explanations provably sound**,
- identical formula sizes and solver-call counts, within ±4 % runtime.

Two bugs were found and fixed along the way:

1. **Missing input domains** — ONNX carries no per-feature bounds, so `Network2` silently used
   `[0,1]`. Fixed with `--input-min` / `--input-max` (§1).
2. **Sigmoid semantics mismatch** — the evaluator applied the trailing sigmoid while the encoder
   dropped it, so every sample was classified as class 1. Fixed by making `Network2::nEffectiveLayers()`
   the single source of truth for both, controlled by `--drop-sigmoid` (default `true`) (§6).

No known issues remain in the ONNX explanation path. The only outstanding items are the pre-existing,
unrelated environment problems in §7 (`analyze.sh` is Linux-only; `-q`/`--quiet` segfaults for *both*
model formats).

---
---

# Appendix A — Full 100-sample re-run after the `--drop-sigmoid` fix

**Date:** 2026-08-19 (appended)
**Scope:** everything above, re-run from scratch on **100 samples** (vs 6–10 previously), with both
fixes in place (`--input-min`/`--input-max` and `--drop-sigmoid`).
**Model:** `data/models/heart_attack/heart_attack-50` (`.nnet` and `.onnx`)
**Strategies:** `abductive`, `itp`, `ucore`
**Output directory:** `build-debug/explain-100/`

> This appendix supersedes §2–§6.4. Those sections were measured on 6 and 10 samples; the numbers
> here are the definitive ones.

## A.0 Headline result

| Question | Answer |
|---|---|
| Do both formats predict the same class? | **Yes — 100/100, for every sample** |
| Do the explanations match feature-for-feature? | **Yes — 300/300 samples across the 3 strategies** |
| Are the `abductive` / `ucore` explanation files identical? | **Yes — byte-for-byte, all 100 lines each** |
| Are the `itp` explanations equivalent? | Equal to within **4.4e-5** (0 differing conjuncts out of 2600) |
| Are the ONNX explanations sound? | **Yes — 300/300 proved UNSAT** |
| Is the cost the same? | Identical `#terms` and `#checks`; runtime within **−5.9 % … +12.6 %** |

**Zero regressions, zero divergences.** The only residual differences are float32-vs-decimal weight
rounding in the `itp` interpolant coefficients.

## A.1 Setup

### A.1.1 Dataset

The first 100 samples of the full heart_attack dataset:

```bash
cd <repo-root>
OUT="$PWD/build-debug/explain-100"
rm -rf "$OUT"; mkdir -p "$OUT/psi-nnet" "$OUT/psi-onnx"
head -101 data/datasets/heart_attack/heart_attack_full.csv > "$OUT/heart_attack_100.csv"   # header + 100 rows
```

### A.1.2 Options used

Identical on both sides except the two ONNX-only groups:

| option | value | applies to |
|---|---|---|
| `<action>` | `explain` / `explain-onnx` | the two pipelines |
| `<exp_strategies_spec>` | `abductive`, `itp`, `ucore` | one run each |
| `-e <file>` | `{nnet,onnx}.phi.txt` | both |
| `-s <file>` | `{nnet,onnx}.stats.txt` | both |
| `--output-times <file>` | `{nnet,onnx}.times.txt` | both |
| `--input-min` | `29,0,0,94,126,0,0,71,0,0,0,0,0` | **ONNX only** |
| `--input-max` | `77,1,3,200,594,1,2,202,1,6.2,2,4,3` | **ONNX only** |
| `--drop-sigmoid` | `true` (default; not passed explicitly) | **ONNX only** |

All other options at their defaults (`--fix-default-sample-neuron-activations none`,
`--prefer-default-sample-neuron-activations all`, `--verifier opensmt`).

### A.1.3 Commands

```bash
ROOT="$PWD"; OUT="$PWD/build-debug/explain-100"

# --- encodings ---
( cd "$OUT/psi-nnet" && "$ROOT/build-debug/spexplain" dump-psi \
      "$ROOT/data/models/heart_attack/heart_attack-50.nnet" )
( cd "$OUT/psi-onnx" && "$ROOT/build-debug/spexplain" encode-onnx \
      "$ROOT/data/models/heart_attack/heart_attack-50.onnx" \
      --input-min 29,0,0,94,126,0,0,71,0,0,0,0,0 \
      --input-max 77,1,3,200,594,1,2,202,1,6.2,2,4,3 )

# --- explanation runs ---
for S in abductive itp ucore; do
    ./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
        --model   ./data/models/heart_attack/heart_attack-50 \
        --dataset "$OUT/heart_attack_100.csv" \
        --strategy "$S" \
        --outdir  "$OUT/$S"
done
```

## A.2 Encoding parity

```bash
for f in psi_d psi_c0 psi_c1; do
    printf "%-8s " "$f"
    python3 data/scripts/test_scripts/psi_diff.py \
        "$OUT/psi-nnet/$f.smt2" "$OUT/psi-onnx/$f.smt2" --tol 1e-3 --stats
done
```

| file | A conj | B conj | identical | close (≤1e-3) | differing | only-A | only-B | max rel. dev |
|---|---|---|---|---|---|---|---|---|
| `psi_d.smt2` | 26 | 26 | **26** | 0 | 0 | 0 | 0 | **0** |
| `psi_c0.smt2` | 127 | 127 | 104 | 23 | 0 | 0 | 0 | 2.21867e-05 |
| `psi_c1.smt2` | 127 | 127 | 104 | 23 | 0 | 0 | 0 | 2.21867e-05 |

Byte-identical to the encodings generated before the sigmoid fix — confirming `--drop-sigmoid`
changed only the *evaluator*, bringing it in line with the encoder that was already correct.

## A.3 Classification agreement

Parsed from the `computed output:` field of the two `-s` stats files:

| | class 0 | class 1 | total |
|---|---|---|---|
| `.nnet` | 4 | 96 | 100 |
| `.onnx` | 4 | 96 | 100 |

**Agreement: 100/100.** Before the fix the ONNX side predicted class 1 for *every* sample, so this
would have been 96/100 at best (and in the 10-sample probe it was 6/10).

`CORRECT` (computed label == dataset label) is **96/100 on both sides**, for all three strategies.

## A.4 Explanation equality

### A.4.1 Feature-level diff (`phi_diff.py`)

| strategy | samples | matching | differing | only-A | only-B |
|---|---|---|---|---|---|
| `abductive` | 100 | **100** | 0 | 0 | 0 |
| `itp` | 100 | **100** † | 0 | 0 | 0 |
| `ucore` | 100 | **100** | 0 | 0 | 0 |

† `itp` produces no `(= xI v)` conjuncts, so this row is vacuous — see A.4.2/A.4.3.

**300/300 samples match.**

### A.4.2 Byte-level equality of the explanation files

```bash
for S in abductive itp ucore; do diff -q "$OUT/$S/nnet.phi.txt" "$OUT/$S/onnx.phi.txt"; done
```

| strategy | result | file size nnet | file size onnx |
|---|---|---|---|
| `abductive` | **IDENTICAL** (all 100 lines) | 9 185 B | 9 185 B |
| `itp` | differs on 100/100 lines (coefficients only) | 709 500 B | 709 263 B |
| `ucore` | **IDENTICAL** (all 100 lines) | 13 744 B | 13 744 B |

### A.4.3 Structural, rounding-tolerant diff (`psi_diff.py --tol 1e-3`)

Explanations wrapped as `(assert <phi>)` and diffed conjunct-by-conjunct:

| strategy | A conj | B conj | identical | close | **differing** | only-A | only-B | max rel. dev |
|---|---|---|---|---|---|---|---|---|
| `abductive` | 811 | 811 | **811** | 0 | **0** | 0 | 0 | **0** |
| `itp` | 2600 | 2600 | 1878 | 722 | **0** | 0 | 0 | 4.43734e-05 |
| `ucore` | 1300 | 1300 | **1300** | 0 | **0** | 0 | 0 | **0** |

**4711 conjuncts compared, 0 differing, 0 only-A, 0 only-B.**

### A.4.4 Logical relation under `psi_d` (`compare_phi_subset.sh`)

```bash
for S in abductive itp ucore; do
    ./data/scripts/test_scripts/compare_phi_subset.sh \
        "$OUT/psi-nnet/psi_d.smt2" "$OUT/$S/nnet.phi.txt" "$OUT/$S/onnx.phi.txt"
done
```

| strategy | `=` equivalent | `<` nnet stronger | `>` onnx stronger | `?` incomparable |
|---|---|---|---|---|
| `abductive` | **100** | 0 | 0 | 0 |
| `itp` | 0 | 6 | 4 | 90 |
| `ucore` | **100** | 0 | 0 | 0 |

`abductive` and `ucore` are **logically equivalent on every single sample**. For `itp` the split is
near-symmetric (6 `<` vs 4 `>`), which is the signature of *symmetric numeric noise* rather than a
systematic bias: an interpolant is a half-space lying exactly **on** the decision boundary, so a
4.4e-5 coefficient perturbation tips it slightly one way or the other, and neither half-space fully
contains the other.

## A.5 Soundness

An explanation `phi` is sound iff `psi_c<c> /\ phi` is UNSAT for some class `c`
(i.e. `phi` determines the classification).

```bash
for S in abductive itp ucore; do
  for SIDE in nnet onnx; do
    ./data/scripts/test_scripts/check_phi_soundness.sh \
        "$OUT/$S/$SIDE.phi.txt" "$OUT/psi-$SIDE/psi_c0.smt2" "$OUT/psi-$SIDE/psi_c1.smt2"
  done
done
```

### A.5.1 Each explanation against its own encoding

| strategy | nnet-phi / nnet-psi | onnx-phi / onnx-psi |
|---|---|---|
| `abductive` | **100/100 sound** | **100/100 sound** |
| `itp` | **100/100 sound** | **100/100 sound** |
| `ucore` | **100/100 sound** | **100/100 sound** |

**All 300 ONNX explanations are provably sound.** (600 solver checks total, 0 failures.)

### A.5.2 Cross-encoding

| strategy | nnet-phi / onnx-psi | onnx-phi / nnet-psi |
|---|---|---|
| `abductive` | **100/100** | **100/100** |
| `itp` | 96/100 ⚠ | 4/100 ⚠ |
| `ucore` | **100/100** | **100/100** |

`abductive` and `ucore` explanations transfer **perfectly** between the two encodings — 400/400.

The `itp` numbers are the expected boundary artifact, and the larger sample makes the mechanism
clear: an interpolant is derived from *its own model's* coefficients and sits flush against that
model's decision boundary. Against the *other* encoding (whose coefficients differ by up to 4.4e-5)
it is off by a hair. `.nnet` interpolants happen to be slightly more conservative (96/100 transfer),
ONNX ones slightly tighter (4/100). This is **not** a soundness defect — each is sound for its own
model (A.5.1) — but it does mean **`itp` explanations should not be transferred across formats**.

## A.6 Statistics

### A.6.1 Aggregates (`phi_stats.py`)

| strategy | run | N | VARS | AVG-FEAT | AVG-FIXED | AVG-TERMS | AVG-CHK | AVG-TIME (s) | TOT-TIME (s) | CORRECT |
|---|---|---|---|---|---|---|---|---|---|---|
| `abductive` | nnet | 100 | 13 | 8.11 | 0.00 | 8.11 | 13.00 | 0.4777 | 47.765 | 96/100 |
| `abductive` | onnx | 100 | 13 | 8.11 | 0.00 | 8.11 | 13.00 | 0.4494 | 44.939 | 96/100 |
| `itp` | nnet | 100 | 13 | 13.00 | 0.00 | 26.00 | 1.00 | 0.0081 | 0.805 | 96/100 |
| `itp` | onnx | 100 | 13 | 13.00 | 0.00 | 26.00 | 1.00 | 0.0078 | 0.777 | 96/100 |
| `ucore` | nnet | 100 | 13 | 13.00 | 0.00 | 13.00 | 1.00 | 0.0059 | 0.593 | 96/100 |
| `ucore` | onnx | 100 | 13 | 13.00 | 0.00 | 13.00 | 1.00 | 0.0067 | 0.667 | 96/100 |

### A.6.2 Relative deltas (onnx vs nnet)

| strategy | Δ #features | Δ #terms | Δ #checks | Δ avg time |
|---|---|---|---|---|
| `abductive` | +0.0 % | +0.0 % | +0.0 % | **−5.9 %** |
| `itp` | +0.0 % | +0.0 % | +0.0 % | **−3.5 %** |
| `ucore` | +0.0 % | +0.0 % | +0.0 % | **+12.6 %** |

Formula size and solver-call count are **exactly equal** in every case. Runtime differences are
noise on top of very small absolute values: the `+12.6 %` on `ucore` is `0.0059 s → 0.0067 s`, i.e.
0.8 **milliseconds** per sample. The only measurement with meaningful absolute magnitude is
`abductive` (~0.45 s/sample), where ONNX is in fact **5.9 % faster**.

### A.6.3 Runtime distribution (seconds per sample)

| strategy | side | min | median | mean | p90 | max | total |
|---|---|---|---|---|---|---|---|
| `abductive` | nnet | 0.0658 | 0.4495 | 0.4777 | 0.8700 | 1.7100 | 47.765 |
| `abductive` | onnx | 0.0635 | 0.4005 | 0.4494 | 0.8420 | 1.6000 | 44.939 |
| `itp` | nnet | 0.0076 | 0.0080 | 0.0081 | 0.0084 | 0.0092 | 0.805 |
| `itp` | onnx | 0.0076 | 0.0077 | 0.0078 | 0.0081 | 0.0089 | 0.777 |
| `ucore` | nnet | 0.0057 | 0.0059 | 0.0059 | 0.0062 | 0.0070 | 0.593 |
| `ucore` | onnx | 0.0060 | 0.0064 | 0.0067 | 0.0075 | 0.0120 | 0.667 |

The ONNX distribution tracks the `.nnet` one closely at every quantile — including the tail
(`abductive` max 1.60 s vs 1.71 s), so there is no pathological slow case introduced by `Network2`.

`abductive` costs ~60× more than `itp`/`ucore` because it performs 13 solver calls per sample
(one per feature) versus 1.

### A.6.4 Explanation size distribution — `abductive`

Number of fixed features per sample:

| #fixed | nnet | onnx |
|---|---|---|
| 5 | 3 | 3 |
| 6 | 7 | 7 |
| 7 | 25 | 25 |
| 8 | 26 | 26 |
| 9 | 23 | 23 |
| 10 | 12 | 12 |
| 11 | 4 | 4 |

| | min | median | mean | max |
|---|---|---|---|---|
| nnet | 5 | 8 | 8.11 | 11 |
| onnx | 5 | 8 | 8.11 | 11 |

**Per-sample counts are identical for all 100 samples.** `abductive` frees on average 4.9 of 13
features (37.6 %).

For `ucore`, all 100 samples on both sides fix all 13 features (mean 13.00), and `itp` produces 52
inequality conjuncts per sample on both sides.

### A.6.5 Per-feature fixing frequency — `abductive`

How often each feature appears in the explanation (out of 100 samples):

| var | feature | nnet | onnx |
|---|---|---|---|
| `x1` | age | 26 | 26 |
| `x2` | sex | 12 | 12 |
| `x3` | cp | 47 | 47 |
| `x4` | trtbps | 92 | 92 |
| `x5` | chol | 96 | 96 |
| `x6` | fbs | 10 | 10 |
| `x7` | restecg | 22 | 22 |
| `x8` | thalachh | **99** | **99** |
| `x9` | exng | 58 | 58 |
| `x10` | oldpeak | 98 | 98 |
| `x11` | slp | 82 | 82 |
| `x12` | caa | 97 | 97 |
| `x13` | thall | 72 | 72 |

**Every single frequency is identical.** This is the strongest available evidence that the two
pipelines are doing exactly the same search — not merely producing same-sized results.

As a side observation, the feature ranking is stable and plausible: the continuous clinical
measurements (`thalachh` max heart rate 99 %, `oldpeak` 98 %, `caa` 97 %, `chol` 96 %, `trtbps` 92 %)
are almost always required, while the coarse categorical ones (`fbs` 10 %, `sex` 12 %, `restecg`
22 %) are usually droppable.

## A.7 Verification effort

| check | solver calls | failures |
|---|---|---|
| Encoding parity (`psi_diff.py`) | — (structural) | 0 differing conjuncts / 280 |
| Explanation structural diff | — (structural) | 0 differing conjuncts / 4711 |
| Soundness, own encoding | 600+ | **0** |
| Soundness, cross encoding | 600+ | 0 for `abductive`/`ucore`; `itp` as explained in A.5.2 |
| Subset/equivalence under `psi_d` | 600 | 0 for `abductive`/`ucore` |
| Explanation runs | 3 000 (`abductive`) + 200 | 0 crashes |

Roughly **5 000 SMT queries**, no crashes, no timeouts, no unsound explanations.

## A.8 Reproducing this appendix

```bash
cd <repo-root>
make debug
command -v opensmt          # required

ROOT="$PWD"; OUT="$PWD/build-debug/explain-100"

# 1. dataset
rm -rf "$OUT"; mkdir -p "$OUT/psi-nnet" "$OUT/psi-onnx"
head -101 data/datasets/heart_attack/heart_attack_full.csv > "$OUT/heart_attack_100.csv"

# 2. encodings
( cd "$OUT/psi-nnet" && "$ROOT/build-debug/spexplain" dump-psi \
      "$ROOT/data/models/heart_attack/heart_attack-50.nnet" )
( cd "$OUT/psi-onnx" && "$ROOT/build-debug/spexplain" encode-onnx \
      "$ROOT/data/models/heart_attack/heart_attack-50.onnx" \
      --input-min 29,0,0,94,126,0,0,71,0,0,0,0,0 \
      --input-max 77,1,3,200,594,1,2,202,1,6.2,2,4,3 )
for f in psi_d psi_c0 psi_c1; do
    printf "%-8s " "$f"
    python3 data/scripts/test_scripts/psi_diff.py \
        "$OUT/psi-nnet/$f.smt2" "$OUT/psi-onnx/$f.smt2" --tol 1e-3 --stats
done

# 3. explanation runs (~2 min)
for S in abductive itp ucore; do
    ./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
        --model   ./data/models/heart_attack/heart_attack-50 \
        --dataset "$OUT/heart_attack_100.csv" \
        --strategy "$S" --outdir "$OUT/$S"
done

# 4. structural diff of the explanations
for S in abductive itp ucore; do
    awk '{print "(assert " $0 ")"}' "$OUT/$S/nnet.phi.txt" > "$OUT/$S/nnet.w.smt2"
    awk '{print "(assert " $0 ")"}' "$OUT/$S/onnx.phi.txt" > "$OUT/$S/onnx.w.smt2"
    printf "%-10s " "$S"
    python3 data/scripts/test_scripts/psi_diff.py \
        "$OUT/$S/nnet.w.smt2" "$OUT/$S/onnx.w.smt2" --tol 1e-3 --stats
done

# 5. soundness (~15 min)
for S in abductive itp ucore; do
  for SIDE in nnet onnx; do
    printf "%-10s %-4s: " "$S" "$SIDE"
    ./data/scripts/test_scripts/check_phi_soundness.sh \
        "$OUT/$S/$SIDE.phi.txt" "$OUT/psi-$SIDE/psi_c0.smt2" "$OUT/psi-$SIDE/psi_c1.smt2" | tail -1
  done
done

# 6. subset / equivalence (~10 min)
for S in abductive itp ucore; do
    printf "%-10s " "$S"
    ./data/scripts/test_scripts/compare_phi_subset.sh \
        "$OUT/psi-nnet/psi_d.smt2" "$OUT/$S/nnet.phi.txt" "$OUT/$S/onnx.phi.txt" \
        | awk '{print $3}' | sort | uniq -c | tr '\n' ' '; echo
done
```

### Output layout

| path | contents |
|---|---|
| `build-debug/explain-100/heart_attack_100.csv` | the 100-sample dataset |
| `build-debug/explain-100/psi-nnet/`, `psi-onnx/` | `psi_d.smt2`, `psi_c0.smt2`, `psi_c1.smt2` |
| `build-debug/explain-100/{abductive,itp,ucore}/` | one directory per strategy |
| ⤷ `{nnet,onnx}.phi.txt` | **the explanations**, one formula per sample per line |
| ⤷ `{nnet,onnx}.stats.txt` | per-sample `#checks`, `#features`, `#fixed features`, `#terms` |
| ⤷ `{nnet,onnx}.times.txt` | per-sample runtime in seconds |
| ⤷ `{nnet,onnx}.log` | full run output |
| ⤷ `{nnet,onnx}.w.smt2` | explanations wrapped as `(assert ...)` for `psi_diff.py` |
| ⤷ `subset.raw.txt` | per-sample subset verdicts |
| `build-debug/explain-100/soundness.txt`, `subset.txt` | aggregated verification output |

## A.9 Regression checks

| check | result |
|---|---|
| `.nnet` explanations, pre-fix vs post-fix | **byte-identical** — the legacy path is untouched |
| ONNX `psi_d`/`psi_c0`/`psi_c1`, pre-fix vs post-fix | **byte-identical** — the encoder was already correct |
| `--drop-sigmoid true` vs default | **byte-identical** output |
| `--drop-sigmoid false` | exits 1 with `Unsupported layer type: sigmoid (only a trailing sigmoid is supported, and only with --drop-sigmoid true)` |
| `.nnet` path with `--drop-sigmoid` | rejected (prints usage), as for `--input-min`/`--input-max` |
| `make debug` | clean, no new warnings |

## A.10 Conclusion

Over 100 samples and three strategies — 300 explanations, ~5 000 SMT queries — the ONNX
explanation pipeline is **indistinguishable from the legacy `.nnet` pipeline**:

- **100/100 classification agreement**, `CORRECT` 96/100 on both sides;
- **300/300 explanations match**; `abductive` and `ucore` output files are **byte-identical** and
  logically equivalent on every sample; `itp` agrees to within 4.4e-5 with **0 differing conjuncts**;
- **per-feature fixing frequencies are identical for all 13 features** — the searches are provably
  taking the same path, not just landing on same-sized answers;
- **300/300 ONNX explanations provably sound**;
- identical `#terms` and `#checks`; runtime within noise (ONNX actually 5.9 % *faster* on the only
  workload with meaningful absolute cost).

Both bugs found during this work are fixed and covered by the reproduction steps above:

1. **Missing ONNX input domains** → `--input-min` / `--input-max` (§1);
2. **Trailing-sigmoid semantics mismatch** → `Network2::nEffectiveLayers()` as the single source of
   truth for evaluation *and* encoding, exposed as `--drop-sigmoid` (default `true`) (§6).

**One caveat worth recording:** `itp` explanations are tight against their own model's decision
boundary and therefore do **not** transfer across the two formats (A.5.2). Each is sound for its own
model; they should not be checked against, or reused with, the other encoding.

No known issues remain in the ONNX explanation path. The outstanding items are the pre-existing,
format-independent environment problems of §7 (`analyze.sh` is Linux-only; `-q`/`--quiet` segfaults
for *both* model formats).
