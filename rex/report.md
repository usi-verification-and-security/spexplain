# GTSRB models for ReX Activation/Deactivation

What was asked: train GTSRB CNNs, make them work with ReX's A/D method, compute
explanations, and record which neurons A/D switches on and off in a file format
matching `neuron_activations/mnist_cnn_deactives.txt`.

Everything new lives in `gtsrb_ad/`, except the neuron files, which go to
`neuron_activations/` next to the MNIST one.

---

## 0. Two things found before starting

**The reference file was empty.** `neuron_activations/mnist_cnn_deactives.txt`
is 0 bytes, so I could not read the format off it. Instead I found the consumer
and read the format out of the parser — see step 5. If that file was supposed
to have content, something went wrong when it was written.

**You already had most of the infrastructure.** Searching for GTSRB turned up:

- the dataset, unpacked, at `~/.cache/spexplain-data/gtsrb` (26,640 train /
  12,630 test);
- `spexplain/neuron-activation/spexplain/python_scripts/train_cnn_family.py`,
  which already trains GTSRB CNNs and exports ONNX under spexplain's exact
  conventions;
- `spexplain`'s `--input-fix-sample-neuron-activations <file>` flag, which is
  what the neuron file feeds.

So rather than write a trainer from scratch, I **import your
`train_cnn_family.py` and reuse its `Net`, `train`, `export` and
`verify_onnx`**. That guarantees the models obey the conventions
`explain-onnx` needs, which are easy to break silently:
input `[1,3,32,32]`, plain `ToTensor()` values in `[0,1]` with no
normalisation, only Conv/Relu/MaxPool/Flatten/Gemm nodes, no trailing softmax,
CSV samples in CHW row-major order.

---

## 1. Training the models

**File: `gtsrb_ad/train_gtsrb_family.py`**

You asked for two models below 9k ReLUs, one around 9k, and two around 15k. I
also mixed two-conv and three-conv topologies so the A/D mask can be compared
across depths. ReLU-neuron count is the right size knob here because it drives
SMT cost far more than parameter count does.

Architectures are given in your token grammar (`c<ch>[s<stride>][k<kernel>]`,
`p`, `f<units>`; a final Linear to 43 classes is implied):

| model | architecture | ReLU neurons | params | test acc | ONNX rel. diff |
|---|---|---|---|---|---|
| gtsrb_a | c8s2k5-c16-p-f64-fc | 3,936 | 41,499 | 0.855 | 1.27e-05 |
| gtsrb_b | c12s2k4-c24-p-c32-p-f128-fc | 7,396 | 32,207 | 0.8324 | 1.48e-05 |
| gtsrb_c | c16s2k4-c32-p-f128-fc | 9,136 | 158,555 | 0.8814 | 1.07e-05 |
| gtsrb_d | c12-p-c24-p-f96-fc | 14,952 | 90,163 | 0.8624 | 3.32e-05 |
| gtsrb_e | c20s2k5-c48-c40-p-f128-fc | 14,960 | 161,203 | 0.9164 | 2.79e-05 |

All five use ReLU after **every** conv, not just after the FC layers, as
requested, and every model has at least one max-pool stage.

Two decisions worth flagging:

- **Data caching.** GTSRB images are variable-sized JPEGs, and the PIL resize
  dominates epoch time. I materialised the resized tensors once into the
  scratchpad (`gtsrb_{train,test}_32.pt`). The transform is exactly
  `Resize((32,32)) + ToTensor()` — identical to what `train_cnn_family.raw_datasets`
  builds for gtsrb, so this is a pure speed-up, not a change of preprocessing.
  A first attempt with `num_workers=4` died with `BrokenPipeError` because
  spawn-based workers cannot start in a detached shell; `num_workers=0` fixed it.
- **12 epochs, Adam, lr 1e-3, batch 128.** Accuracies land in the 83–92% range.
  That is well short of state of the art for GTSRB (>98% with augmentation and
  bigger nets) but is deliberate: you asked for models that are not too big,
  and accuracy was not the objective. Every model is far above the 2.3% chance
  rate for 43 classes, so the explanations are explaining something real.

Outputs in `gtsrb_ad/models/`: `<name>.pt` (weights + tokens + metadata),
`<name>.onnx`, `manifest.csv`, plus `gtsrb_test_s100.csv` — 100 test samples in
spexplain CSV layout — and `gtsrb_test_s100_indices.json`, which records which
dataset rows those are.

**The ONNX export is verified**, not assumed: `verify_onnx` compares torch
against onnxruntime on 16 probes and the max relative difference is ~1e-5 for
every model.

---

## 2. Preparing the models for A/D

**File: `gtsrb_ad/interpretable_gtsrb.py`**

This is the ConvAD forward pass from Chanchal/Kelly/Chockler (arXiv:2510.01038),
implemented for these Conv/ReLU/MaxPool/Flatten/Linear stacks. A binary spatial
mask (1 = visible) travels alongside the activations, and at each checkpoint the
activations are multiplied by it, so any unit whose whole receptive field was
masked contributes nothing downstream. The mask advances as:

- **conv** — convolve the mask with a mean-value kernel of the same geometry;
  the result is the visible fraction of each unit's receptive field; threshold
  at `tau` (0 by default, as in the paper).
- **maxpool** — max-pool the mask. A pooled unit survives if any input in its
  window survives, which is exactly the `tau = 0` rule.
- **flatten + linear** — an FC unit reads the entire feature map, so its visible
  fraction is one global scalar.

### Two deviations from the authors' reference implementation

**1. Bias.** Their ResNet code pushes the mask through the conv *module*
(`bottleneck.conv1(mask)`). That is only sound because torchvision's ResNet
convs have `bias=False`. `nn.Conv2d` defaults to `bias=True`, and your
`train_cnn_family.py` uses the default — so copying their approach would add
the bias to the mask and the mask would never reach zero. I use `F.conv2d` with
an explicit ones-kernel and `bias=None`.

For the same reason their comment that zeroing the input pixels is "truly
faithful to the model" does **not** hold here: a fully masked receptive field
would still emit the bias. The explicit deactivation step after each conv is
what makes it faithful, and it is applied unconditionally.

**2. FC layers are essentially never deactivated, and that is correct.** At
`tau = 0` an FC unit dies only when the *entire* input is masked. This shows up
in the results as a `0` line for the last hidden layer on nearly every sample.
It is not a bug: A/D is a statement about spatial locality, and a fully
connected layer has none. This is the same point as section 5 of the LaTeX
report — the published algorithm is vacuous on fully connected layers.

---

## 3. Verifying the A/D implementation

**File: `gtsrb_ad/verify_ad.py`**

An A/D implementation that is subtly wrong would still produce plausible-looking
files, so I checked it four ways rather than eyeballing output:

1. **Theorem 4.1** — with an all-ones mask the AD pass must equal the plain
   network. Max absolute difference **0.0** on all five models.
2. **Empty mask** — with an all-zeros mask every unit must be dead. **0 active**
   in every layer of every model.
3. **Monotonicity** — shrinking the visible region can only deactivate more
   units, never fewer. Holds on all five.
4. **Receptive-field ground truth** — the real test. For sampled hidden units I
   compute the true receptive field by autograd (which input pixels have
   non-zero gradient), using absolute weights and zero bias so a zero gradient
   means *structurally disconnected* rather than numerically cancelled, and
   average-pooling in place of max-pooling since the two have the same support.
   A unit must be marked dead exactly when every pixel of that receptive field
   is masked. **96–144 unit/mask checks per model, all correct.**

Check 4 matters because it validates the mean-value-kernel propagation against
a value computed a completely different way. Checks 1–3 would all pass for
several wrong implementations.

As a side-benefit, the layer sizes recovered by walking the network sum exactly
to the `relu_neurons` count your `Net` computes analytically.

---

## 4. Running ReX

**Files: `gtsrb_ad/rex_script_gtsrb.py` (ReX glue), `gtsrb_ad/run_rex.py` (driver)**

ReX is driven in-process — patch `sys.argv`, call `get_all_args`, then
`explanation` — rather than through the CLI, because the sufficiency mask is
needed as an array, not a rendered PNG.

Two things to know about this setup:

**The patched ReX checkout is used, not the installed one.** `run_rex.py` puts
`rex/ReX` first on `sys.path`. The copy in the conda env still has the five
defects found earlier in this project (script `model_shape` treated as a
variable rather than called; `preprocess` invoked with the wrong arity; missing
`--mask_value` flag; a visualisation crash for any input that is not already
model-sized; `--analyze` reading a misspelled attribute). Without the patched
checkout, none of this runs.

**`--mask_value -1` is deliberate.** The AD glue recovers the mask by diffing
each mutant against the original input, because ReX's prediction function
receives mutated pixels rather than the mask itself. That recovery is only
exact if the masking value cannot occur in the data. GTSRB tensors live in
`[0,1]`, so `-1` can never collide. ReX's default of `0` would make every
genuinely black pixel look masked — the same leak measured earlier on the
authors' own scripts, where it shifted AD confidence by ~22%. Since the AD path
ignores pixel *values* entirely and uses only the mask, the fact that -1 is
out-of-domain costs nothing.

Settings: 100 samples per model — the same 100 images as
`models/gtsrb_test_s100.csv`, in the same order — at confidence threshold
gamma = 0.5.

| model | samples | explained | mean size (px) | mean size (%) | min-max px | mean conf | mean s/img |
|---|---|---|---|---|---|---|---|
| gtsrb_a | 100 | 100 | 57.1 | 5.6% | 10-280 | 0.757 | 2.0 |
| gtsrb_b | 100 | 100 | 91.3 | 8.9% | 10-570 | 0.718 | 2.4 |
| gtsrb_c | 100 | 100 | 52.4 | 5.1% | 10-270 | 0.784 | 2.2 |
| gtsrb_d | 100 | 100 | 63.2 | 6.2% | 10-400 | 0.772 | 2.1 |
| gtsrb_e | 100 | 100 | 64.3 | 6.3% | 10-810 | 0.805 | 2.3 |

Explanations are small: a few percent of the 1024 pixels. That is plausible for
GTSRB, where sign class is carried by a small, highly discriminative region, and
it is consistent with the small MSPS sizes reported for this dataset elsewhere.

Per-image explanation PNGs are in `gtsrb_ad/out/<model>_ad_g0.5/`, the input
crops in `gtsrb_ad/out/imgs/`, per-sample sizes in
`gtsrb_ad/out/<model>_ad_g0.5_sizes.csv`, and the masks themselves in
`<model>_ad_g0.5_masks.pt`.

`make_figure.py` produces a readable montage per model
(`out/<model>_explanations.png`; 32x32 is unreadable at native size, so it
upscales with nearest-neighbour so pixel boundaries stay honest). Spot-checking
`gtsrb_a`: the mask for the priority-road sign lands on the yellow diamond, the
70-limit sign on the digits, and sign 37 on the arrow. So the explanations are
landing on sign content rather than background, which is the qualitative
behaviour the AD paper reports.

---

## 5. Which neurons are active and deactive

**Files: `gtsrb_ad/neuron_report.py` (writer), `gtsrb_ad/validate_format.py` (checker)**

Per your answer, **deactive means the A/D mask is 0** at that unit — its whole
receptive field lies outside the ReX explanation mask, so ConvAD has switched
that part of the model off. This is the A/D notion, not the sign of the ReLU
pre-activation.

### The format, recovered from the consumer

Since the reference file was empty, I read the format out of
`spexplain/src/spexplain/framework/Config.cpp`,
`parseSampleNeuronActivationsImpl`, which is what
`--input-fix-sample-neuron-activations` calls:

```cpp
while (ifs >> parsedSampleIdx) {
    for (layerIdx = 1; layerIdx < nHiddenLayers + 1; ++layerIdx) {
        for (;;) {
            ifs >> parsedNode;
            if (parsedNode == 0) { break; }
            ...
        }
    }
}
```

So each record is:

```
<sample index>
<node> <node> ... 0      hidden layer 1
<node> <node> ... 0      hidden layer 2
...                      exactly nHiddenLayers lines
```

This matches your description. Details that the parser pins down and that are
easy to get wrong:

- **Both indices are 1-based.** `parsedSampleToIdx` does `sampleIdx - 1` and
  rejects anything `<= 0`; `parsedNodeToIdx` does `abs(node) - 1` and rejects
  `0`, since `0` is the terminator.
- **A layer with nothing to report must still be written as a bare `0`.** The
  parser reads exactly `nHiddenLayers` terminated lists per sample and would
  otherwise consume the next sample's index as a node. This is why the FC layer
  line is usually just `0`.
- **A negative index sets a `negated` flag** (`negated = node < 0`). I emit only
  positive indices and distinguish phase by filename, matching your two-file
  description.
- **Hidden layers are the ReLU layers**, in forward order: one per conv plus one
  per hidden FC. The final Linear to 43 logits is the output layer and is not
  counted. So models with 2 convs + 1 hidden FC have 3 lines per sample, and
  those with 3 convs + 1 hidden FC have 4.
- **Node order is NCHW row-major**, `index = c*H*W + h*W + w`, +1 for 1-based.
  That is what `nn.Flatten` and the ONNX `Flatten` spec produce, and
  `OnnxParser.cpp` works in NCHW, so the numbering lines up with spexplain's.

**No comments or headers.** I initially wrote a `#` provenance header and then
removed it: the parser is a bare `ifs >> long` loop, so any non-numeric byte
makes it throw `Expected sample index`. All metadata goes to a sidecar
`<model>_ad_meta.json` instead.

### Which layers are recorded

Only layers **immediately followed by a ReLU** are written. For these models
that means the convs and the hidden FC; the max-pool stages and the output
Linear are left out. This is also exactly spexplain's notion of a hidden layer,
so the default output is the parseable one.

`interpretable_gtsrb.py` now makes this explicit rather than implicit: the
forward pass records a mask at every unit-producing layer (`Conv2d`,
`MaxPool2d`, `Linear`), each tagged with `followed_by_relu`, and
`neuron_report.py` filters on that tag. `Flatten` is never recorded -- it
reshapes but creates no new units.

`--include-non-relu` writes all of them instead. For `gtsrb_b` that is 7 layers
rather than 4:

```
default:  Conv2d, Conv2d, Conv2d, Linear
          sizes [2700, 4056, 512, 128]
--include-non-relu:
          Conv2d, Conv2d, MaxPool2d (no ReLU), Conv2d,
          MaxPool2d (no ReLU), Linear, Linear (no ReLU)
          sizes [2700, 4056, 864, 512, 128, 128, 43]
```

Files from that mode get an `_all` suffix (`<model>_ad_deactives_all.txt`) so
they cannot be confused with the default ones, and their metadata carries
`"spexplain_compatible": false`.

**That flag makes the file unreadable by spexplain**, and silently so: its
`nHiddenLayers` counts only ReLU layers, so the extra lines would be consumed
as the *next* sample's index and the parse would go wrong rather than error
cleanly. Use it only for analysis outside the SMT pipeline.

### Readability

Samples are separated by a blank line. This is safe for spexplain: its parser
is `ifs >> long`, and `operator>>` skips all whitespace, so blank lines are
invisible to it. A record now reads:

```
1
1 2 3 4 5 ... 0      <- conv1
1 2 3 4 5 ... 0      <- conv2
13 14 15 16 ... 0    <- conv3
0                    <- FC (nothing deactivated)

2
...
```

`--no-blank-lines` restores the compact form. I regenerated all ten files and
diffed them against the previous versions ignoring blank lines: **identical**,
so this changed formatting only.

### Output files

In `neuron_activations/`, per model:

- `<model>_ad_deactives.txt` — units A/D switched **off**
- `<model>_ad_actives.txt` — units A/D left **on**
- `<model>_ad_meta.json` — layer count, layer sizes and shapes, sample count,
  per-layer means, provenance

The two txt files are exact complements: for each sample and layer, every index
in `1..layer_size` appears in exactly one of them.

**Sample index `k` in these files is row `k` of `models/gtsrb_test_s100.csv`,**
so the neuron files and the spexplain sample CSV can be used together directly.

| model | hidden layers | layer sizes | total ReLU | samples | mean deactive | % deactive | per-layer deactive |
|---|---|---|---|---|---|---|---|
| gtsrb_a | 3 | [1568, 2304, 64] | 3,936 | 100 | 2613 | 66.4% | 1273, 1340, 0 |
| gtsrb_b | 4 | [2700, 4056, 512, 128] | 7,396 | 100 | 4609 | 62.3% | 2178, 2377, 54, 0 |
| gtsrb_c | 3 | [3600, 5408, 128] | 9,136 | 100 | 6856 | 75.0% | 3139, 3717, 0 |
| gtsrb_d | 3 | [10800, 4056, 96] | 14,952 | 100 | 12188 | 81.5% | 9511, 2677, 0 |
| gtsrb_e | 4 | [3920, 6912, 4000, 128] | 14,960 | 100 | 8231 | 55.0% | 3148, 3935, 1148, 0 |

### What the numbers say

Two things stand out, and the second is the useful one.

**Deactivation is substantial**: 55–82% of all ReLU units are switched off on an
average sample. These are units that are provably zero under AD, so in the SMT
setting they are exactly the ones that could be fixed or eliminated.

**Deactivation falls with depth, in every model without exception:**

| model | layer 1 | layer 2 | layer 3 | layer 4 |
|---|---|---|---|---|
| gtsrb_a | 81% | 58% | 0% (FC) | - |
| gtsrb_b | 81% | 59% | 10% | 0% (FC) |
| gtsrb_c | 87% | 69% | 0% (FC) | - |
| gtsrb_d | 88% | 66% | 0% (FC) | - |
| gtsrb_e | 80% | 57% | 29% | 0% (FC) |

This is receptive-field growth, not an artefact. A layer-1 unit sees a 3x3 or
5x5 patch, so a small explanation mask easily misses it entirely; by layer 3 a
unit sees most of the image and is almost always touched by something visible.
By the FC layer the receptive field is everything and nothing is ever
deactivated.

It also explains why the totals are not ordered by model size. `gtsrb_d` is the
most deactivated (81.5%) despite not being the largest, because its stride-1
3x3 first conv gives 10,800 units with tiny receptive fields. `gtsrb_e` is the
least (55.0%) at almost the same ReLU count, because its 5x5 stride-2 first
conv plus two further convs grows receptive fields fast. **What drives
deactivation is receptive-field size relative to the explanation, not model
size.**

For the SMT plan this is directly actionable: the neurons you can fix are
concentrated in the early conv layers -- which happen to be the layers with the
most units, so that is the favourable case. But it also means A/D will buy you
nothing on the FC stage, and progressively less as you go deeper.

One sample in `gtsrb_e` has *zero* deactivated neurons: its explanation is
spread out enough that every unit's receptive field catches at least one
visible pixel. The file records this correctly as four `0` lines.

### Format validation

`validate_format.py` re-implements the C++ parser in Python, token for token,
and additionally range-checks every index against its layer size, checks the
one-layer-per-line convention, and (with `--complement`) checks that the actives
and deactives files partition `1..layer_size` exactly.

All ten files pass:

```
gtsrb_a  100 samples, 3 layers each   complement OK -- 300/300 sample-layers partition exactly
gtsrb_b  100 samples, 4 layers each   complement OK -- 400/400
gtsrb_c  100 samples, 3 layers each   complement OK -- 300/300
gtsrb_d  100 samples, 3 layers each   complement OK -- 300/300
gtsrb_e  100 samples, 4 layers each   complement OK -- 400/400
```

This is a re-implementation rather than the real parser, so on its own it would
only prove the files are self-consistent. I therefore also checked the two
things it cannot: the layer count and the layer sizes, straight from the C++.

`Config.cpp` settles both:

```cpp
std::size_t Framework::Config::nHiddenLayers() const {
    std::size_t reluLayers = 0;
    for (auto const & layerPtr : getNetwork2().getLayers())
        if (layerPtr and layerPtr->getType() == "relu") { ++reluLayers; }
    return reluLayers;
}
```

so `nHiddenLayers` is exactly the number of ReLU layers, and
`hiddenLayerSize(layer)` returns that ReLU layer's `getOutputSize()` — the
flattened `C*H*W`. Both match what `interpretable_gtsrb.py` derives by walking
the network, so the per-sample line count and the index ranges are right.

I also ran the real binary on the exported model:

```
$ spexplain read-onnx models/gtsrb_a.onnx
... Conv, Relu, Conv, Relu, MaxPool, Flatten, Gemm, Relu, Gemm
ONNX is parsed successfully!
Network2 built with 9 layers.
```

All nine nodes are the supported kinds and three of them are ReLU, matching the
three lines per sample this model's files use. Note that the "9 layers" figure
counts every layer, not hidden ones — it is not `nHiddenLayers`.

---

## Files created

| path | what |
|---|---|
| `gtsrb_ad/train_gtsrb_family.py` | trains the 5 models, exports `.pt` + `.onnx` + manifest + sample CSV |
| `gtsrb_ad/interpretable_gtsrb.py` | ConvAD forward pass; `selftest()` for Theorem 4.1 |
| `gtsrb_ad/verify_ad.py` | the four correctness checks |
| `gtsrb_ad/rex_script_gtsrb.py` | ReX `--script` glue (model chosen by `GTSRB_MODEL`) |
| `gtsrb_ad/run_rex.py` | runs ReX over N samples, saves masks + sizes |
| `gtsrb_ad/neuron_report.py` | writes the actives/deactives files (`--include-non-relu`, `--no-blank-lines`) |
| `gtsrb_ad/validate_format.py` | re-implements spexplain's parser to check them |
| `gtsrb_ad/make_figure.py` | montage of inputs / AD masks / explanations |
| `gtsrb_ad/make_report_tables.py` | regenerates this report's tables from artifacts |
| `gtsrb_ad/run_all.sh` | the whole pipeline |
| `gtsrb_ad/models/` | `.pt`, `.onnx`, `manifest.csv`, `gtsrb_test_s100.csv` |
| `gtsrb_ad/out/` | explanation PNGs, mask tensors, per-sample size CSVs |
| `gtsrb_ad/logs/` | training and per-model ReX logs |
| `neuron_activations/gtsrb_*_ad_{actives,deactives}.txt` | the requested files |
| `neuron_activations/gtsrb_*_ad_meta.json` | sidecar metadata |

## Reproducing

```bash
PY=/Users/labbaf/miniconda3/envs/rex/bin/python
cd /Users/labbaf/Documents/rex/gtsrb_ad
$PY train_gtsrb_family.py --epochs 12     # needs the cached tensors; see script
$PY verify_ad.py gtsrb_a gtsrb_b gtsrb_c gtsrb_d gtsrb_e
./run_all.sh                              # ReX + neuron files + validation
$PY make_report_tables.py
```

## One process note

`logs/run_all.log` ends with a shell error. That is my fault and not a result
problem: I edited `run_all.sh` to add the complement check *while bash was still
executing it*, and bash reads scripts incrementally, so it resumed at a byte
offset inside the inserted text. All five models had already completed by then
(100/100 each, all files written); only the trailing validation block was
garbled, and I ran it separately afterwards. The script on disk is correct and
runs cleanly from scratch.

## Caveats and what I would do next

- **Accuracy is 83–92%,** not tuned. Fine for explanation work, but do not quote
  these as GTSRB results.
- **`tau = 0` everywhere,** matching the paper. This is the strictest setting and
  the framework's only hyperparameter is otherwise unexplored; `neuron_report.py`
  and the wrapper both take `--tau`, so sweeping it is a one-liner.
- **The FC layer is almost always fully active.** Expected, explained above, but
  it means the deactives file carries no information about the FC stage. If you
  want fixable FC neurons you need the ReLU-phase notion instead, or one of the
  fully-connected A/D analogues from section 5 of the LaTeX report.
- **The files have not been round-tripped through spexplain itself.** The
  format is verified against the parser source and the ONNX loads cleanly, but
  I did not run a full `explain-onnx` with
  `--input-fix-sample-neuron-activations`, so the end-to-end path is unproven.
  That is the one check I would do before a long solver run.
- **The 100 samples are a fixed random draw** (seed 0) from the test split, not
  filtered to correctly-classified images. Some will be misclassified, and ReX
  explains the predicted class, not the true one. If you want only correct
  predictions, filter on the `label` column in the sizes CSV.
