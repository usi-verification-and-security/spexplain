# explviz

Visualizing **space explanations** of neural-network decisions.

An explanation is an SMT2 formula — a conjunction of linear halfspaces over
`x1..xn` — describing a convex subspace of a model's input space. `explviz`
parses such a formula, reduces it to a 2D region, and draws it against the
model's class regions and decision boundary.

The library reads no data of its own: every function takes its file paths as
parameters, so you can point it anywhere.

## Install

Python >= 3.9.

```bash
python3 -m pip install -r explviz/requirements.txt       # runtime
python3 -m pip install -r explviz/requirements-dev.txt   # + pytest, for the tests
```

There is no `pyproject.toml`, so `explviz` is imported from whatever directory
you run in — put the project root on the path:

```bash
PYTHONPATH=. python3 my_script.py
PYTHONPATH=. python3 -m pytest explviz/tests -q
```

(If you would rather `pip install` it, add a `pyproject.toml` beside `explviz/`
with `packages = ["explviz"]`; nothing in the import graph prevents it.)

## Where the data comes from

Put a `data/` folder next to the package and nothing needs to be configured:

```
your-project/
    data/
        explanations/      <- .phi.txt files
        datasets/          <- .csv
        models/            <- .nnet
    explviz/
    my_script.py
```

Resolve paths through the helpers, and they are found relative to that folder:

```python
from explviz.paths import data_path, explanations_path

X, _ = load_points(data_path("datasets/points.csv"))          # data/datasets/points.csv
formulas = load_explanations(explanations_path("my_method/exp.phi.txt"))
clf = Classifier.from_nnet(data_path("models/model.nnet"))    # data/models/model.nnet
```

`explanations_path(...)` is just `data_path("explanations", ...)`.

### How the data root is resolved

Fresh on every call, first hit wins:

1. **`$EXPLVIZ_DATA`**, if set — an explicit override, always respected.
2. **`<explviz>/../data`**, if that directory exists — the layout above.
3. **The current working directory** — the fallback.

Step 2 is anchored on the package's own location, not the working directory, so
your scripts work **no matter where you run them from**. Use `EXPLVIZ_DATA` only
when the data lives somewhere other than beside the package:

```bash
EXPLVIZ_DATA=/path/to/some/data PYTHONPATH=. python3 my_script.py
```

If a file is missing, the error names which root was used and how to change it.
`explviz.paths.describe_root()` reports the same thing on demand.

**None of this is required.** The helpers are a convenience — absolute paths work
just as well:

```python
X, _ = load_points("/wherever/you/keep/points.csv")
```

## Snapshot vs projection

These are the two ways to get from an n-dimensional explanation to a picture,
and choosing between them is the main decision you make.

| | **snapshot** | **projection** |
|---|---|---|
| Question | "On the plane *through this sample point*, what does the explanation look like?" | "What is the *shadow* of the whole explanation on these two axes?" |
| Other variables | pinned to the sample point's values | free |
| Computed by | substitution + exact halfplane intersection | LP support sweep + convex hull |
| Cost | ~0.3 ms | ~0.3 s |
| Class regions | drawn (same plane) | never drawn — not comparable |
| Needs a classifier | yes, for the background | no |

A snapshot is always contained in the projection. A snapshot is **empty** when
the explanation does not contain its own sample point — `explviz` warns about
exactly this case rather than silently drawing nothing (see `validate.py`).

```python
Panel(point=x, axes=(9, 0), explanations=[...], mode="snapshot")   # or "projection"
```

## Quick start

```python
from explviz import (Classifier, Explanation, Panel, load_points,
                     load_explanations, plot_grid, save_figure)
from explviz.paths import data_path, explanations_path
from explviz.specs import HEART_ATTACK_SCALED as spec

X, _ = load_points(data_path("datasets/heart_attack_s100_scaled.csv"))
formulas = load_explanations(
    explanations_path("HA/50x1/fix-all/itp_aweak_bstrong.phi.txt"))
clf = Classifier.from_nnet(data_path("models/heart_attack_50x1.nnet"))

panel = Panel(
    point=X[71],
    axes=(9, 0),                       # 0-based feature indices: Oldpeak vs Age
    explanations=[Explanation(formulas[71], "Fix-all", "fix-all")],
    mode="snapshot",
)
fig = plot_grid([panel], spec, ncols=1, clf=clf)
save_figure(fig, "plots/demo")
```

Axis indices are **0-based**; `spec.axis_labels` supplies the names. `save_figure`
takes a path without a suffix and writes both `.png` and `.pdf`, creating
directories as needed.

## Backgrounds

The background shows the model's class regions on the plotted plane.

- `background="auto"` (the default) — `"contour"` for a snapshot, `"none"` for
  a projection. See the note below.
- `background="contour"` evaluates the model on a grid in **one batched forward
  pass** (300×300 ≈ 20 ms) and draws filled class regions plus a real
  decision-boundary contour.
- `background="scatter"` draws uniformly sampled points coloured by predicted
  class. Pass `seed=` for reproducibility.
- `background="none"` skips it.

**Projection mode never draws a background.** The class regions are the model
evaluated on the plane *through the sample point* — a slice. A projection is the
shadow of the whole polytope onto that plane. The two describe different things,
so overlaying them would suggest a correspondence that isn't there. Asking for
one anyway is ignored, with a warning.

A useful consequence: **projections need no classifier at all.**

```python
plot_explanation(exp, x, (9, 0), spec, mode="projection")   # no clf=
```

## Modules

| Module | Responsibility |
|---|---|
| `config.py` | `DatasetSpec`, `Explanation`, `Panel` — what to plot |
| `specs.py` | ready-made specs (`HEART_ATTACK_RAW`, `HEART_ATTACK_SCALED`) |
| `style.py` | `Style`: palette, markers, label map, `usetex` |
| `datasets.py` | load points, load `.phi.txt`, index maps |
| `paths.py` | where scripts look for data (`data_path`, `explanations_path`) |
| `parsing/smt2.py` | SMT2 formula → `Polytope` list |
| `geometry/polytope.py` | `Polytope` + `substitute()` (pins variables, drops columns) |
| `geometry/halfplanes.py` | exact 2D vertex enumeration |
| `geometry/region2d.py` | `region_2d(...)` — snapshot and projection |
| `models/` | `.nnet` reader and batched `Classifier` |
| `background.py` | class regions as a grid contour or a sampled scatter |
| `plot/panel.py` | **one Axes**: `draw_panel`, `plot_explanation`, `plot_comparison` |
| `plot/figure.py` | **many Axes**: `plot_grid`, `save_figure` |
| `plot/legend.py` | one deduplicated, ordered legend per figure |
| `validate.py` | does an explanation actually contain its sample point? |

`plot_explanation` is the one-element case of `plot_comparison`; both call
`draw_panel`, so there is a single drawing implementation.

## Bundled examples

Four reference scripts, each config-only — the drawing lives in the package:

| Example | Shows |
|---|---|
| `minimal_example` | the smallest useful script: one explanation, one axis pair |
| `ATVA26_methods_compare` | a 1×3 grid comparing five methods, with a shared legend |
| `single_capture_explanation` | one explanation drawn as a projection |
| `CAV25_presentation_figures` | three figure configurations sharing helper functions |

```bash
PYTHONPATH=. python3 -m explviz.examples.minimal_example
```

They resolve everything through `data_path` / `explanations_path`, so they follow
the `data/` layout above — explanation files are read from `data/explanations/`.
They expect the heart-attack data, under these names:

```
data/
    explanations/HA/<model>/<method>/<variant>.phi.txt
    datasets/heart_attack_s100_scaled.csv
    models/heart_attack_50x1.nnet
```

(`single_capture_explanation` and `CAV25_presentation_figures` reproduce older
figures and additionally read `data/examples/…` and two files at the data root,
`heartAttack.csv` and `heart_attack-50.nnet`.)

If your data lives elsewhere, point the root at it:

```bash
EXPLVIZ_DATA=/path/to/data PYTHONPATH=. python3 -m explviz.examples.ATVA26_methods_compare
```

Otherwise read them as reference and write your own — that is the expected use.
The test suite needs no data at all.

## Adapting to a different dataset or model

Nothing here is needed for the heart-attack data and a `.nnet` binary classifier,
but for the record:

| If you change… | Touch |
|---|---|
| dataset (features, names, bounds) | write your own `DatasetSpec`; `specs.py` is data, not logic, and nothing else imports it |
| model format (ONNX, TorchScript) | `Classifier.__init__` already accepts any `nn.Module`; add a constructor beside `from_nnet` |
| number of classes | binary is assumed in `models/classifier.py` (sigmoid, 0/1), `background.py`, `style.py` (`class_colors`/`class_labels`) and `plot/panel.py` (`levels=[0, 0.5, 1]`) |
| variable naming (not `x1`, `x2`, …) | `parsing/smt2.py::variables_in` only; `parse_formula` already takes explicit `var_names` |

Unaffected by any of the above: `config.py`, `datasets.py`, `paths.py`,
`validate.py`, all of `geometry/`, `parsing/smt2.py`, `plot/figure.py`,
`plot/legend.py`, and the whole test suite.

## Provenance

`explviz` replaces ~16 copy-pasted scripts in the `HaslerPolytopes` repo, each of
which re-declared the model wrapper, the sampling and the two plotting functions.
It was validated against that original implementation:

- the parser produces **bit-identical** matrices (max abs diff `0.0`, 328 polytopes);
- snapshot regions match the old equality-row + 1000-LP pipeline to **6.6e-15**,
  while being ~900× faster;
- projection regions match to **1.7e-9** using 360 angles instead of 1000.

Behaviour that deliberately differs from the original:

- the `is_partitioned` flag is gone — `mode` says what you mean, and pinning a
  variable absent from a formula is a harmless no-op;
- a hardcoded VeriX polygon for one panel is **not** reproduced; that explanation
  genuinely does not contain its sample point, and `explviz` warns instead;
- `text.usetex` is off unless you ask for it (`Style(usetex=True)`);
- an unknown method key falls through to itself instead of raising `KeyError`;
- a legend label with no artist is skipped instead of inserting a `None` handle.

