[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# SpExplAIn

Copyright 2024 - present Tomáš Kolárik <tomaqa@gmail.com>  
Copyright 2024 Martin Blicha <martin.blicha@gmail.com>  

SpExplAIn is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

SpExplAIn is a tool that automatically generates Space Explanations of neural network classification.
It gives provable guarantees of the behavior of the network in continuous areas of the
input feature space.
The tool leverages [OpenSMT2](https://github.com/usi-verification-and-security/opensmt) solver compatible
with a range of flexible Craig interpolation algorithms and unsatisfiable core generation.

To try the tool hands-on, from computing explanations to interpreting them, follow the [tutorial](tutorial).

## Download

Download from the [GitHub](https://github.com/usi-verification-and-security/spexplain) repository.

The preferred option is cloning the repository using Git.
You can also directly download a zipped archive with the source codes,
but it is not documented nor recommended.

To clone just to the most recent version, run

```
git clone --single-branch --depth 1 https://github.com/usi-verification-and-security/spexplain.git
```
This will *not* fetch submodules, which is currently just `./data/explanations`
(see `./data/README.md`)
that is not needed for running the tool nor for reproducing the experiments;
it only contains explanations of earlier experiments.
To also fetch all submodules, either run
```
git clone --recurse-submodules --single-branch --depth 1 https://github.com/usi-verification-and-security/spexplain.git
```
or clone/download the nested submodules manually.
Note that the submodules will download several GBs of data.


## Building from source

To build the system from the source code repository, you need a C++20
compliant compiler and the following tools, libraries and headers installed:

 - CMake, GNU Make, git (fetches OpenSMT2), bison, flex
 - gmp
 - for `make marabou` additionally wget and the Python 3 headers
   (Marabou downloads and builds its own versions of Boost, OpenBLAS and protobuf)

Running the tool by `./spexplain` (see [Running the tool](#running-the-tool)) requires Python 3 and,
depending on the action:

 - `interpret` and `list`: the Python packages numpy, scipy, matplotlib and pyparsing
 - `interpret` with the visualization `mathexpr` written as PDF: `pdflatex` with the LaTeX packages
   `standalone` and `varwidth`; showing the PDF (`--show`) uses `xdg-open`
 - `explain` with `--fix-default-sample-neuron-activations adversarial`: the Python packages pandas and torch

For example, on Debian 13, all of them are installed by
```
sudo apt-get install build-essential cmake make git bison flex libgmp-dev wget python3 python3-dev \
    python3-numpy python3-scipy python3-matplotlib python3-pyparsing python3-pandas python3-torch \
    texlive-latex-base texlive-latex-extra xdg-utils
```
If the Python packages are not available from your system, install them with `pip`,
e.g. into a virtual environment:
```
python3 -m venv .venv
. .venv/bin/activate
pip install numpy scipy matplotlib pyparsing pandas torch
```

The [Docker image](docker/tacas27) contains all of the above, with SpExplAIn already built.

SpExplAIn uses CMake as a build system generator.
We use a wrapper `Makefile` (i.e. GNU Make build system) that allows straightforward building of SpExplAIn.

To configure and build the project, run the following command inside the SpExplAIn directory:
```
$ make
```
This will run `cmake -B <build_dir>` and `cmake --build <build_dir>`.
The default `<build_dir>` is `build`, but it can be changed using the command line option `RELEASE_BUILD_DIR`, for example:
`make RELEASE_BUILD_DIR=build-release`.

If the command is not run for the first time, it only rebuilds the sources that are not up-to-date. In the case the `<build_dir>` was removed, it creates it again.

The option `CMAKE_FLAGS` may be used for any additional arguments to be passed to `cmake -B <build_dir>`.
Similarly,
the option `CMAKE_BUILD_FLAGS` may be used for any additional arguments to be passed to `cmake --build <build_dir>`.

### Changing build type

The default build type is Release and `make` is in fact just an alias to `make release`.
In order to build in Debug mode, use
```
$ make debug
```
In this case, options that are related to debug build type should use the `*DEBUG*` variants instead of `*RELEASE*`.
For example `DEBUG_BUILD_DIR=<build_dir>`.
The default `<build_dir>` in debug mode is `build-debug`.

In order to build with an additional verifier [Marabou](https://github.com/NeuralNetworkVerification/Marabou) that is more efficient for simplistic explanations such as abductive or interval explanations than OpenSMT2, run
```
make marabou
```
which will by default use `build-marabou` build directory.
To make the corresponding debug type, use `make marabou-debug`.

In order to also process networks in the [ONNX](https://onnx.ai) format (e.g., convolutional networks), run
```
make onnx
```
which builds a separate engine into the `build-onnx` build directory.
This engine is less trusted than the default one:
it is the frozen version used in the previous experiments with ONNX networks (the git tag `onnx-engine-v1`).
It requires [Protocol Buffers](https://protobuf.dev) (installed on the system, or it is built from source).
To make the corresponding debug type, use `make onnx-debug`.

In order to build all types, run `make all` (it does not include `make onnx`).

### Clearing the build

In case one for example needs to rebuild the project from scratch, it can be removed at first:
```
make clean
```
(or `make clean-release`) and then built again with `make` (or `make release`).
In the case of debug mode, one must run `make clean-debug` and `make debug`.
It works the same way with `*-marabou*` and `*-onnx*`.
To remove all, run `make clean-all`.

### Examples

Build with the default values:
```
make
```

Build Debug:
```
make debug
```

Build with Marabou:
```
make marabou
```

## Running the tool

For a guided introduction by examples, see the [tutorial](tutorial).

The tool is run as `./spexplain <action>` from the root of the repository.
The script `./spexplain` is the user interface that integrates the particular components of the tool;
it requires Python 3.
Running `./spexplain` with no arguments yields a help message:

```
USAGE: spexplain [--engine <path>] [<action>] <args> [<options>]
ACTIONS: [explain] dump-psi interpret list
    explain:    <nn_model_fn> <dataset_fn> [<exp_strategies_spec>]     Run the Explanation Computation Engine
    dump-psi:   <nn_model_fn>                                          Print the SMT-LIB encoding of the NN
    interpret:  [<explanation>...] <technique_spec> [<dataset>]        Run the Explanation Interpreter
    list:       vars|samples [<sel>...] <dataset> | classes <dataset>  List the variables, the samples, or the classes of the dataset
                layers <nn_model_fn>                                   List the layers of the model
Use 'spexplain <action> --help' to print the usage of a particular action.
...
```

The tool currently supports four actions:
`explain` (default), `dump-psi`, `interpret`, and `list`.

The actions `interpret` and `list` treat the misused options as follows:
an option that has no effect on the result (e.g., it would only add information) is ignored with a warning,
whereas an option that conflicts with other options, that cannot be carried out,
or that selects or limits what is processed or listed, is an error if it does not apply.

The actions `explain` and `dump-psi` are implemented by the Explanation Computation Engine,
the executable `engine` in the build directory (e.g., `./build/engine`),
which `./spexplain` runs with the forwarded arguments.
By default, `./build/engine` is used, or `./build-marabou/engine` if the verifier `marabou` is selected,
or `./build-onnx/engine` if the network is in the ONNX format;
this can be overridden by the option `--engine <path>` or the environment variable `SPEXPLAIN_ENGINE`.
Running `./spexplain explain --help` also prints the help message of the engine:

```
USAGE: ./build/engine [<action>] <args> [<options>]
ACTIONS: [explain] dump-psi
ARGS:
    explain:    <nn_model_fn> <dataset_fn> [<exp_strategies_spec>]
    dump-psi:   <nn_model_fn>
STRATEGIES SPEC: '<spec1>[; <spec2>]...'
Each spec: '<name>[ <param>[, <param>]...]'
Strategies and possible parameters:
         nop
   abductive
       trial: n <int> (default: n 4)
       ucore: interval, min, vars x<i>...
         itp: weak, strong, weaker, stronger, bweak, bstrong, aweak, astrong, aweaker, astronger, afactor <factor>, vars x<i>... (default: aweak, bstrong)
       slice: [vars] x<i>...
Default strategy: itp
VERIFIERS: opensmt
OPTIONS:
    ...
```

### `explain`

Generates an explanation for each particular input sample point in the provided dataset.
Alternatively,
it can also explain over already existing explanations
provided by the option `--input-explanations`.

The action requires the following arguments:
* `<nn_model_fn>`:
Filename to the model, that is, a neural network classifier. The main supported file format is [NNet](https://github.com/sisl/NNet) that specifies a fully connected feedforward ReLU network.
Networks in the [ONNX](https://onnx.ai) format (e.g., convolutional networks) are processed by the less trusted engine from `make onnx` (see above).
An example can be found in `./data/models/toy.nnet`.
* `<dataset_fn>`:
Filename to the dataset, that is, a collection of sample points with the expected classification outcomes,
in the CSV format with comma-separated values:
the first line is a header with the names of the features followed by the name of the classification
(it is always skipped, so it must be present),
and each further line is a sample point, that is, the values of the features
followed by the expected class (an integer from 0).
The column of the expected class must be present,
but it is only used to filter the sample points (`--filter-samples`) and in the statistics (`-s`);
the explanations are computed for the class computed by the network.
The names of the features are used by `interpret` (e.g., for the axes of `fspace`),
and so are the names of the classes if given in the header of the classification separated by `|`,
in the order of the classes, that is, starting with the name of the class 0.
An example can be found in `./data/datasets/toy.csv`,
with the header `Sweetness,Softness,Acidity,Unripe|Ripe`.
* `<exp_strategies_spec>`:
The specification of strategies in the format `<spec1>[; <spec2>]...`, that is, each strategy specification is separated by `;`.
The first strategy takes the dataset as its input,
and if multiple strategies are used, each consumes the previous output as its input.
Every particular strategy is specified by its name and optionally with parameters that are separated by `,`.
The default strategy is `itp`, and default parameters of strategies are specified in the brackets.

The neuron activations of the sample points can be fixed or used to guide the search
by the options `--input-fix-sample-neuron-activations <file>` and `--input-guide-sample-neuron-activations <file>`,
or by the options `--fix-default-sample-neuron-activations` and `--guide-default-sample-neuron-activations`.
The value `adversarial` of `--fix-default-sample-neuron-activations`
(processed by `./spexplain` itself)
selects the hybrid configuration:
it fixes the activations of neurons not affected by adversarial counterexamples,
using the file of the model in `./data/neuron_activations`
(e.g., `./data/neuron_activations/mnist/mnist_200x1.txt` for the model `./data/models/mnist/mnist_200x1.nnet`),
which is computed if it does not exist yet.
Computing the file requires [PyTorch](https://pytorch.org)
and the trained PyTorch model next to the NNet model with the extension `.pth`
(e.g., `./data/models/mnist/mnist_200x1.pth`);
the scripts that compute the file are in `./python_scripts/adversarial`.

### `dump-psi`

Generates the encoding of the provided network in the SMT-LIB format:
file `psi_d.smt2`,
which encodes the domains of particular features,
and files `psi_c*.smt2`,
which encode the constraint that the classification from the class `c*` cannot change
in the provided network.
These files are useful for the analysis of the generated explanations.
Refer to the directory `./data` for more information.

<!-- The produced encoding can be modified by the `--encoding-*` options. -->

The action requires the following arguments:
* `<nn_model_fn>`:
Same as in the `explain` action.

It is recommended to also apply the sed script `./data/scripts/polish_psi.sed` to the produced encodings (use the sed option `-i` to apply the changes inline on the files).

### `interpret`

Interprets stored explanations of the sample points
using an extraction technique,
e.g., `slice vars x1 x5`, `diam vars x1 x5`, or `norm` (optionally `min`, `inv`, or both, e.g., `norm min, inv`),
or directly using a visualization technique, `fspace`, `pixel`, or `mathexpr`.
An extraction technique can be followed by a visualization,
which is triggered by the option `--visualize`;
the default visualization is determined by the extraction technique
(`pixel` for `norm`, otherwise `fspace`).
The variables `x<i>` used by the techniques, together with the names of their features,
as well as the samples and the classes of a dataset can be listed by the action `list` (see below).

The action accepts the following arguments:
* `<explanation>...`:
Filename to the explanations in the SMT-LIB format (e.g., the default output file `phi.txt` of `explain`),
one explanation per line for each sample point of the dataset.
It can be omitted with a visualization technique, which then visualizes the sample points.
Multiple files (each at most once) are supported only with the visualizations `fspace` and `pixel`.
* `<technique_spec>`:
The technique and optionally its parameters in the format `<name>[ <param>[, <param>]...]`.
* `<dataset>`:
Same as in the `explain` action.
If given, it must contain as many sample points as there are explanations.
It is currently required by all implemented techniques.

All explanations are processed unless a subset of the sample points is selected
with the option `-i <sel>[,<sel>]...`,
where each `<sel>` is either a single sample point `<idx>`, or a range `[<idx>]-[<idx2>]`
(indexed from 1, the bounds default to the first and the last sample point, respectively),
and with the option `-n <int>`, which limits the number of processed sample points.

The extracted explanations are written into the file given by the option `-e <file>`;
if the option is not given, they are written into the file `tphi.txt` unless they are visualized,
with the kind of the points of `norm` added to the name (e.g., `tphi_min.txt`).
The file contains one line for each sample point,
with `<null>` for the sample points that are not processed.
With more kinds of points of `norm` (e.g., `norm min, inv`), each kind is written into its own file
with the kind added to the name, also of the file given by `-e` (e.g., `out_min.txt` and `out_inv.txt`).
Explanations `<null>` (e.g., when `explain` reached the time limit) are skipped.

Note that `explain` currently writes the explanations only of the processed sample points,
so with its options `-i`, `-n`, or `--filter-samples`, the explanations do not match the dataset
and `interpret` fails; use `explain` on the whole dataset instead.
The visualizations are written into files, one per sample point,
named according to the option `--output-visualization <file>`
(default: `timg.png` for `pixel` and `tphi.tex` for `mathexpr`)
where `_s<idx>` is added to the name (e.g., `timg_s3.png`) and the extension determines the format.
If only one sample point is processed, the option `--show` also shows the visualization.
With a visualization technique, the option `--samples-only` visualizes the sample points and ignores the explanations,
which is the same as omitting `<explanation>`;
with an extraction technique followed by `--visualize`, the extraction technique is ignored as well.
With the visualizations `fspace` and `pixel`, the sample point is shown together with the explanations
unless the option `--exclude-samples` is given (which conflicts with visualizing only the sample points).
Options that are not used by the given techniques are ignored with a warning,
whereas conflicting options are errors (see above),
and `--visualize` without a visualization technique does not take the following argument if it is not one
(e.g., `--visualize <dataset>`).
As in the engine, the option `-q` (`--quiet`) does not print the information to the standard output
(the warnings and the errors are still printed to the standard error),
and the option `-v` (`--verbose`) currently has no effect; if both are given, the last one applies.
With the visualizations `fspace` and `mathexpr`, the option `--visualization-domains <file>`
shows the values in the real units given by a CSV file with the columns `Min`, `Max`,
and optionally `Var` and `Feature` (e.g., `x1,Age,29,77`),
which map the normalized domain [0,1] of each listed feature linearly to [`Min`,`Max`]
(e.g., `./data/domains/heart_attack/heart_attack.csv`);
without `Var`, the rows are the features in their order,
and without `Feature`, the names of the features are those in the header of the dataset.
The explanations are still assumed to be normalized, and the extracted explanations (`-e`) stay normalized.

Currently, the following techniques are implemented:
* `slice`:
Computes the slice of the explanation in the two features given by the parameter `vars x<i> x<j>`
through the sample point, that is, the values of the sample point are substituted for all other features.
Only 2D slices are supported.
The extracted explanation contains the constraints of the slice
and the equalities of the other features to the sample point.
* `diam`:
Computes the maximal diameter of the slice (see `slice`) within the domains [0,1],
that is, the segment between the two farthest points of the slice,
using the parameter `vars x<i> x<j>` and optionally `max` (the default).
The extracted explanation is the segment, with the other features equal to the sample point.
* `norm`:
Computes the point of the explanation with the minimal (L2) norm using the parameter `min` (the default),
or with the maximal norm using the parameter `inv`;
with both parameters (e.g., `norm min, inv`), it computes both points in the order of the parameters.
Assumes that the domains of all features are normalized to [0,1].
* `fspace`:
Shows the slice of the explanation (see `slice`) in the 2D feature space of the two features
given by the parameter `vars x<i> x<j>`,
together with the sample point (unless `--exclude-samples`) and, if extracted by `diam`, the diameter.
After `slice` or `diam`, the features are those of the extraction technique.
With multiple explanations files, they are shown together in one plot (also after `slice` or `diam`),
labelled by their filenames without the common prefixes and suffixes delimited by `/` or `_`
(e.g., `guide-all/itp_aweak_bstrong.phi.txt` and `fix-all/itp_aweak_bstrong.phi.txt`
result in `guide-all` and `fix-all`)
and without a default basename `phi_` or `tphi_` at the start of their names
(e.g., `test.txt` and `tphi_inv.txt` result in `test` and `inv`),
or by the option `--labels <label>[,<label>]...`;
a single explanations file is labelled `Explanation`,
unless its name starts with a default basename (e.g., `tphi_min.txt` results in `min`);
the explanations `<null>` are omitted but still labelled.
The axes are labelled by the names of the features (see `--visualization-domains`),
and with `--visualization-domains`, they show the values in the real units.
With the option `--model <nn_model_fn>`, the plot also shows the decision regions of the classifier
in the 2D feature space through the sample point,
labelled by the names of the classes from the header of the dataset (see `explain`),
by the option `--class-labels <label>[,<label>]...` (which must give all the classes),
or by default `Class <c>`.
The plot is captioned by the number of the sample point (from 1, as with `-i`) and its class:
the class computed by the classifier with `--model`, followed by the expected class if they differ
(e.g., `Sample 2 class Low risk (expected: High risk)`),
otherwise only the expected class (e.g., `Sample 3 (expected class: Low risk)`),
using the names of the classes as above, or the numbers of the classes.
* `pixel`:
Shows points as gray-scale images side by side in one image,
assuming square images with a single channel (e.g., MNIST) and values in [0,1].
It shows the sample point first (unless `--exclude-samples`) and then the explanations, which must be points
(e.g., extracted by `norm`, including all its points);
with the parameter `center`, the sample point is placed in between the points (after the first half of them,
e.g., `norm min, inv` with `pixel center` shows the minimal point, the sample point, and the maximal point).
Multiple explanations files are supported as in `fspace`.
The sample point is titled `sample <idx>`, and the points by the labels of the explanations as in `fspace`
(`explanation` by default), with the kinds of the points of `norm` (only the kinds for a single explanations file
without `--labels`); the explanations `<null>` are shown empty.
* `mathexpr`:
Shows an explanation that is a conjunction of linear constraints,
e.g., an interval explanation (including a point extracted by `norm`), a segment, or a bounded plane,
as a human-readable math expression in the LaTeX syntax (also usable in Markdown math).
By default, it is the explicit form, a conjunction of the bounds of the features,
where the features with full domains [0,1] are omitted,
followed by the other linear constraints,
where an equality is solved for its last feature (e.g., `x_{5} = 4 x_{1} - 27`)
and an inequality is normalized to the first coefficient 1 (e.g., `x_{1} + 32.11 x_{10} \le 115.13`).
With the parameter `param`, a segment is shown in the parametric form,
`x_{i} = a + b\,t` for each feature with `t \in [0, 1]`,
which requires the equalities of the segment to be given explicitly (by `=`).
The numbers are rounded to at most 2 decimals, or to `digits <n>` decimals using the parameter `digits`,
and very small numbers are written in a semi-logarithmic form (e.g., `1.23 \cdot 10^{-5}`).
The parameter `vars x<i>...` restricts the expression to the given features,
including ranges of features `x<i>-x<j>` (e.g., `'mathexpr vars x1 x10-x20, digits 3'`);
the constraints that involve any of the given features are shown entirely.
The expression is written into a `.tex` file,
and with the extension `.pdf` (e.g., `--output-visualization tphi.pdf`), also rendered into a `.pdf` file.
The option `--show` shows the rendered PDF without storing it unless the extension is `.pdf`.
Rendering requires `pdflatex` and showing requires `xdg-open`.

Visualizing an extracted explanation using `--visualize`
is equivalent to first extracting it into a file and then visualizing the file directly
(e.g., using `pixel`).

The action requires the Python packages NumPy, SciPy, pyparsing, and Matplotlib.

### `list`

Lists the items of a dataset or of a model, one row per item, given by the target:
* `vars [<var>...] <dataset>`:
The variables `x<i>` of the dataset (indexed from 1, as in the parameter `vars` of the techniques and the strategies)
with the names of their features from the header of the dataset (`-` if missing) and their domains,
which are normalized to [0,1] unless the option `--domains <file>` gives them in the real units
by a CSV file of the same format as `--visualization-domains` of `interpret`
(e.g., `./data/domains/heart_attack/heart_attack.csv`), also with the names of the features.
The variables can be selected by `x<i>` or `x<i>-x<j>` before the dataset,
or by the option `--vars <var>[,<var>]...`:
```
$ ./spexplain list vars x3 x4 data/datasets/heart_attack/heart_attack_s100_scaled.csv
    Var  Feature         Domain
    x3   Chest pain      [0, 1]
    x4   Blood pressure  [0, 1]
```
* `samples [<sel>...] <dataset>`:
The samples of the dataset (indexed from 1) with the values of their variables and their expected classes,
shown by their names from the header of the dataset followed by their numbers (e.g., `High risk (0)`),
or by `Class <c>` by default.
The samples are selected as by the option `-i` of `interpret`, before the dataset, separated by spaces or commas,
and their number can be limited by the option `-n <int>`.
The variables, that is, the columns, are selected by the option `--vars <var>[,<var>]...`;
by default, at most the first 20 of them, followed by a column `...` if there are more (e.g., MNIST).
The second row of the header shows the domains of the variables;
with `--domains <file>`, the values and the domains are rescaled to the real units,
which is reported on the standard error since the values are not those of the dataset.
With the option `--model <nn_model_fn>`, the samples also show the classes computed by the classifier,
and the misclassified samples are marked by `!`:
```
$ ./spexplain list samples 1-3 98 data/datasets/heart_attack/heart_attack_s100_scaled.csv --vars x1-x4 \
    --model data/models/heart_attack/heart_attack_50x1.nnet --domains data/domains/heart_attack/heart_attack.csv
spexplain: info: The values are rescaled to the domains of data/domains/heart_attack/heart_attack.csv, they are not the values of the dataset
    Sample        x1      x2      x3         x4  Expected       Computed
            [29, 77]  [0, 1]  [0, 3]  [94, 200]
    1             57       1       0        150  High risk (0)  High risk (0)
    2             59       1       3        170  High risk (0)  Low risk (1)   !
    3             57       1       2        150  Low risk (1)   Low risk (1)
    98            52       1       0        108  Low risk (1)   High risk (0)  !
```
* `classes <dataset>`:
The classes of the dataset (indexed from 0) with their names from the header of the dataset (see `explain`),
or `Class <c>` by default, as in the legend of `fspace`.
The number of the classes is given by the classifier of the option `--model <nn_model_fn>`,
otherwise by the names in the header of the dataset;
otherwise, it is only estimated by the largest expected class of the dataset, which is reported on the standard error
(e.g., a class that is missing in the dataset is not counted if it is the last one).
* `layers <nn_model_fn>`:
The layers of the model in the NNet format with their sizes and activations,
and how the outputs determine the class:
a single output is a binary classification (class 1 iff the output is at least 0),
otherwise the class is the one of the maximal output.

The values of the datasets are assumed to be normalized to [0,1] (even with `--domains`),
so `vars` and `samples` print a warning if the values of the listed variables are outside their domains
in any sample of the dataset, not only in the listed ones.

If the standard output is a terminal, the output goes through a pager as in git:
`$PAGER`, or `less` with `LESS=FRSX` unless `LESS` is set
(it quits at once if the output fits on one screen, and it chops long lines instead of wrapping them);
the option `--no-pager` prints the output directly.

The action requires the Python package NumPy.

### Strategies and their parameters

* `nop`:
Just forwards the input to the output.
* `abductive`:
Mimics the computation of abductive explanations,
using a naive elimination of particular input features
following the order of features.
Only accepts interval-like explanations on input
(which a sample point is a special case).
* `trial`:
Trial-and-error approach similar to `abductive`,
but instead of entirely eliminating particular features,
tries to relax the bounds on the feature,
using at most `n` attempts of the binary search
(default: 4),
resulting in interval explanations.
Only accepts interval-like explanations on input.
* `ucore`:
Computes an unsatisfiable core.
Only compatible with verifiers that support this feature.
Accepts any explanation on input
but only has effect on conjunctive explanations.
    * `interval`:
    splits all equations into a pair of inequalities,
    each of which can be eliminated separately.
    * `min`:
    exhaustively reduces the core into an irreducible one.
    * `vars`:
    tries to eliminate only those conjuncts where the listed features appear.
* `itp`: Applies a Craig interpolation algorithm -
a combination of a propositional
(`bstrong => bweak`)
and arithmetic
(`astronger => astrong => afactor => aweak => aweaker`)
interpolation algorithm with different logical strengths.
Only compatible with verifiers that support this feature.
Accepts any explanation on input.
    * `weak`: shortcut for `aweak, bweak`
    * `strong`: shortcut for `astrong, bstrong`
    * `weaker`: shortcut for `aweaker, bweak`
    * `stronger`: shortcut for `astronger, bstrong`
    * `bweak`: uses dual McMillan propositional algorithm (-> disjunctions)
    * `bstrong` (default): uses McMillan propositional algorithm (-> conjunctions)
    * `aweak` (default): uses dual Farkas interpolant arithmetic algorithm;
    considerably expands the input
    * `astrong`: uses Farkas interpolant arithmetic algorithm;
    moderately expands the input
    * `aweaker`: uses dual decomposed Farkas interpolant arithmetic algorithm (can produce nested disjunctions);
    significantly expands the input
    * `astronger`: uses decomposed Farkas interpolant arithmetic algorithm (can produce nested conjunctions);
    often does not expand the input at all
    * `afactor <factor>`: uses arithmetic algorithm in between `astrong` and `aweak` parametrized by `<factor>` $\in [0,1)$
    * `vars`:
    only generalizes those conjuncts where the listed features appear
* `slice`:
Makes a cut in the feature space into the selected features,
fixing all the others to the values of the original sample point.

### Verifiers

The tool supports a few so-called verifiers as the computational symbolic engines:
* `opensmt`:
The native engine that supports all strategies.
Applies SMT solving in the `QF_LRA` SMT-LIB logic.
* `marabou`:
Efficient NN verifier that reasons over floating-point numbers.
It is suitable for interval-like explanations
(i.e., including abductive explanations).
Does not support unsatisfiable core extraction nor Crag interpolation.
Only available if built with a `make marabou*` rule.

### Options

The tool accepts numerous options as described in the help message.

For example, the explanations are by default written in the SMT-LIB format (`-S`), one explanation per line;
alternatively, in the form of intervals (`-I`) or bounds (`-B`).
A subset of the sample points can be selected with the option `-i`,
either a single sample point `<idx>`, or a range `[<idx>]-[<idx2>]`
(indexed from 1, the bounds default to the first and the last sample point, respectively).
Note that only the explanations of the processed sample points are written (`<null>` when reaching the time limit),
so with `-i`, `-n`, or `--filter-samples`, the explanations do not correspond to all sample points of the dataset
(see `interpret`).

### Examples

```
./spexplain ./data/models/toy.nnet ./data/datasets/toy.csv
```
Generates explanations into the default output explanation file `./phi.txt`
for the sample points in the dataset `./data/datasets/toy.csv`
using the toy network `./data/models/toy.nnet`
and the default strategy (`aweak, bstrong`).
The explanations are written in the SMT-LIB format (the default), one explanation per line.

```
./spexplain explain data/models/toy.nnet data/datasets/toy.csv abductive -V marabou -B -e data/explanations/toy.phi.txt
```
Generates explanations into `./data/explanations/toy.phi.txt`
using the strategy `abductive`
and the verifier `marabou`
(requires the build `build-marabou` from `make marabou`).
The explanations are in this case written in the bounds format (`-B`),
a human-readable format where each line corresponds to an equation of a feature to the corresponding sample value
and the explanations are separated by an empty line.

```
./spexplain data/models/toy.nnet data/datasets/toy.csv 'ucore interval, min' -R -e toy.phi.txt
```
Generates explanations into the file `./toy.phi.txt`
using the strategy `ucore` with parameters `interval` and `min`,
following the reversed order of features (`-R`) when relaxing the constraints.

```
./spexplain data/models/toy.nnet data/datasets/toy.csv 'itp aweaker, bstrong; ucore'
```
Generates explanations
using the strategy `itp` with parameters `aweaker` and `bstrong`,
followed by the strategy `ucore`,
which will eliminate some conjuncts.

```
./spexplain data/models/toy.nnet data/datasets/toy.csv 'trial n 3' -n2 -s stats.txt
```
Generates explanations for only the first two sample points in the dataset (`-n2`),
using the strategy `trial` with at most 3 attempts,
and also outputs the statistics about particular samples and explanations
into the file `./stats.txt`.

```
./spexplain dump-psi data/models/toy.nnet
```
Generates the encoding files `./psi_d.smt2` and `./psi_c*.smt2`
for the provided network in the SMT-LIB format.


## Publications

* Space Explanations of Neural Network Classification, CAV 2025 ([link](https://link.springer.com/chapter/10.1007/978-3-031-98682-6_15))

## Contact
If you have questions, bug reports, or feature requests, please refer to our [GitHub](https://github.com/usi-verification-and-security/spexplain/issues) issue tracker or send us an email.
