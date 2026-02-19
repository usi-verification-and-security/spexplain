[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

# SpEXplAIn

Copyright 2024 - present Tomáš Kolárik <tomaqa@gmail.com>  
Copyright 2024 Martin Blicha <martin.blicha@gmail.com>  

SpEXplAIn is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.

SpEXplAIn is a tool that automatically generates Space Explanations of neural network classification.
It gives provable guarantees of the behavior of the network in continuous areas of the
input feature space.
The tool leverages [OpenSMT2](https://github.com/usi-verification-and-security/opensmt) solver compatible
with a range of flexible Craig interpolation algorithms and unsatisfiable core generation.

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
that is not needed for running the tool
but contains reference explanations.
To also fetch all submodules, either run
```
git clone --recurse-submodules --single-branch --depth 1 https://github.com/usi-verification-and-security/spexplain.git
```
or clone/download the nested submodules manually.
Note that the submodules will download several GBs of data.


## Building from source

To build the system from the source code repository, you need a C++20
compliant compiler and the following libraries and headers installed:

 - gmp

SpEXplAIn uses CMake as a build system generator.
We use a wrapper `Makefile` (i.e. GNU Make build system) that allows straightforward building of SpEXplAIn.

To configure and build the project, run the following command inside the SpEXplAIn directory:
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

In order to build all types, run `make all`.

### Clearing the build

In case one for example needs to rebuild the project from scratch, it can be removed at first:
```
make clean
```
(or `make clean-release`) and then built again with `make` (or `make release`).
In the case of debug mode, one must run `make clean-debug` and `make debug`.
It works the same way with `*-marabou*`.
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

Using the `./build` build directory, running `./build/spexplain` with no arguments yields a help message:

```
USAGE: ./build/spexplain [<action>] <args> [<options>]
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

The tool currently supports two actions:
`explain` (default), and `dump-psi`.

### `explain`

Generates an explanation for each particular input sample point in the provided dataset.
Alternatively,
it can also explain over already existing explanations
provided by the option `--input-explanations`.

The action requires the following arguments:
* `<nn_model_fn>`:
Filename to the model, that is, a neural network classifier. The only supported file format is currently [NNet](https://github.com/sisl/NNet) that specifies a fully connected feedforward ReLU network.
An example can be found in `./data/models/toy.nnet`.
* `<dataset_fn>`:
Filename to the dataset, that is, a collection of sample points, optionally with the expected classification outcome, in the CSV format.
An example can be found in `./data/datasets/toy.csv`.
* `<exp_strategies_spec>`:
The specification of strategies in the format `<spec1>[; <spec2>]...`, that is, each strategy specification is separated by `;`.
The first strategy takes the dataset as its input,
and if multiple strategies are used, each consumes the previous output as its input.
Every particular strategy is specified by its name and optionally with parameters that are separated by `,`.
The default strategy is `itp`, and default parameters of strategies are specified in the brackets.

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

### Examples

```
./build/spexplain ./data/models/toy.nnet ./data/datasets/toy.csv
```
Generates explanations into the default output explanation file `./phi.txt`
for the sample points in the dataset `./data/datasets/toy.csv`
using the toy network `./data/models/toy.nnet`
and the default strategy (`aweak, bstrong`).
The explanations are in this case written in the SMT-LIB format.

```
./build-marabou/spexplain explain data/models/toy.nnet data/datasets/toy.csv abductive -e data/explanations/toy.phi.txt
```
Generates explanations into `./data/explanations/toy.phi.txt`
using the strategy `abductive`
and the build `build-marabou` (from `make marabou`)
where the verifier `marabou` is used implicitly when possible.
The explanations are in this case written in a human-readable format
where each line corresponds to an equation of a feature to the corresponding sample value.

```
./build/spexplain data/models/toy.nnet data/datasets/toy.csv 'ucore interval, min' -RS -e toy.phi.txt
```
Generates explanations into the file `./toy.phi.txt`
in the SMT-LIB format (`-S`)
using the strategy `ucore` with parameters `interval` and `min`,
following the reversed order of features (`-R`) when relaxing the constraints.

```
./build/spexplain data/models/toy.nnet data/datasets/toy.csv 'itp aweaker, bstrong; ucore'
```
Generates explanations
using the strategy `itp` with parameters `aweaker` and `bstrong`,
followed by the strategy `ucore`,
which will eliminate some conjuncts.

```
./build/spexplain data/models/toy.nnet data/datasets/toy.csv 'trial n 3' -n2 -s stats.txt
```
Generates explanations for only the first two sample points in the dataset (`-n2`),
using the strategy `trial` with at most 3 attempts,
and also outputs the statistics about particular samples and explanations
into the file `./stats.txt`.

```
./build/spexplain dump-psi data/models/toy.nnet
```
Generates the encoding files `./psi_d.smt2` and `./psi_c*.smt2`
for the provided network in the SMT-LIB format.


## Publications

* Space Explanations of Neural Network Classification, CAV 2025 ([link](https://link.springer.com/chapter/10.1007/978-3-031-98682-6_15))

## Contact
If you have questions, bug reports, or feature requests, please refer to our [GitHub](https://github.com/usi-verification-and-security/spexplain/issues) issue tracker or send us an email.
