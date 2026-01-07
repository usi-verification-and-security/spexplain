# Experiment scripts

This folder contains several scripts to automate and simplify experimentation with SpEXplAIn.

## `run1.sh`

```
USAGE: ./run1.sh <output_dir> <exp_strategies_spec> [<name>] [reverse] [<max_samples>] <args>...
```
Except of these, it also accepts several environment variables, described below.

This is a proxy script for the `spexplain` executable that automatically uses some options suitable for running the experiments (e.g. `--quiet`) and adjusts the destination directory `<output_dir>` for the generated explanations based on the parameters and outputs most of the available data into files (including e.g. statistics, time splits, etc.).
It only runs one experiment, that is, produces one output file with explanations (along with statistics etc.), not multiple.

### `<output_dir>`

The target directory name for the explanations.
Some arguments below may modify this path.
The path `<output_dir>` also determines the input model and dataset, as specified in `spec/models_datasets`.

### `<exp_strategies_spec>`

Specification of strategies - same as in `spexplain`

### `<name>`

Optional name that describes the used strategies and that will be used as the basename for the generated files.
If omitted, it will be generated automatically from `<exp_strategies_spec>`.

### `reverse`

If specified, reverses the order of features (refers to option `-R` of `spexplain`).
Adds `reverse` into `<output_dir>`.

### `<max_samples>`

If specified, computes explanations only for randomly selected `<max_samples>` samples (refers to options `-r` and `-n` of `spexplain`).
Modifies `<output_dir>`.

### `<args>...`

Any other arguments that will be forwarded to `spexplain`.

### Environment variables

* `CMD`: sets the `spexplain` executable (default: `../../build/spexplain`)
* `TIMEOUT`: overall timeout for the execution (default: `30m`); the duration is a number with an optional suffix: `s` for seconds (default), `m` for minutes, `h` for hours or `d` for days
* `SRC_EXPERIMENT`: sets already computed explanations with a given name (corresponding to `<name>` at that time) as the input, instead of using the original sample points (refers to option `-E` of `spexplain`)

### Examples

In directory `data/`:

```
./scripts/run1.sh explanations/heart_attack/full 'itp astrong, bstrong; ucore'
```
will generate explanations into `explanations/heart_attack/full/ucore_itp_astrong_bstrong.phi.txt`,
statistics into `explanations/heart_attack/full/ucore_itp_astrong_bstrong.stats.txt`,
time splits into `explanations/heart_attack/full/ucore_itp_astrong_bstrong.times.txt`
and total time into `explanations/heart_attack/full/ucore_itp_astrong_bstrong.time.txt`.
It will use model `models/heart_attack/heart_attack-50.nnet` and dataset `datasets/heart_attack/heart_attack_full.csv` (as specified in `spec/models_datasets`).

```
TIMEOUT=2h ./scripts/run1.sh explanations/heart_attack/quick 'abductive; itp aweaker, bstrong; ucore min' test_name -e phi.txt
```
will use timeout of `2h` and generate explanations into `./phi.txt` (due to the extra arguments `-e phi.txt`)
but statistics into `explanations/heart_attack/quick/test_name.stats.txt`, etc.
It will use dataset `datasets/heart_attack/heart_attack_quick.csv`.

```
CMD=../build-marabou/spexplain SRC_EXPERIMENT=abductive ./scripts/run1.sh explanations/heart_attack/quick 'trial n 4'
```
will use the executable `../build-marabou/spexplain`
and compute explanations into `explanations/heart_attack/quick/trial_n_4__abductive.phi.txt`
using `explanations/heart_attack/quick/abductive.phi.txt` as starting points (which must already exist) instead of sample points.
The resulting explanations are the same as what would compute the strategy `abductive; trial n 4` if starting from sample points, but it does not re-compute `abductive` that has already been computed.

## `run-experiments.sh`

```
USAGE: ./run-experiments.sh <output_dir> <experiments_spec> [consecutive] [[+]reverse] [<max_samples>] [<filter_experiments_regex>] [-h|-n]
   <output_dir> must be specified in ./spec/models_datasets
   <experiments_spec> is one of: all base itp
CONSECUTIVE_EXPERIMENTS are not run unless 'consecutive' is provided

OPTIONS:
   -h    Display help message and exit
   -n    Dry mode - only print what would have been run
```

This script runs a collection of experiments in parallel, using the script `run1.sh` as a subroutine.
First, the file `spec/experiments/all` defines experiment names and their associated strategies.
Second, all other files in the directory `spec/experiments` define certain collections (i.e. a subset) of experiments by referring to their names (as defined in `spec/experiments/all`), using the array `EXPERIMENT_NAMES`. They can also define the array `CONSECUTIVE_EXPERIMENTS` which define certain experiments that are run on top of others (refer to `SRC_EXPERIMENT` description of the `run1.sh` script above).

### `<output_dir>`

Refer to the description of `<output_dir>` of the `run1.sh` script above.

### `<experiments_spec>`

The name of the collection of experiments to run.
It corresponds to the basename of the specification files in the `spec/experiments` directory.
It also accepts `all` as the collection of all available experiments.

### `consecutive`

If specified, runs `CONSECUTIVE_EXPERIMENTS` instead of those specified in `EXPERIMENT_NAMES`.
This assumes that the input explanations are already computed; their automatic pre-computation is not supported.

### `reverse`

Refer to the description of `reverse` of the `run1.sh` script above.
Without `+`, it runs only experiments with reversed order of variables.
With `+`, it runs both.

### `<max_samples>`

Refer to the description of `<max_samples>` of the `run1.sh` script above.

### `<filter_experiments_regex>`

Only experiments with the name that matches the given regular expression will be run.
(You can test what would be run using the option `-n`.)

### Environment variables

* `CPU_PERCENTAGE`: the percentage of used CPUs (default: `60`)
* `OPTIONS`: additional arguments passed to the `run1.sh` script
* Variables `CMD` and `TIMEOUT` as described for the `run1.sh` script above

### Examples

In directory `data/`:

```
./scripts/run-experiments.sh explanations/heart_attack/full base
```
will run all experiments specified in `spec/experiments/base` (e.g. `abductive`),
using the model `models/heart_attack/heart_attack-50.nnet` and dataset `datasets/heart_attack/heart_attack_full.csv` (as specified in `spec/models_datasets`).

```
CMD=../build-marabou/spexplain ./scripts/run-experiments.sh explanations/heart_attack/full base consecutive
```
will run all consecutive experiments specified in `spec/experiments/base`, given that the previous collection above has already been run,
and using the executable `../build-marabou/spexplain`.

```
TIMEOUT=2m OPTIONS='--filter-samples incorrect' ./scripts/run-experiments.sh explanations/obesity/short itp '^itp'
```
will use the timeout of `2m` per experiment (i.e., per a run of `run1.sh`)
and will run experiments specified in `spec/experiments/itp` (e.g. `itp_astrong_bstrong`),
using the model `models/obesity/obesity-10-20-10.nnet` and dataset `datasets/obesity/obesity_short.csv`.
Additionally, it passes the arguments `--filter-samples incorrect` to the underlying script `run1.sh` (i.e., consequently, to `spexplain`).

## `analyze.sh`


```
USAGE: ./analyze.sh <action> <psi> <f> [<f2>] [<max_rows>]
ACTIONS: check|count-fixed|compare-subset
```
Except of these, it also accepts several environment variables, described below.

It normally requires at least one of `opensmt`, `cvc5`, `z3`, or `mathsat` SMT solvers installed system-wide.
Currently, it uses only `opensmt` as a *fast* solver. A fast solver is used by default.
Currently, the only exception is action `check` that intentionally ensures to use a third-party solver to increase trust.
Note that using a different solver than a fast one usually quite slower. 
If one wants to use a particular solver, e.g. locally built, use environment variable `SOLVER`.

### `<action>`

* `check` (unary): verifies that an explanation file is correct, i.e. the classification indeed cannot change.
* `count-fixed` (unary): precisely computes the ratio of the total number of features that are fixed to a single value by the explanation (e.g., `astrong` explanations sometimes still fix some features)
* `compare-subset` (binary): computes the subset relation between the corresponding explanations in two files
(`<` means: \#explanations from `<f>` that are a subset of those from `<f2>`; `=` means equivalency, `>` stands for supersets, and `?` means uncomparable - neither subsumes another)

### `<psi>`

A "psi" file is an encoding of the classification and/or the domains (produced by the `dump-psi` action in `spexplain`).
Action `check` requires the classification file `psi_c*.smt2` (use any one of them), the others use the domain file `psi_d.smt2`

### `<f> [<f2>]`

One or two explanation files ("phi"), depending whether the action is unary or binary.
The phi files contain the formulas of the produced explanations in SMT-LIB format (`-e` option in `spexplain`).

Use either your locally generated explanations or those already computed and stored in `data/explanations`.

### `[<max_rows>]`

Processes at most the given number of explanations.

### Examples

From directory `data/`, to check explanations, run e.g.:
```
./scripts/analyze.sh check explanations/heart_attack/psi_c0.smt2 explanations/heart_attack/quick/itp_astrong_bstrong.phi.txt
```
which should output:
```
OK!
```
If one wants to use a particular solver:

```
SOLVER=<path_to_solver> ./scripts/analyze.sh check explanations/heart_attack/psi_c0.smt2 explanations/heart_attack/quick/itp_astrong_bstrong.phi.txt
```

To count fixed features, run e.g.:
```
./scripts/analyze.sh count-fixed explanations/heart_attack/psi_d.smt2 explanations/heart_attack/quick/itp_astrong_bstrong.phi.txt
```
which should output:
```
avg #fixed features: 10.0%
```
Running on just 2 upmost explanations:
```
./scripts/analyze.sh count-fixed explanations/heart_attack/psi_d.smt2 explanations/heart_attack/quick/itp_astrong_bstrong.phi.txt 2
```
should output:
```
avg #fixed features: 50.0%
```
meaning that the second explanation fixes all features (for just the first explanation, the average is `0%`).

To compare two explanation files, run:
```
./scripts/analyze.sh compare-subset explanations/heart_attack/psi_d.smt2 explanations/heart_attack/quick/itp_astrong_bstrong.phi.txt explanations/heart_attack/quick/itp_aweak_bstrong.phi.txt
```
which should output:
```
<: 10 =: 0 >: 0 | ?: 0
```


## `analyze-experiments.sh`

```
USAGE: ./analyze-experiments.sh <action> <explanations_dir> <experiments_spec> [[+]consecutive] [[+]reverse] [<max_samples>] [<filter_regex>] [<filter_regex2>] [-h|-f]
ACTIONS: check|count-fixed|compare-subset
   [<filter_regex2>] is only to be used with binary actions
```

This script is built on top of `analyze.sh` and runs an action across a series of phi files.
It implicitly uses explanations from `explanations`.
Which explanations are used is specified in `spec/experiments`.
For example, to run the subset comparisons on all `itp` explanations from the `heart_attack`-`quick` dataset:
```
./scripts/analyze-experiments.sh compare-subset explanations/heart_attack/quick/ itp
```

Beware, in this case, it compares all pairs of files, which is quite a lot.

### Filtering

For example, to only compare all `astrong` `itp` variants with `aweak`:
```
./scripts/analyze-experiments.sh compare-subset explanations/heart_attack/quick/ itp astrong_ aweak_
```

The filters are treated as extended regular expressions. This will still result in many pairs.

The script also supports pairwise pattern matching as done in the `sed` tool. For example, to only compare `itp_astrong` with `itp_aweak`, `ucore_itp_astrong` with `ucore_itp_aweak`, and `ucore_min_itp_astrong` with `ucore_min_itp_aweak` (i.e. always the same "ucore category"):
```
./scripts/analyze-experiments.sh compare-subset explanations/heart_attack/quick/ itp '^(|ucore(|_min)_)itp_astrong_' '^\1itp_aweak_'
```

### Caching

The script also automatically supports caching. It stores what has been computed and tries to reuse already available results.
So you may first pre-compute larger superset of results and later only filter out what you are currently interested in.
It does not work perfectly :)

The option `-f` enforces to compute everything from scratch and disables loading the results from cache.
