# Experiment scripts

This folder contains several scripts to automate and simplify experimentation with SpEXplAIn.


## `run1.sh`

This is a proxy script for the `spexplain` executable that automatically uses some options suitable for running the experiments (e.g. `--quiet`) and automatically adjusts the destination directory for the generated explanations based on the inputs and parameters:
```
USAGE: ./run1.sh <nn_model_fn> <dataset_fn> <exp_strategies_spec> [<name>] [<max_samples>] <args>...
```
Except of these, it also accepts several environment variables, described below.
The inputs `<nn_model_fn>`, `<dataset_fn>` and `<exp_strategies_spec>` are just forwarded to `spexplain`.
The script outputs various data into files (including e.g. statistics, time splits, etc.) in the destination directory.
It only runs one experiment, that is, produces one output file with explanations (along with statistics etc.), not multiple.

### `<name>`

Optional name that describes the used strategies and that will be used as the basename for the generated files.
If omitted, it will be generated automatically from `<exp_strategies_spec>`.

### `<max_samples>`

If specified, computes explanations only for randomly selected `<max_samples>` samples (refers to options `-r` and `-n` of `spexplain`).
Modifies the destination directory path.

### `<args>...`

Any other arguments that will be forwarded to `spexplain`.
See the description of the environment variable `VARIANT` if you need to store different configurations separately.

### Environment variables

* `CMD`: sets the `spexplain` executable (default: `../../build/spexplain`)
* `VARIANT`: specific identifier for the used options and arguments (default: `default`); the destination directory path depends on `VARIANT`, hence it distinguishes the results when using different configurations
* `TIMEOUT`: optional overall timeout for the execution; the duration is a number with an optional suffix: `s` for seconds (default), `m` for minutes, `h` for hours or `d` for days
* `TIMEOUT_PER`: optional timeout per explanation (refers to option `-t` of `spexplain`); the duration format is the same as in `TIMEOUT`
* `SRC_EXPERIMENT`: sets already computed explanations with a given name (corresponding to `<name>` at that time) as the input, instead of using the original sample points (refers to option `-E` of `spexplain`)

### Examples

In directory `data/`:

```
./scripts/run1.sh models/heart_attack/heart_attack-50.nnet datasets/heart_attack/heart_attack_full.csv 'itp astrong, bstrong; ucore'
```
will generate:
* explanations into `explanations/heart_attack/50/full/default/ucore_itp_astrong_bstrong.phi.txt`,
* statistics into `explanations/heart_attack/50/full/default/ucore_itp_astrong_bstrong.stats.txt`,
* time splits into `explanations/heart_attack/50/full/default/ucore_itp_astrong_bstrong.times.txt`,
* and total time into `explanations/heart_attack/50/full/default/ucore_itp_astrong_bstrong.time.txt`.

To run a variant with the reversed order of features:
```
VARIANT=reverse ./scripts/run1.sh models/heart_attack/heart_attack-50.nnet datasets/heart_attack/heart_attack_full.csv 'itp astrong, bstrong; ucore' --reverse-var
```
which generates files into the destination `explanations/heart_attack/50/full/reverse`.
(The order of features is insignificant for `itp` strategies and has a much higher impact on `abductive` and `trial` strategies.)

Another example:
```
VARIANT=fix-all ./scripts/run1.sh models/heart_attack/heart_attack-50.nnet datasets/heart_attack/heart_attack_full.csv 'itp astrong, bstrong; ucore' --fix-default-sample-neuron-activations all
```
which by default fixes all neuron activations
and generates files into the destination `explanations/heart_attack/50/full/fix-all`.

To use a timeout of `2h`:
```
TIMEOUT=2h ./scripts/run1.sh models/heart_attack/heart_attack-50.nnet datasets/heart_attack/heart_attack_quick.csv 'abductive; itp aweaker, bstrong; ucore min' test_name -e phi.txt
```
In addition,
this generates explanations into `./phi.txt` (due to the extra arguments `-e phi.txt`)
but statistics into `explanations/heart_attack/50/quick/default/test_name.stats.txt`, etc.

```
CMD=../build-marabou/spexplain SRC_EXPERIMENT=abductive ./scripts/run1.sh models/heart_attack/heart_attack-50.nnet datasets/heart_attack/heart_attack_quick.csv 'trial n 4'
```
will use the executable `../build-marabou/spexplain`
and compute explanations into `explanations/heart_attack/50/quick/default/trial_n_4__abductive.phi.txt`
using `explanations/heart_attack/50/quick/default/abductive.phi.txt` as starting points (which must already exist) instead of sample points.
The resulting explanations are the same as what would compute the strategy `abductive; trial n 4` if starting from sample points, but it does not re-compute `abductive` that has already been computed.

```
CMD=../build-marabou/spexplain VARIANT=OpenSMT ./scripts/run1.sh models/heart_attack/heart_attack-50.nnet datasets/heart_attack/heart_attack_quick.csv 'abductive; trial n 4' -V opensmt
```
forces to use the `opensmt` verifier
and explicitly sets the variant name to `OpenSMT`,
storing the explanations into `explanations/heart_attack/50/quick/OpenSMT/trial_n_4_abductive.phi.txt`.

```
TIMEOUT=1h TIMEOUT_PER=30s ./scripts/run1.sh models/mnist/mnist-200.nnet datasets/mnist/mnist_short.csv 'itp'
```
will compute explanations into `explanations/mnist/200/short/default/itp.phi.txt`
using the time limit of `30s` per each explanation
and `1h` for the whole computation of all explanations.


## `run-experiments.sh`

```
USAGE: ./run-experiments.sh (<nn_model_fn> <dataset_fn>)... <experiments_spec> [consecutive] [<max_samples>] [<filter_experiments_regex>] [-h|-n]
   <experiments_spec> is one of: all base itp
CONSECUTIVE_EXPERIMENTS are not run unless 'consecutive' is provided

OPTIONS:
   -h    Display help message and exit
   -n    Dry mode - only print what would have been run
```

This script runs a collection of experiments in parallel while restricting to the available resources,
using the script `run1.sh` as a subroutine.
First, the file `spec/experiments/all` defines experiment names and their associated strategies.
Second, all other files in the directory `spec/experiments` define certain collections (i.e. a subset) of experiments by referring to their names (as defined in `spec/experiments/all`), using the array `EXPERIMENT_NAMES`. They can also define the array `CONSECUTIVE_EXPERIMENTS` which define certain experiments that are run on top of others (refer to `SRC_EXPERIMENT` description of the `run1.sh` script above).

The arguments `<nn_model_fn>` and `<dataset_fn>` are the same as in `run1.sh`,
except that here multiple pairs are possible,
all running in parallel.

### `<experiments_spec>`

The name of the collection of experiments to run.
It corresponds to the basename of the specification files in the `spec/experiments` directory.
It also accepts `all` as the collection of all available experiments.

### `consecutive`

If specified, runs `CONSECUTIVE_EXPERIMENTS` instead of those specified in `EXPERIMENT_NAMES`.
This assumes that the input explanations are already computed; their automatic pre-computation is not supported.

### `<max_samples>`

Refer to the description of `<max_samples>` of the `run1.sh` script above.

### `<filter_experiments_regex>`

Only experiments with the name that matches the given regular expression will be run.
(You can test what would be run using the option `-n`.)

### Environment variables

* `CPU_PERCENTAGE`: the percentage of used CPUs (default: `60`)
* `OPTIONS`: additional arguments passed to the `run1.sh` script
* Variables `CMD`, `VARIANT`, `TIMEOUT` and `TIMEOUT_PER` as described for the `run1.sh` script above

### Examples

In directory `data/`:

```
./scripts/run-experiments.sh models/heart_attack/heart_attack-50.nnet datasets/heart_attack/heart_attack_full.csv base
```
will run all experiments specified in `spec/experiments/base` (e.g. `abductive`),
generating the explanations into `explanations/heart_attack/50/full/default`.

```
CMD=../build-marabou/spexplain ./scripts/run-experiments.sh models/heart_attack/heart_attack-50.nnet datasets/heart_attack/heart_attack_full.csv base consecutive
```
will run all consecutive experiments specified in `spec/experiments/base`, given that the previous collection above has already been run,
and using the executable `../build-marabou/spexplain`.

```
TIMEOUT=2m TIMEOUT_PER=30s OPTIONS='--reverse-var' VARIANT=reverse ./scripts/run-experiments.sh models/obesity/obesity-10-20-10.nnet datasets/obesity/obesity_short.csv itp '^itp'
```
will use the timeout of `2m` per experiment (i.e., per a run of `run1.sh`)
and of `30s` per explanation,
will run experiments specified in `spec/experiments/itp` that match the regex filter `^itp` (e.g. `itp_astrong_bstrong`),
and will generate the explanations into `explanations/obesity/10-20-10/short/reverse`.
It uses the variant `reverse`
and passes the arguments `--reverse-var` to the underlying script `run1.sh` (i.e., consequently, to `spexplain`).

Example of another variant is
```
OPTIONS='--fix-default-sample-neuron-activations all' VARIANT=fix-all ./scripts/run-experiments.sh models/obesity/obesity-10-20-10.nnet datasets/obesity/obesity_short.csv itp '^itp'
```
which by default fixes all neuron activations.


## `collect_stats.sh`

```
USAGE: ./collect_stats.sh <explanations_dir> <experiments_spec> [[+]consecutive] [<max_samples>] [<filter_regex>] [<OPTIONS>]
OPTIONS:
   --exclude-column <name>    Exclude given column
   --average [<regex>]     Average columns for all rows [matching the regex] (can be repeated)
```

### Examples

In directory `data/`:

```
./scripts/collect_stats.sh explanations/heart_attack/50/full/default base '^itp_a' --exclude-column '%features' --exclude-column '%fixed' --exclude-column '#checks'
```

```
./scripts/collect_stats.sh explanations/heart_attack/50/full/default base +consecutive '(abductive|^itp_aweak_bstrong)' --exclude-column '%features' --exclude-column '%fixed'
```

```
./scripts/collect_stats.sh explanations/heart_attack/50/full/default itp '(slice_|itp_vars_)' \
   --exclude-column '%features' --exclude-column '%fixed' --exclude-column '%dimension' --exclude-column '#checks' \
   --average '^itp_vars' --average '^ucore_itp_vars' --average 'ucore_min_itp_vars' \
   --average 'slice_.*[0-9]_itp_aweak_bstrong' --average 'slice_.*_ucore_itp_aweak_bstrong' --average 'slice_.*_ucore_min_itp_aweak_bstrong'
```


## `analyze.sh`

Analyzes all explanations in an explanation file, using certain queries to an SMT solver:
```
USAGE: ./analyze.sh <action> <psi> <f> [<f2>] [<max_rows>]
ACTIONS: check|check-sat|count-fixed|compare-subset
```
Except of these, it also accepts several environment variables, described below.

It normally requires at least one of `opensmt`, `cvc5`, `z3`, or `mathsat` SMT solvers installed system-wide.
Currently, it uses only `opensmt` as a *fast* solver. A fast solver is used by default.
Currently, the only exception is action `check` that intentionally ensures to use a third-party solver to increase trust.
Note that using a different solver than a fast one is usually quite slower.
If one wants to use a particular solver, e.g. locally built, use environment variable `SOLVER`.

### `<action>`

* `check-sat` (unary): verifies that all explanations in the file are satisfiable
* `check` (unary): extends `check-sat` by also verifying that the explanations cannot change the classification
* `count-fixed` (unary): precisely computes the ratio of the total number of features that are fixed to a single value by the explanation (e.g., `astrong` explanations sometimes still fix some features)
* `compare-subset` (binary): computes the subset relation between the corresponding explanations in two files
(`<` means: \#explanations from `<f>` that are a subset of those from `<f2>`; `=` means equivalency, `>` stands for supersets, and `NC` means uncomparable - neither subsumes another)

### `<psi>`

A "psi" file is an encoding of the classification and/or the domains (produced by the `dump-psi` action in `spexplain`).
Action `check` requires the classification file `psi_c*.smt2` (use any one of them), the others use the domain file `psi_d.smt2`

### `<f> [<f2>]`

One or two explanation files ("phi"), depending whether the action is unary or binary.
The phi files contain the formulas of the produced explanations in SMT-LIB format (`-e` option in `spexplain`).

Use either your locally generated explanations or those already computed and stored in `data/explanations`.

### `[<max_rows>]`

Processes at most the given number of explanations.

### Environment variables

* `SOLVER`: sets a particular SMT solver to be used for the queries
* `TIMEOUT_PER`: as in `run1.sh`, but used for particular queries to the SMT solver

### Examples

From directory `data/`, to check explanations, run e.g.:
```
./scripts/analyze.sh check explanations/heart_attack/50/psi_c0.smt2 explanations/heart_attack/50/quick/default/itp_astrong_bstrong.phi.txt
```
which should output:
```
OK!
```

If one wants to use a particular solver:
```
SOLVER=<path_to_solver> ./scripts/analyze.sh check explanations/heart_attack/50/psi_c0.smt2 explanations/heart_attack/50/quick/default/itp_astrong_bstrong.phi.txt
```

To restrict the run of particular check queries:
```
TIMEOUT_PER=10 ./scripts/analyze.sh check explanations/heart_attack/50/psi_c0.smt2 explanations/heart_attack/50/quick/default/itp_astrong_bstrong.phi.txt
```
Possible output:
```
OK! (60.0%)
```
which means that only `60.0%` of explanations have been verified, due to the time limit per query.

To count fixed features, run e.g.:
```
./scripts/analyze.sh count-fixed explanations/heart_attack/50/psi_d.smt2 explanations/heart_attack/50/quick/default/itp_astrong_bstrong.phi.txt
```
which should output:
```
avg #fixed features: 10.0%
```

Running on just 2 upmost explanations:
```
./scripts/analyze.sh count-fixed explanations/heart_attack/50/psi_d.smt2 explanations/heart_attack/50/quick/default/itp_astrong_bstrong.phi.txt 2
```
should output:
```
avg #fixed features: 50.0%
```
meaning that the second explanation fixes all features (for just the first explanation, the average is `0%`).

To analyze all but restrict the run of particular check queries:
```
TIMEOUT_PER=0.2 ./scripts/analyze.sh count-fixed explanations/heart_attack/50/psi_d.smt2 explanations/heart_attack/50/quick/default/itp_astrong_bstrong.phi.txt
```
Possible output:
```
avg #fixed features: 9.2% (?: 20.8%)
```
which means that `20.8%` out of all features of all explanations have not completed
and do not participate in the `10.0%` average.

To compare two explanation files, run:
```
./scripts/analyze.sh compare-subset explanations/heart_attack/50/psi_d.smt2 explanations/heart_attack/50/quick/default/itp_astrong_bstrong.phi.txt explanations/heart_attack/50/quick/default/itp_aweak_bstrong.phi.txt
```
which should output:
```
Total: 10
<: 10 =: 0 >: 0 | NC: 0
```

To use a time limit, run:
```
TIMEOUT_PER=0.05 ./scripts/analyze.sh compare-subset explanations/heart_attack/50/psi_d.smt2 explanations/heart_attack/50/quick/default/itp_astrong_bstrong.phi.txt explanations/heart_attack/50/quick/default/itp_aweak_bstrong.phi.txt
```
Possible output:
```
Total: 10
<: 9 =: 0 >: 0 | NC: 0 ?: 1
```
which means that one of the comparisons timed out.


## `analyze-experiments.sh`

```
USAGE: ./analyze-experiments.sh <action> <explanations_dir>... <experiments_spec> [[+]consecutive] [<max_samples>] [<filter_regex>] [<filter_regex2>] [-h|-f]
ACTIONS: check|check-sat|count-fixed|compare-subset
   [<filter_regex2>] is only to be used with binary actions
```
If `<filter_regex2>` is omitted, it is set to `<filter_regex>`.

This script is built on top of `analyze.sh` and runs an action across a series of phi files.
Which strategies are used is specified by `<experiments_spec>` that refers to `spec/experiments` (see `run-experiments.sh`).

Multiple `<explanations_dir>`s are supported as long as they differ only in the variant (cf. `VARIANT`) and nothing else (e.g., dataset or model).

### Examples

To check that all `itp` explanations from `explanations/heart_attack/50/quick/default` are satisfiable:
```
./scripts/analyze-experiments.sh check-sat explanations/heart_attack/50/quick/default itp
```

To run the subset comparisons on all `itp` explanations from `explanations/heart_attack/50/quick/default`:
```
./scripts/analyze-experiments.sh compare-subset explanations/heart_attack/50/quick/default itp
```

Beware, in the cases above, it compares all pairs of files, which is quite a lot - see section on filtering below.

To compare not only the `default` variant, but all computed variants in that destination, run:
```
./scripts/analyze-experiments.sh compare-subset explanations/heart_attack/50/quick/* itp
```

### Filtering

For example, to only compare all `astrong` `itp` strategies with `aweak`:
```
./scripts/analyze-experiments.sh compare-subset explanations/heart_attack/50/quick/default itp astrong_ aweak_
```

The filters are treated as extended regular expressions.
Note that this will still result in many pairs.

The script also supports pairwise pattern matching as done in the `sed` tool.
For example, to only compare `itp_astrong` with `itp_aweak`, `ucore_itp_astrong` with `ucore_itp_aweak`, and `ucore_min_itp_astrong` with `ucore_min_itp_aweak` (i.e. always the same "ucore category"):
```
./scripts/analyze-experiments.sh compare-subset explanations/heart_attack/50/quick/default itp '^(|ucore(|_min)_)itp_astrong_' '^\1itp_aweak_'
```

### Caching

The script also automatically supports caching. It stores what has been computed and tries to reuse already available results.
So you may first pre-compute larger superset of results and later only filter out what you are currently interested in.
It should also recover gracefully after failing or interrupting the computation.
It does not work perfectly :)

The option `-f` enforces to compute everything from scratch and disables loading the results from cache.
