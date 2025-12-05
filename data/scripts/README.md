# Experiment scripts

This folder contains several scripts to automate and simplify experimentation with SpEXplAIn.

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

A "psi" file is an encoding of the classification and/or the domains (produced by the `dump-psi` action in `build/spexplain`).
Action `check` requires the classification file `psi_c*.smt2` (use any one of them), the others use the domain file `psi_d.smt2`

### `<f> [<f2>]`

One or two explanation files ("phi"), depending whether the action is unary or binary.
The phi files contain the formulas of the produced explanations in SMT-LIB format (`-e` option in `build/spexplain`).

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
Which explanations are used is specified in `data/scripts/spec/experiments`.
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
