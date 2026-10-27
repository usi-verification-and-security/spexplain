# Input and Output Data

This folder gathers the inputs and outputs of the tool.

## `models`

Specification files of trained neural networks in several formats (e.g., `.nnet`).

## `datasets`

Collections of sample points together with their expected classifications, in the form of `.csv` files.
Typically, a dataset name has a suffix such as `_full`, `_short` or `_quick`. The latter represent a random subset of the full dataset.
The recent experiments use the datasets `<name>_s100_scaled.csv` (100 samples, normalized to [0, 1])
and their subsets, e.g., `<name>_s100_scaled_quick.csv`.

## `scripts`

Refer there for explicit documentation.
These include scripts for the analysis of the generated explanations.

## `explanations`

This is a submodule, that is, a separate repository: https://gitlab.com/usi-verification-and-security/spexplain/explanations

If you cloned with `--recurse-submodules`, it should have been fetched automatically. Otherwise, you need to run `git submodule add https://gitlab.com/usi-verification-and-security/spexplain/explanations`

The submodule is not needed for running the tool or for reproducing the experiments.
It contains explanations of earlier experiments, only for some of the networks and datasets,
and none of the networks and datasets of the recent experiments (e.g., `*_s100_scaled*.csv` and the CNNs).
The explanations are in the form of logical formulas in the SMT-LIB format;
they only consist of formulas, without any `assert` or `declare-const` commands.

The scripts of the experiments write the explanations into this directory too,
whether the submodule is fetched or not (see `run1.sh` and `run-experiments.sh` in [`scripts`](scripts)).

The submodule also contains standalone SMT-LIB files that encode
the domains of particular input features (`psi_d.smt2`)
and
the constraint that a given classification cannot change (`psi_c*.smt2`),
based on an input model from `models`.
These files are produced by the `dump-psi` action in `build/engine`.
