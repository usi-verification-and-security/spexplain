# Input and Output Data

This folder gathers the inputs and outputs of the tool.

## `models`

Specification files of trained neural networks in several formats (e.g., `.nnet`).

## `datasets`

Collections of sample points together with their expected classifications, in the form of `.csv` files.
Typically, a dataset name has a suffix such as `_full`, `_short` or `_quick`. The latter represent a random subset of the full dataset.

## `scripts`

Refer there for explicit documentation.

## `explanations`

This is a submodule, that is, a separate repository: https://gitlab.com/usi-verification-and-security/spexplain/explanations

If you cloned with `--recurse-submodules`, it should have been fetched automatically. Otherwise, you need to run `git submodule add https://gitlab.com/usi-verification-and-security/spexplain/explanations`

This repository contains reference explanations computed by SpEXplAIn, in the form of logical formulas in the SMT-LIB format.
It only consists of formulas, it does not contain any `assert` or `declare-const` commands.

The repository also contains standalone SMT-LIB files that encode
the domains of particular input features (`psi_d.smt2`)
and
the constraint that a given classification cannot change (`psi_c*.smt2`),
based on an input model from `models`.
These files are produced by the `dump-psi` action in `build/spexplain`.
