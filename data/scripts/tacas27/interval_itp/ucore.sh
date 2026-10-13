#!/usr/bin/env bash
## Unsat cores on top of the interpolants with OpenSMT:
## {ucore,ucore_min}_itp_{aweak,afactor_0.5,astrong}_bstrong (see spec/experiments/ucore_itp),
## timeout 20m per sample, at most 60 processes. Refer to ./common for the env. variables.
##
## By default: naive on 50x{1,2} and fix-all on 50x{1..4}. A model runs with a variant only if it is
## both in MODEL_IDXS and in the variant's list: NAIVE_MODEL_IDXS, FIX_ALL_MODEL_IDXS.
##
## USAGE: [DATASETS=...] [VARIANTS=...] [WRAPPER=~/srun_commands/sbatch.sh] ./ucore.sh

PHASE=ucore

VARIANTS=${VARIANTS:-naive fix-all}
declare -A VARIANT_MODEL_IDXS=(
    [naive]="${NAIVE_MODEL_IDXS:-1 2}"
    [fix-all]="${FIX_ALL_MODEL_IDXS:-1 2 3 4}"
)

source "$(dirname "$(realpath "$0")")/common"
