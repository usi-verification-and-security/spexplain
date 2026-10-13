#!/usr/bin/env bash
## Phase 2: generalize the interval explanations of ./intervals.sh by interpolation with OpenSMT:
## itp_{aweak,afactor_0.5,astrong}_bstrong__trial_n_4, timeout 5m per sample,
## at most 60 processes. Refer to ./common for the env. variables.
##
## USAGE: [DATASETS=...] [WRAPPER=~/srun_commands/sbatch.sh] ./itp.sh

PHASE=itp

source "$(dirname "$(realpath "$0")")/common"
