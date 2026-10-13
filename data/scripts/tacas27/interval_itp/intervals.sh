#!/usr/bin/env bash
## Phase 1: interval explanations 'trial n 4' (trial_n_4) with Marabou,
## timeout 10m per sample, at most 60 processes. Refer to ./common for the env. variables.
##
## USAGE: [DATASETS=...] [WRAPPER=~/srun_commands/sbatch.sh] ./intervals.sh

PHASE=intervals

source "$(dirname "$(realpath "$0")")/common"
