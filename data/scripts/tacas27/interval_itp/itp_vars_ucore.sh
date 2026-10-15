#!/usr/bin/env bash
## Interpolation restricted to selected features and unsat cores on top of it with OpenSMT, all samples:
##   itp_<ITP>_vars_<VARS>            'itp <ITP_PARAMS>, vars <VARS>'              timeout ITP_TIMEOUT_PER
##   ucore_itp_<ITP>_vars_<VARS>      'itp <ITP_PARAMS>, vars <VARS>; ucore'       timeout UCORE_TIMEOUT_PER
##   ucore_min_itp_<ITP>_vars_<VARS>  'itp <ITP_PARAMS>, vars <VARS>; ucore min'   timeout UCORE_MIN_TIMEOUT_PER
## e.g. itp_aweak_bstrong_vars_x4_x10, ucore_itp_aweak_bstrong_vars_x4_x10, ucore_min_itp_aweak_bstrong_vars_x4_x10.
## The unsat cores recompute the interpolant within the same run, so all three experiments run
## at once: all samples of all models, variants and experiments share one queue of at most MAX_JOBS (60) processes.
##
## By default: heart_attack 50x{1,4}, naive and fix-all. Refer to ./common for the other env. variables.
##
## Environment variables (all optional):
##   VARS                   features of the interpolant (default: 'x4 x10')
##   ITP_PARAMS             parameters of the itp strategy (default: 'aweak, bstrong')
##   ITP_TIMEOUT_PER        (default: 5m)
##   UCORE_TIMEOUT_PER      (default: 5m)
##   UCORE_MIN_TIMEOUT_PER  (default: 10m)
##
## USAGE: [WRAPPER=~/srun_commands/sbatch.sh] ./itp_vars_ucore.sh

PHASE=itp_vars

DATASETS=${DATASETS:-heart_attack}
MODEL_IDXS=${MODEL_IDXS:-1 4}
VARIANTS=${VARIANTS:-naive fix-all}

VARS=${VARS:-x4 x10}
ITP_PARAMS=${ITP_PARAMS:-aweak, bstrong}
ITP_STRATEGY="itp $ITP_PARAMS, vars $VARS"

PHASE_STRATEGIES=(
    "$ITP_STRATEGY"
    "$ITP_STRATEGY; ucore"
    "$ITP_STRATEGY; ucore min"
)
PHASE_TIMEOUTS=(
    "${ITP_TIMEOUT_PER:-5m}"
    "${UCORE_TIMEOUT_PER:-5m}"
    "${UCORE_MIN_TIMEOUT_PER:-10m}"
)

source "$(dirname "$(realpath "$0")")/common"
