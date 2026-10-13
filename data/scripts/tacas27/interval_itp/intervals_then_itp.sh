#!/usr/bin/env bash
## Both phases in one job (one node): ./intervals.sh, then ./itp.sh on whatever intervals finished.
## Refer to ./common for the env. variables; set the timeouts per phase with INTERVALS_TIMEOUT_PER
## and ITP_TIMEOUT_PER (default: 10m and 5m), TIMEOUT_PER would apply to both.
##
## USAGE: [DATASETS=...] [WRAPPER=~/srun_commands/sbatch.sh] ./intervals_then_itp.sh

SCRIPT=$(realpath "$0")
DIR=$(dirname "$SCRIPT")

## Resubmit through the wrapper, so that both phases share one job
[[ -n $WRAPPER ]] && exec env -u WRAPPER $WRAPPER "$SCRIPT" "$@"

"$DIR/intervals.sh"
status=$?

"$DIR/itp.sh" || status=1

exit $status
