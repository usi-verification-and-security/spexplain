#!/usr/bin/env bash
## Re-run the leftover TACAS'27 experiments in parallel within one job (one node):
##   loan  200x8  guide-all
##   cifar 200x3  guide-all
##   loan  50x8   fix-all
##   gtsrb 200x3  guide-all (timeout per sample 14m)
## Each run is one line in RUNS: env. assignments followed by a <KEYWORD>.sh script
## (relative to this directory) and its arguments, as in ./commands_list but without WRAPPER.
##
## USAGE: [NAME=<job_name>] WRAPPER=~/srun_commands/sbatch.sh ./leftovers.sh
##   or submit it directly: NAME=leftovers ~/srun_commands/sbatch.sh ./data/scripts/tacas27/leftovers.sh
##
## The output of each run goes to $LOG_DIR/<idx>.log (default: ./parallel_logs/<NAME>).
## Env. variables:
##   WRAPPER         if set, submit this script through it (e.g. sbatch.sh) instead of running it
##   LOG_DIR         directory for per-run logs
##   CPU_PERCENTAGE  percentage of CPUs for each run (default: 25, i.e. the 4 runs share the node)
##   DRY_RUN=1       forwarded to each run

RUNS=(
    'VARIANTS_SPEC=guide_only                  MODEL_IDXS="8" ./LO200.sh'
    'VARIANTS_SPEC=guide_only                  MODEL_IDXS="3" ./CIF200.sh'
    'VARIANTS_SPEC=fix_only                    MODEL_IDXS="8" ./LO50.sh'
    'VARIANTS_SPEC=guide_only  TIMEOUT_PER=14m MODEL_IDXS="3" ./GTS200.sh'
)

SCRIPT=$(realpath "$0")
TACAS_DIR=$(dirname "$SCRIPT")

## Resubmit through the wrapper, so that all runs share one job
[[ -n $WRAPPER ]] && exec env -u WRAPPER $WRAPPER "$SCRIPT" "$@"

## Each <KEYWORD>.sh overwrites NAME, so keep the job name for the logs here
LOG_DIR=${LOG_DIR:-$PWD/parallel_logs/${NAME:-leftovers}}
mkdir -p "$LOG_DIR" || exit 1
unset NAME WRAPPER

## Split the node between the runs
export CPU_PERCENTAGE=${CPU_PERCENTAGE:-25}

pids=()
for idx in ${!RUNS[@]}; do
    ## Resolve the script relative to this directory, but keep the working directory
    run="${RUNS[$idx]/.\//$TACAS_DIR/}"
    log="$LOG_DIR/$idx.log"
    printf "[%d] %s\n    -> %s\n" $idx "$run" "$log"
    ( eval "$run" ) >"$log" 2>&1 &
    pids+=($!)
done

status=0
for idx in ${!pids[@]}; do
    wait ${pids[$idx]}
    s=$?
    printf "[%d] finished with status %d\n" $idx $s
    (( $s )) && status=$s
done

exit $status
