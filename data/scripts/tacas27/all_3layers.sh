#!/bin/bash

## realpath before dirname, so that the script works from any directory
SCRIPTS_DIR=$(dirname "$(dirname "$(realpath "$0")")")
DATA_DIR=$(dirname "$SCRIPTS_DIR")
ROOT_DIR=$(dirname "$DATA_DIR")

RUN_SCRIPT="$SCRIPTS_DIR/run-experiments.sh"

## If one wishes to use a wrapper such as sbatch, use the WRAPPER env. var.

## Use `DRY_RUN=1` to only print what would have been done

## The benchmarks run concurrently by default; use `PARALLEL_BENCHMARKS=0` for
## the previous behaviour of waiting for each one to finish before the next

NAMES=(
heart_attack
obesity
)

KEYWORDS=(
HA50
OB50

)

TIMEOUT_PERS=(
5m
5m
)

export CPU_PERCENTAGE=${CPU_PERCENTAGE:-50}

export EXPERIMENTS_SPEC=${EXPERIMENTS_SPEC:-plain_itp}
export VARIANTS_SPEC=${VARIANTS_SPEC:-neuron_activations}

OPTIONS=()

[[ $DRY_RUN == 1 ]] && {
    OPTIONS+=(-n)
}

[[ -z $PARALLEL_BENCHMARKS ]] && PARALLEL_BENCHMARKS=1

## Without a WRAPPER all the benchmarks share the cores of a single machine,
## so the CPU budget has to be split among them; with a WRAPPER such as sbatch
## each one gets its own allocation and keeps the full budget.
(( $PARALLEL_BENCHMARKS )) && [[ -z $WRAPPER ]] && {
    CPU_PERCENTAGE=$(( ($CPU_PERCENTAGE + ${#KEYWORDS[@]}/2) / ${#KEYWORDS[@]} ))
    (( $CPU_PERCENTAGE < 1 )) && CPU_PERCENTAGE=1
    export CPU_PERCENTAGE

    printf "Sharing one machine: CPU_PERCENTAGE per benchmark is %s\n\n" "$CPU_PERCENTAGE"
}

PIDS=()

for i in ${!KEYWORDS[@]}; do
    KWD=${KEYWORDS[$i]}
    NAME="${NAMES[$i]}"
    TIMEOUT_PER=${TIMEOUT_PERS[$i]}

    declare -n lMODELS=${KWD}_MODELS
    lMODELS=("$DATA_DIR/models/$NAME/${NAME}_"*"x3.nnet")

    DATASET="$DATA_DIR/datasets/$NAME/${NAME}_s100_scaled.csv"

    declare -n lARGS=${KWD}_ARGS
    lARGS=()
    for MODEL in "${lMODELS[@]}"; do
        lARGS+=("$MODEL")
        lARGS+=("$DATASET")
    done

    export TIMEOUT_PER

    if (( $PARALLEL_BENCHMARKS )); then
        $WRAPPER "$RUN_SCRIPT" "${lARGS[@]}" $EXPERIMENTS_SPEC ${OPTIONS[@]} &
        PIDS+=($!)
        printf "Started in the background: %s (pid %s)\n" "$NAME" "$!"
    else
        $WRAPPER "$RUN_SCRIPT" "${lARGS[@]}" $EXPERIMENTS_SPEC ${OPTIONS[@]}
    fi
done

## Do not leave the benchmarks running if the script is interrupted
trap 'kill ${PIDS[@]} 2>/dev/null' INT TERM

STATUS=0
for i in ${!PIDS[@]}; do
    wait ${PIDS[$i]} || {
        printf "%s failed!\n" "${NAMES[$i]}" >&2
        STATUS=1
    }
done

exit $STATUS

