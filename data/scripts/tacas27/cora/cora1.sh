#!/usr/bin/env bash
## The explanations of CORA (neuralNetwork/explain, abstract+refine) of all samples of a dataset, with the timeout
## TIMEOUT_PER each; the counterpart of verix/verix1.sh for CORA (see runme.sh run-cora).
## USAGE: ./cora1.sh <model.nnet> <dataset.csv> <epsilon> <experiment>
## E.g.:  TIMEOUT_PER=10m ./cora1.sh data/models/mnist/mnist_50x1.nnet data/datasets/mnist/mnist_s100_scaled.csv 1 eps100
## The explanations go to data/explanations/<name>/<model>/<dataset>/cora/<experiment>.phi.txt,
## and the results of CORA of each sample (as of verix1.py, see to_phi.py) to the directory <experiment>/ next to it,
## made by cora_explain_dataset.m (here), all samples in one process: either the application compiled by build.sh
## on the MATLAB Runtime (CORA_BIN, as in the image docker/tacas27-cora, no license needed), or MATLAB itself
## (its run time per sample is measured inside, without the startup).
## Returns 0 even if some explanations timed out (they are '<null>'), 1 if some failed.
##
## Env. variables:
##   TIMEOUT_PER   the timeout per explanation (as of the command timeout)
##   CORA_BIN, MCR_ROOT, CORA_DIR, MATLAB   how CORA runs (the compiled application or MATLAB), see backend
##   CORA_RESUME   if 1, keep the results of the samples that already finished
##   CORA_PIN      if not 0 (default: 1), run on one CPU of those allowed, by the job slot of GNU parallel
##                 (PARALLEL_JOBSLOT, e.g. of runme.sh), so that the JIT compiler of MATLAB, which runs in extra
##                 threads, does not slow down the other jobs; without the slot, no pinning

CORA_SCRIPTS_DIR=$(dirname "$(realpath "$0")")
SCRIPTS_DIR=$(dirname "$(dirname "$CORA_SCRIPTS_DIR")")

source "$SCRIPTS_DIR/lib/run"

function usage {
    awk 'NR > 1 && !/^##/ { exit } /^##/ { sub(/^## ?/, ""); print }' "$0"

    [[ -n $1 ]] && exit $1
}

(( $# == 4 )) || usage 1 >&2

VARIANT=cora set_output_dir_from_model_dataset "$1" "$2" || usage $? >&2
EPSILON=$3
EXPERIMENT=$4

source "$CORA_SCRIPTS_DIR/backend"
set_cora_backend || exit 1

## <timeout as of the command timeout> -> seconds
function to_secs {
    local t=$1
    case $t in
    *d) printf "%d" $(( ${t%d}*86400 )) ;;
    *h) printf "%d" $(( ${t%h}*3600 )) ;;
    *m) printf "%d" $(( ${t%m}*60 )) ;;
    *s) printf "%d" ${t%s} ;;
    *)  printf "%d" $t ;;
    esac
}

## The benchmark, e.g. mnist
NAME=$(basename "$(dirname "$MODEL")")
## The shape of its samples [H W C] (the tabular ones are [<features> 1 1], the default),
## and the channels of the images (as in verix1.sh)
case $NAME in
mnist)       INPUT_SIZE='[28 28 1]'; CHANNELS=1 ;;
gtsrb|cifar) INPUT_SIZE='[32 32 3]'; CHANNELS=3 ;;
*)           INPUT_SIZE=;            CHANNELS=1 ;;
esac

RESULTS_DIR="$OUTPUT_DIR/$EXPERIMENT"
## Otherwise the old results would count as finished
[[ $CORA_RESUME == 1 ]] || rm -rf "$RESULTS_DIR" "$RESULTS_DIR.log" || exit $?
mkdir -p "$RESULTS_DIR" || exit $?

## The first line is the header
n_samples=$(( $(awk 'END { print NR }' "$DATASET") - 1 ))

TIMEOUT_SECS=Inf
TIMEOUT_ALL_CMD=
[[ -n $TIMEOUT_PER ]] && {
    TIMEOUT_SECS=$(to_secs $TIMEOUT_PER)
    ## explain stops by itself before its timeout, this is only against a hang (e.g. in one verification),
    ## with the startup of MATLAB and the warm-up (one more explanation, see cora_explain_dataset.m);
    ## the samples that did not finish count as timeouts
    TIMEOUT_ALL_CMD="timeout --signal=KILL $(( ($n_samples + 1)*2*$TIMEOUT_SECS + 300 ))"
}

ARGS=("$(realpath "$MODEL")" "$(realpath "$DATASET")" $EPSILON "$(realpath "$RESULTS_DIR")" Timeout $TIMEOUT_SECS Resume true)
[[ -n $INPUT_SIZE ]] && ARGS+=(InputSize "$INPUT_SIZE")

## The CPUs allowed to this process (e.g. '0-31' or '0-3,8-11'; on Eiger, the allocation) -> one per line
function allowed_cpus {
    awk '/^Cpus_allowed_list:/ {
        n = split($2, ranges, ",")
        for (i = 1; i <= n; ++i) {
            m = split(ranges[i], bounds, "-")
            for (c = bounds[1]; c <= bounds[m]; ++c) print c
        }
    }' /proc/self/status
}

## The allowed CPUs, first one per physical core (the first of its hardware threads), then the other threads,
## so that the jobs share no core while there are enough cores
function allowed_cpus_by_core {
    local cpu siblings first
    local -a primary=() secondary=()
    for cpu in $(allowed_cpus); do
        siblings=$(cat /sys/devices/system/cpu/cpu$cpu/topology/thread_siblings_list 2>/dev/null)
        first=${siblings%%[,-]*}
        if [[ -z $siblings || $first == $cpu ]]; then
            primary+=($cpu)
        else
            secondary+=($cpu)
        fi
    done
    printf "%s\n" "${primary[@]}" "${secondary[@]}"
}

## One CPU per job, by its slot (at most one job per slot at a time, and fewer slots than CPUs)
PIN_CMD=
if [[ $CORA_PIN != 0 && -n $PARALLEL_JOBSLOT ]] && command -v taskset >/dev/null; then
    cpus=($(allowed_cpus_by_core))
    (( ${#cpus[@]} )) && {
        cpu=${cpus[$(( ($PARALLEL_JOBSLOT - 1) % ${#cpus[@]} ))]}
        PIN_CMD="taskset -c $cpu"
        printf "CPU: %d (job slot %d)\n" $cpu $PARALLEL_JOBSLOT >>"$RESULTS_DIR.log"
    }
fi

## One thread, as each job of the queue (as those of the engine and of VeriX-plus); the compiled application
## is built so (build_cora_explain.m)
if [[ -n $CORA_BIN ]]; then
    ## The extracted application, per job, so that the parallel jobs do not extract it into the same place
    export MCR_CACHE_ROOT=$(mktemp -d)
    $TIMEOUT_ALL_CMD $PIN_CMD "$CORA_BIN" "$MCR_ROOT" "${ARGS[@]}" >>"$RESULTS_DIR.log" 2>&1
    status=$?
    rm -rf "$MCR_CACHE_ROOT"
else
    ## The arguments as MATLAB char literals
    MATLAB_ARGS=$(printf ",'%s'" "${ARGS[@]}")
    ## -batch would start a virtual display (Xvfb)
    $TIMEOUT_ALL_CMD $PIN_CMD "$MATLAB" -nosplash -nodesktop -nojvm -singleCompThread \
        -r "addpath(genpath('$CORA_DIR')); addpath('$CORA_SCRIPTS_DIR'); try, cora_explain_dataset(${MATLAB_ARGS#,}); catch e, disp(getReport(e)); exit(1); end; exit" </dev/null >>"$RESULTS_DIR.log" 2>&1
    status=$?
fi
(( $status )) && printf "CORA exited with %d on %s, see %s\n" $status "$DATASET" "$RESULTS_DIR.log" >&2

python3 "$SCRIPTS_DIR/tacas27/verix/to_phi.py" "$DATASET" $EPSILON "$RESULTS_DIR" $CHANNELS >"$OUTPUT_DIR/$EXPERIMENT.phi.txt" || exit $?

n_timeouts=0
n_failed=0
for (( idx=0; idx < $n_samples; ++idx )); do
    if [[ -r $RESULTS_DIR/$idx/error.txt ]]; then
        printf "CORA failed on sample %d (from 0) of %s, see %s\n" $idx "$DATASET" "$RESULTS_DIR/$idx/error.txt" >&2
        (( ++n_failed ))
    elif [[ -r $RESULTS_DIR/$idx/time.txt ]]; then
        :
    elif [[ -r $RESULTS_DIR/$idx/timeout.txt ]] || (( $status == 137 )); then
        ## Stopped by itself, or the sample was not reached before the kill (TIMEOUT_ALL_CMD)
        (( ++n_timeouts ))
    else
        ## E.g., CORA did not start at all
        (( ++n_failed ))
    fi
done

printf "%s: %d explanations, %d timeouts, %d failed\n" "$OUTPUT_DIR/$EXPERIMENT.phi.txt" $n_samples $n_timeouts $n_failed

(( $n_failed || ($status && $status != 137) )) && exit 1
exit 0
