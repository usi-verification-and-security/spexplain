#!/usr/bin/env bash
## The explanations of VeriX-plus of all samples of a dataset, one at a time with the timeout TIMEOUT_PER each,
## as in the original experiments; the counterpart of run1.sh for VeriX-plus (see runme.sh run-verix-plus).
## USAGE: ./verix1.sh <model.nnet> <dataset.csv> <epsilon> <experiment>
## E.g.:  TIMEOUT_PER=10m ./verix1.sh data/models/mnist/mnist_50x1.nnet data/datasets/mnist/mnist_s100_scaled.csv 1 eps100
## The networks for VeriX-plus are made from the .nnet, into the directory <experiment>/models/.
## The explanations go to data/explanations/<name>/<model>/<dataset>/verix/<experiment>.phi.txt,
## and the results of VeriX-plus of each sample to the directory <experiment>/ next to it.
## Returns 0 even if some explanations timed out (they are '<null>'), 1 if some failed.
##
## Env. variables:
##   TIMEOUT_PER   the timeout per explanation (as of the command timeout)
##   VERIX_DIR     VeriX-plus (default: ~/VeriX-plus)
##   VERIX_ORDER   the traversal order (default: random, see verix1.py)
##   VERIX_PYTHON  the Python of VeriX-plus (default: $VERIX_DIR/venv/bin/python)

VERIX_SCRIPTS_DIR=$(dirname "$(realpath "$0")")
SCRIPTS_DIR=$(dirname "$(dirname "$VERIX_SCRIPTS_DIR")")

source "$SCRIPTS_DIR/lib/run"

function usage {
    awk 'NR > 1 && !/^##/ { exit } /^##/ { sub(/^## ?/, ""); print }' "$0"

    [[ -n $1 ]] && exit $1
}

(( $# == 4 )) || usage 1 >&2

VARIANT=verix set_output_dir_from_model_dataset "$1" "$2" || usage $? >&2
EPSILON=$3
EXPERIMENT=$4

export VERIX_DIR=${VERIX_DIR:-$HOME/VeriX-plus}
VERIX_PYTHON=${VERIX_PYTHON:-$VERIX_DIR/venv/bin/python}

set_timeout
## Marabou installs its handler of SIGTERM in the process with the first query, which the queries of VeriX-plus
## (as MILP) never check, so SIGTERM does not stop it (or crashes it, the handler refers to finished queries)
[[ -n $TIMEOUT_PER ]] && TIMEOUT_PER_CMD="timeout --signal=KILL $TIMEOUT_PER"
## The exit status of timeout with SIGKILL
TIMEOUT_KILL_STATUS=$(( 128 + 9 ))

## The benchmark, e.g. mnist
NAME=$(basename "$(dirname "$MODEL")")
## The channels of its images (as BENCHMARKS in verix1.py)
case $NAME in
gtsrb|cifar) CHANNELS=3 ;;
*)           CHANNELS=1 ;;
esac

RESULTS_DIR="$OUTPUT_DIR/$EXPERIMENT"
mkdir -p "$RESULTS_DIR" || exit $?
## The models for VeriX-plus are made from the .nnet by verix1.py, anew in each job
rm -rf "$RESULTS_DIR/models"

## The first line is the header
n_samples=$(( $(awk 'END { print NR }' "$DATASET") - 1 ))

n_timeouts=0
n_failed=0
for (( idx=0; idx < $n_samples; ++idx )); do
    rm -rf "$RESULTS_DIR/$idx"

    $TIMEOUT_PER_CMD "$VERIX_PYTHON" "$VERIX_SCRIPTS_DIR/verix1.py" \
        $NAME "$MODEL" "$DATASET" $idx $EPSILON "$RESULTS_DIR" >"$RESULTS_DIR/$idx.log" 2>&1
    case $? in
    0) ;;
    $TIMEOUT_STATUS|$TIMEOUT_KILL_STATUS)
        (( ++n_timeouts ))
        ;;
    *)
        printf "VeriX-plus failed on sample %d (from 0) of %s, see %s\n" $idx "$DATASET" "$RESULTS_DIR/$idx.log" >&2
        (( ++n_failed ))
        ;;
    esac
done

python3 "$VERIX_SCRIPTS_DIR/to_phi.py" "$DATASET" $EPSILON "$RESULTS_DIR" $CHANNELS >"$OUTPUT_DIR/$EXPERIMENT.phi.txt" || exit $?

printf "%s: %d explanations, %d timeouts, %d failed\n" "$OUTPUT_DIR/$EXPERIMENT.phi.txt" $n_samples $n_timeouts $n_failed

(( $n_failed )) && exit 1
exit 0
