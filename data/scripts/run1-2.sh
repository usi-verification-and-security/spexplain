#!/bin/bash

## ONNX counterpart of `run1.sh`: runs a single experiment for one ONNX model and dataset.
##
## Same arguments, output layout, variants, neuron-activation files, SRC_EXPERIMENT chaining and
## TIMEOUT/TIMEOUT_PER handling as `run1.sh`. The differences are:
##   * uses `lib/run2` (models may live in subdirectories, e.g. models/mnist/cnn-bench/S1.onnx),
##   * invokes the `explain-onnx` action instead of the default `explain`,
##   * passes the ONNX-only `--input-min`/`--input-max` (and optionally `--drop-sigmoid`).

SCRIPTS_DIR=$(dirname "$0")

source "$SCRIPTS_DIR/lib/run2"

function usage {
    printf "USAGE: %s <onnx_model_fn> <dataset_fn> <exp_strategies_spec> [<name>] [<max_samples>] <args>...\n" "$0"
    printf "\nENVIRONMENT (in addition to those of run1.sh):\n"
    printf "\tINPUT_MIN, INPUT_MAX\tInput domain bounds: one number for all features, or one value per\n"
    printf "\t\t\t\tfeature (comma-separated). Default: the entry of the model in\n"
    printf "\t\t\t\t%s, else [0,1]\n" "$MODELS_DATASETS2_SPEC"
    printf "\tDROP_SIGMOID\t\ttrue|false, passed as --drop-sigmoid (default: unset, i.e. true)\n"
    printf "\tQUIET\t\t\tSet to 0 to omit --quiet (default: 1)\n"

    [[ -n $1 ]] && exit $1
}

[[ -z $1 || -z $2 ]] && usage 1 >&2

set_output_dir_from_model_dataset "$1" "$2" || usage $? >&2
shift 2

[[ -z $1 || $1 == short ]] && usage 1 >&2
STRATEGIES="$1"
shift

if [[ -z $1 || $1 == short || $1 =~ ^- ]]; then
    set_experiment_name_from_strategies EXPERIMENT "$STRATEGIES"
else
    EXPERIMENT="$1"
    shift
fi

maybe_read_max_samples "$1" && shift

[[ $1 == short ]] && usage 1 >&2

set_cmd
set_action
set_timeout
set_input_bounds

[[ -n $VARIANT ]] && {
    maybe_find_options_for_variant "$VARIANT" VAR_OPTIONS
}

activations="${MODEL/models\//neuron_activations\/}"
activations="${activations%.*}.txt"
[[ -r $activations ]] && {
    for options_var in VAR_OPTIONS OPTIONS; do
        declare -n lOPTIONS=$options_var
        for opt in --input-{fix,prefer}-sample-neuron-activations; do
            [[ $lOPTIONS =~ ${opt}=\"\" ]] || continue
            lOPTIONS="${lOPTIONS//${opt}=\"\"/${opt}=\"$activations\"}"
        done
    done
}

declare -a options
options=(
    --format=smtlib2
)

## Opt-out-able, as --quiet was known to segfault in some builds.
[[ -z $QUIET ]] && QUIET=1
(( $QUIET )) && options=(--quiet "${options[@]}")

append_onnx_options options

[[ -n $MAX_SAMPLES ]] && {
    OUTPUT_DIR+=/$MAX_SAMPLES_NAME
    options+=(--shuffle-samples --max-samples=$MAX_SAMPLES)
}

[[ -n $TIMEOUT_PER ]] && {
    [[ $TIMEOUT_PER =~ ^[0-9]+(\.[0-9]*)?[smhd]?$ ]] || {
        printf "Unrecognized timeout per explanations: %s\n" "$TIMEOUT_PER" >&2
        usage 1 >&2
    }
    if [[ $TIMEOUT_PER =~ d$ ]]; then
        TIMEOUT_PER_MS=$(bc -l <<<"${TIMEOUT_PER%d} * 1000 * 60 * 60 * 24")
    elif [[ $TIMEOUT_PER =~ h$ ]]; then
        TIMEOUT_PER_MS=$(bc -l <<<"${TIMEOUT_PER%h} * 1000 * 60 * 60")
    elif [[ $TIMEOUT_PER =~ m$ ]]; then
        TIMEOUT_PER_MS=$(bc -l <<<"${TIMEOUT_PER%m} * 1000 * 60")
    else
        TIMEOUT_PER_MS=$(bc -l <<<"${TIMEOUT_PER%s} * 1000")
    fi
    TIMEOUT_PER_MS=${TIMEOUT_PER_MS%.*}

    options+=(--time-limit-per=$TIMEOUT_PER_MS)
}

mkdir -p "$OUTPUT_DIR" >/dev/null || exit $?

function set_file {
    local file_var=$1
    local experiment="$2"
    local type=$3

    local -n lfile=$file_var
    lfile="${OUTPUT_DIR}/${experiment}.${type}.txt"
}

for t in out err phi stats times time; do
    set_file ${t}_file "$EXPERIMENT" $t
done

[[ -n $SRC_EXPERIMENT ]] && {
    set_file src_phi_file "$SRC_EXPERIMENT" phi

    options+=(--input-explanations=\"$src_phi_file\")
}

options+=(
    --output-explanations=\"$phi_file\"
    --output-stats=\"$stats_file\"
    --output-times=\"$times_file\"
)

exec $TIMEOUT_CMD bash -c "{ time ${CMD} $ACTION \"$MODEL\" \"$DATASET\" \"$STRATEGIES\" ${options[*]} $VAR_OPTIONS $OPTIONS "'"$@"'" >\"$out_file\" 2>\"$err_file\" ; } 2>\"$time_file\"" spexplain "$@"
