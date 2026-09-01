#!/bin/bash

## ONNX counterpart of `run1.sh`: runs a single experiment for one ONNX model.
##
## Differences from `run1.sh`:
##   * uses `lib/run2` (ONNX model/dataset spec `spec/models_datasets2`),
##   * invokes the `explain-onnx` action instead of the default `explain`,
##   * passes `--input-min`/`--input-max` (and optionally `--drop-sigmoid`), which are ONNX-only.
## Everything else -- output file naming, --format/--max-samples/--reverse-var/--time-limit-per
## handling, SRC_EXPERIMENT chaining -- is identical.

DIRNAME=$(dirname "$0")

source "$DIRNAME/lib/run2"

function usage {
    printf "USAGE: %s <output_dir> <exp_strategies_spec> [<name>] [reverse] [<max_samples>] <args>...\n" "$0"
    printf "\t<output_dir> must be specified in %s\n" "$MODELS_DATASETS2_SPEC"
    printf "\nENVIRONMENT:\n"
    printf "\tONNX_MODEL_OVERRIDE\tUse this .onnx file instead of the spec entry\n"
    printf "\tINPUT_MIN_OVERRIDE\tComma-separated per-feature input minimums\n"
    printf "\tINPUT_MAX_OVERRIDE\tComma-separated per-feature input maximums\n"
    printf "\tDROP_SIGMOID\t\ttrue|false, passed as --drop-sigmoid (default: unset, i.e. true)\n"
    printf "\tQUIET\t\t\tSet to 0 to omit --quiet (default: 1)\n"

    [[ -n $1 ]] && exit $1
}

[[ -z $1 ]] && usage 1 >&2

read_output_dir "$1" || usage $? >&2
shift

[[ -z $1 || $1 =~ ^(reverse|short)$ ]] && usage 1 >&2
STRATEGIES="$1"
shift

if [[ -z $1 || $1 =~ ^(reverse|short)$ ]]; then
    set_experiment_name_from_strategies EXPERIMENT "$STRATEGIES"
else
    EXPERIMENT="$1"
    shift
fi

[[ $1 == reverse ]] && {
    REVERSE=1
    shift
}

maybe_read_max_samples "$1" && shift

[[ $1 =~ ^(reverse|short)$ ]] && usage 1 >&2

set_cmd
set_action
set_timeout

declare -a OPTIONS
OPTIONS=(--format=smtlib2)

## `--quiet` is known to segfault in the current build, so make it opt-out-able.
[[ -z $QUIET ]] && QUIET=1
(( $QUIET )) && OPTIONS=(--quiet "${OPTIONS[@]}")

append_onnx_options OPTIONS

[[ -n $MAX_SAMPLES ]] && {
    OUTPUT_DIR+=/$MAX_SAMPLES_NAME
    OPTIONS+=(--shuffle-samples --max-samples=$MAX_SAMPLES)
}

[[ -n $REVERSE ]] && {
    OUTPUT_DIR+=/reverse
    OPTIONS+=(--reverse-var)
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

    OPTIONS+=(--time-limit-per=$TIMEOUT_PER_MS)
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

    OPTIONS+=(--input-explanations=\"$src_phi_file\")
}

OPTIONS+=(
    --output-explanations=\"$phi_file\"
    --output-stats=\"$stats_file\"
    --output-times=\"$times_file\"
)

exec $TIMEOUT_CMD bash -c "{ time ${CMD} $ACTION \"$MODEL\" \"$DATASET\" \"$STRATEGIES\" ${OPTIONS[*]} "'"$@"'" >\"$out_file\" 2>\"$err_file\" ; } 2>\"$time_file\"" spexplain "$@"
