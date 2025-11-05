#!/bin/bash

DIRNAME=$(dirname "$0")

source "$DIRNAME/lib/run"

function usage {
    printf "USAGE: %s <output_dir> <exp_strategies_spec> <name> [reverse] [<max_samples>] <args>...\n" "$0"

    [[ -n $1 ]] && exit $1
}

[[ -z $1 ]] && usage 1 >&2

read_output_dir "$1" || usage $? >&2
shift

[[ -z $1 || $1 =~ ^(reverse|short)$ ]] && usage 1 >&2
STRATEGIES="$1"
shift

[[ -z $1 || $1 =~ ^(reverse|short)$ ]] && usage 1 >&2
EXPERIMENT="$1"
shift

[[ $1 == reverse ]] && {
    REVERSE=1
    shift
}

maybe_read_max_samples "$1" && shift

[[ $1 =~ ^(reverse|short)$ ]] && usage 1 >&2

set_cmd
set_timeout

declare -a OPTIONS
OPTIONS=(--quiet --format=smtlib2)

[[ -n $MAX_SAMPLES ]] && {
    OUTPUT_DIR+=/$MAX_SAMPLES_NAME
    OPTIONS+=(--shuffle-samples --max-samples=$MAX_SAMPLES)
}

[[ -n $REVERSE ]] && {
    OUTPUT_DIR+=/reverse
    OPTIONS+=(--reverse-var)
}

mkdir -p "$OUTPUT_DIR" >/dev/null || exit $?

function set_file {
    local file_var=$1
    local experiment="$2"
    local type=$3

    local -n lfile=$file_var
    lfile="${OUTPUT_DIR}/${experiment}.${type}.txt"
}

for t in out err phi stats time; do
    set_file ${t}_file "$EXPERIMENT" $t
done

[[ -n $SRC_EXPERIMENT ]] && {
    set_file src_phi_file "$SRC_EXPERIMENT" phi

    OPTIONS+=(--input-explanations=\"$src_phi_file\")
}

OPTIONS+=(
    --output-explanations=\"$phi_file\"
    --output-stats=\"$stats_file\"
)

exec timeout $TIMEOUT bash -c "{ time ${CMD} \"$MODEL\" \"$DATASET\" \"$STRATEGIES\" ${OPTIONS[*]} "'"$@"'" >\"$out_file\" 2>\"$err_file\" ; } 2>\"$time_file\"" spexplain "$@"
