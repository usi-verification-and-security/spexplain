#!/bin/bash

export DIRNAME=$(dirname "$0")

source "$DIRNAME/lib/run"

function usage {
    local experiments_spec_ary=($(ls "$EXPERIMENTS_SPEC_DIR"))

    ##+ it is still not possible to run multiple variants (different options)
    printf "USAGE: %s (<nn_model_fn> <dataset_fn>)... <experiments_spec> [consecutive] [<max_samples>] [<filter_experiments_regex>] [-h|-n]\n" "$0"
    printf "\t<experiments_spec> is one of: %s\n" "${experiments_spec_ary[*]}"
    printf "CONSECUTIVE_EXPERIMENTS are not run unless 'consecutive' is provided\n"
    printf "\nOPTIONS:\n"
    printf "\t-h\t\tDisplay help message and exit\n"
    printf "\t-n\t\tDry mode - only print what would have been run\n"

    [[ -n $1 ]] && exit $1
}

[[ -z $1 || -z $2 ]] && usage 1 >&2
[[ ! $1 =~ / ]] && {
    printf "Model filename does not contain '/': %s\n" "$1" >&2
    usage 1 >&2
}

MODELS=()
DATASETS=()
while [[ $1 =~ / ]]; do
    MODELS+=("$1")
    DATASETS+=("$2")
    shift 2
done

read_experiments_spec "$1" || usage $? >&2
shift

maybe_read_consecutive "$1" && shift
maybe_read_max_samples "$1" && shift

[[ -n $1 && ! $1 =~ ^- ]] && {
    export FILTER="$1"
    shift
}

export DRY_RUN=0
[[ $1 =~ ^- ]] && {
    if [[ $1 == -n ]]; then
        DRY_RUN=1
    elif [[ $1 == -h ]]; then
        usage 0
    else
        printf "Unrecognized option: %s\n" "$1" >&2
        usage 1 >&2
    fi
    shift
}

[[ -n $1 ]] && {
    printf "Additional arguments: %s\n" "$*" >&2
    usage 1 >&2
}

set_cmd
set_timeout

if [[ -z $INCLUDE_CONSECUTIVE ]]; then
    EXPERIMENT_NAMES_VAR=EXPERIMENT_NAMES
else
    (( $CONSECUTIVE_ONLY )) || {
        printf "Only isolated run of consecutive experiments is supported.\n" >&2
        usage 1 >&2
    }

    EXPERIMENT_NAMES_VAR=CONSECUTIVE_EXPERIMENTS_NAMES
fi
export EXPERIMENT_NAMES_VAR

(( $DRY_RUN )) && printf "DRY RUN - only printing what would be run\n\n"

[[ -n $OPTIONS ]] && export OPTIONS

for idx in ${!MODELS[@]}; do
    model="${MODELS[$idx]}"
    dataset="${DATASETS[$idx]}"

    set_output_dir_from_model_dataset "$model" "$dataset" || usage $? >&2

    printf "Model: %s\n" "$MODEL"
    printf "Dataset: %s\n" "$DATASET"
    printf "Output directory: %s\n" "$OUTPUT_DIR"
    printf "\n"
done

function cleanup {
    local code=$1

    [[ -n $code && $code != 0 ]] && {
        pkill -P $$
        wait
    }

    [[ -n $code ]] && exit $code
}

trap 'cleanup 9' INT TERM

function run1 {
    source "$DIRNAME/lib/run"
    read_experiments_spec "$EXPERIMENTS_SPEC"

    local model="$1"
    local dataset="$2"
    local exp_idx=$3

    set_output_dir_from_model_dataset "$model" "$dataset" || return $?

    local -n lexperiment_names=$EXPERIMENT_NAMES_VAR

    local experiment=${lexperiment_names[$exp_idx]}
    local experiment_full="${OUTPUT_DIR}/${experiment}"
    experiment_full="${experiment_full#explanations/}"

    [[ -n $FILTER && ! $experiment =~ $FILTER ]] && {
        printf "Skipping: %s\n" "$experiment_full"
        return 0
    }

    local {src,dst}_experiment
    [[ -n $INCLUDE_CONSECUTIVE ]] && {
        src_experiment=${CONSECUTIVE_EXPERIMENTS_SRC_NAMES[$exp_idx]}
        dst_experiment=${CONSECUTIVE_EXPERIMENTS_DST_NAMES[$exp_idx]}
    }

    local experiment_strategies
    if [[ -z $INCLUDE_CONSECUTIVE ]]; then
        find_strategies_for_experiment $experiment experiment_strategies $exp_idx
    else
        find_strategies_for_experiment $dst_experiment experiment_strategies
    fi

    printf "Running in the background: %s\n" "$experiment_full"

    (( $DRY_RUN )) && return 0

    ##+ allow running multiple variants
    SRC_EXPERIMENT=$src_experiment "$DIRNAME/run1.sh" "$MODEL" "$DATASET" "$experiment_strategies" $experiment $MAX_SAMPLES $OPTIONS &

    wait -n
    case $? in
    0)
        printf "Finished: %s\n" "$experiment_full"
        return 0
        ;;
    $TIMEOUT_STATUS)
        printf "Timeout: %s\n" "$experiment_full"
        return 0
        ;;
    *)
        printf "%s failed!\nUsed command: %s\n" $experiment \
            "SRC_EXPERIMENT=$src_experiment \"$DIRNAME/run1.sh\" \"$MODEL\" \"$DATASET\" \"$experiment_strategies\" $experiment $MAX_SAMPLES $OPTIONS &" >&2
        return 1
        ;;
    esac
}
export -f run1

[[ -z $CPU_PERCENTAGE ]] && CPU_PERCENTAGE=60

declare -n lEXPERIMENT_NAMES=$EXPERIMENT_NAMES_VAR

if (( ${#lEXPERIMENT_NAMES[@]} )); then
    parallel --halt soon,fail=1 --line-buffer --jobs ${CPU_PERCENTAGE}% 'run1 {} {} {}' ::: "${MODELS[@]}" :::+ "${DATASETS[@]}" ::: ${!lEXPERIMENT_NAMES[@]}
else
    printf "Nothing to run.\n"
fi

(( $? )) && {
    printf "\nFailed.\n"
    exit 1
}

printf "\nSuccess.\n"
exit 0
