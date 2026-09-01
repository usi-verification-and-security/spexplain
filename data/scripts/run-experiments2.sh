#!/bin/bash

## ONNX counterpart of `run-experiments.sh`.
##
## Takes exactly the same positional arguments and forwards exactly the same spexplain options
## (e.g. --allow-neuron-vars-in-explanations, --fix-default-sample-neuron-activations,
## --prefer-default-sample-neuron-activations), but runs the `explain-onnx` action on an ONNX model
## resolved through `spec/models_datasets2` instead of the `explain` action on a `.nnet` model.
##
## Because ONNX files carry no per-feature input domain, the bounds come from INPUT_MINS/INPUT_MAXS
## in `spec/models_datasets2`, or from --input-min/--input-max on the command line.

export DIRNAME=$(dirname "$0")

source "$DIRNAME/lib/run2"

function usage {
    local experiments_spec_ary=($(ls "$EXPERIMENTS_SPEC_DIR"))

    printf "USAGE: %s <output_dir> <experiments_spec> [consecutive] [[+]reverse] [<max_samples>] [<filter_experiments_regex>] [<options>...] [-- <spexplain_args>...]\n" "$0"
    printf "\t<output_dir> must be specified in %s\n" "$MODELS_DATASETS2_SPEC"
    printf "\t<experiments_spec> is one of: %s\n" "${experiments_spec_ary[*]}"
    printf "CONSECUTIVE_EXPERIMENTS are not run unless 'consecutive' is provided\n"
    printf "\nOPTIONS:\n"
    printf "\t-h\t\t\tDisplay help message and exit\n"
    printf "\t-n\t\t\tDry mode - only print what would have been run\n"
    printf "\t--onnx <file>\t\tUse this .onnx model instead of the one from the spec\n"
    printf "\t--input-min <v1,v2,...>\tPer-feature input domain minimums (overrides the spec)\n"
    printf "\t--input-max <v1,v2,...>\tPer-feature input domain maximums (overrides the spec)\n"
    printf "\t--drop-sigmoid true|false\n"
    printf "\t\t\t\tDrop trailing sigmoid layer(s) (default: true)\n"
    printf "\t--no-quiet\t\tDo not pass --quiet to spexplain\n"
    printf "\nAny other option, and anything after '--', is forwarded verbatim to spexplain, e.g.\n"
    printf "\t--allow-neuron-vars-in-explanations true\n"
    printf "\t--fix-default-sample-neuron-activations all\n"
    printf "\t--prefer-default-sample-neuron-activations active\n"
    printf "The OPTIONS environment variable is still honoured, as in run-experiments.sh.\n"

    [[ -n $1 ]] && exit $1
}

[[ -z $1 ]] && usage 1 >&2

read_output_dir_arg="$1"
shift

read_experiments_spec "$1" || usage $? >&2
shift

maybe_read_consecutive "$1" && shift
maybe_read_reverse "$1" && shift
maybe_read_max_samples "$1" && shift

[[ -n $1 && ! $1 =~ ^- ]] && {
    export FILTER="$1"
    shift
}

export DRY_RUN=0
declare -a PASSTHROUGH_OPTIONS
PASSTHROUGH_OPTIONS=()

while [[ -n $1 ]]; do
    case "$1" in
    -h)
        ## `read_output_dir` has not run yet, but usage only needs the spec paths.
        usage 0
        ;;
    -n)
        DRY_RUN=1
        shift
        ;;
    --onnx)
        [[ -z $2 ]] && { printf "Expected a file after %s\n" "$1" >&2; usage 1 >&2; }
        export ONNX_MODEL_OVERRIDE="$2"
        shift 2
        ;;
    --onnx=*)
        export ONNX_MODEL_OVERRIDE="${1#*=}"
        shift
        ;;
    --input-min)
        [[ -z $2 ]] && { printf "Expected a value list after %s\n" "$1" >&2; usage 1 >&2; }
        export INPUT_MIN_OVERRIDE="$2"
        shift 2
        ;;
    --input-min=*)
        export INPUT_MIN_OVERRIDE="${1#*=}"
        shift
        ;;
    --input-max)
        [[ -z $2 ]] && { printf "Expected a value list after %s\n" "$1" >&2; usage 1 >&2; }
        export INPUT_MAX_OVERRIDE="$2"
        shift 2
        ;;
    --input-max=*)
        export INPUT_MAX_OVERRIDE="${1#*=}"
        shift
        ;;
    --drop-sigmoid)
        [[ -z $2 ]] && { printf "Expected true|false after %s\n" "$1" >&2; usage 1 >&2; }
        export DROP_SIGMOID="$2"
        shift 2
        ;;
    --drop-sigmoid=*)
        export DROP_SIGMOID="${1#*=}"
        shift
        ;;
    --no-quiet)
        export QUIET=0
        shift
        ;;
    --)
        shift
        PASSTHROUGH_OPTIONS+=("$@")
        break
        ;;
    --*)
        ## Any other long option (e.g. --allow-neuron-vars-in-explanations) goes to spexplain.
        ## Support both '--opt value' and '--opt=value'.
        if [[ $1 == *=* || -z $2 || $2 =~ ^- ]]; then
            PASSTHROUGH_OPTIONS+=("$1")
            shift
        else
            PASSTHROUGH_OPTIONS+=("$1=$2")
            shift 2
        fi
        ;;
    *)
        printf "Unrecognized option: %s\n" "$1" >&2
        usage 1 >&2
        ;;
    esac
done

read_output_dir "$read_output_dir_arg" || usage $? >&2

set_cmd
set_action
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

## Merge the OPTIONS environment variable (as in run-experiments.sh) with the parsed passthrough
## options; both end up as trailing spexplain arguments.
[[ -n $OPTIONS ]] && PASSTHROUGH_OPTIONS=($OPTIONS "${PASSTHROUGH_OPTIONS[@]}")
export EXTRA_OPTIONS="${PASSTHROUGH_OPTIONS[*]}"

printf "Output directory: %s\n" "$OUTPUT_DIR"
printf "ONNX model: %s\n" "$MODEL"
printf "Dataset: %s\n" "$DATASET"
printf "Input min: %s\n" "${INPUT_MIN:-<unset, defaults to 0 per feature>}"
printf "Input max: %s\n" "${INPUT_MAX:-<unset, defaults to 1 per feature>}"
[[ -n $DROP_SIGMOID ]] && printf "Drop sigmoid: %s\n" "$DROP_SIGMOID"
[[ -n $EXTRA_OPTIONS ]] && printf "Extra options: %s\n" "$EXTRA_OPTIONS"
printf "\n"

[[ -n $INCLUDE_REVERSE ]] && {
    printf "Running reversed-order experiments "
    if (( $REVERSE_ONLY )); then
        printf "only"
    else
        printf "as well"
    fi
    printf "\n\n"
}

(( $DRY_RUN )) && printf "DRY RUN - only printing what would be run\n\n"

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
    source "$DIRNAME/lib/run2"
    read_experiments_spec "$EXPERIMENTS_SPEC"

    local exp_idx=$1

    local -n lexperiment_names=$EXPERIMENT_NAMES_VAR

    local experiment=${lexperiment_names[$exp_idx]}
    [[ -n $FILTER && ! $experiment =~ $FILTER ]] && {
        printf "Skipping %s ...\n" $experiment
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

    printf "Running %s in the background ...\n" $experiment

    (( $DRY_RUN )) && return 0

    reverse_args=('')
    [[ -n $INCLUDE_REVERSE ]] && {
        (( $REVERSE_ONLY )) && reverse_args=()
        reverse_args+=(reverse)
    }

    for rev in "${reverse_args[@]}"; do
        SRC_EXPERIMENT=$src_experiment "$DIRNAME/run1-2.sh" "$OUTPUT_DIR" "$experiment_strategies" $experiment $rev $MAX_SAMPLES $EXTRA_OPTIONS &
    done

    wait -n
    case $? in
    0)
        printf "Finished %s\n" $experiment
        return 0
        ;;
    $TIMEOUT_STATUS)
        printf "Timeout %s\n" $experiment
        return 0
        ;;
    *)
        printf "%s failed!\nUsed command: %s\n" $experiment \
            "SRC_EXPERIMENT=$src_experiment \"$DIRNAME/run1-2.sh\" \"$OUTPUT_DIR\" \"$experiment_strategies\" $experiment $rev $MAX_SAMPLES $EXTRA_OPTIONS &" >&2
        return 1
        ;;
    esac
}
export -f run1

[[ -z $CPU_PERCENTAGE ]] && CPU_PERCENTAGE=60

[[ -n $INCLUDE_REVERSE ]] && (( ! $REVERSE_ONLY )) && CPU_PERCENTAGE=$(( $CPU_PERCENTAGE/2 ))

declare -n lEXPERIMENT_NAMES=$EXPERIMENT_NAMES_VAR

if (( ${#lEXPERIMENT_NAMES[@]} )); then
    parallel --halt soon,fail=1 --line-buffer --jobs ${CPU_PERCENTAGE}% 'run1 {}' ::: ${!lEXPERIMENT_NAMES[@]}
else
    printf "Nothing to run.\n"
fi

(( $? )) && {
    printf "\nFailed.\n"
    exit 1
}

printf "\nSuccess.\n"
exit 0
