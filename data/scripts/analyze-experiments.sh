#!/bin/bash

SCRIPTS_DIR=$(dirname "$0")

ANALYZE_SCRIPT="$SCRIPTS_DIR/analyze.sh"

source "$SCRIPTS_DIR/lib/experiments"
source "$SCRIPTS_DIR/lib/run"

function usage {
    printf "USAGE: %s <action> <explanations_dir> <experiments_spec> [[+]consecutive] [[+]reverse] [<max_samples>] [<filter_regex>] [<filter_regex2>] [-h|-f]\n" "$0"
    $ANALYZE_SCRIPT |& grep ACTIONS
    printf "\t[<filter_regex2>] is only to be used with binary actions\n"

    [[ -n $1 ]] && exit $1
}

[[ -z $1 ]] && usage 1 >&2

ACTION=$1
shift

[[ -z $1 ]] && {
    printf "Provide phi data directory.\n" >&2
    usage 1 >&2
}

EXPLANATIONS_DIR="$1"
shift
[[ -d $EXPLANATIONS_DIR && -r $EXPLANATIONS_DIR ]] || {
    printf "'%s' is not a readable directory.\n" "$EXPLANATIONS_DIR" >&2
    usage 1 >&2
}

#+ do not make this implicit assumption
PHI_DIR="$EXPLANATIONS_DIR/.."

PSI_FILE="$PHI_DIR/psi_"
case $ACTION in
check*)
    PSI_FILE+=c0
    ;;
*)
    PSI_FILE+=d
    ;;
esac
PSI_FILE+=.smt2
[[ -r $PSI_FILE ]] || {
    printf "Psi file '%s' is not readable.\n" "$PSI_FILE" >&2
    usage 2 >&2
}

read_experiments_spec "$1" || usage $? >&2
shift

maybe_read_consecutive "$1" && shift
maybe_read_reverse "$1" && shift
maybe_read_max_samples "$1" && shift

[[ -n $1 && ! $1 =~ ^- ]] && {
    FILTER="$1"
    shift
}

case $ACTION in
compare-subset)
    [[ -n $1 && ! $1 =~ ^- ]] && {
        FILTER2="$1"
        shift
    }
esac

FORCE_COMPUTE=0
[[ $1 =~ ^- ]] && {
    if [[ $1 == -f ]]; then
        FORCE_COMPUTE=1
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

set_timeout

#++ store consecutive cache to separate files to avoid destroying it
if [[ -z $INCLUDE_CONSECUTIVE ]]; then
    declare -n lEXPERIMENT_NAMES=EXPERIMENT_NAMES
    declare -n lMAX_EXPERIMENT_NAMES_LEN=MAX_EXPERIMENT_NAMES_LEN
elif (( $CONSECUTIVE_ONLY )); then
    declare -n lEXPERIMENT_NAMES=CONSECUTIVE_EXPERIMENTS_NAMES
    declare -n lMAX_EXPERIMENT_NAMES_LEN=MAX_CONSECUTIVE_EXPERIMENTS_NAMES_LEN
else
    declare -n lEXPERIMENT_NAMES=EXPERIMENT_NAMES_WITH_CONSECUTIVE
    declare -n lMAX_EXPERIMENT_NAMES_LEN=MAX_EXPERIMENT_NAMES_WITH_CONSECUTIVE_LEN
fi

## Require at least one extra space before the experiment names
case $ACTION in
compare-subset)
    VS_STR='vs.'
    EXPERIMENT_MAX_WIDTH=$(( 1 + ${#VS_STR}+2 + $lMAX_EXPERIMENT_NAMES_LEN*2 ))
    ;;
*)
    EXPERIMENT_MAX_WIDTH=$(( 1 + $lMAX_EXPERIMENT_NAMES_LEN ))
    ;;
esac

case $ACTION in
count-fixed|compare-subset)
    printf "%${EXPERIMENT_MAX_WIDTH}s" experiment
    ;;
esac

case $ACTION in
count-fixed)
    FIXED_CAPTION='%fixed'
    FIXED_MAX_WIDTH=${#FIXED_CAPTION}

    DIMENSION_CAPTION='%dimension'
    DIMENSION_MAX_WIDTH=${#DIMENSION_CAPTION}

    printf " | %s" "$FIXED_CAPTION"
    printf " | %s" "$DIMENSION_CAPTION"
    ;;
compare-subset)
    COMPARE_SUBSET_MAX_WIDTH=5
    SUBSET_CAPTION='<'
    SUBSET_MAX_WIDTH=$COMPARE_SUBSET_MAX_WIDTH
    EQUAL_CAPTION='='
    EQUAL_MAX_WIDTH=$COMPARE_SUBSET_MAX_WIDTH
    SUPSET_CAPTION='>'
    SUPSET_MAX_WIDTH=$COMPARE_SUBSET_MAX_WIDTH
    UNCOMPARABLE_CAPTION='NC'
    UNCOMPARABLE_MAX_WIDTH=$COMPARE_SUBSET_MAX_WIDTH

    printf " | %${SUBSET_MAX_WIDTH}s" $SUBSET_CAPTION
    printf " | %${EQUAL_MAX_WIDTH}s" $EQUAL_CAPTION
    printf " | %${SUPSET_MAX_WIDTH}s" $SUPSET_CAPTION
    printf " | %${UNCOMPARABLE_MAX_WIDTH}s" $UNCOMPARABLE_CAPTION
    ;;
esac

[[ -n $TIMEOUT_PER_CMD ]] && {
    case $ACTION in
    count-fixed)
        UNKNOWN_CAPTION='%unknown'
        UNKNOWN_MAX_WIDTH=${#UNKNOWN_CAPTION}
        ;;
    compare-subset)
        UNKNOWN_CAPTION='?'
        UNKNOWN_MAX_WIDTH=$COMPARE_SUBSET_MAX_WIDTH
        ;;
    esac
    case $ACTION in
    count-fixed|compare-subset)
        printf " | %${UNKNOWN_MAX_WIDTH}s" $UNKNOWN_CAPTION
        ;;
    esac
}

case $ACTION in
count-fixed|compare-subset)
    printf "\n"
    ;;
esac

function set_phi_filename {
    local experiment=$1
    local -n lphi_file=$2

    local experiment_stem=$experiment
    (( $do_reverse )) && experiment_stem=reverse/$experiment_stem
    [[ -n $MAX_SAMPLES ]] && experiment_stem=$MAX_SAMPLES_NAME/$experiment_stem

    lphi_file="${EXPLANATIONS_DIR}/${experiment_stem}.phi.txt"
}

[[ -n $FILTER ]] && {
    KEPT_IDXS=()
    for exp_idx in ${!lEXPERIMENT_NAMES[@]}; do
        experiment=${lEXPERIMENT_NAMES[$exp_idx]}
        [[ $experiment =~ $FILTER ]] && KEPT_IDXS+=($exp_idx)
    done
}

function maybe_replace_subexp {
    local -n dst_filter=$1
    local src_filter="$2"
    local src_str="$3"

    for i in {1..9}; do
        [[ $dst_filter =~ \\$i ]] || continue
        local match=$(sed -rn "s/^.*${src_filter}.*$/\\$i/p" <<<"$src_str")
        dst_filter="${dst_filter//\\$i/$match}"
    done

    return 0
}

MODEL="$(basename $(realpath "$PHI_DIR"))/$(basename "$EXPLANATIONS_DIR")"

SCRIPT_NAME=$(basename -s .sh "$0")
SCRIPT_OUTPUT_CACHE_FILE_REVERSE="$SCRIPTS_DIR/cache/$MODEL/$MAX_SAMPLES_NAME/reverse/${SCRIPT_NAME}.${ACTION}.txt"
SCRIPT_OUTPUT_CACHE_FILE="${SCRIPT_OUTPUT_CACHE_FILE_REVERSE/reverse\//}"

mkdir -p $(dirname "$SCRIPT_OUTPUT_CACHE_FILE_REVERSE") >/dev/null || exit $?

function get_cache_line {
    local -n lcache_line=$1
    local experiment=$2
    local experiment2=$3
    local try_swap=$4

    [[ -z $CACHE ]] && return 1

    local grep_cmd=(grep -Em1)

    local prefix=' '
    local suffix=' '

    case $ACTION in
    check*)
        prefix=/
        suffix='\.'
        ;;
    esac

    local experiment_regex="${prefix}${experiment}${suffix}"

    case $ACTION in
    check*|count-fixed)
        lcache_line=$(${grep_cmd[@]} "${experiment_regex}" <<<"$CACHE")
        ;;
    compare-subset)
        local experiment2_regex="${prefix}${experiment2}${suffix}"
        lcache_line=$(${grep_cmd[@]} "${experiment_regex}${VS_STR}${experiment2_regex}" <<<"$CACHE")
        [[ -n $try_swap && -z $lcache_line ]] && {
            lcache_line=$(${grep_cmd[@]} "${experiment2_regex}${VS_STR}${experiment_regex}" <<<"$CACHE")
            [[ -n $lcache_line ]] && {
                ##! works only if the regex contains directly printable characters!
                lcache_line="${lcache_line/${experiment2_regex}/|${experiment2_regex}}"
                local IFS='|'
                local array=($lcache_line)

                array[1]="${experiment_regex}${VS_STR}${experiment2_regex}"
                local tmp="${array[2]}"
                array[2]="${array[4]}"
                array[4]="$tmp"
                lcache_line="${array[*]}"
                lcache_line="${lcache_line/|/}"
            }
        }
        ;;
    esac

    [[ -n $lcache_line ]]
}

do_reverse_args=(0)
[[ -n $INCLUDE_REVERSE ]] && {
    (( $REVERSE_ONLY )) && do_reverse_args=()
    do_reverse_args+=(1)
}

for do_reverse in ${do_reverse_args[@]}; do
    if (( $do_reverse )); then
        declare -n lSCRIPT_OUTPUT_CACHE_FILE=SCRIPT_OUTPUT_CACHE_FILE_REVERSE
    else
        declare -n lSCRIPT_OUTPUT_CACHE_FILE=SCRIPT_OUTPUT_CACHE_FILE
    fi

    unset CACHE
    (( ! $FORCE_COMPUTE )) && [[ -r $lSCRIPT_OUTPUT_CACHE_FILE ]] && {
        CACHE=$(<"$lSCRIPT_OUTPUT_CACHE_FILE")
    }

    case $ACTION in
    *)
        [[ -e "${lSCRIPT_OUTPUT_CACHE_FILE}" ]] && mv "${lSCRIPT_OUTPUT_CACHE_FILE}" "${lSCRIPT_OUTPUT_CACHE_FILE}.bak"
        exec 3>&1
        exec > >(tee -i "${lSCRIPT_OUTPUT_CACHE_FILE}")
        >"${lSCRIPT_OUTPUT_CACHE_FILE}"
        [[ -n $FILTER ]] && FILTERED_OUTPUT_CACHE_FILE=$(mktemp)
        ;;
    esac

    case $ACTION in
    count-fixed|compare-subset)
        if (( $do_reverse )); then
            printf "REVERSE:\n"
        else
            printf "REGULAR:\n"
        fi
        ;;
    esac

    for exp_idx in ${!lEXPERIMENT_NAMES[@]}; do
        experiment=${lEXPERIMENT_NAMES[$exp_idx]}

        set_phi_filename $experiment phi_file

        case $ACTION in
        compare-subset)
            filter2="$FILTER2"
            maybe_replace_subexp filter2 "$FILTER" $experiment

            ARGS=(${lEXPERIMENT_NAMES[@]:$(($exp_idx+1))})
            [[ -n $FILTER ]] && {
                aux=(${lEXPERIMENT_NAMES[@]::$exp_idx})
                for fidx in ${KEPT_IDXS[@]}; do
                    (( $fidx >= $exp_idx )) && break
                    unset -v aux[$fidx]
                done
                ARGS=(${aux[@]} ${ARGS[@]})
            }
            ;;
        *)
            ARGS=(dummy)
            ;;
        esac

        any=0
        for arg in ${ARGS[@]}; do
            phi_files=("$phi_file")
            unset experiment2

            case $ACTION in
            compare-subset)
                experiment2=$arg

                set_phi_filename $experiment2 phi_file2
                phi_files+=("$phi_file2")
                ;;
            esac

            [[ -n $FILTER && ! $experiment =~ $FILTER ]] && {
                get_cache_line cache_line $experiment $experiment2 && printf "%s\n" "$cache_line" >>"$FILTERED_OUTPUT_CACHE_FILE"
                continue
            }

            case $ACTION in
            compare-subset)
                [[ -n $filter2 && ! $experiment2 =~ $filter2 ]] && {
                    get_cache_line cache_line $experiment $experiment2 1 && printf "%s\n" "$cache_line" >>"$FILTERED_OUTPUT_CACHE_FILE"
                    continue
                }
                ;;
            esac

            any=1

            get_cache_line cache_line $experiment $experiment2 1 && {
                printf "%s\n" "$cache_line"
                continue
            }

            args=()

            case $ACTION in
            check*)
                printf "Analyzing %s ... " "$phi_file"
                ;;
            count-fixed)
                printf "%${EXPERIMENT_MAX_WIDTH}s" $experiment
                ;;
            compare-subset)
                printf "%${EXPERIMENT_MAX_WIDTH}s" "$experiment $VS_STR $experiment2"

                ## Be flexible and accept even incomplete phi files (as in other actions) for any argument
                args+=(max)
                ;;
            esac

            out=$($ANALYZE_SCRIPT $ACTION "$PSI_FILE" "${phi_files[@]}" "${args[@]}") || {
                printf '%s\n' "$out" >&2
                exit $?
            }

            case $ACTION in
            check*)
                if [[ $out =~ ^OK ]]; then
                    printf "%s" "$out"
                else
                    printf "\n%s" "$out" >&2
                    exit 4
                fi
                ;;
            count-fixed)
                perc_fixed_features=$(sed -n 's/^.*#fixed features: \([^%]*\)%.*$/\1/p'  <<<"$out")

                if [[ -n $TIMEOUT_PER_CMD ]]; then
                    unknown_perc=$(sed -n 's/^.*#fixed features:.*(?: \([^%]*\)%.*)$/\1/p'  <<<"$out")

                    perc_dimension=$(bc -l <<<"100 - $perc_fixed_features - $unknown_perc")
                else
                    perc_dimension=$(bc -l <<<"100 - $perc_fixed_features")
                fi

                printf " |%${FIXED_MAX_WIDTH}.1f%%" $perc_fixed_features
                printf " |%${DIMENSION_MAX_WIDTH}.1f%%" $perc_dimension
                ;;
            compare-subset)
                outs=($(sed -n 's/[^:]*: \([0-9]*\)/\1 /pg' <<<"$out"))
                cnt=${outs[0]}
                subset_cnt=${outs[1]}
                equal_cnt=${outs[2]}
                supset_cnt=${outs[3]}
                uncomparable_cnt=${outs[4]}

                subset_perc=$(bc -l <<<"100 * ($subset_cnt / $cnt)")
                equal_perc=$(bc -l <<<"100 * ($equal_cnt / $cnt)")
                supset_perc=$(bc -l <<<"100 * ($supset_cnt / $cnt)")
                uncomparable_perc=$(bc -l <<<"100 * ($uncomparable_cnt / $cnt)")

                [[ -n $TIMEOUT_PER_CMD ]] && {
                    unknown_cnt=${outs[5]}

                    unknown_perc=$(bc -l <<<"100 * ($unknown_cnt / $cnt)")
                }

                printf " |%${SUBSET_MAX_WIDTH}.1f%%" $subset_perc
                printf " |%${EQUAL_MAX_WIDTH}.1f%%" $equal_perc
                printf " |%${SUPSET_MAX_WIDTH}.1f%%" $supset_perc
                printf " |%${UNCOMPARABLE_MAX_WIDTH}.1f%%" $uncomparable_perc
                ;;
            esac
            [[ -n $TIMEOUT_PER_CMD ]] && {
                case $ACTION in
                count-fixed|compare-subset)
                    printf " |%${UNKNOWN_MAX_WIDTH}.1f%%" $unknown_perc
                    ;;
                esac
            }
            printf "\n"
        done

        case $ACTION in
        compare-subset)
            (( $any )) && printf "\n"
            ;;
        esac
    done

    case $ACTION in
    *)
        exec >&3 3>&-
        [[ -n $FILTER ]] && {
            cat "$FILTERED_OUTPUT_CACHE_FILE" >>"${lSCRIPT_OUTPUT_CACHE_FILE}"
            rm "$FILTERED_OUTPUT_CACHE_FILE"
        }
        ;;
    esac
done

exit 0
