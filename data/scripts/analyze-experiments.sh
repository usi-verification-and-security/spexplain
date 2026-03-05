#!/bin/bash

SCRIPTS_DIR=$(dirname "$0")
SCRIPT_NAME=$(basename -s .sh "$0")

ANALYZE_SCRIPT="$SCRIPTS_DIR/analyze.sh"

source "$SCRIPTS_DIR/lib/experiments"
source "$SCRIPTS_DIR/lib/run"

function usage {
    printf "USAGE: %s <action> <explanations_dir>... <experiments_spec> [[+]consecutive] [[+]reverse] [<max_samples>] [<filter_regex>] [<filter_regex2>] [-h|-f]\n" "$0"
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

[[ ! $1 =~ / ]] && {
    printf "Phi data directory does not contain '/': %s\n" "$1" >&2
    usage 1 >&2
}

unset EXPLANATIONS_DIR_BASE
unset PHI_DIR
EXPLANATIONS_DIRS=()
VARIANTS=()
while [[ $1 =~ / ]]; do
    EXPLANATIONS_DIR=$(realpath --relative-to="$SCRIPTS_DIR/.." "$1")
    shift

    [[ -d $EXPLANATIONS_DIR && -r $EXPLANATIONS_DIR ]] || {
        printf "'%s' is not a readable directory.\n" "$EXPLANATIONS_DIR" >&2
        usage 1 >&2
    }
    EXPLANATIONS_DIRS+=("$EXPLANATIONS_DIR")

    explanations_dir_base=$(dirname "$EXPLANATIONS_DIR")
    #+ do not make this implicit assumption
    phi_dir=$(dirname "$explanations_dir_base")
    if [[ -z $EXPLANATIONS_DIR_BASE ]]; then
        EXPLANATIONS_DIR_BASE="$explanations_dir_base"
        PHI_DIR="$phi_dir"
    elif [[ $explanations_dir_base != $EXPLANATIONS_DIR_BASE ]]; then
        printf "%s is not compatible with %s\n" "$EXPLANATIONS_DIR" "${EXPLANATIONS_DIRS[0]}" >&2
        exit 2
    fi

    VARIANT=$(basename "$EXPLANATIONS_DIR")
    VARIANTS+=("$VARIANT")
done

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
    if [[ -n $1 && ! $1 =~ ^- ]]; then
        FILTER2="$1"
        shift
    else
        FILTER2="$FILTER"
    fi
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
    local explanations_dir="$1"
    local experiment=$2
    local -n lphi_file=$3

    local experiment_stem=$experiment
    (( $do_reverse )) && experiment_stem=reverse/$experiment_stem
    [[ -n $MAX_SAMPLES ]] && experiment_stem=$MAX_SAMPLES_NAME/$experiment_stem

    lphi_file="${explanations_dir}/${experiment_stem}.phi.txt"
}

[[ -n $FILTER ]] && {
    FILTER_IDXS=()
    FILTER_IDXS_C=()

    for exp_idx in ${!lEXPERIMENT_NAMES[@]}; do
        experiment=${lEXPERIMENT_NAMES[$exp_idx]}
        if [[ $experiment =~ $FILTER ]]; then
            FILTER_IDXS+=($exp_idx)
        else
            FILTER_IDXS_C+=($exp_idx)
        fi
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

function cleanup {
    local code=$1

    [[ -n $code && $code != 0 ]] && {
        pkill -P $$
        wait

        sed -i '$d' "${lSCRIPT_OUTPUT_CACHE_FILE}"
    }

    close_cache

    [[ -n $code && $code != 0 ]] && {
        [[ -n $CACHE ]] && {
            local tmp=$(mktemp)
            { cat "${lSCRIPT_OUTPUT_CACHE_FILE}" && echo "$CACHE"; } | sort | uniq >$tmp
            cat $tmp >"${lSCRIPT_OUTPUT_CACHE_FILE}"
            rm $tmp
        }
    }

    [[ -n $code ]] && exit $code
}

trap 'cleanup 9' INT TERM

CACHE_DIR_BASE="${EXPLANATIONS_DIR_BASE}/"
for VARIANT in "${VARIANTS[@]}"; do
    CACHE_DIR_BASE+="${VARIANT}_"
done
CACHE_DIR_BASE="${CACHE_DIR_BASE%_}"

[[ ${#VARIANTS[@]} == 1 && $CACHE_DIR_BASE != $EXPLANATIONS_DIR ]] && {
    printf "Unexpected mismatch of CACHE_DIR_BASE and EXPLANATIONS_DIR:\n%s != %s\n" "$CACHE_DIR_BASE" "$EXPLANATIONS_DIR" >&2
    exit 7
}

SCRIPT_OUTPUT_CACHE_FILE_REVERSE="$SCRIPTS_DIR/cache/$CACHE_DIR_BASE/$MAX_SAMPLES_NAME/reverse/${SCRIPT_NAME}.${ACTION}.txt"
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

function close_cache {
    exec >&3 3>&-

    [[ -n $FILTER && -e $FILTERED_OUTPUT_CACHE_FILE ]] && {
        cat "$FILTERED_OUTPUT_CACHE_FILE" >>"${lSCRIPT_OUTPUT_CACHE_FILE}"
        rm "$FILTERED_OUTPUT_CACHE_FILE"
    }
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

    for vidx in ${!VARIANTS[@]}; do

    variant="${VARIANTS[$vidx]}"
    explanations_dir="${EXPLANATIONS_DIRS[$vidx]}"

    ##+ add indentation vvv
    for exp_idx in ${!lEXPERIMENT_NAMES[@]}; do
        experiment=${lEXPERIMENT_NAMES[$exp_idx]}
        experiment_full="${variant}/${experiment}"

        set_phi_filename "$explanations_dir" $experiment phi_file

        case $ACTION in
        compare-subset)
            filter2="$FILTER2"
            maybe_replace_subexp filter2 "$FILTER" $experiment

            ARGS=()
            for vidx2 in ${!VARIANTS[@]}; do
                variant2="${VARIANTS[$vidx2]}"
                ## reconstructed later
                # explanations_dir2="${EXPLANATIONS_DIRS[$vidx2]}"

                pre_args=()
                post_args=()
                for eidx in ${!lEXPERIMENT_NAMES[@]}; do
                    exp=${lEXPERIMENT_NAMES[$eidx]}
                    arg="${variant2}/${exp}"
                    if (( $eidx < $exp_idx )); then
                        pre_args+=("$arg")
                    elif (( $eidx == $exp_idx )); then
                        mid_arg="$arg"
                    else
                        post_args+=("$arg")
                    fi
                done
                all_args=("${pre_args[@]}" "$mid_arg" "${post_args[@]}")
                filter_c_pre_args=()
                filter_c_post_args=()
                [[ -n $FILTER ]] && {
                    ## Keep those that are filtered out for caching
                    for fidx in ${FILTER_IDXS_C[@]}; do
                        if (( $fidx < $exp_idx )); then
                            filter_c_pre_args+=("${pre_args[$fidx]}")
                        elif (( $fidx > $exp_idx )); then
                            filter_c_post_args+=("${all_args[$fidx]}")
                        fi
                    done
                }

                if (( $vidx < $vidx2 )); then
                    ARGS+=("${all_args[@]}")
                elif (( $vidx == $vidx2 )); then
                    ARGS+=("${filter_c_pre_args[@]}")
                    ARGS+=("${post_args[@]}")
                else
                    ARGS+=("${filter_c_pre_args[@]}")
                    ARGS+=("${filter_c_post_args[@]}")
                fi
            done
            ;;
        *)
            ARGS=(dummy)
            ;;
        esac

        any=0
        for arg in "${ARGS[@]}"; do
            phi_files=("$phi_file")
            unset experiment2

            case $ACTION in
            compare-subset)
                experiment2_full="$arg"
                experiment2="${experiment2_full#*/}"
                variant2="${experiment2_full%/*}"
                explanations_dir2="${EXPLANATIONS_DIR_BASE}/${variant2}"

                set_phi_filename "$explanations_dir2" $experiment2 phi_file2
                phi_files+=("$phi_file2")
                ;;
            esac

            [[ -n $FILTER && ! $experiment =~ $FILTER ]] && {
                get_cache_line cache_line "$experiment_full" "$experiment2_full" && printf "%s\n" "$cache_line" >>"$FILTERED_OUTPUT_CACHE_FILE"
                continue
            }

            case $ACTION in
            compare-subset)
                [[ -n $filter2 && ! $experiment2 =~ $filter2 ]] && {
                    get_cache_line cache_line "$experiment_full" "$experiment2_full" 1 && printf "%s\n" "$cache_line" >>"$FILTERED_OUTPUT_CACHE_FILE"
                    continue
                }
                ;;
            esac

            any=1

            get_cache_line cache_line "$experiment_full" "$experiment2_full" 1 && {
                printf "%s\n" "$cache_line"
                continue
            }

            args=()

            case $ACTION in
            check*)
                printf "Analyzing %s ... " "$phi_file"
                ;;
            count-fixed)
                printf "%${EXPERIMENT_MAX_WIDTH}s" "$experiment_full"
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
    ##+ add indentation ^^^

    done

    case $ACTION in
    *)
        close_cache
        ;;
    esac
done

exit 0
