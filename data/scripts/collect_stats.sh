#!/bin/bash

SCRIPTS_DIR=$(dirname "$0")

STATS_SCRIPT="$SCRIPTS_DIR/stats.awk"

source "$SCRIPTS_DIR/lib/experiments"

function usage {
    printf "USAGE: %s <explanations_dir> <experiments_spec> [[+]consecutive] [[+]reverse] [<max_samples>] [<filter_regex>] [<OPTIONS>]\n" "$0"
    printf "OPTIONS:\n"
    printf "\t--exclude-column <name>\t\tExclude given column\n"
    printf "\t--average [<regex>]\t\tAverage columns for all rows [matching the regex] (can be repeated)\n"

    [[ -n $1 ]] && exit $1
}

[[ -z $1 ]] && {
    printf "Provide stats data directory.\n" >&2
    usage 1 >&2
}

STATS_DIR="$1"
shift
[[ -d $STATS_DIR && -r $STATS_DIR ]] || {
    printf "'%s' is not a readable directory.\n" "$STATS_DIR" >&2
    usage 1 >&2
}

read_experiments_spec "$1" || usage $? >&2
shift

maybe_read_consecutive "$1" && shift
maybe_read_reverse "$1" && shift
maybe_read_max_samples "$1" && shift

[[ -n $1 && ! $1 =~ ^-- ]] && {
    FILTER="$1"
    shift
}

AVERAGE_FILTERS=()

declare -A {SUM_STR,COUNT}_perc_features
declare -A {SUM_STR,COUNT}_perc_fixed_features
declare -A {SUM_STR,COUNT}_perc_dimension
declare -A {SUM_STR,COUNT}_nterms
declare -A {SUM_STR,COUNT}_nchecks
declare -A {SUM_STR,COUNT}_avg_time_s

FORMAT_perc_features='%.1f%%'
FORMAT_perc_fixed_features='%.1f%%'
FORMAT_perc_dimension='%.1f%%'
FORMAT_nterms='%.1f'
FORMAT_nchecks='%.1f'
FORMAT_avg_time_s='%.2f'

declare -A EXCLUDE_COLUMNS

while [[ -n $1 ]]; do
    opt="$1"
    shift
    case "$opt" in
        --exclude-column)
            [[ -z $1 ]] && {
                printf "Option '%s': expected column name.\n" "$opt" >&2
                usage 2 >&2
            }
            EXCLUDE_COLUMNS["$1"]=1
            shift
            ;;

        --average)
            if [[ -n $1 ]]; then
                AVERAGE_FILTERS+=("$1")
            else
                AVERAGE_FILTERS+=('.*')
            fi
            shift
            ;;
    esac
done

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

EXPERIMENT_MAX_WIDTH=$(( 1 + $lMAX_EXPERIMENT_NAMES_LEN ))

FEATURES_CAPTION='%features'
FEATURES_MAX_WIDTH=${#FEATURES_CAPTION}

FIXED_CAPTION='%fixed'
FIXED_MAX_WIDTH=${#FIXED_CAPTION}

DIMENSION_CAPTION='%dimension'
DIMENSION_MAX_WIDTH=${#DIMENSION_CAPTION}

TERMS_CAPTION='#terms'
TERMS_MAX_WIDTH=${#TERMS_CAPTION}

CHECKS_CAPTION='#checks'
CHECKS_MAX_WIDTH=${#CHECKS_CAPTION}

function compute_term_size {
    local phi_file="$1"
    local n_lines=$2

    tr ' ' '\n' <"$phi_file" | grep -c '=' | { tr -d '\n'; printf "/$n_lines\n"; } | bc -l | xargs -i printf '%.1f' {}
}

PRINTED_HEADER=0
PRINTED_REVERSE=0
PRINTED_REGULAR=0

function print_header {
    local reverse=$1
    local dataset_size=$2
    local n_features=$3

    (( $PRINTED_HEADER )) || {
        printf 'Dataset size: %d\n' $dataset_size
        printf 'Number of features: %d\n' $n_features
        printf '\n'

        printf "%${EXPERIMENT_MAX_WIDTH}s" experiments
        [[ -z ${EXCLUDE_COLUMNS[$FEATURES_CAPTION]} ]] && printf " | %s" "$FEATURES_CAPTION"
        [[ -z ${EXCLUDE_COLUMNS[$FIXED_CAPTION]} ]] && printf " | %s" "$FIXED_CAPTION"
        [[ -z ${EXCLUDE_COLUMNS[$DIMENSION_CAPTION]} ]] && printf " | %s" "$DIMENSION_CAPTION"
        [[ -z ${EXCLUDE_COLUMNS[$TERMS_CAPTION]} ]] && printf " | %s" "$TERMS_CAPTION"
        [[ -z ${EXCLUDE_COLUMNS[$CHECKS_CAPTION]} ]] && printf " | %s" "$CHECKS_CAPTION"
        printf " | time [s]\n"

        PRINTED_HEADER=1
    }


    if (( $reverse )); then
        (( $PRINTED_REVERSE )) || {
            printf "REVERSE:\n"
            PRINTED_REVERSE=1
        }
    else
        (( $PRINTED_REGULAR )) || {
            printf "REGULAR:\n"
            PRINTED_REGULAR=1
        }
    fi
}

do_reverse_args=(0)
[[ -n $INCLUDE_REVERSE ]] && {
    (( $REVERSE_ONLY )) && do_reverse_args=()
    do_reverse_args+=(1)
}

ERR_FILE=$(mktemp)

function cleanup {
    rm $ERR_FILE

    exit $1
}

function store_var_into_str_array {
    local var_id=$1

    local -n lvar=$var_id

    local -n lformat=FORMAT_${var_id}
    local -n lstr_array=${var_id}_str_array

    local str
    if [[ $lvar == X ]]; then
        str="$lvar"
    else
        str=$(printf "${lformat}" "$lvar")
    fi

    lstr_array+=("$str")
}

function postprocess_str_var {
    local str_var_id=$1
    local experiment="$2"

    local -n lstr=$str_var_id
    [[ $lstr =~ ^@ ]] || return 0
    case $lstr in
        @SKIP)
            return 1
            ;;
        @AVG)
            local var_id=${str_var_id%_str}
            local -n lsum_str_array=SUM_STR_${var_id}
            local -n lcnt_array=COUNT_${var_id}

            local filter="$experiment"
            local sum_str=${lsum_str_array["$filter"]}
            local cnt=${lcnt_array["$filter"]}

            local -n lformat=FORMAT_${var_id}
            if [[ $sum_str =~ X ]]; then
                lstr=X
            else
                local avg=$(bc -l <<<"(${sum_str}0)/$cnt")
                lstr=$(printf "${lformat}" $avg)
            fi

            return 0
            ;;
    esac
}

for do_reverse in ${do_reverse_args[@]}; do
    experiment_array=()
    perc_features_str_array=()
    perc_fixed_features_str_array=()
    perc_dimension_str_array=()
    nterms_str_array=()
    nchecks_str_array=()
    avg_time_s_str_array=()

    for experiment in ${lEXPERIMENT_NAMES[@]}; do
        experiment_stem=$experiment
        [[ -n $FILTER && ! $experiment =~ $FILTER ]] && continue

        (( $do_reverse )) && experiment_stem=reverse/$experiment_stem
        [[ -n $MAX_SAMPLES ]] && experiment_stem=$MAX_SAMPLES_NAME/$experiment_stem

        stats_file="${STATS_DIR}/${experiment_stem}.stats.txt"
        [[ -r $stats_file ]] || {
            printf "File '%s' is not a readable.\n" "$stats_file" >&2
            cleanup 1
        }

        time_file="${STATS_DIR}/${experiment_stem}.time.txt"
        [[ -r $time_file ]] || {
            printf "File '%s' is not a readable.\n" "$time_file" >&2
            cleanup 1
        }

        phi_file="${STATS_DIR}/${experiment_stem}.phi.txt"
        [[ -r $phi_file ]] || {
            printf "File '%s' is not a readable.\n" "$phi_file" >&2
            cleanup 1
        }

        stats=$($STATS_SCRIPT "$stats_file" 2>$ERR_FILE)
        size=$(sed -n 's/^Total:[^0-9]*\([0-9]*\)$/\1/p' <<<"$stats")
        features=$(sed -n 's/^Features:[^0-9]*\([0-9]*\)$/\1/p' <<<"$stats")

        print_header $do_reverse $size $features

        time_str=$(sed -n 's/^user[^0-9]*\([0-9].*\)$/\1/p' <"$time_file")
        if [[ -z $time_str ]]; then
            perc_features=X
            perc_fixed_features=X
            perc_dimension=X
            nterms=X
            nchecks=X
            avg_time_s=X
        else
            [[ -s $ERR_FILE ]] && {
                cat $ERR_FILE >&2
                cleanup 3
            }

            perc_features=$(sed -n 's/^.*#any features: \([^%]*\)%.*$/\1/p' <<<"$stats")
            perc_fixed_features=$(sed -n 's/^.*#fixed features: \([^%]*\)%.*$/\1/p' <<<"$stats")
            nterms_stats=$(sed -n 's/^.*#terms: \(.*\)$/\1/p' <<<"$stats")
            nchecks=$(sed -n 's/^.*#checks: \(.*\)$/\1/p' <<<"$stats")

            perc_dimension=$(bc -l <<<"100 - $perc_fixed_features")

            nterms=$(compute_term_size "$phi_file" $size)
            [[ -z $nterms_stats ]] && {
                ##! fragile
                nterms_stats=$(bc -l <<<"($features * $perc_features)/100")
                nterms_stats=$(printf '%.1f' $nterms_stats)
            }
            [[ $nterms == $nterms_stats ]] || {
                printf "%s: encountered inconsistency: stats.termSize != termSize(phi): %s != %s\n" $experiment_stem "$nterms_stats" "$nterms" >&2
                cleanup 3
            }

            time_min=${time_str%%m*}
            time_s=${time_str##*m}
            time_s=${time_s%s}
            total_time_s=$(bc -l <<<"${time_min}*60 + ${time_s}")
            avg_time_s=$(bc -l <<<"${total_time_s}/${size}")
        fi

        for avg_filter in "${AVERAGE_FILTERS[@]}"; do
            [[ $experiment =~ $avg_filter ]] || continue

            SUM_STR_perc_features["$avg_filter"]+="$perc_features+"
            SUM_STR_perc_fixed_features["$avg_filter"]+="$perc_fixed_features+"
            SUM_STR_perc_dimension["$avg_filter"]+="$perc_dimension+"
            SUM_STR_nterms["$avg_filter"]+="$nterms+"
            SUM_STR_nchecks["$avg_filter"]+="$nchecks+"
            SUM_STR_avg_time_s["$avg_filter"]+="$avg_time_s+"
            (( COUNT_perc_features["$avg_filter"] ++ ))
            (( COUNT_perc_fixed_features["$avg_filter"] ++ ))
            (( COUNT_perc_dimension["$avg_filter"] ++ ))
            (( COUNT_nterms["$avg_filter"] ++ ))
            (( COUNT_nchecks["$avg_filter"] ++ ))
            (( COUNT_avg_time_s["$avg_filter"] ++ ))

            if (( ${COUNT_perc_features["$avg_filter"]} == 1 )); then
                perc_features_str_array+=(@AVG)
                perc_fixed_features_str_array+=(@AVG)
                perc_dimension_str_array+=(@AVG)
                nterms_str_array+=(@AVG)
                nchecks_str_array+=(@AVG)
                avg_time_s_str_array+=(@AVG)
            else
                perc_features_str_array+=(@SKIP)
                perc_fixed_features_str_array+=(@SKIP)
                perc_dimension_str_array+=(@SKIP)
                nterms_str_array+=(@SKIP)
                nchecks_str_array+=(@SKIP)
                avg_time_s_str_array+=(@SKIP)
            fi

            experiment_array+=("$avg_filter")

            continue 2
        done

        experiment_array+=($experiment)

        store_var_into_str_array perc_features
        store_var_into_str_array perc_fixed_features
        store_var_into_str_array perc_dimension
        store_var_into_str_array nterms
        store_var_into_str_array nchecks
        store_var_into_str_array avg_time_s
    done

    for idx in ${!experiment_array[@]}; do
        experiment="${experiment_array[$idx]}"

        perc_features_str="${perc_features_str_array[$idx]}"
        perc_fixed_features_str="${perc_fixed_features_str_array[$idx]}"
        perc_dimension_str="${perc_dimension_str_array[$idx]}"
        nterms_str="${nterms_str_array[$idx]}"
        nchecks_str="${nchecks_str_array[$idx]}"
        avg_time_s_str="${avg_time_s_str_array[$idx]}"

        postprocess_str_var perc_features_str "$experiment" || continue
        postprocess_str_var perc_fixed_features_str "$experiment" || continue
        postprocess_str_var perc_dimension_str "$experiment" || continue
        postprocess_str_var nterms_str "$experiment" || continue
        postprocess_str_var nchecks_str "$experiment" || continue
        postprocess_str_var avg_time_s_str "$experiment" || continue

        printf "%${EXPERIMENT_MAX_WIDTH}s" $experiment
        [[ -z ${EXCLUDE_COLUMNS[$FEATURES_CAPTION]} ]] && printf " | %${FEATURES_MAX_WIDTH}s" $perc_features_str
        [[ -z ${EXCLUDE_COLUMNS[$FIXED_CAPTION]} ]] && printf " | %${FIXED_MAX_WIDTH}s" $perc_fixed_features_str
        [[ -z ${EXCLUDE_COLUMNS[$DIMENSION_CAPTION]} ]] && printf " | %${DIMENSION_MAX_WIDTH}s" $perc_dimension_str
        [[ -z ${EXCLUDE_COLUMNS[$TERMS_CAPTION]} ]] && printf " | %${TERMS_MAX_WIDTH}s" $nterms_str
        [[ -z ${EXCLUDE_COLUMNS[$CHECKS_CAPTION]} ]] && printf " | %${CHECKS_MAX_WIDTH}s" $nchecks_str
        printf " | %s" $avg_time_s_str
        printf "\n"
    done
done

cleanup 0
