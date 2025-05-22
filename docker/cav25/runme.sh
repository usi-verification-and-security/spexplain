#!/bin/bash

ACTIONS=(run get-tables compute-set-tables verify)
MODES=(quick full)

function usage {
    printf "USAGE: %s <action> <mode>\n" $(basename "$0")
    printf "ACTIONS: %s\n" "${ACTIONS[*]}"
    printf "MODES: %s\n" "${MODES[*]}"

    [[ -n $1 ]] && exit $1
}

CMDLINE_ARGS="$*"

cd $(dirname "$0")/data

ROOT_DIR=../
SCRIPTS_DIR=./scripts
EXPLANATIONS_DIR=./explanations

set -e

[[ -z $1 ]] && usage 0
ACTION="$1"
shift

[[ -z $1 ]] && usage 0
MODE="$1"
shift

found=0
for a in ${ACTIONS[@]}; do
    [[ $a == $ACTION ]] && found=1
done
(( $found )) || {
    printf "Unrecognized action: %s\n" "$ACTION" >&2
    usage 1 >&2
}

found=0
for a in ${MODES[@]}; do
    [[ $a == $MODE ]] && found=1
done
(( $found )) || {
    printf "Unrecognized mode: %s\n" "$MODE" >&2
    usage 1 >&2
}

MODELS=(heart_attack obesity mnist)
SPECS=(base itp itp)

SUITES_QUICK=(quick quick quick)
SUITES_FULL=(full short short)

case $MODE in
    quick)
        declare -n SUITES=SUITES_QUICK
        TIMEOUT=8m
        ;;
    full)
        declare -n SUITES=SUITES_FULL
        TIMEOUT=2h
        ;;
esac

case $ACTION in
    run)
        for idx in ${!MODELS[@]}; do
            model=${MODELS[$idx]}
            spec=${SPECS[$idx]}
            suite=${SUITES[$idx]}

            for mode in '' consecutive; do
                CMD="$ROOT_DIR/build-marabou/xspace" TIMEOUT=$TIMEOUT "$SCRIPTS_DIR/run-xspace-all.sh" "$EXPLANATIONS_DIR/$model/$suite" $spec $mode
            done
        done
        ;;

    get-tables)
        for table in table-{1,2,3,5}; do
            for idx in ${!MODELS[@]}; do
                model=${MODELS[$idx]}
                spec=${SPECS[$idx]}
                suite=${SUITES[$idx]}

                ofile="$table-$model-$MODE.txt"
                ofile_report=$(realpath --relative-base="$ROOT_DIR" $ofile)

                case $table in
                    table-1)
                        "$SCRIPTS_DIR/collect_stats.sh" "$EXPLANATIONS_DIR/$model/$suite" $spec '^itp_a' >"$ofile"
                        ;;
                    table-2)
                        "$SCRIPTS_DIR/collect_stats.sh" "$EXPLANATIONS_DIR/$model/$suite" $spec '^[^s].*tp_a' >"$ofile"
                        ;;
                    table-3)
                        [[ $model != heart_attack ]] && continue

                        "$SCRIPTS_DIR/collect_stats.sh" "$EXPLANATIONS_DIR/$model/$suite" $spec +consecutive '(abductive|^itp_aweak_bstrong)' >"$ofile"
                        ;;
                    table-5)
                        "$SCRIPTS_DIR/collect_stats.sh" "$EXPLANATIONS_DIR/$model/$suite" $spec '(slice_|itp_vars_)' --average '^itp_vars' --average '^ucore_itp_vars' --average 'ucore_min_itp_vars' --average 'slice_.*[0-9]_itp_aweak_bstrong' --average 'slice_.*_ucore_itp_aweak_bstrong' --average 'slice_.*_ucore_min_itp_aweak_bstrong' >"$ofile"
                        ;;
                esac

                printf "Table stored in file %s\n" "$ofile_report"
            done
        done
        ;;

    compute-set-tables)
        for table in table-4; do
            for idx in ${!MODELS[@]}; do
                model=${MODELS[$idx]}
                spec=${SPECS[$idx]}
                suite=${SUITES[$idx]}

                ofile="$table-$model-$MODE.txt"
                ofile_report=$(realpath --relative-base="$ROOT_DIR" $ofile)

                case $table in
                    table-4)
                        [[ $model != heart_attack ]] && continue
                        ;;
                esac

                printf "Computing table %s ...\n" "$ofile_report"

                case $table in
                    table-4)
                        "$SCRIPTS_DIR/analyze-all.sh" compare-subset "$EXPLANATIONS_DIR/$model/$suite" $spec '^(|ucore(|_min)_)itp_vars_(x[0-9]+_x[0-9]+)' '^slice_\3_\1itp_aweak_bstrong' >"$ofile"
                        ;;
                esac

                printf "Table stored in file %s\n" "$ofile_report"
            done
        done
        ;;

    verify)
        for idx in ${!MODELS[@]}; do
            model=${MODELS[$idx]}
            spec=${SPECS[$idx]}
            suite=${SUITES[$idx]}

            [[ $model == mnist ]] && continue

            "$SCRIPTS_DIR/analyze-all.sh" check "$EXPLANATIONS_DIR/$model/$suite" $spec
        done
        ;;
esac

printf "\nArtifact script '%s' completed successfully.\n" "$0 $CMDLINE_ARGS"
