#!/bin/bash

ACTION_REGEX='check|count-fixed|compare-subset'

function usage {
    printf "USAGE: %s <action> <psi> <f> [<f2>] [<max_rows>]\n" "$0"
    printf "ACTIONS: %s\n" "$ACTION_REGEX"

    [[ -n $1 ]] && exit $1
}

[[ -z $1 ]] && usage 0

ACTION=$1
shift

[[ $ACTION =~ ^($ACTION_REGEX)$ ]] || {
    printf "Expected an action, got: %s\n" "$ACTION" >&2
    usage 1
}

PSI_FILE="$1"
shift

[[ $PSI_FILE =~ \.smt2$ ]] || {
    printf "Expected an SMT-LIB psi file, got: %s\n" "$PSI_FILE" >&2
    usage 1
}

case $ACTION in
check)
    [[ $PSI_FILE =~ _c[0-9][^0-9] ]] || {
        printf "Expected class-related psi file, got: %s\n" "$PSI_FILE" >&2
        usage 1
    }
    ;;
count-fixed|compare-subset)
    [[ $PSI_FILE =~ _d ]] || {
        printf "Expected domain psi file, got: %s\n" "$PSI_FILE" >&2
        usage 1
    }
    ;;
esac

[[ -r $PSI_FILE ]] || {
    printf "\nNot readable psi file: %s\n" "$PSI_FILE" >&2
    usage 1
}

FILE="$1"
shift

unset STATS_FILE
if [[ $FILE =~ \.phi\.txt$ ]]; then
    PHI_FILE="$FILE"
elif [[ $FILE =~ \.stats\.txt$ ]]; then
    STATS_FILE="$FILE"
    PHI_FILE="${FILE/.stats./.phi.}"
elif [[ -r $FILE ]]; then
    PHI_FILE="${FILE}"
else
    PHI_FILE="${FILE%.*}.phi.txt"
fi

[[ -r $PHI_FILE ]] || {
    printf "\nNot readable formula data file: %s\n" "$PHI_FILE" >&2
    usage 1
}

case $ACTION in
check)
    [[ -z $STATS_FILE ]] && STATS_FILE="${PHI_FILE%.*}.stats.txt"
    [[ -r $STATS_FILE ]] || STATS_FILE="${PHI_FILE/phi/stats}"

    [[ -r $STATS_FILE ]] || {
        printf "\nNot readable stats data file: %s\n" "$STATS_FILE" >&2
        usage 1
    }
    ;;
esac

case $ACTION in
compare-subset)
    PHI_FILE2="$1"
    shift

    [[ -r $PHI_FILE2 ]] || {
        printf "\nNot readable second formula data file: %s\n" "$PHI_FILE2" >&2
        usage 1
    }
    ;;
esac

MAX_LINES=$1
shift

N_LINES=$(wc -l <"$PHI_FILE")

case $ACTION in
compare-subset)
    N_LINES1=$N_LINES
    N_LINES2=$(sed '/^ *$/d' <"$PHI_FILE2" | wc -l)
    unset N_LINES

    if [[ -z $MAX_LINES ]]; then
        (( $N_LINES1 != $N_LINES2 )) && {
            printf "The number of lines of the data files do not match: %d != %d\n" $N_LINES1 $N_LINES2 >&2
            exit 2
        }
        N_LINES=$N_LINES1
    else
        if (( $N_LINES1 <= $N_LINES2 )); then
            N_LINES=$N_LINES1
        else
            N_LINES=$N_LINES2
        fi

        if [[ $MAX_LINES == max ]]; then
            MAX_LINES=$N_LINES
        else
            [[ $MAX_LINES =~ ^[1-9][0-9]*$ ]] || {
                printf "Expected max. no. lines, got: %s\n" "$MAX_LINES" >&2
                usage 2
            }

            (( $MAX_LINES > $N_LINES )) && {
                printf "The entered max. no. lines is greater than the min. no. lines of the files: %d > %d\n" $MAX_LINES $N_LINES >&2
                exit 2
            }
        fi
    fi
    ;;
*)
    [[ -n $MAX_LINES ]] && {
        (( $MAX_LINES > $N_LINES )) && {
            printf "The entered max. no. lines is greater than the actual no. lines: %d > %d\n" $MAX_LINES $N_LINES >&2
            exit 2
        }
    }
    ;;
esac

FAST_SOLVER=opensmt

OTHER_SOLVERS=(
    cvc5
    z3
    mathsat
)

function is_executable {
    local cmd="$1"

    command -v "$cmd" &>/dev/null
}

unset OTHER_SOLVER
for s in "${OTHER_SOLVERS[@]}"; do
    is_executable "$s" || continue
    OTHER_SOLVER="$s"
    break
done

if [[ -z $SOLVER ]]; then
    is_executable "$OTHER_SOLVER" || {
        printf "None of the solvers is executable: %s\n" "${OTHER_SOLVERS[*]}" >&2
        exit 2
    }

    is_executable "$FAST_SOLVER" || FAST_SOLVER="$OTHER_SOLVER"

    case $ACTION in
    check)
        ## Prefer a trusted third-party solver for verification of the results
        SOLVER="$OTHER_SOLVER"
        ;;
    *)
        ## Prefer faster solver in the rest cases
        SOLVER="$FAST_SOLVER"
        ;;
    esac
else
    is_executable "$SOLVER" || {
        printf "The provided solver is not executable: %s\n" "$SOLVER" >&2
        exit 2
    }
fi

IFILES=()
OFILES=()

IFILES_SUBSET=()
IFILES_SUPSET=()

OFILES_SUBSET=()
OFILES_SUPSET=()

# set -e

function cleanup {
    local code=$1

    [[ -n $code && $code != 0 ]] && {
        kill 0
        wait
    }

    for idx in ${!IFILES[@]}; do
        local ifile=${IFILES[$idx]}
        local ofile=${OFILES[$idx]}
        rm -f $ifile
        rm -f $ofile
    done

    for idx in ${!IFILES_SUBSET[@]}; do
        local ifile_subset=${IFILES_SUBSET[$idx]}
        local ofile_subset=${OFILES_SUBSET[$idx]}
        local ifile_supset=${IFILES_SUPSET[$idx]}
        local ofile_supset=${OFILES_SUPSET[$idx]}
        rm -f $ifile_subset
        rm -f $ofile_subset
        rm -f $ifile_supset
        rm -f $ofile_supset
    done

    [[ -n $code ]] && exit $code
}

N_CPU=$(nproc --all)
MAX_PROC=$(( 1 + $N_CPU/2 ))
N_PROC=0

PIDS=()

[[ $ACTION == count-fixed ]] && {
    VARIABLES=($(sed -rn '/declare-fun/s/.*declare-fun ([^ ]*) .*/\1/p' <$PSI_FILE))
}

case $ACTION in
count-fixed)
    ARGS=(${VARIABLES[@]})
    ;;
*)
    ARGS=(dummy)
    ;;
esac

case $ACTION in
check)
    FILE2="$STATS_FILE"
    ;;
compare-subset)
    FILE2="$PHI_FILE2"
    ;;
*)
    FILE2=/dev/null
    ;;
esac

cnt=0
while true; do
    while read line && [[ -z ${line// } ]]; do :; done
    [[ -z $line ]] && break

    case $ACTION in
    check)
        while read oline <&3; do
            [[ $oline =~ "computed output:" ]] && break
        done
        out_class=${oline##*: }
        [[ $out_class =~ ^[0-9]+$ ]] || {
            printf "Unexpected output class from row '%s' in %s: %s\n" "$oline" "$STATS_FILE" "$out_class" >&2
            cleanup 3
        }
        psi_file="${PSI_FILE/_c[0-9]/_c$out_class}"
        ;;
    count-fixed)
        psi_file="$PSI_FILE"
        ;;
    compare-subset)
        while read line2 <&3 && [[ -z ${line2// } ]]; do :; done
        [[ -z $line2 ]] && {
            printf "Unexpected missing data from the second file at cnt=%d\n" $cnt >&2
            cleanup 9
        }
        ;;
    esac

    for arg in ${ARGS[@]}; do
        case $ACTION in
        compare-subset)
            ifile_subset=$(mktemp --suffix=.smt2)
            ofile_subset=${ifile_subset}.out
            IFILES_SUBSET+=($ifile_subset)
            OFILES_SUBSET+=($ofile_subset)
            ifile_supset=$(mktemp --suffix=.smt2)
            ofile_supset=${ifile_supset}.out
            IFILES_SUPSET+=($ifile_supset)
            OFILES_SUPSET+=($ofile_supset)

            cp "$PSI_FILE" $ifile_subset
            cp "$PSI_FILE" $ifile_supset
            ;;
        *)
            ifile=$(mktemp --suffix=.smt2)
            ofile=${ifile}.out
            IFILES+=($ifile)
            OFILES+=($ofile)

            cp "$psi_file" $ifile
            ;;
        esac


        case $ACTION in
        check)
            ## + we do not check that psi alone and also phi alone are both SAT
            printf "(assert %s)" "$line" >>$ifile
            ;;
        count-fixed)
            var=$arg
            var2=${var}-2
            sed_str="s/${var}([^a-zA-Z0-9\-_])/${var2}\1/g"
            line2=$(sed -r "$sed_str" <<<"$line")

            printf "(declare-fun %s () Real)\n" $var2 >>$ifile
            printf "(declare-fun C1 () Real)\n" >>$ifile
            printf "(declare-fun C2 () Real)\n" >>$ifile

            sed -n '/assert/,$p' <"$psi_file" | sed -r "${sed_str}" >>$ifile

            printf "(assert %s)\n(assert %s)\n" "$line" "$line2" >>$ifile
            printf "(assert (and (= %s C1) (= %s C2) (not (= C1 C2))))\n" $var $var2 >>$ifile
            ;;
        esac

        case $ACTION in
        compare-subset)
            printf "(assert (and %s (not %s)))\n(check-sat)\n" "$line" "$line2" >>$ifile_subset
            printf "(assert (and (not %s) %s))\n(check-sat)\n" "$line" "$line2" >>$ifile_supset
            ;;
        *)
            printf "(check-sat)\n" >>$ifile
            ;;
        esac

        while (( $N_PROC >= $MAX_PROC )); do
            # wait -n -p pid || {
            #     printf "Process %d exited with error.\n" $pid >&2
            wait -n || {
                printf "Process exited with error.\n" >&2
                cleanup 4
            }
            (( --N_PROC ))
        done

        case $ACTION in
        compare-subset)
            $SOLVER $ifile_subset &>$ofile_subset &
            PIDS+=($!)
            (( ++N_PROC ))
            $SOLVER $ifile_supset &>$ofile_supset &
            PIDS+=($!)
            (( ++N_PROC ))
            ;;
        *)
            $SOLVER $ifile &>$ofile &
            PIDS+=($!)
            (( ++N_PROC ))
            ;;
        esac
    done

    [[ -z $MAX_LINES ]] && continue
    (( ++cnt ))
    (( $cnt >= $MAX_LINES )) && break
done <"$PHI_FILE" 3<"$FILE2"

case $ACTION in
compare-subset)
    [[ -z $MAX_LINES ]] && cnt=${#IFILES_SUBSET[@]}

    [[ -z $MAX_LINES && $cnt != $N_LINES ]] && {
        printf "Unexpected mismatch of the # processed lines: %d != %d\n" $cnt $N_LINES >&2
        cleanup 9
    }
    [[ -n $MAX_LINES && $cnt != $MAX_LINES ]] && {
        printf "Unexpected mismatch of the bounded # processed lines: %d != %d\n" $cnt $MAX_LINES >&2
        cleanup 9
    }
    ;;
esac

wait || {
    printf "Error when waiting on the rest of processes.\n" >&2
    cleanup 4
}

case $ACTION in
compare-subset)
    SUBSET_CNT=0
    SUPSET_CNT=0
    EQUAL_CNT=0
    UNCOMPARABLE_CNT=0
    for idx in ${!IFILES_SUBSET[@]}; do
        ofile_subset=${OFILES_SUBSET[$idx]}
        ofile_supset=${OFILES_SUPSET[$idx]}

        subset_result=$(cat $ofile_subset)
        supset_result=$(cat $ofile_supset)

        if [[ $subset_result == unsat ]]; then
            if [[ $supset_result == unsat ]]; then
                (( ++EQUAL_CNT ))
            elif [[ $supset_result == sat ]]; then
                (( ++SUBSET_CNT ))
            else
                printf "Unexpected output of supset query: %s\n" $supset_result >&2
                less $ofile_supset
                cleanup 3
            fi
        elif [[ $subset_result == sat ]]; then
            if [[ $supset_result == unsat ]]; then
                (( ++SUPSET_CNT ))
            elif [[ $supset_result == sat ]]; then
                (( ++UNCOMPARABLE_CNT ))
            else
                printf "Unexpected output of supset query: %s\n" $supset_result >&2
                less $ofile_supset
                cleanup 3
            fi
        else
            printf "Unexpected output of subset query: %s\n" $subset_result >&2
            less $ofile_subset
            cleanup 3
        fi
    done

    sum=$(( $SUBSET_CNT + $SUPSET_CNT + $EQUAL_CNT + $UNCOMPARABLE_CNT ))
    (( $cnt == $sum )) || {
        printf "Unexpected mismatch of particular counts with the total count: %d != %d\n" $cnt $sum >&2
        cleanup 3
    }
    ;;
*)
    cnt=0
    for idx in ${!IFILES[@]}; do
        ifile=${IFILES[$idx]}
        ofile=${OFILES[$idx]}
        result=$(cat $ofile)
        if [[ $result == unsat ]]; then
            case $ACTION in
            check)
                ;;
            count-fixed)
                (( ++cnt ))
                ;;
            esac
        elif [[ $result == sat ]]; then
            case $ACTION in
            check)
                printf "NOT space explanation [%d]:\n" $idx >&2
                sed -n '$p' $ifile >&2
                cp -v $ifile $(basename $ifile) >&2
                cleanup 3
                ;;
            count-fixed)
                ;;
            esac
        else
            printf "Unexpected output of query:\n%s\n" "$result" >&2
            less $ofile
            cleanup 3
        fi
    done
    ;;
esac

case $ACTION in
check)
    printf "OK!\n"
    ;;
count-fixed)
    printf "avg #fixed features: %.1f%%\n" $(bc -l <<<"($cnt / ${#IFILES[@]})*100")
    ;;
compare-subset)
    printf "Total: %d\n" $cnt
    printf "<: %d =: %d >: %d | ?: %d\n" $SUBSET_CNT $EQUAL_CNT $SUPSET_CNT $UNCOMPARABLE_CNT
    ;;
esac

cleanup 0
