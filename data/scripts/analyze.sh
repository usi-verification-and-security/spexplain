#!/bin/bash

ACTION_REGEX='check|check-sat|count-fixed|compare-subset'

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

CHECK_SAT_ONLY=0
case $ACTION in
check-sat)
    ACTION=check
    CHECK_SAT_ONLY=1
    ;;
esac

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

WAIT_SUPPORTS_P=$(bc -l <<<"${BASH_VERSION%.*} >= 5.2")

IFILES=()
OFILES=()

IFILES_SAT=()
OFILES_SAT=()

IFILES_SUBSET=()
IFILES_SUPSET=()

OFILES_SUBSET=()
OFILES_SUPSET=()

# set -e

function cleanup {
    local code=$1
    local pid=$2

    [[ -n $code && $code != 0 ]] && {
        pkill -P $$
        wait
    }

    [[ -n $pid ]] && {
        for pidx in ${!PIDS[@]}; do
            local p=${PIDS[$pidx]}
            local ofile
            if [[ $p == $pid ]]; then
                case $ACTION in
                compare-subset)
                    ofile=${OFILES_SUBSET[$pidx]}
                    ;;
                *)
                    ofile=${OFILES[$pidx]}
                    ;;
                esac
            else
                p=${PIDS2[$pidx]}
                [[ $p == $pid ]] || continue
                case $ACTION in
                check)
                    ofile=${OFILES_SAT[$pidx]}
                    ;;
                compare-subset)
                    ofile=${OFILES_SUPSET[$pidx]}
                    ;;
                esac
            fi

            cat $ofile >&2
            break
        done
    }

    for idx in ${!IFILES[@]}; do
        local ifile=${IFILES[$idx]}
        local ofile=${OFILES[$idx]}
        local ifile_sat=${IFILES_SAT[$idx]}
        local ofile_sat=${OFILES_SAT[$idx]}
        rm -f $ifile
        rm -f $ofile
        rm -f $ifile_sat
        rm -f $ofile_sat
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
PIDS2=()

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

function set_input_output_file {
    local ifile_var=$1
    local ofile_var=$2
    local ifiles_var=$3
    local ofiles_var=$4
    local psi_file="$5"

    local -n lifile=$ifile_var
    local -n lofile=$ofile_var

    lifile=$(mktemp --suffix=.smt2)
    lofile=${lifile}.out
    eval $ifiles_var'+=($lifile)'
    eval $ofiles_var'+=($lofile)'

    cp "$psi_file" $lifile
}

function exec_solver {
    local solver=$1
    local ifile=$2
    local ofile=$3
    local pids_var=$4

    $solver $ifile &>$ofile &
    eval $pids_var'+=($!)'
    (( ++N_PROC ))
}

i=-1
cnt=0
any_timeout_cnt=0
while true; do
    (( ++i ))
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
        psi_sat_file="${PSI_FILE/_c[0-9]/_d}"
        ;;
    count-fixed)
        psi_file="$PSI_FILE"
        ;;
    compare-subset)
        while read line2 <&3 && [[ -z ${line2// } ]]; do :; done
        [[ -z $line2 ]] && {
            printf "Unexpected missing data from the second file at i=%d\n" $i >&2
            cleanup 9
        }
        psi_file="$PSI_FILE"
        ;;
    esac

    [[ $line == '<null>' ]] && {
        (( ++any_timeout_cnt ))
        continue
    }
    case $ACTION in
    compare-subset)
        [[ $line2 == '<null>' ]] && {
            (( ++any_timeout_cnt ))
            continue
        }
        ;;
    esac

    for arg in ${ARGS[@]}; do
        case $ACTION in
        check)
            set_input_output_file ifile ofile IFILES OFILES "$psi_file"
            set_input_output_file ifile_sat ofile_sat IFILES_SAT OFILES_SAT "$psi_sat_file"
            ;;
        count-fixed)
            set_input_output_file ifile ofile IFILES OFILES "$psi_file"
            ;;
        compare-subset)
            set_input_output_file ifile_subset ofile_subset IFILES_SUBSET OFILES_SUBSET "$psi_file"
            set_input_output_file ifile_supset ofile_supset IFILES_SUPSET OFILES_SUPSET "$psi_file"
            ;;
        esac


        case $ACTION in
        check)
            ## + we do not check that psi alone is SAT
            (( $CHECK_SAT_ONLY )) || printf "(assert %s)\n(check-sat)\n" "$line" >>$ifile
            printf "(assert %s)\n(check-sat)\n" "$line" >>$ifile_sat
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
            printf "(check-sat)\n" >>$ifile
            ;;
        compare-subset)
            printf "(assert (and %s (not %s)))\n(check-sat)\n" "$line" "$line2" >>$ifile_subset
            printf "(assert (and (not %s) %s))\n(check-sat)\n" "$line" "$line2" >>$ifile_supset
            ;;
        esac

        while (( $N_PROC >= $MAX_PROC )); do
            if (( $WAIT_SUPPORTS_P )); then
                wait -n -p pid || {
                    printf "Process %d exited with error.\n" $pid >&2
                    cleanup 4 $pid
                }
            else
                wait -n || {
                    printf "Process exited with error.\n" >&2
                    cleanup 4
                }
            fi
            (( --N_PROC ))
        done

        case $ACTION in
        check)
            (( $CHECK_SAT_ONLY )) || exec_solver $SOLVER $ifile $ofile PIDS
            exec_solver $SOLVER $ifile_sat $ofile_sat PIDS2
            ;;
        count-fixed)
            exec_solver $SOLVER $ifile $ofile PIDS
            ;;
        compare-subset)
            exec_solver $SOLVER $ifile_subset $ofile_subset PIDS
            exec_solver $SOLVER $ifile_supset $ofile_supset PIDS2
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

    [[ -z $MAX_LINES ]] && (( $cnt + $any_timeout_cnt != $N_LINES )) && {
        printf "Unexpected mismatch of the # processed and timeouted lines: %d + %d = %d != %d\n" $cnt $any_timeout_cnt $(($cnt+$any_timeout_cnt)) $N_LINES >&2
        cleanup 9
    }
    [[ -n $MAX_LINES ]] && (( $cnt + $any_timeout_cnt != $MAX_LINES )) && {
        printf "Unexpected mismatch of the bounded # processed and timeouted lines: %d + %d = %d != %d\n" $cnt $any_timeout_cnt $(($cnt+$any_timeout_cnt)) $MAX_LINES >&2
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
    (( $CHECK_SAT_ONLY )) || for idx in ${!IFILES[@]}; do
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
                printf "NOT space explanation [%d]: classification change not unsat:\n" $idx >&2
                tail -n 2 $ifile | head -n 1 >&2
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
    for idx in ${!IFILES_SAT[@]}; do
        ifile=${IFILES_SAT[$idx]}
        ofile=${OFILES_SAT[$idx]}
        result=$(cat $ofile)
        if [[ $result == unsat ]]; then
            printf "NOT space explanation [%d]: explanation not sat:\n" $idx >&2
            tail -n 2 $ifile | head -n 1 >&2
            cp -v $ifile $(basename $ifile) >&2
            cleanup 3
        elif [[ $result != sat ]]; then
            printf "Unexpected output of sat query:\n%s\n" "$result" >&2
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
