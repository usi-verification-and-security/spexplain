#!/bin/bash

export DIRNAME=$(dirname $(realpath "$0"))
export SCRIPTS_DIR=$(dirname "$DIRNAME")
export DATA_DIR=$(realpath "$SCRIPTS_DIR/..")
export ROOT_DIR=$(realpath "$DATA_DIR/..")

source "$SCRIPTS_DIR/lib/common"

shopt -qs extglob

ACTIONS=(gen run)

function usage {
    printf 'USAGE: %s <action> [<solver>] <model_fn> <dataset_fn> <bench_dir> [<results_file>]\n' "$0"
    printf 'ACTIONS: %s\n' "${ACTIONS[*]}"

    [[ -n $1 ]] && exit $1
}

function set_filename {
    local -n var=$1
    local sample_id=$2

    var="$BENCH_DIR/$MODEL_BASE/${MODEL_NAME}-${DATASET_NAME}-sample_${sample_id}"
    [[ -n $EPSILON ]] && var+=-eps$EPSILON
    [[ -n $PROOF ]] && var+=-proof
    [[ -n $PREFER ]] && var+=-prefer

    var+=.smt2
}
export -f set_filename

[[ -z $INTERACTIVE ]] && {
    [[ -t 1 ]] && INTERACTIVE=1
}
export INTERACTIVE

[[ -z $1 ]] && {
    printf 'Expected an action\n' >&2
    usage 1 >&2
}

__contains ACTIONS "$1" || {
    printf 'Unknown action: %s\n' "$1" >&2
    usage 1 >&2
}

ACTION=$1
shift

[[ $EPSILON == 0 ]] && unset EPSILON
[[ $PROOF == 0 ]] && unset PROOF
[[ $PREFER == 0 ]] && unset PREFER
export EPSILON
export PROOF
export PREFER

[[ -n $EPSILON ]] && ( [[ ! $EPSILON =~ ^[0-9.]+$ || $(bc -l <<<"$EPSILON < 1") != 1 ]] ) && {
    printf 'Expected float epsilon<1, got: %s\n' "$EPSILON" >&2
    usage 2 >&2
}

case $ACTION in
run)
    source "$SCRIPTS_DIR/lib/run"

    command -v "$1" >/dev/null || {
        printf "Expected an SMT solver executable, got: %s\n" "$1" >&2
        usage 1 >&2
    }

    export SOLVER="$1"
    shift

    [[ $SOLVER =~ ^[./] ]] && SOLVER=$(realpath "$SOLVER")

    SOLVER_BASE=$(basename "$SOLVER")

    [[ $SOLVER_BASE =~ ^(z3|cvc[45]|mathsat|opensmt)$ ]] || {
        printf "Expected an actual SMT solver, got: %s\n" "$1" >&2
        usage 1 >&2
    }

    export SOLVER_PROCESS_NAME=xai_smt_solver-${SOLVER_BASE}

    [[ -z $TIMEOUT ]] && TIMEOUT=2m
    export TIMEOUT

    time_to "$TIMEOUT" TIMEOUT_S || exit 1
;;
esac

[[ -z $1 ]] && usage 1 >&2
MODEL=$(realpath "$1")
shift

[[ -z $1 ]] && usage 1 >&2
DATASET=$(realpath "$1")
shift

[[ -z $1 ]] && usage 1 >&2

case $ACTION in
run)
    [[ -d $1 ]] || {
        printf "Expected a directory, got: %s\n" "$1" >&2
        usage 1 >&2
    }
;;
esac

BENCH_DIR=$(realpath "$1")
shift

if [[ -n $1 ]]; then
    RESULTS_FILE=$(realpath "$1")
    shift

    >"$RESULTS_FILE" || exit $?
else
    RESULTS_FILE=/dev/null
fi

case $ACTION in
gen)
    cd "$ROOT_DIR"

    mkdir -p local/$$ >/dev/null || exit $?
    cd local/$$
;;
esac

export SCRIPTS_DIR=$(realpath "$SCRIPTS_DIR" --relative-to "$PWD")
export DATA_DIR=$(realpath "$DATA_DIR" --relative-to "$PWD")

RESULTS_FILE=$(realpath "$RESULTS_FILE" --relative-to "$PWD")

export MODEL=$(realpath "$MODEL" --relative-to "$PWD")
export DATASET=$(realpath "$DATASET" --relative-to "$PWD")
export BENCH_DIR=$(realpath "$BENCH_DIR" --relative-to "$PWD")

export MODEL_BASE=$(basename $(dirname "${MODEL}"))
export MODEL_NAME=$(basename -s .nnet "${MODEL}")
export DATASET_NAME=$(basename -s .csv "${DATASET}")
DATASET_NAME=${DATASET_NAME#${MODEL_BASE}[_-]}

export BUILD_DIR=$(realpath "$ROOT_DIR/build" --relative-to "$PWD")

function cleanup {
    local code=$1

    [[ -n $code && $code != 0 ]] && {
        pkill -TERM -P $$
        pkill -TERM -f $SOLVER_PROCESS_NAME
        wait
    }

    rm -f $tmp
    rm -f $TMP_SOLVED
    rm -f $TMP_TOUTED

    [[ -n $n_samples ]] && for ((i=1; i<=$n_samples; ++i)); do
        set_time_filename tfile $i
        rm -f $tfile
    done

    [[ $(basename "$PWD") == $$ ]] && {
        cd ..
        rm -fr $$
    }

    [[ -n $code ]] && exit $code
}

trap 'cleanup 9' INT TERM

case $ACTION in
gen)
    mkdir -p >/dev/null "$BENCH_DIR/$MODEL_BASE" || exit $?
;;
esac

case $ACTION in
gen)
    [[ -n $EPSILON ]] && {
        lo_line=$(sed -n '5p' "$MODEL")
        hi_line=$(sed -n '6p' "$MODEL")

        _split_string $lo_line lo_array ,
        _split_string $hi_line hi_array ,

        declare -a d_array
        for idx in ${!lo_array[@]}; do
            lo=${lo_array[$idx]}
            hi=${hi_array[$idx]}
            [[ $lo$hi =~ ^[0-9.]+$ && $(bc -l <<<"$hi >= $lo") == 1 ]] || {
                printf 'Unexpected lo/hi at index %d: %s %s\n' $idx "$lo" "$hi" >&2
                exit 3
            }
            d_array+=( $(bc -l <<<"$hi - $lo") )
        done
    }

    "$BUILD_DIR/spexplain" dump-psi "$MODEL" --encoding-neuron-vars true --encoding-output-vars true --encoding-neuron-activation-vars true --encoding-relu-lower-bounds false

    for psi in psi_c*.smt2; do
        "$SCRIPTS_DIR/polish_psi.sed" -i $psi
    done

    k=0
    tail -n +2 "$DATASET" | while read sample; do
        (( ++k ))

        ## *correct* class -> misclassifications (~sat) could occur
        _split_string $sample array ,
        correct_class=${array[-1]}
        unset array[-1]
        [[ -n $EPSILON ]] && {
            _check_array_sizes array d_array
        }

        psi=psi_c${correct_class}.smt2

        tmp=$(mktemp)
        [[ -n $PROOF ]] && {
            printf '(set-option :produce-proofs true)\n' >$tmp
        }

        cat $psi >>$tmp

        [[ -n $EPSILON ]] && _float_to_rat $EPSILON eps_rat

        for idx in ${!array[@]}; do
            i=$(( idx + 1 ))
            _float_to_rat ${array[$idx]} x_rat

            if [[ -z $EPSILON ]]; then
                printf '(assert (! (= x%d %s) :named %s))\n' $i "$x_rat" "ex$i" >>$tmp
            else
                d=${d_array[$idx]}
                _float_to_rat $d d_rat

                lo="(- $x_rat (* $eps_rat $d_rat))"
                hi="(+ $x_rat (* $eps_rat $d_rat))"
                printf '(assert (! (>= x%d %s) :named %s))\n' $i "$lo" "lx$i" >>$tmp
                printf '(assert (! (<= x%d %s) :named %s))\n' $i "$hi" "ux$i" >>$tmp
            fi
        done

        [[ -n $PREFER ]] && {
            ## hack
            "$BUILD_DIR/spexplain" "$MODEL" "$DATASET" nop -n1 -i$i --encoding-neuron-vars true --encoding-output-vars true --encoding-neuron-activation-vars true --encoding-relu-lower-bounds false >/dev/null 2>>$tmp
        }

        printf '\n(check-sat)\n' >>$tmp
        # printf '(get-proof)\n' >>$tmp

        set_filename file $k
        mv $tmp "$file"
    done || exit $?
;;

run)
    function get_n_solved {
        local -n var=$1

        var=$(wc -c <$TMP_SOLVED)
    }
    export -f get_n_solved

    function get_n_touted {
        local -n var=$1

        var=$(wc -c <$TMP_TOUTED)
    }
    export -f get_n_touted

    function set_time_filename {
        local -n var=$1
        local sample_id=$2

        var=${TMP_PREFIX}_time_${sample_id}
    }
    export -f set_time_filename

    function run1 {
        local sample_id=$1

        local file
        set_filename file $sample_id

        local tmp_so=$(mktemp)
        local tmp_se=$(mktemp)
        local tmp_t
        set_time_filename tmp_t $sample_id

        timeout $TIMEOUT bash -c "\\time -f %e -o \"$tmp_t\" \"$SOLVER\" \"$file\" >\"$tmp_so\" 2>\"$tmp_se\"" $SOLVER_PROCESS_NAME
        local ret=$?

        case $ret in
        0)
            printf '.' >>$TMP_SOLVED
            ;;
        $TIMEOUT_STATUS)
            printf '.' >>$TMP_TOUTED
            ;;
        *)
            printf 'Unexpected ERROR at %s:\n' "$file" >&2
            cat $tmp_se >&2
            [[ -s $tmp_so ]] && {
                printf '\nSolver output:\n' >&2
                cat $tmp_so >&2
            }
            ;;
        esac

        rm -f $tmp_so
        rm -f $tmp_se

        case $ret in
        0|$TIMEOUT_STATUS);;
        *) return 1;;
        esac

        [[ -n $INTERACTIVE ]] && {
            local n_solved
            local n_touted
            get_n_solved n_solved
            get_n_touted n_touted
            local n_total=$(( $n_solved + $n_touted ))
            printf '\r%d (S:%d, T:%d)' $n_total $n_solved $n_touted
        }

        return 0
    }
    export -f run1

    export TMP_PREFIX=$(mktemp)
    export TMP_SOLVED=${TMP_PREFIX}_solved
    export TMP_TOUTED=${TMP_PREFIX}_touted
    rm $TMP_PREFIX
    >$TMP_SOLVED
    >$TMP_TOUTED

    n_samples=$(wc -l <"$DATASET")
    (( --n_samples ))
    [[ -n $MAX_N ]] && (( $n_samples > $MAX_N )) && n_samples=$MAX_N

    [[ -z $CPU_PERCENTAGE ]] && CPU_PERCENTAGE=60

    if [[ -z $SLURM_CPUS_PER_TASK ]]; then
        JOBS=${CPU_PERCENTAGE}%
    else
        JOBS=$(( ($SLURM_CPUS_PER_TASK*$CPU_PERCENTAGE + 50)/100 ))
    fi

    parallel --halt soon,fail=1 --jobs "${JOBS}" 'run1 {}' ::: $(seq $n_samples)
    ret=$?

    [[ -n $INTERACTIVE ]] && printf '\r\033[K'

    (( $ret )) && {
        printf "Failed.\n" >&2
        cleanup 1
    }

    get_n_solved n_solved
    get_n_touted n_touted

    TIME='0'
    for ((i=1; i<=$n_samples; ++i)); do
        set_time_filename tfile $i
        [[ -s $tfile ]] || continue
        time=$(<$tfile)
        TIME+="+$time"
    done

    PAR2="$TIME"
    (( $n_touted )) && PAR2+="+ $n_touted*2*$TIMEOUT_S"

    TIME=$(bc -l <<<"($TIME)/$n_samples")
    PAR2=$(bc -l <<<"($PAR2)/$n_samples")

    printf '%d %.2f %.2f\n' $n_solved $TIME $PAR2 | tee -a "$RESULTS_FILE"
;;
esac

case $ACTION in
gen)
    printf 'Done.\n' | tee -a "$RESULTS_FILE"
;;
esac

cleanup 0
