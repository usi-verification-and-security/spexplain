#!/bin/bash

DIRNAME=$(dirname "$0")
SCRIPTS_DIR=$(dirname "$DIRNAME")
DATA_DIR=$(realpath "$SCRIPTS_DIR/..")
ROOT_DIR=$(realpath "$DATA_DIR/..")

MODEL_DIR="$DATA_DIR/models"
DATASET_DIR="$DATA_DIR/datasets"

source "$SCRIPTS_DIR/lib/common"

ACTIONS=(gen run plot)

function usage {
    printf 'USAGE: %s <action> <args>...\n' "$0"
    printf 'ACTIONS: %s\n' "${ACTIONS[*]}"
    printf 'ARGUMENTS:\n'
    printf '\tgen|run:\t\t<bench_dir> <results_dir>\n'
    printf '\tplot:\t\t<results_dir>\n'

    [[ -n $1 ]] && exit $1
}

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

case $ACTION in
plot);;
*)
    [[ -z $1 ]] && {
        printf "Expected a benchmarks directory\n" >&2
        usage 1 >&2
    }

    case $ACTION in
    run)
        [[ -d $1 ]] || {
            printf "Expected a benchmarks directory, got: %s\n" "$1" >&2
            usage 1 >&2
        }
    ;;
    esac

    BENCH_DIR="$1"
    shift
;;
esac

[[ -z $1 ]] && {
    printf "Expected a results directory\n" >&2
    usage 1 >&2
}

RESULTS_DIR="$1"
shift

PLOT_DIR="$RESULTS_DIR/plots"

case $ACTION in
plot)
    mkdir -p "$PLOT_DIR" >/dev/null || exit $?
;;
*)
    mkdir -p "$RESULTS_DIR" >/dev/null || exit $?
;;
esac

[[ -n $SOLVER ]] && SOLVERS=$SOLVER
if [[ -n $SOLVERS ]]; then
    SOLVERS=($SOLVERS)
else
    SOLVERS=(
        opensmt
        /home/tomaqa/Data/Software/mathsat-5.6.12-prefer-linux-x86_64/bin/mathsat
        z3
    )
fi

[[ -n $MODEL ]] && MODELS=$MODEL
if [[ -n $MODELS ]]; then
    MODELS=($MODELS)
else
    MODELS=(
        heart_attack
    )
fi

[[ -n $WIDTH ]] && WIDTHS=$WIDTH
if [[ -n $WIDTHS ]]; then
    WIDTHS=($WIDTHS)
else
    WIDTHS=(
        50
        200
    )
fi

[[ -n $DEPTH ]] && DEPTHS=$DEPTH
if [[ -n $DEPTHS ]]; then
    DEPTHS=($DEPTHS)
else
    DEPTHS=(
        1
        2
        3
        4
        5
        6
    )
fi

[[ -n $EPSILON ]] && EPSILONS=$EPSILON
if [[ -n $EPSILONS ]]; then
    EPSILONS=($EPSILONS)
else
    EPSILONS=(
        # 0
        0.001
        0.005
        0.025
    )
fi

[[ -n $PROOF ]] && PROOFS=$PROOF
if [[ -n $PROOFS ]]; then
    PROOFS=($PROOFS)
else
    PROOFS=(
        0
        # 1
    )
fi

[[ -n $PREFER ]] && PREFERS=$PREFER
if [[ -n $PREFERS ]]; then
    PREFERS=($PREFERS)
else
    PREFERS=(
        0
        1
    )
fi

[[ -z $CPU_PERCENTAGE ]] && CPU_PERCENTAGE=80
export CPU_PERCENTAGE

[[ -z $TIMEOUT ]] && TIMEOUT=20m
export TIMEOUT

case $ACTION in
plot)
    source "$SCRIPTS_DIR/lib/run"

    time_to "$TIMEOUT" TIMEOUT_S || exit 1
;;
esac

function cleanup {
    local code=$1

    [[ -n $code && $code != 0 ]] && {
        pkill -P $$
        wait
    }

    [[ -n $code ]] && exit $code
}

trap 'cleanup 9' INT TERM

function set_results_file {
    local -n var=$1
    local solver_base="$2"
    local concrete_model="$3"
    local eps=$4
    local proof=$5
    local pref=$6

    var="$RESULTS_DIR/"

    if [[ -n $solver_base ]]; then
        var+="${solver_base}"
    else
        var+=$ACTION
    fi

    var+="-${concrete_model}-eps${eps}"

    (( $proof )) && var+='-proof'
    (( $pref )) && var+='-pref'
}

function set_plot_data_file {
    local -n var=$1
    local solver_base="$2"
    local model_width="$3"
    local eps=$4
    local proof=$5
    local pref=$6

    var="$PLOT_DIR/"
    var+="${solver_base}"
    var+="-${model_width}-eps${eps}"

    (( $proof )) && var+='-proof'
    (( $pref )) && var+='-pref'

    var+=.dat
}

function set_plot_file {
    local -n var=$1
    local model_width="$2"
    local eps=$3
    local proof=$4

    var="$PLOT_DIR/"
    var+="${model_width}-eps${eps}"

    (( $proof )) && var+='-proof'

    var+=.svg
}

function gen_gnuplot_script_header {
    local -n var=$1
    local pfile="$2"

    var=$(mktemp)

    local title=$(basename "$pfile")
    title=${title%.*}

    local ymax=$(($TIMEOUT_S*2))
    local ymid=$TIMEOUT_S

    cat <<-END >$var
set term svg dynamic font "Arvo,12" size 800,480 noenhance background "white"
set output "$pfile"

set title "$title"
set xlabel "Number of hidden layers"
set ylabel "Average PAR-2 [s]"

set style data linespoints
# set logscale y

set xrange [${DEPTHS[0]}:${DEPTHS[-1]}]
set yrange [0:$ymax]

set xtics 1

set arrow 1 from graph 0, first $ymid to graph 1, first $ymid nohead dt 5 lc rgb "#888888"
set link y2
set y2tics ("$ymax (max. PAR-2)" $ymax, "$ymid (timeout)" $ymid)
set rmargin 21

set key right bottom outside reverse Left opaque

plot \\
END
}

function to_pdf {
    local file="$1"

    local ofile="${file%.*}.pdf"

    rsvg-convert -f pdf -o "$ofile" "$file"
}

case $ACTION in
gen) AUX=(dummy);;
*) AUX=("${SOLVERS[@]}");;
esac

for aux in "${AUX[@]}"; do
    ARGS=()
    case $ACTION in
    gen) unset solver_base;;
    *)
        solver="$aux"
        ARGS+=("$solver")
        solver_base=$(basename "$solver")
        echo "$solver_base"
    ;;
    esac

    for model in ${MODELS[@]}; do
    for width in ${WIDTHS[@]}; do
        model_width=${model}_${width}
        echo $model_width

        for eps in ${EPSILONS[@]}; do
        for proof in ${PROOFS[@]}; do
        for pref in ${PREFERS[@]}; do
            echo "eps=$eps proof=$proof pref=$pref"

            case $ACTION in
            plot)
                set_plot_data_file pdfile "$solver_base" "$model_width" $eps $proof $pref
                rm -f "$pdfile"
            ;;
            esac

            for depth in ${DEPTHS[@]}; do
                concrete_model=${model_width}x${depth}
                case $ACTION in
                plot);;
                *)
                    echo $concrete_model
                ;;
                esac

                set_results_file rfile "$solver_base" "$concrete_model" $eps $proof $pref

                case $ACTION in
                plot)
                    [[ -r $rfile ]] || {
                        printf '\nReading results %s failed!\n' "$rfile" >&2
                        cleanup 10
                    }

                    {
                        printf '%d ' $depth
                        cat "$rfile"
                    } >>"$pdfile"
                ;;
                *)
                    EPSILON=$eps PROOF=$proof PREFER=$pref $WRAPPER "$DIRNAME"/benchmarks.sh $ACTION "${ARGS[@]}" "$MODEL_DIR"/$model/$concrete_model.nnet "$DATASET_DIR"/$model/${model}_s100_scaled.csv "$BENCH_DIR" "$rfile" &
                    (( $? )) && {
                        printf '\nUnexpected fork failure!\n' >&2
                        cleanup 10
                    }

                    [[ -n $PARALLEL ]] && continue

                    wait $!
                    ret=$?
                    (( $ret )) && {
                        printf '\nFailed!\n' >&2
                        cleanup $ret
                    }
                ;;
                esac
            done
        done
        done
        done
    done
    done

    echo
done

case $ACTION in
plot)
    for model in ${MODELS[@]}; do
    for width in ${WIDTHS[@]}; do
        model_width=${model}_${width}

        for eps in ${EPSILONS[@]}; do
        for proof in ${PROOFS[@]}; do
            set_plot_file pfile "$model_width" $eps $proof

            gen_gnuplot_script_header gpfile "$pfile"

            for solver_idx in ${!SOLVERS[@]}; do
                solver="${SOLVERS[$solver_idx]}"
                solver_base=$(basename "$solver")

                for pref in ${PREFERS[@]}; do
                    set_plot_data_file pdfile "$solver_base" "$model_width" $eps $proof $pref

                    title="${solver_base}"
                    (( $pref )) && title+=-pref

                    lc=$(($solver_idx+1))
                    dt=$(($pref+1))
                    pt=$lc

                    printf '  "%s" using 1:4 title "%s" lc %d dt %d pt %d, \\\n' "$pdfile" "$title" $lc $dt $pt >>"$gpfile"
                done
            done

            printf '  1/0 notitle\n' >>"$gpfile"

            echo "Generating ${pfile%.*}.pdf ..."

            # cat "$gpfile"
            gnuplot "$gpfile" || cleanup $?
            rm -f "$gpfile"

            to_pdf "$pfile" || cleanup $?
            rm -f "$pfile"
        done
        done
    done
    done
;;
*)
    [[ -n $PARALLEL ]] && {
        printf 'Waiting on the background jobs ...\n'
        wait
        ret=$?
        (( $ret )) && {
            printf 'Failed!\n' >&2
            cleanup $ret
        }
    }
;;
esac

printf '\nSuccess.\n'
cleanup 0
