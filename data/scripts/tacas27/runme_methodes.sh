#!/usr/bin/env bash
## Fig. 6: slices of the naive, guide-all and fix-all-no-vars explanations of heart_attack and loan,
## from the results of the mode
## USAGE: ./runme_methodes.sh [<mode>]   (default: smoke)
## The figures go to data/output/<mode>/

SCRIPT=$(realpath "$0")
TACAS_DIR=$(dirname "$SCRIPT")
export SCRIPTS_DIR=$(dirname "$TACAS_DIR")
export DATA_DIR=$(dirname "$SCRIPTS_DIR")
export EXPLANATION_DIR="$DATA_DIR/explanations"

export ROOT_DIR=$(dirname "$DATA_DIR")
export SPEXPLAIN="$ROOT_DIR/spexplain"
## spexplain interpret needs matplotlib and numpy
PYTHON=${PYTHON:-$ROOT_DIR/venv/bin/python}

MODE=${1:-smoke}
source "$TACAS_DIR/modes.sh"

export OUTPUT_DIR="$DATA_DIR/output/$MODE"

mkdir -p "$OUTPUT_DIR" >/dev/null || exit $?

width=50

## The variants shown together in one figure
VARIANTS=(
    fix-all-no-vars
    guide-all
    naive
)
strategy=itp_aweak_bstrong

## '<model_base> <sample> <var1> <var2>', the samples are those of the original s100 dataset (from 1);
## mode_sample gives their index in the mode's dataset
PICKS=(
    ## s97 in the paper
    'heart_attack 98 x3 x4'
    ## s62 in the paper
    'heart_attack 63 x4 x13'
    ## s7 in the paper
    'loan 8 x5 x7'
    ## s69 in the paper
    'loan 70 x1 x5'
)

failed=0

for l in 1 4; do
    pick_k=0
    for pick in "${PICKS[@]}"; do
        pick_char=$(printf "\\$(printf '%03o' $((97+$pick_k)))\n")
        (( ++pick_k ))

        read -r model_base orig_s var1 var2 <<<"$pick"

        dataset=$(mode_dataset $model_base)
        dataset_base=$(basename "$dataset" .csv)
        dataset_base=${dataset_base#${model_base}_}
        s=$(mode_sample $model_base $orig_s) || {
            printf 'Skipping %s sample %s: not in %s\n' $model_base $orig_s "$dataset"
            continue
        }
        model="$DATA_DIR/models/$model_base/${model_base}_${width}x$l.nnet"
        domains="$DATA_DIR/domains/$model_base/${model_base}.csv"

        fexps=()
        for variant in "${VARIANTS[@]}"; do
            fexp="$EXPLANATION_DIR/$model_base/${width}x$l/$dataset_base/$variant/${strategy}.phi.txt"
            [[ -r $fexp ]] || {
                printf 'Skipping %s ...\n' "$fexp"
                continue
            }
            fexps+=("$fexp")
        done
        (( ${#fexps[@]} )) || continue

        fimg="$OUTPUT_DIR/fig-6_${model_base}_${width}x${l}_${pick_char}-sample-${orig_s}_${strategy%%_*}_vars-${var1}-${var2}.png"
        printf 'Generating %s ...\n' "$fimg"
        "$PYTHON" "$SPEXPLAIN" interpret "${fexps[@]}" "slice vars $var1 $var2" "$dataset" --model "$model" --visualize fspace --samples $s --output-visualization "$fimg" -q || (( ++failed ))
    done
done

(( $failed )) && {
    printf 'Failed: %d\n' $failed >&2
    exit 1
}
exit 0
