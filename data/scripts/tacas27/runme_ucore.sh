#!/usr/bin/env bash
## Fig. 10: slices of the weak interpolant and of its minimal unsat core of heart_attack 50x1,
## naive and fix-all, from the results of the mode
## (as explviz: python_scripts/visualization/plots/tacas27/heart_attack_ucore_weak_methods.pdf)
## USAGE: ./runme_ucore.sh [<mode>]   (default: smoke)
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

model_base=heart_attack
width=50
l=1

## One figure per variant, in the order of the paper
VARIANTS=(
    naive
    fix-all
)
## The explanations shown together in one figure; the largest region first, i.e. underneath
STRATEGIES=(
    ucore_min_itp_aweak_bstrong
    itp_aweak_bstrong
)

## '<sample> <var1> <var2>', the samples are those of the original s100 dataset (from 1);
## mode_sample gives their index in the mode's dataset
PICKS=(
    ## s83 in the paper
    '84 x8 x11'
    ## s54 in the paper
    '55 x5 x7'
)

dataset=$(mode_dataset $model_base)
dataset_base=$(basename "$dataset" .csv)
dataset_base=${dataset_base#${model_base}_}
model="$DATA_DIR/models/$model_base/${model_base}_${width}x$l.nnet"

failed=0

pick_k=0
for pick in "${PICKS[@]}"; do
    pick_char=$(printf "\\$(printf '%03o' $((97+$pick_k)))\n")
    (( ++pick_k ))

    read -r orig_s var1 var2 <<<"$pick"

    s=$(mode_sample $model_base $orig_s) || {
        printf 'Skipping %s sample %s: not in %s\n' $model_base $orig_s "$dataset"
        continue
    }

    variant_k=0
    for variant in "${VARIANTS[@]}"; do
        variant_char=$(printf "\\$(printf '%03o' $((97+$variant_k)))\n")
        (( ++variant_k ))

        fexps=()
        for strategy in "${STRATEGIES[@]}"; do
            fexp="$EXPLANATION_DIR/$model_base/${width}x$l/$dataset_base/$variant/${strategy}.phi.txt"
            [[ -r $fexp ]] || {
                printf 'Skipping %s ...\n' "$fexp"
                continue
            }
            fexps+=("$fexp")
        done
        (( ${#fexps[@]} )) || continue

        fimg="$OUTPUT_DIR/fig-10_${model_base}_${width}x${l}_${pick_char}-sample-${orig_s}_${variant_char}-${variant}_itp_vars-${var1}-${var2}.png"
        printf 'Generating %s ...\n' "$fimg"
        "$PYTHON" "$SPEXPLAIN" interpret "${fexps[@]}" "slice vars $var1 $var2" "$dataset" --model "$model" --visualize fspace --samples $s --output-visualization "$fimg" -q || (( ++failed ))
    done
done

(( $failed )) && {
    printf 'Failed: %d\n' $failed >&2
    exit 1
}
exit 0
