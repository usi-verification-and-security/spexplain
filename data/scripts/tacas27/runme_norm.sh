#!/usr/bin/env bash
## Fig. 7: the norm of the mnist explanations, from the results of the mode
## USAGE: ./runme_norm.sh [<mode>]   (default: smoke)
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

model_base=mnist
dataset=$(mode_dataset $model_base)
dataset_base=$(basename "$dataset" .csv)
dataset_base=${dataset_base#${model_base}_}
width=50

failed=0

## The samples of the original s100 dataset; mode_sample gives their index in the mode's dataset
# for orig_s in 1 7; do
for orig_s in 1; do
    s=$(mode_sample $model_base $orig_s) || {
        printf 'Skipping sample %s: not in %s\n' $orig_s "$dataset"
        continue
    }
    # for l in 1 4; do
    for l in 1; do
        variant_k=0
        for variant in guide-all fix-all-no-vars verix-eps{5,100}; do
            variant_char=$(printf "\\$(printf '%03o' $((97+$variant_k)))\n")
            (( ++variant_k ))

            if [[ $variant =~ ^verix ]]; then
                strategy=${variant#*-}
                variant=${variant%%-*}
            else
                strategy=itp_aweak_bstrong
            fi

            fexp="$EXPLANATION_DIR/$model_base/${width}x$l/$dataset_base/$variant/${strategy}.phi.txt"
            [[ -r $fexp ]] || {
                printf 'Skipping %s ...\n' "$fexp"
                continue
            }

            fimg="$OUTPUT_DIR/fig-7_sample-${orig_s}_${variant_char}-${variant}_${strategy%%_*}_layer-${l}.png"
            printf 'Generating %s ...\n' "$fimg"
            ##++ this should probably by scheduled
            "$PYTHON" "$SPEXPLAIN" interpret "$fexp" 'norm min, inv' "$dataset" --visualize 'pixel center' --samples $s --output-visualization "$fimg" >/dev/null || (( ++failed ))
        done
    done
done

(( $failed )) && {
    printf 'Failed: %d\n' $failed >&2
    exit 1
}
exit 0
