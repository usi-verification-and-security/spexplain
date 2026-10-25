#!/usr/bin/env bash

SCRIPT=$(realpath "$0")
TACAS_DIR=$(dirname "$SCRIPT")
export SCRIPTS_DIR=$(dirname "$TACAS_DIR")
export DATA_DIR=$(dirname "$SCRIPTS_DIR")
export EXPLANATION_DIR="$DATA_DIR/explanations"

export ROOT_DIR=$(dirname "$DATA_DIR")
export SPEXPLAIN="$ROOT_DIR/spexplain"

export OUTPUT_DIR="$DATA_DIR/output"

mkdir -p "$OUTPUT_DIR" >/dev/null || exit $?

model_base=mnist
dataset_base=s100_scaled_smoke
width=50

dataset="$DATA_DIR/datasets/$model_base/${model_base}_${dataset_base}.csv"

# for s in 1 7; do
for s in 1; do
    # for l in 1 4; do
    for l in 1; do
        variant_k=0
        ##++ fix-all needs to be replaced by the "no-vars" configuration
        for variant in guide-all fix-all verix-eps{5,100}; do
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

            fimg="$OUTPUT_DIR/fig-7_sample-${s}_${variant_char}-${variant}_${strategy%%_*}_layer-${l}.png"
            printf 'Generating %s ...\n' "$fimg"
            ##++ this should probably by scheduled
            "$SPEXPLAIN" interpret "$fexp" 'norm min, inv' "$dataset" --visualize 'pixel center' --samples $s --output-visualization "$fimg" >/dev/null
        done
    done
done
