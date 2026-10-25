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

dataset_base=s100_scaled_smoke
width=50

HA_VARS=(
    'x1 x5'
    'x5 x8'
    'x1 x8'
    'x10 x11'
)
LOAN_VARS=(
    'x4 x7'
    'x7 x8'
    'x1 x8'
    'x1 x5'
)

##++ only in short
# for model_base in heart_attack loan; do
for model_base in heart_attack; do
    l=1
    dataset="$DATA_DIR/datasets/$model_base/${model_base}_${dataset_base}.csv"
    model="$DATA_DIR/models/$model_base/${model_base}_${width}x$l.nnet"
    domains="$DATA_DIR/domains/$model_base/${model_base}.csv"

    variant=guide-all
    strategy=itp_aweak_bstrong

    fexp="$EXPLANATION_DIR/$model_base/${width}x$l/$dataset_base/$variant/${strategy}.phi.txt"

    if [[ $model_base == heart_attack ]]; then
        ## s20 in the paper, 21 in the full s100 dataset
        s=1
        VARS=("${HA_VARS[@]}")
    else
        ## s89 in the paper, 90 in the full s100 dataset
        ##++ only in short
        s=8
        VARS=("${LOAN_VARS[@]}")
    fi

    vars_k=0
    for vars in "${VARS[@]}"; do
        vars_char=$(printf "\\$(printf '%03o' $((97+$vars_k)))\n")
        (( ++vars_k ))

        var1=${vars% *}
        var2=${vars#* }

        fimg="$OUTPUT_DIR/fig-11_${model_base}_sample-${s}_${variant}_${strategy%%_*}_${vars_char}-vars-${var1}-${var2}.png"
        fimg_mathexpr="${fimg%.png}.pdf"
        fimg_mathexpr="${fimg_mathexpr/fig-11_/fig-11_mathexpr_}"
        printf 'Generating %s ...\n' "$fimg"
        "$SPEXPLAIN" interpret "$fexp" "diam vars $vars" "$dataset" --model "$model" --visualize --visualization-domains "$domains" --samples $s --output-visualization "$fimg" -q
        printf 'Generating %s ...\n' "$fimg_mathexpr"
        "$SPEXPLAIN" interpret "$fexp" "diam vars $vars" "$dataset" --visualize mathexpr --visualization-domains "$domains" --samples $s --output-visualization "$fimg_mathexpr" -q
    done
done
