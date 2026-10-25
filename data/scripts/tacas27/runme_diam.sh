#!/usr/bin/env bash
## Fig. 11: the diameter of the heart_attack and loan explanations, from the results of the mode
## USAGE: ./runme_diam.sh [<mode>]   (default: smoke)
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

for model_base in heart_attack loan; do
    l=1
    dataset=$(mode_dataset $model_base)
    dataset_base=$(basename "$dataset" .csv)
    dataset_base=${dataset_base#${model_base}_}
    model="$DATA_DIR/models/$model_base/${model_base}_${width}x$l.nnet"
    domains="$DATA_DIR/domains/$model_base/${model_base}.csv"

    variant=guide-all
    strategy=itp_aweak_bstrong

    fexp="$EXPLANATION_DIR/$model_base/${width}x$l/$dataset_base/$variant/${strategy}.phi.txt"
    [[ -r $fexp ]] || {
        printf 'Skipping %s ...\n' "$fexp"
        continue
    }

    ## The samples of the original s100 dataset; mode_sample gives their index in the mode's dataset
    if [[ $model_base == heart_attack ]]; then
        ## s20 in the paper
        orig_s=21
        VARS=("${HA_VARS[@]}")
    else
        ## s89 in the paper; not in the smoke dataset
        orig_s=90
        VARS=("${LOAN_VARS[@]}")
    fi
    s=$(mode_sample $model_base $orig_s) || {
        printf 'Skipping %s sample %s: not in %s\n' $model_base $orig_s "$dataset"
        continue
    }

    vars_k=0
    for vars in "${VARS[@]}"; do
        vars_char=$(printf "\\$(printf '%03o' $((97+$vars_k)))\n")
        (( ++vars_k ))

        var1=${vars% *}
        var2=${vars#* }

        fimg="$OUTPUT_DIR/fig-11_${model_base}_sample-${orig_s}_${variant}_${strategy%%_*}_${vars_char}-vars-${var1}-${var2}.png"
        fimg_mathexpr="${fimg%.png}.pdf"
        fimg_mathexpr="${fimg_mathexpr/fig-11_/fig-11_mathexpr_}"
        printf 'Generating %s ...\n' "$fimg"
        "$PYTHON" "$SPEXPLAIN" interpret "$fexp" "diam vars $vars" "$dataset" --model "$model" --visualize --visualization-domains "$domains" --samples $s --output-visualization "$fimg" -q &
        printf 'Generating %s ...\n' "$fimg_mathexpr"
        "$PYTHON" "$SPEXPLAIN" interpret "$fexp" "diam vars $vars" "$dataset" --visualize mathexpr --visualization-domains "$domains" --samples $s --output-visualization "$fimg_mathexpr" -q &
    done
done

wait || {
    printf 'Failed: %d\n' $failed >&2
    exit 1
}
exit 0
