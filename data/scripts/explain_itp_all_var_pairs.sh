#!/bin/bash

DIRNAME=$(realpath $(dirname "$0"))
DATA_DIR=$(realpath "$DIRNAME/../")
ROOT_DIR=$(realpath "$DATA_DIR/../")
BUILD_DIR=$(realpath "$ROOT_DIR/build")
CMD=$(realpath "$BUILD_DIR/spexplain")
export CMD

[[ -z $N ]] && N=13
[[ -z $MODEL ]] && MODEL="$DATA_DIR/models/heart_attack/heart_attack-50.nnet"
[[ -z $DATASET ]] && DATASET="$DATA_DIR/datasets/heart_attack/heart_attack_full.csv"
export MODEL
export DATASET

[[ -z $OUTPUT_DIR ]] && OUTPUT_DIR=$(realpath "$ROOT_DIR/itp_pairs")
export OUTPUT_DIR

[[ -d $OUTPUT_DIR ]] && {
    printf 'Output directory %s already exists, overwrite? [y] ' "$OUTPUT_DIR" >&2
    read a
    if [[ ${a,,} == y ]]; then
        rm -fr "$OUTPUT_DIR"
    else
        exit 0
    fi
}

[[ -z $CPU_PERCENTAGE ]] && CPU_PERCENTAGE=50

set -e

mkdir "$OUTPUT_DIR"

PAIRS=()
for (( x1=1; x1<=N; x1++ )); do
    for (( x2=x1+1; x2<=N; x2++ )); do
        PAIRS+=("$x1 $x2")
    done
done

function run1 {
    local x1="$1"
    local x2="${x1#* }"
    x1="${x1% *}"

    local aitp="$2"
    local bitp=bstrong

    local alabel="$aitp"
    [[ $alabel =~ afactor ]] && alabel=amid
    local basename="itp_${alabel}_${bitp}_vars_x${x1}_x${x2}"
    "$CMD" explain "$MODEL" "$DATASET" "itp $aitp, $bitp, vars x$x1 x$x2" -Sq -e "$OUTPUT_DIR/${basename}.phi.txt" -s "$OUTPUT_DIR/${basename}.stats.txt"
}
export -f run1

parallel --halt soon,fail=1 --jobs ${CPU_PERCENTAGE}% 'run1 {} {}' ::: "${PAIRS[@]}" ::: astrong 'afactor 0.5' aweak
