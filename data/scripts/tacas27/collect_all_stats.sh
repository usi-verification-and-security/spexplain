#!/usr/bin/env bash
## Run collect_stats.sh for every variant directory found under an explanations tree
## and write all tables into one file, each preceded by a header line, e.g. "cifar 50x1 naive".
##
## A variant directory is any directory that directly contains *.stats.txt files, at any depth.
## If such a directory is named Sn<N> (results with max. N samples), its parent is the variant
## and N is passed to collect_stats.sh as <max_samples>.
## The header line is the path relative to the explanations root, without the sampling directory
## (the parent of the variant, e.g. s100_scaled or test100), e.g.:
##   cifar/50x1/s100_scaled/naive               -> cifar 50x1 naive
##   gtsrb/cnn-bench/P4/s100_scaled/hybrid/Sn20 -> gtsrb cnn-bench P4 hybrid Sn20
##
## USAGE: collect_all_stats.sh [<output_file>] [<experiments_spec>] [<filter_regex>]
##   defaults: stats_all.txt, itp, /itp_a
## Can be run from any directory.
## Env. variables:
##   EXPLANATIONS_ROOT  root of the explanations tree (default: data/explanations)
##   DATASETS           space-separated list of top-level directories to include (default: all)
##   ALLOW_PARTIAL=1    accept experiments that processed only the first samples of the dataset

DATA_DIR=$(realpath "$(dirname "$0")/../..")

OUTPUT_FILE=${1:-stats_all.txt}
# absolute path (the output file need not exist yet, which BSD realpath does not allow)
OUTPUT_FILE="$(cd "$(dirname "$OUTPUT_FILE")" && pwd)/$(basename "$OUTPUT_FILE")" || exit 1
SPEC=${2:-itp}
FILTER=${3:-/itp_a}

EXPLANATIONS_ROOT=$(realpath "${EXPLANATIONS_ROOT:-$DATA_DIR/explanations}") || exit 1

# collect_stats.sh resolves paths relative to data/
cd "$DATA_DIR" || exit 1

if [[ -n $DATASETS ]]; then
    search_dirs=()
    for dataset in $DATASETS; do
        search_dirs+=("$EXPLANATIONS_ROOT/$dataset")
    done
else
    search_dirs=("$EXPLANATIONS_ROOT")
fi

: >"$OUTPUT_FILE"

n_ok=0
n_failed=0
# natural order: 50x1, 50x2, ..., 50x10, ..., 200x1, ...
while IFS= read -r stats_dir; do
    variant_dir="$stats_dir"
    max_samples=()
    suffix=
    if [[ $(basename "$stats_dir") =~ ^Sn([0-9]+)$ ]]; then
        variant_dir=$(dirname "$stats_dir")
        max_samples=("${BASH_REMATCH[1]}")
        suffix=" $(basename "$stats_dir")"
    fi

    rel_path=${variant_dir#$EXPLANATIONS_ROOT/}
    sampling_parent=$(dirname "$(dirname "$rel_path")")
    header="${sampling_parent//\// } $(basename "$rel_path")${suffix}"

    printf "%s\n" "$header" >>"$OUTPUT_FILE"
    if scripts/collect_stats.sh "$variant_dir/" "$SPEC" "${max_samples[@]}" "$FILTER" >>"$OUTPUT_FILE" 2>&1; then
        (( n_ok++ ))
    else
        (( n_failed++ ))
        printf "FAILED: %s (see %s)\n" "$header" "$OUTPUT_FILE" >&2
    fi
    printf "\n" >>"$OUTPUT_FILE"
done < <(find "${search_dirs[@]}" -name '*.stats.txt' -exec dirname {} \; | sort -u | sort -V)

printf "Done: %d succeeded, %d failed. Output: %s\n" $n_ok $n_failed "$OUTPUT_FILE" >&2
