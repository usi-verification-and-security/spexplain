#!/usr/bin/env bash
## Interpolation restricted to selected features, only for selected samples:
## 'itp <ITP_PARAMS>, vars x<i> x<j>' with OpenSMT, one experiment per pair of features.
##
## Each experiment explains only its samples; all other samples are recorded as '<null>'
## (as the skipped samples of interval_itp/common), so that the files have the usual format
## <output_dir>/itp_<ITP_PARAMS>_vars_x<i>_x<j>.{phi,stats,times,time,out,err}.txt (e.g. itp_aweak_bstrong_vars_x1_x11)
## (<output_dir> = explanations/<dataset>/<W>x<idx>/s100_scaled/<variant>) and the list of
## the not selected samples goes to <experiment>.skipped.txt.
##
## Environment variables (all optional):
##   DATASET       dataset (default: heart_attack)
##   WIDTH         neurons per layer (default: 50)
##   MODEL_IDXS    <WIDTH>x<idx> models (default: '1 4')
##   VARIANTS      neuron-activation variants (default: 'naive guide-all fix-all')
##   SELECTION     '<sample>:x<i>,x<j>' entries, samples and features 1-based (default: see below)
##   ITP_PARAMS    parameters of the itp strategy (default: 'aweak, bstrong')
##   TIMEOUT_PER   timeout per sample, <int>[smh]; empty = no limit (default: no limit)
##   OPENSMT_CMD   engine built with OpenSMT (default: <repo>/build/engine)
##   OUTPUT_ROOT   root of the outputs (default: explanations, i.e. data/explanations)
##
## USAGE: ./itp_vars_samples.sh

SCRIPTS_DIR=$(dirname "$(dirname "$(realpath "$0")")")
DATA_DIR=$(dirname "$SCRIPTS_DIR")
ROOT_DIR=$(dirname "$DATA_DIR")

cd "$DATA_DIR" || exit 1

source "$SCRIPTS_DIR/lib/run"

DATASET=${DATASET:-heart_attack}
WIDTH=${WIDTH:-50}
MODEL_IDXS=${MODEL_IDXS:-1 4}
VARIANTS=${VARIANTS:-naive guide-all fix-all}
SELECTION=${SELECTION:-84:x1,x11 65:x1,x3 98:x3,x4 47:x3,x10 84:x3,x5 77:x1,x12}
OPENSMT_CMD=${OPENSMT_CMD:-$ROOT_DIR/build/engine}
OUTPUT_ROOT=${OUTPUT_ROOT:-explanations}
ITP_PARAMS=${ITP_PARAMS:-aweak, bstrong}
set_experiment_name_from_strategies ITP_NAME "itp $ITP_PARAMS"

TIMEOUT_OPTS=()
TIMEOUT_PER_S=
[[ -n $TIMEOUT_PER ]] && {
    [[ $TIMEOUT_PER =~ ^([0-9]+)([smh]?)$ ]] || {
        printf "Unrecognized TIMEOUT_PER (expected <int>[smh]): %s\n" "$TIMEOUT_PER" >&2
        exit 1
    }
    case ${BASH_REMATCH[2]} in
        h) TIMEOUT_PER_S=$(( BASH_REMATCH[1]*3600 )) ;;
        m) TIMEOUT_PER_S=$(( BASH_REMATCH[1]*60 )) ;;
        *) TIMEOUT_PER_S=${BASH_REMATCH[1]} ;;
    esac
    TIMEOUT_OPTS=(--time-limit-per=$(( TIMEOUT_PER_S*1000 )))
}

[[ -x $OPENSMT_CMD ]] || { printf "Not an executable: %s\n" "$OPENSMT_CMD" >&2; exit 1; }

dataset_fn="datasets/$DATASET/${DATASET}_s100_scaled.csv"
[[ -r $dataset_fn ]] || { printf "Cannot read the dataset: %s\n" "$dataset_fn" >&2; exit 1; }

## experiment -> its samples
declare -A EXP_SAMPLES=() EXP_VARS=()
EXPERIMENTS=()
for entry in $SELECTION; do
    k=${entry%%:*}
    vars=${entry#*:}
    experiment="${ITP_NAME}_vars_${vars//,/_}"
    [[ -z ${EXP_VARS[$experiment]} ]] && EXPERIMENTS+=("$experiment")
    EXP_VARS[$experiment]="${vars//,/ }"
    EXP_SAMPLES[$experiment]+=" $k"
done

FAILED=0
function fail {
    printf "!!! %s\n" "$*" >&2
    FAILED=1
}

## Variant options as in spec/variants/all, with the neuron-activations file filled in as in run1.sh
function variant_options_of {
    local variant=$1 model=$2
    local -n lopts=$3

    local opts
    maybe_find_options_for_variant "$variant" opts

    local activations="${model/models\//neuron_activations\/}"
    activations="${activations%.*}.txt"
    local opt
    for opt in --input-{fix,prefer}-sample-neuron-activations; do
        opts="${opts//${opt}=\"\"/${opt}=$activations}"
    done
    opts="${opts//\"/}"

    lopts=($opts)
}

## Prints the sample header lines (sample, expected output, computed output) of sample k
function nop_head_of {
    local nop_stats=$1 k=$2
    awk -v k=$k '
        $0 ~ "^sample \\[" k "/" { p = 1 }
        p && /^$/ { exit }
        p && (/^sample \[/ || /^expected output:/ || /^computed output:/) { print }
    ' "$nop_stats"
}

for idx in $MODEL_IDXS; do
    model="models/$DATASET/${DATASET}_${WIDTH}x$idx.nnet"
    [[ -r $model ]] || { fail "Cannot read the model: $model"; continue; }

    for variant in $VARIANTS; do
        out_dir="$OUTPUT_ROOT/$DATASET/${WIDTH}x$idx/s100_scaled/$variant"
        work_dir="$out_dir/.work/itp_vars_samples.$$"
        mkdir -p "$work_dir" || exit 1

        variant_options_of "$variant" "$model" var_opts
        opts=(--quiet --format=smtlib2 -V opensmt "${TIMEOUT_OPTS[@]}" "${var_opts[@]}")

        ## Headers of all samples, for the samples that are not explained
        nop_stats="$work_dir/nop.stats.txt"
        "$OPENSMT_CMD" "$model" "$dataset_fn" nop "${opts[@]}" -e "$work_dir/nop.phi.txt" -s "$nop_stats" \
            >/dev/null 2>"$work_dir/nop.err.txt" && grep -q '^sample \[' "$nop_stats" || {
            fail "$out_dir: nop preflight failed, see $work_dir/nop.err.txt"
            continue
        }
        n_samples=$(sed -n 's/^Dataset size: *//p' "$nop_stats")

        for experiment in "${EXPERIMENTS[@]}"; do
            name="${out_dir#$OUTPUT_ROOT/}/$experiment"
            strategies="itp $ITP_PARAMS, vars ${EXP_VARS[$experiment]}"
            selected=" ${EXP_SAMPLES[$experiment]} "
            tmp="$work_dir/$experiment"
            rm -rf "$tmp"
            mkdir -p "$tmp" || exit 1

            {
                awk '/^-+$/ { exit } /^(Starting from|Ending at) sample/ { next } /^Timeout per sample/ { next } { print }' "$nop_stats"
                [[ -n $TIMEOUT_PER_S ]] && printf "Timeout per sample [s]: %s\n" $TIMEOUT_PER_S
                printf -- '-%.0s' {1..60}
                printf "\n"
            } >"$tmp/stats"
            : >"$tmp/phi"; : >"$tmp/times"; : >"$tmp/out"; : >"$tmp/err"; : >"$tmp/skipped"
            : >"$tmp/time"

            n_ok=0 n_timeout=0 n_error=0
            for k in $(seq 1 $n_samples); do
                if [[ $selected != *" $k "* ]]; then
                    printf "<null> \n" >>"$tmp/phi"
                    printf "<null>\n" >>"$tmp/times"
                    { printf "\n"; nop_head_of "$nop_stats" $k; printf "#checks: 0\n<timeout>\n"; } >>"$tmp/stats"
                    printf "%d not-selected\n" $k >>"$tmp/skipped"
                    continue
                fi

                f="$tmp/s$k"
                { time "$OPENSMT_CMD" "$model" "$dataset_fn" "$strategies" "${opts[@]}" -i $k,$k \
                    --output-explanations="$f.phi.txt" --output-stats="$f.stats.txt" --output-times="$f.times.txt" \
                    >"$f.out.txt" 2>"$f.err.txt" ; } 2>"$f.time.txt"
                code=$?

                if (( code == 0 )) && [[ $(wc -l <"$f.phi.txt" | tr -d ' ') == 1 ]] && grep -q '^sample \[' "$f.stats.txt"; then
                    cat "$f.phi.txt" >>"$tmp/phi"
                    cat "$f.times.txt" >>"$tmp/times"
                    awk 'p { print } /^-+$/ { p = 1 }' "$f.stats.txt" >>"$tmp/stats"
                    if grep -q '^<timeout>' "$f.stats.txt"; then
                        (( ++n_timeout ))
                    else
                        (( ++n_ok ))
                    fi
                else
                    printf "<null> \n" >>"$tmp/phi"
                    printf "<null>\n" >>"$tmp/times"
                    { printf "\n"; nop_head_of "$nop_stats" $k; printf "#checks: 0\n<timeout>\n"; } >>"$tmp/stats"
                    (( ++n_error ))
                    fail "$name: sample $k failed with exit code $code, see $f.err.txt"
                fi
                [[ -s $f.out.txt ]] && { printf "=== sample %d\n" $k; cat "$f.out.txt"; } >>"$tmp/out"
                [[ -s $f.err.txt ]] && { printf "=== sample %d\n" $k; cat "$f.err.txt"; } >>"$tmp/err"
                { printf "=== sample %d" $k; cat "$f.time.txt"; } >>"$tmp/time"
                printf "[%s] %s sample %d: %s\n" "$(sed -n 's/^real[[:space:]]*//p' "$f.time.txt")" "$name" $k "$strategies"
            done

            n_lines=$(wc -l <"$tmp/phi" | tr -d ' ')
            (( n_lines == n_samples )) || { fail "$name: $n_lines explanations instead of $n_samples, not saved"; continue; }

            [[ -e $out_dir/$experiment.phi.txt ]] && printf "=== %s: overwriting the existing results\n" "$name"
            for t in stats times time out err skipped; do
                mv "$tmp/$t" "$out_dir/$experiment.$t.txt" || exit 1
            done
            ## The phi file last: its existence marks a finished experiment
            mv "$tmp/phi" "$out_dir/$experiment.phi.txt" || exit 1
            (( n_error )) || rm -rf "$tmp"

            printf "=== %s: %d explained, %d timeouts, %d errors (samples:%s)\n" \
                "$name" $n_ok $n_timeout $n_error "${EXP_SAMPLES[$experiment]}"
        done
        (( FAILED )) || rm -rf "$work_dir"
        rmdir "$out_dir/.work" 2>/dev/null
    done
done

if (( FAILED )); then
    printf "\nDone with problems (see '!!!' above).\n"
    exit 1
fi
printf "\nSuccess.\n"
exit 0
