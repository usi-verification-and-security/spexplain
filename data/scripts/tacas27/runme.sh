#!/usr/bin/env bash
## TACAS'27 experiments over all benchmarks and both widths (50x, 200x).
## All jobs (model x variant x experiment) share one GNU parallel queue, so at most
## CPU_PERCENTAGE of the CPUs are busy at any time, as with a single run-experiments.sh call.
## Within a job, the samples are always processed sequentially.
##
## USAGE: ./runme.sh <action> <mode> [--resume] [-n] [-h]
##
## Actions:
##   run            run the experiments of the mode
##   run-verix-plus run VeriX-plus on the mode's models and datasets (VERIX_NAMES in modes.sh: all),
##                  with the mode's timeouts and the perturbations epsilon 1, 0.05 and 0.005,
##                  as jobs of the same kind of queue as run (see verix/verix1.sh), configured as the original
##                  experiments: the random traversal, Marabou without Gurobi (as in the image docker/tacas27):
##                  data/explanations/<name>/<width>x<depth>/<dataset>/verix/eps{100,5,0.5}.phi.txt
##   collect-stats  collect the stats of the mode's finished experiments with collect_stats.sh,
##                  one file per benchmark (50x and 200x): <STATS_DIR>/<mode>/<name>.txt,
##                  and of the CNN models: <STATS_DIR>/<mode>/<name>_cnn.txt;
##                  only naive/guide-all/fix-all/hybrid and the itp experiments (see STATS_FILTER)
##   plot-performance
##                  plot the PAR-2 per model of the collected stats with plot_fc_stats.py
##                  and, of the CNN models, plot_cnn_stats.py:
##                  data/output/<mode>/fig-*.pdf, with the tables as
##                  data/output/<mode>/fig-5-performance-plots_<name>.csv and
##                  data/output/<mode>/fig-9_CNN_performance-plots_<name>_cnn.csv (run collect-stats first);
##                  with the results of run-verix-plus of the mode, also their run times as
##                  data/output/<mode>/fig-8_comparison-plots_<name>[200]_verix.csv and the comparison
##                  with plot_compare_with_verix_plus.py
##   interpret      make the interpretation figures from the mode's explanations with
##                  runme_norm.sh (Fig. 7, mnist), runme_diam.sh (Fig. 11, heart_attack and loan)
##                  runme_methodes.sh (Fig. 6, heart_attack and loan)
##                  and runme_ucore.sh (Fig. 10, heart_attack):
##                  data/output/<mode>/*; run also does this at the end
##
## Modes (see ./modes.sh):
##   In every mode but smoke, run also runs the unsat cores of interpolants 'itp aweak, bstrong; ucore min'
##   (ucore_min_itp_aweak_bstrong) of heart_attack 50x1, naive and fix-all, as jobs of the queue,
##   with the timeout per explanation 20m
##   smoke     dataset <name>_s100_scaled_smoke.csv (the important samples);
##             50x1-50x6 only, without cifar and gtsrb, guide-all/fix-all/hybrid/fix-all-no-vars,
##             itp_aweak_bstrong,
##             timeout per explanation 1m (tabular) or 5m (mnist)
##   short     as long, but dataset <name>_s100_scaled_short.csv (8 samples, the first 8 of long)
##             and timeouts per explanation 200s (tabular), 300s (mnist), 400s (cifar) and 600s (gtsrb)
##   long      dataset <name>_s100_scaled_long.csv (20 samples: those of short, then 12 more);
##             50x: everything, 200x: itp_aweak_bstrong only
##   full      dataset <name>_s100_scaled.csv (100 samples); everything
## smoke, short and long leave out the deep models where naive or guide-all mostly times out.
## fix-all-no-vars runs only for the 50x1-50x4 models of heart_attack, loan and mnist, in all modes.
## CNN (ONNX) models, as jobs of the same queue, in every mode (see CNN_* in ./modes.sh):
##   mnist, cifar and gtsrb (smoke: mnist only), models/<name>/cnn-bench/{NS,NAB}1-6.onnx
##   (smoke: up to NS4 and NAB4), naive/guide-all/fix-all/hybrid (smoke: no naive),
##   itp_aweak_bstrong; the dataset and the timeouts of the mode, as for the .nnet models.
##   They run cnn/run1.sh with the engine of 'make onnx' (build-onnx/engine), and the results go to
##   data/explanations/<name>/cnn-bench/<id>/<dataset>/<variant>/.
## The results go to data/explanations/<name>/<width>x<depth>/<dataset>/<variant>/,
## e.g. s100_scaled_short. In every mode, run also runs commands_fix-all_no_neuron_vars.sh at the end
## (explanations for interpretation, into data/explanations_for_interpretation/). The smaller datasets, the maps of their samples to the original ones
## (data/datasets/mode_sample_maps.csv) and their neuron activations are made by make_mode_datasets.py.
##
## Options:
##   --resume  run: skip the jobs that already finished in the previous run of the same mode
##   -n        dry run - only print the jobs and the worst-case estimate
##   -h        display this help and exit
##
## Env. variables (all optional):
##   CPU_PERCENTAGE    percentage of CPUs to use (default: 80)
##   CPUS              CPUs to take the percentage of (default: SLURM_CPUS_PER_TASK or all CPU threads)
##   EXPERIMENTS_SPEC  experiments collection of 'everything' (default: plain_itp)
##   VARIANTS_SPEC     variants collection of 'everything' (default: tacas27)
##   FILTER            regex over '<name>/<width>x<idx>/<variant>/<experiment>' job identifiers
##   LOG_DIR           directory for the job list and the joblog (default: data/logs/runme/<mode>)
##   STATS_DIR         collect-stats, plot-performance: directory of the stats files (default: ./stats)
##   STATS_FILTER      collect-stats: regex over the job identifiers of the jobs to collect
##                     (default in modes.sh: naive/guide-all/fix-all/hybrid, itp_* experiments)
##   PYTHON            plot-performance: Python with matplotlib and pandas
##                     (default: venv/bin/python of the repository, see requirements.txt)
##   DRY_RUN=1         same as -n
##   VERIX_DIR         run-verix-plus: VeriX-plus (default: ~/VeriX-plus)
##   VERIX_ORDER       run-verix-plus: the traversal order of VeriX-plus (default: random,
##                     as in the original experiments; e.g. bounds)
##   WRAPPER           if set, submit this script through it (e.g. sbatch.sh) instead of running it

(( ${BASH_VERSINFO[0]} >= 4 )) || {
    printf "Bash >= 4 is required, got %s. Run it e.g. with /opt/homebrew/bin/bash.\n" "$BASH_VERSION" >&2
    exit 1
}

## GNU parallel runs the jobs with its own choice of shell (e.g. $SHELL, the old /bin/bash of macOS),
## and they use $BASH, so make it run them with this bash
export PARALLEL_SHELL=$BASH

SCRIPT=$(realpath "$0")
TACAS_DIR=$(dirname "$SCRIPT")
export SCRIPTS_DIR=$(dirname "$TACAS_DIR")
export DATA_DIR=$(dirname "$SCRIPTS_DIR")

## Resubmit through the wrapper, so that all jobs share one allocation
[[ -n $WRAPPER ]] && exec env -u WRAPPER $WRAPPER "$SCRIPT" "$@"

function usage {
    ## The header comment, up to the first line that is not a '##' comment
    awk 'NR > 1 && !/^##/ { exit } /^##/ { sub(/^## ?/, ""); print }' "$SCRIPT"

    [[ -n $1 ]] && exit $1
}

[[ -z $PYTHON ]] && {
    command -v python3 >/dev/null && PYTHON=python3
}
export PYTHON

source "$TACAS_DIR/modes.sh"

ACTIONS=(run collect-stats plot-performance interpret run-verix-plus)

ACTION=
MODE=
RESUME=0
for arg; do
    case $arg in
    --resume)
        RESUME=1
        ;;
    -n)
        DRY_RUN=1
        ;;
    -h|--help)
        usage 0
        ;;
    -*)
        printf "Unrecognized option: %s\n\n" "$arg" >&2
        usage 1 >&2
        ;;
    *)
        if [[ -z $ACTION ]]; then
            ACTION=$arg
        elif [[ -z $MODE ]]; then
            MODE=$arg
        else
            printf "Additional argument: %s\n\n" "$arg" >&2
            usage 1 >&2
        fi
        ;;
    esac
done

[[ -n $ACTION && -n $MODE ]] || {
    printf "Expected an action and a mode.\n\n" >&2
    usage 1 >&2
}

[[ " ${ACTIONS[*]} " == *" $ACTION "* ]] || {
    printf "Unrecognized action: %s (expected one of: %s)\n\n" "$ACTION" "${ACTIONS[*]}" >&2
    usage 1 >&2
}

[[ " ${MODES[*]} " == *" $MODE "* ]] || {
    printf "Unrecognized mode: %s (expected one of: %s)\n\n" "$MODE" "${MODES[*]}" >&2
    usage 1 >&2
}

export CPU_PERCENTAGE=${CPU_PERCENTAGE:-80}

export EXPERIMENTS_SPEC=${EXPERIMENTS_SPEC:-plain_itp}
export VARIANTS_SPEC=${VARIANTS_SPEC:-tacas27}

source "$SCRIPTS_DIR/lib/run"

read_experiments_spec "$EXPERIMENTS_SPEC" || exit $?
read_variants_spec "$VARIANTS_SPEC" || exit $?
ALL_EXPERIMENTS=("${EXPERIMENT_NAMES[@]}")
ALL_VARIANTS=("${VARIANT_NAMES[@]}")

function to_secs {
    local t=$1
    case $t in
    *h) printf "%d" $(( ${t%h}*3600 )) ;;
    *m) printf "%d" $(( ${t%m}*60 )) ;;
    *s) printf "%d" ${t%s} ;;
    *)  printf "%d" $t ;;
    esac
}

## <name> <dataset file> -> the directory of the results of the dataset, as named by run1.sh:
## the dataset without the "<name>_" prefix, e.g. s100_scaled_short
function dataset_dir {
    local name=$1
    local dataset=$(basename "$2" .csv)

    printf "%s" "${dataset#${name}_}"
}

## Prints a job of make_jobs: <worst-case secs> <job id> <timeout per> <model> <dataset> <variant>
## <experiment> <strategies> <samples> <benchmark idx> <width> <depth> <variant idx>
function print_job {
    printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%02d.%04d.%03d.%02d\n" "$@"
}

## Writes the jobs of the mode to JOBS_TMP, one per line, tab-separated:
## <worst-case secs> <job id> <timeout per> <model> <dataset> <variant> <experiment> <strategies>
## <samples> <order: benchmark.width.depth.variant>
## For run-verix-plus, the variant is verix, the experiments are VERIX_EXPERIMENTS
## and the strategies are their epsilons.
## The longest jobs go first, so that they do not prolong the end of the run.
function make_jobs {
    local width i name timeout_per dataset n_samples worst_secs
    ## not 'strategies': find_strategies_for_experiment writes through a reference to this name,
    ## and has a local variable of that name itself
    local model model_id depth vidx variant experiment job_id job_strategies job_timeout_per job_worst_secs
    for width in ${WIDTHS[@]}; do
        set_mode_for_width $width || continue

        for i in ${!NAMES[@]}; do
            name=${NAMES[$i]}
            mode_includes_benchmark $name || continue
            [[ $ACTION == run-verix-plus && " ${VERIX_NAMES[*]} " != *" $name "* ]] && continue

            timeout_per=$(mode_timeout_per $i)
            dataset=$(mode_dataset $name)

            [[ -r $dataset ]] || {
                printf "Cannot read the dataset: %s\n" "$dataset" >&2
                printf "The datasets of the modes are made by %s\n" "$TACAS_DIR/make_mode_datasets.py" >&2
                exit 1
            }

            ## The first line is the header
            n_samples=$(( $(awk 'END { print NR }' "$dataset") - 1 ))
            worst_secs=$(( $n_samples * $(to_secs $timeout_per) ))

            for depth in $(model_depths $name); do
                model_id=${width}x$depth
                model="$DATA_DIR/models/$name/${name}_$model_id.nnet"

                [[ -r $model ]] || {
                    printf "Cannot read the model: %s\n" "$model" >&2
                    exit 1
                }

                [[ $ACTION == run-verix-plus ]] && {
                    is_excluded $name $width $depth verix && continue

                    for vidx in ${!VERIX_EXPERIMENTS[@]}; do
                        experiment=${VERIX_EXPERIMENTS[$vidx]}
                        job_id="$name/$model_id/verix/$experiment"
                        [[ -n $FILTER && ! $job_id =~ $FILTER ]] && continue

                        print_job $worst_secs "$job_id" $timeout_per "$model" "$dataset" \
                            verix $experiment ${VERIX_EPSILONS[$vidx]} $n_samples \
                            $i $width $depth $vidx
                    done
                    continue
                }

                for vidx in ${!MODE_VARIANTS[@]}; do
                    variant=${MODE_VARIANTS[$vidx]}
                    is_excluded $name $width $depth $variant && continue

                    for experiment in "${MODE_EXPERIMENTS[@]}" $(ucore_experiments $name $width $depth $variant); do
                        job_id="$name/$model_id/$variant/$experiment"
                        [[ -n $FILTER && ! $job_id =~ $FILTER ]] && continue

                        find_strategies_for_experiment $experiment job_strategies

                        job_timeout_per=$timeout_per
                        job_worst_secs=$worst_secs
                        is_ucore_experiment $experiment && {
                            job_timeout_per=$(mode_ucore_timeout_per)
                            job_worst_secs=$(( $n_samples * $(to_secs $job_timeout_per) ))
                        }

                        print_job $job_worst_secs "$job_id" $job_timeout_per "$model" "$dataset" \
                            $variant $experiment "$job_strategies" $n_samples \
                            $i $width $depth $vidx
                    done
                done
            done
        done
    done >"$JOBS_TMP.unsorted"

    ## VeriX-plus runs only on the fully connected models
    [[ $ACTION == run-verix-plus ]] || make_cnn_jobs >>"$JOBS_TMP.unsorted"

    sort -s -t$'\t' -k1,1nr "$JOBS_TMP.unsorted" >"$JOBS_TMP"
}

## As make_jobs, for the CNN (ONNX) models (CNN_* in modes.sh), to stdout. The job id is
## <name>/<id>/<variant>/<experiment>, e.g. mnist/NAB1/hybrid/itp_aweak_bstrong, and in the order,
## the width is 0 and the depth is <family index>*10 + <depth>
function make_cnn_jobs {
    set_cnn_mode

    local i name timeout_per dataset n_samples worst_secs
    local fidx family depth model_id model vidx variant experiment job_id job_strategies
    for name in ${CNN_NAMES[@]}; do
        mode_includes_benchmark $name || continue

        for i in ${!NAMES[@]}; do
            [[ ${NAMES[$i]} == $name ]] && break
        done

        timeout_per=$(mode_timeout_per $i)
        dataset=$(mode_dataset $name)

        [[ -r $dataset ]] || {
            printf "Cannot read the dataset: %s\n" "$dataset" >&2
            printf "The datasets of the modes are made by %s\n" "$TACAS_DIR/make_mode_datasets.py" >&2
            exit 1
        }

        ## The first line is the header
        n_samples=$(( $(awk 'END { print NR }' "$dataset") - 1 ))
        worst_secs=$(( $n_samples * $(to_secs $timeout_per) ))

        for fidx in ${!CNN_FAMILIES[@]}; do
            family=${CNN_FAMILIES[$fidx]}
            for depth in ${CNN_DEPTHS[@]}; do
                model_id=$family$depth
                model="$DATA_DIR/models/$name/$CNN_MODEL_SET/$model_id.onnx"

                [[ -r $model ]] || {
                    printf "Cannot read the model: %s\n" "$model" >&2
                    exit 1
                }

                for vidx in ${!MODE_CNN_VARIANTS[@]}; do
                    variant=${MODE_CNN_VARIANTS[$vidx]}
                    is_cnn_excluded $name $family $depth $variant && continue

                    for experiment in "${MODE_CNN_EXPERIMENTS[@]}"; do
                        job_id="$name/$model_id/$variant/$experiment"
                        [[ -n $FILTER && ! $job_id =~ $FILTER ]] && continue

                        find_strategies_for_experiment $experiment job_strategies

                        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%02d.%04d.%03d.%02d\n" \
                            $worst_secs "$job_id" $timeout_per "$model" "$dataset" \
                            $variant $experiment "$job_strategies" $n_samples \
                            $i 0 $(( $fidx*10 + $depth )) $vidx
                    done
                done
            done
        done
    done
}

JOBS_TMP=$(mktemp)
trap 'rm -f "$JOBS_TMP" "$JOBS_TMP.unsorted"' EXIT

make_jobs || exit $?

N_JOBS=$(wc -l <"$JOBS_TMP")
(( $N_JOBS )) || {
    printf "Nothing to run.\n" >&2
    exit 1
}

if [[ -z $CPUS ]]; then
    if [[ -n $SLURM_CPUS_PER_TASK ]]; then
        CPUS=$SLURM_CPUS_PER_TASK
    else
        CPUS=$(parallel --number-of-threads)
    fi
fi
SLOTS=$(( ($CPUS*$CPU_PERCENTAGE + 50)/100 ))
(( $SLOTS >= 1 )) || SLOTS=1

printf "Action:      %s\n" "$ACTION"
printf "Mode:        %s\n" "$MODE"
if [[ $ACTION == run-verix-plus ]]; then
    printf "Benchmarks:  %s\n" "${VERIX_NAMES[*]}"
    printf "Epsilons:    %s\n" "${VERIX_EPSILONS[*]}"
    ## The default as in verix/verix1.py
    printf "Order:       %s\n" "${VERIX_ORDER:-random}"
else
    printf "Experiments: %s\n" "$EXPERIMENTS_SPEC"
    printf "Variants:    %s\n" "$VARIANTS_SPEC"
fi
[[ -n $FILTER ]] && printf "Filter:      %s\n" "$FILTER"
printf "\n"

########################################################################
## run

## Per benchmark and width, and the worst case if every explanation hit the timeout.
## The wall time simulates the queue: the longest jobs first, each to the first free slot.
function print_estimate {
    awk -F'\t' -v slots=$SLOTS '
    function dur(s) { return sprintf("%dh %02dm", int(s/3600), int(s%3600/60)) }
    {
        split($2, id, "/")
        ## The CNN models have no width
        if (index(id[2], "x")) group = id[1] " " substr(id[2], 1, index(id[2], "x"))
        else group = id[1] " cnn"
        if (!(group in jobs)) order[++n_groups] = group
        jobs[group]++
        expl[group] += $9
        tout[group] = $3

        m = 1
        for (i = 2; i <= slots; i++) if (load[i] < load[m]) m = i
        load[m] += $1
        total += $1
        n_jobs++
        n_expl += $9
        if ($1 > longest) longest = $1
    }
    END {
        printf "%-18s %6s %13s %12s\n", "benchmark", "jobs", "explanations", "timeout/per"
        for (g = 1; g <= n_groups; g++) {
            printf "%-18s %6d %13d %12s\n", order[g], jobs[order[g]], expl[order[g]], tout[order[g]]
        }
        for (i = 1; i <= slots; i++) if (load[i] > wall) wall = load[i]
        printf "%-18s %6d %13d\n\n", "total", n_jobs, n_expl
#        printf "Worst case (every explanation times out):\n"
#        printf "  CPU time:     %s\n", dur(total)
#        printf "  longest job:  %s\n", dur(longest)
#        printf "  wall time:    %s\n\n", dur(wall)
    }' "$JOBS_TMP"
}

## The neuron activations are indexed by the samples of the dataset, so the smaller datasets
## need their own (see run1.sh); without them, run1.sh would use those of the original dataset.
## Of the CNN models, it only warns about the missing ones
function check_activations {
    local missing=()
    local model dataset name activations
    while IFS=$'\t' read -r model dataset; do
        [[ $model == *.onnx ]] && {
            name=$(basename "$(dirname "$(dirname "$model")")")
            activations="$DATA_DIR/neuron_activations/$name/$CNN_MODEL_SET"
            [[ $(dataset_dir $name "$dataset") == s100_scaled ]] || activations+="/$(dataset_dir $name "$dataset")"
            activations+="/$(basename "$model" .onnx).txt"
            [[ -r $activations ]] || printf "WARNING: neuron activations not found: %s\n" "$activations" >&2
            continue
        }

        name=$(basename "$(dirname "$model")")
        activations="$DATA_DIR/neuron_activations/$name/$(basename "$model" .nnet).txt"
        [[ -r $activations ]] || continue
        [[ $(dataset_dir $name "$dataset") == s100_scaled ]] && continue

        activations="$(dirname "$activations")/$(dataset_dir $name "$dataset")/$(basename "$activations")"
        [[ -r $activations ]] || missing+=("$activations")
    done < <(cut -f4,5 "$JOBS_TMP" | sort -u)

    (( ${#missing[@]} )) || return 0

    printf "Missing neuron activations of the dataset:\n" >&2
    printf "  %s\n" "${missing[@]}" >&2
    printf "They are made by %s\n" "$TACAS_DIR/make_mode_datasets.py" >&2
    return 1
}

## The CNN jobs need the frozen engine for ONNX networks; only a warning in a dry run
function check_onnx_engine {
    cut -f4 "$JOBS_TMP" | grep -q '\.onnx$' || return 0

    local engine="$(dirname "$DATA_DIR")/build-onnx/engine"
    [[ -x $engine ]] && return 0

    printf "Missing the engine for ONNX networks: %s\nMake it with: make onnx\n" "$engine" >&2
    [[ $DRY_RUN == 1 ]] && return 0
    return 1
}

## One job of the run action, or of run-verix-plus (the variant verix, the strategies are the epsilon)
function run_job {
    local job_id="$1"
    local timeout_per=$2
    local model="$3"
    local dataset="$4"
    local variant=$5
    local experiment=$6
    local strategies="$7"

    printf "Running: %s\n" "$job_id"

    if [[ $variant == verix ]]; then
        TIMEOUT_PER=$timeout_per \
            "$BASH" "$SCRIPTS_DIR/tacas27/verix/verix1.sh" "$model" "$dataset" $strategies $experiment
    else
        ## The unsat cores of interpolants (see UCORE_EXPERIMENTS in modes.sh) need OpenSMT
        local options=$OPTIONS
        [[ $experiment == ucore_* ]] && options+=" -V opensmt"

        if [[ $model == *.onnx ]]; then
            ## The frozen engine for ONNX networks, with ../cnn/spec/variants
            VARIANT=$variant TIMEOUT_PER=$timeout_per OPTIONS=$options \
                INPUT_MIN=$CNN_INPUT_MIN INPUT_MAX=$CNN_INPUT_MAX \
                "$BASH" "$SCRIPTS_DIR/cnn/run1.sh" "$model" "$dataset" "$strategies" $experiment
        else
            VARIANT=$variant TIMEOUT_PER=$timeout_per OPTIONS=$options \
                "$BASH" "$SCRIPTS_DIR/run1.sh" "$model" "$dataset" "$strategies" $experiment
        fi
    fi

    case $? in
    0)
        printf "Finished: %s\n" "$job_id"
        return 0
        ;;
    $TIMEOUT_STATUS)
        printf "Timeout: %s\n" "$job_id"
        return 0
        ;;
    *)
        printf "%s failed!\n" "$job_id" >&2
        return 1
        ;;
    esac
}
export -f run_job

function action_run {
    printf "Parallel:    %d jobs at once (%s%% of %d CPUs)\n\n" $SLOTS $CPU_PERCENTAGE $CPUS

    print_estimate
    check_activations || exit 1
    check_onnx_engine || exit 1

    run_jobs "${LOG_DIR:-$DATA_DIR/logs/runme/$MODE}"
}

## Runs all jobs in one GNU parallel queue, with the job list and the joblog in <log dir>
function run_jobs {
    local log_dir="$1"

    [[ $DRY_RUN == 1 ]] && {
        printf "DRY RUN - the longest jobs, in the order they would start:\n"
        cut -f2-3,9 "$JOBS_TMP" | head -20 | column -t
        (( $N_JOBS > 20 )) && printf "... (%d more)\n" $(( $N_JOBS - 20 ))
        exit 0
    }

    local joblog="$log_dir/joblog.txt"
    mkdir -p "$log_dir" || exit 1

    local parallel_opts=(--jobs $SLOTS --line-buffer --colsep '\t' --joblog "$joblog")

    if (( $RESUME )); then
        ## GNU parallel resumes by the job numbers, so the job list must be the same
        [[ ! -e $log_dir/jobs.txt ]] || cmp -s "$JOBS_TMP" "$log_dir/jobs.txt" || {
            printf "Cannot resume: the jobs differ from those of the previous run in %s\n" "$log_dir/jobs.txt" >&2
            exit 1
        }
        parallel_opts+=(--resume)
    elif [[ -e $joblog ]]; then
        mv "$joblog" "$log_dir/joblog.$(date +%Y%m%d_%H%M%S).txt"
    fi
    cp "$JOBS_TMP" "$log_dir/jobs.txt"

    ## The smoke test is to find problems, so stop at the first one
    if is_smoke_mode; then
        parallel_opts+=(--halt soon,fail=1)
    else
        parallel_opts+=(--halt never)
    fi

    printf "Jobs:   %s\nJoblog: %s\n\n" "$log_dir/jobs.txt" "$joblog"

    cut -f2-8 "$log_dir/jobs.txt" | parallel "${parallel_opts[@]}" run_job {1} {2} {3} {4} {5} {6} {7}
    local status=$?

#    ## >>> explanations for interpretation (added) >>>
#    ## In every mode, also compute the explanations without inner-neuron variables for the interpretation
#    ## (loan, obesity, 50x1-4, naive/guide-all/fix-all, on the _short datasets, whatever the mode is)
#    ## into data/explanations_for_interpretation/; see commands_fix-all_no_neuron_vars.sh
#    printf "\nRunning: explanations for interpretation (%s)\n" "$TACAS_DIR/commands_fix-all_no_neuron_vars.sh"
#    "$BASH" "$TACAS_DIR/commands_fix-all_no_neuron_vars.sh" || {
#        printf "explanations for interpretation failed!\n" >&2
#        (( status++ ))
#    }
#    ## <<< explanations for interpretation (added) <<<

    ## SECONDS counts from the start of the script
    local elapsed=$(printf "%dh %02dm %02ds" $(( $SECONDS/3600 )) $(( $SECONDS%3600/60 )) $(( $SECONDS%60 )))

    (( $status )) && {
        printf "\nFailed: %d (jobs: see the non-zero Exitval in %s; or the interpretation)\n" $status "$joblog" >&2
        printf "Overall running time: %s\n" "$elapsed" >&2
        exit 1
    }

    printf "\nSuccess.\n"
    printf "Overall running time: %s\n" "$elapsed"
    exit 0
}

########################################################################
## collect-stats

## Collects the stats of the finished experiments of one variant directory into one table;
## the output goes to <out_file>, the counts '<collected> <not finished> <failed>' to <out_file>.status
function collect_one {
    local out_file="$1"
    local variant_dir="$2"
    local variant=$3
    local experiments=($4)

    ## The unsat cores of interpolants last: the table keeps the header of the first experiment,
    ## with its timeout per sample, which plot_fc_stats.py and plot_cnn_stats.py use for the benchmark
    local e others=() ucores=()
    for e in "${experiments[@]}"; do
        if [[ $e == ucore_* ]]; then ucores+=($e); else others+=($e); fi
    done
    experiments=("${others[@]}" "${ucores[@]}")

    local err_file=$(mktemp)
    local table= rows= notes=
    local n_ok=0 n_unfinished=0 n_failed=0
    local experiment out spec
    for experiment in "${experiments[@]}"; do
        ## The unsat cores of interpolants are not in EXPERIMENTS_SPEC (see UCORE_EXPERIMENTS in modes.sh)
        spec=$EXPERIMENTS_SPEC
        [[ $experiment == ucore_* ]] && spec=$UCORE_EXPERIMENTS_SPEC

        ## The time file is written at the end of the experiment, so it is empty while running
        grep -q '^real' "$variant_dir/$experiment.time.txt" 2>/dev/null || {
            notes+="Not finished: $variant/$experiment"$'\n'
            (( n_unfinished++ ))
            continue
        }

        ## collect_stats.sh resolves the paths relative to data/
        if out=$(cd "$DATA_DIR" && "$BASH" "$SCRIPTS_DIR/collect_stats.sh" \
                "$variant_dir/" "$spec" "^$variant/${experiment//./\\.}\$" 2>"$err_file"); then
            ## The first experiment keeps the header lines of the table, the others add only their row
            if [[ -z $table ]]; then
                table="$out"$'\n'
            else
                rows+=$(tail -n 1 <<<"$out")$'\n'
            fi
            [[ -s $err_file ]] && notes+=$(cat "$err_file")$'\n'
            (( n_ok++ ))
        else
            notes+="Failed: $variant/$experiment: $(tr '\n' ' ' <"$err_file")$out"$'\n'
            (( n_failed++ ))
        fi
    done
    rm -f "$err_file"

    printf "%s%s%s" "$table" "$rows" "$notes" >"$out_file"
    printf "%d %d %d\n" $n_ok $n_unfinished $n_failed >"$out_file.status"
}
export -f collect_one

## Runs collect_stats.sh for every finished experiment of the mode's jobs and writes one table
## per variant directory, and one file per benchmark with both widths
function action_collect_stats {
    local stats_dir=${STATS_DIR:-$PWD/stats}/$MODE

    ## One task per variant directory, in the order benchmark, width, depth, variant:
    ## <idx> <stats file> <header> <variant dir> <variant> <experiments>
    ## The CNN models are in <name>/<CNN_MODEL_SET>/<id> and their stats in <name>_cnn.txt
    local tasks=$(mktemp)
    local job_id model dataset experiment name model_id variant key
    local -A experiments stats_files headers
    local keys=()
    while IFS=$'\t' read -r job_id model dataset experiment; do
        [[ $job_id =~ $STATS_FILTER ]] || continue
        IFS=/ read -r name model_id variant _ <<<"$job_id"
        if [[ $model == *.onnx ]]; then
            key="$name/$CNN_MODEL_SET/$model_id/$(dataset_dir $name "$dataset")/$variant"
            stats_files[$key]="$stats_dir/${name}_cnn.txt"
        else
            key="$name/$model_id/$(dataset_dir $name "$dataset")/$variant"
            stats_files[$key]="$stats_dir/$name.txt"
        fi
        headers[$key]="$name $model_id $variant"
        [[ -n ${experiments[$key]} ]] || keys+=("$key")
        experiments[$key]+="$experiment "
    done < <(sort -s -t$'\t' -k10,10 "$JOBS_TMP" | cut -f2,4,5,7)

    local idx
    for idx in ${!keys[@]}; do
        key=${keys[$idx]}
        variant=${key##*/}
        printf "%d\t%s\t%s\t%s\t%s\t%s\n" $idx "${stats_files[$key]}" "${headers[$key]}" \
            "$DATA_DIR/explanations/$key" $variant "${experiments[$key]% }"
    done >"$tasks"

    local n_tasks=$(wc -l <"$tasks")
    printf "Stats filter: %s\nVariant directories: %d\nStats:  %s\n\n" "$STATS_FILTER" $n_tasks "$stats_dir"

    [[ $DRY_RUN == 1 ]] && {
        printf "DRY RUN - the first stats files and variant directories:\n"
        cut -f2,4 "$tasks" | sed "s|$stats_dir/||; s|$DATA_DIR/||" | head -20 | column -t -s$'\t'
        (( $n_tasks > 20 )) && printf "... (%d more)\n" $(( $n_tasks - 20 ))
        rm -f "$tasks"
        exit 0
    }

    mkdir -p "$stats_dir" || exit 1
    local tmp_dir=$(mktemp -d)

    ## The stats of different directories are independent, so collect them in parallel
    cut -f1,4-6 "$tasks" | parallel --jobs $SLOTS --colsep '\t' collect_one "$tmp_dir/{1}.txt" {2} {3} {4}

    ## Assemble the files in order, each table preceded by its header, e.g. "cifar 50x1 naive"
    local n_ok=0 n_unfinished=0 n_failed=0
    local out_file header ok unfinished failed
    local -A truncated
    while IFS=$'\t' read -r idx out_file header _; do
        [[ -n ${truncated[$out_file]} ]] || {
            : >"$out_file"
            truncated[$out_file]=1
        }

        printf "%s\n" "$header" >>"$out_file"
        cat "$tmp_dir/$idx.txt" >>"$out_file"
        printf "\n" >>"$out_file"

        read -r ok unfinished failed <"$tmp_dir/$idx.txt.status"
        (( n_ok += ok, n_unfinished += unfinished, n_failed += failed ))
        (( $failed )) && printf "FAILED: %s (see %s)\n" "$header" "$out_file" >&2
    done <"$tasks"

    rm -rf "$tmp_dir"

    printf "\nExperiments: %d collected, %d not finished, %d failed.\n" $n_ok $n_unfinished $n_failed
    printf "Stats files:\n"
    cut -f2 "$tasks" | uniq | sed 's/^/  /'
    rm -f "$tasks"

    (( $n_failed )) && exit 1
    exit 0
}

########################################################################
## plot-performance

## Plots the stats of collect-stats with plot_fc_stats.py and plot_cnn_stats.py
## Writes the CSVs of the run times of run-verix-plus of the mode into <output dir>, as produce_csv.py
## of the original experiments: fig-8_comparison-plots_<name>_verix.csv (50x models) and
## fig-8_comparison-plots_<name>200_verix.csv (200x), with the columns Algorithm,epsilon,layers,PAR2:
## the average PAR-2 of the samples per epsilon (e.g. 5%) and model depth (layers, from 1).
## The time of a sample is its time.txt of verix/verix1.py; a sample without it (a timeout or a failure),
## or with at least the timeout per explanation of the mode, counts as twice the timeout.
## The algorithm is IR, i.e. the random traversal order, as run-verix-plus.
## The models without the results of run-verix-plus are left out. Prints the written CSVs (only them with -n).
function write_verix_csvs {
    local output_dir="$1"
    local width i name timeout_secs dataset n_samples depth vidx experiment results_dir csv rows s par2
    for width in ${WIDTHS[@]}; do
        set_mode_for_width $width || continue

        for i in ${!NAMES[@]}; do
            name=${NAMES[$i]}
            mode_includes_benchmark $name || continue
            [[ " ${VERIX_NAMES[*]} " == *" $name "* ]] || continue

            timeout_secs=$(to_secs $(mode_timeout_per $i))
            dataset=$(mode_dataset $name)
            [[ -r $dataset ]] || continue
            ## The first line is the header
            n_samples=$(( $(awk 'END { print NR }' "$dataset") - 1 ))

            rows=()
            for vidx in ${!VERIX_EXPERIMENTS[@]}; do
                experiment=${VERIX_EXPERIMENTS[$vidx]}
                for depth in $(model_depths $name); do
                    is_excluded $name $width $depth verix && continue

                    results_dir="$DATA_DIR/explanations/$name/${width}x$depth/$(dataset_dir $name "$dataset")/verix/$experiment"
                    [[ -d $results_dir ]] || continue

                    par2=$(for (( s=0; s < $n_samples; ++s )); do
                        if [[ -r $results_dir/$s/time.txt ]]; then cat "$results_dir/$s/time.txt"; else echo -; fi
                    done | awk -v t=$timeout_secs '{ sum += ($1 == "-" || $1 >= t) ? 2*t : $1 } END { printf "%.2f", sum/NR }')
                    ## e.g. eps5 -> 5%
                    rows+=("IR,${experiment#eps}%,$depth,$par2")
                done
            done
            (( ${#rows[@]} )) || continue

            ## The prefix as output_prefix of plot_compare_with_verix_plus.py
            csv="$output_dir/fig-8_comparison-plots_${name}$( (( $width == 50 )) || printf %s $width )_verix.csv"
            printf "%s\n" "$csv"
            [[ $DRY_RUN == 1 ]] && continue

            mkdir -p "$output_dir" || exit 1
            {
                printf "Algorithm,epsilon,layers,PAR2\n"
                printf "%s\n" "${rows[@]}"
            } >"$csv" || exit 1
        done
    done
}

function action_plot_performance {
    local stats_dir=${STATS_DIR:-$PWD/stats}/$MODE
    local python=${PYTHON:-$(dirname "$DATA_DIR")/venv/bin/python}
    local output_dir=$DATA_DIR/output/$MODE
    local plot_scripts=(plot_fc_stats.py plot_cnn_stats.py)

    printf "Stats:  %s\nPlots:  %s\n\n" "$stats_dir" "$output_dir"

    local stats_files=("$stats_dir"/*.txt)
    [[ -e ${stats_files[0]} ]] || {
        printf "No stats files in %s; run first: %s collect-stats %s\n" "$stats_dir" "$0" $MODE >&2
        exit 1
    }

    "$python" -c 'import matplotlib, pandas' 2>/dev/null || {
        printf "%s lacks matplotlib or pandas; create the venv (python3 -m venv venv && venv/bin/pip install -r requirements.txt)\nin the repository or set PYTHON to a Python that has them.\n" "$python" >&2
        exit 1
    }

    ## The comparison of our results with VeriX-plus, plot_compare_with_verix_plus.py, if there are the results
    ## of run-verix-plus of the mode: it reads our stats like plot_fc_stats.py (also their timeouts),
    ## and the CSVs of VeriX-plus of write_verix_csvs from the output directory.
    ## It writes output_dir/fig-8_comparison-plots_PAR-2_ours_vs_verix_plus_<layout>_<width>.pdf.
    local verix_csvs
    verix_csvs=$(write_verix_csvs "$output_dir") || exit 1
    if [[ -n $verix_csvs ]]; then
        printf "VeriX-plus run times%s:\n%s\n\n" "$( [[ $DRY_RUN == 1 ]] && printf " (would write)" )" "$(sed 's/^/  /' <<<"$verix_csvs")"
        plot_scripts+=(plot_compare_with_verix_plus.py)
    else
        printf "No results of run-verix-plus of the mode, no comparison with VeriX-plus.\n\n"
    fi

    local script
    [[ $DRY_RUN == 1 ]] && {
        printf "DRY RUN - would run:\n"
        for script in "${plot_scripts[@]}"; do
            printf "  %s\n" "$python $TACAS_DIR/$script $stats_dir $output_dir"
        done
        exit 0
    }

    for script in "${plot_scripts[@]}"; do
        "$python" "$TACAS_DIR/$script" "$stats_dir" "$output_dir" || {
            printf "\nPlotting failed: %s\n" $script >&2
            exit 1
        }
    done

    printf "\nPlots:\n"
    ls -1 "$output_dir"/fig-{5,8,9}[-_]*.pdf 2>/dev/null | sed 's/^/  /'
    exit 0
}

########################################################################
## interpret

INTERPRET_SCRIPTS=(runme_norm.sh runme_diam.sh runme_methodes.sh runme_ucore.sh)

## Runs the interpretation scripts on the mode's explanations; returns 1 if any failed
function run_interpretation {
    local script
    local failed=0

    printf "\nInterpretation: %s\n" "$DATA_DIR/output/$MODE"
    for script in "${INTERPRET_SCRIPTS[@]}"; do
        printf "\nRunning: %s %s\n" "$script" $MODE
        "$BASH" "$TACAS_DIR/$script" $MODE || {
            printf "%s failed!\n" "$script" >&2
            failed=1
        }
    done

    return $failed
}

function action_interpret {
    [[ $DRY_RUN == 1 ]] && {
        printf "DRY RUN - would run:\n"
        printf "  %s $MODE\n" "${INTERPRET_SCRIPTS[@]/#/$TACAS_DIR/}"
        exit 0
    }

    run_interpretation || exit 1
    exit 0
}

########################################################################
## run-verix-plus

## Returns 1 unless VeriX-plus can run as in the original experiments: installed, with Marabou without Gurobi
function check_verix {
    local verix_dir=${VERIX_DIR:-$HOME/VeriX-plus}
    local python=${VERIX_PYTHON:-$verix_dir/venv/bin/python}

    [[ -r $verix_dir/VeriX.py && -x $python ]] || {
        printf "VeriX-plus not found in %s (with %s), see the image docker/tacas27.\n" "$verix_dir" "$python" >&2
        return 1
    }

    ## With Gurobi, Marabou would solve the queries, which the original experiments did not
    local marabou_core=("$verix_dir"/Marabou/maraboupy/MarabouCore*.so)
    [[ -r ${marabou_core[0]} ]] || {
        printf "Marabou of VeriX-plus not found: %s\n" "${marabou_core[0]}" >&2
        return 1
    }
    ldd "${marabou_core[0]}" 2>/dev/null | grep -q libgurobi && {
        printf "Marabou of VeriX-plus is built with Gurobi, unlike in the original experiments: %s\n" "${marabou_core[0]}" >&2
        return 1
    }

    return 0
}

function action_run_verix_plus {
    printf "Parallel:    %d jobs at once\n\n" $SLOTS

    print_estimate

    check_verix || {
        [[ $DRY_RUN == 1 ]] || exit 1
        printf "VeriX-plus cannot run, see above.\n\n" >&2
    }

    run_jobs "${LOG_DIR:-$DATA_DIR/logs/runme/$MODE}/verix"
}

########################################################################

case $ACTION in
run)
    action_run
    ;;
collect-stats)
    action_collect_stats
    ;;
plot-performance)
    action_plot_performance
    ;;
interpret)
    action_interpret
    ;;
run-verix-plus)
    action_run_verix_plus
    ;;
esac
