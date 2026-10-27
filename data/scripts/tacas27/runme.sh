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
##   collect-stats  collect the stats of the mode's finished experiments with collect_stats.sh,
##                  one file per benchmark (50x and 200x): <STATS_DIR>/<mode>/<name>.txt;
##                  only naive/guide-all/fix-all/hybrid and the itp experiments (see STATS_FILTER)
##   plot-performance
##                  plot the PAR-2 per model of the collected stats with plot_fc_stats.py:
##                  data/output/<mode>/performance-plots_*.pdf, with the tables as
##                  data/output/<mode>/performance-plots_<name>.csv (run collect-stats first)
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
##   smokeLong a smoke test for us, will be removed later: smoke plus naive (heart_attack
##             50x1-50x4 only) and the unsat cores; the smoke dataset and results folder
##   quick     as short, with timeouts per explanation 200s (tabular), 300s (mnist),
##             400s (cifar) and 600s (gtsrb)
##   short     dataset <name>_s100_scaled_short.csv (8 samples, the same as quick);
##             50x: everything, 200x: itp_aweak_bstrong only
##   full      dataset <name>_s100_scaled.csv (100 samples); everything
## smoke(Long), quick and short leave out the deep models where naive or guide-all mostly times out.
## fix-all-no-vars runs only for the 50x1-50x4 models of heart_attack, loan and mnist, in all modes.
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

ACTIONS=(run collect-stats plot-performance interpret)

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

## Writes the jobs of the mode to JOBS_TMP, one per line, tab-separated:
## <worst-case secs> <job id> <timeout per> <model> <dataset> <variant> <experiment> <strategies>
## <samples> <order: benchmark.width.depth.variant>
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

                        printf "%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%02d.%04d.%03d.%02d\n" \
                            $job_worst_secs "$job_id" $job_timeout_per "$model" "$dataset" \
                            $variant $experiment "$job_strategies" $n_samples \
                            $i $width $depth $vidx
                    done
                done
            done
        done
    done >"$JOBS_TMP.unsorted"

    sort -s -t$'\t' -k1,1nr "$JOBS_TMP.unsorted" >"$JOBS_TMP"
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
printf "Experiments: %s\n" "$EXPERIMENTS_SPEC"
printf "Variants:    %s\n" "$VARIANTS_SPEC"
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
        group = id[1] " " substr(id[2], 1, index(id[2], "x"))
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
## need their own (see run1.sh); without them, run1.sh would use those of the original dataset
function check_activations {
    local missing=()
    local model dataset name activations
    while IFS=$'\t' read -r model dataset; do
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

## One job of the run action
function run_job {
    local job_id="$1"
    local timeout_per=$2
    local model="$3"
    local dataset="$4"
    local variant=$5
    local experiment=$6
    local strategies="$7"

    printf "Running: %s\n" "$job_id"

    ## The unsat cores of interpolants (see UCORE_EXPERIMENTS in modes.sh) need OpenSMT
    local options=$OPTIONS
    [[ $experiment == ucore_* ]] && options+=" -V opensmt"

    VARIANT=$variant TIMEOUT_PER=$timeout_per OPTIONS=$options \
        "$BASH" "$SCRIPTS_DIR/run1.sh" "$model" "$dataset" "$strategies" $experiment

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

## Runs all jobs in one GNU parallel queue
function action_run {
    printf "Parallel:    %d jobs at once (%s%% of %d CPUs)\n\n" $SLOTS $CPU_PERCENTAGE $CPUS

    print_estimate
    check_activations || exit 1

    [[ $DRY_RUN == 1 ]] && {
        printf "DRY RUN - the longest jobs, in the order they would start:\n"
        cut -f2-3,9 "$JOBS_TMP" | head -20 | column -t
        (( $N_JOBS > 20 )) && printf "... (%d more)\n" $(( $N_JOBS - 20 ))
        exit 0
    }

    local log_dir=${LOG_DIR:-$DATA_DIR/logs/runme/$MODE}
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
#    ## (loan, obesity, 50x1-4, naive/guide-all/fix-all, on the _quick datasets, whatever the mode is)
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
    ## with its timeout per sample, which plot_fc_stats.py uses for the benchmark
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
    local tasks=$(mktemp)
    local job_id dataset experiment name model_id variant key
    local -A experiments
    local keys=()
    while IFS=$'\t' read -r job_id dataset experiment; do
        [[ $job_id =~ $STATS_FILTER ]] || continue
        IFS=/ read -r name model_id variant _ <<<"$job_id"
        key="$name/$model_id/$(dataset_dir $name "$dataset")/$variant"
        [[ -n ${experiments[$key]} ]] || keys+=("$key")
        experiments[$key]+="$experiment "
    done < <(sort -s -t$'\t' -k10,10 "$JOBS_TMP" | cut -f2,5,7)

    local idx
    for idx in ${!keys[@]}; do
        key=${keys[$idx]}
        IFS=/ read -r name model_id _ variant <<<"$key"
        printf "%d\t%s\t%s\t%s\t%s\t%s\n" $idx "$stats_dir/$name.txt" "$name $model_id $variant" \
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

## Plots the stats of collect-stats with plot_fc_stats.py
function action_plot_performance {
    local stats_dir=${STATS_DIR:-$PWD/stats}/$MODE
    local python=${PYTHON:-$(dirname "$DATA_DIR")/venv/bin/python}
    local output_dir=$DATA_DIR/output/$MODE
    local plot_cmd=("$python" "$TACAS_DIR/plot_fc_stats.py" "$stats_dir" "$output_dir")

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

    [[ $DRY_RUN == 1 ]] && {
        printf "DRY RUN - would run:\n  %s\n" "${plot_cmd[*]}"
        exit 0
    }

    "${plot_cmd[@]}" || {
        printf "\nPlotting failed.\n" >&2
        exit 1
    }

    printf "\nPlots:\n"
    ls -1 "$output_dir"/performance-plots_*.pdf | sed 's/^/  /'
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
esac
