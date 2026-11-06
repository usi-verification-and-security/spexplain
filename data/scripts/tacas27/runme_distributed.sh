#!/usr/bin/env bash
## The distributed run of runme.sh on a cluster: the jobs of the action and the mode as batches of whole models,
## each batch one submission through WRAPPER (one node), all running at the same time.
## The models are packed into the batches so that they fill the nodes, up to BATCH_TIME_TARGET of NODE_TIME_LIMIT
## (more only with a model whose worst case alone is over it), and each batch gets the time limit of its
## worst case (every explanation times out), at most NODE_TIME_LIMIT.
##
## USAGE: WRAPPER=<script> ./runme_distributed.sh <action> <mode> [--resume] [-n] [-h]
##
## Actions: run, run-verix-plus, run-cora (see runme.sh)
##
## A batch is runme.sh with FILTER='^(<name>/<model id>|...)/', e.g. '^(gtsrb/50x1|mnist/NAB1)/', so it has its own
## log directory (see LOG_DIR in runme.sh), and the time limit TIME (HH:MM:SS) for the wrapper (e.g. sbatch -t).
## Its worst case is the longest job if it has at most as many jobs as slots, otherwise the total of the jobs over
## the slots plus the longest job (it holds whatever the actual run times), plus TIME_MARGIN.
## The jobs whose worst case alone exceeds NODE_TIME_LIMIT (e.g. the ucore jobs of full) are listed by id,
## and their batches capped to it: they may be killed with the batch, and --resume restarts them from scratch;
## also those that exceed it only with TIME_MARGIN are reported.
## The batches go to the index data/logs/runme/<mode>/distributed-<action>.txt (also printed).
##
## Options:
##   --resume  resubmit only the batches of the index with remaining jobs, each with --resume
##             (with NPROC and CPU_PERCENTAGE of the index)
##   -n        dry run - only print the batches and their time limits;
##             with --resume, the finished, failed and remaining jobs of each batch
##   -h        display this help and exit
##
## Env. variables:
##   WRAPPER           (required) the script that submits a batch, e.g. ~/slurm/sbatch.sh,
##                     with TIME as its time limit; on a single machine, use runme.sh instead
##   NPROC             the hardware threads of a node, as nproc prints there (default: 256)
##   CPU_PERCENTAGE    percentage of NPROC to use (default: 50)
##   NODE_TIME_LIMIT   the maximum time limit of a batch, e.g. 24h or 30m (default: 24h)
##   TIME_MARGIN       percentage added to the worst case of a batch (default: 10)
##   BATCH_TIME_TARGET percentage of NODE_TIME_LIMIT that a batch takes, more only for a model whose worst case
##                     alone is over it; lower makes more, shorter batches running at the same time (default: 25)
##   FILTER            regex over the job ids: only the models with a matching job (all their jobs run)
## The others, e.g. ENV or DEBUG of the wrapper, pass through to the batches; LOG_DIR is not allowed.

SCRIPT=$(realpath "$0")
TACAS_DIR=$(dirname "$SCRIPT")
DATA_DIR=$(dirname "$(dirname "$TACAS_DIR")")
RUNME="$TACAS_DIR/runme.sh"

function usage {
    ## The header comment, up to the first line that is not a '##' comment
    awk 'NR > 1 && !/^##/ { exit } /^##/ { sub(/^## ?/, ""); print }' "$SCRIPT"

    [[ -n $1 ]] && exit $1
}

DISTRIBUTED_ACTIONS=(run run-verix-plus run-cora)

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

[[ " ${DISTRIBUTED_ACTIONS[*]} " == *" $ACTION "* ]] || {
    printf "Unrecognized action: %s (expected one of: %s)\n\n" "$ACTION" "${DISTRIBUTED_ACTIONS[*]}" >&2
    usage 1 >&2
}

[[ -n $WRAPPER ]] || {
    printf "WRAPPER is not set: the batches are to be submitted on a cluster; on a single machine, use runme.sh\n" >&2
    exit 1
}

[[ -z $LOG_DIR ]] || {
    printf "LOG_DIR is not allowed: each batch has its own log directory, by its FILTER\n" >&2
    exit 1
}

NPROC=${NPROC:-256}
export CPU_PERCENTAGE=${CPU_PERCENTAGE:-50}
NODE_TIME_LIMIT=${NODE_TIME_LIMIT:-24h}
TIME_MARGIN=${TIME_MARGIN:-10}
BATCH_TIME_TARGET=${BATCH_TIME_TARGET:-25}
## The batches compute the same slots (see runme.sh)
export CPUS=$NPROC
SLOTS=$(( ($NPROC*$CPU_PERCENTAGE + 50)/100 ))
(( $SLOTS >= 1 )) || SLOTS=1

function to_secs {
    local t=$1
    case $t in
    *h) printf "%d" $(( ${t%h}*3600 )) ;;
    *m) printf "%d" $(( ${t%m}*60 )) ;;
    *s) printf "%d" ${t%s} ;;
    *)  printf "%d" $t ;;
    esac
}

LIMIT_SECS=$(to_secs $NODE_TIME_LIMIT)
TARGET_SECS=$(( $LIMIT_SECS*$BATCH_TIME_TARGET/100 ))

INDEX="$DATA_DIR/logs/runme/$MODE/distributed-$ACTION.txt"

## <index> -> the batches, without the comments: <batch> <time> <jobs/slots> <worst case> <filter>
function read_batches {
    grep -v '^#' "$1"
}

## Packs the models of the jobs into the batches; stdin: <worst secs> <job id> of all jobs, the selected models
## in MODELS (one per line); stdout: the index
function pack {
    awk -F'\t' -v slots=$SLOTS -v limit=$LIMIT_SECS -v target=$TARGET_SECS -v margin=$TIME_MARGIN -v models_file=<(printf "%s\n" "$MODELS") '
    function bound(n, sum, max) { return (n <= slots ? max : sum/slots + max) * (1 + margin/100) }
    ## Rounded up to 15 minutes, at most the limit
    function time_limit(b) {
        b = int((b + 899)/900)*900
        if (b > limit) b = limit
        return sprintf("%02d:%02d:%02d", int(b/3600), int(b%3600/60), b%60)
    }
    function hours(s) { return sprintf("%.1fh", s/3600) }
    function time_secs(t,   a) { split(t, a, ":"); return a[1]*3600 + a[2]*60 + a[3] }
    BEGIN { while ((getline m <models_file) > 0) if (m != "") selected[m] = 1 }
    {
        split($2, id, "/")
        m = id[1] "/" id[2]
        if (!(m in selected)) next
        if (!(m in n)) models[++n_models] = m
        n[m]++
        sum[m] += $1
        if ($1 > max[m]) max[m] = $1

        if ($1 > limit) over_jobs = over_jobs " " $2 " (" hours($1) ")"
        else if ($1*(1 + margin/100) > limit) {
            n_margin_jobs++
            if ($1 > margin_max) margin_max = $1
            if (!(m in margin_models)) { margin_models[m] = 1; margin_list = margin_list " " m }
        }
    }
    END {
        ## Decreasing by the longest job, then by the total (insertion sort, a few hundred models)
        for (i = 2; i <= n_models; i++) {
            m = models[i]
            for (j = i - 1; j >= 1 && (max[models[j]] < max[m] || (max[models[j]] == max[m] && sum[models[j]] < sum[m])); j--)
                models[j+1] = models[j]
            models[j+1] = m
        }

        ## First fit: a batch takes a model if its worst case stays within the target, or does not grow
        n_batches = 0
        for (i = 1; i <= n_models; i++) {
            m = models[i]
            for (b = 1; b <= n_batches; b++) {
                old = bound(bn[b], bsum[b], bmax[b])
                new = bound(bn[b] + n[m], bsum[b] + sum[m], (max[m] > bmax[b] ? max[m] : bmax[b]))
                if (new <= (old > target ? old : target)) break
            }
            if (b > n_batches) n_batches = b
            bn[b] += n[m]
            bsum[b] += sum[m]
            if (max[m] > bmax[b]) bmax[b] = max[m]
            bmodels[b] = bmodels[b] (bmodels[b] == "" ? "" : "|") m
        }

        printf "# runme_distributed.sh %s %s: NPROC=%d CPU_PERCENTAGE=%d (%d slots), NODE_TIME_LIMIT=%s, BATCH_TIME_TARGET=%d%%, TIME_MARGIN=%d%%\n", \
            action, mode, nproc, cpu_percentage, slots, node_time_limit, batch_time_target, margin
        printf "# batch\ttime\tjobs/slots\tworst case\tfilter\n"
        for (b = 1; b <= n_batches; b++) {
            t = time_limit(bound(bn[b], bsum[b], bmax[b]))
            worst = hours(bound(bn[b], bsum[b], bmax[b]))
            if (bound(bn[b], bsum[b], bmax[b]) > limit) worst = worst ", capped to " t
            printf "%d\t%s\t%d/%d\t%s\t^(%s)/\n", b, t, bn[b], slots, worst, bmodels[b]
            node_secs += time_secs(t)
            n_jobs += bn[b]
        }
        ## As against a batch per model
        for (i = 1; i <= n_models; i++) {
            m = models[i]
            model_node_secs += time_secs(time_limit(bound(n[m], sum[m], max[m])))
        }
        printf "# total: %d batches, %d jobs, %d models; reserved %.0f node-hours (a batch per model: %.0f)\n", \
            n_batches, n_jobs, n_models, node_secs/3600, model_node_secs/3600
        if (over_jobs != "")
            printf "# WARNING: these jobs cannot finish within NODE_TIME_LIMIT if every explanation times out (they are killed with their batch, and --resume restarts them from scratch):%s\n", over_jobs
        if (n_margin_jobs)
            printf "# WARNING: %d jobs fit NODE_TIME_LIMIT only without TIME_MARGIN (up to %s), of the models:%s\n", \
                n_margin_jobs, hours(margin_max), margin_list
    }' action=$ACTION mode=$MODE nproc=$NPROC cpu_percentage=$CPU_PERCENTAGE node_time_limit=$NODE_TIME_LIMIT batch_time_target=$BATCH_TIME_TARGET -
}

## <filter> -> runme.sh --resume -n of the batch, locally
function batch_state {
    env -u WRAPPER FILTER="$1" "$RUNME" "$ACTION" "$MODE" --resume -n </dev/null 2>&1
}

## <filter> <time> -> submits the batch through WRAPPER (by runme.sh)
function submit_batch {
    local resume_opt=()
    (( $RESUME )) && resume_opt=(--resume)
    TIME=$2 FILTER="$1" "$RUNME" "$ACTION" "$MODE" "${resume_opt[@]}" </dev/null
}

if (( $RESUME )); then
    [[ -r $INDEX ]] || {
        printf "Cannot resume: no index of the batches %s\n" "$INDEX" >&2
        exit 1
    }
    printf "Index: %s\n\n" "$INDEX"
    cat "$INDEX"
    printf "\n"

    ## The batches resume with the slots that they were packed for
    NPROC=$(sed -n '1s/.* NPROC=\([0-9]*\) .*/\1/p' "$INDEX")
    export CPU_PERCENTAGE=$(sed -n '1s/.* CPU_PERCENTAGE=\([0-9]*\) .*/\1/p' "$INDEX")
    export CPUS=$NPROC

    ## Only the batches with remaining jobs
    to_submit=()
    n_failed_total=0
    n_remaining_total=0
    while IFS=$'\t' read -r batch time jobs worst filter; do
        state=$(batch_state "$filter") || {
            printf "Batch %d: %s\n" $batch "$(tail -n 1 <<<"$state")" >&2
            exit 1
        }
        n_failed=$(awk '/^Failed:/ { print $2 }' <<<"$state")
        n_remaining=$(awk '/^Remaining:/ { print $2 }' <<<"$state")
        (( n_failed_total += n_failed, n_remaining_total += n_remaining ))
        printf "Batch %d: %d failed, %d remaining\n" $batch $n_failed $n_remaining
        [[ $DRY_RUN == 1 ]] && (( $n_failed + $n_remaining )) &&
            sed -n '/^Failed jobs:/,$p' <<<"$state" | sed '/./s/^/  /'
        (( $n_remaining )) && to_submit+=("$batch")
    done < <(read_batches "$INDEX")

    printf "\nTotal: %d failed, %d remaining; batches to resubmit: %s\n" \
        $n_failed_total $n_remaining_total "${to_submit[*]:-none}"
    (( $n_failed_total )) && printf "The failed jobs are not rerun by --resume: rerun them in runme.sh by FILTER of their ids, e.g. '^(<id>|<id>)$'\n" >&2
    [[ $DRY_RUN == 1 ]] && exit 0
    (( ${#to_submit[@]} )) || exit 0
else
    ## All jobs of the mode, also those of the selected models that do not match FILTER
    jobs=$(env -u WRAPPER -u FILTER "$RUNME" "$ACTION" "$MODE" -l) || exit $?

    MODELS=$(while IFS=$'\t' read -r _ job_id; do
        [[ -n $FILTER && ! $job_id =~ $FILTER ]] && continue
        cut -d/ -f1-2 <<<"$job_id"
    done <<<"$jobs" | sort -u)
    [[ -n $MODELS ]] || {
        printf "Nothing to run.\n" >&2
        exit 1
    }

    index=$(pack <<<"$jobs")
    printf "%s\n\n" "$index"
    [[ $DRY_RUN == 1 ]] && exit 0

    mkdir -p "$(dirname "$INDEX")" || exit 1
    [[ -e $INDEX ]] && mv "$INDEX" "${INDEX%.txt}.$(date +%Y%m%d_%H%M%S).txt"
    printf "%s\n" "$index" >"$INDEX"
    printf "Index: %s\n\n" "$INDEX"
fi

n_failed_submissions=0
while IFS=$'\t' read -r batch time jobs worst filter; do
    (( $RESUME )) && [[ " ${to_submit[*]} " != *" $batch "* ]] && continue

    printf "Batch %d (%s, %s jobs): %s\n" $batch $time "$jobs" "$filter"
    submit_batch "$filter" $time || {
        printf "Batch %d: the submission failed\n" $batch >&2
        (( ++n_failed_submissions ))
    }
done < <(read_batches "$INDEX")

(( $n_failed_submissions )) && {
    printf "\nFailed submissions: %d\n" $n_failed_submissions >&2
    exit 1
}
exit 0
