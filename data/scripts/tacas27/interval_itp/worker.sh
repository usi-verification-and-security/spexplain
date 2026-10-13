#!/usr/bin/env bash
## Computes the explanation of one sample - one GNU parallel job of ./common.
##
## USAGE: worker.sh <chunk_dir> <sample_idx>
##
## <chunk_dir>/job.env (written by ./common) holds the command and its options.
## Outputs <chunk_dir>/s<k>.{phi,stats,times,time,out,err}.txt and, last, s<k>.status:
##   ok        spexplain finished (its own timeout, if any, is recorded in its outputs)
##   timeout   killed by the outer timeout (Marabou does not respect spexplain's '-t')
##   skipped:  the input interval explanation is a timeout or empty (phase itp only)
##   error:<c> spexplain failed twice with exit code <c>
## A sample that already has a status is not run again.

chunk_dir=$1
k=$2

status_file="$chunk_dir/s$k.status"
[[ -s $status_file ]] && exit 0

source "$chunk_dir/job.env" || exit 1

f="$chunk_dir/s$k"

## Phase itp: nothing to generalize if the interval explanation is missing or empty
[[ -n $src_phi ]] && {
    line=$(sed -n "${k}p" "$src_phi")
    reason=
    [[ $line =~ ^\<null\> ]] && reason=interval-timeout
    [[ $line =~ ^\(and\)[[:space:]]*$ ]] && reason=interval-empty
    [[ -n $reason ]] && {
        echo "skipped:$reason" >"$status_file"
        printf "[skipped: %s] %s sample %d\n" "$reason" "$name" $k
        exit 0
    }
}

for attempt in 1 2; do
    rm -f "$f".{phi,stats,times,time,out,err}.txt
    { time $TIMEOUT_BIN -k 30 $HARD_TIMEOUT_S "$CMD" "$model" "$dataset_fn" "$strategies" "${opts[@]}" -i $k,$k \
        --output-explanations="$f.phi.txt" --output-stats="$f.stats.txt" --output-times="$f.times.txt" \
        >"$f.out.txt" 2>"$f.err.txt" ; } 2>"$f.time.txt"
    code=$?

    case $code in
        0)
            if [[ $(wc -l <"$f.phi.txt" 2>/dev/null | tr -d ' ') == 1 ]] && grep -q '^sample \[' "$f.stats.txt" 2>/dev/null; then
                status=ok
            else
                status="error:incomplete-output"
            fi
            ;;
        124|137)
            status=timeout
            ;;
        *)
            status="error:$code"
            ;;
    esac

    [[ $status == error:* ]] || break
done

secs=$(sed -n 's/^real[[:space:]]*//p' "$f.time.txt")
echo "$status" >"$status_file"
printf "[%s %s] %s sample %d\n" "$status" "$secs" "$name" $k

## A failed sample is recorded and does not stop the other jobs
exit 0
