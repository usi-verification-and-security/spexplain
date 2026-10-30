#!/bin/bash
# Runs all command blocks of tutorial/README.md in order, from the root of the repository (~/spexplain in Docker),
# with the outputs into $OUTPUT/tutorial ($OUTPUT: default data/output in Docker); each block is timed,
# the run stops at the first failure.
# The setup blocks (docker/run.sh, choosing OUTPUT) are skipped; 'ucore min' (about 4 minutes) only with SLOW=1;
# '--show' is removed (it opens windows) unless SHOW=1.
set -eu
ROOT=$(git rev-parse --show-toplevel 2>/dev/null || pwd)
cd "$ROOT"
TUTORIAL=${TUTORIAL:-tutorial/README.md}
[[ -n ${OUTPUT:-} ]] || OUTPUT=$(mktemp -d)
mkdir -p "$OUTPUT"
export OUTPUT

BLOCKS=$(mktemp -d)
awk -v dir="$BLOCKS" '
    /^```/ { if (inside) { inside = 0; close(f) } else { inside = 1; n++; f = sprintf("%s/%03d.sh", dir, n) } next }
    inside { print > f }
' "$TUTORIAL"

printf "Outputs: %s\n" "$OUTPUT/tutorial"
for f in "$BLOCKS"/*.sh; do
    if grep -q 'docker/run.sh\|^OUTPUT=\|^mkdir -p \$OUTPUT' "$f"; then
        printf "\n=== %s: SKIPPED (setup)\n" "$(basename "$f")"
        continue
    fi
    if grep -q 'ucore min' "$f" && [[ -z ${SLOW:-} ]]; then
        printf "\n=== %s: SKIPPED (slow, set SLOW=1)\n" "$(basename "$f")"
        continue
    fi
    [[ -n ${SHOW:-} ]] || sed -i 's/ --show$//' "$f"
    printf "\n=== %s\n" "$(basename "$f")"
    cat "$f"
    printf -- "---\n"
    start=$SECONDS
    # The variables (MODEL, DATASET, ...) are kept between the blocks
    source "$f"
    printf -- "--- %ds\n" $((SECONDS - start))
done

printf "\n=== outputs\n"
ls -l "$OUTPUT/tutorial"
rm -rf "$BLOCKS"
echo "TUTORIAL PASSED"
