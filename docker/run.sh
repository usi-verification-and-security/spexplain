#!/bin/bash

source $(dirname "$0")/lib

ARGS=()
[[ -n $PLATFORM ]] && ARGS+=(--platform "$PLATFORM")
# A host directory to inspect the outputs, mounted at the output directory of the image
# (its variable OUTPUT, e.g. ~/spexplain/data/output), or at /output if the image does not set it
[[ -n $OUTPUT ]] && {
    mkdir -p "$OUTPUT" || exit $?
    output=$(docker image inspect --format '{{range .Config.Env}}{{println .}}{{end}}' $IMAGE | sed -n 's/^OUTPUT=//p')
    ARGS+=(-v "$(realpath "$OUTPUT"):${output:-/output}")
}

exec docker run -it "${ARGS[@]}" $IMAGE
