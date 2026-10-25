#!/bin/bash

source $(dirname "$0")/lib

ARGS=()
[[ -n $TARGET ]] && ARGS+=(--build-arg "TARGET=$TARGET")
# The branch or tag of the repository (if supported by the Dockerfile)
[[ -n $SPEXPLAIN_GIT_TAG ]] && ARGS+=(--build-arg "SPEXPLAIN_GIT_TAG=$SPEXPLAIN_GIT_TAG")
# E.g. linux/arm64
[[ -n $PLATFORM ]] && ARGS+=(--platform "$PLATFORM")

printf "Building Docker image %s located at %s ...\n" $IMAGE "$DIR"

exec docker build -t $IMAGE "${ARGS[@]}" "$DIR"
