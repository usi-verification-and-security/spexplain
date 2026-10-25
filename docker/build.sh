#!/bin/bash

source $(dirname "$0")/lib

ARGS=()
# The build arguments of the Dockerfile (e.g. TARGET, or SPEXPLAIN_GIT_TAG) are passed from the variables
# of the same name if they are set, even to an empty value
for name in $(sed -n 's/^[[:space:]]*ARG[[:space:]]\+\([A-Za-z_][A-Za-z0-9_]*\).*/\1/p' "$DIR/Dockerfile"); do
    [[ -v $name ]] && ARGS+=(--build-arg "$name=${!name}")
done
# E.g. linux/arm64
[[ -n $PLATFORM ]] && ARGS+=(--platform "$PLATFORM")
# Additional arguments of `docker build`, split by words
[[ -n $DOCKER_BUILD_ARGS ]] && {
    read -r -a build_args <<<"$DOCKER_BUILD_ARGS"
    ARGS+=("${build_args[@]}")
}

printf "Building Docker image %s located at %s ...\n" $IMAGE "$DIR"

exec docker build -t $IMAGE "${ARGS[@]}" "$DIR"
