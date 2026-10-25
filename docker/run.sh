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
# Connect to the X11 display of the host, e.g. for --show
[[ -n $X11 ]] && {
    if [[ $(uname) == Darwin ]]; then
        # XQuartz, accessed via TCP from the VM of Docker Desktop
        ARGS+=(-e DISPLAY=host.docker.internal:0)
        printf "X11: XQuartz must be running with 'Allow connections from network clients' enabled\n" >&2
        printf "X11: (in its settings, then restarted) and the access allowed with: xhost +localhost\n" >&2
    else
        [[ -n $DISPLAY ]] || {
            printf "X11: DISPLAY is not set\n" >&2
            exit 1
        }
        ARGS+=(-e "DISPLAY=$DISPLAY" -v /tmp/.X11-unix:/tmp/.X11-unix:ro)
        if command -v xauth >/dev/null; then
            # The cookie of the display for any host name, since the container has its own one
            XAUTH=$(mktemp "${TMPDIR:-/tmp}/docker-xauth.XXXXXX") || exit $?
            xauth nlist "$DISPLAY" | sed 's/^..../ffff/' | xauth -f "$XAUTH" nmerge - || exit $?
            ARGS+=(-v "$XAUTH:/tmp/.docker.xauth:ro" -e XAUTHORITY=/tmp/.docker.xauth)
        else
            printf "X11: xauth not found, the access may need to be allowed, e.g. with: xhost +local:\n" >&2
        fi
    fi
}
# Additional arguments of `docker run`, split by words
[[ -n $DOCKER_RUN_ARGS ]] && {
    read -r -a run_args <<<"$DOCKER_RUN_ARGS"
    ARGS+=("${run_args[@]}")
}

[[ -z $XAUTH ]] && exec docker run -it "${ARGS[@]}" $IMAGE

docker run -it "${ARGS[@]}" $IMAGE
ret=$?
rm -f -- "$XAUTH"
exit $ret
