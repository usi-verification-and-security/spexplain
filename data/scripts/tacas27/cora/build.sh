#!/usr/bin/env bash
## Compiles CORA (cora_explain_dataset.m, see build_cora_explain.m) into the application that runme.sh run-cora
## runs on the MATLAB Runtime, as in the image docker/tacas27-cora (see its README.md).
## USAGE: ./build.sh <CORA dir> [<output dir>]
## <CORA dir>     CORA of the release CORA_GIT_TAG with docker/tacas27-cora/cora-explain.patch (the options
##                InputBounds and Plot of neuralNetwork/explain); if it does not exist, it is cloned and patched
## <output dir>   default: docker/tacas27-cora/cora_explain (the build context of the image)
## Requires MATLAB of the release of the image (R2026a, the same update as its MATLAB Runtime), with MATLAB Compiler
## and the products of CORA for neural networks (see docker/tacas27-cora/README.md).
##
## Env. variables:
##   CORA_GIT_TAG   the release of CORA (default: v2026.1.0)
##   MATLAB         MATLAB (default: matlab)

CORA_SCRIPTS_DIR=$(dirname "$(realpath "$0")")
ROOT_DIR=$(realpath "$CORA_SCRIPTS_DIR/../../../..")
DOCKER_DIR="$ROOT_DIR/docker/tacas27-cora"
PATCH="$DOCKER_DIR/cora-explain.patch"

CORA_URL=https://github.com/TUMcps/CORA.git
CORA_GIT_TAG=${CORA_GIT_TAG:-v2026.1.0}
MATLAB=${MATLAB:-matlab}

function usage {
    awk 'NR > 1 && !/^##/ { exit } /^##/ { sub(/^## ?/, ""); print }' "$0"

    [[ -n $1 ]] && exit $1
}

(( $# == 1 || $# == 2 )) || usage 1 >&2

CORA_DIR=$1
OUT_DIR=${2:-$DOCKER_DIR/cora_explain}

set -e

if [[ ! -e $CORA_DIR ]]; then
    printf "Cloning CORA %s into %s ...\n" $CORA_GIT_TAG "$CORA_DIR"
    git -c advice.detachedHead=false clone -q --depth 1 --branch $CORA_GIT_TAG $CORA_URL "$CORA_DIR"
    git -C "$CORA_DIR" apply --whitespace=nowarn "$PATCH"
else
    ## The release, with the patch applied
    [[ $(git -C "$CORA_DIR" rev-parse HEAD) == $(git -C "$CORA_DIR" rev-parse "$CORA_GIT_TAG^{commit}") ]] || {
        printf "CORA in %s is not at %s\n" "$CORA_DIR" $CORA_GIT_TAG >&2
        exit 1
    }
    git -C "$CORA_DIR" apply --whitespace=nowarn --check --reverse "$PATCH" 2>/dev/null || {
        printf "CORA in %s is without %s\n" "$CORA_DIR" "$PATCH" >&2
        exit 1
    }
fi
CORA_DIR=$(realpath "$CORA_DIR")
mkdir -p "$OUT_DIR"
OUT_DIR=$(realpath "$OUT_DIR")

## -batch would start a virtual display (Xvfb)
"$MATLAB" -nosplash -nodesktop -nojvm \
    -r "addpath(genpath('$CORA_DIR')); addpath('$CORA_SCRIPTS_DIR'); try, build_cora_explain('$CORA_DIR','$OUT_DIR'); catch e, disp(getReport(e)); exit(1); end; exit" </dev/null
