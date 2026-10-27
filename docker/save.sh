#!/bin/bash

source $(dirname "$0")/lib

# ARCHIVE=${REPO}_${TAG}.tar.gz
# With the tag, so that the images of different platforms (e.g. latest-arm64) do not overwrite each other
ARCHIVE=${REPO}_${TAG}.tar

printf "Saving Docker image %s to %s ...\n" $IMAGE "$ARCHIVE"

# exec docker save $IMAGE | gzip >"$ARCHIVE"
exec docker save $IMAGE >"$ARCHIVE"
