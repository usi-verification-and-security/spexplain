#!/bin/bash
## TACAS'27 experiments for the GTSRB benchmark (gtsrb), CNN-bench ONNX models.
## Refer to ./common-cnn for the shared driver and the supported env. variables.

NAME=gtsrb
TIMEOUT_PER=${TIMEOUT_PER:-14m}

source "$(dirname "$(realpath "$0")")/common-cnn"
