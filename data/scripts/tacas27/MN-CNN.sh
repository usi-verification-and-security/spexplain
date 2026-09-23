#!/bin/bash
## TACAS'27 experiments for the MNIST benchmark (mnist), CNN-bench ONNX models.
## Refer to ./common-cnn for the shared driver and the supported env. variables.

NAME=mnist
TIMEOUT_PER=${TIMEOUT_PER:-10m}

source "$(dirname "$(realpath "$0")")/common-cnn"
