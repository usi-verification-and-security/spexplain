#!/bin/bash
## TACAS'27 experiments for the CIFAR-10 benchmark (cifar10), CNN-bench ONNX models.
## Refer to ./common-cnn for the shared driver and the supported env. variables.

NAME=cifar10
TIMEOUT_PER=${TIMEOUT_PER:-10m}

source "$(dirname "$(realpath "$0")")/common-cnn"
