#!/bin/bash
## TACAS'27 experiments for the CIFAR-10 benchmark (cifar), CNN-bench ONNX models.
## Refer to ./common-cnn for the shared driver and the supported env. variables.

NAME=cifar
## No DATASET override: the default resolves to data/datasets/cifar/cifar_s100_scaled.csv,
## which matches data/models/cifar/ as lib/run2's model-vs-dataset type check requires.
TIMEOUT_PER=${TIMEOUT_PER:-10m}

source "$(dirname "$(realpath "$0")")/common-cnn"
