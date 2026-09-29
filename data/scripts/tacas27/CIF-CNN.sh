#!/bin/bash
## TACAS'27 experiments for the CIFAR-10 benchmark (cifar10), CNN-bench ONNX models.
## Refer to ./common-cnn for the shared driver and the supported env. variables.

NAME=cifar10
## The CIFAR-10 CSV keeps its historical location and prefix: data/datasets/cifar/cifar_*.csv
DATASET=${DATASET:-$(dirname "$(realpath "$0")")/../../datasets/cifar/cifar_s100_scaled.csv}
TIMEOUT_PER=${TIMEOUT_PER:-10m}

source "$(dirname "$(realpath "$0")")/common-cnn"
