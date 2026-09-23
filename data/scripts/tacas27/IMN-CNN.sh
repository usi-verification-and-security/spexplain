#!/bin/bash
## TACAS'27 experiments for the Imagenette-64 benchmark (imagenette), CNN-bench ONNX models.
## Refer to ./common-cnn for the shared driver and the supported env. variables.

NAME=imagenette
TIMEOUT_PER=${TIMEOUT_PER:-1h}
## Imagenette has no *_s100_scaled.csv; use the 100 held-out test images instead
DATASET_BASE=${DATASET_BASE:-imagenette_test100}

source "$(dirname "$(realpath "$0")")/common-cnn"
