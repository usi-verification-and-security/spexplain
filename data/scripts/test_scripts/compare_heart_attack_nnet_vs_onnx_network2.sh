#!/usr/bin/env bash
set -euo pipefail

# Compare heart_attack-50 in NNet vs ONNX/Network2 on the heart_attack_full dataset.
#
# Usage:
#   ./data/scripts/test_scripts/compare_heart_attack_nnet_vs_onnx_network2.sh [tol] [limit]
#
# Example:
#   ./data/scripts/test_scripts/compare_heart_attack_nnet_vs_onnx_network2.sh 5e-5 0

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
NNET_PATH="${ROOT_DIR}/data/models/heart_attack/heart_attack-50.nnet"
ONNX_PATH="${ROOT_DIR}/data/models/heart_attack/heart_attack-50.onnx"
DATASET_PATH="${ROOT_DIR}/data/datasets/heart_attack/heart_attack_full.csv"
BIN_PATH="${ROOT_DIR}/build-debug/compare-nnet-onnx"

TOL="${1:-5e-5}"
LIMIT="${2:-0}"

cd "${ROOT_DIR}"
make debug >/dev/null

"${BIN_PATH}" "${NNET_PATH}" "${ONNX_PATH}" "${DATASET_PATH}" "${TOL}" "${LIMIT}"

