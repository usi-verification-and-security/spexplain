#!/usr/bin/env bash
set -euo pipefail

# Compare outputs of heart_attack .h5, .onnx, and Network2 on the CSV dataset.
#
# Usage:
#   ./data/scripts/test_scripts/check_heart_attack_h5_onnx_network2.sh [tolerance] [limit]
#
# Examples:
#   ./data/scripts/test_scripts/check_heart_attack_h5_onnx_network2.sh
#   ./data/scripts/test_scripts/check_heart_attack_h5_onnx_network2.sh 5e-5
#   ./data/scripts/test_scripts/check_heart_attack_h5_onnx_network2.sh 5e-5 20
#
# Env:
#   AUTO_INSTALL_PY_DEPS=1   Install missing Python deps automatically

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODELS_DIR="${ROOT_DIR}/data/models/heart_attack"
DATASET="${ROOT_DIR}/data/datasets/heart_attack/heart_attack_full.csv"
ONNX_EVAL="${ROOT_DIR}/build-debug/onnx-eval"
PY_SCRIPT="${ROOT_DIR}/python_scripts/compare_h5_onnx_network2.py"
REQS="${ROOT_DIR}/python_scripts/requirements-heart-attack-compare.txt"
AUTO_INSTALL_PY_DEPS="${AUTO_INSTALL_PY_DEPS:-0}"
TOL="${1:-5e-5}"
LIMIT="${2:-0}"

cd "${ROOT_DIR}"
make debug >/dev/null

if ! python3 - <<'PY' >/dev/null 2>&1
import numpy, onnxruntime, tensorflow  # noqa: F401
PY
then
  if [[ "${AUTO_INSTALL_PY_DEPS}" == "1" ]]; then
    python3 -m pip install -r "${REQS}"
  else
    echo "Missing python deps for comparison." >&2
    echo "Install with: python3 -m pip install -r ${REQS}" >&2
    echo "Or rerun with AUTO_INSTALL_PY_DEPS=1" >&2
    exit 2
  fi
fi

python3 "${PY_SCRIPT}" \
  --models-dir "${MODELS_DIR}" \
  --dataset "${DATASET}" \
  --onnx-eval "${ONNX_EVAL}" \
  --tol "${TOL}" \
  --limit "${LIMIT}"

