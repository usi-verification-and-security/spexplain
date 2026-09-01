#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./data/scripts/test_scripts/test_network2_cnn_max_mnist2.sh [num_tests] [tolerance]
#
# Example:
#   ./data/scripts/test_scripts/test_network2_cnn_max_mnist2.sh 50 5e-5
#
# Optional environment variables:
#   AUTO_INSTALL_PY_DEPS=0|1   Install python deps automatically when missing
#   SEED=<int>                 Seed for deterministic random test generation

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH_A="${ROOT_DIR}/data/models/onnx/cnn_max_mnist2.onnx"
MODEL_PATH_B="${ROOT_DIR}/data/models/onnx/cnn_max_mninst2.onnx"
BIN_PATH="${ROOT_DIR}/build-debug/onnx-eval"
PY_COMPARE="${ROOT_DIR}/python_scripts/compare_network2_vs_onnx.py"
PY_REQS="${ROOT_DIR}/python_scripts/requirements-onnx-test.txt"

NUM_TESTS="${1:-50}"
TOL="${2:-5e-5}"
SEED="${SEED:-7}"
AUTO_INSTALL_PY_DEPS="${AUTO_INSTALL_PY_DEPS:-0}"

if [[ -f "${MODEL_PATH_A}" ]]; then
  MODEL_PATH="${MODEL_PATH_A}"
elif [[ -f "${MODEL_PATH_B}" ]]; then
  MODEL_PATH="${MODEL_PATH_B}"
else
  echo "Model not found. Checked:" >&2
  echo "  ${MODEL_PATH_A}" >&2
  echo "  ${MODEL_PATH_B}" >&2
  exit 2
fi

cd "${ROOT_DIR}"
make debug >/dev/null

if ! python3 - <<'PY' >/dev/null 2>&1
import numpy, onnxruntime  # noqa: F401
PY
then
  if [[ "${AUTO_INSTALL_PY_DEPS}" == "1" ]]; then
    python3 -m pip install -r "${PY_REQS}" >/dev/null
  else
    echo "Missing python deps for ONNX comparison." >&2
    echo "Install with: python3 -m pip install -r ${PY_REQS}" >&2
    echo "Or rerun with AUTO_INSTALL_PY_DEPS=1" >&2
    exit 2
  fi
fi

python3 "${PY_COMPARE}" "${MODEL_PATH}" "${BIN_PATH}" --num-tests "${NUM_TESTS}" --tol "${TOL}" --seed "${SEED}"


