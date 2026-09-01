#!/usr/bin/env bash
set -euo pipefail

# Unified Network2 vs ONNX Runtime comparison for any single-input/single-output ONNX model.
#
# Usage:
#   ./data/scripts/test_scripts/test_network2_any.sh --model <onnx_path> [options]
#
# Options:
#   --model  <path>     Path to the ONNX file (required)
#   --tests  <int>      Number of test inputs (default: 50)
#   --tol    <float>    Absolute tolerance for comparison (default: 5e-5)
#   --seed   <int>      RNG seed for deterministic input generation (default: 7)
#   --help              Show this message and exit
#
# Environment variables:
#   AUTO_INSTALL_PY_DEPS=1   Automatically install Python deps if missing
#
# Examples:
#   ./data/scripts/test_scripts/test_network2_any.sh --model ./data/models/onnx/fc1.onnx --tol 1e-5
#   ./data/scripts/test_scripts/test_network2_any.sh --model ./data/models/onnx/conv.onnx --tests 100
#   ./data/scripts/test_scripts/test_network2_any.sh --model ./data/models/onnx/cnn_max_mninst2.onnx --tol 5e-5

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
BIN_PATH="${ROOT_DIR}/build-debug/onnx-eval"
PY_COMPARE="${ROOT_DIR}/python_scripts/compare_network2_vs_onnx.py"
PY_REQS="${ROOT_DIR}/python_scripts/requirements-onnx-test.txt"

MODEL_PATH=""
NUM_TESTS=50
TOL="5e-5"
SEED="${SEED:-7}"
AUTO_INSTALL_PY_DEPS="${AUTO_INSTALL_PY_DEPS:-0}"

usage() {
  grep '^#' "$0" | grep -v '^#!/' | sed 's/^# \{0,1\}//'
  exit 0
}
while [[ $# -gt 0 ]]; do
  case "$1" in
    --model)  MODEL_PATH="$2"; shift 2 ;;
    --tests)  NUM_TESTS="$2"; shift 2 ;;
    --tol)    TOL="$2"; shift 2 ;;
    --seed)   SEED="$2"; shift 2 ;;
    --help|-h) usage ;;
    *)
      echo "Unknown option: $1" >&2
      echo "Run with --help for usage." >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODEL_PATH}" ]]; then
  echo "Error: --model <onnx_path> is required." >&2
  echo "Run with --help for usage." >&2
  exit 1
fi

if [[ ! -f "${MODEL_PATH}" ]]; then
  echo "Error: Model file not found: ${MODEL_PATH}" >&2
  exit 2
fi

cd "${ROOT_DIR}"

# Print effective config
echo "Model:   ${MODEL_PATH}"
echo "Tests:   ${NUM_TESTS}"
echo "Tol:     ${TOL}"
echo "Seed:    ${SEED}"
echo ""

echo "Building Network2 binary..."
make debug >/dev/null

if ! python3 - <<'PY' >/dev/null 2>&1
import numpy, onnxruntime  # noqa: F401
PY
then
  if [[ "${AUTO_INSTALL_PY_DEPS}" == "1" ]]; then
    echo "Installing Python deps..."
    python3 -m pip install -r "${PY_REQS}" >/dev/null
  else
    echo "Missing python deps for ONNX comparison." >&2
    echo "Install with: python3 -m pip install -r ${PY_REQS}" >&2
    echo "Or rerun with AUTO_INSTALL_PY_DEPS=1" >&2
    exit 2
  fi
fi

python3 "${PY_COMPARE}" "${MODEL_PATH}" "${BIN_PATH}" \
  --num-tests "${NUM_TESTS}" \
  --tol "${TOL}" \
  --seed "${SEED}"



