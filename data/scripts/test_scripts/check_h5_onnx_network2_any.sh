#!/usr/bin/env bash
set -euo pipefail

# Compare outputs of .h5, .onnx, and Network2 for any dataset + matching model directory.
#
# Usage:
#   ./data/scripts/test_scripts/check_h5_onnx_network2_any.sh \
#       --models-dir <dir_with_h5> \
#       --dataset <csv_path> \
#       [--onnx-dir <dir_with_onnx>] \
#       [--label-col <name>|--all-columns] \
#       [--pattern <glob>] \
#       [--tol <float>] \
#       [--limit <int>]
#
# Examples:
#   ./data/scripts/test_scripts/check_h5_onnx_network2_any.sh \
#       --models-dir ./data/models/heart_attack \
#       --dataset ./data/datasets/heart_attack/heart_attack_full.csv \
#       --label-col output --tol 5e-5
#
#   ./data/scripts/test_scripts/check_h5_onnx_network2_any.sh \
#       --models-dir ./data/models/heart_attack \
#       --dataset ./data/datasets/heart_attack/heart_attack_full.csv \
#       --pattern 'heart_attack-50.h5' --limit 20
#
# Env:
#   AUTO_INSTALL_PY_DEPS=1   Install missing Python deps automatically

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ONNX_EVAL="${ROOT_DIR}/build-debug/onnx-eval"
PY_SCRIPT="${ROOT_DIR}/python_scripts/compare_h5_onnx_network2.py"
REQS="${ROOT_DIR}/python_scripts/requirements-heart-attack-compare.txt"
AUTO_INSTALL_PY_DEPS="${AUTO_INSTALL_PY_DEPS:-0}"

MODELS_DIR=""
ONNX_DIR=""
DATASET=""
LABEL_COL="output"
USE_ALL_COLUMNS=0
PATTERN="*.h5"
TOL="5e-5"
LIMIT="0"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --models-dir) MODELS_DIR="$2"; shift 2 ;;
    --onnx-dir) ONNX_DIR="$2"; shift 2 ;;
    --dataset) DATASET="$2"; shift 2 ;;
    --label-col) LABEL_COL="$2"; shift 2 ;;
    --all-columns) USE_ALL_COLUMNS=1; shift ;;
    --pattern) PATTERN="$2"; shift 2 ;;
    --tol) TOL="$2"; shift 2 ;;
    --limit) LIMIT="$2"; shift 2 ;;
    --help|-h)
      sed -n '1,80p' "$0" | sed 's/^# \{0,1\}//' | sed '/^!/d'
      exit 0
      ;;
    *)
      echo "Unknown option: $1" >&2
      exit 1
      ;;
  esac
done

if [[ -z "${MODELS_DIR}" || -z "${DATASET}" ]]; then
  echo "Error: --models-dir and --dataset are required." >&2
  exit 1
fi

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

CMD=(
  python3 "${PY_SCRIPT}"
  --models-dir "${MODELS_DIR}"
  --dataset "${DATASET}"
  --onnx-eval "${ONNX_EVAL}"
  --pattern "${PATTERN}"
  --tol "${TOL}"
  --limit "${LIMIT}"
)

if [[ -n "${ONNX_DIR}" ]]; then
  CMD+=(--onnx-dir "${ONNX_DIR}")
fi

if [[ "${USE_ALL_COLUMNS}" == "1" ]]; then
  CMD+=(--all-columns)
else
  CMD+=(--label-col "${LABEL_COL}")
fi

"${CMD[@]}"

