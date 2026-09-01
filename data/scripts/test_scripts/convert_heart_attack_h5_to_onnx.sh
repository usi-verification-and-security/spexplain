#!/usr/bin/env bash
set -euo pipefail

# Convert all .h5 models in data/models/heart_attack to .onnx in-place.
#
# Usage:
#   ./data/scripts/test_scripts/convert_heart_attack_h5_to_onnx.sh [--overwrite]
#
# Env:
#   AUTO_INSTALL_PY_DEPS=1   Install missing Python deps automatically

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_DIR="${ROOT_DIR}/data/models/heart_attack"
CONVERTER="${ROOT_DIR}/python_scripts/convert_h5_to_onnx.py"
REQS="${ROOT_DIR}/python_scripts/requirements-h5-to-onnx.txt"
AUTO_INSTALL_PY_DEPS="${AUTO_INSTALL_PY_DEPS:-0}"

OVERWRITE_FLAG=""
if [[ "${1:-}" == "--overwrite" ]]; then
  OVERWRITE_FLAG="--overwrite"
fi

if ! python3 - <<'PY' >/dev/null 2>&1
import tensorflow, tf2onnx  # noqa: F401
PY
then
  if [[ "${AUTO_INSTALL_PY_DEPS}" == "1" ]]; then
    python3 -m pip install -r "${REQS}"
  else
    echo "Missing python deps for conversion." >&2
    echo "Install with: python3 -m pip install -r ${REQS}" >&2
    echo "Or rerun with AUTO_INSTALL_PY_DEPS=1" >&2
    exit 2
  fi
fi

python3 "${CONVERTER}" --dir "${MODEL_DIR}" ${OVERWRITE_FLAG}

