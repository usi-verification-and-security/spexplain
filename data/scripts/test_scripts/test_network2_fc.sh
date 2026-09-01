#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./data/scripts/test_scripts/test_network2_fc.sh [comma_separated_input]
#
# Example:
#   ./data/scripts/test_scripts/test_network2_fc.sh 0.1,0.2,0.3,0.4
#
# Optional environment variables:
#   TOL=1e-5                  Absolute tolerance for output comparison
#   AUTO_INSTALL_PY_DEPS=0|1  Install python deps automatically when missing

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
MODEL_PATH="${ROOT_DIR}/data/models/onnx/fc1.onnx"
BIN_PATH="${ROOT_DIR}/build-debug/onnx-eval"
PY_HELPER="${ROOT_DIR}/python_scripts/run_onnx_direct.py"
PY_REQS="${ROOT_DIR}/python_scripts/requirements-onnx-test.txt"
TOL="${TOL:-1e-5}"
AUTO_INSTALL_PY_DEPS="${AUTO_INSTALL_PY_DEPS:-0}"

cd "${ROOT_DIR}"
make debug >/dev/null

if ! python3 - <<'PY' >/dev/null 2>&1
import numpy, onnx, onnxruntime  # noqa: F401
PY
then
  if [[ "${AUTO_INSTALL_PY_DEPS}" == "1" ]]; then
    python3 -m pip install -r "${PY_REQS}" >/dev/null
  else
    echo "Missing python deps for ONNX direct execution." >&2
    echo "Install with: python3 -m pip install -r ${PY_REQS}" >&2
    echo "Or rerun with AUTO_INSTALL_PY_DEPS=1" >&2
    exit 2
  fi
fi

NETWORK2_ARGS=("${BIN_PATH}" "${MODEL_PATH}")
ONNX_ARGS=(python3 "${PY_HELPER}" "${MODEL_PATH}")

if [[ $# -ge 1 ]]; then
  NETWORK2_ARGS+=("$1")
  ONNX_ARGS+=("--" "$1")
fi

N2_LOG="$(mktemp)"
ORT_LOG="$(mktemp)"
trap 'rm -f "${N2_LOG}" "${ORT_LOG}"' EXIT

echo "=== Network2 evaluation ==="
"${NETWORK2_ARGS[@]}" | tee "${N2_LOG}"

echo "=== ONNX Runtime direct evaluation ==="
"${ONNX_ARGS[@]}" | tee "${ORT_LOG}"

N2_OUTPUT="$(grep -E '^Output:' "${N2_LOG}" | tail -1 | sed 's/^Output:[[:space:]]*//')"
ORT_OUTPUT="$(grep -E '^Output:' "${ORT_LOG}" | tail -1 | sed 's/^Output:[[:space:]]*//')"

if [[ -z "${N2_OUTPUT}" || -z "${ORT_OUTPUT}" ]]; then
  echo "Failed to parse outputs from one of the runs." >&2
  exit 3
fi

python3 - "${N2_OUTPUT}" "${ORT_OUTPUT}" "${TOL}" <<'PY'
import ast
import math
import sys

n2 = ast.literal_eval(sys.argv[1])
ort = ast.literal_eval(sys.argv[2])
tol = float(sys.argv[3])

if len(n2) != len(ort):
    print(f"Mismatch: output lengths differ ({len(n2)} vs {len(ort)})", file=sys.stderr)
    sys.exit(4)

max_abs_diff = 0.0
for a, b in zip(n2, ort):
    d = abs(float(a) - float(b))
    max_abs_diff = max(max_abs_diff, d)

print(f"Max abs diff: {max_abs_diff:.10g}")
if max_abs_diff > tol:
    print(f"FAIL: max abs diff {max_abs_diff:.10g} > tolerance {tol:.10g}", file=sys.stderr)
    sys.exit(5)

print(f"PASS: outputs match within tolerance {tol:.10g}")
PY


