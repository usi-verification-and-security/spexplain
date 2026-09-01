#!/usr/bin/env bash
set -euo pipefail

# Compare the *explanations* produced from an ONNX model (Network2 + OpenSMTVerifier2)
# against the explanations produced from the equivalent NNet model (Network + OpenSMTVerifier),
# for the same dataset and the same explanation strategy.
#
#   ONNX side : spexplain explain-onnx <model.onnx> <dataset.csv> <strategy>  -> mainExplainOnnx()
#   NNet side : spexplain explain      <model.nnet> <dataset.csv> <strategy>  -> mainExplain()
#
# Unlike compare_psi_onnx_vs_nnet.sh (which diffs the *encoding*), this script diffs the
# *output* of the explanation search itself, so it is sensitive to any behavioral differences
# between OpenSMTVerifier and OpenSMTVerifier2 (e.g. missing input domain bounds for ONNX).
#
# Usage:
#   ./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh --model <basename-without-extension> --dataset <csv>
#   ./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh --onnx <path.onnx> --nnet <path.nnet> --dataset <csv>
#
# Options:
#   --model <path>    Common path prefix; expands to <path>.onnx and <path>.nnet
#   --onnx <path>      Explicit path to the .onnx model
#   --nnet <path>      Explicit path to the .nnet model
#   --dataset <csv>    Dataset CSV shared by both runs (required)
#   --strategy <spec>  Explanation strategy spec forwarded to both runs (default: abductive)
#   --input-min <v1,v2,...>
#                      Per-feature input domain minimums, forwarded as `--input-min` to the
#                      ONNX side only (ONNX carries no input domain; .nnet stores it natively).
#   --input-max <v1,v2,...>
#                      Per-feature input domain maximums, forwarded as `--input-max` to the
#                      ONNX side only.
#   --outdir <dir>     Where to place the generated files (default: build-debug/explain-compare)
#   --keep             Do not wipe an existing output directory
#   --                 Everything after this is forwarded to both spexplain invocations
#   -h, --help         Show this help
#
# `abductive` is used by default because it produces one `(= xI v)` conjunct per fixed
# feature, which is trivial to compare quantitatively (which/how many features are fixed) --
# unlike `itp`, whose result is a generic linear formula that is not directly comparable
# feature-by-feature.
#
# Without real input bounds, Network2 (the ONNX side) silently falls back to a [0,1] domain
# for every feature, which makes the abductive/itp search trivially "succeed" after freeing
# almost any variable and produces drastically weaker explanations than the .nnet side. If
# neither --input-min nor --input-max is given and the model looks like the heart_attack
# model, this script supplies the known heart_attack feature bounds automatically.
#
# Example:
#   ./data/scripts/test_scripts/compare_explain_onnx_vs_nnet.sh \
#       --model ./data/models/heart_attack/heart_attack-50 \
#       --dataset ./data/datasets/heart_attack/heart_attack_quick.csv

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN="${ROOT_DIR}/build-debug/spexplain"

ONNX_PATH=""
NNET_PATH=""
DATASET=""
STRATEGY="abductive"
OUTDIR="${ROOT_DIR}/build-debug/explain-compare"
KEEP=0
INPUT_MIN=""
INPUT_MAX=""
EXTRA_ARGS=()

# Known per-feature (min,max) domain for the heart_attack dataset/model, since the ONNX
# format carries no input-range metadata at all. Order matches the 13 heart_attack features.
HEART_ATTACK_INPUT_MIN="29,0,0,94,126,0,0,71,0,0,0,0,0"
HEART_ATTACK_INPUT_MAX="77,1,3,200,594,1,2,202,1,6.2,2,4,3"

usage() { sed -n '3,50p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)     ONNX_PATH="$2.onnx"; NNET_PATH="$2.nnet"; shift 2 ;;
        --onnx)      ONNX_PATH="$2"; shift 2 ;;
        --nnet)      NNET_PATH="$2"; shift 2 ;;
        --dataset)   DATASET="$2"; shift 2 ;;
        --strategy)  STRATEGY="$2"; shift 2 ;;
        --input-min) INPUT_MIN="$2"; shift 2 ;;
        --input-max) INPUT_MAX="$2"; shift 2 ;;
        --outdir)    OUTDIR="$2"; shift 2 ;;
        --keep)      KEEP=1; shift ;;
        --)          shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help)   usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${ONNX_PATH}" || -z "${NNET_PATH}" || -z "${DATASET}" ]]; then
    echo "ERROR: an ONNX model, an NNet model, and a dataset are all required." >&2
    usage >&2
    exit 2
fi

for f in "${ONNX_PATH}" "${NNET_PATH}" "${DATASET}"; do
    [[ -f "${f}" ]] || { echo "ERROR: file not found: ${f}" >&2; exit 2; }
done

# Auto-supply the known heart_attack bounds unless the caller already gave explicit ones.
if [[ -z "${INPUT_MIN}" && -z "${INPUT_MAX}" && "${ONNX_PATH}" == *heart_attack* ]]; then
    echo "==> Detected heart_attack model: defaulting --input-min/--input-max to its known feature bounds"
    INPUT_MIN="${HEART_ATTACK_INPUT_MIN}"
    INPUT_MAX="${HEART_ATTACK_INPUT_MAX}"
fi

ONNX_EXTRA_ARGS=(${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"})
if [[ -n "${INPUT_MIN}" ]]; then ONNX_EXTRA_ARGS+=(--input-min "${INPUT_MIN}"); fi
if [[ -n "${INPUT_MAX}" ]]; then ONNX_EXTRA_ARGS+=(--input-max "${INPUT_MAX}"); fi

ONNX_PATH="$(cd "$(dirname "${ONNX_PATH}")" && pwd)/$(basename "${ONNX_PATH}")"
NNET_PATH="$(cd "$(dirname "${NNET_PATH}")" && pwd)/$(basename "${NNET_PATH}")"
DATASET="$(cd "$(dirname "${DATASET}")" && pwd)/$(basename "${DATASET}")"

cd "${ROOT_DIR}"
echo "==> Building (make debug)"
make debug >/dev/null

if [[ "${KEEP}" -eq 0 ]]; then rm -rf "${OUTDIR}"; fi
mkdir -p "${OUTDIR}"

ONNX_PHI="${OUTDIR}/onnx.phi.txt"
NNET_PHI="${OUTDIR}/nnet.phi.txt"

echo "==> ONNX    : ${ONNX_PATH}"
echo "==> NNet    : ${NNET_PATH}"
echo "==> Dataset : ${DATASET}"
echo "==> Strategy: ${STRATEGY}"
echo "==> Outdir  : ${OUTDIR}"
if [[ -n "${INPUT_MIN}" || -n "${INPUT_MAX}" ]]; then
    echo "==> ONNX input-min: ${INPUT_MIN:-<unset>}"
    echo "==> ONNX input-max: ${INPUT_MAX:-<unset>}"
fi
echo

echo "==> Explaining NNet model (explain / mainExplain)"
"${BIN}" explain "${NNET_PATH}" "${DATASET}" "${STRATEGY}" -e "${NNET_PHI}" \
    -s "${OUTDIR}/nnet.stats.txt" --output-times "${OUTDIR}/nnet.times.txt" \
    ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} > "${OUTDIR}/nnet.log" 2>&1 \
    || { echo "FAILED: explain (nnet, see ${OUTDIR}/nnet.log)" >&2; tail -20 "${OUTDIR}/nnet.log" >&2; exit 3; }

echo "==> Explaining ONNX model (explain-onnx / mainExplainOnnx)"
"${BIN}" explain-onnx "${ONNX_PATH}" "${DATASET}" "${STRATEGY}" -e "${ONNX_PHI}" \
    -s "${OUTDIR}/onnx.stats.txt" --output-times "${OUTDIR}/onnx.times.txt" \
    ${ONNX_EXTRA_ARGS[@]+"${ONNX_EXTRA_ARGS[@]}"} > "${OUTDIR}/onnx.log" 2>&1 \
    || { echo "FAILED: explain-onnx (see ${OUTDIR}/onnx.log)" >&2; tail -20 "${OUTDIR}/onnx.log" >&2; exit 4; }
echo

echo "==> Comparing explanations"
set +e
python3 "${SCRIPT_DIR}/phi_diff.py" "${NNET_PHI}" "${ONNX_PHI}"
rc=$?
set -e

echo
echo "==> Aggregate statistics (from the -s stats files)"
python3 "${SCRIPT_DIR}/phi_stats.py" \
    --label nnet --stats "${OUTDIR}/nnet.stats.txt" --times "${OUTDIR}/nnet.times.txt" \
    --label onnx --stats "${OUTDIR}/onnx.stats.txt" --times "${OUTDIR}/onnx.times.txt"

echo
echo "NNet explanations: ${NNET_PHI}"
echo "ONNX explanations: ${ONNX_PHI}"
exit "${rc}"
