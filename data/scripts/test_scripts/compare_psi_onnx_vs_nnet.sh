#!/usr/bin/env bash
set -euo pipefail

# Compare the SMT-LIB2 encoding produced from an ONNX model (Network2 + OpenSMTVerifier2)
# against the encoding produced from the equivalent NNet model (Network + OpenSMTVerifier).
#
#   ONNX side : spexplain encode-onnx <model.onnx>   -> readEncodeONNX()
#   NNet side : spexplain dump-psi    <model.nnet>   -> mainDumpPsi()
#
# Both actions write psi_d.smt2 and psi_c<N>.smt2 into the *current working directory*,
# so each run is executed inside its own output folder.
#
# Usage:
#   ./data/scripts/test_scripts/compare_psi_onnx_vs_nnet.sh --model <basename-without-extension>
#   ./data/scripts/test_scripts/compare_psi_onnx_vs_nnet.sh --onnx <path.onnx> --nnet <path.nnet>
#
# Options:
#   --model <path>   Common path prefix; expands to <path>.onnx and <path>.nnet
#   --onnx <path>    Explicit path to the .onnx model
#   --nnet <path>    Explicit path to the .nnet model
#   --outdir <dir>   Where to place the generated files (default: build-debug/psi-compare)
#   --tol <t>        Relative tolerance below which a numeric difference counts as rounding
#                    noise rather than a real difference (default: 1e-3)
#   --ignore-bounds  Do not let the input-domain-bounds assertion (#0) affect the verdict.
#                    ONNX carries no input ranges, so Network2 falls back to its 0/1 defaults
#                    while the .nnet model stores the real per-feature min/max.
#   --textual-diff   Additionally produce plain `diff -u` output per file
#   --polish         Run data/scripts/polish_psi.sed on the outputs before diffing
#   --normalize      Canonicalise solver-internal .iteNNN_1 variable names in the files
#                    themselves (only affects --textual-diff; the structural comparison
#                    always canonicalises them internally)
#   --keep           Do not wipe an existing output directory
#   --               Everything after this is forwarded to both spexplain invocations
#   -h, --help       Show this help
#
# Examples:
#   ./data/scripts/test_scripts/compare_psi_onnx_vs_nnet.sh \
#       --model ./data/models/heart_attack/heart_attack-50
#
#   ./data/scripts/test_scripts/compare_psi_onnx_vs_nnet.sh \
#       --model ./data/models/heart_attack/heart_attack-50 --ignore-bounds

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../../.." && pwd)"
BIN="${ROOT_DIR}/build-debug/spexplain"
POLISH_SED="${ROOT_DIR}/data/scripts/polish_psi.sed"

ONNX_PATH=""
NNET_PATH=""
OUTDIR="${ROOT_DIR}/build-debug/psi-compare"
POLISH=0
NORMALIZE=0
TEXTUAL=0
IGNORE_BOUNDS=0
TOL=1e-3
KEEP=0
EXTRA_ARGS=()

usage() { sed -n '3,40p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; }

while [[ $# -gt 0 ]]; do
    case "$1" in
        --model)  ONNX_PATH="$2.onnx"; NNET_PATH="$2.nnet"; shift 2 ;;
        --onnx)   ONNX_PATH="$2"; shift 2 ;;
        --nnet)   NNET_PATH="$2"; shift 2 ;;
        --outdir) OUTDIR="$2"; shift 2 ;;
        --polish) POLISH=1; shift ;;
        --normalize) NORMALIZE=1; shift ;;
        --textual-diff) TEXTUAL=1; shift ;;
        --ignore-bounds) IGNORE_BOUNDS=1; shift ;;
        --tol) TOL="$2"; shift 2 ;;
        --keep)   KEEP=1; shift ;;
        --)       shift; EXTRA_ARGS=("$@"); break ;;
        -h|--help) usage; exit 0 ;;
        *) echo "Unknown option: $1" >&2; usage >&2; exit 2 ;;
    esac
done

if [[ -z "${ONNX_PATH}" || -z "${NNET_PATH}" ]]; then
    echo "ERROR: both an ONNX and an NNet model are required." >&2
    usage >&2
    exit 2
fi

for f in "${ONNX_PATH}" "${NNET_PATH}"; do
    [[ -f "${f}" ]] || { echo "ERROR: model not found: ${f}" >&2; exit 2; }
done

# The spexplain binary writes its output into the current working directory, so each action is
# executed inside its own folder; model paths must therefore be absolute.
ONNX_PATH="$(cd "$(dirname "${ONNX_PATH}")" && pwd)/$(basename "${ONNX_PATH}")"
NNET_PATH="$(cd "$(dirname "${NNET_PATH}")" && pwd)/$(basename "${NNET_PATH}")"

cd "${ROOT_DIR}"
echo "==> Building (make debug)"
make debug >/dev/null

ONNX_OUT="${OUTDIR}/onnx"
NNET_OUT="${OUTDIR}/nnet"

if [[ "${KEEP}" -eq 0 ]]; then rm -rf "${OUTDIR}"; fi
mkdir -p "${ONNX_OUT}" "${NNET_OUT}"

echo "==> ONNX  : ${ONNX_PATH}"
echo "==> NNet  : ${NNET_PATH}"
echo "==> Outdir: ${OUTDIR}"
echo

echo "==> Encoding ONNX model (encode-onnx / readEncodeONNX)"
( cd "${ONNX_OUT}" && "${BIN}" encode-onnx "${ONNX_PATH}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} ) \
    > "${ONNX_OUT}/run.log" 2>&1 \
    || { echo "FAILED: encode-onnx (see ${ONNX_OUT}/run.log)" >&2; tail -20 "${ONNX_OUT}/run.log" >&2; exit 3; }
grep -E '^Dumped' "${ONNX_OUT}/run.log" || true
echo

echo "==> Encoding NNet model (dump-psi / mainDumpPsi)"
( cd "${NNET_OUT}" && "${BIN}" dump-psi "${NNET_PATH}" ${EXTRA_ARGS[@]+"${EXTRA_ARGS[@]}"} ) \
    > "${NNET_OUT}/run.log" 2>&1 \
    || { echo "FAILED: dump-psi (see ${NNET_OUT}/run.log)" >&2; tail -20 "${NNET_OUT}/run.log" >&2; exit 4; }
grep -E '^Dumped' "${NNET_OUT}/run.log" || true
echo

if [[ "${POLISH}" -eq 1 ]]; then
    echo "==> Polishing with $(basename "${POLISH_SED}")"
    for f in "${ONNX_OUT}"/*.smt2 "${NNET_OUT}"/*.smt2; do
        sed -f "${POLISH_SED}" "${f}" > "${f}.polished" && mv "${f}.polished" "${f}"
    done
    echo
fi

# Compare the two sets of generated files.
if [[ "${NORMALIZE}" -eq 1 ]]; then
    echo "==> Normalising solver-internal .ite variable names"
    for f in "${ONNX_OUT}"/*.smt2 "${NNET_OUT}"/*.smt2; do
        python3 - "${f}" <<'PYEOF'
import pathlib, re, sys
path = pathlib.Path(sys.argv[1])
mapping = {}
def canon(m):
    return mapping.setdefault(m.group(0), ".iteN%d_1" % len(mapping))
path.write_text(re.sub(r"\.ite\d+_1", canon, path.read_text()))
PYEOF
    done
    echo
fi

# NOTE: `mapfile` is unavailable in the bash 3.2 shipped with macOS, so read the lists portably.
FILES=()
while IFS= read -r line; do FILES+=("${line}"); done < <(cd "${ONNX_OUT}" && ls -1 *.smt2 2>/dev/null | sort)
NNET_FILES=()
while IFS= read -r line; do NNET_FILES+=("${line}"); done < <(cd "${NNET_OUT}" && ls -1 *.smt2 2>/dev/null | sort)

if [[ ${#FILES[@]} -eq 0 ]]; then echo "ERROR: ONNX side produced no .smt2 files" >&2; exit 5; fi

PSI_DIFF="${SCRIPT_DIR}/psi_diff.py"
if [[ ! -f "${PSI_DIFF}" ]]; then
    echo "ERROR: helper not found: ${PSI_DIFF}" >&2
    exit 5
fi

IGNORE_ARGS=()
if [[ "${IGNORE_BOUNDS}" -eq 1 ]]; then IGNORE_ARGS=(--ignore-assert 0); fi

echo "==> Structural comparison (tolerance ${TOL} relative)"
echo "    Conjuncts of each assertion are matched by structure; the numeric literals of"
echo "    matched conjuncts are then compared individually."
echo "      identical  equal term by term          close      differ only within the tolerance"
echo "      differing  a literal is really off     only-A/B   no structural counterpart"
echo
printf '%-16s %6s %6s %10s %7s %10s %7s %7s %12s  %s\n' \
    "FILE" "A-CONJ" "B-CONJ" "IDENTICAL" "CLOSE" "DIFFERING" "ONLY-A" "ONLY-B" "WORST-REL" "STATUS"

identical=0
differing=0
missing=0

for f in "${FILES[@]}"; do
    a="${ONNX_OUT}/${f}"
    b="${NNET_OUT}/${f}"
    if [[ ! -f "${b}" ]]; then
        printf '%-16s %6s %6s %10s %7s %10s %7s %7s %12s  %s\n' \
            "${f}" "-" "-" "-" "-" "-" "-" "-" "-" "ONLY-ONNX"
        missing=$((missing+1))
        continue
    fi

    report="${OUTDIR}/${f}.report"
    set +e
    stats=$(python3 "${PSI_DIFF}" "${a}" "${b}" --tol "${TOL}" \
        ${IGNORE_ARGS[@]+"${IGNORE_ARGS[@]}"} --stats 2>"${report}.err")
    rc=$?
    set -e
    if [[ ${rc} -ge 2 ]]; then
        printf '%-16s %6s %6s %10s %7s %10s %7s %7s %12s  %s\n' \
            "${f}" "-" "-" "-" "-" "-" "-" "-" "-" "PARSE-ERROR"
        cat "${report}.err" >&2
        differing=$((differing+1))
        continue
    fi
    rm -f "${report}.err"

    read -r na nb ident close diff_n only_a only_b worst <<<"${stats}"
    if [[ ${rc} -eq 0 && "${close}" -eq 0 ]]; then
        status="IDENTICAL"
        identical=$((identical+1))
    elif [[ ${rc} -eq 0 ]]; then
        status="EQUIVALENT"
        identical=$((identical+1))
    else
        status="DIFFERS"
        differing=$((differing+1))
    fi
    printf '%-16s %6s %6s %10s %7s %10s %7s %7s %12s  %s\n' \
        "${f}" "${na}" "${nb}" "${ident}" "${close}" "${diff_n}" "${only_a}" "${only_b}" \
        "${worst}" "${status}"

    # Always keep the detailed per-file report for inspection.
    python3 "${PSI_DIFF}" "${a}" "${b}" --tol "${TOL}" \
        ${IGNORE_ARGS[@]+"${IGNORE_ARGS[@]}"} > "${report}" 2>&1 || true
done

# Report files that only the NNet side produced.
for f in ${NNET_FILES[@]+"${NNET_FILES[@]}"}; do
    if [[ ! -f "${ONNX_OUT}/${f}" ]]; then
        printf '%-16s %6s %6s %10s %7s %10s %7s %7s %12s  %s\n' \
            "${f}" "-" "-" "-" "-" "-" "-" "-" "-" "ONLY-NNET"
        missing=$((missing+1))
    fi
done

if [[ "${TEXTUAL}" -eq 1 ]]; then
    echo
    echo "==> Textual diff"
    for f in "${FILES[@]}"; do
        b="${NNET_OUT}/${f}"
        [[ -f "${b}" ]] || continue
        diffout="${OUTDIR}/${f}.diff"
        if diff -u "${b}" "${ONNX_OUT}/${f}" > "${diffout}"; then
            rm -f "${diffout}"
            printf '  %-16s identical\n' "${f}"
        else
            nd=$(grep -cE '^[+-][^+-]' "${diffout}" || true)
            printf '  %-16s %s changed lines -> %s\n' "${f}" "${nd}" "${diffout}"
        fi
    done
fi

echo
echo "--- Summary ---"
echo "matching files:  ${identical}   (IDENTICAL = literally equal, EQUIVALENT = equal within ${TOL})"
echo "differing files: ${differing}"
echo "missing files:   ${missing}"
echo
echo "ONNX outputs: ${ONNX_OUT}"
echo "NNet outputs: ${NNET_OUT}"
echo "Per-file reports: ${OUTDIR}/<file>.report"
if [[ "${IGNORE_BOUNDS}" -eq 0 ]]; then
    echo
    echo "Hint: the input-domain-bounds assertion (#0) is expected to differ, because ONNX"
    echo "      carries no input ranges and Network2 falls back to its 0/1 defaults."
    echo "      Pass --ignore-bounds to exclude it from the verdict."
fi

if [[ ${missing} -gt 0 ]]; then exit 6; fi
if [[ ${differing} -gt 0 ]]; then exit 1; fi
echo
echo "PASS: encodings agree (within ${TOL} relative tolerance)."
