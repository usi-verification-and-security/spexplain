#!/usr/bin/env bash
set -uo pipefail

# Compare two `.phi.txt` explanation files *logically* (not textually), sample by sample,
# under the input-domain constraints from a `psi_d.smt2` file.
#
# For each sample i it decides which of these hold:
#   psi_d /\ A_i  =>  B_i     (A is at least as strong / a subset of B)
#   psi_d /\ B_i  =>  A_i
# and reports  =  (equivalent), <  (A stronger), >  (B stronger), ?  (incomparable).
#
# This replaces `./data/scripts/analyze.sh compare-subset`, which does not run on macOS
# (BSD `mktemp` has no --suffix, and bash 3.2 has no `local -n` / `wait -n`).
#
# Usage:
#   ./data/scripts/test_scripts/compare_phi_subset.sh <psi_d.smt2> <A.phi.txt> <B.phi.txt> [<solver>]

PSI="${1:-}"
A="${2:-}"
B="${3:-}"
SOLVER="${4:-opensmt}"

if [[ -z "${PSI}" || -z "${A}" || -z "${B}" ]]; then
    sed -n '4,17p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi
for f in "${PSI}" "${A}" "${B}"; do
    [[ -f "${f}" ]] || { echo "ERROR: file not found: ${f}" >&2; exit 2; }
done
command -v "${SOLVER}" >/dev/null || { echo "ERROR: solver not found: ${SOLVER}" >&2; exit 2; }

BODY="$(grep -Ev '^\(check-sat\)$|^\(exit\)$' "${PSI}")"

# opensmt requires a .smt2 file extension, so use a scratch dir with a fixed name.
TMPDIR_SUB="$(mktemp -d /tmp/phi_subset.XXXXXX)"
trap 'rm -rf "${TMPDIR_SUB}"' EXIT

# implies <lhs> <rhs>  -> exit 0 iff  psi_d /\ lhs  =>  rhs
implies() {
    local q res
    q="${TMPDIR_SUB}/q.smt2"
    {
        printf '%s\n' "${BODY}"
        printf '(assert %s)\n(assert (not %s))\n(check-sat)\n(exit)\n' "$1" "$2"
    } > "${q}"
    res="$("${SOLVER}" "${q}" 2>/dev/null | tr -d '[:space:]')"
    [[ "${res}" == "unsat" ]]
}

paste -d'\t' "${A}" "${B}" | {
    n=0
    while IFS=$'\t' read -r a b; do
        [[ -z "${a}" ]] && continue
        n=$((n + 1))
        ab=1; ba=1
        implies "${a}" "${b}" || ab=0
        implies "${b}" "${a}" || ba=0
        if   [[ "${ab}" -eq 1 && "${ba}" -eq 1 ]]; then echo "  sample ${n}: =  (logically equivalent)"
        elif [[ "${ab}" -eq 1 ]];                  then echo "  sample ${n}: <  (A stronger / subset of B)"
        elif [[ "${ba}" -eq 1 ]];                  then echo "  sample ${n}: >  (B stronger / subset of A)"
        else                                            echo "  sample ${n}: ?  (incomparable)"
        fi
    done
}
