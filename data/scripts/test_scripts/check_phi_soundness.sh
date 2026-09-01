#!/usr/bin/env bash
set -uo pipefail

# Verify that every explanation in a `.phi.txt` file is a *sound* explanation w.r.t. the
# class-encoding files `psi_c<label>.smt2` produced by `dump-psi` / `encode-onnx`.
#
# `psi_c<c>.smt2` encodes "the network output can be something other than class <c>".
# An explanation phi determines the classification iff there is some class c with
#   psi_c<c>  AND  phi   UNSAT.
# That class is then the one phi explains, and it is reported per sample.
#
# Pass every psi_c*.smt2 file of the model; with a single file the check degenerates to
# "is phi an explanation for exactly that class".
#
# This replaces `./data/scripts/analyze.sh check`, which does not run on macOS
# (BSD `mktemp` has no --suffix, and bash 3.2 has no `local -n` / `wait -n`).
#
# Usage:
#   ./data/scripts/test_scripts/check_phi_soundness.sh <phi.txt> <psi_c0.smt2> [<psi_c1.smt2> ...]
#
# Environment:
#   SOLVER   SMT solver to use (default: opensmt; z3 also works)

SOLVER="${SOLVER:-opensmt}"

PHI="${1:-}"
shift || true
PSIS=("$@")

if [[ -z "${PHI}" || "${#PSIS[@]}" -eq 0 ]]; then
    sed -n '4,24p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//' >&2
    exit 2
fi
for f in "${PHI}" "${PSIS[@]}"; do
    [[ -f "${f}" ]] || { echo "ERROR: file not found: ${f}" >&2; exit 2; }
done
command -v "${SOLVER}" >/dev/null || { echo "ERROR: solver not found: ${SOLVER}" >&2; exit 2; }

# opensmt requires a .smt2 file extension, so use a scratch dir with a fixed name.
TMPD="$(mktemp -d /tmp/phi_sound.XXXXXX)"
trap 'rm -rf "${TMPD}"' EXIT

# Strip the trailing (check-sat)/(exit) of each psi so we can append our own assertion.
BODIES=()
LABELS=()
for psi in "${PSIS[@]}"; do
    b="${TMPD}/body_${#BODIES[@]}.smt2"
    grep -Ev '^\(check-sat\)$|^\(exit\)$' "${psi}" > "${b}"
    BODIES+=("${b}")
    lbl="$(basename "${psi}")"; lbl="${lbl#psi_c}"; lbl="${lbl%.smt2}"
    LABELS+=("${lbl}")
done

n=0; ok=0; bad=0
while IFS= read -r line; do
    [[ -z "${line}" ]] && continue
    n=$((n + 1))
    determined=""
    for i in "${!BODIES[@]}"; do
        q="${TMPD}/q.smt2"
        { cat "${BODIES[$i]}"; printf '(assert %s)\n(check-sat)\n(exit)\n' "${line}"; } > "${q}"
        if [[ "$("${SOLVER}" "${q}" 2>/dev/null | tr -d '[:space:]')" == "unsat" ]]; then
            determined="${LABELS[$i]}"
            break
        fi
    done
    if [[ -n "${determined}" ]]; then
        ok=$((ok + 1))
    else
        bad=$((bad + 1))
        echo "  sample ${n}: SAT for every class -- NOT a sound explanation"
    fi
done < "${PHI}"

echo "  ${ok}/${n} sound, ${bad} unsound"
[[ "${bad}" -eq 0 ]]
