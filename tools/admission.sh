#!/usr/bin/env bash
# admission.sh — run `engine/gate.py` once per fixture in `data/refs`, and assert this run's own count.
#
#     tools/admission.sh <gemm-route> [extra gate flags…]          # the whole population
#     tools/admission.sh lt
#     tools/admission.sh hand --attn-tile 0
#     tools/admission.sh lt --fixture dec64f --fixture wide520f    # a subset; every name is validated
#     ADMISSION_REFS=/tmp/nowhere tools/admission.sh lt             # enumeration control (see below)
#
# Why this exists, and the exact mistake it is built to catch. `bench/performance-status.md` §15.2 and
# decision D40: three admissions this session claimed "18/18 gates — nine canonical fixtures × both
# routes" from a fixture list typed from memory. SEVEN of those nine names do not exist in
# `data/refs/`; `gate.py` exited `fixture X has no manifest` for each of them, and the harness printed
# `PASS=0` — which reads as a numerics result and was an error exit from the thing being counted. The
# defect was structural, not careless: a harness that greps a log cannot tell "no PASS line" from "no
# run". So the rules of this script are:
#
#   * the population is ENUMERATED from `data/refs/*/manifest.json`, never recited;
#   * an unknown name ABORTS (the §15.2 names would have died here), and so does an EMPTY population —
#     a table of nothing must not be able to print a clean `ran == enumerated`;
#   * every row is PASS, FAIL, SKIPPED (a carried record `gate.py` names and declines — see
#     `engine/gate.py::NOT_A_PIN_FIXTURE`), or NO-VERDICT. NO-VERDICT is §15.2's case: no verdict line
#     at all. It is reported as a harness defect and never as a failure of the engine, and never as a
#     pass, and it makes this script exit non-zero;
#   * the last line asserts `ran == enumerated`, so a fixture that vanished between enumeration and
#     the run cannot be missed, and every directory that is NOT a fixture is named with its reason.
#
# No timing. This is a correctness gate: it prints no `ms`, no wall per fixture and no throughput,
# because a figure taken inside a gate run is a by-product the gate itself disclaims
# (`bench/performance-status.md` §6 owns measurements, in an exclusive window, run by the main agent).
#
# GPU: `engine/gate.py` does NOT self-acquire `tools/gpu_lock.sh` (`grep -c gpu_lock engine/gate.py`
# is 0), so this script holds the lock once for the whole run and releases it on EXIT with the tag it
# acquired. Note the AGENTS.md trap the other way round: a self-locking driver
# (`prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`) must never
# be wrapped inside a held lock — check `grep -c gpu_lock <driver>` before adding one here.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 1

GATE=(.venv/bin/python engine/gate.py)
refs=${ADMISSION_REFS:-data/refs}
logs=tmp/admission_logs

# A directory under `data/refs/` with no `manifest.json` is not a fixture and `gate.py` cannot see it,
# so it cannot be named-and-skipped by `NOT_A_PIN_FIXTURE`. It is named HERE instead, with the same
# rule (`tools/reference_dump.py::CARRIED_NOT_PINNED`: a carried thing is named and skipped, never
# silently dropped). Anything not listed aborts.
NO_MANIFEST_DIRS=("corpus48engine-smoke")
no_manifest_why() {
    printf '%s' "empty scratch directory — no manifest.json and no dumps in it"
    printf '%s' " (ls data/refs/$1), so it is not a fixture. If a manifest ever appears in it, drop"
    printf '%s' " it from NO_MANIFEST_DIRS in tools/admission.sh and it joins the population."
}

route="${1:?usage: admission.sh <gemm-route hand|library|lt> [extra gate flags…]}"
shift
case "$route" in hand | library | lt) ;;
    *) printf 'ADMISSION ABORT — --gemm route %s is not one of hand|library|lt\n' "$route" >&2
       exit 2 ;;
esac
extra=()
want=()
while [ $# -gt 0 ]; do
    case "$1" in
        --fixture)
            shift
            [ $# -gt 0 ] || { echo "ADMISSION ABORT — --fixture needs a name" >&2; exit 2; }
            want+=("$1") ;;
        --fixture=*) want+=("${1#--fixture=}") ;;
        *) extra+=("$1") ;;
    esac
    shift
done

if [ ! -d "$refs" ]; then
    printf 'ADMISSION ABORT — no fixture directory %s\n' "$refs" >&2
    exit 2
fi

# ---- enumerate the population. Nothing here is remembered. --------------------------------------
mapfile -t all_dirs < <(find "$refs" -mindepth 1 -maxdepth 1 -type d | sort)
fixtures=()
no_manifest=()
for d in "${all_dirs[@]}"; do
    if [ -f "$d/manifest.json" ]; then
        fixtures+=("$(basename "$d")")
    else
        no_manifest+=("$(basename "$d")")
    fi
done

unlisted=()
for n in "${no_manifest[@]}"; do
    hit=0
    for ok in "${NO_MANIFEST_DIRS[@]}"; do [ "$ok" = "$n" ] && hit=1; done
    [ $hit -eq 0 ] && unlisted+=("$n")
done
if [ ${#unlisted[@]} -gt 0 ]; then
    printf 'ADMISSION ABORT — %s: directory under %s with no manifest.json\n' "${unlisted[*]}" "$refs" >&2
    printf '  Not every name it holds is accounted for: give it a manifest, or name it in\n' >&2
    printf '  NO_MANIFEST_DIRS in tools/admission.sh with a reason. Skipping silently is the §15.2 bug.\n' >&2
    exit 2
fi

if [ ${#want[@]} -gt 0 ]; then
    unknown=()
    for n in "${want[@]}"; do
        [ -f "$refs/$n/manifest.json" ] || unknown+=("$n")
    done
    if [ ${#unknown[@]} -gt 0 ]; then
        printf 'ADMISSION ABORT — unknown fixture name(s): %s\n' "${unknown[*]}" >&2
        printf '  none has %s/<name>/manifest.json. The population is: %s\n' \
               "$refs" "${fixtures[*]:-<empty>}" >&2
        exit 2
    fi
    fixtures=("${want[@]}")
fi

enumerated=${#fixtures[@]}
if [ "$enumerated" -eq 0 ]; then
    printf 'ADMISSION ABORT — enumerated 0 fixtures under %s\n' "$refs" >&2
    printf '  An admission that runs nothing is not a pass; it is a harness that cannot fail.\n' >&2
    printf '  (Checked before the header prints, so an empty population cannot reach a table.)\n' >&2
    exit 2
fi

mkdir -p "$logs"
# The lock serialises DEVICE work. `--no-engine` and `--dry-run` start no binary (gate.py's own help
# says so) and AGENTS.md says not to hold the lock during CPU-only work, so an enumeration check over
# the whole population runs unlocked — and prints that it did, so nobody reads it as a device admission.
device=1
for flag in ${extra[@]:+"${extra[@]}"}; do
    case "$flag" in --no-engine | --dry-run) device=0 ;; esac
done
if [ "$device" -eq 1 ]; then
    tag="admission $route"
    tools/gpu_lock.sh acquire "$tag" || exit 1
    trap 'tools/gpu_lock.sh release "$tag"' EXIT
    lockline="gpu_lock held as $tag — $(tools/gpu_lock.sh util 2>/dev/null | tr -d '\n')"
else
    lockline="no gpu_lock taken: these flags ask for a no-device run (--no-engine/--dry-run)"
fi

printf 'ADMISSION route=%s extra=[%s] refs=%s enumerated=%d\n' \
       "$route" "${extra[*]:-}" "$refs" "$enumerated"
printf '  %s\n' "$lockline"
printf '  %s\n' "correctness only — no timing is taken or printed here"
printf '%-20s %-12s %s\n' FIXTURE CLASS DETAIL
printf '%-20s %-12s %s\n' -------------------- ---------- ------

ran=0
pass=0
fail=0
skipped=0
noverdict=0
for name in "${fixtures[@]}"; do
    log="$logs/$name.log"
    # The verdict lines are the ONLY thing read back. Exit code alone cannot do it: `gate.py` exits 1
    # both for a red numerics result and for a `SystemExit` refusal (AGENTS.md: a check whose input can
    # be empty cannot fail; a check whose error and result share an exit code cannot adjudicate).
    "${GATE[@]}" --fixture "$name" --gemm "$route" ${extra[@]:+"${extra[@]}"} >"$log" 2>&1
    rc=$?
    ran=$((ran + 1))
    if grep -q '^ENGINE GATE PASS' "$log"; then
        class=PASS
        detail=$(grep -m1 '^ENGINE GATE PASS' "$log")
        pass=$((pass + 1))
    elif grep -q '^ENGINE GATE FAIL' "$log"; then
        class=FAIL
        detail=$(grep -m1 '^ENGINE GATE FAIL' "$log")
        fail=$((fail + 1))
    elif grep -q '^ENGINE GATE SKIPPED' "$log"; then
        class=SKIPPED
        detail=$(grep -m1 '^ENGINE GATE SKIPPED' "$log" | cut -c1-96)
        skipped=$((skipped + 1))
    else
        class=NO-VERDICT
        detail="rc=$rc and no verdict line — read $log; this is a harness gap, NOT an engine result"
        noverdict=$((noverdict + 1))
    fi
    printf '%-20s %-12s %s\n' "$name" "$class" "$detail"
done

for n in "${no_manifest[@]}"; do
    printf '%-20s %-12s %s\n' "$n" NO-MANIFEST "$(no_manifest_why "$n")"
done

printf '%-20s %-12s %s\n' -------------------- ---------- ------
printf 'ADMISSION ran=%d enumerated=%d  pass=%d fail=%d skipped-not-a-pin=%d no-verdict=%d\n' \
       "$ran" "$enumerated" "$pass" "$fail" "$skipped" "$noverdict"
if [ "$ran" -ne "$enumerated" ]; then
    printf 'ADMISSION FAIL — ran != enumerated: %d fixture(s) never reached a verdict classifier\n' \
           "$((enumerated - ran))"
    exit 2
fi
if [ "$noverdict" -gt 0 ]; then
    printf 'ADMISSION FAIL — %d fixture(s) produced no verdict line. Do not read those as engine' "$noverdict"
    printf ' results (§15.2): engine/gate.py raised or refused before report(), so the fix is in the\n'
    printf ' harness — read the named log above, and note that their exit code alone cannot say this.\n'
    exit 3
fi
if [ "$fail" -gt 0 ]; then
    printf 'ADMISSION RED — %d fixture(s) adjudicated and failed; see %s\n' "$fail" "$logs"
    exit 1
fi
printf 'ADMISSION CLEAN — every enumerated fixture reached a verdict and none of them is red\n'
exit 0
