#!/usr/bin/env bash
# check_r0.sh — the 20-second reproduction check. Run this BEFORE the matrix.
#
#   bash prototype/mlp_insitu/check_r0.sh
#
# Four rows, one lock hold, nothing backgrounded:
#   R0   --layout blob3 --drain burst                          the shipped 48-launch shape, alone
#   R0f  ... --geometry fat                                    the shape the published 2.275 ms was taken in
#   R0b  ... --token-burst 8                                   were R0's 48 launches actually pipelined?
#   R1   ... --drain each                                      the drain cross-check (ledger: 48 x 7.2 us)
#
# WHY each row is here. The first window aborted on canary C2 with R0 = 2.9523 ms (179 GB/s demanded)
# against the published 2.275 ms (232 GB/s) at the same 504 MiB, so P was 0.323 ms — below the
# pre-registered 0.35 ms NO-DIAGNOSIS floor, and nothing downstream was admissible. Reading
# prototype/mlp_gemv/{run.py,mlp_gemv.hip.cpp} for the provenance of the published row (progress/bench/
# evidence/ledger-2026-10-16/commands-and-provenance.md §6: `--layers all --tile 24 --mapping contig`)
# found the shape is not the same shape: mlp_gemv derives `rows` from the WEIGHT FILE SIZE
# (`rows = wbytes / (in*2)`, mlp_gemv.hip.cpp:219), so with `--tile 24` that burst is TWO launches of
# rows=86,016 / rows2=24,576 (grids 10,752 / 3,072) per 24-layer iteration, and run.py divides the
# iteration by 24 to call it per token. The engine emits 48 launches of grid 448/128. R0f is the fat
# shape; R0 is the shipped one; the gap between them is launch granularity, and it is a named term now.
#
# It prints milliseconds, so it is the main agent's to run: zero subagents, blocking, one lock hold.
set -uo pipefail
cd "$(dirname "$0")/../.."
export LC_ALL=C
RIG=${RIG:-prototype/mlp_insitu/mlp_insitu}
TAG=${TAG:-d2-check-r0}
REPS=${REPS:-3}
ITERS=${ITERS:-6}
QUOTED_BURST=${QUOTED_BURST:-2.275}     # QUOTED: engine/performance-status.md §4 (232 GB/s of 504 MiB)
SERIAL_LO=${SERIAL_LO:-0.25}            # pre-registered band on T_serial (48 x 7.2 us = 0.3456 ms)
SERIAL_HI=${SERIAL_HI:-0.45}
OUT=${OUT:-tmp/d2checkr0}; mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd)

fail() { echo "ABORT: $*" >&2; echo "NO-TIMING: a millisecond figure from this run is not usable" >&2
         tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
num() { case "$1" in ''|*[!0-9.]*|*..*) return 1;; *) return 0;; esac; }

[ -x "$RIG" ] || bash prototype/mlp_insitu/build.sh >/dev/null || fail "build failed"
"$RIG" --selftest-geom > "$OUT/geom.log" 2>&1 || fail "--selftest-geom red (see $OUT/geom.log)"
grep -q 'SELFTEST-GEOM ok fails=0' "$OUT/geom.log" || fail "--selftest-geom did not print ok fails=0"

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
echo "== lock held: $(tools/gpu_lock.sh status 2>&1 | head -1) ; util: $(tools/gpu_lock.sh util 2>&1 | tail -1)"

declare -A MS
row() {
    local tag="$1"; shift
    local log="$OUT/$tag.log"
    "$RIG" --time --tag "$tag" --reps "$REPS" --iters "$ITERS" "$@" > "$log" 2>&1 \
        || { sed 's/^/#   /' "$log" >&2; fail "row $tag exited non-zero"; }
    local line ms verify
    line=$(grep -m1 "^INSITU tag=$tag " "$log") || fail "row $tag printed no INSITU line"
    ms=$(echo "$line" | sed -n 's/.* ms_per_token=\([0-9.]*\).*/\1/p')
    verify=$(echo "$line" | sed -n 's/.* verify=\([A-Za-z0-9,_]*\).*/\1/p')
    num "$ms" || fail "row $tag: ms_per_token did not parse ('$ms')"
    [ "$verify" = ok ] || fail "row $tag: verify=$verify — do not use a number from unaccounted bytes"
    MS[$tag]="$ms"
    printf '  %-4s %8s ms  %6s GB/s demanded  geom=%s token_burst=%s launches=%s  [%s]\n' \
        "$tag" "$ms" "$(echo "$line" | sed -n 's/.* gbs_mlp_demanded=\([0-9.]*\).*/\1/p')" \
        "$(echo "$line" | sed -n 's/.* geom=\([a-z]*\).*/\1/p')" \
        "$(echo "$line" | sed -n 's/.* token_burst=\([0-9]*\).*/\1/p')" \
        "$(echo "$line" | sed -n 's/.* launches_mlp=\([0-9]*\).*/\1/p')" "$*"
}

echo "== rows (median of $REPS reps x $ITERS tokens) =="
row R0  --layout blob3 --drain burst
row R0f --layout blob3 --drain burst --geometry fat
row R0b --layout blob3 --drain burst --token-burst 8
row R1  --layout blob3 --drain each
tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1
trap - EXIT

echo
echo "== verdicts =="
d=$(awk -v f="${MS[R0f]}" -v q="$QUOTED_BURST" 'BEGIN{d=(f-q)/q*100; if(d<0)d=-d; printf "%.2f", d}')
num "$d" || fail "reproduction delta did not parse ('$d')"
awk -v q="$QUOTED_BURST" -v f="${MS[R0f]}" -v t="${MS[R0]}" -v b="${MS[R0b]}" -v e="${MS[R1]}" \
    -v lo="$SERIAL_LO" -v hi="$SERIAL_HI" 'BEGIN{
  printf "  C2 reproduction   R0f(fat) %.4f vs QUOTED %.4f = %+.1f %%\n", f, q, (f-q)/q*100
  printf "  T_geo  = R0(thin) - R0f(fat)        = %+.4f ms  (%.1f %% of R0)\n", t-f, (t-f)/t*100
  printf "  T_pipe = R0(burst 1) - R0b(burst 8) = %+.4f ms\n", t-b
  printf "  T_serial = R1 - R0                  = %+.4f ms  band %.2f-%.2f : %s\n", e-t, lo, hi, \
         (((e-t)>=lo) && ((e-t)<=hi) ? "inside (rig drain == stage-clock drain)" : "OUTSIDE — the ledger constant is the finding")
  printf "  QUOTED ledger arithmetic: in-situ 3.3628 - published burst 2.275 = 1.0878 ms.\n"
  printf "  Re-anchor: P_ship = R7 - R0(thin). The published burst is a 2-launch shape the engine never emits.\n"
}'
echo
if awk -v x="$d" 'BEGIN{exit (x<=3.0)?0:1}'; then
    echo "C2 PASS — the rig reproduces the published floor at the published shape. The matrix is"
    echo "admissible, and P must be computed against R0 (thin, shipped shape), which ab_d2_sweep.sh does."
else
    fail "C2: R0f is $d % off the quoted 2.275 ms — the rig cannot reproduce the published burst even at its own 2-launch geometry. Report NO-RESULT and do NOT run the matrix: the 1.11 ms penalty would need re-measuring before anything is funded."
fi
echo
echo "# logs: $OUT/*.log   (single-window medians; quote the spread from the row's own NOTE line)"
echo "# if C2 PASSED: the matrix is admissible, and P must be defined against R0 (thin), not 2.275."
echo "# if C2 FAILED: report NO-RESULT — the published floor is not reproducible from this rig at all."
