#!/usr/bin/env bash
# prototype/mtp_headbatch/run_locked.sh — the device pass for the batched tied head. MAIN AGENT ONLY:
# this prints milliseconds, and only the main agent runs with the box provably exclusive.
#
# One tools/gpu_lock.sh hold, acquired and released inside this one shell, blocking, never backgrounded.
# Arms are adjacent in pairs and the whole set runs twice, so pass-to-pass spread is stated rather than
# assumed (AGENTS: a ruler re-taken in a different window is not an A/B).
#
#   bash prototype/mtp_headbatch/run_locked.sh            # preflight + gate + two timed passes
#   bash prototype/mtp_headbatch/run_locked.sh --admit    # preflight + gate + arm canaries, no ms
#   .venv/bin/python prototype/mtp_headbatch/price_head.py --evidence /tmp/mtp_headbatch/evidence
#
# WHAT IS BEING PRICED. P7 plan A assumes the second candidate row rides the tied table read the
# first row already pays. R = (best 2-row pass) / (best 1-row pass), both hb arms, same window.
#
# PRE-REGISTERED BANDS — fixed before this file could print a millisecond; identical words in
# price_head.py's header and HEADBATCH-HANDOVER.md, arithmetic in price_head.py.
#   R <= 1.10        PLAN A STANDS AS PRICED. Head marginal <= 0.217 ms, i.e. within D1's own
#                    pre-registered +/-5 %-on-the-head margin around the 0.165 ms plan A priced from
#                    the measured proj 1->2 marginal (+7.6 %, D3-RESULT).
#   1.10 < R <= 1.35 PLAN A SURVIVES, NARROWER CLAIM: "P7 clears the standing bar with one batched
#                    body and one batched head at >= 1.17x (>= +1.9 ms/step)", NOT "1.23x", and NOT
#                    "the second row rides the read for free". D3's lm_head row gets re-emitted with
#                    the measured marginal instead of the proj transfer.
#   R > 1.35         THE BATCHED HEAD IS NOT FREE. Re-price in route_price.py before any kernel is
#                    written. Structural note from price_head.py: at R >= 1.862 a batched head no
#                    longer beats calling the head twice by the standing 0.30 ms/step bar, so the
#                    requirement dissolves rather than being renegotiated. And R_KILL = 2.2279, above
#                    two cold passes, so no head number alone can kill P7 — only shrink it.
#   PREDICTION IF THE TABLE IS READ ONCE: R(2) ~ 1.00-1.03 (plateau arithmetic in price_head.py: the
#                    extra logits row is 0.19 % of the bytes, the extra dots take the pass from 3.2 %
#                    to 6.4 % of the quoted 7.30 TFLOP/s scalar v_fma_f32 roof, and at rpb 8 neither
#                    M moves blocks/CU). A measured R far above that is a mechanism not accounted
#                    for, and it is reported as one.
#
# CANARIES (AGENTS 2026-10-17: a stale or wrong-shaped arm reads a clean null, and every parity field
# inside the binary under test is blind to it). Each arm's RESULT line must carry:
#   rows_served / out_elems   MUST move with M          (the arm really served M rows)
#   grid / table_bytes        MUST NOT move with M       (one walk of the table, not one per row)
#   fetch_per_weight          MUST be 1.000000 at every M (bytes actually fetched, counted inside the
#                             kernel and checked against a host walk of the intended range)
#   poison_left               MUST be 0                  (every output element was written)
# Every arm is canaried TWICE: once in --mode check during admission (which prints no ms at all, so
# the canary is proved to fire before a timing window is spent) and again on the timed line. The
# admission phase also counts the arm evidence files and fails if the count is wrong: a runner whose
# arm loop silently stopped early would otherwise print a green admission (empty-parse class).
# No ms is printed for an arm whose canary did not move, and no timing runs at all if admission is
# red (burst_m16.sh: refuse to print a millisecond figure when the admission set is red).
#
# LOCK DISCIPLINE: prototype/mtp_headbatch/{gate.py,mtp_headbatch} carry no lock code
# (`grep -c gpu_lock` is 0 for both), so holding the lock around them is correct. This script invokes
# no self-locking run.py inside the hold.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export LC_ALL=C

WORK=${WORK:-/tmp/mtp_headbatch}
EV=$WORK/evidence
PY=.venv/bin/python
BIN=prototype/mtp_headbatch/mtp_headbatch
GATE=prototype/mtp_headbatch/gate.py
V=248320
ADMIT_ONLY=0
[ "${1:-}" = "--admit" ] && ADMIT_ONLY=1

[ -x "$BIN" ] || bash prototype/mtp_headbatch/build.sh || exit 1
"$PY" prototype/mtp_headbatch/make_inputs.py --verify || {
  echo "inputs failed verification; refusing to run"; exit 1; }
if [ "$(grep -c gpu_lock "$BIN" "$GATE" | grep -v ':0$' | wc -l)" != "0" ]; then
  echo "refusing to hold the lock around a self-locking driver"; exit 1
fi

# ------------------------------------------------------------------ the arm table
# "M rpb" pairs. rpb is rows per BLOCK (lm_head_gemv's --rows), not query rows. M=4 at rpb 8 is
# LDS-bound to 4 blocks/CU (16 KiB/block of staged x), so rpb 16/32 are there to give it a fair arm.
ARMS="1 8|1 16|1 32|2 8|2 16|2 32|4 8|4 16|4 32"
IFS='|' read -r -a ARM_LIST <<< "$ARMS"
ARM_FILE() { echo "$EV/$1-$2-$3.txt"; }              # $1 phase, $2 M, $3 rpb
grid_for() { echo $(( (V + $1 - 1) / $1 )); }        # blocks: a function of rpb ONLY, never of M —
                                                    # which is precisely the claim the canary tests
result_line() { grep '^RESULT' "$1" | tail -1; }
canary_for() {                                      # $1 M, $2 rpb; matched field by field against
  local M=$1 rpb=$2 g; g=$(grid_for "$rpb")        # the RESULT line alone, so a PLAN echo cannot
  echo "arm=hb([^0-9]|$) M=$M([^0-9]|$) rows_served=$M([^0-9]|$) \
out_elems=$((M * V))([^0-9]|$) grid=$g([^0-9]|$) rpb=$rpb([^0-9]|$) \
fetch_per_weight=1.000000([^0-9]|$) poison_left=0([^0-9]|$)"
}
check_canary() {                                    # $1 file, $2 M, $3 rpb
  local rl; rl=$(result_line "$1")
  [ -n "$rl" ] || { echo "    no RESULT line in $1"; return 1; }
  local kv
  for kv in $(canary_for "$2" "$3"); do
    printf '%s\n' "$rl" | grep -qE -- "$kv" || { echo "    canary '$kv' absent from: $rl"; return 1; }
  done
  return 0
}

FAILS=0
run_arm() {                                         # $1 phase (admit|p1|p2), $2 M, $3 rpb
  local ph=$1 M=$2 rpb=$3 f ms rc t0 t1
  f=$(ARM_FILE "$ph" "$M" "$rpb")
  t0=$(date +%s.%N)
  # The admission arm runs the SAME geometry in --mode check, which prints no ms in any path, and
  # the SAME canary: that proves the canary fires on this arm's real output before a timing window
  # is spent. A canary that silently matches nothing is the empty-parse failure AGENTS records three
  # times on this box.
  if [ "$ph" = admit ]; then
    "$BIN" --w "$WORK/W.bin" --x "$WORK/x$M.bin" --M "$M" --rpb "$rpb" --mode check > "$f" 2>&1
  else
    "$BIN" --w "$WORK/W.bin" --x "$WORK/x$M.bin" --M "$M" --rpb "$rpb" --mode burst --iters 25 > "$f" 2>&1
  fi
  rc=$?
  t1=$(date +%s.%N)
  if [ $rc -ne 0 ]; then
    echo "  FAIL $ph hb M=$M rpb=$rpb rc=$rc wall=$(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.1f",b-a}')s (see $f)"
    FAILS=$((FAILS + 1)); return
  fi
  if ! check_canary "$f" "$M" "$rpb"; then
    echo "  NO-TIMING $ph hb M=$M rpb=$rpb — canary did not move as required"; FAILS=$((FAILS + 1)); return
  fi
  # NOTE: a bare `grep -oE 'ms=[0-9.]+'` also matches inside `out_elems=` and prints the element
  # count as if it were a timing, so the match is anchored on a preceding space.
  ms=$(result_line "$f" | grep -oE '[[:space:]]ms=[0-9.]+' | tail -1 | tr -d ' ')
  echo "  ok $ph hb M=$M rpb=$rpb $(result_line "$f" | grep -oE 'grid=[0-9]+') fetch=$(result_line "$f" | grep -oE 'fetch_per_weight=[0-9.]+' | cut -d= -f2) ${ms:-no-ms}"
}
run_shipped() {                                     # the M=1 arm the engine ships today
  local ph=$1 f ms rc
  f="$EV/$ph-shipped.txt"
  if [ "$ph" = admit ]; then
    "$BIN" --w "$WORK/W.bin" --x "$WORK/xrow0.bin" --M 1 --arm shipped --mode check > "$f" 2>&1
  else
    "$BIN" --w "$WORK/W.bin" --x "$WORK/xrow0.bin" --M 1 --arm shipped --mode burst --iters 25 > "$f" 2>&1
  fi
  rc=$?
  if [ $rc -ne 0 ]; then echo "  FAIL $ph shipped rc=$rc"; FAILS=$((FAILS + 1)); return; fi
  local rl; rl=$(result_line "$f")
  printf '%s\n' "$rl" | grep -qE 'arm=shipped([^0-9]|$)' \
    && printf '%s\n' "$rl" | grep -qE 'rows_served=1([^0-9]|$)' \
    && printf '%s\n' "$rl" | grep -qE 'out_elems=248320([^0-9]|$)' || {
      echo "  NO-TIMING $ph shipped — canary absent from: $rl"; FAILS=$((FAILS + 1)); return; }
  ms=$(printf '%s\n' "$rl" | grep -oE '[[:space:]]ms=[0-9.]+' | tail -1 | tr -d ' ')
  echo "  ok $ph shipped (the M=1 shipping kernel, same process shape as the arms) ${ms:-no-ms}"
}

echo "-- preflight: --mode plan for every arm (host-only: no allocations, no launches) --"
for pair in "${ARM_LIST[@]}"; do
  M=${pair%% *}; rpb=${pair##* }; g=$(grid_for "$rpb")
  f=$WORK/preflight-$M-$rpb.txt
  "$BIN" --w "$WORK/W.bin" --x "$WORK/x$M.bin" --M "$M" --rpb "$rpb" --mode plan > "$f" 2>&1 || {
    echo "PREFLIGHT FAIL M=$M rpb=$rpb (see $f)"; exit 1; }
  grep -qE "arm=hb M=$M V=$V rpb=$rpb grid=$g threads=" "$f" || {
    echo "PREFLIGHT FAIL: M=$M rpb=$rpb geometry not as expected: $(cat "$f")"; exit 1; }
  echo "  ok M=$M rpb=$rpb: grid=$g out_elems=$((M * V)) intended_fetch_per_weight=1.000000"
done
"$BIN" --w "$WORK/W.bin" --x "$WORK/xrow0.bin" --M 1 --arm shipped --mode plan | grep -q 'arm=shipped' || {
  echo "PREFLIGHT FAIL: shipped baseline arm"; exit 1; }

tools/gpu_lock.sh acquire "p7-headbatch-ratio" || { echo "lock busy"; exit 1; }
trap 'tools/gpu_lock.sh release "p7-headbatch-ratio"' EXIT
U0=$(tools/gpu_lock.sh util 2>/dev/null | tr -s ' ')
echo "== batched head | lock=p7-headbatch-ratio util@acquire=$U0 | work=$WORK =="
mkdir -p "$EV"; rm -f "$EV"/*.txt

echo "-- admission 1: gate.py — correctness only, one process per case, prints no ms --"
t0=$(date +%s.%N)
"$PY" "$GATE" > "$EV/gate.txt" 2>&1
grc=$?
t1=$(date +%s.%N)
if [ $grc -ne 0 ]; then
  echo "NO-TIMING: gate.py is red (rc=$grc, wall $(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.0f", b-a}')s, see $EV/gate.txt)"
  tail -25 "$EV/gate.txt"
  tools/gpu_lock.sh util | sed 's/^/util at exit: /'
  exit 1
fi
echo "  ok gate.py green in $(awk -v a=$t0 -v b=$t1 'BEGIN{printf "%.0f", b-a}')s of wall (wall is not a measurement)"
grep -E '^(PASS|   argmax)' "$EV/gate.txt" | sed 's/^/  /'

echo "-- admission 2: every timing arm's geometry and canary, in --mode check (prints no ms) --"
for pair in "${ARM_LIST[@]}"; do
  M=${pair%% *}; rpb=${pair##* }
  run_arm admit "$M" "$rpb"
done
run_shipped admit
n_admit=$(find "$EV" -maxdepth 1 -name 'admit-*' -name '*.txt' ! -name 'gate.txt' | wc -l)
want_admit=$(( ${#ARM_LIST[@]} + 1 ))
if [ "$n_admit" -ne "$want_admit" ]; then
  echo "NO-TIMING: admission wrote $n_admit arm files, expected $want_admit — an arm loop that"
  echo "  stopped early cannot fail, so the count is checked."
  exit 1
fi
if [ "$FAILS" -gt 0 ] || [ "$grc" -ne 0 ]; then
  echo "NO-TIMING: admission is red ($FAILS arm(s)). Fix the harness before spending a timing window."
  tools/gpu_lock.sh util | sed 's/^/util at exit: /'
  exit 1
fi
echo "  ok all $want_admit arms admitted with their canaries fired"
if [ "$ADMIT_ONLY" = 1 ]; then
  echo "-- --admit done: admission green, no timing taken, no ms printed --"
  exit 0
fi

for pass in p1 p2; do
  echo "-- timed pass $pass (arms adjacent, whole set twice) --"
  for pair in "${ARM_LIST[@]}"; do
    M=${pair%% *}; rpb=${pair##* }
    run_arm "$pass" "$M" "$rpb"
  done
  run_shipped "$pass"
done

U1=$(tools/gpu_lock.sh util 2>/dev/null | tr -s ' ')
echo "== done | util@acquire=$U0 util@release=$U1 | fails=$FAILS | evidence: $EV =="
if [ "$FAILS" -gt 0 ]; then
  echo "NO-VERDICT: $FAILS arm(s) failed or were refused; price_head.py must not be run on partial evidence."
  exit 1
fi
echo "Next: $PY prototype/mtp_headbatch/price_head.py --evidence $EV"
echo "These ARE timings: they may enter engine/performance-status.md only because the main agent took"
echo "them, blocking, with zero subagents active — which is the precondition for running this file."
exit 0
