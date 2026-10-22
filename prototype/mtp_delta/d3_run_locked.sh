#!/usr/bin/env bash
# D3 — `delta` (what verifying TWO query rows per step costs over ONE): the device pass, MAIN AGENT.
#
# One `tools/gpu_lock.sh` hold, acquired and released inside this one shell, blocking, never backgrounded.
# Arms are adjacent in pairs so each A/B is decided inside a single clock window (AGENTS: "a ruler
# re-taken in a different window is not an A/B"), and the whole set runs twice so pass-to-pass spread is
# stated rather than assumed.
#
#   bash prototype/mtp_delta/d3_run_locked.sh            # admission + two timed passes
#   bash prototype/mtp_delta/d3_run_locked.sh --admit    # admission only: correctness, no timings
#
# PRE-REGISTERED LINES (fixed before this file could print a millisecond; identical in D3-HANDOVER.md §0)
#   KILL  delta > 0.379 of a step = 4.143 ms (route_price.py: FULL head, c = 0.2237, a = 0.7329,
#         4,032 lanes, quoted step 10.9357 ms)
#   WIN   the FULL head still clears the standing +0.30 ms/step bar (speedup >= 1.027) at the measured
#         delta. For the FULL head KILL and WIN are the SAME line: d_max = (1+a)/(BAR*(1+c)) - 1 =
#         0.3788 is exactly where 1.416x decays to 1.027x. So report the distance below the kill line;
#         do not invent a comfort band.
#   every verdict is an ESTIMATE OF delta FROM RIG COMPONENTS, never a measurement of the engine's
#   verify path, which cannot exist until the draft path is built.
#
# CANARY (AGENTS 2026-10-17: a stale or wrong-shaped arm reads a clean null, and that is this box's
# signature failure). Each arm asserts a field the 2-row form is REQUIRED to move, read from the rig's
# own output in the same breath as the timing; a failed canary prints NO-TIMING and aborts:
#   proj            the plan echo's `tokens=` moves 1 -> 2; `launches=48` must NOT move
#   mlp_*           the RESULT line's own `tokens=` moves; `rows=` and `in=` must NOT move
#   attn            `rows=` goes 6 -> 12 and `core_grid=` goes 8x8x1 -> 8x8x2 (grid.z IS the row)
#   proj_i1024      `shared 8192 B/block` -> `shared 32768 B/block` (the tile-width control)
# No ms is printed for an arm whose canary did not move, and no timing runs at all if the admission pass
# is red (the `burst_m16.sh` rule: refuse to print a number when the admission set is red).
#
# LOCK DISCIPLINE: the `run.py` drivers for proj_gemv, glue, gdn_gated_norm, attn_decode, attn_m2,
# gdn_collapse and issue_rate_probe self-acquire the lock, so this script invokes the BINARIES only.
# Never hold the lock and then call a self-locking driver: it parks silently and looks like a hung box.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export LC_ALL=C

WORK=${WORK:-/tmp/mtp_delta}
EV=prototype/mtp_delta/evidence
PY=.venv/bin/python
BIN_PROJ=prototype/proj_gemv/proj_gemv
BIN_MLP=prototype/mlp_gemv/mlp_gemv
BIN_ATTN=prototype/attn_decode/attn_decode
MLP_LAYERS=${MLP_LAYERS:-"0 3 11 23"}     # sampled layers; weight shapes are identical across layers
SPLIT=8                                   # ceil(4033/512): the lane grid BOTH attention arms share

ADMIT_ONLY=0
[ "${1:-}" = "--admit" ] && ADMIT_ONLY=1

for b in "$BIN_PROJ" "$BIN_MLP" "$BIN_ATTN"; do
  [ -x "$b" ] || { echo "building $b"; bash "${b%/*}/build.sh" || exit 1; }
done
[ -d "$WORK" ] || { echo "making inputs in $WORK"; "$PY" prototype/mtp_delta/make_inputs.py || exit 1; }
"$PY" prototype/mtp_delta/make_inputs.py --verify || {
  echo "inputs failed verification; refusing to run"; exit 1; }
mkdir -p "$EV" "$EV/admit"; rm -f "$EV"/*.txt "$EV"/admit/*.txt

TAG=p7-d3-delta
# Device-free geometry preflight, OUTSIDE the lock: `--dump-plan` stages nothing and launches nothing
# (the binary says so in its own comment), so a wrong-shaped plan is caught before a window is spent,
# and this line is also the proof that the two timing arms really differ.
echo "-- preflight: plan geometry via --dump-plan (no launches, no allocations, safe under any lock) --"
for arm in 1 2; do
  if ! "$BIN_PROJ" --mode once --plan "$WORK/plan_proj_t$arm.txt" --tokens "$arm" \
        --variant v0-baseline --rows 8 --dump-plan > "$WORK/preflight-proj-t$arm.txt" 2>&1; then
    echo "PREFLIGHT FAIL plan_proj_t$arm.txt (see $WORK/preflight-proj-t$arm.txt)"; exit 1
  fi
  grep -qE "tokens=$arm([^0-9]|$)" "$WORK/preflight-proj-t$arm.txt" \
    || { echo "PREFLIGHT FAIL: plan_proj_t$arm.txt does not echo tokens=$arm"; exit 1; }
  grep -q 'launches=48' "$WORK/preflight-proj-t$arm.txt" \
    || { echo "PREFLIGHT FAIL: plan_proj_t$arm.txt is not the 48-launch engine plan"; exit 1; }
  echo "  ok plan_proj_t$arm: launches=48 tokens=$arm"
done
tools/gpu_lock.sh acquire "$TAG" || { echo "lock busy"; exit 1; }
trap 'tools/gpu_lock.sh release "$TAG"' EXIT
U0=$(tools/gpu_lock.sh util 2>/dev/null | tr -s ' ')
echo "== D3 delta harness | lock=$TAG util@acquire=$U0 | work=$WORK | mlp layers: $MLP_LAYERS =="
echo "commands, verbatim:"
echo "  $BIN_PROJ --mode burst --plan $WORK/plan_proj_t{1,2}.txt --tokens {1,2} --variant v0-baseline --rows 8 --iters 12 --reps 3 --warm 2 --preburn 2 --label d3"
echo "  $BIN_PROJ --mode burst --plan $WORK/plan_proj_i1024_t1.txt --tokens 1 [--tpb 4] ...   (tile control)"
echo "  $BIN_ATTN --mode burst --plan $WORK/plan_attn_n{1,2}.txt --launch staged --softmax online --split $SPLIT --iters 20"
echo "  $BIN_MLP --mode gateup --in 1024 --tokens {1,2,4} --x \$WORK/x/x1024_tT.bin --wg/--wu <layer L##>"
echo "  $BIN_MLP --mode down   --in 3584 --tokens {1,2}   --x \$WORK/x/x3584_tT.bin --wd <layer L##>"
echo

# ---------------------------------------------------------------- the arms
arm_proj() {                     # $1 = t1 | t2
  "$BIN_PROJ" --mode burst --plan "$WORK/plan_proj_$1.txt" --tokens "${1#t}" \
    --variant v0-baseline --rows 8 --iters 12 --reps 3 --warm 2 --preburn 2 --label "d3-$1"
}
arm_proj_ctrl() {                # $1 = 1 | 4  -> same plan, same one-token work, wider x tile only
  "$BIN_PROJ" --mode burst --plan "$WORK/plan_proj_i1024_t1.txt" --tokens 1 --tpb "$1" \
    --variant v0-baseline --rows 8 --iters 12 --reps 3 --warm 2 --preburn 2 --label "d3-i1024tpb$1"
}
arm_attn() {                     # $1 = n1 | n2
  "$BIN_ATTN" --mode burst --plan "$WORK/plan_attn_$1.txt" --launch staged --softmax online \
    --split "$SPLIT" --iters 20
}
arm_mlp_gateup() {               # $1 = tokens, $2 = layer
  local l; l=$(printf 'L%02d' "$2")
  "$BIN_MLP" --mode gateup --in 1024 --tokens "$1" --x "$WORK/x/x1024_t$1.bin" \
    --wg "$WORK/weights/${l}_mlp_gate_proj.bin" --wu "$WORK/weights/${l}_mlp_up_proj.bin" \
    --out "$WORK/mlp_out.bin" --iters 30
}
arm_mlp_down() {                 # $1 = tokens, $2 = layer
  local l; l=$(printf 'L%02d' "$2")
  "$BIN_MLP" --mode down --in 3584 --tokens "$1" --x "$WORK/x/x3584_t$1.bin" \
    --wd "$WORK/weights/${l}_mlp_down_proj.bin" --out "$WORK/mlp_out.bin" --iters 30
}
# One evidence file per component+arm, carrying one RESULT line per sampled layer; the roll-up in
# components.py sums them and scales by 24/|layers|, and the layer count travels IN the file so the
# scale factor cannot drift from what was actually run.
arm_gateup() { local t=$1; for l in $MLP_LAYERS; do arm_mlp_gateup "$t" "$l" || return 1; done
               echo "# layers=$(echo "$MLP_LAYERS" | wc -w)"; }
arm_down()   { local t=$1; for l in $MLP_LAYERS; do arm_mlp_down   "$t" "$l" || return 1; done
               echo "# layers=$(echo "$MLP_LAYERS" | wc -w)"; }

# ---------------------------------------------------------------- canaries
# Space-separated EREs, all of which must appear in the arm's output. `([^0-9]|$)` guards `tokens=1`
# from being satisfied by `tokens=12`.
canary_for() {                   # $1 = component, $2 = arm, $3 = phase (admit arms run --mode once,
  local comp=$1 arm=$2 phase=${3:-run}   # which prints no `shared` line, so that field is run-only)
  case "$comp/$arm" in
    proj/t1)        echo 'tokens=1([^0-9]|$) launches=48' ;;
    proj/t2)        echo 'tokens=2([^0-9]|$) launches=48' ;;
    proj_i1024/tpb1)
      echo 'tokens=1([^0-9]|$) launches=24'"$([ "$phase" != admit ] && echo ' shared 8192 B/block')" ;;
    proj_i1024/tpb4)
      echo 'tokens=1([^0-9]|$) launches=24'"$([ "$phase" != admit ] && echo ' shared 32768 B/block')" ;;
    attn/n1)        echo 'rows=6([^0-9]|$) core_grid=8x8x1' ;;
    attn/n2)        echo 'rows=12([^0-9]|$) core_grid=8x8x2' ;;
    mlp_gateup/t1)  echo 'tokens=1([^0-9]|$) rows=3584 in=1024' ;;
    mlp_gateup/t2)  echo 'tokens=2([^0-9]|$) rows=3584 in=1024' ;;
    mlp_gateup/t4)  echo 'tokens=4([^0-9]|$) rows=3584 in=1024' ;;
    mlp_down/t1)    echo 'tokens=1([^0-9]|$) rows=1024 in=3584' ;;
    mlp_down/t2)    echo 'tokens=2([^0-9]|$) rows=1024 in=3584' ;;
    *)              echo '' ;;
  esac
}
check_canary() {                 # $1 = file, $2 = space-separated EREs
  local f=$1 kv
  for kv in $2; do
    grep -qE -- "$kv" "$f" || { echo "    canary '$kv' ABSENT in $f"; echo "    line 1: $(head -1 "$f")"; return 1; }
  done
  return 0
}
run_phase() {                    # $1 = "admit" | pass label. One dispatch, so admit and timing
  local phase=$1 comp pair       # cannot drift apart, which is the whole point of admitting first
  local dir="$EV"
  # Clear only THIS phase's files. The original `rm -f "$dir"/*.txt` deleted the previous pass's
  # evidence before anyone read it (a lost window that looked like a passing run), but the evidence
  # directory is also a flat namespace that components.py discovers by globbing - so the fix is a
  # per-phase prefix, not a per-phase subdirectory.
  mkdir -p "$dir"; rm -f "$dir"/$phase-*.txt
  # NOTE on the dropped control arm `proj_i1024 tpb4`: this binary cannot run it in burst mode - its
  # occupancy probe enumerates (in=2048, tpb=4), which is not instantiated (1024 takes tpb 1/4, 2048
  # takes 1/2), so it exits 1 at the probe even though the plan is 100 % in=1024, and burst mode has
  # no --in switch to aim it. components.py guards that control with len(by["proj_i1024"]) >= 2, so
  # omitting it costs the tile-width de-confounder only; the proj marginal comes from t1/t2 below.
  local -a plan=( \
    "proj t1" "proj t2" \
    "proj_i1024 tpb1" \
    "attn n1" "attn n2" \
    "mlp_gateup t1" "mlp_gateup t2" "mlp_gateup t4" \
    "mlp_down t1" "mlp_down t2")
  for pair in "${plan[@]}"; do
    comp=${pair%% *}; arm=${pair##* }
    local f="$dir/$phase-$comp-$arm.txt"
    local t0 t1 wall rc
    t0=$(date +%s.%N)
    if [ "$phase" = admit ]; then
      case "$comp" in
        proj)         "$BIN_PROJ" --mode once --plan "$WORK/plan_proj_$arm.txt" --tokens "${arm#t}" \
                        --variant v0-baseline --rows 8 ;;
        proj_i1024)   "$BIN_PROJ" --mode once --plan "$WORK/plan_proj_i1024_t1.txt" --tokens 1 \
                        --tpb "${arm#tpb}" --variant v0-baseline --rows 8 ;;
        attn)         "$BIN_ATTN" --mode once --plan "$WORK/plan_attn_$arm.txt" --launch staged \
                        --softmax online --split "$SPLIT" ;;
        mlp_gateup)   arm_mlp_gateup "${arm#t}" 0 ;;
        mlp_down)     arm_mlp_down   "${arm#t}" 0 ;;
      esac > "$f" 2>&1
    else
      case "$comp" in
        proj)         arm_proj "$arm" ;;
        proj_i1024)   arm_proj_ctrl "${arm#tpb}" ;;
        attn)         arm_attn "$arm" ;;
        mlp_gateup)   arm_gateup "${arm#t}" ;;
        mlp_down)     arm_down   "${arm#t}" ;;
      esac > "$f" 2>&1
    fi
    rc=$?
    t1=$(date +%s.%N); wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.1f", b-a}')
    if [ $rc -ne 0 ]; then
      echo "  FAIL $phase $comp/$arm rc=$rc wall=${wall}s (see $f)"; FAILS=$((FAILS+1)); continue
    fi
    grep -q '^RESULT' "$f" || { echo "  FAIL $phase $comp/$arm: no RESULT line"; FAILS=$((FAILS+1)); continue; }
    if ! check_canary "$f" "$(canary_for "$comp" "$arm" "$phase")"; then
      echo "  NO-TIMING $phase $comp/$arm — canary did not move as required"; FAILS=$((FAILS+1)); continue
    fi
    local ms="no-ms"
    [ "$phase" = admit ] || ms=$(grep '^RESULT' "$f" | tail -1 | grep -oE 'ms(_per_plan)?=[0-9.]+' | tail -1)
    echo "  ok $phase $comp/$arm wall=${wall}s (wall is a by-product, not a measurement) $ms"
  done
}

FAILS=0
echo "-- admission: correctness only. A red line here means NO-TIMING for the whole run. --"
run_phase admit
if [ "$FAILS" -gt 0 ]; then
  echo "NO-TIMING: admission is red ($FAILS arm(s)). Fix the harness before spending a timing window."
  tools/gpu_lock.sh util | sed 's/^/util at exit: /'
  exit 1
fi
if [ "$ADMIT_ONLY" = 1 ]; then
  echo "-- --admit done: no timing taken, no ms figure printed --"
  exit 0
fi

for pass in p1 p2; do
  echo "-- timed pass $pass --"
  run_phase "$pass"
done

U1=$(tools/gpu_lock.sh util 2>/dev/null | tr -s ' ')
echo "== done | util@acquire=$U0 util@release=$U1 | fails=$FAILS | evidence: $EV =="
echo "Next: .venv/bin/python prototype/mtp_delta/components.py  — folds these into delta, states every"
echo "transfer it had to make, and prints the verdict against the pre-registered kill line."
echo "These ARE timings: they may enter engine/performance-status.md only because the main agent took"
echo "them, blocking, with zero subagents active — which is the precondition for running this file."
[ "$FAILS" -eq 0 ]
