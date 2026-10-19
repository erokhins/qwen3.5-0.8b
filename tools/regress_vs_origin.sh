#!/usr/bin/env bash
# regress_vs_origin.sh — phase-by-phase (per-family) A/B of the shipping engine against `origin/master`.
#
# WHY THIS EXISTS. The WMMA attention round is 28 commits deep and origin/master is its parent era. A
# total-only before/after would hide the regression that is actually likely on a change like that: something
# OUTSIDE the changed kernel (the one-pass `attn_slim_wave` rows, the split/combine launches, casts, and the
# launcher-side row-block rule of §34, which was already half-committed once). So this compares families,
# paired inside ONE lock hold, because §6 and §36 both established that a ruler re-taken in a different window
# is not an A/B.
#
# MAIN-AGENT DEVICE WINDOW: blocking, zero subagents, one `tools/gpu_lock.sh` hold. It does not self-acquire
# around any other self-locking driver — the only programs it runs are `decode_run` (no self-lock) and, with
# --numerics, `engine/gate.py` (no self-lock either).
#
# Usage:
#   bash tools/regress_vs_origin.sh                 # timing leg, widths 4096 1024, reps 3
#   WIDTHS="4096" REPS=2 bash tools/regress_vs_origin.sh
#   bash tools/regress_vs_origin.sh --numerics      # also run the baseline binary through its own gate.py
#   BASE=origin/master bash tools/regress_vs_origin.sh
#
# CANARIES (the D30 rule: a stale or wrong-binary A/B reads as a CLEAN NULL, so the window asserts a
# behavioural difference BEFORE it reports any number):
#   A  the baseline must NOT understand `--attn-fattn` and the shipping binary must;
#   B  the shipping run must print its `ATTN_FATTN` attribution line;
#   C  the `attn` family must differ between sides by more than 5 %, or the window is DISCARDED, not reported.
set -uo pipefail
cd "$(dirname "$0")/.."
REPO=$PWD
BASE=${BASE:-origin/master}
BASEDIR=${BASEDIR:-/tmp/base-orig}
BIN_NEW=engine/build/decode_run
BIN_OLD=$BASEDIR/engine/build/decode_run
MODEL=models/Qwen3.5-0.8B
IDS=$(cat progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt)
WIDTHS=${WIDTHS:-"4096 1024"}
REPS=${REPS:-3}
TAG=${TAG:-regress-origin}
OUT=tmp/gcc/regress
mkdir -p "$OUT"
RUN_NUMERICS=0
[ "${1:-}" = "--numerics" ] && RUN_NUMERICS=1

COMMON="--steps 0 --time-waves --gemm lt --gemm-pins all --gdn-loop 512"
OLDARGS="--steps 0 --time-waves --gemm lt --gemm-pins all --gdn-loop 512"   # no --attn-fattn: it does not exist there

clk() { rocm-smi --showclocks 2>/dev/null | tr -d '\r' | grep -iE "sclk" | tr '\n' ' '; }
fam() { grep -h "WAVE gran=stage family=$1 " "$2" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; }
tot() { grep -h "WAVE gran=stage TOTAL" "$1" | sed -E 's/.*wall_ms=([0-9.]+).*/\1/' | tail -1; }
res() { grep -h "^RESULT" "$1" | sed -E 's/.*ms_total=([0-9.]+).*/\1/' | tail -1; }
median() { sort -g "$@" | awk '{a[NR]=$1} END{if(NR==0){print "NA";exit} print (NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2}'; }
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

[ -x "$BIN_NEW" ] || fail "shipping binary missing: run engine/build.sh"
if [ ! -x "$BIN_OLD" ]; then
    echo "building baseline $BASE into $BASEDIR (CPU only)"
    [ -d "$BASEDIR" ] || git worktree add "$BASEDIR" "$BASE" --detach -q || fail "worktree add failed"
    ( cd "$BASEDIR" && bash engine/build.sh >/dev/null ) || fail "baseline build failed"
fi

if [ "${SUMMARIZE:-0}" != "1" ]; then
# --- canary A, outside the timing hold: the two binaries must be distinguishable by their flags -------
"$BIN_OLD" --attn-fattn 1 --model "$MODEL" --ids "1 2 3" --steps 0 >/dev/null 2>"$OUT/canaryA_old.err"
grep -q "unknown argument --attn-fattn" "$OUT/canaryA_old.err" \
    || fail "canary A: the BASELINE accepted --attn-fattn. Either $BASE is not the pre-round tree or the wrong binary is under test; nothing is timed."
"$BIN_NEW" --attn-fattn 0 --model "$MODEL" --ids "1 2 3" --steps 0 --out "$OUT/canary_new" >/dev/null 2>&1 \
    || fail "canary A: the SHIPPING binary refused --attn-fattn. Is $BIN_NEW built at all?"
echo "CANARY A ok: baseline rejects --attn-fattn, shipping binary accepts it"
fi

if [ "${SUMMARIZE:-0}" != "1" ]; then
tools/gpu_lock.sh acquire "$TAG" || exit 1
echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
echo "CLK@start: $(clk)"
for W in $WIDTHS; do
  # `--ids` is COMMA-separated (`parse_ids`), so a space-joined `seq` feeds the engine ONE token and the run
  # measures a 1-lane wave. Canary B caught exactly this the first time the script was run.
  IDS_W=$( [ "$W" = "4096" ] && printf '%s' "$IDS" || seq -s, 1 "$W" )
  for R in $(seq 1 "$REPS"); do
    sleep 5
    "$BIN_OLD" --model "$MODEL" --ids "$IDS_W" $OLDARGS --out "$OUT/o_${W}_$R" > "$OUT/o_${W}_$R.log" 2>&1 \
        || fail "baseline run W=$W rep=$R did not finish"
    sleep 5
    "$BIN_NEW" --model "$MODEL" --ids "$IDS_W" $COMMON --out "$OUT/n_${W}_$R" > "$OUT/n_${W}_$R.log" 2>&1 \
        || fail "shipping run W=$W rep=$R did not finish"
    # canary B: the shipping side must say which form it ran
    grep -q "ATTN_FATTN" "$OUT/n_${W}_$R.log" || fail "canary B: shipping log W=$W rep=$R has no ATTN_FATTN line; not timed"
  done
done
echo "CLK@end: $(clk)"
echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
tools/gpu_lock.sh release "$TAG"
fi

# --- summary: per-family medians, paired by width ------------------------------------------------------
FAMS="glue proj mlp_gateup mlp_down gdn_conv gdn_gates gdn_rec gdn_gated_norm attn cast copy lm_head host_tail"
echo
echo "== per-family medians of $REPS reps, one window, paired sides =="
printf "%-16s %10s %10s %8s %8s\n" family base_ms new_ms ratio new_share_pct
for W in $WIDTHS; do
  echo "-- width $W --"
  for F in $FAMS; do
    B=$(for R in $(seq 1 "$REPS"); do fam "$F" "$OUT/o_${W}_${R}.log"; done | grep -v '^$' | median -)
    N=$(for R in $(seq 1 "$REPS"); do fam "$F" "$OUT/n_${W}_${R}.log"; done | grep -v '^$' | median -)
    [ "$B" = "NA" ] || [ "$N" = "NA" ] && continue
    printf "%-16s %10s %10s %8s %8s\n" "$F" "$B" "$N" \
        "$(awk -v a="$B" -v b="$N" 'BEGIN{printf "%.3f", a/b}')" \
        "$(grep -h "family=$F " "$OUT/n_${W}_1.log" | sed -E 's/.*share_pct=([0-9.]+).*/\1/' | tail -1)"
  done
  BT=$(for R in $(seq 1 "$REPS"); do tot "$OUT/o_${W}_${R}.log"; done | median -)
  NT=$(for R in $(seq 1 "$REPS"); do tot "$OUT/n_${W}_${R}.log"; done | median -)
  BR=$(for R in $(seq 1 "$REPS"); do res "$OUT/o_${W}_${R}.log"; done | median -)
  NR=$(for R in $(seq 1 "$REPS"); do res "$OUT/n_${W}_${R}.log"; done | median -)
  printf "%-16s %10s %10s %8s   (wave wall_ms)\n" TOTAL_wall "$BT" "$NT" "$(awk -v a="$BT" -v b="$NT" 'BEGIN{printf "%.3f", a/b}')"
  printf "%-16s %10s %10s %8s   (RESULT ms_total; tok/s new=%s)\n" RESULT \
      "$BR" "$NR" "$(awk -v a="$BR" -v b="$NR" 'BEGIN{printf "%.3f", a/b}')" \
      "$(awk -v n="$NR" -v w="$W" 'BEGIN{printf "%.0f", w*1000/n}')"
  # canary C: the attn family must actually differ
  AB=$(for R in $(seq 1 "$REPS"); do fam attn "$OUT/o_${W}_${R}.log"; done | median -)
  AN=$(for R in $(seq 1 "$REPS"); do fam attn "$OUT/n_${W}_${R}.log"; done | median -)
  awk -v a="$AB" -v b="$AN" -v w="$W" 'BEGIN{d=(a-b)/a*100; if (d<5) {print "CANARY C FAIL: attn moved only " d "% at width " w "; sides may be the same build. NOT REPORTED."; exit 1}}' \
      || fail "canary C failed at width $W"
done
echo "CANARY C ok: attn differs by > 5 % on every width measured"

if [ "$RUN_NUMERICS" = "1" ] && [ "${TIMING_ONLY:-0}" != "1" ]; then
    echo "== numerics leg: the BASELINE binary through its OWN gate.py (HEAD's gate passes --attn-fattn, which it lacks) =="
    for D in data models; do [ -e "$BASEDIR/$D" ] || ln -s "$REPO/$D" "$BASEDIR/$D"; done
    for FX in wide520f wide1024f; do
        echo "-- $FX (baseline) --"
        ( cd "$BASEDIR" && "$REPO/.venv/bin/python" engine/gate.py --fixture "$FX" ) 2>&1 | tail -3
    done
fi
echo "DONE. logs in $OUT/"
