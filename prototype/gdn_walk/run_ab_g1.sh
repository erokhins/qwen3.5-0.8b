#!/usr/bin/env bash
# run_ab_g1.sh — the timing A/B for bet G1 (`--gdn-norm inline|hoist`): three sides, one window.
#
# MAIN-AGENT DEVICE WINDOW ONLY. Blocking, zero subagents, one `tools/gpu_lock.sh` hold, and it runs only
# programs that do NOT self-lock (`decode_run`; `engine/gate.py` self-locks nothing today, but this script
# does not call it). A worker must not run it: it takes timings, and `AGENTS.md` keeps every timing on the
# main agent because this box serves its own tokens on the same iGPU.
#
# WHY THREE SIDES. `base` (HEAD, which has no `--gdn-norm` at all) vs `new inline` must be a NULL — that is
# the leg that proves adding the knob did not move the shipping path. `new inline` vs `new hoist` is the bet
# itself, inside one binary, so no build difference can leak into it. Reporting only the second pair would
# leave the first question ("did the default move?") unanswered, and reporting only a total would hide a
# family that moved the wrong way.
#
# PRE-REGISTERED OUTCOME (from the bet brief, against §37's 266.24 ms at 4,096 lanes):
#     live  <= 250 ms   ·   in the middle 250-258 ms = no verdict   ·   dead > 258 ms
# Compare the `hoist` side's RESULT ms_total. Say which side the number came from, always.
#
# Usage:
#   bash prototype/gdn_walk/run_ab_g1.sh                 # REPS=3, width 4096
#   REPS=2 bash prototype/gdn_walk/run_ab_g1.sh
#   BASE=<sha-or-ref> bash prototype/gdn_walk/run_ab_g1.sh
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
REPO=$PWD
BASE=${BASE:-HEAD}
BASEDIR=${BASEDIR:-/tmp/base-g1}
BIN_NEW=engine/build/decode_run
BIN_OLD=$BASEDIR/engine/build/decode_run
MODEL=models/Qwen3.5-0.8B
W=4096
IDS=$(cat progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt)
REPS=${REPS:-3}
TAG=${TAG:-g1-ab}
OUT=tmp/gcc/g1ab
mkdir -p "$OUT"
ARGS="--steps 0 --time-waves --gemm lt --gemm-pins all --wave-rows 512 --gdn-loop 512"
# Expected `gdn_rec` launch counts at this shape, from prototype/gdn_walk/plan_bill_check.py: 8 bodies of
# 512 rows, one span each, 18 GDN layers -> 18*8 walks inline, +18*8 pre-passes hoisted.
EXPECT_INLINE_LAUNCHES=144
EXPECT_HOIST_LAUNCHES=288

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
    for D in data models; do [ -e "$BASEDIR/$D" ] || ln -s "$REPO/$D" "$BASEDIR/$D"; done
    ( cd "$BASEDIR" && bash engine/build.sh >/dev/null ) || fail "baseline build failed"
fi

if [ "${SUMMARIZE:-0}" != "1" ]; then
# --- canary A, outside the timing hold: the two binaries must be distinguishable -------------------------
"$BIN_OLD" --gdn-norm hoist --model "$MODEL" --ids "1 2 3" --steps 0 >/dev/null 2>"$OUT/canaryA_old.err"
grep -q "unknown argument --gdn-norm" "$OUT/canaryA_old.err" \
    || fail "canary A: the BASELINE accepted --gdn-norm. Either $BASE is not the pre-G1 tree or the wrong binary is under test; nothing is timed."
"$BIN_NEW" --gdn-norm inline --model "$MODEL" --ids "1 2 3" --steps 0 --out "$OUT/canary_new" >/dev/null 2>&1 \
    || fail "canary A: the SHIPPING binary refused --gdn-norm inline. Is $BIN_NEW built at all?"
echo "CANARY A ok: baseline rejects --gdn-norm, shipping binary accepts it"

# --- bit-identity leg, outside the hold: a deterministic check needs no exclusive device -----------------
# The claim being re-proved in THIS window (not carried from prototype/gdn_walk/G1-result.md): base's
# default bits == new's default bits == new's hoisted bits. A short prompt, both prefill logit rows and the
# state fingerprints, byte-compared.
GATE_IDS=$OUT/ids46.txt
.venv/bin/python -c "
import numpy as np, sys
ids = np.load('data/refs/p64/input_ids.npy').reshape(-1)
open('$GATE_IDS','w').write(','.join(str(int(x)) for x in ids))
" || fail "could not build the gate ids from data/refs/p64/input_ids.npy"
for SIDE in base newinline newhoist; do
    case $SIDE in
        base)      BIN=$BIN_OLD;   NORM="" ;;
        newinline) BIN=$BIN_NEW;   NORM="--gdn-norm inline" ;;
        newhoist)  BIN=$BIN_NEW;   NORM="--gdn-norm hoist" ;;
    esac
    "$BIN" --model "$MODEL" --ids "$(cat "$GATE_IDS")" --steps 0 --prefill-logits all --state-fp \
        $NORM --wave-rows 16 --gdn-loop 8 --out "$OUT/bi_$SIDE" > "$OUT/bi_$SIDE.log" 2>&1 \
        || fail "bit-identity leg: $SIDE did not finish"
done
for f in "$OUT/bi_base"/*.bin; do
    b=$(basename "$f")
    cmp -s "$f" "$OUT/bi_newinline/$b" || fail "bit-identity: --gdn-norm inline differs from the BASELINE on $b. The default path moved; stop."
    cmp -s "$f" "$OUT/bi_newhoist/$b"  || fail "bit-identity: --gdn-norm hoist differs from the BASELINE on $b. The bet is not bit-neutral; do not time it, report it."
done
for k in newinline newhoist; do
    grep -h "^STATEFP" "$OUT/bi_base.log" > "$OUT/bi_base.fp"
    grep -h "^STATEFP" "$OUT/bi_$k.log" > "$OUT/bi_$k.fp"
    cmp -s "$OUT/bi_base.fp" "$OUT/bi_$k.fp" || fail "bit-identity: STATEFP differs on the $k side (the GDN state moved)."
done
echo "BIT-IDENTITY ok: base == inline == hoisted on every logit row and every state fingerprint"

tools/gpu_lock.sh acquire "$TAG" || exit 1
echo "WINDOW $TAG util@acquire=$(tools/gpu_lock.sh util | tail -1)"
echo "CLK@start: $(clk)"
for R in $(seq 1 "$REPS"); do
    sleep 5
    "$BIN_OLD" --model "$MODEL" --ids "$IDS" $ARGS --out "$OUT/base_$R" > "$OUT/base_$R.log" 2>&1 \
        || fail "baseline run rep=$R did not finish"
    sleep 5
    "$BIN_NEW" --model "$MODEL" --ids "$IDS" $ARGS --gdn-norm inline --out "$OUT/inl_$R" > "$OUT/inl_$R.log" 2>&1 \
        || fail "inline run rep=$R did not finish"
    sleep 5
    "$BIN_NEW" --model "$MODEL" --ids "$IDS" $ARGS --gdn-norm hoist --out "$OUT/hoi_$R" > "$OUT/hoi_$R.log" 2>&1 \
        || fail "hoist run rep=$R did not finish"
    # canary B, inside the window: the sides must say which form they ran, and submit what the plan says.
    grep -q "gdn_norm=inline" "$OUT/inl_$R.log" || fail "canary B: rep=$R inline side did not echo gdn_norm=inline"
    grep -q "gdn_norm=hoist"  "$OUT/hoi_$R.log" || fail "canary B: rep=$R hoist side did not echo gdn_norm=hoist"
    for PAIR in "inl_$R $EXPECT_INLINE_LAUNCHES" "hoi_$R $EXPECT_HOIST_LAUNCHES"; do
        set -- $PAIR
        LOG=$1; WANT=$2
        GOT=$(grep -h "WAVE gran=stage family=gdn_rec " "$OUT/$LOG.log" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1)
        PLAN=$(grep -h "WAVE gran=stage family=gdn_rec " "$OUT/$LOG.log" | sed -E 's/.* plan_launches=([0-9]+).*/\1/' | tail -1)
        MATCH=$(grep -h "WAVE gran=stage family=gdn_rec " "$OUT/$LOG.log" | sed -E 's/.* counts_match=([01]).*/\1/' | tail -1)
        [ "$GOT" = "$WANT" ] || fail "canary B: $LOG observed gdn_rec launches=$GOT, expected $WANT. The sides are not the two forms; nothing is reported."
        [ "$PLAN" = "$WANT" ] || fail "canary B: $LOG plan bill says $PLAN, expected $WANT (stage_clock.cpp's derivation is wrong)."
        [ "$MATCH" = "1" ] || fail "canary B: $LOG counts_match=0: the plan and the submissions disagree."
    done
    grep -q "TOTAL.*plan_match=1" "$OUT/hoi_$R.log" || fail "canary B: hoist side plan_match!=1 at rep=$R"
done
echo "CLK@end: $(clk)"
echo "util@release=$(tools/gpu_lock.sh util | tail -1)"
tools/gpu_lock.sh release "$TAG"
fi

# --- summary: per-family medians across the three sides, paired by rep -----------------------------------
echo
echo "== $W lanes, $REPS reps, one window. Outcome line: hoist RESULT ms_total vs base (266.24 ms at §37)."
printf "%-16s %10s %10s %10s %8s %8s\n" family base_ms new_inline_ms new_hoist_ms ratio_new/base new_share_pct
FAMS="gdn_rec gdn_conv gdn_gates gdn_gated_norm glue proj mlp_gateup mlp_down attn cast copy lm_head host_tail"
for F in $FAMS; do
    B=$(for R in $(seq 1 "$REPS"); do fam "$F" "$OUT/base_$R.log"; done | grep -v '^$' | median -)
    I=$(for R in $(seq 1 "$REPS"); do fam "$F" "$OUT/inl_$R.log"; done | grep -v '^$' | median -)
    H=$(for R in $(seq 1 "$REPS"); do fam "$F" "$OUT/hoi_$R.log"; done | grep -v '^$' | median -)
    [ "$B" = "NA" ] && continue
    printf "%-16s %10s %10s %10s %8s %8s\n" "$F" "$B" "$I" "$H" \
        "$(awk -v a="$B" -v b="$H" 'BEGIN{printf "%.3f", a/b}')" \
        "$(grep -h "family=$F " "$OUT/hoi_1.log" | sed -E 's/.*share_pct=([0-9.]+).*/\1/' | tail -1)"
done
BT=$(for R in $(seq 1 "$REPS"); do tot "$OUT/base_$R.log"; done | median -)
IT=$(for R in $(seq 1 "$REPS"); do tot "$OUT/inl_$R.log"; done | median -)
HT=$(for R in $(seq 1 "$REPS"); do tot "$OUT/hoi_$R.log"; done | median -)
BR=$(for R in $(seq 1 "$REPS"); do res "$OUT/base_$R.log"; done | median -)
IR=$(for R in $(seq 1 "$REPS"); do res "$OUT/inl_$R.log"; done | median -)
HR=$(for R in $(seq 1 "$REPS"); do res "$OUT/hoi_$R.log"; done | median -)
printf "%-16s %10s %10s %10s   (wave wall_ms)\n" TOTAL_wall "$BT" "$IT" "$HT"
printf "%-16s %10s %10s %10s   (RESULT ms_total; tok/s hoist=%s)\n" RESULT \
    "$BR" "$IR" "$HR" "$(awk -v n="$HR" -v w="$W" 'BEGIN{printf "%.0f", w*1000/n}')"
# BET.md SS3 pre-registered the lines against the `gdn_rec` FAMILY ROW (base 266.24 ms), not against
# RESULT. The first version judged RESULT here and printed "dead: 707.88 vs 717.27", which is not the
# quantity the lines were written for: an outcome line and its metric have to be the same number, or the
# pre-registration is decoration.
GR_B=$(for R in $(seq 1 "$REPS"); do fam gdn_rec "$OUT/base_$R.log"; done | grep -v '^$' | median -)
GR_I=$(for R in $(seq 1 "$REPS"); do fam gdn_rec "$OUT/inl_$R.log"; done | grep -v '^$' | median -)
GR_H=$(for R in $(seq 1 "$REPS"); do fam gdn_rec "$OUT/hoi_$R.log"; done | grep -v '^$' | median -)
# The externality a single row cannot see: cost that lands in families OTHER than the one improved.
EXT=0
for F in $FAMS; do
    [ "$F" = "gdn_rec" ] && continue
    fb=$(for R in $(seq 1 "$REPS"); do fam "$F" "$OUT/base_$R.log"; done | grep -v '^$' | median -)
    fh=$(for R in $(seq 1 "$REPS"); do fam "$F" "$OUT/hoi_$R.log"; done | grep -v '^$' | median -)
    [ "$fb" = "NA" ] || [ "$fh" = "NA" ] && continue
    EXT=$(awk -v e="$EXT" -v b="$fb" -v h="$fh" 'BEGIN{printf "%.3f", e + (h - b)}')
done

awk -v b="$GR_B" -v i="$GR_I" -v h="$GR_H" -v bw="$BT" -v hw="$HT" -v ext="$EXT" 'BEGIN{
    printf "pre-registered metric = gdn_rec family row: base %.2f | new inline %.2f | hoist %.2f\n", b, i, h;
    printf "   inline-vs-base null : %+.2f ms (%+.2f %%)  must be inside window noise, else the change moved the shipped path\n", i - b, 100*(i-b)/b;
    printf "   hoist-vs-base       : %+.2f ms (%+.2f %%)  against the lines live<=250, dead>258\n", h - b, 100*(h-b)/b;
    printf "   cost outside gdn_rec: %+.2f ms summed over every other family\n", ext;
    printf "   wave wall_ms        : %.2f -> %.2f (%+.2f %%)  what the whole change wins\n", bw, hw, 100*(hw-bw)/bw;
    if (h <= 250)      printf "VERDICT live: gdn_rec %.2f ms <= 250\n", h;
    else if (h > 258)  printf "VERDICT dead: gdn_rec %.2f ms > 258\n", h;
    else               printf "VERDICT UNRESOLVED: gdn_rec %.2f ms is in the pre-registered 250-258 no-verdict band -> do not ship, do not narrate\n", h;
}'

echo "Also report: the inline-vs-base NULL (RESULT medians must agree within window noise) — if it is not a"
echo "NULL, the change moved the shipping path and the hoist number means nothing. Quote the pair, the spread"
echo "and the clocks; §6's dispatch-bound bimodality swallows short-prompt rows."
echo "DONE. logs in $OUT/"
