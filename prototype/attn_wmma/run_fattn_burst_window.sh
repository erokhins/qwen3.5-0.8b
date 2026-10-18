#!/usr/bin/env bash
# run_fattn_burst_window.sh — the device window that decides claim (iii) of D62's funded probe
# (progress/2026-10-18/llama-fattn-anatomy.md §4.2): does `m16_fattn_score` cut the engine's `attn`
# family row by >= 1.5x at 4,096 lanes?
#
# WRITTEN TO BE RUN BY THE MAIN AGENT, blocking, zero subagents, one `gpu_lock` hold. NOTHING HERE HAS
# BEEN RUN. No number in this file, in fattn_score_handover.md, or in fattn_census.output.txt is a
# measurement. Shape is copied from `prototype/attn_wmma/run_attn_attrib_window.sh` for one reason: that
# script already encodes this box's rules (one lock hold for a paired A/B, `gran=stage family=attn` as the
# claim rather than a harness ratio, abort on a null parse, refuse an inert flag).
#
# IT REQUIRES A PORT FIRST. `m16_fattn_score` today is a compile-only probe kernel with no launcher and no
# engine path (see fattn_score_handover.md §5a for the port as one commit). CANARY A below is written to
# fail loudly on an unported binary rather than time something inert.
#
# CANARY RULE (AGENTS.md, "a paired A/B against a stale binary reads as a clean null"). `plan_match` is
# compiled into `decode_run` and cannot see staleness, and `engine/build.sh` is not bit-reproducible here,
# so `cmp` on two binaries proves nothing. Three checks, each read in the same window that produces the
# number:
#   A  flag liveness: `--attn-fattn 1` accepted, `--attn-fattn 7` refused.
#   B  form banner WITH FIELDS: `new` prints ATTN_FATTN form= vgpr= lds= vtiles= and `base` prints none;
#      the vgpr/lds fields must equal the listing's own note block, which catches a binary built from an
#      older copy of the form (the D30 1.7-prime failure: a pre-change binary timed as `new`, every
#      self-consistency field green).
#   C  a bit-difference witness: one logits row per side must DIFFER. An f16 16x16x16 tile dot cannot
#      reproduce a BF16 serial 256-term dot bit for bit, so identical bytes would prove one side stale.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=models/Qwen3.5-0.8B
IDSDIR=progress/bench/evidence/prefill-occupancy-2026-10-16
BIN=engine/build/decode_run
# The listing this price came from. CANARY B reads the expected VGPR/LDS out of it rather than out of this
# script, so a form that moved cannot agree with a stale script.
LIST=tmp/attn_wmma_fattn/wmma_fattn_score-hip-amdgcn-amd-amdhsa-gfx1151.s
OUT=tmp/gcc/attn_fattn/w
TAG=attn-fattn-62
REPS=${REPS:-3}
WIDTHS=${WIDTHS:-"1024 4096"}          # 4,096 is the width §4.2's bar names; 1,024 is the pair
mkdir -p "$OUT"
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
ids_of() { cat "$IDSDIR/ids$1.txt"; }
gmsof() { grep -h "gran=stage family=attn " "$1" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; }
lnsof() { grep -h "gran=stage family=attn " "$1" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1; }
field() { grep -oh "ATTN_FATTN .*" "$1" | head -1 | sed -E "s/.*$2=([0-9m_a-zA-Z]+).*/\1/"; }

[ -f "$BIN" ] || fail "no $BIN - run engine/build.sh first"
[ -f "$LIST" ] || fail "no $LIST - run bash prototype/attn_wmma/build_fattn_score.sh; CANARY B needs the listing's own VGPR/LDS figures"
# The expected values come from the listing's note block, never from this script, so a form that moved
# cannot agree with a stale script.
EXP_VGPR=$(python3 - "$LIST" m16_fattn_score <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for chunk in re.split(r"\n\s*- \.args:", text):
    if re.search(r"^\s*\.name:\s+" + sys.argv[2] + r"$", chunk, re.M):
        print(re.search(r"\.vgpr_count:\s+(\d+)", chunk).group(1)); break
PY
) || fail "cannot read .vgpr_count for m16_fattn_score out of $LIST"
EXP_LDS=$(python3 - "$LIST" m16_fattn_score <<'PY'
import re, sys
text = open(sys.argv[1]).read()
for chunk in re.split(r"\n\s*- \.args:", text):
    if re.search(r"^\s*\.name:\s+" + sys.argv[2] + r"$", chunk, re.M):
        print(re.search(r"\.group_segment_fixed_size:\s+(\d+)", chunk).group(1)); break
PY
) || fail "cannot read .group_segment_fixed_size out of $LIST"

tools/gpu_lock.sh acquire "$TAG" || exit 1
tools/gpu_lock.sh util | tail -1

# ---------------------------------------------------------------------------------------------
echo "== CANARY A: the knob exists and refuses garbage (an unported binary aborts here)"
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-fattn 1 >/dev/null 2>&1 \
    || fail "--attn-fattn is not accepted: the form is not ported (fattn_score_handover.md §5a). Nothing to time."
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-fattn 7 >/dev/null 2>&1 \
    && fail "--attn-fattn accepted 7: the flag is parsed but inert. Refusing to time an inert knob."
echo "CANARY A OK"

# ---------------------------------------------------------------------------------------------
echo
echo "== CANARY B: the form banner, with fields that must match the listing being priced"
for SIDE in 0 1; do
    "$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-fattn "$SIDE" --time-waves \
        > "$OUT/canaryB_$SIDE.log" 2>&1 || fail "CANARY B side $SIDE did not finish"
done
grep -q "ATTN_FATTN" "$OUT/canaryB_1.log" \
    || fail "side --attn-fattn 1 printed no ATTN_FATTN banner: which form did this binary run?"
grep -q "ATTN_FATTN" "$OUT/canaryB_0.log" \
    && fail "side --attn-fattn 0 also printed the banner: the switch does not select the shipped path."
GOT_VGPR=$(field "$OUT/canaryB_1.log" vgpr); GOT_LDS=$(field "$OUT/canaryB_1.log" lds)
GOT_FORM=$(field "$OUT/canaryB_1.log" form)
[ "$GOT_FORM" = "m16_fattn_score" ] || fail "banner says form=$GOT_FORM, expected m16_fattn_score"
[ "$GOT_VGPR" = "$EXP_VGPR" ] || fail "banner vgpr=$GOT_VGPR, listing says $EXP_VGPR: this binary was built from a different copy of the form. This is exactly the stale-'new'-side failure the canary exists for."
[ "$GOT_LDS" = "$EXP_LDS" ] || fail "banner lds=$GOT_LDS, listing says $EXP_LDS: same failure as vgpr, stated separately so a one-field drift names itself"
echo "CANARY B OK (form=$GOT_FORM vgpr=$GOT_VGPR lds=$GOT_LDS, listing agrees)"

# ---------------------------------------------------------------------------------------------
echo
echo "== CANARY C: a bit-difference witness — identical bytes would mean one side is stale"
# 1,024 lanes, not 4,096: `--prefill-logits all` at 4,096 lanes writes ~2.5 GB per side for a claim one
# row already proves (the same cut run_attn_attrib_window.sh W1 makes).
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-fattn 0 --prefill-logits all \
    --out "$OUT/canC_0" > "$OUT/canC_0.log" 2>&1 || fail "CANARY C base run did not finish"
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-fattn 1 --prefill-logits all \
    --out "$OUT/canC_1" > "$OUT/canC_1.log" 2>&1 || fail "CANARY C new run did not finish"
same=0
for d in 0 1 2 3 4 5 6 7; do
    a="$OUT/canC_0/logits$d.bin"; b="$OUT/canC_1/logits$d.bin"
    [ -f "$a" ] && [ -f "$b" ] || fail "CANARY C: logits row $d missing on one side; the witness would be vacuous"
    cmp -s "$a" "$b" && same=$((same + 1))
done
[ "$same" -lt 8 ] || fail "CANARY C: all 8 sampled logits rows are byte-identical across the two forms. A 16x16x16 f16 tile dot cannot reproduce a BF16 serial dot exactly, so this means one side ran the other side's kernel. Not timed."
echo "CANARY C OK ($same of 8 sampled rows identical, expected far fewer than 8)"

# ---------------------------------------------------------------------------------------------
echo
echo "== CLAIM (iii): the attn family row, paired inside one lock hold, best-of-$REPS"
# Admission first, and the rule from D59 applies: a timing harness must refuse to print a millisecond
# figure when the numerics admission is red. The admission is a separate command (handover §5b) because
# it is a gate run, not a timing; run it and confirm green before believing any line printed below.
for R in $(seq 1 "$REPS"); do
    for W in $WIDTHS; do
        for SIDE in 0 1; do
            "$BIN" --model "$MODEL" --ids "$(ids_of "$W")" --steps 0 --time-waves --gemm lt \
                --gemm-pins all --attn-fattn "$SIDE" --out "$OUT/t_${W}_${SIDE}" \
                > "$OUT/t_${W}_${SIDE}_r${R}.log" 2>&1 \
                || fail "timing T=$W side=$SIDE rep=$R did not finish"
            grep -q "attn_fattn=$SIDE" "$OUT/t_${W}_${SIDE}_r${R}.log" \
                || fail "T=$W side=$SIDE: the run does not echo which side it was. Attributing a number to the wrong side is the classic false win; not timed."
        done
    done
done
for W in $WIDTHS; do
    M0=$(for R in $(seq 1 "$REPS"); do gmsof "$OUT/t_${W}_0_r$R.log"; done | sort -g | head -1)
    M1=$(for R in $(seq 1 "$REPS"); do gmsof "$OUT/t_${W}_1_r$R.log"; done | sort -g | head -1)
    [ -n "$M0" ] && [ -n "$M1" ] || fail "T=$W: no family=attn ms parsed; a null parse must not read as a null result"
    python3 - "$M0" "$M1" "$W" <<'PY'
import sys
m0, m1, w = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3]
r = m0 / m1
print(f"T={w}: attn family {m0:.4f} ms (shipped) -> {m1:.4f} ms (m16_fattn_score) = {r:.4f}x")
if r < 1.5:
    sys.exit(f"CLAIM (iii) FAILS at T={w}: {r:.4f}x against a 1.5x bar. Report it as a failure; do not"
             " re-quote the CPU-side census as the answer, and do not re-take the row hoping for a better"
             " window - quote the pair and the spread (bench/performance-status.md: a ruler re-taken in a"
             " different window is not an A/B).")
print(f"CLAIM (iii) PASSES at T={w}")
PY
done

# ---------------------------------------------------------------------------------------------
echo
echo "== CLOSE: release the lock, print the util seen"
tools/gpu_lock.sh util | tail -1
tools/gpu_lock.sh release "$TAG"
echo "done. Numbers above are the only ms figures this artifact produces, and only if every canary above printed OK."
