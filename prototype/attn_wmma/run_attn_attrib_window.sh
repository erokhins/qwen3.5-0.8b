#!/usr/bin/env bash
# run_attn_attrib_window.sh — the device window that closes queue item 2.6's attribution.
#
# WRITTEN TO BE RUN BY THE MAIN AGENT, blocking, zero subagents, one `gpu_lock` hold. Nothing here
# has been run; no number in this file or in HANDOFF_attn_attrib.md is a measurement taken by this
# script. It is CPU-side work: census, occupancy arithmetic, and reading the trace the D52 window
# already left on disk.
#
# WHY EACH COMMAND EXISTS (detail and the arithmetic: tmp/gcc/HANDOFF_attn_attrib.md)
#   W1  Re-derives the touch-bill elasticity IN SITU. Every share I quote for "how much of `attn`
#       is the pass-1 score stream" is currently derived from a prototype harness
#       (prototype/attn_valu_slim/results.md §7.1: 116 -> 45 slots/touch bought 1.52-1.57x) and
#       from a frozen-witness A/B, not from the engine's own family row. §10/§24's standing rule is
#       that the family row is the instrument and a harness ratio is an attribution.
#   W2  Settles a DENOMINATOR COLLISION that is load-bearing for three prior prices:
#       `prototype/attn_valu_slim` bills a touch at 116 *slots* (pass-1 loop only, of which 55 are
#       VALU), and `progress/2026-10-16/prefill-occupancy-results-2026-10-16.md` §3 measured 114.6
#       *VALU warp-inst per touch* for the WHOLE kernel (2,888 M / 25,190,400 touches). Those two
#       numbers differ by 1.2 % and mean different things, and `prototype/attn_score_dot2/price.py`
#       uses the second as the denominator while the slim elasticity uses the first. Two mutually
#       exclusive predictions for `attn_slim_wave` at 1024 lanes: 2,207 M (whole-kernel VALU,
#       114.6-27 = 87.6/touch) vs 705 M (pass-1 VALU only, 28/touch). 3.1x apart: one counter run
#       decides it, and it decides whether the pass-1 share of `attn` is ~56 % or ~24 %.
#       CALIBRATION FIRST, per the GL2C lesson in that same document: the `--attn-touch 0` side must
#       reproduce the published 2,888 M within 5 % before the slim side's number is believed.
#   W3  Realized occupancy. My host-side calculator says 5 blocks / 20 wavefronts per CU (62.5 % of
#       the 32 the part allows), LDS-bound. A counter witness is owed because §3 of the occupancy
#       document already retracted one occupancy claim (`SQ_WAVE_CYCLES/SQ_BUSY_CYCLES` read 41
#       "against a max of 64" when the architectural max is 32).
#   W4  Corrects a launch-count claim that was inferred, not counted: §22's row "attention launches
#       per pass — 48 both sides, identical" is `6 attn layers x 8 waves`, which is the engine's own
#       counted SITE (launchstats::attn records one per layer per body). Counting dispatches in the
#       D52 trace that is already on disk gives 138 per pass (48 attn_prep + 48 attn_slim_wave +
#       42 attn_combine). The prize is unaffected (138 x 1.72 us = 0.24 ms of a 763 ms wave), but
#       the row in the ledger is wrong and it was load-bearing for "this is not a launch-count item".
#   W5  Refutes the one queue candidate that is not a speed lever at all: `kAttnMaxSplit = 9` bounds
#       the LANE CEILING at MAXSC x 9 = 4,608, and the funded ruler is 4,096 lanes (split = 8), so
#       raising it cannot move a single millisecond at the width we measure. It is a long-context
#       enabler and must be argued as one.
#
# CANARY RULE (AGENTS.md, "a paired A/B against a stale binary reads as a clean null"): every timed
# side asserts a field the change is REQUIRED to move, in the window that produces the number, and
# aborts the script when it does not read as expected. `plan_match` is never the canary.
set -uo pipefail
cd "$(dirname "$0")/../.."
MODEL=models/Qwen3.5-0.8B
IDSDIR=progress/bench/evidence/prefill-occupancy-2026-10-16
BIN=engine/build/decode_run
OUT=tmp/gcc/attn_attrib/w
TAG=attn-attrib-26
mkdir -p "$OUT"
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
ids_of() { cat "$IDSDIR/ids$1.txt"; }
gmsof() { grep -h "gran=stage family=attn " "$1" | sed -E 's/.* ms=([0-9.]+).*/\1/' | tail -1; }
lnsof() { grep -h "gran=stage family=attn " "$1" | sed -E 's/.* launches=([0-9]+).*/\1/' | tail -1; }

[ -f "$BIN" ] || fail "no $BIN — run engine/build.sh first"

tools/gpu_lock.sh acquire "$TAG" || exit 1
tools/gpu_lock.sh util | tail -1

# ---------------------------------------------------------------------------------------------
echo "== CANARY 0: the two A/B knobs are live and refuse garbage (a pre-knob binary aborts here)"
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-touch 1 >/dev/null 2>&1 \
    || fail "--attn-touch is not accepted by this binary; W1/W2 have no two sides to compare"
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-touch 7 >/dev/null 2>&1 \
    && fail "--attn-touch accepted 7: the flag is inert, not parsed. Refusing to time."
"$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --attn-tile 0 >/dev/null 2>&1 \
    || fail "--attn-tile is not accepted either; the frozen witness side cannot be run"
echo "CANARY 0 OK"

# ---------------------------------------------------------------------------------------------
echo
echo "== W1: paired in-situ A/B of the touch spelling, family row = the claim"
# Bit-identity first, loud, before any timing: the slim/shipped pair's whole admission is that no
# value moved, so a band is not consulted and one moved byte voids the window. It runs at 1,024
# lanes ONLY: `--prefill-logits all` at 4,096 lanes would write ~2.5 GB per side (4,096 rows x the
# full vocab in f32) for a claim that one width already proves, and §24's shipped admission was taken
# on 1,024 rows too.
for W in 1024; do
    for SIDE in 0 1; do
        "$BIN" --model "$MODEL" --ids "$(ids_of "$W")" --steps 0 --time-waves --gemm lt \
            --gemm-pins all --attn-touch "$SIDE" --prefill-logits all \
            --out "$OUT/w1_logits_$W$SIDE" > "$OUT/w1_${W}_$SIDE.log" 2>&1 \
            || fail "W1 run T=$W side=--attn-touch $SIDE did not finish"
    done
    grep -q "attn_touch=1" "$OUT/w1_${W}_1.log" || fail "$OUT/w1_${W}_1.log does not echo attn_touch=1 - which side is it?"
    grep -q "attn_touch=0" "$OUT/w1_${W}_0.log" || fail "$OUT/w1_${W}_0.log does not echo attn_touch=0"
    for d in $(seq 0 $((W - 1))); do
        # Main-agent fix, this window's first run: the original line read "$OUT/w1_logits_$W$1/...",
        # so under `set -u` the script died on an unbound positional; and the engine writes each row as
        # `logits<i>.bin` (`decode_run.cpp:1489`), not `prefill_logits_<i>.npy` — the naming the gate
        # uses. Both fixed here; the row count is still required to exist, so the leg cannot be vacuous.
        a="$OUT/w1_logits_${W}0/logits$d.bin"; b="$OUT/w1_logits_${W}1/logits$d.bin"
        [ -f "$a" ] && [ -f "$b" ] || fail "T=$W: a prefill logits row is missing (row $d); the" \
            "bit-identity leg cannot be vacuous (AGENTS: a must-catch set derived from geometry)"
        cmp -s "$a" "$b" || fail "T=$W row $d: --attn-touch moved a logit. The pair is no longer" \
            "bit-identical and nothing downstream of this line may be priced as band-free."
    done
    echo "T=$W: all $W prefill logits rows byte-identical between the two spellings"
done
for R in 1 2 3; do
  for W in 1024 4096; do
    for SIDE in 0 1; do
        "$BIN" --model "$MODEL" --ids "$(ids_of "$W")" --steps 0 --time-waves --gemm lt \
            --gemm-pins all --attn-touch "$SIDE" --out "$OUT/w1t_${W}_${SIDE}" \
            > "$OUT/w1t_${W}_${SIDE}_r${R}.log" 2>&1 \
            || fail "W1 timing run T=$W side=$SIDE rep=$R did not finish"
    done
  done
done
for W in 1024 4096; do
    M0=$(for R in 1 2 3; do gmsof "$OUT/w1t_${W}_0_r$R.log"; done | sort -g | head -1)
    M1=$(for R in 1 2 3; do gmsof "$OUT/w1t_${W}_1_r$R.log"; done | sort -g | head -1)
    L0=$(lnsof "$OUT/w1t_${W}_0_r1.log"); L1=$(lnsof "$OUT/w1t_${W}_1_r1.log")
    [ -n "$M0" ] && [ -n "$M1" ] || fail "T=$W: no family=attn ms parsed; a null parse must not read as a null result"
    [ "$L0" = "$L1" ] || fail "T=$W: launches= moved ($L0 vs $L1) across an A/B that only changes a" \
        "spelling inside the kernel. Something else moved; the pair is not one-window."
    python3 - "$M0" "$M1" "$W" "$L0" <<'PY'
import sys
m0, m1, w, l = float(sys.argv[1]), float(sys.argv[2]), sys.argv[3], sys.argv[4]
r = m0 / m1
print(f"T={w}: attn family {m0:.4f} ms (tile, --attn-touch 0) -> {m1:.4f} ms (slim, 1) = {r:.4f}x"
      f"   launches={l} (identical both sides)")
if r < 1.05:
    sys.exit(f"ABORT: T={w}: the family row moved only {r:.4f}x; the knob is inert in situ and every"
             " share derived from the harness elasticity must be re-priced before it is used again")
PY
done
echo "W1 OK"

# ---------------------------------------------------------------------------------------------
echo
echo "== W2: counter census that settles the per-touch denominator (CALIBRATED, both counters)"
for SIDE in 0 1; do
    rm -rf "$OUT/pmc$SIDE"
    # The regex is `attn_` on purpose, not `attn_slim_wave`: attn_prep and attn_combine are 90 of the
    # 138 dispatches this family submits per pass and nobody has ever priced their share of the row.
    # Getting their SQ_INSTS for free, in the same window, is the only route to the prep/core/combine
    # split that every share in this attribution silently assumes.
    timeout 1200 rocprofv3 --kernel-include-regex "attn_" --pmc \
        SQ_INSTS SQ_INSTS_VALU SQ_WAVES -f csv -d "$OUT/pmc$SIDE" -o "pmc$SIDE" -- \
        "$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --time-waves --gemm lt \
        --gemm-pins all --attn-touch "$SIDE" > "$OUT/w2_$SIDE.log" 2>&1
    echo "side $SIDE rc=$? -> $(find "$OUT/pmc$SIDE" -name '*counter*.csv' | head -1)"
done
cat > "$OUT/w2_expect.py" <<'PY'
# Dedup by Dispatch_Id (the §4 lesson: Correlation_Id is per-API-call and repeats across kernels),
# then adjudicate the two competing models. No timing is read, and no ms figure is printed.
import csv, glob, sys
TOUCHES = 1024 * 1025 // 2 * 8 * 6      # sum_lanes x 8 query heads x 6 attn layers = 25,190,400
EXPECT = {   # warp-inst per whole kernel, per touch, x touches
    "calibration attn_tile_wave (published row)": 2_888e6,
    "model A  attn_slim_wave, whole-kernel VALU": (114.6 - 27) * TOUCHES,
    "model B  attn_slim_wave, pass-1 VALU only":  28 * TOUCHES,
}
print(f"touches at 1024 lanes = {TOUCHES:,}")
for k, v in EXPECT.items():
    print(f"  {k:44} {v/1e6:9.1f} M")
rows = []
for f in glob.glob(sys.argv[1] + "/*counter*.csv"):
    with open(f, newline="") as fh:
        rd = list(csv.DictReader(fh))
    seen, tot = set(), {}
    for r in rd:
        k = r.get("Kernel_Name") or r.get("kernel_name") or "?"
        i = int(float(r.get("SQ_INSTS_VALU", 0)))
        key = (r.get("Dispatch_Id") or r.get("dispatch_id"), k, r.get("Tag") or r.get("tag") or 0)
        if key in seen:
            continue
        seen.add(key)
        tot[k] = tot.get(k, 0) + i
    for k, v in tot.items():
        rows.append((f.rsplit("/", 1)[-1], k, v))
if not rows:
    sys.exit("ABORT: the pmc CSV yielded zero kernel rows — a check whose input list can be empty "
             "cannot fail (AGENTS), so nothing here is adjudicated")
print()
for f, k, v in rows:
    print(f"  {f:16} {k:28} SQ_INSTS_VALU = {v/1e6:9.1f} M")
cal = [v for _, k, v in rows if "tile_wave" in k]
if not cal:
    sys.exit("ABORT: no attn_tile_wave row, so the counter is uncalibrated and the slim side means nothing")
if abs(cal[0] - 2_888e6) > 0.05 * 2_888e6:
    sys.exit(f"ABORT: attn_tile_wave read {cal[0]/1e6:.1f} M against the published 2,888 M — the"
             " counter/window does not reproduce a known row, so per the GL2C lesson it cannot"
             " adjudicate the slim side either")
slim = [v for _, k, v in rows if "slim_wave" in k]
if not slim:
    sys.exit("ABORT: no attn_slim_wave row (the include regex matched nothing on the slim side)")
n, a, b = slim[0], EXPECT["model A  attn_slim_wave, whole-kernel VALU"], EXPECT["model B  attn_slim_wave, pass-1 VALU only"]
print(f"\nCALIBRATED. attn_slim_wave = {n/1e6:.1f} M; model A {a/1e6:.0f} M, model B {b/1e6:.0f} M")
print("VERDICT:", "A (whole-kernel VALU ~87.6/touch -> pass-1 is ~56% of attn)"
      if abs(n - a) < abs(n - b) else "B (pass-1 slots ~45/touch -> the 114.6 denominator is not whole-kernel)")
PY
python3 "$OUT/w2_expect.py" "$OUT/pmc1" || echo "W2 did not adjudicate (see the reason above); do not" \
    "carry any pass-1 share forward until it does"

# ---------------------------------------------------------------------------------------------
echo
echo "== W3: realized occupancy witness (bounded; retract it if it cannot be calibrated)"
rm -rf "$OUT/occ"
timeout 1200 rocprofv3 --kernel-include-regex "attn_slim_wave" --pmc \
    SQ_LEVEL_WAVES SQ_WAVES SQ_BUSY_CYCLES SQ_WAVE_CYCLES GRBM_COUNT -f csv -d "$OUT/occ" \
    -o occ -- "$BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --time-waves --gemm lt \
    --gemm-pins all --attn-touch 1 > "$OUT/w3.log" 2>&1
echo "rc=$? files: $(find "$OUT/occ" -name '*counter*.csv' | wc -l)"
echo "HAND RULES (no threshold is invented here, and a counter that cannot reproduce a known"
echo " population is void per the GL2C lesson):"
echo "  1. SQ_WAVES must equal blocks x 4 wavefronts, where blocks is read from the SAME trace's"
echo "     grid fields (grid X is in THREADS in rocprofv3, so blocks = Grid_Size_X/Workgroup_Size_X"
echo "     x Grid_Size_Y x Grid_Size_Z). If it does not, the instance set is unknown and §3's"
echo "     retracted occupancy claim repeats."
echo "  2. whatever wavefronts/CU comes out must be <= 20, the static cap this window computes from"
echo "     KFD lds_size_in_kb 64 / 12,288 B per block = 5 blocks x 4 warps."
echo "  3. if SQ_LEVEL_WAVES is not a valid counter on gfx1151, say so in the ledger; do not"
echo "     substitute SQ_WAVE_CYCLES/SQ_BUSY_CYCLES, which is the ratio already retracted."

# ---------------------------------------------------------------------------------------------
echo
echo "== W4: launch-count census, counted instead of inferred (both engines, same hold)"
for E in engine llama; do
    rm -rf "$OUT/trace_$E"
    timeout 1200 rocprofv3 --kernel-trace -f csv -d "$OUT/trace_$E" -o t -- \
        $( [ "$E" = engine ] && printf '%s' \
            "$BIN --model $MODEL --ids $(ids_of 4096) --steps 0 --time-waves --gemm lt --gemm-pins all" \
           || printf '%s' \
            'engines/llama.cpp/build-hip/bin/llama-bench -m models/gguf/Qwen3.5-0.8B.f16.gguf -p 4096 -n 0 -ub 512 -fa on -ngl 99 -t 32 -b 2048 -r 1 -o csv' \
           ) > "$OUT/w4_$E.stdout" 2>&1
    echo "$E rc=$?"
done
for E in engine llama; do
    CSV=$(find "$OUT/trace_$E" -name '*kernel_trace*.csv' | head -1)
    [ -n "$CSV" ] || { echo "W4 $E: no trace"; continue; }
    echo "-- $E attention-family dispatch instances (deduped by Dispatch_Id, printed count included)"
    python3 - "$CSV" <<'PY'
import csv, sys
seen, tot = set(), {}
for r in csv.DictReader(open(sys.argv[1], newline="")):
    if r["Kind"] != "KERNEL_DISPATCH":
        continue
    if (r["Dispatch_Id"], r["Kernel_Name"]) in seen:
        continue
    seen.add((r["Dispatch_Id"], r["Kernel_Name"]))
    tot[r["Kernel_Name"]] = tot.get(r["Kernel_Name"], 0) + 1
hit = {k: v for k, v in tot.items() if "attn" in k or "flash" in k}
print("   total distinct dispatches:", len(seen))
for k, v in sorted(hit.items(), key=lambda kv: -kv[1]):
    print(f"   {k:46} {v}")
print("   attention-family total:", sum(hit.values()))
print("   MUST-READ: ours must be >= 3x the 6 layers x 8 waves the ledger inferred (138 on the D52"
      " trace); a result of exactly 48 means the dedup collapsed the passes, not that the count is 48")
PY
done

# ---------------------------------------------------------------------------------------------
echo
echo "== W5: refutation, not a measurement — kAttnMaxSplit is a lane CEILING, not a throttle"
echo "   4,096 lanes runs (split = 8 of kAttnMaxSplit = 9); 4,609 must be REFUSED by name."
"$BIN" --model "$MODEL" --ids "$(ids_of 20480)" --steps 0 > "$OUT/w5.log" 2>&1 \
    && echo "  UNEXPECTED: a 20,480-lane prompt was accepted - the lane ceiling moved, re-read the source"
grep -qiE "kAttnMaxSplit|lane ceiling" "$OUT/w5.log" \
    && echo "  as expected: refused with the named lane-ceiling error ($(grep -m1 -oiE 'kAttnMaxSplit=[0-9]+' "$OUT/w5.log"))" \
    || echo "  W5 did not print the named refusal; read $OUT/w5.log before concluding anything"
echo "  No timing here and none taken: this row exists to show raising 9 cannot move 4,096 lanes."

tools/gpu_lock.sh util | tail -1
tools/gpu_lock.sh release "$TAG"
echo "WINDOW DONE. Every ms figure in this window is a traced or timed number for the MAIN AGENT to"
echo "read and enter in engine/performance-status.md; nothing in this script is a ledger row."
