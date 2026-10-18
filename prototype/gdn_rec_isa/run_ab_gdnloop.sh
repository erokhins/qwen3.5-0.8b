#!/usr/bin/env bash
# Paired in-situ A/B for the D53 engine port: `gdn_rec` with N token-steps inside ONE launch against the
# shipping one-launch-per-token form. WRITTEN AND NEVER RUN — the engine port it names does not exist
# yet (the outline is in tmp/gcc/HANDOFF_gdn_looplaunch.md, §6), and this script refuses to time anything
# until the binary proves the flag is live.
#
# THE CLAIM IS THE `gdn_rec` FAMILY ROW, not the wall and not a harness ratio (§19/D51: the family row
# is the instrument, and a paired window's ratio is an attribution, not a headline).
#
# WHY THESE CANARIES, AND WHY NONE OF THEM IS THE PLAN BILL.
#   1. FLAG IS LIVE. A pre-port binary must be caught, not timed (AGENTS: "a paired A/B against a stale
#      binary reads as a clean null"). The base side prints `gdn_loop=1`, the looped side `gdn_loop=512`;
#      an unknown value must be REFUSED. A binary without the flag exits non-zero on `--gdn-loop 512`,
#      and that is the abort.
#   2. BEHAVIOURAL COUNT, NOT A DERIVED BILL. The looped side's *counted* stage-clock row must read
#      ~18 x bodies and the base side's 18 x tokens — `WAVE gran=kernel family=gdn_rec launches=` is the
#      counter at the call site, so a stale binary prints the old number and is caught across the two
#      sides. It is NOT `plan_launches`/`plan_match`, which is a derivation the binary checks against
#      itself and therefore cannot see its own staleness. Both sides must still end `plan_match=1`, which
#      is the port's own obligation, asserted below and not used as the canary.
#   3. BIT-IDENTITY, IN THE WINDOW, LOUD. Every prefill logits row of a real prompt, dumped on both sides
#      with `--prefill-logits all` and byte-compared (`cmp`), before a single timing. If the looped side
#      moved one byte the script aborts and says so; a numerics band is not consulted, because the claim
#      is that no value moved (D53: "if the loop changes a value, the item is wrong, not the tolerance").
#   4. DECODE MUST NOT MOVE. A one-row body must keep the per-token form by the `resolve` rule, exactly as
#      `--attn-tile`/`--gemm` give a one-row body the shipped bits; the decode pair is timed and must be
#      flat, and its `gdn_rec` count must be identical on both sides.
#
# Run blocking, zero subagents, both sides inside ONE `gpu_lock` hold, alternating every rep (§20's
# shape). Nothing here has been run; no number in this file is a measurement.
set -u
cd "$(dirname "$0")/../.."
MODEL=models/Qwen3.5-0.8B
IDSDIR=progress/bench/evidence/prefill-occupancy-2026-10-16
BASE_BIN=engine/build/decode_run          # rebuild from a tree at `--gdn-loop 1` default
NEW_BIN=engine/build/decode_run          # same binary: the loop is a flag, so both sides are one build
LOOP_SPAN=512                            # kMaxWaveRows: one launch per (layer, body)
OUT=tmp/gcc/gdnloop_ab
mkdir -p "$OUT"
TAG=gdnloop_ab
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

ids_of() { cat "$IDSDIR/ids$1.txt"; }

run_side() {  # $1=side(looped|base) $2=width $3=rep $4...=extra flags
    local side=$1 width=$2 rep=$3; shift 3
    local span=1
    [ "$side" = looped ] && span=$LOOP_SPAN
    "$NEW_BIN" --model "$MODEL" --ids "$(ids_of "$width")" --steps 0 --time-waves \
        --gemm lt --gemm-pins all --gdn-loop "$span" --out "$OUT/o_${side}_${width}_${rep}" \
        "$@" > "$OUT/run_${side}_${width}_${rep}.log" 2>&1 || return 1
    echo "$OUT/run_${side}_${width}_${rep}.log"
}

echo "== CANARY 1: the flag is live on this binary (a pre-port build aborts here, loudly)"
"$NEW_BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --gdn-loop 512 >/dev/null 2>&1 \
    || fail "this decode_run does not accept --gdn-loop: the port is not in the binary. Nothing is timed."
"$NEW_BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --gdn-loop notanumber >/dev/null 2>&1 \
    && fail "--gdn-loop accepted a non-number: the flag is inert, not parsed. Refusing to time."
echo "CANARY 1 OK — --gdn-loop is parsed and refused when wrong"

echo
echo "== CANARY 2: the counted submission site moves (behavioural, not the plan bill)"
W=4096
run_side base "$W" pre >/dev/null || fail "pre run, per-token side"
run_side looped "$W" pre >/dev/null || fail "pre run, looped side"
LB="$OUT/run_base_${W}_pre.log"
LL="$OUT/run_loop_${W}_pre.log"
for L in "$LB" "$LL"; do
    grep -q "gdn_loop=" "$L" || fail "$L did not echo gdn_loop= - which side is it?"
    grep "gran=wave TOTAL" "$L" | grep -q "plan_match=1" \
        || fail "$L: plan_match=0. The port must also teach wave_plan_bill the looped shape (outline 6.4)"
done
rec_count() { grep -E 'family=gdn_rec ' "$1" | head -1 | grep -oE 'launches=[0-9]+' | cut -d= -f2; }
base_n=$(rec_count "$LB"); loop_n=$(rec_count "$LL")
echo "counted gdn_rec launches: base=$base_n  looped=$loop_n  (want 18x4096=73728 vs 18x8=144 at $W lanes)"
echo "   base:   $(grep -E 'family=gdn_rec ' "$LB" | head -1)"
echo "   looped: $(grep -E 'family=gdn_rec ' "$LL" | head -1)"
[ "${base_n:-0}" -gt $((10 * ${loop_n:-1})) ] || fail "the looped side did not cut the counted launch"
    " count by 10x - the flag moved nothing, so every timing below would be a clean null about a stale"
    " binary (AGENTS: assert a behavioural canary inside the window)"
echo "CANARY 2 OK - counted submissions fell $base_n -> $loop_n"

echo
echo "== CANARY 3: bit-identity on every prefill row, byte-compared, BEFORE any timing"
rm -rf "$OUT/id_base" "$OUT/id_loop"
"$NEW_BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --prefill-logits all \
    --gemm lt --gemm-pins all --gdn-loop 1  --out "$OUT/id_base" > "$OUT/id_base.log" 2>&1 \
    || fail "identity run, per-token side"
"$NEW_BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 0 --prefill-logits all \
    --gemm lt --gemm-pins all --gdn-loop "$LOOP_SPAN" --out "$OUT/id_loop" > "$OUT/id_loop.log" 2>&1 \
    || fail "identity run, looped side"
nfiles=$(ls "$OUT/id_base"/logits*.bin 2>/dev/null | wc -l)
[ "$nfiles" -gt 1 ] || fail "only $nfiles logits rows on the per-token side: --prefill-logits all did not" \
    " price the wave, so this comparison would be vacuous (compare one row against thousands proves nothing)"
ndiff=0
for f in "$OUT/id_base"/logits*.bin; do
    cmp -s "$f" "$OUT/id_loop/$(basename "$f")" || { ndiff=$((ndiff+1)); echo "DIFFERS: $(basename "$f")"; }
done
[ "$ndiff" -eq 0 ] || fail "NOT BIT-IDENTICAL: $ndiff of $nfiles prefill logits rows differ between" \
    " the per-token and looped forms. The item claims zero value movement — re-transcribe the loop body" \
    " (check_recloop_transcription.py), do not widen a band."
echo "CANARY 3 OK — $nfiles/$nfiles prefill logits rows byte-identical between the two forms"

echo
echo "== lock"
tools/gpu_lock.sh acquire "$TAG" || fail "gpu_lock acquire"
tools/gpu_lock.sh util | tee "$OUT/util.txt"

echo
echo "== the paired wave A/B: medians of 3, both sides inside this one hold, alternating per rep"
for w in 4096 1024; do
    for rep in 1 2 3; do
        for side in base looped; do
            log=$(run_side "$side" "$w" "$rep") || fail "run $side w=$w rep=$rep"
            fam=$(grep -E "family=gdn_rec " "$log" | head -1)
            ctl=$(grep -E "family=attn " "$log" | head -1)
            tot=$(grep -E "gran=wave TOTAL" "$log" | head -1)
            echo "AB w=$w rep=$rep side=$side"
            echo "   ${fam}"
            echo "   ${ctl}"
            echo "   ${tot}"
        done
        tools/gpu_lock.sh util | tee -a "$OUT/util.txt"
    done
done

echo
echo "== decode must be untouched: a one-row body keeps the per-token form by rule, so time it"
for rep in 1 2 3; do
    for span in 1 "$LOOP_SPAN"; do
        "$NEW_BIN" --model "$MODEL" --ids "$(ids_of 1024)" --steps 192 --wave-rows 1 \
            --gemm lt --gemm-pins all --gdn-loop "$span" --out "$OUT/o_dec_${span}_$rep" \
            > "$OUT/dec_${span}_$rep.log" 2>&1 || fail "decode run span=$span rep=$rep"
        echo "decode span=$span rep=$rep: $(grep -oE 'ms_per_token=[0-9.]+|total_ms=[0-9.]+' \
            "$OUT/dec_${span}_$rep.log" | head -2 | tr '\n' ' ')"
    done
done

tools/gpu_lock.sh release "$TAG" || fail "gpu_lock release"
tools/gpu_lock.sh status

echo
echo "== how to read it"
cat <<'TXT'
The claim is the `gdn_rec` family row: median of 3 paired reps, base ms over looped ms.
  > 1.02  clears D25's kill line; the family row is the ship's evidence, and the wall is reported
          separately and honestly (D51: an attribution inside a paired window is not a headline).
  <= 1.02 do not ship. The first suspect is the loop's own +156 slots/token-step (1,530 vs 1,374), whose
          cheapest lever is an address-spelling variant — increment the six row pointers per iteration
          instead of recomputing t*CH — which moves only integer addresses and so stays bit-identical.
Controls that must be flat: `attn` (untouched form), the wall's `plan_match=1`, and the decode pair.
A looped side that wins at 4096 and loses at 1024 is a body-count effect (8 bodies vs 2) and must be
reported that way, not averaged away.
TXT

echo
echo "NEVER RUN BY ITS AUTHOR. Every line above is for the main agent's device window; no timing in this"
echo "file has been read by the worker who wrote it."
