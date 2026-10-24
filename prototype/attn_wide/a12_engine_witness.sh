#!/usr/bin/env bash
# A12 engine-plumbing witness: the knob-0 inertness byte-compare, and the knob-1 canary line.
#
# WHAT IT PROVES, one per arm:
#   A  the PRE-change binary, default flags        — the reference side of the inertness claim
#   B  the POST-change binary, default flags      — must be BYTE-IDENTICAL to A: knob 0 is inert
#   C  the POST-change binary, --attn-fattn-wide 1 — the `ATTN_WIDE` line must appear, and the run must
#      finish and write its logits (a launch smoke, NOT a numerics verdict and NOT a timing)
#
# WHY A PRE-CHANGE BINARY IS NEEDED AT ALL: `engine/build.sh` is not bit-reproducible on this box, so a
# cmp of two binaries proves only that they are two builds (AGENTS.md, the 2026-10-17 stale-binary row).
# The inertness witness is therefore a dump comparison between two builds run on the same prompt, and the
# canary is a field the change is REQUIRED to move and that a stale binary cannot fake: `ATTN_WIDE` is
# absent from a pre-change log entirely, and the NUMERICS field `attn_fattn_wide=` moves with the flag.
#
# CORRECTNESS ONLY. No timing is taken from this script and none of its numbers may enter a timing doc:
# it runs inside whatever window the caller holds the lock for, and the RESULT lines' ms figures are
# incidental echoes of that. Fixture defaults to `wide520f` (499 prompt rows) because that width is a
# ONE-PASS wave — lanes <= MAXSC — which is the geometry where knob 0 is bit-exact against the pins and
# knob 1 is not, so it exercises both statements in one pair of runs. A fixture above MAXSC (e.g.
# wide1024f) is the pair that additionally shows combine-launched-vs-not, and is the slower one.
#
# Usage (from the repo root; call it holding `tools/gpu_lock.sh`, or let it acquire — it self-acquires and
# releases, unlike `engine/gate.py` which does neither):
#   PRE=tmp/a12plumb/decode_run.prechange FIXTURE=wide520f bash prototype/attn_wide/a12_engine_witness.sh
set -u
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)
cd "$ROOT" || exit 1
PRE=${PRE:?set PRE to a pre-change decode_run binary to compare against}
NEW=${NEW:-engine/build/decode_run}
FIXTURE=${FIXTURE:-wide520f}
STEPS=${STEPS:-2}
OUT=${OUT:-tmp/a12witness}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
TAG=${TAG:-a12-engine-plumbing}

IDS=$(.venv/bin/python -c "import json;print(','.join(str(i) for i in json.load(open('data/refs/$FIXTURE/manifest.json'))['prefill_ids']))")
echo "fixture=$FIXTURE ids=$(echo -n "$IDS" | tr ',' '\n' | wc -l) (argv field $(echo -n "$IDS" | wc -c) bytes; a single argv field above 131,072 B would E2BIG before the binary ran, so a wider fixture must go through --ids-file)"

tools/gpu_lock.sh acquire-noblock "$TAG" || { echo "gpu_lock held by another; nothing ran"; exit 1; }
trap 'tools/gpu_lock.sh release "'"$TAG"'" >/dev/null 2>&1' EXIT
echo "util at acquire: $(tools/gpu_lock.sh util)   (recorded for attribution, NOT as a measurement)"

for arm in A B C; do rm -rf "$OUT/$arm"; mkdir -p "$OUT/$arm"; done

echo "=== A: pre-change binary, default flags ==="
"$PRE" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$IDS" --steps "$STEPS" --out "$OUT/A" > "$OUT/A/log.txt" 2>&1; echo "A exit=$?"
echo "=== B: post-change binary, default flags (knob 0) ==="
"$NEW" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$IDS" --steps "$STEPS" --out "$OUT/B" > "$OUT/B/log.txt" 2>&1; echo "B exit=$?"
echo "=== C: post-change binary, --attn-fattn-wide 1 (knob 1) ==="
"$NEW" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids "$IDS" --steps "$STEPS" --out "$OUT/C" --attn-fattn-wide 1 > "$OUT/C/log.txt" 2>&1; echo "C exit=$?"

echo "=== knob-0 inertness: every dumped file, A vs B ==="
DIFFS=0; N=0
for f in $(cd "$OUT/A" && ls *.bin sampled.txt 2>/dev/null); do
  N=$((N + 1))
  cmp -s "$OUT/A/$f" "$OUT/B/$f" || { echo "DIFFERS: $f"; DIFFS=$((DIFFS + 1)); }
done
echo "files compared: $N, differing: $DIFFS"
[ "$DIFFS" = 0 ] && echo "WITNESS PASS: knob 0 is byte-identical to the pre-change binary on every dump" \
                  || echo "WITNESS FAIL: knob 0 moved bytes it had no right to move"
echo "--- geometry lines must match between A and B ---"
diff <(grep -h "^ATTN_\|^SPLITCAP" "$OUT/A/log.txt" | sort -u) \
     <(grep -h "^ATTN_\|^SPLITCAP" "$OUT/B/log.txt" | sort -u) && echo "geometry lines identical"
grep -h "^NUMERICS" "$OUT/A/log.txt" "$OUT/B/log.txt"
echo "=== knob 1: the canary ==="
grep -h "^ATTN_WIDE" "$OUT/C/log.txt" || { echo "NO ATTN_WIDE LINE: knob 1 did not launch the wide block"; DIFFS=$((DIFFS + 1)); }
grep -h "^NUMERICS" "$OUT/C/log.txt"
cmp -s "$OUT/B/sampled.txt" "$OUT/C/sampled.txt" \
    && echo "NOTE: sampled.txt identical B/C — expected only on a fixture where the wide form cannot differ" \
    || echo "sampled differ B/C (expected: the wide round re-associates the softmax)"
echo "util at release: $(tools/gpu_lock.sh util)"
echo "no timing taken; the RESULT lines' ms fields above are incidental and must not be quoted as a measurement"
exit "$DIFFS"
