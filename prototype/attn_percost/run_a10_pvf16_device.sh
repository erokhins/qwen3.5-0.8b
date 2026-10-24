#!/usr/bin/env bash
# Device pass for the f16 P·V accumulator, one held lock, blocking, zero subagents.
#
#  A. INERTNESS — the new binary at default must be byte-identical to the pre-change binary
#     (/tmp/ab_res/base_run, built from the same tree before the knob existed) on the same prompt. Inertness
#     is witnessed by byte compare, per D4 of A10-F16-IMPL.md, never asserted.
#  B. NUMERICS — same prompt, knob 1 vs knob 0: cos / meanrel / argmax on every logits file plus the sample
#     stream. No band is applied or widened here; the numbers go to the report and to gate.py.
#  C. THROUGHPUT — run separately by run_a10_pvf16_timing.py (paired, interleaved, same binary).
#
# Prompt is 2,048 ids (~8 KB in one argv field), which is both above the row count the WMMA form owns and far
# under the 131,072 B single-argument limit recorded in AGENTS.md.
set -u
cd /home/erokhins/erokhins_projects/qwen3.5-0.8b
PY=.venv/bin/python
tools/gpu_lock.sh acquire pvf16 || { echo "LOCK FAILED"; exit 1; }
echo "util at acquire: $(tools/gpu_lock.sh util 2>/dev/null)"
IDS=$($PY -c "print(','.join(['40'] * 2048))")
run() { # tag binary extra...
  local tag=$1 bin=$2; shift 2
  local out=/tmp/pv_out/$tag; rm -rf "$out"; mkdir -p "$out"
  "$bin" --mtp-accept 0 --mtp-draft 0 --model models/Qwen3.5-0.8B --ids "$IDS" --steps 4 --attn-fattn 1 --out "$out" "$@" \
      >"$out/stdout.txt" 2>&1
  echo "=== $tag rc=$? fattn-owned-lines=$(grep -cv ATTN_FATTN_SKIP "$out/stdout.txt" 2>/dev/null) skip-lines=$(grep -c ATTN_FATTN_SKIP "$out/stdout.txt")"
}
rm -rf /tmp/pv_out
run old0 /tmp/ab_res/base_run
run new0 engine/build/decode_run
run new1 engine/build/decode_run --attn-fattn-pvf16 1
grep -h "NUMERICS" /tmp/pv_out/*/stdout.txt | sed 's/gdn.*//'
echo "--- A. byte compare: pre-change binary vs new default ---"
for f in $(ls /tmp/pv_out/old0 | grep -v stdout); do
  cmp -s "/tmp/pv_out/old0/$f" "/tmp/pv_out/new0/$f" && echo "IDENTICAL $f" || echo "DIFFERS   $f"
done
echo "--- B. numerics: knob 0 vs knob 1 ---"
$PY prototype/attn_percost/a10_pvf16_numerics.py /tmp/pv_out/new0 /tmp/pv_out/new1 2>&1 | tail -10
echo "util at release: $(tools/gpu_lock.sh util 2>/dev/null)"
tools/gpu_lock.sh release pvf16
echo "AB-B-COMPLETE (timing is run_a10_pvf16_timing.py)"
