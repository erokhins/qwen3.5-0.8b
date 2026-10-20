#!/usr/bin/env bash
# A13 default admission — engine/gate.py over the DERIVED population with NO form flags, i.e. the form the
# engine now ships by default. Companion to `run_a12_admission.sh ARMS="0 1 2"`, which priced the knobs
# against each other; this one proves the default itself, because after a default flip an explicit-flag run
# no longer demonstrates anything about what ships.
#
# Rules carried over verbatim from that harness, both of which were earned the hard way: the population is
# enumerated from `a12_engine_admission.py` (which applies gate.py's own NOT_A_PIN_FIXTURE exclusions), never
# typed by hand; and `rc == 0` is the ONLY PASS, with the explicit `ENGINE GATE FAIL` line required to call a
# FAIL — gate.py's prose contains the literal sentence "18 gate runs, all ENGINE GATE PASS", so a substring
# match calls a red fixture green. A run with no verdict line is NO-VERDICT with its refusal reason carried,
# never a PASS and never a silent skip.
set -u
cd "$(dirname "$0")/../.."
PY=${PY:-.venv/bin/python}
BIN=${BIN:-$PWD/engine/build/decode_run}
TAG=${TAG:-a13-default-admit}
FIXLIST=$($PY prototype/attn_wide/a12_engine_admission.py 2>/dev/null | grep -oE -- '--fixture [A-Za-z0-9_.-]+' | awk '{print $2}' | sort -u)
N=$(echo "$FIXLIST" | wc -w)
echo "default admission (no form flags): $N fixtures, binary $BIN"
tools/gpu_lock.sh acquire "$TAG" || { echo "LOCK FAILED"; exit 1; }
echo "util at acquire: $(tools/gpu_lock.sh util 2>/dev/null)"
pass=0; fail=0; over=0
for f in $FIXLIST; do
  out=$($PY engine/gate.py --fixture "$f" --binary "$BIN" 2>&1); rc=$?
  echo "$out" > "tmp/a13ab/default_gate_$f.txt"
  if [ "$rc" -eq 0 ]; then v=PASS; pass=$((pass+1))
  elif echo "$out" | grep -qE "^ENGINE GATE FAIL"; then v=FAIL; fail=$((fail+1))
  else v="NO-VERDICT (rc=$rc): $(echo "$out" | grep -vE '^\s*$' | tail -1 | cut -c1-110)"; over=$((over+1)); fi
  # The echo is the proof the default routed where we think it did. A default flip that did not land in BOTH
  # binaries (engine and gate) shows up right here, as a side whose echo is `wide` or `fattn`. We count the
  # form names gate.py surfaced rather than claiming silently: 0 means gate.py's report did not carry the name,
  # which is recorded as such and is NOT read as a pass of this check.
  ek=$(grep -cE "wide_llama|wide128llama" "tmp/a13ab/default_gate_$f.txt")
  echo "$f -> $v (form-name lines in gate report: $ek)"
done
echo "util at release: $(tools/gpu_lock.sh util 2>/dev/null)"
tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1
echo "DEFAULT ADMISSION DONE: PASS=$pass FAIL=$fail NO-VERDICT=$over of $N"
[ "$fail" -eq 0 ] && [ "$over" -eq 0 ]
