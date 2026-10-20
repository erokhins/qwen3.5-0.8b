#!/usr/bin/env bash
# A12 numerics admission: engine/gate.py over the DERIVED fixture population, both arms, one held lock.
# Main agent only, blocking, zero subagents. The population is enumerated from the admission script's own
# derived list (which applies gate.py's NOT_A_PIN_FIXTURE exclusions), never typed by hand — three gate
# admissions in this repo have already claimed counts off a hand-listed population that did not exist.
# A fixture whose gate cannot be parsed is reported NO-VERDICT, never counted as PASS.
set -u
cd /home/erokhins/erokhins_projects/qwen3.5-0.8b
PY=.venv/bin/python
BIN=$PWD/engine/build/decode_run
FIXLIST=$($PY prototype/attn_wide/a12_engine_admission.py 2>/dev/null | grep -oE -- '--fixture [A-Za-z0-9_.-]+' | awk '{print $2}' | sort -u)
N=$(echo "$FIXLIST" | wc -w)
echo "derived population: $N fixtures"
tools/gpu_lock.sh acquire a12-admit || { echo "LOCK FAILED"; exit 1; }
echo "util at acquire: $(tools/gpu_lock.sh util 2>/dev/null)"
for f in $FIXLIST; do
  # A13 reuses this harness rather than forking a second one: run it as `ARMS="0 1 2"` to judge the
  # llama-ported form beside the shipper and A12 in one lock hold, from one derived population.
  for arm in ${ARMS:-0 1}; do
    if [ "$arm" = "0" ]; then out=$($PY engine/gate.py --fixture "$f" --binary "$BIN" 2>&1); else out=$($PY engine/gate.py --fixture "$f" --binary "$BIN" --attn-fattn-wide "$arm" 2>&1); fi
    rc=$?
    # The exit code is the verdict; a substring match on the report text is not. This line was a live false
    # positive in its first version: gate.py's long census paragraph contains the literal words
    # "all ENGINE GATE PASS" (describing a historical 18-run set), so grep -q "GATE PASS" called a red fixture
    # PASS while its rc said 1 — the exact "a parser that matches something unrelated reports PASS" trap this
    # repo has already recorded three times. rc==0 is PASS; anything else carries the rc.
    if [ "$rc" -eq 0 ]; then v=PASS
    elif echo "$out" | grep -qE "^ENGINE GATE FAIL"; then v=FAIL
    else v="EXIT $rc (no verdict line)"; fi
    # A `no verdict line` row is the one this table used to make unreadable: gate.py refused the fixture
    # before starting the binary, so the row said only `EXIT 1` and a reader had to open the log to learn
    # that not one step was judged. Carry the refusal's own first line (and the recorded reason, when the
    # fixture is on gate.py's expected-red list) into the row, so a provenance gap cannot be mistaken for a
    # numerics result. The rc is still the verdict; this only adds the reason beside it.
    if [ "$v" != "PASS" ] && [ "$v" != "FAIL" ]; then
      reason=$(echo "$out" | grep -E "^  \[expected-red\]" | head -1)
      if [ -z "$reason" ]; then reason=$(echo "$out" | grep -E "^gate:" | head -1); fi
      v="$v: $(echo "$reason" | cut -c1-200)"
    fi
    echo "$f wide=$arm -> $v (rc=$rc)"
  done
done
echo "util at release: $(tools/gpu_lock.sh util 2>/dev/null)"
tools/gpu_lock.sh release a12-admit
echo "ADMISSION DONE"
