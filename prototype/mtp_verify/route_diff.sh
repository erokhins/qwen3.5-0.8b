#!/usr/bin/env bash
# route_diff.sh — what does `--mtp-route amortised` cost the body's NUMERICS? REPORTED, not gated.
#
# WHY. `engine/mtp_verify.h`'s route forcing (hand GEMV + sequential scan) is what keeps row 0 of the
# 2-row body the same kernel chain the one-row shipping step ran, and `prototype/mtp_accept/ROW-SCALE.md`
# measured that the forcing is ~1.0 of the round's 2.085-step cost. Pricing the round on the amortising
# route needs a number for what the cheap route costs the ROW, and that number is not this file's to
# decide: no band is asserted here, no threshold is imported as a verdict, and nothing exits non-zero on a
# metric. Every row pair is printed with `prototype/mtp_verify/rowdiff.py`'s own metrics — ndiff, max-ULP,
# n over/under the cancellation band, 1-cos, meanrel, and whether the argmax agrees — plus the engine's own
# `VERIFY2 … mtp_route= gemm= gdn_scan=` echo, so the row and the route that produced it travel together
# (`AGENTS.md`: a paired A/B against a stale binary reads as a clean null).
#
# WHAT IT RUNS, per width, in one GPU-lock hold:
#   plain     no MTP flag. Its `logits<s>.bin` is the row a one-row step produces at that position.
#   hand      `--mtp-verify 1 --verify-candidate sampled`            -> today's forced body
#   amort     same plus `--mtp-route amortised`                     -> the scratch's own resolved route
# Row 0 pairs with `logits<s>.bin` (the pairing `prototype/mtp_verify/gate.py` §(D) uses: the body's row 0
# is fed the token the step was fed, at the step's own position). `sampled.txt` is byte-compared across all
# three, which is the "the side pass changed no output" claim stated as a byte count rather than a promise.
#
# NO TIMING. Nothing here prints or reads a millisecond figure; timing the same arms is
# `prototype/mtp_accept/verify_body_scale.sh` with EXTRA="--mtp-route amortised", main agent only.
#
# Usage: prototype/mtp_verify/route_diff.sh [TAG] [width...]     # default: dec8 1024
#   "dec8" means the fixture's own prefill_ids; a number means a cycled fixture of exactly that many lanes.
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

BIN=${BIN:-engine/build/decode_run}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
PY=${PY:-.venv/bin/python3}
TAG=${1:-mtp-route}
shift 2>/dev/null || true
STEPS=${STEPS:-8}
OUT=${OUT:-tmp/route_diff/$TAG}
mkdir -p "$OUT"; rm -rf "$OUT"/* 2>/dev/null

ids_from_fixture() { $PY -c "import json,sys;print(','.join(map(str,json.load(open('data/refs/dec8/manifest.json'))['prefill_ids'])))"; }
cycle_ids() { awk -v n="$1" 'BEGIN{RS="[,\n]"} {ids[c++]=$0} END{if(c<1){exit 1} for(i=0;i<n;i++) printf "%s%s", ids[i%c], (i+1<n?",":"")}' "$SRC"; }
fields() { awk -F',' '{print NF}' "$1"; }

# dec8's own ids, as a csv file (an argument is capped at one argv slot; --ids-file splits on commas).
SRC="$OUT/dec8_ids.csv"; ids_from_fixture > "$SRC" || { echo "ABORT: cannot read dec8 prefill_ids"; exit 1; }
DEC8_N=$(fields "$SRC")

arm() { # $1 width-namespace $2 label $3 ids-file $4 steps $5 extra
  local spec="$1" label="$2" ids="$3" steps="$4" extra="$5" d
  d="$OUT/$spec/$label"
  rm -rf "$d"; mkdir -p "$d"
  "$BIN" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids-file "$ids" --steps "$steps" --out "$d" --no-stop --gdn-loop 1 \
      $extra > "$d/stdout.txt" 2>&1 || { echo "ABORT: $label run failed (see $d/stdout.txt)"; return 1; }
  echo "$d"
}

echo "== route_diff: BIN=$BIN out=$OUT dec8 has $DEC8_N fields; steps=$STEPS"
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release '"$TAG"' >/dev/null 2>&1' EXIT
echo "== util at acquire: $(tools/gpu_lock.sh util | tail -1)"

SPECS="$*"
[ -z "$SPECS" ] && SPECS="dec8 1024"
for spec in $SPECS; do
  if [ "$spec" = dec8 ]; then
    ids="$SRC"; steps=$STEPS; want=$DEC8_N
  else
    ids="$OUT/ids$spec.csv"; cycle_ids "$spec" > "$ids"; steps=4; want=$spec
  fi
  got=$(fields "$ids")
  [ "$got" = "$want" ] || { echo "ABORT: $ids has $got fields, wanted $want"; exit 1; }
  echo
  echo "=== lanes=$spec (ids file $ids, $got fields) ==="
  arm "$spec" plain "$ids" "$steps" ""            > /dev/null || exit 1
  arm "$spec" hand  "$ids" "$steps" "--mtp-verify 1 --verify-candidate sampled" > /dev/null || exit 1
  # `hand` named explicitly, against `hand` reached by the default. The flag's whole default-side claim is
  # "naming hand reproduces what a run that named nothing does", and that is a byte-compare or it is nothing
  # — a `--mtp-route` that silently did less than its name would otherwise be invisible in every row below.
  arm "$spec" handx "$ids" "$steps" "--mtp-verify 1 --verify-candidate sampled --mtp-route hand" > /dev/null || exit 1
  arm "$spec" amort "$ids" "$steps" "--mtp-verify 1 --verify-candidate sampled --mtp-route amortised" > /dev/null || exit 1

  echo "-- route echo, per arm (the engine's own line, not this script's inference)"
  for label in hand handx amort; do
    grep -m1 '^VERIFY2' "$OUT/$spec/$label/stdout.txt" | sed "s/^/   $label: /"
    grep -m1 '^VERIFY2DONE' "$OUT/$spec/$label/stdout.txt" | sed "s/^/   $label census: /"
  done

  echo "-- output byte-identity across the arms (sampled.txt is the run's own emitted stream)"
  for label in hand handx amort; do
    if cmp -s "$OUT/$spec/plain/sampled.txt" "$OUT/$spec/$label/sampled.txt"; then
      echo "   plain vs $label: sampled.txt IDENTICAL"
    else
      echo "   plain vs $label: sampled.txt DIFFERS  <-- report, do not fix by loosening anything"
    fi
  done
  # The default-side byte-compare: same flags, `--mtp-route hand` named vs not named, EVERY dump byte-compared
  # (not just the token stream: the body's own logits rows are the object, and a token stream can survive a
  # moved row by luck at a near-tie).
  ndiff_bytes=0; nfiles=0
  for f in "$OUT/$spec/hand"/*.bin; do
    b=$(basename "$f"); nfiles=$((nfiles + 1))
    cmp -s "$f" "$OUT/$spec/handx/$b" || { echo "   hand vs handx: $b DIFFERS"; ndiff_bytes=$((ndiff_bytes + 1)); }
  done
  echo "   hand vs handx: $nfiles dump(s) byte-compared, $ndiff_bytes differed (0 = naming hand is exactly today's run)"

  echo "-- row 0 of the body against the one-row step, per anchor (rowdiff.py; no band applied)"
  # The anchors come off the engine's own VERIFY2 lines, not off a counted range: a run's generation steps
  # are numbered from the prompt length (`logits<s>.bin` / `verify2_<s>.bin` share the step number), so a
  # `seq 1 steps` loop would name files that never existed — a parse that matched nothing and printed
  # nothing, which is the failure class this repo keeps recording.
  anchors=$(grep '^VERIFY2 ' "$OUT/$spec/hand/stdout.txt" | tr ' ' '\n' | sed -n 's/^step=//p')
  [ -n "$anchors" ] || { echo "   ABORT: no VERIFY2 step= anchors parsed; nothing was compared"; exit 1; }
  for s in $anchors; do
    ref="$OUT/$spec/plain/logits$s.bin"
    [ -f "$ref" ] || continue
    for label in hand amort; do
      v="$OUT/$spec/$label/verify2_$s.bin"
      [ -f "$v" ] || continue
      printf '   %-5s anchor %5s vs-step  ' "$label" "$s"
      "$PY" prototype/mtp_verify/rowdiff.py "$v" "$ref" --row-a 0 --row-b 0 --rows 2 --rows-b 1 \
            --label "row0-vs-step/$label@$s" 2>&1 | sed 's/^[^ ]*: //'
    done
    printf '   both  anchor %5s hand-vs-amort  ' "$s"
    "$PY" prototype/mtp_verify/rowdiff.py "$OUT/$spec/hand/verify2_$s.bin" \
          "$OUT/$spec/amort/verify2_$s.bin" --row-a 0 --row-b 0 --rows 2 --rows-b 2 \
          --label "body-row0 hand-vs-amort@$s" 2>&1 | sed 's/^[^ ]*: //'
  done
done
echo
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
echo "== no timing taken or printed by this script."
