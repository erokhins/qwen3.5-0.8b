#!/usr/bin/env bash
# A13 long-context identity — the check the short fixtures could not do.
#
# WHY. `a13_byte_identity.py` proves knob 2 == knob 1 on five fixtures up to 1,024 rows, but the timing
# window that motivated this script found knob 2 at -26.96 % of the WHOLE wave at 16,384 lanes, and a win
# that size from a -14.5 % round-slot cut is exactly the number that demands a skip-work hypothesis. The
# ported form splits each block's rounds into predicate-free full rounds plus a specialised tail
# (`nfull = lanes_of(row0)/64`); at 16k a block runs ~250 full rounds where the pinned fixtures run ~16.
# If the full/tail split is wrong at long prefix, the short witness cannot see it (AGENTS: "a route no
# fixture reaches has no tests, including tests of the gate").
#
# WHAT IT COMPARES. The engine's whole greedy continuation (`sampled.txt`, one line per step, so a 16k
# prompt compares 16,384 decisions) plus every `logits*.bin` row the run dumps, byte for byte, between
# `--attn-fattn-wide 1` and `2`. Correctness only: no millisecond figure is printed or kept.
#
# CANARIES per side: `NUMERICS attn_fattn_wide=`, `ATTN_WIDE form=`, and `PREFILL tokens=` equal to the
# fixture width, because a window that silently prefills 1 token compares two empty runs (this harness's
# own predecessor did exactly that and printed a table of plausible zeros).
set -uo pipefail
cd "$(dirname "$0")/../.."
BIN=${BIN:-engine/build/decode_run}
OUT=${OUT:-tmp/a13ab}
TAG=${TAG:-a13-longctx}
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

one() {   # one <knob> <lanes> <dir>
  local k="$1" lanes="$2" dir="$3" want_core want_form
  case "$k" in 0) want_core=fattn; want_form=- ;; 1) want_core=wide; want_form=wide128 ;; 2) want_core=wide_llama; want_form=wide128llama ;; esac
  rm -rf "$dir"; mkdir -p "$dir"
  "$BIN" --model models/Qwen3.5-0.8B --ids-file "$OUT/ids$lanes.txt" --steps 1 \
         --attn-fattn 1 --attn-fattn-wide "$k" --out "$dir" > "$dir/log" 2>&1 \
      || { echo "run failed knob=$k lanes=$lanes (rc=$?)" >&2; return 1; }
  local got
  # The fixture builder rounds a 32,000-token prompt down to whatever whole passages fit (31,825), so the
  # expected width is read off the ids file, not from the nominal name — otherwise the canary aborts a run
  # that did exactly the right thing.
  local want_tk; want_tk=$(awk -F',' '{print NF}' "$OUT/ids$lanes.txt")
  got=$(grep -E "^PREFILL" "$dir/log" | head -1 | sed -E 's/.*tokens=([0-9]+).*/\1/')
  [ "$got" = "$want_tk" ] || { echo "canary tokens: knob $k prefilled $got, wanted $want_tk" >&2; return 1; }
  got=$(grep -E "^NUMERICS" "$dir/log" | head -1 | sed -E 's/.*attn_fattn_wide=([a-z_]+).*/\1/')
  [ "$got" = "$want_core" ] || { echo "canary core: knob $k echoed $got, wanted $want_core" >&2; return 1; }
  if [ "$k" != 0 ]; then
    got=$(grep -E "^ATTN_WIDE" "$dir/log" | head -1 | sed -E 's/.*form=([a-z0-9]+).*/\1/')
    [ "$got" = "$want_form" ] || { echo "canary form: knob $k echoed $got, wanted $want_form" >&2; return 1; }
  fi
  echo "  knob $k: $(grep -c . "$dir/sampled.txt") sampled tokens, $(ls "$dir"/logits*.bin 2>/dev/null | wc -l) logits rows, sha1 $(cat "$dir"/sampled.txt "$dir"/logits*.bin 2>/dev/null | sha1sum | cut -c1-16)"
}

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1

rc=0
for lanes in ${LANES:-4096 16384}; do
  echo "=== lanes=$lanes  knob 1 vs knob 2, byte-for-byte (no timing taken) ==="
  one 1 "$lanes" "$OUT/id1_$lanes" || { fail "knob 1 at $lanes failed"; }
  one 2 "$lanes" "$OUT/id2_$lanes" || { fail "knob 2 at $lanes failed"; }
  if diff -r -q --exclude=log "$OUT/id1_$lanes" "$OUT/id2_$lanes" > "$OUT/diff_$lanes.txt" 2>&1; then
    echo "  RESULT: BYTE-IDENTICAL at $lanes lanes ($(ls "$OUT/id2_$lanes"/logits*.bin | wc -l) logits rows + full sampled sequence)"
  else
    echo "  RESULT: DIFFERS at $lanes lanes:"; head -5 "$OUT/diff_$lanes.txt"; rc=1
  fi
done
exit $rc
