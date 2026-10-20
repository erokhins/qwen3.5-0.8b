#!/usr/bin/env bash
# R1b — the question R1 was a precondition for: does the A12 wide shape pay once a body can give it the
# residency the rig had?
#
# WHY. The attribution window measured the wide launch at 32 workgroups per launch on a 40-CU machine
# (1 block/CU on 32 CUs, 8 wavefronts/CU) and the rig at 2 blocks/CU x 40 CUs = 16 wavefronts/CU, and the
# shape's whole +25 % lived in that gap. `kMaxWaveRows = 2048` makes `2 x ceil(2048/32) = 128` blocks —
# 3.2 per CU requested against an LDS bound of 2, i.e. EXACTLY the rig's 2 blocks/CU = 16 wf/CU, inside
# the engine. R1 priced that body width on the shipping form at +2.79 % at 16,384 lanes, so the wide arm
# has to beat its own body tax, and the four arms below say by how much.
#
# CANARIES, all read off the run's own lines before its number is kept: `NUMERICS wave_rows_cap=` (which
# binary), `PREFILL wave_rows=` (which body), `NUMERICS attn_fattn_wide=` (which core), and for a wide
# run `ATTN_WIDE grid=(...)` — which is the residency claim itself: (2,1,16) at 512 rows, (2,1,64) at
# 2048. Timing only: numerics are the gate's, and the wide form at a 2048-row body has never been
# admitted.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=${OUT:-tmp/wave_rows}
mkdir -p "$OUT"
TAG=${TAG:-r1b-wide-residency}
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

# arms: <binary> <wave_rows> <wide> <want_grid>
ARMS="w512 512 0 -   w512 512 1 (2,1,16)   w2048 2048 1 (2,1,64)   w2048 2048 0 -"

run() {   # run <bin> <rows> <wide> <want_grid> <lanes> -> ms
  local bin="$OUT/decode_run.$1" rows="$2" wide="$3" want="$4" lanes="$5"
  "$bin" --model models/Qwen3.5-0.8B --ids-file "$OUT/ids$lanes.txt" --steps 1 --wave-rows "$rows" \
         --attn-fattn 1 --attn-fattn-wide "$wide" --out "$OUT/tmpout" > "$OUT/run.log" 2>&1 \
      || fail "run $1 rows=$rows wide=$wide lanes=$lanes rc=$?"
  local cap capwant=fattn got
  cap=${1#w}; [ "$cap" = 2048 ] && cap=2048
  got=$(grep -E "^NUMERICS" "$OUT/run.log" | head -1 | sed -E 's/.*wave_rows_cap=([0-9]+).*/\1/')
  [ "$got" = "$cap" ] || fail "canary cap: lanes=$lanes $1 printed $got, built $cap"
  got=$(grep -E "^PREFILL" "$OUT/run.log" | head -1 | sed -E 's/.*wave_rows=([0-9]+).*/\1/')
  [ "$got" = "$rows" ] || fail "canary rows: lanes=$lanes $1 printed $got, asked $rows"
  [ "$wide" = 1 ] && capwant=wide
  got=$(grep -E "^NUMERICS" "$OUT/run.log" | head -1 | sed -E 's/.*attn_fattn_wide=([a-z]+).*/\1/')
  [ "$got" = "$capwant" ] || fail "canary core: lanes=$lanes wide=$wide echoed $got, wanted $capwant"
  if [ "$want" != "-" ]; then
    got=$(grep -E "^ATTN_WIDE" "$OUT/run.log" | head -1 | sed -E 's/.*grid=(\([0-9,]+\)).*/\1/')
    [ "$got" = "$want" ] || fail "canary grid: lanes=$lanes rows=$rows printed grid=$got, wanted $want"
  fi
  grep -E "^RESULT" "$OUT/run.log" | sed -E 's/.*ms_total=([0-9.]+).*/\1/'
}

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1

for lanes in 16384 32000; do
  reps=3; [ "$lanes" = 32000 ] && reps=2
  : > "$OUT/wide_$lanes.txt"
  for rep in $(seq "$reps"); do
    set -- $ARMS
    while [ "$#" -ge 4 ]; do
      echo "$1/$2/wide$3 $(run "$1" "$2" "$3" "$4" "$lanes")" >> "$OUT/wide_$lanes.txt"
      shift 4
    done
  done
  echo "=== lanes=$lanes (best of $reps; canaries cap/rows/core/grid asserted per run) ==="
  for arm in $(awk '{print $1}' "$OUT/wide_$lanes.txt" | sort -u); do
    best=$(awk -v a="$arm" '$1==a{if(m==""||$2<m)m=$2}END{printf "%.1f", m}' "$OUT/wide_$lanes.txt")
    base=$(awk '$1=="w512/512/0"{if(m==""||$2<m)m=$2}END{printf "%.1f", m}' "$OUT/wide_$lanes.txt")
    printf "  %-16s %8s ms  %6s %% vs shipping@512   %6s tok/s\n" "$arm" "$best" \
           "$(awk -v a="$best" -v b="$base" 'BEGIN{printf "%.2f", (a-b)/b*100}')" \
           "$(awk -v b="$best" -v n="$lanes" 'BEGIN{printf "%.0f", n/b*1000}')"
  done
done
