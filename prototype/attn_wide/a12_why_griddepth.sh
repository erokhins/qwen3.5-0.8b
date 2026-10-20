#!/usr/bin/env bash
# W3 — two legs, one lock hold, both main-agent-only and blocking.
#
# LEG A  the paired wall A/B again with 5 interleaved reps (the tip commit ran 3), so the engine-level
#        difference is a distribution rather than a pair of bests.
# LEG B  the grid-depth probe the answer turns on: `--wave-rows` is the ONE engine knob that changes how
#        many rows a body carries at a fixed total lane count, and the wide launch's block count is
#        2 x ceil(wave_rows/32) while the shipping launch's is 8 x split x ceil(wave_rows/48). So this
#        sweeps the wide arm's blocks-per-launch through and below the 40-CU population and asks whether
#        its time moves with it. Total work is identical at every wave_rows; only the grouping moves.
set -uo pipefail
cd "$(dirname "$0")/../.."
OUT=tmp/a12why
mkdir -p "$OUT"
BIN=engine/build/decode_run
TAG=${TAG:-a12-why-wall}
REPS=${REPS:-5}
fail() { echo "ABORT: $*" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }

# run <lanes> <wide> <wave_rows> -> prints ms_total
run() {
  "$BIN" --model models/Qwen3.5-0.8B --ids-file "$OUT/ids$1.txt" --steps 1 \
         --attn-fattn 1 --attn-fattn-wide "$2" --wave-rows "$3" \
         --out "$OUT/tmpout" > "$OUT/run.log" 2>&1 || fail "run lanes=$1 wide=$2 rows=$3 rc=$?"
  grep -E "^RESULT" "$OUT/run.log" | sed -E 's/.*ms_total=([0-9.]+).*/\1/'
}

stats() {   # stats <file> <wide> -> "best median  reps"
  awk -v w="$2" '$1==w{a[n++]=$2} END{
      n2=n; for(i=0;i<n2;i++)for(j=i+1;j<n2;j++) if(a[j]<a[i]){t=a[i];a[i]=a[j];a[j]=t}
      s=""; for(i=0;i<n2;i++) s=s sprintf("%.1f ", a[i]);
      printf "best=%.1f median=%.1f reps: %s", a[0], a[int(n2/2)], s }' "$1"
}

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
tools/gpu_lock.sh util | tail -1

echo "=== LEG A: paired wall, $REPS interleaved reps (wide0 then wide1 each rep) ==="
for lanes in 16384 4096; do
  : > "$OUT/legA_$lanes.txt"
  for rep in $(seq "$REPS"); do
    for wide in 0 1; do echo "$wide $(run "$lanes" "$wide" 512)" >> "$OUT/legA_$lanes.txt"; done
  done
  echo "lanes=$lanes"
  for wide in 0 1; do echo "  wide=$wide  $(stats "$OUT/legA_$lanes.txt" "$wide")"; done
  awk -v f="$OUT/legA_$lanes.txt" 'BEGIN{while((getline l < f)>0){split(l,x," ");if(x[1]=="0"){b0=x[2];if(m0==""||x[2]<m0)m0=x[2]}else{b1=x[2];if(m1==""||x[2]<m1)m1=x[2]}}
       printf "  wide1 vs wide0: best %.2f %%  (median %.2f %%)\n", (m1-m0)/m0*100, (b1-b0)/b0*100}'
done

echo "=== LEG B: grid-depth probe — wave_rows at fixed 16,384 lanes, 2 reps, best of 2 ==="
printf "%9s %11s %14s %14s %14s %8s\n" wave_rows wide_wgs ship_wgs wide0_ms wide1_ms ratio
for rows in 512 256 128 64; do
  bw=$(( 2 * ( (rows + 31) / 32 ) ))
  bs=$(( 8 * 32 * ( (rows + 47) / 48 ) ))
  : > "$OUT/legB.txt"
  for rep in 1 2; do
    for wide in 0 1; do echo "$wide $(run 16384 "$wide" "$rows")" >> "$OUT/legB.txt"; done
  done
  m0=$(awk '$1==0{if(m==""||$2<m)m=$2}END{print m}' "$OUT/legB.txt")
  m1=$(awk '$1==1{if(m==""||$2<m)m=$2}END{print m}' "$OUT/legB.txt")
  printf "%9s %11s %14s %14s %14s %8s\n" "$rows" "$bw" "$bs" "$m0" "$m1" \
         "$(awk -v a="$m1" -v b="$m0" 'BEGIN{printf "%.4f", a/b}')"
done
