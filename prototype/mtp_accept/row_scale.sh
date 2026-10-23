#!/usr/bin/env bash
# row_scale.sh — what does an EXTRA ROW cost, per kernel family, in a body the engine already runs?
#
# WHY THIS EXISTS. `prototype/mtp_accept/verify_body_scale.sh` found that an accept round costs 2.085 steps
# at 4k and that the cost is inside the body, not the bookkeeping. That number is a fact about ONE route,
# though: `engine/mtp_verify.cpp` forces `WaveGemm::kHand` + `GdnScan::kSeq` so row 0 stays bit-comparable
# to the one-row step. The question this answers is whether the extra row is intrinsically ~1 step (k=2 is
# then dead, because a round buys at most 3 tokens) or whether it is the route (k=2 is then the cheapest
# big win left in decode). The engine prices both for free: a PREFILL pass chunked with `--wave-rows` IS a
# sequence of n-row bodies through the amortising route, and `--time-waves` gives the per-family split.
#
# WHAT IS BEING COMPARED. Not "prefill vs decode" — the same `submit_wave_body`, at n = 1,2,4,8 rows, twice:
# once at a 4-row-short KV (`--tiny`, KV effectively empty, so this is pure weight-walk behaviour) and once
# over a populated ~4k KV (`--long`, where the KV scan is what a second row might have to re-pay). The
# per-body figure is `burst_wall / bodies`, and the number that matters is its slope against rows.
#
# CAVEATS THIS SCRIPT CANNOT REMOVE, stated where they are read:
#   * a prefill body at position p sees p lanes, so `long` mixes body cost with the causal ramp; the slope
#     1->2 rows is still the honest per-row marginal, and `attn` is the family it affects most;
#   * `--time-waves` SERIALIZES the stream (see `WAVEGAP exposed_ms` on each run), so per-family ms are
#     in-band but inflated relative to the burst wall; the burst wall is the column to quote as a cost, the
#     family split is for attribution only;
#   * `priced_rows=1` on a prefill means the head is priced once per pass, not per body — `lm_head` in the
#     `tiny` rows is the pass's single head launch, which is why it looks flat.
#
# Protocol: main agent, blocking, one lock hold, util at both ends and discounted, one pass per row count
# (these are wall-clock bodies, not the medians-of-5 a timing claim would need — see the doc that cites it).
#
# Usage: prototype/mtp_accept/row_scale.sh [short|long]
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1

BIN=${BIN:-engine/build/decode_run}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
LONG_IDS=${LONG_IDS:-tmp/verify_body_scale/ids4032.csv}
ROWCOUNTS=${ROWCOUNTS:-"1 2 3 4 8 16"}
mkdir -p /tmp/row_scale

tools/gpu_lock.sh acquire main-agent row-scale || exit 1
trap 'tools/gpu_lock.sh release main-agent >/dev/null 2>&1' EXIT
echo "== util at acquire: $(tools/gpu_lock.sh util | tail -1); row counts: $ROWCOUNTS"

one() { # $1 label $2 ids-file $3 cap $4 wave_rows  -> one priced body pass
  engine/build/decode_run --model "$MODEL" --ids-file "$2" --steps 0 --cap "$3" --time-waves \
      --wave-rows "$4" --out /tmp/row_scale/out 2>&1 |
  awk -v lbl="$1" -v wr="$4" '
    /^WAVEPASS gran=burst/ { for (i=1;i<=NF;i++) { if ($i ~ /^wall_ms=/) w=substr($i,9); if ($i ~ /^bodies=/) b=substr($i,8) } }
    /^WAVE gran=stage family=/ { fam=""; ms=""; for (i=1;i<=NF;i++) { if ($i ~ /^family=/) fam=substr($i,8); if ($i ~ /^ms=/) ms=substr($i,4) } if (fam != "") f[fam]=ms }
    /^WAVEGAP/ { for (i=1;i<=NF;i++) if ($i ~ /^exposed_ms=/) e=substr($i,12) }
    END {
      printf "%-6s rows %-2s  burst %9.3f ms / %5s bodies = %8.4f ms/body   (serialized exposes %.3f ms)\n",
             lbl, wr, w, b, w/b, e;
      n = split("proj mlp_gateup mlp_down attn gdn_rec gdn_conv glue lm_head", fams, " ");
      printf "         ";
      for (i=1;i<=n;i++) if (fams[i] in f) printf " %s=%s", fams[i], f[fams[i]];
      printf "\n";
    }'
}

case "${1:-short}" in
  short)
    echo "=== SHORT KV: a body of n rows over an empty-ish cache (weight-walk behaviour) ==="
    for n in $ROWCOUNTS; do
      awk -v n="$n" 'BEGIN{for(i=0;i<n;i++) printf "%d%s", 100+i, (i<n-1?",":"")}' > /tmp/row_scale/ids.csv
      one short /tmp/row_scale/ids.csv 80 "$n"
    done ;;
  long)
    [ -f "$LONG_IDS" ] || { echo "no long fixture $LONG_IDS (generate it: prototype/mtp_accept/verify_body_scale.sh makes ids4032.csv)"; exit 1; }
    want=$(awk -F',' '{print NF}' "$LONG_IDS")
    echo "=== LONG KV: a 4,032-token pass chunked into bodies of n rows (KV scan is live) ==="
    for n in $ROWCOUNTS; do
      [ $((want % n)) = 0 ] || { echo "ABORT: $LONG_IDS has $want fields, not divisible by wave_rows=$n (a ragged tail body would price a different shape)"; exit 1; }
      one long "$LONG_IDS" $((want + 300)) "$n"
    done ;;
  *) echo "usage: $0 [short|long]"; exit 1 ;;
esac
echo "== util at release: $(tools/gpu_lock.sh util | tail -1)"
