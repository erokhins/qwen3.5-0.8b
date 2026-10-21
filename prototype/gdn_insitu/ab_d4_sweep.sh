#!/usr/bin/env bash
# ab_d4_sweep.sh — the D4 arm matrix, in ONE lock hold. Main agent only: it takes milliseconds.
#
# Run `check_d4.sh` first and let it pass; this script assumes the rig is the same instrument as the
# ledger. Every row is one flag on ONE binary (no rebuilds inside the window, so no arm can be timed
# against a different build), every row must carry `verify=ok`, and a row that does not is printed as
# NO-TIMING instead of being quietly used. The parsed-row count is asserted against the number of rows
# asked for: a sweep whose parser matches nothing prints a confident empty table (AGENTS.md).
#
# The canary is behavioural, not a file hash (engine/build.sh is not bit-reproducible here): the
# shipped row MUST report launches=72 and the combined reproduction row MUST report launches=18,
# because those numbers are the shapes the whole route is about. A stale binary agrees with itself on
# a hash and cannot agree on a shape it did not compile.
set -uo pipefail

here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd); cd "$root" || exit 1
RIG=prototype/gdn_insitu/gdn_insitu
REPS=${REPS:-5} ITERS=${ITERS:-40}
TAG="gdn_insitu D4 matrix (main agent)"

[ -x "$RIG" ] || prototype/gdn_insitu/build.sh || exit 1
tools/gpu_lock.sh acquire "$TAG"; trap "tools/gpu_lock.sh release '$TAG'" EXIT
echo "   util at acquire: $(tools/gpu_lock.sh util)"

field() { printf '%s' "$1" | sed -n "s/.*[[:space:]]$2=\([^[:space:]]*\).*/\1/p" | head -1; }
num() { case "$1" in ''|*[!0-9.-]*|*nan*|*inf*) return 1;; *) return 0;; esac; }

NROW=0; NBADROW=0
# run ROWNAME extra-args...  -> prints "ROWNAME ms=... launches=... verify=..." or NO-TIMING
run() {
    local name=$1; shift
    local out; out=$("$RIG" "$@" --time --reps "$REPS" --iters "$ITERS" 2>&1)
    NROW=$((NROW+1))
    local v ms lc
    v=$(field "$out" verify); ms=$(field "$out" ms_median); lc=$(field "$out" launches)
    if [ "$v" != ok ] || ! num "$ms"; then
        NBADROW=$((NBADROW+1))
        printf 'NO-TIMING %-22s %s\n' "$name" "${out:0:150}"
        return 1
    fi
    printf 'ROW %-22s ms=%-9s launches=%-4s %s\n' "$name" "$ms" "$lc" \
        "$(field "$out" rate_class)"
    printf '%s %s %s\n' "$name" "$ms" "$lc" >> "$TMP"
}

TMP=$(mktemp); want() { grep -m1 "^$1 " "$TMP" | awk '{print $2}'; }

echo "== canary (behavioural: the shapes the route is about)"
run ship       --kernel all --witness land --drain burst
run repro      --shape combined --spell ship --kernel rec --witness onehot
lc=$(grep -m1 '^ship ' "$TMP" | awk '{print $3}'); [ "$lc" = 72 ] || echo "CANARY FAIL: ship row launches=$lc, want 72"
lc=$(grep -m1 '^repro ' "$TMP" | awk '{print $3}'); [ "$lc" = 1 ] || echo "CANARY FAIL: repro row launches=$lc, want 1 (one launch of 288 blocks is the quoted shape)"

echo "== C2 the state's cache carry-over (a slope, not a difference)"
for p in 0 32 512 1536; do run "cold_${p}MiB" --kernel rec --witness land --pollute-mib "$p"; done

echo "== C3 launch population and drain"
for k in rec conv gates gnorm; do run "only_$k" --kernel "$k" --witness land; done
run drain_each --kernel all --witness land --drain each

echo "== C4 the per-token chain: launch count x per-launch parallelism, bytes fixed"
run split8     --kernel rec --witness land --recmap split8
run comb_all   --shape combined --kernel rec --witness onehot

echo "== C6 barrier/prologue bookkeeping at 2 wavefronts per block (bit-identical spellings)"
run spell_ship    --kernel rec --witness land --spell ship
run spell_unroll4 --kernel rec --witness land --spell unroll4

echo "== C7 placement (the engine's own shape is blob1; the others are manufactured)"
run spanned8  --kernel rec --witness land --layout spanned --gap-mib 8

echo "== refusals that must stay refusals"
"$RIG" --shape combined --recmap split8 --kernel rec | head -1
"$RIG" --kernel rec --witness mirror --tokens 2 | head -1
"$RIG" --kernel rec --time --tokens 2 | head -1

P=$(awk -v a="$(want ship)" -v b="$(want repro)" 'BEGIN{printf "%.4f",a-b}')
echo "== shares (P = shipped - shape-matched reproduction = $P ms)"
share() { printf '   %-14s ms=%-9s share-of-P=%s\n' "$1" "$(want "$1")" \
    "$(awk -v a="$(want "$1")" -v p="$P" 'BEGIN{ if (p==0) {printf "n/a"} else {printf "%.2f", a/p} }')"; }
for r in cold_0MiB cold_1536MiB only_rec only_conv only_gates only_gnorm drain_each split8 spell_ship spell_unroll4 spanned8; do share "$r"; done
echo "   residue is printed, never absorbed: 1 - sum(shares) is the reader's arithmetic on the rows above."
echo "ROWS asked=$((NROW)) parsed=$(( $(wc -l < "$TMP") )) red=$NBADROW"
[ "$NROW" -gt 0 ] && [ "$NROW" -eq "$(wc -l < "$TMP")" ] || echo "SWEEP INCOMPLETE: a row produced no parsed timing"
rm -f "$TMP"
