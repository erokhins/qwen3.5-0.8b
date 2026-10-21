#!/usr/bin/env bash
# check_d4.sh — the ~20 s handback check. Run this BEFORE the matrix is paid for.
#
# This script takes milliseconds, so it is the MAIN AGENT's to run (AGENTS.md: a worker may run
# correctness gates, never timing). One gpu_lock hold, blocking, nothing backgrounded. It answers four
# questions in order and stops at the first red, because a matrix measured on a rig that is not the
# same instrument as the quoted ledger is a table of NO-RESULTs:
#
#   G-witness  does the rig verify itself, on every shape it can be set to?  (--selftest, device, ~4 s)
#   G-repro    does its combined/warm/`ship` row reproduce the quoted 0.0646 ms within +-5 %?
#   G-synth    does the shipped shape, state forced cold, cost >= 0.08 ms MORE than that row?
#   G-drain    is `--drain each - --drain burst` inside a band DERIVED from the row's own sync
#              population times the 5.64 us this family actually measured (stage vs kernel
#              granularity, same window, section 0(i)). CORRECTED by the main agent, not loosened:
#              the original band assumed 36 drains, but `gdn_insitu.hip.cpp:649` syncs once per
#              LAUNCH and the shipped row launches 72 (the engine's stage clock closes 36 SEGMENTS,
#              two kernels each - see `STAGE gran=stage family=gdn launches_per_step=72
#              segments_per_step=36`). Measured 0.3827 ms over 72 = 5.32 us/sync, i.e. inside the
#              quoted 5.64 us once the population is right. The constant survived; the assumed
#              population did not.
#
# Quoted reference for G-repro: prototype/gdn_decode/results.md, `--mode all`, 0.0646 ms / 18 layers.
# A miss is still a NO-DIAGNOSIS, never a widened band: both gates below compare against the
# reference's own instrument (same --iters scale; same per-drain unit), and REPRO_QUOTED_MS and
# DRAIN_*_US are labelled quotes, not tuning knobs.
set -uo pipefail
# Not cosmetic: this box's locale uses a comma as the decimal separator, so awk printed the drain term
# as "0,3827" above. Parsing happened to survive, but a numeric field read under a comma locale is the
# D2 script's recorded failure mode, so state the locale instead of trusting it.
export LC_ALL=C

here=$(cd "$(dirname "$0")" && pwd); root=$(cd "$here/../.." && pwd); cd "$root" || exit 1
RIG=prototype/gdn_insitu/gdn_insitu
REPRO_QUOTED_MS=0.0646          # prototype/gdn_decode/results.md (quoted, labelled)
DRAIN_LO_US=4.17 DRAIN_HI_US=7.20   # per-drain bounds around the quoted 5.64 us family constant
# REPS/ITERS are NOT free here: they must match the instrument the reference row was taken with.
# `prototype/gdn_decode/results.md` quotes 0.0646 ms BEST / 0.0650 worst, spread 0.5 %, from
# `gdn_decode --mode all --iters 200 --trials 5`. At --reps 3 --iters 40 this rig's repro row came in at
# best 0.0643 (correct, -0.5 %) but median 0.0741 (+14.7 %), because a launch-bound sub-0.1 ms row is
# bimodal when under-sampled (AGENTS: dispatch-bound timings do not repeat). At --reps 5 --iters 200 it
# reads median 0.0639 / best 0.0639 / worst 0.0642 — the same 0.5 % spread as the reference, -1.1 % on
# the median. The gate was failing on its own sampling, not on the rig: the fix is the scale, not the band.
REPS=5 ITERS=200
TAG="gdn_insitu D4 check (main agent)"

fail() { printf 'NO-DIAGNOSIS: %s\n' "$1"; exit 1; }
num() { case "$1" in ''|*[!0-9.-]*|*nan*|*inf*) return 1;; *) return 0;; esac; }
field() { printf '%s' "$1" | sed -n "s/.*[[:space:]]$2=\([^[:space:]]*\).*/\1/p" | head -1; }

[ -x "$RIG" ] || prototype/gdn_insitu/build.sh || fail "rig will not build"

echo "== host geometry (no device, no ms)"
"$RIG" --selftest-geom | tail -1 | grep -q '^SELFTEST-GEOM ok fails=0$' || fail "G-geom"
"$RIG" --kernel none | grep -q 'NO-TIMING' || fail "no-launch arm printed a timing instead of NO-TIMING"
"$RIG" --pages cold >/dev/null 2>&1 && fail "--pages cold did not refuse"
"$RIG" --layout blob18 >/dev/null 2>&1 && fail "--layout blob18 did not refuse (it would price a scatter the engine does not have)"

tools/gpu_lock.sh acquire "$TAG"; trap "tools/gpu_lock.sh release '$TAG'" EXIT
echo "   util at acquire: $(tools/gpu_lock.sh util)"

echo "== G-witness (device, correctness only)"
sw=$("$RIG" --selftest | tail -1)
printf '   %s\n' "$sw"
printf '%s' "$sw" | grep -q '^SELFTEST ok' || fail "G-witness: $sw"
np=$(field "$sw" parsed); nc=$(field "$sw" cases)
[ "$np" = "$nc" ] || fail "G-witness parsed $np of $nc cases"

# A timed row: prints `KEY=... ms_median=...`; refuses to emit ms when its verify is not ok.
row() { "$RIG" "$@" --time --reps "$REPS" --iters "$ITERS"; }
get_ms() { field "$1" ms_median; }
want_ok() { case "$1" in *"verify=ok"*) return 0;; *) return 1;; esac; }

echo "== G-repro (shape-matched: combined 1x288 blocks, warm, shipped 'ship' spelling)"
r=$(row --shape combined --spell ship --kernel rec --witness onehot); printf '   %s\n' "$r"
want_ok "$r" || fail "G-repro row is not verified"
repro=$(get_ms "$r"); num "$repro" || fail "G-repro ms unparsed"
# A reproduction row with a fat tail is not a calibrated instrument even when its median lands, so assert
# the spread too: a bimodal rig must fail here rather than poison every row downstream of it.
rbest=$(field "$r" ms_best); rworst=$(field "$r" ms_worst)
awk -v b="${rbest:-0}" -v w="${rworst:-9}" 'BEGIN{exit (b>0 && w/b<=1.03)?0:1}' \
    || fail "G-repro is bimodal (best ${rbest:-?} worst ${rworst:-?}): a row with a >3 % tail cannot calibrate a sub-0.1 ms family"
awk -v a="$repro" -v q="$REPRO_QUOTED_MS" 'BEGIN{d=(a-q)/q*100; if (d<0) d=-d; exit (d<=5.0)?0:1}' \
    || fail "G-repro $repro ms is >5% from the quoted $REPRO_QUOTED_MS ms: the rig is another instrument"

echo "== G-synth (shipped shape, state forced cold: the rig must synthesize an in-situ penalty)"
s=$(row --kernel all --witness land --pollute-mib 1536); printf '   %s\n' "$s"
want_ok "$s" || fail "G-synth row is not verified"
syn=$(get_ms "$s"); num "$syn" || fail "G-synth ms unparsed"
awk -v s="$syn" -v r="$repro" 'BEGIN{exit ((s-r)>=0.08)?0:1}' \
    || fail "G-synth: shipped-cold minus shape-matched warm = $(awk -v a="$s" -v b="$r" 'BEGIN{printf "%.4f",a-b}') ms < 0.08 ms: no in-situ penalty to decompose"

echo "== G-drain (band derived from the row's own sync population)"
e=$(row --kernel all --witness land --drain each); printf '   %s\n' "$e"
b=$(row --kernel all --witness land --drain burst); printf '   %s\n' "$b"
want_ok "$e" && want_ok "$b" || fail "G-drain rows are not verified"
ems=$(get_ms "$e"); bms=$(get_ms "$b"); num "$ems" && num "$bms" || fail "G-drain ms unparsed"
dpop=$(field "$e" launches); num "$dpop" || fail "G-drain sync population unparsed"
awk -v d="$ems" -v b="$bms" -v n="$dpop" -v lo="$DRAIN_LO_US" -v hi="$DRAIN_HI_US" 'BEGIN{
    t=d-b; ps=t*1000/n; seg=ps*36;
    printf "   drain term = %.4f ms over %d rig launch-syncs = %.2f us/sync (want %.2f-%.2f)\n", t, n, ps, lo, hi;
    printf "   engine-comparable: 36 stage-clock segments x %.2f us = %.4f ms of the gdn family line is per-segment drain\n", ps, seg/1000;
    exit (ps>=lo && ps<=hi)?0:1 }' \
    || fail "G-drain: outside the derived per-sync band; the rig's drain shape or the attribution's per-segment constant is wrong"

dr=$(awk -v a="$ems" -v b="$bms" 'BEGIN{printf "%.4f",a-b}')
echo "CHECK-D4 PASS  repro=$repro ms  shipped_cold=$syn ms  drain=$dr ms"
