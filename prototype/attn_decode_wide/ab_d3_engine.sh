#!/usr/bin/env bash
# ab_d3_engine.sh — the paired 4k engine A/B for the D3 `lean` port. MAIN AGENT ONLY: this takes
# milliseconds, and a worker may not (AGENTS.md, "GPU execution is serialized and belongs to the main
# agent"). It acquires tools/gpu_lock.sh itself and holds it for the whole window; run it BLOCKING,
# never backgrounded, with zero subagents active.
#
#   bash prototype/attn_decode_wide/ab_d3_engine.sh
#   REPS=5 GATE_FIXTURES="dec8 wide4096f" bash prototype/attn_decode_wide/ab_d3_engine.sh
#
# ============================================================================ THE PROJECTION
# Quoted, not measured here. From prototype/attn_decode_wide/results.md §"What X comes out as":
#   the isolated per-layer core body went 0.3201 -> 0.1753 ms (-45.2 %, quoted) at 4,104 lanes, which
#   prices X — the tile walk's share of the engine's 3.023 ms `attn` line at 4,032 lanes — at X ~= 0.64,
#   and the projection the engine A/B exists to FALSIFY is:
#       lean alone saves ~= 0.87 ms/step at 4k:  12.68 -> ~= 11.81 ms/token = ~= 84.7 tok/s at 4,032
#      lanes, from the ledger's 78.9 tok/s (both quoted: engine/performance-status.md §1 and
#       prototype/attn_decode_wide/results.md). `lean` + split 18 was projected at ~= 87.7 tok/s and is
#       NOT what this script times: the split lever is held pending the streaming-pin ruling.
#
# PRE-REGISTERED LINES, from HANDOFF-D3.md's head, on the `attn` line at 4,032 lanes (ms/step, stage
# granularity, median of reps):
#     WIN   <= 1.50 ms
#     KILL  >= 2.55 ms
#     between the lines is neither, and gets reported as neither.
# At 128 lanes the WIN line was 0.40 ms and HANDOFF-D3.md §4.3 declared it OUT OF REACH of this route in
# advance: 8 blocks on 40 CUs is one round at either residency, so neither the residency lever nor the
# population lever can bind there. THEREFORE A FLAT 128-LANE ROW IS A PREDICTED PASS, NOT A MISS. Say
# so in the report; do not read it as a failed route.
#
# ============================================================================ THE CANARIES
# `engine/build.sh` is not bit-reproducible on this box, so `cmp` between two builds proves only that
# they are two builds, and any self-check compiled into the binary under test is blind to the one
# failure mode that matters (AGENTS.md, 2026-10-17: the first D30 A/B timed a pre-change binary as the
# `new` side and reported a clean null while every in-binary parity field stayed green). So three
# fields the change is REQUIRED to move are read in the same breath as the measurement, and the script
# aborts rather than timing if any reads wrong:
#
#   1. CODE OBJECT, read from each side's own build product (`--save-temps=obj`, field
#      `.amdhsa_group_segment_fixed_size` + `.amdhsa_next_free_vgpr` via
#      prototype/attn_percost/a11_read_vgpr.py — NOT `.vgpr_count`, NOT a driver query, per the
#      AGENTS 2026-10-20 row):
#           base  attn_core  LDS must read 38944 B   (next_free_vgpr 53)
#           new   attn_core  LDS must read  6176 B   (next_free_vgpr 45)
#      BASE IS BUILT FROM A SCRATCH `git worktree add` AT HEAD, never from the touched working tree,
#      because the working tree is exactly where the uncommitted half of a fact hides (AGENTS, the
#      two-header row). If base reads 6176 the worktree is stale; if new reads 38944 the port did not
#      land and every number below this line would be a fact about one kernel measured twice.
#   2. EXERCISED PATH, read from the run's own echo lines, so a flat result is attributable instead of
#      mysterious: `PREFILL tokens=` must equal the id FILE's field count (`--ids-file` splits on
#      commas; a nominal width that is not the field count is the AGENTS E2BIG/1-token trap), the
#      attention launch must be the decode one at the width being timed (`SPLITCAP split=8 ...
#      rows=1` with a `lanes=4032` line at 4,032; `split=1` at 128), the `attn` family must have been
#      timed at all (`STAGE gran=stage family=attn`), and the run's own replay must be reproducible
#      (`STAGEPARITY ... tokens_identical=1 logits_bitidentical=1`).
#   3. NUMERICS FIRST, before any ms is spent: `engine/gate.py` on the same fixtures for both binaries.
#      The port is a code-object claim, so a red numerics line OUTRANKS the timing entirely and the
#      window aborts before it starts (`NO-TIMING:` — the prototype/attn_m2/burst_m16.sh pattern,
#      which refuses to print a millisecond figure over a red admission).
#
# WHAT IS TIMED, AND HOW. `engine/build/decode_run --time-stages`, the ledger's own instrument
# (progress/2026-10-21/decode-attribution-2026-10-21.md §2): 4,000 prompt lanes + 32 steps = 4,032
# lanes at the last step, 112 + 16 = 128. Arms are INTERLEAVED rep by rep so clock drift shows as
# disagreement between reps of one arm rather than as a window effect, and every arm is reported as a
# median with its spread, never as a best. Two rows per arm: the pre-registered `attn` line
# (`STAGE gran=stage family=attn ms_per_step`) and the headline ms/token
# (`STAGE gran=burst family=TOTAL ms_per_step`, the pipelined production shape).
#
# SHORT-PROMPT CAVEAT that still applies to the 128-lane row: dispatch-bound rows are bimodal on this
# box (engine/performance-status.md §6), so quote the pair and state the spread and let the 4,032 row
# carry the verdict.
set -uo pipefail
cd "$(dirname "$0")/../.."
# LC_ALL=C is not cosmetic here. This box's locale uses a comma as the decimal separator, which makes
# `sort -n` stop a numeric key at the '.' and makes awk's strtod read "3.0230" as 3 — so the medians,
# the spreads and the WIN/KILL verdict would all have been computed off truncated integers while still
# printing plausible digits. The dry run caught it: `attn line: 3,0000 -> 1,0000 ms/step (-66,67 %)`.
export LC_ALL=C
VENV=${VENV:-.venv/bin/python}
OUT=${OUT:-tmp/d3ab}
TAG=${TAG:-d3-lean-ab}
REPS=${REPS:-3}                       # reps per arm, per width (>=3, pre-registered)
GATE_FIXTURES=${GATE_FIXTURES:-dec8}  # space-separated data/refs/<name>; add wide4096f for the pin
MODEL=${MODEL:-models/Qwen3.5-0.8B}
BASE_BIN="$OUT/base_decode_run"; NEW_BIN="$OUT/new_decode_run"
BASE_S="$OUT/base_kernels_attn.s";   NEW_S="$OUT/new_kernels_attn.s"
BASE_WT="$OUT/base"
mkdir -p "$OUT"
# OUT must be ABSOLUTE. The BASE build runs `( cd "$BASE_WT" && ... > "$OUT/build_base.log" )`, and a
# relative OUT cannot resolve from inside the worktree, so the redirect died with "No such file or
# directory" and the script aborted as "base build failed" without ever compiling anything.
OUT=$(cd "$OUT" && pwd) || { echo "ABORT: cannot resolve OUT" >&2; exit 1; }
NUM_OK=0
fail() { echo "ABORT: $*" >&2; echo "NO-TIMING: no millisecond figure is printed from this run" >&2; tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1; exit 1; }
med() { sort -n "$1" | awk '{a[NR]=$1} END{ if (NR==0) {print "nan"; exit} m=(NR%2)?a[(NR+1)/2]:(a[NR/2]+a[NR/2+1])/2; printf "%.4f", m }'; }
spread() { sort -n "$1" | awk '{if(m==""||$1<m)m=$1; if($1>x)x=$1} END{printf "%s..%s", m, x}'; }

# ---------------------------------------------------------------- code-object canary (CPU only) ---
# Built BEFORE the lock is taken: a compile is not device work and must not eat the window.
echo "== building NEW from the working tree (temps kept: the code object is the canary) =="
EXTRA_FLAGS="--save-temps=obj" bash engine/build.sh > "$OUT/build_new.log" 2>&1 \
    || fail "engine/build.sh (new) failed, see $OUT/build_new.log"
cp engine/build/decode_run "$NEW_BIN" || fail "no $NEW_BIN"
cp engine/build/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s "$NEW_S" || fail "no new code-object listing"
if [ -x "$BASE_BIN" ] && [ -s "$BASE_S" ] && [ "${REBUILD_BASE:-0}" != 1 ]; then
    echo "== reusing $BASE_BIN from an earlier run (REBUILD_BASE=1 to force a fresh HEAD build) =="
else
    echo "== building BASE from HEAD in a scratch worktree (never from the touched working tree) =="
    git worktree remove "$BASE_WT" --force >/dev/null 2>&1; git worktree prune >/dev/null 2>&1
    rm -rf "$BASE_WT"
    git worktree add "$BASE_WT" HEAD --detach > "$OUT/worktree.log" 2>&1 || fail "worktree add failed"
    ( cd "$BASE_WT" && EXTRA_FLAGS="--save-temps=obj" bash engine/build.sh > "$OUT/build_base.log" 2>&1 ) \
        || fail "engine/build.sh (base) failed, see $OUT/build_base.log"
    cp "$BASE_WT/engine/build/decode_run" "$BASE_BIN" || fail "no $BASE_BIN"
    cp "$BASE_WT/engine/build/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s" "$BASE_S" || fail "no base listing"
fi
echo "== the two witness lines (field: .amdhsa_next_free_vgpr, .amdhsa_group_segment_fixed_size) =="
base_line=$($VENV prototype/attn_percost/a11_read_vgpr.py "$BASE_S" attn_core | awk '$1=="attn_core(CoreArgs)"')
new_line=$(  $VENV prototype/attn_percost/a11_read_vgpr.py "$NEW_S"  attn_core | awk '$1=="attn_core(CoreArgs)"')
echo "  base  $base_line"
echo "  new   $new_line"
base_lds=$(echo "$base_line" | awk '{print $4}'); new_lds=$(echo "$new_line" | awk '{print $4}')
base_vgpr=$(echo "$base_line" | awk '{print $2}'); new_vgpr=$(echo "$new_line" | awk '{print $2}')
[ "${base_lds:-x}" = 38944 ] || fail "base attn_core reserves ${base_lds:-?} B, wanted 38944 — the BASE build is not HEAD's pre-port object"
[ "${new_lds:-x}"  = 6176 ] || fail "new attn_core reserves ${new_lds:-?} B, wanted 6176 — THE PORT DID NOT LAND on the shipping path; nothing below this line would be an A/B"
[ "${new_vgpr:-x}" -lt "${base_vgpr:-999}" ] 2>/dev/null || fail "new next_free_vgpr=${new_vgpr:-?} not below base ${base_vgpr:-?} — same object twice?"
echo "  CANARY 1 ok: two distinct code objects, 38944 -> 6176 B, vgpr $base_vgpr -> $new_vgpr"

# -------------------------------------------------------------------- prompt fixtures (CPU only) --
$VENV prototype/attn_decode_wide/mkids_d3.py "$OUT" > "$OUT/mkids.log" 2>&1 || fail "mkids_d3.py failed"
want_n() { awk -F',' '{print NF}' "$1"; }
echo "== fixtures: ids128.csv=$(want_n "$OUT/ids128.csv") ids, ids4032.csv=$(want_n "$OUT/ids4032.csv") ids =="

# ------------------------------------------------------------------- numerics, before any ms -----
echo "== numerics on identical fixtures, both binaries (fixture list: $GATE_FIXTURES) =="
for fx in $GATE_FIXTURES; do
    for arm in base new; do
        [ "$arm" = base ] && bin="$BASE_BIN" || bin="$NEW_BIN"
        if $VENV engine/gate.py --binary "$bin" --fixture "$fx" --work "$OUT/gate_${arm}_${fx}" \
                > "$OUT/gate_${arm}_${fx}.log" 2>&1; then
            echo "  gate $arm $fx: $(grep -E '^ENGINE GATE' "$OUT/gate_${arm}_${fx}.log" | tail -1)"
        else
            echo "  gate $arm $fx: $(grep -E '^ENGINE GATE' "$OUT/gate_${arm}_${fx}.log" | tail -1)"
            fail "engine/gate.py $arm $fx went red — an numerics red outranks this timing window entirely"
        fi
    done
done

# Byte claim on the tokens themselves, same prompt, same step count, both binaries. The engine's logits
# bits are the gate's business; `sampled.txt` is the cheap direct evidence that the port moved no
# decision. This block used to sit INSIDE the fixture loop while always running `ids128.csv --steps 16`,
# so it printed an identical "112+16 lanes" line once per fixture and never compared one thing at 4k —
# a check whose independent variable is not the variable under test (AGENTS). It is now driven by the
# WIDTHS, and it prints the width it actually ran rather than a literal.
for w in 128 4032; do
    if [ "$w" = 128 ]; then bids="$OUT/ids128.csv"; bsteps=16; else bids="$OUT/ids4032.csv"; bsteps=32; fi
    for arm in base new; do
        [ "$arm" = base ] && bin="$BASE_BIN" || bin="$NEW_BIN"
        mkdir -p "$OUT/byte_${w}_$arm"
        "$bin" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids-file "$bids" --steps "$bsteps" --out "$OUT/byte_${w}_$arm" \
            > "$OUT/byte_${w}_$arm.log" 2>&1 || fail "byte-identity run ($arm, $w lanes) failed"
    done
    ntok=$(want_n "$bids")
    if diff -q "$OUT/byte_${w}_base/sampled.txt" "$OUT/byte_${w}_new/sampled.txt" >/dev/null 2>&1; then
        echo "  BYTE-IDENTICAL sampled.txt, both binaries, at $w lanes: ${ntok} prompt tokens + $bsteps steps, $(wc -l < "$OUT/byte_${w}_new/sampled.txt") lines, last step $((ntok + bsteps)) lanes"
    else
        fail "sampled.txt differs between base and new at $w lanes — the port is not decision-neutral"
    fi
done
NUM_OK=1

# ------------------------------------------------------------------------------------ timing -----
tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT
echo "== lock held: $(tools/gpu_lock.sh status 2>&1 | head -1) ; util at acquire: $(tools/gpu_lock.sh util 2>&1 | tail -1)"

run() {   # run <arm> <idsfile> <steps> <want_split> <want_last_lanes> -> "attn_ms total_ms"
    local arm="$1" ids="$2" steps="$3" wsplit="$4" wlanes="$5"
    local bin="$BASE_BIN" log od
    [ "$arm" = new ] && bin="$NEW_BIN"
    log="$OUT/run_${arm}_$(basename "$ids" .csv)_${REPCUR}.log"
    od="$OUT/out_${arm}_$(basename "$ids" .csv)_${REPCUR}"; mkdir -p "$od"
    "$bin" --mtp-accept 0 --mtp-draft 0 --model "$MODEL" --ids-file "$ids" --steps "$steps" --time-stages --out "$od" \
        > "$log" 2>&1 \
        || { echo "CANARY-FAIL: $arm run rc=$? (see $log)" >&2; return 1; }
    # --- the exercised-path canary, asserted on EVERY run, not just the first -------------------
    local tk want_tk
    want_tk=$(want_n "$ids")
    tk=$(grep -E '^PREFILL' "$log" | head -1 | sed -E 's/.*tokens=([0-9]+).*/\1/')
    [ "${tk:-x}" = "$want_tk" ] || { echo "CANARY-FAIL: $arm prefilled tokens=$tk, the id file holds $want_tk fields" >&2; return 1; }
    grep -q "^SPLITCAP split=$wsplit " "$log" \
        || { echo "CANARY-FAIL: $arm printed no 'SPLITCAP split=$wsplit' line — the decode attention did not run the geometry being timed" >&2; return 1; }
    grep -q "^SPLITCAP split=$wsplit .* lanes=$wlanes " "$log" \
        || { echo "CANARY-FAIL: $arm ran split=$wsplit but never at lanes=$wlanes — the lane label on this row would be a guess" >&2; return 1; }
    grep -q 'STAGEPARITY .*tokens_identical=1 logits_bitidentical=1' "$log" \
        || { echo "CANARY-FAIL: $arm's own replay is not reproducible (no STAGEPARITY 1/1) — the box or the run is unwell" >&2; return 1; }
    local attn tot
    attn=$(grep 'STAGE gran=stage family=attn' "$log" | head -1 | sed -E 's/.*ms_per_step=([0-9.]+).*/\1/')
    # The burst headline line is `STAGEPASS gran=burst steps=N ms_mean=...`. There is NO `family=TOTAL`
    # line at ANY granularity — the only TOTAL line is `STAGE gran=stage TOTAL ... ms_per_step=` — so a
    # parser hunting `gran=burst family=TOTAL` matches nothing. Without the empty-parse guard below that
    # hole would have printed a blank headline column; with it, it aborts, which is what it just did.
    tot=$(grep 'STAGEPASS gran=burst' "$log" | head -1 | sed -E 's/.*ms_mean=([0-9.]+).*/\1/')
    case "$attn" in ''|*[!0-9.]*) echo "CANARY-FAIL: $arm no attn ms parsed from $log" >&2; return 1;; esac
    case "$tot"  in ''|*[!0-9.]*) echo "CANARY-FAIL: $arm no burst TOTAL ms parsed from $log" >&2; return 1;; esac
    grep 'STAGE gran=stage family=attn' "$log" | head -1 | sed 's/^/#   /' >&2
    echo "$attn $tot"
}

for width in 4032 128; do
    [ "$width" = 4032 ] && { ids="$OUT/ids4032.csv"; steps=32; wsplit=8; wlanes=4032; } \
                        || { ids="$OUT/ids128.csv";  steps=16; wsplit=1; wlanes=128; }
    : > "$OUT/attn_$width.txt"; : > "$OUT/tot_$width.txt"
    for rep in $(seq "$REPS"); do
        REPCUR="$rep"
        for arm in base new; do
            line=$(run "$arm" "$ids" "$steps" "$wsplit" "$wlanes") \
                || fail "window aborted: $arm at $width lanes (CANARY-FAIL above) — NO-TIMING"
            echo "$arm $line" >> "$OUT/attn_$width.txt"
            echo "$arm $line" | awk '{print $1, $3}' >> "$OUT/tot_$width.txt"
        done
        echo "  rep $rep/$REPS at $width lanes done"
    done
done
tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1
trap - EXIT

# ----------------------------------------------------------------------- the report ---------------
[ "$NUM_OK" = 1 ] || { echo "NO-TIMING: numerics did not pass"; exit 1; }
for width in 4032 128; do
    [ "$width" = 4032 ] && label="4,032 lanes (4,000 prompt + 32 steps), split 8 — the pre-registered width" \
                        || label="128 lanes (112 prompt + 16 steps), split 1 — FLAT HERE IS A PREDICTED PASS (HANDOFF-D3.md §4.3: 8 blocks on 40 CUs is one round at either residency)"
    echo
    echo "=== $width lanes — $label"
    printf "%-6s %12s %14s %12s %14s %10s\n" arm attn_ms/step spread tok_per_s burst_ms/tok spread
    for arm in base new; do
        awk -v a="$arm" '$1==a{print $2}' "$OUT/attn_$width.txt" > "$OUT/.a_$arm"
        awk -v a="$arm" '$1==a{print $2}' "$OUT/tot_$width.txt"  > "$OUT/.t_$arm"
        am=$(med "$OUT/.a_$arm"); as=$(spread "$OUT/.a_$arm")
        tm=$(med "$OUT/.t_$arm"); ts=$(spread "$OUT/.t_$arm")
        printf "%-6s %12s %14s %12s %14s %10s\n" "$arm" "$am" "$as" \
            "$(awk -v m="$tm" 'BEGIN{printf "%.1f", 1000/m}')" "$tm" "$ts"
    done
    base_med=$(awk '$1=="base"{print $2}' "$OUT/attn_$width.txt" > "$OUT/.a_b"; med "$OUT/.a_b")
    new_med=$(awk '$1=="new"{print $2}' "$OUT/attn_$width.txt" > "$OUT/.a_n"; med "$OUT/.a_n")
    awk -v n="$new_med" -v b="$base_med" -v w="$width" 'BEGIN{
        printf "  attn line: %s -> %s ms/step (%+.2f %%)\n", b, n, (n-b)/b*100
        if (w != 4032) { print "  128-lane row: no pre-registered verdict (0.40 ms was declared out of reach in advance)"; exit }
        if (n+0 <= 1.50)   print "  VERDICT at 4,032: WIN  (pre-registered <= 1.50 ms)"
        else if (n+0 >= 2.55) print "  VERDICT at 4,032: KILL (pre-registered >= 2.55 ms)"
        else print "  VERDICT at 4,032: NEITHER (between 1.50 and 2.55 ms) — report it as neither"
        x = 2*(3.023 - n)
        printf "  X implied by this row, from X = 2*(3.023 - lean): %.2f  (projection said ~0.64)\n", x
    }'
done
echo
echo "# quote each arm as a pair with its spread, never as a best (engine/performance-status.md §6)."
echo "# raw per-rep files: $OUT/attn_*.txt, $OUT/tot_*.txt; every run's full log: $OUT/run_*.log"
echo "# canaries seen: code object ($base_lds -> $new_lds B), PREFILL tokens=, SPLITCAP, STAGEPARITY,"
echo "#   engine/gate.py per fixture, sampled.txt byte-diff. All asserted inside this window."
