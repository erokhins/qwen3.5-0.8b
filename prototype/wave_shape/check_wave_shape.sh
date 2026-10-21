#!/usr/bin/env bash
# check_wave_shape.sh — the correctness gate for this rig. Takes NO timings.
#
# A builder may run this (AGENTS: short, correctness-only, answers PASS/FAIL, one device hold, under a
# minute). Everything it checks is either pure host arithmetic or a deterministic device witness; no
# burst, no best-of-N, no ms figure is produced or quoted. The timed path is sweep.sh, main-agent only.
#
# It also witnesses the CHECKERS, not just the rig: a gate that cannot fail is decoration, so section C
# feeds fit_shape.py inputs it must refuse, and section D feeds it inputs it must accept and name a
# verdict for. If section D ever goes green while the expected verdicts change colour, the fitter has
# become vacuous.
set -u
export LC_ALL=C
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
BIN="$HERE/wave_shape"
OUT="${OUT:-$HERE/out}"
mkdir -p "$OUT"
PASS=0; FAIL=0; N=0
ck() {   # ck <name> <expected-token> <command...>
    local name=$1 want=$2; shift 2
    local out rc
    out=$("$@" 2>&1); rc=$?
    N=$((N+1))
    if printf '%s' "$out" | grep -q -- "$want"; then
        PASS=$((PASS+1)); echo "  ok   $name"
    else
        FAIL=$((FAIL+1)); echo "  FAIL $name (want '$want', rc=$rc)"; printf '%s\n' "$out" | head -3 | sed 's/^/         /'
    fi
}
ckabsent() {  # the check itself is only real if the bad token is absent AND the good one present
    local name=$1 absent=$2 want=$3; shift 3
    local out
    out=$("$@" 2>&1)
    N=$((N+1))
    if printf '%s' "$out" | grep -q -- "$absent"; then
        FAIL=$((FAIL+1)); echo "  FAIL $name (forbidden token '$absent' present)"
    elif printf '%s' "$out" | grep -q -- "$want"; then
        PASS=$((PASS+1)); echo "  ok   $name"
    else
        FAIL=$((FAIL+1)); echo "  FAIL $name (want '$want')"; printf '%s\n' "$out" | head -3 | sed 's/^/         /'
    fi
}

echo "== A. build and geometry (host only, no HIP)"
ck build "^built " "$HERE/build.sh"
ck selftest-geom "SELFTEST-GEOM ok" "$BIN" --selftest-geom
ck plan-cross-check "PLAN-DIFF ok" python3 "$HERE/plan_table.py"
ck device-facts-nonzero "cus=40" "$BIN" --print-device

echo "== B. the rig refuses a cell that cannot be booked (host only)"
ck refuse-cold-pages REFUSED "$BIN" --launches 16 --grid 8 --pages cold
ck refuse-indivisible-cell ABORT "$BIN" --launches 7 --grid 7
ck refuse-bad-split ABORT "$BIN" --down two --split 3
ck refuse-bad-wpr ABORT "$BIN" --down intra --wpr 3
ck refuse-no-arm ABORT "$BIN" --reps 3
ck refuse-unknown-arg ABORT "$BIN" --launches 16 --grid 8 --bogus 1
ck refuse-bad-neg ABORT "$BIN" --launches 16 --grid 8 --neg not_a_control
ck arm-axis-listed "reachable=1" "$BIN" --print-plan

# A refusal that cannot be triggered proves nothing, so prove each of these is a real code path by
# running the neighbour of the refused input and requiring a different outcome.
ckabsent neighbour-of-refused-cell "reachable=0" "reachable=1" python3 -c \
   "import subprocess,sys; sys.stdout.write('reachable=1' if subprocess.run(['$BIN','--launches','48','--grid','32'],capture_output=True).returncode==0 else 'reachable=0')"
ck refuse-msg-says-why "does not divide the pinned read total" "$BIN" --launches 48 --grid 24

echo "== C. the fitter refuses inputs that cannot carry a claim"
# The format strings are kept on ONE physical line each: a backslash-newline inside a single-quoted
# printf format is not a line continuation, so the first draft emitted a literal backslash and split
# every fixture row in half, and the fitter correctly reported "no RESULT lines" for input it had
# mangled itself. A fixture that cannot be parsed teaches nothing about the parser.
mk_result() {  # mk_result <family> <arm> <L> <G> <blocks_total> <ms> <pass-tag>
    printf 'RESULT arm=%s family=%s launches=%s grid=%s blocks_total=%s wavefronts_total=%s wavefronts_per_cu=25.60 read_mib=1464.0000 write_mib=22.8750 demand_mib=1486.8750 verify=ok launches_executed=%s syncs_executed=1 ms_min=%s ms_median=%s ms_max=%s per_launch_us=%s per_block_us=%s reps=5 iters=3 %s\n' "$2" "$1" "$3" "$4" "$5" "$(( ${5} * 8 ))" "$3" "$6" "$6" "$6" "$(python3 -c "print(round($6 * 1000.0 / $3, 4))")" "$(python3 -c "print(round($6 * 1000.0 / $5, 4))")" "$7"
}
mk_down() {    # mk_down <arm> <ms> <pass-tag>
    printf 'RESULT arm=%s family=down launches=1 grid=128x1 blocks_total=128 wavefronts_total=1024 wavefronts_per_cu=25.60 slices=1 act=f32 weight_mib=7.0000 stage_kib=1792.0 part_kib=0.0 verify=ok mirror=fma vs_ship_ndiff=0 max_ulp=0 launches_executed=1 syncs_executed=1 ms_min=%s ms_median=%s ms_max=%s per_launch_us=%s per_block_us=%s reps=5 iters=3 %s\n' "$1" "$2" "$2" "$2" "$(python3 -c "print(round($2 * 1000.0, 4))")" "$(python3 -c "print(round($2 * 1000.0 / 128, 4))")" "$3"
}
GOOD="$OUT/good_cells.txt"; GOODD="$OUT/good_downs.txt"
ONE="$OUT/onepass_cells.txt"; ONED="$OUT/onepass_downs.txt"
kill_fixture() {   # every mat cell at the same ms: diagonals flat and no shape term -> KILL.
    # Writes into $GOOD/$GOODD, which the caller points wherever it wants the fixture to land.
    local ms=$1 ms2=$2 splitms=$3 passes=${4:-2}
    local p a tag msv
    : > "$GOOD"; : > "$GOODD"
    for tag in 1 2; do
        [ "$tag" -le "$passes" ] || continue
        msv=$ms; [ "$tag" = 2 ] && msv=$ms2
        for p in "16 8" "16 16" "16 64" "16 256" "32 32" "32 128" "48 8" "48 16" "64 16" "64 64" \
                 "128 8" "128 32"; do
            set -- $p
            printf 'CASE cell-%sx%s\n' "$1" "$2" >> "$GOOD"
            mk_result mat "mat|$1|$2|b$(( $1 * $2 ))" "$1" "$2" "$(( $1 * $2 ))" "$msv" "pass=$tag" >> "$GOOD"
        done
        for a in "down|ship|f32" "down|intra4|f32" "down|two4|f32"; do
            printf 'CASE %s\n' "$a" >> "$GOODD"; mk_down "$a" "$msv" "pass=$tag" >> "$GOODD"
        done
    done
    # The split-K arms are the ones whose gain decides WIN/KILL, so they alone get $splitms, which the
    # caller sets just below (kill 9.95 -> 0.05 ms gain, inconclusive 9.50 -> 0.50 ms gain) against the
    # pre-registered 0.15 ms line. shipped down ms stays 10.000 like every other row.
    sed -i "/down|intra4/s/ms_min=[0-9.]*/ms_min=$splitms/;s/ms_median=[0-9.]*/ms_median=$splitms/;s/ms_max=[0-9.]*/ms_max=$splitms/;s/per_launch_us=[0-9.]*/per_launch_us=$splitms/" "$GOODD"
    sed -i "/down|two4/s/ms_min=[0-9.]*/ms_min=$splitms/;s/ms_median=[0-9.]*/ms_median=$splitms/;s/ms_max=[0-9.]*/ms_max=$splitms/;s/per_launch_us=[0-9.]*/per_launch_us=$splitms/" "$GOODD"
}
# A refusal test is only real if the same fixture WITHOUT the defect passes, so every bad fixture below
# is a good one plus exactly the defect under test, and every good fixture is also asserted to produce a
# verdict. A fitter that only ever refuses is as useless as one that only ever passes.
GOOD="$ONE"; GOODD="$ONED"; kill_fixture 10.000 10.050 0.900 1
GOOD="$OUT/good_cells.txt"; GOODD="$OUT/good_downs.txt"
kill_fixture 10.000 10.050 9.950 2
ck fitter-accepts-a-valid-matrix "RESULT-LINE ok" python3 "$HERE/check_result_line.py" "$(head -2 "$GOOD" | tail -1)"
ck fitter-verdict-kill "VERDICT KILL" python3 "$HERE/fit_shape.py" "$GOOD" "$GOODD" --quiet --json "$OUT/kill.json"
kill_fixture 10.000 10.050 9.500 2
ck fitter-splitk-too-big-to-kill "VERDICT INCONCLUSIVE" python3 "$HERE/fit_shape.py" "$GOOD" "$GOODD" --quiet
kill_fixture 10.000 10.050 9.950 2
cat "$GOOD" "$GOODD" > "$OUT/allgood.txt"
{ cat "$OUT/allgood.txt"; printf 'CASE cell-orphan-with-no-result\n'; } > "$OUT/orphan.txt"
{ cat "$OUT/allgood.txt"; printf 'NO-TIMING arm=mat|16|8|b128 reason=verify_not_ok\n'; } > "$OUT/notiming.txt"
{ printf 'CASE cell-drift\n'; printf 'RESULT arm=mat|drift family=mat launches=16 grid=8 blocks_total=128 wavefronts_total=1024 wavefronts_per_cu=25.6 read_mib=99.0000 write_mib=22.8750 demand_mib=1486.8750 verify=ok launches_executed=16 syncs_executed=1 ms_min=10.0 ms_median=10.0 ms_max=10.0 per_launch_us=625.0 per_block_us=78.1 reps=5 iters=3 pass=1\n'; } > "$OUT/drift.txt"
printf 'RESULT arm=mat|16|8|b128 family=mat launches=16 grid=8 blocks_total=128 wavefronts_total=1024 read_mib=1464.0000 write_mib=22.8750 demand_mib=1486.8750 verify=ok launches_executed=16 syncs_executed=1 ms_min=1.0 ms_median=1.0 ms_max=1.0 per_launch_us=62.5 per_block_us=7.8 reps=5 iters=3\n' > "$OUT/trunc.txt"
ck fitter-refuse-empty "REFUSE empty" python3 "$HERE/fit_shape.py" /dev/null
ck fitter-refuse-result-without-case "REFUSE partial" python3 "$HERE/fit_shape.py" "$OUT/trunc.txt"
ck fitter-refuse-case-without-result "REFUSE partial" python3 "$HERE/fit_shape.py" "$OUT/orphan.txt"
ck fitter-refuse-no-timing "REFUSE partial" python3 "$HERE/fit_shape.py" "$OUT/notiming.txt"
ck fitter-refuse-single-pass-no-noise "REFUSE noise" python3 "$HERE/fit_shape.py" "$ONE" "$ONED"
ck fitter-refuse-drifted-footprint "REFUSE footprint" python3 "$HERE/fit_shape.py" "$OUT/drift.txt" "$GOOD"
ck fitter-refuse-count-mismatch "REFUSE count" python3 "$HERE/fit_shape.py" "$GOOD" "$GOODD" <(echo "CELLS-PRINTED=999")
ck check-result-line-rejects-short-line "RESULT-LINE BAD" python3 "$HERE/check_result_line.py" "RESULT arm=x family=mat launches=2 ms_min=1.0"

echo "== D. device witnesses (one lock hold, correctness only, no timing)"
cd "$HERE/../.." || exit 2
TAG="wave_shape item2 correctness gate (worker, no timing)"
tools/gpu_lock.sh acquire "$TAG" >/dev/null || { echo "ABORT: GPU lock busy"; exit 2; }
ck device-selftest "SELFTEST ok" "$BIN" --selftest
ckabsent device-selftest-has-no-FAIL "FAIL" "SELFTEST ok" "$BIN" --selftest
ck cell-one-block-per-launch "CLEAN arm=mat|1024|1" "$BIN" --launches 1024 --grid 1
ck cell-thousand-blocks-per-launch "CLEAN arm=mat|48|1024" "$BIN" --launches 48 --grid 1024
ck cell-iso-pop "CLEAN arm=mat|32|32" "$BIN" --launches 32 --grid 32
ck down-ship-bit-exact "vs_ship_ndiff=0 max_ulp=0" "$BIN" --down ship
ck down-intra1-bit-exact "vs_ship_ndiff=0" "$BIN" --down intra --wpr 1
ck down-two4-runs-clean "CLEAN arm=down|two4" "$BIN" --down two --split 4
ck untimed-cell-prints-no-ms "NO-TIMING" "$BIN" --launches 16 --grid 8
tools/gpu_lock.sh release "$TAG" >/dev/null

echo
echo "CHECK-WAVE-SHAPE $( [ "$FAIL" -eq 0 ] && echo ok || echo FAILED ) checks=$N pass=$PASS fail=$FAIL"
echo "no timing taken: this gate prints no millisecond figure; the timed path is sweep.sh (main agent)."
[ "$FAIL" -eq 0 ] || exit 1
