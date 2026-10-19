#!/usr/bin/env bash
# a7_parse_selftest.sh — prove `a7_host_split.sh`'s readers report the right number for a real
# line shape, and CANNOT report a number for a line that is absent.
#
# Why this exists: the driver's whole claim is "every named phase produced a number". A reader
# that hands back "" for a missing line turns an absent `WAVEPASS gran=burst` into a 0 ms warm
# wave, and the gap then equals `ms_total` exactly — a plausible-looking 119.830 ms constant
# invented out of nothing. That is the empty-parse failure `AGENTS.md` records for
# `gdn_collapse/run.py` (regex matched zero CASE lines, printed PASS) and `gate.py` (1023
# missing rows, still checked floors).
#
# Fixture: `a7_fixture.log`, whose field order is transcribed from a real run's output. The
# negatives are the fixture with one line class deleted, so a fixture edit cannot quietly stop
# exercising a reader.
#
# No device work, no timings:
#   bash prototype/perprompt/a7_parse_selftest.sh
set -uo pipefail
export LC_ALL=C
cd "$(dirname "$0")/../.."

HERE=prototype/perprompt
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

GOOD=$TMP/good.log
cp "$HERE/a7_fixture.log" "$GOOD"
# §49's own arithmetic is the witness: cold 119.830 − warm burst 48.733 = 71.097.
COLD=119.830 WARM=48.7330 WANT_GAP=71.097

NOWAVE=$TMP/no_wavepass.log;      grep -v '^WAVEPASS' "$GOOD" > "$NOWAVE"
NORESULT=$TMP/no_result.log;     grep -v '^RESULT' "$GOOD" > "$NORESULT"
NOLOAD=$TMP/no_load.log;         grep -v '^loaded ' "$GOOD" > "$NOLOAD"
NOPHASE=$TMP/no_a7phase.log;     grep -v '^A7PHASE' "$GOOD" > "$NOPHASE"
NOREPEAT=$TMP/no_a7repeat.log;   grep -v '^A7REPEAT' "$GOOD" > "$NOREPEAT"

die() { echo "die: $*" >&2; exit 70; }
# `die` must exit the caller, and `die` is defined before the lib is sourced for that reason.
. "$HERE/a7_lib.sh"

fails=0
expect() {   # expect LABEL GOT WANT
    if [ "$2" = "$3" ]; then echo "ok   $1 = $2"
    else echo "FAIL $1 = '$2', want '$3'"; fails=$((fails + 1)); fi
}
expect_abort() {   # expect_abort LABEL FUNC ARGS...
    local label=$1; shift
    if ( "$@" ) >/dev/null 2>&1; then
        echo "FAIL $label returned a value from a log with that line class deleted"
        fails=$((fails + 1))
    else
        echo "ok   $label aborts instead of inventing a number"
    fi
}
count_is() {   # count_is LABEL FILE PREFIX WANT
    local got
    got=$(grep -c "^$3" "$2")
    expect "$1" "$got" "$4"
}

echo "-- positive: every reader finds its field in the fixture --"
expect load_ms          "$(load_ms "$GOOD")"                                 "524.3"
expect ms_total         "$(kv "$GOOD" ms_total RESULT)"                      "$COLD"
expect warm_burst_wall  "$(wave_wall "$GOOD" burst legA)"                    "$WARM"
expect warm_stage_wall  "$(wave_wall "$GOOD" stage legA)"                    "52.1040"
expect parity_bitexact  "$(kv "$GOOD" logits_bitidentical WAVEPARITY)"       "1"
expect gap_eq_49        "$(diff_ms "$COLD" "$WARM")"                         "$WANT_GAP"
expect a7phase_name     "$(grep '^A7PHASE name=scratch_alloc ' "$GOOD" | sed -E 's/.*name=([a-z_]+) .*/\1/')" "scratch_alloc"
expect a7phase_ms       "$(grep '^A7PHASE name=scratch_alloc ' "$GOOD" | sed -E 's/.* ms=([0-9.]+)/\1/')" "3.910"
expect a7body_submit    "$(grep '^A7BODY' "$GOOD" | head -1 | sed -E 's/.*submit_ms=([0-9.]+)/\1/')" "6.402"
expect a7sync1_device   "$(grep '^A7SYNC1' "$GOOD" | head -1 | sed -E 's/.*device_ms=([0-9.]+)/\1/')" "113.4100"
expect median_of_three  "$(printf '%s\n' 71.0 70.9 64.9 | med)"              "70.9"
count_is repeat_wave_n  "$GOOD" "A7REPEAT kind=repeat" 3
count_is phase_line_n   "$GOOD" "A7PHASE" 8

echo
echo "-- negative: a deleted line class must abort, never read as zero --"
expect_abort wavepass-absent   wave_wall "$NOWAVE" burst legA
expect_abort result-absent     kv "$NORESULT" ms_total RESULT
expect_abort loadline-absent   load_ms "$NOLOAD"
expect_abort a7phase-absent    sh -c "grep -c '^A7PHASE' '$NOPHASE' | grep -qx 8"
expect_abort a7repeat-absent   sh -c "grep -c '^A7REPEAT kind=repeat' '$NOREPEAT' | grep -qx 3"
expect_abort unknown-key       kv "$GOOD" definitely_not_a_key x

echo
echo "-- the specific §49 lie this rules out --"
if [ "$(diff_ms "$COLD" 0)" = "$COLD" ]; then
    echo "ok   reading an absent warm wall as 0 would print a ${COLD} ms gap; wave_wall dies first"
else
    echo "FAIL diff_ms"; fails=$((fails + 1))
fi
if (printf '\n' | med) >/dev/null 2>&1; then
    echo "FAIL med took a blank rep list as a median"; fails=$((fails + 1))
else
    echo "ok   med refuses an empty rep list"
fi

echo
if [ "$fails" = 0 ]; then echo "SELFTEST PASS — readers are total, not partial"; exit 0; fi
echo "SELFTEST FAIL: $fails check(s)"
exit 1
