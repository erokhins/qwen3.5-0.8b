#!/usr/bin/env bash
# meter_identity.sh — the meter must not change what the engine DECIDES. Correctness only, no timing.
#
# WHY. `--mtp-meter 1` bounds a phase that is pure async submissions by adding a `hipDeviceSynchronize()`
# the shipping path does not perform (`engine/mtp_meter.h`). The claim being defended is the strong one: an
# added WAIT is not an added SUBMISSION — it moves no byte and touches no state — so the emitted token
# stream, the per-round verdicts, the accept statistics and the state fingerprints must be IDENTICAL with the
# instrument on and off. Anything weaker is a meter that changed the machine it reports on, and this repo has
# already watched a flag that moved the target's token stream read itself as a numerics result
# (`prototype/mtp_accept/M3-RESULT.md` §3).
#
# WHAT IT COMPARES, per shape (verify side pass, accept loop), OFF vs ON:
#   emitted.txt / sampled.txt   the token stream, byte for byte
#   ACCEMIT / STEP lines        every emitted token and its position, in order
#   ACCEPT / VERIFY2 lines      the per-round and per-step verdicts, logit VALUES included — a body whose
#                               operands had drifted shows up in `row1_top` long before it shows up in a token
#   MTPSTAT line                rounds / accepted / rejected / emitted / accept_rate / bytes
#   STATEFP lines               the §08 state fingerprints at prompt end and run end (`--state-fp`)
#   DECWALL / RESULT lines      compared FIELD BY FIELD, not as strings: the difference is classified into
#                               `[timing]` (the meter's own fields and the duration fields a device wait is
#                               allowed to move) and `[DECISION]` (everything else — `last_sampled`,
#                               `prefix_len`, `stopped`, `steps`, `bodies`, `rounds`, `plain`, `accepted`,
#                               `rejected`, `suppressed`, `emitted`, `stop`, `ended`, `census`). Any
#                               `[DECISION]` difference fails the run.
#
# WHY THE FIELD SPLIT AND NOT A STRING DIFF. A string compare fails on the one thing the instrument is
# permitted to change — wall-clock fields move because the meter adds waits and one calibration burst — and a
# check that fails for the expected difference is one people learn to ignore. Blanket-stripping the durations
# would be weaker still: it could not tell "only the durations moved" from "a counter moved too". So the
# durations are not hidden, they are NAMED on the output beside their old and new values, and every field that
# is not a duration has to match.
#
# NO LOCK HERE, on purpose. Like `engine/gate.py`, this script does not self-acquire `tools/gpu_lock.sh`: it
# is meant to be driven inside one lock hold the caller owns — a caller that holds the lock and then runs a
# self-locking child deadlocks it (`AGENTS.md`, the self-locking-runners row). So the caller holds the lock.
#
# Usage: prototype/mtp_accept/meter_identity.sh [ids.csv]     # default /tmp/dec8_ids.csv
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1
IDS=${1:-/tmp/dec8_ids.csv}
[ -f "$IDS" ] || { echo "ABORT: $IDS missing. Make one with: .venv/bin/python3 -c \"import json;print(','.join(map(str,json.load(open('data/refs/dec8/manifest.json'))['prefill_ids'])))\" > /tmp/dec8_ids.csv"; exit 2; }
N=$(awk -F',' '{print NF}' "$IDS")
[ "$N" -gt 1 ] || { echo "ABORT: $IDS has $N fields; a one-token prompt runs no attention wave and proves nothing"; exit 2; }
BIN=${BIN:-engine/build/decode_run}
MODEL=${MODEL:-models/Qwen3.5-0.8B}
OUT=${OUT:-tmp/meter_identity}
STEPS=${STEPS:-12}
rm -rf "$OUT"; mkdir -p "$OUT"
echo "== meter identity: $IDS ($N fields), steps=$STEPS, bin=$BIN"

run() { # $1 name $2... flags
    local name="$1"; shift
    mkdir -p "$OUT/$name"
    "$BIN" --mtp-accept 0 --model "$MODEL" --ids-file "$IDS" --steps "$STEPS" --out "$OUT/$name" \
           --no-stop --state-fp --gdn-loop 1 "$@" > "$OUT/$name.log" 2>&1
    local rc=$?
    [ "$rc" = 0 ] || { echo "ABORT: $name exited $rc"; tail -5 "$OUT/$name.log"; exit 1; }
    echo "   ran $name (rc 0)"
}

# field_diff: classify a one-line pair's differences. Anything that is not a duration or one of the meter's
# own fields is a DECISION difference and fails. A field present on only ONE side counts as a difference too,
# which is how the meter's appended fields are accounted for rather than quietly absent from the comparison.
# The 4th argument adds extra allowed names for that line only: `rounds` is the meter's own field on `RESULT`
# and a decision counter on `DECWALL`, and a shared allow-list would excuse the second one.
field_diff() { # $1 label $2 off-line $3 on-line [$4 extra allowed names]
    local out rc extra="${4:-}"
    out=$(awk -v label="$1" -v a="$2" -v b="$3" -v extra="$extra" 'BEGIN {
        ne = split("meter sum_ms_per_round wall_ms_per_round coverage ms_total ms_per_step prefill_ms decode_ms ms_per_token tok_per_s", t, " ")
        for (i = 1; i <= ne; i++) allowed[t[i]] = 1
        nx = split(extra, e, " "); for (i = 1; i <= nx; i++) if (e[i] != "") allowed[e[i]] = 1
        n = split(a, x, " "); for (i = 1; i <= n; i++) { k = x[i]; v = k; sub(/=.*/, "", k); sub(/^[^=]*=/, "", v); A[k] = v; seen[k] = 1 }
        m = split(b, y, " "); for (i = 1; i <= m; i++) { k = y[i]; v = k; sub(/=.*/, "", k); sub(/^[^=]*=/, "", v); B[k] = v; seen[k] = 1 }
        bad = 0; nd = 0; nt = 0
        for (k in seen) {
            if ((k in A) && (k in B) && A[k] == B[k]) continue
            nd++
            if (allowed[k]) { nt++; printf "       [timing] %s: %s -> %s\n", k, (k in A) ? A[k] : "(absent)", (k in B) ? B[k] : "(absent)" }
            else { printf "       [DECISION] %s: %s -> %s\n", k, (k in A) ? A[k] : "(absent)", (k in B) ? B[k] : "(absent)"; bad = 1 }
        }
        if (nd == 0) printf "   %-9s %-9s identical field-for-field\n", "", label
        else if (!bad) printf "   %-9s %-9s identical apart from %d duration/meter field(s), named above\n", "", label, nt
        exit bad
    }')
    rc=$?
    printf '%s\n' "$out"
    return $rc
}

# line_is: every line with this prefix must be byte-identical between the two logs. `present` is decided from
# the metered log: a line the OFF run never printed is not a difference, it is a line this shape has none of.
compare() { # $1 shape $2 off-log $3 on-log $4 dump-dir-off $5 dump-dir-on
    local shape="$1" off="$2" on="$3" doff="$4" don="$5" bad=0 f
    for f in ACCEMIT ACCEPT STEP VERIFY2 STATEFP MTPSTAT PREFILL; do
        if grep -q "^$f " "$on"; then
            if diff -q <(grep "^$f " "$off") <(grep "^$f " "$on") > /dev/null; then
                printf '   %-9s %-8s identical (%s lines)\n' "$shape" "$f" "$(grep -c "^$f " "$on")"
            else
                printf '   %-9s %-8s DIFFERS\n' "$shape" "$f"
                diff <(grep "^$f " "$off") <(grep "^$f " "$on") | head -6 | sed 's/^/       /'
                bad=1
            fi
        fi
    done
    for f in emitted.txt sampled.txt; do
        if [ -f "$doff/$f" ] && [ -f "$don/$f" ]; then
            if diff -q "$doff/$f" "$don/$f" > /dev/null; then
                printf '   %-9s %-8s identical (%s lines)\n' "$shape" "$f" "$(wc -l < "$don/$f")"
            else
                printf '   %-9s %-8s DIFFERS\n' "$shape" "$f"
                diff "$doff/$f" "$don/$f" | head -6 | sed 's/^/       /'
                bad=1
            fi
        fi
    done
    field_diff "DECWALL" "$(grep -m1 '^DECWALL ' "$off")" "$(grep -m1 '^DECWALL ' "$on")" || bad=1
    field_diff "RESULT"  "$(grep -m1 '^RESULT mode=' "$off")" "$(grep -m1 '^RESULT mode=' "$on")" rounds || bad=1
    return $bad
}

# The meter's own `rounds=` on RESULT is excused from the field compare (it exists only when the meter is on),
# so it has to be ADJUDICATED rather than excused: the meter must have counted exactly the rounds the shape
# ran, or every `ms_per_round` on its table divides a population the run never had.
meter_rounds() { grep -m1 '^RESULT mode=' "$1" | tr ' ' '\n' | sed -n 's/^rounds=//p'; }
adjudicate_rounds() { # $1 shape $2 on-log $3 want
    local got; got=$(meter_rounds "$2")
    if [ "${got:-none}" = "$3" ]; then
        printf '   %-9s %-8s meter rounds=%s, same population as the run\n' "$1" "rounds" "$got"
    else
        printf '   %-9s %-8s meter rounds=%s, the run did %s — the meter and the run counted different populations\n' "$1" "rounds" "${got:-none}" "$3"
        return 1
    fi
}

FAILED=0
echo "-- shape 1: the read-only side pass (--mtp-verify 1)"
run verify_off --mtp-verify 1 --mtp-route hand
run verify_on  --mtp-verify 1 --mtp-route hand --mtp-meter 1
grep -q '^MTPMETER role=verify on=1' "$OUT/verify_on.log" || { echo "ABORT: the metered arm printed no header line, so nothing was metered"; exit 1; }
compare verify "$OUT/verify_off.log" "$OUT/verify_on.log" "$OUT/verify_off" "$OUT/verify_on" || FAILED=1
adjudicate_rounds verify "$OUT/verify_on.log" "$(grep -c '^VERIFY2 ' "$OUT/verify_on.log")" || FAILED=1

echo "-- shape 2: the accept loop (--mtp-draft 1 --mtp-accept 1)"
run accept_off --mtp-draft 1 --mtp-accept 1 --mtp-route hand
run accept_on  --mtp-draft 1 --mtp-accept 1 --mtp-route hand --mtp-meter 1
grep -q '^MTPMETER role=accept on=1' "$OUT/accept_on.log" || { echo "ABORT: the metered arm printed no header line, so nothing was metered"; exit 1; }
# Rounds must be > 0, because comparing two logs in which no round ran is a vacuous pass — the same empty-list
# failure this tree records for every gate driver (`AGENTS.md`, the `gdn_collapse` row).
R=$(grep -m1 '^MTPSTAT ' "$OUT/accept_on.log" | tr ' ' '\n' | sed -n 's/^rounds=//p')
[ "${R:-0}" -gt 0 ] 2>/dev/null || { echo "ABORT: accept_on ran 0 rounds; the identity below would compare nothing"; exit 1; }
adjudicate_rounds accept "$OUT/accept_on.log" "$R" || FAILED=1
compare accept "$OUT/accept_off.log" "$OUT/accept_on.log" "$OUT/accept_off" "$OUT/accept_on" || FAILED=1

echo
[ "$FAILED" = 0 ] && echo "METER IDENTITY PASS — the instrument changed no decision in either shape" \
                  || { echo "METER IDENTITY FAIL — the instrument moved the machine; see the [DECISION] lines"; exit 1; }
