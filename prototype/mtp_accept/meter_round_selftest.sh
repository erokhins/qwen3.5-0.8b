#!/usr/bin/env bash
# meter_round_selftest.sh — CPU-only. Feeds `meter_table.awk` synthetic `--mtp-meter` logs and asserts that
# every refusal fires on the log it is meant to fire on, and stays quiet on a well-formed one.
#
# WHY THIS EXISTS. `meter_table.awk` decides whether a phase attribution gets published. A refusal that has
# never been seen to fire is indistinguishable from a refusal that cannot fire, and this repo has already been
# bitten three times by a check whose input list could be empty (the `gdn_collapse` driver that parsed zero
# CASE lines and printed PASS; the coverage gate that judged one row out of 1024; the counter census that
# summed nothing and reported 0 %). So each case below is a log built to be WRONG in exactly one way, and the
# assertion is the refusal's identity, not merely that something failed — a case that fails for the wrong
# reason would let the real defect through.
#
# No device, no engine, no timing: this reads text files. Run it any time `meter_table.awk` changes.
#
# Usage: prototype/mtp_accept/meter_round_selftest.sh
set -u
export LC_ALL=C
cd "$(dirname "$0")/../.." || exit 1
HERE=$(cd "$(dirname "$0")" && pwd)
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT

# A well-formed accept-role run: measured phases 3.75 ms/round, derived overhead 0.25, wall 4.00, coverage
# 0.9375 — inside the band, and sum + overhead == wall exactly. It carries two `status=subphase` rows
# (draft_vocab_d2h + draft_argmax_host, 0.1500 + 0.1000 of `draft`'s own 0.2500): a sub-phase is a breakdown
# of milliseconds its parent already contributed, so it is out of the sum for a different reason than
# `derived` is, and `subphase_charged_into_sum` below is the case that shows the exclusion is checked.
good_accept() { cat <<'LOG'
MTPACCEPT on=1 rows=2 checkpoint_bytes=20201472 status_bytes=16 draft_sched=loop per_step_dumps=0
MTPMETER role=accept on=1 idle_sync_us=24.00 sync_added_phases_inflate=1 plan=prototype/mtp_accept/METER-PLAN.md
ACCEPT round=1 anchor=8 fed=100 cand=200 cand_pos=9 accept=1 verifier=100 bonus=200 emitted=2 pos=8->10 restored_bytes=0
MTPMETER role=accept phase=body ms_total=8.0000 rounds=4 ms_per_round=2.0000 sync_added=1 runs=4 ms_per_run=2.0000 note=submit_wave_body
MTPMETER role=accept phase=mid_checkpoint status=not_metered note=2_extra_submissions_per_GDN_layer
MTPMETER role=accept phase=head ms_total=3.0000 rounds=4 ms_per_round=0.7500 sync_added=1 runs=4 ms_per_run=0.7500 note=one_pass_over_the_tied_table
MTPMETER role=accept phase=verdict_kernel ms_total=1.0000 rounds=4 ms_per_round=0.2500 sync_added=1 runs=4 ms_per_run=0.2500 note=on_device
MTPMETER role=accept phase=verdict_readback ms_total=1.0000 rounds=4 ms_per_round=0.2500 sync_added=0 runs=4 ms_per_run=0.2500 note=four_ints
MTPMETER role=accept phase=accept_path ms_total=0.5000 rounds=4 ms_per_round=0.1250 sync_added=0 runs=2 ms_per_run=0.2500 note=ran_on_an_accept_only
MTPMETER role=accept phase=rollback ms_total=0.5000 rounds=4 ms_per_round=0.1250 sync_added=1 runs=2 ms_per_run=0.2500 note=ran_on_a_reject_only
MTPMETER role=accept phase=lane_bookkeeping ms_total=0.0000 rounds=4 ms_per_round=0.0000 sync_added=0 runs=4 ms_per_run=0.0000 note=moves_0_bytes
MTPMETER role=accept phase=draft ms_total=1.0000 rounds=4 ms_per_round=0.2500 sync_added=1 runs=4 ms_per_run=0.2500 note=next_round_proposal
MTPMETER role=accept phase=draft_vocab_d2h ms_total=0.6000 rounds=4 ms_per_round=0.1500 sync_added=0 runs=4 ms_per_run=0.1500 status=subphase inside=draft note=the_vocabulary_download
MTPMETER role=accept phase=draft_argmax_host ms_total=0.4000 rounds=4 ms_per_round=0.1000 sync_added=0 runs=4 ms_per_run=0.1000 status=subphase inside=draft note=the_host_argmax
MTPMETER role=accept phase=report_prints ms_total=0.0000 rounds=4 ms_per_round=0.0000 sync_added=0 runs=4 ms_per_run=0.0000 note=the_run_prints
MTPMETER role=accept phase=overhead ms_total=1.0000 rounds=4 ms_per_round=0.2500 sync_added=0 runs=4 ms_per_run=0.2500 status=derived note=derived_not_timed
MTPMETER role=accept phase=summary ms_total=15.0000 rounds=4 ms_per_round=3.7500 sync_added=0 runs=4 ms_per_run=3.7500 status=derived wall_ms_per_round=4.0000 coverage=0.9375 note=the_meter_verdict
MTPMETER role=accept phase=instrument windows=4 rounds=4 plain_steps=1 syncs_added=12 idle_sync_us=24.00 sync_errors=0 note=windows_are_the_wall
MTPSTAT route=hand gemm=hand gdn_scan=seq rounds=4 accepted=2 rejected=2 emitted=6 accept_rate=0.5000 mean_tokens_per_round=1.5000 body_rows=8 captured_bytes=0 restored_bytes=40402944 status_bytes=64 split_subs=0
DECWALL mode=run prefill_ms=1.000 decode_ms=16.000 bodies=5 rounds=4 plain=1 accepted=2 rejected=2 suppressed=0 emitted=6 stop=0 ended=length ms_per_token=2.667 tok_per_s=375.0 census=ok
RESULT mode=run steps=13 ms_total=17.000 ms_per_step=1.308 last_sampled=999 prefix_len=14 stopped=0 meter=ok rounds=4 sum_ms_per_round=3.7500 wall_ms_per_round=4.0000 coverage=0.9375
LOG
}

run_case() { # $1 name $2 want(NORMAL|NO-TABLE) $3 want-fragment $4 role $5 generator
    local name="$1" want="$2" frag="$3" role="$4" gen="$5" log out rc
    log="$TMP/$name.log"
    : > "$log"
    [ "$gen" = "empty" ] || good_accept > "$log"
    case "$name" in
        # The one-field mutations, each built by rewriting exactly one line of the good log so every OTHER
        # field still agrees — otherwise a case fails for an unrelated reason and proves nothing.
        bad_coverage)   good_accept | sed 's/coverage=0.9375/coverage=0.5000/g' > "$log" ;;
        bad_identity)   good_accept | sed 's/^MTPMETER role=accept phase=overhead ms_total=1.0000 rounds=4 ms_per_round=0.2500/MTPMETER role=accept phase=overhead ms_total=2.0000 rounds=4 ms_per_round=0.5000/' > "$log" ;;
        bad_sum)        good_accept | sed '/^RESULT mode=/ s/sum_ms_per_round=3.7500/sum_ms_per_round=4.2500/' > "$log" ;;
        # The SAME two breakdown rows, with one field changed: `status=subphase` dropped, so the awk reads
        # them as ordinary measured phases. This is the must-catch for the exclusion — if the exclusion were
        # broken (or the engine charged a sub-phase into its parent's total as well), this is what the table
        # would look like, and the identity check must refuse it rather than publish a 4.15 ms round.
        subphase_charged_into_sum) good_accept | sed 's/^MTPMETER role=accept phase=draft_vocab_d2h \(.*\) status=subphase inside=draft/MTPMETER role=accept phase=draft_vocab_d2h \1/' > "$log" ;;
        bad_meter)      good_accept | sed 's/meter=ok/meter=out_of_band/' > "$log" ;;
        no_coverage_field) good_accept | sed 's/ coverage=0.9375//' > "$log" ;;
        wrong_role)     good_accept | sed 's/^MTPMETER role=accept/MTPMETER role=verify/' > "$log" ;;
    esac
    out=$(awk -v role="$role" -v w=4032 -v logname="$log" -f "$HERE/meter_table.awk" "$log" 2>&1)
    rc=$?
    if [ "$want" = NORMAL ]; then
        [ "$rc" = 0 ] || { echo "FAIL $name: expected the table, awk exited $rc"; printf '%s\n' "$out" | sed 's/^/    /'; FAILED=1; return; }
        printf '%s\n' "$out" | grep -qF "$frag" || { echo "FAIL $name: table printed without '$frag'"; printf '%s\n' "$out" | sed 's/^/    /'; FAILED=1; return; }
    else
        [ "$rc" != 0 ] || { echo "FAIL $name: expected a refusal and got a table"; printf '%s\n' "$out" | sed 's/^/    /'; FAILED=1; return; }
        printf '%s\n' "$out" | grep -qF "$frag" || { echo "FAIL $name: refused for the wrong reason, wanted '$frag'"; printf '%s\n' "$out" | sed 's/^/    /'; FAILED=1; return; }
    fi
    printf '%s\n' "$out" | head -1 | sed 's/^/    /'
    echo "ok   $name"
}

FAILED=0
echo "== meter_table.awk selftest — synthetic logs, no device"

# 1. The positive control. Without this every refusal below could be passing because the program cannot print
#    anything at all. The rendered table is printed here so a reader sees what a good one looks like.
echo "   -- the well-formed accept table --"
good_accept > "$TMP/good_accept.log"
awk -v role=accept -v w=4032 -v logname="$TMP/good_accept.log" -f "$HERE/meter_table.awk" "$TMP/good_accept.log" \
    || { echo "FAIL: the good log did not render"; FAILED=1; }
grep -q "mid_checkpoint status=not_metered" "$TMP/good_accept.log" || { echo "FAIL: the good log lost its not_metered row"; FAILED=1; }
echo "ok   good_accept (and its mid_checkpoint row carries no millisecond figure)"

# 3. Each refusal, on a log wrong in exactly one way.
run_case bad_coverage       NO-TABLE "outside 0.8-1.2"           accept good
run_case bad_identity       NO-TABLE "double-charging a phase"   accept good
run_case bad_sum            NO-TABLE "charged and not printed"   accept good
# The positive half of the sub-phase exclusion: the good log above carries two `status=subphase` rows
# totalling 0.2500 ms/round INSIDE `draft`, and the table still balances at wall 4.000. If the awk summed
# them the identity check below would fire, so `subphase_charged_into_sum` is the same log with that status
# field removed and is refused for exactly that reason. Pair, not solo claim.
run_case subphase_rows_excluded NORMAL   "identity ok"            accept good
run_case subphase_charged_into_sum NO-TABLE "double-charging a phase" accept good
run_case bad_meter          NO-TABLE "meter=out_of_band"         accept good
run_case no_coverage_field  NO-TABLE "no coverage= field"        accept good
run_case wrong_role         NO-TABLE "printed no phase lines"    accept good

# 4. An empty log: a harness pointed at a run that died before printing anything must not get a table.
: > "$TMP/empty.log"
if awk -v role=accept -v w=4032 -v logname="$TMP/empty.log" -f "$HERE/meter_table.awk" "$TMP/empty.log" > "$TMP/empty.out" 2>&1; then
    echo "FAIL empty: an empty log produced a table"; FAILED=1
else
    grep -q "printed no phase lines" "$TMP/empty.out" || { echo "FAIL empty: refused for the wrong reason"; cat "$TMP/empty.out"; FAILED=1; }
    echo "ok   empty"
fi

echo
[ "$FAILED" = 0 ] && echo "SELFTEST PASS — every refusal fires on its own defect and the well-formed log renders" \
                  || { echo "SELFTEST FAIL"; exit 1; }
