#!/usr/bin/env bash
# Compare two directories of gate logs and answer one question: did any DECISION move?
#
#     tools/gate_log_diff.sh <dir-before> <dir-after> [fixture ...]
#     tools/gate_log_diff.sh <dir-before> <dir-after> --diff       # also print the residual text diff
#
# Default fixture list is the eight the gate ships. Per fixture it prints the `[PASS]` / `[FAIL]` counts
# and `[SLACK]`-family line counts on both sides, then whether the md5 of the DECISION lines is identical:
# every `step j: metrics|cos|meanrel|argmax` and `generation step ...` line, i.e. the whole set of lines by
# which a step is judged. That is the check that a reporting-only change reported only. If a band, a rule or
# a margin moved, this md5 moves, and the honest reading of a moved md5 is "look at the diff", never
# "normalize it away".
#
# `--diff` prints what still differs after two normalisations, both of them facts about the machine rather
# than about the code: the engine's own ms echo (`ms_total=`, `ms_per_step=`, `resident in ... ms`) and the
# `decode_run` command lines, which carry a per-run dump directory. Provenance headers written by
# `tools/gate_locked_run.sh` are dropped as well. Anything left is a real text difference and stays visible
# here on purpose: this tool has no ignore list, because an ignore list is how a changed verdict gets
# explained away.
set -uo pipefail

if [ "$#" -lt 2 ]; then
    echo "usage: gate_log_diff.sh <dir-before> <dir-after> [--diff] [fixture ...]" >&2
    exit 2
fi
before="$1"; after="$2"; shift 2

show_diff=0
fixtures=""
for arg in "$@"; do
    if [ "$arg" = "--diff" ]; then
        show_diff=1
    else
        fixtures="$fixtures $arg"
    fi
done
if [ -z "$fixtures" ]; then
    fixtures="dec8 dec64f dec64t dec300f mt32f p128 wide520f chunk64b"
fi

pass_count() { grep -c '\[PASS\]' "$1"; }
fail_count() { grep -c '\[FAIL\]' "$1"; }
slack_count() { grep -c 'SLACK' "$1"; }
decision_md5() {
    grep -E '\[PASS\] (step [0-9]+: (metrics|cos|meanrel|argmax)|generation step)' "$1" \
        | md5sum | cut -d' ' -f1
}
norm() {
    grep -v '^#' "$1" \
        | sed -E 's/(ms_total|ms_per_step)=[0-9.]+/\1=X/g' \
        | sed -E 's/resident in [0-9.]+ ms/resident in X ms/' \
        | grep -v '^  \$ '
}

rc=0
for fx in $fixtures; do
    a="$before/$fx.log"
    b="$after/$fx.log"
    if [ ! -f "$a" ] || [ ! -f "$b" ]; then
        echo "$fx: MISSING log ($a or $b)"
        rc=1
        continue
    fi
    ma=$(decision_md5 "$a")
    mb=$(decision_md5 "$b")
    same="DIFFER -> inspect with --diff"
    if [ "$ma" = "$mb" ]; then
        same="IDENTICAL"
    fi
    echo "$fx: pass $(pass_count "$a") -> $(pass_count "$b")" \
         "| fail $(fail_count "$a") -> $(fail_count "$b")" \
         "| slack-lines $(slack_count "$a") -> $(slack_count "$b")" \
         "| decision-md5 $same"
    if [ "$show_diff" = "1" ] && [ "$ma" != "$mb" ]; then
        diff <(norm "$a") <(norm "$b") | grep -E '^[<>]' | sed 's/^/    /' | cut -c1-200
    fi
done
exit "$rc"
