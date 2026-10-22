#!/usr/bin/env bash
# gate_p1.sh — the cheap gate for the P1 prefetch package. A worker may run this (AGENTS: correctness-
# only device gates are allowed if they are under ~60 s wall, print no timing, and report what they ran).
#
# It runs, in order:
#   1. build (both binaries, -O3);
#   2. the CPU geometry selftest (no device);
#   3. the fitter's own selftest, which asserts that every refusal actually refuses — a fitter whose
#      parser matches nothing prints PASS over a matrix it never read (AGENTS: gdn_collapse/run.py);
#   4. the CPU arithmetic table, which must not REFUSE;
#   5. the rig's device witness matrix, one forked child per case, inside ONE gpu_lock hold, blocking.
#
# Nothing in this file prints an ms or GB/s figure. `--time` is never passed; the timed matrix is
# sweep.sh and belongs to the main agent.
set -u
export LC_ALL=C
HERE=$(cd -- "$(dirname -- "$0")" && pwd)
ROOT=$(cd -- "$HERE/../.." && pwd)
cd "$ROOT" || exit 2
FAIL=0
note() { printf '%s\n' "$*"; }
step() { printf '\n== %s\n' "$*"; }

step "1 build"
if "$HERE/build.sh" 2>&1 | tail -3; then note "build ok"; else note "BUILD FAILED"; FAIL=1; fi

step "2 rig --selftest-geom (host only)"
"$HERE/p1_rig" --selftest-geom | tee /tmp/p1_geom.txt
grep -q "^SELFTEST-GEOM ok=1" /tmp/p1_geom.txt || { note "GEOM GATE RED"; FAIL=1; }

step "3 fit_prefetch --selftest (refusals must refuse)"
.venv/bin/python "$HERE/fit_prefetch.py" --selftest | tee /tmp/p1_fit.txt
grep -q "^FIT-SELFTEST ok=1" /tmp/p1_fit.txt || { note "FITTER GATE RED"; FAIL=1; }

step "4 size_targets.py (CPU arithmetic must not REFUSE)"
.venv/bin/python "$HERE/size_targets.py" > /tmp/p1_size.txt 2>&1
rc=$?
tail -4 /tmp/p1_size.txt
[ $rc -eq 0 ] || { note "SIZE TARGETS REFUSED (rc=$rc)"; FAIL=1; }
grep -q "^REFUSE" /tmp/p1_size.txt && { note "SIZE TARGETS REFUSED"; FAIL=1; }

step "5 p1c_query (device query; sizes and booleans, no timing)"
"$HERE/run_locked.sh" "p1c query via gate_p1, correctness only, no timing" \
    "$HERE/p1c_query" > /tmp/p1_query.txt 2>&1
grep -E "VERDICT|co-residency|arm \(b\)" /tmp/p1_query.txt | head -8
grep -q "UTIL-AT-RELEASE" /tmp/p1_query.txt || { note "P1C DID NOT RUN"; FAIL=1; }

step "6 rig --selftest (device witnesses, forked per case, no timing)"
"$HERE/run_locked.sh" "p1_rig witness matrix via gate_p1, correctness only, no timing" \
    "$HERE/p1_rig" --selftest > /tmp/p1_st.txt 2>&1
grep -E "^SELFTEST|^NOT-CAUGHT|^CLEAN-CASE-RED|^CHILD-KILLED" /tmp/p1_st.txt
grep -q "^SELFTEST ok " /tmp/p1_st.txt || { note "RIG SELFTEST RED"; FAIL=1; }

printf '\n== gate_p1 %s\n' "$([ "$FAIL" -eq 0 ] && echo PASS || echo FAIL)"
printf 'no timing taken by this gate; util was recorded in /tmp/p1_query.txt and /tmp/p1_st.txt\n'
exit $FAIL
