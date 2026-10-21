#!/usr/bin/env bash
# check_s1.sh — the S1 GATE ZERO: everything that must be true before anyone prices a millisecond.
#
# WHAT THIS IS. The S1 brief requires a gate that can tell "the engine pays for launches" from "the
# engine pays for syncs" from "the rig is not accounting for its own bytes", BEFORE any slope is
# quoted. So this script runs every correctness surface of the rig and refuses to proceed when any of
# them is red, and it prints NO millisecond figure anywhere: the timed pass is a separate, main-agent
# command (s1_sweep.sh --run). The one part that touches the device is a PASS/FAIL gate that finishes
# in seconds, as AGENTS.md allows for a worker.
#
# It asserts, in order:
#   A build                  the rig compiles at -O3 for gfx1151 (never -O0)
#   B census                 the engine arm's launch bill, segment bill and per-family bytes agree with
#                            engine/state.h and tools/step_traffic_model.py (census.py, host only)
#   C geometry selftest      host-only: every arm shape's index arithmetic, the derived must-catch /
#                            must-not-catch sets non-empty on both sides, the three structural claims,
#                            and the no-launch arm's bookkeeping
#   D refusals               every forbidden configuration aborts with a reason (a check that cannot
#                            fail is decoration; an admission list typed from memory is not an admission)
#   E device selftest        forked children: 7 positives CLEAN, 8 negatives CAUGHT against their
#                            derived sets, 1 blind case provably blind; parsed == |case list|
#   F no-timing discipline   the no-launch arm and a full un-timed engine row print NO-TIMING, and the
#                            whole device log contains no ms figure at all
#   G the fitter itself       fit_slope.py is fed fabricated stub rows and must (i) refuse them without
#                            --allow-stub, (ii) recover a known slope, (iii) refuse a partial ladder,
#                            (iv) refuse a bimodal ladder, (v) refuse an empty log
#
# Nothing here changes a gate threshold, an engine file, or the ledger.
set -uo pipefail
export LC_ALL=C
cd "$(dirname "$0")"
BIN=./s1_launch_price
fails=0
note() { printf '%s\n' "$*"; }
bad()  { printf '  >>> FAIL: %s\n' "$*"; fails=$((fails + 1)); }
step() { printf '\n== %s\n' "$*"; }

command -v timeout >/dev/null 2>&1 && TK="timeout 120" || TK=""

step "A build (-O3, gfx1151)"
if ./build.sh >/tmp/s1_build.log 2>&1; then
    note "  built ok"
else
    bad "build failed"; tail -20 /tmp/s1_build.log; echo "CHECK-S1: FAIL"; exit 1
fi

step "B census: engine arm against state.h + step_traffic_model.py (host only)"
if $TK python3 census.py >/tmp/s1_census.log 2>&1; then
    note "  $(tail -1 /tmp/s1_census.log)"
else
    bad "census.py red"; tail -25 /tmp/s1_census.log
fi

step "C geometry selftest (host only, no device)"
if $TK $BIN --selftest-geom >/tmp/s1_geom.log 2>&1; then
    note "  $(tail -1 /tmp/s1_geom.log)  ($(grep -c ' ok$' /tmp/s1_geom.log) checks ok)"
    grep -q 'SELFTEST-GEOM ok fails=0' /tmp/s1_geom.log || bad "geom tail is not fails=0"
else
    bad "--selftest-geom red"; tail -30 /tmp/s1_geom.log
fi

step "D refusals (each must abort, with a reason)"
# A refusal that silently exits 0 would let a meaningless configuration reach the ledger, so the exit
# code AND the reason text are both required. The population is enumerated here, not remembered.
refuse_case() { # name, expected-substring, args...
    local name=$1 want=$2; shift 2
    local out rc
    out=$($BIN "$@" 2>&1); rc=$?
    if [ "$rc" -eq 0 ]; then
        bad "refusal '$name' exited 0: it did not refuse"
    elif ! printf '%s' "$out" | grep -q "$want"; then
        bad "refusal '$name' aborted without its reason (wanted /$want/): ${out:0:120}"
    else
        note "  $name refused as designed (rc=$rc)"
    fi
}
refuse_case syncs-zero          'times submission alone'             --launches 48 --syncs 0
refuse_case syncs-gt-launches   'syncs > --launches is not a step'   --launches 4 --syncs 9
refuse_case engine-wrong-syncs  'takes --syncs 305'                  --pattern engine --body-mix engine --syncs 7
refuse_case engine-bad-count    'IS the shipped launch population'   --pattern engine --launches 400 --syncs 305 --body-mix engine
refuse_case engine-needs-mix    'pairs with --body-mix engine'       --pattern engine --syncs 305
refuse_case mix-needs-engine    'needs --pattern engine'             --body-mix engine
refuse_case pages-cold          'REFUSED --pages cold'               --pages cold
refuse_case empty-drain         'drains a step that has nothing'     --launches 0 --syncs 1
refuse_case threads-not-wave    'must be a multiple of 64'           --launches 8 --threads 100
refuse_case passes-zero         'passes must be >= 1'                --passes 0
refuse_case total-vs-engine     'pick one'                           --pattern engine --body-mix engine --syncs 305 --total-mib 64
refuse_case bad-flag            'unknown argument'                   --nope 1
refuse_case launches-negative   'must be >= 0'                       --launches -1

step "E+F device selftest and no-timing discipline (one lock hold, seconds)"
cd ../..
tools/gpu_lock.sh acquire "S1 gatezero check (worker, correctness only)"
note "  util at acquire: $(tools/gpu_lock.sh util)"
cd prototype/launch_price
{ $TK $BIN --selftest; echo "selftest_rc=$?"; } >/tmp/s1_dev.log 2>&1
{ $BIN --launches 0 --syncs 0; echo "zero_rc=$?"; } >>/tmp/s1_dev.log 2>&1
{ $BIN --pattern engine --body-mix engine; } >>/tmp/s1_dev.log 2>&1
{ $BIN --launches 340 --syncs 1 --body tiny; } >>/tmp/s1_dev.log 2>&1
cd ../..
note "  util at release: $(tools/gpu_lock.sh util)"
tools/gpu_lock.sh release "S1 gatezero check (worker, correctness only)"
cd prototype/launch_price

if grep -q '^SELFTEST ok' /tmp/s1_dev.log; then
    note "  $(grep '^SELFTEST ok' /tmp/s1_dev.log)"
else
    bad "device selftest red or incomplete"; grep -E '^(CASE|SELFTEST)' /tmp/s1_dev.log | tail -20
fi
# The case list the host advertised must equal the case list the children ran: a driver whose parser
# matched nothing prints PASS over an empty list (prototype/gdn_collapse/run.py).
want=$($BIN --print-cases | sed -n 's/^CASE-ROWS total=//p')
got=$(grep -c '^CASE ' /tmp/s1_dev.log)
if [ "${want:-x}" = "$got" ] && [ -n "$got" ]; then
    note "  case list agreement: advertised $want == ran $got"
else
    bad "case list disagreement: --print-cases says ${want:-none}, the selftest ran $got"
fi
for want in 'launches=0 syncs=0 NO-TIMING no_launch=1'; do
    grep -q "$want" /tmp/s1_dev.log && note "  no-launch arm: $want" || bad "missing: $want"
done
grep -q 'launches=340 syncs=305.*verify=ok' /tmp/s1_dev.log \
    && note "  engine row: 340 launches / 305 segments, verify=ok" \
    || bad "engine row did not report a clean 340/305 window"
if grep -qE 'ms_median|GB/s' /tmp/s1_dev.log; then
    bad "the correctness log contains a timing figure; it must not"
else
    note "  no ms figure anywhere in the device log (correctness only)"
fi

step "G the fitter itself, against fabricated rows (host only)"
# fit_slope.py decides the verdict, so it gets tested the way D2 §6 tested its parser: with numbers
# that are made up on purpose, so a parser that matches nothing cannot hide.
python3 - <<'PY' > /tmp/s1_stub.log 2>&1 || bad "stub generator failed"
# A stub row is the rig's own RESULT shape plus a trailing `stub=1`, and the arm name is the real arm
# key, so the fitter's grouping, ladder arithmetic, sync leg and merge reading are all exercised on
# the exact string format the timed run will use. The marker is what keeps them out of a verdict.
def row(arm, pattern, body, n, s, us, demand=5.4785, recs=None):
    return (f"RESULT arm={arm} pattern={pattern} body={body} launches={n} syncs={s} passes=8 "
            f"pages=warm threads=256 grid_max=8 recs={recs if recs is not None else n} "
            f"read_mib=5.3125 write_mib=0.1660 demand_mib={demand} seg_lps_min=1 seg_lps_max={s} "
            f"verify=ok launches_executed={n} syncs_executed={s} ms_min={us/1000:.6f} "
            f"ms_median={us/1000:.6f} ms_max={us/1000:.6f} reps=5 iters=5 stub=1")

for n in (48, 165, 305, 340, 680):
    print(row(f"uniform|tiny|p|{n}|1|8|warm", "uniform", "tiny", n, 1,
              5.0 * n + 100.0))                       # slope exactly 5.0 us/launch
for s in (340, 170, 43):
    # launch count pinned at 340, sync count swept: slope exactly 4.0 us per segment on top of the
    # same 340-launch base, so the S=1 point agrees with the ladder's 340|1 arm (a dedup check too)
    print(row(f"uniform|tiny|p|340|{s}|8|warm", "uniform", "tiny", 340, s, 5.0 * 340 + 100.0 + 4.0 * s))
print(row("engine|engine|p|340|305|8|warm", "engine", "engine", 340, 305, 2000.0,
          demand=1484.5963, recs=64695))
# the merge pair, all at the SAME demand: the fitter asserts equal footprints, so an unequal pair
# would be a FIT FAIL rather than a silently meaningless gain
for n, us in ((340, 1000.0), (8, 400.0), (1, 300.0)):
    print(row(f"uniform|tiny|t64|{n}|1|8|warm", "uniform", "tiny", n, 1, us, demand=64.0))
PY
if python3 fit_slope.py --in /tmp/s1_stub.log >/tmp/s1_fit_stub.txt 2>&1; then
    bad "fit_slope accepted stub rows without --allow-stub"
else
    note "  fit_slope refuses stub rows unless told otherwise"
fi
if python3 fit_slope.py --in /tmp/s1_stub.log --allow-stub \
        --sync-leg-syncs 340,170,43 \
        --engine-arm 'uniform|tiny|t64|340|1|8|warm' \
        --merge-arms 'uniform|tiny|t64|8|1|8|warm,uniform|tiny|t64|1|1|8|warm' \
        >/tmp/s1_fit_stub.txt 2>&1; then
    slopes=$(awk '/^  slope     = /{print $3}' /tmp/s1_fit_stub.txt)
    slope=$(printf '%s\n' "$slopes" | sed -n '1p')
    sslope=$(printf '%s\n' "$slopes" | sed -n '2p')
    if awk -v s="$slope" 'BEGIN{exit !(s>4.999 && s<5.001)}'; then
        note "  fit recovers the launch slope: $slope (expected 5.0000 us/launch)"
    else
        bad "fit recovered launch slope=[$slope], expected 5.0000 (slope lines: $(printf '%s' "$slopes" | tr '\n' ' '))"
    fi
    if awk -v s="$sslope" 'BEGIN{exit !(s>3.999 && s<4.001)}'; then
        note "  fit recovers the sync slope: $sslope (expected 4.0000 us/segment)"
    else
        bad "fit recovered sync slope=[$sslope], expected 4.0000"
    fi
    grep -q 'gain 600.00 us' /tmp/s1_fit_stub.txt \
        && note "  merge gain read correctly: 600.00 us for 332 launches removed" \
        || bad "the merge gain is not the expected 600.00 us"
    grep -q 'BANNER: --allow-stub' /tmp/s1_fit_stub.txt && note "  stub banner printed" \
        || bad "the stub banner is missing"
else
    bad "fit_slope red on a valid stub matrix"; tail -8 /tmp/s1_fit_stub.txt
fi
# The RESULT line's field ORDER is a contract between two files, and the fitter's regex is
# order-sensitive, so a field inserted on one side only breaks the timed pass — which is the main
# agent's run, not this gate. Compare the two spellings directly, in the source.
if python3 - >/tmp/s1_contract.txt 2>&1 <<'PY'
import importlib.util, pathlib, re, sys
here = pathlib.Path(".")
spec = importlib.util.spec_from_file_location("fit_slope", here / "fit_slope.py")
fit = importlib.util.module_from_spec(spec); spec.loader.exec_module(fit)
regex_fields = re.findall(r"([a-z_][a-z0-9_]*)=\(", fit.RESULT_RE.pattern)
src = (here / "s1_launch_price.hip.cpp").read_text()
i = src.index('std::printf("RESULT ')
j = src.index('\\n"', i)
fmt = "".join(re.findall(r'"((?:[^"\\]|\\.)*)"', src[i:j + 4]))
print_fields = re.findall(r"([a-z_][a-z0-9_]*)=", fmt)
print(f"regex field literals ({len(regex_fields)}): {' '.join(regex_fields)}")
print(f"printf fields ({len(print_fields)}): {' '.join(print_fields)}")
if print_fields != regex_fields:
    miss, extra = set(regex_fields) - set(print_fields), set(print_fields) - set(regex_fields)
    sys.exit(f"MISMATCH missing_in_printf={sorted(miss)} extra_in_printf={sorted(extra)} "
             f"(order matters even when the sets match)")
PY
then
    note "  RESULT-line contract: printf field order == the fitter's regex order"
else
    bad "RESULT-line contract broken between the rig and fit_slope.py"; cat /tmp/s1_contract.txt
fi

# A partial ladder must be refused rather than fitted over.
head -4 /tmp/s1_stub.log > /tmp/s1_stub_partial.log
if python3 fit_slope.py --in /tmp/s1_stub_partial.log --allow-stub \
        --slope-ladder 48,165,305,340,680 >/dev/null 2>&1; then
    bad "fit_slope fitted a partial ladder instead of refusing it"
else
    note "  fit_slope refuses a partial ladder"
fi
# A bimodal arm must be refused, not averaged: the same arm key twice, 139 % apart, which is the
# spread engine/performance-status.md §6 records for launch-bound rows on this box.
cp /tmp/s1_stub.log /tmp/s1_stub_bim.log
python3 - <<'PY' >> /tmp/s1_stub_bim.log 2>&1
lin = ("RESULT arm=uniform|tiny|p|680|1|8|warm pattern=uniform body=tiny launches=680 syncs=1 passes=8 "
       "pages=warm threads=256 grid_max=8 recs=680 read_mib=5.3125 write_mib=0.1660 demand_mib=5.4785 "
       "seg_lps_min=1 seg_lps_max=340 verify=ok launches_executed=680 syncs_executed=1 "
       "ms_min=%.6f ms_median=%.6f ms_max=%.6f reps=5 iters=5 stub=1")
for us in (3560.0, 1500.0):          # same arm, 139% apart: the bimodality this box is known for
    print(lin % (us / 1000, us / 1000, us / 1000))
PY
if python3 fit_slope.py --in /tmp/s1_stub_bim.log --allow-stub >/dev/null 2>&1; then
    bad "fit_slope averaged a bimodal arm instead of refusing it"
else
    note "  fit_slope refuses a bimodal arm"
fi
: > /tmp/s1_stub_empty.log
if python3 fit_slope.py --in /tmp/s1_stub_empty.log >/dev/null 2>&1; then
    bad "fit_slope succeeded on an empty log"
else
    note "  fit_slope refuses an empty log"
fi

step "verdict"
if [ "$fails" -eq 0 ]; then
    echo "CHECK-S1 PASS — gate zero is green; the rig may now be timed."
    cat <<'NEXT'

NEXT — the timed pass is the main agent's (AGENTS: only the main agent measures). Inside one
gpu_lock hold, blocking, zero subagents:

    prototype/launch_price/s1_sweep.sh --run

then fit it (host only; the merge comparison is asserted equal-footprint, so both arms of a pair share
a --total-mib):

    python3 prototype/launch_price/fit_slope.py --in /tmp/s1_sweep.log \
        --engine-arm 'uniform|tiny|t1461|340|1|8|warm' --merge-arms 'uniform|tiny|t1461|48|1|8|warm'
    python3 prototype/launch_price/fit_slope.py --in /tmp/s1_sweep.log \
        --engine-arm 'uniform|tiny|t64|340|1|8|warm' \
        --merge-arms 'uniform|tiny|t64|8|1|8|warm,uniform|tiny|t64|1|1|8|warm'

Read, in this order: the slope and its residual, then whether the intercept is negative, then the
merge gain. Bands it is judged against are printed by the fitter and each names its probe.
NEXT
else
    echo "CHECK-S1 FAIL — $fails failed check(s). No S1 number may be quoted until this is green."
fi
exit $((fails > 0))
