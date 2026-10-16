#!/usr/bin/env bash
# CPU-only validation for prototype/gemm_algo_sweep. No case here may reach a HIP call: the mode
# early-returns and the argument/band validation all run before main() touches the device.
#
# That last claim is not trusted, it is asserted: every --time case must be refused by CALIBRATION-NEEDED
# or ARG_ERROR, checked from the first line of output rather than only from the exit code. A case that
# quietly fell through to the device would otherwise look green here (exit 0) while being an uncounted,
# un-locked GPU run - which is the mistake this script once made via a pipe.
B=./prototype/gemm_algo_sweep/gemm_algo_sweep
run() {  # run <expect_exit> <label> <args...>
  local want="$1" label="$2"; shift 2
  local out rc
  out="$("$B" "$@" 2>&1)"; rc=$?
  printf '%-46s exit=%s(want %s)\n' "$label" "$rc" "$want"
  [ "$rc" != "$want" ] && { printf '   MISMATCH: %s\n' "$out"; fail=1; }
  local first
  first="$(printf '%s\n' "$out" | head -1)"
  printf '   %s\n' "$(printf '%s\n' "$first" | cut -c1-150)"
  # No --time case may get as far as the device.
  case "$*" in
    *--time*) case "$first" in
                CALIBRATION-NEEDED*|ARG_ERROR*) ;;
                *) printf '   LEAKED-TO-DEVICE: a --time case was not refused before the device: %s\n' "$first"; fail=1 ;;
              esac ;;
  esac
}
fail=0
# "calibration absent" must be expressed as a path that cannot exist, not as the default path: the
# default is ./floor_calibration.tsv beside the binary, and once a real calibration is committed there
# the first two cases below would walk straight through to the device. This happened once, caught only by
# the LEAKED-TO-DEVICE check under run().
rm -f /tmp/cal_absent.tsv
echo "--- fixtures: remade so no case can be green against a stale format ---"
# Fixtures are remade every run. They were once left stale at the previous calibration format, and the
# two cases that were supposed to test "key missing" and "row unuseable" were instead both testing the
# version guard - green, and measuring nothing.
python3 prototype/gemm_algo_sweep/make_selftest_fixtures.py >/dev/null || { echo "FIXTURE GENERATION FAILED"; exit 1; }
echo "--- regression: existing host-only modes ---"
run 0 "--list"                                   --list
run 0 "--selftest-layout"                       --selftest-layout
run 0 "--selftest-sys fx_ok"                    --selftest-sys --algo-file /tmp/fx_ok.txt
run 1 "--selftest-sys fx_bad"                   --selftest-sys --algo-file /tmp/fx_bad.txt
run 3 "--selftest-sys no file"                  --selftest-sys
echo "--- band: every source must be measured, and named ---"
run 2 "time, no calibration file"               --time --check-only --rows 512 --check-rows 8 --calibration /tmp/cal_absent.tsv
run 2 "time, key missing from CAL"              --time --check-only --rows 512 --check-rows 16 --calibration /tmp/cal_partial.tsv
run 2 "time, cos floor unuseable (0.0)"         --time --check-only --rows 512 --check-rows 8 --calibration /tmp/cal_badrow.tsv
run 2 "time, cancel band unuseable (0.0)"       --time --check-only --rows 512 --check-rows 8 --calibration /tmp/cal_badband.tsv
run 2 "time, CAL is v1 (no band measured)"      --time --check-only --rows 512 --check-rows 8 --calibration /tmp/cal_v1.tsv
# The rule v2 added: --cos-floor overrides the cosine half only, so it cannot stand in for a calibration
# that also carries cancel_rel and cancel_tol. This case is the difference between "an operator may A/B a
# floor" and "an operator may skip measuring the band".
run 2 "--cos-floor does not excuse calibration" --time --check-only --rows 512 --check-rows 8 --cos-floor 0.99 --calibration /tmp/cal_absent.tsv
echo "--- band: argument validation ---"
run 2 "--cos-floor out of range"                --time --check-only --cos-floor 1.5
run 2 "--cos-floor 0"                           --time --check-only --cos-floor 0
run 2 "--max-ulp 0"                             --time --check-only --max-ulp 0
run 2 "--floor-margin 0"                        --time --check-only --floor-margin 0
run 2 "--cancel-margin 50 (band=measurement)"   --time --check-only --cancel-margin 50
run 2 "--cancel-margin 0.5 (tighter than measured)" --time --check-only --cancel-margin 0.5
run 2 "--cancel-rel 0"                          --time --check-only --cancel-rel 0
run 2 "--cancel-tol -1"                         --time --check-only --cancel-tol -1
run 2 "--fullmag-rel 0"                         --time --check-only --fullmag-rel 0
run 2 "--calibration read only by --time"       --verify --calibration /tmp/cal_full.tsv
run 2 "--calibration-out on --time"             --time --check-only --calibration-out /tmp/x.tsv
echo "--- probe validation ---"
run 2 "floor-probe needs --calibration-out"     --floor-probe
run 2 "floor-probe --seeds 0"                   --floor-probe --calibration-out /tmp/c.tsv --seeds 0
run 2 "floor-probe --check-rows 1"              --floor-probe --calibration-out /tmp/c.tsv --check-rows 1
run 2 "floor-probe + --calibration (read)"      --floor-probe --calibration-out /tmp/c.tsv --calibration /tmp/cal_full.tsv
echo
[ "$fail" = 0 ] && echo "CPU VALIDATION PASS" || echo "CPU VALIDATION FAIL"
exit $fail
