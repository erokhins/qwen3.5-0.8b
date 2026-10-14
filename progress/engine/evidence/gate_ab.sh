#!/usr/bin/env bash
# One `engine/gate.py` fixture run with the GPU lock held inside the same call, a NAMED binary, and
# the provenance header `tools/gate_locked_run.sh` writes — plus a log name that suits
# `tools/margin_delta.py` (`<side>_<fixture>.log`).
#
#     progress/engine/evidence/gate_ab.sh <side> <fixture> [binary]        # binary defaults to engine/build/decode_run
#
# `<side>` is the A/B label the route check is run under (hand|library), it becomes the log prefix, so
# `tmp/margin/hand_dec300f.log` and `progress/engine/evidence/margin/library_dec300f.log` are directly comparable with
# `tools/margin_delta.py tmp/margin/hand tmp/margin/library dec300f`.
#
# **`<side>` is the ONLY route control, and it reaches `gate.py`, not the binary.** The first version of
# this script put `--gemm <side>` in a wrapper binary, which did nothing: `gate.py` appends its own
# `--gemm <route>` AFTER the wrapper's flag and `decode_run`'s parser is last-wins, so every run of
# `gate_ab.sh library <fx> progress/engine/evidence/decode_run_library.sh` silently ran the HAND route (the logs proved it:
# a `library_dec300f.log` whose NUMERICS line said `gemm=hand`). Passing a wrapper still works — the
# gate's flag is the one that lands, and it is the one the gate judges under — but a wrapper is no
# longer how the route is chosen. Every verdict line below therefore prints its NUMERICS line: after
# that incident the header is the only thing in the log that proves which route ran.
#
# Same honesty rules as `tools/gate_locked_run.sh`: this is a correctness gate, the wall printed is
# this script's own span, and nothing here belongs in `bench/performance-status.md`.
set -uo pipefail
root=$(cd "$(dirname "$0")/.." && pwd)
cd "$root" || exit 1
side="${1:?usage: gate_ab.sh <side> <fixture> [binary]}"
fx="${2:?usage: gate_ab.sh <side> <fixture> [binary]}"
binary="${3:-engine/build/decode_run}"
case "$side" in hand|library) ;; *) echo "side must be hand|library, got '$side'"; exit 2 ;; esac
out="tmp/margin"
mkdir -p "$out"
tag="gate-${side}-$fx"
tools/gpu_lock.sh acquire "$tag" || exit 1
trap "tools/gpu_lock.sh release '$tag'" EXIT
u0=$(tools/gpu_lock.sh util 2>/dev/null | tr -s ' ')
t0=$(date +%s.%N)
.venv/bin/python engine/gate.py --fixture "$fx" --binary "$binary" --gemm "$side" > "$out/${side}_$fx.log" 2>&1
rc=$?
u1=$(tools/gpu_lock.sh util 2>/dev/null | tr -s ' ')
t1=$(date +%s.%N)
wall=$(awk -v a="$t0" -v b="$t1" 'BEGIN{printf "%.2f", b-a}')
{ echo "# cmd: .venv/bin/python engine/gate.py --fixture $fx --binary $binary --gemm $side"
  echo "# side=$side gpu_lock util at acquire:$u0 at release:$u1 | wall=${wall}s (by-product, not a measurement; no timing taken)"
  cat "$out/${side}_$fx.log"; } > "$out/${side}_$fx.log.tmp" && mv "$out/${side}_$fx.log.tmp" "$out/${side}_$fx.log"
echo "${side}/$fx rc=$rc util_acquire=$u0 util_release=$u1 wall=${wall}s log=$out/${side}_$fx.log"
echo "PASS lines: $(grep -c '\[PASS\]' "$out/${side}_$fx.log")  FAIL lines: $(grep -c '\[FAIL\]' "$out/${side}_$fx.log")"
grep -E '^      (NUMERICS|GEMM route)|^  \[(TIE|TIE-NOTE|SLACK|SLACK-NOTE|expected-red|NEW-FAILURE|FAIL)\]|^ENGINE GATE' "$out/${side}_${fx}.log"
exit "$rc"
