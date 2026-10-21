#!/usr/bin/env bash
# burst_m16.sh — the M = 16 geometry tax, timed. MAIN AGENT / USER ONLY: this takes milliseconds, and a
# worker may not (AGENTS.md, "GPU execution is serialized"). It does NOT acquire tools/gpu_lock.sh: three
# gate drivers in this repo self-lock and a caller that holds the lock and runs one of them deadlocks it,
# so the CALLER owns the lock and this script only prints what the box is doing.
#
# WHAT IS BEING TIMED, AND AGAINST WHAT.
#   new   = prototype/attn_m2/attn_m16  --burst N   (form `m16`      = attn_m16_wave, the 16-row re-tile)
#   base  = the same binary's `shipslim` column    (form `shipslim`  = attn_slim1_wave, the kernel the
#                                                  engine ships today), timed INSIDE the same rep
#   echo  = prototype/attn_m2/attn_m2 --slim 1     (the §13 artifact; `AL ALSO_M2=1 to print it)
# `shipslim` and `m16` are launched alternately inside one rep, so a mid-window disturbance hits both and
# the ratio is one ruler rather than two windows — the paired-A/B rule this repo learned in D30 1.7-prime.
# The base invocation this is compared with in `prototype/attn_m2`'s own ledger is
#     prototype/attn_m2/attn_m2 --plan <plan> --split <split> --slim 1 --burst 40 --reps 5
# at the widths below; `attn_slim1_wave` is the same kernel in both binaries (same header, same TU rules),
# and the gate prints `ndiff_aout_shipslim=0` every run to show the copy in this TU agrees.
#
# BEFORE RUNNING IT
#   1. zero subagents, box quiet, run BLOCKING (never backgrounded: flash_serve busy-polls the same iGPU).
#   2. tools/gpu_lock.sh acquire <TAG>   (this script will not do it for you)
#   3. admit first:  .venv/bin/python prototype/attn_m2/gate_m16.py device   # 2 s, no timing
#      The harness itself also refuses to time a red admission ("NO-TIMING: admission is red"), but it
#      cannot check what it was never asked to check, so run the gate anyway.
#   4. a short-prompt row is dispatch-bimodal on this box (engine/performance-status.md §6): quote a PAIR,
#      state the spread, and let the long-context row carry the headline.
#
# USAGE
#   bash prototype/attn_m2/burst_m16.sh                    # 40 launches x 5 reps, all widths
#   BURST=80 REPS=7 bash prototype/attn_m2/burst_m16.sh
#   ALSO_M2=1 bash prototype/attn_m2/burst_m16.sh          # echo the attn_m2 artifact's own numbers
set -euo pipefail
cd "$(dirname "$0")/../.."
BURST=${BURST:-40}
REPS=${REPS:-5}
ALSO_M2=${ALSO_M2:-0}
BIN=${BIN:-prototype/attn_m2/attn_m16}   # override to point at a differently-built GRP, e.g.
#   BIN=prototype/attn_m2/attn_m16_grp2 bash prototype/attn_m2/burst_m16.sh
# It was hardwired before §28, which is how the shipped-parity-occupancy shape (GRP=2) sat built-and-untimed
# while a tax was attributed to the occupancy of the shape that WAS timed.
[ -x "$BIN" ] || bash prototype/attn_m2/build_m16.sh

echo "# gpu_lock status: $(tools/gpu_lock.sh status 2>&1 || true)  util: $(tools/gpu_lock.sh util 2>&1 || true)"
echo "# burst=${BURST} reps=${REPS} new=${BIN} (form m16) base=shipslim in the same rep"
echo "# plans (re)written host-only, no device touched by this step:"
PLANLINES=$(.venv/bin/python prototype/attn_m2/gate_m16.py plans | awk '$1=="PLAN" && $6=="burst"')
echo "$PLANLINES" | sed 's/^/#   /'

# The `full=0` rows below are the STREAMING route at a short width, forced: under `full = 1` a row at or
# below MAXSC is one-pass and this form refuses it (the score staging for 16 live pairs would need the
# whole LDS). Read those two lines as "16-row tile, streaming", NOT as the shipping short-row route.
while read -r _ name plan split full kind extra; do
    [ "${name:-}" = "" ] && continue
    tag="full=1 shipping one-pass rule"
    [ "$full" = "0" ] && tag="full=0 STREAMING FORCED — not the shipping short-row route"
    echo
    echo "### $name  split=$split  $tag"
    "$BIN" --plan "$plan" --split "$split" --full "$full" --burst "$BURST" --reps "$REPS" \
        | grep -E '^(RESULT|SPEEDUP|NO-TIMING|REFUSED)'
    if [ "$ALSO_M2" = "1" ] && [ -x prototype/attn_m2/attn_m2 ]; then
        echo "--- echo only, different process and window: no ratio is computed from this line"
        prototype/attn_m2/attn_m2 --plan "$plan" --split "$split" --slim 1 --burst "$BURST" \
            --reps "$REPS" 2>/dev/null | grep -E '^(RESULT|NO-TIMING|REFUSED)' || true
    fi
done <<< "$PLANLINES"

echo
echo "# ratios to read: SPEEDUP a=m16 b=shipslim on the SAME case, quoted as a pair (best-of-${REPS} of"
echo "#   burst-${BURST}), never as a single window. The occupancy context the number must be read in is"
echo "#   in prototype/attn_m2/results.md §23: the shipped form runs 5 blocks / 20 wavefronts per CU and"
echo "#   this re-tile, at 134 VGPR per lane, is register-bound at about 3 blocks / 12 wavefronts."
