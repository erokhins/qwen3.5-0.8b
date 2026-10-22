#!/usr/bin/env bash
# P7 gate zero — the device pass, handed over whole.  **The author of this file ran none of it.**
#
# Every number in prototype/mtp_gatezero/MTP-GATEZERO.md is quoted or CPU-measured; the items below are the
# device reads that turn the priced routes into funded ones. They are timings, so they belong to the main
# agent (AGENTS.md: GPU execution is serialized and belongs to the main agent; a worker may run only
# short correctness-only gates, and there is nothing on the device to check for correctness here yet).
#
#   bash prototype/mtp_gatezero/DEVICE-HANDOVER.sh --run D1        # run item D1 only
#   bash prototype/mtp_gatezero/DEVICE-HANDOVER.sh --list          # what each item needs
#
# Protocol, per AGENTS.md:
#   * blocking, never backgrounded, zero subagents active;
#   * one lock hold for the whole window so paired items share it (D1's two bursts MUST be in one hold);
#   * `lm_head_gemv/run.py` does NOT self-lock (`grep -c gpu_lock prototype/lm_head_gemv/run.py` = 0), so
#     THIS script holds the lock around it. Do not add a self-locking driver (`prototype/glue/run.py`,
#     `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`) inside a hold: that deadlocks.
#   * best-of-N / burst timing per `engine/performance-status.md` §6, `steady_clock` burst for anything
#     under ~1 ms, and `hipEventElapsedTime` reads exactly 0 below ~1 ms on this part.
#   * report the command, wall time, `tools/gpu_lock.sh util` at the time, and which window each number
#     came from. A number from a different window is not an A/B (the `prefill_sweep` lesson).
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
PY=.venv/bin/python
DOC=prototype/mtp_gatezero/MTP-GATEZERO.md

list() {
  cat <<'EOF'
D1  second consecutive tied-table pass == the first one?
    Why: c_full is one quoted lm_head pass (2.1685 ms) on top of the step. Nothing in a decode step can
         leave 485 MiB resident (fast tier ~32 MiB), so it should be a cold read again; confirm, do not assume.
    Priced: 2.1685 ms. Margin: +-5 % moves c_full 0.218 -> 0.239 and speedup 1.416x -> 1.392x (not decisive).
    Runnable today: YES, with the existing rig. See run_D1.

D2  the draft block at engine shape: 39.0107 MiB over 15 tensors + ~18 launches.
    Runnable today: NO - there is no MTP kernel, and the block's shapes (shared [1024,2048] fc, a second
    o_proj-shaped GEMV, rope at positions 0/1) are not what the existing rigs take.
    Interim: price from the ledger's achieved GEMV efficiencies (mlp 70 %, proj 75 %) = 0.22-0.25 ms.
    Re-check when the kernel exists; if the block lands at 2x the priced value (0.55 ms) the FULL head is
    still 1.39x, so do not fund a block-cost micro-optimisation before the accept rate is on engine bits.

D3  delta: what verifying TWO rows per step costs over ONE at 4,032 lanes.
    Runnable today: NO - `decode_run` has no switch that submits a 2-row body over a populated KV.
    This is the funded work's FIRST measurement, not a gate-zero one.
    Bound already computed: delta_max = 0.379 of a step for the FULL head (`route_price.py`). Above that
    the route is under the 0.30 ms bar and P7 dies on the verify side instead of the draft side.

D4  a on the ENGINE's bits: engine drafts, engine verifies, real prose.
    Runnable today: NO - needs the draft path.
    Protocol when it exists: the 49 lines of speccheck/fixtures/corpus.txt, >= 24 generated tokens each,
    stop on id 248046 in the GENERATION phase only (never while feeding a prompt), and state the index
    alignment any comparison uses (the free-run/`feed`-`gen` trap). Carry `draft_survival.py`'s position
    join, not a re-derived accept loop. Carried value 0.7329; kill line a_needed = 0.257 (batched verify).
    Also record accept by depth (measured RISING: 0.588 at positions 0-7 -> 0.808 at 32-63), because a
    k>1 draft would be priced off that profile, not off the single number.

D5  device top-k over the previous logits row (993 KB, on the host today at 0.7163 ms quoted host_tail).
    Runnable today: NO - not built; shares its mechanism with queue P2, so price it there once, not twice.
    Kills the restricted route alone if it costs more than ~0.3 ms/step; the FULL head stands regardless.
EOF
}

run_D1() {
  local tag=p7-gatezero-d1
  echo "== D1: two back-to-back tied-table bursts, one lock hold, best-of-N =="
  tools/gpu_lock.sh acquire "$tag" || { echo "lock busy"; return 1; }
  trap "tools/gpu_lock.sh release '$tag'" EXIT
  echo "util at acquire: $(tools/gpu_lock.sh status 2>&1 | tr -d '\n')"
  local i
  for i in 1 2; do
    echo "-- burst $i (same window, same hold) --"
    # RESULT line first, CPU baseline last, so grep RESULT and not ms (AGENTS trap).
    "$PY" prototype/lm_head_gemv/run.py --iters 20 --rows 8 2>&1 | grep -E "RESULT|MiB"
  done
  cat <<EOF
READ THIS AS: the two bursts are the first and second head pass of one speculative step. If they agree,
c_full = (2.1685 + 0.2217 + 0.0558) / 10.9357 = 0.2237 and the FULL-head speedup stands at 1.416x
at delta = 0 (see MTP-GATEZERO.md S4). If the second is materially slower, re-run route_price.py with the
measured second-pass ms before quoting any headline.
NO TIMING FROM THIS SCRIPT MAY ENTER engine/performance-status.md unless the main agent took it.
EOF
}

case "${1:---list}" in
  --list) list ;;
  --run) case "${2:-}" in D1) run_D1 ;; *) echo "only D1 is runnable today; see --list"; list ;; esac ;;
  *) echo "usage: $0 [--list | --run D1]"; exit 2 ;;
esac
