#!/usr/bin/env bash
# session_gates.sh — the entry point for the `--session --mtp-accept` gates. One command, one verdict.
#
# WHY THIS EXISTS. Each session driver below is a working tool that takes `tools/gpu_lock.sh` ITSELF, and
# until now each had to be invoked by hand, so "did you run the session gates?" had no answer that did not
# depend on somebody remembering the list. This file IS the list, and it does not hold the lock: a caller
# that holds it and then runs a self-locking driver deadlocks the box into a silent, CPU-less, GPU-less
# park (AGENTS.md, the 2026-10-12 re-take). Check with `grep -c gpu_lock <driver>`.
#
# WHAT EACH DRIVER OWNS:
#   session_lossless.sh   every DERIVED fixture plus the structural cases: the accept round must emit the
#                         plain stream. Exact token equality. Includes `cross-then-feed`, the case that
#                         crosses MAXSC while generating and then feeds.
#   session_longturn.sh   the seven-case localization A-G (SESSION-ACCEPT.md §8). Exact token equality.
#   session_trigger.sh    the geometry sweep with the DECISION MARGIN per position (SESSION-ACCEPT.md §9):
#                         every flip must sit under one bf16 ULP of its row's top logit, and the
#                         `no-rounds` control must be bit-exact. This is the driver that can say WHY the two
#                         exact-equality gates are red, and it is red only if a flip is too wide to be a tie.
#
# KNOWN STATE OF THIS FILE: the band is adopted (§9.7 taken as (c) on 2026-10-25), so the three drivers
# now own three different predicates and a green run means all three are satisfied:
#   session_lossless.sh  stays EXACT on every fixture. It holds: no derived-prompt case has ever needed a
#                        waiver, including `cross-then-feed`. If it ever goes red, that is a new fact and
#                        the answer is `session_rowdiff.py` on dumped rows, never a loosened predicate.
#   session_longturn.sh  exact, OR BAND: a differing case must show EVERY flip adjudicated from logits
#                        rows dumped on BOTH arms (runner-up in the other row, margin under one bf16 ULP).
#   session_trigger.sh   the margin sweep; red if any flip sits at an ordinary margin or the no-rounds
#                        control is not bit-exact.
# Correctness only: no millisecond figure is printed by anything here.
# SELF-LOCKING DRIVERS, DO NOT HOLD tools/gpu_lock.sh WHILE RUNNING THIS.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$root" || exit 1

drivers=(session_lossless.sh session_longturn.sh session_trigger.sh)
declare -a extra_args=("" "" "--at 535")   # the trigger sweep is the slow one; keep its default slice short
sel=${1:-all}
rc_all=0
for d in "${drivers[@]}"; do
  if [[ "$sel" != all && "$sel" != "$d" ]]; then continue; fi
  if ! grep -q gpu_lock "$here/$d"; then
    echo "ABORT $d: expected a self-locking driver, found no gpu_lock call - wiring it in here would run"
    echo "       an unlocked device process (or dead-lock if a caller holds the lock). Check the file."
    rc_all=1
    break
  fi
  idx=0
  for i in "${!drivers[@]}"; do [[ "${drivers[$i]}" == "$d" ]] && idx=$i; done
  echo "################ $d ${extra_args[$idx]:+(${extra_args[$idx]})}"
  # shellcheck disable=SC2086
  bash "$here/$d" ${extra_args[$idx]}
  rc=$?
  echo "################ $d exit=$rc"
  [[ $rc -ne 0 ]] && rc_all=1
done
echo "== session gates: $([ $rc_all = 0 ] && echo 'all green' || echo 'RED (see the driver above with a nonzero exit)')"
exit $rc_all
