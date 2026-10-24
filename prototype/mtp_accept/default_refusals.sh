#!/usr/bin/env bash
# The accept loop's refusal battery, AS THE DEFAULT FLIP LEFT IT — the record `prototype/mtp_accept/
# K2-RESULT.md` §7 called "every combination exits 2 with a message naming the reason", re-run in one
# place so a default move cannot quietly retire one of them.
#
# WHY A SCRIPT AND NOT A TABLE IN A DOC. §7's table was hand-run. Six of its rows are now rows a bare
# `decode_run` reaches by DEFAULT (accept is on, k=2, the GDN span resolved rather than named), so the
# question "does this combination still refuse, and with the message that names it?" has to be answerable
# by anyone, in one command, against the binary that ships. A refusal no caller can reach is an untestable
# branch (`engine/decode_run.cpp`'s note beside the `--mtp-k` block), and a refusal that was retired by
# accident is the same failure with a green face.
#
# WHAT IS NOT HERE: any millisecond figure. Every case below exits at the flag parser, before the 1.55 GB
# load, so this is a correctness gate and not a timing run — it takes a few seconds and no device window
# is priced by it (`AGENTS.md`: short, correctness-only, device-gate rules). It also never runs a case to
# completion: the two POSITIVE cases print the engine's mode echo from a `--sizes-only`-style early exit
# or a one-token run, and everything else is a refusal.
#
# Usage: prototype/mtp_accept/default_refusals.sh [path/to/decode_run]
set -uo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
BIN=${1:-$root/engine/build/decode_run}
MODEL=${MODEL:-$root/models/Qwen3.5-0.8B}
IDS=${IDS:-1,2,3}

[[ -x "$BIN" ]] || { echo "no binary at $BIN"; exit 1; }

pass=0; fail=0

# refused CASE-NAME EXPECTED-NEEDLE [args...]  — must exit non-zero AND print the needle.
# A non-zero exit alone is not enough: the usage block is printed on every parse failure, so a case that
# exited 2 for the WRONG reason (an unknown flag, a bad token list) would look identical to the harness
# and pass. The needle is the half that names the reason (AGENTS.md: a check whose parse can come back
# empty cannot fail).
refused() {
  local name="$1"; shift
  local needle="$1"; shift
  local out rc
  out="$("$BIN" --model "$MODEL" "$@" 2>&1)"; rc=$?
  if [[ $rc -eq 0 ]]; then
    echo "FAIL  $name: expected a refusal, got rc=0"; fail=$((fail+1)); return
  fi
  if [[ "$out" != *"$needle"* ]]; then
    echo "FAIL  $name: rc=$rc but no message naming it. Wanted '$needle', got: $(head -1 <<<"$out")"
    fail=$((fail+1)); return
  fi
  echo "ok    $name: rc=$rc  $(head -1 <<<"$out" | cut -c1-120)"
  pass=$((pass+1))
}

# runs CASE-NAME NEEDLE [args...] — must exit 0 and echo the needle.
runs() {
  local name="$1"; shift
  local needle="$1"; shift
  local out rc
  out="$("$BIN" --model "$MODEL" "$@" 2>&1)"; rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAIL  $name: expected rc=0, got rc=$rc: $(head -1 <<<"$out")"; fail=$((fail+1)); return
  fi
  if [[ "$out" != *"$needle"* ]]; then
    echo "FAIL  $name: rc=0 but no '$needle' line: $(head -1 <<<"$out")"
    fail=$((fail+1)); return
  fi
  echo "ok    $name: $(grep -m1 -F "$needle" <<<"$out" | cut -c1-140)"
  pass=$((pass+1))
}

# env_runs CASE-NAME NEEDLE ENVVAR=VALUE [args...] — the same, with one environment assignment, because
# `QWEN_MTP_ACCEPT` / `QWEN_MTP_DRAFT` are the other half of the override contract and a flag case cannot
# witness them (the flag outranks the environment by construction).
env_runs() {
  local name="$1"; shift
  local needle="$1"; shift
  local assignment="$1"; shift
  local out rc
  out="$(env "$assignment" "$BIN" --model "$MODEL" "$@" 2>&1)"; rc=$?
  if [[ $rc -ne 0 ]]; then
    if [[ "$out" == *"$needle"* ]]; then
      echo "ok    $name: rc=$rc, refused naming it: $(head -1 <<<"$out" | cut -c1-120)"
      pass=$((pass+1)); return
    fi
    echo "FAIL  $name: expected rc=0, got rc=$rc: $(head -1 <<<"$out")"; fail=$((fail+1)); return
  fi
  if [[ "$out" != *"$needle"* ]]; then
    echo "FAIL  $name: rc=0 but no '$needle' line: $(head -1 <<<"$out")"
    fail=$((fail+1)); return
  fi
  echo "ok    $name: $(grep -m1 -F "$needle" <<<"$out" | cut -c1-140)"
  pass=$((pass+1))
}

echo "== $BIN  ($(date -u +%FT%TZ))  refusals only, no timing taken"
echo
echo "-- the k=2 round's own refusals (K2-RESULT.md §7) --"
refused "mtp-k 3 is unfunded"            "--mtp-k takes"                 --ids "$IDS" --steps 1 --mtp-k 3
refused "mtp-k with no round running"    "no round runs"                 --ids "$IDS" --steps 1 --mtp-accept 0 --mtp-k 2
refused "k=2 needs a 3-row scratch"      "needs a scratch at least 3"    --ids "$IDS" --steps 1 --wave-rows 2 --mtp-accept 1 --mtp-k 2
refused "a round needs a proposer"       "there is no draft block"       --ids "$IDS" --steps 1 --mtp-accept 1 --mtp-k 2 --mtp-draft 0
refused "a round pins the gdn span"      "pins --gdn-loop 1"             --ids "$IDS" --steps 1 --mtp-accept 1 --mtp-k 2 --gdn-loop 512
echo
echo "-- the per-step instruments a round is not (the [sibling] rows of §7) --"
# WHICH refusal fires changed with the default, and the harness records that rather than papering over it.
# With `--mtp-draft` now on by default, a bare `--time-stages` trips the DRAFT clock's own refusal (the
# stage clock cannot bill a non-uniform round schedule) BEFORE the accept one, and `--time-waves` likewise.
# Both are still refusals with a message that names the reason; the accept-vs-clock refusal is reached by
# naming the uniform schedule, which is the only spelling under which the two could ever have met.
refused "the stage clock vs a draft round"  "one draft round per step"   --ids "$IDS" --steps 1 --time-stages
refused "the wave clock vs a draft round"   "prices a prefill pass"      --ids "$IDS" --steps 1 --time-waves
refused "round vs the stage clock"       "cannot share a run"            --ids "$IDS" --steps 1 --mtp-sched every --mtp-accept 1 --mtp-k 2 --time-stages
refused "round vs the wave clock"        "prices a prefill pass"         --ids "$IDS" --steps 1 --mtp-accept 1 --mtp-k 2 --time-waves
refused "round vs a replay"              "no alignment"                  --ids "$IDS" --steps 1 --mtp-accept 1 --mtp-k 2 --replay 1,2,3
refused "round vs a session"             "hands the token budget"        --session --mtp-accept 1 --mtp-k 2
refused "round vs sizes-only"            "runs no steps"                 --sizes-only --mtp-accept 1 --mtp-k 2
refused "round vs the verify probe"      "not also armed"                --ids "$IDS" --steps 1 --mtp-accept 1 --mtp-k 2 --mtp-verify 1
refused "round vs a drafted ladder"      "makes its own proposals"       --ids "$IDS" --steps 1 --mtp-accept 1 --mtp-k 2 --mtp-draft-k 2
echo
echo "-- what the DEFAULT FLIP added: the same refusals are reachable by default, and the defaults run --"
# The span refusal now has to name BOTH halves: the flag that contradicts it, and the fact that an
# unnamed span would have been resolved rather than refused.
refused "a NAMED span still refuses"     "Drop the flag"                 --ids "$IDS" --steps 1 --gdn-loop 512
# ...and the two positive cases are the point of the whole file: the bare command the brief calls the
# shipping config must RUN, must say which mode it is in, and must say who chose the span. `--steps 0`
# so nothing decodes: the mode echo is printed before the first body, and a 3-token prefill is the
# cheapest run that reaches it.
runs "bare run is the accept path"       "MTP mode=accept k=2"           --ids "$IDS" --steps 0
runs "bare run names who set the span"   "gdn_span=1 forced_by=--mtp-accept-default" --ids "$IDS" --steps 0
runs "k=1 is still nameable"             "MTP mode=accept k=1"           --ids "$IDS" --steps 0 --mtp-k 1
runs "accept off says so"                "MTP mode=off"                  --ids "$IDS" --steps 0 --mtp-accept 0
runs "a named span is honoured, not forced" "gdn_span=1 forced_by=--gdn-loop" --ids "$IDS" --steps 0 --gdn-loop 1
# The environment still overrides BOTH ways, which the flag cases cannot show (a flag always outranks it).
env_runs "QWEN_MTP_ACCEPT=0 turns it off" "MTP mode=off" QWEN_MTP_ACCEPT=0 --ids "$IDS" --steps 0
env_runs "QWEN_MTP_ACCEPT=0 at the default k" "MTP mode=off draft=1" QWEN_MTP_ACCEPT=0 --ids "$IDS" --steps 0
env_runs "QWEN_MTP_DRAFT=0 alone refuses" "there is no draft block" QWEN_MTP_DRAFT=0 --ids "$IDS" --steps 1
env_runs "QWEN_MTP_DRAFT=0 with accept off runs" "MTP mode=off draft=0" QWEN_MTP_DRAFT=0 --ids "$IDS" --steps 0 --mtp-accept 0
echo
echo "== $pass ok, $fail FAIL"
[[ $fail -eq 0 ]] || exit 1
