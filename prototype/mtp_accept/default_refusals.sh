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
# WHAT IS NOT HERE: any millisecond figure. The refusal cases exit at the flag parser, before the 1.55 GB
# load, so this is a correctness gate and not a timing run — no device window is priced by it (`AGENTS.md`:
# short, correctness-only, device-gate rules). The POSITIVE cases do load the model and prefill 3 tokens,
# because the mode echo they assert is printed after the load and before the first body; `--steps 0` is the
# cheapest run that reaches it.
#
# CORRECTED 2026-10-25 (option C, `prototype/mtp_accept/DEFAULT-ON.md` §7). This file used to assert that a
# round pins `--gdn-loop 1`, in both spellings. That pin was taken run-wide and taxed every prefill wave the
# shipping default submitted, so the forcing moved onto the round's own bodies and the CLI refusal was
# retired. A battery that demands a refusal the engine now correctly allows is a dead gate, so the span rows
# became POSITIVE cases that assert the echoed spans. The last refusal in this file — `--mtp-midsplit` — is
# the same requirement kept on purpose: that probe byte-compares a split body against an UNSPLIT one, and the
# unsplit side arms no checkpoint, so nothing forces it and the flag still has to.
#
# CORRECTED AGAIN 2026-10-25 (`prototype/mtp_accept/ADAPTIVE.md` §7, `DEFAULT-ON.md` §8). The adaptive depth
# became the shipping default, so its three `silent` cases — which proved default-OFF by the ABSENCE of the
# policy's config line — are now `runs` cases asserting its PRESENCE on a bare run, at the default depth and
# at a k=1 ceiling. `silent` did not leave the file: it judges the NAMED-off arms (`--mtp-adaptive 0`,
# `QWEN_MTP_ADAPTIVE=0`, and a run with no round at all), which is the half the flag-off control depends on
# and the half a presence-only battery could not see.
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
runs() {  local name="$1"; shift
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

# session_runs CASE-NAME NEEDLE [args...] — `runs` for `--session`, with stdin at /dev/null. A session
# reads its command protocol from stdin, so a case that only wants the geometry lines printed BEFORE the
# first command must not inherit the harness's stdin (and must not hang waiting for one). `session_loop`
# sees EOF, exits, and exits 0 — which is what makes an rc check meaningful here.
session_runs() {
  local name="$1"; shift
  local needle="$1"; shift
  local out rc
  out="$("$BIN" --model "$MODEL" "$@" </dev/null 2>&1)"; rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAIL  $name: expected rc=0, got rc=$rc: $(head -1 <<<"$out")"; fail=$((fail+1)); return
  fi
  if [[ "$out" != *"$needle"* ]]; then
    echo "FAIL  $name: rc=0 but no '$needle' line: $(head -1 <<<"$out")"; fail=$((fail+1)); return
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

# env_silent CASE-NAME ABSENT-NEEDLE ENVVAR=VALUE [args...] — the environment's half of the OFF side. The
# env names the knob just as the flag does, and a control arm that set `QWEN_MTP_ADAPTIVE=0` while the binary
# silently kept the policy would print a rate for a configuration that never ran.
env_silent() {
  local name="$1"; shift
  local needle="$1"; shift
  local assignment="$1"; shift
  local out rc
  out="$(env "$assignment" "$BIN" --model "$MODEL" "$@" 2>&1)"; rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAIL  $name: expected rc=0, got rc=$rc: $(head -1 <<<"$out")"; fail=$((fail+1)); return
  fi
  if [[ "$out" == *"$needle"* ]]; then
    echo "FAIL  $name: rc=0 but the run printed '$needle' although $assignment named it off"
    fail=$((fail+1)); return
  fi
  echo "ok    $name: rc=0 and no '$needle' line under $assignment"
  pass=$((pass+1))
}

# silent CASE-NAME ABSENT-NEEDLE [args...] — must exit 0 AND must NOT print the needle. This is the shape
# that admits a DEFAULT-OFF flag, and since 2026-10-25 it admits only the arms that are named off: the
# adaptive policy is ON by default now, so ABSENCE proves "the caller named `--mtp-adaptive 0` and the engine
# honoured it" and presence proves "the default armed it" (`runs` cases below). Keeping both shapes is what
# keeps the flag honest in either direction: a battery that asserted absence by default is a dead gate that
# goes red on the correct build, and one that asserted only presence could not see `--mtp-adaptive 0` stop
# being honoured (`prototype/mtp_accept/ADAPTIVE.md` §7 owns the flip and its measurements).
silent() {
  local name="$1"; shift
  local needle="$1"; shift
  local out rc
  out="$("$BIN" --model "$MODEL" "$@" 2>&1)"; rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "FAIL  $name: expected rc=0, got rc=$rc: $(head -1 <<<"$out")"; fail=$((fail+1)); return
  fi
  if [[ "$out" == *"$needle"* ]]; then
    echo "FAIL  $name: rc=0 but the run printed '$needle', so the policy armed on a run whose"
    echo "        caller named it off: $(grep -m1 -F "$needle" <<<"$out" | cut -c1-140)"
    fail=$((fail+1)); return
  fi
  echo "ok    $name: rc=0 and no '$needle' line — the caller named the policy off, and the log is what"
  echo "        proves it honoured that (a silent run is the only witness an off arm leaves)"
  pass=$((pass+1))
}

echo "== $BIN  ($(date -u +%FT%TZ))  refusals only, no timing taken"
echo
echo "-- the k=2 round's own refusals (K2-RESULT.md §7) --"
refused "mtp-k 3 is unfunded"            "--mtp-k takes"                 --ids "$IDS" --steps 1 --mtp-k 3
refused "mtp-k with no round running"    "no round runs"                 --ids "$IDS" --steps 1 --mtp-accept 0 --mtp-k 2
refused "k=2 needs a 3-row scratch"      "needs a scratch at least 3"    --ids "$IDS" --steps 1 --wave-rows 2 --mtp-accept 1 --mtp-k 2
refused "a round needs a proposer"       "there is no draft block"       --ids "$IDS" --steps 1 --mtp-accept 1 --mtp-k 2 --mtp-draft 0
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
refused "round vs sizes-only"            "runs no steps"                 --sizes-only --mtp-accept 1 --mtp-k 2
refused "round vs the verify probe"      "not also armed"                --ids "$IDS" --steps 1 --mtp-accept 1 --mtp-k 2 --mtp-verify 1
refused "round vs a drafted ladder"      "makes its own proposals"       --ids "$IDS" --steps 1 --mtp-accept 1 --mtp-k 2 --mtp-draft-k 2
echo
echo "-- the session: admitted by the round driver, with the instruments it cannot own still refused --"
# THE `--session` REFUSAL WAS RETIRED 2026-10-26 and its replacement is asserted in both directions.
# `engine/mtpaccept::session_body` drives rounds on the session path now, so a battery that still demanded
# "round vs a session" would be a dead gate reddening the correct build (the `--gdn-loop` rows above are the
# same trap, one default earlier). What IS still refused is each instrument whose object a session does not
# have — and every one of them became REACHABLE by the retirement, because until then `--session` + a round
# was a refusal at the flag parser and nothing downstream could be exercised. `--mtp-meter` and
# `--mtp-dense` in particular were unreachable-by-transitivity before; they are inert-on-the-session-path
# unless refused, which is why they are rows here at all.
session_runs "session admits the round"          "MTPGEOM k=2 rows=3"        --session --mtp-accept 1 --mtp-k 2
session_runs "session states its geometry"       "MTPACCEPT on=1 rows=3"     --session
session_runs "session arms the depth policy"     "MTPADAPT on=1 k_start=2"   --session
session_runs "a k=1 session is nameable"         "MTPGEOM k=1 rows=2"        --session --mtp-k 1
session_runs "accept-off session echoes nothing" "READY cap="               --session --mtp-accept 0 --mtp-draft 0
refused    "session vs the dense lane"           "stages the skipped lane"   --session --mtp-dense 1
refused    "session vs prompt-lane seeding"      "no single prompt wave"     --session --mtp-prompt-lanes 8
refused    "session vs the draft ladder"         "re-anchors the position map" --session --mtp-draft-k 2
refused    "session vs the verify probe"         "has no such list"          --session --mtp-verify 1
refused    "session vs the round meter"          "no run window to close"    --session --mtp-meter 1
refused    "session vs the stage clock"          "one draft round per step"  --session --time-stages
refused    "session vs the wave clock"           "prices a prefill pass"     --session --time-waves
refused    "session vs a replay"                 "no alignment"              --session --mtp-accept 1 --replay 1,2,3
refused    "sizes-only still refuses"            "runs no steps"             --sizes-only --mtp-accept 1
echo
echo "-- what the DEFAULT FLIP added: the same refusals are reachable by default, and the defaults run --"
# THE GDN SPAN, as of 2026-10-25. Two batteries ago this file asserted `--gdn-loop 512 --mtp-accept 1`
# REFUSED ("a round pins the gdn span", "a NAMED span still refuses"). The round still needs the
# one-token-step span — its rollback restores a mid-round checkpoint, and at any wider span that state
# exists at no address — but the requirement is the ROUND's, so the forcing moved to the round's bodies
# (`engine/decode_step.h::kMtpRoundGdnSpan`, applied by `submit_wave_body` wherever the checkpoint is
# armed) and the CLI-level refusal was RETIRED. Asserting the old refusal now would be a dead gate: it
# would demand a refusal the engine correctly no longer gives. So the span cases below are POSITIVE, and
# each one asserts the echoed spans rather than a mere exit code — a `runs` case that only checked rc=0
# could not tell "legal" from "legal but silently re-pinned", which is the failure the echo exists to make
# visible. What the echo CANNOT see is whether the round's body actually walked one step per launch; that
# is witnessed numerically, not by a flag: an unforced round at span 3 writes no checkpoint slot, so a
# reject restores uninitialised memory and the emitted stream diverges from the plain one — which is what
# `prototype/mtp_accept/m3_loop_rate.sh` adjudicates on all 49 corpus lines.
refused "the midsplit arm still needs span 1" "pins the reference arm"   --ids "$IDS" --steps 1 --mtp-accept 0 --mtp-verify 1 --mtp-midsplit 1 --gdn-loop 512
runs "bare run is the accept path"       "MTP mode=accept k=2"           --ids "$IDS" --steps 0
runs "the default span is the caller's, not 1" "gdn_span_round=1 gdn_span_other=512 forced_by=default" --ids "$IDS" --steps 0
runs "a NAMED span is legal and echoed"  "gdn_span_round=1 gdn_span_other=512 forced_by=--gdn-loop" --ids "$IDS" --steps 0 --mtp-accept 1 --mtp-k 2 --gdn-loop 512
runs "any span is nameable, not just 1 and 512" "gdn_span_round=1 gdn_span_other=7 forced_by=--gdn-loop" --ids "$IDS" --steps 0 --mtp-accept 1 --mtp-k 2 --gdn-loop 7
runs "an explicit span 1 is honoured"    "gdn_span_round=1 gdn_span_other=1 forced_by=--gdn-loop" --ids "$IDS" --steps 0 --gdn-loop 1
runs "k=1 is still nameable"             "MTP mode=accept k=1"           --ids "$IDS" --steps 0 --mtp-k 1
runs "accept off echoes ONE span"        "MTP mode=off draft=1 gdn_span=512" --ids "$IDS" --steps 0 --mtp-accept 0
# The environment still overrides BOTH ways, which the flag cases cannot show (a flag always outranks it).
env_runs "QWEN_MTP_ACCEPT=0 turns it off" "MTP mode=off" QWEN_MTP_ACCEPT=0 --ids "$IDS" --steps 0
env_runs "QWEN_MTP_ACCEPT=0 at the default k" "MTP mode=off draft=1" QWEN_MTP_ACCEPT=0 --ids "$IDS" --steps 0
env_runs "QWEN_MTP_DRAFT=0 alone refuses" "there is no draft block" QWEN_MTP_DRAFT=0 --ids "$IDS" --steps 1
env_runs "QWEN_MTP_DRAFT=0 with accept off runs" "MTP mode=off draft=0" QWEN_MTP_DRAFT=0 --ids "$IDS" --steps 0 --mtp-accept 0
echo
echo "-- the adaptive depth: --mtp-adaptive (engine/mtp_adaptive.h), ON by default whenever a round runs --"
# The refusal that defines the flag. The policy consumes the verdict readback a ROUND already pays, so with
# no round running it has no input and no effect; it is keyed on the VALUE rather than on `given`, because
# `--mtp-adaptive 0 --mtp-accept 0` is a truthful pair that must run (unlike `--mtp-k`, which is keyed on
# `given` because naming a depth you never run is a knob that does nothing). Note that the DEFAULT now
# follows `--mtp-accept` rather than being a flat 1: `parse_args` resolves an unnamed policy to the accept
# flag, so a bare `--mtp-accept 0` run is a run with no policy and no refusal, and this refusal is reached
# only by an explicit ask — which is the case worth refusing and the one asserted here.
refused "adaptive with no round running" "--mtp-adaptive adapts the depth" --ids "$IDS" --steps 1 --mtp-accept 0 --mtp-adaptive 1
refused "adaptive is a 0/1 switch"      "--mtp-adaptive takes 1"          --ids "$IDS" --steps 1 --mtp-adaptive 2
# The per-step instruments. These refuse for the ROUND's sake, not the policy's — the policy adds no
# instrument of its own — and the battery asserts WHICH message fires, because the accept default makes
# several of them reachable by a bare run and a wrong-reason exit is invisible from the status code alone.
refused "adaptive vs the stage clock"   "one draft round per step"        --ids "$IDS" --steps 1 --time-stages --mtp-adaptive 1
refused "adaptive vs the stage clock, named schedule" "cannot share a run" \
                                          --ids "$IDS" --steps 1 --mtp-sched every --mtp-adaptive 1 --time-stages
refused "adaptive vs the wave clock"    "prices a prefill pass"           --ids "$IDS" --steps 1 --time-waves --mtp-adaptive 1
refused "adaptive vs sizes-only"        "runs no steps"                   --sizes-only --mtp-adaptive 1
refused "adaptive vs a replay"          "no alignment"                    --ids "$IDS" --steps 1 --mtp-adaptive 1 --replay 1,2,3
refused "adaptive vs the verify probe"  "not also armed"                  --ids "$IDS" --steps 1 --mtp-adaptive 1 --mtp-verify 1
# The positive side. When it IS asked, the run states the whole policy in its own log BEFORE any body
# runs — every threshold it will apply, the ceiling, and `re_promote=0` — so a later reader can re-derive
# the decision from the transcript instead of trusting the binary, and so `engine/gate.py` section (m) can
# parse its arithmetic off the engine rather than re-typing a constant that could drift.
runs "adaptive on states its policy"     "MTPADAPT on=1 k_start=2"        --ids "$IDS" --steps 0 --mtp-adaptive 1
runs "adaptive at a k=1 ceiling"         "MTPADAPT on=1 k_start=1"        --ids "$IDS" --steps 0 --mtp-adaptive 1 --mtp-k 1
runs "adaptive leaves the geometry echo alone" "MTPGEOM k=2 rows=3 mid_slots=2" --ids "$IDS" --steps 0 --mtp-adaptive 1
runs "adaptive echoes no re-promotion"   "re_promote=0"                   --ids "$IDS" --steps 0 --mtp-adaptive 1
# THE DEFAULT, proved by PRESENCE. These three cases used to be `silent` cases proving the default was OFF,
# and a battery asserting the old default is worse than no battery: it reddens the correct build and greens
# the regressed one. The flip (`prototype/mtp_accept/ADAPTIVE.md` §7) makes the policy ship, so all three now
# assert what the run PRINTS, at three different depths and by three different routes to them, because the
# interesting regressions differ: a bare run (the shipping path), an explicit 1 at the default depth, and an
# explicit 1 at a k=1 ceiling — where the policy is armed but may only abandon, never widen.
runs "the DEFAULT arms the policy"        "MTPADAPT on=1 k_start=2"       --ids "$IDS" --steps 0
runs "the DEFAULT echoes adaptive=on"     "adaptive=on"                   --ids "$IDS" --steps 0
runs "an explicit --mtp-adaptive 1 is the same shape" "MTPADAPT on=1 k_start=2" --ids "$IDS" --steps 0 --mtp-adaptive 1
runs "a k=1 default arms the policy at that ceiling" "MTPADAPT on=1 k_start=1"  --ids "$IDS" --steps 0 --mtp-k 1
# And the OFF side, still proved by ABSENCE, because "nameable and honouring every existing path" is a claim
# about the flag-off arm and the log is the only witness it leaves. The mode echo says `adaptive=off` so a
# reader of an old log can still tell a static k=2 run from an adaptive one that never fired — with the
# policy on by default, the ABSENCE of an event line no longer distinguishes them.
silent "an explicit --mtp-adaptive 0 is silent"  "MTPADAPT"          --ids "$IDS" --steps 0 --mtp-adaptive 0
runs "an explicit --mtp-adaptive 0 echoes adaptive=off" "adaptive=off"   --ids "$IDS" --steps 0 --mtp-adaptive 0
env_silent "QWEN_MTP_ADAPTIVE=0 is silent" "MTPADAPT" QWEN_MTP_ADAPTIVE=0 --ids "$IDS" --steps 0
silent "no round, no policy line"               "MTPADAPT"          --ids "$IDS" --steps 0 --mtp-accept 0
echo
echo "== $pass ok, $fail FAIL"
[[ $fail -eq 0 ]] || exit 1
