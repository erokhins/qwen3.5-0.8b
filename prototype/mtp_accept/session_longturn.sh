#!/usr/bin/env bash
# session_longturn.sh — the multi-turn-past-MAXSC repro, as a durable gate-shaped tool.
#
# WHAT IT ADJUDICATES. `session_lossless.sh` proved the session's accept round emits the plain
# stream on 25 cases, every one of them a SHORT generation. This drives the geometry those cases
# never reached — a turn long enough to cross the one-pass attention ceiling (MAXSC = 512 lanes)
# and then a second `feed` — and it reproduces the defect recorded in `SESSION-ACCEPT.md` §8:
# with rounds on the session emits a DIFFERENT token stream than with rounds off, from the first
# round that decides a position above the ceiling onward. Each case prints LOSSLESS or DIFFERS
# with the first differing line, so a fix is admitted by this file going clean, not by prose.
#
# The case set is the localization itself, kept because each case rules one cause out:
#   A  one turn past MAXSC            -> width of GENERATED lanes alone is not the trigger
#   B  past MAXSC, then a second feed -> THE DEFECT
#   C  one long wave, short turn      -> a wave-built prefix above the ceiling is not the trigger
#   D  second feed, everything below MAXSC
#   E  prefilled above MAXSC, two short turns
#   F  three turns, all below MAXSC
#   G  case B with `--gdn-loop 1` on BOTH arms, which is the span the round forces anyway: if this
#      goes LOSSLESS the two forms were the round's forced GDN span and nothing else
#   E  a wave prefill above MAXSC, then a second feed
#
# THE FILE IS RED TODAY, ON PURPOSE: case B (and G) DIFFER, and that is §8. A fix is admitted when
# every case prints ok, which is why no case here is exempt from setting the exit code.
#
# WHAT RED MEANS HERE, since §9: the two arms' plain STEP rows agree bit for bit, and every token that
# differs sits at a top1-top2 margin below one bf16 ULP of the row's own top logit (`session_trigger.sh`
# prints the ratio). So this file's exact-stream equality is a stronger demand than the format can meet
# once a round decides a position above MAXSC; it stays red until the band decision in §9.7 is taken, and
# a green here would mean the tie was hidden rather than resolved.
#
# THE POPULATION IS THE SEVEN CASES AND ALL SEVEN RUN. A case whose fixture is missing prints DIFFER/no
# verdict and sets the exit code; nothing is skipped silently.
#
# NO TIMING. No millisecond figure is read or printed. Correctness only (AGENTS.md).
# SELF-LOCKING: do not run this while holding tools/gpu_lock.sh.
set -uo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/../.." && pwd)"
cd "$root" || exit 1
B=${B:-engine/build/decode_run}
M=${M:-models/Qwen3.5-0.8B}
TAG=${TAG:-"mtp session long turn (session_longturn.sh)"}

# The prompt is DERIVED from the pinned fixtures (AGENTS.md: an admission list typed from memory is
# not an admission). It must be long enough that a 240-token turn crosses 512 lanes.
PROMPT=$("$root/.venv/bin/python3" - "$here" <<'PY' 2>/dev/null || true
import json, pathlib, sys
refs = pathlib.Path(sys.argv[1]).parents[1] / "data" / "refs"
best = None
for d in sorted(refs.glob("*/manifest.json")):
    try:
        j = json.loads(d.read_text())
    except (OSError, ValueError):
        continue
    ids = None
    for key in ("sources", "prompts"):
        for row in j.get(key) or []:
            if isinstance(row, dict) and isinstance(row.get("ids"), list):
                ids = row["ids"]; break
        if ids:
            break
    if ids is None:
        ids = j.get("ids")
    if isinstance(ids, list) and 250 <= len(ids) <= 900:
        best = ids; break
print(" ".join(map(str, best)) if best else "")
PY
)
if [[ -z "$PROMPT" ]]; then
  echo "NO VERDICT: no fixture under data/refs/ yielded a prompt of 250-900 ids"
  exit 1
fi
# A second, longer prompt so case E can put a wave prefill ABOVE the ceiling rather than below it.
WIDE=$("$root/.venv/bin/python3" - "$here" <<'PY' 2>/dev/null || true
import json, pathlib, sys
refs = pathlib.Path(sys.argv[1]).parents[1] / "data" / "refs"
for d in sorted(refs.glob("*/manifest.json")):
    try:
        j = json.loads(d.read_text())
    except (OSError, ValueError):
        continue
    ids = None
    for key in ("sources", "prompts"):
        for row in j.get(key) or []:
            if isinstance(row, dict) and isinstance(row.get("ids"), list):
                ids = row["ids"]; break
        if ids:
            break
    if ids is None:
        ids = j.get("ids")
    if isinstance(ids, list) and 520 <= len(ids) <= 1200:
        print(" ".join(map(str, ids))); break
print("")
PY
)

tools/gpu_lock.sh acquire "$TAG" || exit 1
trap 'tools/gpu_lock.sh release "$TAG" >/dev/null 2>&1' EXIT INT TERM
echo "== lock held as '$TAG'; util at acquire: $("$root/tools/gpu_lock.sh" util | tail -1); no timing taken"
echo "== prompt: $(awk -v s="$PROMPT" 'BEGIN{print gsub(/ /,"",s)+1}' ) ids"

fail=0
run_pair() {   # $1 name  $2 transcript(\n-escapes)  $3 extra flags for BOTH arms
  local name=$1 script=$2 extra=${3:-}
  local f_off="--mtp-accept 0 --mtp-draft 0" f_on="--mtp-accept 1"
  printf "$script" | "$B" $f_off $extra --model "$M" --session --cap 2048 > "/tmp/sl.$name.off" 2>&1
  printf "$script" | "$B" $f_on  $extra --model "$M" --session --cap 2048 > "/tmp/sl.$name.on"  2>&1
  local a b
  for a in off on; do grep -E '^(GEN|END|PREFIX|FED|RESET) ' "/tmp/sl.$name.$a" > "/tmp/sl.$name.$a.s"; done
  a=$(grep -c '^ACCEPT' "/tmp/sl.$name.on")
  # END/PREFIX agreement is checked separately: it is the state's claim and it can agree while the
  # tokens diverge, which is exactly what the defect looks like.
  if diff -q "/tmp/sl.$name.off.s" "/tmp/sl.$name.on.s" >/dev/null; then
    printf 'ok      %s: LOSSLESS (%s protocol lines, %s rounds)\n' "$name" "$(wc -l < "/tmp/sl.$name.on.s")" "$a"
  else
    printf 'DIFFER  %s: first at: %s   (%s of %s lines differ, %s rounds)\n' "$name" \
      "$(diff "/tmp/sl.$name.off.s" "/tmp/sl.$name.on.s" | sed -n 2p)" \
      "$(diff "/tmp/sl.$name.off.s" "/tmp/sl.$name.on.s" | grep -c '^[<>]')" \
      "$(wc -l < "/tmp/sl.$name.on.s")" "$a"
    # NO case is exempt: B is the recorded defect, and a gate that forgives the case it exists to
    # catch cannot report a fix.
    fail=1
  fi
  if ! diff <(grep -E '^(END|PREFIX|FED|RESET) ' "/tmp/sl.$name.off") \
             <(grep -E '^(END|PREFIX|FED|RESET) ' "/tmp/sl.$name.on") >/dev/null; then
    printf 'DIFFER  %s: END/PREFIX disagree (state, not tokens)\n' "$name"; fail=1
  fi
}

run_pair A "reset\nfeed $PROMPT\ngen 300\n"
run_pair B "reset\nfeed $PROMPT\ngen 240\nfeed 13 198 760\ngen 20\n"
run_pair C "reset\nfeed $PROMPT 13 198 760\ngen 20\n"
run_pair D "reset\nfeed $PROMPT\ngen 100\nfeed 13 198 760\ngen 40\n"
run_pair F "reset\nfeed $PROMPT\ngen 60\nfeed 13 198\ngen 60\nfeed 760\ngen 60\n"
if [[ -n "$WIDE" ]]; then
  run_pair E "reset\nfeed $WIDE\ngen 20\nfeed 13 198 760\ngen 20\n"
else
  # Not a skip. A case that cannot run is a hole in the population, and a hole reported as `skip` is how a
  # gate starts passing for the reason that nothing was checked (AGENTS.md: an admission over a list typed
  # from memory). Either the fixture is there or the run has no verdict.
  echo "DIFFER  E: NO VERDICT - no fixture of 520-1200 ids under data/refs/, so case E cannot be adjudicated"
  fail=1
fi
run_pair G "reset\nfeed $PROMPT\ngen 240\nfeed 13 198 760\ngen 20\n" "--gdn-loop 1"

echo "== verdict: $([ $fail = 0 ] && echo 'LOSSLESS on every case' || echo 'RED: token streams differ (B/G are §8)')"
exit $fail
