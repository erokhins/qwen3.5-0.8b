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
#
# THE BAND, ADOPTED (§9.7 taken as (c) on 2026-10-25). A case may print BAND instead of ok: the streams
# differ, and `session_rowdiff.py` adjudicated EVERY flip from LOGITS ROWS DUMPED ON BOTH ARMS as the
# runner-up in the other row at a margin below one bf16 ULP of that row's top logit - two legal bf16
# spellings of one computation breaking a tie the format cannot see. That is admissible and does not set
# the exit code. What does set it: a flip at an ordinary margin (STRUCTURAL), a flip at or below MAXSC
# (below the ceiling a round's row is bit-identical to the step's, §9.4, so nothing may excuse a
# difference there), and a differing case whose dump window MISSED the flip (NO-FLIP / NO VERDICT) - a
# case that passes because nothing was checked is this repo's oldest failure class.
#
# A green run therefore means: every case identical, or every difference witnessed as a tie from rows.
# The margins are in /tmp/sl.<case>.band, the per-position JSON beside the dumps in /tmp/sldump.<case>*.
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
run_pair() {   # $1 name  $2 transcript(\n-escapes, %DUMP% placeholder)  $3 extra flags  $4 dump window "lo hi"|empty
  local name=$1 script=$2 extra=${3:-} dumpspec=${4:-}
  local f_off="--mtp-accept 0 --mtp-draft 0" f_on="--mtp-accept 1"
  # THE BAND (§9.7, adopted as (c)). Dumps are armed ONLY for the above-ceiling part of a case, because
  # that is the only place the two forms may differ: below MAXSC a round's row is bit-identical to the
  # step's (§9.4), so a flip down there is a defect and must reach the red path with nothing available to
  # excuse it. No window, or a window that missed the flip, keeps the case red: a pass that happened
  # because nothing was checked is this repo's oldest failure class.
  local d_off="/tmp/sldump.$name.off" d_on="/tmp/sldump.$name.on" dumpcmd="" dopts=()
  if [[ -n "$dumpspec" ]]; then
    dumpcmd="dump $dumpspec\n"; rm -rf "$d_off" "$d_on"
    dopts=(--out "$d_off")
  fi
  local s="${script//%DUMP%/$dumpcmd}"
  printf "$s" | "$B" $f_off $extra --model "$M" --session --cap 2048 ${dopts[@]+"${dopts[@]}"} > "/tmp/sl.$name.off" 2>&1
  local d_on_opt=(--out "$d_on"); [[ -n "$dumpspec" ]] || d_on_opt=()
  printf "$s" | "$B" $f_on  $extra --model "$M" --session --cap 2048 ${d_on_opt[@]+"${d_on_opt[@]}"} > "/tmp/sl.$name.on"  2>&1
  local a b
  for a in off on; do grep -E '^(GEN|END|PREFIX|FED|RESET) ' "/tmp/sl.$name.$a" > "/tmp/sl.$name.$a.s"; done
  a=$(grep -c '^ACCEPT' "/tmp/sl.$name.on")
  # END/PREFIX agreement is checked separately: it is the state's claim and it can agree while the
  # tokens diverge, which is exactly what the defect looks like.
  if diff -q "/tmp/sl.$name.off.s" "/tmp/sl.$name.on.s" >/dev/null; then
    printf 'ok      %s: LOSSLESS (%s protocol lines, %s rounds)\n' "$name" "$(wc -l < "/tmp/sl.$name.on.s")" "$a"
  else
    local vd="no-dump-window"
    if [[ -n "$dumpspec" ]] && "$root/.venv/bin/python3" "$here/session_rowdiff.py" --off-log "/tmp/sl.$name.off" \
         --on-log "/tmp/sl.$name.on" --off-dir "$d_off" --on-dir "$d_on" \
         --json "/tmp/sldump.$name.json" > "/tmp/sl.$name.band" 2>&1; then
      vd=NEAR-TIE
    elif [[ -n "$dumpspec" ]]; then
      vd="$(grep -oE '^(STRUCTURAL|NO VERDICT|NO-FLIP)' "/tmp/sl.$name.band" | head -1)"; vd=${vd:-ERROR}
    fi
    if [ "$vd" = NEAR-TIE ]; then
      printf 'BAND    %s: streams differ at %s, but every flip is the runner-up in the other arm at a margin under one bf16 ULP (§9.7 band) - %s rounds\n' \
        "$name" "$(diff "/tmp/sl.$name.off.s" "/tmp/sl.$name.on.s" | sed -n 2p | tr -s ' ')" "$a"
    else
      printf 'DIFFER  %s: first at: %s   (%s of %s lines differ, %s rounds) band=%s\n' "$name" \
        "$(diff "/tmp/sl.$name.off.s" "/tmp/sl.$name.on.s" | sed -n 2p)" \
        "$(diff "/tmp/sl.$name.off.s" "/tmp/sl.$name.on.s" | grep -c '^[<>]')" \
        "$(wc -l < "/tmp/sl.$name.on.s")" "$a" "$vd"
      fail=1
    fi
  fi
  if ! diff <(grep -E '^(END|PREFIX|FED|RESET) ' "/tmp/sl.$name.off") \
             <(grep -E '^(END|PREFIX|FED|RESET) ' "/tmp/sl.$name.on") >/dev/null; then
    printf 'DIFFER  %s: END/PREFIX disagree (state, not tokens)\n' "$name"; fail=1
  fi
}

NIDS=$(awk -v s="$PROMPT" 'BEGIN{print gsub(/ /,"",s)+1}')
run_pair A "reset\nfeed $PROMPT\ngen 300\n%DUMP%" "" "504 600"
run_pair B "reset\nfeed $PROMPT\ngen 240\nfeed 13 198 760\n%DUMP%\ngen 20\n" "" "$((NIDS+236)) $((NIDS+275))"
run_pair C "reset\nfeed $PROMPT 13 198 760\ngen 20\n%DUMP%" "" "504 $((NIDS+28))"
run_pair D "reset\nfeed $PROMPT\ngen 100\nfeed 13 198 760\n%DUMP%\ngen 40\n" "" ""
run_pair F "reset\nfeed $PROMPT\ngen 60\nfeed 13 198\ngen 60\nfeed 760\ngen 60\n" "" ""
if [[ -n "$WIDE" ]]; then
  run_pair E "reset\nfeed $WIDE\ngen 20\nfeed 13 198 760\n%DUMP%\ngen 20\n" "" "$(( $(awk -v s="$WIDE" 'BEGIN{print gsub(/ /,"",s)+1}') - 2 )) $(( $(awk -v s="$WIDE" 'BEGIN{print gsub(/ /,"",s)+1}') + 55 ))"
else
  # Not a skip. A case that cannot run is a hole in the population, and a hole reported as `skip` is how a
  # gate starts passing for the reason that nothing was checked (AGENTS.md: an admission over a list typed
  # from memory). Either the fixture is there or the run has no verdict.
  echo "DIFFER  E: NO VERDICT - no fixture of 520-1200 ids under data/refs/, so case E cannot be adjudicated"
  fail=1
fi
run_pair G "reset\nfeed $PROMPT\ngen 240\nfeed 13 198 760\n%DUMP%\ngen 20\n" "--gdn-loop 1" "$((NIDS+236)) $((NIDS+275))"

echo "== verdict: $([ $fail = 0 ] && echo 'every case LOSSLESS or BAND-adjudicated' || echo 'RED: a difference the band does not cover (band= names why)')"
exit $fail
