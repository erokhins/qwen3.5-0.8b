#!/usr/bin/env bash
# Regenerate the canonical §10.0 fixture set, and install each fixture only if it
# reproduces the pinned tensors **byte for byte** (tools/check_dump_reproduction.py).
#
# Why this script exists: the exact prompt texts are part of a fixture's identity, but
# before now they lived only inside the gitignored manifests, so a fixture could not be
# regenerated from the repository. They are committed under speccheck/fixtures/ and
# referenced below. `data/long.txt` (also gitignored) is the same text as p300.txt.
#
#   ./tools/dump_fixtures.sh                 # every fixture
#   ./tools/dump_fixtures.sh smoke dec8      # just these
#
# The three boundary fixtures at the bottom (p128, chunk64b, wide520f — plan Q5 package P3) are
# sized to a TOKEN COUNT, not just a topic: their whole point is where the prompt ends relative to
# the §04.7 chunk size of 64, and one that drifts by a token guards nothing. `speccheck/fixtures/
# *.txt` are therefore exact, and the count each renders to is written in its recipe. The count is
# checkable without loading the model, tokenizer only:
#
#   .venv/bin/python -c "import sys;sys.path[:0]=['tools'];from transformers import AutoTokenizer;from chat_render import render_transcript as r;t=AutoTokenizer.from_pretrained('models/Qwen3.5-0.8B');print(len(t(r(t,[{'role':'user','content':open('speccheck/fixtures/p128.txt').read().strip()}],thinking=False),add_special_tokens=False)['input_ids']))"
#
# None of the three is priced: they carry no `route_allowance.json`, so they ride the flat §10.1
# floors. That is not an oversight — `tools/band_history.py` keys on the file path, so a fixture
# copy that carried margin values would read as NEW band keys with no ledger line and the gate
# would refuse to run. Pricing them is a main-agent derivation (`engine/route_allowance.py`), and
# a new fixture's own margin is a derivation rather than a band move (D-2).
#
# Each generation goes to $TMP first; a fixture whose old pins do not reproduce stops
# the run with the offending names printed, and the pinned directory is left untouched.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=.venv/bin/python
FIX=speccheck/fixtures
TMP=${TMP:-/tmp/refs_regen}
mkdir -p "$TMP"

run() {
  local tag=$1; shift
  if [ ! -d "data/refs/$tag" ]; then
    echo "no pinned fixture at data/refs/$tag - refusing (this script verifies reproduction," \
         "it does not create fixtures)" >&2
    return 1
  fi
  rm -rf "$TMP/$tag"
  echo "=== $tag: reference_dump.py $*"
  "$PY" tools/reference_dump.py --out "$TMP/$tag" "$@"
  "$PY" tools/check_dump_reproduction.py "data/refs/$tag" "$TMP/$tag"
  cp "$TMP/$tag"/*.npy "$TMP/$tag/manifest.json" "data/refs/$tag/"
  echo "=== $tag installed ($(ls "data/refs/$tag"/*.npy | wc -l) pins)"
}

SEL="${*// /,}"
want() { [ -z "$SEL" ] || case ",$SEL," in *",$1,"*) return 0;; *) return 1;; esac; }

# The three boundary fixtures are sized to a rendered TOKEN COUNT, and that count is a property of the
# prompt file — nothing in the dumper or the gate checks it. Regenerating from an edited prompt file
# produces a perfectly self-consistent fixture that guards a different lane count, which reads as
# coverage and is worse than none. So re-count here before dumping (tokenizer only, ~2 s, no model).
rendered_tokens() {
  "$PY" - "$1" <<'PY'
import sys
sys.path[:0] = ["tools", "speccheck"]
from chat_render import render_transcript
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("models/Qwen3.5-0.8B")
msgs = [{"role": "user", "content": open(sys.argv[1]).read().strip()}]
print(len(tok(render_transcript(tok, msgs, thinking=False), add_special_tokens=False)["input_ids"]))
PY
}

want_tokens() {  # <prompt-file> <expected rendered token count>
  local got; got=$(rendered_tokens "$1")
  if [ "$got" != "$2" ]; then
    echo "FATAL: $1 renders to $got tokens, but the recipe says $2 — the boundary case is gone." >&2
    echo "       Either restore the prompt, or re-derive the fixture's lane arithmetic and both numbers." >&2
    exit 1
  fi
  echo "  prompt renders to $2 tokens, as the recipe claims: $1"
}

if want smoke; then
  run smoke --message "The capital of France is" --decode 2
fi
if want dec8; then
  run dec8 --message "The capital of France is" --decode 8
fi
if want prefix24dec8; then
  run prefix24dec8 --prompt "$(cat "$FIX/prefix24dec8.txt")" --decode 8
fi
if want p8; then
  run p8 --prompt "$(cat "$FIX/p8.txt")" --decode 0
fi
if want p64; then
  run p64 --message "$(cat "$FIX/p64.txt")" --decode 0
fi
if want p300; then
  run p300 --message "$(cat "$FIX/p300.txt")" --decode 0
fi

# ---- the P3 boundary fixtures (bench/plan-prefill-2026-10-12.md §5.1, finding F3) ---------------
# What none of the five shipped fixtures can say: no prompt of theirs ends at a multiple of 64
# (prompt lens are 17/17/27/232/295), and nothing ever crossed the attention stage's 512-lane
# ceiling. These three are pins for the cases §04.7 and §07.3 keep referring to, written BEFORE any
# batched kernel exists to be gated by them, so P4-P8 inherit gates instead of inventing them.
# Each carries the §08 state FINGERPRINTS at every step plus ONE full per-layer step (--deep-steps),
# which is what buys the depth: one full step is 25.96 MiB and one BF16 logits vector 0.4738 MiB,
# so a ~60 MB budget buys ~71 pinned steps with a full step, ~126 without one.

if want p128; then
  want_tokens "$FIX/p128.txt" 128
  # Prompt renders to EXACTLY 128 tokens = 64·2, so the prefill end-condition sits precisely on a
  # chunk boundary: no ragged tail, no zero-β padding anywhere in the prompt, and `state_new`/the
  # state fingerprints are the no-padding case of §04.7. 8 replayed steps take the run end to 136
  # lanes, i.e. 8 lanes into the third chunk, so the boundary-exact prompt and a chain that has left
  # it are both in one fixture. Probed first: the stop id 248046 first appears at generated step 92,
  # far outside the 8-step window (§10.5/§07.5: a deep fixture states its own first-stop index).
  run p128 --message "$(cat "$FIX/p128.txt")" --decode 8 --lean --deep-steps 7 --logits-bits bf16 \
      --first-stop "92 (probed to 300 generated tokens; legacy stop 248044 at 94)"
fi
if want chunk64b; then
  want_tokens "$FIX/chunk64b.txt" 120
  # The RAGGED half, and the §04.7 case the plan's row was ambiguous about — the task said to cover
  # both readings, so this one does:
  #   * the PROMPT is 120 tokens = 64·1 + 56, so the last prompt chunk is partial and HF's zero-β /
  #     zero-g padding is real here (this is what p64/p300 never exercise: their prompts end at 46
  #     and 277 inside a single partial chunk, with no full chunk carried across a boundary);
  #   * the 30 forced steps cross 64·2 = 128 at step 8 (lane 127 is the step-8 token), so the chunk
  #     a token lands in changes mid-chain and the run ends at 150 = 64·2 + 22 — a ragged tail at
  #     both ends of the fixture, with a chunk carry in the middle.
  # Probed first: stop id 248046 first at generated step 129, well outside the 30-step window.
  run chunk64b --message "$(cat "$FIX/chunk64b.txt")" --decode 30 --lean --deep-steps 29 \
      --logits-bits bf16 --first-stop "129 (probed to 300 generated tokens; legacy stop 248044 at 131)"
fi
if want wide520f; then
  want_tokens "$FIX/wide520f.txt" 500
  # The >512-lane case: 500 prompt tokens + 24 forced steps = 524 lanes, above `MAXSC` (read from
  # `prototype/attn_decode/attn_decode_kernels.h`, enforced by `engine/kernels_attn.cpp`), which no
  # fixture has ever crossed — the deepest was dec300f at 423.
  # THE CROSSING IS PROMPT-SIDE BY DESIGN, which is a deliberate change to plan §5.1's sketch ("~250
  # tokens + ~275 forced steps, crossing inside the forced chain"). The budget decides it: a pinned
  # step costs 0.4738 MiB of logits and the gate pairs every pinned step against its own logits, so a
  # 275-step chain is 130 MiB before anything else is pinned. A 500-token prompt crosses at step 12
  # (the step-11 token is lane 512; step 12 is the first token the one-pass stage cannot serve) and
  # leaves 12 pinned steps beyond the ceiling, at ~56 MiB total. It also puts the over-cap lane count
  # where P8 needs it — in the prefill wave, not only in decode.
  # IT IS INERT UNTIL P7: `gate.py` refuses this fixture's device run at its lane-ceiling preflight
  # today, which is the expected state and is what the plan means by "MAXSC no longer caps at 512".
  # Step names here are the gate's 0-based ones: `decode11` reads exactly 512 lanes (the last the
  # one-pass stage serves), `decode12` is the first it cannot, and `decode12`..`decode23` are above it.
  # What can be checked now without a device is that the pin is self-consistent:
  #   .venv/bin/python engine/gate.py --fixture wide520f --no-engine
  # Probed first, and the strongest of the three: no stop id in 300 generated tokens (and none of the
  # legacy 248044 either), so nothing in the 24-step window is a turn the model had already finished.
  run wide520f --message "$(cat "$FIX/wide520f.txt")" --decode 24 --lean --deep-steps 23 \
      --logits-bits bf16 --first-stop "none in 300 (probed to 300 generated tokens; legacy stop 248044 also none)"
fi
