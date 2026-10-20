#!/usr/bin/env bash
# Chat and document summarisation on the HIP engine, with prefill/decode rates.
#
#   engine/chat.sh                      # REPL: /stats /reset /raw <text> /quit
#   engine/chat.sh --prompt "..."       # one answer, then exit
#   engine/chat.sh --summary doc.md     # summarize a document (or `-` for stdin)
#   engine/chat.sh --summary a.md b.md --max-new 400
#   engine/chat.sh --verbose            # also echo the engine's own diagnostic lines
#   engine/chat.sh --max-new 32         # cap generation length
#   engine/chat.sh --warmup 0           # no throwaway prefill: turn 1 is then the cold one
#   engine/chat.sh --ignore-lock        # start even while the GPU lock is held
#
# This drives the COMPILED ENGINE — `engine/build/decode_run --session` — not the torch
# CPU reference. The weights and the per-request state (GDN state + conv window + KV,
# engine/state.h) live on the GPU for the whole session, so a turn is one decode_step per
# token with no reload and no re-prefill of the transcript. `speccheck/chat.sh` is the
# other implementation's driver and the one that validates specification/02..07.
#
# OUTPUT. The engine writes its own report (`STATE`, `LAUNCH`, `NUMERICS`, and a `GEMM-LT`
# line per hipBLASLt shape it prices, several of them mid-turn) to the same stdout as the
# session protocol, so `chat.py` separates them by the protocol's verb list rather than by
# position — see the `TERMINAL` note there. By default you see the answer and one rate line
# per turn; `--summary` prints the summary and then the prefill and generation rates, and
# `--verbose` puts the engine's own lines back.
#
# TWO PHASES, PRINTED APART (see the module docstring of `chat.py` for the whole argument).
# The `load` line is the one-time cost per process — tokenizer build, then engine start, and
# inside the engine what loading 1.55 GB cost versus everything else about start-up. The rate
# lines are the steady state, and they are WARM by default: `--warmup N` (512 = one full wave
# of kMaxWaveRows rows) prefills N throwaway tokens, generates 2 and resets before the first
# turn you care about, and prints its own cold rate on the way. Without it the first turn is
# labelled `cold` and excluded from the warm figure — on this box a first prefill measured
# 0.09s for a 21-token turn against 0.01s for a same-sized second one, and a 173-token
# document 1,767 tok/s against 11,306 tok/s warm.
#
# The tok/s and TTFT this prints are the engine's own wall clock on a box that is also
# serving this session's tokens. They are NOT ledger numbers: bench/performance-status.md
# §6 defines the clean window a number has to come from (zero subagents, exclusive,
# blocking), and a chat session is none of those.
#
# One device, one holder: there is no thread knob here (the torch CPU original sets
# OMP_NUM_THREADS because torch oversubscribes this box; the engine's parallelism is its
# kernel block counts). What it does need is the one-device rule — the GPU is already
# running flash_serve, and a measurement taken alongside a chat session is contaminated in
# both directions. So chat.py takes tools/gpu_lock.sh for its lifetime and refuses to
# start, or to keep going, while someone else holds it.
set -euo pipefail

here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(dirname "$here")"
py="$root/.venv/bin/python"

if [[ ! -x "$py" ]]; then
  echo "no venv at $py — create it first:" >&2
  echo "  python3 -m venv .venv && .venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu && .venv/bin/pip install transformers safetensors numpy" >&2
  exit 1
fi
if [[ ! -d "$root/models/Qwen3.5-0.8B" ]]; then
  echo "no checkpoint at $root/models/Qwen3.5-0.8B — get it with:" >&2
  echo "  hf download Qwen/Qwen3.5-0.8B --local-dir models/Qwen3.5-0.8B" >&2
  exit 1
fi
if [[ ! -x "$here/build/decode_run" ]]; then
  echo "no engine binary at $here/build/decode_run — build it with:" >&2
  echo "  bash engine/build.sh" >&2
  exit 1
fi

exec "$py" "$here/chat.py" "$@"
