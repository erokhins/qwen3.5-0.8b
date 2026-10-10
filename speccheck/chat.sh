#!/usr/bin/env bash
# Interactive chat on the speccheck (specification) path, with per-turn stats.
#
#   speccheck/chat.sh                      # REPL
#   speccheck/chat.sh --prompt "..."       # one answer, then exit
#   speccheck/chat.sh --max-new 32         # cap generation length
#   speccheck/chat.sh --attn-softmax online
#   speccheck/chat.sh --mtp 2                # speculative decoding via the S09 MTP
#                                            # block; each turn reports accept stats
#
# torch CPU, bf16, greedy. This is the implementation that validates
# specification/02..07 — the HIP engine that replaces it will have its own driver
# and its own G9 numbers.
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

# one thread by default: torch CPU oversubscribes badly on this box and the
# per-token latency gets spiky, which would make the decode numbers noisy.
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
exec "$py" "$here/chat.py" "$@"
