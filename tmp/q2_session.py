"""Throwaway (device driver): does `decode_run --session` carry a two-turn transcript correctly?

The question `engine/chat.sh` found by running: a session advances the state with `feed` and then
generates with `gen`, and the invariant is that the last token shown to the caller is the one still
pending in the state. If that bookkeeping is off by one the transcript in the cache is a token short or
long and turn 2 is judged a different prompt than the reference saw.

So: feed `mt32f`'s transcript, generate 32, and compare the GEN ids with the reference's own forced
continuation. The alignment is `gen[i] == greedy_continuation[i]` — the session emits the sample that
the last fed token produced as its first GEN line, which run mode prints on the last PROMPT line
instead. Same fact, two line protocols, one index of difference: this is the off-by-one that makes a
free-run comparison read 1/64.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path("/home/erokhins/erokhins_projects/qwen3.5-0.8b")
TAG = sys.argv[1] if len(sys.argv) > 1 else "mt32f"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32

man = json.loads((ROOT / "data" / "refs" / TAG / "manifest.json").read_text())
ids, greedy = man["prefill_ids"], man["greedy_continuation"]
script = "feed " + " ".join(map(str, ids)) + "\n" + f"gen {N}\nquit\n"
p = subprocess.run([str(ROOT / "engine" / "build" / "decode_run"), "--session"],
                   input=script, capture_output=True, text=True)
out = p.stdout
(ROOT / "tmp" / f"q2_session_{TAG}.log").write_text(out + "\n[stderr]\n" + p.stderr)
gen = [int(ln.split()[1]) for ln in out.splitlines() if ln.startswith("GEN ")]
fed = [ln for ln in out.splitlines() if ln.startswith(("READY", "FED", "END", "RESET", "ERROR"))]
print("\n".join(fed))
want = greedy[:len(gen)]
first = next((i for i, (a, b) in enumerate(zip(gen, want)) if a != b), None)
print(f"session generated {len(gen)} ids; want greedy_continuation[:{len(gen)}];"
      f" first mismatch: {first}")
print("  engine:", gen[:12], "...", gen[-4:] if gen else [])
print("  pinned:", want[:12], "...", want[-4:] if want else [])
if first is not None and first < len(gen):
    print(f"  at {first}: engine {gen[first]} vs pinned {want[first]}")
from transformers import AutoTokenizer   # noqa: E402

tok = AutoTokenizer.from_pretrained(str(ROOT / "models" / "Qwen3.5-0.8B"))
print("  engine text:", repr(tok.decode(gen))[:300])
print("  pinned text:", repr(man["greedy_text"])[:300])
print("SESSION " + ("PASS" if gen and first is None else "FAIL"))
print("  engine text:", repr(tok.decode(gen))[:300])
print("  pinned text:", repr(man["greedy_text"])[:300])
print("SESSION " + ("PASS" if gen and first is None else "FAIL"))
