#!/usr/bin/env python3
"""A CPU-only stand-in for `decode_run`, used to witness engine/gate.py's prefill-row COVERAGE gate.

Not a fixture and not a checker: a throwaway argv sink that prints the protocol lines the gate parses
and writes row dumps in whichever shape the environment asks for, so the coverage bail can be watched
on a CPU with no engine. Three knobs, all via environment:

  STUB_PREFILL=last|all    what the stub's PREFILL line CLAIMS it ran (independent of what it writes,
                           so the §17 defect — asked `all`, echoed `last` — is reproducible)
  STUB_WRITE=sparse|last|none   what it actually writes: every row as a sparse file of the right size,
                           only §06.5's row, or nothing
  STUB_VOCAB=N             row width in floats, so a truncated corpus is expressible
"""
import os
import sys
from pathlib import Path

argv = sys.argv[1:]


def flag(name, default=None):
    return argv[argv.index(name) + 1] if name in argv else default


shape = os.environ.get("STUB_PREFILL", "last")
write = os.environ.get("STUB_WRITE", "last")
vocab = int(os.environ.get("STUB_VOCAB", "248320"))
out = Path(flag("--out", "."))
prompt_len = len(flag("--ids", "").split(","))
attn = flag("--attn-tile", "0")

print("STATE cap=999 gdn=18874368 conv=663552 kv=12288 total=1 per_token=12288")
print("LAUNCH glue=49 proj=114 gdn=72 attn=6 mlp=48 cast=187 other=20 total=496")
print(f"NUMERICS wave_rows_cap=512 wave_rows=512 gemm=library hipblas=3.6.0 attn_tile={attn}")
print(f"PREFILL tokens={prompt_len} logits={shape} prefix_len={prompt_len} wave_rows=512 pos=0..{prompt_len - 1}")
print(f"STEP kind=prefill index=0 input={flag('--ids', '0').split(',')[0]} sampled=1"
      " top_logit=1 pos_before=0 pos_after=1 stop=0")

out.mkdir(parents=True, exist_ok=True)
rows = range(prompt_len) if write == "sparse" else ([prompt_len - 1] if write == "last" else [])
if write != "none":
    for i in rows:
        p = out / f"logits{i}.bin"
        p.touch()
        os.truncate(p, vocab * 4)
print("RESULT rc=0 ms_total=0.0 ms_per_step=0.0 stopped_by=length")
