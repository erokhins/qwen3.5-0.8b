#!/usr/bin/env python3
"""Write the stage-clock prompt fixtures as comma-separated id files.

Same recipe as `tools/context_sweep.py`'s `build_prompt` (repeated prose, truncated, stop-id free),
so "128 lanes" here means what it means in the ledger rows: 112 prompt lanes + 16 steps, and
4,000 + 32 = 4,032. Written by file rather than by argv because of the MAX_ARG_STRLEN trap
(AGENTS.md): a 4,000-id argument is a single 19 KB argv field, and a file cannot be truncated.
A harness must assert the field count it wrote, so this prints it.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(ROOT / "tools"))
from context_sweep import build_prompt  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

OUT = Path(__file__).resolve().parent
tok = AutoTokenizer.from_pretrained(ROOT / "models/Qwen3.5-0.8B")
for n, name in ((112, "ids128.csv"), (4000, "ids4032.csv")):
    ids = build_prompt(tok, n)
    p = OUT / name
    p.write_text(",".join(str(t) for t in ids) + "\n")
    nf = len(p.read_text().strip().split(","))
    assert nf == len(ids), f"{name}: wrote {len(ids)} ids, file reads {nf} fields"
    print(f"{name}: {nf} ids")
