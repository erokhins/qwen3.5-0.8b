#!/usr/bin/env python3
"""Write the token-id fixtures this window needs (CPU-only: tokenizer only, no device)."""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from context_sweep import build_prompt  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

tok = AutoTokenizer.from_pretrained(str(ROOT / "models/Qwen3.5-0.8B"))
for n in (4096, 16384):
    ids = build_prompt(tok, n)
    assert len(ids) == n, (len(ids), n)
    out = ROOT / f"tmp/a12why/ids{n}.txt"
    out.write_text(",".join(str(t) for t in ids))
    print(n, "wrote", len(ids), "ids,", len(",".join(map(str, ids))), "bytes ->", out)
