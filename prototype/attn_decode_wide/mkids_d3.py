#!/usr/bin/env python3
"""mkids_d3.py — write the two prompt fixtures `ab_d3_engine.sh` times, as id FILES.

CPU only: it loads a tokenizer, writes two files, prints the field count, touches no device.

SAME RECIPE AS THE LEDGER. `tools/context_sweep.py`'s `build_prompt` — repeated prose, truncated to
exactly N tokens, asserted free of the stop id (by id, 248046, which is checked not assumed). That is
the same input the decode-attribution window used, so "4,032 lanes" here means what that row means:
4,000 prompt lanes + 32 steps, and 112 + 16 = 128. Reusing the function rather than copying the
paragraph is deliberate: two prompts that differ in prose are two rulers.

WHY FILES AND NOT `--ids`. A 4,000-id argument is one argv field of ~19 KB and the per-field kernel
limit is 131,072 B (AGENTS.md), so it fits here — but it does not fit at 32k, and the failure at the
limit is `exec` refusing with E2BIG in 0 s, which reads as "the engine crashes at 32k". Files also
cannot be silently truncated by a shell quoting accident, and a `--ids-file` splits on COMMAS, so the
file is written comma-separated and the writer asserts the field count it wrote. A harness that trusts
the filename's nominal width instead of the file's field count is how a 32,000-token fixture quietly
became 1,024 tokens.

Usage: mkids_d3.py OUTDIR          (writes OUTDIR/ids128.csv and OUTDIR/ids4032.csv)
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from context_sweep import build_prompt  # noqa: E402  the ledger's own prompt recipe
from transformers import AutoTokenizer  # noqa: E402

out = Path(sys.argv[1] if len(sys.argv) > 1 else ROOT / "tmp" / "d3ab")
out.mkdir(parents=True, exist_ok=True)
tok = AutoTokenizer.from_pretrained(ROOT / "models" / "Qwen3.5-0.8B")
# (prompt lanes, file name). The step counts that turn these into 128 and 4,032 lanes live in the
# A/B script; the two must move together or the lane labels lie.
for n, name in ((112, "ids128.csv"), (4000, "ids4032.csv")):
    ids = build_prompt(tok, n)
    p = out / name
    p.write_text(",".join(str(t) for t in ids) + "\n")
    nf = len(p.read_text().strip().split(","))
    assert nf == len(ids), f"{name}: wrote {len(ids)} ids, the file reads {nf} fields"
    print(f"{name}: {nf} ids")
