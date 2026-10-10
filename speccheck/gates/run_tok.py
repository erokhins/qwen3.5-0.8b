#!/usr/bin/env python3
"""A-11 / S02.2 gate - the engine's own chat renderer and tokenizer reproduce the
reference prompt ids token-exactly.

Until this gate existed the speccheck implementation *consumed* pinned ids from the
manifest, so tokenization and chat rendering were entirely unverified: a wrong
template, a BOS sneaking in, or special tokens merged with adjacent text would have
been invisible. The real engine has to build the prompt string itself, so this is
the gate that makes that safe.

Checks per fixture:
  1. render_chat(user message) tokenized with add_special_tokens=False == source ids
  2. the rendered strings match the manifest's pinned chat_template_render_* probes
  3. A-10 stop ids: tokenizer eos == 248046, text_config eos == 248044, and the
     engine's stop set is exactly those two.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from speccheck import textio  # noqa: E402
from speccheck.textio import Tokenizer, render_chat  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--fixtures", default="smoke,p64,p300,dec8")
    args = ap.parse_args()

    t = Tokenizer(args.model)
    fails = []
    for tag in [x for x in args.fixtures.split(",") if x]:
        m = json.loads((ROOT / "data" / "refs" / tag / "manifest.json").read_text())
        for src in m.get("sources", []):
            text = src["text"]
            if src["kind"] == "chat":
                got = t.render_ids([{"role": "user", "content": text}])
            else:
                got = t.encode(text)
            want = list(src["ids"])
            ok = got == want
            print("%-6s %-5s %-34s ids %s (%d/%d)"
                  % (tag, src["kind"], text[:34].replace("\n", " "),
                     "exact" if ok else "MISMATCH", len(got), len(want)))
            if not ok:
                first = next((i for i in range(min(len(got), len(want)))
                              if got[i] != want[i]), min(len(got), len(want)))
                fails.append("%s[%s]: first difference at %d: engine %s vs ref %s"
                             % (tag, src["kind"], first, got[first:first + 6],
                                want[first:first + 6]))
        probe = m.get("chat_template_render_probe")
        if probe:
            mine = render_chat([{"role": "user", "content": "PING"}])
            mine_think = render_chat([{"role": "user", "content": "PING"}], thinking=True)
            for name, want in (("probe", probe),
                               ("think", m.get("chat_template_render_think"))):
                got = mine if name == "probe" else mine_think
                if want and got != want:
                    fails.append("%s: render(%s) mismatch\n     engine: %r\n     ref   : %r"
                                 % (tag, name, got, want))
    # A-10 stop ids (text_config lives in the model config, not the manifest copy)
    txt_cfg = json.loads((Path(args.model) / "config.json").read_text()).get("text_config", {})
    checks = [("tokenizer eos_token_id", t.eos_id, textio.EOS_PRIMARY_ID),
              ("text_config eos_token_id", int(txt_cfg["eos_token_id"]), textio.EOS_LEGACY_ID),
              ("think token literals", (t.think_ids[0], t.think_ids[1]),
               (textio.THINK_START_ID, textio.THINK_END_ID)),
              ("engine stop set", tuple(sorted({248046, 248044})), tuple(sorted(textio.EOS_IDS)))]
    for name, got, want in checks:
        ok = got == want
        print("A-10 %-24s got=%s want=%s %s" % (name, got, want, "ok" if ok else "FAIL"))
        if not ok:
            fails.append("A-10 %s: %s != %s" % (name, got, want))
    if fails:
        print("A-11/A-10 FAILED:")
        for f in fails:
            print("  ", f)
        return 1
    print("PASS: engine chat renderer + tokenizer reproduce the reference prompts "
          "token-exactly, and the stop ids match A-10")
    return 0


if __name__ == "__main__":
    sys.exit(main())
