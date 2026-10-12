#!/usr/bin/env python3
"""One chat-template renderer, shared by the probe and the dumper, so they cannot disagree.

`tools/stop_probe.py` decides whether a prompt is fit to pin and `tools/reference_dump.py` pins it.
The probe's answer is only a statement about the fixture if both render the same tokens — and that is
a rule about code, not about intent. Each file used to hand-build a single-user message, which worked
while every fixture had exactly one user turn; a two-turn transcript (the multi-turn fixture) cannot be
rendered that way at all, and a probe that renders differently from the dumper would report a first-stop
index for a prompt the fixture does not contain.

So the render lives here: `apply_chat_template(..., add_generation_prompt=True, enable_thinking=False)`,
i.e. the byte-identical rendering `data/refs/<fixture>/manifest.json` was made with, in one place, with
the multi-turn case as its normal case rather than an add-on. Special tokens come out of the template;
nothing here builds one by string concatenation, per the repo rule that specials are referred to by id
(`speccheck/textio.py` has the ids, and the engine has `engine::kStopToken`).

`parse_turn` is the CLI half: `--turn role:text` on either script's command line, split on the FIRST
colon so a transcript line may contain its own colons.
"""
from __future__ import annotations

ROLES = ("user", "assistant", "system")


def parse_turn(spec: str) -> dict:
    """One `--turn` argument as a message dict. Split on the first colon only, so the content may
    contain colons (times, URLs, "Note: ...") and only the role part is constrained."""
    role, sep, content = spec.partition(":")
    if not sep or role.strip() not in ROLES:
        raise SystemExit(f"--turn expects {('|'.join(ROLES))}:TEXT, got {spec!r}")
    if content == "":
        raise SystemExit(f"--turn {spec!r} has an empty content; a real empty turn is legal but"
                         " write it as a space so this cannot be a typo for a missing colon")
    return {"role": role.strip(), "content": content}


def render_transcript(tok, messages: list[dict], thinking: bool = False) -> str:
    """The transcript as the fixtures render it: chat template, generation header on, thinking off
    (unless asked). Returns text; tokenising is the caller's, because the probe and the dumper both
    need `add_special_tokens=False` (A-11) and neither should get it by accident."""
    rendered = tok.apply_chat_template(list(messages), add_generation_prompt=True,
                                       tokenize=False, enable_thinking=thinking)
    if isinstance(rendered, list):        # some templates return one string per message
        rendered = "".join(rendered)
    return rendered


def mask_specials(text: str, tok) -> str:
    """Every chat-template special in `text`, replaced by its `[id]`. For DISPLAY only, and the rule it
    serves is operational: the checkpoint serving this box stops on its own end-of-turn id, so a
    literal special token printed into a tool result or an agent reply truncates that message where it
    stands (AGENTS.md; two agents were cut off this way on 2026-10-11). Printed by id, the line still
    says everything it said.

    The literals come from the tokenizer's own table, never from a copy of them here, so the masking
    cannot fall behind a vocabulary change. Pin *files* are deliberately not run through this: a pinned
    `greedy_text` is a record of what the reference emitted and `--verify` compares it byte for byte.
    """
    for tid, added in getattr(tok, "added_tokens_decoder", {}).items():
        literal = getattr(added, "content", None)
        if literal and literal in text:
            text = text.replace(literal, f"[{tid}]")
    return text
