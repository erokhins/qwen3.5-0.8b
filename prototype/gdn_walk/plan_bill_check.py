#!/usr/bin/env python3
"""Cross-check the `gdn_rec` plan bill for `--gdn-norm hoist` against the span loop the dispatcher runs.

WHY THIS EXISTS. `engine/stage_clock.cpp` derives the planned `gdn_rec` launch count from the shape
(`tokens`, `wave_rows`, `gdn_loop_span`, `gdn_norm_hoisted`), and `--time-waves` reports `plan_match=1`
when the plan agrees with what was counted. That comparison is the audit, but it is compiled into the
binary that is being audited — the same self-agreement blind spot `AGENTS.md` records for
`wave_plan_bill` as a canary — so the derivation gets an independent spelling here, in the shapes a run
can actually reach, before a device window spends 60 s discovering an off-by-one.

Run: `.venv/bin/python prototype/gdn_walk/plan_bill_check.py`
"""
import sys

G = 18  # GDN layers (kGdnLayerCount / kGdnLayerEvery from engine/state.h: 18 of the 24)


def resolve_gdn_loop(want: int, rows: int) -> int:
    """Mirror of `kernels::resolve_gdn_loop` (engine/kernels_gdn.cpp:197), spelled exactly:
    `(rows <= 1 || want <= 1) ? 1 : want`. A span WIDER than the body is deliberately left alone —
    `ceil(rows/span) == 1` covers it — so the mirror must not clamp it, or the tail arithmetic below
    would describe a shape the dispatcher never builds."""
    return 1 if (rows <= 1 or want <= 1) else want


def body_widths(tokens: int, wave_rows: int):
    """The dispatcher slices the prompt into bodies of `wave_rows`, the last one ragged."""
    n = tokens
    t = 0
    while t < n:
        rows = wave_rows if wave_rows < n - t else n - t
        yield rows
        t += rows


def launched(tokens: int, wave_rows: int, want_span: int, hoisted: bool) -> int:
    """What the dispatcher SUBMITS on the `gdn_rec` row, mirrored from engine/decode_step.cpp.

    `gdn_span > 1` takes the span branch: ceil(rows/span) spans, and `--gdn-norm hoist` puts a pre-pass
    beside each, so 2 per span. `gdn_span == 1` — a one-row body, or `--gdn-loop 1` — takes the
    per-token loop over EVERY token of the body (engine/decode_step.cpp's final `else`), one launch each
    and no pre-pass, because those kernels keep their own prologue.
    A span of exactly one row inside a wider body aborts in the launcher, so such shapes are excluded
    below rather than counted — that abort is pre-existing (`G1-result.md` §5).
    """
    subs = 0
    for rows in body_widths(tokens, wave_rows):
        span = resolve_gdn_loop(want_span, rows)
        if span == 1:
            subs += rows
            continue
        spans = (rows + span - 1) // span
        tail = rows - span * (spans - 1)
        if tail == 1:
            return -1  # unreachable shape: the tail span refuses tokens < 2
        subs += spans * (2 if hoisted else 1)
    return G * subs


def planned(tokens: int, wave_rows: int, want_span: int, hoisted: bool) -> int:
    """Mirror of the `stage_clock.cpp` derivation (spans_of / pre_passes_of), independently spelled.

    `want <= 1` is not a span case at all: the bill's `else if (sh.gdn_loop_span > 1)` guard is not taken,
    so the row keeps its `G * tokens` default. Mirroring the guard, not just the formula, is the point."""
    if want_span <= 1:
        return G * tokens
    widths = list(body_widths(tokens, wave_rows))
    full = widths[:-1]
    last = widths[-1]
    span = want_span

    def spans_of(rows):
        return 1 if rows <= 1 else (rows + span - 1) // span

    def pre_of(rows):
        return 0 if rows <= 1 else (rows + span - 1) // span

    subs = sum(spans_of(w) for w in full) + spans_of(last)
    if hoisted:
        subs += sum(pre_of(w) for w in full) + pre_of(last)
    return G * subs


CASES = [
    # (tokens, wave_rows, span, note)
    (46, 512, 512, "the gate prompt: one body, one span"),
    (46, 16, 8, "the gate's many-span shape (bodies 16/16/14)"),
    (4096, 512, 512, "THE A/B SHAPE: 8 bodies, one span each"),
    (1024, 512, 512, "two bodies"),
    (1030, 512, 512, "ragged last body"),
    (513, 512, 512, "the one-row final body (per-token route)"),
    (4096, 1, 512, "wave_rows 1: every body is the per-token form"),
    (4096, 128, 32, "narrow waves, narrow spans"),
    (2048, 256, 250, "span just under a body"),
    (4096, 512, 1, "span 1: the per-token loop over every token of the body"),
]

# `--gdn-norm hoist` with `--gdn-loop 1` is refused at parse time, so this pair is never billed together.
REFUSED = [(4096, 512, 1, True)]

SKIPPED = [
    (46, 16, 13, "1-row tail span: aborts in the launcher, pre-existing"),
]


def main() -> int:
    bad = 0
    for tokens, wr, span, note in CASES:
        for hoisted in (False, True):
            if (tokens, wr, span, hoisted) in REFUSED:
                print(f"ok   tokens={tokens} wave_rows={wr} span={span} hoist=1: refused at parse,"
                      " never billed   [" + note + "]")
                continue
            got, want = launched(tokens, wr, span, hoisted), planned(tokens, wr, span, hoisted)
            if got == -1:
                bad += 1
                print(f"FAIL {tokens}/{wr}/{span} hoist={hoisted}: shape unreachable ({note})")
                continue
            tag = "ok  " if got == want else "FAIL"
            if got != want:
                bad += 1
            print(f"{tag} tokens={tokens:5d} wave_rows={wr:4d} span={span:4d} hoist={int(hoisted)}"
                  f"  launched={got:5d} planned={want:5d}   [{note}]")
    for tokens, wr, span, note in SKIPPED:
        got = launched(tokens, wr, span, True)
        if got != -1:
            bad += 1
            print(f"FAIL {tokens}/{wr}/{span}: expected the unreachable shape, got {got}")
        else:
            print(f"ok   tokens={tokens} wave_rows={wr} span={span} confirmed unreachable: {note}")
    print("RESULT plan_bill_check", "PASS" if bad == 0 else f"FAIL ({bad})")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
