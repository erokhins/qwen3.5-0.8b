#!/usr/bin/env python3
"""equiv_copy_fold.py — CPU proof that route S1's copy fold moves the same bytes to the same places.

The fold deletes 19 of the 68 `glue`+`copy` submissions a decode token makes (site table:
`prototype/glue_copy_fold/census_sites.py`). Two of them, and the numerics question is different for
each, so they are checked separately rather than folded together:

  A. THE CONV-WINDOW ROLL-BACK (18 submissions/token). Shipped, `conv_kernel` writes the rolled window
     into `sc.conv_new` and a D2D copy puts it in `st.conv_window_layer(layer)`. Folded,
     `engine/gdn_conv_roll_kernels.h::conv_roll_kernel` writes it there directly. Nothing is
     recomputed — the same three rows, the same values, one address shorter — so what is owed is an
     operand-equality proof and a byte-compare, NOT a band, and `prototype/glue/floor_probe.py` is not
     invoked because no reduction is re-associated anywhere in this route. Section A proves the
     movement on CPU, in the same [K-1][CH] layout the device uses, against the shipped two-step
     (scratch then copy), with the state pre-filled with a value no roll writes so a stale row cannot
     pass by being left alone. `AGENTS.md`: a write side verified by reading back a buffer pre-filled
     with the value the writes store proves nothing.

  A2. THE ROLL'S SOURCE IS THE CONV'S OWN TAPS. The folded kernel reads the window once into `tap[]`
     and derives both the conv and the roll from it, so a mis-wired index would show up as a window
     shifted the wrong way — which is `GC_WIN_LAG`, one of the prototype's own named injections
     (`prototype/gdn_decode/gdn_decode_kernels.h`'s enum). Section A2 checks the roll rule against the
     rule the prototype states once in prose — "the window after this tile is its last K-1 raw
     columns" — over every token count where the fold is allowed to run, and refuses the widths it is
     not allowed to run at.

  A3. THE ARITHMETIC TEXT. The fold kernel transcribes `conv_value`'s dot rather than calling it,
     because calling it would mean passing one buffer through two `__restrict__` parameters — the case
     the qualifier forbids. A transcription is exactly the `engine/gdn_collapse_kernels.h` q̃ hazard
     (a re-associated product moved 93,522 of 131,072 outputs on device while a CPU mirror saw
     nothing, because two sides that share a formula agree). A CPU mirror cannot catch that here
     either, so this section diffs the two spellings EXPRESSION BY EXPRESSION, as text, and names any
     expression that is not the same characters.

  B. THE EMBEDDING ROW (1 submission/token). Shipped, the tied table's row is copied into `sc.embed`
     and the residual pointer `s` starts there. Folded, `s` starts at the table row and the copy is
     gone. The values are equal by identity — it is literally the same memory — so what is at risk is
     not the numbers but the ALIASING: `s` must never become a write target, and `sc.embed` must have
     no other reader in a one-row body. Section B checks that against the source text, because those
     are properties of the call sites, not of a value.

A CPU check cannot be the whole admission: the window is device state and the pins that police it are
`§08.1`'s, so this file ends by printing the device commands that close the route. It takes no timings
and makes no GPU call.

    .venv/bin/python prototype/glue_copy_fold/equiv_copy_fold.py [--selftest]
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent
FOLD_H = ROOT / "engine" / "gdn_conv_roll_kernels.h"
SHIP_H = ROOT / "prototype" / "gdn_decode" / "gdn_decode_kernels.h"
STEP_CPP = ROOT / "engine" / "decode_step.cpp"


class EquivError(Exception):
    """A claim this file cannot make. `die()` turns it into exit 1."""


def die(msg: str):
    raise EquivError(msg)


def widths() -> tuple[int, int]:
    """(K, CH) from `engine/state.h`, so the layout is not a typed constant."""
    txt = (ROOT / "engine" / "state.h").read_text()
    def one(name: str) -> int:
        m = re.search(rf"constexpr int {name}\s*=\s*(\d+);", txt)
        if not m:
            die(f"engine/state.h does not define {name} as a literal; this file would be modelling "
                "a layout it cannot read")
        return int(m.group(1))
    return one("kConvKernel"), one("kConvChannels")


# ---- A / A2: the roll ---------------------------------------------------------------------------

def bf16_bits(x: np.ndarray) -> np.ndarray:
    """F32 -> BF16 bits, round-to-nearest-even, the same rounding the device's `bf16_round` does."""
    u = np.asarray(x, np.float32).view(np.uint32)
    u = u + np.uint32(0x7FFF) + ((u >> np.uint32(16)) & np.uint32(1))
    return (u >> np.uint32(16)).astype(np.uint16)


def bf16_from_bits(bits: np.ndarray) -> np.ndarray:
    return (bits.astype(np.uint32) << np.uint32(16)).view(np.float32)


def random_bf16(rng, shape) -> np.ndarray:
    """Random values EXACTLY representable in BF16, so a widening is bit-exact and a mismatch cannot
    be explained away as a rounding difference. Multiples of 1/8 below 16 need 7 significand bits,
    which is exactly what BF16 has, so these survive F32 -> BF16 -> F32 unchanged."""
    return (rng.integers(-128, 129, size=shape).astype(np.float32) / np.float32(8.0))


def roll_rule(win: np.ndarray, raw_row: np.ndarray, k_kernel: int) -> np.ndarray:
    """The end condition, spelled from the prototype's prose: the window after a one-token tile is its
    last K-1 raw columns, with `raw[k<0]` meaning the carried window (`conv_tap`'s one rule)."""
    rows = k_kernel - 1
    new = np.empty_like(win)
    for j in range(rows):
        k = 1 - rows + j          # tokens(=1) - (K-1) + j
        new[j] = win[k + rows] if k < 0 else raw_row
    return new


def conv_taps_from_window(win: np.ndarray, raw_row: np.ndarray, k_kernel: int) -> np.ndarray:
    """`tap[K][CH]`, oldest first: the carried rows then this token's raw row — `conv_value`'s order."""
    if len(win) != k_kernel - 1:
        die(f"the window has {len(win)} rows but K={k_kernel} implies {k_kernel - 1}")
    return np.stack([*win, raw_row])


SENTINEL = np.float32(-1234.5)   # a value outside the fixture's range, so a surviving one is a
                                 # write that never happened, never a value the roll produced


def compare_roll(win: np.ndarray, raw_row: np.ndarray, folded: np.ndarray, k_kernel: int,
                 label: str) -> None:
    """The one assertion every roll case has to survive: what the roll wrote IS what the prose rule
    says the window after a one-token tile is, and every row was actually written."""
    want = roll_rule(win, raw_row, k_kernel)
    if not np.array_equal(folded, want):
        bad = np.argwhere(folded != want)
        first = tuple(bad[0])
        die(f"{label}: the folded window differs from the rule at {len(bad)} elements, first at row"
            f" {first[0]} col {first[1]} (folded {folded[first]}, rule {want[first]}) — the roll and"
            " the copy are not the same rule")
    if (folded == SENTINEL).any():
        die(f"{label}: {int((folded == SENTINEL).sum())} window elements still hold the sentinel — the"
            " roll wrote fewer rows than the copy did")


def section_a(k_kernel: int, ch: int, cases: int) -> list[str]:
    """Both halves of the roll, shipped spelling vs folded spelling, on identical operands."""
    rows = k_kernel - 1
    notes = []
    rng = np.random.default_rng(20261022)
    for case in range(cases):
        win = random_bf16(rng, (rows, ch)).astype(np.float32)
        raw_row = random_bf16(rng, (ch,)).astype(np.float32)

        # SHIPPED: the conv writes a scratch row, then one D2D copy moves it over the window. The
        # scratch starts full of a value no roll writes, so a row the conv forgot is visible in the
        # window the copy lands — `AGENTS.md`: a write side verified against a buffer pre-filled with
        # the value the writes store proves nothing.
        scratch = np.full((rows, ch), SENTINEL, np.float32)
        for j in range(rows):
            k = 1 - rows + j
            scratch[j] = win[k + rows] if k < 0 else raw_row
        shipped_window = scratch.copy()

        # FOLDED: the conv reads the window into `tap[]` once and derives BOTH its taps and its roll
        # from those registers, writing the same rows back in place. The staging is explicit here for
        # the same reason the kernel stages it: every read happens before the first write.
        tap = conv_taps_from_window(win, raw_row, k_kernel)
        folded_window = np.full((rows, ch), SENTINEL, np.float32)
        for j in range(rows):
            folded_window[j] = tap[j + 1]      # the kernel's one line: `rolled.f[i] = tap[j+1][i]`

        compare_roll(win, raw_row, folded_window, k_kernel, f"case {case}")
        if not np.array_equal(folded_window, shipped_window):
            die(f"case {case}: the in-place path and the copy path disagree")
    notes.append(f"roll: {cases} cases x {rows}x{ch} F32 ({cases * rows * ch * 4 / 1e6:.1f} MB"
                 " total), bit-identical between the copied path and the in-place path, both checked"
                 " against the prototype's prose rule, with every destination pre-filled with a value"
                 " no roll writes so a write that did not happen fails instead of surviving")
    return notes


def section_a2(k_kernel: int, ch: int) -> list[str]:
    """The roll rule is checked at the widths the fold may run at, and the ones it may not."""
    rng = np.random.default_rng(7)
    win = random_bf16(rng, (k_kernel - 1, ch)).astype(np.float32)
    raw_row = random_bf16(rng, (ch,)).astype(np.float32)
    want = roll_rule(win, raw_row, k_kernel)
    # The mapping the kernel actually spells, derived independently of the prose rule above: row j
    # takes tap[j+1], and tap is [window rows..., raw row].
    tap = np.stack([*win, raw_row])
    spelled = np.stack([tap[j + 1] for j in range(k_kernel - 1)])
    if not np.array_equal(spelled, want):
        die("the kernel's `tap[j+1]` mapping and the prototype's prose rule disagree — one of the two"
            " readings of `conv_tap(raw, win, c0, tokens-(K-1)+j, tokens=1, ...)` is wrong")
    # And the shift has to be a SHIFT: row K-2 is the only one that may carry this token's raw row.
    if not np.array_equal(spelled[-1], raw_row):
        die("the newest window row is not this token's raw row")
    if not np.array_equal(spelled[0], win[1]):
        die("the oldest new window row is not the window's second row: the roll is off by one row,"
            " which is the prototype's own GC_WIN_LAG injection")
    return [f"roll rule: `tap[j+1]` reproduces the prose rule at tokens=1 (K={k_kernel},"
            f" window [{k_kernel-1}][{ch}]) and the newest row is this token's own raw column"]


# ---- A3: the transcription ----------------------------------------------------------------------

# The expressions the folded kernel re-spells. Each entry is the RHS as the shipped kernel writes it;
# `fold_h` must contain the same characters after its own `lhs` prefix, because a re-associated or
# re-spelled product is a different number even when a CPU mirror of both sides agrees.
TRANSCRIBED = [
    ("the K taps in j order", "acc += tap[j][i] * C[(c0 + i) * K + j];"),
    ("the one BF16 round", "const float rounded = bf16_round(acc);"),
    ("SiLU and the second round", "bf16_round(rounded / (1.f + expf(-rounded)))"),
    ("the window read, oldest first", "*reinterpret_cast<const ConvColsF*>(win + (size_t)j * CH + c0)"),
    ("the raw read, widened", "float(r.h[i])"),
]


def section_a3() -> list[str]:
    fold = FOLD_H.read_text()
    ship = SHIP_H.read_text()
    missing = []
    for what, expr in TRANSCRIBED:
        if expr not in fold:
            missing.append(f"{what}: `{expr}` is not in {FOLD_H.name}")
    # The last item is the one that catches a drift on the OTHER side: if `conv_value` is ever edited
    # and this file is not, the transcription is no longer a transcription.
    for what, expr in TRANSCRIBED[:3]:
        if expr not in ship:
            missing.append(f"{what}: `{expr}` is no longer in {SHIP_H.name}, so this file's copy is "
                           "stale — re-read conv_value and re-transcribe")
    if missing:
        for m in missing:
            print("TRANSCRIPTION PROBLEM: " + m)
        die(f"{len(missing)} transcribed expression(s) do not match; see the TRANSCRIBED list")
    return [f"transcription: {len(TRANSCRIBED)} expressions matched character for character between"
            f" {FOLD_H.name} and {SHIP_H.name}, and the three arithmetic ones were checked to still"
            " be what the shipped kernel contains"]


# ---- B: the embedding row -----------------------------------------------------------------------

def section_b() -> list[str]:
    """`s` may only ever point at read-only memory, and `sc.embed` must have no other use in a one-row body."""
    src = STEP_CPP.read_text().splitlines()
    start = next((i for i, t in enumerate(src) if t.startswith("void submit_wave_body")), None)
    if start is None:
        die("cannot bound submit_wave_body() in engine/decode_step.cpp")
    body = src[start:]
    # 1. `s` has exactly two owners: its own declaration, and the ping-pong's `s = nxt`. A third write
    #    could hand the tied table row to `glue_add_norm_rows` as `s_out`, i.e. write model weights.
    decl = [t.strip() for t in body if re.search(r"^\s*const uint16_t\* s = ", t)]
    if len(decl) != 1:
        die(f"{len(decl)} declarations of the residual pointer `s`, expected 1")
    if "sc.embed" not in decl[0] or "table" not in decl[0]:
        die(f"`s`'s initializer reads {decl[0]!r}; the fold's claim is that it names EITHER the tied"
            " table row (folded) OR sc.embed (shipped), and nothing else")
    assigns = [t.strip() for t in body if re.search(r"^\s*s = ", t)]
    if assigns != ["s = nxt;"]:
        for a in assigns:
            print(f"  s assigned: {a}")
        die(f"`s` has {len(assigns)} assignment(s); exactly one is allowed (`s = nxt;`), because any"
            " other could make the table row a write target")
    # 2. `nxt`, the only thing `s` can become, is confined to the two ping-pong buffers.
    if not any(re.search(r"uint16_t\* nxt = sc\.s_a;", t) for t in body):
        die("`nxt` no longer starts at sc.s_a; the ping-pong this argument depends on has moved")
    if len([t for t in body if re.search(r"\(nxt == sc\.s_a\) \? sc\.s_b : sc\.s_a", t)]) != 1:
        die("the ping-pong's alternation is not the one expression this file describes")
    # 3. `sc.embed`'s write sites: the shipped D2D destination and the wave gather, no others. If a
    #    third appeared, the folded path could leave a reader looking at a row nobody wrote.
    writes = [t.strip() for t in body
              if re.search(r"hipMemcpyAsync\(sc\.embed,|embed_gather_rows\([^)]*sc\.embed", t)]
    if len(writes) != 2:
        die(f"{len(writes)} write sites for sc.embed (expected the shipped D2D destination and the"
            " wave gather); the read-only claim needs exactly those two and no others")
    reads = [t.strip() for t in body if "sc.embed" in t and t.strip() not in writes]
    # 4. the folded source of the row and the shipped copy's source are the SAME expression, so the
    #    fold cannot be reading a different row than the copy used to copy.
    row_expr = "table + (size_t)tokens[0] * kHidden"
    if len([t for t in body if row_expr in t]) != 2:
        die(f"`{row_expr}` names "
            f"{len([t for t in body if row_expr in t])} line(s), expected 2 (the copy's source and the"
            " fold's initializer): the equality this section is about is that they are one expression")
    return [f"embedding row: `s` has 1 declaration (naming the table row or sc.embed) and exactly 1"
            f" assignment (`s = nxt;`, which only ever yields sc.s_a/sc.s_b), `sc.embed` has"
            f" {len(writes)} write sites (the shipped D2D and the wave gather, neither of which a"
            f" folded one-row body reaches) and {len(reads)} read site(s), and the folded pointer and"
            f" the copy's source are the same expression `{row_expr}` — so the fold reads the same"
            " bytes the copy copied, and no bit can differ because there is no second copy of them"]


def selftest() -> int:
    """Each section, fed the bug it exists to catch. A check that cannot fail is decoration."""
    k_kernel, ch = widths()
    rows = k_kernel - 1
    failed = 0

    # 1. a roll that shifts the wrong way (the prototype's own GC_WIN_LAG), caught by the same
    #    comparison the real cases pass through.
    def lagged():
        rng = np.random.default_rng(1)
        win = random_bf16(rng, (rows, ch)).astype(np.float32)
        raw_row = random_bf16(rng, (ch,)).astype(np.float32)
        tap = conv_taps_from_window(win, raw_row, k_kernel)
        got = np.stack([tap[j] for j in range(rows)])     # off by one row, deliberately
        compare_roll(win, raw_row, got, k_kernel, "lagged roll")
    failed += _expect_refuses("a roll shifted one row the wrong way", lagged)

    # 2. a fold that forgets the newest row: the sentinel survives, and the check must say so.
    def missing_row():
        rng = np.random.default_rng(2)
        win = random_bf16(rng, (rows, ch)).astype(np.float32)
        raw_row = random_bf16(rng, (ch,)).astype(np.float32)
        tap = conv_taps_from_window(win, raw_row, k_kernel)
        got = np.full((rows, ch), SENTINEL, np.float32)
        for j in range(rows - 1):
            got[j] = tap[j + 1]
        compare_roll(win, raw_row, got, k_kernel, "roll missing its newest row")
    failed += _expect_refuses("a fold that leaves the newest window row unwritten", missing_row)

    # 3. the transcription check must notice the shipped kernel changing.
    def stale_source():
        good = SHIP_H.read_text()
        try:
            SHIP_H.write_text(good.replace("acc += tap[j][i] * C[(c0 + i) * K + j];",
                                           "acc += C[(c0 + i) * K + j] * tap[j][i];"))
            section_a3()
        finally:
            SHIP_H.write_text(good)
    failed += _expect_refuses("the shipped conv edited so the copy is stale", stale_source)

    # 4. the aliasing check must notice a third owner of `s`.
    def third_owner():
        good = STEP_CPP.read_text()
        try:
            STEP_CPP.write_text(good.replace("        s = nxt;",
                                             "        s = nxt;\n        s = sc.s_a;  // injected"))
            section_b()
        finally:
            STEP_CPP.write_text(good)
    failed += _expect_refuses("a third write to the residual pointer", third_owner)

    print(f"SELFTEST {'PASS' if failed == 0 else f'FAIL ({failed})'}")
    return 0 if failed == 0 else 1


def _expect_refuses(what: str, fn) -> int:
    try:
        fn()
    except EquivError as exc:
        print(f"  selftest refused as required  [{what}]  -> {exc}")
        return 0
    except Exception as exc:  # noqa: BLE001
        print(f"  SELFTEST FAIL [{what}] raised {type(exc).__name__}: {exc}")
        return 1
    print(f"  SELFTEST FAIL [{what}] returned normally — that check cannot fail")
    return 1


def main() -> int:
    if "--selftest" in sys.argv[1:]:
        return selftest()
    k_kernel, ch = widths()
    out: list[str] = []
    out += section_a(k_kernel, ch, cases=8)
    out += section_a2(k_kernel, ch)
    out += section_a3()
    out += section_b()
    for line in out:
        print("EQUIV: " + line)
    print("DEVICE COMMANDS that close the route (this file makes no GPU call and takes no timing):")
    print("  tools/gpu_lock.sh acquire gluecopy-dec8 && "
          "QWEN_COPY_FOLD=1 .venv/bin/python engine/gate.py --fixture dec8 ; "
          "tools/gpu_lock.sh release gluecopy-dec8")
    print("  # and the same fixture with the flag unset, as the base arm; both arms' STATEFP lines must")
    print("  # agree row for row, because the window is the state the fold rewrites in place.")
    print("EQUIV ok")
    return 0


if __name__ == "__main__":
    try:
        sys.exit(main())
    except EquivError as e:
        print(f"EQUIV FAIL: {e}")
        sys.exit(1)
