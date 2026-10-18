#!/usr/bin/env python3
"""check_m16_transcription.py — CPU-only witness that `attn_m16.hip.cpp` transcribes the shipped form.

WHY THIS EXISTS AS A SEPARATE FILE. `AGENTS.md` carries the trap twice over: a CPU mirror of a kernel's
FORMULA cannot see what a device implementation actually does (FMA contraction, the device intrinsics,
and — the expensive one, `prototype/gdn_collapse/results.md` §2 — ASSOCIATION), and a hoisted scalar
that made a different product tree moved 93,522 of 131,072 elements on device while leaving the state
bit-exact. The M = 16 re-tile claims bit-identity with the shipped streaming form at `ndiff = 0`, and the
only instrument that can price that claim BEFORE a device run is the source text: if a shipped
expression appears here with its operands in the same order, the bits follow; if it was re-spelled
"equally", this file says so.

So this is not a numerical mirror. It compares the frozen shipped source against the re-tile under an
alias map that renames only what the re-tile legitimately renamed (per-row state is `st[i][0]` here
because GRP rows are live; the shipped block has one row and calls it `m`). Every check names the shipped
line it is comparing against, so a reader can open both files beside each other.

The alias map is the whole trick and the whole risk, so it is deliberately tiny and every entry maps a
name in THIS file to the shipped spelling. Anything not in the map must match character for character.

Runs with no device, no lock, no HIP. `--expect N` asserts this file ran N checks: a check list whose
patterns stopped matching would otherwise report success over an empty list (the
`prototype/gdn_collapse/run.py` trap — a parser that matched zero records printed PASS).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
SHIP = ROOT / "prototype" / "attn_decode" / "attn_decode_kernels.h"
MINE = HERE / "attn_m16.hip.cpp"

# Renames the re-tile is entitled to. Each one exists because the shipped name is singular and this form
# has GRP of them. Anything else has to match exactly.
ALIASES = [
    ("m16_lanes_of", "lanes_of"),
    ("m16_blocks_for_row", "blocks_for_row"),
    ("m16_row_onepass", "one_pass"),
    ("attn_m16_split", "split_for_lanes"),
    ("ATTN_M16_G", "ATTN_TILE_G"),
    ("ATTN_M16_M", "ATTN_TILE_M"),
    ("st[i][0]", "m"),    ("st[i][1]", "l"),
    ("acc[i]", "acc"),
    ("g[i].row", "row"),
    ("g[i].pos", "row_pos"),
    ("g[i].n", "nrow"),
    ("g[i].s_row", "s_row"),
    ("g[i].span", "span"),
    ("g[i].blo", "blo"),
    ("g[i].bhi", "bhi"),
    ("g[i].exists", "exists"),
    ("g[i].active", "active"),
    ("g[i].chunks", "chunks"),
    ("qrow", "row"),   # the parameter name only; it runs last so it cannot rewrite anything else
]


def norm(text: str) -> str:
    """Source collapsed to one line, comments gone, aliases mapped to the shipped spelling."""
    text = re.sub(r"//.*", "", text)
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    text = re.sub(r"\s+", " ", text).strip()
    for a, b in ALIASES:
        text = text.replace(a, b)
    return text


def body_of(src: str, signature: str) -> str:
    """The text of the function whose definition starts with `signature`, brace-balanced."""
    i = src.find(signature)
    if i < 0:
        raise AssertionError(f"no definition starting {signature!r}")
    j = src.index("{", i)
    depth = 0
    for k in range(j, len(src)):
        if src[k] == "{":
            depth += 1
        elif src[k] == "}":
            depth -= 1
            if depth == 0:
                return src[i:k + 1]
    raise AssertionError(f"unbalanced braces after {signature!r}")


def strip_comments(text: str) -> str:
    """For counts: a sentence in the header comment that says `__shfl_down_sync` is not a second
    implementation of the fold, and pretending otherwise would fail a file for its prose."""
    text = re.sub(r"/\*.*?\*/", " ", text, flags=re.S)
    return re.sub(r"//.*", "", text)


def body_only(fn: str) -> str:
    """Just the `{ ... }` of a function definition: the storage-class and the parameter names are this
    file's business (`m16_lanes_of` has to be `__host__` too, because the census walks it on the host),
    the statements inside are the shipped form's."""
    return fn[fn.index("{"):]


class Checks:
    def __init__(self) -> None:
        self.n = 0
        self.bad = 0

    def fail(self, what: str, detail: str) -> None:
        self.n += 1
        self.bad += 1
        print(f"  FAIL {what}\n{detail}")

    def eq(self, what: str, ship_src: str, mine_src: str, why: str = "") -> None:
        """Both sides reduce to one normalized line each and must be identical after the alias map."""
        self.n += 1
        a, b = norm(ship_src), norm(mine_src)
        if a == b:
            print(f"  ok   {what}")
        else:
            self.bad += 1
            print(f"  FAIL {what}\n        shipped: {a}\n        m16   : {b}\n        {why}")

    def has(self, what: str, haystack: str, needle: str, why: str = "") -> None:
        """`needle` must appear in `haystack`: 'this exact expression, this operand order'."""
        self.n += 1
        if norm(needle) in norm(haystack):
            print(f"  ok   {what}")
        else:
            self.bad += 1
            print(f"  FAIL {what}\n        needed: {norm(needle)}\n        {why}")

    def raw(self, what: str, ok: bool, detail: str = "") -> None:
        self.n += 1
        if ok:
            print(f"  ok   {what}")
        else:
            self.bad += 1
            print(f"  FAIL {what}\n{detail}")


def barrier_walk(kern: str) -> tuple[int, list[int]]:
    """(barrier count, positions of the barriers that sit inside an if/else/switch).

    The claim under test is "every quantity bounding a barriered loop is per-block", and the way to break
    it is a barrier inside a conditional. A textual walk is enough: track which keyword opened each brace
    level and ask what encloses each `__syncthreads()`.
    """
    depth, stack, seen, inside = 0, [], [], []
    for m in re.finditer(r"\b(if|for|while|switch|else)\b|__syncthreads\s*\(\s*\)|[{;}]", kern):
        tok = m.group(0)
        if tok in ("if", "for", "while", "switch", "else"):
            stack.append((tok, depth))
        elif tok == "{":
            depth += 1
        elif tok == "}":
            depth -= 1
            while stack and stack[-1][1] >= depth:
                stack.pop()
        elif tok.startswith("__syncthreads"):
            seen.append(m.start())
            if [t for t, _ in stack if t in ("if", "switch", "else")]:
                inside.append(m.start())
    return len(seen), inside


def main(expect: int) -> int:
    ship = SHIP.read_text()
    mine = MINE.read_text()
    C = Checks()

    # ---- 1. the three row rules, transcribed ------------------------------------------------------
    # `attn_m16_split` is allowed to be THINNER than a transcription: it calls the shipped function, which
    # is the strongest form of "one authority". Both accepted spellings are asserted, so a third one — a
    # re-typed `(lanes + MAXSC - 1) / MAXSC` — fails here.
    C.n += 1
    sp = norm(body_only(body_of(mine, "__device__ __host__ inline int attn_m16_split")))
    if sp in ("{ return split_for_lanes(lanes_last); }",
              norm("{ return (lanes_last + MAXSC - 1) / MAXSC; }")):
        print("  ok   attn_m16_split is the shipped split_for_lanes, called not re-typed")
    else:
        C.bad += 1
        print(f"  FAIL attn_m16_split is a second spelling of the split arithmetic: {sp}")
    for what, ship_sig, mine_sig, why in [
        ("lanes_of body", "__device__ __forceinline__ int lanes_of",
         "__device__ __host__ inline int m16_lanes_of",
         "the causal mask INCLUDING the two injections it carries: a re-tile that dropped the "
         "NEG_READPAST / NEG_DROPCUR arms would make those injection tests meaningless"),
        ("blocks_for_row body", "__device__ __forceinline__ int blocks_for_row",
         "__device__ __host__ inline int m16_blocks_for_row",
         "one-pass is `full && lanes <= MAXSC` in the shipped form and must stay that here, because "
         "one-pass is the route this form refuses"),
    ]:
        C.eq(what, body_only(body_of(ship, ship_sig)), body_only(body_of(mine, mine_sig)), why)

    # ---- 2. the slice geometry: the shipped line, then this form's line ---------------------------
    C.has("span = (nrow + s_row - 1) / s_row", mine, "(g.n + g.s_row - 1) / g.s_row",
          "shipped: `const int span = (nrow + s_row - 1) / s_row;`")
    C.has("blo = blockIdx.y * span", mine, "const int blo = g.span * y;",
          "shipped: `const int blo = blockIdx.y * span;` — the slice is anchored at the LANE BLOCK, so "
          "the lane block index is the multiplier, not an offset")
    C.has("bhi = min(nrow, blo + span)", mine, "std::min(g.n, blo + g.span)",
          "shipped: `const int bhi = std::min(nrow, blo + span);` — spelled as a `std::min` here too, so "
          "the transcription check compares two lines rather than two opinions about a clamp")
    C.has("chunks = ceil(ns / tile)", mine, "(hi - blo + tile - 1) / tile",
          "shipped: `for (int base = blo; base < bhi; base += tile)`. Chunking at the same boundary is "
          "what makes `expf(m - mn)` the same number (§07.3 of that file).")

    # ---- 3. the per-chunk algebra, one line at a time ---------------------------------------------
    for ship_line, mine_line, why in [
        ("float mt = -1e30f;", "float mt = -1e30f;", "the sentinel is part of the max"),
        ("const float mn = fmaxf(m, mt);", "const float mn = fmaxf(st[i][0], mt);",
         "the running max enters before the chunk max, in that order"),
        ("const float rescale = expf(m - mn);", "const float rescale = expf(st[i][0] - mn);",
         "the non-associative step §07.3 says is part of the number"),
        ("l = l * rescale + ls;", "st[i][1] = st[i][1] * rescale + ls;",
         "rescale BEFORE the chunk sum is added, and in that association"),
        ("for (int u = 0; u < 8; ++u) acc[u] *= rescale;", "for (int u = 0; u < 8; ++u) acc[i][u] *= rescale;",
         "the whole accumulator is rescaled before the chunk's PV adds into it"),
        ("m = mn;", "st[i][0] = mn;", "the state advances at the END of the chunk, after PV"),
        ("dot += bf(k[t]) * q[t];", "for (int t = 0; t < 8; ++t) q[t] = bf(qq[t]);",
         "the query is decoded to F32 once per (row, head) in the shipped form; this form re-reads it per "
         "round from the same bytes — same values, more traffic, which is the global-A cost being priced"),
        ("if (lane == 0) s_sc[h][rr] = fb(bf(fb(dot)) * SCALE);",
         "if (lane == 0) s_sc[sp * M16_L + slot] = fb(bf(fb(dot)) * SCALE);",
         "one BF16 round then the scale. A hoisted `* SCALE` into the query would be a different product "
         "tree — the gdn_collapse q-tilde trap, device-measured at 93,522 moved elements"),
        ("float* pp = a.part + (((size_t)row * N_Q + head) * a.split + blockIdx.y) * (DH + 2);",
         "float* pp = a.part + (((size_t)g[i].row * N_Q + head) * a.split + y) * (DH + 2);",
         "keyed by the FULL head, strided by the LAUNCH split. §20.1 of "
         "prototype/attn_decode/results.md: writer and reader must share one address expression, which is "
         "why this is compared rather than paraphrased"),
    ]:
        C.has(f"streaming step: {norm(ship_line)}", mine, mine_line, why)
    C.has("e is sized TILE here as well", mine, "float e[TILE], ls = 0.f;",
          "the shipped form sizes it `e[TILE]` for the same reason: `nt <= tile <= TILE`. Keeping the "
          "same size is what kept four unrolled row bodies inside the register file")
    C.has("tile is clamped to [1, TILE]", mine,
          "const int tile = (a.tile > 0 && a.tile <= TILE) ? a.tile : TILE;",
          "`e[TILE]` is only in bounds because of this clamp; the shipped form says so in its own comment")

    # ---- 4. the folded dot and fold are CALLED, not re-spelled ------------------------------------
    C.raw("vslim_dot / vslim_fold / vslim_permute_addr are called from the shipped slim header",
          all(x in mine for x in ("vslim_dot<8>(", "vslim_dot<4>(",
                                  "vslim_fold<VSLIM_DOT | VSLIM_FOLD>", "vslim_permute_addr(")),
          "the dot and the fold must be `prototype/attn_valu_slim/attn_valu_slim_kernels.h`'s, reached "
          "through their names. Re-spelling them is a second implementation of the reduction order, and "
          "the handoff's constraint was explicit: the shipped VALU dot and warp fold, unchanged.")
    code = strip_comments(mine)
    C.raw("exactly one __shfl_down_sync in the code, and it is the bit-diff helper's",
          code.count("__shfl_down_sync") == 1 and "warp_sum(" not in code,
          f"{code.count('__shfl_down_sync')} occurrences. The butterfly fold lives in `vslim_fold`; a "
          "second one here means the fold was copied, and the diff kernel's own butterfly is the only "
          "other one this file should contain.")

    # ---- 5. barriers: block-uniform, and never after a per-thread early exit ----------------------
    kern = body_of(mine, "__global__ void attn_m16_wave")
    nbar, inside = barrier_walk(kern)
    C.raw(f"{nbar} barriers, none inside an `if`/`else`/`switch`", nbar >= 4 and not inside,
          f"{len(inside)} of them inside a conditional at offsets {inside}. A barrier a subset of the "
          "block can skip is a hang on this part, not a wrong answer, and the re-tile's central argument "
          "is that its barriers are block-uniform.")
    first_barrier = kern.find("__syncthreads")
    guards = {name: kern.find(name) for name in
              ("if (onepass)", "if (nact == 0)", "if (widest > M16_L)")}
    C.raw("the three per-group refusals return or continue BEFORE this group's first barrier",
          first_barrier > 0 and all(v > 0 and v < first_barrier for v in guards.values()),
          f"first barrier at {first_barrier}, guards at {guards}. `__syncthreads()` after a `return` that "
          "only some threads took deadlocks the queue (`attn_m2.hip.cpp`'s diff-kernel comment).")
    guard_src = strip_comments(kern[kern.index("for (int gi"):first_barrier])
    C.raw("the quantities deciding those early exits mention no thread or lane",
          "threadIdx" not in guard_src and "lane" not in guard_src,
          "a pre-barrier guard that reads a lane index makes the early exit per-thread, which is the "
          "divergent-barrier failure above in disguise. Checked from the group loop to its first barrier: "
          "the block prologue before the loop is allowed per-thread values, because nothing there decides "
          "an early exit.")

    print(f"\n{'FAIL' if C.bad else 'PASS'}: check_m16_transcription — {C.n - C.bad}/{C.n} checks against "
          f"{SHIP.relative_to(ROOT)}")
    if expect and C.n != expect:
        print(f"FAIL: expected {expect} checks, ran {C.n} — a check list that shrinks silently cannot "
              "fail (prototype/gdn_collapse/run.py)")
        return 1
    return 1 if C.bad else 0


if __name__ == "__main__":
    ns = argparse.ArgumentParser().parse_known_args()[0]
    argv = sys.argv[1:]
    want = 0
    for i, a in enumerate(argv):
        if a == "--expect":
            want = int(argv[i + 1])
    sys.exit(main(want))
