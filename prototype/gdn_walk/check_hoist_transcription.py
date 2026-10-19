#!/usr/bin/env python3
"""Does the hoisted walk still contain the SHIPPING state-walk arithmetic, line for line?

WHY THIS EXISTS. Bet G1 (`prototype/gdn_walk/BET.md` §3) admits on BIT-IDENTITY, and the argument is a
source-text argument: the loop in `rec_loop_normed_kernel` must be `rec_loop_kernel`'s loop with the norm
prologue deleted — same operands, same order, same `#pragma unroll 4` — because bit-identity is owed to
the shipped SOURCE TEXT and not to the algebra. `prototype/gdn_collapse/results.md` §2, quoted in
`engine/gdn_rec_fused_kernels.h`, is the case that proved a CPU mirror cannot see this class of change: a
hoisted scalar that was algebraically identical moved 93,522 of 131,072 `o` elements on device, because
what a compiler contracts into an FMA is a property of the expression it is shown. So the cheap pre-device
check is whether the expression text moved, and this file is that check.

WHAT IS COMPARED.

  WALK     `rec_loop_kernel`'s per-token region from `float* Sh = S +` through `o[h * D + j] = acc;`
           versus `rec_loop_normed_kernel`'s same region. They must be identical with whitespace collapsed
           and comments stripped — and the `#pragma unroll 4` lines ARE compared, because a dropped or
           doubled pragma is a different listing (the two walks have been bill-matched since §18/D46 for
           measured reasons). The region starts at `Sh` rather than at the prologue because the prologue
           is exactly what the bet MOVES: the walk's `gam`/`bet` arrive as two loads instead of as an
           `expf` and a load, and that difference is checked separately below rather than being allowed to
           blur this one.
  PROLOGUE the shipped prologue's expressions — `kq`, `qq`, the `block_sum_pair_f32(...)` call, `kn`,
           `qn`, `gam`, `bet` — must appear in `gdn_norm_hoist_kernel` verbatim. The STORES differ by
           design (`kn[h * D + j] = …` where the shipped form wrote `ks[j] = …`), which is why this half
           compares expressions rather than lines.

A CHECK THAT CANNOT FAIL IS DECORATION (`AGENTS.md`, stated several times; this directory's sibling
`norm_tree_cpu.py` carries its own must-catch set for the same reason), so `--selftest` applies mutations
that are each a real numerics change — the `q̃` hoist of §2, a reordered add, a swapped `ks`/`qs`, a
dropped pragma, a shifted row index, a re-associated partial fold — and requires every one to be reported.
It also refuses short extractions, because an empty comparison passes vacuously (the
`prototype/gdn_collapse/run.py` parser failure, one directory over).

Usage:
    prototype/gdn_walk/check_hoist_transcription.py            # the real diff
    prototype/gdn_walk/check_hoist_transcription.py --selftest # the must-catch sets
Exit 0 = identical (and, under --selftest, every mutation caught).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SHIPPED = ROOT / "engine/gdn_rec_loop_kernels.h"
HOISTED = ROOT / "engine/gdn_rec_hoist_kernels.h"

WALK_START = "float* Sh = S +"
WALK_END = "o[h * D + j] = acc;"
MIN_WALK_LINES = 20        # the shipped walk is 23 lines (loop B puts three statements on one line);
                           # anything shorter is an extractor failure, not a match

# The prologue's expressions. Each is a substring that must appear, whitespace-collapsed, in BOTH kernels:
# in the shipped one because this table claims to describe it (a table nobody re-checks rots), and in the
# pre-pass because that is the admission.
PROLOGUE_EXPRS = [
    ("kq/qq declared off the same two rows",
     "float kq = k[h * D + j], qq = q[h * D + j];"),
    ("the fused pair reduction",
     "block_sum_pair_f32(kq * kq, qq * qq, redk, redq, &sk, &sq);"),
    ("kn", "kq * rsqrtf(sk + 1e-6f)"),
    ("qn", "qq * rsqrtf(sq + 1e-6f) * Q_SCALE"),
    ("gam", "expf(g[h])"),
    ("bet", "beta[h]"),
]


def kernel_body(path: Path, kernel: str) -> str:
    """The text of one `__global__ void <kernel>(…)` definition, brace-balanced."""
    text = path.read_text()
    m = re.search(r"__global__\s+void\s+" + kernel + r"\s*\(", text)
    if not m:
        raise SystemExit(f"EXTRACT FAIL: no kernel named `{kernel}` in {path}")
    i = text.index("{", m.end())
    depth = 0
    for j in range(i, len(text)):
        if text[j] == "{":
            depth += 1
        elif text[j] == "}":
            depth -= 1
            if depth == 0:
                return text[i:j + 1]
    raise SystemExit(f"EXTRACT FAIL: unbalanced braces in `{kernel}`")


def code_lines(body: str, start: str, end: str) -> list:
    """The region from the line containing `start` through the line containing `end`, normalised.

    Comments are stripped (a comment cannot move a value); blank lines are dropped; whitespace runs are
    collapsed so the looped body's extra indentation is not a difference. `#pragma` lines are KEPT: they
    are instructions to the compiler, not prose.
    """
    out, seen = [], False
    for raw in body.splitlines():
        line = raw.split("//", 1)[0].strip()
        if not line:
            continue
        if not seen:
            if line.startswith(start):
                seen = True
            else:
                continue
        out.append(" ".join(line.split()))
        if line.startswith(end):
            break
    if not seen:
        raise SystemExit(f"EXTRACT FAIL: no line starting `{start}` in the kernel body")
    if not out[-1].startswith(end):
        raise SystemExit(f"EXTRACT FAIL: no line starting `{end}` after the start line")
    if len(out) < MIN_WALK_LINES:
        raise SystemExit(f"EXTRACT FAIL: only {len(out)} lines extracted, minimum {MIN_WALK_LINES}"
                         " — a short extraction is a parse failure, not a match")
    return out


def prologue_present(body: str) -> dict:
    """Which prologue expressions appear in `body`, verbatim and TERMINATED.

    A substring test with a terminator requirement, deliberately. The two forms STORE to different places
    (`ks[j] = kn` there, `kn[h * D + j] = …` here) and the pre-pass names its output pointers `kn`/`qn`
    too, so an extractor that tried to line up "the kn line" would match a pointer declaration instead of
    the arithmetic. What is owed is that the EXPRESSION is present, character for character, and that
    nothing has been appended to it — a bare substring match would still pass on
    `expf(g[h]) * 1.0f`, which is the §2 failure shape exactly, so the match must be followed by `;`, `,`,
    `)` or the end of the body.
    """
    flat = " ".join(body.split())
    # `rstrip(";")` because two of the table entries already carry their terminator; the class after the
    # expression is what proves nothing was appended to it.
    return {name: bool(re.search(re.escape(want.rstrip(";")) + r"\s*[;,)]", flat))
            for name, want in PROLOGUE_EXPRS}


def diff(a: list, b: list) -> int:
    """The first differing line index, or -1 when the two lists are equal."""
    for i, (x, y) in enumerate(zip(a, b)):
        if x != y:
            return i
    return -1 if len(a) == len(b) else min(len(a), len(b))


def walk_of(path: Path, kernel: str) -> list:
    return code_lines(kernel_body(path, kernel), WALK_START, WALK_END)


WALK_MUTATIONS = [
    ("qn's acc line re-associated (the §2 `q̃` shape)",
     lambda ls: ["acc += (qs[i + 0] * s0) * 1.0f;" if "acc += qs[i + 0] * s0;" in l else l
                 for l in ls]),
    ("a reordered add inside the update",
     lambda ls: [l.replace("s0 += ks[i + 0] * dv", "s0 += dv * ks[i + 0]") for l in ls]),
    ("ks/qs swapped in loop A",
     lambda ls: [l.replace("* ks[i + 0]", "* qs[i + 0]") for l in ls]),
    ("the unroll pragma dropped",
     lambda ls: [l for l in ls if l != "#pragma unroll 4"]),
    ("the unroll factor moved to 8",
     lambda ls: [l.replace("#pragma unroll 4", "#pragma unroll 8") for l in ls]),
    ("a shifted state row index",
     lambda ls: [l.replace("Sh[(i + 0) * D + j]", "Sh[(i + 0) * D + j + 1]") for l in ls]),
    ("loop A's first add re-associated",
     lambda ls: [l.replace("r += Sh[(i + 1) * D + j] * ks[i + 1]",
                           "r = r + (Sh[(i + 1) * D + j] * ks[i + 1] + 0.f)") for l in ls]),
    ("the store padded with a no-op add",
     lambda ls: [l.replace("o[h * D + j] = acc;", "o[h * D + j] = acc + 0.f;") for l in ls]),
]

PROLOGUE_MUTATIONS = [
    # The exact §2 failure: algebraically the same product, different tree, 93,522 elements moved.
    ("qn's product tree re-associated (the §2 `q̃` shape)",
     lambda b: b.replace("qq * rsqrtf(sq + 1e-6f) * Q_SCALE", "qq * (rsqrtf(sq + 1e-6f) * Q_SCALE)")),
    ("the epsilon moved outside the rsqrtf",
     lambda b: b.replace("rsqrtf(sk + 1e-6f)", "rsqrtf(sk) + 1e-6f")),
    ("the pair reduction split into two single-row sums",
     lambda b: b.replace("block_sum_pair_f32(kq * kq, qq * qq, redk, redq, &sk, &sq);",
                         "sk = block_sum(kq * kq, redk); sq = block_sum(qq * qq, redq);")),
    ("gam arrives pre-scaled",
     lambda b: b.replace("expf(g[h])", "expf(g[h]) * 1.0f")),
    ("the read is off by one column",
     lambda b: b.replace("float kq = k[h * D + j], qq = q[h * D + j];",
                         "float kq = k[h * D + j + 1], qq = q[h * D + j];")),
]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()

    if args.selftest:
        base = walk_of(SHIPPED, "rec_loop_kernel")
        print(f"MUST-CATCH on the walk ({len(base)} lines of shipped text)")
        ok = True
        for name, mut in WALK_MUTATIONS:
            i = diff(base, mut(list(base)))
            print(f"  {'caught  ' if i >= 0 else 'INVISIBLE'} {name}"
                  + (f" (first diff at line {i + 1})" if i >= 0 else ""))
            if i < 0:
                ok = False
        hoist_body = kernel_body(HOISTED, "gdn_norm_hoist_kernel")
        clean = prologue_present(hoist_body)
        missing = [k for k, v in clean.items() if not v]
        if missing:
            ok = False
            print(f"  the UNMUTATED hoisted prologue is missing {missing}: the mutations below would"
                  " be 'caught' by an extractor bug rather than by the check")
        print("MUST-CATCH on the prologue expressions")
        for name, mut in PROLOGUE_MUTATIONS:
            gone = [k for k, v in prologue_present(mut(hoist_body)).items() if not v]
            print(f"  {'caught  ' if gone else 'INVISIBLE'} {name}"
                  + (f" -> no longer present: {gone}" if gone else ""))
            if not gone:
                ok = False
        print("RESULT " + ("check_hoist_transcription SELFTEST PASS" if ok
                           else "check_hoist_transcription SELFTEST FAIL"))
        return 0 if ok else 1

    ok = True
    shipped = walk_of(SHIPPED, "rec_loop_kernel")
    hoisted = walk_of(HOISTED, "rec_loop_normed_kernel")
    i = diff(shipped, hoisted)
    if i < 0:
        print(f"WALK   identical: {len(shipped)} lines of `rec_loop_kernel`'s state walk appear"
              " unchanged in `rec_loop_normed_kernel` (pragmas included)")
    else:
        ok = False
        print(f"WALK   DIFFERS at line {i + 1} (shipped {len(shipped)} lines, hoisted {len(hoisted)}):")
        print(f"       shipped: {shipped[i] if i < len(shipped) else '<eof>'}")
        print(f"       hoisted: {hoisted[i] if i < len(hoisted) else '<eof>'}")

    sp = prologue_present(kernel_body(SHIPPED, "rec_loop_kernel"))
    hp = prologue_present(kernel_body(HOISTED, "gdn_norm_hoist_kernel"))
    for name, want in PROLOGUE_EXPRS:
        print(f"PROLOG {'ok  ' if sp[name] and hp[name] else 'FAIL'} {name}: `{want}`")
        if not (sp[name] and hp[name]):
            ok = False
            print(f"       present in shipped: {sp[name]}; present in hoisted pre-pass: {hp[name]}")

    print("RESULT " + ("check_hoist_transcription PASS" if ok else "check_hoist_transcription FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
