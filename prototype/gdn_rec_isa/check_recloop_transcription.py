#!/usr/bin/env python3
"""Does `recloop_kernel` still contain the SHIPPING per-token arithmetic, line for line?

WHY THIS EXISTS. D53's item claims that moving the token index inside the launch moves no arithmetic.
The CPU mirror in the harness cannot witness that claim: both of its sides call the same per-token
arithmetic, so two spellings that share a formula agree by construction (`prototype/gdn_collapse/
results.md` §2 — the `q̃` row, quoted in `engine/gdn_rec_fused_kernels.h` — is that exact failure, and
it moved 93,522 of 131,072 `o` elements on device while a mirror watched). The device gate is the only
instrument that can see a compiler-side re-association, but it is a device run and it is expensive, and
the first thing a reviewer should be able to check cheaply is whether the SOURCE TEXT moved. That is
this file: extract the per-token core of three kernels and require it identical.

WHAT IS COMPARED. The "core" of a kernel is the code from its first `float kq = ` line through its
`o[h * D + j] = acc;` line — i.e. everything after the shared-memory declaration and before the
closing brace. `recloop_kernel`'s core is expected to be identical to:
  * `rec2c4_kernel`'s core, in this directory's `gdn_rec_isa_kernels.h` (the harness copy),
  * `rec_fused_kernel`'s core, in `engine/gdn_rec_fused_kernels.h` (the shipping copy), and
  * `rec_loop_kernel`'s core, in `engine/gdn_rec_loop_kernels.h` (the PORTED copy, D54's `--gdn-loop`:
    the same loop, in the engine tree, submitted by `engine/kernels_gdn.cpp::gdn_rec_span`). It is a
    fourth kernel rather than a shared `#define` precisely because a header shared between the
    prototype and the engine would let one side's edit be the other side's silent change; being two
    copies, only this checker can say they agree, so the pair is compared against everything else
    rather than only against its own parent.
Lines are compared with whitespace collapsed, because the looped body is indented four spaces deeper
and indentation is not arithmetic. Comments are stripped, since a comment cannot move a value — but
the `#pragma unroll 4` lines ARE compared: they survived the §18/D46 merge for a measured reason and a
dropped pragma is a different listing.

A CHECK THAT CANNOT FAIL IS DECORATION (`prototype/mlp_gemv`, `tools/mixed_bw`,
`prototype/attn_tile_model.py`, and this directory's own `--selftest-lane`). So:
  * the number of compared core lines must be at least `MIN_CORE_LINES`, and the checker exits non-zero
    if either extraction came back short or empty;
  * `--selftest` mutates the extracted looped core in ways that are each a REAL numerics change (a
    hoisted constant, a reordered add, a dropped pragma, a swapped store) and requires each mutation
    to be reported as a mismatch. A mutation nobody can see means the comparison is not comparing.

Usage:
    prototype/gdn_rec_isa/check_recloop_transcription.py            # the real diff
    prototype/gdn_rec_isa/check_recloop_transcription.py --selftest # the must-catch set
Exit 0 = identical (and, under --selftest, every mutation caught).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HARNESS_KERNELS = ROOT / "prototype/gdn_rec_isa/gdn_rec_isa_kernels.h"
ENGINE_KERNELS = ROOT / "engine/gdn_rec_fused_kernels.h"
ENGINE_LOOP_KERNELS = ROOT / "engine/gdn_rec_loop_kernels.h"

MIN_CORE_LINES = 30          # the shipped body is ~40 code lines; a short extraction is a parse failure

CORE_START = "float kq ="
CORE_END = "o[h * D + j] = acc;"

# The ONE spelling difference allowed between the harness copy and the engine copy, and it is a NAME,
# not an expression: `engine/gdn_rec_fused_kernels.h` renames `block_sum_pair_comp` to
# `block_sum_pair_f32` (its own header says so: "now named `block_sum_pair_f32` for the engine"), and
# the two bodies are otherwise the same helper. Aliasing the name here is not loosening the check — the
# call's ARGUMENTS are still compared, and any other difference still fails. A second alias would need
# the same kind of citation.
NAME_ALIASES = {"block_sum_pair_f32": "block_sum_pair_comp"}


def canonical(line: str) -> str:
    for a, b in NAME_ALIASES.items():
        line = line.replace(a, b)
    return line


def strip_comments(text: str) -> str:
    text = re.sub(r"/\*.*?\*/", "", text, flags=re.S)
    return re.sub(r"//[^\n]*", "", text)


def kernel_body(path: Path, name: str) -> str:
    """The brace-matched body of `__global__ void <name>(...)`."""
    src = strip_comments(path.read_text())
    m = re.search(r"__global__\s+void\s+" + re.escape(name) + r"\s*\([^)]*\)\s*\{", src)
    if not m:
        raise SystemExit(f"{path.name}: no `__global__ void {name}(...)` found — the checker would be"
                         " comparing nothing, which is a FAIL, not a pass")
    i = m.end() - 1
    depth = 0
    for j in range(i, len(src)):
        if src[j] == "{":
            depth += 1
        elif src[j] == "}":
            depth -= 1
            if depth == 0:
                return src[i + 1:j]
    raise SystemExit(f"{path.name}: unbalanced braces in {name}")


def core_lines(body: str) -> list[str]:
    """Whitespace-collapsed code lines from CORE_START through CORE_END, inclusive."""
    out: list[str] = []
    inside = False
    for raw in body.splitlines():
        line = " ".join(raw.split())
        if not line:
            continue
        if not inside and line.startswith(CORE_START):
            inside = True
        if inside:
            out.append(canonical(line))
            if line.startswith(CORE_END):
                return out
    raise SystemExit(f"core not found: no line starting `{CORE_START}` ... `{CORE_END}` in a"
                     f" {len(body.splitlines())}-line body — the extraction is broken")


def diff(a: list[str], b: list[str], an: str, bn: str) -> list[str]:
    problems = []
    if len(a) != len(b):
        problems.append(f"{an} core is {len(a)} lines, {bn} core is {len(b)} — different shape")
    for k in range(min(len(a), len(b))):
        if a[k] != b[k]:
            problems.append(f"line {k}:\n    {an}: {a[k]}\n    {bn}: {b[k]}")
    return problems


def apply_reordered(ls: list[str]) -> list[str]:
    """Swap the two adds of one unrolled body: same four products, different serial order."""
    out = list(ls)
    for i, l in enumerate(out):
        if l.startswith("acc += qs[i + 0] * s0;") and i + 1 < len(out):
            out[i] = out[i + 1]
            out[i + 1] = l
            return out
    raise SystemExit("mutation `add_reordered` found nothing to mutate — the core shape changed")


def apply_qscale_hoist(ls: list[str]) -> list[str]:
    return [("float qn = qq * (rsqrtf(sq + 1e-6f) * Q_SCALE);"
             if l.startswith("float qn =") else l) for l in ls]


def apply_pragma_drop(ls: list[str]) -> list[str]:
    return [l for l in ls if not l.startswith("#pragma unroll 4")]


def apply_store_drop(ls: list[str]) -> list[str]:
    return [(l.replace("s0 += ks[i + 0] * dv; Sh[(i + 0) * D + j] = s0;",
                       "s0 += ks[i + 0] * dv;") if l.startswith("float s0 =") else l) for l in ls]


MUTATIONS = {
    "qscale_hoisted": (apply_qscale_hoist,
                       "`(qq*rsqrt)*Q_SCALE` collapsed into one product: the §19 q̃ class, a"
                       " different product tree, same algebra"),
    "add_reordered": (apply_reordered,
                      "two adjacent serial adds swapped: same four products, different accumulation"
                      " order — exactly the re-association a bit-identity claim may not survive"),
    "pragma_dropped": (apply_pragma_drop,
                       "`#pragma unroll 4` removed: LLVM fully unrolls loop A and the listing shape"
                       " the census billed is gone (constraint recorded in the shipping header)"),
    "store_dropped": (apply_store_drop,
                      "one state store removed from the unrolled body: a byte of the state stops"
                      " being written, which a value comparison alone would call a numerics change"),
}


def selftest(looped: list[str], label: str = "recloop_kernel") -> int:
    """Every mutation must be reported as a mismatch against the looped core's own source."""
    bad = 0
    caught = 0
    for name, (fn, why) in MUTATIONS.items():
        try:
            mutated = fn(list(looped))
        except SystemExit as e:
            print(f"MUST-CATCH {name}: NOT APPLICABLE — {e}")
            bad += 1
            continue
        problems = diff(mutated, looped, "mutated", "original")
        ok = bool(problems)
        caught += 1 if ok else 0
        if not ok:
            bad += 1
        print(f"MUST-CATCH {name:16} {'CAUGHT' if ok else 'MISSED  '} ({len(looped)} vs"
              f" {len(mutated)} lines)  {why}")
    print(f"SELFTEST transcription target={label} mutations={len(MUTATIONS)} caught={caught}"
          f" bad={bad} ok={0 if bad else 1} claim=the line-by-line comparison below can see a hoisted"
          " constant, a reordered add, a dropped pragma and a dropped store")
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--selftest", action="store_true", help="run the must-catch mutations only")
    args = ap.parse_args()

    looped = core_lines(kernel_body(HARNESS_KERNELS, "recloop_kernel"))
    harness_ref = core_lines(kernel_body(HARNESS_KERNELS, "rec2c4_kernel"))
    engine_ref = core_lines(kernel_body(ENGINE_KERNELS, "rec_fused_kernel"))
    engine_loop = core_lines(kernel_body(ENGINE_LOOP_KERNELS, "rec_loop_kernel"))

    if args.selftest:
        # Both looped copies, not just the harness one: the mutations have to be visible in the file
        # the engine actually submits, or a defect in the ported kernel is proven invisible.
        rc = selftest(looped)
        rc |= selftest(engine_loop, label="engine/gdn_rec_loop_kernels.h::rec_loop_kernel")
        return rc

    for name, core in (("recloop_kernel", looped), ("rec2c4_kernel", harness_ref),
                       ("rec_fused_kernel", engine_ref), ("rec_loop_kernel", engine_loop)):
        if len(core) < MIN_CORE_LINES:
            raise SystemExit(f"{name}: only {len(core)} core lines extracted (<{MIN_CORE_LINES})"
                             " — refusing to certify an empty comparison")

    problems = diff(looped, harness_ref, "recloop_kernel", "rec2c4_kernel")
    problems += diff(looped, engine_ref, "recloop_kernel", "rec_fused_kernel")
    problems += diff(harness_ref, engine_ref, "rec2c4_kernel", "rec_fused_kernel")
    # The ported kernel against all three: its parent (`rec_fused_kernel`, the engine's per-token
    # spelling), and the harness loop it was transcribed from. Each is a separate diff so a reader
    # sees WHICH pairing moved rather than reading one merged list.
    problems += diff(engine_loop, engine_ref, "rec_loop_kernel", "rec_fused_kernel")
    problems += diff(engine_loop, looped, "rec_loop_kernel", "recloop_kernel")
    problems += diff(engine_loop, harness_ref, "rec_loop_kernel", "rec2c4_kernel")
    print(f"TRANSCRIPTION recloop_core={len(looped)} rec2c4_core={len(harness_ref)}"
          f" rec_fused_core={len(engine_ref)} rec_loop_core={len(engine_loop)}"
          f" lines_compared={len(looped) + len(engine_loop) + len(harness_ref) + len(engine_ref)}"
          f" (must be >= {MIN_CORE_LINES * 4})")
    if problems:
        print("TRANSCRIPTION FAIL — the looped body is not the shipping body:")
        for p in problems:
            print("  " + p)
        print("  The §19 rule: bit-identity is owed to the shipped SOURCE TEXT, not to the algebra."
              " Re-transcribe the loop body from `rec2c4_kernel` (harness) / `rec_fused_kernel`"
              " (engine) instead of editing the arithmetic here — and fix BOTH looped copies, since"
              " one of them is what the engine submits.")
        return 1
    print("TRANSCRIPTION OK — the per-token core of `recloop_kernel`, `rec2c4_kernel`,"
          " `engine/gdn_rec_fused_kernels.h::rec_fused_kernel` and"
          " `engine/gdn_rec_loop_kernels.h::rec_loop_kernel` is identical: same `#pragma unroll 4`"
          " lines, same order of every multiply, same stores. What this does NOT prove is that the"
          " COMPILER kept them identical: that is the device bit-identity gate (`--modes rec0,recloop`,"
          " and for the ported copy `engine/gate.py` section (j)'s byte-compare of the looped side),"
          " and it is the reason this file is a pre-flight and not an admission.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
