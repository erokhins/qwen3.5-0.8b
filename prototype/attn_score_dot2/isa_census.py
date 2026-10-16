#!/usr/bin/env python3
"""isa_census.py — count issued instructions in a GCN code object's loops, for pricing an arithmetic swap.

WHY THIS EXISTS. The question "how much of `attn`'s instruction stream is the QK score dot" cannot be
answered by counting FMAs, because on this ISA the dot is cheap and the things around it are not: one
bf16->f32 conversion is one `v_lshlrev_b32`/`v_and_b32_e32` (so 8 per touch for the K row), one
`__shfl_down_sync` is ~5 VALU ops plus an LDS permute because the compiler cannot prove `FULL_MASK` is the
full exec mask, and the score round is a `v_bfe` + a rounding add + a shift. A price built from MACs alone
understates the loop by ~3x. So: disassemble, and count what the machine is told to do.

CPU-ONLY. `hipcc -c` compiles and `--save-temps` keeps the `.s`; nothing is launched and no gpu_lock is
taken. Regenerate the input like this (repo root):

    cd prototype/attn_decode && hipcc -O3 --offload-arch=gfx1151 --save-temps -c \
        <(echo '#include "attn_decode_kernels.h"') ...   # see HANDOFF.md for the exact two-line TU

HANDOFF.md names the commands. The counts below are printed, never asserted from prose.

WHAT COUNTS AS WHAT.
  * `v_*`                -> VALU. A `v_dual_X :: v_dual_Y` line is ONE issue slot carrying TWO ops
                            (issue_rate_probe/results.md §3: "the census counts a v_dual_* packet as 2
                            ops in its op column while issue slots count it once"), so both are reported.
  * `ds_*`               -> LDS pipe (includes `ds_bpermute`/`ds_swizzle`, which are how a shuffle issues).
  * `global_*`, `scratch_*`, `image_*` -> VMEM pipe.
  * `s_*`                -> SALU, with `s_delay_alu`/`s_waitcnt` reported separately as encode-only
                            bookkeeping (they retire but do no ALU work; `s_delay_alu` is the dual-issue
                            schedule hint).
  * branches/labels      -> control.
A loop's per-iteration cost is reported as the sum of the blocks annotated
`;   in Loop: Header=BBx_n Depth=m` plus the header block itself, which is the shape LLVM emits when it
peels a prologue: the peeled copy is a different block and is NOT included. Loops are selected by
fingerprint (`--loop-has`), never by the compiler's own `BBn_m` numbering, because that numbering changes
when unrelated source changes.

Usage:
    .venv/bin/python prototype/attn_score_dot2/isa_census.py <file.s> --kernels attn_tile_wave
    .venv/bin/python prototype/attn_score_dot2/isa_census.py <file.s> --kernels attn_tile_wave \
        --loop-has ds_bpermute_b32
"""
from __future__ import annotations

import argparse
import re
from collections import Counter
from pathlib import Path

CLASSES = {
    "VALU": ("v_",),
    "LDS": ("ds_",),
    "VMEM": ("global_", "scratch_", "image_"),
    "SALU": ("s_",),
}
SKIP_AS_WORK = ("s_delay_alu", "s_waitcnt", "s_nop")
LABEL_RE = re.compile(r"^(\.?L?[A-Za-z0-9_]+):")
BB_COMMENT_RE = re.compile(r"^; %bb\.(\d+):")
INSTR_RE = re.compile(r"^\t([a-z][a-z0-9_.]*)")
LOOP_ANN_RE = re.compile(r"in Loop: Header=(\S+) Depth=(\d+)")
HDR_ANN_RE = re.compile(r"This (?:Inner |Sub)?\s*Loop Header: Depth=(\d+)")


def norm_key(label: str) -> str:
    """`LLVM's `.LBB1_27` label and its `in Loop: Header=BB1_27` comment name the same block."""
    key = label.rstrip(":")
    return key[2:] if key.startswith(".L") else key


def classify(mnemonic: str) -> str:
    if mnemonic.startswith("v_dual"):
        return "VALU"
    for cls, prefixes in CLASSES.items():
        if mnemonic.startswith(prefixes):
            return cls
    return "other"


def split_kernels(lines: list[str]) -> dict[str, list[str]]:
    """Kernel name -> its assembly lines. A kernel starts at its mangled label and ends at .Lfunc_endN."""
    out: dict[str, list[str]] = {}
    name = None
    for ln in lines:
        # `_Z14attn_tile_wave8CoreArgs:` (a C++ mangled kernel) or `score_scalar:` (extern "C").
        m = re.match(r"^_Z\d+([A-Za-z0-9_]+)\d+[A-Za-z0-9_]*:\s*$", ln) or \
            re.match(r"^([A-Za-z_][A-Za-z0-9_]*):\s*(?:;.*)?$", ln)
        if m:
            name = m.group(1)
            out.setdefault(name, [])
            continue
        if name is None:
            continue
        if ln.startswith(".Lfunc_end"):
            name = None
            continue
        out[name].append(ln)
    return out


def split_blocks(body: list[str]) -> list[dict]:
    """Basic blocks of one kernel, each with its instructions and its loop annotations."""
    blocks: list[dict] = []
    cur: dict | None = None
    pending: list[str] = []
    for ln in body:
        lab = LABEL_RE.match(ln) or BB_COMMENT_RE.match(ln)
        if lab:
            if cur:
                blocks.append(cur)
            cur = {"label": lab.group(1), "insts": [], "header_of": None, "in_loop": None,
                   "prologue_comment": " ".join(pending)}
            pending = []
            continue
        if ln.strip().startswith(";"):
            pending.append(ln.strip())
            if cur:
                hm = HDR_ANN_RE.search(ln)
                if hm:
                    cur["header_of"] = norm_key(cur["label"])
                lm = LOOP_ANN_RE.search(ln)
                if lm:
                    cur["in_loop"] = (lm.group(1), int(lm.group(2)))
            continue
        if cur is None:
            continue
        im = INSTR_RE.match(ln)
        if not im:
            continue
        mn = im.group(1)
        cur["insts"].append(mn)
    if cur:
        blocks.append(cur)
    return blocks


def tally(insts: list[str]) -> Counter:
    t: Counter = Counter()
    for mn in insts:
        if mn.startswith(SKIP_AS_WORK):
            t["encode-only"] += 1
            continue
        cls = classify(mn)
        t[cls] += 1
        # A dual packet carries a second op in the same slot.
        if mn.startswith("v_dual") or "::" in mn:
            t["VALU"] += 1
            t["dual_second_op"] += 1
    t["slots"] = sum(t[c] for c in ("VALU", "LDS", "VMEM", "SALU", "other"))
    t["ops"] = t["slots"] + t["dual_second_op"]
    return t


def find_loops(blocks: list[dict], fingerprint: str | None) -> list[dict]:
    """Group blocks into loops keyed by their header label; keep loops whose body matches fingerprint."""
    loops: dict[str, dict] = {}
    for b in blocks:
        key = b["header_of"] or (b["in_loop"][0] if b["in_loop"] else None)
        if key is None:
            continue
        key = norm_key(key)
        loop = loops.setdefault(key, {"header": key, "depth": None, "blocks": [], "insts": []})
        loop["blocks"].append(b["label"])
        loop["insts"].extend(b["insts"])
        if loop["depth"] is None:
            loop["depth"] = b["in_loop"][1] if b["in_loop"] else 1
    out = []
    for loop in loops.values():
        if fingerprint and not any(fingerprint in i for i in loop["insts"]):
            continue
        if len(loop["insts"]) == 0:
            continue
        out.append(loop)
    return out


def report(path: Path, kernel_names: list[str], fingerprint: str | None,
           ops_re: str | None = None) -> int:
    lines = path.read_text().splitlines()
    kernels = split_kernels(lines)
    rc = 0
    for name in kernel_names:
        match = [k for k in kernels if name in k]
        if not match:
            print(f"MISSING kernel {name} in {path} (have: {', '.join(sorted(kernels))})")
            rc = 1
            continue
        for k in match:
            body = kernels[k]
            blocks = split_blocks(body)
            all_insts = [i for b in blocks for i in b["insts"]]
            t = tally(all_insts)
            print(f"\n{k}")
            print(f"  whole kernel, static: ops={t['ops']} issue-slots={t['slots']} "
                  f"VALU={t['VALU']} LDS={t['LDS']} VMEM={t['VMEM']} SALU={t['SALU']} "
                  f"other={t['other']} encode-only={t['encode-only']}")
            mix = Counter(all_insts).most_common(12)
            print("  hottest opcodes: " + ", ".join(f"{m}={n}" for m, n in mix))
            if ops_re:
                pat = re.compile(ops_re)
                counts = Counter(all_insts)
                hit = {m: n for m, n in counts.items() if pat.search(m)}
                tot = sum(hit.values())
                print(f"  opcodes matching /{ops_re}/: {tot} of {len(all_insts)} "
                      f"({tot / max(len(all_insts), 1):.1%} of the static instruction stream)")
                print("    " + ", ".join(f"{m}={n}" for m, n in sorted(hit.items(),
                                                                      key=lambda kv: -kv[1])))
            loops = find_loops(blocks, fingerprint)
            if fingerprint:
                print(f"  loops whose body contains {fingerprint!r}: {len(loops)}")
            for loop in loops:
                lt = tally(loop["insts"])
                print(f"    loop {loop['header']} depth={loop['depth']} blocks={len(loop['blocks'])}: "
                      f"ops/iter={lt['ops']} slots/iter={lt['slots']} VALU={lt['VALU']} "
                      f"LDS={lt['LDS']} VMEM={lt['VMEM']} SALU={lt['SALU']}")
    return rc


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("asm", type=Path, help="a GCN .s produced by hipcc --save-temps -c")
    ap.add_argument("--kernels", default="attn_tile_wave",
                    help="comma-separated kernel names (substring match)")
    ap.add_argument("--loop-has", default=None,
                    help="only price loops whose body contains this opcode substring")
    ap.add_argument("--ops", default=None,
                    help="regex: also report the static share of the kernel's instruction stream "
                         "taken by matching opcodes (e.g. 'v_exp_f32|v_div_|v_rcp_f32')")
    args = ap.parse_args()
    return report(args.asm, [k.strip() for k in args.kernels.split(",")], args.loop_has,
                  args.ops)


if __name__ == "__main__":
    raise SystemExit(main())
