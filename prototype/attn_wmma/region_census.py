#!/usr/bin/env python3
"""HALF-FINISHED — its whole-kernel roll-up is NOT a usable denominator. `--calib` now aborts rather
than printing a plausible number: as handed over it reported `125.5 slots/touch` against the
published 45 for `attn_slim_wave` and **exactly 0.0** against 116 for `attn_tile_wave` while exiting
0, after two late edits that were never re-run clean (the `dot` multiplier 1.0 -> 0.5, because both
`attn_slim_dot<8>` and `<4>` inline onto the same source lines and only one executes, and the
WIDE/SHORT pass-2 route split).

What IS trustworthy here: the static whole-kernel bills in `isa_census.engine.output.txt` (1565 ops /
1524 slots for `attn_slim_wave`, reproducing `attn_valu_slim`'s independent 45 and 116 per-touch
numbers by loop fingerprint) and the classification rules, whose selftest passes. What is NOT: any
per-touch roll-up this script prints until `--calib` exits 0. See `results.md` §6 for why the WMMA
price did not depend on it.

Static bill of the shipped attention wave, attributed to source statements by the compiler's line
table and rolled up into a per-(row, lane, head) TOUCH bill. D30 item 2.4.

WHY ANOTHER CENSUS SCRIPT. `prototype/attn_valu_slim/isa_census.py` prices the pass-1 touch loop and
nothing else, which is why `attn_score_dot2` had to take pass 2 "by subtraction". Loop-fingerprint
grouping cannot be repaired here: LLVM fuses the two passes into one 800-instruction back-edge blob
(`.LBB5_116`, 192 blocks in the engine's own listing), so "which loop is the touch" is not
answerable from the CFG. This script therefore asks the compiler instead: `hipcc
-gline-tables-only` attributes every instruction to the statement it came from, resolved through
the inline stack to the frame inside `engine/attn_slim_kernels.h` (so `bf()`'s conversion is billed
to the dot that called it, not to `amd_hip_bfloat16.h`), and each statement region carries its own
per-touch multiplier read off the source's own loop structure:

    lane     the region advances one KV lane per trip (or is a constant-trip loop over the 8 dims a
             lane owns, which LLVM unrolls)   -> billed once per touch
    lane/4   the region's loop is distributed over the block's 4 warps, 8 lanes per sub-block
             -> 2 trips per 8 touches        -> billed 1/4 per touch
    tile     once per TILE(=8) lanes         -> billed 1/8 per touch
    block    geometry, the q load, the epilogue -> once per block, amortised over `--touches`
             (default 512 = one full MAXSC slice, the engine's wide-wave shape)

CALIBRATION is what makes the roll-up usable: `--calib` re-derives the pass-1 touch bill and checks
it against the number `prototype/attn_valu_slim/isa_census.py` measures independently on the same
listing by loop fingerprint (116 slots/touch shipped, 45 post-slim, D21/D24). If the two disagree
the script **exits non-zero**, and a roll-up of exactly zero — the signature of a region table that
no longer matches the source — is an abort, not a measurement of zero.

Categories: `attn_score_dot2` RULES < `attn_valu_slim` RULES < this file's matrix/transcendental/
cross-lane classes, and both inherited example sets re-run as a selftest first. Anything that does
not classify prints as UNCLASSIFIED and stays visible in the totals.

Usage:
    python3 region_census.py                      # compile the engine TU, both wave forms
    python3 region_census.py --asm FILE.s --kernel attn_slim_wave --calib
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent


def _load(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rec = _load("gdn_rec_isa_census", REPO / "prototype/gdn_rec_isa/isa_census.py")
slim = rec.slim
base = slim.base
ARCH = "gfx1151"
SLIM_H = "engine/attn_slim_kernels.h"

# --- statement regions: (name, multiplier, first line, text that must be on it, last line, text) --
# Line numbers because the regions are inside functions and a text anchor like `for (int rr = s_lo
# + w; ...)` occurs in both passes and would silently bill pass 2's code to pass 1. The must-contain
# text is the guard: move a statement and the script aborts instead of misattributing.
# `x/touch` = 0.0 means "once per block", billed at 1/touches (one MAXSC slice = 512 touches per
# warp), which is the wide-wave shape the funded price is about.
REGIONS = [
    ("dot",         0.5,   100, "template <int ND>",              108, "return dot;"),
    # ^ 0.5, not 1.0: `attn_slim_dot<8>` and `attn_slim_dot<4>` are both inlined and attribute to the
    # same source lines, but only one runs (`nd` is uniform). Billing 1.0 double-counts the dot.
    ("fold-add",    1.0,   120, "attn_slim_permute_add",          127, "bit_cast"),
    ("fold-addr",   0.0,   134, "attn_slim_perm_addr",            136, "return 4u"),
    ("fold",        1.0,   143, "float attn_slim_fold",           146, "return v;"),
    ("geom",        0.0,   157, "__shared__ hip_bfloat16 s_sc",   184, "perm_addr[i]"),
    ("q-load",      0.0,   188, "float q[8];",                    193, "q[t] = bf(qq[t]);"),
    ("stage-k",     0.5,   200, "for (int s_lo = 0; s_lo < ns",   215, "K sub-block published"),
    ("score-store", 1.0,   220, "auto publish = [&]",             240, "s_sc published for pass 2"),
    ("sm-1pass",    1.0,   251, "for (int t = 0; t < ns; ++t) m", 252, "l += expf"),
    ("stage-v-1p",  0.5,   253, "for (int s_lo = 0; s_lo < ns",   265, "__syncthreads();"),
    ("pv-1pass",    1.0,   266, "for (int rr = s_lo; rr < s_hi",  276, "l = 1.0f;"),
    ("stage-v-str", 0.5,   281, "for (int base = blo; base < bhi",293, "__syncthreads();"),
    ("sm-max",      1.0,   294, "float mt = -1e30f;",             297, "expf(m - mn)"),
    ("sm-exp",      1.0,   298, "float e[TILE], ls = 0.f;",       302, "}"),
    ("l-update",    0.125, 303, "l = l * rescale + ls;",          303, "l = l * rescale"),
    ("acc-rescale", 1.0,   304, "acc[u] *= rescale;",             304, "acc[u] *= rescale;"),
    ("pv-tile",     1.0,   305, "for (int slot = 0; slot < nt",   310, "before the next stage"),
    ("epilogue",    0.0,   316, "if (s_row == 1) {",              339, "acc[u];"),
]
P1_REGIONS = ("dot", "fold-add", "fold", "score-store", "stage-k")
# what attn_valu_slim's touch-loop window covers

# Classes the inherited tables have no bucket for; matched first, because `v_exp_f32_e32` would
# otherwise fall through to a generic convert/addr bucket.
EXTRA_RULES = [
    ("v_wmma?",            "MATRIX.wmma"),
    ("v_mfma?",            "MATRIX.mfma"),
    ("v_exp_f32_e32",      "transcend.expf"),
    ("v_ldexp_f32*",       "transcend.expf"),
    ("v_rcp_f32_e32",      "transcend.divide"),
    ("v_div_*",            "transcend.divide"),
    ("v_max_f32*",         "softmax.max"),
    ("v_min_f32*",         "softmax.max"),
    ("v_permlane*",        "fold.permute"),
    ("v_cvt_pk_bf16_bf16", "format.bf16-pack"),
    ("v_cvt_off_src_bf16", "format.bf16-round"),
]
NEW_EXAMPLES = [
    ("v_wmma_f32_16x16x16_bf16 v[24:31], v[8:15], v[16:23], v[24:31]", "MATRIX.wmma"),
    ("v_exp_f32_e32 v2, v1", "transcend.expf"),
    ("v_ldexp_f32 v3, v2, 4", "transcend.expf"),
    ("v_rcp_f32_e32 v5, v4", "transcend.divide"),
    ("v_div_scale_f32 v6, vcc, v2, v5", "transcend.divide"),
    ("v_max_f32_e32 v5, v2, v6", "softmax.max"),
    ("v_permlane_xor_b32 v1, v0, 1, v0, exec", "fold.permute"),
    ("v_cvt_pk_bf16_bf16 v2, v2, v3", "format.bf16-pack"),
    ("v_cvt_off_src_bf16 v1, v1, 0, v0", "format.bf16-round"),
]


def _rx(pat: str) -> str:
    out = []
    for c in pat:
        out.append(".*" if c == "*" else "[^,]*" if c == "?" else r"\s+" if c == " " else re.escape(c))
    return "^" + "".join(out)

EXTRA_C = [(re.compile(_rx(p)), c) for p, c in EXTRA_RULES]


def mn_of(text: str) -> str:
    m = re.match(r"([a-z][a-z0-9_.]*)", text)
    return m.group(1) if m else "?"


def category(mn: str, text: str) -> str:
    for rx, cat in EXTRA_C:
        if rx.match(text):
            return cat
    c = slim.rule_hit(text)
    if c and c != "other":
        return c
    if mn.startswith(base.SKIP_AS_WORK):
        return "encode.bookkeeping"
    for pre, cat in rec.FALLBACK:
        if mn.startswith(pre):
            return cat
    return "UNCLASSIFIED"


def selftest() -> tuple[int, int]:
    """Only attn_valu_slim's examples. gdn_rec_isa carries its own rule table for the GDN kernel,
    which is not part of this file's rule chain, so running its examples here would fail by
    construction and would be evidence about nothing."""
    bad = n = 0
    for pat, cat, ex in slim.RULE_EXAMPLES:
        n += 1
        if category(mn_of(ex), ex) != cat:
            bad += 1
            print(f"  FAIL  {pat:28} -> {category(mn_of(ex), ex)!r}, want {cat!r}")
    for ex, want in NEW_EXAMPLES:
        n += 1
        if category(mn_of(ex), ex) != want:
            bad += 1
            print(f"  FAIL  {ex[:36]:36} -> {category(mn_of(ex), ex)!r}, want {want!r}")
    return bad, n


FILE_RE = re.compile(r'^\s*\.file\s+(\d+)\s+"(.*)"')
LOC_RE = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+")
FRAME_RE = re.compile(r"([A-Za-z0-9_./<>]+):(\d+):\d+")


def check_regions() -> None:
    src = (REPO / SLIM_H).read_text().splitlines()
    prev_end = 0
    for name, _, a, must_a, b, must_b in REGIONS:
        if a <= prev_end:
            raise SystemExit(f"region {name} starts at {a}, inside the previous region — "
                             "overlapping regions would bill one instruction twice")
        for ln, must in ((a, must_a), (b, must_b)):
            if must not in src[ln - 1]:
                raise SystemExit(f"region {name}: line {ln} is {src[ln - 1].strip()[:60]!r}, "
                                 f"expected to contain {must!r} — the header moved, re-read it")
        prev_end = b


def instr_locs(body: list[str], ft: dict[str, str]) -> list[tuple[str, str, str]]:
    """(mnemonic, text, 'path:line') per instruction. The inline stack after `;` lists the innermost
    frame first; this picks the FIRST frame inside the copied attention header, so a helper inlined
    from `attn_core.h` or `hip_bfloat16.h` is billed to the attention statement that called it."""
    out, cur = [], ""
    for ln in body:
        m = LOC_RE.match(ln)
        if m:
            note = ln.split(";", 1)[1].strip() if ";" in ln else ""
            frames = FRAME_RE.findall(note) or []
            hdr = SLIM_H.split("/")[-1]
            pick = next((f"{p}:{l}" for p, l in frames if p.endswith(hdr)), "")
            if not pick:
                pick = (f"{frames[0][0]}:{frames[0][1]}" if frames
                        else f"{ft.get(m.group(1), '?')}:{m.group(2)}")
            cur = pick
            continue
        m = base.INSTR_RE.match(ln)
        if m:
            out.append((m.group(1), ln.strip(), cur))
    return out


def census(name: str, body: list[str], ft, touches: int):
    blocks = slim.split_blocks_full(body)
    live_mask = [not any(mn == "s_trap" for mn, _ in b["insts"]) and not b["flow"] for b in blocks]
    seq_all = instr_locs(body, ft)
    seq, i = [], 0
    for keep, b in zip(live_mask, blocks):
        take = seq_all[i:i + len(b["insts"])]
        i += len(b["insts"])
        if keep:
            seq += take
    src = (REPO / SLIM_H).read_text().splitlines()
    regs: dict[str, dict] = {}
    for mn, text, loc in seq:
        path, _, ln = loc.rpartition(":")
        if not path.endswith(SLIM_H.split("/")[-1]) or not ln.isdigit():
            rname, mult, ln = "OTHER-FILE:" + (path.split("/")[-1] or "?"), -1.0, 0
        else:
            ln = int(ln)
            hit = [(nm, mu, a, b) for nm, mu, a, _, b, _ in REGIONS if a <= ln <= b]
            if not hit:
                rname, mult, ln = "UNATTRIBUTED(slim header)", -1.0, ln
            else:
                rname, mult, _, _ = hit[0]
        d = regs.setdefault(rname, {"mult": mult, "cats": Counter(), "slot": 0, "lines": Counter()})
        d["cats"][category(mn, text)] += 1
        if not mn.startswith(base.SKIP_AS_WORK):
            d["slot"] += 1
        if ln:
            d["lines"][ln] += 1
    return {"seq": seq, "regs": regs, "src": src}


def render(name: str, c: dict, touches: int, w) -> float:
    w(f"===== {name}")
    tot_static = sum(sum(d["cats"].values()) for d in c["regs"].values())
    tot_slot = sum(d["slot"] for d in c["regs"].values())
    w(f"  {tot_static} executed instructions, {tot_slot} issue slots (s_trap and %Flow blocks "
      f"dropped); per-block regions billed at 1/{touches} touches")
    for d in c["regs"].values():
        if d["mult"] < 0:
            d["mult"] = 1.0 / touches
    w(f"  {'region':14} {'x/touch':>8} {'static':>6} {'slots':>6} {'slots/touch':>11}  top categories")
    touch_slot = 0.0
    per_reg_slot: dict[str, float] = {}
    for r, d in sorted(c["regs"].items(), key=lambda kv: -kv[1]["slot"]):
        per = d["slot"] * d["mult"]
        per_reg_slot[r] = per
        touch_slot += per
        cats = "  ".join(f"{k}={v}" for k, v in d["cats"].most_common(4))
        w(f"  {r:14} {d['mult']:8.3f} {sum(d['cats'].values()):6} {d['slot']:6} {per:11.2f}  {cats}")
    w(f"  {'TOTAL':14} {'':8} {tot_static:6} {tot_slot:6} {touch_slot:11.2f}")
    # Only one pass-2 route runs per block: the one-pass form when `s_row == 1 && full`, the streaming
    # form otherwise (`engine/attn_slim_kernels.h:248`). The funded wide-wave price is the streaming
    # column; the pinned short prompts are the one-pass column. Billing both would be a fiction.
    onepass = {"sm-1pass", "stage-v-1p", "pv-1pass"}
    stream = {"stage-v-str", "sm-max", "sm-exp", "l-update", "acc-rescale", "pv-tile"}
    wide = sum(v for k, v in per_reg_slot.items() if k not in onepass)
    short = sum(v for k, v in per_reg_slot.items() if k not in stream)
    w(f"  whole-kernel per-touch, WIDE row (streaming pass 2) = {wide:.2f} slots/touch")
    w(f"  whole-kernel per-touch, SHORT row (one-pass)       = {short:.2f} slots/touch")
    w("")
    w("  PER-TOUCH category rollup, WIDE route — this is the denominator D30 item 2.4 names")
    tc: Counter = Counter()
    for r, d in c["regs"].items():
        if r in onepass:
            continue
        for k, v in d["cats"].items():
            tc[k] += v * d["mult"]
    tt = sum(tc.values())
    for k, v in tc.most_common():
        w(f"    {k:24} {v:8.2f}  {100.0 * v / tt:5.1f}%")
    w("")
    w("  statement detail (line:count, source text) for the dot-shaped regions")
    for r in ("dot", "fold", "score-store", "pv-tile", "pv-1pass", "sm-exp", "sm-max",
              "acc-rescale"):
        d = c["regs"].get(r)
        if not d:
            continue
        w(f"    {r:12} " + " | ".join(
            f"L{ln}:{n} {c['src'][ln - 1].strip()[:46]}" for ln, n in d["lines"].most_common(5)))
    w("")
    return touch_slot, per_reg_slot, tc


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", type=Path)
    ap.add_argument("--kernel", default="attn_slim_wave,attn_tile_wave")
    ap.add_argument("--touches", type=int, default=512)
    ap.add_argument("--calib", action="store_true",
                    help="check the pass-1 rollup against attn_valu_slim's loop-fingerprint number "
                         "(116 shipped / 45 post-slim slots per touch)")
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    bad, n = selftest()
    if bad:
        print(f"SELFTEST FAIL ({bad} of {n})")
        return 1
    print(f"# rule selftest PASS ({n} examples: inherited + {len(NEW_EXAMPLES)} new classes)")
    check_regions()

    if a.asm:
        asm = a.asm
    else:
        td = Path(tempfile.mkdtemp(prefix="aw-"))
        subprocess.run(["hipcc", "-O3", "-gline-tables-only", f"--offload-arch={ARCH}",
                        "--save-temps", "-I", str(REPO), "-I", str(REPO / "engine"), "-c",
                        str(REPO / "engine/kernels_attn.cpp"), "-o", str(td / "k.o")],
                       check=True, capture_output=True, text=True, cwd=td)
        asm = next(td.glob("*-hip-amdgcn-amd-amdhsa-*.s"))
    lines = Path(asm).read_text().splitlines()
    ft = {m.group(1): m.group(2).strip('"') for ln in lines if (m := FILE_RE.match(ln))}
    kern = rec.split_kernels_local(lines)

    out: list[str] = []
    w = out.append
    w(f"listing {asm.name}")
    w(f"regions: {len(REGIONS)} statement ranges in {SLIM_H}, boundaries text-checked at run time")
    w("categories: attn_score_dot2 RULES < attn_valu_slim RULES < this file's EXTRA_RULES")
    w("")
    results = {}
    for want in a.kernel.split(","):
        match = [k for k in kern if want in k]
        if not match:
            w(f"===== {want}: NOT IN LISTING (here: {sorted(kern)})")
            continue
        name = match[0]
        c = census(name, kern[name], ft, a.touches)
        touch_slot, per_reg, tc = render(name, c, a.touches, w)
        results[name] = (touch_slot, per_reg, tc)
    if a.calib:
        w("CALIBRATION against attn_valu_slim's independent loop-fingerprint number")
        failed = []
        for name, (_ts, per_reg, _tc) in results.items():
            p1 = sum(per_reg.get(r, 0.0) for r in P1_REGIONS)
            want = 116.0 if "tile_wave" in name else 45.0 if "slim_wave" in name else None
            if want is None:
                continue
            # A rollup of zero means NO pass-1 region matched at all, which is not a measurement of zero and
            # must never be printed beside a published number as though it were one. Fifth instance this session
            # of the same class (gdn_collapse/run.py, mixed_bw, the wide-wave prefill-row count, the gdn_rec_isa
            # --volume parse): a check whose input can come back empty cannot fail.
            if p1 == 0.0:
                raise SystemExit(
                    f"{name}: pass-1 rollup is exactly 0.0 - no region in P1_REGIONS matched this kernel, so "
                    f"the region boundaries no longer describe the source. Refusing to adjudicate. "
                    f"Re-derive the region table against the current header before trusting any figure here.")
            ok = abs(p1 - want) <= 0.12 * want
            w(f"  {name}: pass-1 rollup {p1:.1f} slots/touch vs published {want:.0f} "
              f"({'CALIBRATED' if ok else 'CALIBRATION-FAIL'})")
            if not ok:
                failed.append((name, p1, want))
        if failed:
            # Hard, not soft: this script's numbers get quoted into prices, and a header that says "soft" is how
            # a wrong denominator survives into a table. The WMMA price (prototype/attn_wmma/results.md) does
            # NOT use this rollup - it uses D20's independently measured 27.9 % share - precisely because it
            # was red at hand-off.
            raise SystemExit("CALIBRATION-FAIL for " + ", ".join(f"{n} ({p:.1f} vs {w0:.0f})"
                                                                 for n, p, w0 in failed)
                             + " - fix the region table before quoting a slot figure from this census")
    txt = "\n".join(out)
    print(txt)
    if a.out:
        a.out.write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
