#!/usr/bin/env python3
"""Census the per-touch instruction bill of the slim attention wave, before and after.

D20 funded "remove the causal-mask guards (20 VALU/touch) and the 8 bf16->f32 conversions" and asked
for it to be "counted in the disassembly before and after, not modelled". This is that count. It
reuses `prototype/attn_score_dot2/isa_census.py` for the generic parts (kernel/block splitting, the
dual-packet slot rule) and adds the two things that op's script deliberately left out: a NAMED
CATEGORY ROLLUP of one (row, lane, head) touch, and the resource census (VGPR / LDS / occupancy),
which is where a lever that trades instructions for registers shows up.

What a "touch" is, mechanically, and why it is not a whole-loop line window:
    one iteration of the pass-1 `for (rr = s_lo; rr < s_hi; ++rr)` loop = one score dot + one
    32-lane fold + one BF16 round + the store. The census finds it as the loop at depth 2 whose body
    contains `ds_bpermute_b32` (the fold's permute), and takes every block whose loop annotations
    put it inside that header. Two corrections that the naive version gets wrong:
      * LLVM's `%Flow` blocks (the `exec != mask` slow paths) are annotated as inside the loop but
        never execute on a full warp, so they are excluded and reported separately.
      * MODE 0/3 keep the score dot as a ROLLED child loop (trip count `nd`, 8 on the shipping
        path). Its body therefore counts `--child-trip` times, not once. Not doing this is how a
        census reports 24 VALU/touch for a touch that costs 48.

Usage:
    python3 isa_census.py                      # compile the TU and census every MODE
    python3 isa_census.py --asm FILE.s         # census an existing listing
    python3 isa_census.py --window BB6_32      # dump one loop's instructions, for the write-up
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "attn_score_dot2"))
import isa_census as base  # noqa: E402  (the generic .s parser and slot-tally rules)

ARCH = "gfx1151"
BASE_KERNEL = "attn_tile_wave"           # the shipped form, present in the TU as a control
DEFAULT_KERNELS = [BASE_KERNEL, "attn_slim0_wave", "attn_slim2_wave", "attn_slim3_wave",
                   "attn_slim1_wave"]   # 0 = copy of shipped, 2 = dot lever, 3 = fold lever,
                                        # 1 = both (the headline A/B)
LABELS = {"attn_tile_wave": "shipped", "attn_slim0_wave": "slim0 copy",
          "attn_slim2_wave": "slim2 dot only", "attn_slim3_wave": "slim3 fold only",
          "attn_slim1_wave": "slim1 both"}

# Category rules, FIRST MATCH WINS, operand-sensitive because "v_and_b32_e32 v, 0xffff0000, v" is a
# BF16->F32 conversion and "v_and_b32_e32 v, 0x7f800000, v" is the BF16 rounding emulation testing
# for a NaN. Mini-language: `?` matches a register name, a space matches one-or-more spaces.
#
# Every rule carries an example line it MUST match, checked at startup. This is not ceremony: the
# first version of this table had a broken space escape, four rules matched nothing, and the rollup
# printed a clean-looking table with `fold.range-select` and `dot.bf16-conv` simply absent — the
# same failure shape as `prototype/gdn_collapse/run.py`'s regex that matched zero CASE lines.
RULES: list[tuple[str, str, str]] = [
    ("ds_bpermute_b32",               "fold.permute",            "ds_bpermute_b32 v10, v10, v9"),
    ("ds_swizzle_b32",                "fold.permute",            'ds_swizzle_b32 v10, v9 offset:swizzle(BITMASK_PERM,"1pppp")'),
    ("s_cmp_eq_u32 exec_lo, -1",      "fold.exec-guard",         "s_cmp_eq_u32 exec_lo, -1"),
    ("s_cbranch_scc0",                "fold.exec-guard",         "s_cbranch_scc0 .LBB1_26"),
    ("v_cmp_gt_u32_e32 vcc_lo, 24,",  "fold.range-select",       "v_cmp_gt_u32_e32 vcc_lo, 24, v12"),
    ("v_cmp_gt_u32_e32 vcc_lo, 28,",  "fold.range-select",       "v_cmp_gt_u32_e32 vcc_lo, 28, v12"),
    ("v_cmp_gt_u32_e32 vcc_lo, 30,",  "fold.range-select",       "v_cmp_gt_u32_e32 vcc_lo, 30, v12"),
    ("v_cmp_ne_u32_e32 vcc_lo, 31,",  "fold.range-select",       "v_cmp_ne_u32_e32 vcc_lo, 31, v12"),
    ("v_cndmask_b32_e64 v?, 0, 16,",  "fold.range-select",       "v_cndmask_b32_e64 v11, 0, 16, vcc_lo"),
    ("v_cndmask_b32_e64 v?, 0, 8,",   "fold.range-select",       "v_cndmask_b32_e64 v11, 0, 8, vcc_lo"),
    ("v_cndmask_b32_e64 v?, 0, 4,",   "fold.range-select",       "v_cndmask_b32_e64 v11, 0, 4, vcc_lo"),
    ("v_cndmask_b32_e64 v?, 0, 2,",   "fold.range-select",       "v_cndmask_b32_e64 v11, 0, 2, vcc_lo"),
    ("v_cndmask_b32_e64 v?, 0, 1,",   "fold.range-select",       "v_cndmask_b32_e64 v11, 0, 1, vcc_lo"),
    ("v_add_lshl_u32",                "fold.addr-math",          "v_add_lshl_u32 v10, v11, v12, 2"),
    ("v_add_co_ci_u32_e64",           "fold.addr-math",          "v_add_co_ci_u32_e64 v11, null, 0, v12, vcc_lo"),
    ("v_add_f32_e32",                 "fold.add",                "v_add_f32_e32 v9, v9, v10"),
    ("v_dual_add_f32",                "fold.add",                "v_dual_add_f32 v9, v9, v10 :: v_dual_lshlrev_b32 v10, 2, v11"),
    ("v_lshlrev_b32_e32 v?, 16,",     "dot.bf16-conv",           "v_lshlrev_b32_e32 v11, 16, v11"),
    ("v_dual_lshlrev_b32 v?, 16,",    "dot.bf16-conv",           "v_dual_lshlrev_b32 v25, 16, v26 :: v_dual_fmac_f32 v2, v14, v25"),
    ("v_mov_b16_e32 v?.l,",           "dot.bf16-conv",           "v_mov_b16_e32 v2.l, v36.l"),
    ("v_and_b32_e32 v?, 0xffff0000,", "dot.bf16-conv",           "v_and_b32_e32 v42, 0xffff0000, v42"),
    ("v_dual_and_b32 v?, 0xffff0000,", "dot.bf16-conv",          "v_dual_and_b32 v3, 0xffff0000, v37 :: v_dual_fmac_f32 v2, v11, v3"),
    ("v_movrels",                     "dot.q-indexed",           "v_movrels_b32_e32 v15, v0"),
    ("v_madrs_f32",                   "dot.fma",                 "v_madrs_f32 v9, v15, v11, v9"),
    ("v_mad_f32",                     "dot.fma",                 "v_mad_f32 v9, v15, v11, v9"),
    ("v_dual_fmac_f32",               "dot.fma",                 "v_dual_fmac_f32 v9, v15, v11 :: v_dual_add_nc_u32 v10, 2, v10"),
    ("v_dual_fma_f32",                "dot.fma",                 "v_dual_fma_f32 v9, v15, v11, v9 :: v_dual_mov_b32 v12, 0"),
    ("v_fmac_f32_e32",                "dot.fma",                 "v_fmac_f32_e32 v13, v26, v43"),
    ("v_fma_f32",                     "dot.fma",                 "v_fma_f32 v9, v15, v11, v9"),
    ("v_mul_f32",                     "other.mul",               "v_mul_f32_e32 v9, v10, v11"),
    ("v_bfe_u32",                     "score.bf16-round",        "v_bfe_u32 v10, v9, 16, 1"),
    ("v_add3_u32",                    "score.bf16-round",        "v_add3_u32 v10, v9, 0x7fff, v11"),
    ("v_and_b32_e32 v?, 0x7f800000,", "score.bf16-round",        "v_and_b32_e32 v10, 0x7f800000, v9"),
    ("v_and_b32_e32 v?, 0x8000,",     "score.bf16-round",        "v_and_b32_e32 v11, 0x8000, v10"),
    ("s_and_saveexec_b32",            "score.bf16-round-guard",  "s_and_saveexec_b32 s48, vcc_lo"),
    ("v_cndmask_b32_e32",             "score.bf16-round",        "v_cndmask_b32_e32 v10, v11, v12, vcc_lo"),
    ("ds_load",                       "dot.lds-load",            "ds_load_u16 v11, v10"),
    ("ds_store",                      "stage.lds-store",         "ds_store_b128 v[10:13], v8"),
    ("s_waitcnt lgkmcnt",             "wait.lgkmcnt",            "s_waitcnt lgkmcnt(0)"),
]
RULE_EXAMPLES = [(pat, cat, ex) for pat, cat, ex in RULES]
# Which category groups belong to D20's two named items, for the headline row of the report.
D20_NAMED = {"fold.exec-guard", "fold.range-select", "fold.addr-math", "dot.bf16-conv"}
TOUCH = ("fold.", "dot.", "score.")     # category prefixes that are part of one touch


def _compile(pat: str) -> str:
    """`?` = a register name, a space = one-or-more spaces. `re.escape` escapes BOTH, so the space
    has to be substituted in its escaped form (`\\ `), not as a bare space — the first version of
    this function replaced a bare space and produced a pattern that demanded a backslash."""
    rx = re.escape(pat)
    rx = rx.replace("\\?", "\x00")
    rx = rx.replace("\\ ", r"\s+")
    return "^" + rx.replace("\x00", "[^,]+")


_COMPILED = [(re.compile(_compile(p)), c) for p, c, _ in RULES]


def selftest_rules() -> int:
    """Every rule must match its own example, and must match it as the category it claims."""
    bad = []
    for pat, cat, ex in RULE_EXAMPLES:
        if not _compile(pat) or not re.match(_compile(pat), ex):
            bad.append(f"{pat!r} does not match {ex!r}")
        elif rule_hit(ex) != cat:
            bad.append(f"{pat!r} matches {ex!r} but is bucketed {rule_hit(ex)}, not {cat}")
    if bad:
        for b in bad:
            print(f"RULE SELFTEST FAIL: {b}", file=sys.stderr)
        raise SystemExit(2)
    return len(RULE_EXAMPLES)


def rule_hit(text: str) -> str:
    for rx, cat in _COMPILED:
        if rx.match(text):
            return cat
    return "other"


selftest_rules()


def split_blocks_full(body: list[str]) -> list[dict]:
    """Like base.split_blocks but keeps full instruction TEXT and the loop-header chain."""
    blocks: list[dict] = []
    cur: dict | None = None
    parents: list[tuple[str, int]] = []
    depth: int | None = None
    label = None
    flow = False
    for ln in body:
        lab = base.LABEL_RE.match(ln) or base.BB_COMMENT_RE.match(ln)
        if lab:
            if cur:
                cur["parents"], cur["depth"] = parents, depth
                blocks.append(cur)
            label, parents, depth = lab.group(1), [], None
            flow = "%Flow" in ln
            cur = {"label": label, "insts": [], "flow": flow}
            continue
        if cur is None:
            continue
        if ln.strip().startswith(";"):
            s = ln.strip()
            m = re.search(r"Parent Loop (\S+) Depth=(\d+)", s)
            if m:
                parents.append((base.norm_key(m.group(1)), int(m.group(2))))
            m = re.search(r"This (?:Inner |Sub)?\s*Loop Header: Depth=(\d+)", s)
            if m:
                depth = int(m.group(1))
                parents.append((base.norm_key(label or ""), depth))
            m = base.LOOP_ANN_RE.search(s)
            if m:
                parents.append((base.norm_key(m.group(1)), int(m.group(2))))
            continue
        im = base.INSTR_RE.match(ln)
        if not im:
            continue
        cur["insts"].append((im.group(1), ln.strip()))
    if cur:
        cur["parents"], cur["depth"] = parents, depth
        blocks.append(cur)
    return blocks


def touch_loop(blocks: list[dict]) -> tuple[str, int] | None:
    """The pass-1 touch loop: the loop header whose blocks contain a permute, voted for by them.

    The permute blocks are not the loop header (LLVM splits the unrolled butterfly across blocks
    that each carry `in Loop: Header=BB1_27 Depth=2`), so the vote comes from the annotations, not
    from which block owns the header label. Ties between candidate loops (slim1 has two: the nd==8
    and nd==4 copies of the score loop) go to the one with more FMAs, i.e. the 8-dim hot copy.
    """
    best: Counter = Counter()
    for b in blocks:
        if b["flow"] or not b["parents"]:
            continue
        if not any(mn.startswith("ds_bpermute") or mn.startswith("ds_swizzle") for mn, _ in b["insts"]):
            continue
        hdr, dep = max(b["parents"], key=lambda p: p[1])
        fmas = sum(1 for mn, _ in b["insts"] if "fma" in mn)
        best[(hdr, dep)] += fmas
    return best.most_common(1)[0][0] if best else None


def loop_body(blocks: list[dict], hdr: str, dep: int, child_trip: int) -> tuple[list, list]:
    """The blocks that run once per touch, plus the never-taken %Flow blocks kept aside.

    A block whose own loop depth exceeds `dep` sits in a nested loop (the rolled score dot) and is
    therefore counted `child_trip` times: the touch executes its body nd = 8 times, which is the
    difference between reporting 24 VALU/touch and the 48 the counter will see.
    """
    mine, flow = [], []
    for b in blocks:
        hits = [p[1] for p in b["parents"] if p[0] == hdr]
        if not hits or min(hits) < dep:
            continue
        if b["flow"]:
            flow.append(b)
            continue
        deeper = (b["depth"] is not None and b["depth"] > dep) or \
                 any(p[1] > dep for p in b["parents"])
        mine.extend([b] * (child_trip if deeper else 1))
    return mine, flow


def tally_text(blks: list[dict]) -> Counter:
    insts = [ln for b in blks for _, ln in b["insts"]]
    t = Counter()
    for ln in insts:
        t[rule_hit(ln)] += 1
    return t


def resources(src: Path, outdir: Path) -> dict[str, dict[str, str]]:
    cmd = ["hipcc", "-O3", f"--offload-arch={ARCH}", "-Rpass-analysis=kernel-resource-usage",
           "-I", str(src.parent), "-c", str(src), "-o", str(outdir / "r.o")]
    p = subprocess.run(cmd, capture_output=True, text=True)
    out: dict[str, dict[str, str]] = {}
    name = None
    for ln in (p.stdout + p.stderr).splitlines():
        m = re.search(r"Function Name: (\S+)", ln)
        if m:
            name = re.sub(r"^_Z\d+|[0-9].*$", "", m.group(1))
            name = m.group(1)
            out[name] = {}
            continue
        if name:
            m = re.search(r"remark:\s+([^:]+): (.+?) \[", ln)
            if m:
                out[name][m.group(1).strip()] = m.group(2).strip()
    return out


def short(n: str) -> str:
    """`_Z14attn_tile_wave8CoreArgs` -> `attn_tile_wave`, without needing a demangler on PATH."""
    m = re.match(r"^_Z\d+([A-Za-z0-9_]+)", n)
    return m.group(1) if m else n


def nm(n: str) -> str:
    return LABELS.get(short(n), short(n))



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=HERE / "isa_tu.cpp")
    ap.add_argument("--asm", type=Path, default=None, help="an existing .s; skips the compile")
    ap.add_argument("--kernels", default=",".join(DEFAULT_KERNELS))
    ap.add_argument("--child-trip", type=int, default=8,
                    help="trip count of the rolled score-dot child loop (nd on the shipping path)")
    ap.add_argument("--window", default=None, help="dump every instruction of one loop header")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    tmp = None
    if args.asm:
        asm = args.asm
        res: dict[str, dict[str, str]] = {}
    else:
        tmp = Path(tempfile.mkdtemp(prefix="vslim-isa-"))
        cmd = ["hipcc", "-O3", f"--offload-arch={ARCH}", "--save-temps", "-I", str(args.src.parent),
               "-c", str(args.src), "-o", str(tmp / "isa_tu.o")]
        # `--save-temps` writes the device listing next to the CWD, not next to `-o`, so the compile
        # runs inside the scratch directory it is meant to fill.
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=str(tmp))
        asm = next(tmp.glob("*-hip-amdgcn-amd-amdhsa-*.s"))
        res = resources(args.src, tmp)
        print(f"# compiled: {' '.join(cmd)}\n# listing: {asm}\n")

    lines = asm.read_text().splitlines()
    kernels = base.split_kernels(lines)
    want = [k for k in args.kernels.split(",") if any(k in n for n in kernels)]
    names = [n for n in kernels if any(k in n for k in want)]
    if args.window:
        for n in names:
            for b in split_blocks_full(kernels[n]):
                if args.window in b["label"]:
                    print(f"--- {nm(n)} {b['label']} parents={b['parents']}")
                    for _, ln in b["insts"]:
                        print("   ", ln, "  ;;", rule_hit(ln))
        return 0

    out = []
    def emit(s: str = "") -> None:
        out.append(s)
        print(s)

    emit("## 1. Resources (from -Rpass-analysis=kernel-resource-usage, no device involved)")
    emit(f"{'kernel':22} {'VGPRs':>6} {'SGPRs':>6} {'LDS B':>7} {'waves/SIMD':>10} {'spill':>6}")
    for n in names:
        r = res.get(n, {})
        if not r:
            continue
        emit(f"{nm(n):22} {r.get('VGPRs', '-'):>6} {r.get('TotalSGPRs', '-'):>6}"
             f" {r.get('LDS Size [bytes/block]', '-'):>7} {r.get('Occupancy [waves/SIMD]', '-'):>10}"
             f" {r.get('VGPRs Spill', '-'):>6}")
    emit()

    emit("## 2. Whole-kernel static bill (base isa_census tally; dual packet = 2 ops in 1 slot)")
    emit(f"{'kernel':22} {'ops':>5} {'slots':>6} {'VALU':>5} {'LDS':>4} {'VMEM':>5} {'SALU':>5}")
    per_kernel: dict[str, Counter] = {}
    for n in names:
        t = base.tally([mn for b in split_blocks_full(kernels[n]) for mn, _ in b["insts"]])
        per_kernel[n] = t
        emit(f"{nm(n):22} {t['ops']:>5} {t['slots']:>6} {t['VALU']:>5} {t['LDS']:>4}"
             f" {t['VMEM']:>5} {t['SALU']:>5}")
    emit()

    emit(f"## 3. One (row, lane, head) touch: the pass-1 loop, rolled child x{args.child_trip},"
         " LLVM %Flow blocks excluded")
    header_rows: dict[str, Counter] = {}
    info: dict[str, dict] = {}
    for n in names:
        blocks = split_blocks_full(kernels[n])
        tl = touch_loop(blocks)
        if tl is None:
            emit(f"{nm(n):22} (no loop containing a permute found)")
            continue
        hdr, dep = tl
        mine, flow = loop_body(blocks, hdr, dep, args.child_trip)
        insts = [ln for b in mine for _, ln in b["insts"]]
        mns = [mn for b in mine for mn, _ in b["insts"]]
        cat = tally_text(mine)
        t = base.tally(mns)
        header_rows[n] = cat
        info[n] = dict(hdr=hdr, nblocks=len(mine), nflow=len(flow), slots=t["slots"],
                       ops=t["ops"], valu=t["VALU"], lds=t["LDS"], salu=t["SALU"])
        emit(f"  {nm(n):22} loop={hdr} depth={dep} blocks={info[n]['nblocks']:3}"
             f" flow-excl={info[n]['nflow']:2} slots/touch={info[n]['slots']:3}"
             f" VALU/touch={info[n]['valu']:3} LDS/touch={info[n]['lds']:2}"
             f" SALU/touch={info[n]['salu']:3}")
    emit()

    cats = sorted({c for t in header_rows.values() for c in t if c != "other"})
    emit("## 4. Per-touch category rollup (instructions per touch; D20's two named items are starred)")
    emit(f"{'category':26}" + "".join(f"{nm(n)[:13]:>14}" for n in header_rows))
    base_k = DEFAULT_KERNELS[0] if DEFAULT_KERNELS[0] in header_rows else next(iter(header_rows), None)
    for c in cats:
        row = f"{('*' if c in D20_NAMED else ' ') + c:26}"
        for n in header_rows:
            row += f"{header_rows[n].get(c, 0):>14}"
        emit(row)
    row = f"{'ALL':26}" + "".join(f"{sum(header_rows[n].values()):>14}" for n in header_rows)
    emit(row)
    emit()
    emit("## 5. Rule table (first match wins; every rule was selftested against its own example)")
    for pat, cat, ex in RULES:
        emit(f"  {cat:26} <- {pat:32}   e.g. {ex}")
    if args.out:
        args.out.write_text("\n".join(out) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
