#!/usr/bin/env python3
"""SUPERSEDED: first source-line attempt; replaced by `region_census.py`, which resolves the `.loc`
inline stack so bf()/fb() conversions bill to the attn call site instead of amd_hip_bfloat16.h.

What fraction of the SHIPPING attention wave is matrix-shaped work? (D30 item 2.4, CPU half.)

The question item 2.4 has to answer before anyone writes a hot kernel: a `v_wmma_f32_16x16x16_bf16`
leg can retire only the DOT PARTS of the attention kernel, so the ceiling on the whole route is the
dot parts' share of the time — and that share has to be counted in the shipping binary, not modelled
(D19), and it has to name its denominator (D33: "an aggregate verdict that does not name its
denominator cannot fail").

Two earlier censuses cannot answer it:
  * `prototype/attn_valu_slim/isa_census.py` prices ONE loop (the pass-1 touch) and knows nothing of
    pass 2, which is why `attn_score_dot2` had to take pass 2 `by subtraction`.
  * `prototype/gdn_rec_isa/isa_census.py` censusies a whole kernel, but finds regions by loop
    fingerprint alone.
So this script does BOTH views and prints them separately:
  (1) a whole-kernel STATIC bill attributed to source regions by the compiler's own line table
      (`hipcc -gline-tables-only`), so a region is a named range of the shipping header and an
      instruction that resolves nowhere is printed as UNATTRIBUTED, never folded into "arithmetic";
  (2) a PER-TOUCH DYNAMIC bill: the pass-1 touch loop's body (the loop that contains the folded
      butterfly, attn_valu_slim's fingerprint) plus the pass-2 per-lane loop bodies found by their
      own shape, each category-tallied with the imported rule tables.

Every instruction class here is from `prototype/attn_score_dot2/isa_census.py` RULES, extended by
`prototype/attn_valu_slim/isa_census.py` RULES, extended by this file's EXTRA_RULES (matrix ops,
transcendentals, cross-lane reduces). The order matters: the more specific imported rule wins, and
`selftest_rules()` runs both parents' examples before this file's own counts are trusted.

Usage:
    python3 isa_census_attn.py                      # compile the engine TU, census both wave forms
    python3 isa_census_attn.py --asm FILE.s --out FILE.txt
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
split_blocks_full = slim.split_blocks_full
rule_hit = slim.rule_hit

ARCH = "gfx1151"
ENGINE_TU = REPO / "engine/kernels_attn.cpp"
SLIM_H = "engine/attn_slim_kernels.h"
OP_H = "prototype/attn_decode/attn_decode_kernels.h"
KERNELS = ["attn_slim_wave", "attn_tile_wave"]        # shipping default first, control second

DEFINES = {}
for _ln in (REPO / OP_H).read_text().splitlines():
    _m = re.match(r"#define\s+(\w+)\s+(\d+)", _ln.strip())
    if _m:
        DEFINES[_m.group(1)] = int(_m.group(2))
# `#define GQA (N_Q / N_KV)` is an expression, not a literal; evaluate the two-literal form rather
# than retyping the value, so a header edit moves the census with it.
_g = re.search(r"#define\s+GQA\s*\(\s*(\w+)\s*/\s*(\w+)\s*\)", (REPO / OP_H).read_text())
if _g:
    DEFINES["GQA"] = DEFINES[_g.group(1)] // DEFINES[_g.group(2)]
DH = DEFINES["DH"]
TILE = DEFINES["TILE"]
ATTN_TILE_L = DEFINES["ATTN_TILE_L"]
ATTN_TILE_M = DEFINES["ATTN_TILE_M"]
N_Q, N_KV, GQA = DEFINES["N_Q"], DEFINES["N_KV"], DEFINES["GQA"]

# --- region anchors: (region, file, start-text, end-text, multiplier) --------------------------
# A multiplier is how the region's code repeats: `touch` once per (row, lane, head) touch, `tile`
# once per TILE lanes, `block` once per block. Statements, not line numbers, so a header edit moves
# the region with the text instead of rotting the census.
ANCHORS = [
    ("geom.derive",       "const int row = gridDim.z - 1 - blockIdx.z;", "unsigned perm_addr[4];", "block"),
    ("prologue.q-load",   "unsigned perm_addr[4];",                       "---- pass 1:",          "block"),
    ("stage.k-load",      "for (int rr = s_lo + w; rr < s_hi; rr += nw) {", "__syncthreads();                                   // K sub-block", "touch"),
    ("dot.score",         "template <int ND>",                            "__device__ __forceinline__ float attn_slim_permute_add", "touch"),
    ("dot.fold",          "__device__ __forceinline__ float attn_slim_permute_add", "auto publish = [&](int rr, float dot) {", "touch"),
    ("score.round-store", "auto publish = [&](int rr, float dot) {",      "LEVER 1 — `nd` is warp-uniform", "touch"),
    ("sm.max-onepass",    "for (int t = 0; t < ns; ++t) m = fmaxf",       "for (int t = 0; t < ns; ++t) l += expf", "touch"),
    ("sm.sumexp-onepass", "for (int t = 0; t < ns; ++t) l += expf",       "for (int s_lo = 0; s_lo < ns; s_lo += ATTN_TILE_L)", "touch"),
    ("stage.v-load",      "for (int rr = s_lo + w; rr < s_hi; rr += nw) {", "__syncthreads();\n", "touch"),
    ("pv.dot-onepass",    "const float wt = bf(fb(expf(bf(s_sc[h][rr]) - m) / l));", "__syncthreads();                               // before the next", "touch"),
    ("stage.v-stream",    "for (int rr = base + w; rr < base + nt; rr += nw) {", "float mt = -1e30f;", "touch"),
    ("sm.max-tile",       "for (int slot = 0; slot < nt; ++slot) mt = fmaxf", "const float rescale = expf(m - mn);", "tile"),
    ("sm.sumexp-tile",    "for (int slot = 0; slot < nt; ++slot) {\n                e[slot] = expf", "l = l * rescale + ls;", "touch"),
    ("acc.rescale",       "l = l * rescale + ls;",                        "for (int slot = 0; slot < nt; ++slot) {\n                const hip_bfloat16* v", "tile"),
    ("pv.dot-tile",       "for (int slot = 0; slot < nt; ++slot) {\n                const hip_bfloat16* v", "m = mn;", "touch"),
    ("epilogue",          "if (s_row == 1) {",                             "the engine's wave entry point", "block"),
]

# Classes the two imported tables have no bucket for. Specific-first matching, as in the parents.
EXTRA_RULES = [
    ("v_wmma*",              "MATRIX.wmma"),
    ("v_mfma*",              "MATRIX.mfma"),
    ("v_exp_f32_e32",        "transcend.expf"),
    ("v_ldexp_f32*",         "transcend.expf"),
    ("v_rcp_f32_e32",        "transcend.divide"),
    ("v_div_*",              "transcend.divide"),
    ("v_max_f32*",           "softmax.max"),
    ("v_min_f32*",           "softmax.max"),
    ("v_permlane*",          "reduce.cross-lane"),
    ("ds_bpermute_b32",      "reduce.cross-lane"),
    ("ds_swizzle_b32",       "reduce.cross-lane"),
    ("v_cvt_pk_bf16_bf16",   "format.bf16-pack"),
    ("v_cvt_off_src_bf16",   "format.bf16-round"),
]
TOUCH_CATS = ("dot.", "fold.", "score.", "reduce.cross-lane")


def _compile(pat: str) -> str:
    rx = re.escape(pat)
    rx = rx.replace("\\?", "\x00")
    rx = rx.replace("\\ ", r"\s+")
    return "^" + rx.replace("\x00", "[^,]+")


EXTRA = [(re.compile(_compile(p)), c) for p, c in EXTRA_RULES]
EXTRA_EXAMPLES = [
    ("v_wmma_f32_16x16x16_bf16 v[24:31], v[8:15], v[16:23], v[24:31]", "MATRIX.wmma"),
    ("v_exp_f32_e32 v2, v1", "transcend.expf"),
    ("v_rcp_f32_e32 v5, v4", "transcend.divide"),
    ("v_max_f32_e32 v5, v2, v6", "softmax.max"),
    ("v_permlane_xor_b32 v1, v0, 1, v0, exec", "reduce.cross-lane"),
    ("ds_bpermute_b32 v0, v2, v0", "reduce.cross-lane"),
    ("v_cvt_pk_bf16_bf16 v2, v2, v3", "format.bf16-pack"),
    ("v_cvt_off_src_bf16 v1, v1, 0, v0", "format.bf16-round"),
]


def selftest_rules() -> int:
    bad = 0
    for pat, cat, ex in base.RULE_EXAMPLES + slim.RULE_EXAMPLES:
        got = category(*base_parse(ex)) if False else None
        del got
        if rule_hit(ex) not in (cat,) and category(mn_of(ex), ex) != cat:
            bad += 1
            print(f"  FAIL  {pat:28} example classified {category(mn_of(ex), ex)!r}, want {cat!r}")
    for ex, want in EXTRA_EXAMPLES:
        got = category(mn_of(ex), ex)
        if got != want:
            bad += 1
            print(f"  FAIL  {ex[:32]:32} classified {got!r}, want {want!r}")
    print(f"SELFTEST {'PASS' if bad == 0 else f'FAIL ({bad})'} — "
          f"{len(base.RULE_EXAMPLES) + len(slim.RULE_EXAMPLES) + len(EXTRA_EXAMPLES)} examples")
    return 0 if bad == 0 else 1


def mn_of(text: str) -> str:
    m = re.match(r"([a-z][a-z0-9_.]*)", text)
    return m.group(1) if m else "?"


def base_parse(text: str):
    return mn_of(text), text


def category(mn: str, text: str) -> str:
    """Extra rules first (matrix/transcendental/cross-lane classes the parents have no bucket for),
    then the imported attention rule, then gdn_rec_isa's prefix fallback. `unclassified` stays
    visible as unattributed."""
    for rx, cat in EXTRA:
        if rx.match(text):
            return cat
    c = rule_hit(text)
    if c and c != "other":
        return c
    if mn.startswith(base.SKIP_AS_WORK):
        return "encode.bookkeeping"
    for pre, cat in rec.FALLBACK:
        if mn.startswith(pre):
            return cat
    return "unclassified"


def slots(insts) -> int:
    return sum(1 for mn, _ in [(i[0], i[1]) for i in insts] if not mn.startswith(base.SKIP_AS_WORK))


# --- listing parsing with the line table -------------------------------------------------------
FILE_RE = re.compile(r'^\s*\.file\s+(\d+)\s+"(.*)"')
LOC_RE = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+")


def file_table(lines):
    return {m.group(1): m.group(2).strip('"') for ln in lines if (m := FILE_RE.match(ln))}


def with_locs(body: list[str], ft) -> list[tuple[str, str, str]]:
    """(mnemonic, text, 'path:line') for each instruction of one kernel, innermost inline frame."""
    out, cur = [], ""
    for ln in body:
        m = LOC_RE.match(ln)
        if m:
            note = ln.split(";", 1)[1].strip() if ";" in ln else ""
            top = note.split("@[", 1)[0].strip()
            pm = re.match(r"^(.*?):(\d+):\d+$", top) if top else None
            cur = f"{pm.group(1)}:{pm.group(2)}" if pm else f"{ft.get(m.group(1), '?')}:{m.group(2)}"
            continue
        m = base.INSTR_RE.match(ln)
        if m:
            out.append((m.group(1), ln.strip(), cur))
    return out


def blocks_with_locs(body: list[str], ft) -> list[dict]:
    """attn_valu_slim's block splitter, but each instruction keeps its source location."""
    blocks = split_blocks_full(body)
    locs = with_locs(body, ft)
    out = []
    i = 0
    for b in blocks:
        take = []
        while i < len(locs) and len(take) < len(b["insts"]):
            take.append(locs[i])
            i += 1
        nb = dict(b)
        nb["loc_insts"] = take
        out.append(nb)
    return out


def region_map() -> dict[str, tuple[str, str]]:
    src = (REPO / SLIM_H).read_text().splitlines()
    out: dict[str, tuple[str, str]] = {}
    for name, start_txt, end_txt, mult in ANCHORS:
        hits = [i for i, l in enumerate(src, 1) if start_txt in l]
        if not hits:
            raise SystemExit(f"anchor start not found in {SLIM_H}: {start_txt!r}")
        a = hits[0]
        if end_txt.startswith("the engine's"):
            b = min(a + 30, len(src))
        else:
            e2 = [i for i, l in enumerate(src, 1) if end_txt in l and i >= a]
            if not e2:
                raise SystemExit(f"anchor end not found in {SLIM_H}: {end_txt!r}")
            b = e2[0]
        for ln in range(a, b + 1):
            out[f"{SLIM_H}:{ln}"] = (name, mult)
    return out


# --- loop structure ----------------------------------------------------------------------------
def loop_groups(blocks: list[dict]) -> list[dict]:
    """Every back-edge loop, with its body blocks and its trip count derived from its own induction
    pair (gdn_rec_isa's rule), plus the shape features used to say WHICH loop it is."""
    pos = {base.norm_key(b["label"]): i for i, b in enumerate(blocks)}
    edges = []
    for i, b in enumerate(blocks):
        for _, ln in b["insts"]:
            m = re.match(r"s_cbranch_\w+\s+(\S+)", ln)
            if m and base.norm_key(m.group(1)) in pos and pos[base.norm_key(m.group(1))] <= i:
                edges.append((pos[base.norm_key(m.group(1))], i))
    owner: dict[int, tuple[int, int]] = {}
    for h, e in edges:
        for i in range(h, e + 1):
            if i not in owner or h > owner[i][0]:
                owner[i] = (h, e)
    groups: dict[tuple[int, int], list[dict]] = {}
    for i, b in enumerate(blocks):
        if i in owner:
            groups.setdefault(owner[i], []).append(b)
    out = []
    for (h, e), mine in groups.items():
        insts = [x for b in mine for x in b["loc_insts"]]
        out.append({"hdr": blocks[h]["label"], "blocks": mine, "insts": insts,
                    "bound": rec.derive_bound(mine), "depth": e - h + 1,
                    "permute": sum(1 for mn, _, _ in insts
                                   if mn.startswith(("ds_bpermute", "ds_swizzle"))),
                    "fma": sum(1 for mn, _, _ in insts if "fma" in mn),
                    "max": sum(1 for mn, _, _ in insts if mn.startswith("v_max")),
                    "exp": sum(1 for mn, _, _ in insts if mn.startswith("v_exp")),
                    "lds": sum(1 for mn, _, _ in insts if mn.startswith("ds_")),
                    "n": len(insts)})
    return sorted(out, key=lambda d: -d["n"])


def label_loop(lp: dict, i: int) -> str:
    if lp["permute"]:
        return "pass1.touch"
    if lp["max"] >= 4:
        return "pass2.max"
    if lp["exp"] >= 2:
        return "pass2.sumexp"
    if lp["fma"] >= 6 and lp["lds"] >= 1:
        return "pass2.pv"
    if lp["fma"] >= 4:
        return "pass2.acc-rescale"
    if lp["lds"] >= 1 and lp["fma"] == 0:
        return "stage.global-load"
    return f"UNATTRIBUTED.loop{i}"


def render(name: str, body: list[str], ft, rm, w) -> None:
    w(f"===== {name}")
    blocks = blocks_with_locs(body, ft)
    live, dead = rec.executed_blocks(blocks)
    flows = [b for b in live if b["flow"]]
    live = [b for b in live if not b["flow"]]
    insts = [x for b in live for x in b["loc_insts"]]
    tot = slots([(mn, t) for mn, t, _ in insts])
    w(f"  static bill: {len(insts)} lines, {tot} issue slots in {len(live)} executed blocks "
      f"({len(dead)} trap blocks and {len(flows)} %Flow blocks excluded)")

    # (1) whole-kernel static bill by source region
    reg: dict[str, dict] = {}
    for mn, text, loc in insts:
        r, mult = rm.get(loc, ("UNATTRIBUTED", "?"))
        d = reg.setdefault(r, {"mult": mult, "n": 0, "cats": Counter()})
        d["n"] += 1
        d["cats"][category(mn, text)] += 1
    w("  (1) STATIC bill by source region (multiplier: how the region repeats)")
    w(f"    {'region':19} {'mult':6} {'lines':>6} {'share':>6}  categories")
    for r in sorted(reg, key=lambda r: -reg[r]["n"]):
        d = reg[r]
        pct = 100.0 * d["n"] / len(insts)
        cats = "  ".join(f"{k}={v}" for k, v in sorted(d["cats"].items(), key=lambda kv: -kv[1])[:4])
        w(f"    {r:19} {d['mult']:6} {d['n']:6} {pct:5.1f}%  {cats}")
    w("")

    # (2) per-touch dynamic bill
    loops = loop_groups(live)
    w("  (2) loops by back edge (trip from the loop's own induction pair; label by shape)")
    for i, lp in enumerate(loops):
        w(f"    {label_loop(lp, i):21} {lp['hdr']:10} static={lp['n']:4} bound={lp['bound']} "
          f"perm={lp['permute']} fma={lp['fma']} max={lp['max']} exp={lp['exp']} lds={lp['lds']}")
    touch_loops = [lp for lp in loops if label_loop(lp, 0) == "pass1.touch"]
    p1 = touch_loops[0]["insts"] if touch_loops else []
    p2labs = ("pass2.max", "pass2.sumexp", "pass2.pv")
    p2 = [lp for lp in loops if label_loop(lp, 0) in p2labs]
    p2i = [x for lp in p2 for x in lp["insts"]]
    tiled = [lp for lp in loops if label_loop(lp, 0) in ("pass2.acc-rescale", "stage.global-load")]
    ti = [x for lp in tiled for x in lp["insts"]]
    for tag, group in (("pass-1 touch body", [("", p1)]),
                       ("pass-2 per-lane bodies", [("", x) for _, x in [("", p2i)]]),
                       ("pass-2 per-tile bodies (divide by 8)", [("", ti)])):
        gi = [x for _, x in group for x in x]
        if not gi:
            continue
        cats = Counter(category(mn, t) for mn, t, _ in gi)
        s = slots([(mn, t) for mn, t, _ in gi])
        w(f"    {tag}: {s} slots  " +
          "  ".join(f"{k}={v}" for k, v in sorted(cats.items(), key=lambda kv: -kv[1])))
    w("")

    # (3) category mix of the whole executed body
    cats = Counter(category(mn, t) for mn, t, _ in insts)
    w(f"  (3) whole-kernel category mix ({tot} slots; unclassified={cats.get('unclassified', 0)})")
    for k, v in sorted(cats.items(), key=lambda kv: -kv[1]):
        w(f"    {k:24} {v:5}  {100.0 * v / len(insts):5.1f}%")
    w("")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--asm", type=Path, default=None)
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args()
    if args.selftest:
        return selftest_rules()

    if args.asm:
        asm = args.asm
    else:
        td = Path(tempfile.mkdtemp(prefix="attnwmma-"))
        cmd = ["hipcc", "-O3", "-gline-tables-only", f"--offload-arch={ARCH}", "--save-temps",
               "-I", str(REPO), "-I", str(REPO / "engine"), "-c", str(ENGINE_TU),
               "-o", str(td / "k.o")]
        subprocess.run(cmd, check=True, capture_output=True, text=True, cwd=td)
        asm = next(td.glob("*-hip-amdgcn-amd-amdhsa-*.s"))
        print(f"# compiled: {' '.join(cmd)}\n# listing: {asm}\n", file=sys.stderr)

    lines = Path(asm).read_text().splitlines()
    kernels = rec.split_kernels_local(lines)
    ft = file_table(lines)
    rm = region_map()
    out: list[str] = []
    w = out.append
    w(f"listing {asm}")
    w(f"geometry from {OP_H}: DH={DH} TILE={TILE} ATTN_TILE_L={ATTN_TILE_L} ATTN_TILE_M={ATTN_TILE_M} "
      f"N_Q={N_Q} N_KV={N_KV} GQA={GQA}")
    w(f"regions: {len(set(v[0] for v in rm.values()))} named ranges anchored on statement text in {SLIM_H};")
    w("categories: attn_score_dot2 RULES < attn_valu_slim RULES < this file's EXTRA_RULES")
    w("")
    for k in KERNELS:
        if k not in kernels:
            w(f"===== {k}: NOT IN LISTING (here: {sorted(kernels)})")
            continue
        render(k, kernels[k], ft, rm, w)
    txt = "\n".join(out)
    print(txt)
    if args.out:
        args.out.write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
