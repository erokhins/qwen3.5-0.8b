#!/usr/bin/env python3
"""Census the per-TOKEN-STEP instruction bill of the shipping GDN recurrence, before and after.

Modelled on `prototype/attn_valu_slim/isa_census.py` — same .s parser and slot rules (imported, not
copied, from `prototype/attn_score_dot2/isa_census.py` and that script), same rule-selftest rule,
same control discipline. What differs is the UNIT, and the difference is itself a finding:

THE UNIT. `engine/kernels_gdn.cpp::gdn_rec` launches `rec_kernel` ONCE PER (TOKEN, GDN LAYER):
`rec_kernel<<<NH=16, D=128>>>`, one block per value head, one thread per d_v column, wave32, so a
launch is 16 blocks x 4 wavefronts = 64 wavefronts and one wavefront's instruction stream is the
whole story. There is no token loop inside the kernel, so
    * the prologue — two `block_sum` block reductions for the k/q norms (20 lowered cross-lane
      permutes, 11 barriers), the LDS staging, the scalar `expf` — runs exactly ONCE per token-step
      and amortises against nothing;
    * the only repeated structure is loop A (step-2 dot `r += Sh[i*D+j]*ks[i]`) and loop B (state
      update plus step-3 dot), each walking D = 128 rows, unrolled by LLVM 16x and 8x.
So: token-step = prologue + trips_A x loopA + trips_B x loopB, per wavefront; a launch bills that
x 64 wavefronts; the family bills that x 18 launches/token x tokens (18,432 launches at 1024 lanes,
73,728 at 4096).

CONTROL DISCIPLINE. `rec0_kernel` is the shipped kernel copied verbatim, compiled in the same TU, and
`--engine-asm` compares the engine's own listing against this TU's copy of the same source. Both must
agree before any lever delta means anything; the script prints the control verdict first and says
VOID if it fails.

Usage:
    python3 isa_census.py                                  # compile isa_tu.cpp, census every MODE
    python3 isa_census.py --asm FILE.s --kernels rec_kernel
    python3 isa_census.py --engine-asm tmp/gdn_isa/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s
    python3 isa_census.py --loop loopB                     # dump a loop body with categories
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


slim = _load("attn_valu_slim_census", REPO / "prototype/attn_valu_slim/isa_census.py")
base = slim.base                       # the generic .s parser and slot tally, reused not restated
split_blocks_full = slim.split_blocks_full

ARCH = "gfx1151"
D = 128                       # gdn_decode_kernels.h: #define D 128 — one thread per column
NH = 16                       # value heads = blocks per launch
WAVE = 32                     # .amdhsa_wavefront_size32 1; block_sum uses blockDim.x >> 5
WF_PER_LAUNCH = (D // WAVE) * NH
SHIPPED = "rec_kernel"
MODE0 = "rec0_kernel"

# Category rules, FIRST MATCH WINS, operand-sensitive where one opcode serves two structures.
# Mini-language (inherited from the slim script): `?` matches a register name, a space matches
# one-or-more spaces. Every rule carries an example line it MUST match, checked at import — four
# silently-unmatched rules once printed a clean-looking attention table with the two interesting
# categories missing, and `prototype/gdn_collapse/run.py` printed PASS over zero parsed CASE lines.
RULES: list[tuple[str, str, str]] = [
    # class: lowered cross-lane reduction (`block_sum`'s __shfl_down_sync chain)
    ("ds_bpermute_b32",               "xlane.permute",        "ds_bpermute_b32 v11, v7, v10"),
    ("ds_swizzle_b32",                "xlane.permute",        'ds_swizzle_b32 v11, v10 offset:swizzle(BITMASK_PERM,"1pppp")'),
    ("s_cmp_eq_u32 exec_lo, -1",      "xlane.exec-guard",     "s_cmp_eq_u32 exec_lo, -1"),
    ("s_cmp_lg_u32 exec_lo, -1",      "xlane.exec-guard",     "s_cmp_lg_u32 exec_lo, -1"),
    ("s_and_saveexec_b32",            "xlane.exec-guard",     "s_and_saveexec_b32 s4, s3"),
    ("s_or_b32 exec_lo",              "xlane.exec-guard",     "s_or_b32 exec_lo, exec_lo, s4"),
    ("s_and_not1_b32",                "xlane.exec-guard",     "s_and_not1_b32 vcc_lo, exec_lo, s0"),
    ("s_and_b32",                     "xlane.exec-guard",     "s_and_b32 vcc_lo, exec_lo, s21"),
    ("s_cbranch_vccnz",               "xlane.exec-guard",     "s_cbranch_vccnz .LBB2_65"),
    ("s_cbranch_vccz",                "xlane.exec-guard",     "s_cbranch_vccz .LBB2_17"),
    ("s_cbranch_scc1",                "xlane.exec-guard",     "s_cbranch_scc1 .LBB2_61"),
    ("s_cbranch_execz",               "xlane.exec-guard",     "s_cbranch_execz .LBB2_12"),
    ("s_cbranch_execnz",              "xlane.exec-guard",     "s_cbranch_execnz .LBB2_12"),
    ("s_mov_b32 s?, -1",              "xlane.exec-guard",     "s_mov_b32 s21, -1"),
    ("v_cmp_gt_u32_e32",              "xlane.range-select",   "v_cmp_gt_u32_e32 vcc_lo, 24, v1"),
    ("v_cmp_ne_u32_e32 vcc_lo, 31,",  "xlane.range-select",   "v_cmp_ne_u32_e32 vcc_lo, 31, v1"),
    ("v_cndmask_b32_e64 v?, 0,",      "xlane.range-select",   "v_cndmask_b32_e64 v9, 0, 8, vcc_lo"),
    ("v_mbcnt_lo_u32_b32",            "xlane.lane-id",        "v_mbcnt_lo_u32_b32 v0, vcc_lo, 0"),
    ("v_add_nc_u32_e32",              "xlane.addr-math",      "v_add_nc_u32_e32 v9, v9, v1"),
    ("v_lshlrev_b32_e32 v?, 2,",      "xlane.addr-math",      "v_lshlrev_b32_e32 v7, 2, v9"),
    ("v_add_f32_e32",                 "xlane.add",            "v_add_f32_e32 v10, v10, v11"),
    ("v_dual_add_f32",                "xlane.add",            "v_dual_add_f32 v9, v9, v10 :: v_dual_lshlrev_b32 v10, 2, v11"),
    ("ds_store_b32",                  "xlane.lds-round-trip", "ds_store_b32 v11, v10 offset:1024"),
    ("ds_store_2addr_stride64_b32",   "xlane.lds-round-trip", "ds_store_2addr_stride64_b32 v5, v6, v4 offset1:2"),
    ("ds_load_b32",                   "xlane.lds-round-trip", "ds_load_b32 v10, v10 offset:1024"),
    ("s_barrier",                     "barrier.sync",         "s_barrier"),
    ("buffer_gl0_inv",                "barrier.invalidate",   "buffer_gl0_inv"),
    ("s_waitcnt vmcnt(0) lgkmcnt(0)", "barrier.drain",        "s_waitcnt vmcnt(0) lgkmcnt(0)"),
    # the two state walks
    ("global_load_b32",               "state.load",           "global_load_b32 v17, v[7:8], off offset:1024"),
    ("global_store_b32",              "state.store",          "global_store_b32 v[24:25], v8, off"),
    ("ds_load_b128",                  "state.coef-lds-load",  "ds_load_b128 v[7:10], v31"),
    ("v_fmac_f32_e32",                "state.fma",            "v_fmac_f32_e32 v1, v15, v7"),
    ("v_fma_f32",                     "state.fma",            "v_fma_f32 v1, -v6, v1, v5"),
    ("v_dual_fmac_f32",               "state.fma",            "v_dual_fmac_f32 v9, v15, v11 :: v_dual_add_nc_u32 v10, 2, v10"),
    ("v_dual_fma_f32",                "state.fma",            "v_dual_fma_f32 v9, v15, v11, v9 :: v_dual_mov_b32 v12, 0"),
    ("v_mul_f32",                     "state.mul",            "v_mul_f32_e32 v9, 0x45800000, v8"),
    ("v_dual_mul_f32",                "state.mul",            "v_dual_mul_f32 v9, v10, v11 :: v_dual_mov_b32 v12, 0"),
    ("v_rsq_f32_e32",                 "norm.rsqrt",           "v_rsq_f32_e32 v8, v8"),
    ("v_exp_f32_e32",                 "norm.expf",            "v_exp_f32_e32 v6, s2"),
    ("v_ldexp_f32",                   "norm.expf",            "v_ldexp_f32 v6, v6, s2"),
    ("s_rndne_f32",                   "norm.expf",            "s_rndne_f32 s5, s2"),
    ("s_fmamk_f32",                   "norm.expf",            "s_fmamk_f32 s3, s4, 0x3fb8aa3b, s3"),
    ("s_mul_f32",                     "norm.expf",            "s_mul_f32 s2, s4, 0x3fb8aa3b"),
    ("s_add_f32",                     "norm.expf",            "s_add_f32 s2, s2, s3"),
    ("s_sub_f32",                     "norm.expf",            "s_sub_f32 s2, s2, s5"),
    ("s_cvt_i32_f32",                 "norm.expf",            "s_cvt_i32_f32 s2, s5"),
    ("s_cmp_nlt_f32",                 "norm.expf",            "s_cmp_nlt_f32 s4, 0xc2ce8ed0"),
    ("s_cmp_ngt_f32",                 "norm.expf",            "s_cmp_ngt_f32 s4, 0x42b17218"),
    ("s_cselect_b32",                 "norm.expf",            "s_cselect_b32 vcc_lo, -1, 0"),
    ("v_cndmask_b32_e32",             "norm.expf",            "v_cndmask_b32_e32 v6, 0, v6, vcc_lo"),
    # address math / loop bookkeeping / issue padding
    ("v_lshl_or_b32",                 "addr.math",            "v_lshl_or_b32 v4, s2, 7, v0"),
    ("v_lshlrev_b64",                 "addr.math",            "v_lshlrev_b64 v[8:9], 2, v[4:5]"),
    ("v_add_co_u32",                  "addr.math",            "v_add_co_u32 v7, vcc_lo, v5, s2"),
    ("v_add_co_ci_u32_e64",           "addr.math",            "v_add_co_ci_u32_e64 v8, null, s3, v6, vcc_lo"),
    ("v_add_lshl_u32",                "addr.math",            "v_add_lshl_u32 v10, v11, v12, 2"),
    ("s_lshl_b32",                    "addr.math",            "s_lshl_b32 s3, s2, 2"),
    ("s_clause",                      "vmem.clause",          "s_clause 0x7"),
    ("s_add_i32",                     "loop.salu",            "s_add_i32 s2, s2, 8"),
    ("s_add_u32",                     "loop.salu",            "s_add_u32 s2, s2, 0x2000"),
    ("s_addc_u32",                    "loop.salu",            "s_addc_u32 s3, s3, 0"),
    ("s_cmpk_eq_i32",                 "loop.salu",            "s_cmpk_eq_i32 s2, 0x80"),
    ("s_cmp_eq_u32",                  "loop.salu",            "s_cmp_eq_u32 s2, 0x10000"),
    ("s_cbranch_scc0",                "loop.salu",            "s_cbranch_scc0 .LBB2_20"),
    ("s_mov_b32",                     "salu.mov",             "s_mov_b32 s2, 0"),
    ("s_mov_b64",                     "salu.mov",             "s_mov_b64 s[0:1], s[2:3]"),
    ("v_mov_b32_e32",                 "valu.mov",             "v_mov_b32_e32 v31, s5"),
    ("s_load_b256",                   "salu.kernarg",         "s_load_b256 s[2:33], s[0:1], 0"),
    ("s_load_b32",                    "salu.kernarg",         "s_load_b32 s0, s[0:1], 0x4c"),
    ("v_readfirstlane_b32",           "salu.offload",         "v_readfirstlane_b32 s4, v1"),
    ("s_bfe_u32",                     "norm.warp-count",      "s_bfe_u32 s0, s0, 0xb0005"),
    ("s_ashr_i32",                    "norm.warp-count",      "s_ashr_i32 s0, s0, 5"),
    ("v_and_b32_e32",                 "norm.lane-id",         "v_and_b32_e32 v0, 31, v0"),
    ("v_lshrrev_b32_e32",             "norm.lane-id",         "v_lshrrev_b32_e32 v11, 3, v0"),
    ("v_cmp_eq_u32_e64",              "norm.lane-id",         "v_cmp_eq_u32_e64 s22, 0, v0"),
    ("v_cmp_gt_u32_e64",              "norm.lane-id",         "v_cmp_gt_u32_e64 s1, s0, v0"),
    ("v_ashrrev_i32_e32",             "norm.expf",            "v_ashrrev_i32_e32 v1, 31, v1"),
]
RULE_EXAMPLES = [(p, c, ex) for p, c, ex in RULES]


def _compile(pat: str) -> str:
    """`?` = a register name, a space = one-or-more spaces (the slim script's mini-language)."""
    rx = re.escape(pat)
    rx = rx.replace("\\?", "\x00")
    rx = rx.replace("\\ ", r"\s+")
    return "^" + rx.replace("\x00", "[^,]+")


_COMPILED = [(re.compile(_compile(p)), c) for p, c, _ in RULES]


def rule_hit(text: str) -> str:
    for rx, cat in _COMPILED:
        if rx.match(text):
            return cat
    return ""


def selftest_rules() -> int:
    bad = []
    for pat, cat, ex in RULE_EXAMPLES:
        if not re.match(_compile(pat), ex):
            bad.append(f"{pat!r} does not match its example {ex!r}")
        elif rule_hit(ex) != cat:
            bad.append(f"{pat!r} matches {ex!r} but buckets as {rule_hit(ex)!r}, not {cat!r}")
    if bad:
        for b in bad:
            print("RULE SELFTEST FAIL:", b, file=sys.stderr)
        raise SystemExit(2)
    return len(RULES)


selftest_rules()

# Anything not matched by a rule still has to be visible. These fallbacks are prefix-based; a
# mnemonic that reaches the end lands in `unclassified` and is printed, never silently dropped.
FALLBACK = [("v_dual", "dual.other"), ("s_nop", "encode.sched"), ("s_delay_alu", "encode.sched"),
            ("s_waitcnt", "encode.wait"), ("s_branch", "salu.branch"), ("s_endpgm", "salu.exit"),
            ("s_trap", "dead.trap"), ("s_mov_rel", "salu.mov"), ("v_movrels", "UNBOUND.dyn-reg-index"),
            ("v_cvt", "UNBOUND.convert"), ("v_lshl", "addr.math"), ("v_add", "addr.math"),
            ("v_lshr", "addr.math"), ("s_mov_b64", "salu.mov"), ("s_nop", "encode.sched")]


def category(mnemonic: str, text: str) -> str:
    c = rule_hit(text)
    if c:
        return c
    if mnemonic.startswith(base.SKIP_AS_WORK):
        return "encode.bookkeeping"
    for pre, cat in FALLBACK:
        if mnemonic.startswith(pre):
            return cat
    return "unclassified"


def tally_cats(insts: list[tuple[str, str]]) -> Counter:
    t: Counter = Counter()
    for mn, ln in insts:
        t[category(mn, ln)] += 1
    return t


def slots(insts: list[tuple[str, str]]) -> int:
    """Issue slots. `s_waitcnt`/`s_delay_alu`/`s_nop` are bookkeeping, and a `v_dual_X :: v_dual_Y`
    line is ONE slot carrying two VALU ops — the base tally's rule, so both ops' authors agree."""
    return sum(1 for mn, _ in insts if not mn.startswith(base.SKIP_AS_WORK))


def class_slots(insts: list[tuple[str, str]]) -> Counter:
    t: Counter = Counter()
    for mn, _ in insts:
        t["encode" if mn.startswith(base.SKIP_AS_WORK) else base.classify(mn)] += 1
    return t


# ----------------------------------------------------------------------------------------------
def executed_blocks(blocks: list[dict]) -> tuple[list[dict], list[dict]]:
    """Which blocks a full, uniform wavefront actually runs.

    The one class of dead code here is LLVM's divergent-`__shfl_down_sync` fallback: every shuffle
    site is guarded by `s_cmp_eq_u32 exec_lo, -1` / `s_cmp_lg_u32 exec_lo, -1` and jumps to a block
    whose first act is `s_trap 2` when exec is not the full wave. `rec_kernel` has no divergence at
    any shuffle site, so those blocks never run. NOTE this is the OPPOSITE of the attention census,
    where the `%Flow` blocks were the untaken path: here the Flow block IS the fast path (it is the
    convergence test after the shuffle), so Flow blocks are KEPT and only trap blocks are dropped.
    """
    live, dead = [], []
    for b in blocks:
        if any(mn == "s_trap" for mn, _ in b["insts"]) or \
           any("divergent unreachable" in a for a in b.get("anns", [])):
            dead.append(b)
        else:
            live.append(b)
    return live, dead


def annotate_loops(body: list[str]) -> list[str]:
    """Move a label line's trailing loop annotation onto its own preceding comment line.

    LLVM prints a block's loop annotation in two places depending on whether the block also carries a
    `%bb.N:` comment: for `rec_kernel`'s two walks it is `.LBB2_20:  ; =>This Inner Loop Header:
    Depth=1` (same line), and the splitter reads annotations from comment lines only, so without this
    the two walks look like straight-line blocks and the roll-up silently bills them once instead of
    8x and 16x. Belt and braces: `find_walks` also finds loops structurally by back edge."""
    out: list[str] = []
    for ln in body:
        m = base.LABEL_RE.match(ln)
        tail = ln.split(";", 1)[1].strip() if ";" in ln else ""
        if m and ("Loop Header" in tail or "in Loop:" in tail):
            out.append("; " + tail)
            out.append(m.group(1) + ":")
        else:
            out.append(ln)
    return out


def find_walks(blocks: list[dict]) -> dict[str, dict]:
    """The two state walks, found structurally and then identified by fingerprint.

    This listing (ROCm 7's printer for this TU) carries no `in Loop: Header=` annotations on member
    blocks — only `; =>This Inner Loop Header: Depth=1` on the header label line — so loop membership
    comes from the CFG: a `s_cbranch_* <label>` whose target is an EARLIER live block is a back edge,
    and the blocks from that target through the branch are the loop body. Nested loops would take the
    innermost header; `rec_kernel` has none, and the shuffles' backward `s_branch` lives in a trap
    block that `executed_blocks` already dropped.

    Then the fingerprint: loop B owns the state stores, loop A loads the state and stores none. Trip
    counts are DERIVED twice and cross-checked — once from "the walk touches each of the D columns
    once" (trips = D / state accesses per trip) and once from the loop's own induction pair
    (`s_add_i32 s2, s2, 8` with `s_cmpk_eq_i32 s2, 0x80`).
    """
    pos = {base.norm_key(b["label"]): i for i, b in enumerate(blocks)}
    loops: list[tuple[int, int]] = []
    for i, b in enumerate(blocks):
        for _, ln in b["insts"]:
            m = re.match(r"s_cbranch_\w+\s+(\S+)", ln)
            if m and base.norm_key(m.group(1)) in pos and pos[base.norm_key(m.group(1))] <= i:
                loops.append((pos[base.norm_key(m.group(1))], i))
    owner: dict[int, tuple[int, int]] = {}
    for h, e in loops:
        for i in range(h, e + 1):
            if i not in owner or h > owner[i][0]:
                owner[i] = (h, e)
    if not owner:                        # no back edge found: fall back on the annotations
        for i, b in enumerate(blocks):
            if b["parents"]:
                owner[i] = (i, i)
    groups: dict[tuple[int, int], list[dict]] = {}
    for i, b in enumerate(blocks):
        if i in owner:
            groups.setdefault(owner[i], []).append(b)

    out: dict[str, dict] = {}
    for (h, e), mine in groups.items():
        insts = [ln for b in mine for _, ln in b["insts"]]
        loads = sum(1 for t in insts if t.startswith("global_load"))
        stores = sum(1 for t in insts if t.startswith("global_store"))
        if stores:
            name, per_trip = "loopB_state_update", stores
        elif loads and any("fma" in t for t in insts):
            name, per_trip = "loopA_state_dot", loads
        else:
            continue
        if per_trip == 0 or D % per_trip:
            raise SystemExit(f"{blocks[h]['label']}: {per_trip} state accesses/trip does not divide D={D}")
        bound = derive_bound(mine)
        trips = D // per_trip
        if bound is not None and bound != trips:
            raise SystemExit(f"{blocks[h]['label']}: derived trips {trips} != loop-bound trips {bound}")
        out[name] = {"hdr": blocks[h]["label"], "depth": 1, "per_trip": per_trip, "trips": trips,
                     "bound": bound, "blocks": mine}
    return out


def derive_bound(blks: list[dict]) -> int | None:
    """trips from the loop's own induction variable: an `s_add* sN, sN, STEP` and a compare against a
    constant against the same register. Returns None when the loop has no such pair (e.g. a fully
    unrolled walk), which the caller must then not treat as 1."""
    adds: dict[str, int] = {}
    cmps: dict[str, int] = {}
    for _, ln in [i for b in blks for i in b["insts"]]:
        m = re.match(r"s_add[ui]?\d*_?(i?32)\s+(s\d+),\s*\2,\s+(0x[0-9a-f]+|\d+)", ln)
        if m:
            adds[m.group(2)] = int(m.group(3), 0)
        m = re.match(r"s_cmp[kl]?[a-z0-9_]*\s+(s\d+),\s*(0x[0-9a-f]+|\d+)", ln)
        if m:
            cmps.setdefault(m.group(1), int(m.group(2), 0))
    for reg in cmps:
        if reg in adds and cmps[reg] % adds[reg] == 0:
            return cmps[reg] // adds[reg]
    return None


def census_kernel(body: list[str]) -> dict:
    blocks = split_blocks_full(annotate_loops(body))
    live, dead = executed_blocks(blocks)
    walks = find_walks(live)
    if len(walks) != 2:
        raise SystemExit(f"expected exactly 2 state walks, found {sorted(walks)} — the fingerprint "
                         f"or the kernel changed; refusing to roll up a partial bill")
    in_loop = {id(b) for v in walks.values() for b in v["blocks"]}
    prologue = [b for b in live if id(b) not in in_loop]

    def insts(blks):
        return [ln for b in blks for ln in b["insts"]]

    pro = insts(prologue)
    la = insts(walks["loopA_state_dot"]["blocks"])
    lb = insts(walks["loopB_state_update"]["blocks"])
    ta, tb = walks["loopA_state_dot"]["trips"], walks["loopB_state_update"]["trips"]
    step = pro + la * ta + lb * tb
    return {"blocks": blocks, "live": live, "dead": dead, "walks": walks,
            "regions": {"prologue": pro, "loopA": la, "loopB": lb},
            "trips": {"loopA": ta, "loopB": tb}, "step": step,
            "dead_lines": sum(len(b["insts"]) for b in dead)}


def split_kernels_local(lines: list[str]) -> dict[str, list[str]]:
    """Kernel label -> its lines, keyed by the demangled kernel NAME.

    Not `base.split_kernels`: its mangled-name regex needs a digit between the kernel name and the
    parameter list, which is true of `_Z14attn_tile_wave8CoreArgs` but NOT of
    `_Z10rec_kernelPKfS0_...` — it backtracks and returns the name `rec_kernelPKfS`. The Itanium
    length prefix is exact, so read it instead: `_Z <n> <n chars>` (or `N ... E` for a namespace).
    """
    out: dict[str, list[str]] = {}
    name = None
    for ln in lines:
        m = re.match(r"^(_Z[A-Za-z0-9_]+):(\s*;.*)?$", ln)
        if m:
            name = demangle_fn(m.group(1))
            out.setdefault(name, [])
            continue
        if name is None:
            continue
        if ln.startswith(".Lfunc_end"):
            name = None
            continue
        out[name].append(ln)
    return out


def demangle_fn(sym: str) -> str:
    """`_Z10rec_kernelPKfS0_...` -> `rec_kernel`; `_ZN6gdn_rc19rec_collapse_kernelE...` ->
    `gdn_rc::rec_collapse_kernel`. Names only — no argument decoding."""
    s, i = sym, 2
    if s[i] == "N":
        i += 1
    parts = []
    while i < len(s) and s[i] != "E":
        m = re.match(r"(\d+)", s[i:])
        if not m:
            break
        n = int(m.group(1))
        parts.append(s[i + len(m.group(1)):i + len(m.group(1)) + n])
        i += len(m.group(1)) + n
    return "::".join(parts)


def render(name: str, c: dict, w) -> None:
    w(f"===== {name}")
    for lname in ("loopA_state_dot", "loopB_state_update"):
        lv = c["walks"][lname]
        w(f"  walk {lname:20} hdr={lv['hdr']} depth={lv['depth']} state-accesses/trip={lv['per_trip']}"
          f" trips={lv['trips']} (D/{lv['per_trip']}, loop-bound says {lv['bound']})")
    w(f"  never-executed lines (divergent-shuffle trap blocks): {c['dead_lines']}")
    for reg in ("prologue", "loopA", "loopB"):
        insts_ = c["regions"][reg]
        mult = 1 if reg == "prologue" else c["trips"][reg]
        w(f"  {reg:9} static {len(insts_):4} lines  slots {slots(insts_):4}  x{mult:3} trips"
          f"  -> {slots(insts_)*mult:6} slots/token-step")
    st = c["step"]
    w(f"  TOKEN-STEP, per wavefront: {len(st)} lines issued, {slots(st)} issue slots")
    w(f"  TOKEN-STEP, per launch   : {slots(st)*WF_PER_LAUNCH:,} issue slots"
      f"  ({WF_PER_LAUNCH} wavefronts = {NH} blocks x {D//WAVE})")
    w(f"  pipe slots               : {dict(sorted(class_slots(st).items()))}")
    cats = tally_cats(st)
    unc = cats.get("unclassified", 0)
    w(f"  category rollup per token-step per wavefront (unclassified={unc}):")
    for k, v in sorted(cats.items(), key=lambda kv: (-kv[1], kv[0])):
        w(f"    {k:<24} {v:6}  {100.0 * v / slots(st):5.1f}%")
    w("")


def compare(a: dict, b: dict, an: str, bn: str, w) -> bool:
    ca, cb = tally_cats(a["step"]), tally_cats(b["step"])
    same = ca == cb and slots(a["step"]) == slots(b["step"])
    w(f"  {an} vs {bn}: {'IDENTICAL' if same else 'DIFFERS'} "
       f"({slots(a['step'])} vs {slots(b['step'])} slots/token-step)")
    for k in sorted(set(ca) | set(cb)):
        if ca.get(k, 0) != cb.get(k, 0):
            w(f"    {k:<24} {ca.get(k, 0):6} vs {cb.get(k, 0):6}   {cb.get(k, 0)-ca.get(k, 0):+6}")
    return same


def resources(src: Path, outdir: Path) -> dict[str, dict[str, str]]:
    return slim.resources(src, outdir)


def compile_tu(src: Path, outdir: Path) -> Path:
    """`--save-temps` writes the intermediates into the CURRENT directory, not next to `-o`, so run
    the compile with cwd=outdir (the first version of this globbed outdir and found nothing)."""
    cmd = ["hipcc", "-O3", f"--offload-arch={ARCH}", "--save-temps", "-I", str(src.parent),
           "-c", str(src), "-o", str(outdir / "isa_tu.o")]
    p = subprocess.run(cmd, capture_output=True, text=True, cwd=outdir)
    if p.returncode:
        print(p.stdout + p.stderr, file=sys.stderr)
        raise SystemExit("compile failed")
    cands = sorted(outdir.glob("*-hip-amdgcn-amd-amdhsa-gfx1151.s"))
    if not cands:
        raise SystemExit(f"no device assembly produced in {outdir}")
    return cands[0]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--src", type=Path, default=HERE / "isa_tu.cpp")
    ap.add_argument("--asm", type=Path, default=None)
    ap.add_argument("--engine-asm", type=Path, default=None)
    ap.add_argument("--kernels", default="")
    ap.add_argument("--out", type=Path, default=None)
    ap.add_argument("--dump", default=None, help="prologue|loopA|loopB: dump lines with categories")
    args = ap.parse_args()

    out: list[str] = []
    w = out.append
    keep = False
    if args.asm:
        asm = Path(args.asm)
    else:
        td = tempfile.TemporaryDirectory()
        asm = compile_tu(args.src.resolve(), Path(td.name))
        keep = True
        if args.out:
            (args.out.parent / (args.out.stem + ".s")).write_text(asm.read_text())

    kernels = split_kernels_local(Path(asm).read_text().splitlines())
    names = [n for n in (args.kernels.split(",") if args.kernels else []) if n] or \
            [n for n in (SHIPPED, MODE0) if n in kernels]
    missing = [n for n in names if n not in kernels]
    if missing:
        raise SystemExit(f"kernel(s) {missing} not in listing; kernels here: {sorted(kernels)}")

    w(f"listing  {asm}")
    w(f"rules    {selftest_rules()} patterns, selftest PASS (each matches its own example line)")
    w(f"unit     token-step = one rec_kernel launch = one (token, GDN layer); grid {NH}x{D} threads,"
      f" wave{WAVE} -> {WF_PER_LAUNCH} wavefronts; trip counts derived from the listing")
    w("")

    if args.dump:
        key = {"prologue": "prologue", "loopA": "loopA", "loopB": "loopB"}[args.dump]
        for n in names[:1]:
            c = census_kernel(kernels[n])
            for mn, ln in c["regions"][key]:
                w(f"  {category(mn, ln):<24} {ln}")
        print("\n".join(out))
        return 0

    cens = {n: census_kernel(kernels[n]) for n in names}

    w("===== CONTROL")
    ok = True
    if SHIPPED in cens and MODE0 in cens:
        ok = compare(cens[SHIPPED], cens[MODE0], "shipped rec_kernel", "MODE-0 copy rec0_kernel", w)
    else:
        w("  (MODE-0 copy not in this listing — control not taken)")
    if args.engine_asm and SHIPPED in cens:
        eng = split_kernels_local(Path(args.engine_asm).read_text().splitlines())
        if SHIPPED not in eng:
            raise SystemExit(f"{SHIPPED} not in the engine listing")
        ok = compare(census_kernel(eng[SHIPPED]), cens[SHIPPED],
                     "engine-TU rec_kernel", "prototype-TU rec_kernel", w) and ok
    elif args.engine_asm:
        w(f"  ({SHIPPED} not in this TU, so the engine listing could not be compared)")
    w(f"  control verdict: {'OK — deltas below mean what they say' if ok else 'FAILED — every delta below is VOID'}")
    w("")

    for n in names:
        render(n, cens[n], w)

    levers = [n for n in names if n not in (SHIPPED, MODE0)]
    if levers:
        w("===== DELTA vs shipped, per token-step per wavefront")
        ref, ref_slots = tally_cats(cens[SHIPPED]["step"]), slots(cens[SHIPPED]["step"])
        for n in levers:
            t, s = tally_cats(cens[n]["step"]), slots(cens[n]["step"])
            w(f"  {n}: {s} slots (shipped {ref_slots}, {(s-ref_slots)/ref_slots:+.1%})")
            for c in sorted(set(t) | set(ref)):
                if t.get(c, 0) != ref.get(c, 0):
                    w(f"    {c:<24} {ref.get(c, 0):6} -> {t.get(c, 0):6}  {t.get(c,0)-ref.get(c,0):+6}")
        w("")

    res = {} if args.asm else resources(args.src, Path(td.name))
    if res:
        w("===== RESOURCES (VGPR / SGPR / LDS / spill, from -Rpass-analysis=kernel-resource-usage)")
        for k, v in res.items():
            short = re.sub(r"^_Z\d+|[0-9].*$", "", k) or k
            w(f"  {short:24} " + "  ".join(f"{kk}={vv}" for kk, vv in sorted(v.items())))
    txt = "\n".join(out)
    print(txt)
    if args.out:
        args.out.write_text(txt + "\n")
    if keep:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
