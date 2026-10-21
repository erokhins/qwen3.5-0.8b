#!/usr/bin/env python3
"""m16_census.py — D57's gate for item 2.6: slots per `(row, lane, head)` touch at M = 16.

COMPILE-ONLY. Reads GCN `.s` listings produced by `build_m16.sh` / `build.sh`. Launches nothing, takes
no timing, and refuses to print a `ms` figure (there is none to print).

THE UNIT, NAMED BEFORE ANY NUMBER IS QUOTED
-------------------------------------------
One `(row, lane, head)` TOUCH = one query row x one KV lane x one head = 256 MACs, the same unit
`prototype/attn_valu_slim/isa_census.py` bills the shipped `attn_slim_wave` touch loop in (45 slots:
28 VALU + 7 LDS + 10 SALU, `prototype/attn_valu_slim/results.md` §2), and the same unit
`prototype/attn_wmma/results.md` §3 converts its per-tile feed into (13.5 slots per 16x16x16 tile / 16
touch-equivalents = 0.84).

A SLOT is one issue slot: one instruction line, with a `v_dual_X :: v_dual_Y` packet counted ONCE and
`s_waitcnt` / `s_nop` / `s_delay_alu` counted as ENCODE-ONLY (not work). That is verbatim
`prototype/attn_score_dot2/isa_census.py`'s `tally()` rule, imported rather than re-spelled, so this
census and the 45 cannot drift in definition. `s_waitcnt` is still reported, as its own category, and
the totals are printed both with and without it — a WMMA kernel issues many more waits than a VALU one,
and a comparison that silently dropped them would be a flattery.

PASS 1, NOT THE WHOLE KERNEL. Both the 45 and these figures are the PASS-1 touch stream. That is the
denominator D57 settled when it closed the whole-kernel collision (the calibrated counter read
`attn_slim_wave` at 2,265.7 M VALU/pass, model A / whole-kernel by 3.2x over pass-1-only, so pass 1 is
~61 % of the kernel's VALU). `prototype/attn_wmma/region_census.py`'s whole-kernel roll-up is NOT used
and `--calib` on it still aborts; see `prototype/attn_wmma/results.md` §7 and the note in `--calib` below.

WHAT IS DIFFERENCED, AND WHY A CATEGORY TABLE ALONE CANNOT ANSWER THIS
---------------------------------------------------------------------
`ds_load_b128` appears in both the operand feed and the drain, so a mnemonic-level category table cannot
say which term is expensive. The variants in `wmma_m16_probe.cpp` differ from each other by ONE leg, so
each leg's price is a difference of two censuses:

    FEED   = m16_none    - m16_empty      (the 16 K-step tiles, A/B loads, lane movement, MMA, waits)
    DRAIN  = m16_bf16rt  - m16_none       (the fold-return: D tile -> shipped BF16 round -> s_sc)
    STRIDE = m16_bf16wide - m16_bf16rt    (does a 512-wide score row cost more than a 16-wide one?)
    FLOOR  = m16_bf16frag - m16_empty     (raw builtin + direct fragment drain; map-ASSUMED, a bound)
    CONTROL= m16_scalar  - m16_empty      (same geometry, shipped VALU dot+fold: the re-tile tax alone)

CANARIES (a census whose parse silently matches nothing reports a plausible number — see the AGENTS.md
rows about empty parses, five instances already recorded):
  * every named kernel must be present and non-empty;
  * every differenced pair must be present, and a marginal of <= 0 slots aborts rather than printing a
    zero or a negative, which is the signature of a baseline that no longer matches;
  * `--calib` re-derives the two ledger figures the census must reproduce (45 from the engine listing,
    13.5 per tile from the feeder listing) and aborts if either moves;
  * the touch denominator is asserted, not assumed: it is the probe's own geometry (16 x 16 per warp).

Usage:
    python3 prototype/attn_wmma/m16_census.py tmp/attn_wmma_m16/wmma_m16_probe-*.s
    python3 prototype/attn_wmma/m16_census.py <m16.s> \
        --feeder-asm tmp/attn_wmma_price/wmma_feeder_probe-*.s \
        --engine-asm <kernels_attn-*.s> --calib
"""
from __future__ import annotations

import argparse
import collections
import importlib.util
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent

# The slot definition is imported from the instrument that priced the 45, not re-typed here.
_spec = importlib.util.spec_from_file_location(
    "dot2_isa_census", REPO / "prototype/attn_score_dot2/isa_census.py")
base = importlib.util.module_from_spec(_spec)
sys.modules["dot2_isa_census"] = base
_spec.loader.exec_module(base)

TOUCHES_PER_ROUND = 256        # 16 rows x 16 KV lanes per warp per stage round (probe geometry)
MROWS, MLANES, DH = 16, 16, 256   # mirror wmma_m16_probe.cpp's MROWS/MLANES/DH; checked against the
                                  # probe source below so the two files cannot drift silently
K_STEPS = 16                   # DH / 16 = 16 K-steps per accumulator
def check_probe_geometry() -> None:
    """The denominators this file divides by are the PROBE's geometry. Read the probe source and make
    sure the constants still match it, rather than trusting a copy: a silent drift here would move every
    slots/touch figure in the report without moving a single line of the ISA."""
    src = (HERE / "wmma_m16_probe.cpp").read_text()
    for name, want in (("MROWS", MROWS), ("MLANES", MLANES), ("DH", DH)):
        if not re.search(rf"^#define {name} +{want}\b", src, re.M):
            raise SystemExit(f"wmma_m16_probe.cpp no longer defines {name} {want}; this census's "
                             "denominators are stale. Fix both files together.")
    if TOUCHES_PER_ROUND != MROWS * MLANES:
        raise SystemExit("TOUCHES_PER_ROUND no longer equals the probe's MROWS x MLANES.")


check_probe_geometry()
# The gate, in the words of D57: "<= 19 slots/touch -> fund it; >= 30 -> D30's verdict stands".
FUND_LINE = 19.0
KILL_LINE = 30.0
GQ_LDM = 2048                   # the probe's global-A row stride = engine ROW_H = N_Q * DH

EXPECTED = ["m16_empty", "m16_none", "m16_f32row", "m16_bf16rt", "m16_bf16wide", "m16_roll",
            "m16_bf16gq", "m16_bf16frag", "m16_scalar", "m16_scalar_ob"]
LEGS = [
    ("FEED",    "m16_none",    "m16_empty",    "A/B operand feed + MMA + their waits"),
    ("DRAIN",   "m16_bf16rt",  "m16_none",     "fold-return: f32 D tile -> shipped BF16 round -> s_sc"),
    ("STRIDE",  "m16_bf16wide", "m16_bf16rt",  "same drain at the shipped 512-wide row stride"),
    ("ROLL",    "m16_roll",    "m16_empty",    "same feed ROLLED (unroll 1) + same drain: the VGPR leg"),
    ("GQ",      "m16_bf16gq",  "m16_empty",    "A read from GLOBAL, no 8 KiB Q stage: the LDS leg"),
    ("FLOOR",   "m16_bf16frag", "m16_empty",   "raw builtin + direct fragment drain (map ASSUMED)"),
    ("CONTROL", "m16_scalar",  "m16_empty",    "same 16x16 geometry, shipped VALU dot+fold (no WMMA)"),
    ("CTRL_OB", "m16_scalar_ob", "m16_empty",  "as CONTROL, plus every lane's folded result stored"),
]

# The matrix instruction is NOT one issue slot. `tally()` gives every instruction one slot, which is
# right for VALU/LDS/SALU and wrong here: a wave32 16x16x16 WMMA issues in one slot and then owns its
# SIMD's matrix pipe for 4096 / (32 lanes x 4 MAC/lane/clk) = 32 clocks at the class roof, 36 at the
# measured sustained rate. Reporting slots/touch without this line would price the matrix instruction
# ~32x cheap — the single most flattering thing this census could do to its own answer — so it is
# printed beside the verdict, not below it.
# ---------------------------------------------------------------------------
WMMA_ROOF_MAC_PER_LANE_CLK = 4.0
WMMA_MEAS_MAC_PER_LANE_CLK = 3.552
WMMA_MACS_PER_LANE = 4096 // 32   # = 128 MACs per lane of a wave32 16x16x16 tile
WARPS_PER_BLOCK = 4               # ATTN_TILE_THREADS / 32 == ATTN_TILE_G

# Category rules, FIRST MATCH WINS. Deliberately coarse on the arithmetic (the ledger's own category
# tables already cover the VALU side) and fine on the four things this gate is about: did the compiler
# move lanes, how wide were the LDS touches, where did the stores go, and how many waits.
CATS = [
    ("v_wmma",            "MMA.wmma"),
    ("v_mfma",            "MMA.mfma"),
    ("ds_swizzle",        "SHUFFLE.lane-movement"),
    ("ds_bpermute",       "SHUFFLE.lane-movement"),
    ("ds_permute",        "SHUFFLE.lane-movement"),
    ("v_permlane",        "SHUFFLE.lane-movement"),
    ("v_mov_b32_dpp",     "SHUFFLE.lane-movement"),
    ("v_alignbyte",       "SHUFFLE.lane-movement"),
    ("ds_load_b128",      "LDS.load_b128"),
    ("ds_load_b64",       "LDS.load_b64"),
    ("ds_load_b32",       "LDS.load_b32"),
    ("ds_load_u16",       "LDS.load_u16"),
    ("ds_load",           "LDS.load_other"),
    ("ds_store_b128",     "LDS.store_b128"),
    ("ds_store_b64",      "LDS.store_b64"),
    ("ds_store_b32",      "LDS.store_b32"),
    ("ds_store_u16",      "LDS.store_u16"),
    ("ds_store_2addr",    "LDS.store_dual"),
    ("ds_store",          "LDS.store_other"),
    ("global_load",       "VMEM.load"),
    ("global_store",      "VMEM.store"),
    ("s_barrier",         "BARRIER"),
    ("s_waitcnt",         "WAIT.encode-only"),
    ("s_nop",             "WAIT.encode-only"),
    ("s_delay_alu",       "WAIT.encode-only"),
    ("v_cvt_pk_bf16",     "CVT.bf16-pack"),
    ("v_cvt_off_src_bf16", "CVT.bf16-round"),
    ("v_mov_b16",         "CVT.bf16-widen"),
    ("v_cvt",             "CVT.other"),
    ("v_bfe",             "CVT.bf16-widen"),
    ("v_lshlrev_b32_e32", "CVT.bf16-widen-or-addr"),
    ("v_and_b32_e32",     "CVT.bf16-widen-or-addr"),
    ("s_",                "SALU"),
    ("v_dual",            "VALU.dual-packet"),
    ("v_",                "VALU"),
]
CAT_RX = [(re.compile("^" + "".join(".*" if c == "*" else "[^,]*" if c == "?" else r"\s+"
                                    if c == " " else re.escape(c) for c in p)), c)
          for p, c in CATS]
LANE_MOVE = "SHUFFLE.lane-movement"

# The block splitter is imported from the instrument that priced the 45, for the same reason the slot
# rule is: so the two cannot drift.
_slim_spec = importlib.util.spec_from_file_location(
    "slim_isa_census", REPO / "prototype/attn_valu_slim/isa_census.py")
_slim = importlib.util.module_from_spec(_slim_spec)
sys.modules["slim_isa_census"] = _slim
_slim_spec.loader.exec_module(_slim)


def category(mn: str) -> str:
    for rx, cat in CAT_RX:
        if rx.match(mn):
            return cat
    return "UNCLASSIFIED." + mn


def split_kernels(src) -> dict[str, list[str]]:
    """`base.split_kernels` wants LINES, so pass text through here rather than feed it a string and get
    one 'kernel' per character."""
    return base.split_kernels(src.splitlines() if isinstance(src, str) else src)


# ---------------------------------------------------------------------------
# STATIC SLOTS ARE NOT ROUND SLOTS.
# `tally` counts each instruction line ONCE. Where LLVM leaves a hot loop ROLLED, one body stands for
# many executions, so dividing the static count by 256 UNDERSTATES that variant — the exact mistake
# `prototype/attn_valu_slim/isa_census.py` names as its `--child-trip` correction ("its body therefore
# counts --child-trip times, not once; not doing this is how a census reports 24 VALU/touch for a
# touch that costs 48"). This bit this census first: `m16_roll` printed 1.10 slots/touch, BELOW the
# unrolled 1.73, and `m16_scalar` printed 2.50 lane-movement ops/touch against the shipped 5, which
# was read as dead-code elimination in the control and is not — the static body simply covers 128 of
# the round's 256 touches, and pays the fold 5 times per touch in the code it does contain.
#
# THE LOOP IS FOUND FROM LLVM'S OWN BACK EDGE, NOT FROM A BLOCK TABLE.
# The first attempt here keyed on `split_blocks_full`'s `in Loop: Header=` annotations and silently
# found nothing on the rolled variants: on this toolchain the WMMA variants' loop bodies carry
# `.LtmpNNN` debug labels with an EMPTY parent chain, so a rule that reads annotations expands
# nothing and prints a plausible, wrong number. A back edge is structural and cannot be missing: a
# loop is a label plus an `s_cbranch_*`/`s_branch` that targets it from a later line. The region is
# the label through the LAST such branch, because a partially unrolled body branches back once per
# unrolled group (`m16_scalar` has 128 back edges to one header; taking the first truncated the body
# to 3 % of itself).
#
# THE TRIP COUNT IS DERIVED FROM A FINGERPRINT COUNT, NOT ASSUMED, so a change in LLVM's unroll factor
# moves the number instead of silently breaking the rule:
#   * WMMA form: `trip = K_STEPS / (v_wmma count in the static code)`. Fully unrolled 16/16 = 1 trip
#     (no expansion); fully rolled 16/1 = 16. `K_STEPS = DH/16` is the probe's own geometry.
#   * VALU form: `trip = TOUCHES / (publish stores in the static code)`, the publish store being the
#     one 2-byte BF16 `ds_store_b16_d16_hi` per touch (staging stores are `ds_store_b128`, a different
#     mnemonic, so the two cannot be confused). `m16_scalar`: 128 stores -> 2 trips.
# Both assert exact divisibility, refuse a kernel carrying both fingerprints, and print the region and
# the derivation, because a check whose input can fail to arrive must be shown to have arrived.
# ---------------------------------------------------------------------------
BRANCH_RX = re.compile(r"^\ts_(?:cbranch_\w+|branch|setpc)\s+([A-Za-z0-9_.$]+)")
LABEL_RX = re.compile("^([A-Za-z0-9_.$]+):")
PUBLISH_MN = "ds_store_b16_d16_hi"   # one per touch: the BF16 score store (staging uses b128)


def loop_regions(body: list[str]) -> dict[str, tuple[int, int, int]]:
    """label -> (first line, last back-edge line, number of back edges) for every back edge."""
    pos: dict[str, int] = {}
    for i, ln in enumerate(body):
        m = LABEL_RX.match(ln)
        if m:
            pos.setdefault(m.group(1), i)
    back: dict[str, list[int]] = collections.defaultdict(list)
    for i, ln in enumerate(body):
        m = BRANCH_RX.match(ln)
        if m and m.group(1) in pos and pos[m.group(1)] < i:
            back[m.group(1)].append(i)
    return {t: (pos[t], max(v), len(v)) for t, v in back.items()}


def score_fingerprint(body: list[str]) -> tuple[list[str], callable, int, int]:
    """(all mnemonics, the hot-loop predicate, n_wmma, n_bpermute). Refuses a kernel carrying both, "
    "because the trip-count rule has one form per form of the probe."""
    mn = [m.group(1) for m in map(base.INSTR_RE.match, body) if m]
    n_w = sum(1 for m in mn if m.startswith("v_wmma"))
    n_b = sum(1 for m in mn if m == "ds_bpermute_b32")
    if n_w and n_b:
        raise SystemExit("a kernel body contains BOTH a WMMA loop and a bpermute loop; the trip-count "
                         "expansion has no rule for that. Split the variant.")
    fp = (lambda m: m.startswith("v_wmma")) if n_w else (lambda m: m == "ds_bpermute_b32")
    return mn, fp, n_w, n_b


def hot_region(body: list[str]) -> tuple[str, int, int, list[str], int] | None:
    """(label, first line, last back-edge line, mnemonics in the region, back edges) for the LARGEST
    back-edge region carrying the score fingerprint, or None. Largest because `m16_scalar` nests three
    loops that each contain a permute and the outer one is the touch loop. Shared by the trip-count rule
    and by §4d's traffic arithmetic so the two cannot disagree about what the hot loop is."""
    _, fp, _, _ = score_fingerprint(body)
    regions = loop_regions(body)
    best = None
    for lab, (lo, hi, nb) in regions.items():
        rms = [r.group(1) for r in map(base.INSTR_RE.match, body[lo:hi + 1]) if r]
        if not any(fp(m) for m in rms):
            continue
        if best is None or (hi - lo) > (best[2] - best[1]):
            best = (lab, lo, hi, rms, nb)
    return best


def round_slots(body: list[str], slots: int) -> tuple[int, str]:
    mn_all, _, n_w, n_b = score_fingerprint(body)
    if not n_w and not n_b:
        return slots, "no score fingerprint at all (the empty baseline): static == round"
    best = hot_region(body)
    if best is None:
        if n_w == K_STEPS:
            return slots, "fully unrolled (16 v_wmma in line): static == round"
        raise SystemExit(f"{n_w} v_wmma in the static code, so the K loop is rolled, but no back edge "
                         "contains one. The expansion cannot be derived; refusing to divide by 256.")
    lab, lo, hi, rms, n_back = best
    loop_slots = sum(1 for m in rms if not m.startswith(base.SKIP_AS_WORK))
    if loop_slots == 0:
        raise SystemExit(f"region {lab} has 0 billable slots; the fingerprint is wrong.")
    if n_w:
        if K_STEPS % n_w:
            raise SystemExit(f"{n_w} v_wmma does not divide K_STEPS={K_STEPS}; the feed geometry moved.")
        trip = K_STEPS // n_w
        why = (f"K-step region {lab} [{lo},{hi}] ({n_back} back edges): {n_w} v_wmma in the "
               f"static code, {loop_slots} slots -> expanded x{trip} (K_STEPS {K_STEPS} / {n_w})")
    else:
        per_static = sum(1 for m in mn_all if m == PUBLISH_MN)
        if per_static == 0:
            raise SystemExit(f"no {PUBLISH_MN} publish store found; cannot derive the unroll factor.")
        if TOUCHES_PER_ROUND % per_static:
            raise SystemExit(f"{per_static} publish stores does not divide {TOUCHES_PER_ROUND} touches; "
                             "the control's geometry moved.")
        trip = TOUCHES_PER_ROUND // per_static
        why = (f"touch region {lab} [{lo},{hi}] ({n_back} back edges): {per_static} publish "
               f"stores in the static code = that code covers {per_static} of "
               f"{TOUCHES_PER_ROUND} touches -> expanded x{trip}")
    if trip == 1:
        return slots, f"fully unrolled ({n_w} v_wmma in line): static == round"
    return slots + (trip - 1) * loop_slots, why


# ---------------------------------------------------------------------------
# THE OCCUPANCY POLICY, CALIBRATED TO `tools/kernel_occupancy_probe.cpp`'s RECORDED OUTPUT.
# A slot/touch figure prices the ISSUE stream only. What decides whether those slots can be issued is
# how many wavefronts of the block are resident, and on this part that is bound by LDS and by VGPR, not
# by the instruction mix: `prototype/gdn_collapse/results.md` §4-§5 is the recorded failure of a cost
# model that assumed issue slots the machine did not have (priced 11.4 ms, measured 112.04 ms, because
# the grid realised ~1.6 working wavefronts per CU).
#
# The constants are the KFD facts that probe reads (`lds_size_in_kb 64`, `simd_per_cu 2`,
# `max_waves_per_simd 16`) and the 512-VGPR-per-SIMD budget with the allocation granularity of 8 its
# own reading line uses ("47 VGPR (granularity 8) allows 10 waves/SIMD"). `SHIPPED` below is the same
# kernel's row in `prototype/attn_wmma/evidence/occupancy.txt`, so the policy is checked against the
# tool's recorded answer instead of being trusted: if `blocks_wavefronts(*SHIPPED)` ever stops printing
# 5 blocks / 20 wavefronts, this file's occupancy column is wrong too.
# ---------------------------------------------------------------------------
LDS_PER_CU = 65536
SIMD_PER_CU = 2
MAX_WAVES_PER_SIMD = 16
REGS_PER_SIMD = 512
VGPR_GRAN = 8
SHIPPED_VGPR, SHIPPED_LDS, SHIPPED_BLOCKS, SHIPPED_WF = 47, 12288, 5, 20


def roundup(x: int, g: int) -> int:
    return ((x + g - 1) // g) * g


def blocks_wavefronts(lds: int, vgpr: int, warps: int) -> tuple[int, int, str]:
    """(blocks/CU, wavefronts/CU, which term binds) for a block of `warps` wavefronts."""
    if lds <= 0 or vgpr <= 0 or warps <= 0:
        raise SystemExit("occupancy needs the real LDS and VGPR figures; 0 means the listing had no "
                         "note block for this kernel.")
    by_lds = LDS_PER_CU // lds
    waves = min(MAX_WAVES_PER_SIMD, REGS_PER_SIMD // roundup(vgpr, VGPR_GRAN))
    by_vgpr = (waves * SIMD_PER_CU) // warps
    by_wf = (MAX_WAVES_PER_SIMD * SIMD_PER_CU) // warps
    blocks = max(0, min(by_lds, by_vgpr, by_wf))
    binds = [n for n, v in (("LDS", by_lds), ("VGPR", by_vgpr), ("wavefront cap", by_wf)) if v == blocks]
    return blocks, blocks * warps, "+".join(binds) or "?"



def split_meta(text: str) -> dict[str, dict[str, int]]:
    """Per-kernel resource metadata out of the listing's YAML note block: vgpr / sgpr / LDS bytes."""
    out: dict[str, dict[str, int]] = {}
    for chunk in re.split(r"\n\s*- \.args:", text):
        nm = re.search(r"^\s*\.name:\s+(\S+)", chunk, re.M)
        if not nm:
            continue
        rec: dict[str, int] = {}
        for key, field in (("vgpr", r"\.vgpr_count:\s+(\d+)"),
                           ("sgpr", r"\.sgpr_count:\s+(\d+)"),
                           ("lds", r"\.group_segment_fixed_size:\s+(\d+)"),
                           ("spill", r"\.vgpr_spill_count:\s+(\d+)")):
            m = re.search(field, chunk, re.M)
            if m:
                rec[key] = int(m.group(1))
        out[nm.group(1)] = rec
    return out


def census_kernel(body: list[str]) -> tuple[collections.Counter, int, int]:
    """(category counts, slots, encode-only) for one kernel body.

    TWO currencies are reported, because the two ledger figures were made in different ones and the
    gate must not mix them:
      * SLOTS  = one instruction line, `s_waitcnt`/`s_nop`/`s_delay_alu` EXCLUDED. This is verbatim
        `prototype/attn_score_dot2/isa_census.py`'s `tally()` rule, i.e. the currency the shipped
        touch is 45 in. Used for every verdict line.
      * INSTR  = slots + encode-only, i.e. `feeder_census.py`'s "executed instructions", which is the
        currency `prototype/attn_wmma/results.md` §3's 13.5 slots/tile was printed in. Kept so that figure reproduces
        exactly and so nobody re-uses 13.5 next to 45 as if they were one unit (they differ by the
        waits, and a WMMA block issues far more waits than a VALU one).
    """
    insts = []
    for ln in body:
        m = base.INSTR_RE.match(ln)
        if m:
            insts.append(m.group(1))
    cats: collections.Counter = collections.Counter()
    slots = enc = 0
    for mn in insts:
        if mn.startswith(base.SKIP_AS_WORK):
            cats["WAIT.encode-only"] += 1
            enc += 1
            continue
        cats[category(mn)] += 1
        slots += 1          # one instruction LINE is one issue slot; a v_dual packet packs two ops
    return cats, slots, enc


def must_have(kern: dict[str, list[str]]) -> None:
    missing = [k for k in EXPECTED if k not in kern or not kern[k]]
    if missing:
        raise SystemExit(f"MISSING KERNELS: {missing}. Present: {sorted(kern)[:12]}. A census that "
                         "silently skips a leg prints a plausible number for the ones it did read.")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("asm", type=Path, help="the m16 listing from build_m16.sh")
    ap.add_argument("--feeder-asm", type=Path,
                    help="wmma_feeder_probe listing, for the 13.5-slots-per-tile calibration")
    ap.add_argument("--engine-asm", type=Path,
                    help="engine/kernels_attn.cpp listing, for the 45-slots-per-touch calibration")
    ap.add_argument("--calib", action="store_true", help="run the two ledger calibrations and abort on miss")
    ap.add_argument("--touches", type=int, default=TOUCHES_PER_ROUND)
    ap.add_argument("--out", type=Path)
    a = ap.parse_args()
    if a.touches != TOUCHES_PER_ROUND:
        print(f"NOTE: --touches {a.touches} overrides the probe geometry {TOUCHES_PER_ROUND}; the "
              f"number printed is then not the gate's denominator.")

    out: list[str] = []
    w = out.append
    lines = a.asm.read_text().splitlines()
    kern = split_kernels(lines)
    meta = split_meta(a.asm.read_text())
    must_have(kern)

    w(f"# m16 census of {a.asm.name}")
    w(f"# denominator: {a.touches} (row, lane, head) touches per warp per stage round (16 x 16).")
    w(f"# slot rule: imported from prototype/attn_score_dot2/isa_census.py tally() — one line = one")
    w(f"#   slot, a v_dual packet is one slot, s_waitcnt/s_nop/s_delay_alu are ENCODE-ONLY (shown in")
    w(f"#   the WAIT column and reported both ways in the verdict line).")
    w("")

    cats, slots, enc, rslots, whyloop, occ = {}, {}, {}, {}, {}, {}
    for k in EXPECTED:
        cats[k], slots[k], enc[k] = census_kernel(kern[k])
        rslots[k], whyloop[k] = round_slots(kern[k], slots[k])
        m = meta.get(k, {})
        occ[k] = blocks_wavefronts(m.get("lds", 0), m.get("vgpr", 0), WARPS_PER_BLOCK)
    sb, sw, _ = blocks_wavefronts(SHIPPED_LDS, SHIPPED_VGPR, WARPS_PER_BLOCK)
    if (sb, sw) != (SHIPPED_BLOCKS, SHIPPED_WF):
        raise SystemExit(f"the occupancy policy no longer reproduces the shipped attn_slim_wave row of "
                         f"prototype/attn_wmma/evidence/occupancy.txt ({sb} blocks / {sw} wf/CU vs "
                         f"{SHIPPED_BLOCKS} / {SHIPPED_WF}). Fix the policy before quoting a column.")

    w("## 1. Per-kernel census, and what each block can actually be resident as")
    out.extend([
        "# slots = STATIC issue slots (a rolled body counted once); rounds = slots actually issued per",
        "#   stage round after the back-edge expansion below. blk/CU and wf/CU come from LDS and VGPR",
        "#   per the policy calibrated on the shipped kernel's recorded row, asserted at the top of this",
        "#   section: " + str(SHIPPED_VGPR) + " VGPR / " + str(SHIPPED_LDS) + " B gives "
        + str(SHIPPED_BLOCKS) + " blocks and " + str(SHIPPED_WF) + " wf/CU.",
    ])
    w(f"{'kernel':14} {'slots':>6} {'rounds':>7} {'r/touch':>8} {'instr':>6} {'lane-mv':>7} {'MMA':>4} "
      f"{'ldsL':>5} {'ldsS':>5} {'VGPR':>5} {'LDS B':>7} {'blk/CU':>7} {'wf/CU':>6} {'binds':>14}")
    for k in EXPECTED:
        c = cats[k]
        m = meta.get(k, {})
        lds_ld = sum(v for kk, v in c.items() if kk.startswith("LDS.load"))
        lds_st = sum(v for kk, v in c.items() if kk.startswith("LDS.store"))
        blk, wf, binds = occ[k]
        w(f"{k:14} {slots[k]:6} {rslots[k]:7} {rslots[k] / a.touches:8.2f} {slots[k]+enc[k]:6} "
          f"{c.get(LANE_MOVE,0):7} "
          f"{c.get('MMA.wmma',0):4} {lds_ld:5} {lds_st:5} "
          f"{m.get('vgpr',-1):5} {m.get('lds',-1):7} {blk:7} {wf:6} {binds:>14}")
    w("")
    w("## 1b. How each rounds figure was derived (a rolled body must be expanded; see the header)")
    for k in EXPECTED:
        w(f"  {k:14} {whyloop[k]}")
    w("")

    w("## 2. Legs — each is a difference of two ROUND-slot censuses, divided by the touch denominator")
    w(f"{'leg':9} {'slots':>6} {'slots/touch':>11} {'instr/touch':>11}   composition")
    legs: dict[str, tuple[float, float]] = {}
    for name, hi, lo, what in LEGS:
        d = rslots[hi] - rslots[lo]
        di = (slots[hi] + enc[hi]) - (slots[lo] + enc[lo])
        if d <= 0:
            raise SystemExit(f"LEG {name}: marginal is {d} slots (<= 0). The baseline {lo} no longer "
                             "is a subset of the variant; the variants drifted apart. Refusing.")
        legs[name] = (d / a.touches, di / a.touches)
        dc = collections.Counter(cats[hi]) - collections.Counter(cats[lo])
        comp = "  ".join(f"{kk}={v}" for kk, v in sorted(dc.items(), key=lambda kv: -kv[1])[:6])
        w(f"{name:9} {d:6} {d / a.touches:11.2f} {di / a.touches:11.2f}   {comp}")
    w("")

    w("## 3. The gate: slots per (row, lane, head) touch on the WMMA path")
    feed, feed_i = legs["FEED"]
    drain, drain_i = legs["DRAIN"]
    control, control_i = legs["CONTROL"]
    w(f"  FEED  (16 K-step tiles, A/B loads, lane movement, MMA)   = {feed:6.2f} slots/touch "
      f"({feed_i:.2f} in the instr currency prototype/attn_wmma/results.md §3 used)")
    w(f"  DRAIN (f32 D tile -> shipped BF16 round -> s_sc)         = {drain:6.2f} slots/touch "
      f"({drain_i:.2f} instr)")
    w(f"  ALL-IN  (FEED + DRAIN + the shared stage round, /touches)= "
      f"{(rslots['m16_bf16rt']) / a.touches:6.2f} slots/touch incl. staging and both barriers")
    w("")
    # THE CONTROL, INSIDE THE VERDICT, WITH A WITNESS THAT IT PAYS THE FOLD.
    # A control whose only observable result is lane 0's answers "did the compiler keep the whole fold?"
    # by argument instead of by witness, so the probe now carries both spellings: `m16_scalar` (shipped
    # publish shape) and `m16_scalar_ob` (every lane's folded value stored). The witness is the
    # lane-movement count per touch: the shipped slim fold is 5 (`fold.permute` 5/touch in
    # `prototype/attn_valu_slim/results.md` §2), and a control that pays fewer has had fold steps
    # removed and cannot adjudicate anything. The earlier version of this census printed 2.50/touch and
    # called that dead-code elimination; it was this file dividing a 128-touch static body by 256 (see
    # §1b), so the canary now divides by the touches the code actually covers and ABORTS if the control
    # still cannot account for its fold.
    ship = 45.0
    SHIPPED_PERM = 5.0        # prototype/attn_valu_slim/results.md §2: fold.permute 5 per touch, both forms
    w(f"  CONTROL, both spellings (same 16x16 geometry, shipped VALU dot+fold, no WMMA):")
    for ck, what in (("m16_scalar", "shipped publish shape (only lane 0 stores)"),
                     ("m16_scalar_ob", "every lane's folded value stored")):
        touches_static = sum(1 for m2 in (base.INSTR_RE.match(l) for l in kern[ck]) if m2
                             and m2.group(1) == PUBLISH_MN)
        if touches_static == 0:
            raise SystemExit(f"{ck} has no {PUBLISH_MN}: the control no longer publishes a score and "
                             "cannot witness the re-tile.")
        perm = cats[ck].get("SHUFFLE.lane-movement", 0) / touches_static
        val = (rslots[ck] - rslots["m16_empty"]) / a.touches
        verdict_bit = (f"{perm:.2f} lane-movement ops/touch, the shipped fold shape" if perm >= SHIPPED_PERM
                       else f"FAILS the fold witness: {perm:.2f} ops/touch vs shipped {SHIPPED_PERM:.0f}")
        w(f"    {ck:16} {val:6.2f} slots/touch — {what}; {verdict_bit}")
        if perm < SHIPPED_PERM:
            raise SystemExit(f"{ck} pays {perm:.2f} lane-movement ops/touch against the shipped "
                             f"{SHIPPED_PERM:.0f}: fold work is not observable, so it cannot sit in the "
                             "verdict. Give every lane's result somewhere to land.")
    control = (rslots["m16_scalar_ob"] - rslots["m16_empty"]) / a.touches
    w(f"  -> CONTROL IN THE VERDICT: {control:.2f} slots/touch (the observable spelling, and the "
      f"{'larger' if control >= (rslots['m16_scalar'] - rslots['m16_empty']) / a.touches else 'smaller'} "
      "of the two).")
    w(f"     Against the shipped M = 1 touch of {ship:.0f}, that is the RE-TILE TAX ALONE: "
      f"{control - ship:+.2f} slots/touch")
    w(f"     on the VALU path, before any instruction class changes. It is the number "
      f"`prototype/attn_m2`")
    w("     measured on device (7-10 % slower at short widths, 16-32 blocks) and it is the debit the")
    w("     WMMA credit has to clear.")
    w("")
    w("  VERDICT LINE. A WMMA pass-1 touch is FEED + DRAIN: the score round and store are inside DRAIN,")
    w("  and at M = 16 there is no per-touch loop left to bill — the loop is 16 K-steps per 256 touches.")
    w("  The two ways to price the residue are printed because D57's own bottom-up assumed it survives:")
    dot_slots, fold_slots = 16.0, 11.0     # 8 conv + 8 fma; 5 permute + 4 add + 2 add-to-wait class
    rows = [
        ("residue = 45 - (dot 16 + fold 11) = 18, as D57 priced it",
         feed + drain + (ship - dot_slots - fold_slots)),
        ("residue = 0, measured: no per-touch loop survives the tile (m16_bf16rt - m16_empty)",
         (rslots["m16_bf16rt"] - rslots["m16_empty"]) / a.touches),
    ]
    for label, tot in rows:
        verdict = ("FUNDABLE" if tot <= FUND_LINE else
                   "DEAD (>= kill line)" if tot >= KILL_LINE else "BETWEEN THE LINES")
        w(f"    {label:70} {tot:6.2f} slots/touch -> {verdict}")
    w(f"  (gate lines: <= {FUND_LINE:.0f} fund, >= {KILL_LINE:.0f} dead; in between is an answer too)")
    w("")
    w("## 3c. The same verdict per variant, because a slot figure from a block that cannot be resident")
    w("#       is a fact about the compiler, not about the kernel (gdn_collapse §4-§5: priced 11.4 ms")
    w("#       from its own instruction census, measured 112.04 ms at ~1.6 working wavefronts per CU).")
    w("#       occ = wavefronts/CU the variant's VGPR and LDS allow, against the shipped 20 (62.5 %).")
    w(f"  {'variant':14} {'slots/touch':>11} {'wf/CU':>6} {'blk/CU':>6} {'VGPR':>5} {'LDS B':>7}  "
      f"{'gate on slots':>13}  note")
    for k in EXPECTED:
        if k == "m16_empty":
            continue
        v = (rslots[k] - rslots["m16_empty"]) / a.touches
        blk, wf, binds = occ[k]
        gate = ("<=19 ok" if v <= FUND_LINE else ">=30 dead" if v >= KILL_LINE else "between")
        note = ("holds shipped occupancy" if wf >= SHIPPED_WF else
                f"{SHIPPED_WF - wf} wf/CU below shipped ({100.0 * wf / SHIPPED_WF:.0f} %): "
                f"bound by {binds}")
        if k == "m16_bf16frag":
            note += "; map-ASSUMED, a bound not a spelling"
        if k in ("m16_scalar", "m16_scalar_ob"):
            note = "the CONTROL: no WMMA in it, so it prices the geometry, not the class"
        w(f"  {k:14} {v:11.2f} {wf:6} {blk:6} {meta.get(k, {}).get('vgpr', -1):5} "
          f"{meta.get(k, {}).get('lds', -1):7}  {gate:>13}  {note}")
    w("")

    w("## 4. Stride probe, the roll leg, and the map-assumed floor (bounds, not shippable spellings)")
    w(f"  STRIDE  {legs['STRIDE'][0]:+.2f} slots/touch for a shipped 512-wide score row vs a 16-wide one")
    roll = legs["ROLL"][0]
    w(f"  ROLL    {roll:.2f} slots/touch with the K loop ROLLED, against {feed + drain:.2f} unrolled "
      f"(FEED+DRAIN): rolling costs {roll - (feed + drain):+.2f} slots/touch of per-K-step control and "
      f"buys the VGPR")
    w(f"  GQ      {legs['GQ'][0]:.2f} slots/touch with A from GLOBAL (engine row stride {GQ_LDM}) and "
      f"no Q LDS stage — see §4c")
    w(f"  FLOOR   {legs['FLOOR'][0]:.2f} slots/touch with hand-fed raw WMMA and a direct fragment drain;")
    w("          legal only if the hardware fragment map is the assumed one (issue_rate_probe §5.1).")
    w("")

    # ------------------------------------------------------------------
    # THE TERM A SLOT COUNT CANNOT PRICE. `tally()` gives every instruction one issue slot, which is
    # right for VALU/LDS/SALU and wrong for a WMMA: the instruction issues in one slot and then owns
    # the matrix pipe for 32 clocks at the class roof. Reporting slots/touch without this line would price the
    # matrix instruction 16x cheap, which is the single most flattering thing this census could do to
    # its own answer, so it is printed next to the verdict rather than below it.
    # ------------------------------------------------------------------
    w("## 4b. The matrix pipe, which is not an issue slot")
    clk_roof = WMMA_MACS_PER_LANE / WMMA_ROOF_MAC_PER_LANE_CLK
    clk_meas = WMMA_MACS_PER_LANE / WMMA_MEAS_MAC_PER_LANE_CLK
    pipe_touch = K_STEPS * clk_meas / a.touches
    w(f"  one v_wmma_f32_16x16x16 (wave32) = {WMMA_MACS_PER_LANE:.0f} MACs/lane = {clk_roof:.0f} clk at "
      f"the WMMA roof, {clk_meas:.1f} clk at the measured sustained {WMMA_MEAS_MAC_PER_LANE_CLK} "
      f"MAC/lane/clk")
    w(f"  so the SCORE tile's pipe time is {K_STEPS} x {clk_meas:.1f} = {K_STEPS * clk_meas:.0f} clk per "
      f"warp per round = {pipe_touch:.2f} clk/touch — i.e. the matrix pipe, not the")
    w(f"  feed, is the floor: {pipe_touch:.2f} clk/touch replaces 45 issue "
      f"slots/touch, and the feed+drain ({feed + drain:.2f}) can be hidden behind it only if other "
      f"wavefronts")
    w("  are resident to issue while the pipe is busy — which is an OCCUPANCY question, §4c.")
    w(f"  The pipe is per-SIMD and {WARPS_PER_BLOCK} warps share a block across {SIMD_PER_CU} SIMDs, so a")
    w(f"  block resident alone on a CU puts ~2 of its 4 warps per SIMD: enough to overlap one warp's "
      f"WMMA")
    w("  with another's feed only if the block count is not itself driven to 1 by VGPR or LDS. That is")
    w("  the term §1's wf/CU column carries, and it is why the verdict is printed per variant in §3c.")
    w("")
    w("## 4c. LDS and VGPR per block, against the §26 binding (5 blocks/CU at 12,288 B)")
    w(f"  policy: blocks/CU = min(65536/LDS, (min(16, 512/round{VGPR_GRAN}(VGPR)) x {SIMD_PER_CU} "
      f"SIMD) / {WARPS_PER_BLOCK} warps, 32-wf cap); calibrated on the shipped row "
      f"({SHIPPED_VGPR} VGPR / {SHIPPED_LDS} B -> {SHIPPED_BLOCKS} blocks / {SHIPPED_WF} wf/CU), "
      f"which this run asserts.")
    w(f"  {'variant':14} {'VGPR':>5} {'spill':>6} {'LDS B':>7} {'blk/CU':>7} {'wf/CU':>6} {'binds':>14}"
      f"  {'vs shipped 20 wf':>16}")
    for k in EXPECTED:
        m = meta.get(k, {})
        blk, wf, binds = occ[k]
        w(f"  {k:14} {m.get('vgpr', -1):5} {m.get('spill', 0):6} {m.get('lds', -1):7} {blk:7} "
          f"{wf:6} {binds:>14}  {wf - SHIPPED_WF:+15d}")
    w(f"  shipped attn_slim_wave: 47 VGPR / 12,288 B -> {SHIPPED_BLOCKS} blocks/CU = "
      f"{SHIPPED_WF} wavefronts/CU (62.5 % of the 32 the part allows), from "
      "prototype/attn_wmma/evidence/occupancy.txt.")
    w("")
    w("## 4d. The traffic a global-A spelling buys and costs, as arithmetic on the probe's own geometry")
    a_step = 16 * 16 * 2                     # one A tile per K-step: 32 lanes x 16 B = one b128 each
    k_stage = MLANES * DH * 2                # one 16-lane K sub-block, staged once per round by the block
    k_stage_ship = 8 * DH * 2                # shipped ATTN_TILE_L = 8 lanes, for 8 touches x 4 heads
    a_touch = K_STEPS * a_step / a.touches
    k_touch = k_stage / (WARPS_PER_BLOCK * a.touches)
    k_touch_ship = k_stage_ship / (8 * WARPS_PER_BLOCK)
    reg = hot_region(kern["m16_bf16gq"])
    n_g = len([m for m in (reg[3] if reg else []) if m.startswith("global_load")])
    w(f"  A read from global: {n_g} global load(s) in the rolled K-step body, so A is re-read EVERY K-step,")
    w(f"  not once per round: {K_STEPS} x {a_step} B = {K_STEPS * a_step} B per warp per round = "
      f"{a_touch:.0f} B/touch.")
    w(f"  K read from global: {k_stage} B per block per round, shared by {WARPS_PER_BLOCK} warps x "
      f"{a.touches} touches = {k_touch:.0f} B/touch — against the shipped {k_stage_ship} B for "
      f"{8 * WARPS_PER_BLOCK} touches = {k_touch_ship:.0f} B/touch.")
    w(f"  -> the tile cuts global K traffic {k_touch_ship / k_touch:.0f}x (one sub-block feeds MROWS rows "
      f"instead of one), and buys it with {a_touch:.0f} B/touch of A re-reads.")
    w("  Whether those re-reads are cheap is an L2 question, not a slot question: the 16 rows are 4 KiB")
    w("  apart (GQ_LDM = ROW_H), so each 16 B lane load lands in a different line and the lines fill out")
    w("  only across the round's 16 K-steps. This compile-only census cannot settle it; it is the first")
    w("  thing the device pass must, and §5's answer depends on it.")
    w("")

    w("## 5. The resource-conditioned answer, derived rather than asserted")
    fit = [k for k in EXPECTED if k != "m16_empty" and occ[k][0] >= 4]
    marg = {k: (rslots[k] - rslots["m16_empty"]) / a.touches for k in EXPECTED}
    w(f"  variants holding >= 4 blocks/CU: {', '.join(f'{k} ({marg[k]:.2f} slots/touch, '
      f'{occ[k][1]} wf/CU)' for k in fit) or 'NONE'}")
    wmma_fit = [k for k in fit if cats[k].get("MMA.wmma", 0)]
    if not wmma_fit:
        w("  -> NO WMMA SPELLING HOLDS 4 BLOCKS/CU. The slot win is real but every spelling that keeps it")
        w("     runs at <= 3 blocks/CU, and a cost model priced at the issue slots of a kernel the machine")
        w("     will not run that densely is the `gdn_collapse` error again. Verdict: NOT FUNDABLE AS")
        w("     SPECIFIED, and the residual design question is the N-split that keeps PV's 16 accumulators")
        w("     over DH = 256 out of the 128 VGPR/lane they otherwise cost (D30 §2).")

    else:
        best = min(wmma_fit, key=lambda k: marg[k])
        b_blk, b_wf, b_binds = occ[best]
        w(f"  -> the WMMA spellings that hold >= 4 blocks/CU: {', '.join(wmma_fit)}; cheapest is "
          f"{best} at {marg[best]:.2f} slots/touch, {b_blk} blocks / {b_wf} wf/CU "
          f"({meta.get(best, {}).get('vgpr', -1)} VGPR, {meta.get(best, {}).get('lds', -1)} B), "
          f"bound by {b_binds}.")
        w(f"     Its VGPR is low ONLY because the K loop is rolled (§1b): the same tile unrolled is "
          f"{meta.get('m16_bf16rt', {}).get('vgpr', -1)} VGPR at "
          f"{occ['m16_bf16rt'][0]} block(s)/CU. Register pressure, not the feeder, is what the "
          "M = 16 tile")
        w("     costs, and that is the same wall D30 §2 priced for the PV leg's accumulators: the score "
          "leg alone,")
        w("     unrolled, already spends more than half the 512-register SIMD budget before PV is "
          "mentioned.")
    w("  the PV N-split, witnessed by the score leg: 16 f32 accumulators over DH = 256 is 128 VGPR/lane "
      "(D30 §2).")
    w(f"     This probe's SCORE leg alone needs {meta.get('m16_bf16rt', {}).get('vgpr', -1)} VGPR "
      f"unrolled and {meta.get('m16_roll', {}).get('vgpr', -1)} rolled. A block that holds both legs "
      "unrolled cannot")
    w("     exist under the 512-register budget, so the N-split is a precondition of any port, not an "
      "optimisation.")
    under = [k for k in EXPECTED if 0 < meta.get(k, {}).get("lds", 1 << 30) <= SHIPPED_LDS]
    w(f"  LDS at or under the shipped {SHIPPED_LDS} B: "
      + (", ".join(f"{k} ({meta[k]['lds']} B, {meta[k]['vgpr']} VGPR, {marg[k]:.2f} slots/touch)"
                   for k in under) or "none"))
    w("  — that set is the answer to 'is the low-LDS spelling also the low-VGPR one', read rather than "
      "argued.")
    w("")

    w("## 6. The cost side, from measured rows only — a model with named inputs, not a measurement")
    BILL, SHIP_TOUCH, ATTN_MS, WAVE_MS = 89.9, 45.0, 264.71, 763.49
    EXPO, EXPHI = 0.39, 0.44
    best = min([k for k in EXPECTED if k != "m16_empty" and occ[k][0] >= 4
                and cats[k].get("MMA.wmma", 0)], default=None)
    if best is None:
        w("  no occupiable WMMA spelling, so there is no cost side to price: see §5.")
        w("")
        vbest = 0.0
    else:
        vbest = marg[best]
    w("  inputs, every one measured somewhere else in this tree:")
    w(f"    whole-kernel bill        {BILL} warp-inst/touch  (progress/2026-10-21/performance-status-2026-10-21.md §26 W2, counter-calibrated)")
    w(f"    shipped pass-1 window    {SHIP_TOUCH:.0f} slots/touch     (prototype/attn_valu_slim/results.md §2; --calib re-reads it)")
    w(f"    slot->time elasticity    time ∝ slots^{EXPO}–{EXPHI}      (D21/D22; the exponent prototype/attn_m2/results.md §8.4 also used)")
    w(f"    attn / wave              {ATTN_MS} ms / {WAVE_MS} ms   (progress/2026-10-21/performance-status-2026-10-21.md §25 and §26; attn = "
      f"{100.0 * ATTN_MS / WAVE_MS:.1f} % of the wave)")
    w("    matrix headroom          ~36 µs of matrix work per wave against 257 ms of attn (D34): the")
    w("                             matrix PIPE is never the constraint here; only the geometry tax can be")
    w(f"  the occupiable spelling is {best}: the score leg goes {SHIP_TOUCH:.0f} -> {vbest:.2f} issue "
      f"slots/touch plus {pipe_touch:.2f}")
    w(f"  clk/touch of matrix pipe, so the whole-kernel bill is {BILL} -> "
      f"{BILL - SHIP_TOUCH + vbest:.1f} (slots only) or "
      f"{BILL - SHIP_TOUCH + vbest + pipe_touch:.1f}")
    w("  if the matrix pipe is charged as an issue-equivalent (the conservative reading).")
    lo, hi = BILL - SHIP_TOUCH + vbest, BILL - SHIP_TOUCH + vbest + pipe_touch
    t_best, t_worst = (lo / BILL) ** EXPHI, (hi / BILL) ** EXPO
    w(f"  elasticity transfer: attn x {t_best:.3f}..{t_worst:.3f}, i.e. "
      f"{100 * (t_best - 1):+.1f} % to {100 * (t_worst - 1):+.1f} % BEFORE any geometry tax")
    for tax_name, tax in (("measured at this class by m2 at short widths", 1.07),
                          ("priced by m2 at the funded ruler", 1.29)):
        a_lo, a_hi = sorted((t_worst * tax, t_best * tax))
        share = ATTN_MS / WAVE_MS
        w_lo, w_hi = sorted((1 - share * (1 - a_lo), 1 - share * (1 - a_hi)))
        t_lo, t_hi = sorted(((1 / w_hi - 1) * 100, (1 / w_lo - 1) * 100))
        w(f"  geometry tax {tax_name} (x{tax:.2f}): attn {100 * (a_lo - 1):+.1f} % to "
          f"{100 * (a_hi - 1):+.1f} %  ->  wave {100 * (w_lo - 1):+.1f} % to "
          f"{100 * (w_hi - 1):+.1f} % of its time  ->  prefill {t_lo:+.1f} % to {t_hi:+.1f} % tok/s")
    w("  D30's kill line is that the route must beat `attn` by more than ~15 %. Read this way the band")
    w("  STRADDLES it: it clears on m2's measured tax and fails on m2's priced one. The gate question")
    w("  (slots per touch) answers decisively; the funding question turns on a geometry tax that is a")
    w("  device number, and this compile-only census does not answer it. What would is the device pass")
    w("  handed over beside this census (paired A/B, one lock hold, behavioural canary first), and the")
    w("  two things it must settle are the A-from-global traffic and the tile-local score staging's effect")
    w("  on pass 2 — neither of which is a slot question.")
    w("")

    if a.calib:
        w("## 7. Calibration — the two figures already in the ledger, re-measured here")
        if a.feeder_asm:
            fk = split_kernels(a.feeder_asm.read_text())
            must = [k for k in ("wm_score_tile", "wm_score_k256") if k not in fk]
            if must:
                raise SystemExit(f"--feeder-asm has no {must}; cannot calibrate the 13.5 figure.")
            _, t1, e1 = census_kernel(fk["wm_score_tile"])
            _, t16, e16 = census_kernel(fk["wm_score_k256"])
            per_tile_instr = ((t16 + e16) - (t1 + e1)) / 15.0
            per_tile_slot = (t16 - t1) / 15.0
            ok = abs(per_tile_instr - 13.5) <= 0.10 * 13.5
            w(f"  feeder: (wm_score_k256 {t16 + e16} - wm_score_tile {t1 + e1}) / 15 = "
              f"{per_tile_instr:.2f} instr/tile vs ledger 13.5 -> "
              f"{'CALIBRATED' if ok else 'CALIBRATION-FAIL'}")
            w(f"  the same figure in the SLOT currency the 45 is in (waits excluded): "
              f"{per_tile_slot:.2f} slots/tile")
            w(f"  same leg inside this probe: FEED x {a.touches // K_STEPS} touch-equivalents per tile "
              f"= {feed * (a.touches // K_STEPS):.2f} slots/tile "
              f"({feed_i * (a.touches // K_STEPS):.2f} instr/tile vs ledger 13.5)")
            if not ok:
                raise SystemExit("feeder calibration failed; do not quote the FEED term.")
        else:
            w("  feeder calibration NOT RUN (no --feeder-asm): the 13.5 figure is unverified here.")
        if a.engine_asm:
            p = subprocess.run([sys.executable, str(REPO / "prototype/attn_valu_slim/isa_census.py"),
                                "--asm", str(a.engine_asm),
                                "--kernels", "attn_tile_wave,attn_slim_wave"],
                               capture_output=True, text=True, cwd=REPO)
            got = dict(re.findall(r"(attn_\w+8CoreArgs) loop=\S+ depth=\d+ blocks=\s*\d+ "
                                  r"flow-excl=\s*\d+ slots/touch=\s*(\d+)", p.stdout))
            if not got:
                raise SystemExit(f"could not parse the slim census output (rc={p.returncode}); "
                                 "refusing to calibrate against an empty parse. Tail:\n" + p.stdout[-400:])
            for want, name in ((116, "attn_tile_wave8CoreArgs"), (45, "attn_slim_wave8CoreArgs")):
                if name not in got:
                    raise SystemExit(f"{name} missing from the slim census output.")
                ok = int(got[name]) == want
                w(f"  engine: {name} slots/touch = {got[name]} vs ledger {want} "
                  f"-> {'CALIBRATED' if ok else 'CALIBRATION-FAIL'}")
                if not ok:
                    raise SystemExit("engine touch calibration failed; the denominator moved.")
        else:
            w("  engine calibration NOT RUN (no --engine-asm): the 45 figure is unverified here.")
        w("  NOTE: prototype/attn_wmma/region_census.py --calib is a DIFFERENT instrument (a whole-")
        w("  kernel line-table roll-up) and still aborts at 125.5 against the same 45. This census does")
        w("  not use it and does not re-green it; see prototype/attn_wmma/results.md §7.")
        w("")

    txt = "\n".join(out)
    print(txt)
    if a.out:
        a.out.write_text(txt + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
