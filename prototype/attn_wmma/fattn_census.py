#!/usr/bin/env python3
"""fattn_census.py — the price of D62's falsifiable form, `m16_fattn_score` (§4.2 of
progress/2026-10-18/llama-fattn-anatomy.md). COMPILE-ONLY: reads the `.s` listing
`build_fattn_score.sh` leaves, launches nothing, takes no timing, refuses to print a `ms` figure.

WHY THIS IS A SIBLING OF `m16_census.py` AND NOT AN EDIT TO IT
`m16_census.py`'s recorded output (`m16_census.output.txt`) is the D57/D58 calibration witness, and its
trip-count rule is keyed to that probe's shapes: it derives the loop expansion from the v_wmma count in
the WHOLE kernel body (`round_slots`, m16_census.py:271), which is right when one kernel carries one
score loop and wrong here, where the score loop is rolled (1 v_wmma in the body, 16 trips) and the P*V
loop is deliberately unrolled (16 v_wmma in line, 1 trip) — its `K_STEPS % n_wmma` test would abort on
32. So this file IMPORTS the parts that must not drift (the slot rule and category table from
`prototype/attn_score_dot2/isa_census.py` through `m16_census`, the back-edge region finder, the note
block reader) and replaces only the trip rule, per region instead of per kernel.

TWO CURRENCIES, BOTH NAMED
  SLOTS          one instruction line, waits EXCLUDED — `m16_census.py`'s currency, and the one D57's
                 "<= 19" and D58's 1.98 figures are in.
  PIPE-WEIGHTED  one instruction line with waits counted as work, and each `v_wmma` billed at 32 clocks
                 of matrix-pipe time instead of one slot. This is §3.1's currency: its round is 3.29 raw
                 and 7.16 pipe-weighted, and the pair only reconciles that way ((1,683 - 64 + 64x32)/512
                 = 7.16). The 32 is the class roof, 4096 MAC / (32 lanes x 4 MAC/lane/clk), from
                 `m16_census.output.txt`'s matrix-instruction note; the measured sustained rate
                 (3.552 MAC/lane/clk -> 36 clocks) is printed as a sensitivity line, never as the claim.
  The §4.2 bar (<= 20) is judged in PIPE-WEIGHTED, since that is the currency §3.1 quotes.

THE CANARY, DERIVED FROM THE GEOMETRY AND NOT FROM A HAND LIST
Every rolled region is expanded so that the per-warp-round instruction counts hit the numbers the
SOURCE GEOMETRY says they must, and any miss aborts before a verdict prints:
    global_load_b128 = 3 objects x (16 rows / 4 warps)          = 12
    ds_store_b128    = Q 4 + K 4                                 =  8
    ds_store_u16     = V transposed, 8 narrow stores x 4 rows    = 32
    ds_load_b128     = score 16x2 operands x2 + PV 16x1 x2       = 96
    v_wmma           = 16 score K-steps + 16 P*V tiles           = 32
    s_barrier        = 3 per round (Q+K published / K read / V published)
These come from `wmma_fattn_score.cpp`'s own constants (MROWS, MLANES, DH, NWARPS, read out of the
source below), which is the AGENTS.md rule that a must-hit set be derived, not remembered.

Usage:
    python3 prototype/attn_wmma/fattn_census.py [tmp/attn_wmma_fattn/wmma_fattn_score-*.s]
"""
from __future__ import annotations

import collections
import importlib.util
import re
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
REPO = HERE.parent.parent
DEFAULT_ASM = REPO / "tmp/attn_wmma_fattn/wmma_fattn_score-hip-amdgcn-amd-amdhsa-gfx1151.s"

# The shared definitions, imported so this census and the shipped 45 cannot drift in what a slot is.
_m16_spec = importlib.util.spec_from_file_location("m16_census", HERE / "m16_census.py")
m16 = importlib.util.module_from_spec(_m16_spec)
sys.modules["m16_census"] = m16
_m16_spec.loader.exec_module(m16)
base = m16.base                       # attn_score_dot2/isa_census.py: INSTR_RE, SKIP_AS_WORK, tally()

SRC = HERE / "wmma_fattn_score.cpp"

# ---------------------------------------------------------------------------
# The geometry, read out of the kernel source so the denominators cannot silently move.
# ---------------------------------------------------------------------------
_CONST: dict[str, int] = {}


def src_const(name: str) -> int:
    """Read a `#define`/`constexpr` integer from the kernel source. Values may be small expressions over
    earlier constants (`#define THREADS (NWARPS * 32)`), which are evaluated only after checking that the
    text contains nothing but names, digits, spaces and arithmetic."""
    if name in _CONST:
        return _CONST[name]
    txt = SRC.read_text()
    m = re.search(rf"^#define\s+{name}\s+(.+)$", txt, re.M) or \
        re.search(rf"constexpr int {name}\s*=\s*([^;]+);", txt)
    if not m:
        raise SystemExit(f"{SRC.name} no longer defines {name}; the denominators moved.")
    expr = m.group(1).split("//")[0].strip()   # a `#define` line usually carries its comment
    if not re.fullmatch(r"[A-Za-z_0-9 \t*+()/\-]+", expr):
        raise SystemExit(f"{name} = {expr!r}: the census can only expand arithmetic over source constants.")
    names = {k: v for k, v in _CONST.items() if re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", k)}
    val = int(eval(expr, {"__builtins__": {}}, names))   # guarded by the fullmatch above
    _CONST[name] = val
    return val


DH = src_const("DH")
MROWS = src_const("MROWS")
MLANES = src_const("MLANES")
NWARPS = src_const("NWARPS")
THREADS = src_const("THREADS")
KTILES = src_const("KTILES")
VTILES = src_const("VTILES")
TOUCHES = MROWS * MLANES                       # 256, the same denominator as m16_census.py
ROWS_PER_WARP_QK = MROWS // NWARPS             # 4 staging passes for Q, 4 for K
ROWS_PER_WARP_V = MLANES // NWARPS
WMMA_CLK = 32          # class roof: 4096 MAC / (32 lanes x 4 MAC/lane/clk)
WMMA_CLK_MEASURED = 36  # sustained 3.552 MAC/lane/clk, sensitivity only
BAR_PER_TOUCH = (WMMA_CLK * 16 * 16) // TOUCHES   # matrix clocks per touch, = 2.0

EXPECTED = {
    "v_wmma_f32_16x16x16_f16": KTILES + VTILES,
    "global_load_b128":       2 * ROWS_PER_WARP_QK + ROWS_PER_WARP_V,   # Q rows + K rows + V rows
    "ds_store_b128":          2 * ROWS_PER_WARP_QK,                     # Q and K land wide
    "ds_store_b16*":          ROWS_PER_WARP_V * 8,                      # V transposed: 8 per row
    "ds_load_b128":           2 * 2 * KTILES + 2 * VTILES,              # score A+B, PV A
    "s_barrier":              3,
}
# Lane-movement, from the transcribed shapes: 1 max fold (fattn-mma-f16.cuh:861-872 with
# offset_first == offset_last), 8 get_half2 exchanges (mma.cuh:729-743, one per C element), 1 rowsum
# combine that sits outside the round (fattn-mma-f16.cuh:1421). The first two are IN the round.
EXPECTED_LANE_MOVE_IN_ROUND = 1 + 8
EXPECTED_LANE_MOVE_TAIL = 1
FORM = "m16_fattn_score"
LEGS = [("m16_fattn_empty", None, "staging + 3 barriers + sink, zero matrix work"),
        ("m16_fattn_scoreleg", "m16_fattn_empty", "score tile + fold + exp + P (change 1 + 2)"),
        ("m16_fattn_score", "m16_fattn_scoreleg", "P*V: 16 independent f32 chains (change 3)")]

LANE_MOVE = m16.LANE_MOVE      # "SHUFFLE.lane-movement", the same category rule as D57


# ---------------------------------------------------------------------------
# Region expansion, PER REGION (see the header for why the kernel-wide rule cannot be reused).
# A back-edge region is found by `m16.loop_regions`; its trip count is derived from what the region
# carries: v_wmma -> KTILES / wmma-in-region, staging (a global load + an LDS store) -> the source's
# rows-per-warp, anything else -> 1 with the reason printed.
# ---------------------------------------------------------------------------
def expand(body: list[str]) -> tuple[collections.Counter, collections.Counter, int, int, list[str]]:
    """(mnemonic counts for ONE WARP-ROUND, category counts, slots, encode-only, derivation lines).

    Positions are tracked by line index, not by `list.index` (which would fold identical instruction
    lines onto the first one and quietly under-count a rolled loop).
    """
    insts = []                      # (position, mnemonic)
    for i, ln in enumerate(body):
        m = base.INSTR_RE.match(ln)
        if m:
            insts.append((i, m.group(1)))
    if not insts:
        raise SystemExit("no instructions recognised in this kernel body; the listing moved.")
    mnemonic: collections.Counter = collections.Counter()
    cats: collections.Counter = collections.Counter()
    lines: list[str] = []
    regions = m16.loop_regions(body)
    spans = []                      # (span, label, lo, hi, nb, trip, why)
    for lab, (lo, hi, nb) in regions.items():
        rms = [mn for pos, mn in insts if lo <= pos <= hi]
        if not rms:
            continue
        n_w = sum(1 for m in rms if m.startswith("v_wmma"))
        n_gl = sum(1 for m in rms if m.startswith("global_load"))
        n_st = sum(1 for m in rms if m.startswith("ds_store"))
        if n_w:
            if KTILES % n_w:
                raise SystemExit(f"{n_w} v_wmma in region {lab} does not divide KTILES={KTILES}.")
            trip, why = KTILES // n_w, f"{n_w} v_wmma in region -> KTILES/{n_w}"
        elif n_gl and n_st:
            trip, why = ROWS_PER_WARP_QK, f"staging region (global load + LDS store) -> rows/warp = {ROWS_PER_WARP_QK}"
        else:
            trip, why = 1, "no expansion fingerprint (straight-line or already unrolled): x1"
        spans.append((hi - lo, lab, lo, hi, nb, trip, why))
    # ONE FACTOR PER INSTRUCTION LINE. LLVM emits nested back edges whose spans differ only by a debug
    # label (`.LBB1_95 [1277,1550]` inside `.LBB1_96 [1279,1551]`), so counting every region that covers a
    # line would bill the unrolled P*V block twice — which is exactly how the first version of this file
    # read 48 v_wmma per round where the geometry says 32. Each line belongs to the WIDEST region that
    # covers it; two overlapping regions that both expand would need a product, and that is refused rather
    # than guessed.
    factor: dict[int, int] = {}
    owner_of: dict[int, str] = {}
    for span, lab, lo, hi, nb, trip, why in sorted(spans, key=lambda s: -s[0]):
        inside = [pos for pos, _ in insts if lo <= pos <= hi]
        clash = [p for p in inside if trip > 1 and factor.get(p, 1) > 1]
        if clash:
            raise SystemExit(f"region {lab} expands x{trip} and overlaps an expanding region "
                             f"({owner_of[clash[0]]}) at line {clash[0]}; the expansion would need a "
                             "product. Refusing to guess: split the kernel so each loop is its own case.")
        for pos in inside:
            if pos not in factor:
                factor[pos] = trip
                owner_of[pos] = lab
        if trip > 1:
            lines.append(f"    region {lab} [{lo},{hi}] ({nb} back edge{'s' if nb > 1 else ''}): "
                         f"{len(inside)} instrs, x{trip} ({why})")
    for pos, mn in insts:
        trip = factor.get(pos, 1)
        mnemonic[mn] += trip
        cats["<wait>" if mn.startswith(base.SKIP_AS_WORK) else m16.category(mn)] += trip
    slots = sum(v for k, v in cats.items() if k != "<wait>")
    enc = cats["<wait>"]
    assert slots > 0
    return mnemonic, cats, slots, enc, lines


# ---------------------------------------------------------------------------
# The occupancy rules. Rule R-512 is `m16_census.blocks_wavefronts`, the D57 policy; the two
# driver-shaped rules differ only in the register file per CU, which is the one figure the recorded
# evidence cannot pin down (every recorded row is LDS-bound). All three are printed, and the driver
# probe `occupancy_fattn_probe.cpp` is named as the authority when they disagree.
# ---------------------------------------------------------------------------
REGS_HIP = 196608       # hipDeviceProp_t::regsPerMultiprocessor, as §3.4 reads it
REGS_HALVED = 98304     # the same figure halved, per AGENTS.md: HIP reports a halved SM count here


def rules(lds: int, vgpr: int, warps: int) -> dict[str, int]:
    out = {}
    out["R-512 (D57 policy)"] = m16.blocks_wavefronts(lds, vgpr, warps)[0] if lds and vgpr else 0
    for name, regs in (("R-halved", REGS_HALVED), ("R-HIP", REGS_HIP)):
        by_lds = m16.LDS_PER_CU // lds if lds else 99
        by_reg = regs // (THREADS * m16.roundup(vgpr, m16.VGPR_GRAN)) if vgpr else 99
        by_wf = 32 // warps
        out[name] = max(0, min(by_lds, by_reg, by_wf))
    return out


def validate_rules() -> None:
    """A rule that cannot reproduce the recorded driver rows is not a rule. The rows are
    prototype/attn_wmma/evidence/occupancy-m16.txt and the shipped row m16_census.py:327 calibrates on."""
    rows = [  # (label, threads, vgpr, lds, recorded blocks/CU)
        ("attn_slim_wave (shipped)", 128, 47, 12288, 5),
        ("m16_bf16gq", 128, 30, 11264, 5),
        ("m16_empty", 128, 21, 16384, 4),
        ("m16_scalar_ob", 128, 152, 18432, 3),
        ("m16_bf16rt", 128, 235, 19456, 3),
    ]
    bad = {"R-512 (D57 policy)": [], "R-halved": [], "R-HIP": []}
    for name, thr, vgpr, lds, want in rows:
        got = rules(lds, vgpr, thr // 32)
        for rule, val in got.items():
            if val != want:
                bad[rule].append(f"{name}: says {val}, recorded {want}")
    print("OCCUPANCY RULE VALIDATION against evidence/occupancy-m16.txt (five recorded driver rows)")
    for rule, fails in bad.items():
        verdict = "reproduces all five rows" if not fails else "MISMATCH: " + "; ".join(fails)
        print(f"  {rule:<22} {verdict}")
    if not bad["R-halved"] and not bad["R-HIP"]:
        print("  note: the five recorded rows are all LDS-bound, so they do NOT separate R-halved from "
              "R-HIP;\n        a form whose register term could bind (this one) needs the driver itself: "
              "build_occupancy_fattn.sh")
    if bad["R-512 (D57 policy)"]:
        print("  => the D57 policy's 512-lane-VGPR-per-SIMD term is retired by this evidence "
              "(it is m16_census.py:325, kept here only as the comparison)")


def main() -> int:
    ap_args = sys.argv[1:]
    driver_blocks = None
    if "--driver-blocks" in ap_args:
        i = ap_args.index("--driver-blocks")
        if i + 1 >= len(ap_args):
            raise SystemExit("--driver-blocks needs a number; it is the driver's own answer from "
                             "build_occupancy_fattn.sh, and it is what decides claim (i).")
        driver_blocks = int(ap_args[i + 1])
        del ap_args[i:i + 2]
    path = Path(ap_args[0]) if ap_args else DEFAULT_ASM
    if not path.exists():
        raise SystemExit(f"no listing at {path}; run: bash prototype/attn_wmma/build_fattn_score.sh")
    text = path.read_text()
    if "ms=" in text or " GB/s" in text:
        raise SystemExit("the listing carries a timing figure; this census must not read one.")
    kern = m16.split_kernels(text)
    meta = m16.split_meta(text)
    want = [k for k, _, _ in LEGS]
    missing = [k for k in want if k not in kern or not kern[k]]
    if missing:
        raise SystemExit(f"MISSING KERNELS {missing}; present {sorted(kern)}. A census that skips a leg "
                         "prints a plausible number for the ones it did read.")
    extra = [k for k in kern if k not in want and not k.startswith("__hip_")]
    if extra:
        raise SystemExit(f"unexpected kernels {extra}; the listing is not the one this census prices "
                         f"(expected exactly {want})")

    print(f"listing          {path}")
    print(f"geometry         DH={DH} MROWS={MROWS} MLANES={MLANES} NWARPS={NWARPS} THREADS={THREADS} "
          f"KTILES={KTILES} VTILES={VTILES}")
    print(f"denominator      {TOUCHES} touches per warp-round (16 Q rows x 16 KV lanes x one head)")
    print(f"matrix bill      1 v_wmma = {WMMA_CLK} clk pipe time (class roof); "
          f"{BAR_PER_TOUCH} clk/touch, {WMMA_CLK * (KTILES + VTILES) / TOUCHES:.1f} for "
          f"{KTILES + VTILES} tiles per round\n")
    validate_rules()
    print()

    tables: dict[str, tuple[collections.Counter, collections.Counter]] = {}
    for name, _, _ in LEGS:
        mn, cats, slots, enc, lines = expand(kern[name])
        tables[name] = (mn, cats)
        m = meta.get(name, {})
        print(f"== {name}   {m.get('vgpr', '?')} VGPR (gran {m16.roundup(m.get('vgpr', 0), 8)}), "
              f"{m.get('sgpr', '?')} SGPR, LDS {m.get('lds', '?')} B, "
              f"vgpr_spill {m.get('spill', '?')}")
        for ln in lines:
            print(ln)
        occ = rules(m.get("lds", 0), m.get("vgpr", 0), NWARPS)
        print(f"   one warp-round: {slots} slots + {enc} encode-only = {slots + enc} instructions "
              f"for {TOUCHES} touches")
        print("   occupancy: " + ", ".join(f"{k} = {v} blocks/CU ({v * NWARPS} wf/CU)"
                                           for k, v in occ.items()))
        print("   " + "-" * 74)
        for cat in sorted(cats):
            print(f"     {cat:<26} {cats[cat]:>6}   {cats[cat] / TOUCHES:>7.3f}/touch")
        print()

    # ---- the derived canary: the expanded per-round counts must equal what the source geometry says
    print("== CANARY: expanded per-warp-round counts vs the numbers the source geometry implies")
    catsum_mn = tables[FORM][0]
    got = {
        "v_wmma_f32_16x16x16_f16": catsum_mn["v_wmma_f32_16x16x16_f16"],
        "global_load_b128":        catsum_mn["global_load_b128"],
        "ds_store_b128":           catsum_mn["ds_store_b128"],
        "ds_store_b16*":           sum(v for k, v in catsum_mn.items() if k.startswith("ds_store_b16")),
        "ds_load_b128":            catsum_mn["ds_load_b128"],
        "s_barrier":               catsum_mn["s_barrier"],
    }
    bad = []
    for key, want_n in EXPECTED.items():
        g = got[key]
        flag = "ok" if g == want_n else "MISMATCH"
        if g != want_n:
            bad.append(f"{key}: listing says {g}, geometry says {want_n}")
        print(f"  {key:<26} {g:>5} want {want_n:>5}  {flag}")
    if bad:
        raise SystemExit("REGION EXPANSION IS WRONG — " + "; ".join(bad) +
                         ". No verdict is printed above this line, and none should be read below it.")
    print("  PASS: the expansion reproduces the derived instruction census exactly")

    # ---- the two claims
    print("\n== CLAIMS")
    for name, _, _ in LEGS:
        _, c = tables[name]
        work = sum(v for k, v in c.items() if k != "<wait>")
        instr = work + c["<wait>"]
        n_w = c["MMA.wmma"]
        pipe = (instr - n_w + WMMA_CLK * n_w) / TOUCHES
        pipe_m = (instr - n_w + WMMA_CLK_MEASURED * n_w) / TOUCHES
        m = meta[name]
        occ = rules(m.get("lds", 0), m.get("vgpr", 0), NWARPS)
        print(f"  {name:<20} slots/touch {work / TOUCHES:>7.3f}   "
              f"instr/touch {instr / TOUCHES:>7.3f}   PIPE-WEIGHTED {pipe:>7.3f} "
              f"(36 clk: {pipe_m:.3f})   blocks/CU {occ['R-halved']}..{occ['R-HIP']} "
              f"({occ['R-halved'] * NWARPS}..{occ['R-HIP'] * NWARPS} wf/CU)")
    c = tables[FORM][1]
    work = sum(v for k, v in c.items() if k != "<wait>")
    instr = work + c["<wait>"]
    n_w = c["MMA.wmma"]
    pipe = (instr - n_w + WMMA_CLK * n_w) / TOUCHES
    m = meta[FORM]
    occ = rules(m.get("lds", 0), m.get("vgpr", 0), NWARPS)
    worst = min(occ["R-halved"], occ["R-HIP"]) if driver_blocks is None else driver_blocks
    print(f"\n  (i)   OCCUPANCY (bar: >= 4 blocks/CU)")
    print(f"        rule range: {occ['R-halved']} blocks/CU on a 98,304 lane-VGPR file ("
          f"HIP's 196,608 halved, per AGENTS.md's halved-SM-count trap) to {occ['R-HIP']} on "
          f"HIP's 196,608 as §3.4 reads it")
    print(f"        {meta[FORM]['vgpr']} VGPR x {THREADS} threads; LDS {meta[FORM]['lds']} B allows "
          f"{m16.LDS_PER_CU // meta[FORM]['lds']}, and that is the ONLY term that binds if the file "
          f"is 196,608")
    if driver_blocks is None:
        print(f"        INDETERMINATE from the rules alone: worst case {worst} -> "
              f"{'PASS' if worst >= 4 else 'FAIL'}. Re-run with --driver-blocks N to record the "
              f"driver's own answer, which is the authority §3.4 names.")
    else:
        print(f"        DRIVER AUTHORITY (hipOccupancyMaxActiveBlocksPerMultiprocessor, "
              f"occupancy_fattn_probe): {driver_blocks} blocks/CU, {driver_blocks * NWARPS} "
              f"wavefronts/CU of the 32 the part allows -> "
              f"{'PASS' if driver_blocks >= 4 else 'FAIL'}")
        if driver_blocks not in (occ["R-halved"], occ["R-HIP"]):
            print(f"        note: the driver matches NEITHER rule; the occupancy model here is wrong "
                  f"and must be re-derived before any block-count claim stands.")
    print(f"  (ii)  PIPE-WEIGHTED {pipe:.3f} slots/touch, bar <= 20 -> "
          f"{'PASS' if pipe <= 20 else 'FAIL'}")
    print(f"        matrix term alone is {WMMA_CLK * n_w / TOUCHES:.2f} of it; non-matrix "
          f"{(instr - n_w) / TOUCHES:.2f} — the room §4.2 bets on is the non-matrix term")

    print("\n== LEG DIFFERENCES (each kernel differs from the one on the right by ONE leg)")
    for name, minus, what in LEGS[1:]:
        a, b = tables[name][1], tables[minus][1]
        keys = set(a) | set(b)
        d = {k: a[k] - b[k] for k in keys}
        dw = sum(v for k, v in d.items() if k != "<wait>")
        di = sum(d.values())
        dn = d.get("MMA.wmma", 0)
        print(f"  {name} - {minus}  ({what})")
        print(f"     slots {dw:>6} = {dw / TOUCHES:6.3f}/touch   instr {di:>6} = {di / TOUCHES:6.3f}/touch"
              f"   pipe-weighted {(di - dn + WMMA_CLK * dn) / TOUCHES:6.3f}/touch")
        top = sorted(((v, k) for k, v in d.items() if v), reverse=True)[:6]
        print("     by category: " + "  ".join(f"{k}={v}" for v, k in top))

    # ---- the items §4.2 / the task asks to be named
    print("\n== FRAGMENT RELABELLING, SPILLS, Q RESIDENCY, ADDRESS DISCIPLINE")
    lm = c[LANE_MOVE]
    cvt = sum(v for k, v in c.items() if k.startswith("CVT"))
    print(f"  lane-movement ops (ds_swizzle/ds_permute/v_permlane/ds_bpermute): {lm} per warp-round "
          f"= {lm / TOUCHES:.4f}/touch")
    print(f"     expected for {FORM}: {EXPECTED_LANE_MOVE_IN_ROUND} inside the round (1 max fold, "
          f"fattn-mma-f16.cuh:861-872 with offset_first == offset_last; + 8 get_half2 exchanges, "
          f"mma.cuh:729-743) + {EXPECTED_LANE_MOVE_TAIL} deferred rowsum combine outside it "
          f"(fattn-mma-f16.cuh:1421) = {EXPECTED_LANE_MOVE_IN_ROUND + EXPECTED_LANE_MOVE_TAIL}")
    lm_marg = tables[FORM][1][LANE_MOVE] - tables["m16_fattn_scoreleg"][1][LANE_MOVE]
    print(f"     the P*V leg alone adds {lm_marg} lane-movement ops over the score leg "
          f"(the get_half2 exchange is what change (3) pays)")
    print(f"  CVT-class ops: {cvt} per warp-round = {cvt / TOUCHES:.3f}/touch "
          f"(the bf16->f16 operand conversion at staging is most of this)")
    spills = {k: meta[k].get("spill", "?") for k, _, _ in LEGS}
    print(f"  vgpr_spill_count per kernel: {spills}")
    print(f"  private_segment (scratch) bytes per thread: "
          f"{ {k: priv_segment(text, k) for k, _, _ in LEGS} }")
    print("     why this is printed: a rolled accumulator loop puts the 128 accumulator registers in")
    print("     scratch at 544 B/thread while clang's own remark still says \"VGPRs Spill: 0\". The")
    print("     note block's private segment is what caught it, so it is checked, not assumed.")
    print(f"  Q in registers? NO, by construction: the score A operand is an LDS read — "
          f"{2 * KTILES * 2} ds_load_b128 per round are the score leg's A+B tiles (2 per operand), "
          f"so Q is LDS-resident, not register-resident. Register-resident Q would be 8 VGPR per "
          f"K-step x {KTILES} steps of live values.")
    discipline()
    return 0


def priv_segment(text: str, name: str) -> int:
    """`.private_segment_fixed_size` for one kernel, out of the same note-block chunks `split_meta` uses.
    Non-zero here means the accumulator spilled to scratch, which `vgpr_spill_count` did not report."""
    for chunk in re.split(r"\n\s*- \.args:", text):
        if re.search(rf"^\s*\.name:\s+{name}$", chunk, re.M):
            m = re.search(r"\.private_segment_fixed_size:\s+(\d+)", chunk)
            if m:
                return int(m.group(1))
    raise SystemExit(f"no private_segment figure for {name}; the note block moved.")


def discipline() -> None:
    """The §20.1 guard: every write and every read of a staged object must go through the one named
    address expression for that object. Printed as a table of sites, because a claim nobody can point at
    is a note and not a forcing."""
    txt = SRC.read_text()
    sites = collections.Counter()
    offenders = []
    for i, ln in enumerate(txt.splitlines(), 1):
        if ln.lstrip().startswith("//") or "__shared__" in ln:
            continue          # a declaration `__shared__ __half s_q[MROWS * DH]` is not an access site
        for obj in ("s_q", "s_kv"):
            for m in re.finditer(rf"\b{obj}\[([^\]]*)\]", ln):
                sites[obj] += 1
                if "_addr(" not in m.group(1):
                    offenders.append(f"{SRC.name}:{i}: {obj}[{m.group(1)}]")
        for m in re.finditer(r"\bdst\[([^\]]*)\]", ln):
            sites["staged dst"] += 1
            if "_addr(" not in m.group(1):
                offenders.append(f"{SRC.name}:{i}: dst[{m.group(1)}]")
    print("  address expressions, one per staged object (AGENTS.md §20.1: two self-consistent layouts")
    print("  joined by a branch are invisible to both sides):")
    for obj, n in sorted(sites.items()):
        print(f"     {obj:<12} {n} sites, all through a *_addr() helper")
    if offenders:
        raise SystemExit("ADDRESS DISCIPLINE BROKEN (a raw index into a staged array): "
                         + "; ".join(offenders))
    print("     PASS: no raw index into a staged array anywhere in the source")
    print("  note: s_kv carries TWO layouts (k_addr for K, v_addr for V transposed) with a __syncthreads")
    print("  between the last K reader and the first V writer. That is the aliasing llama.cpp does at")
    print("  fattn-mma-f16.cuh:1252 when nstages == 0, and it is forced: 3 x 8 KiB of stages is 24 KiB,")
    print("  which is 2 blocks/CU on a 64 KiB LDS.")


if __name__ == "__main__":
    sys.exit(main())
