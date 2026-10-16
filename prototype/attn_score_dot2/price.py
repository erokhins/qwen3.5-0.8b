#!/usr/bin/env python3
"""price.py — bottom-up price for "replace the attention score dot's scalar FMA chain with v_dot2_f32_bf16".

The question this answers, in one sentence: **what fraction of the M = 2 form's measured 2.5402 ms residual
at 4096 lanes could a 3-4x faster SCORE DOT actually remove?** The brief's kill line is ~15 % of the wave;
if the prize is under that, no kernel gets written (decision log D19 asked for this price as the partner
lever to row-tiling).

CPU-only. No device, no timing, no gpu_lock: every input is either a constant of the kernel, a number
another tool already measured and recorded, or arithmetic on those. Inputs name the file they came from.

Run:  .venv/bin/python prototype/attn_score_dot2/price.py
      .venv/bin/python prototype/attn_score_dot2/price.py --isa-dir /tmp/isap    # re-check the census

THE ANSWER, printed below and argued in results.md: **4.3 % to 10.5 % of the wave, depending on which
denominator you allow, against a 15 % line. STOP.** The score dot's arithmetic is small: on the shipped
form a (row, lane, head) touch costs ~115 VALU warp-instructions (measured by counter, not modelled) and
the dot inside it is 16 of them, of which a dot2 removes 12. The rest of the touch is reduction guards,
`expf`, an IEEE divide, BF16 conversions and LDS address arithmetic — none of which a dot2 touches. The
PV dot cannot use dot2 at all: `v_dot2_*` folds two products into ONE accumulator and PV needs eight.
"""
from __future__ import annotations

import argparse
import dataclasses
import re
from pathlib import Path

from isa_census import split_kernels

# ============================================================================================
# 1. Inputs. Every constant names where it comes from; nothing here is a guess dressed as a fact.
# ============================================================================================

# Kernel geometry — prototype/attn_decode/attn_decode_kernels.h.
N_Q, N_KV, GQA, DH, MAXSC = 8, 2, 4, 256, 512
LAYERS = 6                      # engine/state.h: 6 attention layers of 24 total

# Measured arithmetic ceilings, T-MAC/s = TFLOP/s ÷ 2 — prototype/issue_rate_probe/results.md §1.
# "Sustained" is the probe's own best-of-20x5 leg on this part, not a roof: the roofs are 7.41 (scalar,
# 1 MAC/lane/clk), 14.82 (dot2, 2) and 29.70 T-MAC/s (WMMA, 4).
SCALAR_MAC_S = 3.65e12          # v_fma_f32 leg, 49.3 % of its roof
DOT2_BF16_MAC_S = 11.64e12      # v_dot2_f32_bf16 leg, 78.6 % of its roof — 3.19x the scalar leg
WMMA_MAC_S = 26.31e12           # v_wmma_f32_16x16x16_f16 leg, for the "what would pay" section

# Issue roof: 2 warp-instructions per CU per clock. rocminfo says 2 SIMDs per CU, and the probe's roofs
# use 40 CU x 64 FP32 lanes, which is the same fact — decision log D16 item 3 and
# progress/2026-10-16/prefill-occupancy-results-2026-10-16.md §3.
CU, CLK_GHZ = 40, 2.894
ISSUE_ROOF = 2 * CU * CLK_GHZ * 1e9        # = 2.3152e11 warp-inst/s

# THE PIN. `rocprofv3 --pmc SQ_INSTS_VALU` on the shipped form: attn_tile_wave = 2,888 M VALU
# warp-instructions for a 1024-lane prefill, measured 27.23 ms
# (progress/2026-10-16/prefill-occupancy-results-2026-10-16.md §3).
PIN_VALU_1024 = 2_888e6
PIN_MEASURED_S = 27.23e-3
PIN_LANES_1024 = 1024 * 1025 // 2           # Σ_lanes over the 1024 query rows, one layer
PIN_TOUCHES = PIN_LANES_1024 * LAYERS * N_Q  # (row, lane, head) triples in that census
TOUCH_VALU = PIN_VALU_1024 / PIN_TOUCHES     # -> 114.6 warp-inst per touch; the doc says "~115"
ACHIEVED_RATE = PIN_VALU_1024 / PIN_MEASURED_S

# The score loop's own cost, COUNTED from the disassembly of `attn_tile_wave` (see results.md §2 for the
# instruction windows this is read off, and isa_census.py for the tool). VALU warp-instructions per touch.
DOT_SCALAR_VALU = 8 + 8      # 8 bf16->f32 of the staged K row (1 instr each) + 8 v_fmac_f32_e32
DOT_DOT2_VALU = 4            # 4 x v_dot2_f32_bf16, no conversions: compile-proved, results.md §2.2
SCORE_REDUCE_VALU = 20       # 5 shuffle steps x 4 VALU (mask compare, cndmask, addr, add) — the guard
                             # per step exists because the compiler cannot prove FULL_MASK is exec
SCORE_OTHER_VALU = 6         # BF16 round of the dot, x SCALE, index arithmetic
SCORE_LOOP_VALU = DOT_SCALAR_VALU + SCORE_REDUCE_VALU + SCORE_OTHER_VALU

# The measured waves — decision log D19's table, window `attn-m2-2` (paired in one lock hold).
@dataclasses.dataclass(frozen=True)
class Wave:
    tag: str
    first_lane: int
    nrows: int
    ship_ms: float
    m2_ms: float

    @property
    def lanes(self) -> list[int]:
        return [self.first_lane + r for r in range(self.nrows)]

    @property
    def sum_lanes(self) -> int:
        return sum(self.lanes)

    @property
    def touches(self) -> int:
        return self.sum_lanes * N_Q


WAVES = [
    Wave("wave64-at-1024", 1021, 64, 0.5991, 0.6511),
    Wave("wave64-at-2048", 2045, 64, 1.2753, 1.3546),
    Wave("wave64-at-4096", 4033, 64, 2.4163, 2.5402),
    Wave("wave16-at-4096", 4081, 16, 0.7166, 0.7667),
]

PRIZE_LINE = 0.15      # the brief's kill line, as a fraction of the wave


# ============================================================================================
# 2. The arithmetic
# ============================================================================================
def macs(w: Wave) -> dict[str, float]:
    """MACs per wave, split by what the kernel actually does with them."""
    t = w.touches
    return {
        "score dot (q.k, 256 MAC per touch)": t * DH,
        "PV dot (weights x v, 256 MAC per touch)": t * DH,
        # rope + §05.2 norm + §05.7 append run in `attn_prep`, one block per head, 256 dims each:
        # ~6 flops/dim for the norm+rope and ~4 for the append's copies. Not part of the wave launch
        # the burst measured, listed because the brief asked for "everything else".
        "rope + per-head norm (attn_prep)": w.nrows * N_Q * DH * 6,
        "§05.6 epilogue (div, sigmoid gate, 2 rounds)": w.nrows * N_Q * DH * 12,
    }


def instructions(w: Wave) -> dict[str, float]:
    """VALU warp-instructions per wave, same split, read off the measured per-touch census.

    The funded waves run the STREAMING route (`s_row == 0`: at 4096 lanes with split 8 no block owns a
    whole row), where `expf` is paid once per touch and the normalising divide happens once per
    (row, head) in the epilogue. The one-pass route (`s_row == 1 && a.full`, rows at or below `MAXSC`
    lanes) pays `expf` twice per touch plus a full IEEE divide per touch — see results.md §5.
    """
    t = w.touches
    return {
        "score dot (the only dot2-shaped leg)": t * DOT_SCALAR_VALU,
        "score reduction (5 shuffle steps + guards)": t * SCORE_REDUCE_VALU,
        "score round/scale/addressing": t * SCORE_OTHER_VALU,
        "pass 2 by subtraction: expf, V conversions, 8 FMAs, rescale, address math":
            t * (TOUCH_VALU - SCORE_LOOP_VALU),
    }


def price(w: Wave) -> dict[str, float]:
    """Three framings of the same question. Each is an upper bound in a different sense."""
    t = w.touches
    # (A) MAC-roof framing: what the score dot costs at the two sustained ceilings. This is the price
    #     of the multiply-adds themselves and ignores every instruction that is not a MAC.
    a_now = t * DH / SCALAR_MAC_S
    a_dot2 = t * DH / DOT2_BF16_MAC_S
    mac_saving = a_now - a_dot2
    # (B) Instruction framing, and the reason it is an UPPER bound: removing a fraction f of the
    #     instructions cannot remove more than f of the time unless the kernel is issue-saturated and
    #     time is exactly proportional to instructions, which is the equality case. The shipped form
    #     achieves 106 G warp-inst/s of a 231.5 G/s roof (45.8 %), so it is not saturated and the
    #     realistic saving is below this line.
    removable = t * (DOT_SCALAR_VALU - DOT_DOT2_VALU)
    frac_inst = (DOT_SCALAR_VALU - DOT_DOT2_VALU) / TOUCH_VALU
    inst_saving_at_roof = removable / ISSUE_ROOF
    inst_saving_at_achieved = removable / ACHIEVED_RATE
    return {
        "score_at_scalar_s": a_now,
        "score_at_dot2_s": a_dot2,
        "A_mac_saving_s": mac_saving,
        "A_pct_of_m2": mac_saving / (w.m2_ms * 1e-3),
        "A_pct_of_ship": mac_saving / (w.ship_ms * 1e-3),
        "removable_warp_inst": removable,
        "frac_instructions": frac_inst,
        "B_pct_bound": frac_inst,
        "B_saving_at_roof_s": inst_saving_at_roof,
        "B_saving_at_achieved_s": inst_saving_at_achieved,
        "wave_valu_at_roof_s": t * TOUCH_VALU / ISSUE_ROOF,
        "wave_valu_at_achieved_s": t * TOUCH_VALU / ACHIEVED_RATE,
    }


def check_isa(tile_s: Path, toy_s: Path) -> list[str]:
    """Re-verify, from a real disassembly, the two ISA claims the price leans on. Both are static and
    CPU-only; if the files are absent the price still stands on the windows quoted in results.md §2, and
    this says so out loud instead of silently skipping (a check that can be empty cannot fail)."""
    out: list[str] = []
    conv = re.compile(r"v_lshlrev_b32_e32\s+\S+, 16,|v_and_b32_e32\s+\S+, 0xffff0000")
    fma = re.compile(r"v_fma[ck]?_f32")
    if tile_s.exists():
        kernels = split_kernels(tile_s.read_text().splitlines())
        key = next((k for k in kernels if "attn_tile_wave" in k), None)
        body = "\n".join(kernels.get(key, [])) if key else ""
        if not body:
            out.append(f"{tile_s}: no attn_tile_wave found (regenerate per HANDOFF step 1)")
        else:
            out.append(f"{key} static: packed-half bf16->f32 conversions={len(conv.findall(body))}, "
                       f"v_fma*/v_fmac_f32={len(fma.findall(body))}. This is a PRESENCE check only - "
                       f"the static count also covers rope and the epilogue, so the per-touch figure "
                       f"(8 conversions + 8 FMAs) comes from the window quoted in results.md 2.1.")
    else:
        out.append(f"{tile_s} not present; results.md 2.1 quotes the window instead")
    if toy_s.exists():
        kernels = split_kernels(toy_s.read_text().splitlines())
        for name in ("score_scalar", "score_dot2"):
            body = "\n".join(kernels.get(name, []))
            if not body:
                out.append(f"{toy_s}: no {name} (rebuild per HANDOFF step 2)")
                continue
            out.append(f"{name}: v_dot2={len(re.findall('v_dot2_f32_bf16', body))} "
                       f"conversions={len(conv.findall(body))} "
                       f"fma={len(fma.findall(body))} "
                       f"- the scalar form must show conversions+fma, the dot2 form 4 dot2 and 0 conv")
    else:
        out.append(f"{toy_s} not present; build it with the HANDOFF step 2 command")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--isa-dir", default="/tmp/isap",
                    help="directory holding the generated .s files, for a census re-check")
    args = ap.parse_args()

    print("=" * 100)
    print("1. PINS AND CEILINGS (all measured elsewhere; nothing here is a guess)")
    print("=" * 100)
    print(f"  touches in the counter census   = {PIN_TOUCHES:,.0f}  "
          f"(Sigma_lanes {PIN_LANES_1024:,} x {LAYERS} layers x {N_Q} heads)")
    print(f"  VALU warp-inst per touch        = {TOUCH_VALU:.1f}   "
          f"(= {PIN_VALU_1024:,.0f} / touches; the occupancy doc prints '~115')")
    print(f"  achieved VALU rate              = {ACHIEVED_RATE / 1e9:.1f} G warp-inst/s "
          f"= {ACHIEVED_RATE / ISSUE_ROOF:.1%} of the 2/CU/clk roof {ISSUE_ROOF / 1e9:.1f} G")
    print(f"  scalar dot ceiling              = {SCALAR_MAC_S / 1e12:.2f} T-MAC/s (7.30 TFLOP/s)")
    print(f"  dot2-bf16 dot ceiling           = {DOT2_BF16_MAC_S / 1e12:.2f} T-MAC/s (23.28 TFLOP/s) "
          f"= {DOT2_BF16_MAC_S / SCALAR_MAC_S:.2f}x the scalar leg")
    print(f"  score loop, counted from the ISA= {SCORE_LOOP_VALU:.0f} VALU/touch "
          f"(dot {DOT_SCALAR_VALU} + reduction {SCORE_REDUCE_VALU} + round/address {SCORE_OTHER_VALU})")
    print(f"  -> the score loop is {SCORE_LOOP_VALU / TOUCH_VALU:.1%} of a touch's VALU stream, "
          f"and the dot alone is {DOT_SCALAR_VALU / TOUCH_VALU:.1%}")
    for line in check_isa(Path(args.isa_dir) / "isa_tu-hip-amdgcn-amd-amdhsa-gfx1151.s",
                          Path(args.isa_dir) / "dot2_shape-hip-amdgcn-amd-amdhsa-gfx1151.s"):
        print(f"  ISA re-check: {line}")

    print()
    print("=" * 100)
    print("2. MACs PER WAVE, BY PART, AND THE CEILING EACH PART RUNS AT")
    print("=" * 100)
    for w in WAVES:
        p = price(w)
        print(f"\n  {w.tag}: {w.nrows} rows, lanes {w.lanes[0]}..{w.lanes[-1]}, "
              f"Sigma_lanes {w.sum_lanes:,}, touches {w.touches:,}")
        for name, m in macs(w).items():
            print(f"    {name:<52} {m / 1e6:>10.1f} M MAC")
        print(f"    {'score dot at the scalar ceiling':<52} "
              f"{p['score_at_scalar_s'] * 1e3:>10.4f} ms")
        print(f"    {'score dot at the dot2-bf16 ceiling':<52} "
              f"{p['score_at_dot2_s'] * 1e3:>10.4f} ms")
        print(f"    {'PV dot at the scalar ceiling (NOT dot2-shaped: 8 accumulators)':<52} "
              f"{w.touches * DH / SCALAR_MAC_S * 1e3:>10.4f} ms")
        print(f"    {'whole VALU stream at the 2/CU/clk roof':<52} "
              f"{p['wave_valu_at_roof_s'] * 1e3:>10.4f} ms")
        print(f"    {'whole VALU stream at the achieved rate':<52} "
              f"{p['wave_valu_at_achieved_s'] * 1e3:>10.4f} ms  "
              f"(x{p['wave_valu_at_achieved_s'] / (w.ship_ms * 1e-3):.0%} of ship)")
        print(f"    MEASURED: ship {w.ship_ms:.4f} ms | m2 {w.m2_ms:.4f} ms")

    print()
    print("=" * 100)
    print("3. THE PRICE: what a 3-4x faster score dot removes, three ways")
    print("=" * 100)
    print("  (A) MAC-roof: only the multiply-adds move, at the two sustained ceilings.")
    print("  (B) Instruction share: the dot2 removes "
          f"{DOT_SCALAR_VALU - DOT_DOT2_VALU} of {TOUCH_VALU:.1f} VALU warp-inst per touch = "
          f"{(DOT_SCALAR_VALU - DOT_DOT2_VALU) / TOUCH_VALU:.1%} of the stream;")
    print("      that fraction is an UPPER bound on the time saving, because a kernel at 46 % of the")
    print("      issue roof does not convert freed slots into freed time 1:1.")
    for w in WAVES:
        p = price(w)
        print(f"\n  {w.tag}:")
        print(f"    (A) {p['A_mac_saving_s'] * 1e3:.4f} ms = {p['A_pct_of_m2']:.1%} of the m2 residual "
              f"({w.m2_ms:.4f} ms) | {p['A_pct_of_ship']:.1%} of ship ({w.ship_ms:.4f} ms)")
        print(f"    (B) bound = {p['B_pct_bound']:.1%} of the wave = "
              f"{p['B_saving_at_roof_s'] * 1e3:.4f} ms at the roof, "
              f"{p['B_saving_at_achieved_s'] * 1e3:.4f} ms at the achieved rate")
    p4096 = price(WAVES[2])
    lo, hi = p4096["A_pct_of_m2"], p4096["B_pct_bound"]
    need = WAVES[2].m2_ms - 1.25          # D19's target for the pair: ~1.2-1.3 ms per wave at 4096
    print()
    print(f"  PRIZE BAND at the funded width (64 rows at 4096 lanes): "
          f"{lo:.1%} .. {hi:.1%} of the m2 form's {WAVES[2].m2_ms:.4f} ms residual "
          f"({p4096['A_mac_saving_s'] * 1e3:.3f} .. {p4096['B_saving_at_achieved_s'] * 1e3:.3f} ms).")
    print(f"  D19's pair target is ~1.25 ms per wave at 4096, i.e. removing ~{need:.2f} ms. A dot2 removes "
          f"at most {p4096['B_saving_at_achieved_s'] * 1e3:.3f} ms = "
          f"{p4096['B_saving_at_achieved_s'] * 1e3 / (need * 1e3):.0%} of what that target needs.")
    print(f"  ROBUSTNESS: the 114.6 warp-inst/touch census is the SHIPPED form's. The m2 form's per-touch "
          f"count cannot be materially lower (the arithmetic per touch is identical; what the pair shares is")
    print(f"    loads, which are not VALU), and any excess bookkeeping makes the share SMALLER: at 20 % more "
          f"instructions per touch the bound is {12 / (TOUCH_VALU * 1.2):.1%}, not "
          f"{(DOT_SCALAR_VALU - DOT_DOT2_VALU) / TOUCH_VALU:.1%}.")
    print(f"  FALSIFIER (HANDOFF step 5 measures this): the instruction bound is "
          f"{(DOT_SCALAR_VALU - DOT_DOT2_VALU)} / census, so the verdict flips only if attn_m2_wave costs "
          f"fewer than {12 / PRIZE_LINE:.0f} VALU warp-inst per touch.")
    print(f"  KILL LINE: {PRIZE_LINE:.0%} of the wave.  "
          f"{'BELOW -> STOP, do not write the kernel' if hi < PRIZE_LINE else 'AT OR ABOVE -> fund step 2'}")

    print()
    print("=" * 100)
    print("4. THE 0.246 ms CLAIM, CHECKED AGAINST THE PINS")
    print("=" * 100)
    w1024 = WAVES[0]
    p1024 = price(w1024)
    print(f"  At the 1024-lane wave ({w1024.tag}, ship {w1024.ship_ms:.4f} ms) the score dot's real")
    print(f"  arithmetic cost is {p1024['score_at_scalar_s'] * 1e3:.4f} ms at the sustained scalar rate "
          f"({w1024.touches * DH / 1e6:,.1f} M MAC),")
    print(f"  and its whole score LOOP is {SCORE_LOOP_VALU / TOUCH_VALU:.1%} of its VALU stream = "
          f"{w1024.touches * SCORE_LOOP_VALU / ISSUE_ROOF * 1e3:.4f} ms at the roof, "
          f"{w1024.touches * SCORE_LOOP_VALU / ACHIEVED_RATE * 1e3:.4f} ms at the achieved rate.")
    whole_roof = p1024["wave_valu_at_roof_s"]
    print(f"  What DOES come out near 0.246 ms: the ENTIRE per-touch VALU stream of that wave at the")
    print(f"  2-warp-inst/CU/clk roof = {whole_roof * 1e3:.4f} ms "
          f"({w1024.touches:,.0f} touches x {TOUCH_VALU:.1f} warp-inst / {ISSUE_ROOF:.3e}).")
    print("  So the figure is the whole instruction bill, not the score dot's: quoting it as the score's")
    print("  cost overstates the prize by "
          f"~{whole_roof / p1024['score_at_scalar_s']:.0f}x. (The claim named in the brief is not in the")
    print("  tree; this is the only reading of it that reproduces the number, and it is arithmetic, not")
    print("  a citation.) The nearest ancestor that IS in the tree is the 2026-10-14 pre-tile census")
    print("  ('~1,030 warp-inst per touch in the one-pass form, of which the score-tile loop is 190 and")
    print("  the softmax+PV loop ~840', progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md §2). Read")
    pre190 = w1024.touches * 190 / (4 * CU * CLK_GHZ * 1e9)
    print(f"  that census at the then-assumed 4 inst/CU/clk roof gives {pre190 * 1e3:.3f} ms, the same")
    print("  neighbourhood - so the figure is underdetermined, and two ancestries land near it. What is")
    print("  NOT plausible under any reading is that it is the score dot's MAC cost. Note also that 190/1030")
    print("  = 18.4 %: even in the pre-tile form the score loop was under a fifth of the touch, which is")
    print("  the same conclusion this price reaches from the shipped form's census.")

    print()
    print("=" * 100)
    print("5. WHAT THE OTHER 85 % IS (so the next lever is chosen from this table, not from hope)")
    print("=" * 100)
    t = WAVES[2].touches
    for name, n in instructions(WAVES[2]).items():
        print(f"  {name:<62} {n / 1e6:>9.1f} M warp-inst  {n / (t * TOUCH_VALU):>6.1%}")
    print(f"  {'TOTAL (matches the counter census)':<62} "
          f"{t * TOUCH_VALU / 1e6:>9.1f} M warp-inst   100.0%")
    print("""
  The pass-2 pile is where the instructions live. On the streaming route these waves take, its three
  biggest items are all bigger than the dot, and naming them here is a census, not a funding request:
    * `expf` is inlined as ~14 VALU ops (v_mul_f32_e32, v_rndne_f32_e32, v_cvt_i32_f32_e32, v_exp_f32_e32,
      v_ldexp_f32, and the clamp cndmasks) and costs ~15 warp-inst per touch on this route; the one-pass
      route pays it TWICE per touch (`attn_decode
      _kernels.h`: the `l += expf(...)` sum loop, then `expf(...) / l` in the weight loop) plus a full
      IEEE divide (`v_div_scale_f32` / `v_rcp_f32` / `v_div_fixup_f32` / `v_div_fmas_f32`) - 34 per
      touch there. Reusing the first `expf` instead of recomputing it would need `ns` floats per
      thread, which is why the reference form pays it twice;
    * the PV dot is 8 one-instruction BF16 conversions + 8 FMAs = 16 per touch, the same size as the
      score dot, and it CANNOT be a dot2: `v_dot2_*` folds two products into ONE accumulator (its
      builtin returns a single f32) and PV needs eight, one per output dim;
    * the shuffle's per-step mask guard (compare + cndmask + address add + LDS permute, plus a full-exec
      test per step) is ~4 VALU per step for work that is dead when the mask really is `exec`.
  The one lever with real headroom against a 15% line is neither dot: it is `v_wmma_f32_16x16x16_f16`
  at 52.63 TFLOP/s (D19 says the same), which covers BOTH dots - 32 of 114.6 warp-inst per touch, 28 %
  of the stream, against two MFMA instructions per 16x16x16 tile. Pricing WMMA is a separate brief and
  is not answered here.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
