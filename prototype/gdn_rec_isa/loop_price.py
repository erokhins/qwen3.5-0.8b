#!/usr/bin/env python3
"""Price D53 — walk N tokens of the GDN recurrence inside ONE launch — WITHOUT touching the device.

Every input below is a number already in this repo, with its source named on the line. Nothing here is
measured; this is arithmetic on measurements, and it is written down so a reviewer can disagree with a
figure instead of disagreeing with a paragraph.

THE MODEL. `gdn_rec`'s family row has two separable parts:

    family_ms  =  launches x submission_floor      +   token-work
                  (what a launch costs the CPU, and §11 showed it is NOT absorbed)
                                                     (what one token-step costs the device)

The loop moves ONLY the first term — 73,728 launches at 4096 lanes become 18 layers x 8 bodies = 144 —
and it adds instructions to the second, because per-token addresses now depend on a loop index the
compiler cannot fold out. Both terms are computed, and the second is subtracted from the first: an
item priced only on the launch saving is the mistake D21/D46 named.

Three discount anchors are carried for the submission term, because the repo has measured the floor
(1.72 us, `tools/submission_probe.cpp`, §11) and has also measured what removing a launch was actually
worth in situ (Route A at 1024 lanes removed 18,396 launches and the family fell 125.36 -> 112.04 ms,
which is 0.72 us per removed launch — `prototype/gdn_collapse/results.md` §4). For the instruction
term, the elasticities on file are ~0.8 (D25, measured on this very kernel, on the axis that removes
work) and 0.39-0.44 (D21/D22, `time ~ slots^0.39-0.44`); the §10 `rec6` result (a 7.7 % slot cut that
returned 15 %) says a *spelling* delta can run near 2x its bill, so a 2x variant is carried too.

Usage:  .venv/bin/python prototype/gdn_rec_isa/loop_price.py
"""
from __future__ import annotations

# ---- inputs, each with the line it came from --------------------------------------------------------
SUBMISSION_US = 1.72          # §11 / D44: empty-launch floor, `tools/submission_probe.cpp`
COLLAPSE_US = 0.724           # derived: (125.36-112.04) ms / (18432-36) launches, collapse §4
WAVE_MS_4096 = 923.36        # §20, ruler median on the `fused` default
FAMILY_MS_4096 = 428.22      # §20, `gdn_rec` family row, paired `fused` side at 4096 lanes
WAVE_MS_1024 = 186.83        # §12
FAMILY_MS_1024 = 109.20      # §20, `fused` side at 1024 lanes
GDN_LAYERS = 18              # §00 / stage_clock's kNumGdnLayers
MAX_WAVE_ROWS = 512          # engine/decode_step.h:49 kMaxWaveRows — a wave is <= 512 rows per body
MIN_PCT_OF_WAVE = 0.1        # prototype/gemm_algo_sweep/deep_pins.py:80, the refuse bar
BILL_SHIPPING = 1374         # §11 recorded: rec2c4_kernel slots/token-step (== engine rec_fused)
BILL_LOOP = 1530             # this run's census: recloop_kernel slots/token-step (walk shapes 16x8/16x8,
                             # identical to the shipping form, so the delta is loop/addressing only)
# Elasticities on file for slots -> time on THIS kernel. D25 measured ~0.8 on the axis that removes
# work; D21/D22 measured time ~ slots^0.39-0.44; §10's `rec6` returned ~2x its own slot delta, so a
# 1.6 effective elasticity is carried as the adverse-but-defensible corner.
ELASTICITIES = {"D25 measured 0.8": 0.8, "D21/D22 0.44": 0.44, "rec6-class ~1.6": 1.6}


def pct(x: float, of: float) -> float:
    return 100.0 * x / of


def price(tokens: int, wave_ms: float, family_ms: float) -> None:
    seq_launches = GDN_LAYERS * tokens
    bodies = max(1, -(-tokens // MAX_WAVE_ROWS))
    loop_launches = GDN_LAYERS * bodies
    removed = seq_launches - loop_launches
    work_ms = family_ms - seq_launches * SUBMISSION_US / 1000.0
    slot_delta = (BILL_LOOP - BILL_SHIPPING) / BILL_SHIPPING

    print(f"\n=== {tokens} lanes (wave {wave_ms:.2f} ms, `gdn_rec` {family_ms:.2f} ms)")
    print(f"  launches  seq={seq_launches:>7,}  looped={loop_launches:>5,}  ({bodies} bodies x "
          f"{GDN_LAYERS} GDN layers, kMaxWaveRows={MAX_WAVE_ROWS})  removed={removed:,}"
          f"  ratio={seq_launches / loop_launches:.0f}x")
    print(f"  submission at the 1.72 us floor: seq={seq_launches * SUBMISSION_US / 1000:7.2f} ms"
          f"  looped={loop_launches * SUBMISSION_US / 1000:.2f} ms"
          f"  -> saving {removed * SUBMISSION_US / 1000:.2f} ms"
          f" = {pct(removed * SUBMISSION_US / 1000, wave_ms):.2f} % of wave")
    print(f"  the token-work part that REMAINS: {work_ms:.2f} ms"
          f" ({pct(work_ms, family_ms):.1f} % of the family; the rest is submission)")
    print(f"  the loop's own instruction cost: {BILL_SHIPPING} -> {BILL_LOOP} slots/token-step"
          f" = {slot_delta:+.1%} (census: walk shapes identical, 16x8 and 16x8, so all of it is")
    print(f"       per-token addressing + loop control, none of it state math)")
    rows = []
    for sub_name, sub_us in (("floor 1.72 us", SUBMISSION_US),
                             ("collapse-observed 0.72 us", COLLAPSE_US)):
        saving = removed * sub_us / 1000.0
        for el_name, el in ELASTICITIES.items():
            penalty = work_ms * slot_delta * el
            net = saving - penalty
            rows.append((sub_name, el_name, saving, penalty, net))
            print(f"    sub={sub_name:24} x elast={el_name:20}  saving {saving:7.2f} ms"
                  f"  - penalty {penalty:6.2f} ms  = NET {net:+7.2f} ms"
                  f" = {pct(net, wave_ms):+5.2f} % of wave"
                  f"  {'CLEARS' if net > MIN_PCT_OF_WAVE / 100.0 * wave_ms else 'REFUSED'}")
    best = max(r[4] for r in rows)
    worst = min(r[4] for r in rows)
    print(f"  RANGE {worst:+.2f} .. {best:+.2f} ms  =  {pct(worst, wave_ms):+.2f} % .."
          f" {pct(best, wave_ms):+.2f} % of the wave")
    bar = MIN_PCT_OF_WAVE / 100 * wave_ms
    tail = (f"worst corner is {worst:+.2f} ms = ADVERSE, see the note under the table"
            if worst <= 0 else f"worst corner clears by {worst / bar:.0f}x")
    print(f"  refuse bar MIN_PCT_OF_WAVE = {MIN_PCT_OF_WAVE} % = {bar:.2f} ms:"
          f" best corner clears by {best / bar:.0f}x; {tail}")


def main() -> int:
    print("D53 launch-batching price — arithmetic on recorded numbers, no device touched.")
    print(f"submission floor {SUBMISSION_US} us (§11); collapse-observed {COLLAPSE_US} us/launch removed (§4);")
    print(f"bill {BILL_SHIPPING} -> {BILL_LOOP} slots/token-step (isa_census on the built harness listing)")
    price(4096, WAVE_MS_4096, FAMILY_MS_4096)
    price(1024, WAVE_MS_1024, FAMILY_MS_1024)
    print("""
The ONLY corner that comes back adverse pairs `collapse-observed 0.72 us per launch removed` with the
`rec6-class ~1.6` elasticity: launches worth under half the measured floor AND the loop's addressing
costing nearly 2x its bill. Both halves are pessimistic at once, and the first is depressed by Route
A's own defects (4x redundant work per token, nine barriers per token per strip,
`prototype/gdn_collapse/results.md` §5), neither of which this form carries. Read the range as: most
likely +8 to +12 % of the wave, defensibly as low as +3 %, and the falsifier is one paired burst plus
one paired in-situ A/B.
""")
    print("""
WHAT THIS CENSUS CANNOT SEE, and the step that sees each (all device, all main-agent):
 1. whether the COMPILER kept the values identical. No static argument does. The gate is
    `--modes rec0,recloop` bit-identity on `o` AND final state, 4/4 cases, `rerun_ndiff=0`.
 2. register pressure from loop-carried state. Read at compile time, not guessed: rec2c4 59 VGPR /
    24 SGPR / 0 spill / occupancy 16 waves-per-SIMD, recloop 75 VGPR / 40 SGPR / 0 spill / occupancy
    STILL 16 waves-per-SIMD (`hipcc -Rpass-analysis=kernel-resource-usage`), and the launch puts only
    64 wavefronts on the device (16 blocks x 4), so the +16 registers do not bind at this grid. The
    device witness that would catch a wrong reading is the paired burst: a looped side that is slower
    per token-step has register pressure or addressing as its first suspects, not its launch count.
 3. LDS residency across a long loop. LDS is 1,280 B/block in BOTH forms (unchanged, and it is per
    launch, not per token), and the state stays in global memory exactly as it was, so nothing is being
    staged. If a later variant stages state in LDS to skip the global round trip, that is Route A's
    geometry (`engine/gdn_collapse_kernels.h`) and it re-opens the association question.
 4. whether 512 iterations of one launch changes scheduling, L2 residency, or the queue's pipelining.
    Only the in-situ paired A/B sees that; the harness burst at 128 tokens is a screen, and §10 is the
    precedent that a burst winner can be an in-situ loser.
 5. whether the loop's addressing cost is actually ~1x its bill or ~2x. `rec6` says spelling deltas on
    this kernel have returned about 2x their slot delta. The mitigation, if the burst reads thin, is a
    variant that increments the six row pointers instead of recomputing `t*CH` per iteration — integer
    only, so still bit-identical, and it goes through the same transcription + bit-identity gates.
""")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
