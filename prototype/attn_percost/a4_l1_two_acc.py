#!/usr/bin/env python3
"""A4-L1 Gate 1: the two-accumulator spelling of the shipped score roll, as a compile-only variant.

This is the L1 candidate of `prototype/attn_percost/A4-BET.md` §"The claim under test", defined ONCE here as a
text replacement against `engine/attn_fattn_kernels.h` so the experiment's delta is reviewable instead of being
buried in a 754-line copy of a shipping header. **Nothing here is a shipping-path edit and nothing here is a
runtime knob**: `a4_l1_build.sh` copies the engine TU and this header into a scratch directory, applies the
replacement to the copy, and compiles the copy. `engine/` is never written to.

Why two accumulators at all: Gate 0 (§1 of `A4-L1-gates.md`) read the shipped roll as ONE `v_wmma` site in a
16-trip rolled loop whose destination tuple is also its accumulator operand — a 16-long loop-carried recurrence
on a single 8-VGPR tuple with no independent work between trips. L1 gives the roll a second accumulator so two
chains progress at once, at the price of one more live accumulator and a re-associated sum.

    python3 prototype/attn_percost/a4_l1_two_acc.py --out /tmp/a4l1v/engine/attn_fattn_kernels.h

Refuses unless the anchor text occurs exactly once (a replacement that silently matched nothing would compile the
shipped form and report it as the variant — the empty-parse failure class in AGENTS.md).
"""
import argparse
import pathlib

OLD = """        fattn_CRegs acc{};
#pragma unroll 1
        for (int t = 0; t < FATTN_KTILES; ++t) {
            const fattn_ARegs first = fattn_load_tile(&s_kv[fattn_kv_addr(fattn_ab_row(lane), 16 * t)]);
            const fattn_ARegs second =
                fattn_load_tile(&s_q[w * FATTN_MROWS * DH + fattn_q_addr(fattn_ab_row(lane), 16 * t)]);
            acc = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(first, second, acc);
        }
"""

NEW = """        fattn_CRegs acc0{}, acc1{};
        static_assert(FATTN_KTILES % 2 == 0, "the two-accumulator roll pairs k-tiles");
#pragma unroll 1
        for (int t = 0; t < FATTN_KTILES; t += 2) {
            // BOTH tiles are fetched before EITHER matrix op issues, so one LDS round-trip and two independent
            // WMATAs share it: that is the whole lever, and it is why the loads are spelled ahead of the ops
            // rather than adjacent to them as the shipped roll spells them.
            const fattn_ARegs k0 = fattn_load_tile(&s_kv[fattn_kv_addr(fattn_ab_row(lane), 16 * t)]);
            const fattn_ARegs q0 =
                fattn_load_tile(&s_q[w * FATTN_MROWS * DH + fattn_q_addr(fattn_ab_row(lane), 16 * t)]);
            const fattn_ARegs k1 = fattn_load_tile(&s_kv[fattn_kv_addr(fattn_ab_row(lane), 16 * (t + 1))]);
            const fattn_ARegs q1 =
                fattn_load_tile(&s_q[w * FATTN_MROWS * DH + fattn_q_addr(fattn_ab_row(lane), 16 * (t + 1))]);
            acc0 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k0, q0, acc0);
            acc1 = __builtin_amdgcn_wmma_f32_16x16x16_f16_w32(k1, q1, acc1);
        }
        // ONE add after the roll. This re-associates the 16-tile F32 dot — `acc0` sums the even tiles, `acc1`
        // the odd — so `o` moves and bit-identity with the shipped form is gone by construction; A4-BET admits
        // the existing 10.1 cos/meanrel bands as the judge instead.
        fattn_CRegs acc{};
#pragma unroll
        for (int i = 0; i < 8; ++i) acc[i] = acc0[i] + acc1[i];
"""


def apply_variant(src: str) -> str:
    n = src.count(OLD)
    if n != 1:
        raise SystemExit(f"REFUSE: the shipped score-roll anchor occurred {n} times, expected exactly 1. "
                         "The header moved; re-read the source instead of letting this compile the shipped form.")
    return src.replace(OLD, NEW)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--src", default=None, type=pathlib.Path,
                    help="header to patch (default: the tree's engine/attn_fattn_kernels.h)")
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    src = a.src or (repo / "engine/attn_fattn_kernels.h")
    text = src.read_text()
    out = apply_variant(text)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out)
    print(f"wrote {a.out}  ({len(out) - len(text):+d} B against {src})")


if __name__ == "__main__":
    main()
