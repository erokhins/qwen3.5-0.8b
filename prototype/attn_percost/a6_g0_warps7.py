#!/usr/bin/env python3
"""A6 Step 2: G0 — the 7-warp falsifier, as a compile-only variant. `A5-GEOMETRY.md` §3.1, read.

One geometry fact, moved in the TWO headers that both own it (AGENTS' "a geometry fact that lives in two
headers can be committed in one of them" trap, and §34's non-building tip):

  * `engine/attn_fattn_kernels.h`:   `FATTN_WARPS 3 -> 7`, `FATTN_THREADS 96 -> 224`, and the block LDS sum;
  * `prototype/attn_decode/attn_decode_kernels.h`: `FATTN_WARPS_PER_BLOCK 3 -> 7`, which is what
    `fattn_rtiles_of` counts blocks of, so grid.z moves 11 -> 5 at 512 rows as a consequence, not as a second
    decision. The kernel's own `static_assert(FATTN_ROWS_PER_BLOCK == FATTN_WARPS * FATTN_MROWS)` is the
    compile-time guard that the two moved together, and `a6_build.sh` compiles the patched pair, so the guard
    is actually exercised rather than trusted.

Everything else is shipped: M stays 16 rows per warp, Q stays staged, the epilogue stays 16 rows deep, so the
block's LDS is `7 x 8,192 + 8,192 = 65,536 B` = the whole CU budget, i.e. 1 block/CU -> **7 wf/CU** on the KFD
number. G0 is a FALSIFIER, not a bet: A6-BET's rule is that if a 6 -> 7 step moves the `attn` family not at
all, the residency thesis is dead and G1 is not built.

    python3 prototype/attn_percost/a6_g0_warps7.py --out-header DIR/engine/attn_fattn_kernels.h \
            --out-decode DIR/prototype/attn_decode/attn_decode_kernels.h
"""
import argparse
import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import a6_g1_unstage as A6U     # one spelling of the refusal-style helpers


def patch_header(text: str) -> str:
    t = A6U.one(text, "#define FATTN_WARPS 3\n#define FATTN_THREADS 96\n",
                "#define FATTN_WARPS 7      // G0 (A6): the Q-staged family's ceiling, 7 wf/CU at NW=7\n"
                "#define FATTN_THREADS 224\n",
                "FATTN_WARPS / FATTN_THREADS")
    t = A6U.one(
        t,
        "static_assert(FATTN_WARPS * FATTN_MROWS * DH * 2 + FATTN_MLANES * DH * 2 == 32768,",
        "static_assert(FATTN_WARPS * FATTN_MROWS * DH * 2 + FATTN_MLANES * DH * 2 == 65536,",
        "the block LDS sum assert")
    t = A6U.span(
        t,
        "              \"the block declares FATTN_WARPS Q stages and ONE aliased K/V stage: 3 x 8192 + 8192 = 32 KiB, \"",
        "the declarations, not from here.\");",
        "              \"the block declares FATTN_WARPS Q stages and ONE aliased K/V stage: 7 x 8192 + 8192 = \"\n"
        "              \"65,536 B, i.e. the ENTIRE CU LDS budget on this part (KFD lds_size_in_kb 64), so 1 \"\n"
        "              \"block/CU and 7 wavefronts/CU — the ceiling of the Q-staged family and the whole point of \"\n"
        "              \"the falsifier. The census reads this off the declarations, not from here.\");",
        "the block LDS sum assert's message")
    return t


def patch_decode(text: str) -> str:
    return A6U.one(text, "#define FATTN_WARPS_PER_BLOCK 3\n", "#define FATTN_WARPS_PER_BLOCK 7   // G0: one fact "
                                                            "with FATTN_WARPS, asserted in the kernel header\n",
                   "FATTN_WARPS_PER_BLOCK")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-header", required=True, type=pathlib.Path)
    ap.add_argument("--out-decode", required=True, type=pathlib.Path)
    ap.add_argument("--src-header", default=None, type=pathlib.Path)
    ap.add_argument("--src-decode", default=None, type=pathlib.Path)
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    sh = a.src_header or (repo / "engine" / "attn_fattn_kernels.h")
    sd = a.src_decode or (repo / "prototype" / "attn_decode" / "attn_decode_kernels.h")
    for path, out, fn in ((sh, a.out_header, patch_header), (sd, a.out_decode, patch_decode)):
        text = path.read_text()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(fn(text))
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
