#!/usr/bin/env python3
"""A6 Step 1, half (b): the epilogue's F32 transpose stages 8 ROWS at a time instead of 16 — 4 passes, 20,480 B.

This is the route `prototype/attn_wmma/E1-result.md` §4 names first, quoted: *"stage the epilogue 8 rows at a
time instead of 16: `3 x 4,096 + 8,192 = 20,480 B` -> 3 blocks/CU = 9 wf/CU, at the cost of 4 transpose passes
instead of 2 (same barriers x2, half the bytes)"*. It is applied AFTER `a6_g1_unstage.py` — the two are one
candidate (G1), and they are kept as two patchers so a compile can attribute the LDS move: E1 read 32,768 B for
the un-stage alone, so if the pair does not reach 20,480 the failing half is named rather than inferred.

What is owed bit-identity (A6-BET.md "Numerics", G1): same 16 source bytes per tile, same
`bf16 -> f32 -> f16 RN` conversion, same clamp, same order, and this half moves **staging addresses in LDS
only** — each published float still travels from the same `out.t[t][l]` to the same `part` slot, and the
half-of-DH order is preserved (passes are half-major). Extra passes are extra barriers, nothing else.

    python3 prototype/attn_percost/a6_g1_epi8.py --in /tmp/a6/new/engine/attn_fattn_kernels.h --out SAME
"""
import argparse
import pathlib

import a6_g1_unstage as A6U     # reuse the same refusal-style helpers, one spelling of them


def apply_variant(src: str) -> str:
    t = src

    # (0) The file header's placement (D) table, which the un-stage patch relabelled `s_tr` at 16 rows.
    t = A6U.one(
        t,
        " *         s_tr  3 x 16 rows x 128 dims x 4 B = 24576 B   (G1: Q is read from global, so this is the\n"
        " *                                                         epilogue's F32 transpose stage — the SAME\n"
        " *                                                         8 KiB per warp the Q stage occupied)\n"
        " *         s_kv  16 lanes x 256 dims x 2 B     =  8192 B   (ONE stage, amortised over the three row-blocks)\n"
        " *         total                               = 32768 B  -> 2 blocks/CU on this part's 64 KiB = 6 wf/CU\n",
        " *         s_tr  3 x 8 rows x 128 dims x 4 B   = 12288 B   (G1: Q from global, and the epilogue stages\n"
        " *                                                         FATTN_TRG = 8 rows per pass instead of 16, so\n"
        " *                                                         one warp's stage is 4 KiB, not 8)\n"
        " *         s_kv  16 lanes x 256 dims x 2 B     =  8192 B   (ONE stage, amortised over the three row-blocks)\n"
        " *         total                               = 20480 B  -> 3 blocks/CU on this part's 64 KiB = 9 wf/CU\n",
        "placement (D)'s stage table, narrowed")

    # (1a) THE PASS GEOMETRY, defined where `FATTN_TROW` is, because the block LDS sum assert above the publish
    # block already needs it (a define that lives below the assert that uses it is not a define).
    t = A6U.one(
        t,
        "#define FATTN_TROW (DH / 2)                       // floats per query row of the transpose stage\n",
        "#define FATTN_TROW (DH / 2)                       // floats per query row of the transpose stage\n"
        "// G1 (A6): the transpose stages FATTN_TRG rows at a time instead of the warp's whole 16, so the stage is\n"
        "// half the shipped depth and the epilogue runs FATTN_NPASS passes (4) instead of 2. `FATTN_LBLK` is the\n"
        "// 8-dim blocks one lane publishes per PASS — four lanes per staged row where the shipped stage used two\n"
        "// — and it needs `FATTN_TBLK`, defined a few lines below, which is fine because a macro body is expanded\n"
        "// where it is USED, and its first use is the coverage assert there.\n"
        "#define FATTN_TRG  8                       // G1: query rows per transpose pass (was all 16)\n"
        "#define FATTN_NGRP (FATTN_MROWS / FATTN_TRG)        // row groups per half of DH: 2\n"
        "#define FATTN_NPASS (2 * FATTN_NGRP)                // 4 passes instead of 2\n"
        "#define FATTN_LBLK ((FATTN_TRG * (FATTN_TROW / FATTN_TBLK)) / 32)   // blocks per lane per pass: 4\n",
        "the pass geometry")

    # (1b) THE PUBLISH MAP and the coverage assert, in place.
    t = A6U.span(
        t,
        "// HOW THE PARTIAL IS PUBLISHED:",
        "would have three warps publishing into one stage, which is §20.1's aliasing bug wearing a hat\");\n",
        "// HOW THE PARTIAL IS PUBLISHED: `part` wants the row's DH dims in blocks of FATTN_TBLK, the transpose\n"
        "// stages half of DH at a time, and FATTN_PBLK is the blocks one lane publishes per half in the SHIPPED\n"
        "// 16-row stage. G1 shrinks the stage to FATTN_TRG rows, so a pass carries FATTN_TRG rows and every row\n"
        "// of a pass is published by FOUR lanes instead of two: FATTN_LBLK (defined above, beside FATTN_TRG) is\n"
        "// the blocks one lane publishes per PASS. The assert below is the same COVERAGE claim the shipped file\n"
        "// makes — every dim of every staged row written exactly once per half — re-derived for the narrower\n"
        "// stage rather than asserted beside it, because the defect it retires is a publish loop that covers 16\n"
        "// of a row's 256 dims while satisfying every bound-style assert in the file.\n"
        "#define FATTN_TBLK 8\n"
        "#define FATTN_PBLK 8                       // blocks per lane per half, the shipped 16-row stage\n"
        "static_assert(FATTN_MROWS % FATTN_TRG == 0, \"the row groups must tile the warp's 16 row-slots\");\n"
        "static_assert(FATTN_TRG * FATTN_TROW == 32 * FATTN_LBLK * FATTN_TBLK,\n"
        "              \"one warp publishes its OWN row-block's stage, every float exactly once per half: rows x \"\n"
        "              \"dims-per-half must equal 32 lanes x blocks x block-width. The 32 is a warp and not \"\n"
        "              \"FATTN_THREADS, because the stage and the publishing set are both per warp; making it the \"\n"
        "              \"block would have three warps publishing into one stage, which is §20.1's aliasing bug \"\n"
        "              \"wearing a hat\");\n"
        "// The transpose stage's ONE address expression (§20.1), [row within this pass][dim]: writer and reader \"\n"
        "// both go through it, so a narrower stage cannot be addressed two ways by the two halves of the\n"
        "// epilogue. Rows are renumbered per pass, which is the whole content of the change.\n"
        "__device__ __forceinline__ int fattn_tr_addr(int row_in_pass, int d) { return row_in_pass * FATTN_TROW + d; }\n",
        "the publish/coverage block")

    # (2) The two asserts that spell the stage's size and its widest write.
    t = A6U.span(
        t,
        "static_assert(FATTN_MROWS * FATTN_TROW * 4 == FATTN_MROWS * DH * 2,",
        "once, which is what keeps the epilogue from serialising behind one shared stage\");\n",
        "// One staged row is 128 F32 = 256 F16 either way, and the stage is FATTN_TRG rows deep per warp: that\n"
        "// is the byte the residency arithmetic turns on. The shipped stage is 16 rows = 8,192 B per warp, which\n"
        "// is exactly what the Q stage cost, and it is WHY un-staging Q alone bought nothing\n"
        "// (`prototype/attn_wmma/E1-result.md` §4). Three warps still transpose at once, per warp.\n"
        "static_assert(FATTN_TRG * FATTN_TROW * 4 == FATTN_TRG * DH * 2,\n"
        "              \"one warp's transpose staging for one pass: FATTN_TRG rows x 128 F32 = 4,096 B\");\n",
        "the per-warp staging size assert")
    t = A6U.one(
        t,
        "static_assert((FATTN_MROWS - 1) * FATTN_TROW + (16 * (FATTN_VTILES / 2 - 1) + 15) + 1\n"
        "                  == FATTN_MROWS * FATTN_TROW,\n"
        "              \"the widest transpose stage write is the last float of the staging area\");",
        "static_assert((FATTN_TRG - 1) * FATTN_TROW + (16 * (FATTN_VTILES / 2 - 1) + 15) + 1\n"
        "                  == FATTN_TRG * FATTN_TROW,\n"
        "              \"the widest transpose stage write is the last float of the (narrower) staging area\");",
        "the widest-stage-write assert")
    t = A6U.one(
        t,
        "static_assert(FATTN_WARPS * FATTN_MROWS * FATTN_TROW * 4 + FATTN_MLANES * DH * 2 == 32768,",
        "static_assert(FATTN_WARPS * FATTN_TRG * FATTN_TROW * 4 + FATTN_MLANES * DH * 2 == 20480,",
        "the block LDS sum assert (20,480)")
    t = A6U.span(
        t,
        "              \"the block declares FATTN_WARPS Q stages and ONE aliased K/V stage: 3 x 8192 + 8192 = 32 KiB, \"",
        "the declarations, not from here.\");",
        "              \"the block declares FATTN_WARPS transpose stages of FATTN_TRG rows and ONE aliased K/V \"\n"
        "              \"stage: 3 x 4096 + 8192 = 20,480 B, which on this part's 64 KiB of LDS (KFD) is the \"\n"
        "              \"3-blocks/CU boundary, i.e. 9 wavefronts/CU — the number G1 exists to buy. The census \"\n"
        "              \"reads this off the declarations, not from here.\");",
        "the block LDS sum assert's message")

    # (3) The declaration shrinks with the stage.
    t = A6U.one(
        t,
        "    __shared__ __align__(16) float s_tr[FATTN_WARPS * FATTN_MROWS * FATTN_TROW];"
        "  // [warp][row][dim/2], 8 KiB each\n",
        "    __shared__ __align__(16) float s_tr[FATTN_WARPS * FATTN_TRG * FATTN_TROW];"
        "  // [warp][row in pass][dim/2], 4 KiB each\n",
        "the s_tr declaration")
    t = A6U.one(
        t,
        "    // warp's half-DH staging is the whole K/V stage, so three warps transposing through `s_kv` would\n"
        "    // write one another's rows (§20.1's aliasing bug). Same bytes as the Q stage at this step; shrinking\n"
        "    // it to 8 rows per pass is `a6_g1_epi8.py`, and that is what moves the block's LDS.\n",
        "    // warp's stage is still bigger than a third of the K/V stage, so three warps transposing through\n"
        "    // `s_kv` would write one another's rows (§20.1's aliasing bug). G1 shrinks it to FATTN_TRG rows per\n"
        "    // pass — 3 x 4,096 + 8,192 = 20,480 B for the block, i.e. 3 blocks/CU = 9 wf/CU on the KFD number.\n",
        "the s_tr comment")

    # (4) The header sentence that says the staging is 16 rows deep.
    t = A6U.span(
        t,
        "// in F32 through the CALLER's dead Q stage, half of DH at a time: 16 rows x (DH/2) floats = 8192 B = "
        "exactly",
        "// staging fills all of it, so three warps transposing through it would write one another's rows.\n",
        "// in F32 through the CALLER's own per-warp stage, half of DH at a time and G1 rows at a time: "
        "FATTN_TRG\n"
        "// rows x (DH/2) floats = 4,096 B per pass. It is PER WARP on purpose: one warp's staging is larger "
        "than a\n"
        "// third of the shared K/V stage, so three warps transposing through it would write one another's "
        "rows.\n",
        "the epilogue header's staging sentence")

    # (5) THE BODY: 4 passes, each staging FATTN_TRG rows and publishing them.
    t = A6U.span(
        t,
        "    const int srow = lane >> 1;",
        "    __syncthreads();                         // the next launch's stage write must not race the last "
        "read\n",
        "    // G1: FATTN_NPASS passes instead of two. `half` still picks which half of DH this pass stages —\n"
        "// half-major, so a float's destination order is the shipped one — and `rg` picks which group of\n"
        "// FATTN_TRG row-slots it stages, so the stage is half the shipped depth and the published VALUES are\n"
        "// untouched: same source register, same `part` slot, same (m,l) publish above.\n"
        "#pragma unroll\n"
        "    for (int pass = 0; pass < FATTN_NPASS; ++pass) {\n"
        "        const int half = pass / FATTN_NGRP;\n"
        "        const int rg = pass % FATTN_NGRP;\n"
        "        const int row0 = rg * FATTN_TRG;                 // this pass's first row-slot in the warp\n"
        "        __syncthreads();          // the previous pass's reads (and the last P*V read of V) are done\n"
        "        // STAGE. `fattn_c_row(lane)` is the row-slot this lane ACCUMULATED (held by the pair t, t^16),\n"
        "        // so half the warp's lanes hold rows this pass does not stage and write nothing. That is a\n"
        "        // guarded store, not a divergent barrier — the barriers are above and below it, taken by all.\n"
        "        if (fattn_c_row(lane) >= row0 && fattn_c_row(lane) < row0 + FATTN_TRG) {\n"
        "#pragma unroll\n"
        "            for (int t = half * (FATTN_VTILES / 2); t < (half + 1) * (FATTN_VTILES / 2); ++t) {\n"
        "#pragma unroll\n"
        "                for (int l = 0; l < 8; ++l) {\n"
        "                    const int d = 16 * t + fattn_c_col(lane, l) - half * FATTN_TROW;  // [0, 128)\n"
        "                    stage[fattn_tr_addr(fattn_c_row(lane) - row0, d)] = out.t[t][l];\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "        __syncthreads();\n"
        "        // PUBLISH: FATTN_LBLK blocks of FATTN_TBLK per lane, four lanes per staged row where the\n"
        "        // shipped 16-row stage used two — the (row, block) pairs come out exactly once per half either\n"
        "        // way, which is what the FATTN_TRG coverage assert above is for.\n"
        "        const int srow = row0 + (lane >> 2);             // the row-slot THIS lane publishes\n"
        "        const int sblk = (lane & 3) * FATTN_LBLK;        // its first block within that row's half\n"
        "        if (((row_mask >> srow) & 1u) != 0u) {  // no barrier below here inside the loop\n"
        "            float* pp = fattn_part_slot(a, rt * FATTN_MROWS + srow, head0);\n"
        "#pragma unroll\n"
        "            for (int k = 0; k < FATTN_LBLK; ++k) {\n"
        "                const int sbase = (sblk + k) * FATTN_TBLK;   // [0, FATTN_TROW) by the coverage assert\n"
        "                const int dbase = half * FATTN_TROW + sbase; // the dim this block carries, in [0, DH)\n"
        "#pragma unroll\n"
        "                for (int u = 0; u < FATTN_TBLK; ++u) {\n"
        "                    pp[2 + dbase + u] = stage[fattn_tr_addr(lane >> 2, sbase + u)];\n"
        "                }\n"
        "            }\n"
        "        }\n"
        "    }\n"
        "    __syncthreads();                         // the next launch's stage write must not race the last "
        "read\n",
        "the transpose body")
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=pathlib.Path,
                    help="the header to patch — the ALREADY-un-staged copy, not the tree's")
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    text = a.src.read_text()
    if "fattn_load_q_tile" not in text:
        raise SystemExit("REFUSE: the input is not the un-staged variant (no fattn_load_q_tile). The 8-row "
                         "epilogue is the second half of ONE candidate; applying it to the shipped header would "
                         "shrink the buffer the Q stage still lives in.")
    out = apply_variant(text)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out)
    print(f"wrote {a.out}  ({len(out) - len(text):+d} B against {a.src})")


if __name__ == "__main__":
    main()
