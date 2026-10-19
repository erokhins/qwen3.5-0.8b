#!/usr/bin/env python3
"""A6 Step 1, half (a): G1's Q un-stage, as a compile-only variant of the shipping header.

This is `prototype/attn_wmma/evidence/fattn_unstage_q.diff` re-anchored to the CURRENT `engine/attn_fattn_kernels.h`
(that patch no longer applies: `git apply --check` fails on `engine/attn_fattn_kernels.h:445` and
`engine/kernels_attn.cpp:170`, because the `--attn-fattn-roll` knob and A4-L1 landed in between and moved the
score roll behind `fattn_score_q_tile` / `fattn_score_k_tile` helpers). The SEMANTICS are E1's, unchanged:

  * the per-warp Q stage fill is gone; the score leg's SECOND operand is read from global at the use site by
    `fattn_load_q_tile`, which applies the same `bf16 -> f32 -> f16 RN` conversion `fattn_stage_row` applied on
    the way in, so the operand bits are identical (`prototype/attn_wmma/E1-result.md` §5, quoted);
  * the shipped clamp is kept and is still per lane: `qrs = min(qslot, qrmax - 1)`, which resolves to row
    `nrows - 1` when a warp's row-block is wholly past `nrows`, so no lane dereferences a row that does not
    exist;
  * the buffer survives, renamed `s_tr` and typed `float`, because the epilogue's F32 transpose needs it — which
    is WHY this half alone does not reduce the block's LDS: E1 measured it at **32,768 B, 184 VGPR, spill 0**
    (`prototype/attn_wmma/E1-result.md` §1, read). That reproduction is this half's canary;
    `a6_g1_epi8.py` is the half that moves the bytes.

`a6_build.sh` applies this to a COPY in a scratch directory. `engine/` is never written to.

    python3 prototype/attn_percost/a6_g1_unstage.py --out /tmp/a6g1/new/engine/attn_fattn_kernels.h

Refuses unless every anchor occurs exactly the expected number of times (a replacement that silently matched
nothing compiles the shipped form and reports it as the variant — the empty-parse class AGENTS.md records).
"""
import argparse
import pathlib


def span(text: str, start: str, end: str, new: str, what: str) -> str:
    """Replace the ONE span that starts at `start` and ends with `end` (both included)."""
    i = text.find(start)
    if i < 0 or text.find(start, i + 1) >= 0:
        raise SystemExit(f"REFUSE: anchor start for {what} is absent or ambiguous. The header moved; re-read it"
                         " instead of letting this compile the shipped form and call it the variant.")
    j = text.find(end, i)
    if j < 0:
        raise SystemExit(f"REFUSE: anchor end for {what} not found after its start.")
    return text[:i] + new + text[j + len(end):]


def one(text: str, old: str, new: str, what: str, n: int = 1) -> str:
    got = text.count(old)
    if got != n:
        raise SystemExit(f"REFUSE: the anchor for {what} occurred {got} times, expected {n}.")
    return text.replace(old, new)


def apply_variant(src: str) -> str:
    t = src

    # (0) THE FILE HEADER'S PLACEMENT NOTE (D) and the one-address-expression note, both of which state the
    # shipped Q stage as a precondition. A5-GEOMETRY.md §6: G1 "reverse[s] that decision in the open, with the
    # reversal written into the header, not left to a diff" — the same rule, applied to the variant.
    t = one(
        t,
        " *         s_q   3 x 16 rows x 256 dims x 2 B = 24576 B   (one 8 KiB stage per warp: Q is per-head and per-warp,\n"
        " *                                                         so it cannot be shared and cannot be halved)\n",
        " *         s_tr  3 x 16 rows x 128 dims x 4 B = 24576 B   (G1: Q is read from global, so this is the\n"
        " *                                                         epilogue's F32 transpose stage — the SAME\n"
        " *                                                         8 KiB per warp the Q stage occupied)\n",
        "placement (D)'s stage table")
    t = one(
        t,
        " *       The variant that keeps Bet 1's un-staged Q at three warps (V6, 8 KiB/block, 24 wf/CU on paper) is\n"
        " *       REFUTED on measurement: 73 spilled VGPRs, 284 B/lane of scratch.\n",
        " *       The variant that keeps Bet 1's un-staged Q at three warps (V6 ON THE SKELETON: 73 spilled VGPRs,\n"
        " *       284 B/lane of scratch) was re-measured on the real body by `prototype/attn_wmma/E1-result.md` §1\n"
        " *       and lands at 184 VGPR / 0 spill / 0 scratch. What the skeleton got wrong is the LDS: the per-warp\n"
        " *       8 KiB is the epilogue's transpose staging whether or not Q is parked in it, so un-staging Q buys\n"
        " *       a register question and no residency. G1 (A6) is the half that moves the bytes: the epilogue\n"
        " *       staged FATTN_TRG rows at a time.\n",
        "placement (D)'s V6 refutation")
    t = one(
        t,
        " * `fattn_v_addr` are the only way the two stages are indexed (Q has no stage: `fattn_q_frag` is the only way\n"
        " * its operand tiles are produced), and the partial slot expression is the shipped one, verbatim, keyed by\n",
        " * `fattn_v_addr` are the only way the two stages are indexed (Q has no stage in this form: `fattn_q_row`\n"
        " * is the one spelling of its global source row and `fattn_load_q_tile` the only way its operand tiles are\n"
        " * produced), and the partial slot expression is the shipped one, verbatim, keyed by\n",
        "the one-address-expression note")

    # (1) THE ONE SPELLING OF Q'S SOURCE ROW. `fattn_q_addr` indexed a stage; there is no stage, so what needs
    # one spelling is the global row (§20.1: one address expression per staged object, moved with the object).
    t = one(
        t,
        "__device__ __forceinline__ int fattn_q_addr(int q, int d) { return q * DH + d; }      // [row][dim]\n",
        "// G1 (A6): Q IS NOT STAGED, so what needs the one spelling is its GLOBAL source row — the expression\n"
        "// the deleted fill used, now the only way any lane reaches Q. `qrow` is an absolute wave row index and\n"
        "// the clamp that keeps it inside the wave lives at its single call site in the body.\n"
        "__device__ __forceinline__ const hip_bfloat16* fattn_q_row(const CoreArgs& a, int qrow, int head0) {\n"
        "    return a.qrope + (size_t)ROW_H * qrow + (size_t)head0 * DH;                    // [row][head][dim]\n"
        "}\n",
        "fattn_q_addr -> fattn_q_row")

    # (2) THE Q OPERAND TILE, from global. Defined next to its only user, above the roll helpers that call it.
    t = one(
        t,
        "__device__ __forceinline__ fattn_ARegs fattn_score_q_tile(const __half* s_q, int w, int lane, int t) {\n"
        "    return fattn_load_tile(&s_q[w * FATTN_MROWS * DH + fattn_q_addr(fattn_ab_row(lane), 16 * t)]);\n"
        "}\n",
        "// THE Q OPERAND, read from global at the use site (G1, E1's port). The same 16 halves\n"
        "// `fattn_load_tile` returns out of a stage, built from the two uint4s of the lane's row: the\n"
        "// conversion is exactly the `__float2half_rn(fattn_bf(x))` that `fattn_stage_row` applied on the way\n"
        "// in, so the operand VALUES are bit-identical and what moves is where the bytes come from — 8 KiB per\n"
        "// warp per LAUNCH becomes 8 KiB per warp per ROUND, which is a traffic question for the device, not a\n"
        "// bits question.\n"
        "__device__ __forceinline__ fattn_ARegs fattn_load_q_tile(const hip_bfloat16* p) {\n"
        "    const uint4 lo = *reinterpret_cast<const uint4*>(p);\n"
        "    const uint4 hi = *reinterpret_cast<const uint4*>(p + 8);\n"
        "    const hip_bfloat16* s0 = reinterpret_cast<const hip_bfloat16*>(&lo);\n"
        "    const hip_bfloat16* s1 = reinterpret_cast<const hip_bfloat16*>(&hi);\n"
        "    fattn_ARegs r;\n"
        "#pragma unroll\n"
        "    for (int e = 0; e < 8; ++e) r[e] = (short)__half_as_ushort(__float2half_rn(fattn_bf(s0[e])));\n"
        "#pragma unroll\n"
        "    for (int e = 0; e < 8; ++e) r[8 + e] = (short)__half_as_ushort(__float2half_rn(fattn_bf(s1[e])));\n"
        "    return r;\n"
        "}\n"
        "// The roll's Q tile, one spelling, now over the lane's global row instead of the stage slot. Both\n"
        "// roll spellings call this, so neither writes an address (§20.1).\n"
        "__device__ __forceinline__ fattn_ARegs fattn_score_q_tile(const hip_bfloat16* qrow, int t) {\n"
        "    return fattn_load_q_tile(qrow + 16 * t);\n"
        "}\n",
        "fattn_score_q_tile (stage -> global)")

    # (3) The stale note that says there is deliberately no global-Q loader. It is the decision G1 reverses,
    # in the open, per A5-GEOMETRY.md §6 ("G1/G2 reverse that decision in the open").
    t = span(
        t,
        "// THE Q OPERAND, staged: one 32-byte tile out of the warp's own Q stage",
        "// the layout (file header, placement (B)).\n",
        "// THERE IS NOW A GLOBAL-Q LOADER (`fattn_load_q_tile`, above the roll helpers). The file used to say\n"
        "// the Q stage was a precondition, and on that reasoning it was right at the time: the one-warp form\n"
        "// spilled (Bet 1, 50 spilled VGPRs) and the skeleton's three-warp un-stage spilled harder (V6, 73).\n"
        "// `prototype/attn_wmma/E1-result.md` §1 re-measured the un-stage on the REAL body at 184 VGPR and\n"
        "// ZERO spill, so the precondition was a register myth; what it does not buy — and this is the other\n"
        "// half of E1's finding — is any LDS, because the epilogue stands on the same per-warp 8 KiB.\n",
        "the stale no-global-Q-loader note")

    # (4) THE BUFFER: the Q stage becomes the epilogue's transpose stage. Bytes are the same at this step.
    t = span(
        t,
        "    // ONE Q STAGE PER WARP.",
        "  // [warp][row][dim], 8 KiB each\n",
        "    // G1 (A6): NO Q STAGE. The score leg's SECOND operand comes from global per round\n"
        "    // (`fattn_q_row` + `fattn_load_q_tile`), so this buffer is now what the epilogue needs it for: the F32\n"
        "    // output transpose staging, one stage per warp. Keeping it PER WARP is still load-bearing — one\n"
        "    // warp's half-DH staging is the whole K/V stage, so three warps transposing through `s_kv` would\n"
        "    // write one another's rows (§20.1's aliasing bug). Same bytes as the Q stage at this step; shrinking\n"
        "    // it to 8 rows per pass is `a6_g1_epi8.py`, and that is what moves the block's LDS.\n"
        "    __shared__ __align__(16) float s_tr[FATTN_WARPS * FATTN_MROWS * FATTN_TROW];"
        "  // [warp][row][dim/2], 8 KiB each\n",
        "the s_q declaration")

    # (5) THE FILL GOES; THE CLAMP STAYS, per lane.
    t = span(
        t,
        "    // ---- Q, staged once per launch into this warp's own 8 KiB",
        "                        a.qrope + (size_t)ROW_H * (rb * FATTN_MROWS + qrs) + (size_t)head0 * DH, lane);\n"
        "    }\n",
        "    // ---- Q, NOT staged (G1). Each lane reads the row its operand tile carries from global every\n"
        "    // round; the values are bit-identical to the staged form because `fattn_load_q_tile` applies the\n"
        "    // conversion `fattn_stage_row` used on the way in. THE CLAMP IS THE SHIPPED ONE and it is still not\n"
        "    // cosmetic: at 48 rows per BLOCK a whole warp's row-block can be PAST `nrows` (`qrmax <= 0`), and\n"
        "    // `rb * FATTN_MROWS + min(slot, qrmax - 1)` is exactly `nrows - 1` there, so a clamped read lands on\n"
        "    // the last real row instead of before the buffer. The staged form `break`ed out of its fill; this\n"
        "    // form reads a valid row whose cells are masked out below (`live`), so no lane ever dereferences a\n"
        "    // row that does not exist.\n"
        "    const int qrmax = nrows_of(a) - rb * FATTN_MROWS;   // row-slots of this warp that exist; may be <= 0\n"
        "    const int qslot = fattn_ab_row(lane);               // the row-slot THIS lane's operand tile carries\n"
        "    const int qrs = qslot < qrmax ? qslot : qrmax - 1;  // the shipped clamp, per lane\n"
        "    const hip_bfloat16* qrow = fattn_q_row(a, rb * FATTN_MROWS + qrs, head0);\n",
        "the Q stage fill")

    # (6) The three call sites of the roll helper.
    t = one(t, "fattn_score_q_tile(s_q, w, lane, t)", "fattn_score_q_tile(qrow, t)",
            "the roll's Q tile call", n=2)
    t = one(t, "fattn_score_q_tile(s_q, w, lane, t + 1)", "fattn_score_q_tile(qrow, t + 1)",
            "the two-accumulator roll's second Q tile call")

    # (7) The epilogue now takes the float stage it always wanted, and the buffer it stands on is named `s_tr`.
    t = one(
        t,
        "__device__ __forceinline__ void fattn_transpose_out(__half* stage_as_f32, const FattnO& out,\n"
        "                                                    const CoreArgs& a, int rt, int head0,\n"
        "                                                    unsigned row_mask, int lane) {\n"
        "    float* stage = reinterpret_cast<float*>(stage_as_f32);\n",
        "__device__ __forceinline__ void fattn_transpose_out(float* stage, const FattnO& out,\n"
        "                                                    const CoreArgs& a, int rt, int head0,\n"
        "                                                    unsigned row_mask, int lane) {\n",
        "fattn_transpose_out's stage parameter")
    t = one(t,
            "fattn_transpose_out(s_q + w * FATTN_MROWS * DH, out, a, rb, head0, row_mask, lane);",
            "fattn_transpose_out(s_tr + w * FATTN_MROWS * FATTN_TROW, out, a, rb, head0, row_mask, lane);",
            "the transpose call site")

    # (8) THE SIZING ASSERT. This is the assert E1 named: the epilogue needs the buffer either way, so the
    # block's total is unchanged at this step and only the spelling of its largest term moves.
    t = one(
        t,
        "static_assert(FATTN_WARPS * FATTN_MROWS * DH * 2 + FATTN_MLANES * DH * 2 == 32768,",
        "static_assert(FATTN_WARPS * FATTN_MROWS * FATTN_TROW * 4 + FATTN_MLANES * DH * 2 == 32768,",
        "the block LDS sum assert")
    # The leftover check: no CODE line may still name the deleted stage. Comment lines may, and do — the file
    # header records placements that existed, and rewriting history to make a grep clean would be worse.
    leftovers = [ln for ln in t.splitlines()
                 if "s_q" in ln and not ln.strip().startswith(("//", "*", "/*"))]
    if leftovers:
        raise SystemExit("REFUSE: `s_q` survives the un-stage patch in code:\n  " + "\n  ".join(leftovers[:6]))
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=pathlib.Path)
    ap.add_argument("--src", default=None, type=pathlib.Path,
                    help="header to patch (default: the tree's engine/attn_fattn_kernels.h)")
    a = ap.parse_args()
    repo = pathlib.Path(__file__).resolve().parents[2]
    src = a.src or (repo / "engine" / "attn_fattn_kernels.h")
    text = src.read_text()
    out = apply_variant(text)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out)
    print(f"wrote {a.out}  ({len(out) - len(text):+d} B against {src})")


if __name__ == "__main__":
    main()
