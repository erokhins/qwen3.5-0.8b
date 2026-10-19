#!/usr/bin/env python3
"""A6 Step 3: insert the `__launch_bounds__(96, N)` ladder into a QUARANTINED copy of the fattn header.

`prototype/attn_percost/A6-BET.md`'s G0-M0b milestone names this experiment and `prototype/attn_percost/A3-result.md`
§4 named it first and nobody ran it: *"the rung at which the driver's answer stops rising IS the register-file
answer"*. The part reports no register-file field (`A3-result.md` §2b, `prototype/attn_wmma/fattn_regfile_ladder.md`
§3), so the file size can only be read out of the driver's own `hipOccupancyMaxActiveBlocksPerMultiprocessor`
behaviour — and that question has to be asked of a real body, not of a stub.

The rungs are separate `__global__` kernels because `__launch_bounds__` is per-kernel, and they are inserted
INSIDE `namespace engine::kernels` (just outside the anonymous namespace, beside `attn_fattn_wave` itself, which
is why `attn_fattn_body` is reachable from them by unqualified lookup). Nothing here touches `engine/`: the input
is a copy, and `a6_rf_ladder.sh` compiles the copy together with `a6_rf_probe.inc`.

    python3 prototype/attn_percost/a6_rf_ladder.py --in /tmp/a6rf/g1/engine/attn_fattn_kernels.h --out SAME

Two instruments are read out of the resulting object, and §3 of `A6-result.md` pairs them:
  * each rung's `vgpr_count` / `vgpr_spill_count` (compile side — the cost of the squeeze), and
  * the driver's max blocks/CU per rung, plus the LDS-only stubs in `a6_rf_probe.inc` that pin the driver's LDS
    term. The register-file answer is the DIFFERENCE between a stub's blocks and a real kernel's blocks at the
    same LDS figure — a rung plateau alone would be partly self-fulfilling, since a minBlocks hint tells the
    compiler to fit and the driver then reports the fit.
"""
import argparse
import pathlib

RUNGS = 8


def apply_variant(src: str) -> str:
    tail = "}  // namespace engine::kernels\n"
    if src.count(tail) != 1:
        raise SystemExit("REFUSE: the header's closing namespace line is absent or ambiguous. Re-read "
                         "engine/attn_fattn_kernels.h instead of inserting the ladder somewhere convenient.")
    block = ["// A6 STEP 3 (compile-side experiment, not a shipping route): the occupancy ladder's rungs. Same",
             "// body, same grid, one different `minBlocksPerMultiprocessor` hint each, so the driver's own answer",
             "// can be asked per rung. Declared beside `attn_fattn_wave`, which is what makes `attn_fattn_body`",
             "// reachable here by unqualified lookup.",
             "#define A6_LB_LADDER(NAME, N) \\",
             "    __global__ void __launch_bounds__(FATTN_THREADS, N) NAME(CoreArgs a) { attn_fattn_body<false>(a); }"]
    for n in range(1, RUNGS + 1):
        block.append(f"A6_LB_LADDER(a6_lb{n}, {n})")
    block.append("#undef A6_LB_LADDER")
    block.append("")
    i = src.index(tail)
    return src[:i] + "\n".join(block) + src[i:]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="src", required=True, type=pathlib.Path)
    ap.add_argument("--out", required=True, type=pathlib.Path)
    a = ap.parse_args()
    text = a.src.read_text()
    out = apply_variant(text)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(out)
    print(f"wrote {a.out}  ({len(out) - len(text):+d} B, {RUNGS} rungs)")


if __name__ == "__main__":
    main()
