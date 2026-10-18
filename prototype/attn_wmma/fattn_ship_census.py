#!/usr/bin/env python3
"""The SHIPPED-PLACEMENT census for `attn_fattn_wave` — the spill gate and the derived-geometry canary.

WHY THIS EXISTS AND WHAT IT REFUSES TO DO. `prototype/attn_wmma/fattn_census.py` priced the PROBE. The probe
has no head axis, so its LDS figure and its 9.719 pipe-weighted slots/touch describe a kernel that cannot be
routed; carrying those numbers onto the shipping path is how claim (i) stayed believed after it had gone red.
This script reads the note block of the kernel the engine actually launches, fails on any non-zero spill (the
gate the Q placement was decided under), and re-derives its expected instruction sites from the geometry
constants in `engine/attn_fattn_kernels.h` rather than copying them from anywhere — a canary whose expected
values were hand-copied from an older geometry asserts the old geometry and passes vacuously.

    python3 prototype/attn_wmma/fattn_ship_census.py [--asm PATH] [--no-build]

Exit 0 = the shipped placement is spill-free and its instruction skeleton matches its own source. Every
number it prints is a compile-time fact (`.s` note block + static sites); nothing here touches a device.
"""
import argparse
import re
import subprocess
import sys

HEADER = "engine/attn_fattn_kernels.h"
NOTE_KEYS = ("vgpr_count", "vgpr_spill_count", "sgpr_count", "sgpr_spill_count",
             "private_segment_fixed_size", "group_segment_fixed_size")


def geometry():
    """The trip counts, read out of the kernel's own source rather than restated here."""
    src = open(HEADER).read()
    g = {}
    for name in ("FATTN_MROWS", "FATTN_MLANES", "FATTN_THREADS"):
        g[name] = int(re.search(r"^#define\s+%s\s+(\d+)" % name, src, re.M).group(1))
    op = open("prototype/attn_decode/attn_decode_kernels.h").read()
    g["DH"] = int(re.search(r"^#define\s+DH\s+(\d+)", op, re.M).group(1))
    g["KTILES"] = g["VTILES"] = g["DH"] // 16
    body = src[src.index("attn_fattn_body"):]
    body = body[:body.index("__device__ __host__ inline bool fattn_plan_ok")]
    g["score_rolled"] = "#pragma unroll 1" in body
    g["transpose_halves"] = 2
    g["launch_bounds"] = "__launch_bounds__(FATTN_THREADS)" in src
    g["one_warp_per_block"] = g["FATTN_THREADS"] == 32
    return g


def derive(g):
    """STATIC SITE counts the listing must show, derived from the geometry above.

    Sites, not per-round dynamic cost: the score leg is rolled, so its operand loads appear once and run
    KTILES times. Deriving dynamic cost here would fail against the listing for a reason that has nothing to
    do with the kernel — the same class of error as a canary whose expected numbers were copied from a probe
    with a different shape.
    """
    return {
        # one v_wmma site in the rolled score loop, one per tile in the unrolled P*V loop
        "v_wmma": (1 if g["score_rolled"] else g["KTILES"]) + g["VTILES"],
        # 2 operand tiles per score site (4 ds_load_b128), 1 per output tile in the unrolled PV loop, and the
        # three stage-fill sites (Q row, K row, V row). +2 tolerated for whatever the epilogue adds.
        "ds_load_b128": (4 if g["score_rolled"] else g["KTILES"] * 4) + g["VTILES"] * 2 + 3,
        # 5 butterflies each in warp_min and warp_max (outside the round), the round's 1 max fold + 8 C->B
        # exchanges, and the 1 deferred rowsum combine: 10 outside, 9 in the round, 1 after it.
        "cross_lane": 10 + 9 + 1,
        # Barrier SITES. At one warp per block a `__syncthreads()` has no second wavefront to sequence, and
        # the measured listing contains ZERO `s_barrier` instructions — so the third barrier per round that
        # aliasing V over K costs on a multi-warp block is free at this geometry, which is a large part of why
        # placement (B) beat placement (A) on more than registers. If FATTN_THREADS ever exceeds a warp this
        # expectation must become 1 + 3 + 3 (Q published / K published / K consumed / V published, then the
        # transpose), and the wavefronts-per-CU claim has to be re-derived with it.
        "s_barrier": 0 if g["one_warp_per_block"] else 1 + 3 + (2 * g["transpose_halves"] + 1),
        # The ordering cost does not disappear, it moves here: LDS RAW within a wavefront is carried by
        # `s_waitcnt lgkmcnt(0)`, which the compiler inserts whether or not a barrier survives.
        "s_wait_lgkmcnt0": 3,
    }


def counts(asm_text):
    """Per-kernel static site counts, from the kernel's label to its `.Lfunc_end`.

    Splitting the file on a `.text` header (an earlier attempt here) leaves no chunk containing the label, so
    every count comes back 0 and the census FAILs a healthy kernel — the counting twin of the empty-parse
    failure in `prototype/gdn_collapse/results.md` §1.
    """
    out = {}
    for m in re.finditer(r"^(_ZN6engine7kernels\w+):", asm_text, re.M):
        rest = asm_text[m.end():]
        end = rest.find(".Lfunc_end")
        body = rest[:end if end > 0 else len(rest)]
        n = lambda pat: len(re.findall(r"^\s+" + pat + r"(\s|$)", body, re.M))
        out[m.group(1)] = {
            "v_wmma": n(r"v_wmma_\w+"),
            "ds_load_b128": n(r"ds_load_b128"),
            "cross_lane": n(r"ds_bpermute_b32") + n(r"ds_swizzle_b32"),
            "s_barrier": n(r"s_barrier"),
            "s_wait_lgkmcnt0": n(r"s_waitcnt lgkmcnt\(0\)"),
        }
    return out


def note_blocks(asm_text):
    """The AMDHSA YAML note blocks, keyed by mangled and by short name.

    The YAML lists one entry per kernel starting at `  - .args:` and names it later with `.name:`. The
    `.amdhsa_kernel NAME` lines inside `.text` are the ELF section directive, not the note; splitting on
    those (or on `---`) yields zero blocks and a census that reports the kernel is missing when it is right
    there in the file.
    """
    blocks = {}
    if "amdhsa.kernels:" not in asm_text:
        return blocks
    yaml = asm_text[asm_text.index("amdhsa.kernels:"):]
    for blk in re.split(r"^  - \.args:", yaml, flags=re.M)[1:]:
        m = re.search(r"^\s+\.name:\s*([^\s]+)", blk, re.M)
        if not m:
            continue
        full = m.group(1)
        rec = {}
        for k in NOTE_KEYS:
            mm = re.search(r"\.%s:\s+(\S+)" % k, blk)
            rec[k] = int(mm.group(1)) if mm else -1
        blocks[full] = rec
        blocks[re.sub(r"^_ZN6engine7kernels\d+", "", full).split("E")[0]] = rec
    return blocks


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--asm", default="/tmp/fattn_census2/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s")
    ap.add_argument("--no-build", action="store_true")
    a = ap.parse_args()
    if not a.no_build:
        subprocess.run(["bash", "tmp/gcc/fattn_census_build.sh", "/tmp/fattn_census2"], check=True,
                       stdout=subprocess.DEVNULL)
        import shutil, os
        # `--save-temps` writes BOTH the device and the host listing into the CWD; moving whichever
        # `os.listdir` returns last is how this census once read the x86_64 listing and found no note block.
        for f in os.listdir("."):
            if f.startswith("kernels_attn-hip-amdgcn") and f.endswith(".s"):
                shutil.move(f, a.asm)
    text = open(a.asm).read()
    g, exp = geometry(), None
    notes, obs = note_blocks(text), counts(text)
    if "attn_fattn_wave" not in notes:
        sys.exit("no note block for attn_fattn_wave in " + a.asm)
    n = notes["attn_fattn_wave"]
    exp = derive(g)
    print("geometry (%s): MROWS=%d MLANES=%d THREADS=%d KTILES=%d VTILES=%d score_leg=%s launch_bounds=%s"
          % (HEADER, g["FATTN_MROWS"], g["FATTN_MLANES"], g["FATTN_THREADS"], g["KTILES"], g["VTILES"],
             "rolled" if g["score_rolled"] else "unrolled", "yes" if g["launch_bounds"] else "NO"))
    print("note block: " + "  ".join("%s=%s" % (k, n[k]) for k in NOTE_KEYS))
    fails = []
    if not g["launch_bounds"]:
        fails.append("no __launch_bounds__(FATTN_THREADS): without it LLVM budgets a thread 196,608/1024 = "
                     "192 VGPRs and spills the rest, so a spilling result would be about the declaration and "
                     "not the kernel")
    for k, limit in (("vgpr_spill_count", 0), ("private_segment_fixed_size", 0)):
        if n[k] != limit:
            fails.append("%s=%d, must be %d. That is the spill gate: scratch lands on the same dependent "
                         "chain bench/performance-status.md 28 identifies as the mechanism, so --attn-fattn "
                         "stays default 0 until this is 0" % (k, n[k], limit))
    lds_wanted = (g["FATTN_MROWS"] * g["DH"] * 2) + (g["FATTN_MLANES"] * g["DH"] * 2)
    if n["group_segment_fixed_size"] != lds_wanted:
        fails.append("group_segment_fixed_size=%d, but the source's stages are %d B (Q %d + aliased K/V %d)"
                     % (n["group_segment_fixed_size"], lds_wanted, g["FATTN_MROWS"] * g["DH"] * 2,
                        g["FATTN_MLANES"] * g["DH"] * 2))
    key = [k for k in obs if k.endswith("attn_fattn_waveE8CoreArgs")]
    if not key:
        fails.append("no attn_fattn_wave body found in the listing to count instruction sites in")
    else:
        o = obs[key[0]]
        print("derived-vs-listing: " + "  ".join("%s want=%d got=%d" % (k, exp[k], o[k])
                                                 for k in ("v_wmma", "ds_load_b128", "cross_lane",
                                                           "s_barrier", "s_wait_lgkmcnt0")))
        if o["s_barrier"] != exp["s_barrier"]:
            fails.append("s_barrier sites %d, derived %d. At one warp per block LLVM deletes every "
                         "__syncthreads() (measured 0), which is also why aliasing V over K is free here; a "
                         "non-zero count means the block is wider than a warp and the wavefronts-per-CU line "
                         "below is stale too" % (o["s_barrier"], exp["s_barrier"]))
        if o["s_wait_lgkmcnt0"] < exp["s_wait_lgkmcnt0"]:
            fails.append("only %d `s_waitcnt lgkmcnt(0)` sites, expected at least %d: the staged Q, the K "
                         "stage and the V stage each need an LDS read-after-write fence inside the wavefront. "
                         "Deleting barriers does not delete the ordering requirement — if this count ever "
                         "falls to 0 the transposed-V read is unsynchronised and the form is wrong on the "
                         "wavefront level, not just slow" % (o["s_wait_lgkmcnt0"], exp["s_wait_lgkmcnt0"]))
        if o["v_wmma"] != exp["v_wmma"]:
            fails.append("v_wmma sites %d, derived %d from KTILES/VTILES and the roll/unroll pragmas"
                         % (o["v_wmma"], exp["v_wmma"]))
        if o["cross_lane"] != exp["cross_lane"]:
            fails.append("cross-lane sites %d, derived %d: 10 warp min/max + 9 in the round + 1 deferred "
                         "rowsum combine. One fewer is D-4 coming back (a half denominator per row, invisible "
                         "to cosine)" % (o["cross_lane"], exp["cross_lane"]))
        if o["ds_load_b128"] > exp["ds_load_b128"] + 2:
            fails.append("ds_load_b128 sites %d > derived %d + 2: the operand tiles this form reads are a "
                         "property of the staging loops, and reading a score tile from global instead would "
                         "be the §20.5 traffic error all over again" % (o["ds_load_b128"],
                                                                       exp["ds_load_b128"]))
    print("touches per warp-round = %d (MROWS x MLANES), the same denominator fattn_census.py used"
          % (g["FATTN_MROWS"] * g["FATTN_MLANES"]))
    print("LDS-bound blocks per CU = %d (64 KiB / %d B), wavefronts/CU = %d at %d warp(s) per block"
          % (65536 // max(n["group_segment_fixed_size"], 1), n["group_segment_fixed_size"],
             65536 // max(n["group_segment_fixed_size"], 1), g["FATTN_THREADS"] // 32))
    if fails:
        print("\nFAIL — the shipped placement is NOT cleared to become the default")
        for f in fails:
            print("  - " + f)
        return 1
    print("\nPASS: spill-free, LDS matches the source, instruction skeleton matches its own geometry")
    return 0


if __name__ == "__main__":
    sys.exit(main())
