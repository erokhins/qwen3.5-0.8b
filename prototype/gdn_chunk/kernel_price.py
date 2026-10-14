#!/usr/bin/env python3
"""Q2 pricing for the chunked GDN scan: compile-only ISA census + grid/occupancy/work arithmetic.
No device is touched (`hipcc -c` is a cross-compile), no timing is taken.

    .venv/bin/python prototype/gdn_chunk/kernel_price.py isa
    .venv/bin/python prototype/gdn_chunk/kernel_price.py grid
    .venv/bin/python prototype/gdn_chunk/kernel_price.py work
    .venv/bin/python prototype/gdn_chunk/kernel_price.py best
    .venv/bin/python prototype/gdn_chunk/kernel_price.py selftest
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
HEADER = ROOT / "prototype" / "gdn_chunk" / "gdn_chunk_kernels.h"

# ---- anchors, each with the file that measured it (no number here is invented) -----------------
CU = 40                       # devices/strix-halo-compute.md correction (rocminfo; HIP's 20 is wrong)
CLK = 2.206e9                 # same file: sysfs pp_dpm_sclk, the clock it sits at
LDS_CU = 65536                # attn-wave-diagnosis.md §3: RDNA3-class CU, 64 KB LDS
WAVES_CU = 32                 # attn-wave-diagnosis.md §0 anchors
THREADS_CU = 1024
ISSUE_PER_CU = 4              # inst/CU/clk, attn-wave-diagnosis.md §0/§2
RATE_HAND_GEMM = 0.85e12      # devices/strix-halo-compute.md: the engine's own hand GEMMs
RATE_VECTOR = 2.93e12         # same file: fp32 hipBLAS row = what a vector FMA path sustains
RATE_SEQ_ACHIEVED = 33.94e9 / 0.127   # gdn_rec read back as a rate (p9 §5, 127.0 ms re-taken)
DRAM_READ = 237e9             # devices/strix-halo-memory-bandwidth.md
L2_RESIDENT = 918e9           # same, measured for an 18 MiB resident set
LAUNCH_FLOOR = 1.8e-6         # eager launch floor, attn-wave-diagnosis.md §0
GDN_REC_MS = 127.0            # Phase 1b device window (seq run of the same wave)
WAVE_SEQ_MS = 331.2
WAVE_CHUNK_MS = 1953.7
WIN_LINE_MS, KILL_LINE_MS = 63.0, 84.0
C, D, H, LAYERS, WAVE_TOKENS, BODY = 64, 128, 16, 18, 1024, 512
THREADS = 128                 # GC_THREADS
STRIP = 32                    # GC_STRIP


def isa_resources() -> dict:
    """Compile-only: resources + the LDS descriptor + the wave32-sharing instruction census."""
    tmp = Path("/tmp/gdn_chunk_price_isa")
    tmp.mkdir(exist_ok=True)
    (tmp / "one.cpp").write_text(f'#include "{HEADER}"\n')
    r = subprocess.run(["hipcc", "-O3", "--offload-arch=gfx1151", "-I", str(ROOT),
                        "-Rpass-analysis=kernel-resource-usage", "-c", str(tmp / "one.cpp"),
                        "-o", str(tmp / "one.o"), "--save-temps"],
                       capture_output=True, text=True, cwd=tmp)
    if r.returncode:
        raise SystemExit("COMPILE FAILED (compile-only; no device touched):\n" + r.stderr[-3000:])
    out, cur = {}, None
    for line in r.stderr.splitlines():
        m = re.search(r"Function Name: _Z\d+(gdn_chunk_prep|gdn_chunk_carry)\d*", line)
        if m:
            cur = m.group(1)
            out.setdefault(cur, {})
            continue
        m = re.search(r"(TotalSGPRs|VGPRs Spill|VGPRs|ScratchSize|Occupancy|LDS Size)[^:]*: (\d+)", line)
        if m and cur:
            out[cur][m.group(1)] = int(m.group(2))
    text = next(tmp.glob("*gfx1151.s")).read_text()
    bodies, cur = {}, None
    for ln in text.splitlines():
        m = re.match(r"^_Z\d+(gdn_chunk_prep|gdn_chunk_carry)\d*\w*:", ln)
        if m:
            cur = m.group(1)
            bodies.setdefault(cur, [])
            continue
        if cur is not None:
            bodies[cur].append(ln)
    pats = {"ds_load_b128": r"\bds_load_b128", "ds_load_b64": r"\bds_load_b64",
            "ds_load_b32": r"\bds_load_b32", "ds_load_u16": r"\bds_load_u16",
            "ds_store_b128": r"\bds_store_b128", "ds_store_b64": r"\bds_store_b64",
            "ds_store_b32": r"\bds_store_b32",
            "ds_bpermute (cross-lane)": r"\bds_bpermute", "ds_swizzle (cross-lane)": r"\bds_swizzle",
            "v_fma_f32": r"\bv_fma_f32", "v_fmac_f32": r"\bv_fmac_f32",
            "v_cvt ...bf": r"v_cvt[a-z0-9_]*bf|v_cvt_bf16", "v_add_co (bf16 RNE ladder)": r"v_add_co",
            "v_and_b32 (bf16 mask)": r"v_and_b32",
            "v_exp_f32": r"\bv_exp", "s_barrier": r"\bs_barrier",
            "global_load_b32": r"global_load_b32", "global_load_b64": r"global_load_b64",
            "global_load_u16": r"global_load_u16", "global_store_b32": r"global_store_b32",
            "v_mfma (matrix unit)": r"\bv_mfma", "s_waitcnt ds": r"s_waitcnt.*ds"}
    for name in bodies:
        out.setdefault(name, {})
        for k, p in pats.items():
            out[name][k] = sum(1 for ln in bodies[name] if re.search(p, ln))
    return out


def strip_comments(src: str) -> str:
    """Code only — so a comment that QUOTES the defective expression cannot witness a check on it."""
    src = re.sub(r"/\*.*?\*/", " ", src, flags=re.S)
    return re.sub(r"//[^\n]*", "", src)


def residency(lds: int) -> int:
    return LDS_CU // lds if lds else 0


def cmd_isa(args) -> int:
    res = isa_resources()
    for name in ("gdn_chunk_prep", "gdn_chunk_carry"):
        d = res[name]
        lds = d["LDS Size"]
        blocks = residency(lds)
        print(f"{name}:")
        print(f"    VGPRs {d['VGPRs']}  SGPRs {d['TotalSGPRs']}  scratch {d['ScratchSize']} B/lane"
              f"  register-limited occupancy {d['Occupancy']} waves/SIMD")
        print(f"    LDS {lds:,} B/block -> {blocks} block(s)/CU at {LDS_CU:,} B "
              f"(two would need {2*lds:,} B, {2*lds - LDS_CU:,} B over)"
              f" -> {THREADS//32*blocks} of {WAVES_CU} wave slots = "
              f"{100.0*(THREADS//32)*blocks/WAVES_CU:.1f} % thread occupancy")
        keys = ("ds_load_b128", "ds_load_b64", "ds_load_b32", "ds_load_u16", "ds_store_b128",
                "ds_store_b64", "ds_store_b32", "ds_bpermute (cross-lane)",
                "ds_swizzle (cross-lane)", "v_fma_f32", "v_fmac_f32", "v_cvt ...bf",
                "v_add_co (bf16 RNE ladder)", "v_and_b32 (bf16 mask)", "v_exp_f32",
                "s_barrier", "global_load_b32", "global_load_b64", "global_load_u16",
                "global_store_b32", "v_mfma (matrix unit)", "s_waitcnt ds")
        print("    static ISA sites: " + "  ".join(f"{k}={d[k]}" for k in keys))
    print("\n  What the census supports, and only this:")
    print("  * prep moves every LDS element through a 4-byte ds_load_b32/ds_store_b32 — there is not")
    print("    one b64 or b128 LDS site in it — which is what `[C,C]` code that walks `d` serially per")
    print("    thread looks like. The carry DOES vectorise (16 ds_load_b128, 24 ds_store_b128), in")
    print("    the state staging and the `vn` reads. So prep pays one LDS instruction per element;")
    print("    that is 2,453 M LDS lane-reads per wave (see `work`).")
    print("  * The bank behaviour comes from the source geometry, not from a profiler: in the [C,C]")
    print("    loop lane L owns p = tid + k*128, so lanes differ in `s` and read kst[s*GC_D + d] at")
    print("    addresses GC_D=128 words apart — and 128 words is 4 full turns of the 32 LDS banks,")
    print("    so all 32 lanes land in ONE bank (32-way conflict, 32 cycles for a 4-byte read).")
    print("    The paired read `kst[t*GC_D + d]` is warp-uniform and broadcasts for free. Half of")
    print("    prep's LDS reads are therefore conflicted, and padding the pitch to GC_D+1 words")
    print("    (or XOR-ing the row into the column) is what fixes it — an indexing change, not an")
    print("    arithmetic one. It is worth ~14 ms, which is why it is not the story.")
    print("  * ds_bpermute/ds_swizzle: 16+4 sites in prep, ZERO in the carry. That is `gc_block_sum`'s")
    print("    `__shfl_down_sync` tree (the q~/k~ rsqrt reduction), which the carry does not do. It")
    print("    is not an LDS-vectorisation artefact, and it is not a cost: 5 shuffles per row norm.")
    print("  * A BF16 round-trip on gfx1151 is a software ladder (v_add_co_* + v_and_b32 mask), not a")
    print("    single instruction — so `gc_bf()` costs about 3 VALU ops per element on top of the")
    print("    FMA that consumes it. Every MAC in these kernels carries one LDS load and that ladder,")
    print("    which is the instruction-mix reason a hand loop here sits near 0.4 TF/s and not 2.93.")
    print("  * 0 v_mfma in both kernels: this is a vector FMA path, so its ceiling is the fp32 row of")
    print("    devices/strix-halo-compute.md, 2.93 TF/s — and hipBLAS's 4.7-5.9 TF/s bf16 row is")
    print("    matrix units, which this engine does not use anywhere.")
    return 0


def cmd_grid(args) -> int:
    res = isa_resources()
    bodies = WAVE_TOKENS // BODY
    sites = LAYERS * bodies
    chunks_body = BODY // C
    prep_blocks_site = chunks_body * H
    carry_blocks_site = (D // STRIP) * H
    print(f"=== grid: {WAVE_TOKENS}-token wave, {BODY}-row bodies ({bodies}), {sites} "
          f"(layer,body) sites, C={C}, d_k=d_v={D}, {H} value heads")
    for name, blocks_site, lds in (("gdn_chunk_prep", prep_blocks_site, res["gdn_chunk_prep"]["LDS Size"]),
                                   ("gdn_chunk_carry", carry_blocks_site, res["gdn_chunk_carry"]["LDS Size"])):
        blocks_wave = blocks_site * sites
        per_cu = blocks_wave / CU
        r = residency(lds)
        rounds = blocks_site / (CU * r)
        print(f"  {name}: grid {blocks_site} blocks per site "
              f"({'chunks x heads' if 'prep' in name else '(d_v/32) x heads'}) x {sites} sites "
              f"= {blocks_wave:,} blocks/wave = {per_cu:.1f} per CU")
        print(f"      LDS {lds:,} B -> {r} block(s)/CU resident = {r*THREADS} threads = "
              f"{100.0*r*THREADS/THREADS_CU:.1f} % of the {THREADS_CU}-thread CU")
        print(f"      one site's blocks fit in {rounds:.2f} residency rounds; the site is a "
              f"dependency barrier (carry reads what prep wrote)")
    print(f"  serial depth per block: {chunks_body} chunk-steps (was {BODY} token-steps in "
          f"gdn_rec); the carry is 8 dependent steps, each with 3 __syncthreads")
    launches = 2 * sites
    print(f"  launches: {launches} replacing gdn_rec's 18,432 at {LAUNCH_FLOOR*1e6:.1f} us eager "
          f"floor = {launches*LAUNCH_FLOOR*1e3:.2f} ms vs {18432*LAUNCH_FLOOR*1e3:.1f} ms")
    print(f"  wave wall: seq {WAVE_SEQ_MS:.1f} ms -> chunk {WAVE_CHUNK_MS:.1f} ms, so the chunked "
          f"GDN line is ~{WAVE_CHUNK_MS - WAVE_SEQ_MS + GDN_REC_MS:.0f} ms against gdn_rec's "
          f"{GDN_REC_MS:.1f} ms ({(WAVE_CHUNK_MS-WAVE_SEQ_MS+GDN_REC_MS)/GDN_REC_MS:.1f}x) and "
          f"the added wall is {WAVE_CHUNK_MS-WAVE_SEQ_MS:.0f} ms")
    return 0


def work_terms() -> dict:
    """Modelled and executed MACs per (chunk, head), as the code is written."""
    modelled_intra = (2 * C * C * D + 2 * C * C * D + 4 * C * C * D + 2 * C * C * D
                      + 2 * C ** 3 // 3) // 2
    modelled_inter = (6 * C * D * D + D * D) // 2
    # prep: as modelled (D4's evaluated-then-discarded upper half is already inside 2*C*C*D).
    executed_intra = modelled_intra
    # carry: v_new (C*D*D) and out (C*D*D + C*C*D/2) run 32x; the state update (C*D*D) is correct.
    executed_inter = 32 * (C * D * D + C * D * D + C * C * D // 2) + C * D * D
    return dict(modelled_intra=modelled_intra, modelled_inter=modelled_inter,
                executed_intra=executed_intra, executed_inter=executed_inter)


def cmd_work(args) -> int:
    w = work_terms()
    chunk_heads = LAYERS * H * (WAVE_TOKENS // C)
    mod = (w["modelled_intra"] + w["modelled_inter"]) * chunk_heads
    exe = (w["executed_intra"] + w["executed_inter"]) * chunk_heads
    dup = (w["executed_inter"] - w["modelled_inter"]) * chunk_heads
    print(f"=== work per (chunk,head): modelled intra {w['modelled_intra']/1e6:.2f} M MAC, "
          f"inter {w['modelled_inter']/1e6:.2f} M MAC")
    print(f"    executed intra {w['executed_intra']/1e6:.2f} M, inter {w['executed_inter']/1e6:.2f} M "
          f"= {w['executed_inter']/w['modelled_inter']:.1f}x the modelled carry")
    print(f"    reason: `jl` (gdn_chunk_kernels.h:379) is computed and never read, so all 32 lanes")
    print(f"    of a row-group each compute all {STRIP} strip columns of v_new and out.")
    print(f"    per wave ({chunk_heads:,} chunk-heads): modelled {mod*2/1e9:.2f} GFLOP "
          f"(Phase 0 priced 54.03), executed {exe*2/1e9:.1f} GFLOP = {exe/mod:.1f}x")
    print(f"    duplicated work alone: {dup*2/1e9:.1f} GFLOP")
    print("    priced at the two vector-path rates this box has measured for hand loops:")
    for rate, note in ((RATE_SEQ_ACHIEVED, "gdn_rec read back (0.267 TF/s)"),
                       (RATE_HAND_GEMM, "the engine's best hand GEMM (0.85 TF/s)"),
                       (RATE_VECTOR, "fp32 vector ceiling (2.93 TF/s)")):
        print(f"      {rate/1e12:5.3f} TF/s  modelled {mod*2/rate*1e3:7.1f} ms   "
              f"executed {exe*2/rate*1e3:8.1f} ms   duplicated {dup*2/rate*1e3:8.1f} ms   ({note})")
    added = WAVE_CHUNK_MS - WAVE_SEQ_MS
    exec_rate = exe * 2 / ((added + GDN_REC_MS) * 1e-3)
    print(f"    measured added wall {added:.0f} ms (chunked GDN line ~{added+GDN_REC_MS:.0f} ms), so")
    print(f"    the kernel's executed rate read back off the device window is "
          f"{exec_rate/1e12:.3f} TF/s.")
    print("    Priced at the two hand-loop rates this box has measured (0.27 / 0.85 TF/s), the")
    print(f"    executed arithmetic spans {exe*2/RATE_HAND_GEMM*1e3:.0f}-{exe*2/RATE_SEQ_ACHIEVED*1e3:.0f} ms, which contains the")
    print("    measured line: the wall is duplicated arithmetic, not a stall mystery.")
    prep_lane = 4608 * 2080 * 2 * D          # kst[s] and kst[t] reads, one lane-read per (pair,d)
    print(f"    LDS read volume, prep: {prep_lane/1e6:.1f} M lane-reads/wave = {prep_lane/32/1e6:.1f} M LDS")
    print("      instructions, every one of them a 4-byte ds_load_b32 (see `isa`). Half are the")
    print("      warp-uniform kst[t] read and broadcast free; the other half are the kst[s] reads,")
    print("      GC_D=128 words apart per lane = one bank = a 32-way conflict:")
    print(f"      {prep_lane/2/CU/CLK*1e3:.1f} ms conflicted, {prep_lane/2/32/CU/CLK*1e3:.1f} ms with the pitch padded to GC_D+1 words.")
    carry_lane = 36 * 64 * 8 * 4 * 16 * (D + 1) * STRIP   # S[i][j] reads in v_new, per row-group
    print(f"    LDS read volume, carry (v_new only): {carry_lane/1e6:.0f} M lane-reads/wave, but they are")
    print(f"      broadcast-shaped — with `jl` dead, all 32 lanes of a row-group read the SAME word, so")
    print("      there is no bank conflict and the LDS unit is not the carry's cost: about")
    print(f"      {carry_lane/32/CU/CLK*1e3:.0f} ms at one LDS instruction per clock per CU.")
    print("      What the redundancy costs is the FMAs and the BF16 rounding behind those reads,")
    print("      32x over — which is the arithmetic in the table above, not a traffic term.")
    return 0


def cmd_best(args) -> int:
    w = work_terms()
    chunk_heads = LAYERS * H * (WAVE_TOKENS // C)
    mod = (w["modelled_intra"] + w["modelled_inter"]) * chunk_heads * 2
    tri_saved = (2 * C * C * D // 2) * chunk_heads * 2       # D4 fixed: triangle only
    print(f"=== best case for THIS geometry (two fixes: the khat expression, and `jl` wired so the")
    print(f"    carry stops recomputing the strip 32x). Modelled work {mod/1e9:.2f} GFLOP/wave;")
    print(f"    dropping D4's discarded upper half takes it to {(mod-tri_saved)/1e9:.2f} GFLOP.")
    for flops, lbl in ((mod, "with D4 as written"), (mod - tri_saved, "D4 fixed")):
        print(f"    {lbl}:")
        for rate, note in ((RATE_HAND_GEMM, "engine's best hand GEMM"),
                           (RATE_VECTOR, "fp32 vector ceiling")):
            ms = flops / rate * 1e3 + 2 * LAYERS * (WAVE_TOKENS // BODY) * LAUNCH_FLOOR * 1e3
            verdict = "clears the win line" if ms <= WIN_LINE_MS else (
                "middle band" if ms <= KILL_LINE_MS else "over the kill line")
            print(f"      at {rate/1e12:5.2f} TF/s ({note:24s}) {ms:7.1f} ms   {verdict}")
    need_win = mod / (WIN_LINE_MS * 1e-3) / 1e12
    need_kill = mod / (KILL_LINE_MS * 1e-3) / 1e12
    print(f"    required rate: {need_win:.3f} TF/s to hit {WIN_LINE_MS:.0f} ms = "
          f"{100*need_win/(RATE_HAND_GEMM/1e12):.0f} % of the best hand GEMM, "
          f"{100*need_win/(RATE_VECTOR/1e12):.0f} % of the vector ceiling; "
          f"{need_kill:.3f} TF/s for the {KILL_LINE_MS:.0f} ms kill line")
    w = work_terms()
    chunk_heads = LAYERS * H * (WAVE_TOKENS // C)
    exe_rate = (w["executed_intra"] + w["executed_inter"]) * chunk_heads * 2
    print(f"    today's executed rate, read back off the device window: "
          f"{exe_rate/((WAVE_CHUNK_MS-WAVE_SEQ_MS+GDN_REC_MS)*1e-3)/1e12:.3f} TF/s")
    return 0


def cmd_selftest(args) -> int:
    fails = []
    res = isa_resources()
    for k in ("gdn_chunk_prep", "gdn_chunk_carry"):
        if res[k]["LDS Size"] <= 0 or res[k]["VGPRs"] <= 0:
            fails.append(f"{k}: compiler reported no resources")
        if res[k]["v_mfma (matrix unit)"] != 0:
            fails.append(f"{k}: v_mfma appeared, the vector-path ceiling argument changes")
    if residency(res["gdn_chunk_prep"]["LDS Size"]) != 1:
        fails.append("prep residency is not 1 block/CU: re-read the LDS number before quoting occupancy")
    if residency(res["gdn_chunk_carry"]["LDS Size"]) != 2:
        fails.append("carry residency is not 2 blocks/CU")
    w = work_terms()
    if w["executed_inter"] // w["modelled_inter"] < 20:
        fails.append("the executed/modelled carry ratio collapsed; the jl argument changed")
    mod = (w["modelled_intra"] + w["modelled_inter"]) * LAYERS * H * (WAVE_TOKENS // C) * 2
    if not 50e9 < mod < 58e9:
        fails.append(f"modelled wave work {mod/1e9:.2f} GFLOP does not reproduce Phase 0's 54.03")
    src = strip_comments(HEADER.read_text())
    if "const int jl = tid & (GC_STRIP - 1);" not in src:
        fails.append("the `jl` line moved; the 32x argument needs re-deriving")
    # In the code (comments stripped: the header's own D6 block quotes this text), `jl` must appear
    # exactly once — the declaration. A second occurrence means it is now used, which is the fix,
    # and the 32x redundancy argument would no longer hold.
    if len(re.findall(r"\bjl\b", src)) != 1:
        fails.append("`jl` is now read somewhere: re-derive the 32x carry redundancy")
    if "gamC / gam[i]" not in src:
        fails.append("the gamC/gam line is gone from the CODE: the Q1 finding needs re-quoting")
    print(f"selftest: {len(fails)} failures" + "".join("\n  " + f for f in fails))
    return 1 if fails else 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", choices=("isa", "grid", "work", "best", "selftest"))
    args = ap.parse_args()
    raise SystemExit({"isa": cmd_isa, "grid": cmd_grid, "work": cmd_work, "best": cmd_best,
                      "selftest": cmd_selftest}[args.cmd](args))
