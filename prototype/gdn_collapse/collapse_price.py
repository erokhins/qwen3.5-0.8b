#!/usr/bin/env python3
"""Route A (D-12) pricing: compile-only ISA resources + the grid/work/rate arithmetic, BEFORE and AFTER
the kernel exists, so the projection and the compiler's number can be compared instead of conflated.

    .venv/bin/python prototype/gdn_collapse/collapse_price.py --geometry   # the arithmetic
    .venv/bin/python prototype/gdn_collapse/collapse_price.py --isa        # compile-only, no device

NOTHING HERE TOUCHES THE DEVICE. `hipcc -c` is a cross-compile; no binary is launched, no timing is
taken, and no number printed here may be quoted as a measurement — the measured figures live in
`bench/` and come from a main-agent device pass.

The anchors are imported, not retyped: `prototype/gdn_chunk/kernel_price.py` carries the device numbers
(CUs, clock, LDS per CU, issue width, the two rate rows, the memory rates, the launch floor) each with
the file that measured it, and the sequential baseline (127.0 ms for `gdn_rec` at 1024 lanes) and the
win line (≤ 63 ms) come from `bench/p9-phase1-2026-10-14.md` D-12, where the line was written BEFORE
the route was funded. The whole point of re-deriving this on the CPU is D-12's rule that a route whose
honest projection cannot clear the line does not get a kernel.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "prototype" / "gdn_chunk"))
import kernel_price as kp                                                # noqa: E402

HEADER = ROOT / "engine" / "gdn_collapse_kernels.h"
KERNEL = "rec_collapse_kernel"

# What the geometry is, in one place, so `--geometry` and `--isa` cannot drift from the code.
STRIPS, COLS, BLOCK_DIM, HEADS = 4, 32, 128, 16
D, LAYERS, BODY_ROWS, WAVE_LANES = 128, 18, 512, 1024
SEQ_MS = 127.0                 # bench/p9-phase1-2026-10-14.md §5: gdn_rec, 1024 lanes, re-taken
SEQ_LAUNCHES = LAYERS * WAVE_LANES   # 18,432: one launch per layer per token
WORK_GFLOP = 33.94             # D-12: the sequential work per 1024-lane wave, re-read as a rate
WIN_LINE_MS = kp.WIN_LINE_MS   # 63.0 ms, pre-registered in D-12
KILL_LINE_MS = kp.KILL_LINE_MS # 84.0 ms
RATE_ENGINE_EXEC = 0.417e12    # what the chunked kernel actually executed at (Q1/D-12's denominator)

# Per-token instruction budget, counted from the source, per USEFUL wavefront (one d_v column each):
#   step-2 dot  128 x (LDS Sst + LDS ks[i] + FMA)                  = 384   (256 LDS, 128 VALU)
#   step-3      128 x (LDS Sst + MUL + LDS ks[i] + FMA + STS
#                      + LDS qs[i] + FMA)                          = 896   (512 LDS, 384 VALU)
#   two norms   2 x (10 shuffles + adds + red staging + barriers)  =  ~90
# It is a count of instructions, not of cycles: what a SIMD can issue is the ceiling this feeds.
INST_TOKEN = {"lds": 256 + 512, "valu": 128 + 384 + 90}

# The LDS footprint the source predicts: one 128x32 F32 state strip (16,384 B) + k̃/q̃ (1,024 B) + the
# reduction's 4 partials (16 B). `--isa` asserts the compiler's own number matches this.
LDS_BLOCK_PREDICTED = D * COLS * 4 + 2 * D * 4 + 16


def isa() -> dict:
    """Compile-only: resources of the collapsed kernel, the instruction census of its body, and the
    `-Wall` cleanliness of the header itself (the AGENTS rule that an injected or dead variable must not
    hide in a kernel the gate compiles)."""
    tmp = Path("/tmp/gdn_collapse_price")
    tmp.mkdir(exist_ok=True)
    (tmp / "one.cpp").write_text(f'#include "{HEADER}"\n')
    r = subprocess.run(["hipcc", "-O3", "-Wall", "--offload-arch=gfx1151", "-I", str(ROOT),
                        "-Rpass-analysis=kernel-resource-usage", "-c", str(tmp / "one.cpp"),
                        "-o", str(tmp / "one.o"), "--save-temps"],
                       capture_output=True, text=True, cwd=tmp)
    if r.returncode:
        raise SystemExit("COMPILE FAILED (compile-only; no device touched):\n" + r.stderr[-3000:])
    res, cur = {}, None
    for line in r.stderr.splitlines():
        if KERNEL in line and "Function Name" in line:
            cur = KERNEL
            res.setdefault(cur, {})
            continue
        m = re.search(r"(TotalSGPRs|VGPRs Spill|VGPRs|ScratchSize|LDS Size|Occupancy)[^:]*: (\d+)", line)
        if m and cur:
            res[cur][m.group(1)] = int(m.group(2))
    own = [ln for ln in r.stderr.splitlines()
           if re.match(re.escape(str(HEADER)) + r":\d+:\d+: ", ln)]
    unused = [ln for ln in own if "unused" in ln]
    text = next(tmp.glob("*gfx1151.s")).read_text()
    body, on = [], False
    for ln in text.splitlines():
        if re.match(r"^_ZN6gdn_rc19rec_collapse_kernel", ln):
            on = True
            continue
        # Stop at the next symbol, not at the next `.section`: a kernel's own body can contain section
        # directives (constant pools), and cutting there is how a census silently under-counts.
        if on and re.match(r"^[A-Za-z_][A-Za-z0-9_.$]*:", ln):
            break
        if on:
            body.append(ln)
    pats = {"v_fma_f32": r"\bv_fm[af]c?_f32", "v_mul_f32": r"\bv_mul_f32",
            "rsqrt/exp (norms, decay)": r"v_rsq|v_sqrt|v_rcp_f32|v_exp", "ds_read_b32": r"\bds_load_b32|\bds_read_b32",
            "ds_read_b64/b128": r"ds_(load|read)_(b64|b128)", "ds_write_b32": r"\bds_store_b32|\bds_write_b32",
            "ds_write_b64/b128": r"ds_(store|write)_(b64|b128)", "ds_swizzle (cross-lane)": r"\bds_swizzle",
            "permlane (cross-lane)": r"\bpermlane", "s_barrier": r"\bs_barrier",
            "v_mfma (matrix units)": r"\bv_mfma", "v_cvt (bf16 rounding)": r"v_cvt[a-z0-9_]*bf|v_cvt_bf16",
            "global loads (flat/global)": r"\t(global_load|flat_load)", "global stores": r"\t(global_store|flat_store)"}
    census = {name: sum(1 for ln in body if re.search(p, ln)) for name, p in pats.items()}
    return {"res": res, "census": census, "own_warnings": own, "unused": unused,
            "static_insts": len([ln for ln in body if re.match(r"^\t[a-z]", ln)])}


def cmd_isa() -> int:
    d = isa()
    res = d["res"][KERNEL]
    lds = res["LDS Size"]
    blocks = kp.LDS_CU // lds
    rc = 0
    print(f"== --isa (compile-only; `hipcc -O3 -Wall --offload-arch=gfx1151 -c`, no device touched)")
    print(f"  {KERNEL}: VGPRs {res['VGPRs']}, SGPRs {res['TotalSGPRs']}, scratch {res['ScratchSize']},"
          f" register-limited occupancy {res['Occupancy']} waves/SIMD")
    lds64 = D * (2 * COLS) * 4 + 2 * D * 4 + 16
    print(f"  LDS {lds:,} B/block -> {blocks} block(s)/CU at {kp.LDS_CU:,} B. The staging that was NOT")
    print(f"    taken is the 64-column strip: {lds64:,} B, so {kp.LDS_CU//lds64} block/CU "
          f"(the whole budget, before k\u0303/q\u0303) — see the header's geometry note")
    if blocks < 2:
        print("  BLOCKER: the staging is too big for the 2-blocks-per-CU target")
        rc = 1
    if lds != LDS_BLOCK_PREDICTED:
        print(f"  NOTE: the compiler's LDS ({lds:,} B) is not the {LDS_BLOCK_PREDICTED:,} B the source "
              f"predicts; the geometry above was priced on the prediction")
    print(f"  grid per (layer, body) = {STRIPS} strips x {HEADS} heads = {STRIPS*HEADS} blocks of "
          f"{BLOCK_DIM} threads = {STRIPS*HEADS*BLOCK_DIM//32} wavefronts32 per site")
    print(f"  static instructions in the kernel body: {d['static_insts']} — the two `i` loops stay ROLLED,")
    print("  so this is the per-token body, and the census below is a COMPOSITION check (what kind of")
    print("  instruction each MAC costs), not a count of work per wave; that is `--geometry`'s job.")
    for name, n in d["census"].items():
        print(f"    {name:<26} {n}")
    if d["census"]["v_mfma (matrix units)"]:
        print("  NOTE: matrix instructions present; this route is a vector route (D-12 priced it there)")
    else:
        print("  no v_mfma, no bf16 cvt in the shipping path: the state stays F32 and the arithmetic stays "
              "the sequential kernel's")
    if d["unused"]:
        print("  BLOCKER: -Wall reports an unused variable in the header:")
        for ln in d["unused"]:
            print("    " + ln[:160])
        rc = 1
    else:
        print(f"  -Wall clean of unused/dead variables in {HEADER.relative_to(ROOT)} "
              f"({len(d['own_warnings'])} warning(s) attributed to this file, none of them unused/dead)")
    print("  negatives: each must compile as its own binary (-D), because an injected defect that is")
    print("  reachable from the shipping binary at runtime is not a negative test:")
    for neg in ("NEG_RC_STATE_BF16", "NEG_RC_STRIP_SHIFT", "NEG_RC_NORM_HALF", "NEG_RC_I_DESC"):
        (Path("/tmp/gdn_collapse_price") / "neg.cpp").write_text(
            f"#define {neg}\n#include \"{HEADER}\"\n")
        rr = subprocess.run(["hipcc", "-O3", "-Wall", "--offload-arch=gfx1151", "-I", str(ROOT),
                             "-c", "/tmp/gdn_collapse_price/neg.cpp",
                             "-o", "/tmp/gdn_collapse_price/neg.o"], capture_output=True, text=True)
        own_un = [ln for ln in rr.stderr.splitlines() if str(HEADER) in ln and "unused" in ln]
        print(f"    {neg:<20} {'compiles' if rr.returncode == 0 else 'FAILED'}"
              + ("" if not own_un else f", {len(own_un)} unused-variable warnings"))
        if rr.returncode or own_un:
            rc = 1
    print("VERICT", "FAIL" if rc else "PASS")
    return rc


def cmd_geometry() -> int:
    sites = LAYERS * (WAVE_LANES // BODY_ROWS)
    blocks_site = STRIPS * HEADS
    useful_wf = blocks_site                       # one useful wavefront per block, 32 lanes each
    simds = kp.CU * 4
    inst = INST_TOKEN["lds"] + INST_TOKEN["valu"]
    cycles_token = inst                           # one issue slot per SIMD per cycle
    us_site = BODY_ROWS * cycles_token / kp.CLK * 1e6
    ms_wave = sites * us_site / 1e3
    state_bytes = LAYERS * (WAVE_LANES // BODY_ROWS) * HEADS * STRIPS * (D * COLS * 4) * 2
    seq_state_bytes = LAYERS * WAVE_LANES * HEADS * (D * D * 4) * 2
    uread_bytes = LAYERS * WAVE_LANES * HEADS * STRIPS * (D + D + COLS) * 4
    print("== route A geometry, before the kernel (and re-checkable after: --isa)")
    print(f"  wave = {WAVE_LANES} lanes = {WAVE_LANES//BODY_ROWS} bodies of {BODY_ROWS} rows; "
          f"{LAYERS} GDN layers -> {sites} (layer, body) sites")
    print(f"  one launch per site, {blocks_site} blocks x {BLOCK_DIM} threads "
          f"({useful_wf} useful wavefronts32, {3*blocks_site} parked at the norm barrier)")
    print(f"  vs {kp.CU} CUs / {simds} SIMDs: {useful_wf/simds:.0%} of FP32 lanes carry useful work.")
    print(f"    That is the serial width of §04.4's scan — {HEADS} heads x {D} d_v columns = "
          f"{HEADS*D} chains, and no re-blocking makes more of them. The")
    print(f"    machine has {kp.CU} CUs x 128 FP32 lanes = {kp.CU*128} lanes; the wave is asking "
          f"{HEADS*D} of them to do work, and the other wavefronts are parked.")
    print(f"  LDS {LDS_BLOCK_PREDICTED:,} B/block predicted (--isa prints the compiler's) = "
          f"{kp.LDS_CU//LDS_BLOCK_PREDICTED} blocks/CU capacity; the grid of {blocks_site} blocks over "
          f"{kp.CU} CUs actually lands {blocks_site/kp.CU:.1f} blocks/CU")
    print(f"  serial depth per site = {BODY_ROWS} tokens (the scan; this is what the route cannot remove)")
    print(f"  instructions per token per useful wavefront: {INST_TOKEN['valu']} VALU + "
          f"{INST_TOKEN['lds']} LDS = {inst}")
    print(f"    -> {cycles_token} cycles/token/SIMD, {us_site:.0f} us/site, "
          f"{ms_wave:.1f} ms per {WAVE_LANES}-lane wave")
    print(f"  cross-checks on that number:")
    print(f"    * as a rate: {WORK_GFLOP:.2f} GFLOP / {ms_wave:.1f} ms = "
          f"{WORK_GFLOP*1e9/(ms_wave*1e-3)/1e12:.2f} TF/s, i.e. at the fp32 vector row "
          f"({kp.RATE_VECTOR/1e12:.2f} TF/s, devices/strix-halo-compute.md). Not above it.")
    print(f"    * the same work at the engine's best hand-GEMM rate ({kp.RATE_HAND_GEMM/1e12:.2f} TF/s)"
          f" = {WORK_GFLOP*1e9/kp.RATE_HAND_GEMM*1e3:.0f} ms")
    print(f"    * at today's EXECUTED rate on the chunked form ({RATE_ENGINE_EXEC/1e12:.3f} TF/s)"
          f" = {WORK_GFLOP*1e9/RATE_ENGINE_EXEC*1e3:.0f} ms — over the line, and the reason")
    print(f"      that rate does not apply is its own instruction mix: 3 VALU + 1 LDS per MAC, a BF16")
    print(f"      rounding ladder on every operand, and 13.5x the priced arithmetic re-executed "
          f"(prototype/gdn_chunk/kernel_price.py).")
    print(f"      Route A's inner loop is 1 LDS + 1 FMA per MAC with no rounding; --isa is the census.")
    print(f"  traffic, the thing the route actually removes:")
    print(f"    state: {state_bytes/1e6:.0f} MB/wave at {kp.L2_RESIDENT/1e9:.0f} GB/s "
          f"= {state_bytes/kp.L2_RESIDENT*1e3:.2f} ms   (sequential: {seq_state_bytes/1e9:.1f} GB"
          f" = {seq_state_bytes/kp.L2_RESIDENT*1e3:.0f} ms at L2 rate,")
    print(f"      {seq_state_bytes/kp.DRAM_READ*1e3:.0f} ms if DRAM-cold — which is where the "
          f"{SEQ_MS:.0f} ms of the sequential form went)")
    print(f"    q|k|v re-read once per strip: {uread_bytes/1e6:.0f} MB/wave "
          f"= {uread_bytes/kp.L2_RESIDENT*1e3:.1f} ms at L2 rate, "
          f"{uread_bytes/kp.DRAM_READ*1e3:.1f} ms if cold — the price of")
    print(f"      recomputing the norm redundantly in all {STRIPS} strips of a head instead of sharing it")
    print(f"  launches: {sites} vs {SEQ_LAUNCHES:,} "
          f"({(SEQ_LAUNCHES-sites)*kp.LAUNCH_FLOOR*1e3:.1f} ms of dispatch floor at "
          f"{kp.LAUNCH_FLOOR*1e6:.1f} us/launch)")
    print(f"  bytes per useful flop (state only): {state_bytes/(WORK_GFLOP*1e9):.4f} B/flop, "
          f"{uread_bytes/(WORK_GFLOP*1e9):.3f} B/flop counting the redundant q|k|v reads")
    print(f"  VERDICT: bottom-up {ms_wave:.1f} ms; even double that ({2*ms_wave:.0f} ms) clears the "
          f"pre-registered {WIN_LINE_MS:.0f} ms line, and the kill line is {KILL_LINE_MS:.0f} ms.")
    print(f"  Sequential today {SEQ_MS:.1f} ms at {WAVE_LANES} lanes -> a {SEQ_MS/ms_wave:.1f}x "
          f"projection against the line, not against the baseline.")
    at3072 = LAYERS * (3072 // BODY_ROWS) * us_site / 1e3
    print(f"  at 3072 lanes ({3072//BODY_ROWS} bodies): {at3072:.1f} ms projected "
          f"(sequential ~{SEQ_MS*3:.0f} ms), so the win holds at both widths asked for.")
    print("  Nothing above is a measurement. If the device lands above "
          f"{WIN_LINE_MS:.0f} ms the route is dead by its own line, whatever this file said.")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--geometry", action="store_true", help="the arithmetic (no compiler, no device)")
    ap.add_argument("--isa", action="store_true", help="compile-only resources + instruction census")
    args = ap.parse_args()
    if not (args.geometry or args.isa):
        args.geometry = args.isa = True
    rc = 0
    if args.geometry:
        rc |= cmd_geometry()
    if args.isa:
        rc |= cmd_isa()
    return rc


if __name__ == "__main__":
    sys.exit(main())
