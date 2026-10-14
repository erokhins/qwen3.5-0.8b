#!/usr/bin/env python3
"""wave_model.py — why the §05 attention wave runs at 0.236 TFLOP/s, computed from the
kernel's own constants and the ISA the compiler emits for it.

CPU-ONLY (`AGENTS.md`: no device work in a worker). Nothing here runs a kernel or takes a
timing: the two measured inputs are quoted from `bench/performance-status.md` and
`devices/`, and everything else is geometry read out of
`prototype/attn_decode/attn_decode_kernels.h` plus arithmetic on it. Run:

    .venv/bin/python prototype/attn_decode/wave_model.py geometry
    .venv/bin/python prototype/attn_decode/wave_model.py roof
    .venv/bin/python prototype/attn_decode/wave_model.py isa      # compile-only
    .venv/bin/python prototype/attn_decode/wave_model.py --selftest

The shape is the one the ledger prices: a 1024-token prompt at `--wave-rows 512`, i.e. two
bodies of 512 query rows through the six full-attention layers, library route OFF or ON —
`attn` is 109.2 ms of a 328.5 ms wave either way, because the wave never routes attention
through the library (`engine/kernels_attn.cpp`'s launch plan is the only route).

WHAT THE GEOMETRY IS, and the two index conventions everything else follows from:

  * A query ROW is a token: row `qrow` sits at position `pos0 + qrow` and may attend
    `lvalid + qrow` KV lanes (`lanes_of`, body 0 has `lvalid = 1`, body 1 `lvalid = 513`).
  * A (row, query-head) PAIR is a `attn_core` block at `split = 1`, two at `split = 2`.
    Body 0 is entirely one-pass (`lanes_last = 512 = MAXSC`); body 1 is entirely split,
    because every row there has 513..1024 lanes and 513 > MAXSC.
  * The unit the cost accounting uses is the (row, head, LANE) triple — a "touch": one score
    dot of `d_head` terms and one term of the P·V sum. The causal triangle says how many
    touches a body does, and one touch moves exactly 2 × d_head bytes (the K row for the
    score, the V row for the product) however many times that row was already touched.
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
KERNEL_H = ROOT / "prototype/attn_decode/attn_decode_kernels.h"

# ---- measured inputs. Every one is a citation, not an assumption. --------------------
MEASURED_ATTN_MS = 109.2       # bench/performance-status.md, route-(c) attribution table
MEASURED_ATTN_LAUNCHES = 12    # 6 layers x 2 bodies; `engine/launch_stats.h` counts sites
MEASURED_WAVE_MS = 328.5       # same table
DRAM_READ_GBS = 237.0          # devices/strix-halo-memory-bandwidth.md (pure read stream)
L2_SIZED_GBS = 918.0           # same file: 18 MiB working set — a CEILING, not a rate used below
VECTOR_FP32_TF = 2.93          # devices/strix-halo-compute.md: hipBLAS fp32 = the vector path
HAND_GEMM_TF = 0.85            # bench/performance-status.md: the engine's own hand GEMMs
LIB_MIX_M512_TF = 18.84        # devices/strix-halo-compute.md: 24-layer mix, 16-bit C, M=512
CU_COUNT = 40                  # devices/strix-halo-compute.md correction (rocminfo; HIP says 20)
WAVES_PER_CU = 32              # rocminfo: "Max Waves Per CU"
SIMD_ISSUE_SLOTS = 2           # rocminfo: "SIMDs per CU"; 32 waves / 16 waves-per-SIMD
CLOCK_GHZ = 2.206              # devices/strix-halo-compute.md: sysfs pp_dpm_sclk in normal use
# Static per-touch warp-instruction counts, read off the two loop bodies the `isa` census names by
# instruction signature (so a compiler change that renumbers blocks cannot silently repoint them).
# A "touch" is one (query row, query head, KV lane) triple.
#   score-tile loop — one K row load, the 5-step butterfly, two barriers: the smallest matching span
#     is 190 instructions per trip per warp, and the block's 8 warps cover the tile's 8 lanes,
#     so 190 warp-instr/touch.
#   softmax+PV loop — expf, the divide by `l`, one V load per lane, run by EVERY warp over ALL ns
#     lanes: smallest matching span 156 instructions for a 2x-unrolled body = 78 per lane per warp,
#     x 8 warps = 624 warp-instr/touch. Add the max and sum-of-exp loops and the one-pass form is
#     ~1030; the split form fuses them into the streaming loop at ~647.
# These are UPPER bounds on the dynamic count — a loop span includes peel and cold blocks — so the
# issue floor is "at most this", and it could be ~3x below. No CPU-side argument can narrow it; a
# dynamic instruction count is the only thing that settles it, and the one-launch handoff for that
# is in the bench doc.
WARP_INSTR_TOUCH = {"one_pass": 1030, "split": 647}


def kernel_consts() -> dict:
    """Every `#define NAME <int or constant expression>` the model needs, read from the kernel
    header rather than copied (`GQA` is spelled `(N_Q / N_KV)`, so values are evaluated in
    dependency order). A stale copy here would silently mis-price the geometry."""
    text = KERNEL_H.read_text()
    raw = {}
    for name, expr in re.findall(r"^#define\s+(\w+)\s+([^\n]*?)\s*(?://[^\n]*)?$", text, re.M):
        if re.fullmatch(r"[0-9A-Za-z_()/ +*.-]+", expr):
            raw[name] = expr
    vals: dict[str, int] = {}
    for _ in range(len(raw)):
        pending = [(k, v) for k, v in raw.items() if k not in vals]
        for k, expr in pending:
            try:
                vals[k] = int(eval(expr, {"__builtins__": {}}, dict(vals)))
            except Exception:
                continue
        if len(vals) == len(raw):
            break
    missing = [k for k in ("N_Q", "N_KV", "GQA", "DH", "THREADS", "TILE", "MAXSC") if k not in vals]
    if missing:
        raise SystemExit(f"{KERNEL_H.name}: cannot read {missing} — the model's copy is stale")
    vals["MAXW"] = vals["THREADS"] // 32                  # warps per block, wave32
    vals["lanes_per_tile"] = min(vals["TILE"], vals["MAXW"])   # `tile = std::min(TILE, nw)`
    return vals


def kernel_int(name: str) -> int:
    return kernel_consts()[name]


def wave_geometry(tokens: int = 1024, body_rows: int = 512, layers: int = 6,
                  lvalid0: int = 1) -> dict:
    """The launch plan and the touch count for one prompt wave, body by body.

    `lvalid0 = 1` is `decode_step.cpp`'s `a.lvalid = pos0 + 1`: row 0 attends its own lane.
    """
    c = kernel_consts()
    bodies = []
    for pos0 in range(0, tokens, body_rows):
        rows = min(body_rows, tokens - pos0)
        lvalid = pos0 + lvalid0
        lanes = [lvalid + q for q in range(rows)]          # row q's causal width
        split = (max(lanes) + c["MAXSC"] - 1) // c["MAXSC"]  # split_for(lanes_last)
        one_pass_rows = sum(1 for n in lanes if n <= c["MAXSC"])
        touches = sum(n * c["N_Q"] for n in lanes)
        # The block's serial inner loop: a (row, head, part) block walks its own slice of the
        # row's lanes 8 at a time, two whole-block barriers per step.
        per_row_tiles = [((n + split - 1) // split + c["lanes_per_tile"] - 1) // c["lanes_per_tile"]
                         for n in lanes]
        tile_iters = sum(per_row_tiles) * c["N_Q"] * split
        touches_split = sum(n * c["N_Q"] for n in lanes if n > c["MAXSC"])
        touches_one = touches - touches_split
        bodies.append({
            "pos0": pos0, "rows": rows, "lvalid": lvalid,
            "lanes_min": min(lanes), "lanes_max": max(lanes),
            "split": split, "one_pass_rows": one_pass_rows, "split_rows": rows - one_pass_rows,
            "touches": touches, "touches_one_pass": touches_one, "touches_split": touches_split,
            "tile_iters": tile_iters,
            "blocks_prep": (c["N_Q"] + c["N_KV"]) * rows,            "blocks_core": c["N_Q"] * split * rows,
            "blocks_combine": (c["N_Q"] * rows) if split > 1 else 0,
            "warps_core": c["N_Q"] * split * rows * c["MAXW"],
            "bytes_kv": touches * 2 * c["DH"] * 2,   # K row + V row, BF16, per touch
            "flops": touches * 4 * c["DH"],          # 2 dots x d_head MACs x 2 flops
            "tiles_mean_per_block": sum(per_row_tiles) / len(per_row_tiles),
            "tiles_max_per_block": max(per_row_tiles),
        })
    tot = {
        "layers": layers,
        "touches": sum(b["touches"] for b in bodies) * layers,
        "flops": sum(b["flops"] for b in bodies) * layers,
        "bytes_kv": sum(b["bytes_kv"] for b in bodies) * layers,
        "lane_tiles": sum(b["tile_iters"] for b in bodies) * layers,
        "blocks_core": sum(b["blocks_core"] for b in bodies) * layers,
        "blocks_all": sum(b["blocks_prep"] + b["blocks_core"] + b["blocks_combine"] for b in bodies) * layers,
        "warp_instr": sum(WARP_INSTR_TOUCH["one_pass"] * b["touches_one_pass"] +
                          WARP_INSTR_TOUCH["split"] * b["touches_split"] for b in bodies) * layers,
    }
    # The distinct bytes the wave could not avoid reading: each K and V row of each layer, once.
    tot["bytes_compulsory"] = layers * 2 * c["N_KV"] * tokens * c["DH"] * 2
    return {"bodies": bodies, "totals": tot, "consts": c}


def fmt_bytes(n: float) -> str:
    for unit, div in (("GB", 1e9), ("MB", 1e6), ("kB", 1e3)):
        if n >= div:
            return f"{n / div:.2f} {unit}"
    return f"{n:.0f} B"


def cmd_geometry(args) -> None:
    g = wave_geometry(args.tokens, args.wave_rows, args.layers)
    c = g["consts"]
    print(f"constants from {KERNEL_H.name}: N_Q={c['N_Q']} N_KV={c['N_KV']} GQA={c['GQA']} "
          f"DH={c['DH']} THREADS={c['THREADS']} (= {c['MAXW']} warps) TILE={c['TILE']} "
          f"MAXSC={c['MAXSC']} lanes/tile={c['lanes_per_tile']}")
    for i, b in enumerate(g["bodies"]):
        print(f"\nbody {i}: rows={b['rows']} pos0={b['pos0']} lvalid={b['lvalid']} "
              f"lanes {b['lanes_min']}..{b['lanes_max']}  split={b['split']}")
        print(f"  one-pass rows {b['one_pass_rows']} (lanes <= MAXSC), split rows {b['split_rows']}")
        print(f"  blocks: prep={b['blocks_prep']} core={b['blocks_core']} "
              f"combine={b['blocks_combine']}  = {c['N_Q']} heads x split x rows, "
              f"{c['MAXW']} warps each")
        print(f"  touches (row x head x lane) = {b['touches']:,}   "
              f"one-pass {b['touches_one_pass']:,}  split {b['touches_split']:,}")
        print(f"  serial inner loop: {b['tile_iters']:,} lane-tile steps in flight per body, "
              f"mean {b['tiles_mean_per_block']:.1f} and max {b['tiles_max_per_block']} per block"
              f"  (2 whole-block barriers each)")
        print(f"  flops {b['flops'] / 1e9:.2f} GFLOP   K+V bytes read {fmt_bytes(b['bytes_kv'])}"
              f"   = {b['bytes_kv'] / b['flops']:.3f} byte/flop")
    t = g["totals"]
    print(f"\nper {args.tokens}-token wave x {args.layers} layers:")
    print(f"  touches      {t['touches']:,}")
    print(f"  flops        {t['flops'] / 1e9:.2f} GFLOP   (ledger quotes 25.79 GFLOP)")
    print(f"  K+V read     {fmt_bytes(t['bytes_kv'])} for {fmt_bytes(t['bytes_compulsory'])} of "
          f"distinct KV (all {args.layers} layers, {args.tokens} tokens) = "
          f"{t['bytes_kv'] / t['bytes_compulsory']:.0f}x re-read")
    print(f"  byte/flop    {t['bytes_kv'] / t['flops']:.3f}")
    print(f"  attn_core blocks {t['blocks_core']:,} over {CU_COUNT} CUs = "
          f"{t['blocks_core'] / CU_COUNT:.0f} blocks/CU; all attn blocks "
          f"{t['blocks_all']:,}")
    per_body = sum(b["blocks_core"] for b in g["bodies"]) / len(g["bodies"])
    print(f"  in flight at once: {per_body:.0f} core blocks per body against "
          f"{CU_COUNT} x {WAVES_PER_CU // c['MAXW']} = "
          f"{CU_COUNT * (WAVES_PER_CU // c['MAXW'])} blocks the CUs could hold at "
          f"{WAVES_PER_CU} waves/CU and {c['MAXW']} warps/block — so residency per CU, not "
          f"supply of blocks, is the only question left")


def cmd_roof(args) -> None:
    g = wave_geometry(args.tokens, args.wave_rows, args.layers)
    t = g["totals"]
    gflops = t["flops"] / 1e9
    ms = MEASURED_ATTN_MS
    print(f"measured: attn {ms:.1f} ms over {MEASURED_ATTN_LAUNCHES} sites "
          f"({ms / MEASURED_ATTN_LAUNCHES:.2f} ms per 512-row body), "
          f"{100 * ms / MEASURED_WAVE_MS:.1f} % of a {MEASURED_WAVE_MS:.1f} ms wave")
    print(f"arithmetic: {gflops:.2f} GFLOP/wave -> {gflops / ms * 1000 / 1e3:.3f} TFLOP/s "
          f"= the ledger's 0.236")
    print(f"traffic:    {fmt_bytes(t['bytes_kv'])} of K+V per wave -> "
          f"{t['bytes_kv'] / 1e9 / (ms / 1000) / 1e3:.3f} TB/s "
          f"({t['bytes_kv'] / 1e9 / (ms / 1000):.0f} GB/s) at the same arithmetic")
    print(f"issue:      {t['warp_instr'] / 1e9:.2f} G warp-instructions counted STATICALLY per "
          f"touch from the ISA loop bodies (an upper bound: peel and cold blocks are inside the "
          f"spans), so the issue floor is at most "
          f"{t['warp_instr'] / (CU_COUNT * SIMD_ISSUE_SLOTS * CLOCK_GHZ * 1e9) * 1000:.0f} ms "
          f"on {CU_COUNT} CUs x {SIMD_ISSUE_SLOTS} issue slots x {CLOCK_GHZ} GHz at 1 IPC/SIMD, "
          f"and could be ~3x below that")
    print("\nrate menu — what the wave's arithmetic costs at each MEASURED rate, and what that")
    print("means for the traffic term. A row that is BELOW the measured line is headroom;")
    print("a row above it is a wall the arithmetic cannot get under.")
    rows = [
        ("measured today", ms),
        ("same touches at the 237 GB/s read ceiling",
         t["bytes_kv"] / 1e9 / DRAM_READ_GBS * 1000),
        ("same touches if EVERY re-read hit cache (918 GB/s ceiling)",
         t["bytes_kv"] / 1e9 / L2_SIZED_GBS * 1000),
        ("issue floor, static instruction count (upper bound)",
         t["warp_instr"] / (CU_COUNT * SIMD_ISSUE_SLOTS * CLOCK_GHZ * 1e9) * 1000),
        ("issue floor, if that count is 3x pessimistic",
         t["warp_instr"] * 0.34 / (CU_COUNT * SIMD_ISSUE_SLOTS * CLOCK_GHZ * 1e9) * 1000),
        (f"arithmetic at the hand-GEMM rate ({HAND_GEMM_TF} TF/s)", gflops / HAND_GEMM_TF),
        (f"arithmetic at the fp32 vector rate ({VECTOR_FP32_TF} TF/s)", gflops / VECTOR_FP32_TF),
        (f"arithmetic at the hipBLAS wave mix ({LIB_MIX_M512_TF} TF/s, M=512)",
         gflops / LIB_MIX_M512_TF),
    ]
    for i, (label, val) in enumerate(rows):
        note = ("" if i == 0 else "AT the measured line" if abs(val - ms) / ms < 0.10
                else "above it — a wall" if val > ms else "below it — reachable")
        print(f"  {label:<54} {val:8.1f} ms   {note}")
    print("\nRow 2 equaling row 1 is the finding: at 1 byte/flop the kernel is priced identically by")
    print("its flop count and by its byte count at the DRAM read ceiling. Row 3 is the honest caveat")
    print("— IF the re-reads hit cache, the same bytes cost 28 ms, so which of the two walls this is")
    print("depends on a cache question the paper cannot answer (see the bench doc's open items). The")
    print("fix is the same under both: stop re-reading.")
    print("\nTraffic after tiling: a block that serves G query heads of one KV group and M query")
    print("rows per K/V read divides the bytes by G*M and changes no output element.")
    c = g["consts"]
    for G, M in ((1, 1), (c["GQA"], 1), (c["GQA"], 8), (c["GQA"], 16), (1, 16)):
        by = t["bytes_kv"] / (G * M)
        print(f"  G={G} heads x M={M} rows per read: {fmt_bytes(by):>10} -> "
              f"{by / 1e9 / DRAM_READ_GBS * 1000:6.1f} ms of traffic at 237 GB/s; "
              f"arithmetic floor then {gflops / VECTOR_FP32_TF:5.1f} ms (vector) / "
              f"{gflops / LIB_MIX_M512_TF:4.1f} ms (matrix)")
    print(f"\nThe traffic term stops binding once it is under the arithmetic term, i.e. around")
    print(f"G*M = {(t['bytes_kv'] / 1e9 / DRAM_READ_GBS * 1000) / (gflops / VECTOR_FP32_TF):.0f}x "
          "re-use on the vector path. Past that the")
    print("win is limited by how fast the box does 16-bit MACs, not by where they came from.")


def cmd_isa(args) -> None:
    """Compile-only resource report and loop census. No device is touched, no timing is taken;
    `hipcc -c` plus `--save-temps` is a cross-compile for gfx1151 and nothing else."""
    tmp = Path("/tmp/attn_wave_model_isa")
    tmp.mkdir(exist_ok=True)
    (tmp / "one.cpp").write_text('#include "%s"\n' % KERNEL_H)
    r = subprocess.run(["hipcc", "-O3", "--offload-arch=gfx1151",
                        "-Rpass-analysis=kernel-resource-usage", "-c", str(tmp / "one.cpp"),
                        "-o", str(tmp / "one.o"), "--save-temps"],
                       capture_output=True, text=True, cwd=tmp)
    if r.returncode:
        raise SystemExit("COMPILE FAILED (compile-only; no device touched):\n" + r.stderr[-2000:])
    for line in r.stderr.splitlines():
        if any(k in line for k in ("Function Name", "VGPRs", "SGPRs", "LDS Size", "ScratchSize",
                                   "Occupancy")):
            print("  " + line.strip())
    sfiles = list(tmp.glob("*gfx1151.s"))
    if not sfiles:
        raise SystemExit("no assembly emitted to census: " + r.stderr[-500:])
    text = sfiles[0].read_text()
    for name in ("attn_core", "attn_prep", "attn_combine"):
        m = re.search(r"\.amdhsa_kernel _Z%d%s" % (len(name), name), text)
        if not m:
            continue
        seg = text[m.end():m.end() + 1200]
        gsz = re.search(r"\.amdhsa_group_segment_fixed_size (\d+)", seg)
        pss = re.search(r"private_seg_size, (\d+)", seg) or re.search(r"private_segment_size (\d+)", seg)
        print(f"  {name}: group_segment_fixed_size = {gsz.group(1) if gsz else '?'} B/block"
              + (f", private_seg_size = {pss.group(1)} B/lane" if pss else ""))
    c = kernel_consts()
    print(f"  the source's own __shared__ arrays sum to "
          f"{(4 * c['DH'] + 8 + c['MAXSC']) * 4} B: q_rot 4 KB + sq/sk/sv 3 KB + sc "
          f"{c['MAXSC'] * 4} B. Whatever the descriptor reserves is what limits residency.")
    loop_census(text, "attn_core")


FEATURE_PATTERNS = {
    "bexp": r"v_exp", "div": r"v_rcp|v_div",
    "ldg128": r"global_load_b128|global_load_u128",
    "ldg16": r"global_load_u16|global_load_ushort",
    "shfl": r"ds_bpermute|ds_swizzle", "bar": r"s_barrier",
    "lds": r"ds_load|ds_store", "fma": r"v_fma_f32|v_mad_f32|v_add_f32|v_fmac",
    "store": r"global_store",
}


def loop_census(text: str, kernel: str) -> None:
    """Every backward branch in the kernel, with the instruction count of the span it closes.
    A span is an UPPER bound on the loop's dynamic cost: it contains peel code and cold blocks."""
    start = text.index(f"_Z{len(kernel)}{kernel}")
    body = text[start:text.index(".Lfunc_end0:", start)].splitlines()
    order, blocks, cur = [], {}, None
    for line in body:
        s = line.strip()
        m = re.match(r"^(\.LBB[0-9_]*):", s)
        if m:
            cur = m.group(1)
            order.append(cur)
            blocks[cur] = []
            continue
        if cur and s and not s.startswith((".", ";", "#", "/")):
            blocks[cur].append(s)
    pos = {b: i for i, b in enumerate(order)}
    print(f"  {kernel}: {len(order)} basic blocks, "
          f"{sum(len(v) for v in blocks.values())} instruction lines")
    loops = []
    for i, b in enumerate(order):
        blob = " ".join(blocks[b])
        for tgt in re.findall(r"s_cbranch_\w+\s+(\S+)", blob) + re.findall(r"s_branch\s+(\S+)", blob):
            if tgt in pos and pos[tgt] < i and i - pos[tgt] >= 1:
                span = order[pos[tgt]:i + 1]
                blob2 = " ".join(" ".join(blocks[x]) for x in span)
                n = sum(len(blocks[x]) for x in span)
                feats = {k: len(re.findall(p, blob2)) for k, p in FEATURE_PATTERNS.items()}
                loops.append((n, f"{order[pos[tgt]]}..{b}", {k: v for k, v in feats.items() if v}))
    # Named by instruction signature rather than by label, so a compiler change that renumbers
    # blocks cannot silently repoint the per-touch instruction counts in WARP_INSTR_TOUCH.
    NAMED = {
        "score-tile loop (one warp per lane: 1 K row, 5-step butterfly, 2 barriers)":
            lambda f: f.get("ldg128", 0) >= 1 and f.get("shfl", 0) >= 5 and f.get("bar", 0) >= 2,
        "softmax+PV loop (every warp, over all lanes: expf, div, one V load per lane)":
            lambda f: f.get("bexp", 0) >= 1 and f.get("ldg16", 0) >= 1 and f.get("div", 0) >= 1,
    }
    for what, pred in NAMED.items():
        hits = sorted([l for l in loops if pred(l[2])])
        print(f"    {what}")
        for n, span, feats in hits[:2]:
            print(f"      smallest matching span {span:<22} {n:>5} instrs  {feats}")
        if not hits:
            print("      NONE — WARP_INSTR_TOUCH is now unsupported by this ISA, re-derive it")


def cmd_selftest(args) -> None:
    g = wave_geometry()
    t = g["totals"]
    checks = [
        # The geometry must reproduce the one number the ledger already has, independently.
        ("flops reproduce the ledger's 25.79 GFLOP", abs(t["flops"] / 1e9 - 25.79) < 0.02),
        # Body 0 is one-pass, body 1 is split — the split the launcher derives, not a guess.
        ("body 0 split=1, body 1 split=2",
         g["bodies"][0]["split"] == 1 and g["bodies"][1]["split"] == 2),
        ("body 1 has no one-pass rows", g["bodies"][1]["touches_one_pass"] == 0),
        # 1 byte/flop is the whole argument; if a model edit breaks it, the model is wrong.
        ("exactly 1 byte of KV per flop", abs(t["bytes_kv"] / t["flops"] - 1.0) < 1e-9),
        # The traffic/DRAM figure must reproduce the measured line within a few percent.
        ("traffic / 237 GB/s reproduces 109.2 ms",
         abs(t["bytes_kv"] / 1e9 / DRAM_READ_GBS * 1000 - MEASURED_ATTN_MS) < 5.0),
        # The static instruction count must NOT be separable from the measurement on paper:
        # that is the honest statement of the ambiguity, and the reason for the rocprofiler
        # handoff in the bench doc.
        ("static issue floor is within 2x of the measured line",
         0.5 * MEASURED_ATTN_MS < t["warp_instr"] / (CU_COUNT * SIMD_ISSUE_SLOTS * CLOCK_GHZ * 1e9) * 1000
         < 2.0 * MEASURED_ATTN_MS),
    ]
    bad = 0
    for name, ok in checks:
        print(f"  {'ok  ' if ok else 'FAIL'}  {name}")
        bad += 0 if ok else 1
    print(f"selftest: {len(checks) - bad}/{len(checks)} passed")
    sys.exit(1 if bad else 0)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cmd", nargs="?", default="geometry",
                    choices=["geometry", "roof", "isa", "selftest"])
    ap.add_argument("--tokens", type=int, default=1024)
    ap.add_argument("--wave-rows", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    args = ap.parse_args()
    if args.cmd == "selftest":
        cmd_selftest(args)
    {"geometry": cmd_geometry, "roof": cmd_roof, "isa": cmd_isa}[args.cmd](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
