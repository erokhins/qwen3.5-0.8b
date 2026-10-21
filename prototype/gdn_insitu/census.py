#!/usr/bin/env python3
"""census.py — the arithmetic behind D4, recomputed instead of inherited. **CPU-only: NO TIMING TAKEN.**

Three sections, each of which is a division, not a measurement:

  --traffic   the `gdn` family's bytes per token, READ and WRITE separately, parsed out of
              `tools/step_traffic_model.py` (which parses `engine/state.h`; a shape retyped here is a
              guess that will be wrong the day the first file changes) and priced against the two
              ceilings this part actually has. This is where the route line's `0.17 ms floor` and its
              `40.3 MB` both fail to reproduce: the family is 43.067 MB/token and 47 % of it is writes.
  --launches  the same bytes booked PER LAUNCH against the GPU's own dispatch rate, because 54 of the
              72 launches in the segment cannot beat their own dispatch floor no matter how well they
              move bytes. This is the term the byte-only floor hides.
  --codeobj   the register/LDS reservation read from the code object by
              `prototype/attn_percost/a11_read_vgpr.py` (never `.vgpr_count`, never a driver query),
              with blocks/CU RE-DERIVED here at each kernel's real block size — the tool's own column
              assumes 96 threads, which is `attn`'s block, not `gdn`'s — and compared with the blocks
              the launcher can actually offer.

Every rate is labelled DEMANDED or DRAM. The plateau this part cannot exceed is ~237 GB/s of DRAM;
any rate above that is bytes the cache hierarchy absorbed, which is a different claim (AGENTS.md:
"a traffic claim is a property of the staging loops", and the `> 240 GB/s means skipped work` rule).
"""
from __future__ import annotations

import argparse
import re
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent.parent

# Quoted, with their files. Nothing here is measured by this script.
DRAM_PLATEAU_GBS = 237.0        # devices/strix-halo-memory-bandwidth.md, read ceiling
MIX_HARMONIC_GBS = 225.2        # tools/step_traffic_model.py no-cliff mix (reads 237 / writes 210)
WARM_32MIB_READ_GBS = 876.7     # devices/strix-halo-memory-bandwidth.md, 32 MiB tier (L2-resident)
WARM_32MIB_COPY_GBS = 325.7     # same table, copy row; the cliff to 239.4 lands at 64 MiB
DISPATCH_US = 1.85              # devices/strix-halo-memory-bandwidth.md, empty-kernel dispatch
LDS_PER_CU = 65536              # KFD lds_size_in_kb 64 (NOT a HIP capability query; see AGENTS.md)
REGS_PER_CU = 131072            # 65,536 per SIMD x 2 SIMD (prototype/attn_percost/a11_read_vgpr.py)
CU_COUNT = 40                   # rocminfo; hipGetDeviceProperties reports 20 on this SKU

# Threads per block for each GDN kernel, from the launch lines the rig re-derives and asserts against
# (prototype/gdn_insitu/gdn_insitu.hip.cpp geom_check, engine/kernels_gdn.cpp:40,52,72,272,485).
BLOCK_THREADS = {
    "rec_fused_kernel": 128, "rec_kernel": 128, "rec_unroll4_kernel": 128,
    "rec_reg16_kernel": 128, "conv_kernel": 256, "gates_kernel": 16, "gdn_gated_norm": 128,
}
# Blocks one launch offers, at the decode shape (layers 18, tokens 1, 16 value heads).
BLOCKS_OFFERED = {
    "rec_fused_kernel": 16, "rec_kernel": 16, "rec_unroll4_kernel": 16,
    "rec_reg16_kernel": 128, "conv_kernel": 6, "gates_kernel": 1, "gdn_gated_norm": 16,
}


def run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True).stdout


def traffic(args):
    """Parse step_traffic_model.py's own rows; do not retype a single shape."""
    out = run([sys.executable, str(ROOT / "tools/step_traffic_model.py")]
              + (["--lanes", str(args.lanes)] if args.lanes else []))
    rows = []
    for line in out.splitlines():
        f = line.split()
        if len(f) >= 3 and f[0] == "gdn":
            try:
                rows.append((f[1], float(f[2]), line.split("  ", 2)[-1].strip()))
            except ValueError:
                continue
    if not rows:
        sys.exit("census: step_traffic_model.py printed no `gdn` rows - the parser matched nothing, "
                 "which is a red result about this script, not about the engine.")
    rd = sum(b for d, b, _ in rows if d == "read")
    wr = sum(b for d, b, _ in rows if d == "write")
    print("== --traffic   (bytes/token parsed from tools/step_traffic_model.py; NO TIMING TAKEN)")
    for d, b, note in rows:
        print(f"   {d:<6} {b:8.3f} MB   {note[:96]}")
    print(f"   READ {rd:.3f} MB   WRITE {wr:.3f} MB   TOTAL {rd + wr:.3f} MB/token"
          f"   write share {100.0 * wr / (rd + wr):.1f} %")
    print(f"   route line assumed 40.3 MB; the family is {rd + wr:.3f} MB, and the missing 2.8 MB is")
    print("   the conv weight widened to F32 plus the conv window rows, which the §2 table did not book.")
    tot = rd + wr
    ms = lambda mb, gbs: mb * 1.048576 / gbs
    print(f"   floor at the DRAM plateau {DRAM_PLATEAU_GBS} GB/s      : {ms(tot, DRAM_PLATEAU_GBS):.4f} ms")
    print(f"   floor at the no-cliff mix  {MIX_HARMONIC_GBS} GB/s      : {ms(tot, MIX_HARMONIC_GBS):.4f} ms")
    print(f"   floor at the 32 MiB warm tier {WARM_32MIB_READ_GBS} GB/s: {ms(tot, WARM_32MIB_READ_GBS):.4f} ms"
          "   <- DEMANDED/L2 rate, not a DRAM rate; see --rate-prov")
    print(f"   quoted ledger floor for the family: 0.170 ms (bytes/237 with bytes=40.3 MB)")
    return rows


def launches(args, rows):
    """The same bytes booked PER LAUNCH against the dispatch floor.

    The bytes come from the rows --traffic parsed, bucketed by the note `tools/step_traffic_model.py`
    prints for each one, so no shape is retyped here. `gates` has no row of its own (its g/beta writes
    are booked with the conv tile), and that is the honest answer: its bytes are noise and its floor is
    the dispatch rate.
    """
    def bucket(note):
        n = note.lower()
        if "gated" in n:
            return "gated_norm"
        if "conv" in n or "u tile" in n:
            return "conv"
        if "state" in n or "rec_kernel" in n:
            return "rec"
        return "gates"

    per = {}
    for side, mb, note in rows:
        per.setdefault(bucket(note), [0.0, 0.0])
        per[bucket(note)][0 if side == "read" else 1] += mb
    print("\n== --launches   (per-launch floor: max(bytes/237, dispatch 1.85 us). NO TIMING TAKEN)")
    tot = 0.0
    for name in ("rec", "conv", "gates", "gated_norm"):
        rd, wr = per.get(name, (0.0, 0.0))
        n = 18                                     # one launch of each per layer per token
        bpl = (rd + wr) / n
        byte_us = bpl * 1.048576 / DRAM_PLATEAU_GBS * 1000.0
        floor_us = max(byte_us, DISPATCH_US)
        tot += floor_us * n
        binder = "bytes" if byte_us >= DISPATCH_US else "DISPATCH"
        print(f"   {name:<11} {n:>3}/step  r {rd:7.3f} + w {wr:6.3f} MB  = {bpl:7.4f} MB/launch"
              f"  byte floor {byte_us:7.3f} us  floor {floor_us:7.3f} us -> {floor_us * n:7.1f} us [{binder}]")
    print(f"   SUM over 72 launches: {tot / 1000:.4f} ms   (the route line's floor: 0.170 ms)")
    print(f"   prize = 0.4292 (wall share, family's own drain constant) - {tot / 1000:.3f}"
          f" = {0.4292 - tot / 1000:.3f} ms   (route line claims -0.23 ms)")
    print("   `gdn_gated_norm` is already priced launch-bound in the ledger at 2.967 us/launch, 42x its")
    print("   own byte floor: the same statement as the DISPATCH rows above, made once in passing.")
    return tot


def rate_prov(args):
    print("\n== --rate-prov   (what `~876 GB/s` is a rate OF; NO TIMING TAKEN)")
    print(f"   the 32 MiB warm tier measures {WARM_32MIB_READ_GBS} GB/s read / {WARM_32MIB_COPY_GBS} GB/s copy")
    print("   (devices/strix-halo-memory-bandwidth.md) and the same table cliffs to 239.4 GB/s at 64 MiB.")
    print("   The quoted 876 comes from prototype/gdn_decode/results.md dividing DEMANDED bytes")
    print("   (state read twice + written = 56.6 MB) by a 0.0646 ms burst of the standalone op, where")
    print("   the 18.9 MB state fits the big tier between touches. In the step it cannot: between two")
    print("   touches of one layer's slab the step demands ~1.5 GB of other traffic (the weights")
    print("   alone are 1.5457 GB/step), so the state is DRAM-resident and 237 / 225.2 are the honest")
    print("   denominators. Whether the recurrence ALSO demands the state twice is C4's first arm.")
    print("   shape mismatch in the same quote: that 0.0646 ms row is `--mode all` = 1 launch of 288")
    print("   blocks; decode ships 18 launches of 16 blocks. Different geometry, different instrument.")


def codeobj(args):
    sp = sorted((ROOT / "engine/build").glob("kernels_gdn-*gfx1151.s"))
    if not sp:
        sys.exit("census: no kernels_gdn code object under engine/build/ - build the engine first.")
    sp = sp[-1]
    print(f"\n== --codeobj   ({sp.name}; reservation via a11_read_vgpr.py, blocks/CU RE-DERIVED here at")
    print("             each kernel's real block size. NO TIMING TAKEN)")
    text = sp.read_text(errors="replace")
    # One field the tool does not print: whether the driver reserved a private (spill) segment.
    priv = {}
    for mangled, body in re.findall(r"\.amdhsa_kernel\s+(\S+)(.*?)\.end_amdhsa_kernel", text, re.S):
        pv = re.search(r"\.amdhsa_private_segment_fixed_size\s+(\d+)", body)
        en = re.search(r"\.amdhsa_enable_private_segment\s+(\d+)", body)
        priv[mangled] = (int(pv.group(1)) if pv else 0, int(en.group(1)) if en else 0)
    out = run([sys.executable, str(ROOT / "prototype/attn_percost/a11_read_vgpr.py"), str(sp), ""])
    # The demangled name is truncated to 46 columns by the tool, so its closing paren may be gone;
    # match the name at the left and anchor the five numeric fields at the right instead.
    row_re = re.compile(r"^(\w+)\(.*?\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(\d+)\s+(registers|LDS)")
    print(f"   {'kernel':<20} {'vGPU':>5} {'alloc':>6} {'LDS':>6} {'thr':>4} {'blk/CU':>7}"
          f" {'spill':>6} {'offered/CU':>11}   binder")
    seen = []
    for line in out.splitlines():
        m = row_re.search(line)
        if not m:
            continue
        name, vgpr, alloc, lds = m.group(1), int(m.group(2)), int(m.group(3)), int(m.group(4))
        key = next((k for k in BLOCK_THREADS if k == name), None)
        if key is None:
            continue
        thr = BLOCK_THREADS[key]
        by_reg = REGS_PER_CU // (alloc * thr)
        by_lds = (LDS_PER_CU // lds) if lds else 99
        blk = min(by_reg, by_lds)
        pv = max([v[0] for mangled, v in priv.items() if name in mangled] or [0])
        spill = "none" if pv == 0 else f"{pv}B"
        off = BLOCKS_OFFERED[key] / float(CU_COUNT)
        seen.append(key)
        print(f"   {key:<20} {vgpr:>5} {alloc:>6} {lds:>6} {thr:>4} {blk:>7} {spill:>6}"
              f" {off:>11.2f}   {m.group(7)} (regs {by_reg}, LDS {by_lds})")
    missing = [k for k in BLOCK_THREADS if k not in seen]
    if missing:
        sys.exit(f"census: no code-object row for {missing} - the parser matched nothing, which is a red")
    print(f"   Capacity is computed at {REGS_PER_CU} regs/CU (65,536 per SIMD) and {LDS_PER_CU} B LDS/CU from")
    print(f"   KFD; {CU_COUNT} CUs from rocminfo, NOT the 20 a HIP capability query reports on this SKU.")
    print("   `spill none` means .amdhsa_private_segment_fixed_size and .amdhsa_enable_private_segment are")
    print("   0: no spill, no alloca. The recurrence keeps its per-thread state in registers by")
    print("   construction, which is what G4's reg16 form is named for (D3's lesson: read the")
    print("   .amdhsa_kernel block, never .vgpr_count, never the driver's occupancy query).")
    print("   The offered column is the point: at T=1 the launcher offers a FRACTION of a block per CU")
    print("   (rec 16 blocks over 40 CUs), so residency cannot be the limiter however large the file is.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traffic", action="store_true")
    ap.add_argument("--launches", action="store_true")
    ap.add_argument("--rate-prov", action="store_true")
    ap.add_argument("--codeobj", action="store_true")
    ap.add_argument("--lanes", type=int, default=0)
    a = ap.parse_args()
    allsec = not (a.traffic or a.launches or a.rate_prov or a.codeobj)
    rows = []
    if allsec or a.traffic or a.launches:
        rows = traffic(a)
    if allsec or a.launches:
        launches(a, rows)
    if allsec or a.rate_prov:
        rate_prov(a)
    if allsec or a.codeobj:
        codeobj(a)
    print("\nNO TIMING TAKEN: every figure above is arithmetic on quoted inputs, named where they come from.")


if __name__ == "__main__":
    main()
