#!/usr/bin/env python3
"""tabulate.py — turn the probe's raw IRP rows into honest rates.

Why this exists next to the binary's own columns: the binary prints `mac_per_lane_clk` equal to its
`icpcb` column, i.e. it reports WARP-INSTRUCTIONS per CU per clock and calls it MACs per lane per
clock. That is why `pct_of_ceiling` came out at 172 % and 248 % for instructions whose roof is 1 —
those are issue-slot rates (and 1.72 of a 2-slot-per-CU machine is a perfectly sane number), not
arithmetic rates. This script recomputes from first principles:

    MACs per launch   = warps x 32 lanes x iters x ops_per_iter x macs_per_lane_per_op
    MACs per lane/clk = MACs per launch / (best_ms/1000) / (2560 lanes x 2.894 GHz)
    inst per CU/clk   = warps x warp_inst_per_iter x iters / (40 CU x best_ms/1000 x 2.894 GHz)

and prints both, side by side, against the per-class roof, so a reader can see which of the two the
machine is actually near. Usage: tabulate.py RAWFILE [RAWFILE ...]
"""
import re
import sys

LANES, CLK, CU = 2560, 2.894e9, 40
SIMD_PER_CU = 2          # rocminfo: SIMDs per CU = 2 on gfx1151, so 2.0 is the issue ceiling
WAVE = 32

# MACs per lane per clock that one instruction of each class can retire at most
ROOF = {
    "irp_f32_8": 1, "irp_mix_8": 1, "irp_fma_1": 1, "irp_fma_4": 1, "irp_fma_8": 1,
    "irp_fma_16": 1, "irp_fma_32": 1, "irp_serial_8": 1, "irp_serial_32": 1,
    "irp_cvt_8": 0, "irp_cvt_16": 0,
    "irp_pkf16_8": 2, "irp_d2f16_8": 2, "irp_d2bf16_8": 2,
    "irp_d4i8_8": 4, "irp_u4u8_8": 4, "irp_wmma_4": 4,
    "irp_d8i4_8": 8, "irp_u8u4_8": 8,
    "irp_kv_read": 0, "irp_lds_tile": 0,
}
CLASS = {1: "scalar f32 (1 MAC/lane/clk)", 2: "packed fp16 / dot2 (2)",
         4: "dot4-i8 / WMMA (4)", 8: "dot8-i4 (8)", 0: "not an arithmetic leg"}

ROW = re.compile(
    r"kernel=(?P<k>\S+).*?blocks=(?P<blocks>\d+) threads=(?P<thr>\d+) warps=(?P<warps>\d+) "
    r"iters=(?P<iters>\d+) warp_inst_per_iter=(?P<wipi>\d+) ops_per_iter=(?P<ops>\d+)\(isa\) "
    r"macs_per_lane_per_op=(?P<mplo>\d+).*?best_ms_per_launch=(?P<ms>[\d.]+)"
)
EXTRA = re.compile(r"(checksum|guards|suspect|bad_blocks)=(\S+)")


def main(paths):
    rows = {}
    for p in paths:
        for line in open(p, errors="replace"):
            m = ROW.search(line)
            if not m:
                continue
            g = m.groupdict()
            k = g.pop("k")
            d = {kk: (float(vv) if kk == "ms" else int(vv)) for kk, vv in g.items()}
            d["flags"] = dict(EXTRA.findall(line))
            # Keep the BEST RATE, not the smallest ms. Rows for one leg can carry different iters
            # (the WMMA leg was run at 500/2000/8000 to test stability), and per-launch overhead
            # makes a short run look faster per launch while its rate is LOWER: taking min-ms put
            # the WMMA leg at 52.6 TFLOP/s when the 8000-iteration run gives 55.0.
            macs = d["warps"] * WAVE * d["iters"] * d["ops"] * d["mplo"]
            d["rate"] = macs / (d["ms"] / 1e3)
            bad = d["flags"].get("checksum", "ok") != "ok" or int(d["flags"].get("bad_blocks", 0) or 0)
            prev = rows.get(k)
            if prev is None or d["rate"] > prev[0]["rate"]:
                rows[k] = [d, bad or (prev[1] if prev else False)]
            else:
                prev[1] = prev[1] or bad

    hdr = (f"{'leg':15s} {'iters':>7s} {'ms/launch':>10s} {'TMAC/s':>8s} {'T·FLOP/s':>9s} "
           f"{'MAC/lane/clk':>12s} {'%roof':>7s} {'witness':>8s}")
    print(hdr)
    print("-" * len(hdr))
    order = [k for k in ("irp_f32_8", "irp_mix_8", "irp_pkf16_8", "irp_d2f16_8", "irp_d2bf16_8",
                         "irp_d4i8_8", "irp_u4u8_8", "irp_d8i4_8", "irp_u8u4_8", "irp_wmma_4")
             if k in rows]
    order += [k for k in sorted(rows) if k not in order]
    for k in order:
        d, bad = rows[k]
        macs = d["warps"] * WAVE * d["iters"] * d["ops"] * d["mplo"]
        secs = d["ms"] / 1e3
        rate = d["rate"]
        mpl = rate / (LANES * CLK)
        roof = ROOF.get(k, 0)
        pct = f"{100*mpl/roof:6.1f}%" if roof else "     —"
        cls = CLASS.get(roof, "?")
        print(f"{k:15s} {d['iters']:7d} {d['ms']:10.4f} {rate/1e12:8.2f} {2*rate/1e12:9.2f} "
              f"{mpl:12.3f} {pct} {'FAILED' if bad else 'ok':>8s}")
        if roof:
            print(f"{'':15s} class: {cls};  ops/iter(isa)={d['ops']} MAC/op={d['mplo']} "
                  f"useful share of the billed loop = {d['ops']}/{d['wipi']} warp-inst")
    print("\nMAC/lane/clk is arithmetic: MACs per launch / best-time / (2560 lanes x 2.894 GHz).")
    print("Warp-instruction rates are deliberately NOT printed: several legs compute above 2.0")
    print("inst/CU/clk against a 2-slot machine (rocminfo SIMDs per CU = 2), because the census")
    print("counts a v_dual_* packet as 2 ops in its op column while the slot count is what a real")
    print("issue-width claim needs. Until that accounting is separated, only the arithmetic column")
    print("is quotable; see results.md.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:] or ["G_raw_rows.txt"]))
