"""G3 census aid: read the two §38 rocprofv3 kernel traces and print the launch geometry of the
GDN kernels on both engines, so the source-level launch config can be checked against what ran.

CPU-only: parses CSVs already on disk. No device work, no timing of our own; every ms figure it
prints is §38's trace re-printed, not a new measurement. The `wgs/CU` column divides by HIP's
`multiProcessorCount = 20`, which AGENTS.md records as halved against the architecture (KFD
`simd_count 80` / `simd_per_cu 2` = 40 CUs); the ratios are what matter and they are invariant.

    python3 prototype/gdn_walk/g3_trace_geom.py
"""
import csv
import collections

TRACES = {
    "llama": "tmp/gcc/llamaphase/llama_pp4096/pp4096_kernel_trace.csv",
    "ours": "tmp/gcc/llamaphase/our_pp4096/pp4096_kernel_trace.csv",
}
# the GDN kernel-name substrings to pull out of each trace
NEEDLES = {
    "llama": ("delta",),
    "ours": ("rec_loop", "gdn"),
}


def main():
    for engine, path in TRACES.items():
        rows = list(csv.DictReader(open(path)))
        agg = collections.defaultdict(lambda: [0, 0])
        for r in rows:
            name = r["Kernel_Name"]
            if not any(n in name for n in NEEDLES[engine]):
                continue
            key = (
                name[:78],
                int(r["Workgroup_Size_X"]), int(r["Workgroup_Size_Y"]), int(r["Workgroup_Size_Z"]),
                int(r["Grid_Size_X"]), int(r["Grid_Size_Y"]), int(r["Grid_Size_Z"]),
                int(r["VGPR_Count"]), int(r["LDS_Block_Size"]), int(r["Scratch_Size"]),
            )
            agg[key][0] += 1
            agg[key][1] += int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
        print(f"==== {engine}: {path}  ({len(rows)} kernel records) ====")
        print(f"{'n':>5} {'ms(total)':>10} {'wgsize xyz':>14} {'grid xyz (threads)':>22} "
              f"{'blocks':>7} {'wgs/CU':>7} {'vgpr':>5} {'lds':>6} {'scr':>5}  name")
        for k, v in sorted(agg.items(), key=lambda kv: -kv[1][1]):
            name, wx, wy, wz, gx, gy, gz, vgpr, lds, scr = k
            threads = wx * wy * wz
            blocks = (gx * gy * gz) // max(threads, 1)
            print(f"{v[0]:>5} {v[1] / 1e6:>10.2f} {f'{wx}x{wy}x{wz}':>14} "
                  f"{f'{gx}x{gy}x{gz}':>22} {blocks:>7} {blocks / 20:>7.2f} {vgpr:>5} {lds:>6} "
                  f"{scr:>5}  {name}")
        print()


if __name__ == "__main__":
    main()
