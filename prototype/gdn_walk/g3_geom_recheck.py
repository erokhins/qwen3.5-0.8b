"""Independent re-read of the §38 trace on ALL THREE launch axes. Grid_Size_* in a rocprofv3 kernel_trace CSV
is the grid in THREADS, so blocks = grid / workgroup; getting that wrong is how §38 came to say "same
0.80 blocks/CU". Prints every distinct launch shape per kernel from the CSVs already on disk. No device work.
"""
import collections
import csv
import sys

WG = ("Workgroup_Size_X", "Workgroup_Size_Y", "Workgroup_Size_Z")
GR = ("Grid_Size_X", "Grid_Size_Y", "Grid_Size_Z")
SIMDS, CUS = 80, 40


def shapes(path, needle, top=4):
    out = collections.Counter()
    with open(path, newline="") as fh:
        for row in csv.DictReader(fh):
            name = row.get("Kernel_Name") or ""
            if needle in name:
                out[tuple(int(row.get(d) or 0) for d in WG + GR)] += 1
    for key, n in out.most_common(top):
        wg = key[:3]
        gr = key[3:]
        threads = max(1, wg[0] * wg[1] * wg[2])
        tgrid = max(1, gr[0] * gr[1] * gr[2])
        blocks = max(1, tgrid // threads)
        wf = blocks * ((threads + 31) // 32)
        yield wg, gr, threads, blocks, wf, n


def main():
    for side, path, needles in (
        ("llama", "tmp/d52/llama_pp4096/pp4096_kernel_trace.csv", ["gated_delta_net"]),
        ("ours", "tmp/d52/engine_pp4096/pp4096_kernel_trace.csv",
         ["rec_loop", "gdn_", "attn_fattn_wave"]),
    ):
        print(f"--- {side}  ({path})")
        for needle in needles:
            rows = list(shapes(path, needle))
            tot = sum(r[5] for r in rows)
            if not tot:
                print(f"  {needle}: no rows")
                continue
            print(f"  {needle}: {tot} launches across {len(rows)} shape(s)")
            for wg, gr, thr, blocks, wf, n in rows:
                print(f"     wg={wg} grid_threads={gr} -> threads/blk={thr} blocks/launch={blocks} "
                      f"wf/launch={wf} wf/SIMD={wf / SIMDS:.2f} blocks/CU={blocks / CUS:.2f} launches={n}")
        print("  all kernel names seen (top 8 by launches):")
        names = collections.Counter()
        with open(path, newline="") as fh:
            for row in csv.DictReader(fh):
                names[(row.get("Kernel_Name") or "")[:46]] += 1
        for nm, n in names.most_common(8):
            print(f"     {n:6d}  {nm}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
