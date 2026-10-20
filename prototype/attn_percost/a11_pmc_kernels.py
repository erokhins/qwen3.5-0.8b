"""Per-kernel VGPR/LDS/time as the pmc dispatch records actually report them.

Why this exists: A11-attribution.md §1 carries two occupancy readings (static .vgpr_count 185 vs pmc
VGPR_Count 248) and which one is real decides whether the shipping kernel is LDS-bound or register-bound at
3 blocks/CU -- and therefore whether a 64-register saving buys residency on its own (A10-F16-IMPL.md D2).
The compile-time listing and the driver's allocated count are different numbers and only the dispatch record
says what the machine reserved.
"""
import csv
import sys
import collections

path = sys.argv[1]
want = sys.argv[2] if len(sys.argv) > 2 else "attn"
agg = collections.defaultdict(lambda: {"n": 0, "vgpr": collections.Counter(),
                                       "lds": collections.Counter(), "grid": set(), "ms": 0.0})
with open(path, newline="") as fh:
    seen = set()
    for r in csv.DictReader(fh):
        k = r.get("Kernel_Name", "")
        if want not in k:
            continue
        # One pmc row per (Dispatch_Id, Counter_Name), and the occupancy replays the whole wave several times,
        # so a raw sum is a small integer multiple of a pass. Dedup on Dispatch_Id; the replay count is printed
        # as n_rows/n_dispatches rather than assumed to be 1.
        key = (r.get("Dispatch_Id"), k)
        a = agg[k]
        a["rows"] = a.get("rows", 0) + 1
        if key in seen:
            a["dup"] = a.get("dup", 0) + 1
            continue
        seen.add(key)
        a["n"] += 1
        a["vgpr"][r.get("VGPR_Count", "?")] += 1
        a["lds"][r.get("LDS_Block_Size", "?")] += 1
        try:
            a["grid"].add(int(r.get("Grid_Size", 0)))
        except Exception:
            pass
        try:
            a["ms"] += (float(r["End_Timestamp"]) - float(r["Start_Timestamp"])) / 1e6
        except Exception:
            pass
for k, a in sorted(agg.items(), key=lambda kv: -kv[1]["ms"]):
    blocks = sum(g // 96 for g in a["grid"] if g)
    print(f"{k[:48]:48s} n={a['n']:4d} rows={a.get('rows',0):5d} dup={a.get('dup',0):5d} "
          f"sum={a['ms']:9.2f}ms blocks={blocks:7d}\n"
          f"    vgpr={dict(a['vgpr'])} lds={dict(a['lds'])}")
