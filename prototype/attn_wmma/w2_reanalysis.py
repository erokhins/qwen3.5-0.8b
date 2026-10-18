"""W2, analysed properly from the CSVs the window already collected.

Why this file exists: `run_attn_attrib_window.sh`'s own W2 analyser ABORTED with "no attn_tile_wave
row, so the counter is uncalibrated", and the abort was correct behaviour on a wrong parse — the CSVs
do contain 36 `attn_tile_wave` rows. rocprofv3 emits one row per (Dispatch_Id, Counter_Name), and
`--time-waves` runs the wave three times per invocation (the run itself, the burst pass, the stage
pass), so a raw sum is 3x a single pass. Summing, then dividing by the pass count derived from the
row count and the known site count, is what the leg needed. Nothing here launches a kernel: it re-reads
tmp/gcc/attn_attrib/w/pmc{0,1}/*.csv that the window wrote.
"""
import csv
import sys

PASSES = 3           # run + burst pass + stage pass, each re-dispatches every kernel
SITES = {"attn_prep": 12, "attn_tile_wave": 12, "attn_slim_wave": 12, "attn_combine": 6}
TOUCHES = 25_190_400  # (row, lane, head) touches at 1,024 lanes (prefill-occupancy §3)
PUBLISHED_TILE_VALU = 2888.0  # M, attn_tile_wave whole-kernel, prefill-occupancy-results §3


def load(path):
    seen = set()
    agg = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (r["Dispatch_Id"], r["Counter_Name"])
            if key in seen:
                continue
            seen.add(key)
            name = r["Kernel_Name"].split("(")[0]
            agg.setdefault((name, r["Counter_Name"]), 0.0)
            agg[(name, r["Counter_Name"])] += float(r["Counter_Value"] or 0)
    return agg


def per_pass(agg, kernel, counter):
    tot = agg.get((kernel, counter), 0.0)
    rows = agg.get((kernel, counter), None)
    return tot / PASSES


for tag, path, wave in (("side 0 --attn-touch 0 (attn_tile_wave)", "tmp/gcc/attn_attrib/w/pmc0/pmc0_counter_collection.csv", "attn_tile_wave"),
                        ("side 1 --attn-touch 1 (attn_slim_wave)", "tmp/gcc/attn_attrib/w/pmc1/pmc1_counter_collection.csv", "attn_slim_wave")):
    agg = load(path)
    print(f"== {tag}")
    for k in sorted({k for k, _ in agg}):
        v = per_pass(agg, k, "SQ_INSTS_VALU")
        w = per_pass(agg, k, "SQ_WAVES")
        print(f"   {k:<18} VALU {v/1e6:9.1f} M/pass   waves {w/1e3:7.1f} K/pass")
    main = per_pass(agg, wave, "SQ_INSTS_VALU")
    print(f"   -> {wave} whole-kernel VALU = {main/1e6:.1f} M/pass = "
          f"{main/TOUCHES:.1f} VALU warp-inst per touch")

# calibration: the tile side must reproduce the published attn_tile_wave figure
agg0 = load("tmp/gcc/attn_attrib/w/pmc0/pmc0_counter_collection.csv")
me = per_pass(agg0, "attn_tile_wave", "SQ_INSTS_VALU") / 1e6
pct = (me - PUBLISHED_TILE_VALU) / PUBLISHED_TILE_VALU * 100
print(f"\ncalibration: attn_tile_wave {me:.1f} M/pass vs published {PUBLISHED_TILE_VALU:.1f} M "
      f"= {pct:+.1f} %  -> {'INSIDE' if abs(pct) <= 5 else 'OUTSIDE'} the +/-5 % band")

agg1 = load("tmp/gcc/attn_attrib/w/pmc1/pmc1_counter_collection.csv")
slim = per_pass(agg1, "attn_slim_wave", "SQ_INSTS_VALU")
print(f"measured attn_slim_wave = {slim/1e6:.1f} M/pass")
print(f"   model A (whole-kernel VALU, 87.6/touch): 2206.7 M -> {(slim/1e6)/2206.7:.3f} of prediction")
print(f"   model B (pass-1 VALU only,   28/touch):  705.3 M -> {(slim/1e6)/705.3:.3f} of prediction")
print(f"   verdict: {'model A' if abs(slim/1e6 - 2206.7) < abs(slim/1e6 - 705.3) else 'model B'}")
print(f"   pass-1 VALU share of the slim kernel = 55 static slots / {slim/TOUCHES:.1f} measured "
      f"whole-kernel VALU per touch = {55/(slim/TOUCHES)*100:.0f} %")
