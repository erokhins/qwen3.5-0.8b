"""R6, answered from the W4 trace already on disk (no device run).

§26 counted the attention-family dispatches (138/pass) and their instruction share (prep+combine = 3.4 %
of VALU), but instructions are not wall. The engine trace from the same window carries Start/End
timestamps per dispatch, so the wall split can be summed directly: durations per kernel name, deduped
by Dispatch_Id, divided by the pass count derived from `attn_prep` against its known 48 sites/pass.
Overlapping is not a concern here: the engine submits on one stream, so attention-family kernels are
serialised and their durations sum.
"""
import csv
import collections
import glob

files = glob.glob("tmp/gcc/attn_attrib/w/trace_engine/*kernel_trace*.csv")
dur = collections.Counter()
cnt = collections.Counter()
seen = set()
for f in files:
    with open(f) as fh:
        for r in csv.DictReader(fh):
            name = (r.get("Kernel_Name") or "").split("(")[0]
            key = (r.get("Dispatch_Id"), name)
            if key in seen or "attn" not in name:
                continue
            seen.add(key)
            try:
                d = int(r["End_Timestamp"]) - int(r["Start_Timestamp"])
            except (KeyError, ValueError):
                continue
            dur[name] += d
            cnt[name] += 1

passes = cnt["attn_prep"] // 48 or 1
print(f"passes derived from attn_prep ({cnt['attn_prep']} / 48 sites) = {passes}")
tot = sum(dur.values())
print(f"attention-family wall, all passes: {tot/1e6:.2f} ms; per pass {tot/1e6/passes:.2f} ms\n")
for k in sorted(dur, key=lambda k: -dur[k]):
    print(f"   {k:<18} {dur[k]/1e6/passes:8.3f} ms/pass   {cnt[k]/passes:4.0f} launches   "
          f"{dur[k]/tot*100:5.1f} % of family")
