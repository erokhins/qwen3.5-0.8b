#!/usr/bin/env python3
"""DEAD END: LLVM fuses attn pass 1 and pass 2 into single back-edge blobs (one region was 837
instructions over 192 blocks), so loop-graph attribution cannot separate the two passes. Kept as the
record of that finding; `region_census.py` (source-line attribution) replaced it.

Dump the shipped attention wave's loop inventory: one row per back-edge loop, with the opcode
histogram needed to say which source construct it is. Hand assignment of loop -> region follows,
and every instruction must land in exactly one bucket (the rest prints as UNASSIGNED)."""
import importlib.util
import re
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


rec = _load("gdn_rec_isa_census", REPO / "prototype/gdn_rec_isa/isa_census.py")
slim = rec.slim
base = slim.base

td = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(tempfile.mkdtemp(prefix="aw-"))
if len(sys.argv) <= 1:
    subprocess.run(["hipcc", "-O3", "-gline-tables-only", "--offload-arch=gfx1151", "--save-temps",
                    "-I", str(REPO), "-I", str(REPO / "engine"), "-c",
                    str(REPO / "engine/kernels_attn.cpp"), "-o", str(td / "k.o")],
                   check=True, capture_output=True, text=True, cwd=td)
asm = next(td.glob("*-hip-amdgcn-amd-amdhsa-*.s"))
lines = Path(asm).read_text().splitlines()
kern = rec.split_kernels_local(lines)

FILE_RE = re.compile(r'^\s*\.file\s+(\d+)\s+"(.*)"')
LOC_RE = re.compile(r"^\s*\.loc\s+(\d+)\s+(\d+)\s+")
ft = {m.group(1): m.group(2).strip('"') for ln in lines if (m := FILE_RE.match(ln))}

for name in [k for k in kern if "attn_slim_wave" in k or "attn_tile_wave" in k]:
    body = kern[name]
    blocks = slim.split_blocks_full(base.annotate_loops(body) if hasattr(base, "annotate_loops") else body)
    live, dead = rec.executed_blocks(blocks)
    live = [b for b in live if not b["flow"]]
    pos = {base.norm_key(b["label"]): i for i, b in enumerate(live)}
    edges = []
    for i, b in enumerate(live):
        for _, ln in b["insts"]:
            m = re.match(r"s_cbranch_\w+\s+(\S+)", ln)
            if m and base.norm_key(m.group(1)) in pos and pos[base.norm_key(m.group(1))] <= i:
                edges.append((pos[base.norm_key(m.group(1))], i))
    owner = {}
    for h, e in edges:
        for i in range(h, e + 1):
            if i not in owner or h > owner[i][0]:
                owner[i] = (h, e)
    groups = {}
    for i, b in enumerate(live):
        if i in owner:
            groups.setdefault(owner[i], []).append(b)
    inside = set(owner)
    out_insts = [x for i, b in enumerate(live) if i not in inside for x in b["insts"]]
    print(f"===== {name}: {sum(len(b['insts']) for b in live)} lines in {len(live)} live blocks; "
          f"{len(out_insts)} lines OUTSIDE any loop")
    print(f"  outside-loop histogram: " + "  ".join(f"{k}={v}" for k, v in
          Counter(base.classify(m) for m, _ in out_insts).most_common()))
    rows = []
    for (h, e), mine in groups.items():
        insts = [x for b in mine for x in b["insts"]]
        hist = Counter(m for m, _ in insts)
        rows.append({"hdr": live[h]["label"], "n": len(insts), "bound": rec.derive_bound(mine),
                     "blocks": len(mine),
                     "sig": (hist["v_ds_bpermute_b32"] + hist["v_ds_swizzle_b32_b32"],
                             sum(v for k, v in hist.items() if "fma" in k),
                             hist["v_exp_f32_e32"], hist["v_max_f32_e32"],
                             sum(v for k, v in hist.items() if k.startswith("ds_load"))),
                     "hist": hist})
    for r in sorted(rows, key=lambda r: -r["n"]):
        p, f, ex, mx, lds = r["sig"]
        top = "  ".join(f"{k}={v}" for k, v in r["hist"].most_common(9))
        print(f"  {r['hdr']:10} n={r['n']:4} blocks={r['blocks']:2} bound={r['bound']} "
              f"| perm={p} fma={f} exp={ex} max={mx} ds_load={lds}")
        print(f"      {top}")
