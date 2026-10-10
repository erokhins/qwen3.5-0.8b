#!/usr/bin/env python3
"""Weights report for a Qwen3.5-family safetensors checkpoint.

Prints exact tensor shapes and byte accounting straight from the safetensors
header (no torch needed). This is the authority for specification/01: shapes are
read, never assumed.

usage: tools/weights_report.py <path/to/*.safetensors> [filter-substring]
"""
import json
import re
import struct
import sys

ELEMENT_SIZE = {"BF16": 2, "F16": 2, "F32": 4, "F64": 8, "I64": 8, "I32": 4, "I8": 1, "U8": 1, "BOOL": 1}


def header(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        return json.loads(f.read(n).decode())


def size_of(v):
    """Byte size; safetensors data_offsets are already in bytes."""
    assert v["dtype"] in ELEMENT_SIZE, v["dtype"]
    nbytes = v["data_offsets"][1] - v["data_offsets"][0]
    numel = 1
    for d in v["shape"]:
        numel *= d
    assert nbytes == numel * ELEMENT_SIZE[v["dtype"]], (v["shape"], v["dtype"], nbytes)
    return nbytes


def main():
    path = sys.argv[1]
    flt = sys.argv[2] if len(sys.argv) > 2 else ""
    hdr = header(path)
    t = {k: v for k, v in hdr.items() if k != "__metadata__"}
    total = sum(size_of(v) for v in t.values())
    print(f"file={path}\ntensors={len(t)}  total={total/1e9:.4f} GB\n")

    def group(prefix):
        keys = sorted(k for k in t if k.startswith(prefix) and flt in k)
        b = sum(size_of(t[k]) for k in keys)
        print(f"### {prefix}  ({len(keys)} tensors, {b/1e9:.4f} GB)")
        for k in keys:
            v = t[k]
            print(f"  {k.replace(prefix, ''):<58} {v['dtype']:<5} {v['shape']}")
        print()
        return b

    lm = "model.language_model."
    lay = sorted({int(m.group(1)) for k in t if (m := re.search(r"language_model\.layers\.(\d+)\.", k))})
    lin = [i for i in lay if f"{lm}layers.{i}.linear_attn.A_log" in t]
    att = [i for i in lay if f"{lm}layers.{i}.self_attn.q_proj.weight" in t]
    print(f"layers {lay[0]}..{lay[-1]} = {len(lay)} | linear_attn {lin} | full_attn {att}\n")

    group(lm + "layers.0.")
    group(lm + "layers.3.")
    group("mtp.")
    group("model.language_model.embed_tokens")
    group("model.language_model.norm")

    per_lin = sum(size_of(v) for k, v in t.items() if f"{lm}layers.{lin[0]}." in k)
    per_att = sum(size_of(v) for k, v in t.items() if f"{lm}layers.{att[0]}." in k)
    vis = sum(size_of(v) for k, v in t.items() if k.startswith("model.visual."))
    emb = size_of(t[lm + "embed_tokens.weight"])
    mtp = sum(size_of(v) for k, v in t.items() if k.startswith("mtp."))
    lm_total = sum(size_of(v) for k, v in t.items()
                   if k.startswith(lm) and ".layers." not in k and "embed_tokens" not in k)
    print("--- summary")
    print(f"linear_attn layer bytes   {per_lin/1e6:8.2f} MB  x{len(lin)}  = {per_lin*len(lin)/1e9:.4f} GB")
    print(f"full_attn   layer bytes   {per_att/1e6:8.2f} MB  x{len(att)}  = {per_att*len(att)/1e9:.4f} GB")
    print(f"embed (tied, no lm_head)  {emb/1e6:8.2f} MB")
    print(f"lm top-level (final norm) {lm_total/1e6:8.2f} MB")
    print(f"mtp block                 {mtp/1e6:8.2f} MB")
    print(f"vision tower (unused)     {vis/1e6:8.2f} MB")
    print(f"text-only totals          {(total-vis)/1e9:.4f} GB (all tensors minus vision)")
    dtypes = {}
    for v in t.values():
        dtypes[v["dtype"]] = dtypes.get(v["dtype"], 0) + size_of(v)
    print("bytes by dtype:", {k: round(v / 1e6, 2) for k, v in sorted(dtypes.items())})
    f32 = sorted(k for k, v in t.items() if v["dtype"] == "F32")
    print(f"F32 tensors ({len(f32)}):", ", ".join(x.split('.')[-2] + '.' + x.split('.')[-1] for x in f32[:4]), "...")


main()
