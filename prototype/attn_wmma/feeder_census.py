#!/usr/bin/env python3
"""feeder_census.py — read the WMMA feeder probe's ISA and price the fragment feed/store.

Why a script and not eyeballing: the numbers have to be reproducible by someone who is not here,
and the lane-movement question ("did the compiler shuffle lanes to satisfy the fragment map?") is
exactly the kind of claim that must be a count, not an impression.

Classification rules are the same family as `prototype/attn_valu_slim/isa_census.py` (slot-per-
instruction: one issue slot each; VMEM/LDS are slots too), with the shuffle/permute classes broken
out because they are the whole point of this probe.

COMPILE-ONLY: this reads a .s file. It never touches the GPU.

    usage: feeder_census.py <listing.s> [--csv]
"""

from __future__ import annotations

import argparse
import collections
import pathlib
import re
import sys

# (pattern, category). `*` = anything, `?` = up to the next token. Order matters: first match wins.
RULES = [
    # matrix unit
    ("v_wmma?", "MATRIX.wmma"),
    ("v_mfma?", "MATRIX.mfma"),
    # lane movement — the fragment-map tax. Three distinct hardware mechanisms, all counted.
    ("ds_swizzle_b32", "SHUFFLE.ds_swizzle"),
    ("ds_bpermute_b32", "SHUFFLE.ds_bpermute"),
    ("ds_permute_b32", "SHUFFLE.ds_permute"),
    ("v_mov_b32_dpp", "SHUFFLE.dpp"),
    ("v_permlane*", "SHUFFLE.permlane"),
    ("v_alignbyte_b32", "SHUFFLE.alignbyte"),
    ("v_pack_b32_?", "SHUFFLE.pack"),
    ("v_perm_b32", "SHUFFLE.perm"),
    # LDS operand feed / drain
    ("ds_load_b128", "LDS.load_b128"),
    ("ds_load_b64", "LDS.load_b64"),
    ("ds_load_b32", "LDS.load_b32"),
    ("ds_load_u16", "LDS.load_u16"),
    ("ds_load*", "LDS.load_other"),
    ("ds_store_b128", "LDS.store_b128"),
    ("ds_store_b64", "LDS.store_b64"),
    ("ds_store_b32", "LDS.store_b32"),
    ("ds_store_u16", "LDS.store_u16"),
    ("ds_store*", "LDS.store_other"),
    ("ds_read2*", "LDS.read2"),
    ("ds_write2*", "LDS.write2"),
    # global operand feed / drain
    ("global_load_b128", "VMEM.load_b128"),
    ("global_load*", "VMEM.load_other"),
    ("global_store_b128", "VMEM.store_b128"),
    ("global_store_b32", "VMEM.store_b32"),
    ("global_store*", "VMEM.store_other"),
    # the 16-B LDS staging the engine itself does (K/V row per lane)
    ("s_waitcnt", "WAIT"),
    ("s_barrier", "BARRIER"),
    # arithmetic the feed would otherwise need
    ("v_cvt_pk_bf16_bf16", "CVT.bf16_pack"),
    ("v_cvt_off_src_bf16", "CVT.bf16_round"),
    ("v_cvt_*", "CVT.other"),
    ("v_bfe_u32", "CVT.bitfield"),
    ("v_bfi_b32", "CVT.bitfield"),
    ("v_lshlrev_b32_e64", "ADDR.shl"),
    ("v_lshlrev_b32", "ADDR.shl"),
    ("v_lshrrev_b32", "ADDR.shr"),
    ("v_and_b32", "ADDR.and"),
    ("v_and_or_b32", "ADDR.and_or"),
    ("v_or_b32_e32", "ADDR.or"),
    ("v_add_co_u32", "ADDR.add"),
    ("v_add_co_ci_u32_e64", "ADDR.add_ci"),
    ("v_add_u32", "ADDR.add"),
    ("v_add3_u32", "ADDR.add"),
    ("v_sub_u32", "ADDR.sub"),
    ("v_mov_b32_e32", "MOV.reg"),
    ("v_dual_mov_b32", "MOV.reg"),
    ("v_mov_b32*", "MOV.reg"),
    ("s_mov_b32", "SALU.mov"),
    ("s_*", "SALU.other"),
    ("v_*", "VALU.other"),
]

RX = []
for pat, cat in RULES:
    rx = "^" + "".join(".*" if c == "*" else "[^,]*" if c == "?" else r"\s+" if c == " " else re.escape(c)
                       for c in pat)
    RX.append((re.compile(rx), cat))

META_RE = re.compile(r"^\s*\.(vgpr_count|sgpr_count|lds_size|accum_vgpr_count|vgpr_spill_count|"
                     r"sgpr_spill_count|wavefront_size)\s*:\s*(\d+)")
KERN_RE = re.compile(r"^(_?Z?\w+):(?:\s+;.*)?$")


def classify(mnemonic: str) -> str:
    for rx, cat in RX:
        if rx.match(mnemonic):
            return cat
    return "UNCLASSIFIED." + mnemonic


def split_kernels(text: str):
    """Yield (kernel_symbol, Counter[mnemonic], metadata dict)."""
    cur, body, meta = None, collections.Counter(), {}
    in_text = False
    for line in text.splitlines():
        if line.startswith("\t.text"):
            in_text = True
            continue
        m = META_RE.match(line)
        if m and cur:
            meta[m.group(1)] = int(m.group(2))
            continue
        k = re.match(r"^([A-Za-z_][\w.]*):(?:\s+;.*)?$", line)
        if k and in_text:
            if cur:
                yield cur, body, meta
            cur, body, meta = k.group(1), collections.Counter(), {}
            continue
        if not in_text or not cur:
            continue
        s = line.strip()
        if not s or s.startswith((".", ";", "@", "/")) or line.startswith(" "):
            continue
        if not line.startswith("\t"):          # a label or a directive at column 0
            continue
        body[s.split()[0]] += 1
    if cur:
        yield cur, body, meta


LANE_MOVEMENT = {c for _, c in RULES if c.startswith("SHUFFLE")}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("listing")
    args = ap.parse_args()
    text = pathlib.Path(args.listing).read_text()

    print(f"# feeder census of {args.listing}")
    print("# LANE-MOVEMENT categories are the fragment-map tax: they exist only because the")
    print("# hardware's operand layout differs from the layout the loads were written in.\n")
    for name, body, meta in split_kernels(text):
        total = sum(body.values())
        if total == 0:
            continue
        cats = collections.Counter()
        for mn, n in body.items():
            cats[classify(mn)] += n
        body = cats
        lane_mv = sum(v for k, v in body.items() if k in LANE_MOVEMENT)
        lds_ld = sum(v for k, v in body.items() if k.startswith("LDS.load"))
        lds_st = sum(v for k, v in body.items() if k.startswith("LDS.store"))
        wmma = body.get("MATRIX.wmma", 0)
        print(f"== {name}")
        print(f"   {total} executed instructions | MATRIX.wmma={wmma} | "
              f"lane-movement={lane_mv} | LDS.load={lds_ld} LDS.store={lds_st} | "
              + " ".join(f"{k}={v}" for k, v in sorted(meta.items())))
        for cat, n in body.most_common():
            print(f"     {cat:26} {n}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
