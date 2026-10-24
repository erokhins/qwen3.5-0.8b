#!/usr/bin/env python3
"""How many device bytes does the FIRST wave of a fresh process first-touch?

Why this exists. §49's ~65-71 ms per-prompt constant has been read as "first-touch of
fresh pages at ~35 GB/s" (`devices/strix-halo-unified-memory.md`). To use that reading
you need the byte count, and the byte count is NOT the checkpoint: `WeightStore::load()`
runs a warm pass over every resident weight page (`engine/loader.cpp:626-644`) inside
`load()`, which `decode_run` brackets separately (`engine/decode_run.cpp:1800-1795`) and
never adds to `RESULT ms_total`. The bytes that ARE cold at the first wave are the
scratch, the KV, and the hipBLASLt workspace. This script sums them from the same
constants `allocate_scratch` uses, so the page-warming hypothesis has a number attached.

Read-only arithmetic on constants; touches no device and takes no timing. The rates are
quoted, not measured here, and are printed with their source.

    python3 prototype/perprompt/a7_scratch_bytes.py
"""

import argparse

# engine/state.h:41-64 — the model's own constants.
NUM_LAYERS = 24
NUM_GDN_LAYERS = 18
NUM_ATTN_LAYERS = 6
NUM_VALUE_HEADS = 16
DK, DV = 128, 128
CONV_CHANNELS, CONV_KERNEL = 6144, 4
CONV_WINDOW = CONV_KERNEL - 1
NUM_KV_HEADS = 2
NUM_QUERY_HEADS = 8
HEAD_DIM = 256
HIDDEN = 1024
INTERMEDIATE = 3584
VOCAB = 248320
VALUE_DIM = NUM_VALUE_HEADS * DV          # 2048
QKV_ROWS = 3 * VALUE_DIM                  # 6144
QWIDTH = NUM_QUERY_HEADS * HEAD_DIM       # 2048
QPROJ_ROWS = 2 * QWIDTH                   # 4096
KVPROJ_ROWS = NUM_KV_HEADS * HEAD_DIM     # 512
GDN_CHUNK_TOKENS = 64                     # engine/kernels.h:504
EMBED_ID_SLOTS = 8                        # engine/decode_step.h:274
WORKSPACE_BYTES = 128 * 1024 * 1024       # kernels_gemm.cpp: workspace_mib=128

# a_part's size is computed by `attn_part_floats()` inside kernels_attn.cpp; the source
# comment states the value at the 512-row ceiling, so quote it rather than re-derive the
# split-K geometry here (re-deriving it is a different tool's job and would be a second
# spelling of the same layout — the §20.1 trap).
A_PART_BYTES = int(3.38e7)   # "a_part's 33.8 MB", engine/decode_step.cpp:302

u16 = lambda n: 2 * n
f32 = lambda n: 4 * n


def scratch_bytes(rows: int) -> list[tuple[str, int]]:
    """One entry per `alloc<>` in `allocate_scratch` (engine/decode_step.cpp:224-318)."""
    h, i = rows * HIDDEN, rows * INTERMEDIATE
    qkv, vdim = rows * QKV_ROWS, rows * VALUE_DIM
    qprj, kprj = rows * QPROJ_ROWS, rows * KVPROJ_ROWS
    vheads = rows * NUM_VALUE_HEADS
    chunk_rows = (rows + GDN_CHUNK_TOKENS - 1) // GDN_CHUNK_TOKENS
    return [
        ("s_a", u16(h)), ("s_b", u16(h)), ("norm", u16(h)), ("norm_f32", f32(h)),
        ("embed", u16(h)), ("delta", u16(h)),
        ("block_out", u16(vdim)), ("block_out_f32", f32(vdim)),
        ("g_raw", u16(qkv)), ("g_z", u16(vdim)),
        ("g_a", u16(vheads)), ("g_b", u16(vheads)),
        ("g_dt_bias", f32(NUM_GDN_LAYERS * NUM_VALUE_HEADS)),
        ("g_conv_w", f32(NUM_GDN_LAYERS * CONV_CHANNELS * CONV_KERNEL)),
        ("g_g", f32(vheads)), ("g_beta", f32(vheads)),
        ("g_u", f32(qkv)), ("g_o", f32(vdim)),
        ("conv_new", f32(CONV_WINDOW * CONV_CHANNELS)),
        ("g_nv", f32(rows * NUM_VALUE_HEADS * HEAD_DIM)),
        ("g_kcd", u16(rows * NUM_VALUE_HEADS * HEAD_DIM)),
        ("g_qg", u16(rows * NUM_VALUE_HEADS * HEAD_DIM)),
        ("g_kh", u16(rows * NUM_VALUE_HEADS * HEAD_DIM)),
        ("g_intra", u16(rows * NUM_VALUE_HEADS * GDN_CHUNK_TOKENS)),
        ("g_cdec", f32(chunk_rows * NUM_VALUE_HEADS)),
        ("g_kn", f32(vdim)), ("g_qn", f32(vdim)),
        ("g_gsc", f32(rows * 2 * NUM_VALUE_HEADS)),
        ("a_q", u16(qprj)), ("a_k", u16(kprj)), ("a_v", u16(kprj)),
        ("a_q_rope", u16(rows * NUM_QUERY_HEADS * HEAD_DIM)),
        ("a_k_rope", u16(rows * KVPROJ_ROWS)),
        ("a_gate", u16(rows * NUM_QUERY_HEADS * HEAD_DIM)),
        ("a_pre", u16(rows * NUM_QUERY_HEADS * HEAD_DIM)),
        ("a_gated", u16(rows * NUM_QUERY_HEADS * HEAD_DIM)),
        ("a_part", A_PART_BYTES),
        ("mlp_act", u16(i)), ("mlp_act_f32", f32(i)),
        ("mlp_gate_raw", u16(i)), ("mlp_up_raw", u16(i)),
        ("logits", f32(VOCAB)),
        ("token_ids_dev", 4 * EMBED_ID_SLOTS * rows),
    ]


def state_bytes(cap: int) -> dict[str, int]:
    """`state_sizes` (engine/state.cpp:47-60), same formula, same order."""
    gdn = NUM_GDN_LAYERS * NUM_VALUE_HEADS * DK * DV * 4
    conv = NUM_GDN_LAYERS * CONV_WINDOW * CONV_CHANNELS * 4
    kv_layer = cap * NUM_KV_HEADS * HEAD_DIM * 2
    kv = 2 * NUM_ATTN_LAYERS * kv_layer
    return {"gdn": gdn, "conv": conv, "kv": kv, "total": gdn + conv + kv}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--widths", default="512,1024,4096")
    ap.add_argument("--cold-gbs", type=float, default=35.0,
                    help="first GPU pass over fresh pages, GB/s (quoted: "
                         "devices/strix-halo-unified-memory.md)")
    ap.add_argument("--warm-gbs", type=float, default=239.0,
                    help="steady read, GB/s (quoted: same file)")
    args = ap.parse_args()

    print("# derived from engine/state.h:41-64, engine/decode_step.cpp:224-318, "
          "engine/state.cpp:47-60")
    print(f"# rates QUOTED from devices/strix-halo-unified-memory.md: cold {args.cold_gbs} "
          "GB/s (first pass over fresh pages), warm "
          f"{args.warm_gbs} GB/s (steady)")
    sc = scratch_bytes(512)
    total = sum(n for _, n in sc)
    print(f"# scratch at wave_rows=512: {total / 1e6:8.1f} MB over {len(sc)} buffers "
          f"(largest: {max(sc, key=lambda kv: kv[1])[0]})")
    print(f"# hipBLASLt workspace (kernels_gemm.cpp, workspace_mib=128): "
          f"{WORKSPACE_BYTES / 1e6:.1f} MB")
    print()
    hdr = ("lanes  state_MB   KV_MB  cold_GB  cold_ms@cold  cold_ms@warm  extra_ms")
    print(hdr)
    for w in (int(x) for x in args.widths.split(",")):
        st = state_bytes(w + 2)          # decode_run's cap = prompt + steps + 1
        # Cold at first wave = scratch + workspace + KV. The GDN state and conv window
        # are hipMemset'd in allocate_state's `st.reset()` (state.cpp:104,118-126), which
        # is BEFORE the timed region, so they are already committed.
        cold = (total + WORKSPACE_BYTES + st["kv"]) / 1e9
        cold_ms, warm_ms = cold / args.cold_gbs * 1e3, cold / args.warm_gbs * 1e3
        print(f"{w:6d}  {st['total'] / 1e6:7.1f}  {st['kv'] / 1e6:6.1f}  {cold:7.3f}"
              f"   {cold_ms:9.1f}   {warm_ms:9.1f}   {cold_ms - warm_ms:8.1f}")
    print()
    w_text = 1.5457
    print(f"# for scale: if the 1.5457 GB of TEXT WEIGHTS were first-touched inside the timed")
    print(f"# region it would be {w_text / args.cold_gbs * 1e3:.1f} ms cold / "
          f"{w_text / 60.0 * 1e3:.1f} ms at 60 GB/s — but it is not: loader.cpp:626-644 warms")
    print(f"# every resident page inside load(), and decode_run.cpp:1588-1593 brackets that as")
    print(f"# `load_ms` and never adds it to RESULT ms_total.")


if __name__ == "__main__":
    main()
