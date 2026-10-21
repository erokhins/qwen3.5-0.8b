#!/usr/bin/env python3
"""a9_waste_model.py — CPU mirror of the shipped WMMA attention geometry, for the A9 wasted-work census.

WHY. A9 (prototype/attn_percost/A9-LONGCTX-CENSUS.md) must say how much of the attention round's work at
16,384 / 32,768 lanes is padding, causal ragged tail, block-union overhang, or split-path scratch traffic.
None of that is observable in a trace; it is all a consequence of the launch geometry and the round-loop
bound, so it is computed here from the SAME rules the device runs. Every rule below is a transcription of a
`path:line` in the engine, and the transcription is adjudicated by reproducing two numbers A2 published from
the pre-G1 kernel (see --selftest): block-rounds/layer and staged K/V bytes/layer at 4,096 lanes.

Rules transcribed (path:line at tip 6d0512f):
  N_Q=8, N_KV=2, GQA=4, DH=256, MAXSC=512        prototype/attn_decode/attn_decode_kernels.h:23-26, :32
  lanes_of(a,q) = lvalid + q                     attn_decode_kernels.h:203
  blocks_for_row = 1 if full and lanes_of<=512
                   else a.split                  attn_decode_kernels.h:220-223
  fattn_covers   = cover and q<nrows and blocks_for_row>1
                                                 attn_decode_kernels.h:553-555
  body i of a W-lane prefill: pos0=512i, lvalid=pos0+1, nrows=512
                                                 engine/decode_step.cpp:686-688
  split = split_for(lanes_last) = ceil(lanes_last/512), lanes_last = lvalid+nrows-1
                                                 engine/kernels_attn.cpp:86, :239, :256
  grid = (N_Q, split, fattn_rtiles_of(512)); FATTN_ROWS_PER_BLOCK=48 -> rtiles=11
                                                 engine/kernels_attn.cpp:494; attn_decode_kernels.h:545-546, :561
  rt = gridDim.z-1-blockIdx.z; rb = rt*FATTN_WARPS + w; rows rb*16 .. rb*16+15
                                                 engine/attn_fattn_kernels.h:671-672, :461
  row slice: span=ceil(nrow/s_row), blo=y*span, bhi=min(blo+span,nrow)
                                                 engine/attn_fattn_kernels.h:475-479
  round range = block union [lo_all, hi_all) step FATTN_MLANES=16
                                                 engine/attn_fattn_kernels.h:749-763, :771
  K/V stage fill clamps a past-hi_all lane to hi_all-1 (a real global load of a duplicate row)
                                                 engine/attn_fattn_kernels.h:776-780 (K), :894-898 (V)
  cell is live iff covered and blo <= j < bhi    engine/attn_fattn_kernels.h:846-848

Costs booked per block-round (one block = 3 warps x 16 rows, 16 KV lanes):
  issued MAC  = 48 rows x 16 lanes x 2*DH                (score DH + P*V DH; masked cells still compute)
  staged B    = 16 lanes x DH x 2 B x 2 (K then V) = 16,384 B   (A2 §3's per-fill size)
  q operand B = 48 rows x DH x 2 B = 24,576 B demanded from global at the use site (the shipped `_u8`
                spelling reads Q at the use site: engine/attn_fattn_kernels.h:722-729)
  rescale     = 16 tiles x 8 F32 = 128 v_mul slots per lane-round = 128 per wavefront-round
                engine/attn_fattn_kernels.h:879-882 + A2 §2's histogram

Usage:
  python3 prototype/attn_percost/a9_waste_model.py --selftest
  python3 prototype/attn_percost/a9_waste_model.py --widths 4096 16384 32768
"""
import argparse
import math

# --- geometry, one spelling per quantity, each one cited above -------------------------------
N_Q, N_KV = 8, 2
GQA = N_Q // N_KV
DH = 256
MAXSC = 512                 # body width == MAXSC == kMaxWaveRows
FATTN_MROWS = 16            # rows per warp
FATTN_WARPS = 3             # warps per block
ROWS_PER_BLOCK = FATTN_WARPS * FATTN_MROWS      # 48
MLANES = 16                 # FATTN_MLANES: KV lanes per round
K_ATTN = 6                  # full-attention layers per pass (engine/state.h:67)
STAGE_BYTES_PER_ROUND = MLANES * DH * 2 * 2     # 16,384 B (K and V, BF16)
Q_BYTES_PER_ROUND = ROWS_PER_BLOCK * DH * 2     # 24,576 B demanded from global (unstaged Q)
RESCALE_SLOTS = 16 * 8                          # 128 v_mul_f32 per lane-round
PART_SLOT_B = (DH + 2) * 4                      # 1,032 B per (row, head, lane block)


def lanes_of(lvalid: int, q: int) -> int:
    return lvalid + q


def blocks_for_row(lvalid: int, q: int, split: int) -> int:
    return 1 if lanes_of(lvalid, q) <= MAXSC else split


def fattn_covers(lvalid: int, q: int, split: int, nrows: int) -> bool:
    return q < nrows and blocks_for_row(lvalid, q, split) > 1


def bodies_of(width: int):
    """One body per 512 lanes; body i: pos0=512i, lvalid=pos0+1, nrows=512 (widths here are multiples of 512)."""
    assert width % MAXSC == 0, f"width {width} is not a whole number of {MAXSC}-row bodies"
    out = []
    for i in range(width // MAXSC):
        pos0 = MAXSC * i
        out.append({"i": i, "pos0": pos0, "lvalid": pos0 + 1, "nrows": MAXSC})
    return out


def split_for(lanes: int) -> int:
    return (lanes + MAXSC - 1) // MAXSC


def row_slice(lvalid, q, split, y):
    """(covered, blo, bhi) for one row of one lane block — attn_fattn_kernels.h:461-482."""
    nrows = MAXSC
    if not fattn_covers(lvalid, q, split, nrows):
        return False, 0, 0
    s_row = blocks_for_row(lvalid, q, split)
    if y >= s_row:
        return False, 0, 0
    nrow = lanes_of(lvalid, q)
    span = (nrow + s_row - 1) // s_row
    blo = y * span
    bhi = min(blo + span, nrow)
    if bhi <= blo:
        return False, 0, 0
    return True, blo, bhi


def block_rounds(width_lo: int, hi_all: int, mlanes: int = MLANES) -> int:
    return (hi_all - width_lo + mlanes - 1) // mlanes


def census(width: int, warps: int = FATTN_WARPS, mlanes: int = MLANES, verbose=False):
    """Sums ONE attention layer: every body of the prompt and every query head.

    `warps` and `mlanes` are the L-a / L-b geometry knobs: a block carries `warps x 16` query rows and one
    round consumes `mlanes` KV lanes. Both are 3 and 16 in the shipping kernel
    (`engine/attn_fattn_kernels.h:186-187`, `:232-233`); any other value is a WHAT-IF whose output is geometry
    arithmetic, never a timing. The two defaults are what the `--selftest` figures were taken at.

    Each of the 6 full-attention layers walks the whole prompt, so per-pass figures are these x K_ATTN. The
    three traffic fields (partial_write_bytes / partial_read_bytes / merged_row_bytes) already carry that x6
    and are per-pass; the block / round / MAC / stage / Q fields are per-layer. The selftest proves the split:
    block_rounds and staged bytes at 4,096 lanes equal A2-census.md's per-layer figures exactly, and
    partial_write_bytes equals its per-pass figure.
    """
    rows_per_block = warps * FATTN_MROWS
    rtiles = (MAXSC + rows_per_block - 1) // rows_per_block      # 11 at warps=3
    acc = {
        "bodies": 0, "fattn_bodies": 0, "blocks": 0, "block_rounds": 0,
        "issued_mac": 0, "useful_mac": 0, "staged_bytes": 0, "q_bytes": 0,
        "dup_stage_lanes": 0,           # KV lanes loaded as a duplicate of hi_all-1
        "rounds_fully_masked": 0,       # rounds in which no row of the block has a live cell
        "row_slots_issued": 0,         # warp x 16 row-slots that ran rounds
        "row_slots_useful": 0,         # of those, row-slots with at least one covered row
        "row_slots_out_of_range": 0,
        "rows_covered": 0, "rows_one_pass": 0,
        "partial_writes": 0,            # partial slots written (row, head, lane block)
        "merged_rows": 0,               # rows attn_combine rewrites
        "rescale_slot_rounds": 0,       # lane-rounds of rescale == block_rounds x 3 warps
        "waste_union_mac": 0,           # block union wider than the row's own slice
        "waste_quant_mac": 0,           # ceil(union/16) lanes no row of the block needs
        "waste_pad_mac": 0,             # row-slots out of range / not this kernel's, still run the round
        "rounds": 0,
    }
    per_body = []
    for b in bodies_of(width):
        lvalid, nrows = b["lvalid"], b["nrows"]
        lanes_last = lvalid + nrows - 1
        split = split_for(lanes_last)
        acc["bodies"] += 1
        brec = {"i": b["i"], "split": split, "rounds": 0, "blocks": 0,
                "useful_mac": 0, "issued_mac": 0}
        if split < 2:
            # attn_fattn_wave is still launched (it is inert: every block returns at its first ballot),
            # engine/kernels_attn.cpp:354 ATTN_FATTN_SKIP / fattn_covers=0 path -> no rounds.
            acc["rows_one_pass"] += nrows
            per_body.append(brec)
            continue
        acc["fattn_bodies"] += 1
        for y in range(split):
            for rt in range(rtiles):
                for head in range(N_Q):          # each head is its own block column
                    acc["blocks"] += 1
                    brec["blocks"] += 1
                    lo_all, hi_all = 10**9, 0
                    rows = []
                    for rb_off in range(warps):
                        rb = rt * warps + rb_off
                        for q in range(FATTN_MROWS):
                            qrow = rb * FATTN_MROWS + q
                            cov, blo, bhi = row_slice(lvalid, qrow, split, y)
                            rows.append((qrow, cov, blo, bhi))
                            if cov:
                                lo_all = min(lo_all, blo)
                                hi_all = max(hi_all, bhi)
                    if hi_all <= lo_all:
                        continue                 # if (!any_mask) return;
                    rounds = block_rounds(lo_all, hi_all, mlanes)
                    acc["block_rounds"] += rounds
                    acc["rounds"] += rounds
                    brec["rounds"] += rounds
                    acc["rescale_slot_rounds"] += rounds * warps
                    acc["staged_bytes"] += rounds * mlanes * DH * 2 * 2
                    acc["q_bytes"] += rounds * rows_per_block * DH * 2
                    acc["row_slots_issued"] += rounds * warps * FATTN_MROWS
                    n_useful_slots = sum(1 for (_, cov, _, _) in rows if cov)
                    acc["row_slots_useful"] += rounds * n_useful_slots
                    acc["row_slots_out_of_range"] += rounds * (
                        warps * FATTN_MROWS - n_useful_slots)
                    for (qrow, cov, blo, bhi) in rows:
                        if cov:
                            acc["rows_covered"] += 0     # counted once per row below
                    issued = rounds * rows_per_block * mlanes * 2 * DH
                    n_cov = sum(1 for (_, cov, _, _) in rows if cov)
                    own = sum((bhi - blo) for (_, cov, blo, bhi) in rows if cov)
                    useful = own * 2 * DH
                    union_live = hi_all - lo_all
                    acc["issued_mac"] += issued
                    acc["useful_mac"] += useful
                    # Three named components of issued-useful, summing exactly to issued - useful:
                    #  pad  : row-slots that are out of range or owned by another kernel still run the round
                    #  quant: ceil(union/16) stages lanes no row of the block needs this round
                    #  over : the block union is wider than the row's own slice (48 rows share one range)
                    acc["waste_pad_mac"] += (rows_per_block - n_cov) * rounds * mlanes * 2 * DH
                    acc["waste_quant_mac"] += n_cov * (rounds * mlanes - union_live) * 2 * DH
                    acc["waste_union_mac"] += (n_cov * union_live - own) * 2 * DH
                    # KV lanes staged but past every row's need in this block (duplicate loads)
                    acc["dup_stage_lanes"] += rounds * mlanes - union_live
                    brec["useful_mac"] += useful
                    brec["issued_mac"] += issued
        per_body.append(brec)

    # split-path traffic: a covered row writes one partial slot per lane block it owns
    for b in bodies_of(width):
        lvalid, nrows = b["lvalid"], b["nrows"]
        split = split_for(lvalid + nrows - 1)
        if split < 2:
            continue
        for q in range(nrows):
            s_row = blocks_for_row(lvalid, q, split)
            acc["partial_writes"] += s_row
        acc["merged_rows"] += sum(1 for q in range(nrows)
                                  if blocks_for_row(lvalid, q, split) > 1)
    acc["partial_write_bytes"] = acc["partial_writes"] * N_Q * PART_SLOT_B * K_ATTN
    acc["partial_read_bytes"] = acc["partial_write_bytes"]      # attn_combine reads it all back
    acc["merged_row_bytes"] = acc["merged_rows"] * N_Q * DH * 4 * K_ATTN
    acc["one_pass_rows"] = sum(
        1 for b in bodies_of(width) if split_for(b["lvalid"] + b["nrows"] - 1) < 2
        for _ in range(b["nrows"]))
    acc["width"] = width
    acc["per_body"] = per_body
    return acc


# --- machine facts, KFD-sourced (see residency_ladder's docstring) --------------------------------
CUS = 40
SIMD_PER_CU = 2
BLOCKS_PER_CU_SHIPPED = 3        # 65,536 // 20,480 (the shipped `_u8` LDS)

CAUSAL_FLOP_PER_PASS = lambda W: 2 * K_ATTN * N_Q * 2 * DH * (W * (W + 1) // 2)


def split_caps(width, caps):
    """The L-c table: what the split path costs, and what a cap on the launch split would save.

    Capping `split` at c does NOT change the MAC work: a row's lanes are still partitioned into `s_row`
    contiguous slices and every lane is visited once (`span = ceil(nrow / s_row)` grows as `s_row` shrinks,
    engine/attn_fattn_kernels.h:475-479). What moves is (a) the partial round-trip, one 1,032 B slot per
    (row, head, lane block) less, and (b) the block count `8 x s_row x rtiles` against the machine's
    resident-block capacity (40 CU x 3 blocks at the shipped 20,480 B LDS, `progress/2026-10-21/performance-status-2026-10-21.md` §48),
    which is what sets how many independent block-rounds the scheduler has to overlap.

    `wall_rounds` is the widest body's `ceil(blocks / capacity) x rounds_per_block`: the coarse model of how
    long one body takes if blocks are equal-sized and the machine holds `capacity` of them at once. It is a
    packing argument, not a timing, and it ignores that later bodies are bigger; it is reported with its
    baseline ratio because that is the only honest use of it.
    """
    rows = []
    for cap in caps:
        wr = rd = 0
        for b in bodies_of(width):
            split = split_for(b["lvalid"] + b["nrows"] - 1)
            if split < 2:
                continue
            s = min(split, cap)
            wr += b["nrows"] * N_Q * s * PART_SLOT_B * K_ATTN
        rd = wr
        cap_blocks = CUS * BLOCKS_PER_CU_SHIPPED
        # The widest body decides the packing question: it has the most blocks AND the most work per block.
        widest = bodies_of(width)[-1]
        split_w = split_for(widest["lvalid"] + widest["nrows"] - 1)
        s = min(split_w, cap)
        nrow_widest = lanes_of(widest["lvalid"], widest["nrows"] - 1)
        span = (nrow_widest + s - 1) // s
        rounds = (span + MLANES - 1) // MLANES
        blocks = N_Q * s * ((MAXSC + ROWS_PER_BLOCK - 1) // ROWS_PER_BLOCK)
        waves = blocks / cap_blocks
        rows.append({
            "cap": cap, "roundtrip_GB": (wr + rd) / 1e9,
            "widest_blocks": blocks, "resident_capacity": cap_blocks, "block_waves": waves,
            "wall_rounds": math.ceil(waves) * rounds,
            "rounds_per_block": rounds,
        })
    base = rows[0]["wall_rounds"] if rows else 1
    for r in rows:
        r["wall_vs_uncapped_pct"] = 100 * (r["wall_rounds"] / base - 1)
    return rows


def llama_staged_bytes(width):
    """K/V bytes llama.cpp's traced instantiation demands from global per pass, same method as ours.

    Geometry read, not assumed: the traced instantiation is
    `flash_attn_ext_f16<256,256,16,4,false,false,false>` (A2 §1's trace row), so DKQ=DV=256,
    ncols1=16 (query tokens per tile, fattn-mma-f16.cuh:1889 `iter_j`), ncols2=4 (query HEADS per tile,
    :1888 `iter_z_gqa` over `gqa_ratio`), nbatch_fa=64 (KV rows per softmax rescaling, :13 + the RDNA row
    at :164). Stages per KV tile (`nbytes_shared_KV_1stage`, :2033) = nbatch_fa x max(stride_tile_K,
    stride_tile_V) x sizeof(half2) with stride = nbatch_K2+4 = 132 -> 33,792 B for K; V is a SECOND tile of
    the same size in their layout (`tile_V`, :1252) and with nstages=1 the two share the one dynamic
    allocation but both are filled per KV tile, so the DEMANDED bytes per KV tile are K+V = 67,584 B.
    """
    NCOLS1, NCOLS2, NBATCH_FA = 16, 4, 64
    stride = 128 + 4                      # nbatch_K2 + 4 (no swizzle on this part: swz row absent)
    kv_tile_bytes = NBATCH_FA * stride * 4        # 33,792 B, one of K or V
    per_kv = 2 * kv_tile_bytes                    # K and V
    tiles = width // NCOLS1                        # one jt per 16 query tokens
    # a tile's causal bound is its LAST token: chunks = ceil((jt+1)*NCOLS1 / NBATCH_FA)
    chunks = sum((jt * NCOLS1 + NCOLS1 + NBATCH_FA - 1) // NBATCH_FA for jt in range(tiles))
    per_layer = N_KV * chunks * per_kv             # 2 KV head groups; the 4 GQA heads ride in one tile
    return {"per_layer": per_layer, "per_pass": per_layer * K_ATTN,
            "per_token": per_layer * K_ATTN / width, "per_kv_tile_B": per_kv,
            "q_rows_per_tile": NCOLS1 * NCOLS2, "kv_rows_per_tile": NBATCH_FA}


def residency_ladder(warps, threads_per_warp=32, qstage_bytes_per_warp=4096, kv_stage=8192):
    """LDS-bound blocks/CU and wf/CU for a warps-per-block choice, POST-G1 (Q un-staged).

    The ladder in `prototype/attn_wmma/fattn_regfile_ladder.md` §1 (`wf/CU(NW) = NW*floor(8/(NW+1))`, ceiling
    7) was derived while Q was staged at 8,192 B/warp. G1 moved Q to global (`engine/performance-status.md`
    §48), so the per-warp stage is now the 4,096 B epilogue transpose and the ladder is a different function.
    LDS/CU 65,536 B and 40 CU / 80 SIMD are KFD facts (same §, and tools/trace_phase_split.py:kfd_topology).
    Never hipGetDeviceProperties, never clang's constant `Occupancy [waves/SIMD]` (AGENTS trap row).
    """
    LDS_CU = 65536
    lds = warps * qstage_bytes_per_warp + kv_stage
    blocks = LDS_CU // lds
    wf = warps * blocks
    return {"warps": warps, "threads": warps * threads_per_warp, "lds_block": lds,
            "blocks_per_cu": blocks, "wf_per_cu": wf,
            "rows_per_kv_stage": warps * FATTN_MROWS,
            # the register-file check A3 §2b says is un-decidable on this part, stated as the sum anyway
            "vgpr_demand_at_185": blocks * warps * threads_per_warp * 185}


def selftest():
    """Adjudicate the mirror against two figures A2 published from the real kernel."""
    c = census(4096)
    a2_rounds = 94752                    # A2 §2: "94,752 block-rounds/layer"
    a2_staged = 1_552_416_768            # A2 §3: "per LAYER: staged 1,552,416,768 B"
    rounds_per_layer = c["block_rounds"]
    staged_per_layer = c["staged_bytes"]
    print(f"# selftest at 4096 lanes (mirror vs A2-census.md §2/§3)")
    print(f"  block-rounds/layer : mirror {rounds_per_layer:,}  A2 {a2_rounds:,}  "
          f"delta {rounds_per_layer - a2_rounds:+,}")
    print(f"  staged B/layer     : mirror {staged_per_layer:,}  A2 {a2_staged:,}  "
          f"delta {staged_per_layer - a2_staged:+,}")
    ok = rounds_per_layer == a2_rounds and staged_per_layer == a2_staged
    # A2 §3's demand1x per layer = 68,736,253,952 B at 4096 lanes: visits x N_Q x 2 x DH x 2 B
    visits = (4096 * 4097 // 2) * N_Q
    d1 = visits * 2 * DH * 2
    print(f"  demand1x B/layer   : mirror {d1:,}  A2 68,736,253,952  delta {d1 - 68_736_253_952:+,}")
    ok = ok and d1 == 68_736_253_952
    # the flop model of A9 §0 must reproduce A2 §2's 2.062e11 MAC/pass at 4096
    mac = K_ATTN * N_Q * 2 * DH * (4096 * 4097 // 2)
    print(f"  MAC/pass           : mirror {mac:.4e}  A2 2.062e11")
    ok = ok and abs(mac - 2.062e11) < 1e8
    # A2 §2 C2: tile waste "4.2-7.3 % at every body of every width (1.07x at most)" — reported, not gated:
    # A2's C2 counted row-slot padding only, this mirror also charges the block-union overhang and the
    # round quantisation, so its issued/useful is expected to be LARGER. The §2 decomposition shows which.
    waste = c["issued_mac"] / c["useful_mac"]
    print(f"  issued/useful MAC  : mirror {waste:.4f}  (A2 §2 C2 quoted 1.042-1.073 for padding alone; "
          f"this mirror also charges union+quant, see the §2 table)")
    ok = ok and 1.0 < waste <= 1.30
    # A2 §4: partial bytes WRITTEN per pass at 4096 lanes = 887,685,120 B
    a2_part = 887_685_120
    print(f"  partial B/pass     : mirror {c['partial_write_bytes']:,}  A2 {a2_part:,}  "
          f"delta {c['partial_write_bytes'] - a2_part:+,}")
    ok = ok and c["partial_write_bytes"] == a2_part
    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def report(width, c):
    issued, useful = c["issued_mac"], c["useful_mac"]
    print(f"\n=== width {width:,} lanes ===   block/round/MAC/stage rows are PER LAYER (x{K_ATTN} = per pass); "
          f"traffic rows are per PASS")
    print(f"  bodies                        {c['bodies']}   (fattn-eligible {c['fattn_bodies']}, "
          f"one-pass {c['one_pass_rows']} rows)")
    print(f"  fattn blocks (all heads)     {c['blocks']:,}   per pass {c['blocks'] * K_ATTN:,}")
    print(f"  block-rounds                 {c['block_rounds']:,}   per pass {c['block_rounds'] * K_ATTN:,}")
    print(f"  causal FLOP/pass             {CAUSAL_FLOP_PER_PASS(width)/1e12:.4f} TFLOP")
    print(f"  USEFUL MAC/layer (covered rows x own slice) {useful:.5e}")
    print(f"  ISSUED MAC/layer (48 rows x 16 lanes x 2 DH) {issued:.5e}   issued/useful = {issued/useful:.4f}")
    print(f"  issued MAC/pass              {issued * K_ATTN:.5e}   useful MAC/pass {useful * K_ATTN:.5e}"
          f"   model causal MAC/pass {K_ATTN * N_Q * 2 * DH * (width * (width + 1) // 2):.5e}")
    print(f"    waste: block union/mask    {c['waste_union_mac']:.5e} "
          f"({100*c['waste_union_mac']/issued:.2f} % of issued)")
    print(f"    waste components (as % of issued): pad {100*c['waste_pad_mac']/issued:.2f} %  "
          f"round-quant {100*c['waste_quant_mac']/issued:.2f} %  "
          f"union-overhang {100*c['waste_union_mac']/issued:.2f} %")
    print(f"    out-of-range row slots ran {c['row_slots_out_of_range']:,} lane-rounds "
          f"of {c['row_slots_issued']:,} issued")
    print(f"    fully-masked rounds {c['rounds_fully_masked']:,}")
    print(f"  duplicate K/V lane loads     {c['dup_stage_lanes']:,} lanes "
          f"= {c['dup_stage_lanes']*DH*2*2:,} B")
    staged_pass = c["staged_bytes"] * K_ATTN
    q_pass = c["q_bytes"] * K_ATTN
    uniq_pass = K_ATTN * 2 * DH * 2 * width
    print(f"  staged K/V bytes/pass        {staged_pass:,} B ({staged_pass / width / 1e6:.3f} MB/token) "
          f"= {staged_pass / uniq_pass:.1f}x the unique K/V")
    print(f"  unique K/V bytes/pass        {uniq_pass:,} B ({uniq_pass / width / 1e6:.3f} MB/token)")
    print(f"  Q operand B demanded/pass    {q_pass:,} B ({q_pass / width / 1e6:.3f} MB/token) "
          f"[the shipped _u8 spelling reads Q at the use site]")
    print(f"  rescale lane-rounds/pass     {c['rescale_slot_rounds'] * K_ATTN:,} "
          f"= {c['rescale_slot_rounds'] * K_ATTN * RESCALE_SLOTS / width:,.0f} v_mul slots/token")
    print(f"  partial slots written        {c['partial_writes']:,} x 1,032 B x 8 heads x 6 layers"
          f" = {c['partial_write_bytes']/1e9:.3f} GB written, same read back,")
    print(f"                               + merged rows {c['merged_row_bytes']/1e9:.3f} GB written")
    split_total = (c["partial_write_bytes"] + c["partial_read_bytes"] + c["merged_row_bytes"])
    print(f"  split traffic total          {split_total/1e9:.3f} GB/pass")
    print(f"  block-rounds per token       {c['block_rounds'] * K_ATTN / width:.2f}  (x{FATTN_WARPS} warps = "
          f"{c['block_rounds'] * K_ATTN * FATTN_WARPS / width:.2f} wavefront-rounds/token, 6 layers)")
    # rounds per token: every covered row is present in `split` lane blocks; count per row
    print("  per-body detail (first 3, last 3):")
    hdr = f"    {'body':>5} {'split':>6} {'blocks':>8} {'rounds/blk avg':>15} {'issued/useful':>14}"
    print(hdr)
    for brec in (c["per_body"][:3] + c["per_body"][-3:]):
        if brec["blocks"] == 0:
            print(f"    {brec['i']:>5} {brec['split']:>6} {'-':>8} {'(one-pass)':>15} {'-':>14}")
            continue
        print(f"    {brec['i']:>5} {brec['split']:>6} {brec['blocks']:>8} "
              f"{brec['rounds']/brec['blocks']:>15.2f} "
              f"{brec['issued_mac']/max(brec['useful_mac'],1):>14.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--widths", type=int, nargs="+", default=[4096, 16384, 32768])
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--caps", action="store_true", help="the L-c split-cap table at --cap-width")
    ap.add_argument("--cap-width", type=int, default=32768)
    ap.add_argument("--traffic", action="store_true", help="the §3 demanded-traffic comparison")
    ap.add_argument("--variants", action="store_true", help="the L-a / L-b geometry what-if table")
    a = ap.parse_args()
    if a.selftest:
        raise SystemExit(selftest())
    for w in a.widths:
        report(w, census(w))
    if a.caps:
        print(f"\n=== L-c: capping the launch split at width {a.cap_width:,} ===")
        print(f"    {'cap':>4} {'round-trip GB/pass':>19} {'widest-body blocks':>19} "
              f"{'block waves':>12} {'rounds/block':>13} {'widest-body wall':>17} {'vs uncapped':>12}")
        for r in split_caps(a.cap_width, [64, 32, 16, 8, 4, 2]):
            print(f"    {r['cap']:>4} {r['roundtrip_GB']:>19.3f} {r['widest_blocks']:>19} "
                  f"{r['block_waves']:>12.2f} {r['rounds_per_block']:>13} "
                  f"{r['wall_rounds']:>17} {r['wall_vs_uncapped_pct']:>11.2f}%")
    if a.traffic:
        print("\n=== §3: K/V bytes DEMANDED per query row (demanded != DRAM, D21) ===")
        lm = llama_staged_bytes(32768)
        print(f"  (per-touch figures are per causal (row,lane,head) touch = N_Q x W(W+1)/2, one layer)")
        for w in a.widths:
            c = census(w)
            touches = N_Q * (w * (w + 1) // 2)
            ours_layer = c["staged_bytes"] / touches
            q_layer = c["q_bytes"] / touches
            print(f"  width {w:,}: ours staged K/V {c['staged_bytes'] * K_ATTN / 1e9:,.1f} GB/pass "
                  f"= {ours_layer:.2f} B/touch | Q operand {c['q_bytes'] * K_ATTN / 1e9:,.1f} GB/pass "
                  f"= {q_layer:.2f} B/touch | total {(ours_layer + q_layer):.2f} B/touch")
        print(f"  llama: Q tile {lm['q_rows_per_tile']} rows x KV tile {lm['kv_rows_per_tile']} lanes, "
              f"{lm['per_kv_tile_B']:,} B staged per KV tile (K+V, stride-padded) "
              f"= {lm['per_layer']/ (N_Q*(32768*32769//2)):.2f} B/touch")
        c = census(32768)
        touches32 = N_Q * (32768 * 32769 // 2)
        ours32 = c["staged_bytes"]
        print(f"  ratio at 32,768: ours {ours32/touches32:.2f} B/touch vs llama "
              f"{lm['per_layer']/touches32:.2f} B/touch = "
              f"{(ours32/touches32)/(lm['per_layer']/touches32):.3f}x more demanded K/V bytes per touch")
        print(f"  reuse factor: ours {ROWS_PER_BLOCK} query rows share one 16-lane stage; "
              f"llama {lm['q_rows_per_tile']} query rows share one {lm['kv_rows_per_tile']}-lane stage "
              f"-> stage bytes / (row x lane) = {2*DH*2/ROWS_PER_BLOCK:.2f} vs "
              f"{lm['per_kv_tile_B']/lm['q_rows_per_tile']/lm['kv_rows_per_tile']:.2f} B")
        uniq_data = K_ATTN * 2 * DH * 2 * 32768
        uniq_demand = N_Q * 32768 * 2 * DH * 2 * K_ATTN
        print(f"  unique K/V DATA per pass at 32,768 = {uniq_data/1e9:.3f} GB "
              f"(6 layers x 2 tensors x 512 B x W); DRAM floor at 237 GB/s = {uniq_data/237e9*1000:.1f} ms")
        print(f"  minimum DEMANDED by a per-head-block kernel (each of 8 head columns streams its group's "
              f"prefix once) = {uniq_demand/1e9:.3f} GB/pass")
        print(f"  ours staged / unique-data = {ours32*K_ATTN/uniq_data:.1f}x  "
              f"(A2 §3 called this 'uniq' and got 41x at 4,096; that column was the minimum-demand one)")
        print(f"  ours staged / min-demand = {ours32*K_ATTN/uniq_demand:.1f}x ; "
              f"llama staged / min-demand = {lm['per_pass']/uniq_demand:.1f}x")
        print(f"  per token at 32,768: ours K/V {ours32*K_ATTN/32768/1e6:.2f} MB + Q "
              f"{c['q_bytes']*K_ATTN/32768/1e6:.2f} MB ; llama K/V {lm['per_pass']/32768/1e6:.2f} MB")
    if a.variants:
        print(f"\n=== L-a / L-b geometry variants at width {a.cap_width:,} (geometry arithmetic, no timing) ===")
        print(f"    {'warps':>5} {'mlanes':>6} {'LDS B':>7} {'wf/CU':>6} {'rtiles':>7} {'staged GB/pass':>15} "
              f"{'Q GB/pass':>11} {'rescale slots/tok':>18} {'issued/useful':>14} "
              f"{'rounds/token':>13} {'pred. family vs shipped':>24}")
        base = None
        for warps, mlanes in ((3, 16), (4, 16), (5, 16), (6, 16), (7, 16), (3, 32), (3, 64), (6, 32)):
            r = residency_ladder(warps, kv_stage=mlanes * DH * 2)
            c = census(a.cap_width, warps=warps, mlanes=mlanes)
            rescale_tok = c["rescale_slot_rounds"] * K_ATTN * RESCALE_SLOTS / a.cap_width
            fam = None
            if base is None:
                base = (r["wf_per_cu"], rescale_tok)
                pred = "(shipped)"
            else:
                # Two independent, separately-labelled predictions, both quoted from measured moves:
                #  (i) residency: §48's own datapoint, 6 -> 9 wf/CU bought family x0.7791, so the exponent is
                #      log(1/0.7791)/log(9/6) = 0.644 (A2 §2 C6's independent §31 figure of 0.72 is the range).
                #  (ii) slot budget: A4-L1 measured a 7.7 %-of-round slot cut delivering 1.67 % of family,
                #      i.e. a pass-through of 0.217; a slot INCREASE is charged at the same rate.
                # Time factor falls as wf/CU rises, so the exponent is NEGATIVE.
                exp48 = math.log(1 / 0.7791) / math.log(9 / 6)
                res = (r["wf_per_cu"] / base[0]) ** (-exp48)
                slot_cut = 1 - rescale_tok / base[1]
                net = res * (1 - 0.217 * slot_cut)
                pred = (f"res x{res:.3f}, slots {100 * slot_cut:+.2f} % -> x{net:.3f} "
                        f"({100 * (net - 1):+.1f} % family)")
            print(f"    {warps:>5} {mlanes:>6} {r['lds_block']:>7} {r['wf_per_cu']:>6} "
                  f"{(MAXSC + warps * FATTN_MROWS - 1) // (warps * FATTN_MROWS):>7} "
                  f"{c['staged_bytes'] * K_ATTN / 1e9:>15.1f} {c['q_bytes'] * K_ATTN / 1e9:>11.1f} "
                  f"{rescale_tok:>18,.0f} {c['issued_mac'] / c['useful_mac']:>14.4f} "
                  f"{c['block_rounds'] * K_ATTN / a.cap_width:>13.1f} {pred:>24}")
    if a.caps or a.traffic or a.variants:
        print("\n=== residency ladder, post-G1 (Q un-staged, per-warp stage 4,096 B epilogue) ===")
        for nw in (3, 4, 5, 6, 7, 14):
            r = residency_ladder(nw)
            print(f"    warps {r['warps']:>3}  threads {r['threads']:>4}  LDS {r['lds_block']:>7} B "
                  f"-> {r['blocks_per_cu']} block/CU = {r['wf_per_cu']:>3} wf/CU  "
                  f"rows sharing one K/V stage {r['rows_per_kv_stage']:>4}  "
                  f"VGPR demand {r['vgpr_demand_at_185']:,}")


if __name__ == "__main__":
    main()
