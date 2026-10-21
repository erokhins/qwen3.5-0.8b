#!/usr/bin/env python3
"""attribute_d3.py — the arithmetic behind Route D3, in one runnable place.

WHY. The route asks why decode attention costs 3.023 ms/step at 4,032 lanes when the KV it must read is
worth about 0.21 ms. Every number below is derived from geometry and from measurements recorded elsewhere
in this tree (the 237 GB/s read plateau, the 7.30 TFLOP/s scalar-FMA rate, 40 CUs); nothing here is a
fit to the measured 3.023. Running it is cheap and CPU-only.

THE ANSWER IT GIVES, in one sentence: unique and demanded traffic are both far below the machine's read
plateau, and the static instruction bill is several times below its issue rate even at the pure-FMA rate,
so the line is neither bandwidth-bound nor issue-bound — what is left is the serial per-tile dependency
chain of a grid that cannot fill the machine, which is why the two levers this prototype builds (resident
blocks per CU, and serial depth per block) are the ones to measure.
"""
N_Q, N_KV, DH = 8, 2, 256            # §02's shape: 8 query heads, 2 KV heads, 256 dims
GQA = N_Q // N_KV                    # 4 query heads share one KV head
LAYERS = 6                           # attention layers 3,7,11,15,19,23 (§02)
MAXSC, TILE = 512, 8                 # staging ceiling and lanes-per-tile, attn_decode_kernels.h
CUS, GHZ = 40, 1.8e9                 # devices/strix-halo-unified-memory.md; KFD says 80 SIMD / 2 = 40 CU
PLATEAU = 237e9                      # GB/s, the measured read ceiling (devices/, prototype/README.md)
SCALAR_FMA_TFLOPS = 7.30             # v_fma_f32, prototype/issue_rate_probe/results.md §1
WARPS_PER_BLOCK = 256 // 32          # THREADS=256, warpSize=32 on this part

# Static warp-instructions per (query head, lane) touch, from `isa_bill.txt` (this build's own
# annotations). WHICH numbers to use is a trap worth stating: the shipped core's `nd`-bounded dot loops
# are ROLLED and left unannotated, so their bodies fall outside every annotated loop and its per-loop
# bill (226 one-pass, 137+118 streaming) is an UNDER-count — one rolled `v_fmac_f32` is standing there
# for eight. `attn_core_lean`, whose trip counts are literals, is unrolled and bills completely: 270
# one-pass, 187+117 = 304 streaming. The complete bill is the one to price with, and it is the
# candidate's, not the shipped one's; the difference between them is a counting artefact and not a
# speed claim either way (see `isa_bill.txt`'s header).
INSNS_TOUCH_STREAM = 304
INSNS_TOUCH_ONEPASS = 270


def geometry(lanes: int) -> dict:
    split = -(-lanes // MAXSC)                       # split_for_lanes(), the engine's own rule
    span = -(-lanes // split)
    tiles = -(-span // TILE)
    # The makespan is priced PER LAYER, because the six layers are six sequential launches: what a CU
    # must serialise is its share of ONE layer's blocks, not of the step's. Pricing it on the step's
    # block count overstates the walk and hides the lever, which is exactly what a ratio is for.
    return {
        "lanes": lanes, "split": split, "span": span, "tiles": tiles,
        "blocks_per_layer": N_Q * split,             # grid = (head, lane block) for a one-row decode
        "blocks_per_step": LAYERS * N_Q * split,
        "launches_per_step": LAYERS * (3 if split > 1 else 2),
        "rounds_at_1_block_per_CU": -(-(N_Q * split) // CUS),
        "tile_trips_per_CU_per_layer": -(-(N_Q * split) // CUS) * tiles,
    }


def makespan(lanes: int, split: int, residency: int) -> int:
    """Tile-trips one CU must serialise for one layer: ceil(blocks / (CUs x resident blocks)) x tiles.
    The point of writing it out is that at residency 1 it is WORK-CONSERVING — deeper cuts do not help,
    because the rounds grow exactly as fast as the walk shortens — and at residency 4 it stops being so
    for a while, and that is where the lever lives."""
    span = -(-lanes // split)
    tiles = -(-span // TILE)
    return -(-(N_Q * split) // (CUS * residency)) * tiles


def report(lanes: int, measured_ms: float, insns: int, launches: int) -> None:
    g = geometry(lanes)
    kv_unique = LAYERS * lanes * N_KV * DH * 2 * 2            # K and V, both KV heads, every layer
    # One touch is one (query head, lane) pair. It reads ONE K row and ONE V row of the SHARED KV head:
    # 256 dims x BF16 = 512 B each, so 1,024 B per touch. The re-use factor this exposes is the point:
    # the four query heads of a GQA group each fetch the same 1,024 B, so demand is GQA x unique.
    touch_bytes = 2 * (N_KV * DH * 2) // N_KV                 # 1,024 B: one K row + one V row
    demand = LAYERS * N_Q * lanes * touch_bytes
    touches = LAYERS * N_Q * lanes
    warp_insns = touches * insns
    print(f"\n=== {lanes} lanes per decode step (measured attn line: {measured_ms:.3f} ms) ===")
    for k, v in g.items():
        print(f"  {k:26s} {v}")
    print(f"  launches per step (staged)  {launches:9d}   (prep + core per layer, + combine above MAXSC)")
    print(f"  unique KV bytes/step        {kv_unique/1e6:9.2f} MB -> floor "
          f"{kv_unique/PLATEAU*1e3:6.3f} ms at the {PLATEAU/1e9:.0f} GB/s plateau")
    print(f"  demanded bytes/step         {demand/1e6:9.2f} MB -> floor "
          f"{demand/PLATEAU*1e3:6.3f} ms  (re-use {demand/kv_unique:.2f}x, = GQA {GQA}: one K/V row "
          f"read once per query head of the group)")
    print(f"  rate actually demanded      {demand/(measured_ms/1e3)/1e9:9.2f} GB/s "
          f"({100*demand/(measured_ms/1e3)/PLATEAU:.1f} % of the plateau)")
    print(f"  (head,lane) touches/step    {touches:9d}   insns/touch {insns}")
    print(f"  warp-instructions/step      {warp_insns/1e6:9.2f} M")
    fma_rate = SCALAR_FMA_TFLOPS * 1e12 / (2 * 32)            # warp-instr/s at the measured scalar rate
    print(f"  issue floor (at the pure {SCALAR_FMA_TFLOPS} TFLOP/s scalar-FMA rate = "
          f"{fma_rate/1e9:.0f} G warp-inst/s): {warp_insns/fma_rate*1e3:6.3f} ms "
          f"-> {measured_ms/(warp_insns/fma_rate*1e3):5.1f}x headroom")
    for half in (2.0, 3.0):                                   # a mixed LDS/VMEM/barrier mix issues worse
        print(f"    if the mix issues at 1/{half:.0f} of that rate: "
              f"{half*warp_insns/fma_rate*1e3:6.3f} ms -> {measured_ms/(half*warp_insns/fma_rate*1e3):4.1f}x")
    per_trip_ms = measured_ms / (g["tile_trips_per_CU_per_layer"] * LAYERS)
    print(f"  tile-trips per CU per layer {g['tile_trips_per_CU_per_layer']:9d}   -> the whole step divided"
          f" by that is {per_trip_ms*1e3:8.3f} us/trip = {per_trip_ms*1e-3*GHZ:7.0f} cycles at "
          f"{GHZ/1e9:.1f} GHz")
    print("     (an UPPER bound on a tile trip: prep, combine, launch gaps and the norms also pay inside")
    print(f"     the same {measured_ms:.3f} ms). A tile's chain - K row load, 8 serial FMAs, a 5-step")
    print("     butterfly, a barrier, V row load, 8 more FMAs - is a few HUNDRED cycles.")


if __name__ == "__main__":
    print("D3 attribution: geometry, traffic and issue arithmetic for the decode attention line.")
    print("KV row = 256 dims x BF16 = 512 B; a touch reads one K row and one V row of the SHARED kv head.")
    report(4032, 3.023, INSNS_TOUCH_STREAM, LAYERS * 3)
    report(128, 0.496, INSNS_TOUCH_ONEPASS, LAYERS * 2)
    print("\n=== the two levers, as arithmetic only (no timing predicted here) ===")
    print("  makespan = tile-trips one CU serialises per layer: ceil(8*split / (40*resident)) * tiles")
    for residency in (1, 4):
        for s in (8, 16, 18, 32, 36, 64):
            span = -(-4032 // s)
            tiles = -(-span // TILE)
            trips = makespan(4032, s, residency)
            print(f"  residency {residency} blocks/CU, split {s:3d}: span {span:4d} lanes, tiles/block "
                  f"{tiles:3d}, tile-trips/CU/layer {trips:4d}   ratio vs ship's "
                  f"{makespan(4032, 8, 1):3d}: {makespan(4032, 8, 1) / trips:5.2f}x")
    print("  At residency 1 the makespan is work-conserving: a deeper cut shortens the walk and lengthens")
    print("  the round count by the same factor, so it buys ~nothing. At residency 4 the 64 blocks of the")
    print("  shipped cut all fit in one round and a deeper cut shortens that round — the two levers")
    print("  compose, and the arithmetic ceiling of the pair is the ~4x row, not either one alone.")
    print(f"  Cost a deeper cut carries: partial traffic 8 x split x 258 F32 x {LAYERS} layers, i.e. at")
    print(f"  split 36 {8 * 36 * 258 * 4 * LAYERS / 1e6:.2f} MB written and read per step = "
          f"{8 * 36 * 258 * 4 * LAYERS * 2 / PLATEAU * 1e3:.4f} ms at the plateau; no extra launches.")
