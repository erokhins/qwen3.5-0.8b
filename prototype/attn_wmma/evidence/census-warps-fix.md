# Teaching `fattn_ship_census.py` the 3-warp geometry (Bet 2)

Worker log, CPU-only. **No device run, no timing, no ms figure appears here or was taken.** The census is
compile-only by design (`prototype/attn_wmma/fattn_census_build.sh` = `hipcc -O3 --offload-temps -c`), which
is the only thing executed besides the Python.

## 0. The crash, reproduced

```
python3 prototype/attn_wmma/fattn_ship_census.py
AssertionError: cannot evaluate 'FATTN_WARPS'   (eval_expr <- shared_bytes <- geometry, line 140/262/783)
```

`geometry()` already reads `FATTN_MROWS/MLANES/THREADS/TBLK/PBLK/TROW` out of the header, but the block's Q
stage is now declared `s_q[FATTN_WARPS * FATTN_MROWS * DH]` and `eval_expr` could not evaluate a bare name it
had never been given, nor a product of names.

## 1. The note block and the listing, read off the shipped 3-warp body (compile-only, this run)

```
vgpr_count 178   vgpr_spill_count 0   sgpr_count 24   sgpr_spill_count 0
private_segment_fixed_size 0   group_segment_fixed_size 32768
```

**Both spill lines are green unaided**: 0 and 0, so `--ack-spill` is not needed and was not used. That retires
D63's "spill is a reported cost" state for this geometry.

Site counts of `attn_fattn_wave` in the device listing, and every `ds_*`/`global_*` mnemonic it emits:

```
v_wmma 17  lds_read_B 1124  lds_write_B 588  global_read_B 64  global_write_B 520
cross_lane 20  s_barrier 11  s_waitcnt lgkmcnt(0) 43   unclassified: none
ds_load_b128 70, ds_store_2addr_b32 65, global_store_b128 32, ds_bpermute_b32 20,
global_load_b128 4, ds_store_b16 4, ds_store_b16_d16_hi 4, ds_store_b128 3,
ds_store_b32 1, ds_load_b32 1, global_store_b64 1
```

Every one of those mnemonics is already in the census's width tables (`_unclassified` is empty), so no new term
had to be invented to cover the 3-warp listing.

## 2. Attribution of the bytes (all of it read off the disassembly, not assumed)

- `ds_load_b128 70` + `ds_load_b32 1` = **1124 B** = score leg 2 operand tiles (64 B) + P·V leg 16 tiles
  (512 B) + publish reads of the transpose stage (512 B) + **the cross-warp fold's 9 int reads (36 B)**. The
  fold reads are visible at the barrier-1 site: `ds_load_b128 ... offset:24576`, `ds_load_b128 ...
  offset:24592`, `ds_load_b32 ... offset:24608` over `fold[0..8]` — exactly 9 ints, no over-read.
- `ds_store_*` = **588 B**: transpose stage 64 paired writes (`ds_store_2addr_b32`, 512 B) + the fold's 3 int
  writes (`ds_store_2addr_b32 v3, v2, v1 offset1:3` = `fold[w]`,`fold[W+w]`, plus `ds_store_b32 offset:24600`
  = `fold[2W+w]`, 12 B) + Q fill (16 B) + K fill (16 B) + the transposed-V narrow fill (4 × `ds_store_b16` +
  4 × `ds_store_b16_d16_hi` = 16 B) = **572 B demanded by the source**, so **16 B more is emitted than the
  source spells** — see §4.
- `global_load_b128 4` = **64 B** vs the 3 fill sites the source spells (Q fill, K fill, inline V fill) =
  48 B — the same +16 B, see §4.
- `global_store_b128 32` + `global_store_b64 1` = 512 + 8 = **520 B** = the publish loop (2 halves ×
  FATTN_PBLK 8 × FATTN_TBLK 8 × 4 B) + the (m, l) header. Exact.
- `s_barrier 11` = the source's own 11 `__syncthreads()` sites weighted by the loops that statically unroll
  them: 2 before the round + 4 in the rolled round loop + 5 in the epilogue (2 per `half` × the 2 unrolled
  halves, + 1 after the loop).
- `cross_lane 20` = 10 warp min/max + 9 in the round + the 1 deferred rowsum. Unchanged by the warp count:
  each is one site in the one kernel body, and three warps execute the same body.
- `v_wmma 17` = 1 rolled score site + 16 unrolled P·V sites. Unchanged, and unchanged by the warp count for
  the same reason.

## 3. Denominators (the census prints these now, because an undenominated count is the census bug class)

| term | denominator |
|---|---|
| `lds_B` | bytes per BLOCK (3 per-warp Q stages + 1 shared K/V stage) |
| `lds_read_B`, `lds_write_B`, `global_read_B`, `global_write_B` | bytes per LANE, one instruction site per source access site (a rolled loop prices one trip; a `#pragma unroll` loop prices all its trips) |
| `v_wmma`, `cross_lane`, `s_barrier`, `s_wait_lgkmcnt0` | static instruction SITES in the one kernel body (`s_barrier` is per block, the others per warp-execution) |

## 4. The +16 B, root-caused by the main agent on review (the worker's §2 hypothesis, confirmed)

The listing the census actually counts (`/tmp/fattn_census2/…gfx1151.s`, `attn_fattn_wave` body) holds **four**
`global_load_b128` and **three** `ds_store_b128`, and the extra pair is not a new access — it is **one fill loop
emitted twice**:

```
282:  global_load_b128 v[7:10],  v[7:8],    off          326:  ds_store_b128 v15, v[7:10]
283:  global_load_b128 v[11:14], v[11:12],  off          327:  ds_store_b128 v16, v[11:14]
735:  global_load_b128 v[131:134], v[131:132], off       758:  ds_store_b128 v129, v[131:134] offset:24576
1586: global_load_b128 v[173:176], v[132:133], off       (V's fill stores narrow: 8 x ds_store_b16)
```

Two adjacent load/store pairs of identical shape feeding the Q stage, with the destination in a register rather
than an immediate, are one rolled loop's body emitted twice: the per-warp `for (qr = 0; qr < 16; ++qr)` fill
declares no unroll factor, so LLVM picked one. `offset:24576` on the third site is `s_kv[0]` (three 8 KiB Q
stages ahead of it), so that one is the K fill and the fourth site is V's, which stores narrow — so **all three
source fills are present exactly once as logical traffic, and the surplus is +1 body on the Q fill on both the
read side and the write side.** That is why the two errors are the same 16 B: `+16 B global_read_B`,
`+16 B lds_write_B`, never a new destination.

**Why this is not absorbable, and what it costs.** The census compares *bytes demanded by the source* with
*sites emitted*, one site per source access; that basis is sound for a loop whose unroll factor the source
declares (the score leg's `#pragma unroll 1`, which `geometry()` reads) and unsound for a rolled loop whose
factor is a codegen choice. The cost of leaving it red is real and measured by the selftest: two injections
that must go red — removing one `ds_store_b128` and removing one `global_load_b128` — now land exactly on the
surplus and pass. **A surplus that swallows a missing write is not a harmless off-by-one, it is a blind control**
(the `s_wait_lgkmcnt0` "≥ 3 floor" line is the same class, which is why the 13 injections exist).

**The two honest resolutions, both open, neither of which is a tolerance:**
1. *Source declares the contract:* put an explicit unroll factor on the fill loops and let `geometry()` read it,
   exactly as it already reads the score leg's `#pragma unroll 1`. This is consistent with the census's own
   precedent and makes the site basis well-defined — **but it changes codegen on a shipped kernel**, so it owes a
   note-block re-take (`178 / 0 / 0 / 32768`), a bit-parity run, and a fresh timing before §33's numbers could be
   cited against it. Not free, and not a worker's call.
2. *Re-basis the fill terms:* check the fills by **destination coverage** (every stage region written exactly
   once, which is what `mixed_bw` learned to verify — bytes touched, not arithmetic) and keep the exact
   byte-for-byte comparison only where the source declares its unrolling. This changes no kernel and keeps both
   escaped injections fatal, because a missing write removes a destination whether or not a body was duplicated.
