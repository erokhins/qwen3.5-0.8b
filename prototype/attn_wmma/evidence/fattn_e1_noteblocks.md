# E1 note blocks: `attn_fattn_wave` with Q staged (shipped) vs Q read from global (un-staged)

Compile-only and driver-fact-only. **No kernel was launched and no timing was taken**, by anyone, in this file.
Taken 2026-10-19 on the tip `bb3ec13`, working tree otherwise clean.

## 1. The commands, verbatim

```
# (0) the tree builds, both times — shipped and variant
engine/build.sh                                   # 17.4 s, built engine/build/{loader_test,decode_run}

# (1) the listing, per side. `--save-temps` writes the device .s into the CWD named after the SOURCE STEM,
#     not after -o, which is why the file below is kernels_attn-*.s and not k-*.s.
mkdir -p /tmp/e1base && cd /tmp/e1base
hipcc -O3 -Wno-unused-command-line-argument -I$REPO -I$REPO/engine --offload-arch=gfx1151 --save-temps \
    -c $REPO/engine/kernels_attn.cpp -o k.o
# variant: same command after `git apply prototype/attn_wmma/evidence/fattn_unstage_q.diff`, into /tmp/e1qg

# (2) the note block and the derived residency (KFD LDS + rocminfo ceilings; never hipGetDeviceProperties,
#     never clang's constant `Occupancy [waves/SIMD]` line)
python3 prototype/attn_wmma/fattn_residency_read.py --asm /tmp/e1base/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s
python3 prototype/attn_wmma/fattn_residency_read.py --asm /tmp/e1qg/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s
python3 prototype/attn_wmma/fattn_residency_read.py --asm ... --all      # the whole-TU drift table
```

## 2. The device facts actually read

```
$ grep -E "lds_size_in_kb|simd_per_cu|simd_count" /sys/devices/virtual/kfd/kfd/topology/nodes/1/properties
simd_count 80
lds_size_in_kb 64
simd_per_cu 2

$ rocminfo | sed -n '85,135p'          # Agent 2, gfx1151
  Compute Unit:            40
  SIMDs per CU:            2
  Max Waves Per CU:        32(0x20)
  Max Work-item Per CU:    1024(0x400)
  Workgroup Max Size:      1024(0x400)
  Wavefront Size:          32(0x20)
```

So: **LDS = 65,536 B/CU** (KFD), **40 CUs** (rocminfo `Compute Unit`, agreeing with KFD's `simd_count 80 /
simd_per_cu 2`), **32 wavefronts/CU** and **1,024 work-items/CU** ceilings. The **VGPR file per CU is unknown on
this part** (`prototype/attn_wmma/fattn_regfile_ladder.md` §3) and is therefore *not* applied anywhere below.

## 3. The two note blocks, verbatim from the `.s`

Shipped body (Q staged per warp), `/tmp/e1base/...gfx1151.s` lines 15074-15091:

```
    .group_segment_fixed_size: 32768
    .kernarg_segment_size: 456
    .max_flat_workgroup_size: 96
    .name:           _ZN6engine7kernels15attn_fattn_waveE8CoreArgs
    .private_segment_fixed_size: 0
    .sgpr_count:     24
    .sgpr_spill_count: 0
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     178
    .vgpr_spill_count: 0
    .wavefront_size: 32
```

Un-staged body (Q read from global at the use site), `/tmp/e1qg/...gfx1151.s` lines 15027-15045:

```
    .group_segment_fixed_size: 32768
    .kernarg_segment_size: 456
    .max_flat_workgroup_size: 96
    .name:           _ZN6engine7kernels15attn_fattn_waveE8CoreArgs
    .private_segment_fixed_size: 0
    .sgpr_count:     27
    .sgpr_spill_count: 0
    .uniform_work_group_size: 1
    .uses_dynamic_stack: false
    .vgpr_count:     184
    .vgpr_spill_count: 0
    .wavefront_size: 32
```

## 4. The residency read, both sides

```
$ python3 prototype/attn_wmma/fattn_residency_read.py --asm /tmp/e1base/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s
listing                : /tmp/e1base/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s
kernel                 : attn_fattn_wave
  vgpr_count               178
  vgpr_spill_count         0
  sgpr_count               24
  sgpr_spill_count         0
  private_segment_fixed_size 0
  group_segment_fixed_size 32768
  block threads            96
  block warps              3
device facts (must be re-read to re-take this):
  LDS per CU             : 65536 B  [KFD nodes/1/properties lds_size_in_kb 64]
  work items per CU      : 1024   [rocminfo 'Max Work-item Per CU']
  waves per CU           : 32   [rocminfo 'Max Waves Per CU']
  register file per CU   : UNKNOWN on this part (fattn_regfile_ladder.md §3) — not applied
derived blocks/CU        : 2   (LDS=2  threads=10  wavefronts=10)
  binding limit(s)       : LDS
derived wf/CU            : 6   (= blocks/CU 2 x warps/block 3)
  LDS per block / budget : 32768 B / 65536 B  = 0.50 of the CU's LDS
```

```
$ python3 prototype/attn_wmma/fattn_residency_read.py --asm /tmp/e1qg/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s
listing                : /tmp/e1qg/kernels_attn-hip-amdgcn-amd-amdhsa-gfx1151.s
kernel                 : attn_fattn_wave
  vgpr_count               184
  vgpr_spill_count         0
  sgpr_count               27
  sgpr_spill_count         0
  private_segment_fixed_size 0
  group_segment_fixed_size 32768
  block threads            96
  block warps              3
device facts (must be re-read to re-take this):
  LDS per CU             : 65536 B  [KFD nodes/1/properties lds_size_in_kb 64]
  work items per CU      : 1024   [rocminfo 'Max Work-item Per CU']
  waves per CU           : 32   [rocminfo 'Max Waves Per CU']
  register file per CU   : UNKNOWN on this part (fattn_regfile_ladder.md §3) — not applied
derived blocks/CU        : 2   (LDS=2  threads=10  wavefronts=10)
  binding limit(s)       : LDS
derived wf/CU            : 6   (= blocks/CU 2 x warps/block 3)
  LDS per block / budget : 32768 B / 65536 B  = 0.50 of the CU's LDS
```

The shipped side reproduces §33's banner exactly (178 / 0 / 0 / 32768), which is what licenses comparing the
variant against it.

## 5. Drift check: only `attn_fattn_wave` moved

`--all` over both listings, diffed. One line differs, and it is the one the change is required to move:

```
$ diff <(... --asm /tmp/e1base/... --all) <(... --asm /tmp/e1qg/... --all)
<   attn_fattn_wave   vgpr_count=178  vgpr_spill_count=0  sgpr_count=24  sgpr_spill_count=0  private_segment_fixed_size=0  group_segment_fixed_size=32768
>   attn_fattn_wave   vgpr_count=184  vgpr_spill_count=0  sgpr_count=27  sgpr_spill_count=0  private_segment_fixed_size=0  group_segment_fixed_size=32768
```

The rest of the TU is byte-for-byte identical in the note section: `attn_core` 53 VGPR / 38,944 B LDS,
`attn_slim_wave` and `attn_tile_wave` 47 / 12,288, `attn_prep` 30 / 2,080, `attn_combine` 10 / 0,
`rope_table` 16 / 0 — all unchanged, spill 0 everywhere.

## 6. ISA witness that the change landed where it claims (static sites, one kernel body)

```
                     shipped   un-staged
global_load sites        4         4
ds_load sites           71        69
ds_store sites          77        75
v_cvt_f16_f32 sites     48        48
v_wmma sites            17        17
s_barrier sites         11        11
```

The score leg's SECOND operand register pair, from the first `v_wmma` of the rolled KTILES loop:

```
shipped   : 788: v_wmma_f32_16x16x16_f16 v[129:136], v[159:166], v[167:174], ...   # second operand from ds_load
un-staged : 705: global_load_b128 v[168:171], v[160:161], off
                706: global_load_b128 v[172:175], v[160:161], off offset:16
                753: v_wmma_f32_16x16x16_f16 v[129:136], v[160:167], v[168:175], ...   # second operand = v[168:175]
```

Two 16-byte global loads at `offset:16` apart — the 16 halves of one Q row's 16 dims — now feed the tile that the
LDS stage used to feed. `v_cvt_f16_f32` is unchanged at 48 sites, which is the witness that the BF16→F16
conversion merely **moved** (out of the once-per-launch fill, into the per-round loader) rather than being added
to the round; `ds_store` is down by 2 (the Q fill's stores are gone) and `ds_load` by 2 (the Q operand's LDS
reads). `v_wmma` and `s_barrier` counts are untouched, so the round's protocol did not change.

## 7. Where the LDS went, since it did not go anywhere

The block's declared LDS, from the declarations:

| form | Q | epilogue transpose staging | K/V | total |
|---|---|---|---|---|
| shipped | 3 × 8,192 B (`s_q`, one per warp) | rides the dead `s_q` — 16 rows × FATTN_TROW F32 = 8,192 B/warp | 8,192 B | **32,768 B** |
| un-staged | — | 3 × 8,192 B (`s_tr`, one per warp) | 8,192 B | **32,768 B** |

`FATTN_MROWS * FATTN_TROW * 4 == FATTN_MROWS * DH * 2` is the assert that states the equality: one warp's
epilogue staging is exactly one warp's Q stage. The Q stage was never *extra* LDS; the epilogue was borrowing it.
