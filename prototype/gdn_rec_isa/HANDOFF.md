# gdn_rec_isa — handoff

CPU/ISA census of the shipping `gdn_rec` family, modelled on
`prototype/attn_valu_slim/`. Findings and the pricing are in `results.md` (same directory).
Nothing here was run on a device; no timing was taken by this worker. Read `results.md` first.

## State of the tree when this was written

* `engine/`, `prototype/attn_m2/`, `prototype/issue_rate_probe/` — untouched.
* New files, all under `prototype/gdn_rec_isa/`:
  * `isa_census.py` — the census. 76 patterns, each self-tested against its own example line
    (`rules … selftest PASS`). Imports the shipped-classifier machinery from
    `prototype/attn_score_dot2/isa_census.py` through `prototype/attn_valu_slim/isa_census.py`;
    nothing was copied, and the pipe table is the base's (which fixes the `v_movrels_b32`-is-SALU
    mistake `attn_valu_slim` §4 corrects).
  * `gdn_rec_isa_kernels.h` — MODE 0 `rec0_kernel` (the shipped `rec_kernel` copied character for
    character), MODE 1 `rec1_kernel`, MODE 2 `rec2_kernel`, MODE 2 control `rec2c_kernel`.
    MODE 3 (norms out of the launch) is described in the header comment and **not implemented**.
  * `isa_tu.cpp` — compile-only TU that instantiates all four modes.
  * `isa_census.output.txt` / `.s` — the census run and the listing it read.
  * `rec_kernel.ship.s` — the shipped listing, extracted from the engine TU, kept so the numbers
    here can be re-diffed without a rebuild.
* `tmp/gdn_isa/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s` — the engine TU built with
  `--save-temps` (scratch, disposable).
* No harness `.hip.cpp` was written — the bit-identity burst A/B in §3 below is written as commands
  to run, but **the harness that runs them does not exist yet** (see "What is not built").

## What the census says (one line)

A token-step — one `rec_kernel` launch, one (token, GDN layer), 16 blocks × 128 threads, wave32 —
costs **1,523 issue slots per wavefront**, 97,472 per launch; the launch's own issue floor is
~0.7 µs against a measured 6.8–6.9 µs, so the family runs at ~10 % of its issue roof and
**instruction surgery is not the wall**. The xlane spill attention had *is* here (195 slots,
12.8 % of the bill, same instruction names) and is already 10× too small to pay. Details, numbers
and the falsifier: `results.md`.

## Commands for the main agent

Every device command here is run blocking, zero subagents, under `tools/gpu_lock.sh`.
`isa_census.py` is CPU-only and safe to run any time.

### 1. The census itself (no device)

```bash
.venv/bin/python prototype/gdn_rec_isa/isa_census.py \
    --out prototype/gdn_rec_isa/isa_census.output.txt \
    --engine-asm tmp/gdn_isa/kernels_gdn-hip-amdgcn-amd-amdhsa-gfx1151.s
```

If `tmp/gdn_isa/` is gone, rebuild the engine listing first (`prototype/gdn_decode`'s
`--save-temps`; **`--save-temps` writes next to the `-o` path**, so pass a `-o` under `tmp/`):

```bash
mkdir -p tmp/gdn_isa && /opt/rocm/bin/hipcc -x hip --offload-arch=gfx1151 -O3 -Wl,-rpath,/opt/rocm/lib \
    -I engine -I prototype/gdn_decode --save-temps engine/kernels_gdn.cpp -o tmp/gdn_isa/kernels_gdn.o
```

Useful flags: `--dump loopB --kernel rec_kernel` (full instruction dump of a region),
`--categories` (every mnemonic seen, classified or not), `--engine-asm ''` to skip the cross-TU row.
`--help` prints the unit definition and the trip-count derivation.

### 2. Compile check for the modes (no device, ~60 s)

```bash
/opt/rocm/bin/hipcc --offload-arch=gfx1151 -O3 -I prototype/gdn_rec_isa \
    -c prototype/gdn_rec_isa/isa_tu.cpp -o /tmp/isatu.o
```

Passes as of this writing. This is the only check that the hand-written `ds_bpermute_b32` /
`ds_swizzle_b32` operand forms assemble.

### 3. The A/B that decides the lever (device — **harness does not exist yet**)

What is needed: a `gdn_rec_isa.hip.cpp` that, for the same input `u` tile and the same initial
state, runs `rec_kernel` (control) and `rec2_kernel` over a serial run of T tokens on **separate
state buffers**, then compares `o` and the final state **bit for bit**, and bursts each mode
(N launches, one sync, wall/N — `steady_clock`, not `hipEventElapsedTime`, per
`bench/performance-status.md` §6). `prototype/gdn_collapse/` has a harness of that shape to copy
from. Run each mode in its own forked child (`prototype/m2_launch/`'s `run_dev_case` lesson).

The gate, in the tree's words: **bit-identity must be exact** (`o` and state, 0 differing elements,
T = 128 tokens, several seeds) — MODEs 1/2 preserve each value's addition tree and the compiler's
own out-of-range lane rule, so any difference is a bug, not a band. Then the burst: MODE 2 vs
MODE 0 in one lock hold, `util` recorded, same buffers. `results.md` §4 states what kills it
(< 1.02× — the lever is dead; ≥ 1.10× — the barrier chain was on the critical path and the census's
issue-bound ceiling was the wrong model, in which case the prologue is worth re-censusing after
moving the norms out of the launch, which is MODE 3 and is not written).

### 4. If it survives, the counters that explain it

```bash
tools/gpu_lock.sh status
/opt/rocm/bin/rocprofv3 --kernel-name-recapture 0 -f csv -k "rec*_kernel" -- \
    counters --token-batch 1024 --token-seq 1
```

`SQ_WAIT_ANY` against `SQ_BUSY`, plus `SQ_LDS_BANK_CONFLICT` and `SQ_LDS_ACTIVE`. Two things the
census could not decide and this can: whether the wavefront is waiting on LDS or on VMEM, and
whether the 512-byte-stride LDS coefficient read (64 distinct banks × 16 B) actually conflicts.

## What is *not* built, and the two decisions sitting with the main agent

1. **The harness.** Commands in §3 are written but nothing runs them. Either build it, or drop the
   lever on the pricing alone (my reading of `results.md` §4 is that the ceiling is under the tree's
   kill line and the honest move is to record the null and stop).
2. **MODE 3 (norms out of the launch)** is specified in the header but unwritten, because MODE 2
   has to survive first, and because moving the norms changes the launch *count* — which is exactly
   the lever `prototype/m2_launch` D22 measured and falsified for this family (removing 33,792
   launches bought 13 ms of 125). Say the word if you want it written anyway.
3. `.gitignore`: `prototype/gdn_rec_isa/gdn_rec_isa` (a future binary) is **not** ignored, and I did
   not edit `.gitignore` because the main agent was editing it concurrently. Per the trap noted
   there, ignore by exact path, and re-check `git status --ignored` after adding the op's files —
   the census/kernels/TU/output files under `prototype/gdn_rec_isa/` need adding too.
4. Nothing is committed. `git status` will show `prototype/gdn_rec_isa/` untracked and
   `tmp/gdn_isa/` (scratch — disposable, or keep it as the census's input).

## Corrections to anything I said verbally earlier

* The SIMD count: `devices/strix-halo-compute.md`'s own `hipGetDeviceProperties` print says 20 CUs
  (a reporting artifact — use 40), and `rocminfo` there prints "SIMDs per CU: 2" while the roofline
  section of the same file uses 128 FP32 lanes/CU (= 4 × 32). `results.md` §3 carries the roofline
  both ways, because the conclusion (≤1 wavefront per SIMD for a 16-block launch) is the same under
  both readings. I did not resolve which is right and did not invent a third number.
* My first version of `shfl_down_bits` clamped the out-of-range source lane to 31. That is **not**
  what the shipped lowering does and would have changed lane 0's sum; the committed version emits
  the compiler's own clamp (`delta -> 0`, i.e. read your own lane) and keeps the range-select, which
  is why MODE 1 saves less than the raw guard count suggests. Written up at the top of
  `gdn_rec_isa_kernels.h`.
