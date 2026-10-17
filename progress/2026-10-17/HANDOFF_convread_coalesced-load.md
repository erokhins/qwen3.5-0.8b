# HANDOFF — conv read: give `conv_kernel` a coalesced read of its BF16 operands

Task: recover the ~6.5 ms that `62b73e9` gave back (§13 of `bench/performance-status.md`: `gdn_conv`
19.9876 → 26.4833 ms at 4096 lanes, +32.5 %, against a predicted −22 %). Base `9e2d19e`. **No commit
made, nothing staged.** Three source files modified; everything else is scratch under `tmp/conv_isa/`.

Provenance note first: `progress/2026-10-17/HANDOFF_17prime.md`, named as step 1 reading, **does not
exist** in the tree (not in `progress/2026-10-17/`, not in git history) although §13 and D39 cite it.
The measured facts came from §13 and D39 directly; the missing file is a gap in the record, not a gap
in this work.

## 1. ISA attribution — what the listing says the regression is

Generated CPU-only (`hipcc -O3 -I. -Iengine --offload-arch=gfx1151 --save-temps -c
engine/kernels_gdn.cpp`), listings under `tmp/conv_isa/`: `kernels_gdn_base.s` (from `62b73e9^`, the
F32-tile form), `kernels_gdn.s` (HEAD, per-channel BF16 — the regressed form), `kernels_gdn_new.s`
(the change below). Instruments: `meta.py` (YAML `.amdgpu_metadata`; the old directive regex matched
0 kernels, so it now counts what it parsed and exits 1 on a name it did not find), `waitmap.py`
(first `s_waitcnt` after each global load), `blockmap.py`.

HEAD's `conv_kernel` (941 lines, `conv.s`) vs the base it replaced (910 lines, `conv_base.s`):

| | base (F32 tile) | HEAD (BF16, one channel per lane) |
|---|---|---|
| tap loads | 33 `global_load_b32` (4 B/lane = 128 B per wavefront) | 23 `global_load_u16` (2 B/lane = **64 B, half a line**) |
| widening | none | one `v_lshlrev_b32_e32 v?,16,v?` per tap |
| wait shape | `s_waitcnt vmcnt(3)(2)(1)(0)` — **a batch of 4 taps in flight** | **`s_waitcnt vmcnt(0)` immediately after each tap load** — 29 immediate drains vs 22 |
| LDS | none (`ds_*` = 0) | none |

SUPPORTED by the listing, and the cause:
1. the operand load is 2 bytes per lane, so each wavefront requests half a cache line;
2. **memory-level parallelism is gone.** The widening is the load's only consumer and sits in the same
   exec-guarded tap branch, so LLVM must take the wait there and emits the conservative full drain:
   the K taps of one output became K separate memory round trips. The base issued its four `b32` loads
   back to back and drained them with one counted group.

RULED OUT by the same listing: bank conflicts (0 `ds_*` in either form — the kernel touches no LDS);
the ALU cost of the widening (one shift per tap, ~4 of a ~250-slot body, cannot be 6.5 ms); read
amplification (a 256-thread block covers its tap rows contiguously, no line fetched twice — the bytes
went *down* while the time went up, so bytes were never the limiter: base ≈181 GB/s vs new ≈102 GB/s
on the traffic arithmetic, i.e. the regressed form is not even DRAM-bound); and "the compiler failed to
vectorise for lack of an alignment proof" — under one-channel-per-lane there is nothing to merge, the
four operands are 12 288 B apart. **The thread map was the blocker, not an alignment proof.**

## 2. The change (one code path, injections left live on it)

`CONV_VEC = 4`: a lane owns 4 *consecutive* channels and fetches a tile row as ONE aligned word.

- `prototype/gdn_decode/gdn_decode_kernels.h` — `conv_tap`/`conv_value` take a chunk (out-params)
  instead of a channel; `conv_value` fetches all K rows before consuming any (the batching half);
  `conv_kernel` maps `c0 = (blockIdx.x*blockDim.x + threadIdx.x) * CONV_VEC`. Two aligned aggregate
  types make the width real: `struct __attribute__((aligned(8))) ConvCols { hip_bfloat16 h[4]; }`
  (natural align 2 would force 4 × 2-byte loads no matter how the source reads) and the same F32 chunk
  at `aligned(16)`. `static_assert(CH % (256*CONV_VEC) == 0)` + `CH % CONV_VEC == 0`; the wide access
  is legal because every buffer is `hipMalloc`'d at 256 B and a row is 12 288/24 576 B.
  `GC_TAP_STRIDE` keeps a scalar gather **inside the same body** — that injection is a different
  address, not a different width — so all four negatives still run on the shipping path; there is no
  fast/slow split for the injections to hide behind. `gates_kernel` untouched (it reads two values per
  thread and shares none of this).
- `engine/kernels_gdn.cpp` — grid `dim3(CH/(256*CONV_VEC), tokens)` (24 blocks → 6) + the same
  `static_assert`; comment updated.
- `prototype/gdn_decode/gdn_decode.hip.cpp` — the op harness's grid, same divisor.
- New ISA (same recipe as above): `global_load_b64` ×19 (the packed tile rows, 8 B/lane = 256 B per
  wavefront), `global_load_b128` ×20 (window rows + the `u`/`new_win` stores), 16 remaining
  `global_load_u16` all in the `channel_major` injected arms and **batched there in groups of four with
  counted `vmcnt(3)…vmcnt(0)`**; `global_store_b128` ×4; VGPR 17 → **42, `vgpr_spill=0`**; LDS 0.
  Per output element the loads went 4 → 1 and the drains 4 → 1 at 4× the width.

## 3. Bit-identity — the whole admission — is proved twice

**(a) CPU differential, no device, no lock:** `tmp/conv_isa/host_diff/run.sh` (rerunnable). It
compiles BOTH spellings of the kernel header as host code — the old one re-extracted with
`git show 9e2d19e:…` on every run, the new one from the working tree — through a `hip/` stub so the
*real source text* is compiled rather than a mirror (a mirror agrees with itself and proves nothing),
links one side per binary to avoid the ODR collision that would otherwise compare one kernel with
itself, runs `conv_kernel` as a host loop over each side's own grid, and adjudicates three ways
(`cmp.py`): old-vs-new bit-for-bit; a sentinel prefill so an unwritten element cannot pass vacuously
(the new grid is 4× narrower — exactly the change that could strand a column); and per-injection
must-move/must-not-move expectations **derived from geometry**, which reproduce results.md's P6 table
(`tap_stride` invisible at tokens=1 / all 46 rows at 46; `win_lag`/`win_from_u` move the window only;
`zero_pad` confined to rows 0..2, 0 elements at rows 3..45). Result:

    CONV DIFFERENTIAL PASS — bit-identical on every case and element, every element written,
    every injection fires exactly where it lives on both sides.   (tokens ∈ {1,46} × neg ∈ 0..4)

**(b) Device gate 1, run** — `tmp/conv_isa/gate1.sh p64`, i.e.

    tools/gpu_lock.sh acquire "convread gate1 gdn_decode wave gate"; tools/gpu_lock.sh util
    .venv/bin/python prototype/gdn_decode/check.py --tokens p64
    tools/gpu_lock.sh util; tools/gpu_lock.sh release …

`rc=0`, **wall 4 s**, `gpu use: 7 %` at the time (lock free before and after), **correctness only, no
timing taken**. It printed: `q/k/v_post_conv` `max|d| 0 … bitexact=True`; `new window == raw[T-3..T-1]`
bitexact; two-wave boundaries at rows 39 and 41 bitexact both ways; `T=1 launches == the wide launch,
row by row — bitexact=True` (9 probed rows); `tap_stride` 46/46 rows moved with the window unchanged;
`win_lag`/`win_from_u` with `u` unchanged and the window moved; `zero_pad` rows [0,1,2] only.

## 4. What the user must run (device pass 2 — NOT run by me)

**Device gate 2, shipping path.** The stored pre-change dumps are a valid witness and already agree
with each other (`md5sum …/logits499.bin` = `6025e97f8b902aaeab86adeeeb56114e` in both
`tmp/gcc/new42-wide520f-lt` and `tmp/gcc/w-new-wide520f-lt`, 25 `logits*.bin` each; the prior wave's
`tmp/gcc/diff-*-lt.txt` files are 0 bytes = its base/new sides were bit-identical). Run the wide520f
argv from `tmp/gcc/argv_wide520f.sh` with only `--out` changed, under the lock, blocking:

    tools/gpu_lock.sh acquire "convread gate2 wide520f"; tools/gpu_lock.sh util
    bash -c "$(sed 's#--out tmp/gcc/new42-wide520f-lt#--out tmp/conv_isa/g2-new#' tmp/gcc/argv_wide520f.sh)"
    for f in tmp/gcc/new42-wide520f-lt/logits*.bin; do cmp "$f" "tmp/conv_isa/g2-new/$(basename $f)"; done
    tools/gpu_lock.sh util; tools/gpu_lock.sh release "convread gate2 wide520f"

Expect **no output from `cmp`** (bit-identical to the pre-change dumps). If it differs, stop and
attribute before concluding — a stored witness can also be stale in ways unrelated to this change.
`dec64f` (`tmp/gcc/argv_dec64f.sh`, witnesses `tmp/gcc/w-new-dec64f-lt`) is the third run if the
decode shape is in question too.

**Paired A/B (user, in one lock hold per side, with the canary asserted in-window).** §13's `cast`
change moved the launch bill (424→406, cast 115→97); **this change moves no bill field**, so the 1.7′
canary cannot detect a stale binary here — both sides print `cast=97 other=20 total=406`. Use a
structural canary on the exact binaries being timed, asserted before any timing, e.g. for the `new`
side the extracted `conv_kernel` listing must contain `global_load_b64` and no `global_load_u16` in
its hot blocks, and for `base` it must show 23 `global_load_u16` and no `b64`
(`awk '/^_Z11conv_kernel/{p=1} p&&/^\.Lfunc_end/{exit} p' <side>.s | grep -c global_load_b64`).
Template to follow: `tmp/gcc/run_ab_17p_retake.sh`. Family row to read: `gdn_conv` at 4096 lanes.

**The two-sided prediction, written down before the measurement.** If the attribution is right, the
4096-lane paired A/B shows **both**: `gdn_conv` back to ≈19–20 ms (from 26.48, i.e. ≥ −6 ms), **and**
wave wall ≥ 4 ms lower at 4096 than the §13 landing. If `gdn_conv` lands materially above 20 ms, the
row does not move and **the item is dead** — §13 then needs a further correction saying the coalescing
layout was not the loss. Decode is predicted flat (this is a per-launch shape change inside a launch
that was already launch-floor bound); watch `gdn_conv` decode, not the wall, for the thread-count note
below.

## 5. Residual risks / named follow-ups (not decisions I made)

- **The remaining LSUoperand is the conv weights.** `C[(c0+i)*K+j]` is still 4 scalar `b32` loads per
  column = 24 576 loads per token row vs 6 144 packed tap loads, i.e. ~76 % of the kernel's load
  instructions. The same chunk argument covers it exactly (16 consecutive floats per lane, 16 B
  aligned) and it would cut the LSU count again ~2.3×; deliberately left out of scope (the named cause
  was the BF16 operand) and it is the first thing to price if the A/B lands short.
- **Row loads are still one-per-drain inside a chunk** (LLVM places the widening in the load's block).
  Per output that is 1 drain instead of the base's 1 and HEAD's 4, so it is no worse than the base on
  exposed latency; batching the four rows of one chunk would need a hot/cold split of the tap rule and
  I did not add a second spelling of it.
- **Decode parallelism**: `tokens == 1` now runs 6 blocks × 256 threads instead of 24 × 256. Work per
  thread is 4× the instruction count smaller, but if the decode `gdn_conv` row regresses, this is the
  suspect and `CONV_VEC = 2` (12 blocks, 4 B/lane = the base's transaction width) is the dial.
- VGPR 17 → 42 (no spills; 5 blocks/CU at 256 threads), listing 941 → 2112 lines from the ×4 unroll.
- Not touched, per instructions: `specification/`, `bench/`, `devices/`, `progress/`, `engine/gate.py`.
  No band, no timing, no commit. `prototype/gdn_decode/results.md` has **no** new section yet — the ISA
  attribution above belongs there (its sections are named, not numbered; last is "## P6: …"), and the
  kernel comments deliberately cite no section that does not exist.

## 6. State

- Builds: `./prototype/gdn_decode/build.sh` and `./engine/build.sh` both green after the change
  (engine build 17 s).
- `git status --porcelain`: ` M engine/kernels_gdn.cpp`, ` M prototype/gdn_decode/gdn_decode.hip.cpp`,
  ` M prototype/gdn_decode/gdn_decode_kernels.h`. Scratch: `tmp/conv_isa/` (listings, `meta.py`,
  `waitmap.py`, `blockmap.py`, `host_diff/`, `gate1.sh`). Nothing staged, nothing committed.
- Remaining work for the next turn: device gate 2 + the A/B (both above); the `results.md` section; and
  the §13 correction, which is the user's call once the A/B number exists.
