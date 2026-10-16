# issue_rate_probe — handoff (worker run ended mid-task)

Nothing here is committed (`?? prototype/issue_rate_probe/`). No device run has been executed by the
worker: everything below was verified on the CPU only. No timing was taken, none is recorded here.

## What question this answers

`progress/2026-10-16/prefill-occupancy-results-2026-10-16.md` §5 item 1. Question A: the sustained
VALU warp-instruction rate per CU per clock (the shipped `attn_tile_wave` is at 95 % of a floor that
assumes 1 warp-inst/CU/clk; if the real number is 2, three funded routes flip). Question B: achievable
read GB/s against the size of the *resident* working set for KV-shaped access.

## Files

| file | state |
|---|---|
| `irp_kernels.h` | complete: `irp_fma_{1,4,8,16,32}`, `irp_serial_{8,32}`, `irp_cvt_{8,16}`, `irp_lds_tile`, `irp_kv_read`, `publish_block_sum`, fill kernels, guard-band constants |
| `isa_census.py` | compiles `--save-temps`, slices the ISA per kernel, finds the hot loop, emits `irp_isa.gen.h`. **Two loops are mis-picked, see BLOCKERS** |
| `irp_isa.gen.h` | generated, 11 `#define IRP_WI_*` |
| `issue_rate_probe.cpp` | host driver: plan → allocate → warm → burst → read accounts back → one `RESULT` line. `--selftest-geom` green (18/18), `--selftest` (device) **never run** |
| `build.sh` | written, **not run** — it now runs `isa_census.py --check`, which will fail until BLOCKER 1 is fixed |
| `run.py` | **NOT WRITTEN** |
| `README.md`, `results.md` | **NOT WRITTEN** |

## BLOCKERS — read before trusting any number from this directory

1. **The census bills the wrong loop for the two kernels that matter.** `irp_lds_tile` gets
   `WI=18` from the *staging* loop (6 `ds_store_b128` + barriers); the read-back loop is not in the
   disassembly at all — `grep -c ds_read_b16` over the kernel body is **0**, i.e. the LDS reads were
   hoisted out of the counted loop or the loop was rotated away. `irp_kv_read` gets `WI=36` from
   `publish_block_sum`'s reduction (it contains `v_add_f64`, `ds_load_b64`), not the row loop. So
   `warp_inst_total` is wrong for the LDS row of the table and for every KV row. `pick_hot` needs to
   bill the loop that carries the kernel's own memory traffic and to require the expected mnemonic
   per family (fma→`v_fma`, lds→`ds_read`+`ds_store`, kv→`global_load_b128`), refusing to emit if a
   family's signature is absent. The `A`-family counts (7/10/14/22/38 …) look right: `v_fma_f32 ×N`
   plus 6 loop-overhead instructions, and they scale linearly in N.
2. **`__syncthreads()` inside the counted LDS loop is inside the billed span.** The `s_barrier` +
   `buffer_gl0_inv` pairs are counted as instructions per iteration. That is honest for the shipped
   `attn_tile_wave` (it barriers per stage too), but the row must be labelled so a reader does not
   compare it against a barrier-free LDS number.
3. **Device path is untested.** `./issue_rate_probe --selftest` forks one child per case (the
   mixed_bw lesson: one faulting kernel poisons the context and every later case silently never ran)
   and has never been run on this box. Expect geometry or address faults in `irp_kv_read` first — it
   is the only kernel with real addressing.

## What is done well and should be kept

- **The disassembly caught two modeling errors before any GPU time was spent.** The first version
  seeded every lane with the same value and the compiler scalarized the whole loop into
  `s_fmac_f32`/`s_mov_b32` — 29 scalar instructions per iteration and not one VALU instruction. A
  computation every lane performs identically is uniform and RDNA scalarizes it. Fixed by a per-lane
  seed (`+ (float)threadIdx.x`), which is also attn's real shape (`q` per lane, the scale not).
- **Defects are separate builds, not runtime branches** (`-DIRP_NEG=<n> -DIRP_NEG_NAME='"<name>"'`),
  the `gdn_collapse` rule: a branch inside the timed loop would change the instruction stream this
  probe exists to count. `build.sh` builds the clean binary plus 7 defect binaries.
- **Two witnesses, and the blind set is derived, not hand-listed.** `--selftest-geom` derives
  `kv_alias` catchability from the geometry (`caught=40 blind=0` — in this grid each row has exactly
  one reader per pass, so an alias is always caught; the wish that it be blind was wrong and the
  assertion now states the derived fact) and derives the LDS permutation witness
  (`permutation=1 threads_that_differ=128`). The first LDS swap candidate, a row rotation, is
  **invisible to both witnesses** (`differ=0`) because with `nw=4` the warp row-cosets carry equal
  tag sums; the lane-swap (`off ^= 8`) is what makes it a real permutation the block sum cannot see.
- **The witness is an exact F32 integer** (`tag = 1 + e%7`, all values < 2^24) and `plan_build`
  refuses any `--iters`/`--passes` that would let a per-thread running total cross 2^24, so equality
  is exact rather than approximate.
- `publish_block_sum` is a real block reduction (shared-memory halving tree, one `double` per thread).
  The `mixed_bw` trap — `sums[blockIdx.x] = per_thread_acc`, a per-thread store to a per-block slot —
  is what this replaced.

## Commands for the main agent

Every device command below must run under `tools/gpu_lock.sh` in a window with zero subagents.
`run.py` does not exist yet, so the binary is invoked directly and **does not** take the lock
(`grep -c gpu_lock issue_rate_probe.cpp` = 0, and it must stay 0).

```bash
cd prototype/issue_rate_probe
python3 isa_census.py                                   # A0: the bill itself. Re-run after any edit
python3 isa_census.py --emit irp_isa.gen.h --check      # gate: counts must match the header
./issue_rate_probe --selftest-geom                      # CPU only, no device: must print ok, fails=0
./issue_rate_probe --dry-run --kernel irp_fma_32 --iters 20000
./issue_rate_probe --selftest                           # FIRST device run. <60 s. Correctness only.
```

Then, once BLOCKER 1 is fixed, the measurement rows (main agent, one lock hold, blocking):

```bash
# Question A: warp-inst per CU per clock. `--launches` big enough that dispatch is not the floor.
for k in irp_fma_1 irp_fma_4 irp_fma_8 irp_fma_16 irp_fma_32 irp_serial_8 irp_serial_32 \
         irp_cvt_8 irp_cvt_16; do
  ./issue_rate_probe --kernel $k --iters 20000 --blocks 320 --threads 128 --launches 20 --reps 5 \
      --cu-count 40 --clock-mhz 2894 --tag A_$k
done
# Expected shape: one line, `IRP tag=A_irp_fma_32 ... best_ms_per_launch=... icpcb=1.0000 ...`
# icpcb is warp-inst per CU per clock already; ~1.0 means the 1/CU/clk floor is real, ~2.0 halves it.
# Run the same list with --cu-count 20 as well: this box reads 20 CU in HIP and 40 in rocminfo.

# Question A, LDS row (blocked on BLOCKER 1):
./issue_rate_probe --kernel irp_lds_tile --iters 2000 --reuse 1 --launches 20 --reps 5
./issue_rate_probe --kernel irp_lds_tile --iters 2000 --reuse 4 --launches 20 --reps 5

# Question B: read GB/s against resident working-set size, attn's access shape.
for f in 2 8 32 128 512; do
  ./issue_rate_probe --kernel irp_kv_read --footprint-mib $f --passes 0 --min-launch-mib 40 \
      --blocks 320 --threads 128 --launches 6 --reps 5 --tag B_${f}MiB
done
# Expected shape: `IRP tag=B_32MiB ... footprint_mib=32.000 rows=65536 passes=... rd_mib_per_launch=...
# gbs=... pct_of_read_probe=... pct_of_resident_tier=... checksum=ok guards=intact suspect=0`
# `suspect=1` is printed by the binary itself when a KV row exceeds 240 GB/s: for a resident set that
# is the finding (devices/strix-halo-memory-bandwidth.md 0b reads 18 MiB at 918 GB/s), not a bug —
# but it is only believable with checksum=ok.
```

## Fixed at the very end of the run (do not redo)

- The KV family's billed iteration is ONE PASS over the rows, so `plan_build` now sets
  `p.iters = p.passes` and `mirror_kv_warp` takes the resolved pass count instead of `Cfg::passes`.
  Before this the KV plan printed `expected_block_sum=0 expected_total=0` — an account that can never
  fail, i.e. exactly the vacuous-check failure mode this repo keeps hitting.
  `--dry-run --kernel irp_kv_read --footprint-mib 32 --passes 0 --min-launch-mib 40` now reads
  `iters=2 rows=65536 passes=2 rd_mib_per_launch=64.000 expected_block_sum=421376 expected_total=134215168`.

## Restating BLOCKER 1 in terms of the two families, because the fix has to satisfy both

- `A` family (fma/serial/cvt): one counted loop, `iters` trips, `wi` from it. Currently right.
- `lds`: the billed iteration is stage-the-tile + re-read-it `reuse` times, i.e. the outer `i` loop.
  The census must bill THAT loop, and the read side currently does not exist in the ISA (hoisted).
  A compiler barrier (`asm volatile("" ::: "memory")`) at the top of the loop body, or making `reuse`
  a compile-time constant so the reads unroll into the billed span, are the two candidate fixes.
  Whichever is taken, re-run the census and confirm the billed span contains BOTH `ds_store_b128` and
  `ds_read_*`.
- `kv`: the billed iteration is one pass, whose body reads `rows/nwarps` rows. The census must bill
  the loop containing `global_load_b128`, not the block-reduction loop it currently picks, and its
  per-iteration instruction count then depends on `rows/nwarps` — so for this family the honest bill is
  `insts_per_row × rows/nwarps`, with `insts_per_row` read out of the disassembly of the row loop.
  Print the row-loop histogram next to it so a reader can check the arithmetic.
