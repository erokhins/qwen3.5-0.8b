# `issue_rate_probe` — what an arithmetic instruction actually sustains on gfx1151

## Why this directory exists

Three facts, each proved elsewhere in this repo (`tools/isa_probe.sh`,
`devices/strix-halo-official-specs.md` §7, decision log D13/D14):

* there is **no MFMA** on `gfx1151`, and none of the 55 code objects shipped with rocBLAS contains a
  matrix instruction;
* `v_wmma_f32_16x16x16_f16` is the only matrix instruction, and it *is* reachable from our compiler;
* our bf16 path today executes scalar `v_fmac_f32_e32` at **1 MAC/lane/clk**, while the WMMA roof is
  **4 MAC/lane/clk** — four times the vector roof.

So the number this tool exists to produce is: **what fraction of the WMMA roof does a hand-written
WMMA K-loop actually sustain, and where do `v_pk_fma_f16`, `v_dot2_f32_{f16,bf16}`, `v_fma_mix_f32`,
`v_dot4_i32_i8` and `v_dot8_i32_i4` sit between the scalar floor and that roof.** Everything else
here (instruction-per-issue calibration, the KV-shaped read sweep, the LDS tile) supports that
question or is a second-order row; the ordering is deliberate.

## What it measures, and what it refuses to

| measured | refused |
|---|---|
| sustained warp-instructions per CU per clock for a loop whose billed instruction is proved present in the disassembly | any bill taken from the source text. `isa_census.py` reads the counts out of the disassembly and refuses a kernel whose required mnemonic is not inside a loop. |
| MACs per lane per clock and `%` of the booked ceiling, with the conversion printed on the same line | a leg whose op got hoisted, folded, or never emitted. That leg exits non-zero and `run.py` SKIPs it with the census reason. `irp_lds_tile` is in that state now. |
| a witness equality (block sum + per-thread `each`) computed on the host *before* the launch and compared after it | a number from a kernel that did not do its work: `checksum=FAIL`, `bad_blocks>0` or a short launch makes `run.py` exit instead of printing the row. |
| read GB/s against a **resident** KV working set (2…512 MiB) | a GB/s figure above ~240 GB/s believed without the resident-set argument (`devices/strix-halo-memory-bandwidth.md`: an 18 MiB set reads at 918 GB/s). The row prints `suspect=1` and says so. |

## Files

| file | role |
|---|---|
| `issue_rate_probe.cpp` | the driver: plan (grid, bill, expected sums), one timed run per invocation, `--selftest`, `--selftest-geom`, `--wmma-check`. |
| `irp_kernels.h` | traffic kernels: `irp_kv_read` (one warp per 512 B KV row), `irp_lds_tile` (**unbilled**, see below), `irp_fma_*`, `irp_serial_*`, `irp_cvt_*`, plus the defect injections. |
| `irp_arith_kernels.h` | one kernel per arithmetic path, all one shape: 8 independent accumulators, one op per chain per iteration, inside a `_Pragma("unroll 1")` loop. How each mnemonic has to be spelled to reach it (inline asm where the frontend cannot) is recorded in that header. |
| `isa_census.py` | compile-only. Bills each kernel's loop from the disassembly against the mnemonics that kernel declares it needs; `--emit` writes `irp_isa.gen.h`, `--check` compares with `run.py:DECLARED`. |
| `run.py` | the driver wrapper. `--verify` (host-only), `--commands` (the device window), `--device` (main agent). Holds **no** GPU lock — the only `gpu_lock` text in the file is the comment saying it must not be called. |
| `irp_isa.gen.h` | generated: `IRP_WI_*` (warp-inst/iter), `IRP_OPS_*` (ops/iter), `IRP_UNBILLED_*`. |
| `results.md` | the ledger rows. Written by the main agent from a clean window; a worker must not fill it. |

## Build and census flow

```bash
bash build.sh            # census --emit + --check, then the binary + 7 defect binaries
python3 run.py --verify  # host-only: bill, mirrors, plan for every leg at the --iters it will use
```

The census bills a loop only if that loop contains every mnemonic the kernel declares it needs
(e.g. `irp_d2bf16_8` demands eight `v_dot2_f32_bf16`). A kernel that satisfies none of its
requirements makes the build fail with the per-loop counts printed, which is the whole point of the
rewrite: the first version billed `irp_lds_tile`'s staging loop and `irp_kv_read`'s block-reduction
loop and produced two plausible, meaningless numbers.

`irp_lds_tile` is currently listed in `build.sh`'s `--allow-unbilled`, which emits a **0** bill for
it; `run.py` then refuses to run that leg and prints why. Any other kernel that cannot be billed
still fails the build. Deleting the leg is preferable to billing it wrongly, and a 0 with a reason is
how that deletion is recorded without losing the kernel.

## The conversion, printed on every row

```
warp_inst_total   = icpcb * ...      one warp-instruction = 64 lane-slots of work
flop_per_s        = warp_inst_total * 64 * macs_per_lane_per_op * 2 / secs
mac_per_lane_clk  = flop_per_s / (2 * cu_count * 64 * clock_hz)
pct_of_ceiling    = 100 * mac_per_lane_clk / ceiling_mac_per_lane_clk
crosscheck        = icpcb * macs_per_lane_per_op        # must equal mac_per_lane_clk
```

The scalar FMA roof is `icpcb = 1.0000` → 1.0 MAC/lane/clk; the WMMA roof is `icpcb = 0.03125`
(one `v_wmma` per 32 issued slots) → 4.0 MAC/lane/clk. If the cross-check column disagrees with
`mac_per_lane_clk`, the booked `macs_per_lane_per_op` is wrong, not the machine.
`cu_count` defaults to 40 and `clock_mhz` to 2894; this box disagrees with itself (HIP reports 20
CUs on some paths), so both are flags and both are printed.

Occupancy is printed with every row: `waves_requested_per_cu`, `resident_waves_per_cu` (from
`hipOccupancyMaxActiveBlocksPerMultiprocessor`) and the hardware `max_waves_per_cu=32`. A leg running
at 2 resident waves is latency-bound, not issue-bound, and its `%` of ceiling must be read that way.

## Ceilings booked, and the caveats that belong to each

| leg | mnemonic | MAC/lane/clk booked | caveat |
|---|---|---|---|
| `irp_f32_8` | `v_fma_f32` / `v_fmac_f32_e32` | 1 | the floor our bf16 path actually runs at |
| `irp_mix_8` | `v_fma_mix_f32` | 1 | bf16-in/f32-out. Its operand semantics are ambiguous (one product or a 2-element dot), so the `b` operand carries a zero in its high bf16 half and the witness increment is exactly 1 **under either reading** |
| `irp_pkf16_8` | `v_pk_fma_f16` | 2 | fp16 accumulator: exact integers only to 2^11, so `--iters` is capped by the plan, not by taste |
| `irp_d2f16_8` | `v_dot2_f32_f16` | 2 | plain form, f32 accumulator (the `_acc` forms change the accumulate semantics and the per-op MAC count) |
| `irp_d2bf16_8` | `v_dot2_f32_bf16` | 2 | the only bf16 2-MAC path on this part; there is **no** `v_pk_fma_bf16` |
| `irp_d4i8_8`, `irp_u4u8_8` | `v_dot4_i32_i8`, `v_dot4_u32_u8` | 4 | **raw dot rate.** A real int8 GEMM adds zero-point/dequant/scale work and the accumulator's sign handling; this row is the ceiling for the dot alone |
| `irp_d8i4_8`, `irp_u8u4_8` | `v_dot8_i32_i4`, `v_dot8_u32_u4` | 8 | int4, **not** fp4 — there is no fp4 and no fp8 dot on this part. Same "raw dot rate" caveat, plus packing cost |
| `irp_wmma_4` | `v_wmma_f32_16x16x16_f16` | 4 | 4096 MAC per warp-inst, K accumulated over 16. See the layout note below |

`v_pk_fma_bf16` does not exist here, and there are no FP8 dot instructions: "8-bit" on this part means
int8 `v_dot4_i32_i8` / `v_dot4_u32_u8`, and "4-bit" means int4.

## WMMA: what is proved before any WMMA number is believed

`--wmma-check` runs one WMMA on host-supplied raw fragments and writes raw D back
(`irp_wmma_frag`), so nothing in the check hard-codes a fragment layout:

| line | what it proves | must print |
|---|---|---|
| `WMMA onesh` | every D element is exactly 16 → the part really does a 16-deep accumulating dot, all ones in, 16 out | `ok=1 bad=0/128 first=16` |
| `WMMA cadd` | D == 16 + C → the accumulate wiring (C really is read) | `ok=1 bad=0/128` |
| `WMMA feeder` | which lanes feed which D slot: per-slot feeder counts must total 16 for A and 16 for B, and there must be 16 distinct A patterns and 16 distinct B patterns (the rows and the columns) | `ok=1 slots=128 slots_not_16_deep=0 distinct_A_patterns=16 distinct_B_patterns=16` |
| `WMMA tile` | a naive fp16 matmul of a rank-1 tile, compared slot by slot — **only run if each D slot is fed by exactly one A lane and one B lane** | either `ok=1 bad=0/128 worst=0` or an explicit `SKIPPED single_feeder_A=… single_feeder_B=… NOT DECIDABLE` |

The `SKIPPED` branch is a real possibility and is not a failure: the k-slot *pairing* between an A
fragment slot and a B fragment slot is not recoverable from per-lane marginals, so a general tile
check needs a per-(lane,slot) probe (32×16 launches per side). Until that exists, no shipping WMMA
kernel may claim a fragment layout from this probe — it may claim the dot depth, the accumulate, and
the row/column feeder sets. `--wmma-dump` prints the derived map.

## Timing policy

The binary times nothing unless asked, and never takes `tools/gpu_lock.sh`. Every number in
`results.md` comes from a main-agent window with zero subagents, blocking, best-of-N; the rows here
are exactly the kind (bandwidth-bound or dispatch-bound) that `bench/performance-status.md` §6 says
do not repeat across windows. Quote a pair measured in one window, never two runs from two windows.

## Known open item

`irp_lds_tile` is unbilled: LLVM stages `global → LDS` inside the trip loop and peels the read side
into a separate loop that adds constants, so the only billable loop measures staging while the row
would claim it measures LDS reads. See `HANDOFF.md` for the attempts and what a fix would need.
