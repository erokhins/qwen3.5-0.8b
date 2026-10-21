# `issue_rate_probe` — handoff

Worker-authored, 2026-10-17. Everything below was done on CPU: compile-only census runs, host
mirrors, and `--dry-run`/`--selftest-geom`. **No device work, no timing, no `tools/gpu_lock.sh`,
nothing committed.** `results.md` is an empty skeleton on purpose.

## 1. State of the tree right now

| thing | state | how it is known |
|---|---|---|
| `issue_rate_probe.cpp` compiles | **yes** | `hipcc -O3 -Wl,-rpath,/opt/rocm/lib --offload-arch=gfx1151 -o issue_rate_probe issue_rate_probe.cpp` — silent success |
| `bash build.sh` | **green**: census `--emit` + `--check`, main binary + 7 defect binaries | prints `BUILD ok: issue_rate_probe + 7 defect binaries` |
| instruction bills | come from the disassembly, per-kernel mnemonic requirements | `isa_census.py --check`: every leg `ok`, `ISA CENSUS ok` |
| host mirrors + plan rejections | **green** | `./issue_rate_probe --selftest-geom` → `SELFTEST-GEOM ok cases=28 fails=0` |
| driver wrapper | `run.py` exists; `--verify` green, `--commands` prints the window | `python3 run.py --verify` → `VERIFY ok — the bills, the mirrors and the plans agree. No device was opened.` |
| device measurement | **none taken** | `results.md` is empty |
| one leg unbilled | `irp_lds_tile` — census refuses it, `run.py` SKIPs it loudly | see §4 |

Three bugs the rewrite fixed, because a reviewer will want to know what the earlier "green" was worth:
the census billed `irp_lds_tile`'s staging loop and `irp_kv_read`'s block-reduction loop (both wrong
spans); `^v_fmac_f32\b` never matched `v_fmac_f32_e32`; and the kernel slicer mis-attributed every
`namespace irp_arith` kernel to the previous `irp::` kernel, which had hidden all eight arithmetic
legs behind the wrong loop. `irp_kv_read`'s op count is **1**, not the 2 the first version asserted:
one warp reads one 512 B row per iteration and 512 B / 32 lanes is exactly one `global_load_b128`;
the 2 had come from the mis-billed reduction.

One table change worth flagging: the accumulator exactness bound was booked against the per-thread
*tally* (both fp16 halves of 8 accumulators) instead of against **one accumulator**, which made
`irp_pkf16_8` un-runnable at 128 threads (the printed cap came out negative). The bound is now
expressed per accumulator (`acc_seed0`/`acc_seed_lane`/`acc_grow`), and `--selftest-geom` has the
case that proves the rejection still fires on the f32 leg.

## 2. The device window — numbered, hand-me-ready

`python3 run.py --commands` prints exactly this list with the accepted `--iters` already substituted.
Hold `tools/gpu_lock.sh` around the whole list; `run.py` deliberately does not acquire it (a caller
that holds the lock and then runs a self-locking driver parks silently — AGENTS.md trap row).

**0. WMMA correctness, before any WMMA number.**

```
./issue_rate_probe --wmma-check
./issue_rate_probe --wmma-check --wmma-dump
```

| line | must print | what it proves |
|---|---|---|
| `WMMA onesh` | `ok=1 bad=0/128 first=16` | with all-ones A/B and C=0 every D element is exactly 16 → the part executes a **16-deep accumulating dot**, and the built-in `__builtin_amdgcn_wmma_*` we call really lowers to `v_wmma_f32_16x16x16_f16` |
| `WMMA cadd` | `ok=1 bad=0/128` | D == 16 + C → the C operand is actually read, i.e. accumulating over K is real, not overwritten |
| `WMMA feeder` | `ok=1 slots=128 slots_not_16_deep=0 distinct_A_patterns=16 distinct_B_patterns=16` | every D slot is fed by 16 A operands and 16 B operands, and there are exactly 16 distinct A feeder sets and 16 distinct B ones → the rows and the columns of the tile |
| `WMMA tile` | `ok=1 bad=0/128 worst=0` **or** `SKIPPED single_feeder_A=0 single_feeder_B=0 NOT DECIDABLE` | the naive fp16 rank-1 matmul check. It only runs if each D slot has exactly one A feeder lane and one B feeder lane. SKIPPED is an honest outcome, not a failure: the k-slot *pairing* between A and B fragment slots is not recoverable from per-lane marginals |

PASS means: the matrix path works and the accumulate wiring is right. If `tile` runs and is green, the
layout is pinned. If it prints SKIPPED, **no shipping WMMA kernel may claim a fragment layout on this
probe's authority** — that sentence belongs in the commit that ships one, and §5 names the probe that
would settle it. Any `ok=0` here voids every WMMA number in §1.

**0b. `./issue_rate_probe --selftest`** — device byte witnesses, one child process per case, no
timing printed. PASS means every witness reproduces and every guard band is intact; `--selftest-geom`
already proved the host half (28/28).

**1. The headline: arithmetic paths, scalar floor → matrix roof.**

```
python3 run.py --device            # all legs, or --only <leg> for one
```

Per leg it prints `icpcb`, `mac_per_lane_clk`, `ceiling_mac_per_lane_clk`, `pct_of_ceiling`,
`crosscheck_icpcb_x_mac`, `waves_requested_per_cu`, `resident_waves_per_cu`, `max_waves_per_cu=32`,
plus `checksum=ok`, and exits non-zero if a witness fails or a launch ran short. The conversion is on
the row itself: `flop_per_s = warp_inst_total * 64 * macs_per_lane_per_op * 2 / secs`,
`mac_per_lane_clk = flop_per_s / (2 * cu_count * 64 * clock_hz)`, `pct_of_ceiling = 100 * that /
ceiling`.

What each row discriminates:

| leg | ceiling MAC/lane/clk | the reading a green row supports |
|---|---|---|
| `irp_f32_8` | 1 | the floor: our bf16 path's `v_fmac_f32_e32` rate. Should land near 100 % if the loop is issue-bound |
| `irp_mix_8` | 1 | the bf16-in/f32-out path rocBLAS's HS_HPA runs. If it matches `irp_f32_8`, the bf16 penalty is the conversion, not the FMA |
| `irp_pkf16_8` | 2 | first rung above the floor. fp16 accumulator, so `--iters` is capped (512 at 320×128; the plan refuses more and says the max) |
| `irp_d2f16_8`, `irp_d2bf16_8` | 2 | `v_dot2` with an f32 accumulator, f16 and bf16 operands. `d2bf16` is the only bf16 2-MAC path on this part — there is no `v_pk_fma_bf16` |
| `irp_d4i8_8`, `irp_u4u8_8` | 4 | **raw dot rate only.** A real int8 GEMM adds zero-point/dequant/scale work this loop does not contain |
| `irp_d8i4_8`, `irp_u8u4_8` | 8 | int4, not fp4 — there is no fp4 and no fp8 dot here. Same raw-dot caveat, plus packing cost |
| `irp_wmma_4` | 4 | **the number this tool exists for**: what a hand-written WMMA K-loop sustains of the 4-MAC roof. `crosscheck` must equal `mac_per_lane_clk` (icpcb 0.03125 × 128 = 4.0) |

Read the `waves` column before believing a low `%`: at 2 resident waves of a possible 32 the leg is
latency-bound at its realised occupancy, which is `gdn_collapse`'s lesson, not a broken kernel.

**2a. Original question 1 — sustained warp-inst/CU/clk on the attention dot shape** (now second):
`irp_fma_1/4/8/16/32`, `irp_serial_8/32`, `irp_cvt_8/16`. ILP curve vs one dependency chain, and what
a `bf16→f32` conversion feeding every FMA costs. PASS = `icpcb` where §07.4's corrected claim says it
should be; the row says which.

**2b. Original question 2 — read GB/s vs resident KV working set**, `--footprint-mib` 2 / 8 / 32 /
128 / 512, one run each (`run.py --commands` prints them). PASS = `checksum=ok` at every footprint and
a reported cliff position in MiB. Anything above 240 GB/s is the resident-tier finding, not a bug,
and only believable with `checksum=ok` and a sweep that straddles the tier.

## 3. What is finished

`isa_census.py` (requirement-driven billing, dual-issue aware, LDS width-weighted, `--allow-unbilled`,
`--check` against `run.py:DECLARED`); `irp_arith_kernels.h` (one leg per arithmetic path, all one
shape, plus `irp_wmma_frag` for the layout probe); `irp_kernels.h` (staged LDS tile, `fill_tag_k`,
`frag` operand); `issue_rate_probe.cpp` (kernel table with ceilings and per-accumulator bounds,
occupancy reporting, `mirror_arith_thread`, `wmma_check`, traffic rows that print `n/a` instead of a
fake MAC column); `run.py`; `README.md`; `results.md` skeleton; `build.sh` passing the unbilled list.

## 4. Open items, in priority order

1. **`irp_lds_tile` is unbilled.** The census finds no loop containing both the staged LDS read and the
   FMAs that consume it: the billed loop holds `ds_store_b128 ×6` + `global_load_b128 ×6` and zero LDS
   reads; the reads were peeled into a 13-instruction loop of `v_add_f32` (adding constants). Attempts
   so far, in order: phase-varying staged values, an opaque rotation read from device memory, phase-
   varying source tiles, the trip index added into the read address. Each moved the fold; none removed
   it. Not attempted, because the chase changes what the row MEANS and that is the orchestrator's
   call: stage a cycle once (`lds[j] = (j+1) mod S`, `S` slots of `b32`), start each lane at
   `idx = lane`, and make each trip's address the value read last trip, accumulating the values —
   hoisting that would require unrolling the whole trip count, and the witness is a closed form
   (`sum_t (lane+t) mod S`, with `S` small enough to keep the per-thread total under 2^24). The caveat,
   and the reason it was not built unprompted: a per-lane **dependent** chain measures LDS *latency per
   trip*, not the LDS read *width/bandwidth* the leg was written to ask about — attn's LDS reads are
   independent parallel loads, which is precisely the shape LLVM keeps folding away. So the chase buys
   an honest row for a DIFFERENT question, and needs a new family in the plan (`lds_mib_total` is no
   longer `12,288 x (1+reuse)`), a new host mirror, a census entry requiring `ds_read_b32 >= 1` plus the
   accumulate, and `run.py` bookkeeping: about an hour of work if the rebooking is approved. If it is
   not, delete the leg and ask the LDS-width question of `attn_decode`'s own gated kernels instead.
   `run.py` currently SKIPs it with this reason and `build.sh` carries it in `--allow-unbilled`; a 0
   bill cannot be mistaken for a measurement.
2. **WMMA k-slot pairing.** If `--wmma-check` prints `WMMA tile SKIPPED`, settling the layout needs a
   per-(lane,slot) probe: zero all A but one (lane, slot), read which D slots move, 32×16 launches per
   side. `irp_wmma_frag` already accepts raw per-lane fragments, so the probe is host-side work in
   `wmma_check` and needs no new kernel.
3. **`results.md` is empty** and must be filled by a main-agent window, together with
   `bench/performance-status.md` if any of these numbers become ledger rows. Quote pairs from one
   window: these legs are the dispatch-bound kind that `bench/performance-status.md` §6 shows
   spreading 5–10 % across windows.
4. **Per-leg grid is the default 320 × 128 everywhere.** That is right for the scalar legs and probably
   wrong for WMMA (4 D accumulators × 8 f32 = 32 registers/lane of accumulator, so occupancy will be
   low). If `resident_waves_per_cu` comes back small for `irp_wmma_4`, that is the first thing to vary
   (`--blocks`, `--threads`), not the ceiling.
5. **Defect binaries for the LDS injections** (`lds_slot_off`, `lds_short`, `lds_swap`) build but have
   no live leg while §4.1 stands; `results.md` §4 marks them n/a. `lds_swap`'s blindness to the block
   sum is nevertheless proved on the host (`lds_swap_blinds_block_sum ok=1`).
6. **Commit hygiene, flagged not fixed.** `build.sh` leaves `issue_rate_probe` and the seven `neg_*`
   binaries untracked and `.gitignore` does not cover them (`git status --porcelain` lists them), which
   is the same class as the `.gitignore` trap already recorded in `AGENTS.md`. Whoever commits this
   topic should ignore the binaries by exact path, and should split the work into the topics it actually
   is: census rewrite / arithmetic kernels / driver ceilings + WMMA probe / `run.py` / docs.

## 5. CPU work actually run by this worker (wall, and no timing taken)

| command | wall | note |
|---|---|---|
| `isa_census.py --emit/--check` (compile-only, ~24 kernels) | ~60–90 s each, ~8 runs | compile-only; no device |
| `hipcc … -o issue_rate_probe issue_rate_probe.cpp` | ~25–40 s each | first compile since the table/RESULT rewrite: 4 errors (missing `IRP_OPS_*` before `--emit`, bare names for `irp_arith::` kernels, one `irp::`-qualified `irp_wmma_frag`, then the tally-vs-accumulator bound) |
| `bash build.sh` | ~3 min | `BUILD ok: issue_rate_probe + 7 defect binaries` |
| `./issue_rate_probe --selftest-geom` | < 1 s | `ok cases=28 fails=0` |
| `python3 run.py --verify` | ~90 s (dominated by the census) | `VERIFY ok` |
| `python3 run.py --commands` | < 1 s | prints the window |
| `./issue_rate_probe --dry-run --kernel <leg>` × 21 | < 1 s each | host plan only |

Every device-shaped command in §2 is marked as such and **no timing was taken by this worker**.

## 6. If you only do one thing

Run §0 (`--wmma-check`) and §1's `irp_wmma_4` row. Those two answer the question the tool was built
for: whether the only matrix instruction on this part is worth writing a kernel for, and at what
fraction of its 4-MAC roof a hand-written K-loop lands.

---

**Archived 2026-10-21** from `prototype/issue_rate_probe/HANDOFF.md`, where it sat beside the rig it briefed. The device pass it hands over is done and its verdict is in that op's `results.md` and in the ledger, so `progress/README.md` rule 1 puts it here. The filename now carries the op name because a dated folder holds handoffs from several ops at once (rule 3's collision clause); citations naming the old path were re-pointed in the same commit.
