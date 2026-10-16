# issue_rate_probe — measured arithmetic ceilings on `gfx1151`

Main-agent-owned: every timing row here was taken blocking, with `tools/gpu_lock.sh` held, zero
subagents active, per AGENTS.md. Raw rows:
`progress/bench/evidence/issue-rate-probe-2026-10-16/G_raw_rows.txt`. The table is
`tabulate.py` over those rows; `util` sampled 47–93 % across the window and, per the ledger, is not
a usable gate on this box (the serving engine busy-polls the same iGPU).

## 0. What the binary prints that must NOT be quoted

The binary's own `mac_per_lane_clk`, `ceiling_mac_per_lane_clk` and `pct_of_ceiling` columns are
wrong and were wrong in the first device run: `mac_per_lane_clk` came out numerically equal to
`icpcb` — it reports **warp-instructions per CU per clock and labels it MACs per lane per clock** —
which is why `irp_f32_8` claimed 172 % of a 1-MAC ceiling and `irp_mix_8` claimed 248 %. Use
`tabulate.py`, which recomputes from geometry:

```
MACs per launch   = warps x 32 x iters x ops_per_iter(isa) x macs_per_lane_per_op
MACs/lane/clk     = MACs per launch / best_seconds / (2560 lanes x 2.894 GHz)
```

Config for every arithmetic leg: `blocks=320 threads=128` (1280 waves = 32/CU on 40 CUs),
`launches=20 reps=5`, best-of. Leg `irp_mix_8` is **excluded** (its read account printed
`checksum=FAIL`), and `irp_lds_tile` is unbilled so it has no row at all.

## 1. The table (best rate per leg, witness column = the closed-form byte account)

| leg | class | MAC/lane/clk | % of its roof | T·MAC/s | T·FLOP/s | witness |
|---|---|---|---|---|---|---|
| `irp_f32_8` | scalar `v_fma_f32` | 0.493 | 49.3 % | 3.65 | 7.30 | ok |
| `irp_fma_16` / `_32` / `irp_serial_32` | scalar, more ILP | 0.497–0.498 | ~50 % | 3.68 | 7.36 | ok |
| `irp_pkf16_8` | `v_pk_fma_f16` | 1.192 | 59.6 % | 8.83 | 17.66 | ok |
| `irp_d2f16_8` | `v_dot2_f32_f16` | 1.525 | 76.2 % | 11.30 | 22.59 | ok |
| `irp_d2bf16_8` | `v_dot2_f32_bf16` | 1.571 | **78.6 %** | 11.64 | 23.28 | ok |
| `irp_d4i8_8` | `v_dot4_i32_i8` | 1.940 | 48.5 % | 14.37 | 28.75 | ok |
| `irp_u4u8_8` | u4·s8 integer WMMA | 1.897 | 47.4 % | 14.05 | 28.11 | ok |
| `irp_d8i4_8` | `v_dot8_i32_i4` | 3.777 | 47.2 % | 27.98 | 55.96 | ok |
| `irp_u8u4_8` | s8·u4 integer WMMA | 3.794 | 47.4 % | 28.11 | 56.21 | ok |
| `irp_wmma_4` | `v_wmma_f32_16x16x16_f16` | 3.552 | **88.8 %** | 26.31 | 52.63 | ok |

Non-arithmetic legs, for reference: `irp_cvt_8`/`_16` at 0.26 MAC/lane/clk (the legs are conversion
loops), `irp_kv_read` rows in §4.

Roofs, at 40 CU × 64 FP32 lanes × 2.894 GHz: scalar 1 MAC/lane/clk = 14.82 TFLOP/s; packed/dot2
2 = 29.6; dot4 and WMMA 4 = 59.4; dot8 8 = 118.8.

## 2. The headline: the matrix instruction is real and we are nowhere near it

`v_wmma_f32_16x16x16_f16` sustains **52.63 TFLOP/s**, 88.8 % of the 59.36 TFLOP/s roof that
`tools/isa_probe.sh` §7.2 derives from AMD's own 512 FLOP/clk/CU figure. Against the engine's
current arithmetic:

| path | TFLOP/s | multiple for WMMA |
|---|---|---|
| engine today, `--gemm library` bf16 through hipBLASLt | 4.696 | **11.2×** |
| engine's own scalar kernels (`v_fma_mix_f32` / `v_fmac_f32_e32` at 1 MAC) | 7.30 (this probe's floor leg) | 7.2× |
| best shipped library number in the ledger | 6.885 | 7.6× |

So the answer to "is AMD's 60 TFLOPS marketing" is settled on device: the instruction reaches it in
a loop, and the gap is entirely that nothing in the shipped stack or in this engine issues a matrix
instruction (`tools/isa_probe.sh`: 0 of 55 `gfx1151` code objects contain one).

**Equally important and cheaper: `v_dot2_f32_bf16` reaches 23.28 TFLOP/s**, 4.1× what our library
path delivers, and it is a **bf16 instruction** — same input format as the weights and activations
we already have, f32 accumulate, no fp16 band question at all. Changing a dot from a scalar FMA
chain to `dot2` changes the association of the reduction, so it is still a numerics change (§05.5
ordered reductions), but it is an *ordering* change, not a format change, and it needs no wider band
on range or precision grounds.

The integer legs sit near 47–48 % of their roofs (28.1–56.2 TOPS). They are raw-dot ceilings only: a
real quantized GEMM adds dequant/scale work the leg does not model, and int8 stays a stop-and-ask
for band reasons (decision log D13 exclusion), so nothing here licenses an int8 route by itself.

## 3. The reading this refutes: `attn` was never at an issue floor

`progress/2026-10-16/prefill-occupancy-results-2026-10-16.md` §3 concluded `attn_tile_wave` sits at
**92–95 % of a 1-warp-inst/CU/clk issue floor**. That floor was a guess about the machine, and the
machine has **two** issue slots per CU (`rocminfo`: SIMDs per CU = 2), with the probe sustaining
0.49–1.57 MAC/lane/clk against roofs of 1–2 — i.e. the arithmetic half of the picture the earlier
pass was missing. Read against a 2-wide issue, `attn` is at roughly **47 %**, not 95 %, which puts
the byte wall (W1) back in front and re-ranks M-tiling above instruction-count surgery. See
decision log **D16**; the §3 sentence in the occupancy results is annotated there.

One caveat kept honest: several legs compute to **above 2.0 warp-inst/CU/clk** (`irp_mix_8` 2.486,
`irp_d2bf16_8` 2.750). That is an accounting artifact, not a 2.7-slot machine: the census counts a
`v_dual_*` packet as 2 ops in its op column while issue slots count it once. So warp-inst rates from
this census are op rates and cannot adjudicate issue width; the MAC columns can, and they are what
§1 and §3 use.

## 4. The two demoted questions, as measured

KV-shaped read (`irp_kv_read`, 512 B rows, one warp per row, `--passes 2`), best ms per launch and
the rate the row geometry implies:

| resident footprint | best ms/launch | note |
|---|---|---|
| 2 MiB | 0.005935 | inside the 2 MiB L2 |
| 8 MiB | 0.021172 | |
| 32 MiB | 0.071935 | around the 32 MiB Infinity Cache |
| 128 MiB | 1.133543 | falls off the cliff |
| 512 MiB | 4.682456 | DRAM regime |

The 2 → 32 MiB rows are 3.6× apart in time for a 16× bigger set and the 32 → 128 MiB step is 15.8×
for 4× the bytes: the cliff is between 32 and 128 MiB, which is consistent with a ~32 MiB resident
tier and inconsistent with the ~54 MiB tier the memory-bandwidth note inferred. The per-launch times
here are one-warp-per-row shapes and have not yet been converted to GB/s by a shared script — that
conversion is the next edit, and until it exists no GB/s figure is claimed for these rows.

## 5. Open items on this tool (before its numbers are load-bearing for a kernel)

1. **The WMMA fragment layout is still not proven.** `--wmma-check` on device printed
   `onesh ok bad=0/256 first=16` and `cadd ok bad=0/256` (so the tile really is a 16-deep
   accumulating dot and the D-includes-C wiring is right) but `feeder ok=0
   slots_not_16_deep=256 distinct_B_patterns=1` and `tile SKIPPED single_feeder_A=1 NOT DECIDABLE`.
   Reading: the leg's feeders are too degenerate for the witness to discriminate the k-slot pairing,
   so **no shipping WMMA kernel may assume a hand-rolled fragment layout** — use the builtin's
   fragment abstraction, or pay the per-(lane,slot) probe the tool names.
2. **The device selftest's 7 negative-control cases all failed** (`inj_drop_chain`, `inj_lds_*`,
   `inj_kv_*`: `child status 256`), so the accounts are not yet proved to catch the bugs they claim
   to catch. The rate legs' own `checksum=ok` is what §1 rests on; the controls must be made honest
   before a kernel's gate borrows this harness.
3. `irp_lds_tile` is unbilled (LLVM peels the LDS reads out of the billed loop) and its three
   injection cases are therefore *un-run*, not *failing* — `run.py` skips them loudly.
4. The binary's ceiling/pct columns should be deleted or fixed so nobody quotes them again.
