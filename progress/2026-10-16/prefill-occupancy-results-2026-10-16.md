# Prefill occupancy pass — measurements, the refutation of my own first reading, and the funded next step

Main-agent device measurements, 2026-10-15 evening, every row taken blocking in the foreground with
**zero subagents active** and inside one `tools/gpu_lock.sh` hold. Decision log for the framing is
`prefill-decision-log-2026-10-16.md` (D1–D7); this file is the data and what it decides.
Raw rows: `window1-timing.txt`, `window2-pmc.txt`, `window3-occupancy.txt`, `window4-4096.txt` in this
directory; profiler CSVs under `tmp/pmc-occ*/` (regenerable with `tmp/occ/window{1,2,3,4}.sh`).
Binary: `engine/build/decode_run` at HEAD `222b23f`, shipped defaults
(`--gemm library --attn-tile 1 --mlp-map auto --gdn-scan seq`, `kMaxWaveRows` 512).

## 1. The user's question, answered at the widths the engine can reach

A 20k-token prompt **cannot be run today** and the refusal was reproduced, not inferred:

```
decode_run FAILED: attn_decode_layer: widest row is 4608 lanes (nrows=512), needs split=9,
more blocks than the engine's kAttnMaxSplit=8 (lane ceiling 4096 = staging MAXSC 512 x split)
```

At `--steps 0` the 4096-lane prompt wave itself does run (split = 8 is legal); it is the *generated*
token at 4097 lanes that refuses, which is why window 1's `--steps 1` rows at 4096 printed no `RESULT`.

**Family attribution, one window, `--time-waves`, serialized wall (ms, share of wave):**

| family | 1024 lanes | 2048 | 4096 | scaling |
|---|---|---|---|---|
| `gdn_rec` | 128.31 (50.85 %) | 259.74 (46.87 %) | 520.26 (39.78 %) | linear, 6.96–7.06 µs/token/layer |
| **`attn`** | **27.23 (10.79 %)** | **104.32 (18.83 %)** | **412.07 (31.51 %)** | **quadratic in T** |
| `proj` | 39.72 | 76.66 | 155.73 | linear |
| `mlp_gateup` + `mlp_down` | 32.83 | 67.42 | 114.28 | ~linear |
| wall (serialized / burst) | 252.33 / 248.04 | 554.13 / 548.56 | 1307.75 / 1285.57 | |
| `ungaged_ms` (instrument completeness) | 0.0206 | 0.0500 | 0.0980 | ≤ 0.009 % of the wave |
| `exposed_ms` (serialization cost) | 4.29 | 5.57 | 22.19 | 1.7 % of the wave at 4096 |

Witnesses on every row: `WAVEPARITY tokens_identical=1 logits_bitidentical=1`,
`plan_match=1 launchstats_match=1`, `launches = plan_launches` (20,264 / 40,526 / 81,050).

**`exposed_ms` is 1.7 % of a 4096-lane wave and `ungaged_ms` is 0.007 %** — so there is no dispatch-bubble
prize hiding in the wave and nothing is unattributed between the gages. The prefill question is
entirely "why are the families as slow as they are", not "why is the GPU idle".

**Plain walls, best of 3 (`--steps 0` where 4096 needs it):** 1024 → 321.7 ms (3,184 tok/s incl. the
extra step), 2048 → 629.8 (3,252), 4096 → 1,337.9 (3,062 prompt tok/s for the wave at 1,307.8 ms).
Ledger §1's 3,443 / 3,386 at 1024/3072 are a different window and ruler; the §6 rule says a sweep in
another window is not an A/B, so nothing here is a regression claim.

## 2. The byte model fits a third width, and that is not yet proof

Model: `Σ_lanes = T(T+1)/2`, 2 KiB per token per attention layer (`kNumKvHeads=2 × kHeadDim=256 ×
bf16 × K and V`), 6 layers, at the 237 GB/s read probe.

| context | model `attn` | measured | ratio | effective demanded-byte rate |
|---|---|---|---|---|
| 1,024 | 27.2 ms | 27.23 ms | 1.001 | 236.8 GB/s |
| 2,048 | 108.8 ms | 104.32 ms | 0.959 | 247.1 GB/s |
| 4,096 | 435.0 ms | 412.07 ms | 0.947 | 250.2 GB/s |

Three widths, all within 5 %, all landing at 237–250 GB/s of *demanded* K/V bytes. On its own that reads
as "attention is a DRAM sweep and the on-die tier never rescues it", which is the W1 verdict that would
fund row-tiling to divide the sweep by M. Section 4 says why that verdict is not safe yet.

## 3. Counter census on the shipped form: `attn` is also at ~93 % of a 1-wide VALU issue floor

`rocprofv3 --pmc` works for a non-root user on this box (counter groups replay the dispatch; dedup by
`Dispatch_Id`, never by `Correlation_Id`). Clock from `rocm-smi`: sclk 2894 MHz, and `rocminfo` says
**2 SIMDs per CU**, so "4 inst/CU/clk" in the 2026-10-14 §9 census was GCN thinking; a floor computed at
1 warp-inst per CU per clock is therefore a *conservative lower bound*, not the machine's limit.

| context | `attn` VALU warp-inst (all attn kernels) | floor @1 inst/CU/clk | measured | issue utilisation |
|---|---|---|---|---|
| 1,024 | 2,984 M (`attn_tile_wave` 2,888 M) | 25.78 ms | 27.23 ms | **95 %** |
| 2,048 | 11,052 M (`attn_tile_wave` 10,832 M) | 95.47 ms | 104.32 ms | **92 %** |

`attn_tile_wave` geometry: workgroup 128, LDS 12,288 B, VGPR 48, grid 262,144 at 1024 lanes.

**Correction, same day.** An earlier draft of this line read `SQ_WAVE_CYCLES / SQ_BUSY_CYCLES = 41`
wavefronts per CU "against a max of 64" and concluded the form was not starved for parallelism. The
max is not 64: `rocminfo` says **`Max Waves Per CU: 32`** (2 SIMDs × 16) — see
`devices/strix-halo-official-specs.md` §4 — so 41 is *above* the architectural ceiling and the ratio is
therefore not "resident wavefronts per CU" at all; it is summed over a set whose size I have not
established. **No occupancy claim from this pass survives.** That is what the ratio can and cannot say,
and it is a second reason the prototype in §5 item 1 is the right next instrument: occupancy has to be
earned from a bounded measurement, not inferred from a counter whose instance set is unknown.

The two candidate walls both scale with `Σ_lanes`, so **they are indistinguishable by scaling**, and both
fit within 5–8 %. Per (row, lane, head) triple the kernel issues ~115 warp-instructions against ~30 for
the two 256-dim dots, so the instruction stream carries a ~3–4× tax of conversion, LDS addressing,
masking and reduction. That tax is now the same size as the byte bill, and it bounds what any
traffic-side rewrite can buy: row-tiling with M=2 and the VALU path unchanged cannot go below
`max(bytes/M, issue floor)` ≈ 320 ms at 4096, i.e. −22 % of `attn`, not −50 %.

## 4. A trap: my first reading of the cache counters was wrong, and the calibration that says so

Window 2 appeared to show attention barely touching DRAM — `GL2C_EA_RDREQ_64B` × 64 B summed to 4.4 MiB
for `attn_tile_wave` against 6.45 GB of demanded bytes, hit rate 0.75. Take 1 of the analyser reported
that as a refutation of the byte wall. It is void, for two reasons, both now encoded in
`tmp/occ/analyse_pmc2.py`:

1. **Dedup key.** Replays must be collapsed by `Dispatch_Id`; `Correlation_Id` is per-API-call and
   repeats across kernels, which turned "passes per dispatch" into 7879 and every absolute into noise.
2. **Calibrate before believing a byte counter.** The same pass run over every kernel in the wave
   (complete coverage: 21,305 dispatches against the wave's 20,264 launches) reports
   `lm_head_gemv_full` = **5.55 MiB**, a kernel that streams **485 MiB** by construction and measures
   2.13 ms, i.e. at the read probe. `GL2C_EA_RDREQ_64B` on this part is not counting global reads the
   way its name suggests. Therefore *"attention never reaches DRAM" is not established*, the byte model
   stands unfalsified, and the hit-rate ratios (0.75–0.79 for `attn_tile_wave`) are the only part of the
   GL2C output worth carrying.

This is the AGENTS.md pattern again — a check whose clean bill of health measures the population it
scanned. A counter that cannot reproduce a 485 MiB known workload cannot adjudicate a 6.45 GB claim.

## 5. What is funded, and the order

1. **`prototype/issue_rate_probe` (funded now, cheap, decides everything else).** A microprototype that
   measures, on this box, (a) the sustained VALU warp-instruction rate per CU per clock for the dot shape
   `attn_tile_wave` uses, and (b) the achievable GB/s for a KV-shaped streaming read as a function of
   resident working-set size (2/8/32/128/512 MiB). The two open questions in §3 and §2 are exactly
   (a) and (b): if the box sustains 2 warp-inst/CU/clk then `attn`'s "93 % of a 1-wide floor" is really
   ~47 % and bytes are the wall alone; if a 2 MiB KV set can be driven at ~876 GB/s (the rate the `gdn`
   state demonstrably reaches resident) then the shipped sweep is leaving bytes on the table. Everything
   downstream — M-tiling alone, M-tiling + MFMA, or instruction-count surgery — is ranked by these two
   numbers, so nothing else should be built first.
2. **Row-tiled attention (M query rows sharing one K/V walk, diagonal masked)** — the traffic lever,
   priced by `tmp/occ/walls.py`: `attn` at 4096 412 → 320 ms (M=2, issue-capped) or → 217 ms (M=2 with
   the issue tax also halved); at 20k, 10.7 s → 5.4 s (M=2) → 1.4 s (M=8). Bit-identity required on the
   per-row lane order. Funded if (1b) says bytes are being left on the table.
3. **MFMA-backed scoring** — the only lever that removes the issue floor: same FLOPs at the measured
   5.667 TFLOP/s bf16 = 4.6 ms at 1024, 72.8 ms at 4096, 1.82 s at 20480. AMD's own page puts this part
   at **"FP16 AI Compute up to 60 TFLOPS"** (`devices/strix-halo-official-specs.md` §1), ten times the
   rate our best matrix kernel has reached, so this pricing is a floor with an unexplored ceiling above
   it; the open question is recorded as discrepancy 1 in that file. Funded if (1a) confirms the
   issue wall and the M-tile prize turns out capped by it. Not a band-free change: a matrix-unit score
   dot does not preserve the §05.5 ordered reductions, so this is a cascade-band admission like route (c).

   **Corrected the same day, from `tools/isa_probe.sh` — see `devices/strix-halo-official-specs.md` §7.**
   Three things in that paragraph are wrong. (i) There is **no MFMA on `gfx1151`**; the matrix
   instruction here is **WMMA** (`v_wmma_f32_16x16x16_f16`), and it is reachable from our compiler.
   (ii) The 5.667 TFLOP/s was never a matrix measurement: **0 of the 55 `gfx1151` code objects shipped
   with rocBLAS contain a matrix instruction** — the bf16 libraries execute scalar `v_fmac_f32_e32`, so
   that number is 32 % of the *vector* roof (14.82 TFLOP/s) and 7.9 % of the matrix roof
   (59.36 TFLOP/s, = AMD's 60). (iii) The prize is therefore 4× off, not 10× off, and the lever is
   "write a WMMA kernel", not "get closer to a rate the libraries already achieve". The route survives,
   re-priced and renamed; the fast tier it implies is **fp16**, since `v_pk_fma_bf16` does not exist on
   this part.
4. **Lane-ceiling raise** (`kAttnMaxSplit` 8 → 40, `a_part` ~68 → ~338 MiB at `wave_rows` 512) so a real
   20k prompt can be measured and served. Held until 1–3 say the long-context regime is worth its
   numerics admission (ledger §6 debt 4).

## 6. Decode: notes only, explicitly not funded in this pass

- Ledger §2: `attn` in-situ is **2.996 ms/step at 4032 lanes**. Its K/V bill is 6 layers × 4032 lanes ×
  2 KiB = 47.25 MiB per step = **0.21 ms at the read probe**, so decode attention is **~14× its own roofline**.
  M-tiling cannot help (M = 1 by definition in decode), so the candidate causes are the issue/latency
  side and per-block staging waste. It has no isolated burst row beside it (ledger debt 7), so the first
  action there is an instrument, not a kernel. Worth more than anything in §5 if it survives a probe.
- Already parked: §6.2 fp16 on the matrix path — **unblocked 2026-10-16** (no band needed: fp16 is a
  mantissa gain on bf16 inside its range, and nothing pinned exceeds that range, decision log D13/D14);
  its cost is now gated on the WMMA measurement below, not on approval. §6.3 weight
  quantisation (out of scope unless asked), §6.4 MTP on GPU (never priced on device).

## Correction, taken the same day: the 1-wide issue floor this document assumes does not exist

§3's "92–95 % of a 1-warp-inst/CU/clk issue floor" was measured correctly and bounded wrongly: the
part has **two** issue slots per CU (`rocminfo`: SIMDs per CU = 2), and a later device window
measured the arithmetic directly — scalar `v_fma_f32` at 0.493 of a 1-MAC/lane/clk roof, `dot2` at
1.57 of 2, `v_wmma_f32_16x16x16_f16` at 3.55 of 4 (52.63 TFLOP/s, 88.8 % of the matrix roof).
Re-read against 2 slots, `attn_tile_wave` is near 47 %, not 95 %, so the byte wall is back in front
and M-tiling outranks instruction-count surgery. Detail and the numbers that came with it:
decision log **D16**, `prototype/issue_rate_probe/results.md` §1 and §3.
