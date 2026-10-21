# Ledger re-take, 2026-10-16 — main agent, blocking, zero subagents active

Every number below was taken by the main agent in the foreground with no subagent running, each window
inside one `tools/gpu_lock.sh` hold (the lock is held by the calling process, so each window is a single
blocking bash call). `gpu_lock.sh util` is recorded at acquire and release and then discounted, per the
standing rule that util is not a usable gate here (identical clean runs read 24–91 %).

**Durable witness, and its limits.** §3 and §4 are reconstructible from `stage-wave.txt` beside this
file alone — one block per run (`stage128_1` … `wave1024_3`), each run carrying its `LAUNCH`/`NUMERICS`
flag echo, then `LAUNCHSTAT`, `STAGE`/`WAVE` family rows, the `*PARITY` witnesses, `TOTAL`, `GROUP`,
`WAVEGAP` and the `RESULT` line. The `tmp/ledger/*.log` names below are the scratch originals those
lines came from. **§1, §2, §5 and §6 are hand-transcriptions** of what `tools/context_sweep.py`,
`tools/prefill_sweep.py`, `tools/cold_restore.sh` and the `prototype/*` gates printed — best-of-3, the
run list and the spread, copied out of the window output rather than machine-attached. Treat them as
one remove weaker than §3/§4; re-run the named command before quoting one into something load-bearing.

Binary: `engine/build/decode_run` rebuilt from HEAD (`bash engine/build.sh`) at 18:09, after commit
`6fd6e61` (which flipped `--attn-tile` to 1). Shipped defaults on every row unless stated:
`--gemm library`, `--attn-tile 1`, `--mlp-map auto`, `--gdn-scan seq`, `kRowsPerBlock = 32`.

| window | lock tag | util at acquire / release | wall |
|---|---|---|---|
| decode context sweep | `ledger-head-1` | 92 % / 40 % | 38.5 s |
| prefill sweep | (same hold, not released) | 92 % / 40 % | 14.6 s |
| stage clock 128 + 4032 | `ledger-stage` | 88 % / 85 % | 27.4 s |
| wave attribution + cold restore | `ledger-wave`, `ledger-cold` | 90 % / 68 %, 63 % / 14 % | 7.4 s |
| op bursts (lock held by caller) | `ledger-ops1` | 91 % / 5 % | 55.6 s |
| MLP walk/window sweep (caller holds) | `ledger-ops2` | 90 % / 21 % | 3m33s |
| self-locking gates (no outer lock) | self-locked | — | 19.4 s |
| proj rows/block 32 + `--only` split (self-locked) | self-locked | — | 17.1 s |
| read probe | `ledger-probe` | 48 % / 64 % | ~1 min |

## 1. Decode cost vs context — `tools/context_sweep.py --lanes 128,512,1024,2048 --steps 32 --runs 3`

| lanes at last step | marginal ms/token best of 3 | all runs | spread | tok/s |
|---|---|---|---|---|
| 128 | **11.035** | 11.526 / 11.988 / 11.035 | 8.64 % | 90.6 |
| 512 | **12.408** | 12.525 / 12.475 / 12.408 | 0.95 % | 80.6 |
| 1024 | **12.407** | 12.415 / 12.407 / 12.635 | 1.84 % | 80.6 |
| 2048 | **12.382** | 16.490 / 12.382 / 13.378 | 33.17 % | 80.8 |

Medians (the 128 and 2048 rows carry one outlier run each): 128 → 11.526; 512 → 12.475; 1024 → 12.415;
2048 → 13.378.

## 2. TTFT / prefill — `tools/prefill_sweep.py --lanes 64,256,1024,3072 --runs 3`

| prompt lanes | ms to first sampled token, best of 3 | all runs | spread | prompt tok/s |
|---|---|---|---|---|
| 64 | **91.1** | 92.1 / 92.7 / 91.1 | 1.76 % | 702.9 |
| 256 | **140.0** | 143.0 / 140.0 / 142.4 | 2.17 % | 1,828.5 |
| 1024 | **297.4** | 345.3 / 310.1 / 297.4 | 16.09 % | 3,443.1 |
| 3072 | **907.3** | 914.4 / 918.0 / 907.3 | 1.18 % | 3,385.8 |

## 3. Stage clock, decode step — `decode_run --time-stages`, medians of 3

128 lanes (prompt 112 + 16 steps), `tmp/ledger/stage128_{1,2,3}.log`:

| quantity | three runs | median |
|---|---|---|
| burst (pipelined, production shape) | 11.3573 / 11.2735 / 11.5772 | **11.357** |
| stage (323 waits/step) | 13.2385 / 13.2469 / 13.7743 | 13.247 |
| kernel (425 waits/step) | 13.7162 / 14.3092 / 14.3141 | 14.309 |
| exposed (`STAGEGAP exposed_ms`) | 1.889 / 1.981 / 2.208 | **1.98** |
| exposed, other pairing (gaged `TOTAL` − burst) | 1.881 / 1.973 / 2.197 | 1.97 |
| `idle_sync_us` calibration | 0.099 / 0.099 / 0.119 | 0.099 |

Witnesses: `STAGEPARITY … tokens_identical=1 logits_bitidentical=1` at all three granularities on all
three runs; `launches_per_step=424 plan_per_step=424 plan_match=1`; `syncs_per_step` 323 (stage) / 425
(kernel).

In-situ family rows at 128 lanes (stage granularity, ms/step, median of 3):
glue 0.6125 · proj 4.1128 · gdn 0.6375 · attn 0.4847 · mlp 3.3880 · cast 0.9028 · copy 0.2186 ·
lm_head 2.1799 · host_tail 0.7360. **The gages are exhaustive: Σ families − TOTAL ≤ 0.0001 ms in every
run** (verified across all six runs at stage *and* kernel granularity, both contexts — `Σ` is 13.2386
vs TOTAL 13.2385 for stage128_1, and so on). So the stage wall carries no unattributed time, and the
sum must be read per run: a sum of per-family medians is 13.273 against a median wall of 13.247, and
that 0.2 % is a medians-don't-add artifact, not a measurement. What the construction cannot see is
burst-to-step: each named family costs 0.46–1.01 ms more inside the step than in its own isolated burst.

4032 lanes (prompt 4000 + 32 steps), `tmp/ledger/stage4032_{1,2,3}.log`:

| quantity | three runs | median |
|---|---|---|
| burst | 14.0280 / 13.8788 / 14.0995 | **14.028** (= 71.3 tok/s) |
| stage | 16.2757 / 16.2367 / 16.3003 | 16.276 |
| kernel | 17.0259 / 17.2039 / 17.2587 | 17.204 |
| exposed (`STAGEGAP exposed_ms`) | 2.212 / 2.259 / 2.369 | **2.26** |
| exposed, other pairing (gaged `TOTAL` − burst) | 2.248 / 2.358 / 2.200 | 2.248 |
| `idle_sync_us` | 0.143 / 0.143 / 0.142 | 0.143 |

In-situ family rows at 4032 lanes (median of 3): glue 0.706 · proj 4.137 · gdn 0.657 · **attn 2.996** ·
mlp 3.381 · cast 0.988 · copy 0.229 · lm_head 2.156 · host_tail 1.018. The `attn` median is exactly
2.9955 — a rounding midpoint, printed half-up as 2.996; a naive float `round(median, 3)` prints 2.995
and looks like a contradiction of the ledger when it is only a tie-break.
`STAGEPARITY` clean, `plan_match=1`, same sync counts as the 128-lane rows.

## 4. Wave attribution, 1024-token prompt — `decode_run --steps 1 --time-waves`, medians of 3

`tmp/ledger/wave1024_{1,2,3}.log`, ids from `progress/bench/evidence/q10/ids1024.txt`. **Basis:** the
single-family rows are the median of the three runs; **composite rows are sums of per-family medians**
(not a group measurement, and not the median of per-run sums — for `gdn` batched those two bases are
7.1248 and 7.108 ms, and this table prints the former). The two share rows are the median of the three
printed `share_pct`/`matmul_pct` values.

| family | ms (median) | share | runs (ms) |
|---|---|---|---|
| `gdn_rec` | **126.83** | **51.2 %** | 123.08 / 126.84 / 126.83 |
| `proj` | 38.81 | 15.8 % | 38.81 / 39.21 / 38.10 |
| `attn` | **26.74** | **10.9 %** | 26.74 / 27.00 / 26.55 |
| `mlp_gateup` | 24.99 | 10.2 % | 24.67 / 25.42 / 24.99 |
| `mlp_down` | 6.42 | 2.6 % | 6.82 / 6.07 / 6.42 |
| `cast` + `copy` | 9.46 | 3.8 % | — |
| `gdn` batched (conv/gates/gated_norm) | 7.12 | 2.9 % | — |
| `glue` | 3.32 | 1.3 % | — |
| `lm_head` + `host_tail` | 3.04 | 1.2 % | — |
| serialized wall | **245.73** | — | 243.17 / 247.60 / 245.73 |
| burst wall | 235.48 | — | 236.03 / 234.89 / 235.48 |
| `GROUP matmul_ms` | 70.30 | **28.6 %** | 70.30 / 70.69 / 69.52 |

Witnesses: `WAVEPARITY tokens_identical=1 logits_bitidentical=1` at both granularities;
`launches=20264 plan_launches=20264 plan_match=1 launchstats_match=1 within_tol=1`.

**Correction note (2026-10-16, from the review that also produced `tools/citation_rot_check.py`).** Five
composite rows here — `attn` share, `gdn` batched, `cast`+`copy`, `lm_head`+`host_tail`, `GROUP
matmul_pct` — were first transcribed from run 1 under a caption that said medians of 3. They now carry
the median of the three runs on the basis named at the head of this section (`gdn` batched
4.5194+0.4109+2.1945 = 7.1248 → 7.12; `cast`+`copy` 6.8252+2.6335 = 9.4587 → 9.46 and
2.76+1.07 = 3.83 → 3.8 %; `lm_head`+`host_tail` 2.2940+0.7455 = 3.0395 → 3.04 and 0.93+0.31 = 1.24 → 1.2 %;
`attn` share median of 11.00/10.90/10.81 = 10.9 %; `matmul_pct` median of 28.91/28.55/28.29 = 28.6 %),
recomputed from the three committed logs. Nothing else in the section moved; the same cells in
`engine/performance-status.md` §3 were corrected with these.

## 5. Cold weight-restore — `tools/cold_restore.sh --runs 5`

runs (load() wall): 335.8 / 353.9 / 366.8 / 340.9 / 345.8 ms → **best 335.8, median 345.8, worst 366.8,
spread 9.2 %**. Phases of the best run: cold H2D 58.5 ms, pinned staging 58.9 ms, first GPU read pass
8.5 ms. (The committed row it replaces was 454.4 ms best / 77 % spread.)

## 6. Op floors in isolation

- `lm_head_gemv` fused: **2.1309 ms = 238.7 GB/s = 100.7 % of probe** (`--iters 8`, lock held).
- `gdn_decode` state, 18 layers warm: **0.0648 ms best, worst 0.0682, spread 5.2 %**, 882.7 GB/s
  (372 % of probe, i.e. cache-resident); `rec` alone 0.0071 ms. All four functions PASS vs dumps.
- MLP burst, cold, per-token over 24 layers (`prototype/mlp_gemv/run.py --layers all --fixture dec8`):

  | window | walk | per-token MLP | % of probe |
  |---|---|---|---|
  | 132 MB (`--tile 6`) | interleave (the wave walk) | 2.647 ms | 84 % |
  | 132 MB | contig (the one-row walk) | **2.378 ms** | 94 % |
  | 528 MB (`--tile 24`) | interleave | 2.372 ms | 94 % |
  | 528 MB | contig | **2.275 ms** | 98 % |

  Every row also re-gated (`ndiff act 1 out 0`, §10.1 class gates hold).
- `glue` fused, 49 launches: **0.1496 ms @512 t (spread 4.0 %)**, 0.1852 @256 t, 0.2644 @128 t;
  unfused 97 launches: 0.2386 @512 t. Gate: 784 comparisons, 0 failures.
- `gdn_gated_norm` §04.5: **0.4273 ms per 144-job plan = 2.967 µs/launch, spread 0.8 %**;
  `DEVICE GATE PASS`, all six negatives localise.
- `proj_gemv` 48-launch token plan, cold, rows/block 32 (the shipped `kRowsPerBlock`, unpacked `x`):
  **3.2843 ms = 142.3 GB/s = 60 % of probe** (roofline 1.972 ms). At the prototype's default rows/block 8
  the same plan reads **3.5863 ms = 55 %**. Gate 912 comparisons, 0 failures.
  New split rows, same window and config (`--only gdn` / `--only attn`, which closes the §2 debt):
  **GDN projections 2.5969 ms = 146.0 GB/s = 62 %** (roofline 1.600 ms) and
  **attention projections 0.6887 ms = 128.1 GB/s = 54 %** (roofline 0.372 ms); the two sum to 3.2856 ms,
  i.e. 0.04 % off the whole-plan row, so the split is self-consistent.
- Read probe `tools/build/bw_probe_hip 1.0 4`: **best read 239.3 GB/s**, best copy 210.5 GB/s,
  H2D 82.4 GB/s, D2H 69.7 GB/s. (The ledger has used 237 GB/s as the probe since 2026-08; the re-take is
  +1 %, so the "% of probe" columns do not move.)
