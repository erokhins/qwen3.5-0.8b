# A12 — why the wide shape's 25 % does not reach the wave (measured 2026-10-20)

The question came from `6b857d2` (the non-regression window): the rig says the wide block is **1.2507×**
the shipping WMMA kernel, the engine says **−0.69 %**, and the commit said the next question is
*"coverage, not speed: which rows at 16,384 lanes does the wide kernel actually own"*. It was run by the
main agent in four legs, each inside one `tools/gpu_lock.sh` hold, blocking, zero subagents, `util` read
at acquire (89 / 91 / 92 %) and then discounted per `engine/performance-status.md` §6.

## 0. The answer, in four lines

1. **It is not coverage.** The wide block owns **every** row of every body — the engine's own echo reads
   `wide_rows=512/512` at all 32 distinct geometries — so the tip commit's hypothesis is refuted, not
   confirmed.
2. **The shape's rate win does not exist at engine geometry.** Paired body-by-body from device dispatch
   timestamps at 16,384 lanes, the wide core and the shipping core take **the same time**: ratio
   **1.0001** summed over 31 paired bodies, per-body ratios 0.937–1.122 with no trend in body width.
3. **The reason is launch geometry, and it is now measured rather than argued.** One wide launch covers
   `2 × ceil(512/32) = 32` workgroups on a **40-CU** machine (the trace prints `wgs/CU 0.80 *`, the tool's
   own flag for "fewer workgroups than CUs: that kernel cannot fill the machine"), while the shipping
   launch it replaces carries **176 workgroups on average and 2,816 at the last body**. The rig reached
   **2 blocks/CU × 40 CUs = 16 wavefronts/CU**; in the engine the same kernel can host **at most 1 block/CU
   on 32 of 40 CUs = 8 wavefronts/CU, with 8 CUs idle** — *below* the 9 wavefronts/CU the form it replaces
   sustains. The relaxation bought a residency the shipping body cannot spend.
4. **What the wide route does win at the wall is launch elimination, and it is mostly buyable without it.**
   It does not launch `attn_combine` (71.58 ms) or `attn_slim_wave` (13.42 ms) — the split's partial
   round-trip. `--attn-split-cap 8` on the *shipping* kernel recovers **−1.14 %** of the wide arm's
   **−1.46 %**, with no numerics risk at all.

## 1. W1 — the coverage census (`a12_why_census.sh`, echo lines only, no timing)

One prefill of 16,384 lanes per arm, `--attn-fattn 1`, `--attn-fattn-wide 0|1`. Raw lines:
`evidence/echo_16384_wide0.txt`, `evidence/echo_16384_wide1.txt`.

| arm | what the engine printed, once per distinct geometry |
|---|---|
| `wide=0` | `ATTN_FATTN form=fattn16 q=unstaged8 roll=single pv=f32 warps=3 vgpr=185 lds=20480 scratch=0 wf_per_cu=9 wave_rows=512 split=2..32 fattn_rows=512/512 covered_rows_pct=100.0 covered_work_pct=100.0`, plus `ATTN_FATTN_SKIP` for the body at or below `MAXSC` |
| `wide=1` | `ATTN_WIDE form=wide128 warps=8 threads=256 pos_per_block=32 stage=64 acc=f16 vgpr=256 lds=32768 scratch=176 wf_per_cu=16 SPILL grid=(2,1,16) wide_rows=512/512 wave_rows=512 lanes_last=512..16384 split=1..32 split_used=0` |

Three things are readable off those two lines and all three matter:

* `wide_rows=512/512` at **all 32** geometries — coverage is total, so the sliver hypothesis is dead.
* `grid=(2,1,16)` **does not change with lane count**. The wide block covers 32 query positions and no
  lane split, so its workgroup count is fixed by the body's row count: `2 × ceil(wave_rows/32)`. The
  shipping launch's `split` axis is what grows with context (176 → 2,816 wgs), and the wide form has no
  such axis — `split_used=0` says so on its own line.
* `wf_per_cu=16` is a **capacity** (the driver's occupancy answer: 32,768 B LDS ⇒ 2 blocks/CU), not the
  realised figure. Realised occupancy is `min(2, blocks per CU)`, and with 32 blocks and 40 CUs the
  second term is 1 on some CUs and 0 on at least 8 of them. This is the same distinction §55 made for
  registers, one axis over: a capacity that the launch population cannot reach is not a resource you have.

## 2. W2 — in-situ attribution and the per-body pairing (`a12_why_trace.sh`)

`tools/prefill_kernel_trace.py` (rocprofv3 dispatch timestamps), one 16,384-lane prefill per arm in one
hold. Summary tables: `evidence/a12_why_kernels_wide{0,1}.txt`; the core-dispatch rows are in
`evidence/a12_why_core_wide{0,1}.csv` and the body pairing below is reproduced by
`a12_why_body_compare.py` → `evidence/body_by_body.txt`.

| kernel | `wide=0` | share | `wide=1` | share |
|---|---|---|---|---|
| attention core | `attn_fattn_wave_u8` **1,473.15 ms** (186 launches, mean 176 wgs) | 52.35 % | `attn_wide_wave` **1,476.05 ms** (192 launches, 32 wgs, `0.80/CU *`) | 54.31 % |
| `attn_combine` | 71.58 ms | 2.54 % | **not launched** (0.05 ms, one-row body only) | 0.00 % |
| `attn_prep` | 15.88 ms | 0.56 % | 16.72 ms | 0.62 % |
| `attn_slim_wave` | 13.42 ms (inert early-out where the WMMA form covers) | 0.48 % | **not launched** | — |
| `attn_core` (one-row bodies) | 7.37 ms | 0.26 % | 8.44 ms | 0.31 % |
| **attention family** | **1,581.40 ms** | **56.2 %** of Σ 2,814.21 ms | **1,501.26 ms** | 55.2 % of Σ 2,717.71 ms |

So the family is **−5.1 %** and the core is **+0.20 %**. Every millisecond of the family's saving is a
launch the wide route does not submit. And the per-body detail rules out "the average hides a win at the
top": the widest body — 512 rows at 16,384 lanes, the shape the BET was priced on — is
**14,493.8 µs wide against 14,196.2 µs shipping, ratio 1.0210**.

| body (lanes_last) | ship µs/launch | wide µs/launch | ratio | ship wgs | wide wgs |
|---|---|---|---|---|---|
| 2 (1,024) | 900.2 | 1,010.2 | 1.1222 | 176 | 32 |
| 8 (4,096) | 4,007.9 | 4,043.1 | 1.0088 | 704 | 32 |
| 16 (8,192) | 7,496.6 | 7,593.9 | 1.0130 | 1,408 | 32 |
| 24 (12,288) | 10,870.7 | 11,212.6 | 1.0315 | 2,112 | 32 |
| 32 (16,384) | 14,196.2 | 14,493.8 | 1.0210 | 2,816 | 32 |
| **Σ paired** | **245.52 ms** | **245.54 ms** | **1.0001** | | |

## 3. W3 — the wall, and the grid-depth probe (`a12_why_griddepth.sh`)

**LEG A, the wall, 5 interleaved reps (the tip commit ran 3).** `RESULT ms_total`, `--steps 1`:

| lanes | `wide=0` best / median | `wide=1` best / median | wide1 vs wide0 |
|---|---|---|---|
| 16,384 | 2,810 / 2,829 ms | 2,753 / 2,778 ms | **−2.03 % best, −1.76 % median** |
| 4,096 | 509 / 513 ms | 499 / 503 ms | **−1.96 % best** |

The per-rep ranges at 16,384 are **disjoint** (`wide=0` min 2,810 > `wide=1` max 2,789), so this is a
shift, not a spread, and it **corrects the tip commit**: −0.69 % with overlapping ranges understated the
wide route by about 3×. The wide route is a real ~2 % win at long context. It is still ~12× short of the
25 % the rig printed, and §2 says where the difference went.

**LEG B, the grid-depth probe.** `--wave-rows` is the one engine knob that regroups rows at a fixed total
lane count, so the work is identical and only the launch population moves: wide `2 × ceil(rows/32)` wgs,
shipping `8 × split × ceil(rows/48)` (table at the `split = 32` the last bodies run at). Best of 2 at
16,384 lanes, absolute ms are meaningless across rows (fewer rows ⇒ more bodies ⇒ more `attn_prep`), the
**ratio** is the readout:

| `--wave-rows` | wide wgs/launch | ship wgs/launch | wide0 ms | wide1 ms | ratio |
|---|---|---|---|---|---|
| 512 (default) | **32** | 2,816 | 2,830.7 | 2,769.9 | **0.9785** |
| 256 | 16 | 1,536 | 3,136.4 | 3,408.3 | **1.0867** |
| 128 | 8 | 768 | 3,657.3 | 5,698.8 | **1.5582** |
| 64 | 4 | 512 | 5,271.2 | 10,584.5 | **2.0080** |

The wide arm's cost tracks its block count and the shipping arm's does not: halving the wide launch's
population from 32 to 16 turns a −2 % arm into a +9 % arm, and at 4 blocks it costs 2×. That is the
mechanism, measured in the shipping engine rather than inferred from the rig — and it says the engine's
own default body sits near the edge of the cliff, not at the bottom of it.

## 4. W4 — the ~2 % is not the shape's, it is the split's (`a12_why_splitcap_ab.sh`)

Since §2 attributes the whole engine win to skipping `attn_combine` + `attn_slim_wave`, the same win should
be reachable on the **shipping** kernel by shrinking that round-trip — which is what `--attn-split-cap`
(bet A9-L-c) is for. Best of 3, interleaved, one hold, 16,384 lanes, canary: every run printed its own
`SPLITCAP split= cap= …` line:

| arm | best ms | vs ship |
|---|---|---|
| `wide=0`, no cap (ships today) | 2,805.0 | — |
| `wide=1` (A12) | 2,764.0 | **−1.46 %** |
| `wide=0 --attn-split-cap 8` | 2,773.0 | **−1.14 %** |
| `wide=0 --attn-split-cap 16` | 2,777.0 | −1.00 % |
| `wide=0 --attn-split-cap 4` | 2,850.0 | +1.60 % |
| `wide=0 --attn-split-cap 2` | 2,954.0 | +5.31 % |

So ~78 % of what the wide route delivers at the wall is a property of the **partial round-trip**, not of
the 128-row shape, and it is available on the default kernel with a knob that moves no format and no band.
(The cap's own economics are still width-dependent — §53 already recorded L-c as DISPLACED at 32k, and the
cap over-binds below 8 here.)

## 5. What would make the wide shape pay, stated as a requirement

The rig paid out at **80 resident blocks** (2/CU × 40 CUs) and the engine hands it 32. To reach the rig's
residency inside the engine, a 512-row body must emit ≥ 80 workgroups, which means either:

* **≥ 1,280 query rows per body** — i.e. `kMaxWaveRows` above 512 (`engine/decode_step.h:51`), which is a
  wave-width change with its own scratch and numerics consequences, or
* **a lane split in the wide form** — `split` blocks per row-tile writing `part` partials and an
  `attn_combine`, which is exactly the thing `attn_wide_kernels.h` declares it does not have ("NO LANE
  SPLIT … NO COVERAGE RULE"). `split ≥ 3` at 512 rows would put it at 96 blocks.

Until one of those exists, the wide form's rate is pinned at ≤ 8 wavefronts/CU on 32 of 40 CUs, and the
BET's own mechanistic finding — "the wall is latency on re-fetched operands" — explains why the shape does
nothing there: operand re-use pays per resident wavefront, and the engine gives this kernel fewer
wavefronts than the one it replaces.

## 6. Consequences for the default decision (not taken here)

* The default stays `--attn-fattn-wide 0`. Nothing here argues for moving it: the shape's rate win is not
  available at engine geometry, and the numerics admission is still red (`smoke` pooled 1-cos 2.028e-04
  against a 2e-04 band, `A12-ADMISSION.md`), so the widened band currently buys ~1.5 % that is 78 %
  reachable another way.
* `--attn-split-cap 8` as a *default* is a separate question with its own admission (§53 measured it at
  32k and displaced it there; here it is −1.14 % at 16k). This window did not adjudicate it and nothing
  about its numerics changed.
* The honest headline is unchanged: warm **10,637 tok/s at 512 lanes**, **9,790 at 4,096**
  (`progress/2026-10-21/performance-status-2026-10-21.md` §51 and §55). The long-context rows are where either lever would show,
  and neither is shipped.

## 7. Files

| file | what it is |
|---|---|
| `a12_why_census.sh` | W1: the echo/grid census (no timing) |
| `a12_why_trace.sh` | W2: the paired `prefill_kernel_trace.py` window |
| `a12_why_body_compare.py` | W2's per-body pairing, reads the trimmed core CSVs |
| `a12_why_trim_evidence.py` | reduces the 6 MB raw trace CSVs to `a12_why_core_wide*.csv` + the summary tables |
| `a12_why_griddepth.sh` | W3: LEG A (5-rep paired wall) + LEG B (`--wave-rows` grid-depth probe) |
| `a12_why_splitcap_ab.sh` | W4: the split-cap A/B that prices the wide route's real win |
| `a12_why_mkids.py` | the 4,096 / 16,384-lane prompt fixtures (CPU-only) |
| `evidence/` | the echo lines, both per-kernel tables, the per-body table, the per-dispatch core CSVs, and the raw per-rep timing lists |

Every timing leg is main-agent-only by the standing rule; a worker may re-run `a12_why_census.sh` (it takes
no timing) but must not run the other three.
