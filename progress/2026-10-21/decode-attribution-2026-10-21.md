# Where a decode token goes, re-taken on the shipped tree — 2026-10-21

**Why this file exists.** `engine/performance-status.md` §1 and §2 both say a decode re-take is owed:
the banner rows came from a 2026-10-19 window and the family attribution from 2026-10-16, and three
shipped rounds (G4 `reg16`, G1 residency, the A13 default flip) landed after them. This file is that
re-take, plus the floor arithmetic it needs, plus the route list it produces. Nothing here loosens a
gate; every route below gets its win line and kill line **before** a kernel exists, which is the rule
`progress/2026-10-14/speedup-routes-2026-10-14.md` §7 rule 1.

**Window.** Main agent, blocking foreground calls, **zero subagents active**, one
`tools/gpu_lock.sh` hold per window, `util` recorded at acquire and release and then discounted per
§6 of the live ledger. Binary: `engine/build/decode_run` built from `aa2dce4` by `engine/build.sh`
(22.4 s), shipped defaults echoed by every run:
`gemm=lt hipblas=3.6.0 attn_tile=1 attn_touch=1 attn_fattn=1 attn_fattn_qstage=unstaged8
attn_fattn_wide=wide_llama gdn_scan=seq gdn_spell=fused gdn_loop=512 gdn_norm=inline gdn_geom=reg16`.
Scripts and logs: `progress/2026-10-21/evidence/decode-retake-2026-10-21/` (`context_sweep.log`,
`stage128_{1,2,3}.txt`, `stage4032_{1,2,3}.txt` — the gage lines of each run, with the per-token
`STEP`/`STAGESTEP`/`SPLITCAP` records dropped because they are 90 % of the bytes and carry no row of
this table — plus `stage_window.sh`, the exact window, `tabulate_stage.py`, the reader that produced
the medians, and `mkids.py` + `ids128.csv`). util read 93 → 37 % on the sweep hold and 91 → 83 % on
the stage hold.

## 1. The banner: marginal cost of one generated token

`tools/context_sweep.py --lanes 128,512,1024,2048 --steps 32 --runs 3` — differential by
construction (two runs differing only in `--steps`), so the prompt wave cancels.

| lanes at last step | marginal ms/token, best of 3 | all runs | spread | tok/s | was (§1, 2026-10-19) |
|---|---|---|---|---|---|
| 128 | **11.181** | 11.181 / 11.404 / 11.532 | 3.13 % | **89.4** | 11.475 / 87.1 |
| 512 | **12.488** | 12.652 / 12.998 / 12.488 | 4.08 % | 80.1 | 12.928 / 77.4 |
| 1,024 | **12.590** | 12.590 / 12.678 / 12.751 | 1.28 % | 79.4 | 11.922 / 83.9 |
| 2,048 | **12.453** | 12.986 / 12.741 / 12.453 | 4.28 % | 80.3 | 12.558 / 79.6 |

**Read this honestly: decode did not move.** Every row is inside its own spread against the
2026-10-19 window, which is what §37's noise floor (a sub-3 % move on a sub-30 ms row is not
claimable) predicts — G4, G1 and A13 were wave-only rounds, and the decode path was untouched by
construction. The campaign this file opens is the first one aimed at the decode path itself.

## 2. The step, gaged: `decode_run --time-stages`, medians of 3

Same instrument and same two points as the frozen 2026-10-16 window (`progress/bench/evidence/
ledger-2026-10-16/commands-and-provenance.md` §3): 112 prompt lanes + 16 steps = 128 lanes at the
last step; 4,000 + 32 = 4,032. Prompts are `context_sweep.py`'s own recipe, written to id **files**
(the MAX_ARG_STRLEN trap, `AGENTS.md`), with the field count asserted by the writer: 112 and 4,000.

| quantity | 128 lanes, three runs | median | 4,032 lanes, three runs | median |
|---|---|---|---|---|
| **burst** (pipelined, production shape) | 11.4609 / 11.3031 / 11.5801 | **11.461** | 14.2223 / 14.2073 / 14.2241 | **14.222** |
| stage (305 waits/step) | 13.0400 / 13.5840 / 13.7701 | 13.584 | 16.3240 / 16.3705 / 16.3446 | 16.345 |
| kernel (425 waits/step) | 13.6736 / 14.3907 / 14.3660 | 14.366 | 16.9270 / 16.9569 / 16.9760 | 16.957 |
| exposed (`STAGEGAP exposed_ms`) | 1.5791 / 2.2809 / 2.1900 | 2.190 | 2.1016 / 2.1632 / 2.1205 | 2.121 |
| `idle_sync_us` calibration | 0.099 / — / — | 0.099 | 0.142 / — / — | 0.142 |

Witnesses on every run: `STAGEPARITY … tokens_identical=1 logits_bitidentical=1` at all three
granularities, `launches_per_step=406 plan_per_step=406 plan_match=1`, `syncs_per_step=305`. The
launch bill is now `glue=49 proj=114 gdn=72 attn=6 mlp=48 cast=97 other=20 total=406` — the
1036-era `cast=115 / total=424` bill in the frozen window is superseded by `62b73e9`'s packed-row
deletion, and this is the first window to print 406 in the ledger.

Family rows at stage granularity (ms/step, median of 3), with the byte census from
`tools/step_traffic_model.py` and its floor at the measured 237 GB/s read ceiling:

| family | launches/step | segments/step | in-situ ms (128) | bytes/token | floor ms | in-situ ms (4,032) |
|---|---|---|---|---|---|---|
| `proj` | 114 | 48 | **4.234** | 466.7 MB r | 1.973 | **4.195** |
| `mlp` | 48 | 48 | **3.388** | 529.0 MB r/w | 2.244 | **3.392** |
| `lm_head` | 1 | 1 | **2.138** | 509.6 MB r | 2.146 | **2.208** |
| `host_tail` | 0 | 1 | 0.931 | CPU | — | 1.015 |
| `cast` | 97 | 97 | 0.808 | 1.1 MB | 0.005 | 0.859 |
| `glue` | 49 | 49 | 0.700 | 0.5 MB | 0.002 | 0.738 |
| `gdn` | 72 | 36 | 0.657 | 40.5 MB r/w | 0.170 | 0.674 |
| `attn` | 6 | 6 | **0.496** | 1.57 MB (128 lanes) | 0.007 | **3.023** (49.6 MB) |
| `copy` | 19 | 19 | 0.220 | 2.7 MB | 0.011 | 0.234 |
| **Σ = TOTAL** | 406 | 305 | **13.575** | 1,556.7 MB | **6.58** | **16.334** |

`Σ families − TOTAL ≤ 0.0002 ms` in all six runs, at stage and kernel granularity, both contexts:
the gages are still exhaustive, so there is no unattributed time in the step wall.

## 3. The floor, and the two arithmetic corrections needed to use the table above

**The floor is 6.58 ms/token = 152 tok/s**, from the census (`1,532.3 MB read + 24.4 MB written per
token`, read:write = 62.8:1) at the measured 237 GB/s. `tools/step_traffic_model.py` prices the same
thing three ways (reads alone 6.465 ms; no-cliff with writes at 210 GB/s 6.582; writes at half that
6.698) and its falsifier is unchanged: no memory system charges a 63:1 read stream enough write
penalty to explain the residue, so the floor is a read floor and the step's `11.461 / 6.58 = 57 %`
of it is a real inefficiency, not a wrong denominator. §07.4's band with dispatch included
(**145–153 tok/s**) still stands; the engine is at 89 tok/s.

**Correction 1 — a stage-granularity family row contains its own launch overhead.** Each family
segment is delimited by a sync, and the serialized pass pays per-segment launch/idle latency the
pipelined step hides. Total: `exposed 2.190 ms / 305 segments = 7.2 µs per segment`. Attributing it
per segment (not per launch — 406 launches at the 1.85 µs floor is 0.75 ms of that 2.19) gives the
column the burst wall must add up to, and it adds up:

| family | stage ms | − segments × 7.2 µs | **estimated share of the 11.46 ms burst wall** | floor ms | **gap to floor** |
|---|---|---|---|---|---|
| `proj` | 4.234 | −0.346 | **3.89** | 1.973 | **1.92** |
| `mlp` | 3.388 | −0.346 | **3.04** | 2.244 | **0.80** |
| `lm_head` | 2.138 | −0.007 | **2.13** | 2.146 | **≈0 — at ceiling** |
| `attn` | 0.496 | −0.043 | **0.45** | 0.007 | **0.44** |
| `gdn` | 0.657 | −0.259 | **0.40** | 0.170 | **0.23** |
| `glue` | 0.700 | −0.353 | **0.35** | 0.002 | **0.35** |
| `cast` | 0.808 | −0.698 | **0.11** | 0.005 | 0.10 |
| `copy` | 0.220 | −0.137 | **0.08** | 0.011 | 0.07 |
| `host_tail` | 0.931 | — | ≤0.93 (CPU; exposed only if the GPU starves) | — | ? |
| Σ | 13.575 | −2.190 | **10.45 + host_tail 0.93 = 11.38** | 6.58 | **≈4.9** |

The last row is the check: the corrected shares plus host time reproduce the measured burst wall
(11.38 vs 11.461, 0.7 %), which is what licenses using a per-family gap as a prize size. It also
demotes two families the raw stage table flattered: `cast` is 0.81 ms of *stage* time but only
~0.11 ms of wall — 97 tiny cast kernels are almost entirely hidden in the pipelined step, so
deleting them is worth ~1 %, not 7 %. Same for `copy`. **This is the number to plan against, and it
is derived arithmetic on measured rows, so it carries a §6-style caveat: the per-segment constant is
an average over 305 segments of very different sizes.**

**Correction 2 — a burst is not a step, and the two directions of that error are different
families.** `lm_head` (one fat launch, 485 MiB) moves 2.3 % between burst and step and sits at
99.6 % of its roofline; `proj` moves +29 % (3.2843 isolated → 4.234 in-situ) and sits at 47 %.
`mlp` moves +49 % against its own 528 MB `contig` burst (2.275 → 3.388) while sitting at 66 %. So
the two big families are each above their own roofline for a *different* reason: `proj` is slow in
isolation too (60 % of probe at burst), `mlp` is not.

## 4. What the numbers say about the shape of the problem

1. **Decode is a weight walk and it is the only thing that matters at short context.** 1,506 MB of
   the 1,557 MB/token is weights (mlp 528 + lm_head 509 + proj 467). Everything else — state, KV,
   activations, logits, casts — is 51 MB, 3.3 %. The engine reads 1.04 MB of activations and writes
   157 KB per megabyte of weight, so any byte-reduction argument that is not about weights argues
   over 3 % of the traffic.
2. **The bandwidth ceiling is reachable and has been reached, by one kernel.** `lm_head` streams the
   whole tied table in 2.138 ms against a 2.146 ms roofline (99.6 % of probe, reproduced to 0.2 %
   window to window). That is the control: the box can do 237 GB/s on a skinny single-row GEMM at
   these sizes, so a proj burst at 60 % is a shape problem, not a machine limit.
3. **`attn` is the only family whose cost is context, and it is the worst offender per byte.** At
   4,032 lanes it reads 49.6 MB of KV in 3.023 ms = **16 GB/s**, 7 % of the probe; at 128 lanes it
   reads 1.57 MB in 0.496 ms = 3.2 GB/s. The floor is 0.007 and 0.21 ms. Nothing in the prefill
   rounds touched this: `--attn-tile`, `--attn-fattn` and A13 are all wave-path forms, and §36
   measured decode unchanged. The §07.3 decision (one-pass softmax for bit-exactness against the
   pins) is the right *numerics* decision and an unexamined *performance* one.
4. **The context slope is attention and the short-context plateau is proj.** From 128 to 4,032 lanes
   the step moves 11.461 → 14.222 ms (+2.76), and `attn` moves +2.53 of it. Every other family is
   flat to <1 % except `host_tail` (+0.08) and `cast` (+0.05).

## 5. The route list, with the line fixed before the kernel

Each route states the mechanism, the deciding measurement, the win line, the kill line, and what
would have to be true numerically. Prizes are the corrected wall shares from §3, stated at both
context points, because they are different questions.

| # | route | mechanism | prize @128 | prize @4032 | win line | kill line |
|---|---|---|---|---|---|---|
| **D1** | **the one-row projection GEMV** (`proj`) | 114 launches / 466.7 MB / 35.6 µs per launch = 115 GB/s in-situ and 142 GB/s at burst, against a `lm_head` control at 238. One-row `x · Wᵀ` is shape-limited: thin `N`, `K` ∈ {1024, 6144, 16384}, `kRowsPerBlock = 32`. Variants to price: rows/block, vector width (bf16 128-bit loads), split-`K`, grid-stride, one launch per layer-body instead of per weight, and hipBLASLt at `M = 1` (the library route has never been allowed below `rows > 1`) | **−1.9 ms = −17 %** | −1.9 ms = −13 % | proj burst ≤ **2.60 ms** (≥ 76 % of probe) | proj burst ≥ **3.05 ms** (< 9 % of step) |
| **D2** | **the MLP one-row walk** (`mlp`) | in-situ 3.388 vs its own 2.275 burst at the same 528 MB and same `contig` map: a **1.11 ms in-situ penalty on a kernel that is at 98 % alone**. Diagnose before writing: is it interleaving with the rest of the step's traffic (replay the step's traffic mix against the MLP kernel alone), the DRAM page walk over 24 layers' disjoint weight ranges, or the `act`/`delta` writes landing in the same window? | **−0.80 ms = −7 %** | −0.80 ms = −5.6 % | mlp in-situ ≤ **2.90 ms** | mlp in-situ ≥ **3.20 ms** |
| **D3** | **the decode attention form** (`attn`) | 16 GB/s at 4,032 lanes, 3 GB/s at 128. One-pass softmax over `P+1` lanes with `MAXSC = 512` per block, 8 query heads over 2 KV heads. Candidates: lane-block staging (K/V staged once per block as in the prefill tile form), vectorised loads, a `4:1` GQA block so one KV read serves four query heads, and — only if staging cannot reach the floor — the streaming/online form §07.3 already prices numerically (25.9 % of elements moved, cos 0.99999651, meanrel 1.58e-3) | **−0.44 ms = −3.8 %** | **−2.5 ms = −18 %** | attn ≤ **1.50 ms** @4032 **and** ≤ 0.40 @128 | attn ≥ **2.55 ms** @4032 |
| **D4** | **the GDN decode walk** (`gdn`) | 0.40 ms of wall against a 0.17 ms floor; the state is 18.9 MB r + 18.9 MB w and reads warm at ~876 GB/s, so the gap is the per-token chain at 72 launches, not bytes. G4 `reg16` shipped the wave form; the decode step is `T = 1` through the same kernel | **−0.23 ms = −2 %** | −0.23 ms | gdn ≤ **0.50 ms** | gdn ≥ **0.60 ms** |
| **D5** | **the small-op tail** (`glue`+`cast`+`copy`) | 165 of the 406 launches, 1.73 ms of stage time, ~0.54 ms of wall, ~1.2 MB of traffic. §2's ruling that launch count is not a prize binds the *launch* half of this; the *work* half is what the prefill rounds deleted (the F32 seam, the packed GDN cast). Candidates: cast the GEMV input in the GEMV (the 97 casts widen BF16→F32 for `x`), fuse `copy`'s roll-back into the conv ring (§08.2's alternative), fold the norm+gate epilogues into `glue` | **−0.35 ms = −3 %** | −0.35 ms | Σ ≤ **1.20 ms** stage | Σ ≥ **1.50 ms** stage |
| **D6** | **host time per step** (`host_tail` 0.93 → 1.01 ms) | CPU work per step: sampling, the logits D2H, bookkeeping. Exposed only where the GPU starves; today's exposed bill is 2.19 ms and the pipelined step already hides it, so this is a *headroom* item: cutting the GPU side by D1–D3 shrinks the host budget from 28 µs/launch to ~20 µs/launch, which 2.3 µs/launch still clears | not a prize today | not a prize today | measure only; no code until a window shows the GPU idling on the host | — |

**Explicitly not funded here, both because they need a decision that is not this campaign's.**

- **Weight quantization.** It is the only route that can move the floor itself: 8-bit blocks roughly
  halve the 1.55 GB/token and would take the ceiling to ~200 tok/s (§07.4's byte ladder: 4-bit
  everywhere → 243 tok/s). `AGENTS.md`'s numeric-format rule names int8/int4 as a stop-and-ask, and
  the SCOPE section says quantization-for-speed needs an explicit request. Every D1–D5 win together
  buys ~3.4 ms of the 4.9 ms gap; quantization buys the rest and then moves the floor.
- **MTP drafting on the GPU.** §07.4's byte arithmetic says it beats the bandwidth floor structurally
  (MTP k=1 at accept 0.58 → 181 tok/s against 152), and the CPU path measured the opposite (−21 vs
  −24 tok/s, accept 0.66) which is why it is a GPU-path bet that has never been priced. It is a
  feature build (draft layer + verify on the wave machinery), not an optimization of what is there,
  and it needs its own package. `prototype/` has the `mt32f` fixture and §09 has the spec.
- **Batch > 1.** The byte bill amortizes across requests; the ledger prices batch 8 at ~1,080 tok/s
  aggregate. That is a serving-mode change, not a decode-step change.

## 6. Order of work, and the protocol each route runs under

D1 → D3 → D2 → D4 → D5, largest corrected prize first except that D3 precedes D2 because its prize
is the only one that grows with context. Each route runs the same loop, which the box's GPU rule
forces:

1. a worker writes the prototype, the CPU-side reference and the gate, and may run a **correctness-only
   device gate under ~60 s** (PASS/FAIL, no timing, command + wall + util reported);
2. the worker stops, and the **main agent** runs the timing commands it hands over — blocking,
   zero subagents, one lock hold, best-of-N with the spread printed;
3. the number decides against the line above, in those words, before any engine edit;
4. a win graduates into `engine/` behind its flag, passes `engine/gate.py` over the derived fixture
   population with **no band moved**, and lands with its `results.md`;
5. `engine/performance-status.md` §1/§2 rows move only with a window that re-takes them, and the
   re-take is quoted with its command.

## 7. Two tool fixes this window needed

- `tools/instep_family_budget.py` carried a C-style comment (`/* … §07 … */`) on the line that reads
  the per-token launch bill, so it raised `SyntaxError` on import and took
  `tools/step_traffic_model.py` down with it — the traffic census this file depends on could not run.
  The comment is now a Python comment and the fallback `406` is commented as the fallback it is.
- The stage window writes its prompt ids through `--ids-file` from files whose field count the
  writer asserts (`.../decode-retake-2026-10-21/mkids.py`), because a 4,000-id argv is one 19 KB field
  and a comma/space mistake there silently feeds one token (AGENTS.md, the `--ids-file` row).
