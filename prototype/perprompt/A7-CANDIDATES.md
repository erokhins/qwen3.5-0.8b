# A7 — what the ~65–71 ms per prompt that is not the wave is

Task: `progress/2026-10-21/performance-status-2026-10-21.md` §49 measured `RESULT ms_total` − wave `wall_ms` =
**71.1 / 70.9 / 64.9 ms at 512 / 1,024 / 4,096 lanes**, called it a fixed per-prompt cost
outside the wave, and left it unattributed. This file enumerates what it can be, prices each
candidate from numbers already in the ledger or derived from the source, and hands the main
agent the one measurement that settles it. **No device timing was taken here**; every ms
figure below is either quoted with a `§`/file cite or derived arithmetic on quoted rates.

Labels used throughout: **[read]** = a fact read off the source; **[derived]** = arithmetic
on the source's own constants; **[quoted]** = a number taken from this repo's docs/ledger
with its cite.

---

## 0. First, a framing correction: the two numbers in §49's subtraction bracket the SAME work

This changes what the constant *is*, so it goes first.

| fact | where |
|---|---|
| `RESULT ms_total` is `step_ms`, and `step_ms` receives exactly two contributions: the wall of `prefill_wave(...)` and the wall of each `decode_step(...)`. At `--steps 0` (what the llama-comparison leg runs) it is **the prompt wave and nothing else**. | [read] `engine/decode_run.cpp:1797-1795` (timer around `prefill_wave`), `:1886-1890` (per-step), printed at `:1990` |
| The weight load is timed into a **separate** variable `load_ms`, printed as `loaded … in X ms`, and **never added** to `step_ms`. Same for `check_weights`, state/scratch allocation, the hipBLASLt route/plan, `prepare_weight_adapters`, the per-row printing loop and the `--time-waves` report. | [read] `engine/decode_run.cpp:1598-1593`, `:1599`, `:1700-1701`, `:1726-1740`, `:1753`, `:1803-1850`, `:1976` |
| `WeightStore::load()` performs **phase 3: one GPU read pass over every resident page**, unconditionally, before `load()` returns — i.e. inside `load_ms`, outside `ms_total`. | [read] `engine/loader.cpp:626-644` (the `warm_read_kernel` loop, not gated by any option), `engine/loader.h:14-15` (the contract it implements), `loader.h:85` (`warm_ms`) |
| `wall_ms` — the number §49 subtracts — is printed by `wave_pass`, which calls `st.reset()` and then **runs `prefill_wave` again on the same prompt** and times that. So `wall_ms` is the **second** (burst) pass of identical work; the `stage` pass right after it is the **third**. `WAVEPARITY … logits_bitidentical=1` is the engine's own witness that the re-pass is the same work and not a cheaper one. | [read] `engine/decode_run.cpp:1407-1421` (`st.reset()` at `:1406`, `prefill_wave` at `:1415-1416`, `out.wall_ms` at `:1421`), `:1496-1520` (`wave_report` runs burst then stage), `:1466-1472` (parity line) |

**Consequence.** `ms_total − wall_ms` is not "host cost outside the wave". Both operands are
`prefill_wave` of the same prompt in the **same process**; the only thing that can be in the
difference is work the *first* execution of that wave pays and a later execution does not:
first-touch of device pages, first launch of each kernel, first execution of each library
algorithm, and the DVFS ramp at the start of the first sustained GPU burst in a process that
just spent ~0.4–0.5 s loading on the CPU. It is a **per-process first-execution cost that
happens to be paid inside the first prompt's wave**, not per-prompt host work, and no kernel
change can remove it — which is the one part of §49's conclusion that survives intact.

The engine already prints a same-process first-vs-later pair, so the number is cheap to
re-take cleanly: `engine/decode_run.cpp:1992` runs `wave_report` before the `RESULT` print at
`:1990`, so one `--steps 0 --time-waves` invocation yields cold (`RESULT ms_total`), warm
(`WAVEPASS gran=burst wall_ms`) and warm-serialised (`WAVEPASS gran=stage wall_ms`) in one
process — and §36's ruler table is exactly that shape (§36 is `--steps 0`, per its own
command line at `progress/2026-10-21/performance-status-2026-10-21.md:1637-1638`).

**A pair shape to avoid.** It is tempting to lift a first-vs-later pair out of an ordinary
`--steps N` log, and that comparison is invalid: a `mode=run` log such as
`tmp/attn_percost/a6g1ab/base_1.log` prints `ms_total=546.520` for a run that also generated
4,096 steps (`ms_per_step=0.133` — so most of that wall is generation), and subtracting the
*prefill* wave's `wall_ms=441.8548` from it is a category error. The only valid pair is
prompt-only (`--steps 0`) with both operands from the same process, which is what §36 printed
and what the handover leg A prints.

**Attribution caution on §49's own pair.** §49's `119.830 / 167.080 / 494.957 ms` do not
appear in that window's own run logs: `tmp/gcc/llamaphase/our_512_{1,2,3}.log` read
`ms_total=` **147.680 / 144.007 / 144.480** (median 144.5 → the 3,544 tok/s in §49's own
table), and those bench-leg logs contain **no `WAVEPASS` line at all** (the bench leg runs
without `--time-waves`; verified by counting the line in each file). So the 71.1 ms at 512
lanes was computed from a *cold* run and a *warm* run taken in different invocations, and
against the bench-leg medians the same subtraction would read 144.5 − 48.7 = **95.8 ms**.
First job of the handover measurement is therefore to re-derive the pair **inside one
process**, where nothing but warm/cold can differ.

---

## 1. Enumeration: every cost around a prompt, priced

Bracket column: **in-timed** = inside `RESULT ms_total`; **first-only** = inside the timed
region but paid only by the process's first wave (i.e. a candidate for §49's constant);
**outside** = not in `ms_total` at all (so not in the constant, and not in the headline).

### 1a. Costs outside `RESULT ms_total` (not the constant; priced because a real per-prompt budget needs them)

| # | cost | bracket | class | path:line | size |
|---|---|---|---|---|---|
| O1 | weight load: header parse, `hipHostMalloc` staging, 335 `hipMalloc`+`hipMemcpy` uploads, the warm read pass | outside | per-process | `decode_run.cpp:1588-1593`; `loader.cpp:595` (upload), `:626-644` (warm pass) | **[quoted]** measured in the window's own logs: `loaded 335 tensors, 1.5457 GB resident in **380.8–525.4 ms**` (`tmp/gcc/llamaphase/our_*.log`). This is 5–7× the constant, and it is already excluded from the headline. |
| O2 | `check_weights` (CPU walk of 335 shard-header claims) | outside | per-process | `decode_run.cpp:1599` | **[read]** CPU-only, no launches; upper bound ≪ 1 ms next to O1 |
| O3 | `allocate_state`: 4 `hipMalloc` (GDN 18.000 MiB + conv 1.266 MiB + KV 6/12/48 MiB at 512/1,024/4,096) then `st.reset()` = `hipMemset` of GDN+conv | outside | per-process | `decode_run.cpp:1700`; `state.cpp:96-104`, `:118-126` | **[derived]** `hipMalloc` reserves pages and transfers nothing (`devices/strix-halo-unified-memory.md`: "the hipMallocs (page reservation, no transfer)", `loader.h:80`); the memsets commit 19.3 MiB → at the quoted warm device rate 19.3 MiB / 239 GB/s ≈ **0.08 ms** |
| O4 | `allocate_scratch`: ~45 `hipMalloc` + 1 `hipHostMalloc` | outside (allocation) | per-process | `decode_run.cpp:1701`; `decode_step.cpp:210-318` (`alloc<>` helper `:146-150`) | **[derived]** sum of the alloc calls at `wave_rows=512` = **138.5 MB** (`prototype/perprompt/a7_scratch_bytes.py`); allocation itself reserves only |
| O5 | hipBLASLt `build` + `gemm_route_plan(512)`: handle, 128 MiB workspace `hipMalloc` (`kernels_gemm.cpp:124-125`), 7 heuristic queries, printed as the `GEMM`/`GEMM-LT` lines before `PREFILL` | outside | per-process + per-shape | `decode_run.cpp:1726-1740`; `kernels_gemm.cpp:746`, `:756`, `:104-141` | **[read]** runs before `t_wave`; the `GEMM-LT … source=heuristic-rank-0-per-shape heur_returned=8` lines precede `PREFILL` in every log, so the heuristic queries are already excluded. This kills §49's "`--gemm lt` shape/algo work at a new shape despite `--gemm-pins all`" as an explanation of the constant *for these widths* — with one exception in 1c. |
| O6 | `prepare_weight_adapters`: 36 `cast_bf16_to_f32` launches (dt_bias, conv1d weight) + one `hipDeviceSynchronize` | outside | per-process | `decode_run.cpp:1753`; `decode_step.cpp:376-415` (sync at `:413`) | **[derived]** bytes widened = 18 × (16 f32 + 6144×4 f32) = **1.77 MB** written, read 0.89 MB → ≤ 0.03 ms of device time plus one sync |
| O7 | printing the prompt's rows: 2 lines per prompt token (`prefill …` + `STEP kind=prefill …`) to stdout, plus `write_raw` of the priced row's logits (0.99 MB) and `sampled.txt` | outside | **per-prompt** | `decode_run.cpp:1803-1850` (`write_raw` at `:1849`) | **[derived]** 8,192 stdout lines at 4,096 tokens (2 × T) and one 0.99 MB file. Width-proportional, not fixed; excluded from `ms_total`, so it is *free* in the headline and would become visible in any end-to-end wall-clock comparison. |
| O8 | teardown: `gemm_route_shutdown`, `free_scratch`, `free_state` | outside | per-process | `decode_run.cpp:1998-2000` | **[read]** frees only |

### 1b. Costs inside `ms_total` **and** inside `wall_ms` (in both operands → cannot produce the gap)

| # | cost | class | path:line | size |
|---|---|---|---|---|
| B1 | the wave's own launches: 271 device calls at 512 lanes, 2,618 at 4,096 (LAUNCHSTAT / `launches=` on the stage TOTAL line) | per-prompt | `prefill.cpp:73` → `submit_wave_body` `decode_step.cpp:428+` | **[quoted]** host submit is not free but it is bounded: `WAVEGAP … exposed_ms=10.8341 ratio=1.025` for 2,083 serialized waits at 4,096 lanes (`tmp/attn_percost/a6g1ab/base_1.log`) ⇒ a drain costs ~5 µs; §6 also records dispatch-bound rows as bimodal. Whatever this is, both operands pay it. |
| B2 | per-body prompt H2D: `n` ints from the pinned ring slot (`hipMemcpyAsync`), with a `hipEventSynchronize` on ring reuse | per-prompt (per-body) | `decode_step.cpp:506-536` (`hipMemcpyAsync` `:535`, event wait `:529`) | **[derived]** 512 ints = 2 KB per body, 16 KB at 4,096 lanes. At the quoted bulk H2D 76–82 GB/s that is **0.2 µs/KB**; even at the engine's pathological 27.7 GB/s chunked figure (`devices/strix-halo-unified-memory.md`) it is **0.6 ms** at worst, and it is in both operands. |
| B3 | `finish_token`: cast row + `lm_head_full` + **one** blocking D2H of `kVocab` F32 + host argmax over 248,320 floats | per-prompt (1 row under `--prefill-logits last`) | `decode_step.cpp:1023-1060` (D2H `:1046`, argmax `:1048-1050`) | **[derived]** 0.993 MB D2H at the quoted 63–70 GB/s bulk = **14–16 µs**; at the engine's measured 27.7 GB/s chunked = 36 µs; host argmax of 248,320 floats ≲ 0.1 ms. `lm_head` itself is in the wave clock: `WAVE gran=stage family=lm_head ms=2.2003`. |
| B4 | `st.reset()`'s `hipMemset` of GDN+conv inside `wave_pass` | per-pass | `decode_run.cpp:1406`; `state.cpp:118-126` | **[derived]** 19.3 MiB ≈ 0.08 ms **[quoted rate]**, and it sits *before* the pass's own timer, so it is excluded from `wall_ms` — an asymmetry of ~0.1 ms in the wrong direction, not an explanation. |
| B5 | `PrefillRow` vector growth (one row per prompt token, host PODs) | per-prompt | `prefill.cpp:47-66` | **[read]** `reserve`d; no device work |

### 1c. Costs inside `ms_total` and **not** inside `wall_ms` — the only things the constant can be

| # | candidate | class | path:line | bound |
|---|---|---|---|---|
| **F1** | **first touch of the freshly `hipMalloc`'ed device bytes the wave writes**: scratch (138.5 MB), KV (`cap` × 12 KiB/token: 6.3 / 12.6 / 50.3 MB), hipBLASLt workspace (128 MiB, only the part an algo actually uses; `ws=134217728` on the two pinned shapes). GDN+conv state were already committed by O3's memsets; **the weights were already warmed by O1's pass**. | **first-only** | scratch `decode_step.cpp:210-318`; KV `state.cpp:96-101` + not zeroed `:118-126`; workspace `kernels_gemm.cpp:124-125` | **[derived]** `prototype/perprompt/a7_scratch_bytes.py`: cold bytes ≈ **0.279 GB at 512 lanes, 0.323 GB at 4,096** (the middle term is 1,024). At the quoted 35 GB/s first-pass rate vs the quoted 239 GB/s steady rate the *extra* cost is **6.8 / 7.0 / 7.9 ms** at those widths, and ≤ 17.9 ms if the true first-touch rate were 20 GB/s. **This bounds §49's "first-touch page-warming" candidate at under ~8 ms, not ~65.** |
| **F2** | **first execution of each library algorithm** — `hipblasLtMatmul`'s first call per (shape, algo) loads that kernel's code object and sets up its device-side state. The heuristic query is excluded (O5); the *execution* is inside the first wave. 7 shapes × 8 waves at 4,096 lanes, but only 7 distinct algos pay once. | **first-only**, per-shape | `kernels_gemm.cpp:862-885` (`hipblasLtMatmul` call site; lazy plan build for unplanned widths prints its own line) | **[derived]** cannot be bounded from the ledger — no library-first-call number exists here. Falsified/confirmed by the `--gemm hand` leg (§3 H2), which needs no patch. |
| **F3** | **first execution of each engine kernel**: per-kernel driver work on the first dispatch (image map/patch, program-slot binding, per-kernel scratch). The *file-reading* version of this hypothesis is dead — see the evidence column. | **first-only**, per-process | every launcher; first execution happens in `prefill.cpp:73`; kernel inventory = the `--time-kernels` table in `progress/2026-10-21/performance-status-2026-10-21.md:1966-1998` | **[read]** `prototype/perprompt/a7_codeobj_census.py`: the whole device image set is a single **`.hip_fatbin` of 413,624 B (0.39 MiB)** embedded in `engine/build/decode_run`, with `.hipFatBinSegment` and `__hipRegisterFatBinary` present (registration runs **before `main`**) and **no `external_code_object` marker**. So there is no kernel file to read at first launch, and even re-touching the entire image set inside the wave would cost 0.41 MB / 35 GB/s = **0.012 ms**. F3 survives only as *driver bookkeeping per first dispatch*, not as image loading. **[read]** the usual env kill-switch does not exist on this runtime anyway: `HIP_MODULE_LAZY_LOADING` and `HSA_ENABLE_MODULE_LOAD_DELAY` are absent from `libamdhip64.so.7` / `libhsa-runtime64.so.1` (`strings -a`, 2026-10-19). |
| **F4** | **DVFS / clock ramp**: the process spends 0.4–0.5 s loading (O1) with the GPU mostly idle, then the wave is the first sustained burst. §49's own clock line reads `sclk 2573 MHz at start, 951/933 mid/end`. | **first-only** | not in engine code; a property of the window | **[quoted]** §49's clock line. **Disfavoured by the shape of the data**: a clock ramp costs a *fixed time window* of slowed work, so the excess should grow with the wave's own length and saturate; §49 measured 71.1 / 70.9 / 64.9 ms at waves of 48.7 / 96.2 / 430 ms — flat. A 2.7× ramp (951→2573) applied to the whole 48.7 ms warm wave would predict ~83 ms excess at 512 lanes and ≥ 100 ms at 4,096, i.e. it does not fit the 64.9 at 4,096. Falsifier in §3 H4. |
| **F5** | **per-shape hipBLASLt plan build *inside* the wave for a remainder width** — `gemm_bf16` builds and caches a plan for a width `gemm_route_plan` did not prewarm, and prints a `GEMM-LT` line to do it | **first-only**, per-shape | `kernels_gemm.cpp:865-872` (lazy `lt_plan_for`, printed at `:874`) | **[read]** zero for §49's widths: 512 / 1,024 / 4,096 are all exact multiples of `wave_rows=512`, so every body is M=512 and pre-planned. Non-zero for any prompt that is not a multiple of 512 — check `GEMM-LT` line count vs `bodies` in any log. |

---

## 2. The warmup question, answered — **say it loudly**

The four static checks the brief asked for:

**(a) Does the bracket around `RESULT ms_total` include the model load and the first device
touch?** Not the load — `load_ms` is separate and never added (`decode_run.cpp:1588-1593`);
the first device touch of the *weights* is inside `load()` (`loader.cpp:626-644`) so it is
excluded too. What it does include, and what no ledger note has said, is **the first device
touch of everything that is not a weight** — scratch, KV, the hipBLASLt workspace — and the
first execution of every kernel and library algorithm.

**(b) Is there a warmup pass before the timed region?** Partially, and only for weights:
`loader.cpp:626-644` warms weight pages (that is its documented contract, `loader.h:14-15`).
There is **no warmup pass for scratch, the hipBLASLt workspace, kernel code, or the GEMM
library** before `t_wave`. (`st.reset()`'s memsets at `state.cpp:118-126` do commit the GDN and
conv pages as a side effect, so of the device bytes the wave touches only KV, scratch and the
library workspace are truly cold — that is what F1's byte census counts.) So `RESULT ms_total`
is a *semi-cold* number and has always been one.

**(c) Does `--time-waves` bracket something narrower?** Yes, and this is the load-bearing
answer: it brackets **a second execution of the same wave** (`decode_run.cpp:1406-1421`),
i.e. an already-warm pass. §49's subtraction is therefore cold-minus-warm of identical work,
not "work outside the wave".

**(d) Does `llama-bench` exclude warmups?** **Yes, by default, and it is a full-length one.**
`engines/llama.cpp/tools/llama-bench/llama-bench.cpp:2415-2428` runs
`test_prompt(ctx, t.n_prompt, …)` — the *entire* prompt length — plus a `test_gen(ctx, 1, …)`
warmup, before the timed loop; `no_warmup` defaults to `false` (`:422`, flag `:1057`), and the
timer starts at `:2484` after it. `tools/llama_phase_compare.sh` passes no `--no-warmup`, so
every llama row in §21/§38/§46/§49 is a **warm** pass.

### Verdict

**Our headline includes per-process first-execution cost and llama.cpp's excludes it. Both
directions of that error are large at short context.**

* Against us: at 512 lanes, cold `ms_total` ≈ 144.5 ms **[read]** vs warm `wall_ms` = 48.7 ms
  **[quoted §49]** vs llama's warm pp512 ≈ 69 ms **[quoted §49: 7,378.72 tok/s]**. A
  warm-vs-warm comparison reads **10,513 tok/s vs 7,379 tok/s = we are ~1.43× FASTER at 512,
  not 2.08× slower.** **This specific magnitude is conditional on H1a** — it assumes the ~96 ms
  cold/warm difference those two numbers straddle is real at 512 lanes. If leg A instead reads a
  width-proportional gap (H1b), the warm 512 number is nearer 65 ms, the warm-vs-warm ratio is
  roughly parity (~7,900 vs 7,379), and the honest statement is "we are ~1.07×, not 2.08×".
  **Either way §49's short-context ratio is methodology-biased against us**; only the size of
  the correction is open. The §46 `a` coefficient (1.285×, now 1.207×) is inflated by the same
  artifact — §46 already guessed in this direction ("per-process page-in of the weights (the
  35 GB/s first-pass trap)") but attributed it to the wrong pages: the weights are warmed, the
  rest is not.
* Against us in the *other* direction: for a serving engine the first prompt of a cold
  process is a real user-visible cost, so the warm number is a methodology choice, not a
  truth. What must not stand is a headline that mixes the two: **every `RESULT ms_total`
  headline in this ledger (§25's 763.49 ms, §36's banner, §44's 535.105, §48's 7,986 tok/s)
  is a cold-first-wave number, and every llama-bench row is warm.** Re-taking our banner with
  a discarded warmup pass would move the 4,096 headline by ~+13–19 % and the 512 row by ~+3×,
  and the llama ratios with it.

This is a **methodology decision for the user/main agent**, not something I should pick: the
choices are (i) report both (cold-first and warm) and label which a ratio used, (ii) adopt a
warmup in our ruler to match llama-bench's default, or (iii) keep cold and pass
`--no-warmup` to llama-bench (which is the only way to make the *current* headline shape
honest). (iii) is the cheapest and touches no engine code.

---

## 3. Ranked hypotheses, each with the number that must move

Let `gap(W) = RESULT ms_total − WAVEPASS gran=burst wall_ms`, both read from **one process**.
§49's published (instrument-mixed, see H6) numbers: 71.1 / 70.9 / 64.9 ms at W = 512 / 1,024 /
4,096.

**The constant is real, and it has been measured inside one process before.** §36's ruler table
(`progress/2026-10-21/performance-status-2026-10-21.md:1640-1644`) prints both operands **from the same run**, three reps
at 4,096 lanes: 705.863−626.619 = **79.2**, 708.141−627.754 = **80.4**, 714.855−629.504 =
**85.4 ms**; §44's pair (line 2008) gives 535.105−453.65 = **81.5 ms**; §38 (line 1784) names
"the 82 ms between kernel sum and RESULT". So at 4,096 lanes the in-process gap is ~**80 ms** —
bigger than §49's 64.9 — and as a *fraction* of the warm wave it reads 12.6 % (§36), 18.0 %
(§44) and 20.6 % (§48, whose pair is best-of-5 against best-of-3 and so is a shape rather than a
same-run pair). Those two readings, flat-in-ms and ~15 %-of-wave, imply
opposite remedies, which is the fork the measurement turns on.

| rank | hypothesis | predicted magnitude | signature (what moves, by how much) | how falsified |
|---|---|---|---|---|
| **H1a** | **fixed-in-ms first-execution work** inside the first wave (library first call + kernel first launch + page first-touch) | ~65–96 ms at *every* width | `gap(512) ≈ gap(4096)` in one process; a discarded warm wave drops `RESULT ms_total` by that amount at every width (512: 144.5 → ~49 ms = **+195 % tok/s**, 4,096: +13–19 %) | one-process `gap(512)` reads single-digit ms |
| **H1b** | the gap is **proportional** to the wave (~15 % of it): the first pass of the kernels themselves is slower (DVFS ramp, cold caches), and §49's short-width rows came from two different instruments | ~6–10 ms at 512 lanes, ~80 ms at 4,096 | `gap(512)/warm(512) ≈ gap(4096)/warm(4096)`; §49's 71.1 at 512 does **not** reproduce in one process. Then the §49 sentence must be restated as "~15 % of the wave, first pass only", and the short-context ratio does *not* flip sign | one-process `gap(512)` ≈ 70–96 ms |
| **H2** | a **large slice is hipBLASLt first-execution** (F2) rather than pages or kernels | 0–60 ms of the gap | `--gemm hand`: `gap(hand) < 0.2 × gap(lt)` ⇒ F2 owns ≥ 4/5. Compare **differences**, never walls — the hand wave is itself slower, so a ratio of walls would lie | `gap(hand) ≈ gap(lt)` within 20 % → the library is not it |
| **H3** | page first-touch is **small** | ≤ **7.9 ms** at 4,096 lanes, ≤ 6.8 ms at 512 **[derived]**, `a7_scratch_bytes.py`, from 0.28–0.32 GB cold-touched at the quoted 35 GB/s vs 239 GB/s | any pages-only warmup can recover at most this much (≤ 17 ms even at a 20 GB/s first-touch rate) | a pages-only intervention recovers ≫ 8 ms → the byte census is wrong (re-check `a7_scratch_bytes.py` against `decode_step.cpp:210-318`; `a_part`'s 33.8 MB is quoted from a comment, not derived) |
| **H4** | DVFS / clock ramp (F4) — the mechanism behind H1b | proportional, never flat | §49's own clock line reads `sclk 2573 at start, 951/933 mid/end`. If a sustained pre-burst before the timed wave leaves `gap` unchanged, DVFS is out. `WAVEPASS stage > burst` on a later pass would say clocks were still climbing; `WAVEGAP exposed_ms=10.83` **[read]** says the queue is already drained by pass 3 | `gap` unchanged by a pre-burst → not DVFS |
| **H5** | it is **legitimate per-prompt work that cannot be removed** | ≤ ~0.5 ms total from B2+B3+B4 **[derived]** | — | already falsified on paper: every per-prompt cost outside the kernels is ≤ 0.5 ms and sits in *both* operands. The honest form of "legitimate" is H1a: per-process first-execution, real for a cold process, amortisable across prompts in a server — not per-prompt work |
| **H6** | §49's arithmetic mixed instruments | the whole 65–71 ms at the short widths | The §49 window's own bench-leg logs contain **no `WAVEPASS` line** **[read]**, and the medians behind its 3,544 tok/s read 144.5 ms, not 119.83 → that pair is cross-process by construction | the one-process re-take (§4) reproduces ~65–71 ms at 512 lanes |

---

## 4. The handover measurement (main agent runs the timed legs; I did not)

Five files here, all CPU-validated:

| file | what it is | validated how (no timings) |
|---|---|---|
| `a7_host_split.sh` | the driver: legs A–E, one `gpu_lock` hold, medians, falsification table. Legs A/B need no patch; C/D/E take `--worktree BINARY` and set `A7_PHASE=1` (+ `A7_REPEAT=3` / `A7_WARMUP=1` / `A7_BODY=1 A7_SYNC1=1 A7_REPEAT=2`) | run end-to-end against `a7_stub_decode_run.py`; every leg's count assertion fires correctly, including "8 phase lines or abort" |
| `a7_phase.py` | anchor-based instrument for a **scratch worktree** (15 asserted anchors; refuses to write if any anchor is not unique) | `--check` on the clean tree prints `anchors: 15/15`; applied to `git worktree add … HEAD --detach` and built with `engine/build.sh` → `built engine/build/decode_run`, `strings` shows the four `A7*` tags |
| `a7_lib.sh` | the readers (`kv`, `wave_wall`, `load_ms`, `med`, `diff_ms`), each of which **dies** on an empty parse | `a7_parse_selftest.sh` |
| `a7_parse_selftest.sh` + `a7_fixture.log` | positive/negative parse tests, including "the warm wall is missing ⇒ abort, never 0" | prints `SELFTEST PASS` (20 checks) |
| `a7_scratch_bytes.py` | the F1 bound: bytes the first wave first-touches, priced at the quoted cold/warm rates | prints 0.279–0.323 GB → **6.8 / 7.0 / 7.9 ms** extra |
| `a7_codeobj_census.py` | the F3 bound: are kernel images embedded or fetched at first launch | prints the `.hip_fatbin` triple (addr/off/size) plus the registrar/external-marker verdict; on `engine/build/decode_run`: 413,624 B embedded, registered before `main`, no `external_code_object` ⇒ **image load cannot be an in-wave cost (≤ 0.012 ms)**; prints the same verdict structure for `/bin/ls` (all absent) so a wrong-ELF mistake is visible |
| `a7_stub_decode_run.py` | device-free stand-in that shapes `A7*` output from the environment, so the driver's count assertions are testable | used by the end-to-end stub run |

**Commands for the main agent** (blocking, zero subagents, one hold; legs A+B need no build):

```bash
# CPU-side, any time — proves the instrument still matches the source and the readers fail loudly
python3 prototype/perprompt/a7_phase.py --check
bash prototype/perprompt/a7_parse_selftest.sh
bash prototype/perprompt/a7_host_split.sh --dry-run

# DEVICE, one hold, ~4 min: legs A+B with the shipping binary (no patch, no rebuild)
TAG=a7-hostsplit WIDTHS="512 1024 4096" REPS=3 bash prototype/perprompt/a7_host_split.sh

# DEVICE, same hold or a second one: add legs C/D/E from a scratch worktree (never engine/ in place)
WT=$(mktemp -d)/a7wt && git worktree add "$WT" HEAD --detach
python3 prototype/perprompt/a7_phase.py --apply "$WT" && (cd "$WT" && bash engine/build.sh)   # ~20 s
TAG=a7-hostsplit WIDTHS="512 1024 4096" REPS=3 \
  bash prototype/perprompt/a7_host_split.sh --worktree "$WT/engine/build/decode_run"
git worktree remove "$WT"
```

A ready-built one exists from this task: **`tmp/a7wt2/engine/build/decode_run`** (HEAD `277d8c5`
+ the current `a7_phase.py`, built after the `A7_PHASE` gating landed). `tmp/a7wt` is an
**earlier, un-gated build** of the same patch — its `A7PHASE` lines print unconditionally; use
`a7wt2`, or rebuild. The binary was checked to run: legs see `RESULT`, 8 `A7PHASE` lines under
`A7_PHASE=1`, and it reproduces the shipped binary's sampled token and top logit (§4 tail).

The driver holds `tools/gpu_lock.sh` under `TAG` (default `a7-hostsplit`), prints `util` at
acquire and release and `sclk` at start and end, `export LC_ALL=C`, writes only under
`tmp/a7/`, uses no `--ack-spill`, touches no gate, and aborts (releasing the lock) on the first
field that does not parse. Legs C/D/E run the **patched** binary; the patch is env-gated, so
without `A7_*` it prints exactly what the shipped binary prints (verified below).

**Falsification map.** gap(512) ≲ 10 ms ⇒ H1b (proportional), and §49's short-width rows must
be restated. gap(512) ≈ 65–96 ms ⇒ H1a (fixed first-execution). `gap(hand) < 0.2 × gap(lt)` ⇒
H2 (hipBLASLt first call). Any pages-only intervention above ~8 ms falsifies H3's byte census.
`gap` unchanged by a pre-burst ⇒ not H4. leg D's `warm_tok_s` vs §49's llama column decides
whether the short-context ratio flips sign — the methodology question in §2.

**Device check I did run** (correctness-only, own TAG, allowed by the brief). After building the
patched binary in a scratch worktree, two blocking 512-lane runs under
`tools/gpu_lock.sh acquire a7-gate-check` (**util 92 % at acquire, lock free at release; wall 2 s
for both runs together; no timing taken** — the ms fields were never read into this report):

* **The instrument is inert by default.** The patched binary with **no `A7_*` in the environment
  emits zero `A7` lines**, and its `RESULT mode=run steps=512 last_sampled=75 prefix_len=512
  stopped=0` plus `STEP kind=prefill index=511 … sampled=75 top_logit=13.2657185` are identical
  to the shipped binary's. So a patched binary can be handed to a leg without changing what any
  other parser sees.
* **With `A7_PHASE=1 A7_WARMUP=1 A7_REPEAT=3 A7_BODY=1 A7_SYNC1=1`** it emits 8 `A7PHASE` lines
  with exactly the names the driver asserts (`check_weights`, `state_alloc_and_reset`,
  `scratch_alloc`, `gemm_route_init`, `gemm_route_plan`, `prepare_weight_adapters`,
  `row_prints_and_dumps`, `teardown`), 4 `A7REPEAT` (1 warmup + 3 repeats), and 5 `A7BODY` /
  5 `A7SYNC1` (one per wave execution: warmup + run + 3 repeats). The engine's own
  `WAVEPARITY … logits_bitidentical=1` was green in the earlier build of the same patch.

The only line-count difference from the shipped binary was `ATTN_FATTN_SKIP`, which each extra
wave prints by design (`decode_run.cpp`'s own note on that field). Scratch worktrees and stub
logs live under `tmp/`.

## 5. What this cannot see

* **No device number from me.** Every magnitude above is quoted or derived; the ranking of
  F2 vs F3 (library vs kernel first-launch) is not decidable on paper and is exactly what
  legs A/B/C/E decide.
* **`a_part`'s 33.8 MB is quoted from a source comment** (`decode_step.cpp:292`), not
  re-derived from `attn_part_floats`; if the split-K layout moved, F1's byte sum moves with
  it (≤ ~2 ms of the bound).
* The patched leg measures **host-side** phase walls plus one synced first body. It cannot see
  GPU-side per-kernel durations, so a clean split of F3 (first launch) from F1 (pages) inside
  the first wave still needs a `rocprofv3 --kernel-trace` pass with the pass-count discipline
  `AGENTS.md` records (dedup by `(Dispatch_Id, Counter_Name)`, derive the pass count) — a
  separate item whose parser I did not scope.
* The F1 byte census counts a buffer as cold if the wave could touch it; a buffer only
  partially touched costs less than the bound, so 6.8–7.9 ms is an upper bound, not a forecast.
* Clock state is recorded, not controlled; `sclk` on this box is read, not pinned (§6), so H4
  can be bounded by the shape of `gap` but not measured directly here.
* **F3 survives only in its driver-bookkeeping form.** `a7_codeobj_census.py` proves the images
  are embedded and registered before `main`, so the image-*load* variant of F3 is bounded at
  0.012 ms and is out. What it cannot bound is the driver's per-first-dispatch bookkeeping
  (map/patch/slot bind) — unpriceable from source. `HIPLoader` already prints `module
  register/fini` walls and `first_exec_offset_us` per kernel under `A7_VERBOSE=4`
  (`engine/loader.h:59-62`), an option that predates this work and that I did not run; a
  one-line leg setting it would settle F3 without any new instrument.
* **The census tool I wrote first could not fail.** It matched `.amdgpu_fat_*` **section names**
  in the host ELF, found 0, and would have printed "0 code objects" — those names live *inside*
  the `.hip_fatbin` container, not beside it. The rewritten tool prints the section's own
  addr/off/size triple for exactly that reason (and taking the second hex field instead of the
  third reports the file offset 90,112 B as the size, another plausible wrong number). Same
  class as `AGENTS.md`'s "a census must name the files it could read".
* If the engine later ships a warmup-by-default, every `RESULT ms_total` row already in the
  ledger becomes a cold-pass row and needs re-labelling — that is a decision, not a
  measurement, and it belongs to the user.
