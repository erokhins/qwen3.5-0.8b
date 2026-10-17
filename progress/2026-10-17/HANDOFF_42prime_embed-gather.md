# D30 4.2′ — implemented: one gather kernel replaces the 4096 per-token `E[y]` row copies

Base: `HEAD = 63628b8` (4.1-a is `26cd5d6`, inside it). Rebuilt first, then implemented.
**Nothing committed, nothing staged, nothing in `bench/`, no gate file touched, no band touched.**

```
engine/kernels_embed.cpp   NEW (80 lines)   the gather kernel + its launcher
engine/decode_step.cpp     +62/-8           the §02.3 site, the staging alloc/free, the header table note
engine/decode_step.h       +32              the staging fields + why they are pinned and ringed
engine/kernels.h           +14              the launcher declaration + the "two non-graduated launchers" note
engine/stage_clock.cpp     +24/-8           `wave_plan_bill`'s `copy` row learns the row-count rule
engine/build.sh            +1/-1            link kernels_embed.cpp
tools/build.sh             +1/-1            same, for `route_c_stages`
```

---

## Design answer 1 — the token ids on device

**No device-resident id array exists for the wave path.** `submit_wave_body` takes a host
`const int* tokens`; its two callers pass host memory — `prefill.cpp:73` passes
`tokens.data() + begin` out of a host `std::vector<int>`, and `submit_token_body` passes `&token`, a
host local. Grepped `DecodeScratch`: every device buffer there is an activation or a state tile, none
holds ids. So the prescribed shape stands: **one H2D upload of the body's id array (≤512 ints) plus one
gather — 8 uploads + 8 gathers per 4096 wave, replacing 4096 submissions.** No per-token id copy anywhere.

Three things that shape took real work to get right, all of them stated in the code:

1. **The kernel.** `embed_gather_kernel`: one block per wave row, 256 lanes, each lane moves one `uint2`
   (4 BF16). `kHidden = 1024` BF16 = 2048 B per row = exactly 256 chunks, `static_assert`ed, and both a
   source row and a destination row start on a 2048-byte boundary, so every access is naturally aligned
   and covers the row exactly once with no tail. `ids[blockIdx.x]` is one broadcast load per block — that
   is the whole indirection cost. **No arithmetic exists in it**, which is why a bit difference is a bug.
2. **The upload must be from PINNED memory.** `hipMemcpyAsync` from pageable memory is synchronous in
   HIP, and a synchronous H2D on the legacy stream waits on everything queued in front of it — i.e. the
   whole previous body — which would flatten exactly the queue depth this item exists to exploit. So the
   scratch owns a pinned staging buffer (`hipHostMalloc`), written host-side with `std::memcpy`.
3. **A pinned buffer cannot simply be reused by the next body.** The host runs bodies ahead of the device,
   so a single slot would be overwritten while an earlier body's copy still had to read it — a real
   corruption, not a theoretical one (body 3's write lands at ~1 ms host time; body 2's ids are not read
   by the device until the end of body 1, ~120 ms later). The scratch therefore holds **8 slots**, each
   with an event recorded right after its copy, and a body waits on its slot's event before writing it.
   The wait is a correctness bound that caps host look-ahead at 8 bodies; at 512 rows a body is ~120 ms
   of device work, so seven whole bodies stay queued in front of the device at every wait and nothing can
   starve. **The first 8 bodies a scratch serves do not attempt the wait at all, and the shipped 4096-lane
   wave is exactly 8 bodies, so the window this item is measured in is wait-free.** The `wide520f`
   `--wave-rows 5` side (104 wide bodies, 96 ring wraps) exercised the wrap path and came back
   bit-identical.

## Design answer 2 — the one-row path, and the bill deriver

**The deriver can express it; no second source of truth was invented, and `plan_match` is still an exact
integer check.** The `copy` row follows the same row-count rule 4.1-a installed for `cast`, so I hoisted
the one-row-body derivation that `cast` had computed inline and both rows now read it:

```cpp
const long long last_body_rows =
    sh.bodies > 0 ? sh.tokens - (sh.bodies - 1) * sh.wave_rows : 0;
const long long single_row_bodies =
    sh.wave_rows <= 1 ? bodies : ((last_body_rows == 1) ? 1 : 0);
```

### New bill formula

```
old:  p.subs[kCopy] = tokens + G * bodies;
new:  p.subs[kCopy] = single_row_bodies + 2 * (bodies - single_row_bodies) + G * bodies;
                                                    └ one id upload + one gather per WIDE body
                  └ one row copy per ONE-ROW body              └ 18 conv-window rolls per body, unchanged
```

Route-blind, unlike `cast`: which form a body runs depends on `n` and nothing else, so no new
`WaveShape` field was needed. On the body, `n == 1` keeps the single `hipMemcpyAsync` verbatim — a decode
step's submission bill cannot move.

**Measured plan side of the bill, CPU-only probe** (`tmp/gcc/copy_bill_probe.cpp`, built against HEAD and
against the change; `plan_match=0` there is expected — the probe measures nothing, it reads the plan):

| shape | HEAD plan_launches | with 4.2′ | Δ |
|---|---|---|---|
| 1 token / 1 body of 1 row / hand | **424** | **424** | **0** |
| 1 token / 1 body of 1 row / `--gemm lt` | **424** | **424** | **0** |
| 4096 / 8 bodies of 512 / lt (the shipped wave) | 80426 | **76346** | **−4080** |
| 4096 / 8 bodies of 512 / hand | 81050 | **76970** | −4080 |
| 520 / 512+8 / lt | 10532 | 10016 | −516 |
| 513 / 512+1 / lt (single-row tail) | 10477 | 9967 | −510 |
| 4096 / 4096 bodies of 1 row / lt (`--wave-rows 1`) | **1728514** | **1728514** | **0** |

Three things to read off that table. (a) A decode step's bill is **424 on both routes**, and the existing
deriver self-checks `body1_plan_matches_token_plan=1` and
`body1_library_plan_matches_token_plan=1` still print 1. (b) A per-row pass (`--wave-rows 1`, which is one
side of every §11 wave-width comparison in `gate.py`) is **unchanged to the launch**. (c) The 81050 figure
for the hand 4096 wave is exactly the `launches=81050 plan_launches=81050` the census measured on the
pre-4.1-a binary, and 81050 − 624 (4.1-a's cast trim) = 80426 = the pre-change lt figure here. The two
commits' arithmetic reconciles against the census line for line.

`gate.py` needs no change and got none: `copy` has **no launcher counter** in `engine/launch_stats.h`, so
section (h) cannot price it and there is nothing to re-teach — the same fact D37 recorded for `cast` ("a
check over a family with no counter cannot fail"). `planned_launch_counts()`'s `other = 20` is the n = 1
row and is untouched.

---

## Device runs I made — 5, one at a time, under `tools/gpu_lock.sh`, blocking, **no timing taken**

No `ms` figure of mine is in this document, none belongs in `bench/`, and none was read out of the logs I
generated (the `--time-waves` runs were read for **count fields only**: `launches`, `plan_launches`,
`counts_match`, `plan_match`, the two `body1_*` flags).

| # | command | wall | `util` at acquire | result |
|---|---|---|---|---|
| 1 | `.venv/bin/python engine/gate.py --fixture dec64f --gemm lt` | 10.5 s | 94 % | **ENGINE GATE PASS** — with 2 steps TIE (17, 18 — the fixture's own licensed ties, same as base) |
| 2 | `.venv/bin/python engine/gate.py --fixture wide520f --gemm lt --work tmp/gcc/new42-wide520f-lt` | 21.9 s | 93 % | **ENGINE GATE PASS** |
| 3 | `.venv/bin/python engine/gate.py --fixture wide520f --gemm hand --work tmp/gcc/new42-wide520f-hand` | 22.5 s | 94 % | **ENGINE GATE PASS** |
| 4 | same as 3, re-taken on the final binary after the last comment-only edits | 24.0 s | 93 % | **ENGINE GATE PASS** |
| 5 | `engine/build/decode_run --model models/Qwen3.5-0.8B --ids "$(cat progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt)" --steps 0 --time-waves --gemm lt --gemm-pins all --out tmp/gcc/o` | 3.6 s | 91 % | `family=copy launches=160 plan_launches=160 counts_match=1` · `launches=76346 plan_launches=76346 plan_match=1` · both `body1_*=1` |

**160 = 144 conv rolls + 8 gathers + 8 id uploads**, which is the census's predicted `152 + 8`, seen on
device with `counts_match=1` — the counted submissions equal the derived plan exactly, so the deriver and
the body agree about what a wave owes.

### The bit-identity witness (the strict standard)

`gate.py` section (h) on `--gemm hand` compares a wide wave's logits **byte-for-byte** against the same
prompt fed one token per body. The wide side now takes the gather; the one-row side still takes the
memcpy. So that check is gather-against-copy and not gather-against-itself:

```
[PASS] both sides write bit-identical logits — 524 files, 520478720 bytes compared, 0 differing
[PASS] the 100-wave side writes bit-identical logits to the one-wave side — 524 files, 520478720 bytes, 0 differing
```

Independently, the new gate work dirs byte-diff **0 differing** against the pre-change base dirs already
on disk from the 4.1-a window (`tmp/gcc/w-base-wide520f-{hand,lt}`), for all three sides
(`ab_batched`, `ab_perrow`, `ab_carry`) on both routes — 6 × 524 logits files, ~3.1 GB of compared bytes,
0 differing. That chain is transitively bit-identical to the pre-4.1-a binary too, because 4.1-a's own
diff files (`tmp/gcc/diff-*.txt`) are all empty. No numerics band was consulted, because none was
reachable.

---

## Commands for your device pass

Base binary is already saved: **`tmp/gcc/decode_run.pre42prime`** is a build of HEAD taken before I
touched anything. `engine/build/decode_run.base` was not touched and still holds the pre-4.1-a binary —
if you reuse it you are measuring 4.1-a + 4.2′ together, so re-take it from `decode_run.pre42prime`.

```bash
tag="d30 4.2' gather A/B"; tools/gpu_lock.sh acquire "$tag"
trap "tools/gpu_lock.sh release '$tag'" EXIT; tools/gpu_lock.sh util
cp tmp/gcc/decode_run.pre42prime tmp/gcc/dr_42p_base
cp engine/build/decode_run       tmp/gcc/dr_42p_new
```

**§5-C paired A/B** — script written at **`tmp/gcc/run_ab_42prime.sh`** (shape copied from 4.1-a's
`tmp/gcc/run_ab.sh`: paired inside one lock hold, alternating per rep, medians of 3, blocking, zero
subagents). It prints, per rep: `wall_ms`, `copy`/`cast`/`glue` as `ms/launches/counts_match`, `plan_match`
and the two `body1_*` flags.

```bash
bash tmp/gcc/run_ab_42prime.sh 2>&1 | tee tmp/gcc/ab_42p.txt      # then take the medians of 3
```

**The prediction, two-sided, stated before the window is taken.**

* **Confirms (both halves required):** `copy` **10.86 → ≲0.5 ms** (and its `launches` 4240 → 160,
  which is now a `counts_match=1` fact rather than a hope) **AND `wall_ms` ≥ 5 ms lower** at 4096 lanes.
  Then per-op submission cost is on the critical path, 4.2′ is worth its ~8.3 ms ≈ 0.87 % of wave, and the
  other launch-shaped ceilings in D30 keep their standing.
* **Refutes:** the **row collapses and the wall does not** move ≥5 ms. Then submissions are absorbed on a
  busy stream, 4.2′'s real prize is the ~0.07 ms of bytes, and the record has to say that **every
  launch-shaped row in D30 gets re-read** — `glue`'s 0.73 ms "all launches retired" ceiling, the
  `cast` row's 29.25 µs/launch reading (which §8 already says is *not* launch-shaped, so it would survive
  this refutation), and 1.7's "+73 584 submissions = 136.1 ms of dispatch floor" arithmetic, which is the
  one that would need the most careful restatement because it prices submissions that would then be free.
  Record it as a null with the numbers, not as a win: the `copy` row collapsing on its own proves only
  that the submissions were deleted, which was known from the launch counters before any timing.
* Also worth a glance in the same window, and cheap: `glue` and `cast` are the untouched controls, and the
  `--wave-rows 1` side at 1024 lanes should show a **zero** delta in `copy` launches (derived at CPU:
  1728514 → 1728514) — an in-window check that the one-row path did not move.

`wide1024f`, the 9-fixture sweep and every paired A/B are yours; I did not run them.

---

## `ROTTED` list from `tools/citation_rot_check.py` — 37, all mine, nothing in `progress/` edited

```
scanned 408 tracked files against HEAD; checked 780 line-anchor(s); 37 rotted line-citation(s).
```

By file: `engine/decode_step.cpp` 23, `engine/decode_step.h` 10, `engine/kernels.h` 4. Full list in
`tmp/gcc/citation_rot_final.txt`; the pairs are:

```
engine/decode_step.cpp:37   <- progress/2026-10-16/ceiling-based-route-ranking-2026-10-16.md:304
engine/decode_step.cpp:455  <- engine/results.md:2670
engine/decode_step.cpp:469  <- progress/2026-10-16/ceiling-based-route-ranking-2026-10-16.md:304
engine/decode_step.cpp:471  <- engine/results.md:2670
engine/decode_step.cpp:514  <- prototype/gemm_algo_sweep/gemm_algo_sweep.cpp:97 ; prototype/gemm_algo_sweep/README.md:88
engine/decode_step.cpp:517  <- prototype/gemm_algo_sweep/gemm_algo_sweep.cpp:98 ; prototype/gemm_algo_sweep/README.md:89
engine/decode_step.cpp:520  <- prototype/gemm_algo_sweep/gemm_algo_sweep.cpp:98 ; prototype/gemm_algo_sweep/README.md:89
engine/decode_step.cpp:527  <- prototype/gemm_algo_sweep/gemm_algo_sweep.cpp:100; prototype/gemm_algo_sweep/README.md:91
engine/decode_step.cpp:530  <- prototype/gemm_algo_sweep/gemm_algo_sweep.cpp:101; prototype/gemm_algo_sweep/README.md:92
engine/decode_step.cpp:677  <- prototype/gdn_rec_isa/gdn_rec_isa_kernels.h:372 ; prototype/gdn_rec_isa/results.md:337 ;
                               prototype/gdn_rec_isa/traffic_census.output.txt:10,16
engine/decode_step.cpp:682  <- prototype/gdn_rec_isa/traffic_census.output.txt:10,16
engine/decode_step.cpp:687  <- prototype/gdn_rec_isa/gdn_rec_isa_kernels.h:372
engine/decode_step.cpp:722  <- prototype/gemm_algo_sweep/gemm_algo_sweep.cpp:99 ; prototype/gemm_algo_sweep/README.md:90
engine/decode_step.h:49     <- AGENTS.md:315 ; progress/2026-10-16/prefill-decision-log-2026-10-16.md:598 ;
                               progress/2026-10-17/glue-cast-copy-bytes-census.md:76 ;
                               prototype/gemm_algo_sweep/{deep_pins.py:668, results.md:190, README.md:113,
                                                         gemm_algo_sweep.cpp:2794}
engine/decode_step.h:114    <- progress/2026-10-12/plan-prefill-2026-10-12.md:43
engine/decode_step.h:155    <- progress/2026-10-16/prefill-decision-log-2026-10-16.md:26
engine/decode_step.h:178    <- progress/2026-10-14/p9-phase1c-2026-10-14.md:132
engine/kernels.h:48         <- progress/2026-10-16/ledger-prune-2026-10-16.md:102
engine/kernels.h:84         <- progress/2026-10-11/commit-review-2026-10-11.md:13,58
engine/kernels.h:171        <- progress/2026-10-16/ledger-prune-2026-10-16.md:103
```

`wc -l` guard, HEAD → now: `decode_step.cpp` 983→1037, `decode_step.h` 315→347, `kernels.h` 473→487,
`stage_clock.cpp` 519→535; new file `kernels_embed.cpp` 80 lines. **The single most expensive one is
`engine/decode_step.h:49`** (`kMaxWaveRows`, cited 7 times including `AGENTS.md:315`); it rotted only
because the struct now needs `hipEvent_t`, which cost a 2-line `#include <hip/hip_runtime.h>` at the top
of the header — everything below line 17 shifted by 2. Cheapest re-point is that one line. I did **not**
type-pun the events to `void*` to dodge it; clear types beat citation stability, and this is what the
checker is for.

---

## Residual risks / things you should know before you review

1. **Two device ops per wide body are not named by the bill.** The `hipEventRecord` that witnesses each
   id upload is an in-stream command (8 per 4096 wave), and the ring reuse does a `hipEventSynchronize`
   (host-side, zero for the first 8 bodies, immediate thereafter). I did **not** note the event as
   `kCopy`, because it is not a copy and inflating a family named `copy` with a non-copy op is worse than
   an undercount of 8 in 76 346. So the wave's true device-op count is 168 where the clock bills 160. If
   you want the event inside the bill, say so and it is a one-line `note()` and one term in the deriver.
2. **`tools/build.sh route_c_stages` does not link — and did not before my change either.** I had to add
   `kernels_embed.cpp` to its TU list for correctness of the co-change, but the target fails at HEAD too
   with `undefined symbol: hipblasLtCreate` & co: `kernels_gemm.cpp` has needed `-lhipblaslt` since D26
   and that case still passes only `-lhipblas`. Verified pre-existing by building the same target from a
   clean `git archive HEAD` tree (`/tmp/pre42`). Left alone as out of scope; it is a one-word fix
   (`libs="-lhipblas -lhipblaslt"`) whenever you want it.
3. **`tools/marker_audit.py` exits 1 on the tree and did so at HEAD**, with the same 7 flagged sites
   (the `project()` lambda's two launchers, `wave_gemm_is_library`, two `resolve_gdn_scan` tests, and the
   multi-line `mlp_gateup`/`mlp_down` calls). My change moved that count from 31 sites/24 marked to
   33/26 — the same 7 pre-existing entries at new line numbers, and `--break-note 3` still drops `marked`
   to 25, so the audit is not vacuous. Not caused by me and not fixed here.
4. **The ring's blocking reuse is untested for a prompt past 8 bodies at 512 rows** — no shipped fixture
   has one (`wide1024f` is 2 bodies). It is exercised structurally by the `wide520f --wave-rows 5` side
   (104 bodies, 96 wraps, bit-identical) and the derivation is witnessed on device for the mixed case by
   the `--wave-rows 8` / 17-token run: `family=copy launches=59 plan_launches=59 counts_match=1`,
   `plan_match=1` — i.e. the 8+8+1 split bills 2+2+1 copies correctly. A future `--wave-rows 512` run on
   a >4096-token prompt is the missing case; it should be a correctness check only, since the host
   look-ahead cap lands at exactly 8 bodies there.
5. **`hipMemcpyAsync` H2D and `hipEventRecord` behaviour is relied on, not measured.** The correctness
   argument (pinned + event + stream order) is the standard double-buffer pattern and the bit-identity
   evidence above is its witness; no timing claim of mine rests on it.
6. Not touched, deliberately: the conv-window member of `copy` (144 rolls, 0.37 ms ceiling, and an
   aliasing contract at `decode_step.cpp:588-593` that forbids the in-place form), and folding the gather
   into glue's layer-0 launch (0.071 ms, not worth a signature change). `engine/kernels_cast.cpp`'s stale
   per-token header count is still the census's doc fix to make, not mine.
7. `speccheck/run_all.py` was **not** run: `speccheck/specimpl.py` is the torch CPU reference and no gate
   file was touched, so nothing in that suite could be affected by a change to an engine launcher. Per the
   standing budget rule I cited the existing green instead of re-making it. `engine/gate.py` is byte-for-byte
   unchanged.
