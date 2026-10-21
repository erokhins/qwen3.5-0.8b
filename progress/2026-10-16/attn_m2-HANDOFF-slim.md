# `prototype/attn_m2` — handoff for the slim port (device half)

Written 2026-10-16 (the same day as §2–§4's window, later) by the worker that did the port. **Everything
below is a device run and a timing, so it
is the main agent's, per AGENTS.md.** The worker took no timings, no device numbers, and no lock.

`gpu_lock` status at the time of writing (`tools/gpu_lock.sh status` at 2026-10-16T12:54Z, host-side read
only):

```
STALE (holder pid dead): 2026-10-16T09:45:30Z	attn-touch-decode2	pid=3357109 — acquire will reclaim it
```

i.e. the lock is **free** (a dead holder is reclaimed on acquire). `prototype/attn_m2/run.py` does **not**
self-lock — check with `grep -c gpu_lock prototype/attn_m2/run.py`, which returns only comment lines — so a
caller may hold the lock across the whole sequence without deadlocking it. Three other drivers in this tree
do self-lock (`prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`);
none of them is in this sequence.

The reading of §8 of `results.md` you are supplying: **the prediction is that m2-slim will not pay** (1.77–
2.02 ms vs ship-slim's measured 1.5416 ms at wave64-at-4096). Step 3 is what reopens it, step 4 is what
confirms it, and §8.4 names which number means which.

## 0. Before anything, above the knee, zero subagents

```
tools/gpu_lock.sh acquire attn-m2-slim     # then run everything below inside this one hold
tools/gpu_lock.sh util                     # record it; discount it (identical clean runs read 24-91 %)
```

## 1. Build and host-only guards (no device, safe to repeat)

```
bash prototype/attn_m2/build.sh
./prototype/attn_m2/attn_m2 --geom-guard                                        # host-only
.venv/bin/python prototype/attn_m2/m2_model.py selftest                         # 33 checks, 0 failures
.venv/bin/python prototype/attn_m2/run.py selftest                              # 12 stub cases, 12/12
.venv/bin/python prototype/attn_m2/run.py plans                                 # rewrites /tmp/attn_m2/plan-*.txt
```

Expected: `built ./attn_m2`; `geom-guard: PASS` with 13,744 geometries / 518 of 520 / 0 phantom / 0 leak /
16 real plan geometries; both drivers green. The selftest now also refuses a *second* spelling of the score
dot or the fold in `attn_m2_kernels.h` — if that check ever goes red, the port has been forked, not ported.

One of those gates is worth running deliberately, because it witnesses the rule that keeps step 2 from being
decoration: the stub prints only the two forms of record, so at `--slim 1` the driver must call the missing
pairs out rather than average over what arrived:

```
.venv/bin/python prototype/attn_m2/run.py ab --only pair-at-maxsc --bin prototype/attn_m2/stub_harness.py --slim 1
```

Expected, CPU-only: two `verdict=NO-KEY` lines and a non-zero exit. If that command ever prints PASS, the
pair table and the RESULT parser have come apart and step 2 is measuring nothing.

## 2. Bit-identity admission, three pairs in one process (device, correctness only)

```
.venv/bin/python prototype/attn_m2/run.py ab --slim 1
```

Requires, on every row of all 12 plans, `ndiff_aout=0 ndiff_gated=0` for **all three pairs**:

| pair | RESULT keys | what it proves |
|---|---|---|
| m2 vs ship | `ndiff_aout` / `ndiff_gated` | the pair of record (§2), unchanged |
| m2slim vs m2 | `ndiff_aout_m2slim_vs_m2` / `ndiff_gated_m2slim_vs_m2` | the two levers change no value on the m2 body |
| m2slim vs ship | `ndiff_aout_m2slim` / `ndiff_gated_m2slim` | the shipping claim |

`--slim 0` (the default) prints only the pair of record, which is the gate every number in `results.md` was
taken on — re-run it if you want the pre-port gate re-confirmed under the new binary. A `verdict=NO-KEY` or
`verdict=NO-LAUNCH` line is a failure, not a skip; the driver aborts non-zero on both. If any row differs:
**stop and report the geometry and the cause** (`ROWDIFF pair=… case=… row=… pos=… lanes=…`). Do not loosen
a gate to route around it.

Budget note: one process now runs three kernels per case and uploads one extra cache copy per case — each
form needs its own replica because the §05.7 append writes the cache in place (`attn_m2.hip.cpp`, the
`upload_forms` comment). Expect roughly 1.5× the `ab --slim 0` wall time, dominated by uploads, not by the
kernels.

## 3. Negative controls with the slim body in the ring (device)

```
.venv/bin/python prototype/attn_m2/run.py negative --neg dimhalf  --slim 1     # the lever-shaped one
.venv/bin/python prototype/attn_m2/run.py negative --neg maskwide --slim 1     # the m2-range one
.venv/bin/python prototype/attn_m2/run.py negative --neg partrow  --slim 1     # the m2-only one
```

`--slim 1` adds one requirement to each: the slim body's wrong must land on **exactly** the plain body's
rows (`moved_m2slim` is printed beside `moved_m2`, and a mismatch is a problem clause). `dimhalf` is the one
that matters for this port — the dot lever is *about* `nd`, so a `nd`-shaped injection that moves one body
and not the other means the lever changed semantics. If a dump is missing the driver says so instead of
reading a missing file as "nothing moved". §1.2 of `results.md` is where these injections and their derived
sets are written up.

## 4. The bursts — the price, in one ruler per row (device, main agent only)

One plan per command, blocking, never backgrounded. The burst path times **every active form inside one
rep** (form loop inner to rep loop), so the four forms below are measured in one window and the ratios are
pairwise A/B, not rulers from different windows — the trap the ledger's "a ruler re-taken in a different
window is not an A/B" row describes. Answer to the standing question: **yes, the harness alternates more
than two forms inside one rep**; `--forms` takes any subset and `ship` is always included as the reference.

```
for p in wave64-at-1024 wave64-at-2048 wave64-at-4096 wave16-at-4096; do
  ./prototype/attn_m2/attn_m2 --plan /tmp/attn_m2/plan-$p.txt \
      --forms ship,m2,m2slim,shipslim --burst 40 --reps 5
done
```

Lines to keep: `BURST case=… form=… ms=…` (every rep, so spread is visible), then per case

```
RESULT case=… best_ship_ms=… best_m2_ms=… best_m2slim_ms=… best_shipslim_ms=…
SPEEDUP a=m2slim b=m2       speedup_a_over_b=…     # do the levers pay on the m2 body?  predict ~1.4x
SPEEDUP a=m2slim b=shipslim speedup_a_over_b=…     # the shipping question. predict ~0.87x (a loss)
SPEEDUP a=shipslim b=ship   speedup_a_over_b=…     # should reproduce ~1.5x at 4096, or the window is bad
```

`speedup_a_over_b` is `best_b / best_a`, so **> 1 means a is faster**. Do not record a row where
`shipslim/ship` fails to reproduce §valu_slim's ~1.5× at 4096 — that says the window is contaminated, not
that attention got slower.

(a) **m2-slim vs m2-plain**: the `a=m2slim b=m2` line at each width. Prediction ≈ 1.4× at 4096; §8.4 names
the sub-falsifier if it comes in under ~1.11×.
(b) **m2-slim vs ship-slim**: the `a=m2slim b=shipslim` line at wave64-at-1024 / 2048 / 4096 and
wave16-at-4096. This is the pair the ship/don't-ship call turns on, and §8.4 of `results.md` is the one-line
test against `best_shipslim_ms`. wave16-at-4096 is not a repeat of wave64-at-4096: the crossover in §7 of
`results.md` is a block-count story (128 m2 blocks vs 256 tile blocks on 40 CUs), so the two widths can
disagree and that disagreement is itself the answer.

If you want m2-slim vs ship-slim without the m2-plain row crowding the window, `--forms m2slim,shipslim`
still runs `ship` as reference and times three forms in one rep.

## 5. Ledger, and where the numbers go

Timing rows belong in `bench/performance-status.md` and only the main agent writes them; the per-op
`results.md` §4 keeps the prototype-harness ruler. A worker device result never enters either. Record, per
width: the lock tag, `util`, the burst and rep counts, the four bests, and the two speedups — the pairs, not
one side of a pair.

```
tools/gpu_lock.sh release attn-m2-slim
```

## 6. What is still not built, so the next brief does not rediscover it

* No V-split, no occupancy lever. §7 of `results.md` keeps the `s_sc`-into-registers lever (37 % of the
  block's LDS) as the only lever that would change blocks/CU, and it is still unimplemented and unmeasured.
* Nothing in this directory touches `engine/`, and an engine port is a bigger question than the two levers:
  the shipped wave is one warp per head (`engine/attn_slim_kernels.h:163-164`), so a row-tiled engine form
  has to give one block two query rows and re-derive the warp→(row, head) mapping — the mapping this harness
  uses (`attn_m2_kernels.h`, `head = ATTN_M2_G * kv_write + h`) was written for the harness's launch, not for
  the engine's — while keeping §7's block-count effect in view, because halving the grid is a loss at low row
  counts whatever the warp mapping is.
* `--slim` here is deliberately binary (0 = neither lever, 1 = both). Per-lever attribution on the tile form
  already exists as `prototype/attn_valu_slim --slim 2|3`; the header note explains why a second copy of
  that table would be a mistake (the levers share one term and `dot` alone is not bit-identical — an `nd`-of-8
  child loop is not the same summation as `warp_sum` over 4).

---

**Archived 2026-10-21** from `prototype/attn_m2/HANDOFF-slim.md`, where it sat beside the rig it briefed. The device pass it hands over is done and its verdict is in that op's `results.md` and in the ledger, so `progress/README.md` rule 1 puts it here. The filename now carries the op name because a dated folder holds handoffs from several ops at once (rule 3's collision clause); citations naming the old path were re-pointed in the same commit.
