# P1 — cross-phase prefetch into the fast tier: prototype handover

**Status: code complete, CPU half settled, timings not taken (by design — see §5).**
Prototype only. Nothing here touches `engine/`; no gate, ledger or status doc was edited; nothing committed.

Route row: `progress/2026-10-21/generation-idea-queue-2026-10-21.md` P1 / P1a / P1b / P1c.

## 1. What is in this directory

| file | what it is | who runs it |
|---|---|---|
| `p1c_query.cpp`, `p1c_query` | the P1c device query: HIP residency-hint attributes, the access-policy window setter, KFD/HIP cache inventory, `hipMemPrefetchAsync` callability, and a second-stream co-residency probe with a serialized negative control | anyone, correctness-only |
| `size_targets.py` | the CPU half: derives every byte count from `engine/state.h` + `tools/step_traffic_model.py`, holds the quoted rates ([Q1]–[Q11]), prints the idle-fetch table, the per-target ceilings, the residency-headroom table and the aggregate verdict against the bar. Also the single source of slice sizes for the rig and the fitter (`--slice-mib`, `--bytes-mib`) | anyone, CPU only |
| `P1-ARITHMETIC.md` | the argument: quoted inputs, the P1c verdict, the ceiling arithmetic, what survives the intervening traffic, and whether the bar is reachable before any timing | reader |
| `p1_rig.hip.cpp`, `p1_rig` | the three-arm rig (cold / warm / polluted) with byte witnesses, guard bands, a sentinel, four shipped injections, per-case forked children, `--selftest-geom` and `--selftest` | correctness parts: anyone. `--time`: main agent only |
| `fit_prefetch.py` | the only reader allowed to turn timed rows into WIN / KILL-1 / KILL-2 / INCONCLUSIVE, with refusal paths and a `--selftest` that proves the refusals fire | anyone (it takes no timings) |
| `sweep.sh` | the timed matrix: one `tools/gpu_lock.sh` hold, two interleaved passes, admission-gated on the rig's own selftests, a behavioural canary, sizes derived from `size_targets.py` | **main agent only, blocking, zero subagents** |
| `gate_p1.sh` | the correctness gate: build, geometry selftest, fitter selftest, `size_targets.py`, `p1c_query`, rig `--selftest` | anyone |
| `build.sh`, `run_locked.sh` | build (`-O3 --offload-arch=gfx1151 -Wl,-rpath,/opt/rocm/lib`); lock wrapper that records `util` and the exit code and takes no timing | anyone |

## 2. P1c — the mechanism question, answered

Hardware residency hints **cannot** address the tier this route needs:

* `attr PersistingL2CacheMaxSize` = **2 097 152 B**, byte-identical to `attr L2CacheSize` — the carve-out
  is the 2 MiB per-GPC L2, not the tier.
* `prop accessPolicyMaxWindowSize` = **0**, and `hipStreamSetAttribute` with an access-policy window is
  rejected at **every** size tried (1, 2, 8, 16, 32, 64 MiB — `0 of 6 accepted`). ROCm has no
  `hipLimit_PersistingL2CacheSize`, no `hipCtxResetPersistingL2Cache`, no `hipStreamRemoveAttribute`.
* KFD is the only source that names the tier at all: node1 cache **level 3 = 32 768 KiB**, line 64 B
  (HIP has no level-3 spelling; `rocminfo` has no notion of MALL).

**Consequence: every P1 arm must be explicit prefetch.** No gate should be written against
`hipStreamSetAttribute`, `hipDeviceSetCacheConfig` or a `hipDeviceSetLimit` carve-out.

Second finding, because the ledger's own note asked for a re-check: `prop concurrentKernels` is **1**, but
a bounded spinner + a second-stream probe observes real overlap in **5/5 trials**, with a negative control
(both kernels on one stream, which must serialise) correctly reading 0 after burning its whole poll cap. A
second-stream prefetch is implementable. All booleans and poll counts — no duration is measured. The first
draft of that probe released the spinner immediately after launching the probe and flipped between runs;
that was a race in the instrument, not a flaky machine, and the protocol now releases only after the probe
signals completion.

## 3. What CPU settled (all of it derived or quoted; none of it measured)

Full detail and every source tag in `P1-ARITHMETIC.md`. The three results that change how P1 should be
written elsewhere:

1. **No layer's weight block fits the tier.** Derived: one GDN layer's weights 41.076 MiB, one attention
   layer's 35.014 MiB, against a 32.0 MiB tier. The weight-staging variant is retired without a device
   run, and its prize was already measured anyway (the `mlp` memory-state term, ~0.06 ms [Q10]).
2. **Only the per-layer forms survive the intervening traffic.** `staged + intervening ≤ 32 MiB` is the
   test; one KV slice (7.875 MiB) staged in the layer's own q/k/v projections (14.000 MiB) fits with ~10 MiB
   of slack, and one state slice (1.000 MiB) fits easily. Staging **all** 18 states (18.000 MiB) and then
   reading 20.062 MiB of `in_proj` puts 38.06 MiB through a 32 MiB tier — evicted by arithmetic. P1b in its
   per-layer form is the engine-shaped route; P1a whole-state is not.
3. **On the transfer model alone, P1 cannot reach the bar**: state 0.0581 ms + KV 0.1033 ms = **0.161 ms**
   against the **0.30 ms** bar, and [Q9]'s measured +0.070 ms state-coldness penalty says the state term is
   already inside the shipped numbers. The only term that can clear the bar is exposed DRAM latency in a
   latency-bound family (`attn` non-floor 1.4031 ms, `gdn` 0.4558 ms), which no transfer model can price.
   So **P1 is not dead on arithmetic, but its expected value is the transfer column and its upside is the
   latency column** — which is exactly the question the rig asks.

Bandwidth is not the constraint: idle stage time sums to 5.0602 ms of a 10.9357 ms step (≈ 332 MB of free
fetch in `attn` alone, ≈ 170 MB in `host_tail`). The one window with nothing to spare is `lm_head`, at
104 % of its own byte floor — a prefetch riding beside it displaces useful traffic byte for byte.

## 4. Rig semantics — read this before interpreting any delta

```
cold      evict(P)                                then USE(T)      [timed]
warm      evict(P)   PREFETCH(T)                  then USE(T)      [timed]
polluted  evict(P)   PREFETCH(T)   intervene(Q)   then USE(T)      [timed]
```

* **Only `USE(T)` is timed.** P, the prefetch and Q all run before the timer and are synced outside it.
  This is not cosmetic: a prefetch is an *extra* read, so if the whole sequence were timed the warm arm
  would move more bytes than the cold arm and `cold - warm` would be negative whatever the cache did.
  (The first draft did exactly that; caught by re-reading the arm definition, before any timing.)
* `cold - warm` = ceiling, what a staged target is worth if nothing follows it.
  `cold - polluted` = **the engine number**, does the staged slice survive the real gap.
* Because Q is untimed, `polluted ≈ cold` can only mean eviction, not "Q cost time".
* The timed quantity is 0.03–0.2 ms, i.e. sub-ms on a box where `hipEventElapsedTime` returns exactly 0
  below ~1 ms. So every row prints `samples=`, `reps=`, `iters=` and `spread_rel=`; the fitter refuses a row
  whose own min/max spread exceeds 30 % and a stratum whose repeats are bimodal, and compares medians.
* Witnesses are byte-level, not arithmetic: exact per-block double read checksums with the host replaying
  the same index arithmetic over the launch plan, guard bands holding a pattern no kernel stores, a 1e30
  sentinel after every target, and four shipped injections (wrong base, over-read past the sentinel, the
  `mixed_bw` per-thread-store "reduction", wrong content fill) that must all be caught. Every device case
  runs in its own forked child, HIP initialised in the child.
* Current state: `SELFTEST-GEOM ok=1 checks=10 fails=0`, `SELFTEST ok cases=16 parsed=16 children_bad=0
  caught=5/5 clean_red=0`, `FIT-SELFTEST ok=1 fails=0`, `gate_p1 PASS`.

## 5. What the main agent must run (I did not, and must not)

Blocking, foreground, zero subagents active, one lock hold (the script takes and releases it itself):

```bash
prototype/prefetch_tier/sweep.sh                 # ~2-4 min; writes prototype/prefetch_tier/out/results.txt
.venv/bin/python prototype/prefetch_tier/fit_prefetch.py prototype/prefetch_tier/out/results.txt
```

`sweep.sh`'s header holds the **pre-registered** lines, written before any timing existed:

* bar **0.30 ms/step**; envelope **0.2–0.6 ms**; state sub-target **~0.07 ms** [Q9].
* **WIN** — `cold - polluted ≥ 0.30 ms/step` at an engine-shaped stratum, adjudicable against that
  stratum's own relative noise floor.
* **KILL-1** — `|cold - warm| ≤ noise` at every stratum: the mechanism is not present on this part.
* **KILL-2** — `cold - warm` moves somewhere but `cold - polluted` never does: the bytes are fetched and
  evicted before the use site. That is the answer to the question P1 was built to dodge, and it kills the
  route.
* otherwise **INCONCLUSIVE**, reported with the stratum that could not be adjudicated and what a re-take
  needs.

The sweep also carries two diagnostic sub-matrices, because they decide the *shape* of any follow-up:
the `--setup-mib 32/64/256` dose curve (must reproduce D4's saturation at 32 MiB or explain it, otherwise
the cold arm was never cold and every delta is unreadable) and the `--intervene-mib 7/21/43/64` residency
cliff (where the prize dies is the engine's staging budget, and §3's arithmetic predicts it). A
`--regime idle` vs `spin` pair says whether the prize is residency-shaped or window-shaped; if they differ,
the route must be re-filed as an overlap route, not a residency one.

If a row prints no RESULT the sweep writes `NO-TIMING` and exits non-zero, and the fitter refuses — an
admission-red run must not print a millisecond figure (`prototype/attn_m2/burst_m16.sh` precedent).

## 6. Open items and risks

* **A `spin` regime that shares real bandwidth is a model, not the engine.** The rig's spinner is
  LDS/FMA work; the engine's phases are mixed with real DRAM traffic. If `cold - warm` is large only under
  `--regime idle`, the prize is host-tail-shaped and the ceiling must be re-derived against
  `host_tail`'s 0.7163 ms rather than `attn`'s 1.4031 ms.
* **The per-step extrapolation is an extrapolation.** The rig prices one staged set; the fitter multiplies
  by `consumptions per step / slices in the staged set` (18/18 = 1 for the whole state, 6/1 = 6 for one KV
  slice) and refuses to extrapolate a staged set that is not a whole number of derived slices or that is
  larger than the tier. It is printed as an extrapolation and should be quoted as one.
* **`hipMemPrefetchAsync` returns `no error` on a `hipMalloc` pointer.** Accepted is not performed; nothing
  here claims it moves a byte. If anyone wants it as a mechanism the burden is a witness that it changed a
  residency state, not a success code — and AGENTS records that managed memory degrades silently near
  ~32 GiB, which is why the engine's weight path is `hipMalloc` + explicit copy.
* **Sub-ms timed quantity.** Everything above runs at 0.03–0.2 ms per sample. That is the same order D4
  priced successfully, but it is why a single unpaired row here means nothing; only the paired medians
  inside one stratum, inside one lock hold, against that stratum's own floor, are readings.
* **Unit mixing in the queue's P1 row.** "per-layer KV at 4k is 8.3 MiB" is 7.875 MiB (8 257 536 B); the 8.3
  is decimal MB. Same mixing makes `attn`'s "49.55 MiB" actually 47.25 MiB and the state's "18.9 MB"
  exactly 18.000 MiB. Nothing downstream breaks, but a tier-fitting argument is where a 4 % unit error
  shows up. Correcting the queue row is the main agent's call, not mine.
* **`concurrentKernels = 1` is recorded as a device fact in `devices/strix-halo-memory-bandwidth.md`** and
  that document itself says "re-check before relying on it either way". The re-check is §2 above; whether to
  annotate the device doc is the main agent's call (I did not edit it).
