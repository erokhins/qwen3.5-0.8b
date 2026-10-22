# D1′ result plan — the batched tied head: what was built, what was proved, and the bands fixed before any timing

**Status: rig built, admitted on device (correctness only). NO TIMING WAS TAKEN by its author, and no
`ms` figure appears anywhere in this file except figures marked `quoted`.** The timing command is §6
and none of it has been run. Every number here is either imported from
`prototype/mtp_gatezero/route_price.py` or `quoted` from `engine/generation-limits.md` §1 /
`prototype/mtp_delta/D3-RESULT.md`. The four-section P7 build plan this measurement de-risks is beside
it: **`MTP-IMPL.md`** in this directory.

**The question.** `prototype/mtp_delta/D3-RESULT.md` prices plan A's `delta` at **0.1505 of a step** and
the pessimistic arm at **0.4729 — KILLED**, and names the difference:

> **`lm_head_batched` is a plan, not a measurement** — the batched 2-row head does not exist. It is the
> single largest swing term between plan A and the pessimistic arm (±2.17 ms), and it is the first
> thing the funded work must measure, not the last.

Plan A assumes one pass over the 485 MiB tied table produces logits for **both** candidate rows. This
directory is the instrument that measures that assumption instead of carrying it:
`prototype/mtp_headbatch/`, prototype-only — **nothing in `engine/` was touched.**

---

## 0. Pre-registered bands (fixed before this rig could print a millisecond; same words in `run_locked.sh`'s header and `price_head.py`)

Let **R = (best measured M=2 pass) / (best measured M=1 pass)**, both from the same rig, same process
shape, same lock window, best over `--rpb`. Plan A assumes R ≈ 1.0.

| band | verdict | derivation |
|---|---|---|
| **R ≤ 1.10** | **PLAN A STANDS AS PRICED.** The head marginal is ≤ 0.217 ms against the 0.165 ms plan A priced via the measured `proj` 1→2 marginal (+7.6 %, D3-RESULT). The 0.052 ms of slack is *inside* gate zero's own pre-registered margin on this term: “±5 % on the head alone moves `c` 0.2237→0.2336 and the speedup 1.416→1.405× — not decisive” (MTP-GATEZERO §6, D1 row; ±5 % of 2.1685 ms = 0.108 ms). |
| **1.10 < R ≤ 1.35** | **PLAN A SURVIVES WITH A NARROWER CLAIM.** The claim that survives: “P7 clears the standing +0.30 ms/step bar with one batched body and one batched head, at **≥ 1.17×** (≥ +1.9 ms/step)” — which still exceeds every other queue item together (P6 ≤ 0.09 ms, P2 ~0.1 ms, P5 ≤ 1.1 ms). The claims that do **not** survive: “1.23×” and “the second row rides the table read for free” — at R = 1.35 the head alone is 0.76 ms, a third of the entire 2-row verify. Consequence: D3's `lm_head_batched` row is re-emitted with the measured marginal instead of the `proj` transfer, and the D3 headline is restated. |
| **R > 1.35** | **THE BATCHED HEAD IS NOT FREE.** Re-price in `route_price.py` before any kernel is written. Two structural lines come with it: at **R ≥ 1.862** a batched head no longer beats simply calling the head twice by the standing 0.30 ms/step bar, so the requirement *dissolves* rather than being renegotiated; and **R_KILL = 2.2279** — the R at which P7 itself falls to 1.027×, given plan A's other measured terms — is *above* 2.0, i.e. above “the batched head is as bad as two cold passes”. So **no head measurement can kill P7 on its own; it can only shrink it by up to 3×.** That is stated here rather than discovered later, because it says what the number is for: it sizes the prize, it does not gate the route. |

**Prediction if the table really is read once: R(2) ≈ 1.00–1.03.** Plateau arithmetic, in
`price_head.py --bands`, all inputs `quoted` or imported:

```
table once            485.0000 MiB / 237 GB/s = 2.1458 ms   (measured 1-row pass 2.1685 = 1.011 of it)
+ 2nd logits row      0.9475 MiB                            = +0.0042 ms        (+0.19 %)
+ 2nd x row           4 KiB per block, L1-resident          = not a DRAM term
dots  M=1 → M=2       0.509 → 1.017 GFLOP/pass = 235 → 469 GFLOP/s
                      against the quoted scalar v_fma_f32 roof 7.30 TFLOP/s
                      (prototype/issue_rate_probe §1): 3.2 % → 6.4 % of that roof
LDS per block         M × 4 KiB; at rpb 8 both M=1 and M=2 stay thread-bound at 8 blocks/CU
                      (64 KiB/8 KiB = 8 = 2048 threads/256), so no occupancy term at M=2
```

A measured R well above ~1.03 is therefore evidence of a mechanism not in the list above — L2 thrash of
the staged `x` rows, a register spill, or a partial second read of the table — and it is to be reported
as such, not smoothed over. A measured R near 2.0 means the batch is not a batch.

## 1. What the rig is

One warp per vocab row, exactly `chunks = HID/8 = 128` `uint4` fetches per row per warp, and **M dots
per fetched `uint4`** against M staged copies of `x`. The weight byte is fetched once and used M times;
`fetch_per_weight` below is what proves that happened rather than asserting it.

| file | what it is |
|---|---|
| `hb_kernels.h` | `hb_head<M, WIT, CH>`. `#include`s `../lm_head_gemv/lm_head_gemv_kernels.h`, so the M=1 arm is a **generalisation of the shipped kernel, not a copy** (that header's own rule is that nothing in it is edited for another's convenience). Per-output summation order transcribed unchanged (`v = lane; v < HID/8; v += 32`, then `j = 0..7`, then the 5-step butterfly), staged row indexed `hs[q*HID + v*8]`, which at q=0 *is* the shipped `hs[v*8]`. `WIT=false` is the timed instantiation: no injection branch, compile-time trip count, `hs` only. `WIT=true` adds the byte witness and the injections. |
| `mtp_headbatch.hip.cpp` | host. `--mode plan` (host-only, no HIP call at all), `--mode check` (correctness, **prints no `ms` in any path**), `--mode burst` (times, and refuses to print `ms` if the witness is red). Also runs `lm_head_gemv_full` itself as the `--arm shipped` baseline, so the baseline is the shipping code in the same process shape. |
| `make_inputs.py` | builds `W.bin` and the M `x` rows and **proves the fixture pairing before writing anything**: row 0 (`dec8/final_hidden`) × table must reproduce `dec8/decode7.logits_last` (cos ≥ 0.9999 + argmax), and must **not** reproduce the fixture's bare `logits_last` (§10.5's last-writer-wins trap is asserted present, measured here at cos 0.242 vs 1.0000032). Rows are four *distinct* real pinned hidden rows — dec8, smoke, p300 row 100, p128 — because a batched head that ignored the row index would be undetectable on four equal rows. |
| `gate.py` | the correctness gate: 24 cases, one process each, **no ms**. `--selftest` is CPU-only and proves the machinery can fail. |
| `price_head.py` | the band arithmetic (imports `route_price`), `--bands`, `--evidence DIR` (asserts the parse is non-empty before it prints a verdict), `--ratio`. |
| `run_locked.sh` | the main agent's runner: host-only preflight, then **one** lock hold, admission (gate + every arm's canary in `--mode check`), two timed passes, canaries on every timed line, `NO-TIMING` if admission is red. |

## 2. The byte witness, and why it is not arithmetic

The claim under test is a **traffic** claim, so it is counted in the kernel and compared against a
host-side walk of the *intended* range — `AGENTS`' `tools/mixed_bw` / `prototype/gdn_walk` rule that a
device selftest must verify bytes touched, not arithmetic.

* per lane: `weighted` (address-weighted, odd multiplier per vocab row, so a byte delivered to the
  **wrong row** changes it), `raw` (plain byte sum), `n` (fetch count).
* per block: a **real** reduction — 64-bit warp butterfly, one publish per warp to `wsum/wraw/wcnt`,
  `__syncthreads`, then warp 0 folds the ≤ 32 warp partials and thread 0 publishes
  `cs[slot+blockIdx.x]`. A per-thread store to a per-block slot is not a block reduction, and that bug
  made `mixed_bw`'s read checksum report 1/blockDim of the bytes it had read while its GB/s looked
  entirely plausible.
* host: fixed-order integer sum over slots, compared to `hb_expect()`, which walks
  every (row, chunk) of the intended range over the host copy of the table. **It deliberately does not
  mirror the kernel's lane striding**: mirroring it would make the check pass by construction.
* slots are per block **per pass**, and there are **no atomics**: integer addition is associative and
  commutative, so the host sum is exact regardless of order, and a second pass writes its own slots
  instead of racing the first into one address.

`fetch_per_weight = total_fetches / (V × 128)`. Measured, all cases green: **1.000000 at M = 1, 2 and 4**
(one table walk served every query row), 2.000000 for `extra_pass`, 1.250000 for `extra_chunk`,
0.999968 for `short_grid`.

Guard bands: 256 poison words (`0x7FC00001`, a NaN payload — no logit can equal it) before and after the
output region, plus a padded `x` region and a 4 KiB pad past the table. `poison_left` counts logits slots
never written (must be 0), `guard_written` counts guard words that stopped being poison (must be 0). The
poison must be a value writes never store: a buffer pre-filled with the value the writes store proves
nothing.

## 3. The control set, DERIVED from the geometry

The geometry is *one warp per vocab row, M staged query rows read as `hs[q*HID + …]`*. The controls
follow from that and are generated in `gate.py` (`derived_expectations`, `counts()`), not hand-listed.
Counts printed and asserted non-empty on every run: `mc_dropk=6 mnc_dropk=14 mc_dup=4 mnc_dup=2
mc_short_holes=6 mc_extra_pass=2 crossM=6 solo=4`.

| family | must-catch | must-**not**-catch | why it is derivable |
|---|---|---|---|
| `dropk(q)`, q ∈ [0,M) | output row q moves entirely | every other row **bit-identical** | a per-query-row bug moves exactly its own row; Σ M = 6 injections, Σ M(M−1) = 14 frozen-row assertions |
| `row_dup` (every row reads staged row 0) | rows 1..M−1 move | **row 0 bit-identical** | the bug *is* “row 1 rode row 0's data”, and the untouched row is the proof the detector is not flagging every change |
| `row_shift` | all rows move | — | a cyclic relabelling must move every row when M > 1 |
| `short_grid` (grid−1 blocks) | the vocab rows `[(grid−1)·rpb, V)` left **poison** in every query row, rest bit-identical | — | the hole span is computed from the grid **the binary echoed**; a one-pass row has no partial slot to steal, which is the `partalias` lesson re-appearing here |
| `extra_pass` (2 launches, 2 accumulation windows) | `fetch_per_weight == 2.000000`, **values bit-identical** | — | the witness's own calibration: it must detect the exact failure plan A rules out, and the *values* cannot see it (measured `moved=00`) |
| `extra_chunk` (one extra fetch round) | checksum + count + values all move | — | an over-read changes the fetched multiset |

Two instantiations of the same claim are then required to agree: the **witness** kernel (whose numbers
are judged) and the **timed** kernel (the one that gets clocked) must produce bit-identical logits,
asserted every clean case (`differs_from_witness=no`). Otherwise the instrument would be the subject.

And the cross-form identities, all **bit-exact**, because the dot was transcribed rather than
paraphrased:

* `hb M=1` ≡ `lm_head_gemv_full` (the shipping kernel, run in the same process) — 0 differing elements;
* `hb M=2` row 0 ≡ row 0 run alone at M=1, and row 1 ≡ row 1 run alone — 0 differing elements;
* `hb M=4` rows 0..3 ≡ the same four rows run alone — 0 differing elements.

**This is worth more than a timing to §09.4**, which says a batched verify is not bit-lossless in bf16
and that “an engine that must be exactly lossless has to make its verify path reuse the decode kernels
bit for bit”. On this shape the batched head *does*: batching M rows does not move one bit of any row's
result, on real pinned data, at M = 2 and M = 4.

Numerics on the pin (row 0 = `dec8/final_hidden`, judged with `prototype/lm_head_gemv/run.py`'s
per-element rule — imported, not retyped, so this rig owns no thresholds of its own; the band was sized
by `lm_head_gemv/floor_probe.py`):

```
argmax 271 vs 271 | cos 1.000003 | meanrel 0.00140 | ndiff 15 (0.006 %) | n_bad 0
   (band: ulp ≤ 1 outside a 1e-4 cancellation band, ndiff ≤ 0.10 %, top-5 set preserved)
```

`gate.py --selftest` (CPU-only) proves the checker is awake: it feeds the machinery a correct `row_dup`
detection, a `row_dup` that failed to catch, a `row_dup` that also flagged the untouched row, a blind
witness, a correctly-placed coverage hole and a mis-placed one — 3 must-fail, 3 must-pass, all behaved —
and re-asserts that every derived count is non-zero.

## 4. Two instrument facts found while building it, both worth keeping

* **A `--mode` path only one arm takes is invisible to every other case.** The `--arm shipped` baseline has
  no witness slots, so it launched the witness with null pointers and produced
  `HSA_STATUS_ERROR_MEMORY_FAULT … faulting addr = 0x0, kernel: hb_head<1,true,128>`. The gate noticed
  only because it runs that arm. Same class as the AGENTS “device fault on a wave-only path presents as
  a hang” row.
* **A stale binary was run into that fault.** `build.sh` had not been re-run after an edit, and the gate
  judged a build that was not the code on disk. `gate.py` now rebuilds when any source (including the
  included `lm_head_gemv_kernels.h`) is newer than the binary — AGENTS 2026-10-17: a stale arm reads a
  clean null, and here it read a device fault.
* Cheaper, and filed: a bare `grep -oE 'ms=[0-9.]+'` on the RESULT line also matches inside
  `out_elems=` and prints the element count as if it were a timing. The runner's match is anchored on a
  preceding space.

## 5. What the author ran on device (correctness only — no timing taken, no ms reported)

All inside one `tools/gpu_lock.sh` hold each, blocking, never backgrounded, one subagent (this one)
active; `tools/gpu_lock.sh status` reads `free` afterwards. No millisecond figure below, and nothing
here may enter `engine/performance-status.md` or any `results.md` perf row.

1. `bash prototype/mtp_headbatch/run_locked.sh --admit` — **wall 12 s** (run twice, before and after the
   harness fixes in §4), `util` at acquire **36 %**.
   Host-only `--mode plan` preflight for all 9 hb arms + the shipped arm (grid = 31040/15520/7760 as a
   function of `--rpb` only, never of M), then gate.py green, then **10/10 arms admitted with their
   canaries fired** in `--mode check`, which prints no `ms` in any path. `--admit` exits before the
   timing phase by design.
2. `.venv/bin/python prototype/mtp_headbatch/gate.py` — **wall 10.4 s**, `util` at acquire **90 %**.
   24 cases, one process each: 4 solo rows, clean M=1/2/4, the shipped arm, and 16 injection cases; the
   per-row moved/frozen matrix came out exactly as derived (`10/01`, `1000/0100/0010/0001`,
   `row_dup → 0111`, `extra_pass → 0000`), `fetch_per_weight=1.000000` on every clean case,
   `poison_left=0` on every clean case. **No timing taken.**
3. `--mode plan`, `--selftest`, `make_inputs.py --verify` — CPU-only, no lock, no device.

## 6. Handover — the device pass (main agent only; nothing below has been run)

    tools/gpu_lock.sh status                          # must read free
    bash prototype/mtp_headbatch/run_locked.sh --admit   # optional: correctness only, prints no ms
    bash prototype/mtp_headbatch/run_locked.sh           # preflight + admission + TWO timed passes
    .venv/bin/python prototype/mtp_headbatch/price_head.py --evidence /tmp/mtp_headbatch/evidence

Run it **blocking**, zero subagents active, never backgrounded. Arms: `M ∈ {1,2,4} × rpb ∈ {8,16,32}`
plus the `shipped` arm (the M=1 kernel the engine runs today), 25 iterations, whole set twice.
`price_head.py` takes best-over-rpb on both sides, prints R(M=2) and R(M=4), `delta`, the speedup, and
the band verdict; it refuses to print a verdict for an M with no arm, and refuses to run on partial
evidence (the runner exits non-zero if any arm failed). After the passes it also compares
`hb M=1` against `shipped`: **if the generalisation is not within ~5 % of the shipping kernel, the ratio
is suspect** — that would mean the rig is measuring the rig, and the read must be reported against the
shipped baseline instead.

## 7. Named transfers and risks (what this rig does *not* settle)

1. **`x` is F32 here** (`lm_head_full`'s form), while the shipping head is `lm_head_full_xbf16`
   (`decode_step.cpp:1187`, route S1 item 1's last fold: the staged row is BF16). The per-block staged
   cost is a constant either way (4 KiB vs 2 KiB) and the *ratio* — the thing being priced — is
   insensitive to it; and `engine/generation-limits.md` §1's 2.1685 ms row predates the fold, so the
   rig's baseline matches the number it is compared against. If the measured R needs defending, the
   follow-up arm is a BF16-`x` instantiation beside this one.
2. **`gemm_algo_sweep` cannot bound this shape** (no `[248320,1024]` row), which is *why* the rig exists;
   conversely the rig is a hand-written GEMV, not hipBLASLt, so a future engine implementation may get a
   different absolute cost. The claim being priced is a *ratio*, and the `proj` family's measured
   1→2 marginal (+7.6 %) is the closest same-class analogue.
3. **M=4 at rpb 8 is LDS-bound** to 4 blocks/CU (16 KiB/block of staged `x`), so rpb 16/32 arms exist;
   R(M=4) must be read best-over-rpb, or it prices a geometry no engine would ship.
4. **Register pressure is unmeasured.** With M accumulators LLVM may spill, and clang's
   `-Rpass-analysis=kernel-resource-usage` "Occupancy [waves/SIMD]" is a constant on this part (AGENTS
   2026-10-18), so it cannot detect that. The tell is in the measurement: R(M=4) ≫ R(M=2).
5. The rig reads the table with the same warm-up discipline as `lm_head_gemv` (first touch outside the
   burst, best-of-5 bursts), so it is comparable to the ledger's `lm_head` row and to D1.
6. `hipcc` warns `loop not unrolled` for the three **witness** instantiations (runtime chunk count). The
   timed instantiations fold the trip count to `HID/8` exactly as shipped — that is what the template
   parameter `CH` is for.

## 8. What would change this answer

An R in the 1.1–1.35 band means plan A's *number* moves and its *shape* stands (§0). An R > 1.35 means
the second row is not free and `route_price.py` must be re-run before a kernel is written — and note
that the honest kill line for *this* term (R_KILL = 2.2279) is unreachable by any batching failure, so
the correct response to a bad R is “re-price the prize”, not “close P7”. If `hb M=1` disagrees with the
shipped kernel by more than a few per cent, this rig's numbers do not transfer at all and the
measurement has to move into the engine (which is also where `delta`'s remaining terms live).
