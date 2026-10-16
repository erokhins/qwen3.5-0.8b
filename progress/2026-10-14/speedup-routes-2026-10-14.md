# Every speed-up route tried on this engine, and what is left — 2026-10-14

**What this file is.** One index of every route this project has tried to make the engine faster, each with
the mechanism it claimed, the line it had to clear, the number that decided it, and where the full write-up
lives. It exists because the primary record — `bench/performance-status.md`, 1,149 lines frozen as
`progress/2026-10-16/performance-status-2026-10-16.md`, appended to ever since — records each measurement in the order it happened, so "what have we already tried, and
why did it not work?" currently requires reading all of it. Nothing here is a new measurement: every number
is copied from the ledger, a `results.md`, or a `bench/p9-*.md`, and every row points at its source.

**How to read a verdict.**

| verdict | means |
|---|---|
| **SHIPPED** | default on the shipping path, gated, in the headline tables |
| **DEAD** | measured against a pre-registered line and missed it, or its mechanism was refuted by its own control. Not to be re-opened without the named new mechanism |
| **HELD** | priced, judged too small or badly blocked for its cost, decision recorded. Re-pricing needs a new reason, not enthusiasm |
| **UNPRICED** | plausible, never measured, sized here so the next pass can decide whether to fund it |

---

## 1. The ladder: what the headline is, and which route earned each step

Prefill is `RESULT ms_total` for a prompt + `--steps 1` (the ruler in `tools/prefill_sweep.py`); decode is
marginal `ms/token` from `tools/context_sweep.py`. Rows are copied from the ledger sections named in
parentheses; where the paired A/B and the sweep disagree at short prompts, the paired A/B is quoted and
labelled as such.

| route (verdict) | prefill @1024 | prefill @3072 | decode @128 | source |
|---|---|---|---|---|
| starting point, before the prefill era | 636 tok/s | 568 tok/s | 77.2 tok/s | ledger §P8 |
| **P8b** — one attention site per layer per body, attention leaves the per-token loop (**SHIPPED**) | 911 | 746 | unchanged (by construction) | §P8's payoff |
| **Q6** — `kRowsPerBlock` 8 → 32 in the projection launcher (**SHIPPED**) | 971 | 800 | −1.3…−2.7 % | §Q6 |
| **Q10b** — MLP weight-walk split, `--mlp-map auto` (contig one-row / interleave wave) (**SHIPPED**) | 971 (unchanged) | 800 (unchanged) | **85.9** (+8…11 %) | §Q10b/Q10c |
| **route (c)** — hipBLAS `GemmEx` bf16/bf16-C on wave bodies, hand GEMV for one-row bodies, `--gemm library` (**SHIPPED**, D-11 band) | **2,706** | 1,713 | 87.1 | §route (c), §flip tables |
| **tiled attention wave** — `--attn-tile 1`, block-level K+V staging, one row per block (**SHIPPED**) | **3,330** | **3,299** | 89.6 (unchanged by construction) | §tile flip |

The last row is quoted from the **paired** A/B (3,330.5 and 3,299.2 tok/s); the `prefill_sweep` ruler taken in
a separate window reads 3,326 and 3,321 for the same two widths, which is the same number within its own
spread. Both are given once, with their ruler named, because §5 rule 4 says a sweep is not a pair.

**Cumulative: 636 → 3,330 prompt tok/s at 1024 lanes (5.2×) and 568 → 3,299 at 3072 (5.8×); decode 77.2 →
89.6 tok/s (+16 %).** Note the shape of that column: prefill is where every large route has landed, and
decode has moved only through kernel-mapping changes, because decode is weight-walk-bound and prefill is
byte-pattern-bound. At 3072 the rate now equals the rate at 1024 (3,321 vs 3,330) — long prompts stopped
being disproportionately expensive the day the streaming route stopped re-reading 4×.

## 2. The funding instrument: what a 1024-lane prefill wave spends, before and after the last change

| | before the tile flip | now | share now |
|---|---|---|---|
| wave wall | 329.55 ms | **241.00 ms** | |
| `gdn_rec` | 125.45 ms (38 %) | 121–126 ms | **~50 %** |
| `attn` | 107.67 ms (33 %) | **26.70 ms** | **11 %** |
| matmul (`proj` + `mlp_gateup` + `mlp_down`) | 68.66 ms (21 %) | ~70–72 ms | **29 %** |
| everything else (glue, casts, copies, gdn conv/gates/norm, lm_head) | ~24 ms | ~22 ms | ~9 % |

Any new route has to be argued against this table, not against last week's. Two consequences stated flatly:
**half the wave is one kernel family whose two known attacks are both measured dead** (§4.1, §4.2), and
**attention is now worth 11 %**, so a further 20 % win in attention is 2 % of prefill.

Re-take it any time (cheap, and it is the only thing that justifies funding):

```bash
tools/gpu_lock.sh acquire ATTR && ./engine/build/decode_run \
  --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --time-waves --gemm library --gdn-scan seq \
  | grep -E "WAVEPASS|WAVE gran=stage family|GROUP"; tools/gpu_lock.sh release ATTR
```

---

## 3. SHIPPED routes, with what each one cost to prove

### 3.1 P8b — attention leaves the token loop
One attention site per attention layer per body instead of one per query: `attn_prep` + `attn_core` for a
wave, the fused single launch for a one-row body. TTFT −7.5 % at 64 lanes, −30.2 % at 1024, −23.9 % at 3072
(spreads ≤ 2 %), decode unmoved and *required* to be unmoved. Proof: eight fixtures re-gated with the gate
seven checks stricter than before, every count the old count + 7. (`f4a1095..88fd712`, ledger §P8.)

### 3.2 Q6 — projection `kRowsPerBlock` 8 → 32
Bit-identical by construction (a row is owned end-to-end by one wavefront), byte-identical across all 40
`dec8` output files, `margin_delta` ×1.0000 including the two tightest lanes on the tree. Worth 1.3–2.7 % of
a decode token and 5.6–8.0 % of TTFT — the asymmetry is the lesson: **a wider block only helps when there
are many rows to fill it.** The worker's derived hope of −0.50 ms/token did not materialise; the honest
figure is a fraction of that. (`bc61dce`, ledger §Q6.)

### 3.3 Q10/Q10b — the MLP weight walk depends on which path you are on
The decisive negative result (§4.5) said the in-step deficit is the weight-walk *mapping*, not the memory
system. The fix is a template parameter, and it ships as `auto`: a one-row body walks `contig`, a wave walks
`interleave`, because the two paths measurably prefer opposite walks (under `contig` both weight
`global_load_b128`s collapse onto one destination quad, so one weight word is in flight instead of two —
an ISA fact; *why* it helps one path and hurts the other is explicitly not established). Bit-identical by
construction, proven on device (`prototype/mlp_gemv/run.py --mapping-identity`), so the A/B is a timing
question and not a numerical one. Decode **+8…11 %**, 4k −5.8 %, prefill untouched — which was the point of
splitting rather than picking. All three pre-registered confirmations hit within 0.9 %. (`d1703b0`,
ledger §Q10b/Q10c.)

### 3.4 Route (c) — hipBLAS on wave bodies, hand GEMV on one-row bodies
`GemmEx` bf16/bf16-C for wave bodies; the `rows > 1` seam means a decode step never touches the library, so
decode is bit-identical by the seam rather than by luck. **2.85× on prefill.** The cost was not the kernel,
it was the numerics admission: a reassociating route cannot be byte-identical, so D-11 replaced cross-wave-
count byte-identity with the **cascade band**, policed by `tools/band_history.py` (19 keys, `--selftest`
green) and pinned provenance (`engine/numerics_provenance.json`: wave-row ceiling + hipBLAS version). The
audit (`progress/2026-10-14/route-c-audit-2026-10-14.md`) shows every hand↔library difference is reassociation: ≤ 1 ULP
above the cancellation band, `meanrel ≤ 3.9e-5`, 0 of 1,566 greedy decisions with ≥ 0.10 room flipped, and
each injected bug clearing the legal delta by ≥ 492×. It also had a false claim to retract first — the
famous "16/16 both routes" was sixteen hand runs, because a wrapper set `--gemm` and `gate.py` appended it
again (last-wins parser). (ledger §route (c), §correction, §flip tables.)

### 3.5 The tiled attention wave — `--attn-tile 1`
One K **and V** row load per lane per block instead of one per (row, head) pair, one query row per block, and
one partial layout shared with the shipped form. `attn` 107.67 → **26.70 ms** at 1024 lanes (**4.03×**,
against a pre-registered `≤ 40 ms` line and a 27.2 ms traffic prediction), wave wall −26.9 %, prefill
+25.3 % at 1024 and **+94.6 % at 3072**. Proof that it is not just fast but right: `attn_form_band` reports
**136/136 and 520/520 above-`MAXSC` rows bit-identical** between the two forms, 0 differing elements, argmax
moved 0 — bit-identity, not a band — plus `gate.py` PASS on both anchors at `--scope form` *and* `full`,
PASS on `dec8`/`dec64f`/`dec300f`, the op's own wave gate and device negatives, and a CPU mirror whose
negative suite is 7 controls with derived must-catch sets. It also first **died of a mis-diagnosis** (§4.6).
(`6db6ee4`, `6bbee45`, `prototype/attn_decode/results.md` §20–§21.)

---

## 4. DEAD routes — each with the mechanism it claimed and the control that killed it

### 4.1 D-12 — the chunked delta-rule GDN scan, as built
Claim: replace §04.4's sequential recurrence with a chunked scan, 1.96–2.99× faster at C=64/32/16 by
arithmetic at the engine's own worst measured matmul rate, and *closer* to the pinned dumps than the
sequential form (so no band wider than D-11). Both halves of that survive on paper. The device returned
**331.2 → 1,953.7 ms wave** (`gdn_rec` 127.0 → ~1,750 ms) — **5.9× slower than priced**. The named cause
(`progress/2026-10-14/p9-phase1c-2026-10-14.md`): matrix-shaped work executed on a vector path — the carry executed
**730.4 GFLOP against 54.03 priced**, and `v_mfma` appears **nowhere** in the binary. Revival condition: not
another hand-FMA chunk kernel; a chunked form whose matmuls are backed by the matrix unit or by hipBLAS,
with the same arithmetic re-done at the rate that route actually delivers. Numerics are not the obstacle.

### 4.2 Route A — the resident (launch-collapsed) GDN recurrence
Claim: hold the state resident in LDS for a whole body, issue **one launch per layer per body** instead of
one per layer per token, move the state 38.7 GB → 75 MB per wave. Lines fixed *before* the kernel existed:
≤ 63 ms lives, > 84 ms dies. Measured **112.04 ms** against `seq`'s 125.36. It is bit-identical to `seq` on
all ten fixtures end-to-end (after two real device bugs were fixed: a 32-column strip indexed with `tid` over
128 columns, which ran off both the state tensor and its own LDS array; and a hoisted `Q_SCALE` that moved
93,522 of 131,072 `o` elements while leaving the state bit-exact). The durable finding: `seq` spends 6.8 µs
per (token, layer) on the recurrence, `collapse` spends **6.1 µs** while issuing 36 launches instead of
18,432 — so **`gdn_rec` is latency-bound on 2,048 dependent chains, not launch-bound or traffic-bound**, and
no launch-shape change will move it. `--gdn-scan collapse` stays selectable (bit-identity makes it a free A/B
side), and it is not the shipped default. (`prototype/gdn_collapse/results.md`, ledger §route A.)

### 4.3 Dispatch / launch overhead as the in-step gap
The bill was built from counters: **424 launches per token ≈ 1.6 ms of a 12.949 ms token**, and `glue`'s 98
launches measured 0.190 ms against 0.184 derived from the launch-cost curve (3 %). A 42.4 µs/launch ramp
would bill a 12.74 ms step 18.0 ms; it does not. Launch overhead is real, small, and already largely claimed.
(ledger §Q6, §Q6.1.)

### 4.4 Cold windows and "warmth" (Q7 M1)
Claim: the step is slow because it re-walks a huge cold window. The pre-registered test grew the burst's
window 4× at fixed kernel geometry — 132 → 264 → 528 MB, the last being exactly one real token's MLP traffic
— and the MLP got **faster**: 2.630 → 2.457 → **2.408 ms** (201 → 219 GB/s, 93 % of probe). The step's 52 %
of probe is not a coldness effect. (ledger §Q7.)

### 4.5 Mixed read/write traffic and write-back pressure (Q8)
Claim: read-only probes flatter the box; a real step reads 1.5 GB of weights while dirtying activations, GDN
state, KV and ~115 cast outputs. Priced with a 26-row sweep (`devices/strix-halo-mixed-traffic.md`): at the
engine's own 62:1 read:write the box runs at **99.1 % of its read probe**; reading freshly dirtied lines is
**free** (−1.8 %); a single row replaying the whole step's traffic budget achieves **218.9 GB/s**, i.e.
7.06 ms of weight traffic against the engine's 12.4 ms. The memory system offers a floor the engine fails to
reach — which is what led to §4.7 and the shipped `--mlp-map` split. (ledger §Q8.)

### 4.6 "The tiled attention form is structurally wrong" (§19, retracted in §20)
§19 concluded tiled attention could not reproduce the reference above `MAXSC` (row 512, decision room 13.25)
and called it structural. It was not structural: the tile form keyed its split partial by head-**in-group**
while `attn_core` keys by the full head, so above `MAXSC` two lane blocks of one row wrote the same floats
and the merge handed both heads of a KV group one answer. Fixed (one layout, one merge), it is bit-identical
and 4.03×. Filed here as a dead route because a *verdict* was nearly final: **a "structural" conclusion needs
a mechanism attached before it retires a route**, and an address bug in a partial buffer will present as a
numerics failure. (AGENTS.md traps, `results.md` §20.)

### 4.7 Weight-walk *cache locality* (Q9, refuted as locality)
Replaying the engine's per-thread weight arithmetic as pure reads at byte-identical geometry reproduced the
penalty (0.470 / 0.472 / 0.629 of the contiguous sweep), while **reordering an identical 424-launch multiset
changed nothing (1.005×)**. So it is not locality or ordering; it is the mapping (two-stream interleave and
the row tail). That is the finding §3.3 ships. LDS staging was exonerated (`nostage/eng = 1.098`). (ledger
§Q9.)

### 4.8 The skinny argmax / `lm_head` fusion (§07.4's claim)
The specification's "skinny argmax is bandwidth-critical" was wrong for the GPU: `lm_head` measures 2.127 ms
against a 2.146 ms roofline — at ceiling, nothing to take — and the fused variant gained ~1 %. Corrected in
place in §07.4. (ledger §Q6, `prototype/lm_head_gemv/results.md`.)

### 4.9 The fp16 accumulator
`wmma_gemm<fp16, fp16 acc>` runs (247 vs 219 for f32 acc) but is 3e-4…2.9e-3 relative against its own
fp16-accumulated reference and catastrophic on cancelled elements (maxrel 4.7e+01 at K=1024). D-3 pins F32
accumulation; not a candidate. (`progress/2026-10-13/prefill-route-spike-2026-10-13.md`.)

### 4.10 Two instrument-level routes, closed for lack of a handle
**Stream/graph capture (M4)** needs a stream handle at the launch sites; the engine uses the default stream,
so capture is impossible today. **The 4032-lane differential ruler** is unusable (spread 29.7 %, with a "best"
faster than a 128-lane step at 32× the KV — not physical); quote 4k decode from the stage-clock burst only.
(ledger §Q7 M4, §Q10b.)

---

## 5. HELD — priced and deferred, with the price and the blocker

| route | prize, as priced | why it is held |
|---|---|---|
| **M3 — packed projections** (proj 114 → 48 launches) | 0.1–0.25 ms/token, ~2 % of a 12.4 ms step | costs a `gate.py::launch_plan` and §07 table change; buying 2 % with a gate edit was judged a bad trade while the ceiling question was open. The ceiling question is now answered (§4.5), so it is a fair re-price if anything else touches projections |
| **`chunk64b` step 0 sized rule** | not a speed route, but the fixture's 3 D-9 red lines are the only unsized numerics row on the tree | owed a rule, not a kernel |
| **mixed one-pass/streaming wave grid** | priced only by engine self-consistency (no independent reference) | documented limitation, no action taken |

## 6. UNPRICED — future thoughts, ranked by (size × confidence), each with what must be proved first

Ranked honestly: the first item is the only one with a >10 % prefill prize whose mechanism has not already
been refuted. The rest are smaller or blocked on a user decision.

### 6.1 A matrix-unit-backed chunked GDN scan (the only >10 % prize left)
`gdn_rec` is ~50 % of the wave; §4.1 shows the chunked *algorithm* was never refuted, only its vector-path
implementation (730.4 GFLOP executed vs 54.03 priced, zero `v_mfma`), and its numerics were shown to fit
D-11 with margin (state `meanrel` 1.6e-7 vs sequential 5.2e-7; a BF16-operand/F32-accumulator model costs
`meanrel ≤ 9.9e-3`, `1-cos ≤ 3.2e-4`, i.e. 3× and 6× inside the admitted band). **Cheapest first step, and
it is not a kernel:** a counter pass on the shipped `seq` recurrence with `rocprofv3`
(`SQ_INSTS SQ_INSTS_VALU SQ_INSTS_LDS SQ_WAVES` per launch, same census that priced attention) to say
whether it is LDS-latency-bound or genuinely chain-bound. §4.2 already proves the binding term is
dependency, not launches or bytes, so the funding question is: **which reformulation shortens the §04.4
chain?** A chunked form backed by hipBLAS (the engine already has the library route) is the obvious
candidate, and its rate must be re-derived from what hipBLAS delivers at *those* shapes, not from the hand
GEMM rate the Phase 1 spike used. Proposed line, to fix before writing code: the chunked form must take
`gdn_rec` below **90 ms** at 1024 lanes (a 25 % cut ≈ 10 % of prefill) with token flips at 0 among rows with
≥ 0.10 room and no band change; **> 110 ms kills it.** Also record that `--gdn-chunk` currently defaults to
64 for the *existing* scan and means something different from D-12's chunk size — do not overload the flag.

### 6.2 fp16 activations on the matrix path (+21.5 % on the GEMM legs)
`devices/strix-halo-compute.md` measures fp16 **6.885** vs bf16 **5.667** TFLOP/s at the engine's own
`1024×4096×1024`, and `v_wmma_f32_16x16x16_f16` exists. Corrected 2026-10-16 by `tools/isa_probe.sh`:
on `gfx1151` that is the ONLY matrix instruction — bf16 has no WMMA and no `v_pk_fma_bf16` — so the
matrix path is fp16 or nothing, and its roof is 4× the vector rate, not 1.2× (specs §7). Weight
conversion alone is safe and cheap (whole bf16→fp16 weight delta **3.0e-08**), and fp16 beats our
bf16 activations on mantissa (10 bits vs 7) wherever the value fits; every pinned value fits, since
`tools/fp16_range_probe.py` finds activations peaking at 82.5 against a 65504 ceiling, so a switch
cannot lose precision inside range. Matmul is 29 % of the wave: at least −10 % of prefill, more via
WMMA. Admitted under the user's 2026-10-16 rule — decision log D13/D14 — with the gate set as the check.

### 6.3 Weight quantization for decode (largest decode lever, explicitly out of scope today)
Decode is weight-walk-bound: 1.5457 GB of bf16 text weights, and at this box's 237 GB/s read ceiling that is
**~6.5 ms/token = 153 tok/s floor** against today's **11.155 ms / 89.6 tok/s** — i.e. bf16 quantisation to 8
bits would roughly halve the byte bill and is the only route that can plausibly double decode. *Arithmetic on
measured rows, not a measurement.* The repo's SCOPE section says quantization-for-speed is out of scope
without an explicit request, and it is a bigger numerics decision than §6.2 (it quantises the operands the
residual stream actually accumulates, not just a dtype). Listed so it is on the map, not funded.

### 6.4 MTP drafting on the GPU (decode-side, never priced on device)
`specification/09-mtp-draft-layer.md` §09.5: drafting is bit-lossless but **slower than plain greedy on torch
CPU** (~21 vs ~24 tok/s, accept 0.66 at k=1) — it is explicitly a GPU-path bet, and the GPU path has never
been priced. The MTP draft block is 39 MiB and already understood; `mt32f` exists as a fixture. Cheap probe
if anything funds it: k=1 draft/verify on the wave machinery, one accept-rate + one tok/s number against the
89.6 tok/s ruler, kill line stated first. Expect the accept rate to be the whole story: 0.66 accept at k=1
means 1.66 tokens per verify step at best.

### 6.5 A re-price of the wave-row cap now that the streaming route is 4× cheaper
`kMaxWaveRows = 512` was chosen when the wave was 331 ms with attention dominating and the streaming route
reading 4×. The cheapest unpriced knob on the tree: `--wave-rows` is a flag, the gate accepts it, and the
question ("is 512 still the right chunk width, and does the `split_for`/lane-cut interaction prefer a wider
wave now?") is one locked window at 256/512/1024. Small expected prize (a few %), near-zero cost, and it
would retire a constant nobody has re-examined since the wave path was written.

### 6.6 Attention: the lever is spent, and here is the stop rule
26.70 ms is 241.6 GB/s of *demanded* K/V bytes, above the probe's 237 GB/s, which means the census
over-counts what reaches DRAM (caches serve repeated lane rows) — so the byte wall is an upper bound that
has been met, not a ceiling proved. Two named remainders, both small: **M>1 row tiling** is refused by
`static_assert(ATTN_TILE_M == 1)` on an *argument* (two rows in a block share no lane range, so the union of
their stage windows exceeds `MAXSC`) rather than a measurement; and block-order/cache-aware scheduling is
unexplored. At 11 % of the wave, a 20 % attention win is 2 % of prefill: **do not fund attention work
below a 15 % wave win.**

### 6.7 Instruments to build before the next big route (cheap, and they gate everything above)
* **DRAM byte counters.** Every traffic claim in this repo is census arithmetic; `rocprofv3` has never
  produced DRAM bytes on this box. §6.1 and §6.6 would both be settled outright by a real byte counter.
* **A stream handle in the engine** (for capture, and for any overlap experiment) — see §4.10.
* **A usable long-context ruler.** 4032-lane decode can only be quoted from the stage-clock burst; the
  differential harness is void there.
* **Sub-millisecond timing.** `hipEventElapsedTime` returns exactly 0 below ~1 ms on this iGPU; bursts are
  the only timer to trust, which is why sub-1 ms routes (glue, casts) are hard to price.

---

## 7. Rules that came out of the above (they are also in `AGENTS.md`; collected here because they are what
## made these numbers trustworthy)

1. **Fix the line before the kernel exists.** D-12 (63/84 ms), route A (same), the tile win line (40 ms), Q7
   M1 (≥ 3.6 vs < 2.8 ms). Every dead route above is dead against a pre-registered number; nothing was
   retired on vibes, and nothing shipped beat its line by accident and got kept without a mechanism.
2. **Never loosen a gate or a band to route around a failure.** No band was widened for route (c) beyond the
   cascade band D-11 admitted by audit, and the tile form shipped at bit-identity instead of asking for a band.
3. **A negative control must have a non-empty, geometry-derived must-catch set**, and a check whose reference
   side is derived from the variable under test cannot adjudicate it. Three vacuous checks were found this
   way (`partalias`, a dry-run stand-in, a hand-listed must-not-catch set that rotted into "everything").
4. **A paired A/B in one lock window, or it did not happen.** Sweeps taken in different windows disagreed
   with paired runs at dispatch-bound widths (64/256 lanes), and a burst row compared against a differential
   row once looked like an 8.9 % regression that did not exist.
5. **Prefer the arithmetic that can be wrong.** Every "at ceiling" claim here came from comparing a measured
   number against a roofline derived independently of it (`lm_head` 2.127 vs 2.146; attn 26.70 vs 27.2; the
   mixed-traffic row at 218.9 GB/s against 12.4 ms of engine time).

## 8. Available A/B switches right now (each is one flag, and each is gate-visible)

| switch | default | what it is for |
|---|---|---|
| `--gemm library\|hand` | `library` | route (c) vs the hand GEMVs; the `rows > 1` seam means it cannot change a decode bit |
| `--attn-tile 1\|0` | `1` | the tiled wave vs the per-(row,head) witness it was proved against |
| `--mlp-map auto\|interleave\|contig` | `auto` | the MLP weight walk; bit-identical either way, so a pure timing A/B |
| `--gdn-scan seq\|collapse\|chunk` | `seq` | `collapse` is dead but bit-identical (free A/B side); `chunk` **fails closed** by design (D-12) |
| `--gdn-chunk N` | 64 | chunk width of the *existing* scan — not D-12's chunk size, do not overload it |
| `--wave-rows N` | `kMaxWaveRows` (512) | wave scratch width; `1` forces per-token launches (the P4b parity run). See §6.5 |
| `--time-waves` / `--time-stages` | off | the attribution rulers; `--time-stages` is the stage-clock, `--time-waves` the family table |

## 9. Housekeeping for the clean-up

* **`tmp/` was cleaned on 2026-10-14: 126 GB → 83 MB** (disk free went 2.4 TB → 2.5 TB). What went was the
  regenerable payload and nothing else: **132,683 `.bin` per-row logits dumps (121.2 GB) and 526 `.npy`
  stage dumps (3.9 GB)**, then the emptied directories. What stays, and it is all small: the probe and
  doc-builder scripts (`tmp/*.py`, `tmp/*.sh`, `tmp/tilefix/`), the gate and sweep logs, the id fixtures
  (`tmp/q10/ids{64,128,256,300,640,1024,3072}.txt` — several commands in this repo's docs read them), the
  `tmp/pre-route/` source snapshot the route-(c) claims were taken against, and the built A/B witnesses
  (`q6/decode_run_rpb8`, `q10b/decode_run_old`, `q10/decode_run_contig`, …) — about 3 MB of binaries that are
  the only *built* evidence for historical ledger rows, so they are cheaper to keep than to re-explain.
  After the clean, `decode_run` on the kept 1024-id fixture reproduced 305.4 ms (3,353 prompt tok/s, i.e.
  today's number) and `engine/gate.py --fixture dec8` PASSed, so nothing load-bearing went with the dumps.
* **The habit that keeps it small: a `--work` directory is disposable.** Every `engine/gate.py --work <dir>`
  run writes one `.bin` per pinned prefill row, which is ~1 MB × rows × fixtures — `wide1024f` alone is
  1,024 rows × 6 layers × ~1 MB. Delete the work dir when the verdict is written into a doc; the verdict, not
  the dump, is the artifact. (Even the sanity check in this session's clean-up left an 80 MB directory behind
  until it was removed by hand.)
* **Do not delete:** `data/refs/` (3.3 GB of pinned HF dumps, gitignored, the ground truth for every gate —
  and `wide640f`/`wide1024f` are the only >`MAXSC` pins on the box), `tools/band_history.py`'s
  `bench/band-history.jsonl`, or any `results.md`.
* `tmp/tilefix/*.py` are doc-builders that have already been applied (`append_s20.py`, `append_traps.py`,
  and the id-slicers `slice_ids.py` / `repeat_ids.py`, which are worth keeping if long-context A/Bs continue —
  `repeat_ids.py` states on its own output when it has repeated a prompt). Re-running the appenders asserts
  rather than duplicating, which is intended.
* `engine/.pi/` is untracked and pre-existing; keep it out of commits. Nothing in this repo is pushed.
