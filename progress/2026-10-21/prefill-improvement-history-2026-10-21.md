# What actually made prefill faster — the measured history, 2026-10-12 → 2026-10-21

**What this document is.** A single self-contained account of every change that produced a **significant
prefill improvement** on this engine (`Qwen/Qwen3.5-0.8B`, text-only, Strix Halo `gfx1151`), with the
mechanism that made each one fast, the numbers that measured it, the admission that made it shippable, and
the pointer to the raw evidence. It is a synthesis, not a new measurement: every figure below is copied from
`bench/performance-status.md` (the live ledger), its frozen predecessor
`progress/2026-10-16/performance-status-2026-10-16.md`, the decision log
`progress/2026-10-16/prefill-decision-log-2026-10-16.md`, a `prototype/<op>/results.md`, or a commit message,
and each entry names which.

**What it deliberately excludes.** Routes that were measured and refused (route A, the chunked scan, G2,
A3, A4-L1, L-b, L-c, A12, the fp16-accumulator bet), analyses that moved no code, and decode-only wins. A
short version of the refusals is in §16, because a reader told only what worked will re-fund what did not.
Dead-route detail lives in `progress/2026-10-14/speedup-routes-2026-10-14.md` (the route index) and in the
ledger sections cited per entry.

**Date convention.** Sections are dated as the **ledger** dates them. A handful of commits carry a `git` date
one day later, because the deciding device window ran late at night (§8 → `a1cde3b`, §7 → `aaefe01`, §§14-15
→ `99da4c9`/`ec72d94`). Both are given.

---

## 0. How to read these numbers (read this before comparing any two rows)

Six rules, all of them this repo's own, all of them load-bearing for the tables below.

1. **The ruler.** Prefill is `decode_run --steps 0|1 --time-waves` for a prompt: `RESULT ms_total` (cold) or
   `WAVEPASS gran=burst wall_ms` (warm), reported as prompt tokens ÷ seconds. `tools/prefill_sweep.py` is the
   sweep form; `tools/context_sweep.py` is the decode differential. Ids come from committed fixtures
   (`ids1024.txt`, `ids4096.txt`, …), comma-separated, and `--ids-file` for anything above ~8k tokens.
2. **Cold and warm are different quantities and both are in the record.** `RESULT ms_total` is the **first**
   pass of `prefill_wave`; `--time-waves wall_ms` is a `st.reset()` + re-run of the *same* wave, so the
   difference (~90–110 ms) is **first-execution cost, mostly hipBLASLt's first execution per algo** — not
   per-prompt host work (§50). From §51 onward every headline in the ledger is **warm**. This is why the same
   engine carries 7,986 tok/s (§48, cold) and 9,790 tok/s (§51, warm) at 4,096 lanes: **one engine, two
   brackets.** Never compare one engine's cold number against another engine's warm number.
3. **A paired A/B inside one `tools/gpu_lock.sh` hold, or it did not happen.** This box serves its own tokens
   on the same iGPU, so an absolute ms moves a few percent between windows and a short-prompt row is
   bimodal (dispatch-bound). Every ratio quoted below was taken with both forms **in one binary, alternating
   per rep, inside one lock hold**. Cross-window pairs are labelled where they appear. Chained paired ratios
   *are* portable — §12 measured it: two chained ratios predicted 4,425 tok/s and the dedicated ruler came
   back 4,419, **0.14 % apart**.
4. **The family row is the attribution; the wall is the ship gate.** `--time-waves` prints per-family ms. A
   change that makes one family cheaper can push cost into its neighbours (the GDN norm-hoist bet's −5.91 %
   on `gdn_rec` was −2.21 % of `wall_ms`, because +6.06 ms landed in `gdn_gated_norm` and `glue`). From §40
   the ledger books both denominators and ships on the wall.
5. **Admission is usually bit-identity, not a band.** Most of the wins below move no value at all — Q6, the
   tile flip, `--attn-touch`, `unroll4`/`fused`, the F32 seam, the gather, the conv read, `--gdn-loop`,
   `--attn-fattn-qstage` and A13 up to 4,096 lanes are **bit- or byte-identical by construction** and needed no
   numerics decision whatsoever. Only four change values, and each was admitted under bands already in force:
   route (c) (library reassociation — the one route that *bought* its band with an audit), the WMMA form
   (streaming rows judged inside the existing caps, **no new band**), G4 `reg16` (re-associated state dot) and
   A13 above 4,096 lanes (re-associated wide round). **No band was widened to make a win pass**, and the one
   band-shaped key that moved (the flip-room line, 0.10 absolute → 1.50 BF16 ULP) moved because the old one was
   magnitude-blind, not because a route was too slow to pass.
6. **Engine default and gate default move in one commit, every time.** `--attn-tile`, `--gemm`, `--gemm-pins`,
   `--gdn-spell`, `--gdn-loop`, `--gdn-geom`, `--attn-fattn`, `--attn-fattn-qstage`, `--attn-fattn-wide`: each
   flip is one commit touching `decode_run.cpp`, `decode_step.h` and `engine/gate.py`. A half-flipped pair
   makes the gate silently judge a kernel the engine does not ship.

**Model shape, since several mechanisms depend on it.** 24 layers = 18 GDN (gated delta-rule, linear
attention) + 6 full-attention; `n_q = 8`, `n_kv = 2`, GQA group `g = 4`, `d_head = 256`
(`specification/05-gated-full-attention.md`). A wave body is at most `kMaxWaveRows = 512` rows, so a
4,096-lane prompt is **eight sequential 512-row waves**; rows at or below `MAXSC = 512` lanes take the
one-pass attention route while rows above it stream. That split is why several attention wins below cover
"above-`MAXSC` rows only" (the WMMA form covers 87.5 % of rows and 98.4 % of the lane-work, §41) — partial
coverage by design, not a gap.

---

## 1. The ladder

### 1.1 Ranked by size of the win

| # | change | headline move | what it moved |
|---|---|---|---|
| 1 | **G4 `reg16` — the GDN walk keeps its state in registers** | `gdn_rec` −66.6 %, wave −27.9 %, **5,764 → 7,655** cold (+32.8 %) | the per-token state round-trip |
| 2 | **A13 — the llama-ported wide attention round** | warm **−14.9 % at 4,096, −27.4 % at 16k, −35.7 % at 32k**; published warm headline **9,790 → 11,070** at 4,096, **6,125 → 8,542** at 16k (+39.5 %) | non-matrix issue slots per matrix slot |
| 3 | **route (c) — library matmuls on wave bodies** | TTFT 1,077.4 → 384.7 ms at 1,024; matmul 74.4 % → 21.2 % of the wave; **971 → 2,706** at 1,024 (2.79×) | the hand GEMMs |
| 4 | **`--gdn-loop 512` — the token loop moves inside the kernel** | `gdn_rec` 449 → 266 ms (1.63×), ruler 921.56 → 763.49 ms, **4,444 → 5,364** (+20.7 %) | 73,728 launches → 144 |
| 5 | **tiled attention wave `--attn-tile 1`** | `attn` 107.67 → 26.70 ms (4.03×), **+25.3 % at 1,024 and +94.6 % at 3,072** | K/V re-fetch traffic |
| 6 | **the WMMA attention round** | `attn` 263 → 133 ms (1.985×), wave 1.226×, **5,364 → 5,784** cold (+7.8 %) | score and P·V onto the matrix unit; then residency and zero spill |
| 7 | **P8b — attention leaves the per-token loop** | TTFT −30.2 % at 1,024, −23.9 % at 3,072; **636 → 911** at 1,024 | per-query attention launches |
| 8 | **`--gdn-spell unroll4` — a second spelling of the same recurrence** | `gdn_rec` 1.153×, wave 1.091×, **3,927 → 4,284** at 4,096 (+9.1 %) | how the loop gets scheduled |
| 9 | **`--attn-touch` — two per-touch instruction levers** | `attn` 1.466×, wave 1.125×, **3,295 → 3,706** at 4,096 (+12.5 %) | 116 → 45 issue slots per touch |
| 10 | **`--gemm lt` — hipBLASLt with the algo resolved per (shape, M)** | matmul 1.419× at 4,096, wave 1.049×, **3,649 → 3,828** (+4.9 %); **+11.0 % at 1,024** | which library algorithm the engine asks for |
| 11 | **A6-G1 — attention residency: 3 blocks/CU, 9 wavefronts/CU** | `attn` −22.09 %, wave −5.90 % at 4,096 and **−18.61 % at 32,768**; **7,655 → 7,986** | LDS per block (the binder), not registers |
| 12 | **the three deletions: F32 seam, embedding gather, conv coalesced read** | `cast` −57 %, `copy` −85 %, `gdn_conv` −37.7 %; paired wall −1.83 %, −1.38 %, −0.91 %; **4,284 → 4,419 → 4,444** | dead work and per-token submissions |
| 13 | **Q6 — projection `kRowsPerBlock` 8 → 32** | TTFT −5.6…−8.0 % (decode −1.3…−2.7 %) | block width on many-row bodies |

**Cumulative on one ruler.** Cold `RESULT ms_total` at **4,096 lanes**: 3,295 (2026-10-16, before the touch
port) → 3,706 → 3,828 → 4,284 → 4,419 → 4,444 → 5,364 → 5,784 → 7,655 → **7,986** (2026-10-19) = **2.42× in
four days**, before the warm convention and A13. On the **warm** convention the shipped headline is
**11,070 tok/s at 4,096** and **8,542 at 16,384** (§60). At **1,024 lanes** the same history reads 636 → 911 →
971 → 2,706 → 3,330 (5.2× by 2026-10-14) → warm **12,047** today.

**Decode did not move with any of this, and that is the expected result, not an omission.** A one-row body is
routed away from every wave form: the hand GEMV for `rows ≤ 1` (`resolve_wave_gemm`), the per-token embedding
copy, the one-pass attention form, the per-token GDN kernels whatever `--gdn-loop`/`--gdn-geom` say. Decode
moved only through the mapping/width changes (Q6, and Q10b `--mlp-map auto`: decode **+8–11 %**, prefill
neutral by design), and otherwise held at **71.3 tok/s at 4,032 lanes / 87.1 at 128** across the whole
wave-only arc — re-measured, not merely asserted, in §36's banner re-take.

---
## 2. P8b — attention leaves the per-token loop (2026-10-13)

**Before.** Attention was emitted per query token inside the token loop, so a prompt paid an attention launch
set per query position.

**The change.** One attention **site per attention layer per body**, with the plan chosen by row count: the
fused single launch for a one-row body, `attn_prep` + `attn_core` for a wave. Engine half `f4a1095..88fd712`;
the timings are the main agent's, on the same harness and protocol as the tables above it, so the rows are
comparable for row-for-row.

**Measured (`tools/prefill_sweep.py --steps 1`, weight load excluded, spreads ≤ 2 %).**

| prompt lanes | before (P7 tree) | after (P8b) | Δ | prompt tok/s |
|---|---|---|---|---|
| 64 | 89.4 ms | 82.7 | −7.5 % | 716 → 774 |
| 256 | 347.5 | 276.5 | −20.4 % | 737 → 926 |
| 1,024 | 1,610.2 | 1,124.3 | −30.2 % | 636 → **911** |
| 3,072 | 5,407.1 | 4,116.9 | −23.9 % | 568 → **746** |

The gain grows with the prompt because the removed cost is per-query launches. Decode did not move and was
*required* not to: a decode body is one row and takes the fused path. Admission: all eight fixtures re-gated
with the gate seven checks **stricter** than before (each count = the old count + 7, from section `(h)`'s new
attention checks), and `chunk64b` back to exactly its three announced D-9 red lines.

**Why it belongs here rather than in the dead column:** it is the first change in the project that moved a
wall-clock number the user would feel, and it created the shape of everything after it — the wave machinery
(`attn_prep`/`attn_core`, later `attn_tile_wave`, `attn_slim_wave`, the WMMA and wide forms) exists because of
the body-level structure this change introduced. Against the pre-prefill era (~40 s for a 3,072-token prompt)
the sequence to that point was already ~10×.

**Evidence:** frozen ledger §"P8's payoff"; `progress/2026-10-14/speedup-routes-2026-10-14.md` §3.1; commits
`f4a1095..88fd712`, ledger commit `a201ddb`.

---

## 3. Q6 — projection `kRowsPerBlock` 8 → 32 (2026-10-13)

**Before.** The projection launcher ran 8 threads per row-block. Q6's launch bill (424 launches/token ≈ 1.6 ms
of a 12.949 ms token) had just proved the decode step was **not** launch-bound, and named the one real
inefficiency as a **shape** problem: the projections sat at 5 % of probe on the 16-row `in_proj_a`/`in_proj_b`
and 40–49 % elsewhere.

**The change.** `kRowsPerBlock` 8 → 32, with the thread count derived from that one constant so the two
cannot drift (`bc61dce`). Bit-identical by construction: one wavefront owns one row end to end.

**Measured.** Decode −1.3…−2.7 % at ≤2,048 lanes, **TTFT −5.6…−8.0 %**, both binaries in one locked window.
The asymmetry is the lesson: *a wider block only helps when there are many rows to fill it.* A burst four
days later agrees from the other side — 3.2843 ms at rows/block 32 against 3.5863 ms at 8 (−8.4 %). Proof:
byte-identical across all 40 `dec8` output files, identical counters, identical gate counts on all eight
fixtures, `margin_delta` ×1.0000 on the two tightest lanes in the tree. Ladder effect: 911 → **971** prompt
tok/s at 1,024.

**Evidence:** frozen ledger §Q6; speedup-routes §3.2; `bc61dce`.

---

## 4. Route (c) — library matmuls on wave bodies, `--gemm library` (measured 2026-10-13, default 2026-10-14)

**Before.** The engine's own hand GEMVs ran the wave-body matmuls at **0.8–0.9 TFLOP/s** — near the *vector*
path ceiling (~2.9 TFLOP/s) and 8× below the matrix units. Concretely `mlp_down` cost 4.562 ms per 512-row
launch of a shape hipBLAS does in 0.112 ms: a **37× gap**. Matmul was **74.4 % of a 1024-token wave**
(754.0 ms of 1,013.1 ms).

**The change.** `hipblasGemmEx` bf16/bf16-C on wave bodies, with a `rows > 1` seam so a decode step never
touches the library — which is why decode is bit-identical *by the seam* rather than by luck. Implementation
`1155d6a`…`5eac51d` (routed entry points, the adapter, the call-site seam and kill switch), knob
`--gemm library|hand`, default flipped in `504f93b` in binary and gate together.

**Measured (best-of-3, same window, same binary, one flag apart).**

| prompt lanes | TTFT `hand` | TTFT `library` | prompt tok/s `hand` → `library` |
|---|---|---|---|
| 64 | 79.2 ms | 85.7 | 808.5 → 747.0 (**regression**, small-body dispatch) |
| 256 | 263.2 | 130.8 | 972.5 → **1,956.5** |
| 1,024 | 1,077.4 | **384.7** | 950.5 → **2,662.1** |
| 3,072 | 3,869.0 | 1,797.8 | 794.0 → **1,708.7** (2.15×) |

Family shares: matmul 74.4 % → 21.2 % (69.7 ms after); wave wall 1,013.07 → 328.5 ms. The published flip
headline is 971 → **2,705.6** prompt tok/s at 1,024 and 1,713.3 at 3,072. With matmul cheap, the serial GDN
recurrence and attention become 71 % of prefill — which is what funded §§5, 8 and 10.

**The admission is the template for every later route.** A reassociating route cannot be byte-identical, so
D-11 replaced the cross-wave-count byte-identity check with a **cascade band**, priced by an audit rather than
by taste: 1,694 hand-vs-library logits rows over eight fixtures, worst per-row 1−cos 1.9e-3, every library
shape ≤1 ULP above the cancellation band, and — the part that makes a band legitimate rather than permissive —
three injected bugs separating from the legal delta by **492× to 1e8×** on the least flattering statistic. The
band keys and their provenance are recorded in `bench/band-history.jsonl` (rows dated 2026-10-13, decision
"D-11 accepted (option A), commit f3b2f6c"). It also came with a new reproducibility caveat stated rather than
glossed: a library GEMM's reassociation depends on `M`, so a prompt's logits now depend on the wave-row cap.

**A retraction worth keeping.** The first admission claim, "16/16 both routes", was **sixteen hand runs**:
`gate.py` appends its own `--gemm` after the wrapper's and the parser is last-wins, so the library route had
never been gated at all. Fixed by one control point plus re-pointing the route-mismatch check at the engine's
own `NUMERICS gemm=` echo — and the dry-run stand-in was made to echo what the engine actually prints, so the
selftest can now be wrong.

**Evidence:** frozen ledger §route (c), §"Correction — the 16/16 both-routes claim", §"Headline tables after
the route (c) flip"; `progress/2026-10-14/route-c-audit-2026-10-14.md`; speedup-routes §3.4.

---

## 5. The tiled attention wave — `--attn-tile 1` (2026-10-14)

**Before.** `attn` was 107.67 ms of a 329.55 ms wave at 1,024 lanes. The streaming route fetched a K (and V)
row **once per (query row, head) pair**, so with GQA `g = 4` each lane row was fetched four times: the re-use
factor the write-up had booked (4×) was not what the loop did — `for (p = w; p < pairs; p += nw)` is one fetch
per pair, re-use 1×, and V was not staged at all. The honest lever was **G**: one block per row, both K and V
staged per block.

**The change.** `prototype/attn_decode/attn_decode_kernels.h:attn_tile_body` — one K **and V** row load per
lane per **block** instead of per (row, head) pair, one query row per block (`M = 1`), and §19's aliased
split-partial slot replaced by `attn_core`'s own address. Fix `9fb12cf`, ship `6fd6e61` (default 1 in engine
and gate in one commit).

**Why it first looked impossible.** §19 had declared the tiled form *structurally* unable to reproduce the
reference above `MAXSC` (row 512, with 13.25 of decision room). It was not structural: the tile form keyed its
split partial by **head-in-group** while `attn_core` keyed by the **full head**, so above `MAXSC` two lane
blocks of one row wrote the same `2+DH` floats and half of every streaming row came out as the other KV
group's answer, bit for bit. One layout, one merge — and it is bit-identical. The rule it bought: *a
"structural" verdict needs a mechanism attached before it retires a route*, and two self-consistent layouts
joined by a branch are invisible to both sides.

**Measured (paired inside one lock hold, the two forms alternating so nothing can sit between a pair).**

| prompt lanes | shipped form ms | tiled form ms | prompt tok/s | gain |
|---|---|---|---|---|
| 64 | 83.26 | 80.91 | 768.7 → 791.0 | +2.9 % (dispatch-bound) |
| 256 | 131.78 | 122.12 | 1,942.6 → 2,096.3 | +7.9 % |
| 640 | 258.65 | 235.58 | 2,474.4 → 2,716.7 | +9.8 % |
| 1,024 | 385.15 | **307.46** | 2,658.6 → **3,330.5** | **+25.3 %** |
| 3,072 | 1,812.08 | **931.13** | 1,695.3 → **3,299.2** | **+94.6 %** |

The 3,072 row is the largest because it is geometric: 2,560 of 3,072 rows are above `MAXSC`, so nearly the
whole wave sits on the route whose traffic just fell 4×. Family: `attn` **107.67 → 26.70 ms (−75.2 %, 4.03×)**,
wave wall 329.55 → 241.00 ms. The admission line was pre-registered at `attn ≤ 40 ms` and the traffic model
predicted 27.2 ms before the run; the device returned 26.70 ms, 1.8 % inside its own prediction — and that is
241.6 GB/s of *demanded* bytes, above the box's 237 GB/s probe, which the ledger reads correctly as "the
demand model over-counts what reaches DRAM" rather than as a bandwidth record. Read the re-use factor out of
the header, the ledger adds — a traffic claim is a property of the staging loops the source contains.

**Admission.** `attn_form_band.py`: **136/136 and 520/520 above-`MAXSC` rows bit-identical**, 0 elements
differing, argmax moved 0, pooled meanrel 0.000e+00 — bit-identity, not a band. `engine/gate.py` PASS on
`wide640f` and `wide1024f` at both `--scope form` and `--scope full`, plus `dec8`/`dec64f`/`dec300f`; the op's
wave gate and its device negatives; a CPU mirror whose negative suite is 7 controls with **geometry-derived**
must-catch sets. Decode untouched by construction (a one-row body runs the fused single-row form under either
flag), and the differential agrees (12.38 vs 12.42 ms/token).

**Evidence:** frozen ledger §"Headline tables after the tiled attention wave ships";
`prototype/attn_decode/results.md` §20–§21; speedup-routes §3.5; the AGENTS.md trap rows for the aliased
layout and the derived-must-catch discipline.

---

## 6. `--attn-touch` — two per-touch instruction levers (2026-10-16)

**Before.** The tiled wave body spent **116 issue slots per (row, lane, head) touch** (55 VALU / 14 LDS / 47
SALU), and `attn` at 4,096 lanes was 403 ms of a 1,243 ms wave.

**The change.** The engine grew its own copy of the kernel (`engine/attn_slim_kernels.h`, `attn_slim_wave`)
with exactly two edits, both inside the pass-1 score loop, and **both kernels compile into the same TU** so an
ISA census always prints a shipped-vs-slim control pair out of one binary (the prototype header stays frozen —
it is what the §18/§19/§20 witnesses are made against). The levers:

1. **The dot's trip count becomes compile-time.** `nd` was a runtime value, so LLVM kept the score dot rolled:
   per element a scalar `ds_load_u16`, a `v_movrels_b32_e32` for the dynamically indexed `q[t]`, a widen, an
   FMA, an `s_waitcnt lgkmcnt(0)` and a back edge — eight of each per touch. Giving it the shape the PV dot two
   hundred lines below already had (one `ds_load_b128` plus eight unrolled convert+FMA pairs that dual-issue
   with the converts they feed) cut `dot.lds-load` 8 → 1, `dot.q-indexed` 8 → 0, `wait.lgkmcnt` 13 → 6 — same
   eight products, same serial-in-`t` order, same single F32 accumulator.
2. **The warp reduce is folded by hand, with its own wait.** Four of `warp_sum`'s five `__shfl_down_sync` steps
   lower to `ds_bpermute_b32` plus four loop-invariant guards — 16 exec-guard pairs per touch, more than the
   dot they protect. Spelling the permute once as inline asm with the address math hoisted gives
   `fold.exec-guard` 16 → 2, `fold.range-select` 7 → 0, `fold.addr-math` 4 → 0. Delta 16 stays
   `__shfl_down_sync`, because there LLVM already emits an immediate `ds_swizzle_b32` and hand-rolling could
   only lose. The one real risk: `SIInsertWaitcnts` does not recognise an inline-asm DS op, so the
   `s_waitcnt lgkmcnt(0)` is written inside the asm block — a missing wait is nondeterminism, not a stable
   wrong answer, which is why the gate must run the same plans twice and diff them.

Result: **45 slots per touch** (28 / 7 / 10), bit-identical while doing it. Knob `--attn-touch`, default 1,
commit `4b76f40`.

**Measured (paired, one lock hold).** `attn` family **403.38 → 275.19 ms (1.466×)**, whole wave
**1,243.07 → 1,105.20 ms (1.125×)**, prefill at 4,096 tokens **3,295 → 3,706 tok/s (+12.5 %)**; `gdn_rec` did
not move, which is what a control row is for. The 1,024-lane row reads 1.041× and is quoted alongside because
that row is dispatch-bound and bimodal. Re-measured in situ a week later (§26 W1) at 1.5227× / 1.5672× with
1,024/1,024 prefill logits rows byte-identical, and §26 W2 then settled a denominator collision in its favour:
the counter leg reproduces the published `attn_tile_wave` figure at −3.4 % and reads `attn_slim_wave` at 89.9
VALU per touch — pass 1 is the *high* branch, ~61 % of the kernel's VALU.

**Admission.** 4,096-lane dumps **byte-identical** between the two flag values; `gate.py` PASS at both values
with numerically identical aggregates, and the gate now **fails** if the engine's `NUMERICS` line omits
`attn_touch=` (an inert extra field was the silent-pass failure mode). Decode measured unchanged (ratio of
bests 1.029, inside noise) because a one-row body never enters the wave body.

**Evidence:** ledger §"The `--attn-touch` pair"; D22; `prototype/attn_valu_slim/results.md` §2/§7; ledger §26
W1/W2.

---

## 7. `--gemm lt` — hipBLASLt with the algorithm resolved per (shape, M) (window 2026-10-16, shipped 2026-10-17)

**Before.** The library route asked hipBLAS (`hipblasGemmEx`) for a GEMM and took the heuristic's answer. D26
found the gap was **the engine's own call, not the library's kernels**: 1.17× available at the shipping width
with no new code, and three shapes where the heuristic's rank-0 was already the best enumerated kernel yet the
engine's own call sat 1.22–1.63× off it. A sweep at a width the engine never issues would have priced the prize
at 1.66× instead of 1.17× — `kMaxWaveRows = 512` means the engine never emits `M = 4096`.

**The change.** `hipblasLtMatmul` with **one algorithm resolved per `(shape, M)` at plan time** (`2219215`),
plus a pin table so a winning algorithm is *named* rather than hoped for (`--gemm-pins`, `2ad43f0`). Shipped in
`aaefe01` with `library` kept as a first-class A/B value. A later deep-field pass (§9, `a4308be`) admitted four
candidates at burst and shipped **two, and only together**, because **a pin set is adjudicated as a set** and
its *coverage* matters more than which near-equal winner is named: `mlp_gateup` −7.0 % at 4,096 / −9.4 % at
1,024 (≈ −0.5 % of wave), while the two `proj` candidates that won at burst ran **+26 % / +27 % slower in
situ** — burst winners whose change to a neighbouring shape's L2 state turns a win into a loss.

**Measured (both routes in one `gpu_lock` hold per width, medians of 3).**

| width | `GROUP matmul_ms` | wave `wall_ms` | prefill tok/s | ratio |
|---|---|---|---|---|
| **4,096 (headline)** | 276.90 → **195.11** | 1,122.50 → **1,069.89** | 3,649 → **3,828** | **1.049×** |
| 3,072 | 210.10 → **145.26** | 794.96 → **733.24** | 3,864 → **4,190** | 1.084× |
| 1,024 | 71.11 → **47.32** | 238.02 → **214.40** | 4,302 → **4,776** | **1.110×** |

`proj` 1.561×/1.530×, `mlp_gateup` 1.647×/1.524×, and `mlp_down` the one shape that costs a little (0.945×)
because it is the pinned one. Decode is untouched **by construction**: a one-row body is the hand GEMV on
either route.

**Admission.** The single red row (`wide520f` row 180) was attributed on data before the default moved: it is a
**one-BF16-ULP tie**, and the retired absolute flip-room line (0.10 of a logit) could not see that — 0.10 is
0.8 of a BF16 step at logit 20 and 40 whole steps at 0.5. The rule was re-derived in **BF16 ULP at the row's
own top-1** (`flip_room_max_ulp = 1.50`, from a census of 74 flips over 2,951 rows × both routes × both
cross-width pairings, worst room anywhere 0.9985 ULP), and it *bites* where the old line was blind: a 2.00-ULP
injection at top-1 1.0 is red here and was admitted under 0.10. Gate 18/18 on both routes under the new line
with every injection control still biting. `bench/band-history.jsonl` carries both events (the removal of the
absolute key and the application of the ULP key, 2026-10-16).

**Evidence:** ledger §§7, 9 and §9's same-day correction; D26–D29, D35, D36;
`prototype/gemm_algo_sweep/results.md` §3a, §3d, §3f.

---
## 8. `gdn_rec` gets a second spelling — `--gdn-spell unroll4`, then `fused` (2026-10-16 / 2026-10-17)

**Before.** `gdn_rec` was **47.6–51.2 % of the wave** and three stories had been told about why: traffic-bound,
instruction-count-bound, dependency-latency-bound. D32 refuted all three with a falsifier and found the answer
was **how the loop gets scheduled**.

**The change.** A second spelling of the *same operations* with `#pragma unroll 4` on both state walks
(`engine/gdn_rec_unroll4_kernels.h`, knob `--gdn-spell`), shipped in `a1cde3b`. **Bit-identical** — there is
nothing to trade, no value moves, so the A/B is a pure timing question.

**Measured (paired inside one hold per width, medians of 3, blocking).**

| width | `gdn_rec` ms | wave `wall_ms` | prefill tok/s | family ratio | wave ratio |
|---|---|---|---|---|---|
| **4,096** | 496.55 → **430.69** | 1,043.11 → **956.12** | 3,927 → **4,284** | **1.153×** | **1.091×** |
| 1,024 | 121.71 → **110.62** | 209.46 → **197.66** | 4,889 → **5,181** | 1.100× | 1.060× |

Every `unroll4` rep beat every `ship` rep (worst-vs-worst 1.110×), on the family that is half the wave. Decode
moves 0.11 % — four times below the ruler's own spread — so the ledger states it as arithmetic, not as a
re-take, and records that the row is owed a re-take if the decode ruler ever gets tighter than 1 %. Admission:
ten fixtures green across **both** spellings, and `gate.py` section `(j)` **byte-compares the two corpora
inside every one of those runs** (`wide1024f`: 1,032 logits rows / 1,025.1 MB per side, identical) — bit-identity
re-proved per gate rather than inherited.

**Follow-up: `fused` (`c564b47` → `ebeebdb`).** The merged fused-prologue-on-unroll4 form: 1,374 vs 1,406
slots/token-step at compile time (same 59 VGPRs, zero spill), then a paired in-situ A/B at 4,096 lanes giving
`gdn_rec` 455.83 → 428.22 ms, **ratio 1.0645 with 3/3 reps**, decode flat (1.0025). The default moved — and
**the headline did not**: 4,436 vs §17's 4,444, i.e. unchanged. The ledger books the family row at −2.6 %
cross-window, explicitly refuses to quote the paired −2.96 % of wall as a new headline, and states why the two
disagree (the `unroll4` side of that window ran ~2.8 % slow). A real but sub-noise win, labelled as such.

**Evidence:** ledger §§8, 19, 20; D32, D33, D50, D51; `prototype/gdn_rec_isa/results.md` §8–§11.

---

## 9. Three deletions — the F32 seam, the embedding gather, the conv's coalesced read (2026-10-17)

Small individually (+1.8 %, +1.4 %, +0.9 % of the wall), and together they took the 4,096-lane ruler from
**4,284 → 4,419 → 4,444 tok/s**. They matter as much for the mechanism each settled.

### 9.1 The F32 seam deletion — `cast` −57 % for a guard (`26cd5d6`)

Under the library route the wave-body GEMMs are handed a **16-bit** operand, yet the code still widened the
same tiles into F32 scratch anyway: `decode_step.cpp`'s own `project()` picks `x_bf16` under `lib_gemm`, and
the MLP says so in a comment two lines above widening unconditionally. The dead set was **derived from the
readers**, not taken from the census's line numbers, and that mattered: `norm_f32` is dead for attention layers
only, because `in_proj_a`/`in_proj_b` call `proj_gemv` directly at every width, so 18 of the 24 mixer-input
casts per body are live. Derived dead set: **78 of 114 casts per wave body, 4.08 GB/wave.**

Paired in one hold: `cast` 913 → **289** launches and 27.3218 → **11.7545** ms at 4,096 lanes (**−56.98 %**
against a predicted −57 %), wall **−1.83 %**; at 1,024 the row is −54.76 % and the wall −4.65 %. **The row and
the wall move together** — which is exactly what distinguishes a real deletion from a null where the wall sits
still. Deleting a widening cannot move a bit, so no numerics band was consulted; the guard is a route test, so
the hand route had to come back unchanged, and it did (18/18, nine fixtures × both routes).

### 9.2 The embedding gather — `copy` −85 %, and submissions are **not** absorbed (`47274f4`)

`copy` was 4,096 per-token 2 KiB D2D copies of the embedding table, one submission per token. Its bytes are
worth **0.16 ms against its own 10.86 ms row**, so it was never a traffic item: `tools/submission_probe.cpp`
measured **2.05 µs/op for a real copy against 1.72 µs for an empty launch** — 84 % of the op is the submission,
and the same bytes as one operation cost 0.10 ms instead of 8.40 ms. One gather kernel plus one id upload per
body replaces all 4,096: **4,240 → 160 launches, 10.9677 → 1.6240 ms (−85.2 %)**, wall **−1.38 %**, all six
pairs negative across both widths. The prediction had been written before the measurement and survived
two-sided, including its refuting branch: had the row collapsed with a still wall, per-op submission cost would
be absorbed by a busy stream and every launch-shaped row in the D30 inventory would have needed re-reading. **It
is not absorbed, so launch-count reduction is a real lever on this box.**

The host half was the hard part: the kernel needs the body's token ids on device and none lives there for the
wave path, while the host runs ~120 ms ahead of the device, so a single staging buffer would be overwritten
before the previous body read it. The scratch holds **8 slots with an event recorded after each copy**, capping
look-ahead at 8 bodies while still leaving seven queued in front of the device at every wait (`wide520f
--wave-rows 5` runs 96 ring wraps bit-identical). The shipped wave's plan drops 80,426 → 76,346 launches while
a decode step's bill is unchanged at 424 on both routes.

### 9.3 The packed GDN cast, and the conv coalesced read that paid for it (`62b73e9` → `5096a57`)

`62b73e9` deleted the packed BF16→F32 GDN widening by having `conv_kernel`/`gates_kernel` widen their operands
in-register: `cast` **−81.3 %** … and `gdn_conv` **+32.5 %**, where the prediction had it 22 % *faster*. Reason:
the packed F32 tile had been doing a second job as a **coalescing layout** (one contiguous tile read instead of
three strided 2-byte fetches). The byte account was right and its timing consequence was backwards. Net of the
three affected rows: **−0.50 % of wave**, kept, with the recoverable prize named and sized at ~6.5 ms.

`5096a57` collected it: `CONV_VEC = 4` with `aligned(8)`/`aligned(16)` chunk structs (natural alignment 2 would
have emitted four 2-byte loads however the source was written), grid 24 blocks → 6. Before: `global_load_u16`
(64 B per wavefront, half a line) with `s_waitcnt vmcnt(0)` right after every tap — 29 immediate drains. After:
`global_load_b64`/`b128` (256 B) with the whole K-row window fetched before any of it is consumed — 4 loads → 1
and 4 drains → 1 per output element at 4× the width. **`gdn_conv` 26.90 → 16.76 ms (−37.7 %), five of five
pairs negative, wall median −0.91 %.** The attribution is the point: **the win came from latency, not traffic**
— the new form reads 4 bytes per element more than the ideal and is still 37.7 % faster than a form that read
half what the old one did. Ruler after it: **4,444 tok/s** at 4,096.

**Evidence:** ledger §§10–16; D37, D38, D39, D40; `progress/2026-10-17/glue-cast-copy-bytes-census.md` §3/§8;
`progress/2026-10-17/HANDOFF_17prime_gdn-cast-deletion.md`, `HANDOFF_convread_coalesced-load.md`. Two honesty
notes carried with these ships: (a) §13's first A/B printed a clean "nothing moved" because the `new` side was a
**stale binary** — builds on this box are not bit-reproducible and `plan_match=1` is compiled into the binary so
it agrees with itself — which is why every A/B script since carries a **behavioural canary** asserted before any
timing; and (b) §15.2 corrects three of these admissions: the "18/18 — nine canonical fixtures" claim rested on
a fixture list in which **seven of the nine names do not exist** in `data/refs/`, and `PASS=0` from a harness is
an unclassified run, not a result. The population is now derived by enumerating `data/refs/*/manifest.json`.

---

## 10. `--gdn-loop 512` — the token loop moves inside the kernel (2026-10-18)

**Before.** The GDN recurrence relaunched **per token per layer**: 73,728 launches for a 4,096-token prompt,
each carrying a 1.72 µs submission (≈ **127 ms**, 13.8 % of the wave) plus its own prologue and drain. The
comparative trace (§22, `rocprofv3` on both this engine and llama.cpp in one hold) put llama at **8 launches per
layer per 4,096 tokens** — one per 512-token chunk, walking the tokens inside the kernel — against our 4,096:
512× the launches and ≥9.0× the family time.

**The change.** The token index moves **inside the kernel** (`engine/gdn_rec_loop_kernels.h`), one launch per
body. Knob first (`980f926`, `--gdn-loop N`, default 1), default moved on the in-situ evidence (`80930c1`).
**Bit-identical by construction**: the recurrence is already sequential, so the same expression tree runs in the
same order. This is explicitly *not* the chunked delta-rule scan that died as D-12 — it keeps the scan and
removes the relaunching.

**Correctness first, then a burst.** A host mirror of the looped mapping bit-compared **39,444,480 elements**
across spans {1, 2, 3, whole}, `bad=0`, with five injected loop defects each caught exactly where their geometry
predicts (`wrong_catch=0`, `defects_unreachable=0`); a transcription checker proved the 33 core lines identical
to the shipped kernel including the `#pragma unroll 4` lines, and caught all four of its own mutations; device
admission at spans 0/1/3/16 **bit-identical every time**. The screen then read **1.58–1.64×** with per-token
cost falling from 5.43–5.74 µs to 3.43–3.51 µs — above the band that authorised a port — and only then was the
port made.

**In situ (medians of 3, both sides in one hold).**

| shape | base `gdn_rec` | looped | ratios per rep | counted launches |
|---|---|---|---|---|
| 4,096 lanes | 449.07 ms | **266.03** | 1.6002 / 1.6893 / 1.6881 | 73,728 → **144** |
| 1,024 lanes | 109.34 ms | **67.16** | 1.5845 / 1.7281 / 1.6280 | 18,432 → **36** |

All seven pairs favour the looped form (median 1.6280), and the 1,024-lane worry (two bodies instead of eight)
did not materialise. Per-launch cost goes 5.94 µs → 1.90 ms per 512-token launch. Admission: `wide520f` (520
prompt tokens = a full 512 body plus a ragged 8, exactly the cases the tail logic and the plan bill exist for)
ran **370 checks, 0 failures** and byte-compared 524 logits rows / 520.5 MB of the looped body against `ship`;
**1,024/1,024 prefill logits rows byte-identical**; decode flat at median ratio 0.9999 over three paired
one-row reps, which is what the one-row routing predicts and is now a measurement rather than an argument.

**The ruler (§25).** `ids4096.txt`, three reps, one hold: **921.56 → 763.49 ms = 4,444 → 5,364 prompt tok/s,
+20.7 %** — the largest single move to that date. Elastic at every width the engine serves: 64 lanes **+12.0 %**,
256 **+21.0 %**, 1,024 **+22.2 %** on the wall, family ratios 1.574–1.640×. The known cost the harness flagged —
the loop bills **+156 slots/token-step** from recomputing `t*CH` per iteration — was paid here and still came
out 1.63×. (That "+156" was itself later corrected, D60: a flat per-kernel ISA census divides once-per-launch
prologue work by tokens; split by LLVM's own loop annotations it is **−47 per token**.)

**A null that nearly read as a refutation:** the first short-width A/B generated its ids **space**-separated;
`--ids` splits on commas, so `parse_ids` read one token and both sides priced a one-token wave — 12.99 vs
12.93 ms at every "width", which reads exactly like "the loop does nothing below 1,024 lanes". Two guards came
out of it: print `launches=` beside the family `ms`, and assert that a sweep's `tokens=` field matches the width
it names.

**Evidence:** ledger §§22–25; D52–D55; `prototype/gdn_rec_isa/`; commits `980f926`, `80930c1`.

---

## 11. The WMMA attention round (2026-10-18 → 2026-10-19)

**Before.** After §5 and §6, `attn` was 265 ms of a 763 ms wave (~34 %) and §26's census established *why* it
was worth attacking: **zero matrix and zero dot-class instructions in all 40 kernels the engine builds** — every
matmul that reaches the matrix unit on this box reaches it through hipBLASLt, whose tuned kernels are unreadable
CCO containers. Against the right roof (scalar `v_fma_f32` 7.30 TFLOP/s → `v_dot2_f32_bf16` 23.28 → WMMA 52.63),
attention's 1.59 TFLOP/s was **4.6× low**, and llama.cpp's rate on that phase sits *above* the best scalar rate
this machine has ever sustained — so parity was a matrix-class question regardless of what our own disassembly
said. (An earlier framing of "14× below the roof" was retracted: 22.3 TFLOP/s is a *matrix* rate and this kernel
could only occupy a scalar rung. Judging a kernel against the ceiling its own instruction class can occupy is its
own lesson.)

**The change, in four measured steps.**

1. **The WMMA streaming form ships** (`5024f2e`, NaN fix `eb3f4a1`, defaults `748d790` + CLI `9b09056`).
   `v_wmma_f32_16x16x16_f16`: one block owns four query heads × sixteen query rows × a sixteen-lane KV round;
   the score of a whole 16×16 block is one `v_wmma` per 16-dim step, the online `(m, l, acc)` fold is one
   cross-lane max per row per round, the output accumulator is 16 independent F32 chains over `DH = 256`. It
   takes exactly the rows the shipped form already **streams** (above `MAXSC`) and **refuses** the one-pass rows,
   because routing a pinned one-pass row through it would move bits the pins currently judge exactly. The
   fragment map was measured on device (`fattn_map_probe` + `check_fattn_map.py`), because the prototype's first
   transcription put Q first and produced a transposed score that folded every row down the wrong axis.
   Result: `attn` 263.38 → 188.74 ms (**1.3955×**), wave **1.1407×** — under the 1.5× bar that had been set, and
   recorded as a failure of the bar rather than as a near-miss, then adopted on the merits.
2. **Un-staging Q** (`8a7c222`): the block stops staging Q, `group_segment_fixed_size` 16,384 → 8,192 B, and the
   driver's own answer goes **4 → 8 wavefronts/CU** — the family a further 1.20× to 160.65 ms, 1.647× over frozen.
3. **Three warps per block sharing one K/V stage** (`2ac956d`): 96 threads, each warp owning a 16-row block of
   one head — **178 VGPR, 0 spill, 0 scratch, 6 wf/CU**. This is the discriminator: **spill-free at 6 beats
   spilled at 8** by 1.217× on the family, so spill traffic was real and cost more than the interleave it bought.
   Cumulative **1.985× over frozen**, +21.4 % on the wave in that window.
4. **The banner re-take (§36)**: the ruler on the shipped form with the clock measured rather than inferred
   (819–838 MHz idle, 2,423–2,437 MHz under load → ≈2.43 GHz sustained): **708.14 ms = 5,784 prompt tok/s at
   4,096** (+7.8 % over §25). §37 then compared HEAD against `origin/master` family by family in one window:
   `attn` **1.996×** — matching §33's 1.985× from an independent window and harness — wave **1.226×** at 4,096,
   and **every family the round did not touch inside ±3.1 %**, which is how this ledger now states its own noise
   floor: *in a paired window a sub-3 % move on a sub-30 ms row is not claimable in either direction.*

**What it cost to get right — three NaNs and one withdrawn win.** The first 1.3665× was measured on a form whose
covered rows were **entirely non-finite** (248,320/248,320 NaN per row), and the acceptance canary was green
*because* of that: it asserted the two sides differ bit-wise, and a NaN row differs from a finite row
beautifully. Three gate checks could not see it either (`nan > cap` is False; one side did not forward the flag;
a "bit-identical over 1025 MB" claim vacuous over non-finite data). The win was withdrawn the same day (§29
correction, D64) and re-earned. The Bet-2 multi-warp NaN was one loop: each warp filled its own Q stage with the
*shared-buffer* stride, and the `WARPS=1` bisect that "proved" the fill correct was structurally blind to the
only thing that differed above one warp. §34 records that the tip also failed to build for one commit — the
kernel half landed without the launcher half, in a different header — so the rule became **build the committed
tree**, not the working one.

**Admission.** `wide1024f --scope full` PASSES with the default on: 512 streaming rows, 0 outside either cap, 0
non-finite, 0 argmax moved, worst 1−cos 1.99e-04 against the 2.0e-3 cap, boundary row 512 cos deficit 2.8301e-05,
rows 0..511 bit-identical between forms. **No new band was needed.** Decode was re-measured unchanged in §36
rather than declared unchanged.

**Evidence:** ledger §§26–37; D56–D64; `prototype/attn_wmma/{evidence/,…}`; commits `5024f2e`, `eb3f4a1`,
`748d790`, `9b09056`, `8a7c222`, `2ac956d`.

---

## 12. G4 `reg16` — the GDN walk keeps its state shard in registers (2026-10-19)

**Before.** `gdn_rec` was 41.3 % of the wave (266 ms at 4,096 lanes) against llama.cpp's 47.2 ms on **identical
launch counts** (144 each) — a **5.76×** per-kernel gap. Every structural hypothesis had just been measured dead:
norms hoisted −5.9 %, barrier-free one-wavefront geometry **+121.9 %**, state LDS-resident +10.6 %, parallel
scan +493 %, unrolled +0.5 %. What finally cracked it was **reading llama's kernel source**
(`engines/llama.cpp/ggml/src/ggml-cuda/gated_delta_net.cu`, vendored in this repo the whole time) instead of
profiling it: both engines perform the same 4 FMA-class ops per state element (7.73×10¹⁰ MAC/pass), llama issues
**1.28× more** slots per token and finishes 5.76× faster, and the surviving difference is **state elements per
thread — 128 ours, 16 theirs**. At 16 the shard lives in registers, so llama moves no state per token, reduces
rows with 8-lane butterflies needing no barrier, and fields **8× our wavefronts** (6.40 vs 0.80 wf/SIMD). (The
mis-read that caused the detour: `rocprofv3`'s `Grid_Size_*` is in **threads**, not blocks — §38 read one axis
and recorded llama's shape as identical to ours, which inverted the sign of a real measurement.)

**The change (`46139e9`, default `e1df7a7`).** llama's thread map under our recurrence: **128 blocks × 128
threads, 8 lanes per state column, 16 state elements per thread living in VGPRs from the first token of a body
to the last**. The map is forced by arithmetic, not chosen: `wavefronts × 32 × elements/thread == 262,144`, so
raising wavefronts means lowering elements, and 16 is what fits in a register file that also holds `k-hat` and
`q-hat`. k/q row norms are a 3-level butterfly over the 8 lanes of a column group: no block reduction, no
barrier, and — the reason the earlier barrier bets are not repeated — no staging buffers at all. State layout is
unchanged, so every consumer (`state.h`, `STATEFP`, the decode handoff) keeps working. Verified on an independent
compile rather than inherited: `next_free_vgpr 76`, `group_segment 0`, private segment disabled — and **inside the
token loop 9 global loads and exactly one global store (`o`), with all 16 state loads and 16 state stores outside
it.** The stride-512 B first load happens once per 512 tokens: **16 KB/token/layer against the ~3 MB/token/layer
the shipped walk pays.**

**Measured (window `g4-ab`, 4,096 lanes, medians of 3, one hold, against lines pre-registered before the code
existed: `gdn_rec` ≤170 live / >225 dead, `wall_ms` ≤540 live / >607 dead, with `wall_ms` the ship gate because
it is the only denominator that can see cost land in a neighbour).**

| side | `gdn_rec` ms | wave `wall_ms` | `RESULT ms_total` | externality | verdict |
|---|---|---|---|---|---|
| `reg16` vs shipped `block` | 267.09 → **89.22** (−66.6 %) | 628.89 → **453.65** (−27.9 %) | 712.065 → **535.105** | +2.18 ms | **LIVE → shipped** |

**5,764 → 7,655 prompt tok/s at 4,096 lanes** — the largest single-engine move in the ledger. Traced, same
window, same 144 launches/pass: `rec_loop_kernel` 276.99 → `rec_reg16_kernel` **89.22** against llama's
**47.20**, so the gap this section chased goes **5.87× → 1.89×** and the GDN phase falls from 43.8 % to 18.7 % of
the traced wave. Long context re-taken at the new default (§45): 8,192 lanes **6,406 tok/s (+20.6 %)**, 16,384
**4,937 (+19.1 %)**, 32,768 **3,136 (+12.2 %)** — the decay with width is itself the arithmetic check, since a
fixed-fraction win on a shrinking share must look exactly like that.

**Numerics, declared rather than negotiated.** The state dot **re-associates** through 16-element partials and an
8-lane butterfly, so `o` and the state fingerprints move; judged by the existing §10.1 aggregates, no band
widened, no `--ack-spill`. One inherited far-field witness read red (`rms_rel ≤ 1.25 × shipped`, measured
1.250290) and is now **informative rather than binding, on evidence**: the shipped binary's own `rms_rel` to the
pins swings 27.8 % between the very spans where the ratio is measured, the sign flips with span, `reg16` reads
0.694 (closer to the pins than shipped) at wave-rows 16, and a 64-step free run emits 110/110 identical token
ids. A witness whose noise exceeds its margin cannot adjudicate; the constant was left untouched, the check still
prints on every span with its noise quoted beside it, and the flag that could have waved it through was deleted
rather than kept.

**Evidence:** ledger §§38, 42–45; `prototype/gdn_walk/{G3-llama-kernel.md,BET.md}` §12–§13; commits `46139e9`,
`0d1f41c`, `e1df7a7`, `eb43ca1`; `progress/bench/evidence/g4ab-2026-10-19/`.

---
## 13. A6-G1 — attention residency: 3 blocks/CU and 9 wavefronts/CU (2026-10-19)

(Not to be confused with the GDN bet `G1` of §40, which hoists the k/q norms out of the GDN walk and was **not**
shipped. This one is the attention bet `A6-G1`.)

**Before.** The shipping attention wave `attn_fattn_wave_u8` sat at **178 VGPR / 0 spill / 32,768 B LDS** = 2
blocks/CU = **6 wavefronts/CU, LDS-bound**. Three routes to a third resident block had just been closed: A3
derived the LDS floor of the shipped arithmetic at 32,768 B with Q staged at all, and killed half-tile staging
as a work-split change; A4-L1 priced the accumulator dependency chain at **−1.67 % and dead**; A2 bounded the
pipes at 5.56 % of the matrix roof and ≤8.8 % of issue. The one measurement that made residency the live
thesis was G0: warps 3 → 7 and nothing else moved **−10.82 %** on the family, exactly what §31's log-elasticity
of 0.72 had predicted for 6 → 7 wf/CU.

**The change (`2808c13` knob, `40a19ec` default).** Two edits, both staging addresses only: **Q is read from
global at the use site** (the E1 port, re-anchored after the roll knob moved the roll) and **the epilogue's F32
transpose is staged 8 rows at a time**. LDS per block goes `3 × 8,192 + 8,192 = 32,768` → `3 × 4,096 + 8,192 =
20,480` B, which allows **3 blocks/CU = 9 wavefronts/CU** at 185 VGPR, spill 0 — and the driver's own occupancy
query grants 3 blocks, which settles the register-file question A3 had left as a bracket.

**Measured (paired; NULL arm −0.41 %, rep spreads ≤1.56 %).** `attn` family **135.3870 → 105.4753 ms
(−22.09 %)**, traced **−17.54 %**, wave `wall_ms` **451.8840 → 425.2243 (−5.90 %)**, and at long context
**−14.22 % at 16,384** and **−18.61 % at 32,768**. Headline: `RESULT ms_total` best-of-5 **512.908 ms** against
base best-of-3 **531.285** = **+3.46 % → 7,986 prompt tok/s at 4,096**; on §45's denominator, 16,384 lanes goes
4,937 → **5,801 (+17.5 %)** and 32,768 goes 3,136 → **3,838 (+22.4 %)**. The win *grows* with width, which is
§46's claim that the quadratic term is bought with residency, observed.

**Why it cost nothing in numerics:** it moves staging addresses only and is **byte-identical** — base vs default
and base vs `unstaged8` match on every dumped logit file and all 44 `STATEFP` rows at 640, ragged 1,111 and
2,000 ids. `engine/gate.py --fixture wide640f` on the committed tree at the new default: PASS, nothing widened,
no `--ack` of anything. Decode: paired 64-step reads +1.4 %, inside the floor — noted, not claimed. Externality
+2.83 / +20.08 / +5.05 ms: better than G0's +74.72 ms at 16k, but a third resident block is not free for its
neighbours.

**The methodological point the ledger keeps:** three negative results (A2's bound, A3's LDS floor, A4-L1's
dependency price) bought the discriminating one-kernel change G0, and G1 then reached the byte count A3 had
called unreachable **because A3's kill was specific to splitting a 16-row round, which G1 never does**. *A
closed route is a claim about a mechanism, not a byte count.* Corroborated later from the other direction by
§55's pad A/B: deliberately padding LDS back to 32,768 B (same VGPR, same work) costs **+3.36 % at 3,072 lanes
and +4.42 % at 16,384** with disjoint per-rep ranges — which proves LDS binds at 3 blocks/CU, proves the
register file is per-SIMD rather than per-CU, and proves a register saving cannot buy residency on this geometry.

**Evidence:** ledger §§47–48, §55; `prototype/attn_percost/{A6-BET.md,A6-G0-verdict.md,A6-G1-verdict.md}`;
commits `2808c13`, `a2ac3b1`, `40a19ec`.

---

## 14. The wide-attention arc: A12 refused, **A13 shipped** (2026-10-20 → 2026-10-21)

This is the largest long-context win of the project, and it only makes sense as a pair: **A12 was measured,
refused, and produced the constraint that made A13 work.**

### 14.1 Why the first wide shape did not ship (§56–§58)

`--attn-fattn-wide 1` (A12, `engine/attn_wide_kernels.h`, a 128-row wide block) won **+25.1 %** in a rig's single
2,048-workgroup launch and **~0 %** in the engine. Four legs established why, and each closed an axis:

* **Coverage is total** — the engine's own `ATTN_WIDE` echo reads `wide_rows=512/512` at all 32 geometries of a
  16,384-lane prefill, so "it is not covering the rows" is refuted.
* **The rate win is absent, not diluted** — paired per-body dispatch timestamps read the wide core at **+0.20 %**
  against the shipping core, ratio per body 0.937–1.122 with no trend.
* **The mechanism is launch population**: the wide block covers 32 query positions and has no lane split, so a
  `kMaxWaveRows` body emits `2 × ceil(512/32) = 32` workgroups on a **40-CU** machine — ≤1 block/CU on 32 CUs,
  8 wavefronts/CU against the 9 the form it replaces sustains. Sweeping `--wave-rows` at fixed total work made it
  a curve: ratio 0.9785 / 1.0867 / 1.5582 / 2.0080 at 32 / 16 / 8 / 4 blocks per launch.
* **And the residency it needed does not pay**: giving it a 2,048-row body (128 workgroups, the rig's own
  2 blocks/CU) made it **+2.45 % slower at 16,384** — so the rig's +25.1 % was a property of the rig's workload,
  not transferable. `kMaxWaveRows = 512` was separately confirmed to be at its optimum (halving the body costs
  +10.8 %, doubling costs +2.8 %).

The rule these legs installed: **price the geometry the engine actually emits**, and read workgroups per launch on
both sides of any kernel A/B. One residual is worth keeping: ~78 % of the wide route's ~2 % engine win is the
**skipped split round-trip**, and `--attn-split-cap 8` on the *shipping* kernel reproduces −1.14 % of the
−1.46 % without a numerics risk. That knob was itself measured at −4.61 % of attention but only −2.38 % of the
32k wave and **displaced** (§53), so it stays inert.

**What A12 contributed instead of a win** was the census that reframed the whole problem. Same ISA tool, run on
all three kernels: **non-matrix issue slots per matrix slot** — shipping `attn_fattn_wave_u8` **108.3**, wide
`attn_wide_wave` **110.9**, llama's `flash_attn_ext_f16<256,256,16,4>` **51.9**. Their amortization is 2.09×
better against a measured 1.9× rate gap, which closes the account arithmetically, and it explains why A12 moved
the ratio the wrong way (7.5× the matrix work per round but 8.2× the non-matrix bill). It also killed two
plausible-sounding explanations before anyone funded them: address arithmetic (`int_vec` 19.1 vs their 18.1) and
`s_waitcnt` behaviour (4.2 vs 4.2) are **already at their level** — the difference is `move` (19.8 vs 2.9),
`salu` (15.8/21.8 vs 3.7), `delay`, `select_mask`, and the 128-slot accumulator rescale.

### 14.2 A13 — the llama-ported round, `--attn-fattn-wide 2` (shipped)

**The change (`99da4c9`, default `ec72d94`; `engine/attn_wide_llama_kernels.h`, 486 lines).** Four structural
features ported onto the wide block, all aimed at the one quantity that closes the gap — non-matrix slots per
matrix slot, round-scoped **24.05 → 20.56**:

1. **Template-parameter LDS strides with an XOR swizzle**, with per-lane bases hoisted out of both rolls, so no
   address is computed at run time in the steady state. Writer and reader share one expression
   (`wll_k_addr`/`wll_v_addr`) — the §20.1 law applied to a permutation.
2. **The next round's K prefetched into registers** after the score tile is consumed and committed once V is
   done — a *register* pipeline, not an LDS double-buffer, which would have needed 64 KiB against a 32 KiB
   residency claim.
3. **`template <bool kMasked>` full/tail round split**, so the full-tile body carries no per-cell bounds
   predicate at all. This is a *stronger* version of llama's single specialized `last_iter`, justified by
   arithmetic: a wide block spans 32 query positions and `lanes_of` advances by 1 per row, so a block's causal
   spread is 31 lanes, less than one 64-position round.
4. **`SCALE = 2⁻⁴` folded into Q at staging.** Exact, because scaling by a power of two commutes with rounding
   and with the F32 WMMA sum; the only hole is a Q element under ~2⁻¹⁰, which falls into the F16 subnormal range
   once scaled.

**Two corrections the funding plan needed, recorded because they are the reason this is not "just copy llama":**
llama's swizzled `ldmatrix` path and its `cp.async` multi-stage pipeline are both **CUDA-only**
(`TURING_MMA_AVAILABLE`/`CP_ASYNC_AVAILABLE` require `!GGML_USE_HIP`), so on RDNA3 llama **pads rather than
swizzles and stages synchronously** — items 1 and 2 are this project's own engineering wearing llama's hat, and
they are the two that won. A packed `v_pk_mul_f16` rescale, planned as item 3, needed no port: LLVM already
emitted 64 per round in both forms. And a source census that reads the code but not the preprocessor is a census
of the readable subset.

**Measured (warm, §51 convention, median of 3, knob 0 → knob 2, one binary).**

| width | knob 0 warm | tok/s | **knob 2 warm** | **tok/s** | Δ |
|---|---|---|---|---|---|
| 512 | 48 ms | 10,667 | 43 | **11,907** | **−10.42 %** |
| 1,024 | 94 | 10,894 | 87 | **11,770** | −7.45 % |
| 4,096 | 422 | 9,706 | 359 | **11,409** | **−14.93 %** |
| 16,384 | 2,674 | 6,127 | 1,942 | **8,437** | **−27.37 %** |
| 32,000 | 7,960 | 4,020 | 5,116 | **6,255** | **−35.73 %** |

Witnessed again as the **default**, not as a knob, by scripts that pass **no form flags** and assert each run's own
`attn_fattn_wide=`/`form=` echo (§60): **512 → 11,636 tok/s, 1,024 → 12,047, 4,096 → 11,070, 16,384 → 8,542,
32,000 → 6,154** — i.e. **+13.1 % at 4,096 and +39.5 % at 16,384** against §51's published figures. Cold moved
alike and with disjoint ranges (16,384: 2,782 → 2,032 ms). Knob 1 (A12) sat at −1.0…−1.6 % throughout, which is
the number §56/§57 established for it.

**Why it is that big.** §51 had attributed the whole long-context gap to attention and quantified it: at 16k our
attention cost **+751 ms/pass** more than llama's while the total gap was +596 ms; at 32k **+2,815 ms** against
a total gap of +2,489 ms. A13 removes **−732 ms at 16,384** and **−2,844 ms at 32,000** warm — **97.5 % and
~100 % of exactly the excess §51 named**. Against llama's own traced totals this now reads **8.7 % faster at
16k** and **12.4 % faster at 32k** (cross-window against §51's llama figures, and labelled as such). GEMM was
already 0.88–0.90× in our favour, so **the attention question is closed**: any remaining long-context difference
is not attention. §58's amortization model predicted the size from the other side, which is the rarest and best
kind of agreement in this ledger.

**What it cost, and why that is acceptable.** Spilled VGPRs **163 → 920**, private segment **176 → 720 B/thread**
(inside A12's declared 1,024 B budget, printed as `scratch=720 SPILL` on the engine's own echo — and that field
is the behavioural canary the window asserts, since a stale binary agrees with itself). The slot cut pays for it
many times over at every width.

**Admission.** Byte-identical to knob 1 on the five pinned geometries (**1,960 files, 0 differing**) and
bit-identical at 4,096 lanes including all 4,099 free-run decisions. At 16k/32k it is **not** bit-identical, and
that is quantified rather than waved through: **1−cos 1.41e-05 and 6.43e-05**, argmax SAME, top-5 SAME on every
row, and **the entire 16,385 / 31,826-token greedy continuation agrees token-for-token** — with all 32 wave
lines carrying identical `grid`/`wide_rows`/`lanes_last`/`split`, so coverage is disproved independently of the
numbers and this is re-association, not skipped work. Gate over the derived 18-fixture population: **knob 2
18/18, knob 1 18/18, knob 0 (the shipping form) 17/18** — and after the default flipped, **18/18 with no flags
passed, including `chunk64b`**, the batched-vs-one-row token flip (2.925 BF16 ULP of room against a 1.50 line)
the shipping path had carried as a declared red since before A12. The flip **removed** that red rather than
inheriting it. **No band was widened and no pre-authorisation was spent**; the only gate change is that the knob
accepts `2`, the echo map is an explicit dict so a fourth form cannot fall into an `else` and be judged as the
shipper, and `wide_llama` is routed onto the already-adopted wide-scoped pooled line (measured 2.028e-04 against
2.500e-04 — 19 % headroom, which the ledger flags as tight for the next wide-form change).

**Evidence:** ledger §§56–60; `prototype/attn_wide/{A12-WHY-NOT-FASTER.md,A13-RESULT.md,LLAMA-PORT-PLAN.md
(+ correction banner),evidence/a13_*}`; `prototype/attn_percost/R2-amortization-spec.md`; commits `197d041`,
`41d1cb7`, `6b857d2`, `ef38189`, `67ab337`, `99da4c9`, `ec72d94`.

---

## 15. Secondary wins, and one that moved decode only

| change | commit | what it moved | measured | prefill effect |
|---|---|---|---|---|
| **`--mlp-map auto`** (Q10/Q10b) — the MLP weight walk follows the path: `contig` for a one-row body, `interleave` for a wave | `b298e73`, `734b3d9`, `d1703b0` | which two-stream walk the MLP uses | `contig` recovers **0.945 ms of the MLP's 2.034 ms in-step gap** at 128 lanes; as a whole-default flip it cost **+5.1 % TTFT** at 1024, so the split ships | **decode +8–11 %, prefill neutral by design** (bit-identical either way, so a pure timing A/B). The 4k row improved −5.8 %. |
| **deep algo pins** (`--gemm-pins all` = `down_proj` + `deep_gate_proj+up_proj`) | `2ad43f0`, `a4308be` | which hipBLASLt algorithm two shapes get | `mlp_gateup` **−7.0 % at 4,096, −9.4 % at 1,024**, every pair | **≈ −0.5 % of wave**; no tok/s headline claimed, because the wave column is carried by families no pin touches |
| **packed GDN cast deletion** (`62b73e9`) | see §9.3 | per-token launch bill 424 → 406, wave plan 76,346 → 76,202 | `cast` −81.3 %, `gdn_conv` **+32.5 %** | **−0.50 % of wave**; the harm was later recovered by the conv coalesced read |
| **`--gdn-spell fused`** | `ebeebdb` | fused prologue on the unroll4 walks | `gdn_rec` **1.0645** paired at 4,096 (3/3 reps), decode flat | **family −2.6…−6.5 %, headline unchanged**; kept, honestly booked below the ruler's noise |
| **lane-ceiling raises** (`kAttnMaxSplit` 8 → 9 → 64) | `0f5dca8`, `c188aa3` | the maximum lane count attention will plan for: 4,096 → 4,608 → **32,768** | bit-preserving by construction, proved by `tools/attn_ceiling_proof.py` + an old-vs-new binary diff at 1,024 lanes (ndiff 0 in all four pairings) | **no speed claim, deliberately** — these are long-context *enablers*. The price is scratch: `attn_part_floats` 32.25 → 258.00 MiB per scratch instance (~0.78 % of the pool), charged per scratch, not per request. |

---
## 16. What was measured and refused (so this list is not re-litigated)

Every one of these died against a **pre-registered line** or on the control inside its own experiment. The one-
line version, with the number that decided it:

| route | the claim | the number that killed it |
|---|---|---|
| **D-12 chunked delta-rule scan** | 1.96–2.99× by arithmetic | device returned `gdn_rec` 127 → ~1,750 ms: **5.9× slower than priced**; matrix-shaped work on a vector path (730.4 GFLOP executed vs 54.03 priced, no `v_mfma` in the binary) |
| **route A, the LDS-resident recurrence** | one launch per layer per body, state traffic 38.7 GB → 75 MB | **112.04 ms against ≤63 ms live / >84 ms dead**; 6.1 µs/token against `seq`'s 6.8 while issuing 36 launches instead of 18,432 — so `gdn_rec` was latency-bound on the chain, not on launches or bytes |
| **G2, the barrier-free one-wavefront GDN geometry** | llama runs `LDS = 0`, so delete barriers and LDS | **+121.9 % slower** (`wf4`): 0.20–0.40 wf/SIMD against 0.8 leaves nothing to hide the per-token chain. A wrong premise inverted the sign of a real measurement (§43) |
| **M = 16 re-tile (VALU spell)** | 4× less demanded traffic | 2.12–14.3× slower at bit-identical output; and §28 showed the register attribution was wrong too (3 → 5 blocks/CU changes time by ≤5 %), so it is **serial-round latency at low interleave**, not registers or bytes |
| **A3 (half-tile Q staging), A4-L1 (second accumulator), L-b (32 KV lanes), L-d (batched rescale)** | four attention levers | A3 died at the source on a derived LDS floor; A4-L1 measured **−1.67 %** with VGPR 178 → 200 and died; L-b is residency run backwards (9 → 6 wf/CU ⇒ +28 % by the engine's own elasticity); L-d is −1.26 % of family, below any ≥3 % line |
| **L-c, the split cap** | −4.2…−4.7 % of the 32k wave from a *traced* 208 GB/s partial round-trip | measured **−4.61 % of attention, −2.38 % of the wave: DISPLACED**. Even a traced rate × traced bytes overstates payoff by ~1.35× — the calibration point for every future traffic-shaped estimate |
| **A12, the wide shape** | +25.1 % in a rig | ~2 % in the engine, and 78 % of that is what `--attn-split-cap` already buys (§14.1) |
| **f16 P·V accumulator, fp16 activations as a headline route** | free registers / +21.5 % on the GEMM legs | the f16 atom issues at the F32 atom's rate and the change proved **register-neutral**, not the 64 VGPR the probe promised; fp16 remains admissible-by-gate but the GEMMs are already at 97–102 % of roof and hipBLASLt's tuned kernels are what put them there |
| **captured graphs / fusion-for-launch-count** | discrete-card graph-capture win | replay 1.78 µs vs 1.85 µs eager — the limit is this GPU's own dispatch rate. The exposed step gap **is** the submission bill, already hidden by pipelining |
| **A8, warming the GEMM algos inside `load_ms`** | remove ~85–110 ms of first-execution cost from every timed window | **parked, not refused** — a short-context and first-request win, and the user's call was that the remaining gap lived at long context. §50's measurement and the legs in `prototype/perprompt/a7_host_split.sh` still stand, so this is the largest known *unspent* prefill number on the board |

## 17. Three things that changed published numbers without making the engine faster

1. **Cold → warm as the headline convention (§51, user decision 2026-10-19).** Same engine: 7,986 tok/s cold
   (§48) vs 9,790 warm (§51) at 4,096 lanes. A ~23 % "improvement" that is a bracket change, not a code change.
   Cold stays recorded as the real first-request latency a server serves; the two must never be mixed across
   engines.
2. **The cold-vs-warm subtraction was mis-read first (§49 → §50).** `RESULT ms_total − wall_ms` was booked as a
   "~65–71 ms per-prompt host cost"; both operands are the *same* wave call in the same process, so it is
   **first-execution cost of ~90–110 ms, mostly hipBLASLt's first execution per algo** — with `--gemm hand` the
   gap at 512 lanes falls from +91.5 ms to +5.8 ms. Consequence: warm-vs-warm this engine is **faster than
   llama.cpp at short context** (10,637 vs 7,379 tok/s at 512), level at 4,096, and behind only as context grows.
   Related and still open: `tools/prefill_scaling_fit.py` splits the engine's cost into a constant `a` and a
   context-growing `b`; before A13 the ratios vs llama were `a` 1.285× → 1.207× and `b` 2.164× → 1.554×, and G1
   took 21.6 % out of `b` — which is why the 4,096 headline can improve while the 32k ratio gets worse in the
   same window if only `a` moves.
3. **Ruler windows.** The same build reads 64 and 256 lanes *slower* in a fresh sweep than a paired A/B in one
   hold (708 vs 770, 1,893 vs 1,971), and llama.cpp's own matched row moved 8,290 → 10,381 tok/s between two
   windows on the same build. No number above is quoted without its window, and every ratio above is paired.

## 18. The discipline that makes these numbers trustworthy

* **Fix the line before the kernel exists.** Every dead route in §16 died against a number written down first
  (route A: 63/84 ms; the tile flip: ≤40 ms; A13's rewrite pre-gate: 6.5 TFLOP/s; G4: ≤170 ms family / ≤540 ms
  wall). Nothing shipped beat its line by accident without a mechanism attached.
* **Prefer byte-identity, and never loosen a gate to pass.** Most wins moved no value at all. Where values had
  to move, the band was bought by an audit with injected bugs separating by ≥492×, not by taste — and in the
  two later cases (`reg16`, A13) the existing aggregates already admitted the reassociation without a change.
* **A paired A/B in one lock hold, both forms in one binary.** Suggested by §6's bimodality, proven necessary by
  the stale-binary false null (§13) and by §56's four legs.
* **A behavioural canary asserted before any timing.** A field the change is *required* to move, read in the same
  breath as the measurement: the launch bill (`cast=97 other=20 total=406`), the form echo (`form=wide128llama`),
  the scratch field (`scratch=720 SPILL`), the `tokens=` count read off the fixture file, `launches=` beside the
  family ms. A self-consistency check compiled into the binary under test cannot detect a stale build.
* **Both denominators, always.** Family row answers "did the targeted work get cheaper"; `wall_ms` answers "did
  the engine get faster"; only the second one ships. The externality (cost landing in a neighbouring family) is
  printed on every window since §40.
* **Correct in place, and record the correction.** The ledger's own history of retractions — the 16/16 route claim,
  the M=16 register attribution, the 14× roof framing, the §19 "structural" tile verdict, the §49 per-prompt
  constant, §51's 8×-low TFLOP/s, the A13 funding plan's preprocessor blindness — is why the surviving numbers can
  be used as engineering inputs.

## 19. Where prefill time goes now, and what is left

**Shipped defaults as of this writing** (`engine/decode_run.cpp`, `engine/decode_step.h`, mirrored in
`engine/gate.py`): `--gemm lt`, `--gemm-pins all`, `--mlp-map auto`, `--attn-tile 1`, `--attn-touch 1`,
`--attn-fattn 1`, `--attn-fattn-wide 2`, `--attn-fattn-qstage unstaged8`, `--attn-split-cap` inert,
`--gdn-scan seq`, `--gdn-spell fused`, `--gdn-loop 512`, `--gdn-geom reg16`, `kMaxWaveRows = 512`,
`kAttnMaxSplit = 64` (lane ceiling 32,768).

**Headline, warm, as shipped (§60):** 512 → **11,636**, 1,024 → **12,047**, 4,096 → **11,070**, 16,384 →
**8,542**, 32,000 → **6,154** prompt tok/s. Decode: 87.1 tok/s at 128 lanes, 71.3 at 4,032 (cold ruler, §1/§36).

**The shape of the wave, last traced before A13 (§47/§48 base):** at 4,096 lanes `gdn_rec` ~89 ms (18.7 % of the
traced wave after `reg16`), attention ~101–135 ms (~30 %), GEMM ~150 ms traced (and **ours beats llama's** at
154.19 vs 176.83 ms in §38's trace). §60 flags explicitly that **§51's phase table is now stale**: it records an
attention row (1,518.91 ms/pass at 16k) the engine no longer produces, and re-tracing it is named as separate
work, not bookkeeping.

**What is left, as the ledger currently states it:**

* **Re-trace after A13.** Every long-context claim now rests on a phase table that predates the shipping attention
  form. That is the cheapest way to find out what the new largest phase is.
* **A8 (unspent):** warming the GEMM algorithms during `load_ms` would take ~85–110 ms off the cold number at
  every width — 3× on short-context first-token latency, ~20 % at 4,096 — and must be bit-identical to be legal.
  Parked by user decision, not by measurement.
* **GDN's remaining 1.89×** against llama's traced kernel, now that we run llama's thread map at its own
  elements-per-thread: it cannot be geometry or occupancy any more.
* **The wide-form numerics headroom**: the wide-scoped pooled line sits at 2.028e-04 against a 2.500e-04 ceiling,
  so the next wide-form change has ~19 % to work in.
* **Long context is where the remaining leverage is, and it is now measured as being ahead** of llama.cpp on
  traced totals at 16k and 32k (cross-window, §59); the constant-vs-quadratic split in §17 says which lever moves
  which width.

## 20. Evidence index

| topic | where the raw record lives |
|---|---|
| the live measurement ledger, §1–§60 | `bench/performance-status.md` (each section named above by number) |
| the frozen pre-2026-10-16 ledger (P8, Q6, Q10, route (c), route A, tile flip) | `progress/2026-10-16/performance-status-2026-10-16.md` |
| route index with verdicts, kill lines and the 2026-10-14 ladder | `progress/2026-10-14/speedup-routes-2026-10-14.md` |
| the prefill decision log D1–D64 | `progress/2026-10-16/prefill-decision-log-2026-10-16.md` |
| band provenance (every threshold, its derivation and its decision) | `bench/band-history.jsonl`, `engine/gate.py` |
| device-window command lines, util pairs, raw rows | `progress/bench/evidence/` (`ledger-2026-10-16/`, `retake-2026-10-19/`, `regress-2026-10-19/`, `llamaphase-2026-10-19/`, `g4ab-2026-10-19/`, `g1ab-2026-10-19/`) |
| attention tile / slim / WMMA / wide forms | `prototype/attn_decode/results.md` §20–§21, `prototype/attn_valu_slim/results.md`, `prototype/attn_wmma/`, `prototype/attn_wide/{A12-WHY-NOT-FASTER.md,A12-ADMISSION.md,A12-FIXES.md,A13-RESULT.md,LLAMA-PORT-PLAN.md,evidence/}` |
| GDN recurrence work (ISA census, collapse, walk bets) | `prototype/gdn_rec_isa/results.md`, `prototype/gdn_collapse/results.md`, `prototype/gdn_walk/{BET.md,G3-llama-kernel.md}` |
| per-cost attention attribution (A2–A11, R2) | `prototype/attn_percost/` (`A3-result.md`, `A4-L1-verdict.md`, `A6-*-verdict.md`, `A9-LONGCTX-CENSUS.md`, `A9-LC-verdict.md`, `A11-attribution.md`, `R2-amortization-spec.md`) |
| cast/copy byte census and the handoffs behind §9 | `progress/2026-10-17/glue-cast-copy-bytes-census.md`, `HANDOFF_41a_cast-deletion.md`, `HANDOFF_42prime_embed-gather.md`, `HANDOFF_17prime_gdn-cast-deletion.md`, `HANDOFF_convread_coalesced-load.md` |
| per-prompt / cold-vs-warm legs | `prototype/perprompt/{A7-CANDIDATES.md,a7_host_split.sh,a7_scratch_bytes.py,a7_codeobj_census.py}` |
| llama.cpp as a comparison instrument | `tools/llama_phase_compare.sh`, `tools/trace_phase_split.py`, `tools/prefill_scaling_fit.py`, `engines/llama.cpp` (vendored, gitignored) |
| ceilings and device facts | `devices/strix-halo-compute.md`, `devices/strix-halo-memory-bandwidth.md` (incl. the measured size–bandwidth curve: ~32 MiB big tier at 876.7 GB/s), `devices/strix-halo-unified-memory.md`, `prototype/issue_rate_probe/results.md` |

**Commits named in this document,** in date order: `f4a1095..88fd712` (P8b), `bc61dce` (rows/block 32),
`b298e73`/`734b3d9`/`d1703b0` (`--mlp-map`), `1155d6a`…`5eac51d` + `504f93b` (route (c) + default), `cc30153`
(cascade band), `9fb12cf`/`6fd6e61` (tiled wave), `4b76f40` (`--attn-touch`), `2219215`/`2ad43f0`/`aaefe01`
(`--gemm lt`, pins, default), `a4308be`/`db58dce` (deep pins), `65699ea`/`a1cde3b` (`unroll4`), `26cd5d6` (F32
seam), `47274f4` (gather), `62b73e9` (packed cast), `5096a57` (conv read), `c564b47`/`ebeebdb` (`fused`),
`980f926`/`80930c1` (`--gdn-loop`), `5024f2e`/`eb3f4a1`/`748d790`/`9b09056`/`8a7c222`/`2ac956d` (WMMA round),
`c188aa3` (lane ceiling 32,768), `46139e9`/`e1df7a7` (`reg16`), `2808c13`/`40a19ec` (`unstaged8`), `197d041`/
`41d1cb7` (A12 inert), `99da4c9`/`ec72d94` (A13 form, then default).
