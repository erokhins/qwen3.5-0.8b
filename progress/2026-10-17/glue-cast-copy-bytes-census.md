# D30 items 1.7 / 4.1 / 4.2 — the bytes-and-launches census of `glue`, `cast`, `copy` (2026-10-17)

CPU-only. **No timing was taken, no device was touched, no gate was run.** Every `ms` below is parsed
out of a log that already exists; every byte count is re-derived from the engine's own geometry and
cross-checked against the launch counts those logs print. The arithmetic is in
`tmp/gcc_census.py` (scratch, `tmp/` is gitignored per `.gitignore:72`); its output is
`tmp/gcc_out.txt`; the script aborts if its derived launch bills do not equal the logs'.

**The cut the task asked for, in one table.** Per family: measured row, launch count, cost per launch
against the repo's measured dispatch floor, bytes the family must move, and the rate that implies.
Denominator for every `%` in this file is named in the row or in §1.

| family | ms (median of 3) | % of wave | launches | µs/launch = ×floor | GB/wave | GB/s demanded | % of the 210.5 GB/s 1:1 copy probe | floor as % of row | **bound by** |
|---|---|---|---|---|---|---|---|---|---|
| `glue` | 14.0210 | 1.46 % | 392 | 35.77 = 19.3× | 1.627 | 116.1 | 55.1 % | 5.2 % | neither — traffic-shaped, well below the wall |
| `cast` | 26.7073 | 2.77 % | 913 | 29.25 = 15.8× | 7.262 | 271.9 | **129.2 %** | 6.3 % | **bytes** |
| `copy` | 10.8579 | 1.13 % | 4240 | 2.56 = **1.4×** | 0.038 | 3.5 | 1.7 % | **72.2 %** | **launches** |
| `gdn_conv` | 19.2164 | 1.99 % | 144 | 133.45 = 72.1× | 3.659 | 190.4 | **90.5 %** | 1.4 % | **bytes** (at the wall) |
| `gdn_gates` | 1.6717 | 0.17 % | 144 | 11.61 = 6.3× | 0.024 | 14.1 | 6.7 % | 15.9 % | neither — a tiny grid, and 0.17 % of wave |
| `gdn_gated_norm` | 9.1358 | 0.95 % | 144 | 63.44 = 34.3× | 1.209 | 132.4 | 62.9 % | 2.9 % | bytes |

Consequences, before any item is opened:

- **`copy` is not a bytes item and D30 4.2 was framed as one.** The whole family moves 38.0 MB per
  wave; at the 237 GB/s read probe that is **0.16 ms**, i.e. **1.5 % of its own 10.86 ms row**. 72 %
  of the row is at or under the cost of *submitting* 4,240 operations (2.56 µs each against the
  repo's measured 1.85 µs empty-launch floor). Nothing that changes what is *copied* can matter here;
  only deleting the operations can.
- **`cast` is a bytes item and D30 4.1 was framed as one — but the bytes it can retire are the wrong
  ones.** 56.1 % of the family's bytes are stores that **nothing reads** under the shipped
  `--gemm lt --gemm-pins all` default (`engine/decode_run.cpp:106,110`). That is not a fusion, it is
  a deletion.
- **`glue` is bound by neither**, and that is why it has no prize: 5.2 % of its row is dispatch, its
  demanded rate is 55 % of the copy probe, and the only byte it writes that nobody reads is worth
  0.004 % of the wave.
- The three families together are **51.59 ms = 5.36 % of the wave**; the D30 1.7 trio
  (`gdn_conv` + `gdn_gates` + `gdn_gated_norm`) is **30.02 ms = 3.12 %** — and 2 of its 3 members are
  at or near the memory wall, which is what kills the fusion framing (§4).

---

## 1. Basis, denominators, and the instrument's own self-check

**Timing window** — `tmp/pinab/w4096_new_r{1,2,3}.txt`, the `new` side of the §9 `ab4` paired window
(`tmp/pinab/run_ab.sh`: `engine/build/decode_run --model models/Qwen3.5-0.8B --ids <ids4096> --steps 0
--time-waves --gemm lt --gemm-pins all --out tmp/pinab/o`; `--gemm-pins all` is the shipped default).
Medians of 3, as §9 takes its rows. Rep spread is quoted, because it bounds what any falsifier can see:

| row | rep 1 | rep 2 | rep 3 | peak-to-peak |
|---|---|---|---|---|
| `wall_ms` | 971.8253 | 957.5386 | 963.2839 | 1.5 % |
| `glue` | 13.8474 | 16.0021 | 14.0210 | **15.6 %** |
| `cast` | 26.2340 | 26.7073 | 26.8551 | 2.4 % |
| `copy` | 10.8155 | 10.8579 | 10.8669 | 0.5 % |
| `gdn_conv` | 18.0122 | 19.2164 | 19.3973 | 7.7 % |
| `gdn_gates` | 1.6198 | 1.6717 | 1.6785 | 3.6 % |
| `gdn_gated_norm` | 9.0424 | 9.1358 | 9.3446 | 3.3 % |
| `gdn_rec` | 426.6246 | 431.9017 | 439.3376 | 3.0 % |

**Denominators, with sources** (none measured here):

- **963.2839 ms** — one 4096-lane prefill wave at the shipped default (median of the three reps
  above). Every `% of wave` in this file divides by this. Arithmetic on it: 4096 / 0.9632839 =
  4252 prompt tok/s, a derived figure from a log, not a new ledger row.
- **237 GB/s** best read probe (`devices/strix-halo-memory-bandwidth.md`; §4's re-take read 239.3).
- **210.5 GB/s** best **1:1 copy** probe (§4 re-take). This is the right denominator for an
  elementwise read+write family like `cast` or `glue`; the read probe flatters them. `cast` at
  271.9 GB/s demanded is above both, which is the D21 statement in numbers: demanded ≠ DRAM bytes,
  L1/L2 absorb part of it.
- **219 GB/s** conservative mixed step budget (`devices/strix-halo-mixed-traffic.md`,
  `G_step_1p5g_61p7_arena2m`), used as the pessimistic rate.
- **1.85 µs** empty-kernel launch floor (`tools/launch_overhead_hip.cpp`, quoted at
  `engine/performance-status.md` §4). Graph capture replays at **1.78 µs — no gain**, which is why
  HIP graphs cannot rescue a launch-bound row and were retired as a lever
  (`engine/results.md` lever 5, struck).
- **Geometry**: `kMaxWaveRows = 512` (`engine/decode_step.h:49`), 8 bodies for 4096 tokens,
  24 layers / 18 GDN / 6 full-attention (`engine/state.h:41-64`), and one launch per `kCopy`
  including the D2D memcpys (`engine/stage_clock.cpp`'s `wave_plan_bill`).

**Self-check before any arithmetic.** The wave printed `launches=81050 plan_launches=81050
plan_match=1` in all three reps, and every family's `launches=` equals the bill re-derived from
`stage_clock.cpp`: `glue` 392 = 49/body × 8, `cast` 913 = 114/body × 8 + 1 (the final-norm cast in
`finish_token`), `copy` 4240 = 4096 embedding rows + 144 conv-window rolls, the GDN trio 144 each
(18 GDN layers × 8 bodies). `gdn_rec` is 73,728 of the 81,050 launches — **one launch per token per
GDN layer**, which is the fact that kills half of D30 1.7 (§4). `sum_ms` = 963.2122 vs `wall_ms`
963.2839: 0.0717 ms ungaged, so the family rows tile the wave and no row hides an uninstrumented gap.

**Exposure, the one thing the logs cannot pin.** `WAVEGAP` reads `exposed_ms` = 1.5229 / 21.6100 /
**−21.6824** across the three reps (median 11.5664; the third rep's production burst was *slower*
than the drain-per-family pass). So the slack a deeper queue could hide is bounded only to about
**±22 ms**, not to zero. This is the single reason a launch-count ceiling (§5) cannot be turned into a
price from these logs, and it is why D30 4.4's `exposed_ms=7.14, ratio=1.030` should not be quoted as
tight. It does not affect any *bytes* claim: the device pays for bytes in either pass.

---

## 2. The byte accounts, and what in them is redundant

### 2.1 `cast` — 7.262 GB/wave, and 56.1 % of it feeds nothing

One cast launch reads BF16 and writes F32: **6 B per element**. There are six call sites per body
(`engine/decode_step.cpp`, line numbers as of this census) plus one per priced row in `finish_token`:

| call site | launches/body | MB/launch | who reads the F32 destination under the shipped `--gemm lt` |
|---|---|---|---|
| `ds:499` `norm → norm_f32`, mixer input, **attention layers** | 6 | 3.15 | **NONE** |
| `ds:499` `norm → norm_f32`, mixer input, **GDN layers** | 18 | 3.15 | `proj_gemv(in_proj_b)` `ds:532`, `proj_gemv(in_proj_a)` `ds:535` |
| `ds:540` `cast_bf16_to_f32_3` `g_raw│g_a│g_b` | 18 | 18.97 | `gdn_conv`, `gdn_gates` |
| `ds:636` `a_gated → block_out_f32` (attention o_proj input) | 6 | 6.29 | **NONE** |
| `ds:711` `block_out → block_out_f32` (GDN out_proj input) | 18 | 6.29 | **NONE** |
| `ds:739` `norm → norm_f32`, MLP input | 24 | 3.15 | **NONE** |
| `ds:763` `mlp_act → mlp_act_f32` | 24 | 11.01 | **NONE** |
| `ds:820` `norm row → norm_f32` in `finish_token` | 1/wave | 6 KB | `lm_head_full` `ds:822` |

**Unread: 78 of 114 launches per body = 68.4 % of the launches, 4.077 of 7.262 GB/wave = 56.1 % of the
bytes.** `submit_wave_body`'s own header comment (`decode_step.cpp:439-449`) states the rule: a `project()`-routed
dot takes the **BF16** operand on the library route and the **F32 widening** on the hand route
(`decode_step.cpp:451-459`), and `mlp_gateup_gemm` / `mlp_down_gemm` likewise read `sc.norm` and
`sc.mlp_act` directly — `mlp_down_gemm`'s own comment says so ("`sc.mlp_act`, not the widening above:
the library reads a 16-bit `A` operand"). `lib_gemm` is
`wave_gemm_is_library(sc.wave_gemm, n)` (`ds:450`), `resolve_wave_gemm` returns `kHand` only for
`rows <= 1` (`engine/kernels_gemm.cpp:844-850`), and the shipped defaults are `--gemm lt`
(`decode_run.cpp:106`) with `--gemm-pins all` (`decode_run.cpp:110`). So on every 512-row body those
four destinations are written and never read. Two witnesses that this is deliberate-but-unpriced, not
a misread: `decode_step.cpp:447-449` — *"The F32 widenings stay where they are even when this route
makes one of them unread: deleting a launch is a change to the §07 launch bill (`planned_launch_counts()`,
`gate.py` section (d), the wave plan in `stage_clock.cpp`), and §7 W5 prices that separately"* — and the mechanical reader-scan of the three fields:

```
$ grep -rn "block_out_f32\|mlp_act_f32\|norm_f32" engine/*.cpp engine/*.h
```
returns only allocations, frees, the cast writes above, `project()`'s hand branch, `ds:532`/`ds:535`
(the 16-row GDN pair that keeps `norm_f32` alive for GDN layers only), `ds:760`/`ds:771` (the hand MLP
branches) and `ds:822` (`lm_head_full`, hand at any width). **`in_proj_a`/`in_proj_b` are N=16 and call
`proj_gemv` directly at every width**, which is why 18 of the 24 mixer-input casts are live and 6 are not.

The live half is `cast_3` (2.732 GB/wave) plus the 18 GDN mixer-input casts (0.566 GB/wave). It retires
only by changing a *consumer's* operand width (§4, 1.7′), not by deletion.

### 2.2 `copy` — 38.0 MB/wave in 4,240 operations, 96.6 % of which are 2 KiB each

| member | ops/wave | bytes/wave (read + written) | ms at 237 GB/s | ms at the row's own 2.56 µs/op |
|---|---|---|---|---|
| `E[y]` row `hipMemcpyAsync` per token (`ds:465-471`) | 4096 | 16.78 MB | 0.071 | **10.49** |
| conv-window roll, 18 per body (`ds:575-578`) | 144 | 21.23 MB | 0.095 | 0.37 |
| **family** | **4240** | **38.01 MB** | **0.16** | **10.86** (measured row) |

No redundancy exists in the conv-window member and the "in-place" wording cannot reach the embedding
member: `E[table[tokens[t]]] → sc.embed` is a gather, not a staging move. The embedding tile itself is
read twice and written once (`sc.embed` becomes the running residual at `ds:478`, which glue's
layer-0 norm-only launch `ds:490-491` and layer 0's first `add_norm` `ds:733` both read), so deleting
the tile outright — folding the gather into glue — retires
another 16.78 MB/wave, worth 0.071 ms at 237 GB/s: noise. **The prize is 4096 submissions, not bytes.**
`sc.conv_window_layer`'s roll cannot be made in place without breaking a stated contract: `conv_kernel`
declares `win` and `new_win` `__restrict__` and `decode_step.cpp:565-570` says the roll-into-scratch
(`ds:575-578`) is what keeps both pointers `__restrict__`, with double-buffering named as the
alternative ("1.27 MiB of state per request") and priced as a deferral. A ring would be the
"in-place staging" D30 4.2 asked for, and its ceiling is 0.37 ms.

### 2.3 `glue` — 1.627 GB/wave, and nothing in it is spare

`glue_add_norm` (prototype/glue/glue_kernels.h:102-126) holds its `PER` elements in registers and
touches each tile exactly once: reads `s` and `delta`, writes `s_out` and `n_out`. So 4 tiles per
add+norm launch (4 × 512 × 1024 × 2 B = 4.194 MB), 2 for the norm-only layer-0 launch, = 1.627 GB/wave.
Every output is consumed by the next launch (`norm` by the next GEMV or MLP, `s_out` by the next add)
with **one exception**: layer 23's residual write is never read — `finish_token` reads `sc.norm`, not
the residual — so 8.4 MB/wave is a dead store worth **0.035 ms = 0.004 % of wave**. Removing it would
cost a route-dependent special case in `add_norm` for a rounding error, and it would also change
`glue`'s `rows_total` counter that `gate.py`'s `wave_expect` polices. Not recommended; recorded because
a census that names no dead byte has not looked.

`glue`'s row is also the least readable here: 15.6 % peak-to-peak across three reps of the same config,
the §6 dispatch-bimodality signature, which is an independent reason a sub-0.1 % ceiling on this family
cannot be adjudicated on the row.

### 2.4 The D30 1.7 trio — where the round-trips actually are

| item | bytes/wave | demanded rate | first-touch (DRAM-class) share | inter-kernel round-trip a fusion could retire |
|---|---|---|---|---|
| `gdn_conv` (144) | 3.659 GB | 190.4 GB/s = 90.5 % of the copy probe | 3.624 GB (raw in + `u` out) = 15.29 ms at 237 = **79.6 % of its own row** | none — it reads `g_raw_f32`, which is a *widenings* artifact, not a neighbour's output |
| `gdn_gates` (144) | 0.024 GB | 14.1 GB/s = 6.7 % | 0.024 GB | reads `g_a_f32`/`g_b_f32`, 23.6 MB/wave total |
| `gdn_gated_norm` (144) | 1.209 GB | 132.4 GB/s = 62.9 % | 1.209 GB | `g_o`: `gdn_rec` writes 4.194 MB F32 per launch, gated_norm reads it → **1.208 GB/wave round-trip** |

The conv's 4 taps are read per (channel, token) with the reuse spread across `gridDim.y` blocks
(`conv_tap`, `prototype/gdn_decode/gdn_decode_kernels.h:122-127`) — no shared staging — so its *demanded*
total is 9.1 GB/wave (386 GB/s, impossible) and the tap re-reads are cache-served. That is why the
first-touch column is the honest one, and it says the conv is already at 80 % of the read ceiling: it is
not a kernel waiting to be fused, it is a kernel at the wall. `gdn_gates` at 512 blocks × 16 threads
(`<<<tokens, NH>>>`, `engine/kernels_gdn.cpp:32`) is 8192 threads total — its 1.67 ms is grid-shape
latency on 0.024 GB, and the whole family is 0.17 % of the wave whatever is done to it.

---

## 3. Verdicts, D34 form

Kill-band reference used for "does the ceiling clear": **D35 shipped −0.75 % of wave**, and the §9
null test on rows a change cannot touch reads +1.9 %…+3.1 %. A ceiling below ~0.75 % is not
admissible on a family row; a ceiling above ~1.5 % is readable on the row but not on the wall.

### 4.1 `cast` — **FUNDED**, and not as a fusion: it is a deletion of 624 launches that write nothing

**The prize.** 78 of the 114 cast launches per body (624 of 913 per wave, 68.4 %) write an F32
destination that no consumer reads on the shipped route: **4.077 GB/wave = 56.1 % of the family's
bytes**. Priced two ways: byte-share of the family's own row → **14.99 ms = 1.56 % of wave**; at the
1:1 copy probe rate → **19.37 ms = 2.01 % of wave**. Launch-count alone (if the bytes were free) →
1.15 ms = 0.12 %. **Ceiling 1.56–2.01 % of wave**, i.e. 2–2.7× what D35 shipped, and it clears D30
4.1's own booked "≈2.7 % if all of it vanished" without vanishing all of it.

**Why no new band is needed.** Every deleted value is a BF16 widened to F32, which is exact, and the
value is unread anyway: **the deletion cannot move one output bit.** The numerics gate is the witness,
not a new band (the standing numeric-format rule: a green gate set *is* the decision procedure).

**The cost is gate plumbing, not numerics — and it is smaller than the source implies.** Guarding the
four dead sites on the predicate their consumer already uses (`if (!lib_gemm)`) does **not** touch
`planned_launch_counts()`, because `resolve_wave_gemm` forces `kHand` at `rows <= 1`: the per-token bill
`gate.py` section (d) polices is unchanged, and a decode step's 115 casts stay exactly where they are.
What must move together in one commit, or the run fails itself:
`stage_clock.cpp:120`-style `wave_plan_bill` must learn the route (it already takes a `WaveShape` with
resolved-form flags `gdn_chunked`/`gdn_resident`, so a `gemm_library` field is in-shape), and
`gate.py`'s `launch_plan()`/`wave_expect()` (`gate.py:3484,3947`) — whose docstring asserts "49/24/24/114
launches arrive per body at any width" — must stop assuming the wave body owes 114 casts on a library
route. `plan_match` and section (h) are exact-integer checks, so a half-done change aborts rather than
passing quietly. **This is §7 W5's price, and it is now priced: 4 launch sites guarded + 2 bill
derivers taught the route, for 1.56–2.01 % of wave.**

**Second shape, for the record (4.1-b).** Make the hand kernels read BF16 (`PACK_X=true` for
`proj_gemv`; a BF16-staged `mlp_gateup`/`mlp_down`, the deferral at `engine/results.md:146,326`) so the
same casts retire on **both** routes and the bill stays route-blind. Bigger: two graduated kernels
re-gated, and the decode path also wins (48 of 115 per-token casts). But `engine/results.md:571` carries
a *measured* precedent against the consumer-side fold at decode width — `PACK_X` cost **+0.278 ms** over
48 launches to save 48 casts. That precedent does **not** transfer to the wave: at `n = 1` a cast moves
kilobytes and its row is dispatch-shaped; at `n = 512` the same call moves 3.15–11.01 MB and the family
demands 129 % of the copy probe. Worth having, but 4.1-a is the cheap half and the only half this census
can size from the source.

### 4.2 `copy` — **FUNDED NO as written**; the family's real prize is a launch-count item D30 did not name

**The refuting number for "in-place staging where legal":** the entire family's traffic is **0.16 ms at
237 GB/s against a 10.86 ms row (1.5 %)**, and the specific thing the wording reaches — the conv-window
roll — is **144 copies, 21.23 MB/wave, ceiling 0.36 ms = 0.037 % of wave**, on top of an aliasing
contract (`decode_step.cpp:565-570`) that forbids the in-place form as written. Funded no, and the
1.27 MiB/request double-buffer it would require is not free.

**The redirect (`4.2′`), which is funded:** **4096 of the 4240 ops (96.6 %) are 2 KiB `E[y]` row
memcpys**, submitted at 2.56 µs each against a 1.85 µs empty-launch floor, moving bytes worth 0.071 ms.
One gather kernel replaces them: ceiling **7.58 ms at the floor to 10.49 ms at the row's own per-op cost
= 0.79–1.09 % of wave**, plus 16.78 MB/wave if it is folded into glue's layer-0 launch (0.071 ms — not
worth the signature change, so propose the separate kernel: 8 launches + 8 tiny H2D uploads of the token
ids per wave, replacing 4096 submissions). Graph capture is *not* an alternative: replay measures
1.78 µs vs 1.85 µs eager (§4), so only deleting submissions helps. **But whether the submission cost is
on the critical path is not decidable from these logs — see §4 below.** Verdict: funded on its ceiling,
price pending one paired A/B.

### 1.7 the GDN trio — **FUNDED NO as framed** (fuse `gdn_conv` + `gdn_gates` + `gdn_gated_norm` into the rec launch, or into each other)

Three numbers close the item as written, in increasing order of how much they cost to check:

1. **Into `gdn_rec` is refuted by arithmetic before anything is measured.** `gdn_rec` is one launch per
   **token** per GDN layer — 73,728 of the wave's 81,050 launches (measured, `launches=` field). Folding
   the conv (or the gates) into it turns 144 launches into 73,728: **+73,584 submissions × 1.85 µs =
   136.1 ms of dispatch floor = 14.1 % of wave**, to retire a 19.22 ms family. The direction of the
   trade is backwards by 7× the whole wave.
2. **Into each other has nothing to retire.** The trio's own mutual round-trip is nil — `gdn_conv` and
   `gdn_gates` both read `ds:540`'s widening, not each other. Folding the gates into the conv launch
   retires 144 launches (0.27 ms) plus `g_a_f32`/`g_b_f32` (23.6 MB/wave, 0.10 ms) = **0.37 ms = 0.027 %
   of wave**, and the whole `gdn_gates` row it would consume is only 1.67 ms = 0.17 % of wave anyway
   (its cost is a 512-block × 16-thread grid, not traffic).
3. **The one non-refuted piece is the round-trip the census found, and it is asymmetric.** `gdn_rec`
   writes `g_o` F32 (4.194 MB per launch) and `gdn_gated_norm` reads it: **1.208 GB/wave = 5.10 ms at
   237 GB/s**, plus 144 launches → **5.36 ms = 0.56 % of wave**. The geometry is legal — `gdn_rec` is
   `<<<NH=16, D=128>>>` per (token, layer) (`kernels_gdn.cpp:47-49`) and `gdn_gated_norm` is one
   128-thread block per (token, head) reducing 128 elements (`gdn_gated_norm_kernels.h:110-127`), so the
   norm's epilogue fits `rec`'s existing block. It is killed by the denominator instead: that work would
   land on `gdn_rec`, **431.90 ms = 44.8 % of the wave**, the family D32/D33 established is
   latency-bound at ~1.6 working wavefronts per CU — where **1 % of `gdn_rec` = 4.32 ms**, i.e. 80 % of
   the entire prize, and D35 just demonstrated a ±3 % drift on that row between configs nothing touched.
   Risking a 1 % rec regression to bank 0.56 % is not a trade this census can recommend. Funded no.

**What survives in 1.7's neighbourhood is a dtype change, not a fusion — and it is the one item here that
is NOT decidable from the CPU.** Making `gdn_conv` and `gdn_gates` read the BF16 `g_raw`/`g_a`/`g_b`
directly retires `cast_bf16_to_f32_3` (`ds:540`, 144 launches, **2.732 GB/wave**) and halves the conv's
raw read (**1.81 GB/wave**): 4.54 GB/wave = **19.17 ms at 237 GB/s = 1.99 % of wave**, or 20.75 ms at the
conservative 219 step budget. Bit-identity is cheap (the widening is exact; the taps would convert in
place, and `gdn_conv`'s gate already has must-catch negatives for its tap geometry). Why it cannot be
settled here: the conv's *demanded* rate is already 190.4 GB/s = 90.5 % of the copy probe while its
first-touch share is only 79.6 % of its own row — the difference is cache-served traffic, so whether
deleting 4.54 GB of *demanded* traffic buys time is exactly D21's question ("demanded bytes are not DRAM
bytes … cut traffic and see which one moves time"). One measurement decides it (§5).

### `glue` — **FUNDED NO** (D30 4.3's disposition stands; its stated reason does not)

Neither bytes-bound (55.1 % of the copy probe) nor launch-bound (dispatch floor = 5.2 % of the row,
35.77 µs/launch = 19.3× the floor); the launch-count ceiling if *all* 392 launches vanished is
**0.73 ms = 0.08 % of wave**, and the only dead byte in the family is the layer-23 residual write
(8.4 MB/wave = **0.004 %**). There is no fused form left to discover — glue is already the fused form, and
its outputs are all read. **Correction:** 4.3's "dispatch-bound" is a *decode-width* fact (§4's floor row:
49 launches for one token = 0.1496 ms against a 0.0021 ms roofline = 71× roofline; `prototype/glue`'s own
burst ≈ 1.85 µs/launch, i.e. exactly the floor). At 512 rows the same family is traffic-shaped with an
occupancy/tail gap, and its row is the noisiest in the log (15.6 % p2p). Right answer, wrong mechanism —
worth a one-line correction wherever 4.3 is quoted.

---

## 4. The two questions the CPU cannot settle, and the one measurement for each

Both are the same failure mode seen from two sides: **the instrument can price bytes but cannot price
submissions**, and **demanded bytes are not DRAM bytes**.

1. **Is a per-op submission cost on the critical path?** This decides whether `4.2′`'s 7.58–10.49 ms
   (0.79–1.09 %) is real, and by extension whether any launch-count claim in D30 is real. The logs bound
   it only to ±22 ms (`WAVEGAP exposed_ms` = +1.52 / +21.61 / **−21.68**). *One measurement:* the gather
   A/B in §5-C, read as a two-sided prediction, not a hope — **`copy` must collapse (10.86 → ≲0.5 ms)
   and the burst `wall_ms` must drop ≥5 ms.** If the row collapses and the wall does not move, the
   submissions were hidden and the prize is ~0.07 ms of bytes; that outcome is itself a finding, because
   it re-prices every launch-shaped row in D30 (including `glue`'s and the 73,728 `gdn_rec` submissions).
2. **How much of the F32 seam is DRAM traffic rather than L2-served?** This decides whether 4.1's ceiling
   of 1.56–2.01 % and 1.7′'s 1.99–2.15 % are prices or ceilings. The three dead destinations are
   `block_out_f32` 4.19 MB + `norm_f32` 2.10 MB + `mlp_act_f32` 11.01 MB ≈ 17 MB, reused every layer, so
   some of their write traffic may never reach DRAM — and the family's own demanded rate (271.9 GB/s =
   129 % of the 1:1 copy probe) proves some of it is already being absorbed. *One measurement each*
   (§5-A, §5-B), and both are the D21 shape: cut traffic, then ask whether time followed.

---

## 5. Falsifiers, as commands (main agent only — timings, GPU lock, blocking, zero subagents)

No worker may run these: every one is a timing or a build-plus-timing, and AGENTS.md reserves those to
the main agent. Shared prologue:

```bash
tag="d30-1.7-4.1-4.2 census falsifier"; tools/gpu_lock.sh acquire "$tag"
trap "tools/gpu_lock.sh release '$tag'" EXIT; tools/gpu_lock.sh util
mkdir -p tmp/gcc && cp -a engine/build/decode_run engine/build/decode_run.base   # keep the ruler
./engine/build.sh                                                                # hipcc -O3 -Wl,-rpath,/opt/rocm/lib
IDS4096=progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt
IDS1024=progress/bench/evidence/prefill-occupancy-2026-10-16/ids1024.txt
wave() { engine/build/decode_run --model models/Qwen3.5-0.8B --ids "$(cat "$1")" \
           --steps 0 --time-waves --gemm lt --gemm-pins all --out tmp/gcc/o; }
```

**A — 4.1-a, guard the four unread cast sites (`ds:636,711,739,763`) on `lib_gemm`, plus the two bill
derivers.** Cheapest real falsifier in this census: it is a guard, not a kernel.
1. *CPU pre-flight, no device:* `grep -rn "block_out_f32\|mlp_act_f32" engine/*.cpp engine/*.h` must show
   no reader outside the hand branches — if any appears, stop, the dead-set is wrong.
2. *Device, one paired A/B at each width, medians of 3, both configs inside one lock hold* (`decode_run`
   built before/after the guard): the prediction is `launches=913` → **289** on the `cast` row,
   `plan_match=1`, and `cast` **26.71 → ≈11.7 ms** (a 56 % drop, ~24× the row's 2.4 % rep spread). Then
   the wall question: **`wall_ms` must drop ≥10 ms**; a row collapse with a still wall is D21 again and
   must be recorded as a null, not a win.
3. *Numerics admission:* the nine canonical fixtures × both routes, unchanged bands —
   `for f in dec8 dec64f dec64t dec300f mt32f p128 wide520f chunk64b wide1024f; do .venv/bin/python engine/gate.py --fixture $f; done`
   and the same with `--gemm hand`. `--gemm hand` is the control that must be **bit-identical** to
   today (the guard cannot fire at `rows <= 1`), and `wide520f`/`wide1024f`/`chunk64b`/`dec300f` are the
   fixtures that exercise the wide-body route the guard touches. Also required: `LAUNCH` section (d)
   unchanged, section (h) `wave_expect` green after it learns the route.
4. **Kill:** if `cast` does not drop below ~20 ms, or the gate cannot be made green without touching a
   band, stop and report — do not widen a band for a deletion that claims to move no bits.

**B — 1.7′ (the surviving GDN piece): make `gdn_conv` and `gdn_gates` read BF16 `g_raw`/`g_a`/`g_b` and
delete `cast_bf16_to_f32_3`.** Gate first (`prototype/gdn_decode`'s conv negatives, `gdn_gates`' own
gate), then the paired A/B above at 4096 and 1024. Prediction: `cast` −26.71 → −13.9 ms *on top of* A,
`gdn_conv` **19.22 → ≈15 ms** (its first-touch share is 15.29 ms). **Refuting result:** both rows' bytes
cut by 4.54 GB/wave and neither row moves ⇒ the F32 seam was cache-served, the item is worth its launch
count only, and this is the outcome that also caps A.

**C — 4.2′ (the gather).** Replace the `ds:465-471` loop with one gather kernel (ids uploaded once per
body), paired A/B at both widths. Prediction: `copy` **10.86 → ≈0.4 ms**, `launches` 4240 → 152 + 8,
`wall_ms` −5…−10 ms. **Refuting result:** row collapses, wall does not ⇒ submissions are hidden, and
record that as the re-pricing of every launch-shaped D30 row (§4.1).
**C′ — if a number is wanted before any kernel:** one burst of 4096 `hipMemcpyAsync` D2D of 2 KiB into a
2 MiB destination, timed as one wall, `/4096`, in the shape of `tools/launch_overhead_hip.cpp`. If it
reads ≈1.9–2.6 µs, the submission story holds and C is worth its ceiling; if it reads ≤0.5 µs the `copy`
row is *not* submission cost and C is dead. That probe is 10 lines and should be run before C.

**D — 1.7a (gated_norm into `rec`) — do not build it.** If it is ever revisited, the precondition is a
`gdn_rec` tolerance measurement, not a conv experiment: 1 % of `gdn_rec` is 4.32 ms against a 5.36 ms
prize, so the item's own error budget is 1.24 %.

---

## 6. Corrections this census forces on the record

1. **D30 4.3's reason.** "`glue` … dispatch-bound and bimodal" is true at decode width and false at the
   wave width this log prices: dispatch floor is 5.2 % of the row and the family demands 116.1 GB/s.
   The disposition (CLOSED for value) is right; the ceiling to quote for it is **0.73 ms = 0.08 % of wave**
   (all 392 launches retired), not "well under the noise floor".
2. **D30 4.1's shape.** "fuse `cast` work into the producing kernels … ceiling ~2.7 % if all of it
   vanished" — 56.1 % of the family's bytes need no fusion at all, they are stores nobody reads on the
   shipped route. The producer-consumer question is only about the remaining 43.9 %.
3. **D30 1.7's framing.** "fuse … into the rec launch" is the one direction arithmetic forbids (it turns
   144 launches into 73,728). The trio's only real round-trip is `rec → gated_norm`'s `g_o`
   (1.208 GB/wave), and the trio's real neighbour prize is the dtype seam at `ds:540`.
4. **`engine/kernels_cast.cpp`'s header count.** It still describes the per-token bill ("48 of 115
   casts"); the wave bill is **114 per body / 913 per wave**, of which 78 per body are unread. The file
   is a launcher comment, so this is a doc fix, not a code fix.
5. **A new fact worth its own line in the ledger:** under the shipped `--gemm lt --gemm-pins all` default,
   **68.4 % of the engine's cast launches and 96.6 % of its copy operations are shape artifacts of a
   route that no longer reads them**. D30's family rows were right to be suspicious of themselves; two of
   them were wrong for a reason nobody had named.

## 7. What I did not do, and residual risks

- **No timing, no device, no gate, no build** was run here. Every `ms` is parsed from
  `tmp/pinab/w4096_new_r{1,2,3}.txt`; the derived launch bills were asserted equal to those logs'
  `launches=` fields (6/6 match) and `plan_match=1` was asserted in all three, so the census cannot be
  reading a geometry the engine did not run.
- **Byte counts are first-touch unless stated.** `gdn_conv`'s tap re-reads (5.436 GB/wave demanded,
  cache-served) and `cast`'s 129 %-of-probe demand mean the *demanded* totals in §2 overstate DRAM
  traffic. Every ceiling that uses a rate says which rate; every ceiling that uses a byte-share of a row
  is self-limiting. This is §4's open question, not an error, but it is why no verdict here should be
  read as a predicted `ms` saving.
- **Byte account assumptions, stated:** glue touches each tile once (verified in
  `glue_kernels.h:102-126`, values held in registers); the conv reads its 4 taps per output without
  shared staging (verified in `gdn_decode_kernels.h:122-127`); a memcpy costs read + write; `A_log`'s and
  `dt_bias`'s contributions (8 KB + 64 B per launch) were left in the trio's totals but are < 0.05 % of
  them; the `--prefill-logits all` path would add one tiny cast and one `lm_head` launch per priced row,
  which the census's `--steps 0` basis does not include.
- **The unread-cast set is derived, not hand-listed**, from "who reads this destination" at every call
  site (`grep` output in §2.1), and it is **geometry-dependent**: it holds for `rows > 1` on a library
  route with all shapes eligible. A narrower `--gemm-pins`, a `--gemm hand` run, or a partial final body
  (4097 tokens, say) moves the set, so any admission must re-derive it rather than paste the count — and
  the 16-row `in_proj_a`/`in_proj_b` carve-out is exactly the kind of exception that makes a hand-listed
  must-not-catch set rot.
- **Not examined:** `proj`, `mlp_gateup`, `mlp_down`, `attn`, `lm_head`, `gdn_rec` (D30 1.x/2.x/3.x
  territory), and the `copy` of `logits` D2H (host-side, `kHostTail`, 0.84 ms row). `gdn_gates`'s
  8192-thread grid is noted as a latency artifact but is 0.17 % of the wave and was not given a fix.
- **If the review gate disagrees with one thing, let it be this:** 4.1's ceiling is quoted as a byte-share
  of a row that already runs at 129 % of the 1:1 copy probe. If the reviewer wants a *lower* ceiling, the
  number to argue is how much of 4.077 GB/wave is L2-absorbed — and the repo has no measurement of that,
  which is precisely why §5-A exists and why 4.1 is "funded" rather than "shipped".

## Appendix — reproducing the arithmetic

```
$ python3 tmp/gcc_census.py            # tmp/gcc_out.txt holds the full printed tables
```
It re-derives every number above from the geometry constants, the two probe rates, and the three logs,
and aborts on a launch-bill mismatch or a `plan_match != 1`. Sibling tool:
`tools/instep_family_budget.py` does the same comparison for one decode token and deliberately excludes
the cast family; the two byte tables price different geometries and are not merged.

---

## 8. Main-agent addendum: §5-C′ was run, and it settles the question the census could not

The census ended honestly: *"whether the submission cost is on the critical path is not decidable from
these logs."* §5-C′ is 10 lines and a minute of device, so it was run before any gather kernel was
written — `tools/submission_probe.cpp`, built `hipcc -O3 -Wl,-rpath,/opt/rocm/lib`, one lock hold, `util`
46 % at acquire, 5 reps, blocking, zero subagents. Medians of the 5, all five reps within 6.5 %:

| config (4096 ops of the engine's own shape) | ms | µs/op |
|---|---|---|
| 4096 separate 2 KiB D2D copies, one sync at the end | **8.40** (8.35–8.89) | 2.05 |
| 4096 empty kernel launches | **7.04** (7.03–7.12) | **1.72** |
| 4096 tiny kernels doing the same row copies | 8.50 (8.49–10.00) | 2.07 |
| **the same 8 MiB as one copy** | **0.104** (0.036–0.169) | — |
| **one kernel doing all 4096 rows** | **0.102** (0.043–0.169) | — |

**The submission story holds, and the flat of the two lower rows is what proves it.** A real copy costs
**2.05 µs** against an empty launch's **1.72 µs**, so the *bytes* in a 2 KiB row copy are ~0.33 µs and the
*submission* is 1.72 µs — 84 % of the op. Doing the identical work as one operation costs **0.10 ms
instead of 8.40 ms**. The repo's own dispatch floor is confirmed independently at 1.72 µs/op (it had been
carried as 1.85 µs), which is also the first figure this box has given me that is not bimodal.

Two consequences, stated at their width and not wider:

1. **D30 4.2′ is funded at ~8.3 ms of wave ≈ 0.87 %** of the 956 ms wave — inside the census's own
   predicted 0.79–1.09 % band, which is the first D30 item whose ceiling a pre-measurement confirmed
   rather than being discovered by the build. What it does *not* settle is the engine half: here the CPU
   had nothing else to do, so its submit rate was the binding rate, whereas in the wave other work is
   in flight and may absorb the submissions. §5-C (the paired A/B on the real gather kernel) is still the
   gate, and its prediction is now numeric in both directions — row `copy` 10.86 → ≲0.5 ms **and** wall
   ≥5 ms lower, or the record says submissions are hidden on a busy stream and every launch-shaped row in
   D30 must be re-read, `glue` included.
2. **`cast` is not submission-shaped**, and this probe shows it by contrast: 913 cast launches cost
   29.25 µs each, **17× the floor**, so their cost is on the device and 4.1 is a bytes item as booked.
   The same ratio is why `gdn_rec` at 140 µs/launch is nowhere near launch-bound and 4.2′ says nothing
   about it.

**And §5-A's CPU pre-flight passed at the source, not by grep alone.** `decode_step.cpp:451-459` is
`project(weight, x_bf16, x_f32, ...)`, and it picks `x_bf16` under `lib_gemm` and `x_f32` otherwise; the
MLP is spelled the same way at 763-772, with the engine's own comment at 765 saying *"the library reads a
16-bit `A` operand"* while the cast two lines above widens to F32 unconditionally. So the widening is not
merely unread on the library route — the source already knows it and does it anyway. That is the whole of
4.1-a, and it is a guard rather than a kernel.

---

**Dated correction (2026-10-17, later same day).** The line-form cites in this document are *as of* the revision it was written against. `62b73e9` and the docs cascade that followed retired the 424/115 launch bill (cast 97, total 406) and re-pointed the projection call-site cites in `prototype/gemm_algo_sweep/` from the 51x/72x block to 583-605/810, and `specification/ASSUMPTIONS.md`'s A-32 gate cite from 3692 to 3765. Nothing here was rewritten to match - the record of what was true when this was measured is the point of it.
