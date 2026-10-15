# attn_decode — §05 gated full-attention decode (k = 1)

Scope: post-projection inputs (`q_proj` 4096, `k_proj`/`v_proj` 512) + the request's KV cache +
the position → `attn_out_pre_gate` and the gated operand `o_proj` consumes. §05.1 de-interleave,
§05.2 zero-centred per-head RMSNorm, §05.3 partial rope, §05.7 cache append, §05.5 GQA attention,
§05.6 output gate. `o_proj` is `prototype/proj_gemv`'s; the de-interleaved gate feeding §05.6 was
pinned there (§05.1) and is re-proved here as a stage.

**Green: 1764 stage comparisons, 0 failures** — 9 stages × (48 decode cases + 20 long-cache
cases). `.venv/bin/python prototype/attn_decode/run.py --long --negative` reproduces it, plus 12
negative injections each rejected exactly where declared. `speccheck/run_all.py` still 26/26.
Band `|ref|/max ≤ 2e-4`, sized below (not borrowed).

## 1. The rounding policy, and the one line of §05.3 the spec does not say

Measured against the pins, not chosen:

| boundary | what reproduces the pins | bit-exact? |
|---|---|---|
| §05.1 de-interleave | per-head `[q 256 | gate 256]`, gate = `qg[h*512+256:]` | yes, 2048/2048 per case |
| §05.2 norm | `bf16((x·rsqrt(mean(x²)+eps))·(1+w))`, one round | yes |
| §05.3 rope | **cos/sin rounded to BF16 first**, then `bf16(bf16(z·c) + bf16(rot·s))` | 47/48 decode cases, see §4 |
| scores | `bf16(q·k) · 1/16` — scores live in BF16 (§10.3) | n/a, see the note |
| softmax | F32 exp/sum over the BF16 scores, **weights rounded back to BF16** | yes |
| P·V | F32 accumulate over BF16, one round | yes |
| §05.6 gate | `a · bf16(sigmoid(gate))`, plain sigmoid | yes |

**The rope discovery.** §05.3 says "rotate the first `d_rot` dims"; it does not say the cos/sin
table is rounded to the activation dtype before the multiply. The pins say it is. Rounding them in
F32 instead — which is exactly what `speccheck/specimpl.py::apply_rope` does, because `rot*c`
promotes to F32 — misses **52–97 of every 2048 `attn.q_rope` elements** (measured per case in
`run.py`'s verify section, which requires that misreading to fail). With the BF16 policy the
kernel reproduces `attn.q_rope` and `attn.k_rope` bit-for-bit everywhere except the one position in
§4. **Action for speccheck, not for this kernel:** `apply_rope` should cast `cos`/`sin` to the
activation dtype; the pins are the evidence and this gate now refuses the other policy.

Score rounding is *not* pinned by this data: keeping the scores in F32 instead of BF16 reproduces
`attn_out_pre_gate` at ndiff 0 too (48/48 cases, checked in `reference.verify`). §10.3's "scores
live in BF16" is therefore a policy choice the fixtures cannot distinguish at 18–25 lanes, and the
kernel keeps the reference's policy rather than the cheaper one.

## 2. Stage gate (dec8, 6 attention layers × 8 decode steps; band 2e-4)

| stage | vs pin | elements compared | worst ndiff on one case | raw max ULP | large elements >1 ULP |
|---|---|---|---|---|---|
| `q_rope` | bit-exact except §4 | 98 304 | 1 / 2048 | 1 | 0 |
| `k_rope` | bit-exact | 24 576 | 0 | 0 | 0 |
| `v_append` (§05.7) | bit-exact | 24 576 | 0 | 0 | 0 |
| `cache_keys` (read + append) | bit-exact | 528 384 | 0 | 0 | 0 |
| `cache_values` | bit-exact | 528 384 | 0 | 0 | 0 |
| `gate_vec` (§05.1) | bit-exact | 98 304 | 0 | 0 | 0 |
| `rope_cs` (device table) | bit-exact except §4 | 6 144 | 2 / 128 | 1 | 0 |
| `attn_out_pre_gate` | bit-exact on all 48 | 98 304 | 0 | 0 | 0 |
| `gated` (§05.6, `o_proj`'s input) | bit-exact on all 48 | 98 304 | 0 | 0 | 0 |

`cache_keys`/`cache_values` are compared over lanes `[0, lvalid)` of a buffer allocated `cap ≥
lvalid` lanes wide; the slack lanes hold `1.0` poison on purpose, because a causal-mask bug that
reads zeros would merely look small (that is what makes `readpast` observable). The pinned cache
dump is post-append, so `reference.py` cuts the appended lane back off and hands the kernel a
buffer with that lane free — and `verify()` requires the pin's own last lane to equal the pinned
`attn.k_rope`/`attn.v`, which is what proves `pos = lvalid − 1` instead of assuming it (§10.5's
last-writer-wins trap). 48 cases × 9 such claims, all green.

## 3. The softmax form: the engine has to make a two-pass row (the real finding)

The pinned form is one softmax over the row with the weights rounded back to BF16 (HF casts the
softmax output to the activation dtype before P·V). The engine form §07.3 describes — reduce in
blocks with running `(max,sum)`, divide at the end — **cannot reproduce it at any tile size**,
because it would have to round a weight it has not divided yet:

| form | elements differ vs the pin | of them large (>1 ULP) | cos | meanrel |
|---|---|---|---|---|
| two-pass (`--softmax full`) | 0 / 98 304 | 0 | 1.00000000 | 0 |
| streaming (max,sum) (`--online`) | 25 745 / 98 304 (26.2 %) | 6 098 (6.2 %) | ≥ 0.99999134 | ≤ 1.8e-3 |
| streaming, row split 4 ways | 25 744 / 98 304 (26.2 %) | 6 098 | ≥ 0.99999134 | ≤ 1.8e-3 |

cos/meanrel are fine in the streaming row — a whole token would still sample well — but token-level
match is not the gate (§A-18) and 1-ULP drift on 6 % of elements is exactly how the chaos floor
gets reached early. Two consequences, neither of which is this kernel's call to make:

- §07.3's online form should carry the same note §05.5's "scores in BF16" carries: it is a policy,
  and the two-pass form is the one the reference (and therefore the dumps) agrees with. If W5 wants
  streaming for long caches, the cheapest way to stay bit-compatible is to round the *tile* weights
  to BF16 anyway and accept the residual, or to keep two-pass below ~1k lanes.
- `--softmax full` is a single block per query row by construction (the denominator is the whole
  row), so the kernel refuses `--split > 1` under it rather than silently normalising per block.

## 4. Long caches: 1 → 277 lanes, gated

The decode fixtures only reach 25 lanes, so the read side would otherwise be untested where it
matters. Each **row `p` of a prefill dump is a decode-shaped query at position `p` attending lanes
`0..p` of that same pinned cache**, which gives long-cache cases with real references. `run.py
--long` runs rows `0,1,7,8,15,16,31,32,33,63,64,65,127,128,129,191,255,256,257,276` of `p300`
(277-token prefill), layer 3:

- `cache_keys`/`cache_values`/`v_append`/`gate_vec`: bit-exact at every length, 1…277 (141 824
  elements at the longest row).
- `attn_out_pre_gate` and `gated`: **0 ndiff at 18 of 20 lengths**; 2/2048 at `p=191` (1 ULP),
  1/2048 at `p=255` (1 ULP), and `gated` at `p=191` has one element at 2 ULP whose
  `|ref|/max = 1.5e-4` — the element that sets the band.
- The kernel's appended key at `p` equals the prefill's own roped key lane bit-for-bit at every
  length, i.e. §05.7 + §05.3 agree with the prefill path up to 277 positions.
- `rope_cs` differs from torch's table in **2 of 128 entries at position 64** and 0 of 128 at the
  other 67 positions sampled; that moves 1 element of `q_rope` by 1 ULP at `p=64`. **Corrected
  2026-10-11 after `speccheck/rope_table_probe.py`: this is not a floor and not a
  reference-generation boundary.** That probe asks which producer's F32 table reproduces the pin's
  `attn.q_rope` rope dims bit-for-bit, and the answer is torch F32 and a plain glibc `cosf`/`sinf`
  at **0 wrong elements over 277 positions × 6 attention layers — including position 64**. The
  device's own table is the outlier (1 wrong element at `p=64` in layers 3/7, 2 in layer 11). So
  the pins are reachable; our device table is not what produced them. Worse, nudging *any single*
  frequency index's cos or sin by up to 2 F32 ULP does **not** reproduce the pin from torch's table
  either, so the residual is not a 1-ULP transcendental disagreement: the surviving candidate is
  the kernel's own pre-rope reduction — the F32 mean in §05.2's RMSNorm, summed warp-tree here and
  by torch's reduction in the pins. Naming it properly needs `z` (post-norm, pre-rope) dumped from
  the kernel and diffed against torch at that position; that is written up as item **(g)** of the frozen
  ledger's §5, the quiescent device pass (progress/2026-10-16/performance-status-2026-10-16.md:203), because it needs a device run and
  neither this gate nor the probe can produce it. Until then `q_rope` at `p=64` is **1 element off for
  an unattributed reason**, not an accepted floor.
* **Closed 2026-10-12: the `p=64` residual is the rope table's `cos[3]`, not the norm's reduction.**
  `prototype/attn_decode/prerope_probe.py` settles it from both ends; no kernel, band or gate moved.
  It did not need the `z` dump §5(g) proposed, because `z` is already visible in the bits the gate
  writes: §05.3 passes the dims past `ROT_DIM` straight through (`q_rope[d] == z[d]` for d ≥ 64), so a
  difference in the norm's reduction must show in those 192 pass-through dims, while a rope-table
  difference cannot touch them. Reading the gate's own stage dumps (`prerope_probe.py --device-root
  /tmp/attn_decode/out_long`, one `run.py --long` per layer, since `run.py` builds long cases for the
  first layer asked for rather than all of them):

      L3   q_rope vs pin 1 wrong (rope dims 1, pass-through 0) | table 2 of 128 differ | CPU z + device table vs device q_rope: 0 wrong
      L7   1 wrong (rope 1, pass-through 0)     L11  2 wrong (rope 2, pass-through 0)
      L15  0 wrong                              L19  3 wrong (rope 3, pass-through 0)
      L23  2 wrong (rope 2, pass-through 0)
      cos[3]: device 0.0140380859, torch 0.0140991211, -1 BF16 step   (identical in its duplicate cos[35])

  The pass-through dims are clean at all six layers, so the device's post-norm `z` is **bit-identical
  to the pin's** — the reduction order is exonerated by the device's own bits. And the CPU path handed
  the device's measured table reproduces the device's `q_rope` **bit-for-bit, 0 of 2048 wrong at every
  layer**: one frequency's cosine, one BF16 step low, is the whole residual.

  The CPU sweep rules the reduction out independently. Re-permuting the reference's own sum of squares
  on the pinned rows — including `block_sum`'s exact warp tree, the order the kernel really uses —
  moves **0 elements at p=64 in all six layers**, though legal orders do move 20–207 elements at
  19–184 *other* positions, so reassociation carries a real but scattered 1-ULP floor that the device
  did not exhibit here. One F32 ULP on `r = rsqrtf(mean+eps)` moves 48 elements at 44 scattered
  positions and again nothing at p=64. And of the 128 single-frequency, one-BF16-ULP table
  perturbations at p=64, **exactly one** reproduces the measured signature (`cos[3] −1 ULP` →
  {L03:1, L07:1, L11:2, L15:0, L19:3, L23:2}, rope dims only). **The function, not the reduction.**

  Two corrections to this section. The signature is **1/1/2/0/3/2 across the six attention layers**,
  not "1 element, 2 at layer 11": the earlier note had only the layers the gate had happened to run,
  and layer 19 is worse than layer 11. And the F3a sentence *"nudging any single frequency index's cos
  or sin by up to 2 F32 ULP does not reproduce the pin"* could not have worked — the pins round the
  table to BF16 before it multiplies anything, so an F32-domain nudge of 1–2 ULP is invisible and that
  search was vacuous; the nudge belongs after the BF16 round (`prerope_probe.py::bf16_ulp`). So
  `p=64` is the device's `cosf` disagreeing with glibc's at one angle by one BF16 step: a device-libm
  boundary in the class `EXACT_STAGES` deliberately bands and gate-audit G5 records, now with a name
  instead of "unattributed". Reproduce with
  `.venv/bin/python prototype/attn_decode/prerope_probe.py` (CPU, ~1.5 s) and
  `... --device-root /tmp/attn_decode/out_long --layers 3,7,11,15,19,23` after a `--long` run.
* **Closed 2026-10-12: the `p=64` residual is the rope table's `cos[3]`, not the norm's reduction.**
  `prototype/attn_decode/prerope_probe.py` settles it from both ends; no kernel, band or gate moved.
  It did not need the `z` dump §5(g) proposed, because `z` is already visible in the bits the gate
  writes: §05.3 passes the dims past `ROT_DIM` straight through (`q_rope[d] == z[d]` for d ≥ 64), so a
  difference in the norm's reduction must show up in those 192 pass-through dims, while a rope-table
  difference cannot touch them. Reading the gate's own stage dumps
  (`prerope_probe.py --device-root /tmp/attn_decode/out_long`, one `--long` run per layer — `run.py`
  takes long cases at the first layer asked for, not all of them):

      L3   q_rope vs pin: 1 wrong (rope dims 1, pass-through 0) | table: 2 of 128 differ | CPU z + device table vs device q_rope: 0 wrong
      L7   ... 1 wrong (rope 1, pass-through 0)      L11  ... 2 wrong (rope 2, pass-through 0)
      L15  ... 0 wrong                               L19  ... 3 wrong (rope 3, pass-through 0)
      L23  ... 2 wrong (rope 2, pass-through 0)
      cos[3]: device 0.0140380859, torch 0.0140991211, -1 BF16 steps   (same in its duplicate cos[35])

  The pass-through dims are clean at all six layers, so the device's post-norm `z` is **bit-identical
  to the pin's** — the reduction order is exonerated by the device's own bits. And the CPU path handed
  the device's measured table reproduces the device's `q_rope` **bit-for-bit, 0 of 2048 wrong at every
  layer**: one frequency's cosine, one BF16 step low, is the entire residual.

  The CPU sweep rules the reduction out independently of that. Re-permuting the reference's own sum of
  squares on the pinned rows — including `block_sum`'s exact warp tree, which is what the kernel
  computes — moves **0 elements at p=64 in all six layers**, though legal orders do move 20–207
  elements at 19–184 *other* positions, so reassociation has a real but scattered 1-ULP floor that the
  device did not exhibit here; one F32 ULP on `r = rsqrtf(mean+eps)` moves 48 elements at 44 scattered
  positions and again nothing at p=64; and of the 128 single-frequency one-BF16-ULP table
  perturbations at p=64, **exactly one** reproduces the measured signature (`cos[3] −1 ULP` →
  {L03:1, L07:1, L11:2, L15:0, L19:3, L23:2}, rope dims only). So: **the function, not the reduction.**

  Two corrections to this section. The signature is **1/1/2/0/3/2 over all six attention layers**, not
  "1 element, 2 at layer 11" — the earlier note only had the layers the gate had happened to run, and
  §5(g)'s "(2 at layer 11)" understates layer 19. And the F3a sentence *"nudging any single frequency
  index's cos or sin by up to 2 F32 ULP does not reproduce the pin"* could not have worked: the pins
  round the table to BF16 before it multiplies anything, so an F32-domain nudge of 1–2 ULP is invisible
  and the search was vacuous; the nudge has to be applied to the BF16 value (`prerope_probe.py`'s
  `bf16_ulp`). What `p=64` actually is: the device's `cosf` disagreeing with glibc's at one angle, one
  BF16 step — a device-libm boundary in the class `EXACT_STAGES` deliberately bands and gate-audit G5
  records, now with a name instead of "unattributed".

* **Do not "fix" the rope table by computing it in double precision.** The dumps carry F32 libm
  results, not correctly-rounded ones: evaluating the same angle in F64 and rounding down to F32
  misses 3 of 277 positions (`p = 180, 217, 241`, 2–11 elements each) that the plain F32 path
  reproduces exactly (`speccheck/rope_table_probe.py`, run per layer). Higher precision moves the
  table *away* from the pins. Any table produced here must be F32 `cosf`/`sinf` over the F32 angle
  `p · theta^(-2i/64)`, exactly as §05.3 writes it.

## 5. Floors (`floor_probe.py`), each with its sample size

| floor | what was reordered / re-evaluated | sample | result |
|---|---|---|---|
| F1 | score dot: kernel's warp tree (8 dims/lane + 5 shuffle steps) and a pairwise 256-sum, vs torch's matmul order | 15 592 scores → 108 544 output elements | **0 scores flip, 0 elements differ** |
| F2 | softmax *policy*: streaming (max,sum) vs the pinned two-pass | 108 544 elements | 25.9 % differ, 4 491 large — a policy, excluded from the band |
| F2b | softmax *order*: denominator summed forwards vs reversed, same policy | 108 544 elements | 2 differ (0.0018 %), **0 large**, band floor 0 |
| F3 | rope transcendentals re-evaluated f64→f32 vs f32 | 416 cos values | 0 flip |
| F3b | the device's own rope table vs torch's, read back from run.py's outputs | 116 positions, 14 848 entries | 2 differ (0.0135 %), all 1 ULP — but see §4: the *pin* agrees with torch, so this row is our device disagreeing, not the reference |
| F4 | §05.6 sigmoid: torch's libm vs `1/(1+exp(-x))` in f64→f32 | 108 544 gate values → 108 544 outputs | 0 flip, band floor 0 |

Reordering this op's reference arithmetic **needs no band at all** — the products are BF16-exact and
the sums are 256 terms at magnitudes where F32 reassociation stays far inside a BF16 step. That is
a different floor from a 2048-way projection dot, and `proj_gemv`'s 3e-5 would have been a guess.
What actually moves an element past 1 ULP is the device transcendentals (F3/F3b class): the worst
offender anywhere in run.py's 3.6 M elements sits at `|ref|/max = 1.5e-4`, printed by run.py as
"tightest band these results would pass at". **Band = 2e-4** (1.4× margin). Stages that are pure
index math and copies (`cache_keys`, `cache_values`, `v_append`, `gate_vec`) are held to
bit-exactness instead, and are.

## 6. Negatives (12, each a bit-for-bit diff against a clean run under identical knobs)

Comparing an injection against the *pins* is what made the first version of this table vacuous: the
streaming-vs-pinned policy gap then showed up as "caught" for every injection. Comparing against a
clean run at the same knobs localises instead.

| `--neg` | bug | form | caught | must-not, all clean |
|---|---|---|---|---|
| `qswap` | q_proj per-head halves swapped (§05.1) | pinned | `q_rope`, `gate_vec`, `attn_pre_gate`, `gated` | `k_rope`, `v_append`, `cache_*`, `rope_cs` |
| `gatehalf` | gate read as the contiguous half (§05.1) | pinned | `gate_vec`, `gated` | everything else incl. `attn_pre_gate` |
| `ropefull` | rope over all 256 dims (§05.3) | pinned | `q_rope`, `k_rope`, `cache_keys`, `attn_pre_gate`, `gated` | `v_append`, `cache_values`, `gate_vec`, `rope_cs` |
| `ropenone` | no rope | pinned | same five | same four |
| `gqaoff` | κ(i) off by one (§05.5) | pinned | `attn_pre_gate`, `gated` | all stage outputs (the append keeps the honest κ) |
| `nomax` | running max never rescaled (A-17) | online | `attn_pre_gate`, `gated` | everything else |
| `carry` | row seeded from the previous query's state (§07.3) | online | `attn_pre_gate`, `gated` | everything else |
| `readpast` | attend to `cap` lanes — causal mask gone | pinned | `attn_pre_gate`, `gated` | everything else |
| `dropcurrent` | attend to lanes < `pos` — appended token forgotten | pinned | `attn_pre_gate`, `gated` | everything else |
| `dimhalf` | score dot accumulates 128 of 256 dims | pinned | `attn_pre_gate`, `gated` | everything else |
| `slotk` | key appended to the wrong slot (§05.7) | pinned | `cache_keys` **only** | `attn_pre_gate`, `gated`, `cache_values`, … |
| `slotv` | value appended to the wrong slot (§05.7) | pinned | `cache_values`, `v_append` **only** | `attn_pre_gate`, `gated`, `cache_keys`, … |

`nomax`/`carry` are injected under the streaming form because that is the only form that has a
running max; injecting them under the two-pass form and declaring a catch would have been a
vacuous test. `slotk`/`slotv` are declared against the cache alone **on purpose**: the core reads
the lane it appends from shared memory rather than round-tripping it through the cache, so a wrong
slot is invisible to the current query and first shows up in the *next* token's read. `run.py`
cannot express that (each pinned case rebuilds its own cache), so the limitation is stated here
instead of being papered over with a must-catch that does not hold. Closing it needs a two-token
case with a pinned second token — W5's multi-token harness, not this gate.

## 7. Cross-checks between the knobs

- **fused vs staged launch: bit-identical on every stage** (the fused core recomputes its own
  §05.2/§05.3 and appends inline; same bits, one fewer launch). Any difference would mean one of
  the two is wrong, so it is a hard failure in `run.py`.
- `--split 4` under the streaming form: same numbers as the single-block row (3rd row above) — but
  since P7 the *criteria* differ by route, not by knob: §10 records what the streaming route is held
  to and why, and `--split 1` under `--softmax online` is judged the same way.
- `--score-f32` (skip the BF16 score rounding): reported by `reference.verify` as indistinguishable.

## 8. For W5 — launch shape

At these cache lengths the whole attention layer moves ~0.6 MB per layer (512 B × 2 tensors × 2 KV
heads × lanes, plus ~14 KB of stage writes), which at 237 GB/s is **~2.5 µs of memory time** for
277 lanes and ~0.2 µs at 18 lanes. The measured launch floor on this box is ~1.85 µs
(`prototype/README.md`), so:

- **One launch per attention layer is the right shape.** The fused plan (`--launch fused`) does
  §05.1 de-interleave + §05.2 norm + §05.3 rope + §05.7 append + §05.5 attention + §05.6 gate for
  one head per block, 8 blocks, and is bit-identical to the staged plan. The staged plan needs 2
  launches (prep + core) and at 3 launches the split merge would add a third — at 18–25 lanes that
  is pure latency, 3× the floor to do the same work.
- Grid shape that works: `<<<dim3(8, split), 256>>>`, one thread per head dim, warp-per-cache-lane
  score with a `uint4` (8 BF16) load per lane, 8 lanes per tile. 5.2 KB shared per block means the
  8-block grid uses 8 of 20 CUs — for 6 attention layers that is 6 launches touching 8 CUs each,
  so **layer-parallel or two-tokens-per-launch is where the CUs come from**, not splitting rows.
- Splitting the row (`--split > 1`) is only meaningful for long contexts *and* only under the
  streaming form; the two-pass form's denominator is the whole row by definition.
- The `rope_table` launch `run.py` uses to prove the in-kernel transcendentals is a stage check and
  is not in the timed path.

## 9. Not measured here / open

- **No timings.** This run's budget went to the gate; per the queue's rule a provisional number
  from a contended box is worth less than a committed gate. `--mode burst` exists in the binary
  (wall-clock over a burst, per `mlp_gemv`'s trap that `hipEventElapsedTime` reads 0 below ~1 ms)
  and `--tile N` replicates the caches so a long-cache plan is actually cold, but nothing in this
  file quotes a µs. Until someone runs it under `tools/gpu_lock.sh`, treat the launch shape above
  as arithmetic on the roofline, not measurement.
- `speccheck`'s `apply_rope` still rounds cos/sin in F32 (§1). Fixing it is a speccheck change with
  its own re-gating; this directory now proves which policy the dumps carry either way.
- The three-knob matrix (staged/fused × online/full × split) is 8 shapes; only the ones run.py
  cross-checks are claimed equivalent.
- Three kernel bugs were found by the gate rather than by reading: the rope table used frequency
  index `d` instead of `d % 32` for dims 32..63 (44 of 128 entries wrong), the two-pass form
  divided by the softmax sum twice (uniform per-head scale error, cos 1.000 with meanrel 0.7 — the
  most deceptive failure this project has produced), and `--softmax full` with `--split > 1` was
  silently normalising per block. All three are now refused or fixed; the second is worth rereading
  because "cos = 1.0000, meanrel = 0.7" is what a wrong normalisation looks like.

## 10. P7 (2026-10-13): the online route becomes an engine route, and gets the band §07.3 pre-authorized

Plan `progress/2026-10-12/plan-prefill-2026-10-12.md` §3/P7 moves the engine above the score staging: when a row has
more lanes than `MAXSC`, `engine/kernels_attn.cpp` now selects `full=0 --split ceil(lanes/MAXSC)`
instead of refusing the token. **No kernel changed for it** — `attn_core`'s streaming branch,
`attn_combine`'s `(max,sum)` merge and the `part` geometry were already gated here in `--mode once`
shape; what was missing was an engine that asked for them, and a gate that could tell the online
route's output from the pinned form's without asking the impossible of it.

**The judging rule, and where its authority comes from.** §07.3, verbatim: *"When that path is built it
must be gated on a **cos/meanrel band** — §10 G6's thresholds are already of that shape — and not on
agreement with the pins, which it cannot reach."* So `run.py --softmax online` now holds, for the two
stages the reduction produces (`attn_pre_gate`, `gated`):

- **held:** `cos >= 0.9999` and `meanrel <= 1e-2` against the pins — `COS_MIN`/`MEANREL_MAX`, the §10.1
  numbers this file already used. **No new constant was introduced anywhere.**
- **demoted to a reported number:** `ndiff` and the 1-ULP large-element count — i.e. pin agreement,
  which §3 measured unreachable at any tile size (26.2 % differ, 6.2 % large). They are still printed
  for every case and in the band summary; they just no longer decide a verdict that would fail correct
  code.
- **untouched:** every other stage (`q_rope`, `k_rope`, `rope_cs`, `gate_vec`, `cache_keys`,
  `cache_values`, `v_append`) is still required bit-exact, at every length, online or not — the softmax
  form cannot move a copy or a rope. And `--softmax full` — the route all five priced fixtures ride —
  is judged exactly as before, by all three tests, and still reproduces the pins at **0 ndiff**.
- The ids half of G6 (`sampled ids exact`) is discharged one level up, by `engine/gate.py` section (a)
  on `wide520f`. This op has no tokens to sample, and saying so is better than pretending the cos
  covers it.

**Anti-vacuity — the band is only safe if an online-labelled run cannot secretly be the two-pass form.**
Two mechanical checks, both firing:

1. A run whose softmax stages are bit-identical to the pins on *every* case is not streaming. Measured:
   **67 of 68 cases moved** (`attn_pre_gate` 34 883/139 264 = 25.0 %, `gated` 31 670/139 264 = 22.7 %).
   The one that did not move is `prefill000` — a single lane, where the two forms are the same
   arithmetic. It is reported on the gate's own `anti-vacuity` line and a zero would FAIL.
2. `--neg nomax` (A-17's un-rescaled running max) exists **only** in the online branch and is now
   injected at the run's own `--split`, so it proves the split blocks ran the streaming accumulation.
   The merge cannot go unobserved either: above `split 1` the core writes only `part`, so a missing
   `attn_combine` leaves uninitialised memory where the output should be and no aggregate comes near
   0.9999.

**A trap that check found, and it is this directory's own harness, not the kernel.** `nomax` is invisible
inside a *single* tile — within one tile the subtraction constant cancels in the final divide — and a
split block streams `ceil(lvalid/split)` lanes in tiles of `TILE`. The two decode cases the negative
harness used are 18/19 lanes: three tiles at split 1, **one tile at split 4**, where the injection
legitimately moved nothing and `--neg nomax` declared `caught NOTHING`. A negative that cannot fail
because of the case width handed to it is exactly the vacuous test `results.md` §6 forbids, so
`neg_cases()` now switches to the long-cache rows of `--long-fixture` above the split the decode cases
can carry (and `KERNEL_TILE` is read out of `attn_decode_kernels.h`, the way `engine/gate.py` reads
`MAXSC`).

**Re-taken on device, correctness only, `util` 8–9 % as read, no timing taken.** Each command
1380 stage comparisons, 0 failures, ~5 s wall (`--long` = the 20 `p300` rows up to 277 lanes,
`--negative` = all 12):

```
.venv/bin/python prototype/attn_decode/run.py --softmax online --split 2 --long --negative   # PASS
.venv/bin/python prototype/attn_decode/run.py --softmax online --split 4 --long --negative   # PASS
.venv/bin/python prototype/attn_decode/run.py --long --negative                              # PASS (1764 cmp, the pinned route, unchanged)
```

| stage | worst cos (floor 0.9999) | max meanrel (ceiling 1e-2) | elements differing from the pin | judged on |
|---|---|---|---|---|
| `attn_pre_gate` | 0.99999134 | 1.58e-03 | 34 883/139 264 (25.0 %) | cos + meanrel |
| `gated` | 0.99999282 | 1.82e-03 | 31 670/139 264 (22.7 %) | cos + meanrel |
| the other seven stages | 1.00000000 | 0 | 0 | bit-exact, unchanged |

Identical at `--split 2` and `--split 4`, and identical to the §3 row from 2026-10-11 — the split does
not add error on top of the streaming form, which is what §3's third row claimed and what this re-run
re-measures at engine-shaped lane counts. `fused` vs `staged` remains bit-identical at both splits.

**What this is not.** It is not a band move in the ledger's sense: no `BAND` constant, no fixture
margin, and nothing in `bench/band-history.jsonl` came from this section (the one entry the numbers
touch is G6's own 0.9999, which was already published and already this file's `COS_MIN`). The engine's
logits rule for `wide520f` is priced from `wide520f`'s own steps per D-2, not from this table. And the
one-pass route below 512 lanes is still the pinned route: the split exists only where a row does not
fit the staging, which is the blast-radius rule plan §3 states and `engine/results.md` proves with
five fixtures whose numbers did not move.

## 11. P8 (2026-10-13): a whole prefill wave through one launch — `blockIdx.z` is the query row

Plan `progress/2026-10-12/plan-prefill-2026-10-12.md` §2.1/P8 asks for the attention layers of a prefill **wave**
(§06.3: up to 512 query rows, each at its own position, each with one more visible lane than the row
before it) to be priced by ONE launch per attention layer instead of one per row. This section is the
op-level half: what the kernel had to gain, why the grid is shaped the way it is, and what a wave is
now required to agree with.

**`grid.z` over query rows, not a host loop.** `CoreArgs` gained `nrows` and the three kernels gained a
query-row index: `attn_prep<<<N_Q + N_KV, nrows>>>` uses `blockIdx.y`, `attn_core<<<N_Q, split,
nrows>>>` and `attn_combine<<<N_Q, nrows>>>` use `blockIdx.z` and `blockIdx.y`. A host loop over rows
would have been the smaller diff — no `nrows`, no row strides, no per-row slab in `part` — and it would
have failed the thing P8 exists for: §06.3's saving is *launch count*, and 512 launches from a loop are
512 launches however neatly they are written. The gate that has to notice is `engine/gate.py` (h),
which counts launches per batched site and must see the collapse (§06.4's "count the launches, do not
infer them from outputs"); a loop would leave that counter flat. Row position and row width come out of
the row index, one expression each — `a.pos + qrow`, `a.lvalid + qrow` — so the §06.3 causal mask is
per-row by construction and the two ways an engine gets it wrong are still expressible as injections
(`NEG_READPAST` reads `cap`, `NEG_DROPCUR` forgets the row's own appended lane).

**A wave rides the staged plan; a single row may still ride the fused one.** In the fused plan each core
block does its own §05.7 append, and for one query that is safe because nobody else reads what it wrote.
In a wave row *t* reads the lanes rows *0..t-1* appended, and blocks across `grid.z` have no ordering
between them, so a fused wave would be a race that usually works — the worst kind. The staged plan puts
every row's prep (norm, rope, gate extract, append) in a launch that is complete before any core block
starts, which is the only ordering the wave needs. The host therefore **refuses** `--launch fused` for a
case with `nrows > 1` rather than pricing a race, and refuses `--neg carry` for one too (the
previous-query state an injection seeds is undefined when there is no previous query).

**One row's bits must not depend on its neighbours.** That is the claim P8's blast radius rests on, and
it is what the new `--wave` stage measures, three ways:

1. **The wave against the same rows run one at a time, bit for bit.** Same pins, same knobs, five
   stages, every row. The pins cannot make this claim — two forms can each agree with a pin and still
   differ — so it is measured directly. Measured, L03 of `p300`, 277 rows in waves of 64 and in one
   wave of 277 and in waves of 32 on `p64`: **0 of 2 397 450 elements differ.**
2. **Every row against its own pinned prefill row** (`L03.attn.q_rope` / `k_rope` /
   `attn_out_pre_gate`, §10.5's pairing), judged **one row at a time** — a flat 64-row comparison would
   dilute §10.1's ndiff limit by 64 and would not be comparable with the one-row run it is being
   compared against. `gate_cases` therefore skips the reduction stages of a wave case and
   `wave_row_divergences` judges them per row, from both sources, under one rule.
3. **The online route, waves included**: `--softmax online --split 2/4` over the same spans, with the
   anti-vacuity pointed at this shape — an online wave whose reduction stages are bit-identical to the
   same waves run `--softmax full` never streamed. Measured **245 898 / 245 902** elements moved
   (split 2 / split 4); the check fails at zero.

Rows 64..127, 128..191 and 192..255 of `p300` are the multi-wave carry: their lanes below `p0` arrive
from `prefill.L03.cache_*` exactly as a previous wave would have left them, and their first row must
read lanes it did not append. `v_append` and `cache_*` are compared over the wave's whole span, which is
§05.7 checked for every row of it at once, and the §10.5 operand rebuild runs at both ends of every span
(it is a per-row fact; re-running the torch chain once per row would cost more than the gate does).

**A sampling gap this found, and it is not P8's.** Pricing all 277 pinned `p300` rows — which `--wave`
does and `--long` never did, because `LONG_ROWS` is a 20-row boundary sample — turns up **7 of 554**
reduction comparisons over the shipped §10.1 rule: rows 40, 52 (×2 stages), 78, 141 (×2) and 242 of
`p300`, worst 9 ULP at |ref|/max 1.9e-02, plus row 32 of `p64`. The same rows, launched **one at a
time** through the same judge, miss **identically**: same 7, 0 wave-only. So this is the op's long-cache
reduction floor at lane counts nothing in this directory had priced, not a wave effect — which is what
§10.1's own warning predicts (a chain's legal ndiff is set by summation order, and a whole-tensor
1-ULP move is invisible to a max-ULP bound). `LONG_ROWS` is **unchanged** — those rows were never
priced, and a boundary sample that misses them is a sampling gap to report, not a licence to move a
threshold. Reproduce with:

```
.venv/bin/python prototype/attn_decode/run.py --long --long-rows 40,52,78,141,242   # FAIL, same 7
.venv/bin/python prototype/attn_decode/run.py --wave                                # PASS, names them
```

`--long-rows` is new for exactly that reason: `sample` is the shipped list, `all` is every pinned row,
and the difference between "the gate is green" and "the gate is green on the rows it samples" is only
visible if both are sayable.

**One knob became obsolete, and dropping it let the mixed grid be priced.** P7's host refused
`--softmax full --split > 1`, because the two-pass form normalises its weight matrix inside one block
and splitting it was silently wrong. `blocks_for_row` moves that decision from the launch to the ROW — a
full row that fits `MAXSC` takes exactly one block whatever the grid's depth — so the combination is now
meaningful: one-pass wherever a row fits, streaming above it. That is the shape `engine/kernels_attn.cpp`
needs for a wave whose widest row is over the staging, and the wave stage prices it: the same waves at
`--split 2 --softmax full` move **0 elements** versus `--split 1` — the upper lane blocks return before
they touch anything and `attn_combine` skips a row the core already finished. What is still refused, and
is the only remaining route to mis-normalisation, is a row WIDER than the staging served by ONE block
(`lanes_last / split > MAXSC`).

The wave stage's third line — waves at `--split 2 --softmax full` moving 0 elements — is that mechanism
gated: a wave with a 2-deep lane grid in which every row takes the one-pass route. No shipped fixture
has a wave that straddles `MAXSC` (`p128` and `p300` are single waves under the staging; `wide520f`'s
tail wave of 8 rows is entirely above it, and `chunk64b`'s waves are 64 rows wide at 64-lane steps, so
they cross the staging between waves, never inside one), so this is the mechanism priced rather than the
combination; both halves of the combination are gated separately (the streaming rows by §10, the
one-pass rows by the identity in point 1 above).

**Re-taken on device, correctness only, `util` 93 % as read (discount it: the serving process
busy-polls the iGPU), no timing taken.** Waves are 64 rows = plan §2.1's chunk width:

```
.venv/bin/python prototype/attn_decode/run.py --wave                                  # 4112 cmp, 0 fail, 22 s
.venv/bin/python prototype/attn_decode/run.py --wave --wave-rows 300                  # 4084 cmp, 0 fail, 22 s
.venv/bin/python prototype/attn_decode/run.py --wave --long-fixture p64 --wave-rows 32 # 2012 cmp, 0 fail, 2 s
.venv/bin/python prototype/attn_decode/run.py --wave --softmax online --split 2        # 3738 cmp, 0 fail, 21 s
.venv/bin/python prototype/attn_decode/run.py --wave --softmax online --split 4        # 3738 cmp, 0 fail, 23 s
.venv/bin/python prototype/attn_decode/run.py --negative                              # 12/12 caught, 1584 cmp, 4 s
.venv/bin/python prototype/attn_decode/run.py --softmax online --split 2 --long --negative  # 1380 cmp, 0 fail, 5 s
.venv/bin/python prototype/attn_decode/run.py --softmax online --split 4 --long --negative  # 1380 cmp, 0 fail, 5 s
.venv/bin/python prototype/attn_decode/run.py --long --negative                       # 1764 cmp, 0 fail, 3 s
```

The last three are §10's re-gate commands, re-run unchanged after the kernel grew `nrows`: same
comparison counts, same 0 failures, same 67-of-68 anti-vacuity, same 12 negatives localising to the same
stages — including G4's de-interleave (`--neg gatehalf`, caught by `gate_vec`+`gated` and nowhere else)
and `--neg qswap`. A single-row case with `nrows` absent behaves exactly as it did before: the row
strides are the strides a one-row launch already used, and `grid.z = 1` costs nothing.

## 12. P9 Phase 2 (2026-10-14): the tiled wave core — `attn_tile_wave`, gated on CPU, not yet on device

Phase 1 (`progress/2026-10-14/attn-wave-diagnosis-2026-10-14.md`) closed with the byte wall binding: 25.79 GB of K/V
re-read for 25.79 GFLOP, 1.000 byte/flop, 108.8 ms of traffic against 109.2 ms measured. §9's lever is
re-use: one K row loaded once and fed to more than one score dot. This section is the tiled form that
does it, everything that could be proven without a device, and the command list for the device pass.

**What was built.** `attn_tile_wave` (`attn_decode_kernels.h`, after `attn_core`), behind the existing
wave entry point in both launchers:

| knob | what it selects |
|---|---|
| `run.py --wave --tile-kernel` | the op's own A/B; every core launch of the run goes through the tile form |
| `decode_run --attn-tile 0\|1` | the engine's A/B, echoed as `attn_tile=` on the `NUMERICS` line |

Geometry: block = (one KV group's G=4 query heads) x (M=4 consecutive query rows) x (one MAXSC-wide lane
window); grid = `dim3(N_KV, split_for_lanes(lanes_last), ceil(nrows/M))`, `blockIdx.z` inverted so the
widest row tiles launch first (§2's dispatch tail). `L=8` K lanes resident per sub-block. The five
ordered reductions of §05.5 are untouched — same 8 MACs/thread, same 5-step butterfly, same single BF16
round site, same serial `m`/`l`/weight/P·V order — so the contract is bit-for-bit equality with the
shipped form, not agreement within a band.

**Priced, not discovered** (compile-only census, `attn_tile_model.py isa`, which is `hipcc -O3
--offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage -c` on the header — no device touched):

| kernel | VGPR | spill | scratch | occupancy | LDS/block |
|---|---|---|---|---|---|
| `attn_core` (shipped) | 53 | 0 | 0 | 16 waves/SIMD | 38,944 B |
| `attn_tile_wave` | **49** | 0 | 0 | 16 waves/SIMD | **20,480 B** |

* registers came in *cheaper* than §5 priced (49, not 53+32). The +32 was for holding G or GxM query
  rows live; the pair-first work rule (warp w takes pairs w, w+nw, …) means a warp ever holds ONE pair's
  query, 8 floats, so occupancy is unchanged and registers are not the limit.
* LDS is exactly the sum of the source's own `__shared__` arrays — 16,384 B of score staging (G x M x
  MAXSC BF16 scores) + 4,096 B of K at L=8 — so **3 blocks/CU**, while `attn_core` in the same
  translation unit reserves 38,944 B for arrays summing to 6,176 B. That asymmetry is new evidence for
  §3: the 6.3x inflation is kernel-specific, not a code-object or toolchain property, and if §3's
  residency query answers 1 block/CU for `attn_core` then this form also relieves it.
* The score staging is 2 bytes and not 4 because §05.5's score is `bf(fb(dot)) * SCALE` with
  `SCALE = 1/16`, and a BF16 value times a power of two is exact — so the score *is* a BF16 value and
  reads back as the identical F32. That halves §5's budget and is the only reason M=4 fits; it is also
  why `--score-f32` is a *refusal* in `attn_tile_ok` rather than a silent rounding.
* M=16 is not reachable on this boundary: staging is G x M x MAXSC scores, so M=16 is 64 KiB on a CU that
  has 64 KiB total, and shrinking the window below MAXSC to buy M would push rows the shipped form
  computes exactly into §07.3's streaming band — a band decision, not a kernel knob.

**Traffic, as written vs as it could be.** `attn_tile_model.py traffic --tokens 1024 --wave-rows 512`:

```
  touches          25,190,400   flops 25.79 GFLOP (the ledger's 25.79)
  shipped K/V read 25.79 GB  = 1.000 byte/flop
  tiled   K/V read  6.47 GB  = 0.251 byte/flop   (4.0x less, = M: pairs stage per pair)
  block-staged      1.62 GB  = 0.063 byte/flop   (16.0x less, = G*M: the next lever)

  shipped bytes / 237 GB/s       108.8 ms   <- predicts the measured line (109.2)
  tiled bytes / 237 GB/s          27.3 ms
  block-staged / 237 GB/s          6.8 ms
  arithmetic at the hand-GEMM rate 30.3 ms   <- what binds once bytes stop binding
  arithmetic at the vector rate     8.8 ms
```

The honest reading of the first asterisk: this version stages K per *pair*, so the four heads of a group
issue four identical 512-byte loads and write them into the same LDS slots with the same bytes. That is
benign and probably cheap (L1 likely serves three of the four), but a traffic claim may not rest on an L1
hit, so the model books all four and the numbers above are the pessimistic ones: **the re-use actually
booked is M = 4x, not G x M = 16x.** [Superseded by §20: the re-use actually available as shipped was
**1x**, not 4x — staging per pair fetches each lane row G times per block, and pass 2 had no V stage at
all. Block-level staging of both K and V is what makes the honest number G = 4x.] Hoisting the stage to
block level — stage the group's rows once per
sub-block, let a pair substitute only its own appended lane at the use site, the one lane where pairs
differ — is the remaining factor of G, ~15 lines in pass 1 plus a pass-2 sub-block loop, and it is worth
running *only if* the device says bytes still bind, because at M=4 the traffic term (27.3 ms) is already
under the arithmetic term (30.3 ms) and §7's ≤40 ms win line does not depend on it.

**Where bit-identity stops — one place, and it is pre-registered here rather than discovered later.**
For a row wider than MAXSC, the shipped form cuts the row into `split` *equal* slices
(`span = ceil(n/split)`, per row) while the tile form owns *fixed MAXSC-wide* windows, because a window
is what a block shares a K load over. Both are §07.3's streaming form and both sit inside its
pre-authorized band, but the two groupings are not bit-identical to *each other*: different partial
boundaries are a different F32 sum. So the bit-for-bit claim is exact for every row the one-pass route
serves (lanes ≤ MAXSC) and is a **band** claim above it. No pinned fixture reaches 512 lanes — p300 tops
out at 277 — so neither this gate nor `run.py --wave` can exercise it; the engine's 1024-token body 1
(lanes 513..1024) is the first place it exists. If an engine gate step moves there, it is this, it is
inside §07.3's band by construction, and it is a band question for the parent, not a bug to chase. The
mirror models the two groupings separately so it cannot quietly agree with itself.

**CPU gate, run and green** (`prototype/attn_tile_model.py`, CPU-only, no device, no timing; p300 L03,
waves of 64 rows, G=4 M=4 MAXSC=512):

```
gate:            567,296 elements, 0 failures
  per wave:      ref-vs-pin 0 | tile-vs-shipped 0 | mirror-vs-pin ndiff 63/12/111/7/1, cos 1.00000000
selftest:        10/10
gate --negative: 6 injections over the prompt wave and a carried wave, mis-localisations 0
```

Three things in that output are load-bearing and none of them is the headline:

1. **The tile-vs-shipped column is the claim** (0 differing elements over 567,296), and it is a claim
   about the mirror's arithmetic. What makes the mirror worth believing is the `ref-vs-pin` column:
   `reference.attend`'s pinned form reproduces `attn_out_pre_gate` **bit-exactly, 0 of 131,072**, on
   every wave, which proves the positions, the pre-append cache and the row/head layout are right
   (§10.5's trap is exactly a gate that cannot see a mis-pairing). The mirror's own residue against the
   pins — 194 elements, 0.034 %, max 10 ULP, cos 1.00000000, meanrel ≤ 1.6e-6 — is the ordering floor of
   §10.4: it sums `l` and `acc` serially in lane order, which is what the *kernel* does, whereas torch's
   softmax reduces in its own order. Per AGENTS.md the chain aggregate judges a chain, so the gate's
   bar is `cos ≥ 0.9999999 and meanrel ≤ 1e-5` and `ndiff` is printed, not gated. A real arithmetic bug
   moves thousands of elements and drops the cosine.
2. **The poison tripwire is real, not a count.** The mirror reads K/V through an instrumented cache whose
   unbuilt lanes hold NaN, so a mask leak produces a NaN in the output rather than a number a reader
   has to trust (`mixed_bw`'s lesson). It reports 0 poison here for a specific reason worth knowing: on
   these spans `cap == lanes_last`, so `--neg readpast` cannot step outside the built region at all.
3. **Every must-catch set is derived from the geometry, and two of them are not "everything"** — which is
   the `mlp_gemv` rule (a negative that catches everything measures nothing) applied one layer up, after
   two FAILs rather than by inspection:

| injection | caught | blind | blind because |
|---|---|---|---|
| `readpast` (mask gone) | 1008/1024 | 16 | a row that already attended `cap` lanes cannot see the mask disappear |
| `dropcurrent` (§05.7 append forgotten) | 1024/1024 | 0 | — |
| `gqaoff` (whole group reads the neighbour) | 1024/1024 | 0 | — (N_KV=2, offset 1 mod 2 moves both) |
| `dimhalf` (128 of 256 dims) | 1016/1024 | 8 | **softmax of a singleton is 1.0 whatever the score**, so no score-only bug can move row 0 of a span |
| `gqa_head0` (one head reads the neighbour) | 128/1024 | 896 | only head 0's K/V changes — the shared-load wiring test |
| `tilecarry` (state leaks from the previous row of a block) | 768/1024 | 256 | only rows that are not the first of their row tile can inherit |

`tilecarry` is §07.3's bug in its new shape: with M rows per block the leak is *between rows of one
block*, which the shipped per-(row,head) form cannot express — so `run.py --neg carry` is refused under
`--tile-kernel` and its tile analogue lives here instead. [Superseded by §20: one row per block makes a
row-to-row leak unrepresentable, so this row is retired. Its successor `headcarry` leaks a head's state
into the next head of the SAME block, which is the boundary M=1 actually leaves open, and it is localised
by `h % G` rather than by row index.]

**Device pass, hand-off (the parent runs these; `run.py` self-locks, `decode_run` does not).**
Correctness first, then timing, blocking, zero subagents, best-of-3, and §7 item 3's contamination check
(`proj`/`mlp_down`/`mlp_gateup`/`gdn_rec` must not move):

```sh
# 1. the op gate on the tiled form — the pins are the same pins, so this IS the §5 contract
.venv/bin/python prototype/attn_decode/run.py --wave --tile-kernel
.venv/bin/python prototype/attn_decode/run.py --wave --tile-kernel --negative
.venv/bin/python prototype/attn_decode/run.py --wave --tile-kernel --wave-rows 300
# 2. engine A/B, bit-for-bit, one window (default 0 must equal the pre-Phase-2 binary)
tools/gpu_lock.sh acquire "attn phase2 ab"; tools/gpu_lock.sh util
./engine/build/decode_run --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --attn-tile 0 --out tmp/ab/ship
./engine/build/decode_run --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --attn-tile 1 --out tmp/ab/tile
tools/gpu_lock.sh release
diff <(cd tmp/ab/ship && ls) <(cd tmp/ab/tile && ls) && for f in tmp/ab/ship/*; do cmp -s "$f" "tmp/ab/tile/${f##*/}" || echo "DIFF ${f##*/}"; done
# 3. §7 attribution, both sides, best-of-3 (the `attn` line is the measure; the four siblings are the
#    contamination check)
for i in 1 2 3; do ./engine/build/decode_run --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --time-waves \
    --gemm library --attn-tile 0 | grep -E "WAVE gran=stage|WAVEGAP"; done   # then --attn-tile 1
# 4. §9's counters over both cores, to check the read census instead of assuming it
rocprofv3 --kernel-include-regex "attn_tile|attn_core" --pmc SQ_INSTS SQ_INSTS_VALU SQ_INSTS_LDS \
    SQ_WAVES GRBM_COUNT GRBM_RUN_BUSY_STATS --output-format csv -d tmp/rocp2 -o tile \
    -- ./engine/build/decode_run --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --attn-tile 1
```

Pre-registered before any of it runs, as §7 demands: the measure is `WAVE gran=stage family=attn ms` on
the library route at 1024 tokens, `wave_rows=512`, baseline 109.2 ms re-taken in the same window; **a win
is ≤ 40 ms absolute**; 40–80 ms is "traffic fell, something else binds" and must be reported PARTIAL; > 80
ms is not a win. The model's own expectation is **27.3 ms of traffic against a 30.3 ms arithmetic floor**,
i.e. a line that should land in the high-20s to low-40s depending on how well the two overlap — so a
number near 40 ms is the model being right, not the kernel being slow. Falsifier A: bytes fall 4x and
`attn` does not fall at all ⇒ §4's verdict was wrong and the redundant pass-2 loops are the wall.
Falsifier B: `attn` lands 40–80 ms ⇒ traffic fell and something else (issue, LDS, or the 4x-pair staging)
binds ⇒ report PARTIAL and price the block-level stage. `SQ_INSTS_VALU` must not *rise* on the tile form.
No number from a worker run belongs in `bench/performance-status.md`; none was taken here, and the two
`hipcc -c` calls and the Python gate above touched no device.

## 13. §12's device pass, run by the parent (2026-10-14, lock `ATTN-P2`): **RED — the flip is refused**

§12 was green on CPU and said so. The device pass on the same code is not green, and three separate things are
wrong with it. Nothing was flipped: `--attn-tile` stays `0`.

1. **The op gate on the tiled kernel FAILS.** `prototype/attn_decode/run.py --wave --tile-kernel` reports
   `wave L03: 1098790 elements moved when the lane grid was deepened to 2 under --softmax full`, and the same
   failure is the first thing `--negative` prints, so it is not a control artifact — the tiled form is **not
   invariant to how the lane grid is cut**, which is the same class of property §12's whole "bit-identical below
   MAXSC" claim rests on.
2. **The engine A/B is not bit-identical where the claim said it would be, and it flips a token.** `--attn-tile 0`
   vs `--attn-tile 1` on the 1024-token prompt differ in `logits1023.bin` **and in `sampled.txt`** — a changed
   argmax on the last row, not a cosmetic drift. Row 1023 is a >MAXSC row so a *band* difference is admissible
   there by the accepted condition, but a **token flip** is the outcome that condition was written to bound, and
   it arrived as a bare `cmp` difference with no band measured against it.
3. **The win cannot even be measured today.** Under `--time-waves` the tiled route fails closed: `attn_decode_layer:
   the tiled wave form is unavailable: a wave's §05.7 append must fin…`. So there is no `attn` attribution line for
   the tile side at all. Meanwhile the shipped side, re-measured in this window, is **116.29 ms / 33.60 % of a
   346.05 ms wave** (`ms_per_launch 9.69`) — the same 33 % as before, so the byte-wall diagnosis still stands and
   the prize is still there. It is just not collected by this kernel yet.

**What this means and what it does not.** Phase 1's counter result is untouched — attention is still memory-pattern
bound and still worth ~1/3 of prefill. Phase 2 as delivered is **not a candidate for default**: a gate that fails,
a flipped token on the first fixture that exercises the band region, and an instrument that cannot price it. The
correct next move is to fix (1), size (2) against §07.3's band with the worst element's location reported, and make
(3) either work under `--time-waves` or say precisely why the replay path cannot satisfy §05.7.

## 14. §13's three defects, diagnosed and fixed CPU-side (2026-10-14, worker pass: **no device touched, no timing taken**)

All three turn out to be one kernel bug, one guard bug, and two instruments that could not tell a crash from a
measurement. None of them says the tiled core's arithmetic is wrong, and none of them was fixed by moving a band.
`--attn-tile` stays `0`; the device re-take is §14.5.

### 14.1 Defect 1: a **real defect** — `attn_tile_body` indexed its K stage by the window-absolute lane, so the stage walked out of LDS

Position, with the evidence the parent asked for: **the tiled kernel was broken, and the control that reported it
was also mis-wired.** Both halves matter, and the fix went to both.

The bug. `s_k` is `ATTN_TILE_L * DH` = 8 x 256 BF16 = **4,096 B**, inside a kernel that reserves **20,480 B** of LDS
in total (§12's `isa` census). The sub-block loop steps `s_lo` by `ATTN_TILE_L`, but both the stage write and the
score read indexed `s_k[rr * DH + lane * 8]` with `rr` the **window-absolute** lane, not the slot `rr - s_lo`. So any
row tile whose widest row exceeds 8 lanes writes past the array, at `(span - 1) * DH * 2` bytes: **141,824 B** for
p300's 277-lane wave and **262,144 B** for an engine MAXSC-wide window, against a 4,096 B reservation. Out-of-range
LDS does not fault on this part — the writes land on the 16 KiB of score staging next door, so the scores that
survive are K bytes. That is a silent numerics difference produced by an address, and it fires for **every
sub-block after the first**, i.e. for every row above 8 lanes, which is why it looked like the grid rather than the
index. Correct now: `s_k[slot * DH + lane * 8]` with `slot = rr - s_lo`, and the same `(rr - s_lo)` on the read side.

Why the CPU mirror could not see it, which is the trap worth keeping: **the mirror models lanes, not addresses.**
`isa`'s `LDS Size` prints the *declared* sum and was green throughout; a footprint claim has to be priced from the
loop's own index expression. So `attn_tile_model.py` gained `lds`, which reads this kernel source, finds the
`s_k[...] + lane * 8` expressions, classifies the index as sub-block-relative or window-absolute, and reports the
worst byte the stage can reach:

```
$ git show e44ed51:prototype/attn_decode/attn_decode_kernels.h > /tmp/pre.h   # the header as shipped
$ attn_tile_model.py lds --kernel /tmp/pre.h   # index spellings: `rr * DH` -> WINDOW-ABSOLUTE lane
  FAIL  a tile whose widest row is L lanes ...  reaches   4.10 kB (declared 4.10 kB, block 20.48 kB)
  FAIL  p300 --wave --wave-rows 64, widest 277  reaches 141.82 kB
  FAIL  engine 1024-token wave, MAXSC window    reaches 262.14 kB
$ attn_tile_model.py lds                       # index spellings: `slot * DH`, `(rr - s_lo) * DH` -> sub-block slot
  ok    ... all three reach 4.10 kB            lds: PASS
```

Note the first line goes FAIL even where the *size* happens to fit: an absolute index into a slot-indexed array is
wrong by construction, and a census that only complains when it overflows would pass the same bug at a smaller
fixture. Two `selftest` checks (12/12 now) pin both spellings at a 277- and a 512-lane span.

The number §13 saw is the corruption's footprint, not grid dependence. `oob-bound` prices how much the control
*could* have moved if the only damaged rows are the ones an absolute index can reach — a row whose own row-tile is
`<= L` lanes cannot be touched by the overrun whatever it does:

```
$ attn_tile_model.py oob-bound --observed 1098790
row-tiles with span > L (=8 lanes): 68 of 70; rows unreachable: 8 of 277
reduction elements the control compares: 1,134,592
upper bound if ONLY the unreachable rows stayed clean: 1,101,824  (97.11 % of the comparison)
observed / bound                   : 99.7246 %   (bound - observed = 3,034 elements that happened to agree)
oob-bound: CONSISTENT
```

99.72 % of the geometric bound, with the whole shortfall explained as elements that coincidentally agreed, is not
what "deepening the grid changed the arithmetic of a row that fits the staging" would look like; it is what "almost
every row above 8 lanes got hit" looks like. The check is falsifiable in the other direction too — feed it a bigger
number and it says NOT EXPLAINED and exits 1.

The control was wrong as well, and it was **tightened, not relaxed**. Claim 2b used to launch its deep side without
`--tile-kernel`, so the line printed "the lane grid was deepened to 2" about a run of a **different core**: it
compared the tiled form at split 1 against the **shipped** form at split 2, and §13 read a real address bug as grid
dependence. Claim 2b now runs both of its own sides, passes the form flag to both, and reads `core_grid=(x,y,z)` off
each side's own `RESULT` line (new report in `attn_decode.hip.cpp`, on the `plan`, `once` and `burst` lines alike),
because a control that recomputes the depth in Python would reproduce the assumption it exists to test. It can now
fail in three distinct ways instead of one: `grid.x` not the core under test (N_KV tiled, N_Q shipped — that is the
mis-wiring itself, named), `grid.y` not actually deeper (a `--split` the core ignored would formerly have passed by
being the same launch twice), and bits moved at a deeper grid (the §07.3 one-block-per-row property). `--split` also
stopped being a silent no-op on the tiled path: `a.split = std::max(split, tile_split_of(lanes_last))`. Where a
fixture's lane counts already pin `grid.y` at or above the asked split, the line says "control not exercised"
instead of reporting a pass or a vacuity — that geometry has no deeper grid to compare, which is a fact about the
fixture, not a verdict.

CPU state after the fix: `lds` PASS (was FAIL), `selftest` 12/12, `gate` **unchanged at 567,296 elements, 0
failures** — the arithmetic the mirror predicts was never the thing that was broken.

### 14.2 Defect 3: the §05.7 refusal was reading the **one-row plan's** flag; wave replay satisfies §05.7 already

The precondition was never unmet. `attn_decode_layer` asked `attn_tile_ok(c.score_f32, c.fused, ...)`, and `c.fused`
is the **one-row** plan's value (1 = §05.2's fused launch, where the core stages and appends inline, so a second row
could read the cache before the append lands — that is what the refusal protects). A wave cannot run that plan at
all, and does not: the `rows > 1` branch launches `attn_prep`, which stages every row and performs every row's
§05.7 append on the same stream, and then launches the core with `fused = 0` so it reads what prep wrote. Append
finished before any row read the cache — the condition is met by the shape of the launch sequence, not by a waiver.
The fix is one named value, `wave_fused = 0`, used by the guard **and** by the launch, so the guard cannot drift from
the thing it guards. With it the guard's `fused` branch is dead on the wave path by construction, which is the correct
state; the `warp != 32` and `score_f32` refusals still fire.

That bug is also why defect 2 looked like numerics: `--attn-tile 1` threw on the **first wave body of every prefill**
— every prompt goes through wave bodies — in every path, `--time-waves` included, so the tiled side wrote no logits
at all (§14.3) and there was no milliseconds line because there was no run. If the guard ever has to stay shut, the
fallback attribution is the prototype's own burst (`run.py --mode burst --tile-kernel`, x launches/wave) cross-checked
against `rocprofv3` kernel ms for `attn_tile_wave` vs `attn_core` — but `--time-waves` now answers the question
directly, so those are backups and were not needed or used here.

### 14.3 Defect 2: **there was no flip to measure** — the tiled A/B produced no numbers at all

`tmp/ab/tile/` contains **one file**: a 61-byte `sampled.txt` that is the header line and nothing else, and **no
`logits*.bin` at all**. `tmp/ab/ship/` has both logits files and 1,025 sampled rows. `decode_run` opens the sampled
log and writes its header *before* prefill and dumps logits only after prefill returns, so "header-only log, no
logits" is the signature of a throw inside `prefill_wave` — which is exactly §14.2. §13's `cmp` loop then did what a
`cmp` loop does: a missing file and a truncated log both print DIFF, and a crashed run was recorded as a numerics
difference on row 1023.

So: no token flip was observed, no band was entered, and §13 item 2 is **void rather than satisfied**. Row 1023
remains unmeasured, and the claim "bit-identical below MAXSC, banded above" is untested on the device until the
re-take below runs. `--attn-tile` staying `0` is therefore still the only defensible default.

The instrument, so this cannot happen quietly again: **`engine/attn_form_band.py`** (new, CPU-only — it reads two
dump directories and nothing else). Its two design rules are the two ways §13 went wrong:

1. **A missing or short dump is a crashed run, never a band difference.** Every row is required on both sides before
   anything is compared; an absent row prints `MISSING` with the side named and the two `sampled.txt` line counts
   beside it, and exits 1 with "nothing in this A/B was measured". A `diff`/`cmp` loop over two directories cannot
   tell "the bytes differ" from "one side died", and only one of those is a numerics question.
2. **A flip is not a band event until its room has been measured.** Per dumped row it reports cos, meanrel, BF16- and
   F32-lattice max ULP **with the cancellation band applied**, the worst element's index, both sides' values and
   |ref|/max, both sides' top-1 id and top-1/top-2 room, whether the argmax moved, and the verdict. A flip at or
   above `flip_room_max` prints BLOCKER and exits 1 — stop-and-report, not a line to file under "band".

It judges only with numbers imported from the files that own them — `engine/gate.py`'s `CASCADE_BAND` and
`flip_admissible`, `prototype/attn_decode/run.py`'s measured `CANCEL_REL`, `MAXSC` from the kernel header — and
retypes none, so it cannot quietly invent a band. That choice is also what makes it honest about what is *not*
judged: above MAXSC a row is judged by the published per-row shape number `row_1cos_max`, while meanrel and the ULP
columns are **reported, not judged**, because `stage_meanrel_max`/`stage_cos_min` are published per stage tile and no
per-row meanrel or ULP bound exists for a reassociating route; the pooled band rows are what the stage keys judge.
At or below MAXSC any difference at all is a DEFECT, since bit-identity is the claim there (§12). A `sampled.txt`
step that no judged row explains is red rather than band, and the alignment used to decide that is printed, not
assumed — the ±1 between the §10.5 pairing and the REPL's is stated in the output because AGENTS.md records what
getting it wrong once cost. `--selftest` runs 15 checks on synthetic rows, with teeth in both directions: the same
0.2 perturbation is excused at an element under the cancellation band and held against the run at a large one; a
wide-room flip is a blocker and the same flip at coin-toss room is admitted; a one-pass row that differs is a defect
while a clean one stays green; a short `sampled.txt` is a run failure; and the preserved §13 dumps must come back
"crashed run". All 15 pass, on CPU, device untouched.

### 14.4 The two traps, for the index

* **A guard that reads the other plan's flag turns a working path into a crash that looks like numerics.** The wave
  satisfied §05.7 and was refused anyway; the refusal killed the run before it wrote a number, and the A/B reported
  the absence as a difference. Price a precondition against the plan the launch actually uses, and make the guard and
  the launch read one named value.
* **A `cmp`-based A/B cannot distinguish "the bytes differ" from "the other side never wrote a file".** Require the
  row on both sides, and print the row counts of the side's own log, before any number is compared.

### 14.5 Device re-take, hand-off (the parent runs these; blocking, one at a time, zero subagents)

No number here is mine and none of these was run by this pass. `run.py` takes `tools/gpu_lock.sh` itself, so hold the
lock only around the `decode_run` lines (AGENTS.md's deadlock trap; check `grep -c gpu_lock prototype/attn_decode/run.py`).

```sh
bash prototype/attn_decode/build.sh && bash engine/build.sh

# 1. DEFECT 1. Claim 2b now prints both grid depths and both grid.x values. Expect on the tiled core:
#    "wave at grid.y 1 vs 2 under --softmax full (tiled core, L03): 0 elements moved", and
#    `core_grid=2x1x..` / `2x2x..` on the two RESULT lines. A non-zero count now means the one-block-per-row
#    property is really broken, because both sides demonstrably ran `attn_tile_wave`.
.venv/bin/python prototype/attn_decode/run.py --wave --tile-kernel
.venv/bin/python prototype/attn_decode/run.py --wave --tile-kernel --negative
.venv/bin/python prototype/attn_decode/run.py --wave --tile-kernel --wave-rows 300
#    then re-price the diagnosis against whatever the gate prints:
.venv/bin/python prototype/attn_tile_model.py oob-bound --observed <the number the gate printed>

# 2. DEFECT 2. The A/B, with EVERY prompt row dumped on both sides -- `--prefill-logits last` (what §12 used)
#    leaves one row on each side, which is why there was no table to take. ~1 GB per side at 1024 tokens.
tools/gpu_lock.sh acquire "attn p2b ab"; tools/gpu_lock.sh util
./engine/build/decode_run --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --prefill-logits all \
    --attn-tile 0 --out tmp/ab2/ship
./engine/build/decode_run --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --prefill-logits all \
    --attn-tile 1 --out tmp/ab2/tile
tools/gpu_lock.sh release
.venv/bin/python engine/attn_form_band.py --ship tmp/ab2/ship --tile tmp/ab2/tile --prompt-len 1024

# 3. DEFECT 3. The attribution this could not price, both sides, best-of-3. This must now print an `attn`
#    line for the tiled side too; §13's shipped baseline to beat is 116.29 ms / 33.60 % of 346.05 ms.
for i in 1 2 3; do ./engine/build/decode_run --ids "$(cat progress/bench/evidence/q10/ids1024.txt)" --steps 1 --time-waves \
    --gemm library --attn-tile 0 | grep -E "WAVE gran=stage|WAVEGAP"; done        # then --attn-tile 1
```

Order matters, and it is the order the defects depend on: if step 1 is still RED, steps 2 and 3 price a core that
does not hold its own contract, and §12's pre-registered win line (<= 40 ms, with both falsifiers) should not be
quoted against it. If `attn_form_band.py` comes back RED on a flip at room >= `flip_room_max`, that is a
stop-and-report per §07.3 and this pass touched no band, so nothing here widens one.

## 15. §14's fixes, re-run by the parent (2026-10-14, lock `ATTN-P2B`/`ATTN-NEG`): **two fixed, one still blocking — `--attn-tile` stays 0**

| §13 item | re-run result |
|---|---|
| 1. not lane-grid invariant | **FIXED.** `run.py --wave --tile-kernel` → `4112 stage comparisons, 0 failures`, and the negatives are real: `qswap`, `gatehalf`, `ropefull`, `ropenone`, `gqaoff` each caught **only** at their declared stages, so the green is not vacuity. `attn_tile_model.py oob-bound --observed 1098790` → **99.7246 %** of what an `s_k` overrun can reach, which is why "real kernel bug, not grid dependence" is the right call rather than a convenient one. |
| 2. token flip | **VOID as written — §13's own `cmp` loop was the artifact**, and §14 caught it: the tiled run threw inside `prefill_wave`, so `tile/` held a 61-byte header and no logits, and `cmp` called a missing file a numerics difference. Corrected here, in my own section, because I wrote §13. |
| 3. unpriceable under `--time-waves` | **MEASURABLE now.** Both forms produce 1,026-row `sampled.txt` and full logits dumps. |

**But the underlying claim is still RED, and with a real instrument rather than my broken `cmp`.**
`engine/attn_form_band.py --ship tmp/ab2/ship --tile tmp/ab2/tile --prompt-len 1024`:

> `sampled.txt: 1025 steps compared, 2 differ at steps [1023, 1024]` … `2 differing step(s) are NOT explained by any
> judged row flipping within the flip rule` … **RED**

Rows 1023–1024 are the tail of body 1 — the >MAXSC streaming region — so the difference is exactly where §12's
accepted band was supposed to live, and it is **not** inside the flip rule as written. The band question is therefore
**unresolved, not answered**, and the tile form stays opt-in.

**The reason it cannot be resolved yet, and the thing to fund.** `attn_form_band.py` is judging **tile against
shipped**, but above 512 lanes neither of them is anchored to anything: **no pinned fixture exceeds 512 lanes**, so
there is no ground truth for any streaming path in this engine. Two implementations disagreeing in an unanchored
region cannot be settled by more comparison between them — it needs the >512-token HF reference fixture that was
promoted to blocking in §12's acceptance conditions. Until that exists, "the tile form is wrong" and "the shipped
streaming form is wrong" are indistinguishable, and neither is a safe default.

*Correction made while §16 pinned the anchor: "no pinned fixture exceeds 512 lanes" is too broad as written —
`data/refs/wide520f` crosses 512 and pins logits at lanes 513..524. Those are single-token DECODE steps
(§07.3's decode streaming), not prefill-wave rows above the staging ceiling, so the sentence's load-bearing
content — no HF ground truth for prefill rows above `MAXSC` — was true, and §16 is what closed it.*

## 16. The >512-lane HF anchor now exists (2026-10-14, worker pass: **CPU only — no device touched, no timing taken**)

§15's blocking gap was that neither attention form was anchored above 512 lanes. Two fixtures close it, made by
the dumper the repository already had (`tools/reference_dump.py`, extended with `--prefill-logits last|all`, the
switch `decode_run` has carried since §06.5 — no second dumper was invented), with their recipes and their
token-count guards committed in `tools/dump_fixtures.sh`.

| fixture | prompt | what is pinned | on disk | what it makes adjudicable |
|---|---|---|---|---|
| `data/refs/wide640f` | the first 640 tokens of `progress/bench/evidence/q10/ids1024.txt` (640 = 512 + 128 = 5 × §04.7's chunk) | **every prefill row**: `logits_rows.npy` = [640, 248320] BF16-as-uint16, 303.1 MiB — plus 8 replayed steps, `--deep-steps 7`, state fingerprints | 356 MiB | rows 512…639, the first rows the one-pass stage cannot serve |
| `data/refs/wide1024f` | `progress/bench/evidence/q10/ids1024.txt` verbatim — proved, not asserted: the manifest's `sources[0].ids` **is** that list, 1024 of 1024 | [1024, 248320], 485.0 MiB, same decode side | 551 MiB | rows 512…1023, **including row 1023, the row §15's A/B disagreed on** |

Both carry `prefill_logit_rows: all`, `logits_storage: bf16`, the §10.5 stop probe (`none in 300`, probed before
dumping, legacy stop 248044 also absent), and a `provenance` block: dumper, dump path, **transformers 5.19.0**,
torch 2.14.1+cpu, and the normalized recipe (scratch-only flags stripped so an independent run can reproduce it).

**Reproduced before trusted.** Two independent A/B dumps per fixture compared with `tools/check_dump_reproduction.py`:
PASS both, and a third run after installation — `reference_dump.py --verify` with the recorded recipe — came back
`REPRODUCE PASS — every pinned byte and every pinned manifest value came back identical` (548 files each, ~4 s of
CPU per dump). One trap worth recording, because it read as a broken fixture: **`--verify` verifies the flags you
type, not the flags the pin was made with** — verify wide1024f without the recipe and it reports 550 manifest
mismatches on a fixture that is pinned perfectly. `--verify`'s own help now says so.

**The nesting is real, checked, and has a negative control.** `tools/check_fixture_prefix.py data/refs/wide640f
data/refs/wide1024f --rows 640` → **PASS**: rows 0..639 of the 1024-row pin are bit-identical to the 640-row pin,
so HF's chunked scan is length-local over this prompt and the two anchors corroborate each other. Its negative
control, `--shift 1`, **FAILS** as it must (639/639 rows differ, first difference named with both bit patterns) —
without that the PASS would be worth nothing, which is the same rule §12's negatives obey.

**What the reference itself decides.** `tools/prefill_row_rooms.py data/refs/wide1024f`: 2 rows of 1024 tie at
their maximum (61, 84) and 4 rows have top-1/top-2 room under `flip_room_max` 0.10 (61, 73, 84, 104), so **1020 of
1024 rows were decided by HF with room to spare** — median gap 10.44. Row 512's room is 13.25 and **row 1023's is
12.25**. That is the number that makes this anchor able to settle anything: the disputed row is not a coin toss in
the reference, so a form that flips it cannot be excused by the room rule.

**The gate now judges rows, by the floors already in force.** `engine/gate.py` gained section (i): every pinned
row against the engine's own per-row dump (`logits<i>.bin` ↔ row *i*, the §06.5 pairing), judged by the §10.1
`cos_hard` 0.9995 / `meanrel_hard` 0.05 floors, floored on the library route by the existing `CASCADE_BAND`
`row_1cos_max` 2e-3 / `stage_meanrel_max` 3e-2 — **no band moved, nothing was widened, `tools/band_history.py`
still reports its 19 recorded values green.** Rows ≤ MAXSC are labelled one-pass and rows above it the streaming
form of §07.3, and the boundary rows (511, 512, last) are printed by name. A moved argmax goes through the same
`flip_admissible` room rule sections (a)/(b) use; the D-8 tie policy is keyed to a per-step pin file and was
**not** generalized to a row block, so a tied row is judged by the narrower fail-closed clause (the engine's id
must be one of the pinned tied ids) and the tie set is printed. `--attn-tile 0|1` is now passed to every
`decode_run` the gate starts **and asserted against that run's `NUMERICS attn_tile=`** — a log that cannot say
which form ran is evidence about neither. The fixture's row pins force `--prefill-logits all`, because a fixture
that pins rows against a run that dumps one row is a gate that looks like it ran and measured nothing. Its teeth
are five in-process negatives in `--selftest` at vocabulary 8 (bit-identical clean; wide-room flip RED; thin-room
flip ADMITTED with the shape floors demonstrably untouched; shape-only damage RED on cos; missing row reported).

**The two device commands, and the reading decided in advance.** Both belong to the main agent (a 1024-row
comparison is numpy over 254 M elements, and a worker's device budget is a minute):

```bash
tools/gpu_lock.sh acquire "attn-hf-anchor-1024" && tools/gpu_lock.sh util
# Two --work dirs, not one: the gate clears its work directory before every run (a run must be judged
# against the dumps it wrote, §10.5), so one directory cannot hold both forms' rows.
.venv/bin/python engine/gate.py --fixture wide1024f --attn-tile 0 --work tmp/form1024/ship   # SHIPPED form vs HF
.venv/bin/python engine/gate.py --fixture wide1024f --attn-tile 1 --work tmp/form1024/tile   # TILED form vs HF
tools/gpu_lock.sh release "attn-hf-anchor-1024"
# and, if both are green and still disagree with each other, the form-vs-form view of the same bits:
.venv/bin/python engine/attn_form_band.py --ship tmp/form1024/ship --tile tmp/form1024/tile --prompt-len 1024
```

Four outcomes, and they are different findings, not four spellings of one:

- **both inside** → §15's two differing steps are inside HF tolerance and the forms differ only from each other.
  The dispute is then a non-defect and the tile form may be adopted on price alone, with the form-vs-form
  difference recorded as an observation rather than a gate.
- **`--attn-tile 0` outside above 512** → **the shipped streaming form is wrong.** That is a shipping-path defect
  in the P7/P8 prefill wave, with a witness for the first time, and it is worse than anything §13–§15 found;
  stop and report it rather than routing around it.
- **`--attn-tile 1` outside** → the tile form is wrong and §12's band was mis-sized; `--attn-tile` stays 0.
- **both outside, differently** → both wrong; the region has no correct implementation today.

No branch licenses a band change: the floors section (i) uses are the ones already published, and it says so in
its own output. What is still missing after this pass, stated plainly: **none of the four readings above is known
yet**, because nothing in this pass touched a device — the pins, the guards and the instrument are what was built.

## 17. The HF anchor, run on device by the parent (2026-10-14, lock `ANCHOR`): **the anchor is real; the adjudication is not yet possible**

The four pre-registered readings in §16 could not be taken, because the two runs failed in **different,
non-comparable ways**:

| run | result |
|---|---|
| `--attn-tile 0` vs HF, `wide1024f` | **1 FAIL**: `the engine wrote all 1024 prefill row dumps — 1023 missing, first at rows [0..7]` |
| `--attn-tile 1` vs HF, `wide1024f` | **26 FAIL**, headed by the batched-vs-carry cascade-band checks (`1032 rows, 0 of them still bit-identical`) rather than by the HF row comparison |
| `attn_form_band.py` ship-vs-tile | `sampled.txt: 1032 steps compared, **0 differ**`, still **RED** on rows |

Read carefully, because the tempting summary is wrong twice over. **The shipped run produced no prefill-row dumps
at all**, so it says nothing about whether the shipped streaming form is numerically wrong — it is an instrument
gap: §16 says the fixture's row pins force `--prefill-logits all`, and evidently that forcing does not happen on
the path I ran. **The tiled run's failures are the wave-carry checks, not the HF row checks**, so its 26 failures
likewise do not yet price the tile form against HF either. And the `0 differ` on `sampled.txt` while rows are RED
means the two forms now agree on the *continuation* and differ only deeper — the opposite of §15's picture, and a
sign the two runs are not comparing the same thing.

**What the anchor already bought us, which is the reason it was worth building.** `tools/prefill_row_rooms.py`:
of 1024 pinned rows, **1020 are decided by HF with room to spare**, and specifically **row 1023 has room 12.25**
(row 512: 13.25). The disputed row is nowhere near a coin toss, so the form question is *answerable* — whichever
form misses HF at row 1023 is wrong, not marginal. That is exactly the property the unanchored comparison lacked.

**Next step, stated as the fix rather than a re-run attempt:** the gate must actually pass `--prefill-logits all`
when a fixture carries row pins, and the wave-carry checks must be scoped so a form A/B does not trip them for a
reason unrelated to the form. Until then the four readings in §16 stay unopened, and `--attn-tile` stays `0`.

## 18. §17's two instrument defects, fixed CPU-side (2026-10-14, worker pass: **no device touched, no timing taken**)

§17 could not take §16's four readings because the instrument was wrong in two places, not because the engine
was. Both are fixed in `engine/gate.py`, and each fix ships with a witness that has been shown to FAIL — this
session has already produced three checkers that could not fail, and a fourth is one too many.

### 18.1 Defect A — a fixture that pins every prefill row did not make the engine dump them

Two bugs, one of them cosmetic-looking and one of them the reason §17 read as it did:

1. **The forcing fired after the argv was built.** `main` assembled `cmd = [… "--prefill-logits",
   args.prefill_logits, …]` and only on the *next* statement did `if rows_pin is not None:
   args.prefill_logits = "all"`. The engine therefore received `last`, wrote one row, and the gate's own note
   said `all`. The requirement now comes from the manifest (`prefill_logit_rows`) and is applied **before**
   `cmd` exists, and the argv itself is asserted: `cmd_flag(cmd, "--prefill-logits") ==
   args.prefill_logits`. Asserting on `cmd` rather than on `args` is the point — `args` is the variable the
   old bug mutated, so a check on it could not have caught the old bug.
2. **A coverage gap fell through into a numerics verdict.** Section (i) counted the missing rows, failed that
   one line, and carried on to `check(not outside, "every judged prefill row is inside the floors")` — which
   passes when nothing was judged. That is the second half of §17's shipped run: `1023 missing` next to a green
   floor verdict computed over the one row that existed. A coverage gap must read as a coverage gap, so:

   * a new pure predicate `prefill_row_coverage(n_rows, work, vocab, asked, echoed)` distinguishes the three
     ways coverage is lost — the gate asked the wrong shape; the run's own `PREFILL logits=` echo disagrees
     with the ask (the §17 shape, caught at the source rather than inferred from missing files); and the echo
     says `all` but the files are absent or truncated;
   * `main` runs it immediately after the run parses, **before** sections (a)/(b)/(g)/(h), and returns on a
     gap: the failure names the side that owes the rows and the run prints no numerics section at all;
   * section (i) re-proves it as its first line and returns before printing any rule or geometry preamble;
   * the aggregate verdict is now `total == n_rows and not outside`, i.e. non-vacuous by construction, and a
     row that goes unread after coverage passed ends the section instead of thinning the sample.

**Witness.** `selftest_prefill_rows` went from 5 negatives to 16, all CPU, vocabulary 8 (§16's reason for
staying in-process stands: a 1024-row stand-in dump is a disk benchmark). The seven new ones are the §17 shape
(asked `all`, echoed `last`), asked-the-wrong-shape, no `PREFILL logits=` field at all, no rows at all, a
truncated row dump, plus five direct `prefill_row_coverage` cases. The five numerics cases now *also* assert
on what the section prints: `forbid=("was judged and is inside the floors",)`. Falsified by construction, in
the direction that matters — with the coverage predicate stubbed to always pass, six of the new cases FAIL and
two of them print exactly §17's symptom:

```
[PASS] every one of the 2 pinned prefill rows was judged and is inside the floors named above — 2 rows judged …
it printed a verdict it had no business printing: ['was judged and is inside the floors']
```

The wiring was witnessed separately from the predicate, with a throwaway CPU stand-in for `decode_run`
(`argv sink` + sparse row files, no engine, no device): asked `all` / echoed `last` → **1 failure, the coverage
line, and not one numerics section printed**; echoed `all` / wrote nothing → same, with `640 missing, first at
rows [0, 1, 2 …]`; echoed `all` / wrote all 640 → coverage PASSES and the gate proceeds into (a)…(h) (83
failures from the stand-in's all-zero logits, i.e. the coverage gate is not a blanket refusal). That harness
lives in `tmp/` and is not part of the tree; the in-process negatives are.

### 18.2 Defect B — the wave-carry checks were counted as evidence about a form

`--scope {full,form}` was added; the form A/B runs under `form`. Under `form`, section (h)'s **output**
comparisons between wave widths (cross-width ids, the cascade-band rows, the batched-vs-carry bit-identity on
the hand route) run, print as `[DEFERRED-PASS]`/`[DEFERRED-FAIL]`, and are written to
`wave_carry_deferred.json` beside the dumps — but are not counted into that run's verdict, and the summary
line says `PASS — with N wave-width check(s) DEFERRED and belonging to neither form` instead of a bare PASS.

**The choice, stated: deferred, not "same form on both sides" — because that was already true and could not
help.** `run_batched_ab` has passed `--attn-tile` to every side since §13's device pass was misread, so the
batched/one-row/carried sides already differ only in `--wave-rows`. The reason those 26 failures cannot
adjudicate the form question is not a mismatched pair of sides; it is that wave width is not the variable under
test. So the fix is attribution, and it is deliberately not a silencer: nothing is deleted, no band moved, no
check weakened, and `--scope full` — the default, and what every ordinary gate run uses — counts every one of
them again. A carry check that fails identically on both forms stays a finding, and the sidecar makes it a
*checkable* one rather than a rumour about whichever log was read last. The launch-counter half of section (h)
is **not** deferred in any scope: it counts SITES, not lanes, so no attention form can move it, and it is the
only thing in the tree that distinguishes "batching cost nothing" from "batching never ran".

**Witness, on CPU with the existing injection** (`--break-ab-logits 3` on `dec8`, which differs one row on the
one-row side of section (h)):

| run | result |
|---|---|
| `--dry-run --break-ab-logits 3` (scope `full`) | `ENGINE GATE FAIL — 2 failed`, rc 1 — unchanged from before this change |
| `--dry-run --attn-tile 1 --scope form --break-ab-logits 3` | rc 0, two `[DEFERRED-FAIL]` lines, summary names the deferral, sidecar records `n_failed 2 of 10` with `attn_tile: "1"` |

`--selftest` (dec8) is green, including its own `--break-ab-logits` case, which still expects `[FAIL]` under
the default scope; `engine/attn_form_band.py --selftest` is 15/15 (it imports `gate.py` for its bands, so this
also proves the import path still works).

### 18.3 What this pass ran (CPU only, no device, no timing)

| command | result |
|---|---|
| `engine/gate.py --fixture wide1024f --no-engine` | PASS, row-block pairing proofs green |
| `engine/gate.py --fixture wide1024f --dry-run --scope form` | PASS; forcing prints before `cmd`; argv assertion green; 10 wave-width checks deferred |
| `engine/gate.py --fixture wide640f --dry-run [--scope form]` | PASS, same shape — the fix reads the manifest, so it is not keyed to 1024 |
| `engine/gate.py --selftest --fixture dec8` | SELFTEST PASS (all sections, incl. the 16 section-(i) negatives) |
| `engine/attn_form_band.py --selftest` | 15/15 |
| `--break-ab-logits 3`, scope `full` vs `form` | rc 1 vs rc 0 + deferred-2 + sidecar, as above |
| `progress/prototype/evidence/stub_decode_run.py` coverage harness, three cases | see §18.1 |

Nothing above produced a millisecond figure, and none of it entered a timing document.

### 18.4 The device hand-off — exact commands, with §16's four readings restated one line each

Blocking, one at a time, zero subagents, lock held across the pair. `--scope form` is new and required for the
form question; `--prefill-logits all` is **not** typed — the manifest forces it, and the run's echo is checked.

```bash
# 1) the cheap generalisation check first: does the fix hold at 640 rows as well as 1024?
tools/gpu_lock.sh acquire "attn-hf-anchor-640b"; tools/gpu_lock.sh util
.venv/bin/python engine/gate.py --fixture wide640f  --attn-tile 0 --scope form --work tmp/form640/ship
.venv/bin/python engine/gate.py --fixture wide640f  --attn-tile 1 --scope form --work tmp/form640/tile
tools/gpu_lock.sh release "attn-hf-anchor-640b"
.venv/bin/python engine/attn_form_band.py --ship tmp/form640/ship --tile tmp/form640/tile --prompt-len 640

# 2) the one that decides §15
tools/gpu_lock.sh acquire "attn-hf-anchor-1024b"; tools/gpu_lock.sh util
.venv/bin/python engine/gate.py --fixture wide1024f --attn-tile 0 --scope form --work tmp/form1024/ship
.venv/bin/python engine/gate.py --fixture wide1024f --attn-tile 1 --scope form --work tmp/form1024/tile
tools/gpu_lock.sh release "attn-hf-anchor-1024b"
.venv/bin/python engine/attn_form_band.py --ship tmp/form1024/ship --tile tmp/form1024/tile --prompt-len 1024

# 3) is a deferred wave-carry failure a FORM finding or a wave-carry finding? diff the two sidecars:
.venv/bin/python -c "import json;f=lambda p:json.load(open(p));a=f('tmp/form1024/ship/wave_carry_deferred.json');b=f('tmp/form1024/tile/wave_carry_deferred.json');print('ship',a['n_failed'],'of',a['n_checks'],'| tile',b['n_failed'],'of',b['n_checks']);print('same verdicts on both forms:',[(c['check'],c['ok']) for c in a['checks']]==[(c['check'],c['ok']) for c in b['checks']])"
```

Each gate run must print, in this order and no other: `raised to \`all\` BEFORE the command line was built` →
`[PASS] the command line carries the prefill row shape…` → `[PASS] coverage: every prefill row this fixture
pins was dumped…` → the row table → `[DEFERRED-*]` carry lines → a verdict line that names the deferral. If
coverage is what fails, nothing after it prints, and that is the whole finding.

**§16's four readings, one line each, and where a green or a red lands:**

1. **both inside** — every judged row of both forms is inside the §10.1/§10.2 floors (floored by the cascade
   band) with no inadmissible flip: the forms differ only from each other, the dispute is a non-defect, and the
   tile form may be adopted on price with the form-vs-form difference recorded as an observation.
2. **`--attn-tile 0` outside above row 511** — the SHIPPED streaming form is wrong against HF: a shipping-path
   defect in the P7/P8 prefill wave with a witness at last; stop and report, do not route around it.
3. **`--attn-tile 1` outside** — the tile form is wrong and §12's band was mis-sized; `--attn-tile` stays `0`.
4. **both outside, differently** — both are wrong and the >MAXSC region has no correct implementation today.

Two labels that are not readings, so they are not arguments either: a `COVERAGE GAP` line means the instrument,
and the gate deliberately stopped before any numerics — fix the invocation, not the kernel; a
`[DEFERRED-FAIL]` line belongs to neither form, and step 3 above says whether it is a wave-carry finding.
Row 1023 has pin room 12.25 and row 512 has 13.25 (§16's census), so branch 2 or 3, if it fires at row 1023,
fires on a row the reference decided with room — no flip-rule argument is available there.

## 19. §16's four readings, taken (2026-10-14, lock `ADJ`/`ADJ2`): **reading 3 — the tile form is wrong, the shipped streaming form is right**

With coverage enforced, both anchors pass for the **shipped** form and fail for the **tiled** one, on exactly the
streaming rows:

| run | verdict |
|---|---|
| `wide640f --attn-tile 0` | **PASS** |
| `wide640f --attn-tile 1` | **FAIL** — row 636 (streaming): pin top1 421 at **room 11.125**, engine said 393 |
| `wide1024f --attn-tile 0` | **PASS** — `every one of the 1024 pinned prefill rows was judged and is inside the floors` |
| `wide1024f --attn-tile 1` | **FAIL** — rows 512 (**room 13.25**), 518 (8.25), 519 (3.875), 1021 (10.875), 1023 and others |

Three things worth writing down because each one is a decision rather than a result.

**The shipping path is clean.** The shipped streaming form reproduces HF on **all 1024 rows**, including every
row above `MAXSC`, at decision room up to 13.25. So branch 2 of the pre-authorised fix-first rule does **not** fire
— there is no correctness defect above 512 lanes, and nothing queues ahead of P9 on that account.

**The tile form is wrong, not marginal.** Room 13.25 at row 512 is not a rounding argument: a reassociation cannot
move an argmax that far. The failure is structural — the fixed-window partials do not reproduce the reference's
per-row softmax — which means **§12's accepted band was mis-sized for this kernel**, and the honest statement is
that the tiled form must fix its streaming merge before it may be measured again, not that its band needs
re-deriving. `--attn-tile` stays `0` indefinitely, not provisionally.

**And the instrument now does what it was built to do.** `argv carries --prefill-logits 'all'` is asserted on the
argv, the run's own echo is checked against it, and the verdict requires `total == n_rows`, so §17's silent zero-row
coverage is structurally impossible. That is the fourth checker this session made capable of failing, and unlike
the other three it was built *before* it was needed rather than after it lied.

## 20. §19's cause, fixed CPU-side (2026-10-14, worker pass: **no device touched, no timing taken**): the partial slot had no KV-group term

§19 ended with the tile form barred from measurement until its streaming merge was fixed. This section names
the cause, fixes it, and re-establishes the claim on the CPU mirror. **No device was touched and no timing was
taken**; the device commands are handed over in §20.8.

### 20.1 The cause is an address, not a rounding

`attn_core` writes a split row's partial to `((qrow*N_Q + head)*a.split + y)*(DH+2)` — the **full** head
index, so the two KV groups of one row own different slots. §12's tiled form wrote
`((row*ATTN_TILE_G + h)*windows + y)*(DH+2)` — the head-**in-group**, with no KV-group term — and
`attn_combine` had a second branch that read back exactly that layout. Above `MAXSC`, `blockIdx.x = 0` and
`blockIdx.x = 1` of one row therefore wrote the same `(2 + DH)` floats, and the merge read one of them for
**both** heads: half of every streaming row's heads came out as the other KV group's answer, bit for bit, with
the row's own gate applied on top. Two self-consistent layouts, joined by a branch, so nothing could notice.

| site | §12 (what §19 ran) | §20 |
|---|---|---|
| `attn_tile_body` epilogue | `a.part + ((row*ATTN_TILE_G + h)*windows + y)*(DH+2)` | `a.part + ((row*N_Q + head)*a.split + y)*(DH+2)` — `attn_core`'s expression, same file, line 382 vs 702 |
| `attn_combine` | two branches, one per layout, chosen by `a.tile_kernel` | **one** merge; `tile_kernel` decides which CORE ran and must not choose an address |
| `tile_part_floats` | `nrows*ATTN_TILE_G*split*(DH+2)` (half the shipped footprint) | `nrows*N_Q*split*(DH+2)` — the footprint `engine/kernels_attn.cpp:attn_part_floats` already allocates |
| lane cut | `blo = y * MAXSC` (fixed windows) | `blocks_for_row(a, row)` then `span = ceil(n/s_row)` — a CALL into the shipped form's own rule, so the one-pass/streaming boundary cannot drift |

### 20.2 The reproduction had to be built first, because nothing on the box reaches MAXSC

Every pinned prompt is under 512 tokens, so every element §12's gate ever compared came from the one-pass
route — `gate` prints `windows 1` on all five spans, and that line was the whole tell. Built:

* `wide_case` — a **SYNTHETIC-WIDE** wave (named as such in its own output): real pinned L03 stage rows,
  cycled to fill 1024 positions, so geometry, addressing and reduction order are real and the *content* is
  not HF ground truth. It takes a `start` so a wave can be placed exactly on the boundary.
* `decode_steps` — the only HF-pinned rows above `MAXSC` on the box: `wide640f`/`wide1024f`'s `decode7` L03,
  re-shaped as one-row waves at 648 and 1032 lanes, with `attn_out_pre_gate` pinned.
* `PartBuf` — the partial scratch modelled as an **address space** with last-writer-wins and an alias report,
  because the defect was an address; a mirror that hands each row a Python list cannot express two blocks
  writing one slot.

Two defects in the mirror surfaced on the way, and both are the kind that hide behind a green gate:

* `merge_partials` returned `(m, acc)` and its caller divided by 1. On the one-pass route that is harmless
  (the weights were already normalised); on the streaming route it is §07.3's mis-normalisation, and it had
  been sitting in the mirror since §12 because no pinned fixture reaches 512 lanes. The first straddling
  anchor printed `cos 0.449` and that is what gave it up. **A path a gate cannot reach has no tests,
  including tests of the gate.**
* The first version of `compare` modelled the *shipped* side with the layout read from the tile header, so it
  compared two aliased waves and reported the fixed geometry as the one that moved 512 rows. The frozen form
  is the reference and is now always modelled with its own spelling; a check whose reference side is derived
  from the variable under test cannot adjudicate it.

### 20.3 The two candidate causes, priced apart

Same operands, same mirror, four geometries, judged against the published floor `row_1cos_max = 2e-3`
(imported from `engine/gate.py`, not retyped). "group-COLLAPSED" is §12's slot expression, re-injected:

| lane cut / partial layout | rows moved | elements moved | worst row 1-cos | against the floor |
|---|---|---|---|---|
| per-row-slice / group-dimensioned (**as §20 ships**) | 0/1024 | 0/2,097,152 | — | identical |
| fixed-window / group-dimensioned (§12's grouping **alone**) | 64/1024 | 65/2,097,152 | 4.768e-07 | inside, ~4,200x |
| per-row-slice / group-collapsed (the **aliasing alone**) | 512/1024 | 523,929 | 1.137 | OUTSIDE, ~570x |
| fixed-window / group-collapsed (§12 exactly) | 512/1024 | 523,961 | 1.137 | OUTSIDE, ~570x |

So §19's hypothesis was not wrong, it was small: re-windowing a streaming row is a reassociation, worth 65
elements and a worst 1-cos of 4.8e-07 — four decades inside a band it never needed. The aliasing is what
§19's room-13.25 rows were. Three witnesses, all exact rather than banded:

* the collapse writes 4,096 of the 8,192 distinct slots that 8,192 writers ask for — an alias by count;
* every one of the 512 streaming rows comes out with its four head-pairs **bit-identical to each other**, and
  in 500 of them group 0's answer equals group 1's *correct* answer bit for bit (the 12 exceptions are exactly
  the rows where the window grouping also moved a bit, which is itself the proof the causes separate);
* **0** rows at or below `MAXSC` are touched — a one-pass row writes no partial, so it has no slot to steal.
  That is the derived must-not-catch set, and it is why §19's failures were all above 512.

Boundary rows, priced by both forms: row 511 (512 lanes) → 1 block `[(0, 512)]`; row 512 (513 lanes) → 2
blocks `[(0, 257), (257, 513)]`. §12's fixed windows cut the same row at 512/1, which is the grouping
difference in one line.

### 20.4 The geometry, and why `M = 1`

`attn_core` is frozen, so everything below is in `attn_tile_body` and its launcher.

| | §12 | §20 |
|---|---|---|
| block | G=4 heads x M=4 rows, 8 warps, 2 warps per (row,head) pair | G=4 heads of **one** row, 4 warps, **one warp per head** |
| lane range per block | window ∩ widest row of the tile | the row's own slice `[y*span, min(n, (y+1)*span))` |
| K stage | per pair — G identical fetches per lane row | **per block** — one warp stages one lane, all G warps read it |
| V stage | **none**: pass 2 read `a.vcache` at the use site, per pair | staged per block, same shape as K, `static_assert(ATTN_TILE_L >= TILE)` |
| pass 2 | run redundantly by both warps of a pair | run once per head, by that head's warp |
| grid | `(N_KV, split, ceil(rows/M))`, 256 threads | `(N_KV, split, rows)`, 128 threads |

Why no row tiling: with a per-row slice, two rows in one block do not share a lane range — a 1024-lane row and
a 513-lane row both cut into two blocks, but their slice `y=1` is `[512,1024)` and `[257,513)`, a union of 767
lanes, above `MAXSC`. A row tile would size the K/V stage and the score staging by the widest *pair* rather
than by the staging, and would reopen §16's cross-row carry question to buy nothing: the byte lever lives in G,
and G is taken. `static_assert(ATTN_TILE_M == 1)` says so at compile time, and the mirror's
`tile_lane_segments` declines the same request rather than modelling a geometry the kernel refuses.

The bit-identity contract is now stronger than §12's, not weaker: the same five ordered reductions, **and** the
same lane cut, so the tiled form is bit-identical on the streaming route too rather than merely banded. On the
real >MAXSC anchors (`wide1024f` 1032 lanes / 3 blocks, `wide640f` 648 / 2) the two forms differ in
**0 of 2,048** elements each; before the fix it was 1,023 and 1,025. Mirror-vs-pin on those rows is
1529/2,048 at cos 0.99977 and 1790/2,048 at cos 0.99758 — printed and **not judged**: `attn_form_band.py`'s
published rule is that above MAXSC the row is judged by `row_1cos_max` and meanrel is reported, because no
per-row bound exists for a route that reassociates. Judging those columns against `stage_meanrel_max` — which
an earlier version of this line did — invented a per-row bound and turned a legal §07.3 row red.

### 20.5 The traffic claim, re-derived from the staging loops

`stage_copies()` reads which staging loop the header contains instead of assuming one, because traffic is
multiplicity of *global* requests:

| header | staging shape | fetches per lane row per block | re-use |
|---|---|---|---|
| §12, as shipped | `for (int p = w; p < pairs; p += nw)` — per pair | 4 | **1x** |
| §20, as shipped | `for (int rr = s_lo + w; rr < s_hi; rr += nw)` — per block | 1 | **4x = G** |

So §12's booked "M = 4x" was the traffic model's assumption, not the kernel's behaviour, and the header
declared no `s_v` at all: V was a global read per touch, exactly the shipped form's. §20's numbers, from the
same closed-form census (`traffic`): 25.79 GB → **6.45 GB**, 1.000 → 0.250 byte/flop, and the traffic term
108.8 ms → **27.2 ms** at the measured 237 GB/s. The V-per-pair counterfactual — hoist K, leave V where §12
left it — is 16.12 GB / 68.0 ms, i.e. half-staging buys a fifth of the lever, because V is half the bytes.
These are arithmetic on the read census, not measurements.

### 20.6 Negative controls, with derived sets — including one that could not fail

Cases: p300 spans `0-63` and `64-127`, the two real >MAXSC decode anchors, and a **tile-boundary wave**
(`wide_case(start=MAXSC-1)`: rows of exactly 512, 513, 514, 515 lanes). 1,072 (row,head) pairs per control.

| control | caught | blind | why the blind set is blind |
|---|---|---|---|
| `readpast` (mask gone) | 1040/1072 | 32 | a row that already attended `cap` lanes cannot see the mask disappear; 20 reads land on poison |
| `dropcurrent` (§05.7 append forgotten) | 1072/1072 | 0 | — |
| `gqaoff` (whole group reads the neighbour) | 1072/1072 | 0 | — |
| `dimhalf` (128 of 256 dims) | 1064/1072 | 8 | softmax of a singleton is 1.0 whatever the score |
| `gqa_head0` (one head reads the neighbour) | 134/1072 | 938 | only head 0's K/V changes — the shared-stage wiring test |
| `headcarry` (head h seeded from head h-1) | 804/1072 | 268 | a block's first head in a group (`h % G == 0`) has no predecessor to leak from |
| `partalias` (§19's bug, re-injected) | 20/1072 | 1052 | a one-pass row writes no partial, so it has no slot to steal |

Every row is **must-not-hit 0** except where derived, and the whole matrix is derived from geometry: §16's
`tilecarry` is retired because one row per block has no row-to-row state to leak (the host now says so when
asked), and `headcarry` replaces it at the boundary M=1 actually leaves open.

The lesson that cost the most: run over pinned spans alone, `partalias` caught **0 of 1024** with an empty
must-catch set — a negative that cannot fail, in exactly the shape `mlp_gemv` and `mixed_bw` filed. The
streaming controls are only meaningful above `MAXSC`, so the wide anchors and the boundary wave are now
mandatory cases, and `--neg-wide` names them.

### 20.7 Census, compile-only (`hipcc -O3 --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage -c`)

| kernel | VGPRs | SGPRs | scratch | spill | LDS per block |
|---|---|---|---|---|---|
| `attn_core` (frozen, 512 threads) | 53 | 75 | 0 | 0 | 38,944 B |
| `attn_tile_wave` (§12, 256 threads) | — | — | — | — | 20,480 B, and `lds` FAILs it |
| `attn_tile_wave` (§20, 128 threads) | 47 | 52 | 0 | 0 | **12,288 B** |

The descriptor's 12,288 B is exactly the source's own `__shared__` sum — `s_sc` 4,096 + `s_k` 4,096 +
`s_v` 4,096 — so §3's LDS inflation is `attn_core`-specific, and the tile form reserves about a third of the
shipped core's LDS while staging more of the row. `lds` before/after, on the two headers:

```
head.h (§12):  declares s_sc 16.38 kB, s_k 4.10 kB | NOT DECLARED: s_v
               FAIL V is staged and a streaming tile fits it: NO V STAGE AT ALL  -> lds: FAIL
live header:   declares s_sc 4.10 kB, s_k 4.10 kB, s_v 4.10 kB, block LDS 12.29 kB
               ok    reach 4.10 kB of 4.10 kB declared, spellings `(rr - s_lo) * DH`, `slot * DH`
               ok    static_assert(ATTN_TILE_L >= TILE) present                   -> lds: PASS
```

`selftest` is 13/13, including the three `__shared__` sizes derived from their own bracket factors, and
`gate` is green on all five pinned spans (567,296 elements, `tile-vs-shipped 0`, `ref-vs-pin 0`, poison 0;
mirror-vs-pin 194/567,296 at §10.4's ordering floor).

### 20.8 What this pass does **not** claim, and the device hand-off

Not claimed: any timing, any device confirmation, or that M>1 was ever priced on device (it was not, and the
`static_assert` now refuses it). `--attn-tile` stays `0` in the engine default until the four §19 readings come
back for the fixed form. The mirror's bit-identity is between the two mirrored forms; the device gates below
are what decide whether it survives the compiler and the grid.

Locks: `engine/gate.py` and `prototype/attn_decode/run.py` do **not** self-lock (run.py's `--tile-kernel` help
claimed the opposite and has been corrected), so hold the lock across the A/B pair; the self-locking drivers
listed in AGENTS.md are not involved here.

```bash
# 1) the two fixtures §19 used, both forms, one lock per pair so ship and tile cannot be split by another job
tools/gpu_lock.sh acquire "attn-tile-fix-640"; tools/gpu_lock.sh util
.venv/bin/python engine/gate.py --fixture wide640f  --attn-tile 0 --scope form --work tmp/tilefix640/ship
.venv/bin/python engine/gate.py --fixture wide640f  --attn-tile 1 --scope form --work tmp/tilefix640/tile
tools/gpu_lock.sh release "attn-tile-fix-640"
.venv/bin/python engine/attn_form_band.py --ship tmp/tilefix640/ship --tile tmp/tilefix640/tile --prompt-len 640

tools/gpu_lock.sh acquire "attn-tile-fix-1024"; tools/gpu_lock.sh util
.venv/bin/python engine/gate.py --fixture wide1024f --attn-tile 0 --scope form --work tmp/tilefix1024/ship
.venv/bin/python engine/gate.py --fixture wide1024f --attn-tile 1 --scope form --work tmp/tilefix1024/tile
tools/gpu_lock.sh release "attn-tile-fix-1024"
.venv/bin/python engine/attn_form_band.py --ship tmp/tilefix1024/ship --tile tmp/tilefix1024/tile --prompt-len 1024

# 2) the op's own wave gate, including its device-side negatives (§16's refusal is unchanged for `--neg carry`)
tools/gpu_lock.sh acquire "attn-tile-fix-op"; tools/gpu_lock.sh util
.venv/bin/python prototype/attn_decode/run.py --wave --wave-rows 64 --long-fixture p300
.venv/bin/python prototype/attn_decode/run.py --wave --wave-rows 64 --long-fixture p300 --tile-kernel
.venv/bin/python prototype/attn_decode/run.py --wave --wave-rows 64 --long-fixture p300 --tile-kernel --negative
tools/gpu_lock.sh release "attn-tile-fix-op"
```

What each must say: `attn_form_band` must print `bit-identical` on every above-MAXSC row — after §20 the two
forms agree by construction, so a single differing element there is a defect and not a band. §19's five
failing rows (`512, 518, 519, 1021, 1023` at 1024; `636` at 640) must move from OUTSIDE to inside with their
decision room unchanged. If the forms differ on device but not in the mirror, the suspect is the compile, not
the arithmetic: re-take `isa` and diff the descriptor.

### 20.9 Traps, for the index

* **Two layouts joined by a branch are invisible to both sides.** The tile form's slot expression and its merge
  agreed with each other, and the shipped form's agreed with itself, so every per-form check passed and only a
  cross-form comparison could see it — which is why the gate that matters compares the two forms' bits, and why
  `part_layout()` now reads every `a.part +` site out of the header and refuses a row multiplied by anything but
  `N_Q`.
* **A check whose reference side is derived from the thing under test cannot adjudicate it.** The first
  straddling `compare` read the partial layout once and applied it to both sides, and reported the *fixed*
  geometry as the one that moved 512 rows.
* **A negative with an empty must-catch set is not a negative.** `partalias` scored a clean 0/1024 on the
  pinned spans alone; only rows above `MAXSC` can be hit by it, so the cases that make it meaningful are now
  mandatory arguments of the gate rather than an optional extra.


## 21. §20 confirmed on device (2026-10-14, main agent: blocking, lock held, zero subagents) — **4.03×, inside the win line, and the default now flips**

All numbers below were taken by the main agent in the foreground with `tools/gpu_lock.sh` held and **no
subagent active**, in the window 15:32–16:10 UTC. `gpu_lock.sh util` read 91–93 % throughout, which per
`bench/performance-status.md` §6 is not a usable gate for this box (identical clean runs read 24–91 %); the
guarantee being relied on is the zero-subagent foreground shape, not the util figure.

### 21.1 Correctness first, on both fixtures §19 used, both forms

| run | result |
|---|---|
| `run.py --wave --wave-rows 64 --long-fixture p300` (shipped form) | 4112 stage comparisons, **0 failures** |
| same `--tile-kernel` | 4112 stage comparisons, **0 failures** |
| same `--tile-kernel --negative` | **0 failures**, with `carry` SKIPPED with a printed reason (§20.4: one row per block has no row-to-row state; the head→head analogue is the mirror's `headcarry`) |
| `gate.py --fixture wide640f --attn-tile 1 --scope form` | **ENGINE GATE PASS**, 640/640 pinned prefill rows inside the floors (worst cos deficit 4.6100e-04, worst meanrel 3.1672 %, both at row 2) |
| `gate.py --fixture wide1024f --attn-tile 1 --scope form` | **ENGINE GATE PASS**, 1024/1024 rows, same worst rows |
| `gate.py --fixture wide{640,1024}f --attn-tile 1 --scope full` | **ENGINE GATE PASS** on both — the §19 batched-vs-carried wave-width comparisons that `--scope form` deferred also pass with the form held at `1` |
| `gate.py --fixture dec8 / dec64f / dec300f --attn-tile 1` | **PASS / PASS (2 TIE steps, as on the shipped form) / PASS** |

Then the comparison §19 was built for, `attn_form_band.py --ship …/ship0 --tile …/ship1`:

| anchor | rows judged above `MAXSC` | bit-identical | elements differing | argmax moved | pooled over band elements |
|---|---|---|---|---|---|
| `wide640f` (lanes 648) | 136 | **136** | **0** | 0 | cos 1.00000000 (1-cos 1.110e-16), meanrel 0.000e+00 |
| `wide1024f` (lanes 1032) | 520 | **520** | **0** | 0 | cos 1.00000000 (1-cos 0.000e+00), meanrel 0.000e+00 |

§20 predicted this from the mirror and the device agreed: after the slot fix the two forms do not merely sit
inside §07.3's band above `MAXSC`, they produce the **same bits** there. §19's five failing rows
(`512, 518, 519, 1021, 1023` at 1024 lanes; `636` at 640) are gone — not waived, not re-banded: there is
nothing left to judge, because nothing differs. That is a stronger gate than the one §19 applied, and it is
the one §07.3 always wanted.

### 21.2 The measurement: 1024-lane prefill, `attn` row, best-of-3, alternating

`./engine/build/decode_run --ids <1024 ids> --steps 1 --time-waves --gemm library --attn-tile {0,1}`, run
`t0 t1 t0 t1 t0 t1` in one lock hold so no other job can sit between a pair:

| quantity | `--attn-tile 0` (shipped) | `--attn-tile 1` (tiled) | change |
|---|---|---|---|
| `WAVE gran=stage family=attn ms` | 108.340 / 107.672 / 109.352 → **107.67 best** | 27.053 / 26.896 / 26.703 → **26.70 best** | **−75.2 %, 4.03×** |
| `WAVE gran=stage … wall_ms` (whole wave) | 331.588 / 331.716 / 329.549 → **329.55 best** | 247.870 / 240.997 / 246.011 → **241.00 best** | **−26.9 %** |
| `family=gdn_rec ms` (unchanged by this change) | 129.363 / 128.683 / 125.447 | 123.134 / 121.571 / 125.678 | within run-to-run noise |
| `GROUP matmul_ms` | 68.66 / 71.52 / 70.12 | 72.04 / 70.15 / 71.66 | unchanged, as expected |

**The win line (§07.3 admission, `traffic` FALSIFIER pair): attn ≤ 40 ms. It is met at 26.70 ms.** And the
model is what makes this number trustworthy rather than lucky: §20.5 predicted the traffic term at 6.45 GB /
237 GB/s = **27.2 ms**, and the device returned **26.70 ms** — 1.8 % inside a prediction made before the run.
FALSIFIER A (bytes fall 4×, attn does not move) is refuted; FALSIFIER B (40–80 ms, something else binds) is
not reached. §4's "traffic is not the wall" verdict, written when the stage was per-pair and V was never
staged at all, is now superseded: traffic *was* the wall, and the stage was the reason traffic was 4× what
the geometry claimed.

### 21.3 The headline, and where the change does **not** reach

Prefill, `RESULT ms_total` on the same 1024-token prompt, best-of-3: **385.153 → 307.464 ms = 2,660 →
3,330.5 prompt tok/s, +25.3 %.** Today's shipped-side baseline reads 2,659 against the ledger's 2,706 (§6's
spread on an unpinned desktop), so quote the ratio, not the pair, against older rows.

Width sweep, same shape (`--steps 1`, best-of-2 each side), which also bounds the regression risk of flipping
the default for short prompts:

| prompt lanes | `--attn-tile 0` ms | `--attn-tile 1` ms | change |
|---|---|---|---|
| 300 (all rows below `MAXSC`) | 147.274 | 140.151 | −4.8 % |
| 640 | 258.646 | 235.580 | −8.9 % |
| 1024 | 385.153 | 307.464 | −20.2 % |

Every width improved; the gain grows with the share of rows above the staging, exactly as the traffic model
has it. The sub-`MAXSC` gain at 300 lanes is the block-level stage paying for itself even on the one-pass
route, where the shipped form still reads each K row once per (row,head) pair.

**Decode is unchanged, by construction rather than by measurement.** A decode body is one query row, and
`attn_decode_layer` runs the fused single-row form under either flag (`engine/decode_run.cpp:104-109` says so
explicitly), so the ledger's decode ruler is untouched by this commit. The differential confirms it rather
than contradicting it: prefill 1024 + 128 decode steps, `(ms_total(129) − ms_total(1)) / 128` = 12.46 / 12.38
ms/token shipped vs 12.43 / 12.42 tiled — a difference inside the run-to-run noise, and in no case a
regression. The honest headline stays **decode 87.1 tok/s @128 lanes** (ledger, unchanged) and the prefill
ruler is the one that moved.

### 21.4 The default flips to the tiled form, and why that was the pre-registered condition

§20.8: *"`--attn-tile` stays `0` in the engine default until the four §19 readings come back for the fixed
form."* They came back: both anchors, both scopes, the three short fixtures, the op's own wave gate and its
device-side negatives, and the form-vs-form bit-identity on 656 above-`MAXSC` rows. So `decode_run.cpp`
(`int attn_tile`) and `engine/gate.py` (`--attn-tile` default) both move to `1` **in the same commit** — two
defaults that can disagree is the failure mode §19 already filed once, and `gate.py`'s NUMERICS cross-check
(`str(echoed_tile) == str(args.attn_tile)`) is what proves they still agree after the flip.

`--attn-tile 0` remains selectable and is no longer the shipped recommendation; it is the frozen witness the
tile form was proved against, and the A/B stays one flag wide.

### 21.5 What is still open

* **`M > 1` was never priced.** §20.4 refuses it at compile time on the argument that two rows in one block do
  not share a lane range, so row tiling buys nothing once G is taken. That argument is arithmetic, not a
  measurement, and it has not been falsified — but it also has not been tested against the real cost of a
  128-thread block (occupancy is 16 either way per the `isa` census, so the block size is not obviously the
  binding term). If a future pass wants more than 26.70 ms it has to say which term of
  max(27.2 traffic, 30.3 arithmetic-at-hand-GEMM-rate) it is attacking, and 26.70 is already at both of them:
  the remaining headroom in *this* shape is small, and the next win is elsewhere (GDN recurrence, 125 ms, 39 %
  of the wave).
* The CPU suite `speccheck/run_all.py --quick` (11 gates) was re-run after the change and passed, and nothing
  in `speccheck/` consumes this path anyway. `M > 1` was never priced.
