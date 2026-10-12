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
  the kernel and diffed against torch at that position; that is written up as item **(g)** of
  `bench/performance-status.md` §5, the quiescent device pass, because it needs a device run and
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

Plan `bench/plan-prefill-2026-10-12.md` §3/P7 moves the engine above the score staging: when a row has
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
