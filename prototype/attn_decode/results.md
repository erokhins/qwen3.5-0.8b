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
  other 67 positions sampled; that moves 1 element of `q_rope` by 1 ULP at `p=64`. This is the
  device libm versus torch, and it is a floor, not a bug — `floor_probe.py` F3b measures the rate
  (2 of 14 848 entries, 0.0135 %, in 1 of 116 positions).

## 5. Floors (`floor_probe.py`), each with its sample size

| floor | what was reordered / re-evaluated | sample | result |
|---|---|---|---|
| F1 | score dot: kernel's warp tree (8 dims/lane + 5 shuffle steps) and a pairwise 256-sum, vs torch's matmul order | 15 592 scores → 108 544 output elements | **0 scores flip, 0 elements differ** |
| F2 | softmax *policy*: streaming (max,sum) vs the pinned two-pass | 108 544 elements | 25.9 % differ, 4 491 large — a policy, excluded from the band |
| F2b | softmax *order*: denominator summed forwards vs reversed, same policy | 108 544 elements | 2 differ (0.0018 %), **0 large**, band floor 0 |
| F3 | rope transcendentals re-evaluated f64→f32 vs f32 | 416 cos values | 0 flip |
| F3b | the device's own rope table vs torch's, read back from run.py's outputs | 116 positions, 14 848 entries | 2 differ (0.0135 %), all 1 ULP |
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
- `--split 4` under the streaming form: same gate criteria as the single-block row (3rd row above).
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
