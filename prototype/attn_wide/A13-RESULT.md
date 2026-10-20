# A13 result — the llama-shaped round lands: **−27.4 % warm at 16k, −35.7 % at 32k**, 18/18 gate PASS, no band moved (2026-10-20)

`LLAMA-PORT-PLAN.md` was the funding note; this is what came of it. Knob `--attn-fattn-wide 2`
(`engine/attn_wide_llama_kernels.h`, 486 lines — **the shipping default** as of 2026-10-20, see "State" below).
Instruments: `a13_byte_identity.py`,
`a13_longctx_identity.sh`, `a13_identity_metrics.py`, `a13_ab_timing.sh`, `a13_warm_headline.sh`; evidence
in `evidence/a13_*`. Main agent ran every device number in this file, blocking, zero subagents, one
`gpu_lock` hold per leg, canaries asserted on every single run.

## Headline, warm (§51 convention: `WAVEPASS gran=burst wall_ms`, median of 3, one process per width)

| width | knob 0 warm | tok/s | **knob 2 warm** | **tok/s** | Δ |
|---|---|---|---|---|---|
| 512 | 48 ms | 10,667 | 43 ms | **11,907** | **−10.42 %** |
| 1,024 | 94 ms | 10,894 | 87 ms | **11,770** | −7.45 % |
| 4,096 | 422 ms | 9,706 | 359 ms | **11,409** | **−14.93 %** |
| 16,384 | 2,674 ms | 6,127 | 1,942 ms | **8,437** | **−27.37 %** |
| 32,000 | 7,960 ms | 4,020 | 5,116 ms | **6,255** | **−35.73 %** |

The knob-0 column reproduces §51's published warm figures (10,637 / 11,276 / 9,790 / 6,125) within 0.3 %,
3.4 % and 0.03 % at 512 / 1,024 / 16,384 — so the baseline is the same instrument, not a new window.
Cold `RESULT ms_total` moved the same way (16,384: 2,782 → 2,032 ms, ranges `[2782,2855]` vs `[2032,2094]`,
disjoint; 32,000: 7,957 → 5,112 ms, `[7957,7996]` vs `[5112,5121]`, disjoint; 4,096 −9.11 %, 1,024 −3.35 %,
512 −3.01 %), and knob 1 (A12) sat at −1.0…−1.6 % throughout, which is the number §56/§57 established for it.

**Against llama.cpp.** §51's traced table recorded 16,384 lanes at ours 2,721.90 vs llama 2,126.23 ms
(1.28× behind) and 32,768 at 8,328.23 vs 5,839.50 (1.43× behind). Knob 2's warm wave is **1,942 ms at
16,384** and **5,116 ms at 32,000**: **8.7 % faster than their 16k figure and 12.4 % faster than their 32k
figure** — and theirs was taken at 32,768 lanes, so the 32k margin is if anything understated. Cross-window
caveat, stated because §57 makes it a rule: llama's numbers were taken in §51's window, not paired against
this one; ours are paired and interleaved within a single hold.

## Why it is that big, and why that is not a surprise

§51 attributed the whole gap to attention and quantified it: at 16k our attention cost **+751 ms/pass more**
than llama's while the total gap was **+596 ms**; at 32k **+2,815 ms** against a total gap of **+2,489 ms**.
Knob 2's measured saving is **−732 ms at 16,384** and **−2,844 ms at 32,000** (warm). **The port removed
97.5 % and ~100 % of exactly the excess §51 named** — with the caveat that §51's 32k excess was measured at
32,768 lanes and ours at 32,000, so treat the second ratio as ±5 % rather than as a confirmation. §58 then predicted the size from the other direction:
the amortization gap was 2.09× against a measured rate gap of 1.9×, and the ported round bills 20.56
non-matrix slots per matrix slot where A12's round bills 24.05 (round-scoped; the mandated widest-span
instrument reads 67.09 vs 110.91, but the worker correctly flagged that span as contaminated — it encloses
both round instantiations and the whole epilogue).

So the honest characterisation is: **attention is at llama's rate now, and nothing else moved.** GEMM was
already 0.88–0.90× (we win), GDN is untouched. Any remaining long-context difference versus llama is no
longer an attention question.

## Correctness

| check | result |
|---|---|
| byte-identity vs knob 1, 5 pinned geometries (`a13_byte_identity.py`) | **1,960 files, 0 differing**, both sides asserting their own `form=`/`attn_fattn_wide=` |
| byte-identity vs knob 1 at 4,096 lanes (full free run) | **bit-identical**, both logits rows `ndiff=0`, all 4,099 sampled lines equal |
| at 16,384 / 32,000 lanes | **not** bit-identical, and quantified (`a13_identity_metrics.py`): **1−cos 1.41e-05 and 6.43e-05**, argmax SAME, top-5 SAME on every row, **the entire 16,385 / 31,826-token greedy continuation agrees token-for-token** (the two differing lines differ only in a printed max-logit value, 1.5e-03 relative) |
| launch geometry at 16k, knob 1 vs knob 2 | **all 32 wave lines identical** — same `grid=(2,1,16)`, same `wide_rows=512/512`, same `lanes_last` progression 512…16384, same `split` — so no coverage was dropped |
| engine gate, derived population of 18 fixtures × 3 arms (`run_a12_admission.sh` with `ARMS="0 1 2"`) | **knob 2: 18/18 PASS. knob 1: 18/18. knob 0 (ships): 17/18**, the red being `chunk64b`, pre-existing and witnessed against a pre-change binary in `A12-ADMISSION.md` |
| cost side the worker flagged | spill 163 → 920 VGPR, private segment 176 → 720 B/thread (inside A12's 1,024 B budget, visible as `scratch=720 SPILL`). The slot cut pays for it by a wide margin at every width. |

**No band was widened, and none needed widening** — the user's pre-authorisation was not spent. The only gate
change is that the knob accepts `2`: `choices` gains `"2"`, the echo assertion maps `2 → wide_llama` from an
explicit dict (so a future fourth form cannot fall into an `else` and be judged as the shipper), and
`_is_wide_side` accepts `wide_llama` as a wide core, which routes its pairings onto the **already-adopted**
wide-scoped pooled line (2.500e-04) exactly as knob 1's are — measured 2.028e-04 on `smoke`, 19 % inside.

## Corrections to the funding plan, which said "do as they did"

The worker's spec-check found the plan had read the source but not the preprocessor:

* llama's swizzled `ldmatrix` path and its `cp.async` multi-stage pipeline are both **CUDA-only**
  (`TURING_MMA_AVAILABLE` / `CP_ASYNC_AVAILABLE` require `!GGML_USE_HIP`). On RDNA3 llama **pads rather than
  swizzles, and stages synchronously**. So plan items 1 and 2 were **our** engineering, not ports — and they
  are the two items that won.
* item 3 (packed `v_pk_mul_f16` rescale) needed no port: LLVM was already emitting 64 per round in both
  forms.
* item 4's "add a sentinel instead of selecting" is a pessimisation against our single scalar causal bound;
  what landed instead is the `template <bool kMasked>` full/tail split — predicate-free steady-state rounds,
  bounds kept for the tail.
* item 5 landed: `SCALE = 2⁻⁴` folded into Q at staging. Stated limit: a Q element under ~2⁻¹⁰ falls into
  the F16 subnormal range after the fold, so the claim is exact-except-subnormal, not exact.
* `g3_isa_bill.py`'s "widest backward branch" is **not the round** for either wide form, which is why both
  readings are reported and why the round-scoped figure is the one to quote.

## State and what is left

**The default is now 2.** Shipped 2026-10-20 in the same commit across `decode_run.cpp`'s option struct and
`gate.py`'s `--attn-fattn-wide` default, because a half-flipped pair makes the gate silently judge a kernel
the engine does not ship (the trap this repo already records). `AttnLayerArgs`' **struct** default stays 0 on
purpose and says why in its comment: it is the library default that only reaches callers which leave the
field unset — the standalone rigs, whose published baselines were all taken on the inert form — while
`decode_run` assigns the field unconditionally.

Witnessed as the default, not as a knob (`a13_default_admission.sh`, `a13_default_warm_headline.sh`, both
passing **no form flags** and asserting the run's own `attn_fattn_wide=`/`form=`):

| width | warm ms (median of 3) | **prompt tok/s as shipped** | §51 published | Δ |
|---|---|---|---|---|
| 512 | 44 ms `[44.31, 44.61]` | **11,636** | 10,637 | **+9.4 %** |
| 1,024 | 85 ms `[85.09, 86.16]` | **12,047** | 11,276 | +6.8 % |
| 4,096 | 370 ms `[361.8, 373.5]` | **11,070** | 9,790 | **+13.1 %** |
| 16,384 | 1,918 ms `[1894.3, 1922.7]` | **8,542** | 6,125 | **+39.5 %** |
| 32,000 | 5,200 ms `[5158, 5295]` | **6,154** | 3,924 @ 32,768 | ~+57 % |

**Default admission: 18/18 PASS, zero no-verdicts** — and one of those 18 is `chunk64b`, the red the shipping
path carried since before A12 (a batched-vs-one-row token flip at 2.925 BF16 ULP of room, `A12-ADMISSION.md`).
It passes on this form, which is worth saying out loud: the flip did not merely avoid the old red, it removed
one. The gate's own echo cross-check is what proves the routing on every one of those runs, so a default that
had landed in only one of the two binaries could not have produced this table.

Still open, untouched by this work: the `smoke` wide-scoped margin (measured 2.028e-04 against the 2.500e-04
line — the two wide forms sit 19 % under their ceiling, so any future wide-form change has very little
headroom), and the question of whether §51's llama-facing attribution table should now be re-traced, since the
attention row it records (ours 1,518.91 ms/pass at 16k) is no longer what the engine does.
