# D30 4.1-a — the F32 widenings the library route never reads: IMPLEMENTED (not committed)

Repo `/home/erokhins/erokhins_projects/qwen3.5-0.8b`, base `f50d50c`. Nothing staged, nothing committed,
nothing touched under `bench/`, `progress/`, `specification/`.

```
engine/decode_step.cpp | 93 +++++++++++++++++++++++++++++++-----------
engine/stage_clock.cpp | 60 +++++++++++++++++++++-----
engine/stage_clock.h   | 10 ++++++
engine/decode_run.cpp  |  7 ++++
engine/gate.py         | 16 ++++++--
5 files changed, 147 insertions(+), 39 deletions(-)
```

---

## 1. The dead set, derived from the readers at each consumer (not from the census's line numbers)

Readers were enumerated by exhaustively grepping the three scratch fields (`grep -n
"sc.norm_f32\|sc.block_out_f32\|sc.mlp_act_f32" engine/decode_step.cpp` → alloc/free/write/hand-branch
only, no other reader anywhere in `engine/*.cpp|h`), then reading each consumer. Line numbers are
**post-edit** (the file grew).

| # | destination | write site (post-edit) | every reader found, with its line | verdict on a library route at `rows > 1` | per body |
|---|---|---|---|---|---|
| 1 | `norm_f32` (mixer-input tile) | `ds:519` | `project()` hand branch at `ds:532/535/538` (q,k,v) and `ds:545/548` (in_proj_qkv/z); **`proj_gemv` at `ds:551` and `ds:554` (`in_proj_b`, `in_proj_a`) read it at EVERY width** | **dead for attention layers, live for GDN layers** | 6 dead, 18 live |
| 2 | `block_out_f32` (attention gated operand) | `ds:659` | the o_proj `project()` F32 operand, `ds:760` | dead | 6 |
| 3 | `block_out_f32` (GDN gated-norm operand) | `ds:739` | the same `ds:760` call | dead | 18 |
| 4 | `norm_f32` (MLP input) | `ds:774` | `mlp_gateup` hand branch, `ds:796` (`mlp_gateup_gemm` takes `sc.norm`, BF16) | dead | 24 |
| 5 | `mlp_act_f32` | `ds:800` | `mlp_down` hand branch, `ds:810` (`mlp_down_gemm` takes `sc.mlp_act`, BF16) | dead | 24 |
| — | `g_raw_f32`/`g_a_f32`/`g_b_f32` | `ds:559` `cast_bf16_to_f32_3` | `gdn_conv`, `gdn_gates` | **live** | 18 |
| — | `norm_f32` row 0 in `finish_token` | `ds:859` | `lm_head_full`, `ds:861` (a hand kernel at every width) | **live** | 1/wave |

**78 of 114 per body = 624 per 4096-lane wave**, which is the census's booked 78/114 and its 56.1 % of the
`cast` bytes. **No destination in the census's dead set turned out to be live**, so the item is its booked
size. The one correction to the census's shorthand: §5-A.1 names four sites (`ds:636,711,739,763`); the
fifth — the mixer-input widening — is dead **only on attention layers**, because the two 16-row
`in_proj_b`/`in_proj_a` calls keep its GDN half alive at any width. That is not a smaller prize, it is a
guard whose polarity matters (§5 below).

Every guard uses the predicate its consumer already uses — `lib_gemm`
(`= kernels::wave_gemm_is_library(sc.wave_gemm, n)`, `ds:458`) — and site 1 adds only `attn_layer`, which
is derived in the same body. There is no new width test anywhere.

## 2. What changed

**`engine/decode_step.cpp`** — five sites guarded, `stageclock::note(kCast)` moved *inside* each guard so
the instrument cannot count a submission that was never made:

* `ds:517` `if (!lib_gemm || !attn_layer)` — mixer-input tile (site 1)
* `ds:657` `if (!lib_gemm)` — `a_gated → block_out_f32`
* `ds:737` `if (!lib_gemm)` — `block_out → block_out_f32`
* `ds:772` `if (!lib_gemm)` — MLP-input norm tile
* `ds:798` `if (!lib_gemm)` — `mlp_act → mlp_act_f32`

The stale note in the row-count-seam comment ("the F32 widenings stay where they are even when this route
makes one of them unread … §7 W5 prices that separately") is replaced with the derived reader list and the
statement that the two derivers moved in this same commit; the `block_out_f32` paragraph that said the
attention mixer "widens that straight into `block_out_f32`" now says what happens on each route, and keeps
the dec8 `cos 0.42` anecdote that is the reason the BF16 tile has to be named per layer type.

**`engine/stage_clock.h`** — `WaveShape::gemm_library`, documented as *the route the run asked for*, not a
per-body resolution, with the reason: `resolve_wave_gemm` gives a one-row body the hand GEMV whatever the
run asked for, and `wave_plan_bill` derives that exception from the shape. Default `false` = hand =
today's bill, so a caller that forgets the field **over**-bills and reads `plan_match=0`.

**`engine/stage_clock.cpp`** — `wave_plan_bill`'s `cast` row is now route-aware:

```
casts_hand     = 3L + 2G + A = 114      (unchanged)
casts_library  = 2G = 36                (the two per-GDN-layer casts whose readers are not GEMMs)
single         = wave_rows <= 1 ? bodies : (last_body_rows == 1 ? 1 : 0)     // derived, as gdn_chunked's is
cast = casts_hand*single + casts_library*(bodies - single) + priced
```

plus a second self-check on the same line: `body1_library_plan_matches_token_plan=1`, which asserts that a
**one-row body on a library route still bills §07's per-token bill** — i.e. the deriver itself proves the
guard cannot fire on a decode step, and if it ever could the flag reads 0 in every run rather than being a
claim nobody can check. (The lambda that does the comparison replaces the old inline conjunction; a stray
`w1.` in it was caught and fixed.)

**`engine/decode_run.cpp`** — `shape.gemm_library = wave_gemm_is_library(o.wave_gemm, kMaxWaveRows)`, the
same question the `GEMM` banner asks at `:1223`, next to the two `gdn_*` shape flags.

**`engine/gate.py` — docs only, and this is a finding: it needed no arithmetic change.** Section (h)'s
`counts` dict is built at `gate.py:7031` from `launch_plan()` but carries only the eight families that
have a launcher counter (`glue, proj, mlp_gateup, mlp_down, gdn_conv, gdn_gates, gdn_gated_norm, attn`) —
**`cast` is not in it and cannot be**, because `engine/launch_stats.h` has no cast counter and a check over
a family with no counter cannot fail. So `wave_expect()` never priced a per-body cast count, and
`launch_plan()`'s `cast` row is the **per-token** bill (115), which section (d) polices against the
`LAUNCH` line and which does not move with the route. The census's §5-A.3 claim that `wave_expect()` "must
stop assuming a wave body owes 114 casts" misreads its `49/24/24/114` — that 114 is `proj`
(5×18 + 4×6), which is also 114 per body. I added no arithmetic and loosened nothing; I edited the two
docstrings to say (a) `cast` there is per token and why it cannot move with the route, (b) the per-body
cast count is `stage_clock.cpp`'s alone to price, and (c) what `49/24/24/114` counts. Verified in the
device logs: `[PASS] launch count cast — engine 115 vs §07 115`.

## 3. Verification I ran

CPU (all passed):

| command | wall | result |
|---|---|---|
| `./engine/build.sh` (×3) | 16 s each | built, no warnings |
| `.venv/bin/python engine/gate.py --selftest` | 7 s | `SELFTEST PASS` |
| `.venv/bin/python engine/gate.py --fixture dec64f --dry-run` | ~2 s | `ENGINE GATE PASS` |
| `hipcc … tmp/gcc/bill_probe.cpp <engine TUs> -o tmp/gcc/bill_probe` + run | 2 s | table below |
| `.venv/bin/python tools/citation_rot_check.py` | — | 26 ROTTED, listed in §6 |
| `.venv/bin/python tools/doc_ref_check.py` | — | no new unnamed-label cite from my edits (`decode_step.cpp` still 1: `§Q10b`) |

`tmp/gcc/bill_probe` (scratch, `tmp/` is gitignored) links the **real** `wave_plan_bill` and prints only
plan fields — a host-only witness, no device:

| shape (tokens / bodies / wave_rows / priced) | hand `plan_launches` | library `plan_launches` |
|---|---|---|
| 1 / 1 / 1 / 1 | **424** | **424** (both with `body1_plan_matches_token_plan=1 body1_library_plan_matches_token_plan=1`) |
| 4096 / 8 / 512 / 1 | **81050** | **80426** |
| 520 / 2 / 512 / 1 | — | 10532 |
| 513 / 2 / 512 / 1 (1-row tail) | 10555 | 10477 |

* 4096-hand = **81050**, exactly the pre-change log's measured `launches=81050 plan_launches=81050`, so
  the hand bill is untouched; 4096-library = 81050 − **624** = 80426.
* 513 vs 520: the difference is one wide body × 78, i.e. the single-row tail correctly owes the **hand**
  count in a `--gemm lt` run.
* one-row body: 424 on both routes, so a decode step's bill is exactly §07's.

Device — correctness only, one at a time under `tools/gpu_lock.sh`, blocking, **no timing taken, no `ms`
figure taken or reported, nothing entered in `bench/`**. `util` recorded at acquire and discounted
(`flash_serve` busy-polls the same iGPU):

| command | wall | util at acquire | result |
|---|---|---|---|
| `.venv/bin/python engine/gate.py --fixture dec64f --gemm hand --binary engine/build/decode_run.base --work tmp/gcc/w-base-dec64f-hand` | 11.6 s | 93 % | rc 0, PASS (2 TIE) |
| `… --fixture wide520f --gemm hand …decode_run.base…` | 23.0 s | 92 % | rc 0 PASS |
| `… --fixture wide520f --gemm lt …decode_run.base…` | 24.7 s | 92 % | rc 0 PASS |
| `… --fixture dec64f --gemm lt …decode_run.base…` | 10.8 s | 92 % | rc 0 PASS |
| `… --fixture wide520f --gemm hand --binary engine/build/decode_run …` | 22.7 s | 0 % | rc 0 PASS |
| `… --fixture wide520f --gemm lt …decode_run…` | 22.5 s | (same hold) | rc 0 PASS |
| `… --fixture dec64f --gemm hand …decode_run…` | 10.5 s | (same hold) | rc 0 PASS |
| `… --fixture dec64f --gemm lt …decode_run…` | 10.4 s | (same hold) | rc 0 PASS |

`decode_run.base` is the pre-change build (kept at `engine/build/decode_run.base` for your A/B; `cp -a` of
the binary built from clean `f50d50c`, which passed its own gates before I edited).

**Bit-identity control — `diff -rq` over the whole work dir** (every `logits*.bin` of the main run **and**
of section (h)'s wide / `--wave-rows 1` / `--wave-rows 5` carry sides and section (j)'s spell A/B):

```
wide520f hand  BIT-IDENTICAL  (2645 logits files, 2.5 G)     wide520f lt  BIT-IDENTICAL (2645, 2.5 G)
dec64f   hand  BIT-IDENTICAL  ( 470 logits files, 447 M)     dec64f   lt  BIT-IDENTICAL ( 470, 447 M)
```

`--gemm hand` is the control the task asked for and it holds. It is **not** the only control: the shipped
`--gemm lt` default is bit-identical too, i.e. the guard *fires* (widths 1, 5, 8, 17 and 512 were all
exercised across the main run and section (h)) and moves no bits, exactly as the census argued it must.
**No band, allowance or floor moved** — the gate tables are untouched in this diff.

## 4. What is left for you (the main agent)

`plan_match`/`counts_match` on a **real** wide wave run is the one witness I could not take inside my
device budget: the wave report is only printed by `--time-waves`, which is a timing instrument. My bill is
witnessed on the host (§3) and the numerics on the device, but the observed-vs-plan equality is yours.

```bash
tag="d30 4.1-a wave plan + paired A/B"; tools/gpu_lock.sh acquire "$tag"
trap "tools/gpu_lock.sh release '$tag'" EXIT; tools/gpu_lock.sh util
cp -a engine/build/decode_run engine/build/decode_run.41a          # the post-change ruler (base = decode_run.base)
IDS4096=progress/bench/evidence/prefill-occupancy-2026-10-16/ids4096.txt   # both exist
IDS4097=progress/bench/evidence/prefill-occupancy-2026-10-16/ids4097.txt
w(){ engine/build/decode_run --model models/Qwen3.5-0.8B --ids "$(cat "$1")" --steps 0 \
     --time-waves --gemm "$2" --gemm-pins all --out tmp/gcc/o; }
w "$IDS4096" lt      # EXPECT  family=cast launches=289 plan_launches=289 counts_match=1
                     # EXPECT  TOTAL     launches=80426 plan_launches=80426 plan_match=1
                     # EXPECT  body1_plan_matches_token_plan=1 body1_library_plan_matches_token_plan=1
w "$IDS4096" hand    # EXPECT  cast 913/913, TOTAL 81050/81050, plan_match=1  (hand bill did not move)
w "$IDS4097" lt      # 9 bodies = 512x8 + a 1-row tail: EXPECT cast launches=403 plan_launches=403
                     # (36*8 wide + 114 for the one-row tail + 1 priced) and plan_match=1
# numerics admission, unchanged bands, both routes:
for f in dec8 dec64f dec64t dec300f mt32f p128 wide520f chunk64b wide1024f; do .venv/bin/python engine/gate.py --fixture $f; done
for f in dec8 dec64f dec64t dec300f mt32f p128 wide520f chunk64b wide1024f; do .venv/bin/python engine/gate.py --fixture $f --gemm hand; done
# optional, cheap, same predicate: `--gemm library` (legacy hipBLAS side) — I ran only `lt` and `hand`.
```

Then §5-A.2's paired A/B (`decode_run.base` vs `decode_run.41a`, both sides inside one lock hold, medians
of 3): the prediction is `cast` **26.71 → ≈11.7 ms** on the `cast` row with `launches=289`, and the wall
question — `wall_ms` must drop ≥10 ms or the record says the bytes were L2-absorbed and 4.1's price is its
launch count only. If `cast` does not fall below ~20 ms, the census's own kill rule applies.

`tmp/gcc_census.py` is scratch and asserts `cast 913 = 114/body` against the **pre-change** logs; against
a post-change log that assertion will (correctly) fail. It is not part of the tree.

## 5. The one thing that went wrong, because it is the best evidence in this handoff

My first build inverted site 1's polarity (`!lib_gemm || attn_layer` instead of `|| !attn_layer`), which
skipped the mixer-input widening for **GDN** layers and starved `in_proj_b`/`in_proj_a`. Both
`--gemm lt` fixtures went red, loudly and immediately:

```
dec64f   lt: step 0 cos deficit 2.79e-03 (allowance 2.00e-03), meanrel 8.29 %, then 4 diverged ids
wide520f lt: prefill row 1596 vs pinned 760, step 0 cos deficit 1.74e-02, meanrel 17.2 %
```

while both `--gemm hand` fixtures stayed green. That is the negative control this item needed: **the gate
set does discriminate the dead-set**, a wrong dead-set does not sneak through a band, and the failure is
attributable to the route that reads the buffer. Fixed, rebuilt, re-run — §3's four green + four
bit-identical rows are from the corrected build only (`engine/build/decode_run` md5 changed; the
`--wave-rows 1/5` sides exercise the hand route at wide prompts, which is why the hand runs also confirm
the guard cannot fire where it must not).

## 6. ROTTED line-form citations — for you to re-point (I did not edit `progress/` or `specification/`)

`tools/citation_rot_check.py` read 0 rotted before my edits and **26 after**, all of them pure line-shifts
caused by inserting lines into two cited files (22 into `engine/decode_step.cpp`, 4 into `engine/gate.py`).
Full list:

```
engine/decode_step.cpp:455  <- engine/results.md:2670
engine/decode_step.cpp:469  <- progress/2026-10-16/ceiling-based-route-ranking-2026-10-16.md:304
engine/decode_step.cpp:471  <- engine/results.md:2670
engine/decode_step.cpp:514  <- prototype/gemm_algo_sweep/README.md:88 ; gemm_algo_sweep.cpp:97
engine/decode_step.cpp:517  <- prototype/gemm_algo_sweep/README.md:89 ; gemm_algo_sweep.cpp:98
engine/decode_step.cpp:520  <- prototype/gemm_algo_sweep/README.md:89 ; gemm_algo_sweep.cpp:98
engine/decode_step.cpp:527  <- prototype/gemm_algo_sweep/README.md:91 ; gemm_algo_sweep.cpp:100
engine/decode_step.cpp:530  <- prototype/gemm_algo_sweep/README.md:92 ; gemm_algo_sweep.cpp:101
engine/decode_step.cpp:677  <- prototype/gdn_rec_isa/gdn_rec_isa_kernels.h:372 ; results.md:337 ;
                               traffic_census.output.txt:10, :16
engine/decode_step.cpp:682  <- prototype/gdn_rec_isa/traffic_census.output.txt:10, :16
engine/decode_step.cpp:687  <- prototype/gdn_rec_isa/gdn_rec_isa_kernels.h:372
engine/decode_step.cpp:722  <- prototype/gemm_algo_sweep/README.md:90 ; gemm_algo_sweep.cpp:99
engine/gate.py:3692         <- progress/2026-10-16/archive-review-2026-10-16.md:321 ; specification/ASSUMPTIONS.md:28
engine/gate.py:4620         <- progress/2026-10-16/ledger-prune-2026-10-16.md:142
engine/gate.py:5946         <- progress/2026-10-16/archive-review-2026-10-16.md:296
```

Re-pointing is cheap in one direction only: `decode_step.cpp` grew by a net 39 lines (+66/−27),
concentrated in 447–517 and later at 648–660 / 728–740 / 761–774 / 792–806, so cites below 447 are
unaffected and those above need the shift that corresponds to their position. `gate.py` grew a net 12
lines (+14/−2) at ~3497 and ~3955.

## 7. Residual risks / notes

1. **`plan_match` on a real library wave is unverified by me** — see §4. If the engine's observed wide-body
   cast count is not 36/body, `plan_match=0` will say so before any number from that run is readable, which
   is the property the task asked to preserve; I did not soften anything to get there.
2. **Route-dependence is now a property of the wave bill.** `p.subs[kCast]` needs `WaveShape::gemm_library`
   and `decode_run` is its only producer; a second driver that forgets it over-bills and reads
   `plan_match=0`, which is the loud direction. Chosen over a second source of truth for the count.
3. **`engine/kernels_cast.cpp`'s header comment** still says "115 launches/token" and names no per-body
   figure. True for the token bill, silent about the wave bill — census §6.4 already carries it as a doc
   fix; I left it out of this commit rather than widen it.
4. **`--gemm library`** (the legacy hipBLAS spelling) shares `wave_gemm_is_library`, so it inherits both
   the guard and the bill; I ran only `lt` (shipped default) and `hand` on the device. One command in §4.
5. **`gate.py` needed no arithmetic change** (§2) — if you want census §5-A.3 corrected on the record, the
   misread is that `wave_expect`'s "49/24/24/114" 114 is `proj`, not `cast`, and `cast` has no launcher
   counter for section (h) to judge.
6. `wide1024f`, the 9-fixture sweep, every paired A/B, every `ms` figure and every ledger line: yours, per
   the brief. No timings were taken by me and none of my wall-clock numbers above is a device measurement.
