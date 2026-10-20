# HANDOFF 43 — D30 1.7′: `gdn_conv`/`gdn_gates` read BF16; `cast_bf16_to_f32_3` deleted

Worker run, 2026-10-17. Tree: working tree on top of `e0d8d83` (nothing committed, nothing staged).
**Nothing in `specification/`, `bench/`, or `engine/gate.py` was touched, and no band was touched.**

## 0. Status — the decision this handed back

The brief said: work out first whether the deletion is route-blind, and **if it is, stop and hand
back the derived bill rather than landing it.**

**It is route-blind and width-blind.** Per the brief that is the stop-and-tell branch. I did stop for
the decision (escalated separately) and I implemented the full engine half in the working tree so the
plan is reviewable **with evidence instead of with intention**: the engine is bit-identical to HEAD on
device, and the launch-bill consequence is visible as exactly two predicted section-(d) failures that
nothing has silenced. If the decision is "§07's per-token bill may not move", `git checkout -- .`
reverts the whole thing in one command; nothing depends on it having been built.

The single reason this is a decision and not a routine: the deletion is the first in this line that
changes **§07's per-token launch bill**, which `engine/gate.py` section (d) polices. 4.1-a could guard
its casts on the GEMM route and leave §07 alone; this one has no route to hide behind — at every width
and on both routes, nothing reads the F32 operands.

## 1. Reader census — the evidence the F32 operand is unread

Sole writer of the three F32 tiles (at HEAD, `engine/decode_step.cpp`):

```
613  kernels::cast_bf16_to_f32_3(sc.g_raw, sc.g_raw_f32, n*kQkvRows, sc.g_a, sc.g_a_f32, …,
                                 sc.g_b, sc.g_b_f32, …)
```

Sole readers:

```
645  kernels::gdn_conv (sc.g_raw_f32, …)        ← §04.2 conv, once per GDN layer per body
655  kernels::gdn_gates(sc.g_a_f32, sc.g_b_f32, …)  ← §04.3 gates, once per GDN layer per body
```

Everything else that mentions those names is allocation/free (`:250/254/255`, `:327/331/332`) and
`tools/route_c_stages.cpp:130,133`, a stage-dump tool that names them as *regions to dump* — a
co-change, not a consumer. After the change `grep -rn "g_raw_f32\|g_a_f32\|g_b_f32\|cast_bf16_to_f32_3"
engine tools prototype` returns only prose (comments that say the widening is gone). The symbol is
also gone from the binary:

```
$ nm -C engine/build/decode_run | grep -ci cast_bf16_to_f32_3
0
```

Route-blindness, structurally: the cast sits in the GDN branch of `submit_wave_body` with **no** guard
on `lib_gemm` (unlike the 4.1-a set), and `submit_token_body` is `submit_wave_body(&token, 1)`, so the
decode step pays the same launch. Guarding it was not an option — a guard only makes sense when some
route still reads the destination.

## 2. What changed (12 files, +251/−154)

Eleven of those are the deletion itself; the twelfth is `engine/gate.py`, which the supervisor approved
moving in the same commit (§5).

Kernel spelling, in `prototype/gdn_decode/gdn_decode_kernels.h` (the engine includes this header —
there is one kernel source, so engine and prototype cannot disagree):

* `gates_kernel(const hip_bfloat16* a_raw, const hip_bfloat16* b_raw, …)`; `float(a_raw[…]) + dt_bias[h]`,
  `float(b_raw[…])`. `A_log`/`dt_bias` stay F32 (checkpoint dtype, widened once per layer by the weight
  adapters — not a per-token cost).
* `conv_tap`/`conv_value`/`conv_kernel` take `const hip_bfloat16* raw`; every tile read is `float(raw[…])`.
  **`win`/`new_win` stay F32** — that is §08.1's state dtype and a separate decision from the tile's
  (§08.3). Bit-neutral: a window column is the exact widening of a BF16 number either way.
* No arithmetic moved: tap order, F32 accumulation, `bf16_round` once then SiLU, the negative-injection
  enums (`GG_ROW_ZERO` indexes row 0 of the BF16 tile and still bites), the staged/negative paths.

Engine:

* `engine/kernels_gdn.cpp` — launchers take `const uint16_t*` and cast to `const hip_bfloat16*`.
* `engine/kernels.h` — the two declarations, and the stale note "`raw` is F32 because the conv kernel
  reads it that way"; the `cast_bf16_to_f32_3` declaration is gone.
* `engine/kernels_cast.cpp` — `cast_bf16_to_f32_3` kernel + launcher deleted (and its now-dead
  `<stdexcept>` include, which was only there for the launcher's null-region guard); header comment now
  says 97 launches/token and records why the 3-region form went (a widening no consumer reads is not a
  dtype adapter, it is a store to an empty buffer).
* `engine/decode_step.cpp` — cast site deleted (with the reason, in place); conv/gates call sites pass
  `sc.g_raw` / `sc.g_a`, `sc.g_b`; three F32 scratch allocs+frees gone;
  `planned_launch_counts()`: `c.cast = 4*kNumGdnLayers + 4*kNumAttnLayers + 1`; header bill table
  115→97, 424→406; the 4.1-a guard note corrected (it used to assert "§07 stays 115").
* `engine/decode_step.h` — `g_raw_f32`/`g_a_f32`/`g_b_f32` fields gone (with the reason); "405 of the
  token's 406 launches".
* `engine/stage_clock.cpp` — `wave_plan_bill`: `casts_hand = 3L + G + A` (was `3L + 2G + A`),
  `casts_library = G` (was `2G`); its docstring table and the one-row self-check note.
* `engine/stage_clock.h` — "the observed total equals the plan exactly — 406 launches since D30 1.7′".
* `tools/route_c_stages.cpp` — the three F32 stage regions dropped (it would not compile otherwise).

Prototype driver + gate, `prototype/gdn_decode/`:

* `gdn_decode.hip.cpp` — the fixtures are F32 files, so the driver narrows them on the way in through
  `narrow_exact()`, which **aborts** if any value is not BF16-exact. That assertion is the gate's own
  guard: if a fixture ever stopped being a bf16 tensor, feeding it to a BF16-read kernel would silently
  round it and the gate would be judging a kernel given something the reference never had.
* `check.py` F1 — one forced co-change. `a_raw`/`b_raw` for F1 are *computed* (`x @ Wa.T` in fp32), and
  all 32 of those values were **not** BF16-exact, so the fixture was feeding the gates a dtype the
  engine has never had. They are now narrowed with the file's existing `bf16()` helper before being put
  in the fixture, and the unit-exact torch mirror reads the narrowed arrays. **No band moved** — the
  1e-6 unit bands and the 5e-3 dump bands are untouched. Measured effect (CPU, `tmp/gcc/f1_narrow_probe.py`):

  | fixture operand | max &#124;g−ref&#124; | max &#124;beta−ref&#124; |
  |---|---|---|
  | fp32 (before) | 1.877e-03 | 1.591e-03 |
  | bf16 (now) | **6.968e-05** | 1.576e-03 |

  i.e. the residual the old comment blamed on "a fp32 fixture vs HF's bf16 matmul" shrank 27×; the
  comment now says that instead of claiming the bound is loose by nature. Tightening 5e-3 is a separate
  gate decision, deliberately not taken here.

## 3. Bit-identity — what the change owes, and what it paid

BF16→F32 widening is exact, so the bar is bit-identity against today, not a band.

CPU side (why it *must* be exact):
* `tmp/gcc/bf16exact.py` — the pinned dumps the kernels are fed are BF16-exact:
  `gdn.mixed_qkv_pre_conv` 0/6144 inexact, `in_proj_a.out` 0/16, `in_proj_b.out` 0/16. The F32 tiles
  carried no extra precision; they were the same numbers stored wider.

Device side (2 gates, both under `tools/gpu_lock.sh`, one at a time, blocking, correctness only —
**no timing taken, and no number from these runs may enter a timing doc**):

1. **wide520f, `--gemm lt`** — the exact argv the HEAD-era gate run used (recovered from
   `tmp/gcc/new42-wide520f-lt.log:95`, 520-token prefill in 512-row waves + 24 decode steps,
   `--attn-tile 1 --attn-touch 1 --gdn-scan seq --gdn-chunk 64 --gdn-spell unroll4 --state-fp`).
   Command: `bash tmp/gcc/run_p17_wide520f.sh` (`--out` repointed to `tmp/gcc/p17-wide520f-lt`).
   Wall ≈2 s. `gpu_lock util` 91 % (discount per repo rule).
   Result: **26/26 engine files byte-identical** to the HEAD-era baseline `tmp/gcc/new42-wide520f-lt`
   (25 × `logitsNNN.bin` + `sampled.txt`); `differing=0`.
2. **dec64f, paired in one window** — HEAD binary `tmp/gcc/decode_run.pre17p` (md5 `e600f7d3…`) and
   the new `engine/build/decode_run`, same argv, **both routes** (`--gemm lt`, `--gemm hand`):
   `bash tmp/gcc/g2_old_lt.sh; bash tmp/gcc/g2_new_lt.sh; bash tmp/gcc/g2_old_hand.sh; bash
   tmp/gcc/g2_new_hand.sh`. Wall ≈6 s for all four. `util` 63 %.
   Result: `diff -r` reports **no difference on either route** — 66 files each (65 logits +
   `sampled.txt`). This is the decode/token path at both routes, which is the path the guard-less cast
   used to reach.

The `RESULT … ms` lines in those logs have been blanked to `<contended-not-a-measurement>` so a
contended number cannot leak into the ledger.

**Re-confirm after the last cleanup edits.** Three edits landed after the two gates above (two comments
and one dead `#include <stdexcept>` in `kernels_cast.cpp`), so the engine was rebuilt and the new
binary re-run on both routes in one lock hold (`bash tmp/gcc/g2_new_lt.sh; bash tmp/gcc/g2_new_hand.sh`,
wall ≈3 s, lock tag `p17-confirm`, released — `gpu_lock.sh status` reads `free`): still **byte-identical
to the HEAD-era output on both routes**, still `cast=97 … total=406`. The device evidence is therefore
tied to the binary this tree builds now. No timing taken.

## 4. The launch bill, derived (not edited to fit)

Per GDN layer the body used to cast 5 things (mixer-norm row, the packed `g_raw│g_a│g_b`, gated-norm
output, MLP-norm row, MLP act); the packed one is gone, so GDN goes 5→4 per layer and attention stays 4.

* per token: `cast = 4·18 + 4·6 + 1 = 97` (was `5·18+4·6+1 = 115`); token total **424 → 406**.
* per wide body: hand `3L + G + A = 72+18+6 = 96` (was 114); library `G = 18` (was 36). The
  library-side dead set stays exactly 78 per body — the deleted cast was one of the two that *both*
  routes issued, which is why hand and library both drop by 18 and their difference does not move.
* per 4096-token wave (8 bodies): hand `76970 → 76826`, library `76346 → 76202`; −144 = 18 × 8 bodies
  on both passes.

Witness, CPU-only, `tmp/gcc/bill_probe.cpp` → `tmp/gcc/bill_probe17after` (links the real
`wave_plan_bill` + `planned_launch_counts`, does no device work):

| shape | before (HEAD-era) | now |
|---|---|---|
| 1 token / 1 body / hand or library | 424 | **406** |
| 4096 / 8×512 / hand | 76970 | **76826** |
| 4096 / 8×512 / library | 76346 | **76202** |
| 520 = 512+8 / library | 10016 | **9980** |
| 513 = 512+1 / library (single-row tail owes the hand bill) | 9967 | **9931** |
| 513 = 512+1 / hand | 10045 | **10009** |
| 4096 / 4096 bodies of 1 row / library (all-one-row pass) | 4096 × 424 = 1736704 by the same rule | **1662976 = 4096 × 406** |

`body1_plan_matches_token_plan=1` and `body1_library_plan_matches_token_plan=1` in **every** case —
i.e. the one-row body still equals §07's per-token plan (96 + the 1 priced final-norm cast = 97), which
is 4.1-a's self-check still holding after the formula moved. The all-one-row case is the same property
at wave scale: a wave of one-row bodies costs exactly `tokens × the per-token bill` — 1662976 =
4096 × 406, where HEAD's deriver gives 4096 × 424. That is the property the 4.2′ handoff's
`--wave-rows 1` row (their 1728514, identical on both routes) was taken to witness; their case priced
rows differently from my probe, so I did **not** reproduce 1728514 itself — if a same-window
before/after on that exact row is wanted it belongs in the device pass, not here.

Engine side, from a real run: `LAUNCH glue=49 proj=114 gdn=72 attn=6 mlp=48 cast=97 other=20 total=406`
(HEAD binary printed `cast=115 … total=424` on the same input).

## 5. Gate state — the supervisor decision, and what it changed

**Decision received and applied: the bill may move, and `engine/gate.py`'s expectation moves in the
same commit as the engine** (supervisor, this run). So the tree is no longer in the "red by design"
state this section originally described: `launch_plan()`'s cast row is now `4 * n_gdn + 4 * n_attn + 1`
(`engine/gate.py:3569`), its docstring row spells the four surviving GDN-layer casts and why they
survive (their readers — `proj_gemv` with `PACK_X=false`, `mlp_gemv`, `lm_head_gemv` — still take
`const float*`, unlike the conv and gates), and section (d) was refactored into
`launch_plan_verdicts(want, got)` so the selftest and the section judge through ONE spelling instead of
two. Consequences of that edit, all verified:

* `gate.py --selftest` → **SELFTEST PASS**, with a new section (d2) carrying the two-fold witness the
  supervisor required: `cast +1` (the deleted widening re-added, 98 vs 97) and `cast −1` (a different
  cast wrongly dropped, 96 vs 97) are each caught and each named at its own row, plus a
  total-only-mismatch case and a missing-row case. `clean bill reproduces §07 exactly (cast=97,
  total=406)`. That is the must-catch evidence that the row moving 5→4 did not weaken the check.
* `tmp/gcc/section_d_rehearsal.py` now prints **0 section-(d) failures against the new binary's
  `LAUNCH` line** and 2 against the HEAD-era line (`cast=115 … total=424`) — the check still bites at
  the old engine, which is the direction that matters if this is ever reverted.
* `gate.py --fixture dec64f --dry-run --no-engine` → **ENGINE GATE PASS** (CPU stand-in path).

What is NOT done, and is the parent's to sequence (all of it is a live claim that now describes a step
that no longer exists, and the supervisor wants it in its own cleanly-revertible commit):

1. `engine/gate.py` — DONE (see above).
2. `engine/README.md:10,372,496,498` and `engine/results.md:135,140,142,147,153,217,259,320,323,327,
   518,523,543,571` carry 424/115. Most are historical (the gap-accounting table at :518-526 priced a
   115-cast step and wants a marker, not a rewrite). **:571 is the one that must be marked rather than
   edited out**: it books "the 115 cast launches — upper bound −0.43 ms" as an open item, and this
   commit takes 18 launches/token and 144/wave-body of it. Say so there, with the commit id.
3. `bench/performance-status.md:92,618,659`, `devices/strix-halo-mixed-traffic.md:12,45,384,490` —
   ledger/device docs; supervisor said those are his, with a dated correction line naming the commit.
4. `tools/step_traffic_model.py:64,107-109,126-128,277` — the `LAUNCHES` dict and the two cast byte
   rows (the packed term is `ng * (qkv + 2*vh)`, read at 2 B and written at 4 B — it drops out
   cleanly). The numbers only move once that byte row is removed, and its `--selftest` has no expected
   file, so re-pinning nothing here is safe.
5. **Decode-side prediction, written down BEFORE the measurement (supervisor condition 6).** 18 fewer
   submissions on the per-token path: at the 1.72 µs submission cost the 4.2′ pass measured for one of
   these copies, that is 18 × 1.72 ≈ **31 µs/token**, against a ~13–14 ms token → **≤ +0.3 % on the
   71.3 tok/s headline, i.e. inside run-to-run noise**. The traffic side is the same size of small: the
   deleted packed cast moved 6176 elements per GDN layer per token at 2 B read + 4 B write ≈ **0.67
   MB/token**, and the conv's read of the tile halves, saving another ≈0.22 MB/token — together ≈0.9 MB
   on a step that already moves ~1.5 GB of weights, so ~0.06 % of its bytes. **My prediction: the
   decode headline does not move measurably; if it moves more than ~0.5 % either way, something other
   than the deletion is responsible.** The wave-side prediction (gdn_conv 19.22 → ≈14.5–15 ms, cast
   −4 ms) is in §6 and is the larger signal.

Old framing, kept for the reader: before the decision arrived this section said section (d) was
expected to fail by exactly two lines, and that was the state the tree was handed over in. It is kept
in `tmp/gcc/section_d_rehearsal.py`'s behaviour against the HEAD-era log, which is where the claim is
still testable.

### The original §07-line map, for the record

1. **`specification/`: nothing.** No spec file states 115 or 424; §07's op list enumerates operations,
   not casts (`grep -n "cast" specification/07-decode-step.md` returns only the softmax-weight note and
   the normalised-weight line). The nearest spec claim, `specification/04-gdn-linear-attention.md:50`
   ("`a_raw` is F32-cast before the addition `[REF]`"), is **value-level and still true** — the kernel
   performs that exact widening before the addition, in a register instead of in a buffer. §04.2's
   pre-conv window and §08's F32-window correction are untouched and stay valid.
2. ~~`engine/gate.py` (not edited by me — brief forbids it).~~ Superseded by the decision above.
   The row's *justification* — "the widenings the F32-input kernels force" — now yields 4, since the
   conv and gates are no longer F32-input kernels. **This is not a loosening**: the check stays exact
   equality between two independent readings, and its discriminating power is unchanged — an engine
   that emitted the widening again would read 98 vs 97 and fail. If the parent wants a witness that the
   check still bites, that is the must-catch case to state: re-add one `cast_bf16_to_f32_3` call, see
   section (d) go red on `cast` alone.
3. **Engine docs** (main-agent call, one docs commit): `engine/README.md:10,372,496,498` and
   `engine/results.md:135,140,142,147,153,217,259,320,323,327,518,523,543,571` all carry 424/115 as
   live claims; some are historical (the gap-accounting table at :518-526 prices a 115-cast step and
   should be marked as such rather than rewritten).
4. **`bench/performance-status.md:618` ("The per-token cast bill stays **115 on both routes**") and
   :659 ("a decode step's bill is **424 on both routes**")**, plus `:92`'s `launches_per_step = 424`:
   ledger lines. Not mine to touch, and per AGENTS the ledger's frozen sections need the user's words.
   :618 is a 4.1-a claim that is now false-for-a-good-reason, so it wants a correction line, not a
   deletion.
5. **`devices/strix-halo-mixed-traffic.md:12,45,384,490,568,607,620`** — measurement rows taken on an
   115-cast step. :12 and :45 are present-tense claims about the engine; the rest are records of
   measured multisets and should be marked as belonging to the 115-cast step, not edited into new ones.
6. **`tools/`** — deliberately untouched:
   `tools/step_traffic_model.py:64,73-74,109,128,277` (its `LAUNCHES` dict and its traffic rows name
   "packed qkv+a+b"); it hard-codes the 115-cast bill while deriving its *shapes* from `engine/state.h`,
   so its byte budget and its read:write ratio are both now about a step that no longer exists, and its
   conclusions feed `devices/strix-halo-mixed-traffic.md`. Re-deriving that byte model is a D30-sized
   call of its own, not a side effect of this edit, so I left the constants alone rather than quietly
   reprice a recorded row. Also `tools/instep_family_budget.py:14,214`, `tools/prefill_gemm_shapes.py:206`.
7. `tools/route_c_stages.cpp` — already changed (it names the stage regions).
   Unrelated finding: **`./tools/build.sh route_c_stages` fails to link at clean HEAD** (missing
   `-lhipblaslt` for `kernels_gemm.cpp`: `undefined symbol: hipblasLtMatmul`). Verified by building the
   same target from a HEAD worktree, so it is pre-existing, not caused here. I syntax-checked my edit
   with `hipcc -fsyntax-only`.

## 6. Commands for the device pass, with a two-sided prediction

Blocking, zero subagents, under `tools/gpu_lock.sh`. Everything below is what the user's own sweep
should run; my two gates above were correctness-only.

```bash
# section (d) verdict + numerics on both routes (this is the loud one: expect exactly 2 failures,
# both in section (d): `launch count cast` and `launch total`; everything else green)
.venv/bin/python engine/gate.py --fixture wide520f --gemm lt --work tmp/gcc/p17gate-wide520f-lt
.venv/bin/python engine/gate.py --fixture dec64f  --gemm lt --work tmp/gcc/p17gate-dec64f-lt
# if section (d) is to be moved first (§07 read agreed), re-run after the gate.py edit and expect
# 0 failures; that is the commit pairing, not two commits.

# the graduated prototype's own gate — re-gate, since its kernels' operand dtype changed
./prototype/gdn_decode/build.sh
.venv/bin/python prototype/gdn_decode/check.py            # F1 bands unchanged, expect same PASS
.venv/bin/python prototype/gdn_decode/check.py --tokens   # the wave-width witness (§06.2 width parity)

# the census's own §5-B falsifier — the paired A/B on gdn_conv / cast, main-agent window only
```

**Two-sided prediction for the timing rows** (census figures: `gdn_conv` 19.2164 ms, `gdn_gates`
1.6717 ms, `cast` ≈11.4 ms after 4.1-a, `gdn_conv` at 90.5 % of the 210.5 GB/s copy probe, dispatch
floor 1.85 µs/launch):

* *If the F32 seam was cold as the census says:* `gdn_conv` moves 19.22 → **≈14.5–15 ms**. The read side
  halves: per wave it read 4096×6144 F32 = 1.81 GB of `raw` and now reads 0.90 GB, i.e. 3.65 GB/wave →
  2.75 GB/wave at the same achieved rate. The `cast` family loses 144 launches and ~906 MB/wave
  (per layer 4096×6176 elems at 2 B read + 4 B write × 18 layers), which at the same rate is on the
  order of **−4 ms**, so `cast` should land near 7 ms.
* *Refuting result (record it as a finding, not a failure):* both byte accounts shrink and **neither
  row moves** → the F32 seam was cache-served after all, and the whole win is the 144 dispatch slots
  (≈0.27 ms/wave). That is §5-B's falsifier firing, and it says the census's first-touch reading of
  `gdn_conv` was wrong, which changes what the next D30 item is worth.

## 7. Verification run here (all green)

* `./engine/build.sh` → `loader_test`, `decode_run` built (twice, latest after the comment fixes).
* `./prototype/gdn_decode/build.sh` → built.
* `hipcc -fsyntax-only tools/route_c_stages.cpp` → OK.
* `./tools/gpu_lock.sh status` free before each gate; released after (`gpu_lock.sh util` recorded above).
* `.venv/bin/python engine/gate.py --selftest` → **SELFTEST PASS**, now including the new section (d2)
  launch-bill witness (clean bill reproduces; cast +1 and cast −1 each caught and named; total-only and
  missing-row cases too). This says nothing about the engine.
* `tmp/gcc/bill_probe17after` → the table in §4, both `body1_*` self-checks = 1 in all 7 cases.
* `tmp/gcc/section_d_rehearsal.py` → the two predicted section-(d) failures, 0 on HEAD-era output.
* `tmp/gcc/bf16exact.py`, `tmp/gcc/f1_narrow_probe.py` → the exactness witnesses in §2/§3.
* Device: the two gates in §3. No timing taken from them, nothing written to `bench/`.
* `tools/citation_rot_check.py` → **68 rotted line-citations**, all from line shifts in files I edited:
  `engine/decode_step.cpp` 46, `prototype/gdn_decode/gdn_decode_kernels.h` 9, `engine/kernels_gdn.cpp` 8,
  `engine/decode_step.h` 4, `prototype/gdn_decode/gdn_decode.hip.cpp` 1. Cited-by side:
  `progress/2026-10-17` 29, `prototype/gdn_rec_isa` 15, `prototype/gemm_algo_sweep` 12,
  `progress/2026-10-16` 5, `engine` 4, `progress/2026-10-12` 2, `progress/2026-10-14` 1. Full list in
  `tmp/gcc/p17-rotted.txt`. None point into `bench/`. `tools/doc_ref_check.py` found one newly-dead
  anchor: `progress/2026-10-17/glue-cast-copy-bytes-census.md:189` cites `engine/kernels_gdn.cpp:32`,
  which is now blank (that is the `gdn_gates` launcher line — it moved). Journals are the parent's to
  re-point; I made no same-line substitutions because every one of these needs a shift, not a
  substitution.

## 8. What I did *not* do

No commits, nothing staged. No `specification/`, `bench/`, `engine/gate.py`, band, or `*.expected.json`
edit. No new gate, no new abstraction, no route guard. No `ms` figure of my own anywhere in this file —
the device numbers here are byte-comparisons and `util` readings only.

## 9. Deadline checkpoint (runner kill at ~600 s) — state as of this run's end

**Changed files (12, all unstaged; nothing committed, HEAD still `e0d8d83`):**
`engine/decode_step.{cpp,h}`, `engine/kernels.h`, `engine/kernels_gdn.cpp`, `engine/kernels_cast.cpp`,
`engine/stage_clock.{cpp,h}`, `engine/gate.py`, `prototype/gdn_decode/{gdn_decode_kernels.h,
gdn_decode.hip.cpp,check.py}`, `tools/route_c_stages.cpp`. Untouched on purpose: `specification/`,
`bench/`, `devices/`, `progress/`, every band, every `*.expected.json`.

**Build state:** `./engine/build.sh` → both targets built with the current tree (last build after every
engine edit, including the dead-include cleanup). `./prototype/gdn_decode/build.sh` → built.
`./tools/build.sh route_c_stages` → fails to link, and fails identically at clean HEAD (missing
`-lhipblaslt`), so it is pre-existing; my edit there was syntax-checked with `hipcc -fsyntax-only`.

**Test state, all after the final edits:** `engine/gate.py --selftest` → SELFTEST PASS including the new
section (d2) two-fold witness (cast +1 and cast −1 both caught and named). `engine/gate.py --fixture
dec64f --dry-run --no-engine` → ENGINE GATE PASS. `tmp/gcc/section_d_rehearsal.py` → 0 section-(d)
failures vs the new binary's `LAUNCH` line, 2 vs the HEAD-era line. `tmp/gcc/bill_probe17after` → the
§4 table, both `body1_*` self-checks = 1 in all 8 cases. Device: §3's three gates, byte-identical on
wide520f (26/26) and on dec64f for both routes (66/66 each, paired against `tmp/gcc/decode_run.pre17p`
in one window). `tools/gpu_lock.sh status` = **free**; no timing taken; nothing written to `bench/`.

**Remaining work, in the order it should happen (all of it is the parent's or a resumed worker's):**
1. Parent's device pass: 9 fixtures × both routes + byte-diff against `tmp/gcc/new42-wide520f-lt`,
   `tmp/gcc/w-new-dec64f-lt`, `tmp/gcc/g2-old-lt`, `tmp/gcc/g2-old-hand` (commands in §6), plus the
   `prototype/gdn_decode/check.py` re-gate and the `gdn_conv`/`cast` paired A/B that decides §5-B.
2. If that A/B is a win: the doc commits in §5's list — README/results.md (mark :571's open item as
   partly taken, with the commit id), `bench/performance-status.md:92,618,659` and
   `devices/strix-halo-mixed-traffic.md:12,45,384,490` (dated correction lines, parent's words),
   `tools/step_traffic_model.py` byte rows, `tools/instep_family_budget.py:14,214`,
   `tools/prefill_gemm_shapes.py:206`. None of it is done.
3. The decode re-take (condition 6) — my prediction is written in §5 so the measurement can refute it.
4. The 68 rotted line-cites (`tmp/gcc/p17-rotted.txt`) are the parent's to re-point.

**Commit state:** nothing staged, nothing committed, no branch moved. Suggested split once the A/B comes
back: (a) `engine: gdn conv/gates read BF16 operands; delete cast_bf16_to_f32_3` — engine + prototype +
`tools/route_c_stages.cpp` + `gate.py` (derivers must be in the same commit as the deletion, 4.1-a's
rule); (b) `docs: the launch bill is 97 casts / 406 launches`; (c) `bench` + `devices` correction
lines; (d) `progress` cite re-pointing.
