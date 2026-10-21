# S1 item (1) — delete the 97 `cast` launches by folding the widen into the consumer

Handover from the worker that implemented it. Written before the device A/B, which is the main agent's
(`ab_item1.sh`, below). Nothing here is committed; `engine/performance-status.md`, `README.md`,
`engine/generation-limits.md` and everything under `progress/` are untouched, and no millisecond figure
in this file is a measurement this route is entitled to bank — the two that appear are quoted from the
ledger or supplied by the parent's review, each labelled where it comes from.

Status: implemented behind one flag, default = shipped behaviour, CPU-equivalence proved, `dec8` gate
green on both arms, `wide4096f` green with the fold **off** (main agent still owes the fold-on half, see
§6), A/B script and canaries written and dry-run against a HEAD-built BASE.

---

## 1. The population: which 97, and what each one is for

`prototype/launch_price/census.py` bills `cast: 97 launches` and `engine/generation-limits.md` §1 says
the same; neither says which 97. `census_sites.py` derives it from source — the call sites and their
multiplicity from `engine/decode_step.cpp`, the widths from `engine/state.h`, the bytes cross-checked
against `tools/step_traffic_model.py`'s own `cast` rows — and refuses on any mismatch:

```
.venv/bin/python prototype/cast_fold/census_sites.py            # also --selftest (7 broken-source cases)
```

| site | src → dst | launches/token | read MB | write MB | consumer (who reads the F32 copy) | classification |
|---|---|---:|---:|---:|---|---|
| `norm_for_input_proj` | `sc.norm → sc.norm_f32` | 24 | 0.049 | 0.098 | `proj_rows` (grouped, `engine/proj_group_kernels.h`) on the shipped `packed` plan; else `project()`'s hand branch → `proj_gemv` | foldable now (grouped kernel already stages `x`; needed a template arm, not a signature change) |
| `attn_gated_for_o_proj` | `sc.a_gated → sc.block_out_f32` | 6 | 0.025 | 0.049 | `project()` hand branch → `proj_gemv` (`o_proj`) | foldable now (same kernel, `IN` arm) |
| `gdn_out_for_out_proj` | `sc.block_out → sc.block_out_f32` | 18 | 0.074 | 0.147 | `project()` hand branch → `proj_gemv` (`out_proj`) | foldable now |
| `norm_for_mlp_gateup` | `sc.norm → sc.norm_f32` | 24 | 0.049 | 0.098 | `mlp_gateup` hand form (the library form already reads `sc.norm` itself) | needs a kernel signature change → done in an engine-owned sibling header |
| `mlp_act_for_mlp_down` | `sc.mlp_act → sc.mlp_act_f32` | 24 | 0.172 | 0.344 | `mlp_down` hand form | same |
| `final_norm_for_lm_head` | `norm_row_ptr → sc.norm_f32` | 1 | 0.002 | 0.004 | `lm_head_full` | same |
| **total** | | **97** | **0.371** | **0.741** | | **6/6 foldable** |

Two more `cast_bf16_to_f32` calls exist in `prepare_weight_adapters()` (`dt_bias`, `conv1d.weight`, at
`decode_step.cpp:405/419`). They run **once per process**, are not in the 97, and are not folded.

Every consumer was read before it was classified; none of the six is in the "not foldable" class, because
every one of them ends in a dot-product loop that already stages its `x` operand (shared memory or
registers) and widens it at the use site. The two families whose prototype kernels take a bare
`const float*` (`mlp_gemv`, `lm_head_gemv`) are the ones that needed a new entry point.

## 2. What the fold does, and where it is

A BF16 widened to F32 is the same real number — same value, no rounding, no reassociation — so the
consumer can hold the BF16 tile and widen at the use site instead of reading a buffer somebody else
wrote. That is the whole change: the value is identical, the launch disappears.

Flag: **`QWEN_CAST_FOLD=1`** (env, default `0`). It is an environment variable rather than a
`decode_run` option for one reason: `engine/gate.py` builds the engine's argv itself and gate files are
out of bounds for this route, so a new option could not be passed through to the arm under test without
editing the gate. The env reaches the engine by subprocess inheritance, so `QWEN_CAST_FOLD=1
engine/gate.py --fixture …` judges the folded binary with no gate change at all. This is a deliberate
exception to the repo's option style, recorded here so nobody "fixes" it by adding a flag and leaving the
gate unable to reach the fold.

Engine changes (all in `engine/`; `prototype/<op>/*_kernels.h` were **not** edited — the folded variants
are engine-owned siblings that `#include` the prototype headers):

| file | change |
|---|---|
| `engine/decode_step.h` | `DecodeScratch::cast_fold`; `planned_launch_counts(bool cast_fold = false)` |
| `engine/decode_step.cpp` | `fold_cast = sc.cast_fold && (n == 1)`; the six sites guarded; folded consumer calls; the plan row |
| `engine/proj_group_kernels.h` | `proj_group_rows<IN, XBF16>` — `x` becomes `const void*`, staged as BF16 and widened at the use site when `XBF16` |
| `engine/mlp_xbf16_kernels.h` | new: `mlp_gateup_xbf16`, `mlp_down_xbf16` |
| `engine/lm_head_xbf16_kernels.h` | new: `lm_head_gemv_full_xbf16<true>` |
| `engine/kernels.h`, `kernels_proj.cpp`, `kernels_mlp.cpp`, `kernels_lmhead.cpp` | the five new launchers (`proj_rows_xbf16`, `proj_gemv_xbf16`, `mlp_gateup_xbf16`, `mlp_down_xbf16`, `lm_head_full_xbf16`) |
| `engine/stage_clock.{h,cpp}` | the plan-in-force: `Clock::set_plan_cast_fold`, `WaveShape::cast_fold`, `wave_plan_bill` fold rule, new self-check `body1_castfold_plan_matches_token_plan` |
| `engine/decode_run.cpp` | `Options::cast_fold`, the `CASTFOLD` echo line, `step_bill=`, the clock/shape wiring, usage text |

The fold is a **one-row rule** (`sc.cast_fold && (n == 1)`), because that is the body a decode step is.
A wide prefill body folds only `finish_token`'s final-norm widening — the site whose consumer is not
row-batched — so **TTFT is not what this route prices**, and the `CASTFOLD` line says so
(`full_wide=final-norm-only`) instead of leaving it in a comment.

## 3. The launch-plan contract (the parent's review item, fixed)

Deleting 97 submissions is a change to §07's per-token bill, and a first cut left the declared bill at
340 while 243 were submitted, so the engine's own self-check went red on every folded run. Fixed by
moving the declared bill in the same change as the fold, in both places that derive one:

* `planned_launch_counts(bool cast_fold = false)` — the `cast` row is `cast_fold ? 0 : 4·18 + 4·6 + 1`.
  Default argument, so the `LAUNCH` line keeps printing §07's shipped bill (`cast=97 total=340`) in every
  run whatever the flags say. That is not an oversight: `engine/gate.py` section (d) re-derives that
  table in Python, and an echo that moved with the engine's own flags would make the cross-check a
  tautology — the `--attn-tile` / `--proj-plan split` trap. `gate.py` was not touched and is green on
  both arms.
* `stageclock::Clock::set_plan_cast_fold(bool)` — the plan-in-force for a step clock, mirroring
  `set_plan_proj`: the clock has no shape to derive the fold from, so the driver states it. A wave clock
  gets it through `WaveShape::cast_fold`, and `wave_plan_bill` applies it as a rule about the body: a
  one-row body loses all six sites, a wide body loses only `priced` (the `finish_token` widening), both
  derived from `tokens`/`bodies`/`wave_rows` rather than trusted to the caller.
* New self-check `body1_castfold_plan_matches_token_plan` on the `WAVE … TOTAL` line: the folded wave
  bill against `planned_launch_counts(true)`'s folded step bill. It compares the **folded** bills on
  both sides, so it is not decoration.

Measured lines (device, `--time-stages`, 4 prompt ids + 2 steps — geometry stated because these are
plan fields, not prize figures):

```
flag off: CASTFOLD on=0 plan=97 folded=0  kept=97 step_bill=340 | launches_per_step=340 plan_per_step=340 plan_match=1
flag on : CASTFOLD on=1 plan=97 folded=97 kept=0  step_bill=243 | launches_per_step=243 plan_per_step=243 plan_match=1
```

`plan_match=1` on both arms, and 340 with the flag **off**, which is what keeps the canary meaningful.
With the fold on there is no `STAGE gran=stage family=cast` row at all (the row is printed only if a
widening was submitted), and `mlp`'s `segments_per_step` reads 24 instead of 48 — two observations of the
deletion made by the instrument rather than by the launcher.

Both new checks were **witnessed to have teeth** by a temporary break, then reverted:
`planned_launch_counts` forced to answer 97 regardless of the flag reads `plan_match=0` and
`body1_castfold_plan_matches_token_plan=0`; and severing `plan_bill(plan_proj_, plan_cast_fold_)` makes
`prototype/cast_fold/check_equiv.py` exit 1 naming that exact defect on CPU, before any device run.

## 4. CPU equivalence (this is the part CPU can decide)

```
.venv/bin/python prototype/cast_fold/check_equiv.py
```

prints, and refuses on an empty parse of any of it:

```
[A] widening exactness: 65536 of 65536 BF16 patterns checked (NaN-as-NaN: 254)
[B] per-site operand equality: 403968 comparisons over 6 sites
SITES_CHECKED=6 LAUNCHES_COVERED=97 COMPARISONS=403968
EQUIV PASS … EQUIV OK: 6 sites / 97 launches …
```

`equiv_check.cpp` does two things. Over **all 65,536 BF16 bit patterns** it checks
`float(hip_bfloat16)` against the bit-shift oracle `bits << 16` (NaN treated as NaN-vs-NaN), i.e. the
widening is exact, not approximately exact. Then per site it compares the shipped path (write the cast
buffer, read it back) against the folded path (widen at the use site) at **every tile index**, sweeping
every BF16 value through position 0 — 403,968 comparisons — so "the folded site reads the value the
deleted launch wrote" is element-wise, not asserted. The population assertion is in the same file:
`sites checked == 6` and `launches covered == 97`, equal to what `census_sites.py` publishes, or it fails.

What CPU **cannot** decide, and the file says so in its last line: the device codegen of `float(bf16)`
inside a dot's expression tree — contraction and association. That is what §5's byte diffs are for.

## 5. Device evidence taken by a worker (correctness only, no prize figures)

| command | result | wall |
|---|---|---|
| `.venv/bin/python engine/gate.py --fixture dec8` | `ENGINE GATE PASS` | 9 s |
| `QWEN_CAST_FOLD=1 .venv/bin/python engine/gate.py --fixture dec8` | `ENGINE GATE PASS` | 8 s |
| `.venv/bin/python engine/gate.py --fixture wide4096f` | `ENGINE GATE PASS` | **242 s — over my ~60 s budget, disclosed** |
| `QWEN_CAST_FOLD=1 … gate.py --fixture wide4096f` | **not run** (would double the breach) — handed back, §6 | ~240 s expected |
| BASE (HEAD, built in a scratch `git worktree add`) vs the folded binary, `ids128.csv` + 16 steps | `sampled.txt` identical; **17/17 logits rows `cmp`-identical** | ~1 s/run |
| same at `ids4032.csv` + 32 steps (4,000 prompt + 32) | `sampled.txt` identical; **33/33 logits rows `cmp`-identical** | ~2 s/run |
| BASE vs new-with-flag-off, both widths | `sampled.txt` identical (the rebuild alone moved nothing, so the flag is the only variable) | |
| `--time-stages`, `--time-waves`, both arms | plan lines above; `STAGEPARITY tokens_identical=1 logits_bitidentical=1` on every pass | |

So the fold is bit-identical to HEAD on device at both the short and the headline width, including every
logit row, not merely decision-identical. The parent's own review independently reported
`wide4096f` bit-identical at 4,104 rows with burst and stage parity green.

The stage-total comparison is **not** a prize figure and must not be quoted as one: folding also deleted
121 instrument segments and therefore 121 per-segment syncs (`syncs_per_step` 305 → 184 in the runs
above), so a serialized stage total falls mechanically whatever the wall does. The only figure with a
claim on the wall is `STAGEPASS gran=burst`. The pair the parent supplied from a contended worker window
— **9.7525 → 9.5326 ms/step**, i.e. −0.22 ms — was taken at 4 prompt ids + 4 steps (8 lanes), not at
4,032, in a window this session was sitting on; it is indicative only, it is not ledger material, and it
is not the number the pre-registered lines are read at.

## 6. Handed back to the main agent

```bash
# the fold-on half of the wide admission (~4 min each, main-agent-only)
QWEN_CAST_FOLD=1 .venv/bin/python engine/gate.py --fixture wide4096f
# and, to have both sides in one window:
.venv/bin/python engine/gate.py --fixture wide4096f

# the paired wall A/B — takes the GPU lock itself, run BLOCKING with zero subagents
bash prototype/cast_fold/ab_item1.sh
REPS=5 GATE_FIXTURES="dec8" bash prototype/cast_fold/ab_item1.sh   # a shorter admission set
```

`ab_item1.sh` leaves a scratch BASE worktree behind; `rm -rf tmp/cast_fold/ab/base && git worktree prune`
when done with it (the script prunes before it builds, so a re-run is self-cleaning; `git worktree
remove --force` is deliberately not used — it can delete uncommitted work and this box's safety net
refuses it).

## 7. The A/B: pre-registration and canaries (`ab_item1.sh`)

Pre-registered **before** any timing, quoted from `engine/generation-limits.md` §7:

* saves **≥ 0.24 ms** on the step wall → the ~3.8 µs/launch residual is real, and `glue`/`copy` (~116
  more tiny launches) become a funded route;
* saves **≤ 0.05 ms** → launches are free when pipelined, the small-op tail is an instrument artefact,
  and the route closes;
* between → reported as neither.

Lines are on the **4,032-lane decode step wall** (4,000 prompt + 32 steps), median of paired interleaved
reps, each arm reported with its spread, never a best. **Scaling:** the script scales the lines by
`folded/plan` read out of its own runs, not by a remembered number. For this build that scale is
`97/97 = 1.000`, so **the lines stand as written and nothing is scaled down** — all six sites fold for a
one-row body. The scale factor exists because a future refactor that folds fewer sites must not be read
against a win line it no longer earns. At 128 lanes there is **no** line: the S1 brief puts short widths
out of reach of this route in advance, so FLAT THERE IS A PREDICTED PASS and it is reported beside the
headline, never instead.

Structure, following `prototype/attn_decode_wide/ab_d3_engine.sh`: `export LC_ALL=C`; BASE built from HEAD
in a scratch `git worktree add` (never the touched working tree) and asserted to differ from NEW by
bytes; CPU admission (census + equivalence + the §5 wiring checks) before anything else; numerics
(`gate.py`, three arms per fixture) before timing, with `NO-TIMING` and no millisecond printed on any
red; `sampled.txt` byte-identity at **both** 128 and 4,032 lanes against both BASE and the
flag-off arm of NEW; one `tools/gpu_lock.sh` hold acquired and released inside the one shell block, with
an `EXIT` trap; every parser aborting on an empty or non-numeric field.

Behavioural canaries, asserted inside the window before any millisecond is spent — the discipline AGENTS
records after a D30 A/B timed a stale binary as `new` and reported a clean null while every in-binary
parity field stayed green:

1. **the flag took effect, two ways.** `CASTFOLD on=… plan=… folded=… kept=… step_bill=…` exists only in
   the new binary (BASE printing one is treated as "BASE IS NOT HEAD"), `plan=` must equal the untouched
   `LAUNCH cast=` field, and with the knob on `folded=plan kept=0` — with the knob off `folded=0
   kept=97`. The second way is the instrument's own observation rather than a field the launcher
   volunteered: `STAGE gran=stage family=cast` present on BASE and flag-off, **absent** when folded; and
   `launches_per_step` = 340 (BASE/flag-off) vs 243 (folded), each with `plan_match=1`.
   One honest limit, recorded in the script: BASE is HEAD and HEAD has no `CASTFOLD` line, so the
   `folded=0 kept=97` half of that canary is asserted on the **flag-off arm of the new binary** — the arm
   whose whole job is to be HEAD's behaviour — while BASE is asserted on `LAUNCH cast=97`, the absence of
   the line, and `plan_match=1`.
2. **the exercised path**, on every timed run: `PREFILL tokens=` equals the id **file**'s field count
   (`--ids-file` splits on commas; a nominal width that is not the field count is the E2BIG/one-token
   trap), `STAGEPARITY … tokens_identical=1 logits_bitidentical=1`, and the fixtures come from
   `prototype/attn_decode_wide/mkids_d3.py` so "4,032 lanes" means what the ledger's row means.
3. **what a null would mean.** The consumer-side staging has a measured precedent against it in
   `engine/results.md`'s `cast` row, which priced the projection family's `PACK_X` staging as a net loss
   (the number is there, quoted) — measured on `proj_gemv_tokens` at rows/block 32, before route D1's
   grouped kernel, and it halved the LDS per block while it did. So a null or a negative is ambiguous
   between "launches are free" and "the staging costs more than the launches were worth", and the script
   prints both arms' family stage lines next to the verdict so the discriminator is in the same window:
   `proj`/`mlp`/`lm_head` rising while `cast` disappears is the staging's cost; every family flat and the
   wall flat is the launch price being instrument, which is the reading §7 asked for.

## 8. Open risks

* **The prize may be smaller than the launch count.** `cast` is 97 launches of 0.371/0.741 MiB read/write
  a token (quoted, §1) — the bytes are not the point, the submissions are, and whether the submissions
  cost anything once pipelined is exactly what is unmeasured.
* **Staging cost moved, it did not vanish.** A folded consumer stages BF16 and widens in registers; on the
  grouped projection kernel that halves the `x` staging bytes per block but adds a widening inside the
  dot's operand path. `PACK_X` is the precedent (§7).
* **Prefill is not covered.** A wide body folds only the final-norm site, so any TTFT effect is ~1 launch
  and is out of reach by construction.
* **One-row-only means the wave bill needs its own rule**, which it now has (`WaveShape::cast_fold`) and
  which is self-checked (`body1_castfold_plan_matches_token_plan`), but no shipped gate runs
  `--time-waves --fold`; the A/B runs `--time-stages`.
* **`check_equiv.py`'s §5 wiring checks are substring checks.** They catch plumbing being severed (witnessed,
  §3) and they will complain about a deliberate reformat; that is the intended trade, and the message
  names the defect rather than the pattern.
* The `QWEN_CAST_FOLD` env mechanism is how `gate.py` reaches the fold without editing it; if the route
  ships, an explicit `--cast-fold` option plus a one-line `gate.py` pass-through is the tidier end state
  and is a separate decision.
