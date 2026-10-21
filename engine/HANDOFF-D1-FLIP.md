# HANDOFF-D1-FLIP — route D1's defaults are shipped; the ledger windows this flip still owes

Written 2026-10-21 by the worker that took the flip. `engine/HANDOFF-D1-ENGINE.md` §4c records the decision;
`engine/results.md`'s closing section records the evidence in prose. **Nothing here is committed** — see §6.

The flip: `engine/build/decode_run` now defaults to `--proj-plan packed --proj-shape unroll8`, so the shipped
per-token projection bill is 48 where §07 bills 114 and the token's launch bill is **340 where it was 406**.
Every structural check that reads that bill moved in the same set of edits, which is the whole point of doing
it as one set: the trap this is guarding is `--attn-tile` (`AGENTS.md`) — a gate silently judging a different
launch plan than the engine ships.

## 1. What changed, per file

| file | change |
|---|---|
| `engine/kernels.h` | new `kProjPlanDefault = kPacked` / `kProjShapeDefault = kUnroll8` — the one spelling of "the shipped pair", with the near-name trap (`ProjShape::kShipped` is the **pre-D1** shape, not the default) written down. Enum comments mark `kSplit`/`kShipped` as what was measured before D1 and what a flag still replays. |
| `engine/decode_run.cpp` | option defaults take those constants; the unbuildable-pair refusal is re-keyed from `--proj-plan` given to **`--proj-shape` given** (with `packed` as the default it is a bare `--proj-shape shipped` that now asks for the impossible pair, and `--proj-plan split --proj-shape shipped` must stay legal — it does); `--help` rewritten for both flags; the `LAUNCH`/`PROJPLAN` comment now says which line answers for what. |
| `engine/decode_step.cpp` | `planned_launch_counts()` → `proj = proj_launches_per_body(kernels::kProjPlanDefault)`, i.e. **counted off the job list, not typed**; the header table carries both bills (shipped 48/340 and §07's per-operation 114/406 for the `split` plan *and for any wide body*, because `pack_proj = (n == 1) && packed`). |
| `engine/decode_step.h` | `submit_token_body` submits 339 of 340 (was 405 of 406); the plan-as-data section now explains why `LAUNCH` is flag-free and which single constant the three C++ readers share. |
| `engine/stage_clock.{h,cpp}` | `plan_bill()`'s fallback is the shipped (packed) bill, so `set_plan_proj` is now how a **`split`** run departs from it (the polarity inverts, the mechanism does not); the one-row self-check builds its body under the shipped plan and gained `body1_split_plan_matches_token_plan`, so the non-shipped plan stays witnessed too. **A `--time-waves` log line now carries four `body1_*` fields, not three.** |
| `engine/kernels_proj.cpp` | the two comments that said "decode issues 114 of these launches" and "114 with the flag off, 48 with it on" now say which is shipped and which is the flag arm; `parse_proj_plan`'s error lists `packed` first as the default. |
| `engine/gate.py` | `launch_plan()` (shipped, `proj = 2 × n_layers`) and a new `launch_plan_split()` (§07's per-operation table, = the wide-body bill) so section (d) and section (h) are not derived from each other; `selftest_launch_bill()` gained the `proj`-revert must-catch; `wave_expect()` takes a one-row bill override; new `launch_work_rows()`; the cross-side "not less work" check compares tensor-rows; the wave-ratio identity became two per-side derived equalities. **No band, floor, allowance or `attn_form_band` value was touched.** |
| `engine/README.md` | the section-(d) echo line prints both bills on one line; the two other stale bill claims (top of file, deferred-lever list) were marked in place. Kept **line-count-neutral** on purpose: 72 tracked documents cite it by line. |
| `specification/07-decode-step.md` | §07.4's "~200 kernel issues per decode step" is marked as the estimate it was and the audited 406/340 appended. Also line-count-neutral, and the 145–153 tok/s band is untouched — re-pricing a band is a measurement call (§5 below). |
| `engine/results.md` | append-only: one new closing section. No old section edited. |
| `engine/HANDOFF-D1-ENGINE.md` | §1's flag table and "default OFF" line, §2's `LAUNCH`/consequence paragraphs and §4c corrected; §7's "do not flip the default" clause marked as spent. |

## 2. Evidence already taken (worker; correctness only, **no timing taken**)

Every device command below was run blocking, one at a time, with `tools/gpu_lock.sh` held by this worker
(`engine/gate.py` does not self-lock — `grep -c gpu_lock engine/gate.py` is 0), and `util` recorded and then
discounted per `AGENTS.md`.

| command | result | wall / util |
|---|---|---|
| `bash engine/build.sh` | both binaries build | 22.7 s, CPU only |
| `decode_run --dump-proj-plan` (no flags) | `PROJPLAN plan=packed shape=unroll8 launches_per_step=48` | 0 s, device-free |
| `… --dump-proj-plan --proj-plan split` | `launches_per_step=114` | 0 s, device-free |
| `… --dump-proj-plan --proj-shape shipped` | **refused**, message names the pre-D1 pair | 0 s |
| `… --dump-proj-plan --proj-plan split --proj-shape shipped` | `plan=split shape=shipped launches_per_step=114` (frozen replay still reachable) | 0 s |
| `gate.py --selftest` | `SELFTEST PASS`, including `the projection plan reverted to one launch per projection — caught by ['launch count proj', 'launch total']` and `the clean bill reproduces §07 exactly (cast=97, total=340)` | ~90 s, CPU |
| `gate.py --dry-run --fixture <each of 21>` | 16 PASS + the 5 expected reds | ~2 min, CPU |
| `decode_run --ids-file tmp/proj-d1/p128.ids --steps 32 --time-stages` (no proj flags) | `LAUNCH … proj=48 … total=340`; `LAUNCHSTAT family=proj calls=1650 calls_batched=114 rows_total=16128 rows_max=128` = 114×128 + 48×32; `STAGE … family=proj launches_per_step=48` | 11 s for three runs, util 90 %→81 % |
| same on `tmp/proj-d1/w4000.ids` (4,000 fed + 32 steps), default vs `--proj-plan split --proj-shape shipped` | all **4,032 `STEP` lines identical** including `top_logit`; default `calls=2448 rows_total=457536` = 114×4,000 + 48×32, split arm `calls=4560 rows_total=459648` = 114×4,032 | same 11 s window |
| parity against the banked A/B logs | `diff` of the 160 `STEP` lines vs `ab128_pack_1.log`, `ab128_off_1.log`, `ab128_shape_1.log`: **all three identical** (the D1 arms are bit-identical, so a no-flag default run matches every one of them) | CPU |
| `decode_run --ids-file tmp/proj-d1/p128.ids --steps 4 --time-waves` | `… plan_match=1 launchstats_match=1 body1_plan_matches_token_plan=1 body1_library_plan_matches_token_plan=1 body1_split_plan_matches_token_plan=1 body1_packed_plan_matches_token_plan=1`, and the wave bills the wide body's 114 in the same report | 2 s, util 92 %→47 % |
| `gate.py --fixture <all 21>` on device | **21 judged = 13 PASS + 5 FAIL + 3 SKIP**; the 5 are the recorded set | 6–231 s each, util 89–93 % |
| `speccheck/run_all.py --quick` | `11 gates run, 0 failed — ALL GATES PASS` | 138 s, CPU |

The device population was **derived**, not typed: `for d in data/refs/*/; do [ -f $d/manifest.json ] && echo $d; done`
→ 21 fixtures, and the count judged was asserted against it.

- PASS (13): `dec8 p64 p8 chunk64b p128 p300 prefix24dec8 smoke wide640f wide1024f zz wide4096f wide4096p`
- FAIL (5), the expected set: `dec300f dec64f dec64t mt32f wide520f` — four stale `route_allowance.json`
  `specimpl_sha256` hashes, plus `dec64f`'s three §10.2 `meanrel` steps (28, 29, 34). Nothing else is red.
- SKIP (3): `corpus48 corpus48engine corpus48sdpa` — `rc=3`, the gate's own refusal (a token-agreement
  corpus, not a pin set). Recorded so nobody reads three refusals as three passes.

Section (h) on a device fixture, as a worked example (`dec64f`, 17 prompt rows + 64 steps): batched
`proj calls=3186 = 114 + 48×64`, one-row `3888 = 48×81`, carry `3528 = 4×114 + 48×64`; the work comparison
reads **tensor-rows 9,234 on all three sides** (= §07's 114 × 81 tokens) with the raw `rows_total`
(5,010 / 3,888 / 5,010) printed beside it in the same line. `gate_dec*.log` files are in `tmp/proj-d1/`.

## 3. The windows you owe (main agent: blocking, zero subagents, lock held, best-of-N, spread stated)

No millisecond in this file came from this worker. These are the commands, in the order that makes the
ruler cheap first.

**W1 — the `proj` family row at 128 lanes, two arms now instead of three.** The banked window used three arms
(`off` / `shape` / `pack`); with the flip, `off` **is** the shipped arm, so the A/B is now "shipped (no flags)"
against "the frozen pre-D1 pair". `tmp/proj-d1/ab128.sh` still passes `off)` no flags and `pack)` the two
flags — **it must be re-labelled before it is re-run, or its canary (`PROJPLAN launches_per_step` must differ
between arms) reads identical and the window is void** (AGENTS: a paired A/B against a stale arm reads as a
clean null). The two arms and the canary:

```bash
tag="D1 shipped-vs-preD1 proj A/B"; bash tools/gpu_lock.sh acquire "$tag"; bash tools/gpu_lock.sh util
for rep in 1 2 3 4 5 6 7; do for side in ship pred1; do
  case "$side" in
    ship)  extra=() ;;                                        # shipped: packed + unroll8
    pred1) extra=(--proj-plan split --proj-shape shipped) ;;  # the frozen pre-D1 pair
  esac
  timeout 900 ./engine/build/decode_run --ids-file tmp/proj-d1/p128.ids --steps 32 --time-stages \
      "${extra[@]}" > "tmp/proj-d1/flip128_${side}_${rep}.log" 2>&1
done; done
bash tools/gpu_lock.sh util; bash tools/gpu_lock.sh release "$tag"
```
Canary to assert per side, before any ms is read: `PROJPLAN … launches_per_step=` is **48** on `ship` and
**114** on `pred1`; `NUMERICS … proj_plan= proj_shape=` names the side; every `STAGE` line carries
`plan_match`-consistent `launches_per_step`. Feeds: `engine/performance-status.md` §2's `proj` row
(carried: 4.1128 ms/step in-situ, 3.2843 isolated) and the `<main-agent window>` cells of
`engine/results.md`'s D1 section; the banked single-window figures are `proj` 4.2015 → 2.9521 → 2.7316 and
step 11.6967 → 10.1851 ms.

**W2 — the headline context sweep under the new default.**
```bash
tag="D1 context sweep"; bash tools/gpu_lock.sh acquire "$tag"
.venv/bin/python tools/context_sweep.py --lanes 96,128,512,1024,2048,4096 --steps 32 --runs 3
bash tools/gpu_lock.sh release "$tag"
```
Feeds §1's headline decode tok/s (carried: 89.4 tok/s at 128 lanes, marginal 11.181 ms/token) and the top-level
`README.md` headline — both main-agent-owned documents. Quote the spread, and let the long-context row carry
the headline if the short one is dispatch-bimodal (`engine/performance-status.md` §6).

**W3 — the 4,032-lane row**, same two arms as W1 on `tmp/proj-d1/w4000.ids` (4,000 fed + 32 steps), from the
stage-clock burst only. This is the row every short gate cannot see: at 4,032 lanes the prefill is 8 wide
waves plus one-row decode steps, so it prices both halves of the flip. Feeds §3's wave row and §2's
long-context statement. Banked: −10.4 % on the 4,032-lane row.

**W4 — §4's isolation row.** `engine/performance-status.md:176` bills "projections, whole 48-launch token
plan, **rows/block 32** | 3.2843 ms". The shipped one-row geometry is now rows/block 8 at 1,024 threads, so
that row's *geometry label* is stale as well as its number: whichever prototype isolation probe backs it
(`prototype/proj_gemv`) needs re-reading, and the row should be restated for the shipped arm with its own
window. Do not just retype the ms: the row names a geometry that is no longer the default.

**W5 — after W1–W4, the ledger and README.** `engine/performance-status.md` §1/§2/§3/§4 and the top-level
`README.md` headline are yours (both were off-limits to this worker). `engine/README.md` was updated only
where it bills launches, and kept line-count-neutral.

## 4. Citation coordinates this set rotates (commit-time chore, main agent)

`tools/citation_rot_check.py` is blind to drift introduced inside its own commit, so the census was done the
other way round (`tmp/flip_cite_check.py`, which generalizes `tmp/proj_d1_cite_check.py` to `.py`/`.md`
targets; it scanned **705 tracked documents**, found **150 line-form citations into the 12 changed files**, of
which **83 coordinates rotate**):

| cited file | net lines added | citations rotated |
|---|---|---|
| `engine/decode_step.cpp` | +18 | 56 |
| `engine/decode_run.cpp` | +16 | 11 |
| `engine/gate.py` | +137 | 8 |
| `engine/kernels.h` | +14 | 4 |
| `engine/kernels_proj.cpp` | +3 | 4 |
| `engine/README.md`, `specification/07-decode-step.md` | **0** (edited in place) | 0 |
| `engine/results.md` | +77 (append only) | 0 |

Full list, one line per citation, `cited file ⇤ citing doc:line ⇤ old -> new`:
**`tmp/proj-d1/flip_cite_rotation.txt`**; the summary table above is `tmp/proj-d1/flip_cite_summary.txt`.
Most of the load is in `progress/` journals (frozen by rule 2, so re-pointing them is your call and needs
same-line substitutions only) and in `prototype/` comments. `engine/HANDOFF-D1-ENGINE.md` §6 already records
two coordinates (`engine/decode_step.cpp:596,599` in `prototype/gemm_algo_sweep/README.md:91,92`) that were
*already* stale before this change; the +18 shift moves them again — the new coordinates are in the rotation
file. Re-check after any further edit with:

```bash
for f in $(git diff --name-only); do echo "$f $(git show HEAD:$f | wc -l) -> $(wc -l < $f)"; done
timeout 600 .venv/bin/python tmp/flip_cite_check.py
```

## 5. Findings, and the decisions this worker did **not** make

1. **`specification/07-decode-step.md` has no launch table.** The flip brief expected one; there is none, and
   no "five per GDN layer / four per attention layer" sentence. The per-family census lives in
   `engine/decode_step.cpp`'s header and `engine/gate.py`; §07.4's only launch-count claim was the "~200
   kernel issues per decode step" dispatch estimate, which was already ~2× low before route D1. That sentence
   is now marked as an estimate and followed by the audited bill (406 split / 340 shipped). **Its derived
   145–153 tok/s band was left exactly as written** — at the measured 1.85 µs the shipped 340 launches imply
   ≈0.63 ms of dispatch instead of ≈0.36 ms, so the band is ~0.27 ms/token optimistic on dispatch, but
   re-stating a band is a numerics decision, not an edit. Yours to take or waive.
2. **`tools/step_traffic_model.py` was left alone, and is not a launch bill.** Its `LAUNCHES` dict
   (`proj: 114, …`) is **unreferenced dead data** — nothing in the module reads it — and `:77` is provenance
   prose for the *byte* census, which the flip does not change (the same tensors are projected, grouped
   differently). Verified by running the tool: its table prints MB/token only. Anyone who wants it to reflect
   the shipped plan should move `LAUNCHES["proj"]` to 48 and `FAMILY_INSITU_MS["proj"]` from 4.237 to the
   banked 2.7316 in the same edit that re-prices its `ms` column.
3. **`gate.py` section (h) changed shape, not strength.** `rows_total` stopped being a work measure when a
   launch began covering four tensors while reporting width 1, so the "fewer launches, not less work"
   comparison now folds the bundled tensor-rows back (`launch_work_rows`, derived from the two bills and the
   observed counters: 114 − 48 = 66 per one-row body) and compares those; the check name is unchanged so the
   selftest's must-catch/must-not-catch lists did not have to move, and the raw counters are printed in the
   detail line. The wave-ratio identity `r_wave × waves_b == b_wave × waves_r` was **not** true of a correct
   engine under the new plan (a one-row body is now cheaper than a wide one) and became two per-side
   equalities against each side's own derived bill. Nothing was deferred, scoped out, or made conditional.
4. **`gate.py` does not pass `--proj-*` anywhere** — deliberate. Every gate run therefore judges the shipped
   plan, and §07's split bill is only ever reached by an explicit parity run. If a future gate wants both
   plans in one run, the per-side bill has to be threaded through `run_batched_ab` as well as the stand-in;
   today the stand-in carries it (`one_row_plan`) and the device path does not need it.
5. **The new refusal polarity is user-visible**: `--proj-shape shipped` **alone** is now an error (it names the
   pre-D1 kernel while the plan stays `packed`), and the message names the pair to type. `--proj-plan split`
   alone remains legal (shape-only arm), which is what the banked `shape` A/B arm was.
6. **`--time-waves` printed fields grew from three `body1_*` to four.** Nothing parses them today; recorded
   because it is an echo-line change.

## 6. Nothing is committed

`git status` shows 12 modified files and no staged entries. If you want the topical split this repo's rules
ask for, one plausible set is: (a) engine defaults + shipped-plan constants + refusal re-key (`kernels.h`,
`decode_run.cpp`); (b) the bill move (`decode_step.{h,cpp}`, `stage_clock.{h,cpp}`, `kernels_proj.cpp`);
(c) `gate.py`'s launch model and section-(h) restatement; (d) docs (`engine/README.md`,
`specification/07-decode-step.md`, `engine/results.md`, the two HANDOFFs) plus the citation chore. (a)–(c)
must land together — splitting them is precisely the trap this handoff exists to avoid, so either one commit
for (a)+(b)+(c) plus a docs commit, or all four in one push.
