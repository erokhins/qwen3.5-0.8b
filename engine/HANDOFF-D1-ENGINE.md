# HANDOFF-D1-ENGINE — the `prototype/proj_gemv` win, landed in `engine/`, awaiting its timing window

Written 2026-10-21 by the implementation worker. Source decision: `prototype/proj_gemv/results.md` §D1-W4
(`7c432a4`, `f8644b1`, `2534b40`). Everything below marked **(worker, done)** was run on device with the
GPU lock held and **no timing claim** — no number here except the prototype's own burst figures and the
ledger's carried ones belongs to `engine/performance-status.md`. Everything marked **(main agent, owed)**
needs a clean exclusive window: blocking, zero subagents, `tools/gpu_lock.sh` held, spread stated.

## 1. What shipped, and what it is actually called

> **Correction, same day (the default flip was taken).** The two flags below are no longer default OFF: the
> in-situ A/B landed in `engine/performance-status.md`'s favour and `--proj-plan packed --proj-shape unroll8`
> is now what a bare `decode_run` runs — `engine/results.md`'s closing section names the evidence and
> `engine/HANDOFF-D1-FLIP.md` lists the ledger windows still owed. `split`/`shipped` remain selectable, and
> the pair `--proj-plan split --proj-shape shipped` is still the byte-for-byte pre-D1 replay. Everything the
> rest of this file says about "default OFF" is the state at the time it was written; §4c and §2's
> "consequence" paragraph carry the flip itself.

Two flags, both **default OFF** at the time of writing (HEAD's behaviour), on `engine/build/decode_run` — see the correction above for the current defaults:

| flag | values (default at writing; **bold = shipped today**) | what it moves |
|---|---|---|
| `--proj-plan` | `split` · **`packed`** | one-row projection bodies: the 114 launches/token of §07 of `specification/07-decode-step.md` → the prototype's 48-job plan, now the shipped bill |
| `--proj-shape` | `shipped` · **`unroll8`** | the one-row kernel: `proj_gemv_tokens<IN,1,false>` → `proj_group_rows` (compile-time word walk, `rpb=8` at `threads=1024`) |

**`--proj-lpr 32|128` was not built, deliberately.** The handover described the winning arm as "128 lanes
per row at K=1024, one 128-bit load per lane". No timed arm is that. In the prototype's variant table
(`prototype/proj_gemv/proj_gemv.hip.cpp`) the lanes-per-row axis (`lpw`) is 16/8/4 in the four **refuted**
variants `v2/v3/v4/v6` (25–47 % of probe — a row's F32 dot split and re-associated, which also would have
broken bit-identity), and the winner `v1-unroll` carries `lpw=32`, one wavefront per row exactly as shipped,
claim string "compile-time word bound with `#pragma unroll`, nothing else changed". The name says `lpr`;
the measurement says the levers are (a) the unrolled walk and (b) `step = threads/LPW > rpb`, i.e. a quarter
of each block's wavefronts idle at 4× the grid. Naming the engine knob `--proj-lpr` would have advertised the
axis that lost. Nothing ported from `v2/v3/v4/v6`. **Where the memo's sentence came from:** §D1-W4's own
mechanism paragraph reads "at K = 1024, 128 lanes per row is 8 bf16 per lane = exactly one 128-bit load per
lane", and that sentence contradicts the arm table and the kernel printed around it — `lpw=32`,
`lane = threadIdx.x % LPW`, `WORDS/LPW = 128/32 = 4` words per lane, so four 16-B loads and the unroll is
what keeps them in flight. `threads/rpb = 1024/8 = 128` is threads per row *slot*, three quarters of which
have no row to do; that is the arithmetic the sentence inverted. The measured win is untouched by this, but
`prototype/proj_gemv/results.md` carries the wrong mechanism sentence and this worker may not edit
`prototype/`, so it is recorded here for an amendment commit of yours — the next reader will size a route
from that paragraph.

The shape is applied at **both** K=1024 and K=2048: the −24.5 % and −11.7 % are whole-plan figures covering
both widths, and the unrolled walk is bit-preserving at either. The fallback is loud: `proj_shape_available()`
is true only for those two widths — the only two the grouped kernel is instantiated at — and only on a device
whose warp is 32 wide, which is the width the winning geometry was swept at and the one that makes
`step = threads/warp` separate from `rpb` the way the sweep's winning point does. Any one-row body that
cannot use the grouped shape goes through the shipped kernel and prints
`PROJSHAPE-FALLBACK steps=… shapes=… of …`; if **every** one-row body fell back, `main()` throws rather than
letting `--proj-shape unroll8` report a silent no-op.

Packing is **no-copy**, as steered. `engine/loader.h:170` does one `hipMalloc` per tensor, so contiguity does
not exist and concatenation would have to be manufactured. Priced from the checkpoint rather than assumed
(`tmp/proj_d1_bytes.py`: the `--dump-proj-plan` echo against the safetensors headers, and it aborts if the
two disagree), the projection set is **466,747,392 B = 445 MiB = 0.43 GiB across all 24 layers** — 30 % of the
model's 1.5457 GB of resident weights, ~0.93 GB of read+write traffic to copy, and single-digit milliseconds
at the 237 GB/s plateau. So the copy was affordable and is still wrong: the segment list buys the same
48-launch plan for 192 × 24 B ≈ 4.6 KB of descriptors and zero extra traffic, and the prototype measured the
packed plan over segmented bases (`proj_run_packed` takes a per-tensor `Tensor&` + `row0`), so the
measurement never asked for contiguity either.

## 2. Files

| file | change |
|---|---|
| `engine/proj_group_kernels.h` | **new**. `engine::projgroup::proj_group_rows<IN>(ProjSegs, x, n_rows)`: shared F32 `x` staging, per-row segment lookup, `step = blockDim.x/warpSize` trip loop with clamp + masked store, compile-time word walk at warp 32 (runtime walk otherwise), `__shfl_down_sync` ladder (64-bit mask), one BF16 rounding. Dot body transcribed from `prototype/proj_gemv/proj_gemv_kernels.h`, staging from §06.5. |
| `engine/kernels.h` | `ProjPlan{kSplit,kPacked}`, `ProjShape{kShipped,kUnroll8}` + parsers/names (throw on typo), `ProjSeg{weight,out,rows,start}`, `ProjSegs` (`kProjMaxSegs=4`, by value as a kernel arg), `proj_shape_available()`, `proj_rows()`. |
| `engine/kernels_proj.cpp` | `kProjGroupRowsPerBlock=8`, `kProjGroupThreads=1024`; `proj_rows()` owns the `start` running sum, validates (segment count, width ∈ {1024,2048}, nulls, rows>0) and records **one** `launchstats::proj().record(1)` per grouped launch — that counter is the canary. |
| `engine/decode_step.{h,cpp}` | `DecodeScratch::proj_plan/proj_shape/proj_fallbacks`; the shape and plan are resolved on `n == 1` only (waves keep §07's wide path); the b/a pair keeps `library_eligible=false`, which is the carve-out `engine/kernels_proj.cpp` already enforces by refusing `N < 512` in `proj_gemm` (the 16-row GDN projections sit below the arithmetic-intensity ridge at every width, §3.2, and a library call there costs 12–22× the dispatch floor); `ProjPlanJob` + `proj_plan_jobs()` + `proj_launches_per_body()`, device-free and derived. |
| `engine/stage_clock.{h,cpp}` | `Clock::set_plan_proj()`, `plan_bill(int proj_override = 0)`, `WaveShape::proj_packed` (one-row bodies only), new self-check `body1_packed_plan_matches_token_plan`. |
| `engine/decode_run.cpp` | `--proj-plan`, `--proj-shape`, `--dump-proj-plan`; `PROJPLAN` line beside `LAUNCH`; `proj_plan=`/`proj_shape=` on `NUMERICS`; `PROJSHAPE-FALLBACK`; refuses `packed` with `shipped` (the packing lives in the grouped kernel). |
| `engine/results.md` | new closing section `--proj-plan / --proj-shape … (2026-10-21), flags OFF`, with the perf cells left as `<main-agent window>`. |

`LAUNCH` was **deliberately left at the split bill of §07 of `specification/07-decode-step.md`**
(`kLaunchCounts`, proj=114) in the commit that added the flags. gate.py section (d)
compares that static bill against `LAUNCHSTAT` independently of the binary; making `LAUNCH` follow the flag
would have turned the gate's launch-count check into a tautology inside the tested binary — the exact
failure the `body1_*` self-checks exist to avoid. `PROJPLAN` is the claim line for these flags. **That
arrangement survived the flip unchanged in kind and inverted in value**: `LAUNCH` is still flag-free and now
bills the shipped plan, i.e. proj=48/total=340, so on a default run `LAUNCH` and `PROJPLAN` agree and on a
`--proj-plan split` run they deliberately disagree (read `PROJPLAN` there). `planned_launch_counts()` gets
its count from `proj_launches_per_body(kernels::kProjPlanDefault)` rather than typing 48, so the two
spellings still have to earn the same number.

**Consequence for whoever flips the default** — taken on 2026-10-21, in one set of edits, which is what this
paragraph existed to force: with `packed` as the shipped default, `kLaunchCounts::proj` and
`gate.py:launch_plan()` had to move to 48 in the **same** commit, or section (d) and every `--wave-rows 1`
launch-count expectation would go red — the trap class of "`--attn-tile` default and `decode_run.cpp` must
move in one commit". What that turned out to require, beyond the two numbers: section (h)'s wave model had to
learn that a one-row body now bills fewer projections than a wide one (`wave_expect`'s `one_row` parameter),
the "fewer launches, not less work" comparison had to move from `rows_total` to tensor-rows
(`launch_work_rows`), the wave-ratio identity had to become two per-side derived equalities, and
`stage_clock`'s `body1_plan_matches_token_plan` had to build its one-row body under the shipped plan. All of
it, with the evidence, is in `engine/results.md`'s closing section.

## 3. Correctness evidence already on disk (worker, device, no timings)

Bit-identity was the admission contract and it held, so **no band, gate or expectation file was touched**
(`engine/gate.py`, `engine/attn_form_band.py`, `speccheck/` untouched; `prototype/` read-only).

Per fixture, one `decode_run --time-stages --out DIR` process three ways — defaults / `--proj-shape unroll8`
/ `--proj-shape unroll8 --proj-plan packed` — then byte-compared on `sampled.txt` **and** on every `STEP`
line (`top_logit=%.9g` round-trips a float exactly):

```
fixture      prompt ids   verdict                          STAGEPARITY   PROJSHAPE-FALLBACK
dec8              17      bit-identical off=shape=pack     9/9 green     0
p64               46      bit-identical off=shape=pack     9/9 green     0
chunk64b         120      bit-identical off=shape=pack     9/9 green     0
mt32f            232      bit-identical off=shape=pack     9/9 green     0
p300             277      bit-identical off=shape=pack     9/9 green     0
dec300f          295      bit-identical off=shape=pack     9/9 green     0
wide640f         640      bit-identical off=shape=pack     9/9 green     0
4k anchor     4,000+32    bit-identical off=pack           3/3 green     0
```

(`9/9` = three sides × three granularities of `STAGEPARITY … tokens_identical=1 logits_bitidentical=1`.)
Route witnesses on `p64`: `--gemm hand` off vs both flags on → identical bytes, `family=proj calls` 570→306;
`--wave-rows 1` off vs on → identical bytes, 5700→2400. So the hipBLASLt-hand route and the wave's one-row
bodies both survive the packed plan.

Canary readings, required to move and they did — three independent instruments per side:

```
PROJPLAN plan=split  shape=shipped launches_per_step=114   STAGE gran=stage family=proj launches_per_step=114   LAUNCHSTAT family=proj calls=1026
PROJPLAN plan=split  shape=unroll8 launches_per_step=114   (shape lever moves no count, by design)
PROJPLAN plan=packed shape=unroll8 launches_per_step=48    STAGE gran=stage family=proj launches_per_step=48    LAUNCHSTAT family=proj calls=498
```

`--dump-proj-plan` (device-free: no model, no launch, no timing): `114` lines at default, `48` lines +
`tensors=114` when packed, each printing its segment list and output-base offset. Device self-checks print
`body1_plan_matches_token_plan=1 body1_library_plan_matches_token_plan=1 body1_packed_plan_matches_token_plan=1`
and `plan_match=1`; at `--wave-rows 1` the wave bill reads 2208 proj launches for 46 one-row bodies (46×48)
with `counts_match=1`, and at `--wave-rows 512` a wide body still reads 114 — the packed branch touches
one-row bodies only.

Gate: `engine/gate.py` at **its own defaults** (it does not know these flags, which is the point of defaulting
OFF) over the derived population — `for d in data/refs/*/; do [ -f $d/manifest.json ]; done` enumerates 21
directories, 18 of which carry a prompt source; the three `corpus48*` are pin corpora with `sources: []` and
are not engine fixtures, and the count was asserted, not assumed. Result **13 PASS / 5 FAIL**, and the five
FAIL lines are **byte-identical to the same gate run against a HEAD binary** built in a scratch worktree at
`2534b40` (same gate.py, `--binary` pointed at the HEAD build): four are a stale `route_allowance.json`
(`inputs.specimpl_sha256` mismatch — a carried cache needing re-price by `engine/route_allowance.py`, which
touches nothing here) and `dec64f`'s three extra lines are its known `meanrel <= 5%` step check (§10.2 of
`specification/10-reference-dumps-and-gates.md`), already
recorded in `engine/results.md` as that fixture's honest red state. **Pre-existing, not introduced.**

Scripts used (untracked, under gitignored `tmp/`): `tmp/proj_d1_ids.py` (fixture prompt ids → comma-separated
`--ids-file`, asserting the field count it wrote), `tmp/proj_d1_population.py` (population census with an
expected-count assertion), `tmp/proj_d1_parity.sh` (the three-side witness above), `tmp/proj_d1_w4k.sh`.
The gate logs are under `tmp/proj-d1/gate_*.log`, the HEAD baselines `tmp/proj-d1/base_*.log`.

My device commands, as run, each with the lock held and released by the same command, `util` recorded,
nothing backgrounded, no timing taken: `decode_run --ids-file … --steps {4,8,32} [--time-stages] [--out DIR]`
over the eight fixtures above (host wall 2 s for a short fixture, ~4 s per side at 4,032 lanes,
`util` 47–92 %, which is this session's own serving and should be discounted), plus the 18-fixture gate loop
(~8 s per fixture) and the five HEAD-baseline gate runs.

## 4. Owed: the timing window (main agent)

Rules this handoff assumes: blocking, zero subagents, one `tools/gpu_lock.sh` hold for the whole A/B, both
sides inside it (a ruler re-taken in a different window is not an A/B), `util` before and after, no call to a
self-locking driver (`prototype/proj_gemv/run.py` takes the lock itself) while it is held.

### 4a. Paired A/B of the `proj` family at 128 lanes — the decisive one

`tools/context_sweep.py` has **no passthrough** for engine flags (only `--binary/--model/--lanes/--steps/--runs`),
so the A/B cannot run through the ledger's differential harness; it must come from `decode_run --time-stages`
directly. The `proj` row is context-flat in the ledger ("`mlp`, `proj` and `lm_head` were flat to under 1 %"
from 128 to 4,032 lanes), which is why a 128-lane fixture is the right place to price it.

```bash
# fixtures: 128 fed ids + 32 steps -> last decode step sits at 128..160 lanes.
.venv/bin/python tmp/proj_d1_ids.py p128 0 tmp/proj-d1/p128.ids
awk -F',' '{print "fields="NF}' tmp/proj-d1/p128.ids     # assert the file, do not trust the name
TAG="D1 engine proj A/B"
tools/gpu_lock.sh acquire "$TAG"; tools/gpu_lock.sh util
for rep in 1 2 3 4 5 6 7; do
    for side in off shape pack; do
        case "$side" in
            off)   extra=() ;;
            shape) extra=(--proj-shape unroll8) ;;
            pack)  extra=(--proj-shape unroll8 --proj-plan packed) ;;
        esac
        ./engine/build/decode_run --ids-file tmp/proj-d1/p128.ids --steps 32 --time-stages "${extra[@]}" \
            > "tmp/proj-d1/ab128_${side}_$rep.log" 2>&1
    done
done
tools/gpu_lock.sh release "$TAG"; tools/gpu_lock.sh util
```

**Canary first, timings second** — before reading a single `ms`, assert on every log:
`PROJPLAN … launches_per_step=` reads **114 / 114 / 48** for off/shape/pack, `NUMERICS … proj_plan=…
proj_shape=…` names the side you think you are timing, and `STAGE … TOTAL … plan_match=1`. A stale binary
agrees with itself, so the launch count is the field the change is *required* to move; abort the script if any
log fails it. Then take, per log and per side:

- `STAGE gran=stage family=proj ms_per_step` — the headline, plus its `launches_per_step`, `segments_per_step`,
  `us_per_launch`;
- `STAGE gran=kernel family=proj ms_per_step` — the independent second instrument (different sync pattern);
- `STAGEPASS gran=burst ms_mean` — whole-step cost, which is what the ledger's decode row is built from;
- every other family row (`glue`, `gdn_*`, `attn`, `mlp_*`, `lm_head`) — by construction nothing else moved,
  and the run should show that rather than be asked to assume it.

Quote best-of-N **per side with the spread**, and pair sides from the same rep.

**The expected prize, arithmetic shown.** The ledger's current `proj` row
(`progress/2026-10-21/performance-status-2026-10-21.md` §2, 128 lanes, median of 3) is **4.1128 ms/step**
in-situ against a **1.973 ms** roofline, with today's isolated burst at **3.2843 ms** (the 48-launch plan at
rows/block 32) and a burst→in-situ gap of **+0.83 ms**. (The transfer memo carried 4.234 ms; the ledger's §2 row is
newer, so these predictions use it — re-derive them from your own OFF row if it differs again.) From the
prototype: the shape is worth −24.5 % at fixed plan (2.4693 vs 3.2692 ms/plan) and the plan −11.7 % at fixed
shape (2.4942 vs 2.8244), which **compose multiplicatively to −33.6 %**. Three transfers, three predictions:

| what transfers | arithmetic | predicted `proj` ms/step | decode step, from the 11.475 ms marginal |
|---|---|---|---|
| shape + plan (the −33.6 % lands whole) | 4.1128 × 0.664 | **≈2.73** | −1.38 ms ≈ **−12 %**, ≈99 tok/s |
| shape only (plan absorbed — this handoff §5.2) | 4.1128 − 0.80 | **≈3.31** | −0.80 ms ≈ −7 %, ≈94 tok/s |
| plan only | 4.1128 − 0.33 | **≈3.78** | −0.33 ms ≈ −3 % |

The brief's adjudication lines were **WIN ≤ 2.60 ms, KILL ≥ 3.05 ms** — as ratios of the carried OFF row,
≤ 0.632× and ≥ 0.742×. Read them as two verdicts rather than one: 3.05 sits *above* the shape-only
prediction, so a run landing near 3.3 does not mean the win failed, it means the **plan** lever failed and the
**shape** lever paid. That is why the A/B has three sides — adjudicate `shape` vs OFF and `pack` vs `shape`
separately, and do not average them into one number.

### 4b. The 4,032-lane row, from the stage-clock burst only

```bash
.venv/bin/python tmp/proj_d1_ids.py wide4096f 0 tmp/proj-d1/w4096f.ids
cut -d, -f1-4000 tmp/proj-d1/w4096f.ids > tmp/proj-d1/w4000.ids
awk -F',' '{print "fields="NF}' tmp/proj-d1/w4000.ids    # 4000, never a truncated 4096
# 4,000 fed + 32 steps -> last decode step attends 4,032 lanes
# same three sides as 4a, inside one lock hold:
./engine/build/decode_run --ids-file tmp/proj-d1/w4000.ids --steps 32 --time-stages [--proj-shape unroll8 --proj-plan packed]
```

Read `STAGEPASS gran=burst ms_mean` and the `proj` family row only. **Never `context_sweep.py` at ≥4k** —
the differential there produced a 29.7 % spread with a "best" faster than a 128-lane step at 32× the KV.
Expected wall ≈4 s per side (≈1 s of it model load); this witness was already run for bits at the worker's
wall of 7.9 s for both sides.

### 4c. If the numbers land

**They landed, and the flip was taken on 2026-10-21.** The in-situ `proj` family row went 4.2015 → 2.9521 →
2.7316 ms/step (off → `unroll8` → `+ packed`) and the whole step 11.6967 → 10.1851 ms, −12.92 %, with a
−10.4 % 4,032-lane row and bit-identity on the nine fixtures the parity work compared; those are the
banked window's figures and the `<main-agent window>` cells in `engine/results.md`'s D1 section is where the
three columns went. `--proj-plan packed --proj-shape unroll8` is therefore the shipped path, and
`engine/results.md`'s final section records what moved with it: `kLaunchCounts` 406 → 340,
`gate.py:launch_plan()` proj 114 → 48 (with `launch_plan_split()` keeping §07's table and a new
must-catch case), `wave_expect`/`launch_work_rows`/the wave-ratio check in section (h), `stage_clock`'s
one-row self-checks, `engine/README.md`'s section-(d) line and §07.4 of the spec. The remaining debt is
ledger-side, not engine-side, and it is itemised with its commands in `engine/HANDOFF-D1-FLIP.md`:
`tools/context_sweep.py` re-taken at 96/128 and 512/1024/2048/4096, the stage-clock family rows at 128 lanes,
and the decode-attribution `proj` row plus the headline decode tok/s in `engine/performance-status.md`
(§6 owns the window). The flags stay as booked levers: `--proj-plan split --proj-shape shipped` is the
pre-D1 replay and still runs byte-identically (re-witnessed at 4,032 rows after the flip).

## 5. What could make the prize not transfer

Stated with the arithmetic, so the window can falsify it rather than inherit it:

1. **A burst is not a step, and for `proj` the ledger has already sized it: +0.83 ms.** The §2 row of
   `progress/2026-10-21/performance-status-2026-10-21.md` is 3.2843
   isolated against 4.1128 in-situ, so multiplying a burst *ratio* by the in-situ row overstates the absolute
   saving by about that much; the conservative form is the absolute burst delta (3.2692→2.4693 =
   −0.80 ms/step) on top of the ledger's gap, which is the middle row of this handoff §4a's table. The instrument's own
   line to read is the step-level **`STAGEGAP gran=stage exposed_ms`** (serialized mean − burst mean, 1.98 ms
   in the carried window): if packing only removes launches, `exposed_ms` shrinks while `burst_ms_mean`
   barely moves — that pattern is the queue paying the plan lever, not the machine.
2. **The step already hides launches behind execution — a standing ledger finding, not a worry.**
   `engine/performance-status.md` §2: "*The exposed submission gap is the whole launch bill, not a prize* …
   no fusion-for-launch-count, graph capture or wider block buys anything there", with captured graphs
   replaying at 1.78 µs against 1.85 µs eager for the same reason. So the ledger predicts that most of the
   plan lever's 66 removed launches (66 × 1.85 µs ≈ 0.12 ms) will **not** transfer; the plan's measured burst
   delta is 0.33 ms for those launches (≈5 µs each), so at best ≈0.21 ms of it is work distribution rather
   than dispatch. Expect the plan lever at about a third of its burst value in-situ, or nothing. If `shape`
   pays and `pack` does not, report two verdicts, not one average.
3. **The winning shape idles three-quarters of each block.** `rpb=8` at `threads=1024` means 8 of 32 wavefronts
   per block do row work, so blocks-per-CU occupancy and the resident-block budget differ from the shipped
   shape. At burst, with nothing else on the machine, that measured faster. In a step the projection launches
   sit between GDN and MLP launches, and idle wavefronts still hold slots. This is the most likely reason an
   in-situ ratio comes in above 1.0; the per-family rows from 4a will show it as a neighbouring family
   slowing rather than `proj` speeding up.
4. **Shape and plan are not orthogonal on device.** The two prototype deltas were measured one at a time, each
   with the other held; the −33.6 % is their product, not a measurement. 4a's `shape` side is exactly the
   experiment that tests whether they compose.
5. **The launch-shape win was taken at `lanes=1` context in the prototype.** `proj` is context-flat in the
   engine's own ledger, so this is the weak reason — but 4b is the direct check.

## 6. Citation coordinates this change rotates (commit-time chore, main agent)

A `path:NNN` citation is a claim about the cited file's line count (`AGENTS.md`, citation hygiene), and
`tools/citation_rot_check.py` cannot see drift introduced inside the change it checks. Seven engine files
grew, so `tmp/proj_d1_cite_check.py` was written to do that check the other way round — every
`engine/<edited file>:NNN` citation in a tracked document, re-pointed by `git diff -U0` hunk arithmetic.
It scanned 281 documents, found **93** citations into the seven edited files, and **2 rotted**:

| cited as | meaning | where that call site is now |
|---|---|---|
| `engine/decode_step.cpp:596` (`kQkvRows`) | the GDN `in_proj_qkv` projection call | `engine/decode_step.cpp:653`, now `add_in(… kInProjQkv, sc.g_raw, kQkvRows, true)` |
| `engine/decode_step.cpp:599` (`kValueDim`) | the GDN `in_proj_z` projection call | `engine/decode_step.cpp:654`, now `add_in(… kInProjZ, sc.g_z, kValueDim, true)` |

Both live in `prototype/gemm_algo_sweep/README.md:91,92` (and the bare `decode_step.cpp:596/599` in that
directory's `results.md:260,261`), which this worker may not edit. Note they were **already 13 lines stale at
`2534b40`** — HEAD held those calls at 609/612, not 596/599 — so this is pre-existing rot that the +34 shift
made larger, not a new breakage. The other 91 coordinates survived, including `engine/kernels_proj.cpp:34,35`
(the pair `progress/2026-10-16/ledger-prune-2026-10-16.md:101` is careful about), because every insertion in
that file is below them.

## 7. If a re-run goes red

The admission contract is bit-identity, and it held everywhere it was tested. If a future run prints
`tokens_identical=0`, a `DIFFERS` in the byte-compare, or a `PROJSHAPE-FALLBACK` you did not expect, the
route is: stop, keep the logs, and report — do **not** widen a band, do not edit `engine/gate.py`,
`engine/attn_form_band.py` or `speccheck/` (and, since 2026-10-21, there is no default left to flip: the
shipped pair is the packed one, so a red run means reverting the pair by flag, not by editing a default). Diff the `STEP` lines first: the
first step at which the two sides' `top_logit` differs localises the row, and `STAGESTEP gran=burst step=…`
gives the same index in the replay. Two signatures are diagnostic on sight: `ndiff == nrows × width` exactly
is a whole-tensor difference, i.e. a changed work split or a wrong segment index rather than a wrong value;
and a difference that appears only above 512 lanes is the streaming/partial path, which this change does not
touch and which has bitten this tree before.
