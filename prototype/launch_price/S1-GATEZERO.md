# S1 GATE ZERO — pricing launches, segments and the merge prize at the shipped 340 launches

Route **S1** of `progress/2026-10-21/route-S1-segments-2026-10-21.md` §3 is funded to answer one
question before any merging work starts: at the shipping decode step's real launch count and drain
pattern, what does the engine actually pay per launch, what does it pay per segment, and what would
merging buy? This directory is the rig that answers it and the gate that says the answer can be
trusted.

**Status: gate green, no millisecond taken by the builder.** `check_s1.sh` passes (build, census,
geometry selftest, 14 refusals, 16-case device selftest, no-timing discipline, the fitter tested
against fabricated numbers). The timed pass is one command for the main agent, `s1_sweep.sh --run`,
and `fit_slope.py` turns its log into the slope, the intercept, the sync leg, the merge gain and the
verdict. Nothing here touched `engine/`, `engine/performance-status.md`, `README.md` or any gate
threshold, and no number from this directory may enter the ledger except through the main agent's own
measurement.

## 1. The four quantities, and which arms produce each

| quantity | arms | reported as |
|---|---|---|
| launch cost | ladder `--launches N --syncs 1 --body tiny`, N ∈ {1,48,165,305,340,680} | slope in µs per launch, with intercept and worst residual |
| is dispatch hidden behind work? | same ladder at the byte-dominated bodies (`mid`, `fat`) | the slope there; ~0 means the issue cost is already hidden |
| sync cost | leg A: sync count swept (340→1) at pinned launch count. leg B: the engine arm at 1 / 305 / 340 drains, identical launches, bytes and work | slope in µs per segment, per leg |
| merge prize | `--total-mib` pairs: identical bytes and work at different launch counts — 64 MiB at {340, 8, 1}, and the step's own 1461 MiB at {340, 48} | `gain … µs = … µs per launch removed`, with the demand-equality asserted |

leg B is the D2 §5 experiment (serialized 2.557 ms vs burst 0.754 ms at 48 launches, ~37 µs per sync)
re-run at the real 340 launches and 305 segments, which is the ratio the route is actually about.

One launch cannot carry the whole step: the per-launch ceiling is `grid_cap(2048) × threads(256) ×
passes(8) × 8 B = 33.55 MiB`, so 1461 MiB needs at least 44 launches. That is why the step-scale merge
pair bottoms out at N = 48 and the pure launch-count pair goes down to N = 1 at a smaller footprint.

## 2. The decision line, and the bands that describe the cost's shape

The funded decision rule is the route's own, from `progress/2026-10-21/route-S1-segments-2026-10-21.md`
§4, and the fitter prints it first: **slope ≥ 3 µs/launch → S1 is real and the candidate list is
funded; 1–3 µs → marginal, fold it into D5 (merges must also delete work); < 1 µs → S1 is dead** and
the D2/D4 `T_geo`/`T_serial` figures are artefacts of a probe that drains per segment. The rig
measures the slope; it does not get to redefine that line.

| shape band | value | source |
|---|---|---|
| async submit floor | 0.375 µs/launch | `tools/launch_overhead_hip.cpp` |
| empty-kernel dispatch | 1.85 µs per launch-segment | `tools/submission_probe.cpp`, empty 48-launch burst, one sync |
| serialized-sync gap | 5.9 µs per sync | `engine/stage_clock.cpp` rows: serialized minus burst = 1.80 ms over 305 syncs for identical work |

The shape bands say *what the cost is made of* — issue, empty-kernel dispatch, or the drain — which is
what chooses the remedy; the decision bands say whether to fund anything. The fitter prints both, plus
D2 §5's cost test (a per-launch saving must clear ~2 µs to matter), and refuses to let a negative
intercept be read as physics — D2 §4's −0.065 ms is the cautionary case, and the same trap is recorded
for the prefill floor fit.

## 3. Verification architecture — what makes a number here trustworthy

Every byte the timed window touches is witnessed, and every witness is predicted from the launch plan
before the device runs. The discipline is `prototype/mlp_insitu` (D2) and `prototype/gdn_insitu` (D4),
carried one level over to launch accounting.

* **Per-(launch, block) records.** Each block writes `{read-sum, read-count, write-sum, write-count,
  launch tag}` into its own slot. A slot left at the sentinel is a block that never ran; a wrong sum is
  a wrong base, a short range or a dropped block; a wrong tag is a block that ran the wrong launch.
* **The read fill has a closed form.** `word(p) = K·(2p+1)` over 8-byte elements with K odd: the value
  depends on the element's global index, the host predicts any range's sum in closed form with no
  floating-point association risk, and because K is invertible mod 2⁶⁴ a sum also knows its element
  count. That invertibility is asserted in `--selftest-geom`, not assumed — the first value tried was
  not invertible and the assertion caught it.
* **Written values are indexed by offset inside their own launch, not by destination address.** A
  launch pointed at a neighbour's range then stores the values it would have stored at home, so its own
  record stays byte-identical and **only** a byte-content check can see the alias — which is what makes
  the alias control test the byte check instead of the record check. (An address-indexed value made the
  aliased launch's own sum move; the witness was doing the launcher's job.)
* **Chunk arithmetic is booked, not inferred.** The kernel receives `rchunk`/`wchunk` from the plan
  rather than deriving them from `gridDim.x`, so "one block never ran" means exactly that instead of
  "every remaining block silently enlarged its own work".
* **Must-catch and must-not-catch sets are DERIVED** from the plan arithmetic (`rec_class`,
  `welem_class`), never hand-listed, and `--selftest-geom` asserts every control has a non-empty set on
  **both** sides — the AGENTS rule that an empty must-catch set is not a control. One exception is
  proved rather than populated: `permute_reads` rotates a block's own read set, which provably cannot
  move a sum, so it is labelled `blind`, its must-catch set is empty by proof, and the case is refused
  as evidence of anything else. A score-only defect on a single-lane row is the same class.
* **Three structural claims are asserted**, so no check can be decoration: the alias and lie-write
  controls move *no* record (the byte-content check is load-bearing), and the short-read control moves
  records but *no written byte* (the read-sum is load-bearing).
* **Each control runs in its own forked child** and re-derives its own expectation from the same
  functions the positive path uses; the selftest requires `parsed == |case list|`, and
  `check_s1.sh` additionally requires the advertised case list to equal the one that ran.
* **NO-TIMING discipline.** `--launches 0` prints
  `NO-TIMING no_launch=1 reason=timed_window_launches_nothing` and never a number; an un-timed row
  prints `NO-TIMING reason=no_--time_flag`; a row whose witnesses are not clean prints
  `NO-TIMING reason=verify_not_ok`. `check_s1.sh` greps its whole correctness log for
  `ms_median|GB/s` and fails if one appears, and `--pages cold` is refused outright (first touch of
  fresh pages runs ~7× slow).
* **The engine arm's population and byte mix are checked twice**: `plan_invariants` inside the rig (and
  a drift beyond 2 % aborts with both numbers printed), and `census.py` outside it, which re-derives
  the per-family launch bill from `engine/state.h`'s layer counts and the per-family bytes from
  `tools/step_traffic_model.py` — consumed, never retyped — and asserts the totals are the quoted
  340/305.
* **`fit_slope.py` is itself gated.** It dies on an empty parse, on a missing ladder arm, on an arm
  whose reps disagree by more than `--max-spread` (this box's dispatch-bound rows are bimodal), and on
  an arm key that covers two different footprints. Fabricated stub rows carry a `stub=1` marker that
  the fitter refuses unless `--allow-stub` is passed, and prints a banner over when it is.

## 4. What building this found — all five were device-found, none guessed

1. **A grid-short launch is not one dropped block unless the chunk is booked.** Deriving the chunk from
   `gridDim.x` made every surviving block enlarge its work: the control moved 64 records instead of 1,
   and its derived set was a lie about the injection it claimed.
2. **"One element short" has to be short in the block's own chunk.** Subtracting from the launch's end
   left every per-block chunk untouched except the last, so two controls did nothing and reported
   `must=0/2` — a control that catches nothing while printing a plausible line.
3. **An alias that also shortens its own write count moves its record**, which broke the claim that an
   address defect is invisible to the record check. Fix was (write count as booked) + (offset-indexed
   values).
4. **The engine's byte mix was MB mislabelled as MiB**, making the engine arm 4.6 % heavier than the
   census it reproduces (`lm_head` reads 509.552 MB = 485.947 MiB). The rig's own invariant abort
   surfaced it; `census.py` now keeps it honest, and the refusal prints both numbers.
5. **A kind-1 tally that silently inverted `chk.ok` printed `caught=0/8` over a fully green run.** One
   verdict per case, computed once, is now the rule.

## 5. Files

| file | role |
|---|---|
| `s1_launch_price.hip.cpp` | the rig: plan, kernel, witnesses, negative controls, selftests, refusals |
| `build.sh` | `hipcc -O3 --offload-arch=gfx1151`, with `-Wl,-rpath,/opt/rocm/lib` |
| `census.py` | engine arm's launch/segment/byte bill vs `engine/state.h` + `tools/step_traffic_model.py` |
| `check_s1.sh` | **the gate** (~5 s, correctness only, prints no ms): A build · B census · C geometry · D 14 refusals · E device selftest (16 forked cases) · F no-timing discipline · G the fitter under stubs, plus the RESULT-line contract between the rig's printf and the fitter's regex |
| `s1_sweep.sh` | the timed matrix; prints the command list by default, `--run` executes it inside one `gpu_lock` hold — **main agent only** |
| `fit_slope.py` | slope / intercept / residual / sync legs / merge gain + the band verdict, with the refusals above |

## 6. NEXT — the main agent's timed pass

`check_s1.sh` is green, so the rig may be timed. One `tools/gpu_lock.sh` hold, blocking, zero subagents
(AGENTS: only the main agent measures, and this box serves its own tokens on the same iGPU):

```
prototype/launch_price/s1_sweep.sh --run
python3 prototype/launch_price/fit_slope.py --in /tmp/s1_sweep.log \
    --engine-arm 'uniform|tiny|t1461|340|1|8|warm' --merge-arms 'uniform|tiny|t1461|48|1|8|warm'
python3 prototype/launch_price/fit_slope.py --in /tmp/s1_sweep.log \
    --engine-arm 'uniform|tiny|t64|340|1|8|warm' --merge-arms 'uniform|tiny|t64|8|1|8|warm,uniform|tiny|t64|1|1|8|warm'
```

Read order: the launch slope **and its residual** (a bent ladder is not a slope), then whether the
intercept is negative, then the sync leg, then the merge gain against the slope prediction. Report the
per-rep spread you saw: glue's 49-launch path is dispatch-bound and bimodal
(`engine/performance-status.md` §6), and `fit_slope.py` refuses a ladder whose arms disagree by more
than `--max-spread` — the machine refusing its own bad window.

Two things the timed pass will **not** answer, recorded so they are not inferred from it:

* **The merge prize at the engine's own per-family byte mix is not measured directly.** `--body-mix
  engine` fixes the shipped 340-launch population, and `--total-mib` is the knob that changes launch
  count at fixed bytes; the rig refuses to combine them rather than silently averaging two definitions
  of the bytes. The merge pairs therefore hold bytes and work equal at a *uniform* body. If the slope
  and the merge gain come out large enough to fund S1, an engine-mix merge arm is the next build.
* **Whether launches overlap work is not separable from these arms alone.** The engine arm at 1 vs 305
  drains says how much the drain pattern costs at the shipping shape; if that gap is large at a
  byte-dominated body as well as at the tiny one, the cause is the drain, not the issue, and the next
  experiment is a repeat-pass arm (same launches, more work per launch), which is a one-knob addition.
