# D3 — `delta`: what verifying TWO query rows per decode step costs over ONE

**Status: harness built and admitted on device (correctness only). NO TIMING WAS TAKEN by its author.**
Every `ms` figure marked `quoted` comes from `engine/generation-limits.md` §0–§1 (paired window, 4,032
lanes) or from `prototype/mtp_gatezero/` (D1, taken by the main agent). The timing commands are in §6;
none has been run. §7 lists the correctness-only device checks the author did run, with command, wall and
`tools/gpu_lock.sh util`, and states explicitly that no millisecond figure from them is reported.

**Why it still matters.** `route_price.py` kills the FULL-head route above `delta_max = 0.379` of a step
(= **4.143 ms** of the quoted 10.9357 ms step at 4,032 lanes). `components.py` with no rig evidence at all
prints the arms that bracket the answer: the naive "every per-row family doubles" bound is **1.10 of a
step** (killed, §4); the structured worst case — a batched body, but a whole extra attention pass, the
tied head run twice, and the gateup marginal at its floor-of-floors — is **0.567 of a step**, also above
the kill line; the floor of the range (batched everything, marginals near zero) is ~0.02. So `delta`
cannot be closed by argument, and this is **not** the "cannot exceed 0.1 even pessimistically" case where
the census would have settled it free. Once `d3_run_locked.sh` has produced evidence the two arms tighten
— and while any rig term is still missing `components.py` prints **NO-VERDICT** rather than a speedup,
because a verdict computed from a partial sum is the empty-parse failure in a new costume.

**What this document is not.** Every number it eventually prints is an **estimate of `delta` from rig
components**. It is not a measurement of the engine's verify path, and cannot be until the draft path
exists: `decode_run` has no switch that submits a 2-row body over a populated KV, so no shipped binary
executes the shape being priced. Each row of the component table says which rig stood in for it and what
was transferred.

## 0. Pre-registered lines (fixed before any timing, same words in `d3_run_locked.sh`'s header)

* **KILL**: `delta > 0.379` of a step (**4.143 ms** at 4,032 lanes) — from `route_price.py`, route
  `FULL tied head`, batched verify model, `c = 0.2237`, `a = 0.7329`.
* **WIN**: the standing prize bar, `+0.30 ms/step` = speedup `≥ 1.027` (`route_price.BAR`). For the FULL
  head these two lines are **the same number**: `d_max = (1+a)/(BAR·(1+c)) − 1 = 0.3788`, which is where
  1.416× decays to 1.027×. That is pre-registered here because it changes what the harness may claim: for
  the full head there is no "wins but marginally" band, so the only informative output is *where `delta`
  lands below 0.379*, and the harness therefore prints `delta` in ms, as a fraction of a step, the implied
  speedup and ms/step, and the distance to the kill line — and does **not** invent a comfort threshold of
  its own. (The restricted-head rows in the same ladder have their own, lower, `d_max`; unchanged here.)
* **Everything in between is REPORTED, not decided here.**
* Every verdict line must name: the arm it used (batched body vs two sequential 1-row bodies), the form
  the attention number came from, and the fact that it is a rig-component estimate. `components.py` prints
  the label `[estimate of delta from rig components, not a verify-path measurement]` on both verdict lines.

## 1. The component list is DERIVED, not hand-listed

Source of truth: `tools/step_traffic_model.py::rows()` (which itself parses `engine/state.h`) is the
repo's traffic census of one token. It emits ~34 rows tagged `(family, direction, bytes, weight-or-state)`,
and its `--lanes` argument is **KV lanes, not query rows** — its own docstring says "for one token, n = 1"
and `n` is not a parameter. So it cannot emit a 2-row census, and this harness does not pretend otherwise.

What `components.py` does instead: it re-reads that census at runtime and assigns **every** row a scaling
class (`per_step` or `per_row`) from a rule table keyed by a unique substring of the model's own note, then

* **fails if any census row matches zero rules or more than one** (so a new row in the traffic model is a
  hard error here, not a silent omission — the must-catch half of the AGENTS "derived must-catch set" rule), and
* `--selftest` injects a fake census row and requires the classifier to reject it (the must-catch witness),
  and requires a real row to be accepted (the must-not-catch witness).

`per_step` = a second query row cannot change it (weight tables, the cast row, per-launch-only traffic);
`per_row` = a second row demands it again (KV walk, activation tiles, logits row, GDN state walk).
The derived result is the family list in §3 — the same nine families as `engine/generation-limits.md` §1,
reached by counting rather than by typing.

**The structural fact that keeps `delta` small and is easy to get wrong:** a 2-row body does **not** double
the launch count. `submit_wave_body` runs `n` rows through the *same* launches (48 mlp, 48 packed proj,
6 attn, the GDN launches, 49 glue), each doing more work per launch. So `delta` is the extra bytes and
dots *inside* launches whose cost is already paid — pricing it as "another token's worth of work" would
double-count ~340 launches × 3.10 µs ≈ 1.05 ms of per-step launch cost, which is listed in §4's
double-count table instead.

## 2. Which rig can actually express a second query row (read from the usage text, not guessed)

The brief's warning is right: neither GEMV rig takes `--rows` as a batch size and `--fixture` together.
`--rows` in **both** rigs is **rows per block**, a GEMV block geometry, not a query-row count. The
second query row is expressed by a different flag, and the working invocations are:

| family | flag that means "N query rows" | command (see §6 for the full lines) | what the 2-row arm really is |
|---|---|---|---|
| `proj` | `proj_gemv --tokens T` with a plan whose `XFILE` holds `T*IN` elems | `--mode burst --plan … --tokens {1,2}` | `proj_gemv_tokens<IN,TPB>` — weight row read **once per tile**; inner loop has `if (t >= nt) break`, so `--tokens 2` is **exactly 2 dots**, no waste |
| `mlp down` | `mlp_gemv --mode down --in 3584 --tokens 2` | ditto | `mlp_down_tokens<3584,2>` — TPB=2, `nt=2` → **exact** |
| `mlp gateup` | `mlp_gemv --mode gateup --in 1024 --tokens {1,2,4}` | ditto | `mlp_gateup_tokens<1024,4>`: the accumulate loop runs **`TPB`, not `nt`** (no break — `mlp_gemv_kernels.h`, contrast the `proj` kernel and the epilogue, which does break). So `--tokens 2` at `in=1024` pays **4** token dots. It is an **upper bound** on the gateup marginal; `--tokens 4` (same grid, same tile, 4 real dots) is the control that turns it into a linear estimate `(ms(t4)−ms(t1))/3`, and `ms(t2)==ms(t4)` is the witness that this reading of the kernel is right rather than a guess |
| `attn` | `attn_decode --plan … [NROWS]` (12th plan field) | `--mode burst --launch staged --softmax online --split 8` | one `attn_core` launch per attn layer with `grid = 8 × split × nrows`; `NROWS=2` is a second query row over the same populated cache, each row attending its own causal prefix. **This is the rig's split/combine wave form, not the engine's shipping `fattn`/wide wave core** — stated as a transfer in §3 |
| `lm_head` | **cannot be expressed** | — | `lm_head_gemv` takes `--rows` = rows/block and one `--h` row; there is no `--tokens`, and `gemm_algo_sweep`'s shape table has no `[248320,1024]` row and no band for one. So the head's 1→2 marginal is **bounded, not measured**: two arms in §3 (a second full pass, or the batched marginal priced by transferring the measured `proj` in=1024 ratio) |
| `gdn` | `gdn_decode --tokens N` exists but only for `gates`/`conv`; `rec` is one token per launch | not run here | bounded from the quoted family line + the quoted per-token recurrence share (0.1243 of the 0.6458 `gdn` row) |

`prototype/attn_decode/run.py --long-rows` was the brief's candidate and it is **not** enough: it selects
*pinned* prefill rows and the widest pinned long fixture (`p300`) is ~300 lanes, a thirteenth of the
operating point. The binary takes a plan file with a synthetic `[2][cap][256]` cache whose `cap` is
derived from the file size, so `make_inputs.py` writes a **synthetic 4,096-lane cache** and the two arms
are `pos=4031 lvalid=4032 NROWS=1` (lanes_last 4,032 — the ledger's own operating point) and the same
line with `NROWS=2` (lanes_last 4,033, i.e. row 1 reads the lane row 0 appended — the correct causal
geometry for a verify body). Cache *content* is synthetic; attention timing does not depend on values,
and the plan geometry is echoed by the binary (`rows=`, `core_grid=`) rather than asserted by me.

## 3. The component table (what `components.py` prints, with the arms it offers)

| # | family | does a 2nd row change it | how the number is obtained | pessimistic bound (quoted) |
|---|---|---|---|---|
| 1 | `proj` 2.7578 ms | yes — extra dots, same weight bytes | **measured**: the 48-launch engine plan (`plan_proj_t1/t2.txt`: 18 GDN in-proj at packed rows 8224 + 6 attn out-proj at 5120), `--tokens 1` vs `2`; plus a `plan_proj_i1024_t1.txt` control at `--tpb 1` vs `4` isolating the wider staged-x tile (shared 8 KiB → 32 KiB) from the extra dot | 2.7578 ms (only if the body were two 1-row bodies) |
| 2 | `mlp` 3.3628 ms | yes — same | **measured**: `down` `[1024,3584]` at TPB=2 is an exact 2-token tile (it has the `t >= nt` break); `gateup` `[3584,1024]` at TPB=4 runs **4 dots for 2 tokens** (no break in the accumulate loop), so its `t1→t2` arm is an upper bound and `(t4−t1)/3` is the linear estimate — admissible only if the `ms(t2)==ms(t4)` witness (±5 %) holds, which `components.py` *enforces* (on witness failure the bound goes into plan A, not the linear number) | 3.3628 ms (same condition) |
| 3 | `attn` 1.6121 ms | yes — a second walk of every lane | **measured** on a synthetic 4,096-row cache (48 MiB/layer, `pos=4031 lvalid=4032`, the same random K/V rows in both arms): `NROWS=1` → lanes_last **4,032** (the ledger's own width), `NROWS=2` → **4,033** (row 1 reads the lane row 0 appended). Device-verified geometry: `rows=6 → 12`, `core_grid=8x8x1 → 8x8x2`. Bytes cannot price this family — 49.55 MiB of unique KV against a 12.75 MB floor, 13 %, so it is latency-bound, which is exactly the arm the brief warned about. **Transfer:** the rig's split/combine `attn_core` wave, not the engine's shipping `fattn`/wide wave core (§4, A12 row) | 1.6121 ms = 0.1474 step |
| 4 | `lm_head` 2.1685 ms | **only if the head is run twice** | not measurable here: `lm_head_gemv` has `--rows` (rows/**block**) and one `--h` row, no token batching; `gemm_algo_sweep`'s shape table has no `[248320,1024]` row and no band for one. Two arms: (a) second full pass = D1's measured 2.127–2.221 ms — that pass is **cold**, since D1 also measured the tied table at 512.9 ms cold vs 51.5 ms warm; (b) one batched pass, priced by transferring the measured `proj` in=1024 ratio (same kernel class: one wavefront per row, bf16, k=1024, weight read once per tile) | 2.1685 ms = 0.1983 step |
| 5 | `gdn` 0.6458 ms | yes for the state walk (the recurrence re-reads S per token); the batched conv/gates part is bytes-only | bounded: quoted recurrence share **0.1243 ms** + the per-row **NON-state** gdn bytes at the 237 GB/s plateau. The state's own 2 × 18.9 MB is subtracted from the byte term because it is already inside the 0.1243 ms — that subtraction is this line's double-count guard. `gdn_decode --tokens` does not batch `rec`, so it is bounded rather than measured | 0.6458 ms = 0.0591 step |
| 6 | `glue` 0.5593 ms | almost not at all | arithmetic: launch count unchanged (49) in a 2-row body, so only the extra row bytes count: 0.30 MB read + 0.20 MB written → 0.002 ms | 0.5593 ms = 0.0511 step |
| 7 | `copy` 0.2141 ms | only the per-row part of the roll-back copies | arithmetic on the census rows: 2.66 MB r+w extra (the per-token slab copies are shared by both rows) → 0.011 ms | 0.2141 ms = 0.0196 step |
| 8 | `host_tail` 0.7163 ms | yes — a second logits row | arithmetic: second 993,280 B D2H at the quoted 62.8 GB/s device→host rate = 0.0158 ms, + ~7 µs for the second argmax (§07.4 measured fusing it at ~1 %, so the unfused second pass is the bound); `host_tail` is serial per token and not hidden behind compute. `components.py --selftest` **bounds this figure** — a byte÷rate line that divides by `62.8e6` instead of `62.8e9` also prints "a number" (it did: 15.8 ms) | ~0.023 ms |
| 9 | launches | **no** | 0 by construction — same launches do 2 rows each (and see §4: charging a second step's launches here is the double-count) | 0 |

## 4. Double-counting and omissions, stated (this is the part a summed table usually hides)

**Charged once, not twice:** the per-step launch cost (~340 launches × 3.10 µs ≈ 1.05 ms of the step) is
already in the step and a 2-row body does not add launches. `glue`/`copy` therefore contribute their
*bytes*, not their launch cost. The `attn` stage-clock sync inflation (~5.3 µs per segment) is why the
family lines are used as a **ratio denominator** where possible, and no rig's stage line is summed as an
absolute wall figure.

**Missing from `delta` as measured here, and by how much:**
* **State checkpoint/rollback on rejection.** A speculative step must be able to undo the accepted-prefix
  state: per request that is GDN 18.0 MiB + conv 0.63 MiB ≈ **18.66 MiB**, so ~0.079 ms to snapshot and
  ~0.16 ms to restore, and on reject (probability 1−a ≈ 0.27 at `a = 0.733`) that averages **≈ 0.12 ms**.
  This is a *verify-adjacent* cost, not a 2-row-body cost, and whether it belongs inside `route_price.py`'s
  `delta` is a decision for the supervisor, not this harness. It is printed beside the total, never inside it.
* **The draft step itself** is `c`, already priced by `route_price.py`; not re-charged here.
* **Whatever a 2-row body costs that no rig can reach**: the engine's `fattn`/wide wave core at 2 of 32
  query positions live in one block (`prototype/attn_wide/A12-WHY-NOT-FASTER.md`: the wide block covers 32
  query positions and has no lane split, so a 2-row body emits very few workgroups on a 40-CU machine).
  That geometry risk is real and unpriced; the rig arm does not see it. Flagged as the largest named
  unknown in the attention row.
* **`--tokens 2` batching is not free in shared memory**: the staged-x tile grows with the tile width, and
  the control arm's own line proves it — `shared 8192 B/block` at `--tpb 1` versus `shared 32768 B/block`
  at `--tpb 4` on the same 24-launch `in=1024` plan (device-verified geometry, §7), which can lower
  blocks/CU. The `tpb1`/`tpb4` pair measures exactly that term, so it lands in the estimate instead of
  being assumed away.

## 5. Files

| file | what it is |
|---|---|
| `components.py` | the derived census + the pricing arithmetic; CPU-only. `--selftest` proves the classifier can fail (an unclassified row and an ambiguous row must both be caught, a real row must be accepted, the D2H figure must sit in a sane band, and the naive bound must exceed the kill line or the brief's escape hatch applies). `--evidence DIR` folds rig stdout in, refuses on any internally-inconsistent file, and prints NO-VERDICT while any rig term is missing |
| `make_inputs.py` | CPU-only prep: extracts the engine's `proj`/`mlp` weight blobs from the shard (rank- and shape-checked), writes synthetic x tiles and a synthetic `[2][4096][256]` cache per attention layer, and writes the five plan files. `--verify` re-reads every size against every plan line — including "the 2-row x tile is exactly 2× the 1-row tile", the one mistake that would make both arms run the same shape while printing two different ms figures |
| `d3_run_locked.sh` | the runner for the main agent: device-free `--dump-plan` geometry preflight outside the lock, then ONE lock hold, admission first, arms adjacent in pairs, two passes, a behavioural canary per arm, `NO-TIMING` if admission is red |
| `evidence/` | created by the runner: raw per-arm rig stdout. Admission output goes to `evidence/admit/`, which `components.py` deliberately does **not** read, so an admission's throwaway `--iters 1` number can never enter a ratio. It is rig output, not source; nothing in it is quoted in this file |

Nothing in `engine/`, `engine/performance-status.md`, `engine/generation-limits.md`, `progress/`,
`data/refs/` or any gate threshold was touched, and nothing was committed.

## 6. Handover — the device pass (main agent only)

    bash prototype/mtp_delta/d3_run_locked.sh            # preflight + admission + 2 timed passes
    bash prototype/mtp_delta/d3_run_locked.sh --admit    # preflight + admission, no timing at all
    .venv/bin/python prototype/mtp_delta/components.py   # folds evidence into delta, prints the verdict
    .venv/bin/python prototype/mtp_delta/make_inputs.py --verify
    .venv/bin/python prototype/mtp_delta/components.py --selftest

The runner prints its own verbatim command list at the top of its output; the arms are:

    prototype/proj_gemv/proj_gemv --mode burst --plan /tmp/mtp_delta/plan_proj_t1.txt --tokens 1 \
      --variant v0-baseline --rows 8 --iters 12 --reps 3 --warm 2 --preburn 2 --label d3-t1   # t2 = tokens 2
    prototype/proj_gemv/proj_gemv --mode burst --plan /tmp/mtp_delta/plan_proj_i1024_t1.txt \
      --tokens 1 [--tpb 4] --variant v0-baseline --rows 8 ...        # tile / occupancy control
    prototype/attn_decode/attn_decode --mode burst --plan /tmp/mtp_delta/plan_attn_n1.txt \
      --launch staged --softmax online --split 8 --iters 20          # n2 = the NROWS=2 plan
    prototype/mlp_gemv/mlp_gemv --mode gateup --in 1024 --tokens {1,2,4} --x .../x1024_tT.bin \
      --wg .../L##_mlp_gate_proj.bin --wu .../L##_mlp_up_proj.bin --out ... --iters 30  # layers 0 3 11 23
    prototype/mlp_gemv/mlp_gemv --mode down --in 3584 --tokens {1,2} --x .../x3584_tT.bin \
      --wd .../L##_mlp_down_proj.bin --out ... --iters 30

Two properties a reader has to hold in mind to use those numbers:

* The **mlp arms sample 4 layers** (`MLP_LAYERS`; that binary has no plan file, one weight blob per
  process) and each burst re-touches a ~22 MiB working set that is partly L2-resident, so the mlp marginal
  ratios are **warm-arm** ratios. A warm arm overstates a bandwidth-bound marginal, which makes them
  conservative for `delta`. The `proj` arm (48 launches, 379.1 MiB distinct weights) is genuinely cold.
* The **attention arm's `--split 8` is `ceil(4033/512)`**, the lane grid both arms must share so that the
  only difference between them is `grid.z`, the row. Its `--iters 20` and the plan's 48 MiB/layer caches
  keep it inside one window; if the pass is interrupted, both attn arms must be re-taken together (a ruler
  re-taken in a different window is not an A/B).

Lock discipline (AGENTS trap row): `prototype/{proj_gemv,glue,gdn_gated_norm,attn_decode,attn_m2,
gdn_collapse,issue_rate_probe}/run.py` **self-acquire** the lock (`grep -c gpu_lock prototype/*/run.py`).
This runner invokes the **binaries**, which carry no lock code, so holding the lock around them is correct;
it invokes no `run.py` inside the hold. Its `--dump-plan` preflight runs *outside* the hold on purpose (that
mode allocates and launches nothing), so a wrong-shaped plan is caught without spending a window. Check
`tools/gpu_lock.sh status` before starting, and run it blocking, never backgrounded.

## 7. What the author ran (correctness only — no timing taken, no ms reported)

All device work was inside one `tools/gpu_lock.sh` hold each, blocking, and acquired the lock itself (TAG
`p7-d3-delta` / `p7-d3-worker-admit`); `status` reads `free` afterwards. No number below is a timing, and
nothing from these runs may enter `engine/performance-status.md` or any `results.md` perf row.

1. `bash prototype/mtp_delta/d3_run_locked.sh --admit` — **wall 3.3 s**, `util` at acquire **37 %**.
   Preflight ok (`plan_proj_t1/t2` both 48 launches, `tokens=1`/`tokens=2`), then **11/11 arms admitted**:
   `proj/t1 t2`, `proj_i1024/tpb1 tpb4`, `attn/n1 n2`, `mlp_gateup/t1 t2 t4`, `mlp_down/t1 t2`. The script
   printed no millisecond figure and exited before the timing phase by design.
2. A standalone one-hold pass over the same eight `--mode once` invocations before the preflight existed —
   **wall 0.9 s**, `util` at acquire **90 %**, all rc=0. Geometry fields read off the rigs' own lines:
   proj `launches=48` in both arms with `tokens=` moving 1→2; attn `rows=6→12` and `core_grid=8x8x1→8x8x2`
   (so `grid.z` really is the row, which is the claim the attention term rests on); mlp `rows=3584 in=1024`
   and `rows=1024 in=3584` with `tokens=` moving 1→2 and `tile=4`/`tile=2` fixed. **No timing taken**: the
   `RESULT` lines of an admission carry an `ms` field from a single warm `--iters 1` rep under 90 % util; it
   was written to `evidence/admit/`, is discarded, and `components.py` does not read that directory.
3. CPU-only, no lock: `components.py --selftest` PASS (33/33 census rows classified 1:1, `delta_max` 0.3789,
   naive bound 1.1007 of a step, the D2H units witness inside its band); `make_inputs.py --verify` PASS
   (126 rank-checked weight blobs, 529.1 MiB, five plan files, 578 MB in `/tmp/mtp_delta`); and the
   mis-wiring test — editing one `tokens=1` into `tokens=2` inside a `t1` evidence file makes
   `components.py` print `RESULT lines disagree on tokens` and **refuse to price**, which is the guard that
   stops a stale or swapped arm from reading as a clean null.

