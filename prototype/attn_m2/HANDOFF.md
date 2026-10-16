# `prototype/attn_m2` — device handoff

Everything that could be settled without a GPU is settled and green (`results.md` §1). What remains needs
an exclusive box, so it is yours. Steps 1–5 are correctness and take well under a minute each; step 6 is
the only timing step and it is main-agent work by rule.

**Lock protocol.** `run.py` and `attn_m2` do **not** self-lock `tools/gpu_lock.sh` (deliberately: three
gate drivers in this repo do self-lock and a caller that already holds the lock deadlocks on them). So
**you** hold the lock around all of it, and release it before invoking anything that self-locks. Check with
`grep -c gpu_lock prototype/attn_m2/*.py` — the answer is 0.

---

## 0. Take the lock, record util

```bash
tools/gpu_lock.sh acquire attn-m2
tools/gpu_lock.sh util          # record it; identical clean runs have read 24-91 % on this box
```

## 1. Build (compile-only, no device; already done once here)

```bash
bash prototype/attn_m2/build.sh
```

Resource census, if you want it again (`-c`, no device):

```bash
cd prototype/attn_m2 && hipcc -O3 --offload-arch=gfx1151 -Rpass-analysis=kernel-resource-usage \
    -c attn_m2.hip.cpp -o /tmp/m2.o 2>&1 | grep -A8 attn_m2_wave
```

Expected: VGPR 50, SGPR 64, ScratchSize 0, LDS 22,656 B/block → 2 blocks/CU = 16 wavefronts, against the
shipped tile form's 5 blocks = 20. That occupancy cost is the thing most likely to make this route a
wash, so it is worth confirming on device in step 6.

## 2. Write the plans (no device)

```bash
.venv/bin/python prototype/attn_m2/run.py plans
```

Eight plans in `/tmp/attn_m2/` (scratch lives outside the tree, like `attn_decode/run.py` does). Each case
is placed on a boundary M = 2 changes; the content is pinned rows cycled over a populated cache
(`m2_model.wide_plan`), because no fixture on this box has a prompt long enough (§20.2) and a bit-identity
A/B needs identical operands, not HF ground truth.

## 3. THE BIT-IDENTITY RUN — the gate

```bash
.venv/bin/python prototype/attn_m2/run.py ab
```

One process per plan: `attn_prep → attn_tile_wave → attn_combine` and `attn_prep → attn_m2_wave →
attn_combine` over separate cache replicas and separate outputs, then a per-row **bit** diff
(`aout` and `gated` compared as raw 16-bit patterns; the block total is a real warp-shared reduction, not a
per-thread store). `attn_combine` is the shipped merge for both forms — the two forms share the partial
layout slot for slot, which is what §20.1 argues for.

Admissible answer: **`ndiff_aout=0 ndiff_gated=0` on every row of every plan.** The `ROWDIFF` lines name
row, position, lane count and the count, so a partial agreement cannot be read as a pass.

* The case that proves it for the geometry M = 2 changes: **`plan-pair-at-maxsc`** (rows of 512 and 513
  lanes in one pair — one one-pass, one streaming, so the two rows of the block take different routes and
  have different slices) and **`plan-straddles-block-3`** (split 4, four lane blocks per pair).
* The case that proves it against real HF data above MAXSC: **`plan-tail`** (`wide1024f`'s `decode7`, one
  row at 1032 lanes, split 3). It is also the case that shows the route's limit: a lone row has no partner
  and the ratio is ~1.001.

If a row differs here and the CPU mirror says it should not, suspect in this order: barrier uniformity
(the pair's `kmax`/union bounds are block-uniform min/max — a lane-dependent `return` or a barrier inside a
divergent branch would be the bug), the per-row `s_kself`/`s_vself` slots, then the V-stage slot arithmetic
(`(rr - c_lo) * DH`, never a lane index).

## 4. THE NEGATIVES — controls that must fire

```bash
for n in maskwide partrow partalias readpast dropcurrent dimhalf gqaoff slotk slotv; do
    echo "=== $n ==="
    .venv/bin/python prototype/attn_m2/run.py negative --neg $n
done
```

Each must move exactly the derived set, printed per row. The three that only this form can express:

* `maskwide` — a row attends the pair's union instead of its own prefix. The bug M = 2's shared walk
  invites, and the one the admission gate exists to catch. Must move the rows whose own slice is narrower
  than the union in some lane block; must **not** move a row whose slice already equals the union (the CPU
  derivation is `m2_model.maskwide_rows`, and its first hand-written version got this backwards at the
  one-pass/streaming boundary, which is why it walks the geometry now).
* `partrow` — the two rows of a pair write one split-partial slot. Must move every **streaming** row of a
  pair and must not move a one-pass row (a one-pass row writes no partial, so there is no slot to steal —
  the reason `plan-pair-shortest` cannot be the case that proves this one).
* `partalias` — §19's own bug re-injected, keyed by head-in-group. Must move streaming rows' heads 0..G−1.

`slotk`/`slotv` matter more here than in the shipped form: they perturb the cache, and this form reads the
self lane from a dedicated stage slot. If they do not move a row, the shared stage has laundered the
difference and the "every operand comes from the memory the shipped form reads it from" claim is false.

## 5. Refusals (cheap, do not skip)

```bash
.venv/bin/python prototype/attn_m2/run.py ab --split 1     # must be auto-raised, not over-run
```

`attn_m2_ok` must refuse `score_f32` and `fused`, a `split` shallower than `split_for_lanes`, and a union
above `ATTN_M2_STAGE` (520 lanes; worst real pair is 519). A refusal prints a reason and exits 1. There is
no case where over-running the stage is safe on this part — LDS over-run does not fault, it corrupts a
neighbour (§13/§14).

## 6. THE TRAFFIC CLAIM — arithmetic + one paired timing (main agent)

The counters do not settle it on this part: GL2C calibration failed by ~100× (decision D9), so no byte
counter reading is admissible. The claim is instead:

> **Which arithmetic.** The census is fetched **lane rows with multiplicity**: for each block, the width of
> the union of the two rows' lane slices (K, staged once per block) plus, per pass-2 chunk, the width of
> the union of the two rows' current tiles (V), plus each row's self-lane row. Bytes = lane rows × 512 B ×
> 2 KV heads × 2 (K,V) × 6 layers, divided by the measured 237 GB/s read ceiling. The shipped side's census
> is one block per (row, KV group, lane block) staging its own row's slice, i.e. `Σ_lanes` per group — the
> form §20.5 corrected (the re-use it really has is G, at block level, not across rows).
> `prototype/attn_m2/m2_model.py traffic` prints it, and `./attn_m2` prints the same census per case from
> the kernel's own closed forms (`attn_m2_pair_union` / `attn_m2_row_union`), so a disagreement between the
> C++ and Python spellings of the traffic model is visible without a device at all
> (`m2_model.py selftest`, check 1).
>
> The arithmetic reproduces the funding doc's floors (27.2 / 108.8 / 435.0 ms) and its M = 2 prizes
> (13.6 / 49.9 / 194.6 ms) to within 0.3 ms. What it cannot do is say what the kernel *costs*, because M = 2
> also spends occupancy (16 wavefronts/CU vs 20).

So the device half of the traffic claim is **one timing**, taken by you, paired inside a single lock hold:

1. Instrument: `attn_m2.hip.cpp` has **no timer**, on purpose (a worker may not take timings). Add a
   `--burst N` loop around `run_form` — same launches, N iterations, one `hipDeviceSynchronize` at the end,
   `steady_clock` around the burst, not `hipEventElapsedTime` (it returns exactly 0 for sub-ms kernels on
   this part). Divide by N. Keep both forms' bursts alternating inside one process so a mid-run
   disturbance hits both.
2. Widths that carry the headline: **1024 lanes and 4096 lanes.** The byte model says the prize is 13.6 ms
   at 1024 and 194.3 ms at 4096 if the kernel runs at the floor; it also says the arithmetic floor
   (3.53 / 56.5 ms) is untouched by M = 2, so at 1024 lanes the win is bounded by the traffic share.
3. Best-of-N, both forms, same window, quote the pair and the spread — a number re-taken in a different
   window is not an A/B (that trap cost a retracted headline on 2026-10-16).
4. Warm the pages first: the first GPU pass over fresh pages runs ~35 GB/s.

## 7. Release, and paste into `results.md`

```bash
tools/gpu_lock.sh release attn-m2
```

Fill §2 (bit-identity per row), §3 (negatives), §4 (timing pair + realized occupancy). §5 of `results.md`
is the decision rule: any non-zero diff means fix the kernel; bit-identity green but timing flat means the
occupancy cost is binding and the named levers are aliasing `s_k`/`s_v` or a 2-row/4-warp block.

---

### Things deliberately not done, so you are not surprised

* No timing anywhere in the prototype, no `ms` figure in any file here.
* `prototype/attn_decode/` is untouched: no change to `attn_decode_kernels.h`, its `run.py`, or its gate.
  `engine/` is untouched — M = 2 is not wired into the engine's route chooser, and nothing should read this
  as a shipped path until §2 and §6 are green.
* `headcarry` (row 1 of a pair seeded from row 0's softmax state) is **mirror-only**: on a device, two rows
  in one block sharing a state slot would race and the winner is whichever warp landed last, which is not a
  gate witness. The CPU version is the honest witness for that class.
* M = 4 and M = 8 are a byte model in `traffic`, not a kernel. The static_assert in the header says so, and
  the union bound has to be re-derived for a wider tile.
* The shipped mirror's `gqaoff` injection moves a row's self operand while both kernels stage the self row
  from the block's own group, so those two injections are not usable as form-neutral bit-identity checks
  (results.md §1.3, finding 1). Not fixed here, because `prototype/attn_decode` is frozen and its own gate
  only asserts "moved", which the wider set still satisfies. Worth a separate look by whoever owns that op.
