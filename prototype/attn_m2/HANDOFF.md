# `prototype/attn_m2` — device handoff (second window)

The first window (`attn-m2`, 2026-10-16, `results.md` §2–§4) proved bit-identity where it could run, was
blocked from the funded widths by a launcher bug, and its negative controls were vacuous. Both bugs are
fixed and the fixes are CPU-checked (§6, §3). What is left is a device pass, and it is bigger than last
time in two ways: the four previously-refused plans now launch, and the plans that can actually price the
route are new (`wave64-at-*`, §7).

**Lock protocol.** Neither `run.py` nor `attn_m2` self-locks `tools/gpu_lock.sh` — deliberately: three gate
drivers in this repo self-lock and a caller that already holds the lock deadlocks on them (AGENTS.md). **You**
hold the lock around all of it, and release before invoking anything that self-locks. Check (the AGENTS.md
pattern `grep -c gpu_lock` over-counts, because this file *discusses* the lock): what must be zero is a call
site — `grep -n "gpu_lock.sh\s*\(acquire\|status\)\|subprocess.*gpu_lock" prototype/attn_m2/*.py` prints
nothing, and the two `grep -c gpu_lock prototype/attn_m2/run.py` hits are the prose at lines 12 and 46.

Steps 1–4 are correctness and take well under a minute each on device. Step 5 is the timing and is
main-agent work by rule.

---

## 0. Take the lock, record util

```bash
tools/gpu_lock.sh acquire attn-m2-2
tools/gpu_lock.sh util          # record it; identical clean runs have read 24-91 % on this box
```

## 1. Build, then run both guards (no device in either)

```bash
bash prototype/attn_m2/build.sh
.venv/bin/python prototype/attn_m2/m2_model.py guard          # CPU mirror; needs no build
./prototype/attn_m2/attn_m2 --geom-guard                       # the C++ closed forms; host-only
```

Both must print `geom-guard: PASS` with the same numbers: **13,744 geometries**, worst single-block union
**518 of 520**, worst row slice **512 of 512**, **0 phantom refusals, 0 leaks**, **16 real plan geometries
accepted**. If they disagree with each other, stop — two spellings of the geometry have diverged, and the
kernel's launch refusal is one of them. (`m2_model.py selftest` carries the same six assertions, so a
green selftest already implies a green guard.)

## 2. Write the plans and read the control matrix (no device)

```bash
.venv/bin/python prototype/attn_m2/run.py plans        # 12 plans into /tmp/attn_m2/
.venv/bin/python prototype/attn_m2/run.py negmatrix    # which plan proves which injection
.venv/bin/python prototype/attn_m2/run.py selftest     # the driver's verdict logic, against the stub
```

`negmatrix` is derived, not typed: at 1–8 lanes `partrow`/`partalias` score an empty must-catch set and are
excluded from those plans, and a lone 1032-lane row's `maskwide` set is empty. `run.py negative` refuses to
run an injection with no plan that can catch it — an empty list cannot print PASS.

`selftest` is new and it is the reason steps 3–4 can be trusted to mean what they print: it drives the real
driver against `stub_harness.py` (same CLI, same `RESULT` lines, same dump layout, no attention) through
`--bin`, so the adjudication rules are witnessed on CPU before a device window spends itself on them.
Eleven cases; the two that matter here are an m2-only injection printing **PASS** — which was impossible
before the class scoping was fixed — and a form-neutral injection still failing when the two forms disagree.

## 3. THE BIT-IDENTITY RUN — the gate, now over every funded width

```bash
.venv/bin/python prototype/attn_m2/run.py ab
```

One process per plan: `attn_prep → attn_tile_wave → attn_combine` and `attn_prep → attn_m2_wave →
attn_combine` over separate cache replicas and separate outputs, then a per-row **bit** diff; `attn_combine`
is the shipped merge on both sides. It prints one `VERDICT` line per plan and exits non-zero if any plan
differs **or fails to launch** — `verdict=NO-LAUNCH` is a failure, which is the check that would have caught
§2's bug inside the window.

Admissible answer: `ndiff_aout=0 ndiff_gated=0` on every row of all 12 plans. **Check whether this window has
already run before spending a lock hold on it**: decision log D19 (window `attn-m2-2`) records "bit-identity is
green on device at every width including the 1032-lane pinned HF row and 64-row waves", which is this step's
answer — but `results.md` §2 still carries the first window's four `REFUSED` rows and no §2b table, so what
may be missing is the write-up, not the run. Re-run only if the table cannot be filled from that window's log.
New since the first window:

* **`straddles-block-2` (1024–1027, split 3), `straddles-block-3` (1536–1539, split 4),
  `pair-widest-1024` (1021–1024, split 2), `tail` (the real HF row at 1032 lanes, split 3)** —
  refused last time, and the first four rows of `results.md` §2's table are still `NOT MEASURED`.
* **`wave64-at-{1024,2048,4096}` and `wave16-at-4096`** — waves of tens of streaming rows, which
  is the shape the route is funded for and the shape §4 never touched.

Those are the **tags** `--only` takes; the file on disk is `/tmp/attn_m2/plan-<tag>.txt`, so the `plan-`
prefix belongs to the path and not to the filter.

If a row differs here and the CPU mirror says it should not, suspect in this order: barrier uniformity (the
pair's union bounds are block-uniform; a lane-dependent `return` or a barrier in a divergent branch is the
bug — `diff_rows`' own fault was exactly that, and it aborted the queue with `0x1016`), then the per-row
`s_kself`/`s_vself` slots, then the V-stage slot arithmetic (`(rr - c_lo) * DH`, never a lane index).

## 4. THE NEGATIVES — now with the right baseline

```bash
for n in maskwide partrow partalias readpast dropcurrent dimhalf gqaoff slotk slotv; do
    echo "=== $n ==="
    .venv/bin/python prototype/attn_m2/run.py negative --neg $n
done
```

Per plan it prints one line:

```
NEGVERDICT plan=straddles-block-3 neg=partrow class=m2-only rows=4 lanes=1536-1539 split=4 must=32 \
           moved_m2=32 moved_ship=0 blind_hits=0 forms_agree=no \
           forms_check=control-group-must-not-move verdict=PASS
```

and what each field means is not obvious, so:

* **`moved_*` is injected-vs-clean on the SAME form**, from two runs of the plan (`--neg none`, then
  `--neg X`), compared per `(row, head)`. This is the fix for §3's second defect: measured form-vs-form, a
  form-neutral injection is invisible *by construction*, so `readpast`/`dropcurrent`/`dimhalf`/`gqaoff`\
  could never have registered last window at any geometry.
* **`moved_ship=0` is the expected answer for `maskwide`/`partrow`/`partalias`** — the shipped kernel does
  not implement `NEG_M2_*`, so it is the control group. If the shipped form moves under those, something is
  reading memory it should not.
* **`blind_hits=0`**: no row the geometry says cannot be reached may move. This is the half of the control
  that catches an over-wide mask or a leaked shared stage, and it is why a must-catch set alone is not a
  control.
* **`forms_agree` means something only for `class=shared` and `class=cache`** — there the two forms do the
  same wrong thing and must stay bit-identical to each other even while both are wrong (the §1.2 claim).
  For `class=m2-only` the shipped form is the control group, so `forms_agree=no` is the **expected** answer
  and `verdict=PASS` next to it is correct; before this scoping was fixed those three injections could not
  print PASS at all (results.md §3, defect 3). `forms_check` names which rule the verdict applied.
* **`class=cache` (`slotk`/`slotv`)**: injected inside `attn_prep`, so the mirror has no derived set for
  them; the assertion is that **both** forms move, because a shared stage that swallowed a cache
  perturbation would show up here and nowhere else.

`headcarry` and `gqa_head0` are mirror-only (a device version would race an aliased shared slot, and the
winner is whichever warp landed last) — they are covered by `m2_model.py negative`.

## 5. THE TIMING — burst above the bandwidth knee (main agent)

§4's numbers are 0.90× on waves of **16–32 blocks**, and this box's read bandwidth only plateaus at
233–240 GB/s **from ~80 blocks up** (`devices/strix-halo-memory-bandwidth.md`). Below the knee, M = 2's
halved grid cancels its halved bytes by construction, so those numbers say nothing about the route. The
measurement that does is on the four `wave*-at-*` plans, which are above the knee (§7):

```bash
# warm pages first are already handled by the harness (one warm launch per form before the reps)
for p in wave64-at-1024 wave64-at-2048 wave64-at-4096 wave16-at-4096; do
    ./prototype/attn_m2 --plan /tmp/attn_m2/plan-$p.txt --burst 40 --reps 5
done
```

`--burst/--reps` is your path, not the driver's; `RESULT case=… best_ship_ms=… best_m2_ms=… speedup=…` is
the line, both forms alternate inside one rep, and `steady_clock` is the timer (`hipEventElapsedTime`
returns exactly 0 for sub-ms launches here). Quote the pair and the spread, never one number per side.

Predictions to compare against, from `m2_model.py regime` (census ÷ 237 GB/s, so they are arithmetic and not
a guess):

| plan | blocks ship → m2 | floor ship | floor m2 | bytes ratio β | §7 prediction if both reach the plateau |
|---|---|---|---|---|---|
| `wave64-at-1024` | 384 → 192 | 0.582 ms | 0.298 ms | 0.512 | ≈ 1.95× |
| `wave64-at-2048` | 640 → 320 | 1.148 ms | 0.591 ms | 0.515 | ≈ 1.94× |
| `wave64-at-4096` | 1024 → 512 | 2.248 ms | 1.125 ms | 0.500 | ≈ 2.00× |
| `wave16-at-4096` | 256 → 128 | 0.565 ms | 0.283 ms | 0.500 | ≈ 2.00× |

Three outcomes, and they mean different things:

1. **≈ 1.5–2× above the knee** → §7's explanation stands, the route is priced, and §5's decision rule applies
   (ship it behind a width/wave-size gate by lane count, with the numbers that earned it).
2. **≈ 0.9 above the knee too** → §7's plateau explanation is **falsified**, and the occupancy or
   union-residue explanation returns. Then alias `s_k`/`s_v` (22,656 → 18,560 B → 3 blocks/CU → 24
   wavefronts) and re-measure; do not bother with a 2-row/4-warp block, which leaves LDS alone and therefore
   *halves* wavefronts to 8 (§7.1).
3. **Wins at 4096, loses at 1024** → ship behind a width gate, and say which width, from these numbers.

Worth taking while the lock is held, if a profiler is available: realized blocks/CU and wavefronts/CU for
`attn_m2_wave` against `attn_tile_wave` (results.md §4 still says NOT MEASURED). The compile census says
VGPR 50, SGPR 64, scratch 0, LDS 22,656 B → 2 blocks/CU = 16 wavefronts against the shipped form's 5 blocks
= 20, and §7.1 argues that figure is not the binding term — a profiler either confirms that or retires
§7.1.

## 6. Release, then paste into `results.md`

```bash
tools/gpu_lock.sh release attn-m2-2
```

Fill §2's four REFUSED rows and the four new wave rows, §3's re-run column, and §4/§7's `NOT MEASURED`
cells. If the guard's numbers move, they must move in both spellings at once — they are the same assertion
written twice on purpose.

---

### Things deliberately not done, so you are not surprised

* No timing in the driver, and no `--burst` from it. The harness's burst path exists and shares one
  `core_args` spelling with the gated path, so a timing cannot be taken on a different launch plan from the
  one that was gated.
* `prototype/attn_decode/` untouched, `engine/` untouched. M = 2 is not wired into the engine's route chooser.
* `diff_rows`' uniform-reduction fix and the `--burst/--reps` timer are left exactly as you left them.
* The refusals that blocked the last window were a **unit** bug (`total` lane rows summed over lane blocks
  compared against `ATTN_M2_STAGE`, a per-block bound), not the row-index/lane-count mix-up recorded in the
  decision log's D18 item 3. The guard now asserts the bound and the refusal's *attribution* (§6), and the
  bound genuinely holds: worst 518 of 520, at `lanes_last=4094 split=8` — there is no real width where the
  union exceeds the staging, so staging is not a limit on M = 2.
* The byte counters are still not admissible evidence (D9, ~100× calibration failure): the traffic claim
  stays bytes-demanded arithmetic plus the burst above, which is why `regime` prints arithmetic.
