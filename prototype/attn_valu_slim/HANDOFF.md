# HANDOFF — the device pass for `prototype/attn_valu_slim`

Nothing in `prototype/attn_valu_slim/` has touched a device. Every claim in `results.md` is either
static (from `hipcc -O3 --offload-arch=gfx1151`) or CPU-only. The A/B below is the witness, and it is
main-agent work: it must be run blocking, in one `gpu_lock` hold, with **zero subagents active**.
This binary does **not** self-lock (`grep -c gpu_lock prototype/attn_valu_slim/*` returns 0 for the
harness), so the caller holds the lock — and must release it before invoking any self-locking gate
driver (`prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py`).

## 0. Census, no device (already run; re-run to confirm nothing drifted)

```
.venv/bin/python prototype/attn_valu_slim/isa_census.py \
    --out prototype/attn_valu_slim/isa_census.output.txt
.venv/bin/python prototype/attn_valu_slim/touch_probe.py
```

Expected: the `attn_slim0_wave` row equals the `attn_tile_wave` row everywhere (that is the control —
the copy must be instruction-identical to the shipped kernel), and `touch_probe.py` prints
`PROBE summary PASS`. If slim0 ≠ shipped, stop: the copy drifted and every delta below is void.

## 1. Build

```
prototype/attn_valu_slim/build.sh
```

## 2. Plans (CPU-only, reuses m2's writer so the geometries are the ones the wave work already proved)

```
.venv/bin/python prototype/attn_m2/run.py plans        # -> /tmp/attn_m2/plan-<tag>.txt
ls /tmp/attn_m2/plan-*.txt
```

## 3. The bit-identity A/B — the gate

For **every** plan, with both levers:

```
tools/gpu_lock.sh acquire attn-valu-slim && tools/gpu_lock.sh util
for p in /tmp/attn_m2/plan-*.txt; do
    ./prototype/attn_valu_slim/attn_valu_slim --plan "$p" --slim 1
done
tools/gpu_lock.sh release attn-valu-slim
```

**Admissible answer: `ndiff_aout=0 ndiff_gated=0` on every `RESULT case=` line of every plan, and
`cases_with_any_difference=0`.** These levers claim bit-identity, not a band, so there is no
"close enough". `--slim 2` (dot lever alone) and `--slim 3` (fold lever alone) are the same run with
one lever each — required if step 3 fails, to say which lever did it.

## 4. Run it twice — the nondeterminism check

Step 3 must be run **twice** on the same plans and the two logs diffed. The fold lever is an inline
`ds_bpermute_b32` whose `s_waitcnt lgkmcnt(0)` is written by hand, because `SIInsertWaitcnts` does not
insert waits for inline-asm DS ops. A missing or misplaced wait is *racy*, and a racy permute shows up
as run-to-run disagreement, not as a stable wrong answer. If run 1 and run 2 differ, the lever is
dead as written and the fix is the wait, not the gate.

## 5. Every injection, still bit-identical

The two spellings must agree not only on the clean path but in every corner the injections create
(zero scores, huge scores, half-dim dots, wrong cache slots, GQA off). With the same plan loop, add
`--neg NAME` for each of: `qswap gatehalf ropefull ropenone gqaoff nomax carry readpast dropcurrent
dimhalf slotk slotv`. All must still print `ndiff_aout=0`.

`--neg dimhalf` deserves a line of its own: it is the reason `nd` is a runtime value, and lever 0
exists to stop that runtime value from rolling the hot dot. If `dimhalf` is the one case where slim
disagrees with the shipped form, the peel (the second copy of the loop, `vslim_dot<4>`) is wrong and
that is the bug to fix first.

Note the direction of this control: unlike `attn_m2`'s NEGVERDICT, this harness diffs *form against
form under the same injection*, so **the expected answer is 0 everywhere**, and a non-zero is a real
disagreement between two spellings that were argued to be identical.

## 6. The timing that prices the route (main agent, one lock hold, zero subagents)

```
tools/gpu_lock.sh acquire attn-valu-slim-burst && tools/gpu_lock.sh util
for p in wave64-at-1024 wave64-at-2048 wave64-at-4096 wave16-at-4096; do
    ./prototype/attn_valu_slim/attn_valu_slim --plan /tmp/attn_m2/plan-$p.txt --slim 1 \
        --burst 40 --reps 5
done
tools/gpu_lock.sh release attn-valu-slim-burst
```

One burst, both forms alternating inside each rep (the harness does this), best-of-reps,
`steady_clock` — the `hipEventElapsedTime`-returns-exactly-0 trap and the "a re-taken ruler is not an
A/B" trap are both already handled by the harness, so quote the *pair*, not two windows. Record the
`util` at the time and then discount it. A `RESULT case=… burst=… best_ship_ms=… best_slim_ms=…` line
is the number; `results.md` §4 predicts ≤ 23.6 % on the m2 wave and ≤ 0.60 ms of 2.5402 ms at 4096,
with a floor of 0 — and the ship form at 4096 is at the traffic plateau (D19), so a **flat ship-side
result is consistent with D19 and is not a falsification of the lever**; the arithmetic-bound side of
that pair is m2, and if this A/B is flat on ship but the m2 form is where the win belongs, the next
step is applying the same two spellings to `prototype/attn_m2/attn_m2_kernels.h` (its own score loop
and `warp_sum` call are the same shape — `results.md` §2's last row is the census that says so).

## 7. If the A/B passes

The lever is bit-identical, costs no numerics decision, costs no occupancy (VGPRs 47 → 47, LDS
12,288 B → 12,288 B), and needs:

1. `git status --ignored prototype/attn_valu_slim` — the built binary is `prototype/attn_valu_slim/attn_valu_slim`
   and `.gitignore` must not be swallowing the directory (the trap row in `AGENTS.md`).
2. Commit by topic: (a) the kernel + census + probe, (b) the harness + build.sh + this HANDOFF, (c)
   `results.md` with the device numbers filled in, (d) the D20 correction — the "causal-mask guards"
   of D20 item 4 are shuffle-lowering guards, and the shipped form has no per-element causal guard
   (m2 does, at 3 ops).
3. Then, as a separate decision: port the two spellings to `attn_m2_kernels.h` and re-run
   `prototype/attn_m2`'s own gates, which is where D19 says the arithmetic-bound residual lives.
