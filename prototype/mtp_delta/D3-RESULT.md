# D3 result — `delta`, measured from rig components (main agent, 2026-10-22)

Taken by the main agent, blocking, zero subagents active, inside one `tools/gpu_lock.sh` hold per phase.
Command: `bash prototype/mtp_delta/d3_run_locked.sh` (wall 22 s, 32 arms ok across admit + two timed
passes), then `.venv/bin/python prototype/mtp_delta/components.py --evidence prototype/mtp_delta/evidence`.

## The verdict

| arm | delta (ms) | of a step | speedup | vs bar |
|---|---|---|---|---|
| **plan A** — batched 2-row body, batched head, linear gateup marginal | **1.6454** | **0.1505** | **1.231×** (+2.525 ms/step) | **WIN** (bar 1.027×) |
| **pessimistic** — batched body, head run twice, whole extra attn pass, gateup at upper bound | 5.1712 | 0.4729 | 0.961× (−0.421 ms/step) | **KILLED** |
| **two sequential 1-row bodies** | 10.9357 | 1.000 | — | **KILLED**, structurally |

Kill line was pre-registered at `delta_max = 0.379` of a step (`route_price.py`), before any timing.
Plan A sits at 0.1505, under it; the pessimistic arm sits above it. So **P7 lives or dies on the shape of
the verify body, not on the accept rate**: the engine must verify both candidate rows in ONE batched body
and read the tied `lm_head` once for both. The third row above says what happens if it is built as two
one-row calls — that is dead whatever else is true. This matters because `decode_step.h:411-424` already
runs `n`-row layer-major bodies for one sequence (the P8 finding), so the machinery is there; what does
not exist is a path that submits two candidate rows over a populated KV.

## Rig numbers behind it (paired, same window)

| rig arm | t=1 | t=2 | second-row marginal |
|---|---|---|---|
| `proj_gemv` (24 jobs, in=1024) | 3.4388 ms | 3.6989 ms | **+7.6 %** — weights dominate, second row nearly free |
| `mlp_gemv --mode down` | 0.0128 ms | 0.0225 ms | **+76 %** — the skinny `down_proj` is the worst marginal found |
| `mlp_gemv --mode gateup` | 0.0325 ms | 0.0351 ms (t4: 0.0413) | +8 %; `(t4−t1)/3` linear estimate admitted per the handover |
| `attn_decode` (synthetic 4,096-lane cache, `NROWS=1→2`) | 1.5144 ms | 1.4742 ms | **second row measured *faster*** — see caveat |

Two honest caveats, both the fitter's own, not added after the fact:

1. **`attn` is the term this harness cannot measure.** A second row read *faster* than one is not a
   speedup claim; it is the synthetic cache and the split geometry not transferring (the rig is at 13 % of
   its own byte floor, so bytes cannot price it). The harness therefore keeps `attn_whole_pass`
   (1.6121 ms, a whole extra pass) as the pessimistic substitute rather than believing −0.04 ms.
2. **`lm_head_batched` is a plan, not a measurement** — the batched 2-row head does not exist. It is the
   single largest swing term between plan A and the pessimistic arm (±2.17 ms), and it is the first thing
   the funded work must measure, not the last.

Excluded deliberately (handover §4): state checkpoint/rollback, ~0.12 ms per round, which is a *draft-round*
cost, not a verify-row cost, and belongs to the accept-loop pricing, not to `delta`.

## Two instrument defects found while taking it

- **The runner deleted its own first pass.** Each phase began with `rm -f "$dir"/*.txt`, so pass 2 erased
  pass 1's evidence before anyone read it — and the run still looked green. Now it clears only
  `$dir/$phase-*.txt`. (Discovery cost: the first "FAIL p1 … (see …p1-…txt)" pointed at a file that no
  longer existed.)
- **The `proj_i1024 tpb4` occupancy control is not expressible on this rig** and exited 1 at its probe: it
  enumerates `(in=2048, tpb=4)`, which is not instantiated (1024 takes tpb 1/4, 2048 takes 1/2), even
  though the plan is 100 % `in=1024`, and burst mode has no `--in` to aim it. `components.py` already
  guards the control with `len(...) >= 2`, so dropping it costs the tile-width de-confounder and nothing
  else — the `proj` marginal comes from the direct `t1`/`t2` pair. Recorded in the runner, not silently.
