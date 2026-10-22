# HEADBATCH result — the batched tied head is free (main agent, 2026-10-22)

Taken by the main agent, blocking, zero subagents active, one `tools/gpu_lock.sh` hold per phase.
Commands: `bash prototype/mtp_headbatch/run_locked.sh` (wall 18 s; preflight + admission + two timed
passes), then `.venv/bin/python prototype/mtp_headbatch/price_head.py --evidence /tmp/mtp_headbatch/evidence`.
Bands were fixed before any millisecond was printed (`HEADBATCH-HANDOVER.md` §0 and `price_head.py
--bands`, duplicated in the runner header), so nothing here is a post-hoc threshold.

## The number

| arm | best ms (over rows/block 8/16/32) | R vs M=1 |
|---|---|---|
| M=1 (one query row) | 2.1265 | 1.0000 |
| **M=2** | **2.1305** | **1.0019** |
| **M=4** | **2.1538** | **1.0128** |
| shipped `lm_head_gemv` kernel, same process | 2.1272 | — |

**VERDICT: PLAN A STANDS AS PRICED** (band R ≤ 1.10). Folding the measured marginal back into `delta`:
**delta = 1.4847 ms = 0.1358 of a step → 1.247× (+2.699 ms/step)** — slightly *better* than the 1.231×
that D3 estimated via the `proj` transfer.

`fetch_per_weight = 1.000000` on every M arm, with the calibration arms reading 2.000000 (`extra_pass`)
and 1.25 (`extra_chunk`) and `short_grid` opening 16/32 poison holes — so "the table is read once for
every row served" is a witnessed byte fact, not an arithmetic assumption. Cross-M and shipped-kernel bit
identity were 0 differing elements; against the pin, `argmax 271=271 | cos 1.000003 | meanrel 0.00140 |
ndiff 15 (0.006 %) | n_bad 0`.

## What this changes for P7, beyond plan A

**M=4 costs 1.3 %.** That is the cheap answer to a question the queue had not asked yet: the head is not
the obstacle to a **deeper draft** (`k>1`). Since the accept-by-depth profile *rises* (0.588 at positions
0–7 → 0.808 at 32–63), the draft depth is now priced by the draft *block* and the verify body only — the
~485 MiB head no longer scales with `k`. If the block behaves like the head, `k` is nearly free, and the
economics move from `a > c` toward `a·(k+1) / (1 + k·c_block)`.

## What this does NOT settle

- `a` on the engine's own bits (carried 0.7329, kill line 0.257) — still the load-bearing unknown, and it
  needs a draft path to exist (D4 protocol is in `MTP-IMPL.md` §3).
- The draft **block** at engine shape (39.0107 MiB, 15 tensors) — priced by transfer at 0.22–0.25 ms, and
  D2's lesson applies: a rig that stages it differently from the engine measures the rig.
- The batched 2-row **verify body** over populated KV, and the rollback surface D3 deliberately excluded
  (~0.12 ms/round). See `MTP-IMPL.md` §1.
