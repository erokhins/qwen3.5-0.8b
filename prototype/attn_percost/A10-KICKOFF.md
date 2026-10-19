# A10 kickoff prompt — paste this into a fresh session to start the rewrite

```
Start the A10 attention geometry rewrite in /home/erokhins/erokhins_projects/qwen3.5-0.8b.

Read first: prototype/attn_percost/A10-REWRITE-BRIEF.md (the plan: R1..R8 with dependencies, the R1 correction,
and the "what would make this the wrong call" section), bench/performance-status.md §52-§54 (funding line, L-c
DISPLACED, the A11 census and the abort line), A9-LONGCTX-CENSUS.md §3/§5 (traffic + wasted work), AGENTS.md.
Confirm the current tip and engine defaults first — the ledger moves.

Order, cheapest first. Do not skip step 1, and do not run a device window before step 3.
1. SOURCE READ (CPU only): establish from engines/llama.cpp/ggml/src/ggml-cuda/fattn-mma-f16.cuh how 64 query
   rows map onto the 8 waves under a 16x16x16 atom — warps split along KV sharing one query set, or along
   queries. It reprices R2 (Q would live in registers once per BLOCK, not per wave). Write the answer with
   path:line into A10 §3 before anything is built.
2. COMPILE-ONLY LADDER for R1+R2+R3 (8 waves x 16 query rows = 128 rows/block, 256 threads, 64-position KV
   stage, Q in registers, synchronous staging): report .vgpr_count / .vgpr_spill_count /
   group_segment_fixed_size and a __launch_bounds__ occupancy ladder, launching nothing. Kill it at the source
   if 256 VGPR/thread cannot hold 128 accumulator + Q + scores + addressing — a source-level kill is a first-class
   outcome (A3 did one).
3. Only if 2 survives: graduate behind a knob (default inert and byte-identical, witnessed), pre-register the BET
   with its lines in the BET file, then measure at 32,768 lanes in ONE paired window against a pinned pre-change
   sha, with a behavioural canary asserted before any timing.

Hard gates (do not renegotiate them afterwards):
  * ABORT LINE: a prototype must clear 6.5 TFLOP/s at 32k (today 4.47). Below it => record the route dead.
  * NON-REGRESSION: warm 10,637 tok/s at 512 lanes and 9,790 at 4,096 must not fall; route by width if needed,
    and report any trade rather than averaging it away.
  * NUMERICS: a 64-position stage re-associates the softmax, so no bit-identity claim — existing gates stay green,
    no new band, no loosening; a red gate asks whether the band can widen without losing a must-catch.
  * ONE COMMIT for the plumbing: launcher mirror (FATTN_ROWS_PER_BLOCK), the launch bill/canaries, the gate's
    DERIVED case lists at the new row geometry, the default trio (decode_run / decode_step.h / gate.py), and the
    inertness witness.
  * FALSIFIER >= 3% of the deciding row in a paired window; a family win that does not reach the wave is
    DISPLACED, not shipped (§53).

Rules: the main agent owns every measurement (blocking, foreground, zero subagents active, one gpu_lock hold,
record util and clocks); workers are CPU-only and take no timings; reuse instruments, do not build new ones;
commit by topic.
```

**Two things that must not be re-litigated** (both measured): residency is not the gap — llama runs 1.9x our rate
with 8 wf/CU against our 9; and traffic cuts do not convert here — L-c removed 77 % of a 105 GB/pass round-trip and
bought −4.61 % of the family, −2.38 % of the wave (§53). The route's premise is **MAC per LSU byte** (ours 9.6,
llama ≈31), and nothing short of not re-reading Q per round changes that.
