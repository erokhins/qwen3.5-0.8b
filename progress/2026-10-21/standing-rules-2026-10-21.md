# Standing rules for this session and beyond (user, 2026-10-21) — READ THIS FIRST AFTER A COMPACTION

Written on explicit instruction: *"store these rules to the progress folder into md file — so after
context compacter you won't lose info about these rules."* If your context was just compacted, this file
plus `engine/generation-limits.md` and `progress/2026-10-21/route-S1-segments-2026-10-21.md` are the three
documents that tell you where the work stands. Re-read them before deciding anything is finished.

## The three rules, as given

1. **Do not worry about "context almost full".** There is a compactor and it compacts correctly. Context
   pressure is not a reason to shorten a measurement, skip an evaluation, thin a handover, or stop a queue.
   It is not a reason to leave work for a future session that will not know what it was.
2. **Do not stop until every optimization idea for generation has been measured and evaluated.** If the
   idea list looks empty, that is a signal to analyze harder, not to stop — the last four sessions each
   found a new avenue after a previous conclusion looked final (occupancy from a compiler artifact, launch
   price by deletion, the copy fold, the allowance-instrument gap). The live inventory and its method are
   in `generation-idea-queue-2026-10-21.md` beside this file; keep it current, and add to it from the
   sources listed there rather than from memory.
3. **Keep this file and the queue file updated** as decisions land, so a compacted session inherits state
   rather than reconstructing it.

## Rules already in force that these do NOT override

These come from `AGENTS.md` and the user's own earlier instructions; the new rules raise persistence, not
authority:

- **GPU is serialized and owned by the main agent.** No device timing while a subagent is alive (one
  subagent at a time, and "done" means stopped, not idle). Measurements run blocking, never backgrounded,
  one `tools/gpu_lock.sh` hold acquired and released inside a single shell block.
- **Workers never take timing.** Correctness-only device gates ≤ ~60 s, PASS/FAIL, command + wall + `util`
  + "no timing taken". No worker number ever enters a timing doc.
- **A check that cannot fail is a defect.** Must-catch sets derived from geometry; parsers abort on empty
  or odd parses; a null needs a behavioural canary inside the same window; pre-register bands before
  timing and never widen a band after a refusal.
- **Never print chat-template special tokens literally** — ids 248046 (eos/end-of-turn), 248068/248069
  (think boundaries). This has cut off two agents on this box.
- **Line-cited files stay line-count-neutral**: `engine/performance-status.md` (307), `README.md` (248),
  `prototype/attn_decode/attn_decode_kernels.h` (821). `wc -l` before and after; `citation_rot_check.py`
  cannot see drift introduced by the commit it checks.
- **Main agent steers, commits and measures; workers implement.** One topical commit per change, header
  ≤ 72 chars then a body explaining what and why.
- **Quantization (int8/int4) and any band a pinned gate depends on for discriminating power remain user
  decisions.** 16/32-bit arithmetic changes are allowed if the gates stay green.

## State at the time of writing (2026-10-21, late)

- **HEAD**: `775e376` (default flip: both folds ship on; step is 224 launches, was 340). Ledger headline
  **91.4 tok/s at 4,032 lanes** (10.9357 ms/token, cast-fold paired window), 102.2 at 128. Prefill 11,070
  at 4,096. The copy fold's further −0.036/−0.065 ms is a recorded paired delta, not re-based into the
  absolute row (different window; base read 11.32 vs 11.24).
- **Shipped this week**: D3 attention occupancy port (−46 % on `attn`, −9.9 % step), cast fold (97 launches
  deleted, −0.3008 ms, 3.10 µs/launch measured by deletion), copy fold (19 launches, −0.036/−0.065 ms),
  both defaults on.
- **Closed as routes**: cross-layer MLP merge (residual chain forbids it), "fewer, fatter launches" (the
  fat shape is that same illegal merge; measured plateau-then-cliff 7.20 → 311.37 ms at equal bytes),
  split-K on one-row `down_proj` (loses at 2/4/7/8/14-way), glue norm-epilogue fold (394 MB/token cost to
  buy 0.152 ms — 10.9× the prize), per-segment "drain" prizes (the stage clock's own syncs, 5.3 µs each,
  not paid in production), `lm_head` (246 GB/s = at the wall), D2 as scoped.
- **In flight**: worker `c16bb38c` pricing the missing `>MAXSC` route allowances and gating `--split 18`
  (~0.36 ms) against them. Its handover will be `prototype/maxsc_allowance/MAXSC-HANDOVER.md`. **The
  ruling is the parent's**, and the evidence may say the route does not fit.
- **User decisions still unspent**: D5's pre-registered lines (its own KILL already exceeded); the `>MAXSC`
  instrument ruling once W2's evidence lands; any band move the above requires.
