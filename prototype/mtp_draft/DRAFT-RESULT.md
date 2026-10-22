# DRAFT result — the draft block measured against its pre-registered bands (main agent, 2026-10-22)

Taken by the main agent, blocking, zero subagents active, one `tools/gpu_lock.sh` hold, two interleaved
passes (`bash prototype/mtp_draft/run_locked.sh`, wall ~11 s), adjudicated by
`price_block.py --evidence /tmp/mtp_draft/evidence/log.txt`. Bands were fixed in `DRAFT-PRICE.md` before
any millisecond was printed.

| arm | best ms | passes | admission | vs bands |
|---|---|---|---|---|
| **block-k1** | **0.4371** | 0.4371 / 0.4776 (9 % spread) | numeric | **NARROW** (WIN ≤ 0.2986, NARROW ≤ 0.5973) |
| block-k2 | 0.7666 | 0.7666 / 0.8624 | numeric | ratio **1.754** → linear |
| block-k4 | 1.5039 | 1.5039 / 1.6222 | structural only | ratio 3.441; sizes the curve, **not a headline** |

Census prediction was 0.2389 ms block-only, so the block runs **1.83× its byte floor** — which is the same
place the engine's own GEMV families live (proj 75 %, mlp 70 % of floor), not a defect in the block. Depth
is linear, so a deeper draft is a serial chain and not a re-read of the same weights: the `KILL(also)`
superlinearity rule (> 2.35) does **not** fire. Launch floor is *overlapped*, not exposed: the `empty` arm
reads **1.82 µs per launch** (note the unit — the runner prints ms for block arms and µs/launch for the
empty arm on visually identical lines), so 14 launches ≈ 0.026 ms, already inside the block number.

**Projected headline at the measured block cost: 1.4041×** (census model said 1.4289×; the pre-existing
promised figure, priced before the head was measured, was 1.247×). Break-even is far away: the block may
cost **1.918 ms** before the headline falls below 1.247×, and 5.093 ms before the round stops being a win
at all at `a = 0.7329`.

Per the NARROW band's own words: **fund a staging pass, do not fund deeper k.** That is the instruction
this measurement issues to the next increment, and it is a pre-registered instruction, not a post-hoc one.

## Three defects found while taking this, all of the "a check that cannot fail" family

1. **`price_block.py` had no argument handling at all**, yet the handover's command passed it
   `--evidence DIR`. The flag was accepted by silence and ignored, so the tool printed bands it had never
   compared to a number — the empty-parse failure class exactly. It now requires the runner's log, refuses
   a missing/empty/partial parse, and prints the verdict; on the first re-run it correctly refused
   (`parsed arms [] lack block-k1/block-k2`) instead of inventing one.
2. **`run_locked.sh` wrote no evidence**, so the provenance path named in the handover could not exist.
   It now tees its per-arm lines to `$EV/log.txt` (default `/tmp/mtp_draft/evidence/`).
3. My own first regex matched `pass 1` where the log prints `pass1` — which is only visible because
   defect 1's replacement refuses an empty parse rather than adjudicating nothing. Worth recording as the
   paired lesson: a refusal path is what makes a parser bug *visible* instead of silent.

Also corrected before use: the `empty` arm is **µs/launch**, not ms. Read as ms it looks like an empty
chain costing 1.82 ms — 40× the launch floor it exists to measure — and would have been quoted as a
finding about launch cost.
