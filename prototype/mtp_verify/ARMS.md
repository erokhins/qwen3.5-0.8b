# ARMS — the cost side of the batched 2-row verify body (main agent's paired window)

**STATUS: BLOCKED.** These arms are written but must not be run until
`prototype/mtp_verify/gate.py` is green at both candidate settings. Today it is red (§"What device says"
of `VERIFY-HANDOVER.md`): row 0 of the body is not bit-identical to the step it replaces, and the body
dirties the request's own KV lane. Timing a body whose arithmetic is not yet the funded one produces a
number that cannot be quoted, which is the one outcome worse than no number. Nothing here was timed by
the worker who wrote this file.

## What is being priced

`delta` = the cost of one verify round expressed as a fraction of one decode step, where a round is
*two rows through the body, one batched tied-head pass, one logits download, one checkpoint and one
restore*. `prototype/mtp_delta/D3-RESULT.md` priced the funded shape at **0.1505** of a step and the
same body with the head run twice at **0.4729**, and pre-registered the kill line at **0.379**. Those
two numbers are the interpretation band, quoted here rather than invented:

| measured `delta` | reading |
|---|---|
| ≤ 0.1505 | plan A stands; the head-batch increment is funded and the accept loop is the next increment |
| 0.1505 … 0.379 | report, do not decide: the win is real but smaller than priced, and §1's `rho` row needs the re-derivation at 19.27 MiB per round first |
| > 0.379 | KILL, per D3's own line, and say so with the pair that earned it |

`delta` must be computed with the checkpoint/restore arm subtracted out (the row below), because reject
handling is a separate design item with a separate cost and `VERIFY-HANDOVER.md` shows the checkpoint is
19.2656 MiB per round rather than §1's 18.66 MiB.

## Arms — one lock hold, blocking, zero subagents, best-of-N, warm pages first

| arm | command shape | what it prices |
|---|---|---|
| **A — plain step** | `decode_run --ids-file <fixture> --steps N` (flag off) at 128 and at 4,032 lanes | the denominator: one decode step |
| **B — step + verify round** | same plus `--mtp-verify 1 --verify-candidate sampled` | step + body + batched head + D2H + snapshot + restore |
| **C — checkpoint/restore alone** | same as B, with the body's cost removed by timing `take_snapshot`+`restore_snapshot` in isolation (or B minus a body-only arm if a cheaper split is wanted) | the 19.2656 MiB copy pair, which reject handling will pay for whether or not the body does |
| **D — the killed shape** *(optional, for the record)* | two 1-row bodies and two head passes, i.e. `--mtp-verify 1` with the batched launcher replaced by two `lm_head_full_xbf16` calls | whether D3's 0.4729 reproduces in situ; only worth the window if the C column is surprising |

`delta = (B − A − C) / A`, reported per width, with **both** sides of every pair quoted and the spread
stated — a single number from a short-prompt row is not trustworthy on this box (dispatch-bound
bimodality, `engine/performance-status.md` §6). The headline width is 4,032 lanes, per the standing rule
that decode is measured at 4k; the 128-lane row is quoted beside it, not instead of it.

Rules this window has to obey, all of them load-bearing:
blocking foreground runs, never backgrounded; `tools/gpu_lock.sh` acquired once for the whole set and
`util` recorded (then discounted — identical clean runs read 24–91 %); warm before timing, since the
first pass over fresh pages runs ~35 GB/s; no `ms` figure from a worker run may enter a timing doc, so
this table is filled by the main agent alone.

## Behavioural canary — assert before timing anything

`AGENTS.md`'s "a paired A/B against a stale binary reads as a clean null" row applies verbatim: the
`new` side must be shown to carry the change *inside the timing window*. The canary here is the
body's own census, read from the same stdout as the milliseconds:

* every `VERIFY2` line reads `head_passes=1` and `d2h=1` — two of either is the KILLED shape wearing
  the batched shape's clothes;
* `LAUNCHSTAT family=lm_head` grows by exactly **one** call per generation step versus arm A, not two;
* `VERIFY2 row0_eq_step=1` on every round. **This is the canary that is red today** (it read 0 at the
  first anchor), and until it is green there is nothing to time;
* `VERIFY2DONE restore_witness=identical` and `gemm=hand gdn_scan=seq` — if the forced route did not
  take, the arm is pricing a different body than the one §1 priced;
* and the rows-served/bytes-demanded statement the brief asks for: arm B serves two query rows per body
  where arm A serves one, so `demanded bytes per row served` must fall. Derive it from the engine's own
  lines (`LAUNCHSTAT rows_total`, `logits_bytes=`, the KV lane figure) rather than asserting it.

If any canary does not read as expected on both sides, the script must abort before printing a
millisecond figure, in the style of `prototype/attn_m2/burst_m16.sh`'s `NO-TIMING: admission is red`.

## Table to fill

| width | A: step ms | B: step+round ms | C: ckpt+restore ms | delta | band reading |
|---|---|---|---|---|---|
| 128 lanes | | | | | |
| 4,032 lanes | | | | | |

Quote the pair and the spread per row; state the fixture, the `util` at acquire, and that the admission
gate was green in the same window.
