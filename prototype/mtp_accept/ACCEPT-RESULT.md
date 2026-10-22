# ACCEPT result — `a` on engine bits is 0.6641, and the on-side gate is RED for a benign reason (main agent, 2026-10-22)

Reproduced by the main agent, blocking, one `tools/gpu_lock.sh` hold, zero subagents active
(`RUN_GATE=1 OUT=/tmp/mtp_accept_gate bash prototype/mtp_accept/run_accept.sh`, wall 116 s; then the
gate off/on directly under its own hold).

## The number P7 was waiting on

| quantity | value |
|---|---|
| **accept on ENGINE bits** | **615/926 = 0.6641** (proposals adjudicated / made = 926/953) |
| same protocol on **torch** bits (witness) | 627/942 = 0.6656 |
| carried torch-vs-pins figure | 343/468 = 0.7329 |
| kill lines | 0.257 (`delta`=0), 0.4458 (plan A), **0.4568** (plan A + rollback — the real one) |
| **decision room** | **+0.2073** above the binding kill line |

By draft position: 0.640 (n=25) / 0.578 (n=230) / 0.684 (n=462) / 0.718 (n=209) — rising, as the carried
profile predicted. Second schedule `--mtp-sched every` = 0.7248 (1101/1519).

**The witness is what makes the gap interpretable.** Of the −0.0688 distance from the carried 0.7329,
**−0.0673 is protocol** (this harness, run on torch bits, returns 0.6656, and reproduces 343/468
exactly on the carried protocol) and **−0.0015 is the engine's bits**. So A-18's bf16 chaos floor is
*not* eating the accept rate — the mechanism question MTP-IMPL §4 M3 flagged has a number attached to it,
and the answer is that it is nearly nothing. Arithmetic on quoted figures: the speedup is linear in `1+a`,
so **1.4041× × (1.6641/1.7329) = 1.348×**.

## The blocker: flag-on is RED on the launch census, not on numerics

`engine/gate.py --fixture dec8`: **flag off → ENGINE GATE PASS**. **Flag on → rc=1, 15 failures**, every
one the same shape — the observed launch census exceeds §07's plan by a constant +25 per category:

| category | observed (flag on) | §07 plan | delta |
|---|---|---|---|
| glue | 466 | 441 | +25 |
| proj | 523 | 498 | +25 |
| mlp_down / mlp_gateup | 221 / 221 | 216 / 216 | +25 / +25 |

and the gate's own trailer says `[expected-red] none recorded for dec8, so all 15 failure(s) above are new
by construction`. Two things follow, and they are different:

1. **The side-computation property holds.** The identity stage is green independently: `sampled.txt` and
   all 86 STATEFP rows byte-identical flag-off vs flag-on on both `dec8`'s prompt (which contains id
   248046 mid-prompt) and a corpus line. The draft reads and proposes; it does not perturb what the target
   path consumes.
2. **The flag-on launch plan is undeclared, and the gate is right to call that red.** The fix is *not* to
   relax a threshold: it is to make the expected census a function of the declared plan when the flag is
   on — derived from the engine's own launch plan (the draft's per-round launches: `block=1+13
   head=1+1` per the engine's own `MTPDRAFT` echo), never from the observed numbers, and witnessed by
   showing that a *wrong* declared count still reddens it. Until that exists, flag-on is red by
   construction and the engine change must not be committed as though it were green.

## Defect in the handover, third instance of the same class today

`RUN_GATE=1` was read into a variable (`RUN_GATE=${RUN_GATE:-0}`, line 51) and **never used**: no gate
stage exists, and the runner printed its normal stage list as if nothing was requested. The handover
described that stage as "the main agent's step" and the flag as wiring it, so the only un-green admission
looked like it had been run. A flag that is accepted by silence is the same defect as `--evidence` on a
tool with no argument handling (`price_block.py`) and a parser that matches nothing. Fixed here by running
the gate directly; the script still needs the stage actually wired.

## State of the working tree right now

`engine/{build.sh,decode_run.cpp,decode_step.cpp,decode_step.h,kernels.h,kernels_attn.cpp}` modified,
`engine/mtp_draft.{h,cpp}` and `prototype/mtp_accept/` new — **uncommitted on purpose.** Flag off is green
and byte-identical; flag on is red on the census. Next increment: declare the draft's launches in the plan
the gate reads, with the wrong-count witness above.
