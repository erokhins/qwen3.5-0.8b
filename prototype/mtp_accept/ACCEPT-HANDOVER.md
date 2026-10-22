# ACCEPT handover — P7 third funded increment: `a` on the engine's own bits

Written as the **first** artifact of this increment, before any long analysis, so that if this
worker's window is compacted or expires, the plan and the carried numbers are on disk rather than in a
summarised context.

Status legend: `[ ]` not started, `[~]` in progress, `[x]` done and witnessed.

## 0. What this increment is, and what it is not

P7 = MTP speculative drafting, k=1. Everything except `a` is now measured:

| term | value | source |
|---|---|---|
| `R(M=2)`, batched tied head | **1.0019** — head is free | `prototype/mtp_headbatch/HEADBATCH-RESULT.md` |
| draft block, k=1 | **0.4371 ms**, NARROW band, depth linear | `prototype/mtp_draft/DRAFT-RESULT.md` |
| projection transfer | **1.4041×** projected headline | same |
| `delta` (2-row verify marginal) | 0.1358 of a step at plan A | `HEADBATCH-RESULT.md` |
| `rho` (rollback) | 0.12 ms/round = 0.0110 of a step | `MTP-IMPL.md` §3 |
| `a`, carried | **0.7329** on torch-CPU bits | `prototype/mtp_gatezero/draft_survival.json` |
| `a_needed`, pre-existing kill line | **0.257** (gate zero, the `delta = 0` row) | `prototype/mtp_gatezero/MTP-GATEZERO.md` |
| `a_needed`, plan A + rollback | 0.4568 | `MTP-IMPL.md` §0/§4 M3 |

This increment measures `a` **on the engine's own bits**. It does **NOT** build the verify body, the
accept loop for real, the snapshot/restore path, or any 2-row body. Proposal only: the draft exists as a
read-only side computation whose output is a proposal id and nothing else.

The three funded deliverables:

1. `[x]` **D1** — the draft path in `engine/`, behind `QWEN_MTP_DRAFT=0` (default off) + a `decode_run`
   flag, as a strict **side computation**. Admission: `sampled.txt` and every STATEFP fingerprint row
   **byte-identical** with the flag on and off; launch-plan line states any added launches.
2. `[x]` **D2** — `a` on engine bits over the 49 lines of `speccheck/fixtures/corpus.txt`, ≥ 24 generated
   tokens each, stop id **248046** in the generation phase only, with the index alignment printed and
   asserted, per-depth reporting, named denominator, and the **witness** (same harness on the torch-side
   pins must return ≈ 0.7329).
3. `[x]` **D3** — `run_accept.sh`: one `gpu_lock.sh` hold, blocking, admission-gated first, corpus runs
   then tally; expected wall stated; split if > ~10 min.

## 1. Hard rules this increment carries from `AGENTS.md`

- **Index alignment.** §10.5 pins `argmax(decode{s}.logits_last) == greedy_continuation[s+1]`, so a
  proposal at step *s* pairs with the engine's own greedy token at **s+1**. A free run aligned on `[0:]`
  read a false 0/128 in this repo's history. The alignment is printed on the result line, and the join is
  the **position** join carried over from `prototype/mtp_gatezero/draft_survival.py`
  (`accept ⟺ draft == pred_at[p-1]`), not a re-derived token-index loop.
- **Stop token.** id 248046, generation phase only; a prompt fixture legitimately contains it, so the
  stop check must never fire while feeding a prompt (§07.5). A **proposal** of 248046 must not stop a
  round; the stop test runs on accepted tokens only.
- **Never print chat-template special tokens literally** anywhere — reports, logs, gate output, commit
  messages. Refer to them by id.
- **No timings.** Correctness-only device checks ≤ ~60 s each, one at a time, each reporting the exact
  command, wall time, `tools/gpu_lock.sh util`, and "no timing taken". No number from this increment
  enters `engine/performance-status.md`.
- **Gate ownership.** No edit to gate thresholds, floors, `data/refs/` pins, `engine/performance-status.md`,
  `engine/generation-limits.md`, `progress/`, `README.md`. No commit. `export LC_ALL=C`.
- **Do not re-make a result that is already on disk.** `prototype/mtp_draft` and
  `prototype/mtp_headbatch` are measured and admitted; cite them.
- **Self-locking drivers.** `prototype/glue/run.py`, `prototype/proj_gemv/run.py`,
  `prototype/gdn_gated_norm/run.py` acquire the lock themselves — `grep -c gpu_lock` before wrapping.

## 2. Protocol for D2, stated before the number exists

- Population: all 49 corpus lines (verified: `wc -l` = 49), ≥ 24 generated tokens per line.
- Denominator, named on the result line: **proposals adjudicated / proposals made**; every skipped
  proposal counted and explained (a proposal at the last step of a line has no `s+1` to adjudicate
  against, and is *skipped*, not counted as a miss).
- Accept reported **overall and by depth** into the generation, the same buckets `draft_survival.py`
  used (0-7 / 8-15 / 16-31 / 32-63 / 64+), against the carried profile that *rises* 0.588 → 0.808.
- Verdict against **both** reference points: the pre-existing kill line `a_needed = 0.257` and the
  carried `a = 0.7329`. If engine-bit `a` ≪ 0.7329, the interpretation is pre-registered: that is
  A-18's bf16 chaos floor arriving in the accept path — a result **about the model**, not about the
  harness — but only after the witness below has run and come back green.
- **Witness**: the identical harness driven from the torch-side pins (where `pred_at` comes from the
  reference, not the engine) must reproduce ≈ 0.7329. Without a green witness, a low engine number is
  unadjudicable and gets reported as such, not as a kill.

## 3. Stop-and-report condition

If the byte-identity admission (flag on vs off) cannot be achieved, **stop and report**. That means the
draft cannot be a side computation and P7's integration shape changes — a decision for the main agent,
not something to design around.

## 4. Working state (append-only checkpoint)

- `[x]` carried files read: `HEADBATCH-RESULT.md`, `MTP-IMPL.md` §1/§3, `DRAFT-RESULT.md`,
  `draft_survival.py` + `draft_survival.json`, `AGENTS.md` traps table, `speccheck/mtp.py`.
- `[x]` engine read: `decode_step.h/.cpp` (body/finish split, `sc.norm` row 0 = §09.1's `h`),
  `decode_run.cpp` (flag/env surface, `sampled.txt`, STATEFP, session protocol), `prototype/mtp_draft`
  (`td_modes.cpp` chain, `td_attn_tu.cpp` transcription, launcher TUs).

### D1 — engine side: WRITTEN, builds clean (23.7 s full build)

| file | what |
|---|---|
| `engine/mtp_draft.h` / `.cpp` | NEW. The block as a side computation: `resolve()` (15 tensors, rank+shape per
  tensor, thrown names), one arena, `draft_round()` = 1 D2D + 13 launches + `lm_head_full_xbf16` + 1
  D2H = 16 submissions/round, `RoundTracker` (the two schedules), `truncate_lane`, `refusal()` |
| `engine/kernels.h` | NEW decls: `kernels::MtpAttnRow`, `kernels::mtp_attn_row`, `mtp_attn_max_lanes()` |
| `engine/kernels_attn.cpp` | `mtp_attn_row_kernel` — the transcription of `attn_core`'s fused one-pass
  `full` path with the dense-append-slot change, ported from `prototype/mtp_draft/td_attn_tu.cpp`, minus
  the rig's witness dumps (pure stores). Lives here because this is the ONE TU allowed to include
  `attn_decode_kernels.h`: its `__global__`s are strong symbols, so a second TU including it is a
  duplicate-symbol link error (verified with `nm`: `T __device_stub__attn_core`). |
| `engine/decode_step.h` / `.cpp` | `argmax_greedy()` factored out of `finish_token` and reused by the
  draft — §03.4's rule gets one spelling, so the proposal cannot be held to a second standard |
| `engine/decode_run.cpp` | `QWEN_MTP_DRAFT=0|1` env + `--mtp-draft 0|1` (flag outranks env),
  `--mtp-sched loop|every`, `--mtp-cap N`; refusal with `--time-stages`/`--time-waves` and for
  `--mtp-sched` while off; `MtpSide` + `mtp_after_step`/`mtp_close_run`/`mtp_census_line`; hooked into
  BOTH the run-mode generation loop and the session `gen` loop; `reset` resets the draft cache |
| `engine/build.sh` | `engine/mtp_draft.cpp` added to the `decode_run` link |

Default-off is inert by construction: `mtp` is a null `unique_ptr`, nothing is allocated, no submission
is made and no line prints when the flag is off.

**Printed lines** (only when the flag is on): `MTPDRAFT on=1 sched= mode= block=1+13 head=1+1
per_round=16 rounds= adjudicated= accepted= refused= unadjudicated= lanes_max= cap= tensors=15
block_bytes=`, `DRAFT round= draft_pos= adjudicated_by_pred_pos= y_next= proposal= top_logit= lanes=`,
`MTPACCEPT settled_pos= proposal= target_pred= accept=`, `DRAFT-SKIP reason=`,
`DRAFT-UNADJUDICATED adjudicated_by_pred_pos= proposal= reason=`.

### The index alignment, as implemented (asserted, not assumed)

After a generation step whose row predicted position `p` (run mode: `pos_before = p`; session mode:
`prefix_len - 1` after the step — the same number its own `STEP`/`GEN` line prints):

- the round drafts position **p+2**, behind the token just emitted, with `h` = `sc.norm` row 0 (position
  `p`, post-final-norm — A-12 falsified the pre-norm reading);
- that proposal is settled by the target's own prediction **at position p+1**, i.e. by the NEXT step;
- every `DRAFT` line prints `adjudicated_by_pred_pos`, and the harness joins on that printed number
  against `STEP kind=decode pos_before=` / `GEN <id> <pos>`. `RoundTracker::after_step` refuses a
  pending proposal whose recorded position is not the step it was handed, and the run aborts rather
  than printing a rate.

### Two round schedules, both measured rather than one assumed

`loop` (default) reproduces `speccheck/mtp.py`'s `greedy_mtp(k=1)` round geometry on a plain greedy
stream — round at q drafts q+1, next round at q+2 after an accept and q+1 after a reject, the rejected
round's draft-cache lane truncated (`greedy_mtp`'s `m.restore(msnap)`). A plain greedy run emits exactly
the stream that loop emits, so this is the engine counterpart of `verify="steps"`, which is what the
carried 0.7329 was measured with. `every` drafts after every step, so its cache holds every position —
more context than the loop gives a draft. The gap between the two is that difference measured.

**Interpretation call, flagged, not silent**: the round schedule is not the verify body and not the
accept loop (no candidate row is fed to the target, nothing is snapshotted or rolled back, the target
keeps running plain greedy and no byte it consumes moves). It exists because a `a` measured under a
different draft-cache geometry than the carried number's is confounded, and the confound would have
been invisible. Both schedules are printed so nothing rests on the choice.

- `[x]` D1 device admission: flag off vs on, `sampled.txt` + STATEFP byte-identical — 86 STATEFP rows
  and `sampled.txt` identical on `dec8`'s own 17-id prompt (which contains id 248046 mid-prompt) and on
  a corpus line; see §5.
- `[x]` D2 harness `prototype/mtp_accept/run_accept.py` + `witness_torch.py` — see §5.
- `[x]` D3 `run_accept.sh` — one hold, stages `build,gate,identity,corpus,witness`, ran end to end in
  109 s of harness wall.
- `[ ]` `engine/gate.py --fixture dec8` with the flag off and on: **main agent's step**. It is a device
  run of unknown-but-likely-multi-minute wall, which is over a worker's ~60 s budget, so
  `RUN_GATE=1 prototype/mtp_accept/run_accept.sh` is handed over rather than run here. The script
  refuses to call the on-side green unless an `MTPDRAFT on=1` line actually came back, so a flag that
  failed to reach the engine cannot masquerade as a pass.

### Carried fixtures found, worth using

- `data/refs/corpus48/manifest.json` — the 49 corpus prompts **with their token ids** and the HF
  reference's 24-token continuations (`steps: 24, n: 49`, reference run cost 40.6 s of torch CPU).
  So the corpus population is pre-tokenised and the torch side of the witness is minutes, not an hour.
  Reference continuations legitimately contain 248068/248069 — ids only, never rendered.

## 5. RESULT — `a` on the engine's own bits: **0.6641**, and the gap to 0.7329 is protocol, not bits

Population: all 49 lines of `speccheck/fixtures/corpus.txt`, 32 generated tokens each (≥ 24 asked for),
one resident `decode_run --session --mtp-draft 1 --mtp-sched loop`, schedule `loop`.

```
proposals made        953
proposals adjudicated 926
proposals skipped     27   (all: "no later generation step in this turn settles it"; a skip is never a miss)
accept OVERALL        615/926 = 0.6641   [denominator: proposals adjudicated / proposals made = 926/953]
  of which settled by the turn's unprinted final sample: 22/30; excluding those: 593/896 = 0.6618
accept by draft position   0-7:0.640(n=25)  8-15:0.578(n=230)  16-31:0.684(n=462)  32-63:0.718(n=209)
accept by round in line    0-7:0.606(n=343) 8-15:0.691(n=392)  16-31:0.712(n=191)
engine-vs-reference   976/1176 tokens agree with the pinned HF continuations
engine census         rounds=953 adjudicated=926 accepted=615 refused=0 unadjudicated=27 lanes_max=16
census cross-check    harness 953/926/615 == census 953+0/926/615  (independent counters, exact)
```

**Verdict against the quoted bands.** 0.6641 is above the pre-existing kill line 0.257 (gate zero's
`delta = 0` row), above plan A's 0.4458, and above plan A + rollback's **0.4568** — the last is
`MTP-IMPL.md` §0's stated real kill line, and the decision room is **+0.2073**. It sits inside that
document's predicted survival band [0.457, 0.733]. P7 stays funded on the engine's bits.

**Arithmetic on the quoted projection, no new timing.** `speedup = (1+a)/((1+delta)(1+c)+rho)` is linear
in `1+a`, so holding every quoted term fixed: 1.4041× × (1.6641/1.7329) = **1.348×**. That is the same
denominator the `1.4041×` was priced with, not a new measurement.

### The witness, which is the part that makes the number mean something

0.6641 against a carried 0.7329 looks like a −0.069 loss and invites the A-18 story ("the chaos floor
arrived in the accept path"). It is almost entirely **not** that. `witness_torch.py` runs this protocol —
this population, 32 steps, this `RoundTracker` geometry line for line, this settlement rule — on torch
bits:

| measurement | bits | accept |
|---|---|---|
| carried `draft_survival.py` (20 prompts × 40 steps, `greedy_mtp` verify=steps, no stop rule) | torch | **0.7329** (343/468) |
| same, re-run inside `witness_torch.py --protocol carried` | torch | **0.7329** (343/468) — reproduced to the last digit |
| **this protocol** (49 corpus lines × 32 steps, `loop`, generation-phase stop) | torch | **0.6656** (627/942) |
| **this protocol** | **engine** | **0.6641** (615/926) |

So of the −0.069 gap between the carried number and the engine's, **−0.0673 is the protocol** (population,
step budget, round schedule, stop rule) and **−0.0015 is the engine's bits**. The mechanism question
`MTP-IMPL.md` §4 M3 flagged — "the gap between the torch number and the engine's bits is a mechanism
question, not a rounding error" — has an answer: on identical protocol the engine's bf16 bits cost 0.15
points of accept rate. The carried figure was optimistic as a *prediction of this protocol*, which is the
only thing the increment needed to find out.

Context row, not part of `a`: the engine's own greedy tokens agree with the pinned HF continuations on
976/1176 of the compared tokens (83.0 %) at 32 tokens deep, which is A-18's divergence doing what it
does; the accept rate is computed entirely inside the engine's own stream, so that divergence does not
enter the numerator.

### One real finding on the engine side: `LASTSAMPLE`

The session's `gen` loop prints `GEN` **before** it steps, so on the budget path the last step's sample is
a real target prediction that is never printed as a token (the driver closes the turn instead). The draft
had already been settling proposals against it, so the engine's census adjudicated one proposal per line
that no printed line could account for — and the harness's independent join, correctly refusing to guess,
aborted the run. Fixed by printing that sample as `LASTSAMPLE pos= token=` — **draft-only output**, so a
flag-off session emits byte-for-byte what it always did, and `chat.py` treats an unrecognised line as a
diagnostic (its `TERMINAL` filter), so the driver protocol did not change. Both rates are printed: with
those settlements 0.6641, excluding them 0.6618, because "settled by a token the user was never shown" is
a claim a reader may reasonably want excluded rather than averaged in silently.

### D1 admission, as run

`sampled.txt` byte-identical and **86 STATEFP rows** byte-identical with the flag off vs on, on both:
`dec8`'s own 17-id prompt — which contains id 248046 in the middle of the prompt, so it also witnesses
that §07.5's prompt-phase rule is undisturbed — and an ordinary corpus line. The census line is asserted
against the declared launch plan (`block=1+13 head=1+1 per_round=16`), and the flag-on runs print it.
`refused=0` and `lanes_max=16` at this depth: no draft-cache ceiling was reached, so no proposal was lost
to `DRAFT-SKIP` in this pass (`--mtp-cap` defaults to 512 lanes).

Then the stronger form, which off-vs-on inside one binary cannot give: today's **flag-off** output is
byte-identical to `engine/build/decode_run.premtp`, a binary built before any of this landed —
`sampled.txt`, all 86 `STATEFP` rows, and every `STEP` line, on both fixtures. The pair carries a
behavioural canary (`AGENTS.md`: a paired A/B against a stale binary reads as a clean null): that binary
must *reject* `--mtp-draft`, and it does, so the two sides are demonstrably different builds rather than
one build compared with itself. `run_accept.sh` runs this whenever a `.premtp` binary exists and says so
when it does not.

### Commands, walls, and the honesty line

All device work ran under `tools/gpu_lock.sh` in a worker window, util 81-93 % at acquire, blocking,
one run at a time. **No timing is taken from any of it** and nothing here may enter
`engine/performance-status.md`. Walls below are the harness's own minutes.

| command | wall | what it is |
|---|---|---|
| `OUT=/tmp/mtp_accept_run bash prototype/mtp_accept/run_accept.sh` | 109 s total (build 24, identity 4, corpus 19, witness 62) | the whole protocol, one lock hold, witness outside it |
| `.venv/bin/python3 prototype/mtp_accept/run_accept.py --steps 32 --sched loop --json …` | 19 s | the 49-line pass (util 93 % at acquire, 81 % after) |
| `.venv/bin/python3 prototype/mtp_accept/witness_torch.py --protocol same --sched loop` | 62 s | torch-side witness, CPU only |
| `.venv/bin/python3 prototype/mtp_accept/witness_torch.py --protocol carried --limit 20 --steps 40` | 45 s | reproduces 343/468 = 0.7329 |
| `… run_accept.py --steps 32 --sched every --json …` | 21 s | the second schedule, 1 519 adjudications (util 92 % at acquire) |
| `OUT=/tmp/mtp_accept_run STAGES=identity prototype/mtp_accept/run_accept.sh` | 6 s | admission alone, both fixtures, pre-MTP form included (util 91 % at acquire) |

Reproduce: `RUN_GATE=1 prototype/mtp_accept/run_accept.sh` (adds `engine/gate.py --fixture dec8` off and
on — the main agent's step, and the only part of this not yet green).

### The second schedule, measured rather than assumed: `--mtp-sched every` = **0.7248**

`every` drafts after **every** generation step, so the block's cache holds every position instead of
skipping the one a `loop` accept advances past. Same 49 lines × 32 steps, same join, same cross-checks:

```
proposals made 1568   adjudicated 1519   skipped 49   accept 1101/1519 = 0.7248
  excluding the unprinted-final-sample settlements: 1063/1470 = 0.7231
by draft position  0-7:0.581(31)  8-15:0.627(351)  16-31:0.746(784)  32-63:0.788(353)
census             rounds=1568 adjudicated=1519 accepted=1101 refused=0 unadjudicated=49 lanes_max=32
```

The `loop` number (0.6641) is the one comparable with the carried figure and the one that funds P7, because
the loop is the geometry a real k=1 verify visits; `every` is the upper bound that a draft with a hole-free
cache gets, and the +0.061 between them is that hole measured rather than argued about. Neither schedule
aborted on alignment at any of 2,494 rounds across the two passes, and neither hit the draft-cache ceiling
(`refused=0`; `lanes_max` 16 and 32 against `--mtp-cap 512`).
