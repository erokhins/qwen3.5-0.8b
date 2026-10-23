# ROUTE-FLAG — the verify/accept body can now run on the amortising route, behind a named flag

Built 2026-10-23 by a worker, on the strength of `ROW-SCALE.md`. **No timing is in this file**: nothing
here was measured in a clean window, and no figure from a worker run may enter a timing doc
(`AGENTS.md`). The numbers below are *numerics*, which is the other kind of number.

## 0. WORKER CHECKPOINT (appended mid-run, before compaction — the rest of the file is the detail)

**The flag.** `--mtp-route hand|amortised` on `engine/decode_run.cpp`; default `hand`; parse-or-throw; the
engine enum is `engine::mtpverify::Route { kHand, kAmortised }` in `engine/mtp_verify.h`, shared with
`accept_round` the way `kRows` is shared. Echoed as `mtp_route=<flag> gemm=<ran> gdn_scan=<ran>` on every
`VERIFY2` line, on `VERIFY2DONE`, and on `MTPSTAT` — where `gemm=`/`gdn_scan=` are read off the scratch the
body was configured from (`mtpverify::Result::gemm_ran`, `mtpaccept::Round::gemm_ran`), not restated from
the flag.

**Files changed (tree left dirty, nothing staged, nothing committed).**
`engine/mtp_verify.h`, `engine/mtp_verify.cpp`, `engine/mtp_accept.h`, `engine/mtp_accept.cpp`,
`engine/decode_run.cpp` (modified); `prototype/mtp_accept/verify_body_scale.sh` (modified: `EXTRA=`
passthrough onto the body arms + a route canary that was made to fail on purpose); new:
`prototype/mtp_accept/ROUTE-FLAG.md` (this file), `prototype/mtp_verify/route_diff.sh`,
`prototype/mtp_verify/route_diff_summary.py`, `tools/citation_repoint.py` (the coordinate-diff guard, §5),
and `prototype/mtp_accept/evidence/repoint-decode_run-2026-10-23.txt` (its output for this edit).

**What the switch moves, in one line.** At the engine's default flags it moves **only the GEMM axis**
(`gemm=hand` → `gemm=lt`, i.e. `proj`/`mlp_gateup`/`mlp_down` go to hipBLASLt and the widenings those shapes
never read are deleted). `gdn_scan` stays `seq` because `seq` IS the engine default, so `gdn_rec` moves only
if `--gdn-scan collapse` is also named — and that is refused whenever a mid-round checkpoint is armed
(`--mtp-accept`, `--mtp-midsplit`), because the checkpoint is a one-token-step recurrence's end state.

**Flag-off byte identity: PASS.** Command `.venv/bin/python3 engine/gate.py --fixture dec8` (after
`engine/build.sh`, lock held as tag `worker-mtp-route`, released after), **9.6 s wall**, `util` 93 % at
acquire, tail line `ENGINE GATE PASS`; re-run on the final build as tag `worker-mtp-final`, **9.1 s wall**,
`util` 0 % at acquire, same `ENGINE GATE PASS`. Also PASS in the new instrument: `--mtp-route hand` named
explicitly is byte-identical to not naming it — 17 dumps at dec8, 9 at 1024 lanes, 0 differed; and
`sampled.txt` is identical across plain / hand / handx / amortised at both widths. **No timing taken** (the
gate's own ms/step by-product is not quoted here).

**Flag-on row-0 numerics, REPORTED not gated** (row0-vs-step, `rowdiff.py`, dtype read off file size):
dec8 8 anchors and 1024 lanes 4 anchors — argmax agrees on **12/12** rounds (`row0_eq_step=1`) and on all
24 row pairs; 1−cos worst **9.84e-05**, meanrel worst **1.45e-02**, and the hand-vs-amortised gap is the
same order as either arm's own body-vs-step gap. Nothing was bit-identical and nothing claims to be; no band
or gate threshold was touched. Full table + reading in §4.

**Device runs this worker made (all correctness-only, blocking, one at a time, own lock tags, all under
~60 s each, NO TIMING TAKEN or printed):** `engine/gate.py --fixture dec8` (9.6 s, and 9.1 s re-run on the
final build); the accept-arm and
midsplit smoke set (`MTPSTAT route=hand gemm=hand gdn_scan=seq` vs `MTPSTAT route=amortised gemm=lt
gdn_scan=seq`, same 4 rounds / accept_rate 0.75 / `emitted.txt` byte-identical; midsplit still
`split_eq_batch=1 logits_ndiff=0`; a side-pass run with `--gdn-scan collapse` proving the scan axis moves
when no checkpoint is armed); `prototype/mtp_verify/route_diff.sh mtp-route dec8 1024` (38 s wall,
dominated by the CPU-side rowdiff). Any DECWALL `ms` those logs contain is contaminated by this session and
**must not** be read as a measurement.

**What the parent must run (the parent owns every measurement):**

```bash
prototype/mtp_accept/verify_body_scale.sh 128 1024 4032                 # today's forced route
EXTRA='--mtp-route amortised' prototype/mtp_accept/verify_body_scale.sh 128 1024 4032
```

Same lock hold, medians of ≥3, and compare the per-width `round/step` medians of the two prints against
`ROW-SCALE.md` §4's pre-registered line (**≤ 1.55 steps at 4,000 lanes**). The script reads the route out of
`EXTRA`, asserts it on each body arm's own echo, and aborts if a verify arm's `VERIFY2` line or an accept
arm's `MTPSTAT` does not say that route back (`AMORTISED_GEMM=<name>` overrides the `gemm=` expected on the
amortised arm, which is how the canary was made to fail on purpose — §5). The base arm prints a note when it
strips `--mtp-route`: a one-row step has no body to route and the engine refuses the flag with no
verify/accept running.

## 1. What was built

`--mtp-route hand|amortised` on `engine/decode_run.cpp`, default **`hand`**.

| piece | where |
|---|---|
| `enum class Route { kHand, kAmortised }` + `parse_route` / `route_name` | `engine/mtp_verify.h`, `engine/mtp_verify.cpp` |
| the forcing, now conditional | `verify_body(… Route route …)` in `engine/mtp_verify.cpp`; `accept_round(…, Route route)` in `engine/mtp_accept.cpp` |
| the flag, its parse, its refusals, its echo | `engine/decode_run.cpp` |
| the reported-numerics instrument | `prototype/mtp_verify/route_diff.sh` (+ `route_diff_summary.py`), reusing `prototype/mtp_verify/rowdiff.py` |
| the pricing passthrough | `prototype/mtp_accept/verify_body_scale.sh`, `EXTRA=` |

`kHand` applies the old forcing (`sc.wave_gemm = kHand`, `sc.gdn_scan = kSeq`); `kAmortised` writes
nothing, so the body runs whatever the driver resolved — the route a wide body takes everywhere else.
Both restore the scratch on every exit path, throw included, exactly as before.

`--mtp-route hand` is **byte-identical to not naming the flag**: 17 dec8 dumps and 9 1k-lane dumps
(`logitsN.bin`, `verify2_N.bin`, `sampled.txt`) compared pair by pair, 0 differed
(`route_diff.sh`'s `hand` vs `handx` section). That is the claim that makes every flag-off number already
on the books still mean what it says.

### The echo, because a flag that did not take would price one route twice

`VERIFY2` ends with `mtp_route=<flag> gemm=<ran> gdn_scan=<ran>`; `VERIFY2DONE` and `MTPSTAT` carry the
same triple. The `gemm=`/`gdn_scan=` values are **read off the scratch inside the call**
(`mtpverify::Result::gemm_ran` / `scan_ran`, `mtpaccept::Round::gemm_ran` / `scan_ran`), not restated from
the flag, so the echo cannot be satisfied by a binary that parsed the flag and ignored it — `AGENTS.md`'s
"a paired A/B against a stale binary reads as a clean null", and `VERIFY2DONE` used to print a hard-coded
`gemm=hand gdn_scan=seq`, which would have kept asserting the forcing after the forcing became a choice.

What the two arms actually print, at every anchor, on this box:

```
hand    … mtp_route=hand      gemm=hand  gdn_scan=seq
amort   … mtp_route=amortised gemm=lt    gdn_scan=seq
```

## 2. Which families the switch actually moves — and which it cannot

The engine's own default `--gdn-scan` is **already `seq`**, so on a default command line the amortised arm
moves the **GEMM axis only**: `lib_gemm = wave_gemm_is_library(sc.wave_gemm, n)`
(`engine/decode_step.cpp`) flips the ten fat shapes from the hand GEMV to hipBLASLt, i.e. the `proj`,
`mlp_gateup` and `mlp_down` families, and the same `lib_gemm` guard deletes the BF16→F32 widenings those
shapes never read (the `cast` family, which is why `LAUNCHSTAT` — identical between the two arms — counts
neither). `attn` and `lm_head` are untouched by this flag: attention is owned by `--attn-*`, and the batched
head is one pass over the tied table at any row count.

`gdn_rec` moves **only** if `--gdn-scan collapse` is named alongside, and that combination is refused
wherever a mid-round checkpoint is armed (see §3), so **for the accept round specifically the route switch
can only ever move the GEMM** — which is also where `ROW-SCALE.md` put the money (`proj` 2.73 → 4.27 ms
between 1 and 2 rows at 4k, against `gdn_rec` 0.32 → 0.43 ms). Anything that wants the recurrence
amortised has to solve the rollback first, not flip this flag.

## 3. Two refusals, because "amortised" must not silently mean something else

* `--mtp-route` named with neither `--mtp-verify` nor `--mtp-accept` → refused (the `--mtp-sched`
  precedent: a knob for a body that is not running).
* `--mtp-route amortised` + `--gdn-scan collapse` + (`--mtp-accept` or `--mtp-midsplit`) → refused. The
  mid-round checkpoint exists as an address only when the recurrence walks one token-step per launch, so an
  amortised round on `collapse` would restore a state the body never wrote — a wrong token stream, not a
  red comparison. Serving it a silently re-forced `seq` would have made a named flag inert, so it is a
  refusal at the flag; `verify_body` and `accept_round` also throw if a checkpoint is ever armed over a
  non-`seq` scan, for a caller that did not come through argv.

Verified at parse time, no GPU touched: bad value, no-body, collapse+accept and collapse+midsplit all
refuse with the reason; `--mtp-route hand --gdn-scan collapse` is **not** refused, because that pair is
inert by construction (the forcing overwrites the scan anyway).

## 4. Numerics — REPORTED, not gated

No band was widened, no gate threshold edited, no gate file edited
(`prototype/mtp_verify/gate.py` and `engine/gate.py` are untouched). `--mtp-route` defaults to `hand`, so no
pinned gate can see the amortised arm unless someone asks for it.

Instrument: `prototype/mtp_verify/route_diff.sh`, whose row comparison is `rowdiff.py` over the engine's
own dumps. Row 0 of the body pairs with the plain step's `logits<s>.bin` at the same step number — the
pairing `prototype/mtp_verify/gate.py` §(D) uses. Ranges below are min..max over anchors
(dec8: 8 anchors at 17–24; the 1k arm: 4 anchors at 1024–1027, steps 4).

| comparison | lanes | argmax agrees | 1−cos | meanrel |
|---|---|---|---|---|
| hand row 0 vs the one-row step | dec8 | 8/8 | 2.564e-05 … 9.840e-05 | 6.806e-03 … 1.441e-02 |
| **amortised row 0 vs the one-row step** | dec8 | **8/8** | 2.401e-05 … 9.311e-05 | 6.343e-03 … 1.447e-02 |
| hand row 0 vs amortised row 0 | dec8 | 8/8 | 1.985e-05 … 9.735e-05 | 5.802e-03 … 1.422e-02 |
| hand row 0 vs the one-row step | 1024 | 4/4 | 2.006e-05 … 2.942e-05 | 5.846e-03 … 7.767e-03 |
| **amortised row 0 vs the one-row step** | 1024 | **4/4** | 2.030e-05 … 2.942e-05 | 5.927e-03 … 8.331e-03 |
| hand row 0 vs amortised row 0 | 1024 | 4/4 | 1.479e-05 … 2.446e-05 | 4.845e-03 … 6.367e-03 |

Readings, in order of how much they are worth:

1. **The switch moves row 0 by about as much as the body already differed from the step.** The
   hand-vs-amortised gap and either arm's own body-vs-step gap are the same order at both widths — the
   amortising GEMM is not adding a new class of error on top of the wide-vs-fused form gap that is already
   admitted (`CASCADE_BAND["row_1cos_max"]` 2.0e-3, `row_1cos_max_lt` 2.75e-3, `BAND["meanrel_hard"]` 0.05,
   all unchanged, quoted from `engine/gate.py`, which is where they are owned). Worst row here: 9.8e-05.
2. **Every row-0 decision agrees at every anchor on both arms** — `row0_eq_step=1` on all 12 rounds, and
   `rowdiff`'s argmax SAME on all 24 row pairs including the near-ties. `row0_top` does move
   (e.g. dec8 anchor 17: 23.1856 hand vs 23.0724 amortised vs 23.0824 step), so the rows are **not**
   bit-identical and nobody should claim they are; the decision is what agrees.
3. **`ndiff` and max-ULP are the wrong instruments here and should not be quoted as if they were.** Both
   dumps are F32 vocabulary rows, and two legal summation orders of a 24-layer cascade move ~75 % of the
   elements and tens of thousands of BF16 ULPs at a `1−cos` of 3e-05 — which is `AGENTS.md`'s
   "a max-ULP band is blind to whole-tensor 1-ULP error, and ndiff is blind to chains" row, applied to F32.
   The aggregates that mean something on a chain are `1−cos` and `meanrel`, and those are what §(D) judges.
4. `sampled.txt` byte-identical between plain / hand / handx / amortised on both widths, and
   `emitted.txt` byte-identical between the two **accept** arms at dec8 (4 rounds, accept_rate 0.75, same 9
   tokens). One short fixture, so that is a smoke result and not an accept-rate claim: `m3_loop_rate.py`
   over the corpus population remains the instrument for that, and it has not been re-run.
5. `--mtp-verify 1 --mtp-midsplit 1 --mtp-route amortised` still prints
   `split_eq_batch=1 logits_ndiff=0 end_ndiff=0 verdict_eq_host=1` — the split's own bit-identity admission
   is unaffected by the route, as it should be: both sides of that comparison run the same route.

## 5. What the main agent has to do (not done here — no timing taken by a worker)

One lock hold, blocking, zero subagents, medians of ≥3. `EXTRA` is appended to every arm that runs a body.
`--mtp-route` is the reason it exists, and the base arm has it stripped — with a printed note, not silently —
because a one-row step has no body to route and the engine refuses the flag with neither `--mtp-verify` nor
`--mtp-accept` running; a one-row body resolves to the hand GEMV under either setting
(`resolve_wave_gemm`), so the strip does not make the denominator differ between the two runs.

```bash
prototype/mtp_accept/verify_body_scale.sh 128 1024 4032                      # the forced route (2.085) again
EXTRA="--mtp-route amortised" prototype/mtp_accept/verify_body_scale.sh 128 1024 4032
```

**The route canary, and the control that says it bites.** The script reads `want_route` out of `EXTRA` (an
empty `want_route` means "no route was asked for", which is not the same as asserting nothing) and aborts
unless a verify arm's `VERIFY2` lines carry `mtp_route=<want>` and `gemm=<expected>`, and an accept arm's
`MTPSTAT` carries the same pair. `AMORTISED_GEMM=<name>` overrides the `gemm=` the amortised arm is expected
to echo, which is how the canary was witnessed at 128 lanes / `STEPS=2` / `REPS=1`, one lock hold, **no
timing taken**: green with `EXTRA="--mtp-route hand"`, green with `EXTRA="--mtp-route amortised"`, and with
`AMORTISED_GEMM=wrongname EXTRA="--mtp-route amortised"` it aborted —
`ABORT: 128/verify did not echo gemm=wrongname; --mtp-route amortised never reached the GEMM seam`. A canary
nobody has made fail is decoration (`AGENTS.md`, the empty-parse and stale-binary rows together).

The route under each arm is in every arm's log (`VERIFY2 … mtp_route= gemm= gdn_scan=`, and `MTPSTAT`'s
same triple); quoting the pair without naming the route is the misreport this whole flag exists to avoid.
The pre-registered line to judge against is `ROW-SCALE.md` §4: **≤ 1.55 steps at 4,000 lanes**.

Optional follow-ups, none of which this file decided:

* **Citation drift this edit caused, and how to fix it before committing.** `engine/decode_run.cpp` is cited
  by line from journals and prototype docs, and this edit grew it from **3863 to 3952 lines** (the Options
  field, the parser, two refusals, the `VerifySide` fields, the three echo sites, the help entry). `tools/
  citation_rot_check.py` run on the dirty tree reports **31 rotted line-anchors** into that one file, and
  that count goes to **0 the moment the commit lands**, because its baseline is HEAD and HEAD moves with the
  commit — `AGENTS.md`'s "cannot see drift introduced inside the change being checked" row, exactly.
  `tools/citation_repoint.py` (new in this change) prints the new coordinate for every one of them, and its
  output for this edit is recorded in `prototype/mtp_accept/evidence/repoint-decode_run-2026-10-23.txt`:
  `152 -> 167`, `325 -> 340`, `1797 -> 1857`, and so on. Two of them need an eye rather than a sed, because
  the cited text is not unique (`1460 -> 1511`, one of 29 identical `//` lines; and `61` is blank at HEAD, so
  no text match is possible). Journals under `progress/` are frozen and take a same-line substitution, which
  is what a re-point is; `progress/2026-10-21/performance-status-2026-10-21.md` needs the user's explicit
  words before it is touched at all. Re-pointing was left to the commit owner rather than done here, because
  it spans files this change has no other business in.
* Whether the amortised arm should be *gated*. `prototype/mtp_verify/gate.py` has no flag passthrough
  today, so it judges the default arm only. Adjudicating the amortised arm is a gate-policy decision (and
  per `AGENTS.md` the numeric-format rule, a green gate set is the decision procedure).
* `prototype/mtp_verify/ARMS.md`'s canary line said `VERIFY2DONE … gemm=hand gdn_scan=seq` unconditionally;
  it now says the echo reports the route that RAN, so `gemm=lt` is correct on the amortised arm. That edit is
  line-count-neutral on purpose: `prototype/mtp_verify/CROSSROW.md:126` cites `ARMS.md:5` by line.
* The GDN axis on the accept path is still pinned by the checkpoint, so a round cannot get `--gdn-loop 512`
  or `collapse` until the rollback is redesigned. That is the next-cheapest lever after the GEMM and it is
  a design item, not a flag.

---

## 6. The pricing (main agent, 2026-10-23) — the pre-registered route gate FAILS

`prototype/mtp_accept/evidence/route_ab-2026-10-23.txt`. Same script, same fixture, medians of 3, one lock
hold per arm, blocking, zero subagents. The route was read back off the logs rather than trusted from the
flag (`MTPSTAT route=amortised gemm=lt gdn_scan=seq`; `VERIFY2 … mtp_route=amortised gemm=lt`), and the
hand arm reproduced the pre-flag figures within window noise (2.133 vs 2.085 at 4k).

| lanes | round/step, `hand` (default) | round/step, `amortised` | change |
|---|---|---|---|
| 128 | 1.670 | 1.582 | −5.3 % |
| 1,024 | 1.744 | 1.617 | −7.3 % |
| **4,032** | **2.133** | **1.964** | **−7.9 %** |

**The route gate in `ROW-SCALE.md` §4 was ≤ 1.55 steps at 4,000 lanes. The best measured is 1.964, so it
fails by 27 %, and by its own wording no k=2 loop gets built on this round.** The flag is still worth having
(it is the instrument that proved the point, and it is where any future route work will be measured), but
the hypothesis it tested is false at long context.

**Why the sizing was wrong, in one sentence:** `ROW-SCALE.md`'s long-KV column priced bodies *averaged over
their position in a 4,032-token pass* — most of those bodies sit at a few hundred lanes, while a decode
round sits at 4,032 — and the term that dominates a round is exactly the one that grows with position. The
weight-walk half of the claim survives (`lm_head` is flat, and rows beyond the second really are almost free
short of context); the long-context half does not, and a ramp-averaged body is not a body at the end of the
cache.

What the round's second row actually costs at 4k, on the amortised route: `22.235 − 11.319 = 10.9 ms`, i.e.
**0.96 of a step**, against llama.cpp's **0.078 step per extra row**. That ratio — 12×, not 1.15× — is the
distance, and it is not in the GEMM axis: the switch moved `proj`/`mlp_*` to hipBLASLt and bought 0.17 step.
The rest is per-row traffic the round pays per row: the long-context attention scan (`attn` is 3.02 ms of an
11.2 ms step at 4k) and `gdn_rec`, which is one row per launch *by construction*. Closing that gap is a
kernel programme — multi-query attention that shares one KV scan across query rows, and a multi-token GDN
scan — not a loop change.

**Consequences, stated as arithmetic.** At the corpus accept rate a = 0.6556, a k=1 round returns
`1.6556 / round_per_step`: **0.776× on the hand route, 0.843× amortised** — i.e. −22 % improved to −16 %,
still a loss. A third row would add about the same 0.96 step, giving ≈ 2.9 steps for `1 + a₁ + a₁a₂ ≈ 2.0`
tokens at the k=2 rate: **0.69× — k=2 is worse than k=1 at 4k, not better.** At short context the picture
inverts (rows beyond the second cost ~0.15 ms each there, so a 3-row round sizes at ~1.6 steps for ~2.0
tokens ≈ **+23 %**), but the standing rule reports decode at 4,032 lanes and the adopt line is set there,
so a short-context-only speculation win is a scope decision for the user, not a result of this pass.
