# B0 — the per-phase MTP round meter (`engine/mtp_meter.h`)

**Status: built, device-gated, identity-proved. No millisecond figure in this file is a measurement of the
engine — every one of them is the main agent's to take with `meter_round.sh` in one clean lock hold
(`AGENTS.md`: GPU execution is serialized and belongs to the main agent). What this file documents is each
phase boundary, whether it is a hipEvent pair or a stream sync, and what each phase can therefore under- or
over-report.**

## 1. What B0 is for

`ROW-SCALE.md` and `M3-RESULT.md` established the shape of the problem: an accept round costs ~1.96–2.09
plain steps at 4,032 lanes, and the extras anyone could name — the mid-round checkpoint capture and restore
(~19.27 MiB), the verdict kernel plus a 16-byte readback, one draft round — do not add up to the gap. That
is a budget with no line items. B1 ("cut the round's fixed cost") and B2 ("cut its per-lane cost") cannot be
sized against a total; only against a phase big enough to be worth attacking. B0 produces the line items, and
it is deliberately an instrument with a **visible price** rather than a cost accounting: every phase prints
how it was bounded, and where that bound costs the run something the line says so.

## 2. Why a second instrument when `stage_clock.h` exists

`engine/stage_clock.h` prices a **step**: it segments the queue at §07.2 family boundaries, reconciles
observed launches per family against the compiled launch plan (`engine/launch_stats.h`), and is admissible
only when that reconciliation holds (`plan_match=1`). An MTP round is not a step. Its body is one
`submit_wave_body(..., kRows=2)` whose per-token launch bill is already accounted in
`engine/decode_step.h`, and none of the round's own parts — checkpoint copy, shadow re-seed, lane staging,
verdict kernel, 16-byte readback, rollback — is a §07 family. That is precisely why `--mtp-verify` and
`--mtp-accept` already refuse `--time-stages`. So this is a second, small instrument rather than a
contortion of the first, and `--mtp-meter` refuses `--time-stages`/`--time-waves` too: the stage clock drains
the queue at family boundaries to price a step, this drains it at phase boundaries to price a round, and
armed together each serializes the other's subject.

That refusal sits **above** the older `--mtp-verify`/`--mtp-accept` refusals of the same flags, deliberately:
both of those already refuse `--time-stages`, so a meter refusal written below them could never be reached. A
refusal that cannot fire is the failure shape this tree keeps recording. Verified reachable:

```
$ engine/build/decode_run --model models/Qwen3.5-0.8B --ids 1,2,3 --mtp-meter 1 --mtp-verify 1 --time-stages
decode_run: --mtp-meter and --time-stages/--time-waves cannot share a run: ...
$ engine/build/decode_run --model models/Qwen3.5-0.8B --ids 1,2,3 --mtp-meter 1
decode_run: --mtp-meter 1 meters an MTP round and there is none: name --mtp-verify 1 ... or --mtp-draft 1 --mtp-accept 1 ...
$ engine/build/decode_run --model models/Qwen3.5-0.8B --ids 1,2,3 --mtp-meter 2
decode_run: --mtp-meter takes 0 (off, the default) or 1 ...
```

## 3. Why every boundary is a host clock pair and not a `hipEvent`

`AGENTS.md` records that `hipEventElapsedTime` returns **exactly 0.000 ms** for a 2.1 ms kernel on this part,
not only for sub-millisecond ones. Several phases here live in the hundreds-of-microseconds band
(`lane_save`, `lane_revert`, `logits_d2h`, `verdict_readback`, both bulk restores), so device events would
print zeros precisely for the phases B1 cares about. Host `steady_clock` pairs do not have that failure, and
each phase is bounded either by a call that already returns to the host or by a wait this instrument adds, so
a host pair sees the device time either way.

That choice is the whole reason the labelling below exists: **a host pair bounds device work only if something
makes that work finish before the clock read.** Where today's code already does, the phase is honest and free.
Where it does not, the meter adds one `hipDeviceSynchronize()`, prints `sync_added=1`, and the number becomes
an upper bound.

## 4. The arithmetic, and what makes coverage a real check

* **window** — a host span belonging to a round. Windows are disjoint and additive; the round's **wall** is
  their sum. Opening a second window while one is open throws: overlapping windows would double-count the
  wall, and a check that cannot be broken is not a check.
* **round** — the unit every `ms_per_round=` divides by. A phase may only be charged inside a round
  (`round_begin` before any window opens, `round_end` after every window closes; both enforced).
* **phase** — a named span inside a window; `begin()` throws if a phase is already open, so a mis-placed guard
  is a crash and not a silent double charge.
* **`overhead`** — `status=derived`, never timed: `wall − Σ(measured phases)`. Round cost that no site owns
  (candidate arithmetic, route-knob save/restore, scratch-row clearing, driver tallies). It is **excluded from
  the sum**, which is what makes coverage falsifiable instead of 1.000 by construction.
* **`summary`** — `status=derived`, also excluded: the meter's own verdict line.
* **`coverage`** = Σ(measured phases) / wall, band **0.8–1.2**, enforced twice: the engine prints
  `meter=out_of_band` on its own `RESULT` line, and the harness refuses to tabulate at all.
* **`rounds=` vs `runs=`, on every phase line.** `rounds=` is THE RUN'S ROUND COUNT — the denominator
  `ms_per_round=` actually divides by, and the same number on every phase line and on `RESULT`. A phase that
  one reject in six rounds pays therefore reads as the diluted per-round average the funding question wants,
  and `ms_per_run=` is the price of ONE execution of the site. `runs=` is site executions: `runs > rounds`
  means more than one execution per round (`report_prints` prints twice per accept round), `runs < rounds`
  means a branch not every round takes (`rollback`, rejects only), and `runs=0` means the flag that would
  exercise the phase was off. The meter deliberately does NOT print "how many rounds charged this phase":
  that number into `ms_total` is the per-execution price wearing a round's label, which is the mix-up this
  pair exists to prevent.

Output contract, one line per phase per run, `RESULT` still the run's last stdout line:

```
MTPMETER role=verify phase=body ms_total=… rounds=… ms_per_round=… sync_added=1 runs=… ms_per_run=… note=…
MTPMETER role=accept phase=mid_checkpoint status=not_metered note=…          ← deliberately has no ms_total=
MTPMETER role=accept phase=overhead … status=derived …
MTPMETER role=accept phase=summary … status=derived wall_ms_per_round=… coverage=…
MTPMETER role=accept phase=instrument windows=… rounds=… plain_steps=… syncs_added=… idle_sync_us=… sync_errors=…
RESULT mode=run … stopped=… meter=ok rounds=… sum_ms_per_round=… wall_ms_per_round=… coverage=…
```

The verdict rides the run's **own** `RESULT` line rather than being a second `RESULT` line: `engine/gate.py`
keeps the last `RESULT` it sees and every other harness in the tree greps `^RESULT`, so a second one would be
the line they parse. If a device wait fails, `RESULT` says `meter=sync_failed` and the failing string is
printed on the instrument line — a faulted queue must still let the run print its result.

`meter_table.awk` adds two checks the engine cannot make about itself:

1. `Σ(rows) + overhead == wall` (±2 µs/round) — so the derived row cannot absorb a double charge.
2. `Σ(rows) == RESULT sum_ms_per_round` — the engine's accumulator against its printed rows. This is the one
   that catches a phase **charged but not printed**: check 1 passes vacuously whenever `overhead` swallows
   whatever went missing.

`prototype/mtp_accept/meter_round_selftest.sh` (CPU-only, no device, no lock) feeds the renderer synthetic
logs that are wrong in exactly one way each and asserts the *named* refusal: coverage 0.5, overhead bumped,
RESULT sum bumped, `meter=out_of_band`, coverage field absent, phase rows belonging to the wrong role, and an
empty log. It failed twice while being written and found two real defects in the checker itself: `log` is an
awk built-in so `-v log=` aborts the program ("cannot command line assign to log"), and `wall` was being read
from the summary row's `ms_per_round`, which is the *sum*, so every log good or bad failed the identity for
the wrong reason. Both are why the positive control is in the file: without it, every refusal could be
passing because the renderer cannot print anything.

## 5. The phases, their boundaries, and their bias

`Kind` lives in the registry (`engine/mtp_meter.cpp`, one static_assert binding it to the `PhaseId` enum)
rather than at the call sites, so a line cannot be printed with a label its author forgot.

| kind | boundary as shipped | `sync_added` | what `ms_per_round` means |
|---|---|---|---|
| `kNativeBlock` | the phase ends in a HIP call that **already blocks today** — `hipMemcpy` D2D/D2H returns only once the copy landed | 0 | a bound the run was already paying for; costs nothing to know |
| `kHostOnly` | pure host work; nothing to drain | 0 | exact |
| `kSyncAdded` | the phase ends in async submissions, so the meter adds **one** `hipDeviceSynchronize()` | 1 | **upper bound**: it serializes work the round would have overlapped |
| `kNotMetered` | no honest boundary exists at today's call sites | — | no number is printed, by design |
| `kSubphase` | a **breakdown of a span another phase already owns**; its own close adds no wait | 0 | exact for what it times, but it is NOT in the sum: `draft` already contains it, and `meter_table.awk` excludes it for the same reason it excludes `derived` (`meter_round_selftest.sh` reddens a log where that exclusion is broken) |

Only `kSyncAdded` phases pay a wait — that is the content of "do not add a sync to a phase that has none
today without labelling it" — and `phase=instrument` prints `syncs_added=` so the count is auditable. What a
`sync_added=1` phase loses is the **overlap**, not the wait: an empty `hipDeviceSynchronize()` on a drained
queue is sub-microsecond here, and the run measures its own floor and prints it as `idle_sync_us=`.

### role=verify — the read-only side pass (`--mtp-verify 1 --mtp-meter 1`)

Two windows per round (four with `--verify-guard 1`): the pre-step `take_snapshot`, and the side pass itself.
The plain decode step sits **between** them and is not in the wall — it is `--time-stages`' subject.

| phase | boundary | kind | under/over-report |
|---|---|---|---|
| `snapshot_take` | `take_snapshot`: two blocking D2D copies of `sizes()` bytes (19.27 MiB here) | native | none. This is the per-round cost D3 excluded as out of scope rather than solved |
| `shadow_seed` | `restore_snapshot`: checkpoint→shadow, same bytes | native | none; it is the copy a reject *would* pay, charged unconditionally because the side pass pays it unconditionally |
| `witness` | every `--verify-guard`/`--mtp-midsplit` download and byte-compare, at all five sites | native | reads exactly 0.000 with both flags off, which is the only arm a timing pass should run; on an admission arm it is tens of MiB of D2H per round and dominates everything else |
| `lane_save` | one blocking 512 B D2D per KV region per body row — 24 regions here (6 full-attention layers × 2 KV heads × K and V, `kHeadDim` 256 in BF16), so 24 copies and 12 KiB per row, 48 and 24 KiB per 2-row round | native | none, and this is the launch-count-bound phase a fused staging kernel targets: the bytes are trivial, the 48 calls are the cost |
| `body` | `submit_wave_body` at the round's row count | **sync added** | **over-reports**: today the body's last submissions flow into the head launch and the next blocking call, so a wait here removes pipelining the engine relies on. Also carries the mid-checkpoint's extra submissions whenever one is armed |
| `body_with_checkpoint` | only under `--mtp-midsplit 1`: a second body, same anchor and tokens, mid-checkpoint **armed** | **sync added** | `body_with_checkpoint − body` is that checkpoint's price measured in one window at one anchor — the only honest way to ask it. Off ⇒ row prints 0.000 with `runs=0` |
| `lane_revert` | the same copies, other direction | native | none |
| `head` | one `lm_head_batched_xbf16` pass over the 485 MiB tied table, serving both rows | **sync added** | over-reports in a pipelined round, but not for free in a serial one: the logits D2H two lines later already waited for this launch on the same stream, so the wait is relocated rather than added |
| `logits_d2h` | the blocking vocabulary download: `kRows × kVocab` F32 = 1,986,560 B (1.89 MiB) here | native | none. This line beside the accept loop's `verdict_readback` **is** the traffic argument: 1.89 MiB against 16 B |
| `argmax_report` | host `argmax_greedy` over the downloaded rows | host-only | exact, and not free: 2 × 248,320 F32, a memory-bound host loop over ~1.9 MiB. A line item no one proposed, because it had never been visible |
| `report_prints` | the run's per-round lines (`printf` + explicit `fflush`) | host-only | exact; named so a reader can tell whether the reporting is the cost instead of finding it in `overhead` |
| `overhead`, `summary` | **derived** | — | see §4 |

### role=accept — the shipping accept loop (`--mtp-draft 1 --mtp-accept 1 --gdn-loop 1 --mtp-meter 1`)

One window per round, spanning the round branch: the round, the next round's draft, and the run's own prints.

| phase | boundary | kind | under/over-report |
|---|---|---|---|
| `body` | `submit_wave_body` at 2 rows, carrying the mid-checkpoint's submissions | **sync added** | over-reports, same reason as verify's `body`, and by more: this is the body whose pipelining the round depends on |
| `mid_checkpoint` | **no boundary at this level** | not metered | prints no millisecond figure, and guessing one would be the lie. The capture is 2 extra submissions **per GDN layer** interleaved inside `submit_wave_body` (conv-mid derivation plus a per-layer state copy); bounding it means 18 waits inside the call whose pipelining is the thing being priced. Priced instead as role=verify's `body_with_checkpoint − body` under `MIDSPLIT=1` |
| `head` | one batched pass over the tied table, both rows | **sync added** | over-reports: submissions only today; the verdict kernel and readback follow it, and the readback is what actually blocks today |
| `verdict_kernel` | `mtp_accept_verdict` over both logits rows, on device | **sync added** | over-reports (the wait makes the on-device argmax serial). Its true cost is bounded below by this line and above by the vocabulary download it *replaces* — which is why it prints next to `--mtp-verify`'s `logits_d2h` |
| `verdict_readback` | the four-int verdict word, blocking D2H | native | none. This path has **no** logits D2H at all: the vocabulary never leaves the device |
| `accept_path` | decoding the verdict words and checking the candidate echo — runs on **both** branches | host-only | exact. An accept restores nothing, so a near-zero here is the expected result and not a missing phase; `runs=` equals the round count |
| `rollback` | the reject branch's two bulk D2D restores of the mid-checkpoint (19.27 MiB here) | **sync added** | **inflates hardest of any phase**: these are `hipMemcpyAsync` today and the round deliberately lets them overlap the next round's draft, so the meter's wait serializes a copy the real round hides. Read with the accept rate, never alone. `runs=` is the reject count, so `ms_per_round` is diluted by accepts while `ms_per_run` is the price of an actual reject |
| `lane_bookkeeping` | §08.6 `check_invariants` after either branch, plus the rejected round's draft-lane truncate in the driver | host-only | exact; moves 0 bytes. `runs=` = rounds + rejects, which is the honest count of the two sites |
| `draft` | the **next** round's draft round, charged to this round's window because that is where it is paid for; ends in the draft's own blocking vocabulary D2H | **sync added** | mildly over-reports (the D2H already waited for the draft's head pass, so most of the wait is relocated). Plain one-row iterations also draft, and those drafts are charged **nowhere** — a plain iteration is not a round |
| `draft_vocab_d2h` | the draft's blocking download of its own logits row — today `kVocab` F32 = 993,280 B per round | sub-phase of `draft`, native | exact, and out of the sum because `draft` already carries it. This is the second of the two vocabulary crossings `ROUND-TAX.md` §1 names; after the argmax is resolved on device the line bills one small status word, and a 0.000 there **is** the result rather than a missing phase |
| `draft_argmax_host` | the host `argmax_greedy` over that downloaded row: 248,320 F32 scanned on one CPU core | sub-phase of `draft`, host-only | exact. `runs=0` means the site is gone (the device argmax removed it), not that it was missed |
| `report_prints` | the `ACCEPT` line and the `ACCEMIT` emit block | host-only | exact |
| `overhead`, `summary` | **derived** | — | route forcing inside `accept_round`, driver tallies, the draft-cache refusal probe |

### Deliberately outside every window

* The **plain decode step** in the verify shape (§5 header note). `overhead` means "what no phase owns inside
  those two spans", not "the step".
* **Plain one-row iterations** in an accept run — counted in `plain_steps=`, never metered, so a run that
  speculates poorly cannot dilute a per-round figure into a step's cost. The first iteration of every accept
  run is one (nothing is proposed yet), as is every iteration at the context ceiling.
* The **verify shape's draft**: `--mtp-draft`'s own accounting, outside both windows.
* Anything printed after the meter disarms itself, just before `RESULT`.

Structural witnesses from the device run (counts, not timings): verify reported `windows=24` for
`rounds=12` — two windows per round as designed — with `syncs_added=24`, i.e. exactly the two `sync_added=1`
phases per round; accept reported `windows=6 rounds=6 plain_steps=2 syncs_added=25`, the 25th being the one
reject's `rollback`. `lane_bookkeeping runs=7` = 6 invariant checks + 1 lane truncate, and `rollback runs=1`
against `accept_path runs=6` reproduces the accept/reject split independently of `MTPSTAT`. A meter that had
lost a guard cannot produce those coincidences.

## 6. The meter must not change what the engine decides

An added **wait** is not an added **submission**: it moves no byte, allocates nothing, touches no operand and
submits nothing. The claim is therefore the strong one — emitted token stream, per-round verdicts, accept
statistics and §08 state fingerprints identical with the instrument on and off — and it is checked, not
assumed, by `prototype/mtp_accept/meter_identity.sh` (correctness-only; it takes no lock of its own so the
caller holds one, per the self-locking-runners trap):

* both shapes, OFF vs ON, byte-compared: `emitted.txt`/`sampled.txt` and every `ACCEMIT`/`STEP`/`ACCEPT`/
  `VERIFY2`/`STATEFP`/`MTPSTAT`/`PREFILL` line. Verdict lines carry the printed logit **values**, so an
  operand that drifted shows there long before it shows in a token.
* `DECWALL` and `RESULT` compared **field by field** and classified: `[timing]` (the meter's own fields and
  the duration fields a device wait is allowed to move — each printed with its old and new value) versus
  `[DECISION]` (`last_sampled`, `prefix_len`, `stopped`, `steps`, `bodies`, `plain`, `accepted`, `rejected`,
  `suppressed`, `emitted`, `stop`, `ended`, `census`). Any `[DECISION]` difference fails the run. A plain
  string diff would fail on the one expected difference and teach the reader to ignore it; blanket-stripping
  the durations could not tell "only durations moved" from "a counter moved too".
* the meter's `rounds=` is **adjudicated**, not excused: it must equal the run's own count (`MTPSTAT rounds=`
  for the accept loop, one VERIFY2 line per step for the side pass), or every `ms_per_round` on the table
  divides a population the run never had.
* rounds must be > 0, or the comparison would be two empty logs — the empty-parse failure this tree records
  for every gate driver.

Taken on `dec8` (17-field prompt from `manifest.prefill_ids`, 12 generation steps), one lock hold, blocking,
util 92 % at acquire and 27 % at release, **no timing taken**:

* `engine/gate.py --fixture dec8` → **ENGINE GATE PASS** (9 s wall) with the meter compiled in and OFF;
* `meter_identity.sh` → **METER IDENTITY PASS**: token streams identical, all 86 `STATEFP` rows identical per
  shape, `MTPSTAT` identical, `DECWALL`/`RESULT` differing only in the duration fields, which it names;
* the accept loop's own admission, `run_accept.sh STAGES=identity,corpus` (26 s harness wall), reproduced
  its carried `a = 615/926 = 0.6641` on the shipping path with the flag off, census cross-check green — so
  the guards added in `engine/mtp_verify.cpp`/`engine/mtp_accept.cpp` did not move the shipping path either;
* `meter_round_selftest.sh` → **SELFTEST PASS**.

Flag-off cost is one null-pointer read per guard object (`stageclock::note()`'s discipline): no allocation, no
added wait, no line.

## 7. Taking the measurement

```
tools/gpu_lock.sh status                                     # free, and no STALE holder
prototype/mtp_accept/meter_round.sh 128 1024 4032 --route hand
prototype/mtp_accept/meter_round.sh 4032 --route amortised
MIDSPLIT=1 prototype/mtp_accept/meter_round.sh 1024           # adds the arm that prices the mid-checkpoint
prototype/mtp_accept/meter_round_selftest.sh                  # CPU only; re-run whenever meter_table.awk changes
```

`meter_round.sh` self-acquires `tools/gpu_lock.sh` (TAG `main-agent meter-round`), so **do not call it while
holding the lock** — it parks silently. Canaries run before any millisecond is read, and each exists for a
failure that has already happened in this repo: exactly one `RESULT`, and it is the run's last stdout line (a
`tail -1` harness must not read a summary instead); a `MTPMETER … on=1` header, the behavioural canary that
the binary under test actually contains the instrument, since every other field in the log agrees with a
stale build; `PREFILL tokens=` equal to the field count read off the ids **file** (`--ids-file` splits on
commas, so a space-joined fixture reads as one token); `DECWALL census=ok`; the route echoed back
(`VERIFY2 mtp_route=`/`gemm=`, `MTPSTAT route=`/`gemm=`), so a `--mtp-route` that never took cannot report the
arm it did not run; and the meter's `rounds=` equal to `MTPSTAT rounds=`.

`--gdn-loop 1` is carried on **every** arm including the side pass: a flag one arm requires belongs on both, or
the pair prices the flag instead of the round (`AGENTS.md`, the `mtp_wall_ab.sh` row), and the accept loop pins
it.

## 8. Reading the table honestly

1. **A `sync_added=1` phase is an upper bound.** It costs the round the overlap it removed. `rollback` and
   `body` inflate most, and both inflate by de-pipelining a round that genuinely overlaps today.
2. **Compare phases against each other and against `DECWALL`, never against a previous window.** A short prompt
   is dispatch-bound and bimodal on this box; the question B1/B2 asks is which phase *rises with lanes*, and
   that is a paired measurement inside one lock hold, not two runs (`AGENTS.md`: a ruler re-taken in a
   different window is not an A/B).
3. **Never write these numbers into `engine/performance-status.md`.** The ledger's decode headline is the
   4,032-lane stage-clock number; a metered round is a differently-serialized object with a different
   denominator.
4. **Do not sum the two roles.** `--mtp-verify` runs one side pass *per step* while an accept round *replaces*
   a step; the same word "round" names two populations, which is why the roles carry separate phase sets and the
   harness reads the route echo from the shape that ran.
5. **`runs=0` means the flag that would exercise the phase was off** (`witness`, `body_with_checkpoint`), not
   that the phase is free.
6. `overhead` and `summary` are derived; adding either to a phase total double-counts.

## 9. Known limits, and what a successor must decide rather than assume

* **No phase may be added without naming its bias.** A phase closing on an async submission is `kSyncAdded` and
  inflates; there is no kind that is both precise and free for such a phase, and the hipEvent alternative was
  rejected for a measured reason on this part (§3).
* **`overhead` grows silently** when a round gains host work outside the guards. The band is the alarm, the
  derived line is the display, and nothing names the growth because nothing times it. If a future shape makes
  `overhead` a real fraction of the wall, that is a phase waiting to be named — not a band to widen.
* **The accept loop's mid-checkpoint is still not metered in the accept loop.** The `--mtp-midsplit 1` price is
  the *verify* body's delta at the side pass's anchor and route: the same body form, a different call site. The
  accept loop's own number would need per-layer marks inside `submit_wave_body`, which changes the thing being
  measured. That is a decision for the main agent, not an oversight to fix quietly.
* **Per-branch phases are per-branch, not per-round**: quote the accept rate beside `rollback`'s `ms_per_run`,
  or the diluted `ms_per_round` will be read as the price of every round.
* **Windows are host spans.** If a future engine submits from more than one host thread this instrument is
  wrong by construction — stated in its header, as `launch_stats.h` and `stage_clock.h` state the same limit.
* **`--verify-guard` and `--mtp-midsplit` change the shape as well as the witness.** They add windows and extra
  bodies; never mix a guard-on run's phases with a guard-off run's in one argument.

**Why the table tool is in `prototype/` and not `tools/`.** A second digest was proposed for `tools/`; it
was not promoted, and the reason is not style. Everything in `tools/` here is a **checker** that adjudicates
a tree-wide invariant (`citation_rot_check.py`, `doc_ref_check.py`, `gate.py`-adjacent tools) or a **device
probe** that owns a lock and a window (`mixed_bw`, `gpu_lock.sh`). `meter_digest.py` is neither: it reads one
instrument's logs for one experiment directory, its verdicts are the instrument's own (`meter=ok` or refuse),
and it touches no device. The live rule it enforces — **a refused arm is not a measurement** — is already
enforced where the refusal happens, by `meter_round.awk`; this file re-checks it so a digest cannot smuggle
an `out_of_band` arm into a table, which is a property of *this* experiment, not of the repo. Two copies of
the same parse (awk for the gate, Python for the human view) is a real cost and it is named in the tool's own
docstring; the mitigation is that both refuse on the same three fields (`meter=`, `coverage=`, `rounds=` vs
`MTPSTAT rounds=`), so they cannot disagree about which arms count. If a second instrument ever emits
`MTPMETER` lines, that is the moment this belongs in `tools/` — not before.
