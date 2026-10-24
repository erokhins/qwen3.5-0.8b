# MTP speculation is the default — what was decided, what it costs, and what every caller must now do

Status: **landed**. `engine/decode_run` ships `--mtp-accept 1 --mtp-k 2 --mtp-draft 1` as its parsed
defaults. This file is the record of why, the shape of the resolution rules that make a bare run
possible, the caller reconciliation that had to happen in the same move, and the list of things this
flip leaves genuinely unpaid.

No milliseconds are introduced here that were not already on disk: every duration in this file is a
carried citation with its source line, and the numbers this flip still owes are named as owed instead
of being guessed.

---

## 1. The decision

Three knobs move together, because the engine requires them together:

| knob | old default | new default |
|---|---|---|
| `--mtp-accept` | 0 | **1** |
| `--mtp-k` | 1 | **2** |
| `--mtp-draft` | 0 | **1** |
| `--gdn-loop` | 512 | 512 as parsed, **resolved to 1** when accept is on and no span was named — *overtaken 2026-10-25 by §7: 512 stands, and the round's own bodies are forced to 1 inside the engine* |

The option on the table was (A) flip accept only, (B) flip the mutually-required triple, (C) flip the
triple *and* retire the `--gdn-loop` pin. **(B) was chosen.** (C) is not a default change: narrowing
the span requirement to the round body would change what state provenance the accept round is admitted
against, i.e. it needs its own bit-provenance admission through gate (m) and the lossless corpus. It is
a funded follow-up, not a flag.

**Overtaken 2026-10-25.** (C) was funded by measurement and landed; see §7. The diagnosis in that paragraph
was right — it did need its own provenance admission — and the admission came back clean. The table row
above and §2 describe the decision *as it was taken*; §7 says what runs now.

`k=2` is the shipped depth, and `k=3` is not:

* `k=1` measures **+9.75 … +10.38 %**;
* `k=2` measures **+16.53 % / +17.76 %**, and **+20.68 %** with `--mtp-dense --mtp-prompt-lanes 146`
  (`prototype/mtp_accept/K2-RESULT.md` §the k=2 wall table);
* round/step **1.82–1.84** against the pre-registered **1.95** kill line — kill line not hit;
* lossless **49/49** on the full corpus;
* gate (m) admits rounds at **both** `k=1` and `k=2`.
* `k=3` projects ≈ **+18 %**, which is inside `k=2`'s own noise, and is **unimplemented, un-swept and
  un-gated**. It is not the default and nothing in this tree claims it.

`engine/mtp_accept.h` carries the depth as `kDefaultK = kMaxK` with a `kMinK = 1` so `--mtp-k 1` stays
nameable: the parse range is `[kMinK, kMaxK]`, and a range of `[kDefaultK, kMaxK]` would have made the
gate's own `k=1` admission cases illegal.

## 2. Why a bare run needs a span rule, and what the rule is

**THE RULE TODAY (2026-10-25, §7).** Nothing in `decode_run` resolves the span anymore. `--gdn-loop` means
what it says for every body that is not an MTP round, and the round's own bodies are forced to
`decode_step.h::kMtpRoundGdnSpan` = 1 by `submit_wave_body` wherever the mid-round checkpoint is armed — the
same arming that already distinguishes a round's body for `arm_mtp_round_attn_form`. `--gdn-loop <any>
--mtp-accept 1` is legal; the `MTP mode=` line echoes both spans. Everything below this marker is the record
of the rule as it was shipped on 2026-10-24, kept because it is the reason the round is forced at all.

The accept round restores from a mid-round GDN checkpoint, which is a one-token-step recurrence's end
state. A span of 512 batches the recurrence over a prefill body, so its checkpoint is not the state the
round restores into. The engine therefore *pinned* `--gdn-loop 1` whenever accept was on — correct for a
named flag, fatal for a default, because `--gdn-loop`'s parsed default is 512 and a bare
`--mtp-accept 1` would then refuse itself.

The resolution, in `engine/decode_run.cpp`, is deliberately narrower than "force the span":

* if accept is on **and `--gdn-loop` was never named**, the span resolves to 1 and is reported;
* if the user named a span, the name stands and the **refusal stays reachable** — `--gdn-loop 512
  --mtp-accept 1` still exits 2 and says why. An implicit default may satisfy a requirement; it may not
  erase one.
* the resolution is placed **before every gdn cross-check**, so a refusal sees the span that will
  actually run rather than the one that was parsed.

Echoed on every run (`MTP mode=` line, before `NUMERICS`):

```
MTP mode=accept k=2 draft=1 gdn_span=1 forced_by=--mtp-accept-default
MTP mode=off draft=0 gdn_span=512
```

`forced_by=--mtp-accept-default` versus `forced_by=--gdn-loop` is the difference between "the default
chose this" and "you chose this", and `engine/gate.py` now reads the line (see §4).

## 3. What the span costs, stated from numbers already on disk

The span matters to **prefill** only. `resolve_gdn_loop(want, rows)` collapses to per-token for a
one-row body, so decode was already paying the per-token recurrence at either default and does not move.

For prefill, the carried measurement is `gdn_rec` at 4,096 lanes: **449.07 ms** at span 1 versus
**266.03 ms** at span 512 (`progress/2026-10-21/performance-status-2026-10-21.md:1092`, the ratio row
1.6002/1.6893/1.6881; the same pair is cited at `:1126` as the thing the collapse earned). Against a
4k prefill wall of roughly **370 ms** serialized-wave, that is the shape of the penalty the new default
can carry — it is the reason `--gdn-loop` was 512 in the first place.

**That penalty is not measured for this default, and no number in this file pretends it is.** Every wall
number taken so far — the k=1/k=2 ladder, the corpus A/B, gate (m) — passed `--gdn-loop 1` on *both*
arms, so none of them ever compared span 1 against span 512 at the default. The first honest read of
"what does the default cost on prefill" is a paired A/B inside one lock hold, which is a main-agent
measurement (§6). Quoting any existing wall as the default's prefill price would be a category error:
those walls were all taken with the span pinned.

## 4. What changed for every consumer

Three consequences of the flip, each of which is a silent meaning-change if a caller ignores it:

1. **Output shape.** An accept run writes `emitted.txt`, not `sampled.txt` plus `logits<N>.bin` (the
   logits path is gated on `!o.mtp_accept`). A consumer that reads `sampled.txt` after a default run
   reads a file the run never wrote.
2. **Refusals become reachable by default.** `--time-stages`, `--time-waves`, `--session`,
   `--sizes-only`, `--replay` and `--mtp-verify` all exit 2 against default accept. These stay
   **refusals**; none of them was converted into a silent implication of `--mtp-accept 0`. An
   instrumentation harness must say so on its own command line.
3. **`--mtp-draft` is now on by default**, so a harness that wanted "draft off" for a clock comparison
   has to name `--mtp-draft 0`, and a harness whose *whole point* is env-driven draft scheduling must
   not name it at all (`prototype/mtp_accept/m1_stage_witness.sh` case B is that case: it sets
   `QWEN_MTP_DRAFT=1`, and an explicit `--mtp-draft 0` would beat the env and witness nothing).

`engine/gate.py`:

* `MTP_STEP_PIN = ["--mtp-accept", "0", "--mtp-draft", "0"]` at every site that prices logits or counts
  steps (batched-A/B side, gdn-spell side, smoke, the **main run**, section (b2));
* `parse_run` reads the `MTP mode=` line and `mtp_step_path_claim()` reddens a plain-priced run whose
  echo disagrees — the canary that makes a stale or repurposed binary visible instead of quiet;
* section (m) runs the **default-k witness** case: it names no `--mtp-k`, asserts the engine's default,
  the gate's mirror constant and the asked depth all agree, and prints `MTPGEOM k=` so the record says
  which depth ran. That case is deliberately outside the population totals — it is an admission of the
  default, not another sample.

## 5. Caller enumeration, and the classification rule

144 files name `--model`; the live `decode_run` invocation population came to **61 files with no MTP
flag** and **15 already MTP-aware**. 120 explicit pins landed across 61 files. Three classes:

* **instrumentation** (must name `--mtp-accept 0`, and usually `--mtp-draft 0` too, because a bare
  `--time-stages` otherwise meets the *draft* clock refusal first): every timing/attribution rig under
  `prototype/attn_percost`, `attn_wide`, `attn_wmma`, `attn_decode_wide`, `cast_fold`, `gdn_rec_isa`,
  `gdn_walk`, `glue_copy_fold`, `perprompt`, `wave_rows`, and `tools/{bench36_ruler,context_sweep,
  corpus_freerun,llama_phase_compare,longctx_check,prefill_sweep,regress_vs_origin,route_c_audit}`.
* **adjudication-aware** (keeps the accept path, so the pin goes on the *plain* arm or the run shape and
  the accept arm re-names `--mtp-accept 1` later — the last name wins): `engine/gate.py` (m);
  `prototype/mtp_accept/{m3_loop_rate,corpus_wall_ab,draft_cond_probe,draftk_ladder,run_accept}.py`;
  `prototype/mtp_accept/{meter_round,meter_identity,verify_body_scale,wide_body_ab,round_form_identity,
  row_scale,m1_stage_witness}.sh`; `prototype/mtp_verify/{gate,zero_gate}.py` and its `*.sh` dumpers.
* **left alone, on purpose**: `tools/reference_dump.py` (its `--model` is a torch dumper flag; it never
  execs the engine), `tools/prefill_kernel_trace.py` and `tools/prefill_occupancy.py` (they wrap a
  command line the *caller* supplies, so there is nothing to pin inside them),
  `prototype/mtp_accept/ids_from_text.py` (its own `--model`), and every harness line running
  `engine/build/decode_run.premtp` — a pre-MTP binary does not know these flags, so pinning it would
  break a canary that exists to prove the old binary refuses.

Two scripted sweeps did the mechanical half, and their admission tests are the interesting part.
They live in `tmp/` (scratch, not committed — re-derive them from this paragraph if they are needed
again). `pin_callers.py` pins only where the token before `--model` *is* the engine, refusing `case` arms
(`--model) MODEL=$2 ;;`), docstrings, `add_argument`, `ARGS+=(...)`, and a line whose command is a `.py`
harness — `run_accept.sh` passes `--model` to `gate.py`, which has no `--mtp-accept` and would have died
on an unknown flag. `pin_arms.py` pins arm *strings* that name `--mtp-verify 1` / `--mtp-draft 1`
without naming accept, and asserts the expected number of hits per file before writing, so a pattern
that silently matches nothing cannot report success. Any line already naming any `--mtp-` flag was left
to a human: a witness case's intent is not a text pattern. Every touched `.sh` was re-read with `bash -n`
and every touched `.py` with `py_compile`, because the one way a mechanical pin is worse than no pin is a
harness that no longer parses.

## 6. Owed, by this flip, to named owners

1. **The default's prefill price** (span 1 against span 512 at the default, one lock hold, paired). Main
   agent; this is a timing measurement and no worker may take it.
2. **`--mtp-k 3`**: implemented? no. Swept? no. Gated? no. Projected ≈ +18 %, inside `k=2`'s noise. Needs
   its own admission, including the round/step kill line at the new depth.
3. **`kArgmaxChunks`** is unswept at the shipped depth; the chunk count was picked for `k=1`.
4. **Gate (m) depth is bounded at 8** because `--mtp-accept` refuses `--replay`, so a deep round cannot
   be replayed row-by-row for a per-row verdict. Widening that is an instrument change, not a default one.
5. **Full-population lossless at the new default** (`prototype/mtp_accept/m3_loop_rate.sh`) is handed to
   the main agent: it exceeds a worker's one-minute device budget. What was taken instead, at the default
   depth: `gate.py --fixture dec8` → ENGINE GATE PASS, and section (m) at `k=2` over 18 fixtures /
   92 rounds / 188 tokens with **the same two admitted forks as the pre-flip record** (chunk64b stream
   index 0, prefix24dec8 index 3) — i.e. the default introduced no new bit divergence.
6. **`--gdn-loop` narrowing (option C)**, if the prefill price in §3 turns out to matter: funded follow-up
   with its own state-provenance admission, as §1 says. → **DONE 2026-10-25**, §7. The price did matter
   (+37 % at 1,024 lanes, +72 % at 4,032 — `engine/performance-status.md` §7), the admission was run, and it
   is clean.

## 7. Correction, 2026-10-25: the span requirement is the round's, and the whole-run pin is gone

This section overtook §1's "(B) was chosen", §2's run-wide resolution rule, §3's "that penalty is not
measured", and §6 item 6. It is appended rather than woven in because those passages are the record of a
decision taken on stated grounds, and what changed is a measurement, not a judgement.

**What was wrong.** The requirement — a round's rollback restores a mid-round checkpoint, which is a
one-token-step recurrence's end state — is about the ROUND's bodies. The enforcement was about the RUN: an
unnamed span resolved to 1 for the whole process, so the prefill wave walked one token-step per launch to
satisfy a requirement it never had. §3 guessed at the shape of that cost and correctly refused to name a
number; the main agent measured it inside one lock hold: prefill fell **+37 % at 1,024 lanes and +72 % at
4,032** (9,091 → 5,257 tok/s) against `--mtp-accept 0 --gdn-loop 512` — `engine/performance-status.md` §7.
Decode did not move, then or now: `resolve_gdn_loop(want, rows)` collapses a one-row body to 1 whatever the
flag says, so a plain step ran the per-token form at either setting.

**What runs now.** One rule, one spelling, in `engine/decode_step.h::kMtpRoundGdnSpan` (= 1, with a
`static_assert`, because a wider value would capture a state the recurrence never wrote). `submit_wave_body`
takes the span from it when `sc.gdn_ckpt` is armed and from `sc.gdn_loop` otherwise; the armed checkpoint is
the same arming `arm_mtp_round_attn_form` keys on, chosen for the reason that header states — nothing at the
launcher can tell a 3-row round from a 3-row prefill tail, but the caller can, and exactly two callers arm a
round body. The CLI-level refusal `--mtp-accept && gdn_loop != 1` is retired, not weakened: there is no
longer any span at which a round could run, so the predicate became unreachable, and a refusal no caller can
reach is an untestable branch. `--mtp-midsplit` keeps its own version of that refusal on purpose, because
that probe byte-compares a split body against an **unsplit** one, and the unsplit side arms no checkpoint and
so has no forcing to share.

The echo gained the second span, because a log that prints one span while two ran cannot be read:

```
MTP mode=accept k=2 draft=1 gdn_span_round=1 gdn_span_other=512 forced_by=default
MTP mode=off draft=0 gdn_span=512
```

`gdn_span_round` is printed from `kMtpRoundGdnSpan` itself — the same constant the launcher branches on, so
the echo cannot disagree with the launch — and `forced_by=` now names who chose `gdn_span_other`:
`--gdn-loop` when it was named, `default` when it was not. `gdn_span_round=1` is not something a caller can
ask for, and nothing prints `--mtp-accept-default` anymore.

**The admission, run on the scoped build.** §1 named the two instruments this needed, and both now run at
the spans the engine actually ships rather than at a pinned 1:

| command | verdict |
|---|---|
| `engine/gate.py --fixture dec8` (sections (a)–(m), (m) at the default `k=2`) | **ENGINE GATE PASS**, 875 PASS lines, 0 FAIL. (m): 18 fixtures / 91 rounds / 190 tokens, and the fork set is the carried pair — `chunk64b` at stream index 0, `prefix24dec8` at index 3 — so the move introduced no new bit divergence |
| `prototype/mtp_accept/m3_loop_rate.sh` (full population, lossless) | **ADMISSION: PASS** — `lines compared 49 identical 49 flipped 0`, accept arm against the flag-off greedy stream, both arms at the same prefill span. The `a`-vs-kill-line verdict is unchanged (0.6703 against 0.4568) |
| section (m)'s new span assertion, inside the run above | **PASS** — all 18 cases echoed `gdn_span_round=1 gdn_span_other=512` |
| `speccheck/run_all.py --quick`, `prototype/glue/run.py`, `prototype/proj_gemv/run.py`, `prototype/gdn_gated_norm/run.py --gate` | all PASS — untouched by this change, run as the standing regression set |

Why the lossless corpus is the load-bearing witness and the echo is not: if the forcing were missing, a
3-row round at span 3 would write **no** checkpoint slot at all — the capture sites sit in the per-token
branch — so a reject would restore memory the body never wrote and the emitted stream would diverge from the
plain one. 49/49 identical streams is therefore a claim about the launch, not about a printout. The expected
movement did not appear: the walked span is admitted at bit identity
(`prototype/gdn_rec_isa/gdn_rec_isa.hip.cpp --selftest-loop`, re-proved per fixture by gate section (j)), so
feeding a round from a span-512 prefill moves no bit — and both sides of every comparison above share that
span anyway, which is the half of the argument that does not depend on span-invariance holding.

**What moved with it.** `prototype/mtp_accept/default_refusals.sh` flipped its span rows from refusals to
positive cases asserting the echoed pair (a battery demanding a refusal the engine correctly no longer gives
is a dead gate). `engine/gate.py` section (m) stopped pinning `--gdn-loop 1` and forwards its own
`--gdn-loop` instead (default 512, the engine's default), so the section admits the run the engine ships —
the `--attn-tile` rule, moved in one commit; `--gdn-loop 1` on the gate's command line reproduces the old
record — and it now asserts the echoed span pair on every case. `prototype/mtp_accept/m3_loop_rate.py` no
longer forces `--gdn-loop 1` on its arms, for the same reason and with the same escape hatch (`GDN=1`
re-takes the carried whole-run-span figure); both arms take the value from one place, so the pair can never
differ in it.

**Still owed by this change**, and none of it is a worker's to take: the prefill A/B at the scoped default
against `--mtp-accept 0 --gdn-loop 512`, and the decode confirmation that k=2's +16.50 % survives the move.
Both are main-agent measurements (`engine/performance-status.md` §6), and the commands are handed over
verbatim.
