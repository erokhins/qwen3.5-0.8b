# HANDOFF — make the three crashing fixtures adjudicable; self-witnessing admission

Repo `/home/erokhins/erokhins_projects/qwen3.5-0.8b`, base `0966ae9`. Nothing committed. Two files
touched: `engine/gate.py` (modified, +196/−7) and `tools/admission.sh` (new, executable). No file
under `specification/`, `bench/`, `devices/` or `progress/` was touched; no pinned dump was touched.

---

## 1. The three-classification, with the evidence

### 1a. `p300` — class **(b)**, bare-named dumps, and it is the whole `--decode 0` family (p8, p64, p300)

*Manifest keys present* (all of them read by `gate.py`): `model, dtype, seed, torch, transformers,
attn_implementation, sources, prefill_ids, greedy_continuation, greedy_text, special_ids,
chat_template_render_probe, chat_template_render_think, text_config, pins`. So **not (a)** — nothing
the gate reads is missing.

*Dumps on disk*: 485 `.npy`, **zero** carrying a `decode` prefix. The bare names are `input_ids.npy`,
`logits_last.npy`, `pre_final_hidden.npy`, `final_hidden.npy`.

*The geometry*: `greedy_continuation = [27775]`, one token, so `n_steps = len(greedy) − 1 = 0`, and
`greedy[-2]` at the old line 2890 has no referent → `IndexError`. `tools/dump_fixtures.sh:113-120`
shows why: **p8, p64 and p300 are all dumped with `--decode 0`**. Verified the same shape on all three:

| fixture | len(greedy) | n_steps | decode-prefixed files | bare `input_ids.npy` vs `prefill_ids` | argmax(bare `logits_last`) vs greedy[0] |
|---|---|---|---|---|---|
| p8  | 1 | 0 | 0 | equal, all ids | 6511 == 6511 |
| p64 | 1 | 0 | 0 | equal, all ids | 2523 == 2523 |
| p300| 1 | 0 | 0 | equal, all ids | 27775 == 27775 |

So the §10.5 last-writer-wins trap **cannot spring here, and it fires inverted**: with no decode phase
there was never a second writer, so the bare `input_ids.npy` *is* the prompt — the pairing a reader
wants. The code asserted the opposite because it assumed `n_steps ≥ 1`.

`p300` is **not superseded** — §10.5 says "a gate that passes on `smoke` only is not passing:
`p64`/`p300` are required because of the chunk-64 padding in §04.7" — so it was fixed, not excluded.

A **second crash sat behind the first**, on the same family, and fixing one exposes the other:

```
IndexError: list index out of range            gate.py:2890  greedy[-2]            (observed, base gate)
AttributeError: 'NoneType' object has no attribute 'size'                           (section (a), proved by
    direct call: first_pinned_logits([]) returns None, and the base line is
    `vocab = first_pinned_logits(dec_logits).size` — base gate.py:6785)
```

The second never printed a traceback only because the first stopped the run earlier.

### 1b. `corpus48engine` — class **(a)**, and it is the whole `kind: corpus` family (corpus48, corpus48sdpa, corpus48engine)

*Directory contents*: **one file**, `manifest.json` (20,365 B). **Zero `.npy`.**

| | keys |
|---|---|
| present in its manifest | `kind, source, model, binary, reference_corpus, steps, n, start, note, prompts` |
| read by `gate.py` | `sources[0].ids` → **KeyError: 'sources'**; then `greedy_continuation`, `pins`, `text_config` — all absent |

It is not a pin fixture at all: `kind: "corpus"`, and its own manifest names
`binary: …/engine/build/decode_run` and `reference_corpus: corpus48`, i.e. it records **engine** output
(`tools/corpus_freerun.py` writes it). Its siblings `corpus48` (the HF reference corpus, read by
`speccheck/gates/run_g7.py` and `tools/corpus_margin.py`) and `corpus48sdpa` (the sdpa floor run) are
the same class — §15.2 named only `corpus48engine` because only one of the three was in its list.
Correct treatment: **named and skipped**, per `tools/reference_dump.py::CARRIED_NOT_PINNED`.

### 1c. `prefix24dec8` — **none of (a)/(b)/(c)**. It is adjudicable, and it adjudicates RED.

*Evidence it is not (a) or (b)*: 4,485 `.npy`, 4,000 decode-prefixed; `sources`, `pins` (4,485),
`greedy_continuation` (len 9 → 8 steps) all present; section (0) passes in full, including the
last-writer-wins line — `[PASS] bare input_ids.npy is decode7's input … [2526] == greedy_continuation[-2]`.

*Evidence it is not (c)*: under the exact invocation `--gemm lt --attn-tile 1` it **does** print a
verdict — `ENGINE GATE FAIL — 19 failed`, rc 1 — and no caller flag rescues it. That log was captured
before any edit to `gate.py`, so it is the base gate's reading of the base binary, and the later runs
below (taken after the edits, which touch nothing numerical) agree with it:

| run | verdict |
|---|---|
| `--gemm lt --attn-tile 1` | FAIL — 19 |
| `--gemm lt --attn-tile 1 --scope form` (§18's deferral of the wave-carry comparisons) | FAIL — 15 |
| `--gemm hand --attn-tile 1` | **PASS** |
| `--fixture dec8 --gemm lt --attn-tile 1` (control, same 8-free-step geometry, chat prompt) | **PASS** |

Why the log read as "no verdict": on a FAIL run the verdict line is **not the last line** — line 447 of
467 in the captured log, because `report()` prints the failure list and the `[expected-red]` block
*after* it. A harness that reads `tail -1` sees `[expected-red] none recorded for prefix24dec8 …` and
records "no verdict". Same class as the AGENTS row about a `RESULT` line swallowed by `tail -1`.
`tools/admission.sh` greps the marker anywhere in the log for exactly this reason.

**What the 19 are** (this is a finding, not a harness gap, and I did not touch it):
at generation step 2 the pin's margin is 0.0625 = 1.0 BF16 ULP against D-9's own bound 0.137684, so the
slack rule admits the engine's different id (12263 where the pin has 23010) — the pin could not decide
that token. The engine then free-runs from a different token and step 3 lands on a different id where
the **reference had 22.00 BF16 ULP of room**, which the room rule correctly calls RED; steps 4–7 keep the
same tokens but their logits sit at cos 0.948–0.9946 against the 0.9995 floor because the state history
now differs — precisely §10.5 rule 2 ("beyond the first flip the two sides are feeding each other
different tokens, so the tensors are not comparable"). 4 of the 19 are the §18 wave-carry class
(`--scope form` defers them); the other 15 are this divergence and its downstream rows.

---

## 2. What I changed

### `engine/gate.py`

1. **`NOT_A_PIN_FIXTURE` + `EXIT_SKIPPED = 3` + `refuse_non_pin_fixture()`** (new, ~line 207). One
   entry per carried record, each with its printed reason, following `CARRIED_NOT_PINNED`. Called from
   `set_fixture()`, the one choke point every mode passes through (`--selftest`, `--smoke`, `--dry-run`,
   device), so a corpus directory is refused the same way in all four. Three behaviours:
   * listed carried record → prints `ENGINE GATE SKIPPED — <name>: <reason>` and exits **3** (a third
     outcome, so a decline can never be tabled as a numerics result — exit 1 could not carry both);
   * carried record **not** listed → refuses, naming the fixture and telling the operator to list it;
   * listed but the manifest now carries `pins` → refuses, because a stale pardon silently skips a
     fixture the gate can judge.
2. **`selftest_non_pin_fixtures()`**, wired into `--selftest` as section **(k)**. Five cases: listed →
   skipped by name with rc 3; listed-but-pinned → refused; unlisted carried → refused with no verdict
   line; **must-not-catch** — the real pinned manifest of the fixture under test passes through with
   nothing printed and nothing raised; and the **population walk** — every manifest under `data/refs/`
   that reads as a carried record must be on the list, which is what makes the list derived rather than
   remembered.
3. **`pairing_checks`**: the last-writer-wins line is now chosen by the **geometry**, not by a fixture
   name. `n_steps == 0` asserts the fact that is true there — bare `input_ids.npy` **is** the prompt,
   full-list id-by-id equality against `prefill_ids` — and appends a summary note saying a `--decode 0`
   fixture's per-step sections judge zero steps and are not a decode claim. The `n_steps ≥ 1` branch is
   the original check, character for character.
4. **`vocab = vocab_pin`** in section (a): reuses the width `main` already derived, so a fixture with no
   per-step pin falls back to `text_config.vocab_size` instead of `None.size`.

### `tools/admission.sh` (new)

Enumerates `data/refs/*/` directories, classifies each, runs `engine/gate.py` once per fixture, prints
one row per fixture plus the assertion line. Rules: population enumerated, never recited; unknown names
abort (rc 2); **empty population aborts** (rc 2) — "an admission that runs nothing is not a pass; it is
a harness that cannot fail"; every row is `PASS` / `FAIL` / `SKIPPED` / `NO-VERDICT` / `NO-MANIFEST`; a
directory with no manifest must be named in `NO_MANIFEST_DIRS` with a reason or the run aborts; exit
0 clean / 1 red / 2 harness-abort / 3 no-verdict. It reads the verdict **markers anywhere in the log**,
never the tail, and it takes the GPU lock for the run (gate.py does not self-lock — `grep -c gpu_lock
engine/gate.py` is 0) but skips the lock when the flags ask for a no-device run. It prints no `ms`.

---

## 3. Proof that the fixes bite

| control | result |
|---|---|
| `--fixture corpus48sdpa`, entry listed | `ENGINE GATE SKIPPED — corpus48sdpa …` rc 3, row `SKIPPED` |
| **entry removed** from `NOT_A_PIN_FIXTURE` | gate refuses rc 1 ("it is not on …NOT_A_PIN_FIXTURE"); admission row `NO-VERDICT`, `ADMISSION FAIL … rc=3` — never a silent pass. And `--selftest` itself goes red: `SELFTEST FAIL — carried manifests not on NOT_A_PIN_FIXTURE: corpus48sdpa` (section (k)'s population walk). Entry restored, selftest green again. |
| a real pinned fixture (`p64`) added to the list | refused: "it has become a pinned fixture … leaving it here would skip a fixture this gate can judge" |
| must-not-catch for the new pairing check | a scratch fixture shaped like `p300` whose `prefill_ids` differs from the pinned `input_ids.npy` by **one id** (built in a throwaway dir under `data/refs/`, deleted after; pinned dumps untouched) → `[FAIL] bare input_ids.npy IS the prompt …`, `ENGINE GATE FAIL — 1 failed` |
| misspelled names in the admission | `ADMISSION ABORT — unknown fixture name(s): wide3100f prefill_t256f` + the real population printed, rc 2 (the exact §15.2 names) |
| emptied population | `ADMISSION_REFS=/tmp/empty_refs …` → `ADMISSION ABORT — enumerated 0 fixtures …`, rc 2, printed **before** any header, so a table of nothing cannot assert `ran == enumerated` |
| forced NO-VERDICT (`--work /dev/null/nope`) | rows `NO-VERDICT rc=1 and no verdict line … NOT an engine result`, `ADMISSION FAIL`, rc 3 |
| gate's own regression suite | `engine/gate.py --selftest` → `SELFTEST PASS`, 0 `[FAIL]` lines, after all edits |

---

## 4. Commands run (correctness only — **no timing taken, no `ms` figure reported**)

All device runs were one at a time, blocking, each well under a minute; `util` is recorded as required
and is not usable as a gate on this box (the serving session holds it high). Device logs kept in
`tmp/admission_logs/`. The two rows that die inside `gate.py` before any binary starts (the base-gate
`p300` IndexError, and the corpus-record skips) touch no device at all and were run without the lock;
everything that started `decode_run` held it (`admission.sh` takes it itself, tag `admission lt`).

| command | rc | verdict | util at acquire |
|---|---|---|---|
| `engine/gate.py --fixture p300 --gemm lt --attn-tile 1` (base gate) | 1 | IndexError, gate.py:2890 | 91 % |
| `… --fixture prefix24dec8 --gemm lt --attn-tile 1` | 1 | FAIL 19 | 90 % |
| `… --fixture prefix24dec8 --gemm lt --attn-tile 1 --scope form` | 1 | FAIL 15 | 93 % |
| `… --fixture prefix24dec8 --gemm hand --attn-tile 1` | 0 | PASS | 93 % |
| `… --fixture dec8 --gemm lt --attn-tile 1` | 0 | PASS | 93 % |
| `tools/admission.sh lt --attn-tile 1 --fixture p8 --fixture p64 --fixture p300 --fixture dec8 --fixture corpus48engine` | 0 | 4 PASS, 1 SKIPPED, 0 NO-VERDICT; `ran=5 enumerated=5` (wall 27 s) | 91 % |
| `tools/admission.sh lt --no-engine --attn-tile 1` (full population, **CPU only, no device, no lock**) | 1 | 15 PASS, 1 FAIL (`zz`), 3 SKIPPED, 0 NO-VERDICT; `ran=19 enumerated=19` | n/a |

The `--no-engine` row is an **enumeration and classification** check, not a numerics admission: that flag
starts no binary. It is how the whole 19-directory population was walked without spending a device
window.

---

## 5. What I refused to do, and why

1. **Did not re-dump or invent any pinned dump.** None was needed: the `p300` family's dumps are
   correct, the gate's assumption about them was not. If a re-dump were wanted (e.g. `p300` with a
   decode step), that is a reference run and therefore the main agent's call — say the word.
2. **Did not widen, narrow or delete any band or check.** `prefix24dec8` stays red. The tempting
   "fix" — stop judging steps after an admitted flip, as speccheck's `run_gates.py` does per §10.5
   rule 2 — would delete comparisons in order to make a red fixture green. That is a policy decision
   for you, not a harness repair, so it is reported here and shipped nowhere.
3. **Did not exclude `p300`.** §10.5 requires `p64`/`p300` in the gate set; an exclusion would have
   satisfied the symptom and broken the spec. The list is only for genuinely non-pin artifacts.
4. **Did not run the full 19-fixture device admission** — it cannot fit a worker's ~60 s device budget.
   Command line in §6.
5. **Did not silence `zz`.** See §7.
6. Did not commit; did not touch `specification/`, `bench/`, `devices/`, `progress/`.

---

## 6. The command to run for a full admission

```bash
cd /home/erokhins/erokhins_projects/qwen3.5-0.8b
tools/gpu_lock.sh status                    # confirm nothing else holds it (STALE is reclaimable)
tools/admission.sh lt --attn-tile 1         # the shipping route, §17's invocation style
tools/admission.sh hand --attn-tile 1       # the other route, same population
```

Each takes the lock itself, runs all 19 manifests, and ends with
`ADMISSION ran=19 enumerated=19 pass=… fail=… skipped-not-a-pin=3 no-verdict=0` plus
`ADMISSION CLEAN` (exit 0) or a named reason for a non-zero exit: 1 red, 2 harness-abort, 3 NO-VERDICT.
Per-fixture logs land in `tmp/admission_logs/<fixture>.log`. Do not wrap a self-locking driver in it.

---

## 7. Loose ends the main agent should decide on

1. **`prefix24dec8` is red on `--gemm lt` and green on `--gemm hand`**, at base, and it has never been
   in `EXPECTED_RED` (which is empty since P7), so the gate prints "all 19 failure(s) are new by
   construction". Per the fix-first rule this outranks pending speed work until you either call it a
   defect to fix or record it. The precise open question is whether an id flip that D-9's slack rule
   admits (step 2, pin margin 1.0 ULP vs bound 0.137684) should then make the remaining free-run steps
   unjudgeable, as §10.5 rule 2 says for `run_gates.py`, or keep being compared as they are now.
2. **A vacuity the gate still prints, disclosed not removed.** On a `--decode 0` fixture the per-step
   aggregates in (a) and the whole of (b) run over an empty list and print `[PASS] all 0 generated ids
   match …`. Each line does name its denominator, and section (0) now appends a summary note saying
   these judge nothing, so a PASS on `p8`/`p64`/`p300` cannot be misread as a decode claim. Making the
   lines themselves print `[VACUOUS]` instead of `[PASS]` would need a new outcome in `check()` and a
   decision about what a vacuous aggregate means for the exit code — a gate-policy call, so it is left
   for you rather than invented here.
3. **`zz` is red, and was before this change**: its manifest declares `logits_declared_absent =
   ['decode2','decode3']` while `decode2.logits_last.npy` and `decode3.logits_last.npy` are both on
   disk, and the gate's "declaration and disk agree in both directions" check fires. Proved
   pre-existing by running the base `gate.py` (from `git show HEAD:engine/gate.py`) on it: identical
   `ENGINE GATE FAIL — 1 failed` on base and on the new gate. Either re-dump `zz` per its declaration
   or fix its manifest; until then every admission ends `ADMISSION RED — 1 fixture(s)`.
4. A **fourth latent crash** was found behind the three (§1a's `AttributeError`), and the crash class is
   **three fixtures, not one** (p8/p64/p300) and the schema class **three, not one** (corpus48/…sdpa/…engine).
   Worth a line in the ledger when you write this up.
5. If a future dump adds a decode step to the `p` family, the `n_steps == 0` branch retires itself
   automatically — it keys on the geometry, not the names.
