# HANDOFF — `engine/gate.py`: teacher-forced adjudication + ECHO attribution

Status: **code complete and verified; not committed.** Wind-down requested mid-review; nothing was
left half-edited — the working tree parses, `--selftest` is green against the exact bytes on disk, and
the diff below is that same file (`diff -q` of the saved diff against the live `git diff` prints
`DIFF UNCHANGED SINCE SELFTEST`).

Base `99fa8a7`. One file touched: `engine/gate.py` (+715 / −60, now 8191 lines). No pinned dump
touched, no band value changed, no `specification/`/`bench/`/`devices/`/`progress/` edit, no commit,
nothing staged.

---

## 1. What the change is

### 1a. A third verdict bucket — ECHO (machinery `gate.py:1178–1229`, classifier inside `check()` at `gate.py:1230–1258`, alongside `defer()` at `gate.py:1148`)

`echo()` is the sibling of `defer()`, same context-manager shape on purpose so the attribution cannot
be lost at a call site. A check inside an open echo block still runs, still prints, still lands in a
sidecar; only the bucket changes. `check()` classifies in this precedence:

```python
tag = (("[ECHO-PASS] " if ok else "[ECHO-FAIL] ") if ECHO_REASON is not None else
       ("[DEFERRED-PASS] " if ok else "[DEFERRED-FAIL] ") if DEFER_REASON is not None
       else ("[PASS] " if ok else "[FAIL] "))
```

Failed echoes go to `echoes`, never to `fails`/`fail_names`; every echoed check (pass or fail) goes to
`echo_records`; each echoed failure prints `           [ECHO CAUSE] …` under itself.

`echo_first_step(admitted)` (pure) returns `min(admitted) + 1` — the flip's **own** step is still
judged normally, because up to and including it the run was fed the pin's own tokens — and `None` for
an empty list, which is the state where every band failure must stay red. `PREFILL_ROW` (−1, §06.5's
y0) counts as a step so a prefill-row divergence opens the window at step 0.

`admitted_divergences()` (`gate.py:7851`) is a pure pre-pass that runs the **same three predicates**
(tie → slack → room, same precedence) over the whole run before any of their lines print, because the
last of the three can admit the flip that everything downstream must be attributed to. The per-line
admissions computed later are asserted equal to the map, so the two cannot drift:

`[PASS] the echo window was attributed to exactly the steps the three rules admitted on their own lines`

`echo_from = None if replay else echo_first_step(admitted)` — a `--replay` run is fed the
reference's own token at every step, so an admitted flip cannot fork anything and every downstream
failure is real. Echo sites: `gate.py:7341, 7368, 7394, 7409, 7436, 7456, 7550, 7556, 7563, 7569` —
the per-step tie/slack/room lines, the per-step id lines, the aggregate "all N generated ids match"
line (judged on the non-echoed remainder), the fed-chain line, and (b)'s cos/meanrel/argmax lines.
The informational metrics line is never echoed: a printed number must not change colour.

Sidecar `echo_attribution.json` (beside `wave_carry_deferred.json`, same lifecycle — unlinked with the
stale dumps) carries `admitted_step`, `admitted_as`, `engine_token`, `pin_token`, `admitted_steps`,
`echo_from`, `reason`, `n_checks`, `n_failed`, and the per-check records. Verdict lines:

```
ENGINE GATE PASS — with 1 step(s) reported SLACK and 14 check(s) ECHO of an admitted flip (23 judged
downstream of it; see the [ECHO] lines and section (b2))
ENGINE GATE FAIL — 2 failed(1 step(s) admitted by the slack rule); 3 more ECHO of an admitted flip
and not counted
```

### 1b. Section `(b2)` — the same steps at identical conditioning (`teacher_forced_checks`, `gate.py:7886`, called at `7614`)

For a fixture with a decode phase, the engine runs a **second** time with
`--replay <greedy[:-1]>`, built from the pinned fed sequence (`greedy[:-1]` — §10.5's rule 2 and
§07.4's `input_token_id = previous_token_id`; the already-passing section (0) proves
`dec_inputs[j][0] == greedy[j]`, so this is the reference's own chain, not a reconstruction). Judged
per step: token equality, the same three admissions, and `cos deficit` / `meanrel` against the **same**
`rule_for_step` caps. **No band is wider anywhere, and there is no echo window anywhere** — that is the
section's whole point. Its premise is asserted, not assumed: `rc == 0`, `RESULT steps=` == parsed
STEP-line count == prompt + decode count, `mode == replay`, and every fed token == `greedy[:-1]`; on a
premise failure it returns and says so in the notes rather than judging a partial replay. It prints an
info line and does nothing on a fixture whose main run was already a replay, or with no decode phase.

### 1c. Sections (h) and (j) are teacher-forced whenever the fixture has a decode phase (`gate.py:7692`)

`ab_replay = replay or n_decode >= 1`, at the `run_batched_ab` and `run_gdn_spell_ab` call sites.
Their independent variable is a wave width or a GDN spelling; on a free-running fixture the two sides
could each take their own thin-margin token and fork, after which a cross-width line compares two
different questions. This is what fixed p24's four (h) failures honestly rather than echoing them:
pooled meanrel went 4.569e-02 → **9.404e-03** (band 3e-02) and the per-row line came inside 2.75e-03.
Fixtures with `n_decode == 0` (the `wide*f` prefill probes, `p8`/`p64`/`p300`) are unaffected; fixtures
already replayed (`n_decode > 8`) are unaffected.

### 1d. Stand-in accuracy fix (`dry_run`, `gate.py:5179`; the line fixed is `gate.py:5403`)

`RESULT … steps=25` was hardcoded; it is now `len(prompt_ids) + len(dec_logits)`. Without this the new
(b2) count assertion could not see a wrong step count in the one place it is testable on CPU.

### 1e. `--selftest` — section `(b2)` of its own: the echo window's teeth (`echo_injection_target` 5609, `selftest_echo_window` 5665, wired at 6356)

Classifier cases (empty → None, `[PREFILL_ROW]` → 0, `[5,2]`/`[0]` → 3/1); a clean run must produce no
`[ECHO-` line and no sidecar; MUST-NOT-CATCH (slack lift at the derived host step → rc 0, sidecar
`n_failed == 0`); MUST-CATCH (same lift + `--break-runnerup` at the derived late step → rc 1 with the
exact `[ECHO-FAIL]`, `[ECHO]`, **and** `[FAIL] b2 step …` lines); sidecar↔printed count equality in
both directions; and the "nothing admitted upstream" half (rc 1, **zero** ECHO lines, ordinary
`[FAIL] step …`). The carrier and both steps are **derived from geometry** — a walk of every
`data/refs/*/manifest.json` using `slack_region_steps`/`step_slack`, i.e. the same caps the gate
polices — with carried records refused before `set_fixture` (the `kind`/`pins` discriminator behind
exit 3), a census naming every candidate rejected and why, and "no carrier exists" reported as a hole
in the corpus, not a skip. The derived target today is `prefix24dec8`, host 2, late 7.

---

## 2. The four controls — what actually ran

All four ran. Device runs were under `tools/gpu_lock.sh` (tag `p24gate-echo-b2-cpu-verified`, released,
`status` now `free`), correctness-only, blocking, one at a time, **no timing taken and no ms figure
below**; wall seconds are host wall for the gate command, not a kernel timing. `util` was not recorded
per command; the box was serving this session's tokens throughout, which is admissible only because
none of these are timing measurements.

| control | command | result |
|---|---|---|
| **must-not-catch, device, must be green and say why** | `.venv/bin/python engine/gate.py --fixture prefix24dec8 --gemm lt --attn-tile 1` | **RC 0, `ENGINE GATE PASS — with 1 step(s) reported SLACK and 14 check(s) ECHO of an admitted flip (23 judged downstream of it …)`**. Was 19 failures at base. Sidecar: `admitted_step 2`, `admitted_as quantization-slack rule`, `engine_token 12263`, `pin_token 23010`, `echo_from 3`, `n_checks 23`, `n_failed 14`. The one line the flip itself caused is `[PASS] step 2: … quantization-slack rule …`; the room-rule RED at step 3 is `[ECHO-FAIL]` with the cause printed; (b2) is fully green. |
| **`--break-runnerup` still red** | `.venv/bin/python engine/gate.py --dry-run --fixture prefix24dec8 --break-runnerup 7` | **RC 1, 6 failures**, none echoed (0 `[ECHO-FAIL]` lines), including the two new ones: `[FAIL] generation step 7 …` and `[FAIL] b2 step 7 …`. CPU stand-in only — see §3. |
| **late-step injection: caught by (b2), not hidden by the echo allowance** | `.venv/bin/python engine/gate.py --dry-run --fixture prefix24dec8 --slack-alt 2 --break-runnerup 7` | **RC 1 — `2 failed(1 step(s) admitted by the slack rule); 3 more ECHO of an admitted flip and not counted`**. Exactly 3 `[ECHO-FAIL]` free-run lines at step 7 (3 printed == 3 sidecar records, `echo_from 3`), and **exactly 2 real `[FAIL]`s, both `b2 step 7 …`** (`… cascade room rule at identical conditioning`, `b2 step 7 samples greedy_continuation[8] with the reference's own token fed in`). The defect is visible only because of (b2); the echo allowance did not swallow it. Also enforced inside `--selftest` (derived target). |
| **`--selftest` green** | `.venv/bin/python engine/gate.py --selftest` | **RC 0, `SELFTEST PASS`, wall 11 s, CPU only** — includes the new (b2) echo cases above. |

Extra regressions run with the final bytes, same lock hold:

- `--fixture prefix24dec8 --gemm hand --attn-tile 1` → **PASS**, 0 ECHO lines (nothing is admitted at
  the bit-exact route, which is the right answer). wall 5 s.
- `--fixture dec8 --gemm lt --attn-tile 1` → **PASS**, 0 ECHO lines; (b2) runs and is green. wall 7 s.
- `--fixture dec8 --gemm hand --attn-tile 1` → **PASS**. wall 6 s.
- `--fixture prefix24dec8 --gemm lt --attn-tile 1 --scope form` → **PASS**, 12 `[DEFERRED-*]` lines and
  the same 14 ECHO lines — the two attributions coexist. wall 7 s.
- CPU dry-runs still PASS: `--dry-run --fixture prefix24dec8`, `--dry-run --fixture prefix24dec8
  --slack-alt 2` (→ `PASS — with 1 step(s) reported SLACK`), `dec8`, `dec64f`, `p8`.

Device (b2) evidence, `--gemm lt`: `mode=replay`, `steps=32` parsed == 32 printed, all 8 fed tokens ==
`greedy[:-1]`; worst cos deficit **1.2427e-04** against the 2.0000e-03 floor in force, worst meanrel
**1.6730 %** against 5 %; every pinned top-1 reproduced except step 2, which is the admitted slack step
and is named as such at identical conditioning (`b2 step 2 … 12263 vs 23010 — and nothing upstream
admits it, because at identical conditioning there is no upstream flip to blame`).

---

## 3. Named gaps — read these before trusting the word "done"

1. **The late-step must-catch is proven on the CPU stand-in only, and can be proven nowhere else in
   this tree.** `--break-runnerup` is a `dry_run` flag; the device binary has no numeric-injection
   flag, so there is no device command that injects a late defect. What the device run proves is that
   (b2) runs on real engine dumps, asserts its premise, and passes at real conditioning; what it cannot
   prove is the injected-defect direction. Do not describe that control as device-proven.
2. **The `--selftest` census over-claims its own coverage.** `echo_injection_target` returns at the
   first fixture that can carry the case and names only the candidates priced before it (13 today:
   `chunk64b, corpus48, corpus48engine, corpus48sdpa, dec300f, dec64f, dec64t, dec8, mt32f, p128,
   p300, p64, p8`); the population is 19 manifests, so `corpus48engine-smoke, smoke, wide1024f,
   wide520f, wide640f, zz` are never priced and the info line does not say so. I drafted the fix (an
   `unreached` field printed as "stopped at the first carrier and never priced: …") but the edit did not
   apply and I did not re-apply it under the wind-down. Cosmetic, in the selftest's info line only —
   it does not weaken any assertion — but it is exactly the "0 of the readable 55" shape and should land
   before the census is quoted anywhere.
3. **`p128` is the one affected fixture not re-run on device.** The (h)/(j) change touches every
   free-running fixture with ≥1 decode step. That population is `{dec8, prefix24dec8, p128}` (per the
   derived walk's census: `p8`/`p64`/`p300` have 0 decode steps, everything with >8 is replayed,
   `wide*f` carry no `greedy_continuation`). dec8 and prefix24dec8 are device-green; **`p128` was not
   run**. It is a ~7 s gate.
4. **What the check can no longer catch, in one sentence:** a defect that perturbs only the running
   continuation of the engine's own already-admitted different token — a state-carry fault that fires
   only once the engine is fed its own flip — is reported `ECHO` in (a)/(b) and is invisible in (b2)
   too, because (b2) never runs that chain and no pin describes it (§10.5 rule 2: there is no reference
   for a prompt nobody fed). It is still printed, labelled `ECHO`, with the blamed flip and both tokens
   in the line and in `echo_attribution.json`; it is visible, not adjudicable. This is written into the
   `teacher_forced_checks` docstring as well.
5. **The (h)/(j) teacher-forcing is a scope change I made, and it costs something.** For a free-running
   fixture with a decode phase, the cross-width and cross-spelling comparisons are now made only at
   teacher-forced conditioning; the free-running chain is no longer compared across widths. The argument
   in the code comment is that a forked pair proves nothing about width, and a width that changes a
   decision still shows as a RED flip at that row — but if the operator wants the free-running cross-
   width comparison kept as well as the teacher-forced one, that is a second pair of runs, not a revert.
6. `speccheck/run_all.py` was not run: nothing under `speccheck/` is an input to, or an input *of*,
   `engine/gate.py`, and `AGENTS.md`'s budget rule says cite the existing green rather than re-make it.

---

## 4. Exact commands for what remains

```bash
# 0. the diff, and the byte-identity claim in §"Status"
git status --short                       # expect: M engine/gate.py  (nothing staged)
git diff engine/gate.py                  # +715 / -60; also saved at /tmp/p24gate.diff

# 1. the one affected fixture never run (§3.3) — release the lock afterwards
bash tools/gpu_lock.sh acquire "p24gate-p128-check"; tools/gpu_lock.sh util
.venv/bin/python engine/gate.py --fixture p128 --gemm lt --attn-tile 1; echo RC=$?
bash tools/gpu_lock.sh release "p24gate-p128-check"

# 2. the usual admitted matrix, if the operator wants it re-taken with this gate (main agent only)
for f in dec8 prefix24dec8 p128 chunk64b dec64f dec64t dec300f mt32f wide520f wide640f wide1024f p8 p64 p300; do
  .venv/bin/python engine/gate.py --fixture $f --gemm lt --attn-tile 1 | tail -1
done

# 3. CPU controls (seconds each; no lock needed)
.venv/bin/python engine/gate.py --selftest
.venv/bin/python engine/gate.py --dry-run --fixture prefix24dec8                                    # PASS
.venv/bin/python engine/gate.py --dry-run --fixture prefix24dec8 --slack-alt 2                       # PASS + SLACK
.venv/bin/python engine/gate.py --dry-run --fixture prefix24dec8 --break-runnerup 7                  # FAIL, 6, 0 echoes
.venv/bin/python engine/gate.py --dry-run --fixture prefix24dec8 --slack-alt 2 --break-runnerup 7     # FAIL, 2 + 3 ECHO
```

Commit hygiene suggestion (three topics, per `AGENTS.md`): (i) the ECHO bucket + sidecar + (a)/(b)
wiring + the (h)/(j) teacher-forcing, (ii) section (b2) with the derived-population selftest cases,
(iii) the `dry_run` step-count fix. They are separable in the diff; (i) and (ii) are only jointly
useful, so two commits (`dry_run` fix first, then the rest) is also defensible.
