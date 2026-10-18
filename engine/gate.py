#!/usr/bin/env python3
"""Engine gate: does a greedy token come out right?

    .venv/bin/python engine/gate.py                 # dec8; runs engine/build/decode_run
    .venv/bin/python engine/gate.py --fixture dec64f  # the 64-step FORCED pin (P1)
    .venv/bin/python engine/gate.py --smoke 300     # free run, 300 tokens, smoke only
    .venv/bin/python engine/gate.py --no-engine     # fixture pairing + state sizes, zero device
    .venv/bin/python engine/gate.py --selftest      # proves THIS gate discriminates: no engine, no device
    .venv/bin/python engine/gate.py --slack-region   # which steps this fixture's pin could not decide
    .venv/bin/python engine/gate.py --dry-run       # every section against a CPU stand-in
    .venv/bin/python engine/gate.py --build         # run engine/build.sh first
    .venv/bin/python engine/gate.py --binary P --model DIR --work DIR

Two fixtures, because they prove different things and only one of them can be free-running:

  dec8     8 steps, the engine feeding ITSELF. Below A-18's chaos bound a correct engine produces
           the reference's tokens by itself, so the chain can be re-derived rather than replayed.
  dec64f   64 steps, REPLAYED: `decode_run --replay` feeds the REFERENCE's token at every step and
           the engine's own sample is compared, not consumed. That is teacher forcing, and it is the
           only way depth can be pinned at all: A-18 measures two legal bf16 continuations of the
           same prompt agreeing only 0.8776 of the way over 49 prompts x 24 tokens, so a 64-token
           FREE-run id match fails by construction for a correct engine. What depth actually tests
           is state carry-over — GDN state, conv window, KV — 64 deep instead of 8.

`--smoke N` is neither: a free run of N tokens that must not crash, must not emit a NaN, must stay
inside its state, and must end by length or by the stop id. It is not a parity claim and it is not a
timing; parity is claimed only over the forced pins.

`--selftest` re-runs this file in subprocesses against injected bugs — a wrong sampled id, a swapped
top-two logits, 5 % logits noise that leaves the argmax alone, a §07.5 stop check firing during the
prompt phase, a replay step fed the wrong token, and — at a step the pin ties — the engine landing on
an untied id (must FAIL), on the pin's other tied id (must PASS as TIE), and on the pin's own top-1
with its logit 1 BF16 ULP off the tied payload (must PASS, because a step where engine and pin agree
is an ordinary match: D-1) — plus one quantization-slack perturbation aimed three ways: at a step the
pin could not decide, where lifting the pin's runner-up one BF16 ULP over the pinned maximum must PASS
as SLACK, at the nearest pinned id outside that step's margin (must FAIL), and at the same lift on the
nearest step the pin DID decide (must FAIL, because that is the negative that catches a slack bound
sized too generously). It requires each to fail at the step it was injected at, because a negative that
every check would catch proves nothing (`prototype/mlp_gemv/results.md`). It is the only part of a gate
that can be validated before the engine has ever run, and it is why a first device run need not double as
a test of this file.

Sections, in the order the task fixed them plus the ones the deep fixture needed:

  (pre) BAND HISTORY — every number in `BAND`, and every priced fixture's route margin, against
      `bench/band-history.jsonl`. An unrecorded move and this gate refuses to run at all
      (`tools/band_history.py`); it runs before any fixture read or device work, and it is the mechanical
      half of `AGENTS.md`'s "never loosen a gate to make it pass".

  (0) FIXTURE PAIRING, proved rather than assumed (§10.5). See `pairing_checks`. A step whose logits
      the fixture DECLARED absent (`reference_dump --logit-steps`) is skipped for the logits half while
      its feed line is still checked, and the section prints `paired N of M steps, K declared gaps` so
      that a fixture which quietly lost its pairing shows up in the output rather than only in the
      check total. Logits that are merely missing — not declared — still exit 1, and a gapped fixture is
      refused an engine run: sections (a)/(b)/(e) judge every pinned step, and judging fewer of them is
      not a thing this gate does silently. `--selftest` injects each of those four cases.
  (a) SAMPLED IDS — the engine's sample after the prompt's last token plus its generation samples
      must equal the manifest's `greedy_continuation`, exactly. §10.1 G7 requires token-for-token
      exactness, and a token id is an integer: there is no band to argue about. There are TWO named,
      proved exceptions, each printed with every number it used and counted in the summary:
        * TIE — the pin itself is an exact tie AND the engine picked one of the other tied ids, which
          leaves an argmax with no defined winner to agree with (TIE POLICY, next to the band
          rationale). A step where the engine reproduced the pin's own id is never one of its uses.
        * SLACK — the pin has a unique winner, but its margin is smaller than the element error that
          step's own meanrel band licenses, so the pinned token is one the reference never decided
          (QUANTIZATION SLACK, below the tie policy; sized by `engine/slack_size.py`, and the only
          step in the corpus that uses it today is `chunk64b` step 0). Same scope rule as the tie
          policy: an engine that reproduced the pin's id consults neither.
  (b) THE LOGITS — each step's logits against that step's pinned `logits_last.npy`, inside the band
      rule printed for that step: the measured ROUTE ALLOWANCE where `engine/route_allowance.py` has
      priced this fixture, the flat published §10 floors where it has not. The rule in force is named on
      every step line, and the derivation of its margins is in the ROUTE ALLOWANCE block below.
  (b2) THE SAME STEPS AT IDENTICAL CONDITIONING — the fixture re-run with `decode_run --replay` fed the
      reference's own token at every step, so the engine's sample is never fed back and every row is
      comparable with the pin that was computed on that same token (§10.5's alignment rule). NO BAND IS
      WIDER HERE: same `rule_for_step` caps, same G5 §10.1 floor, same §10.2 ceiling, same flip line,
      same three admission rules — only what the engine was fed changes, and that is asserted from the
      run's own STEP lines rather than assumed from the flag. This section is what makes the ECHO
      attribution below safe to have: a defect the echo window legitimately stops counting in (a)/(b) is
      judged here with no echo window at all. It costs one extra engine run on a free-running fixture;
      on a replayed one (a)/(b) were already teacher-forced and it says so instead of running twice, and
      a `--decode 0` fixture has no self-fed chain to re-feed.
  (c) STATE SIZES — the bytes the engine reports for its request state must equal what
      `engine/state_sizes.py` derives from `state.h`; that file is *run*, not re-typed.
  (d) LAUNCH PLAN — one token's launch bill, counted from §07 in Python and compared with the engine's.
  (e) THE CURVE — the per-step logits and state-delta table across the whole forced run, with a drift
      reading, because the point of depth is whether anything degrades with step count. It reports;
      it does not invent a new band, and the reason is in the BAND comment below.
  (f) FIXTURE INTEGRITY of the state deltas — the pinned fingerprints must be finite and the pinned KV
      delta must be exactly 0 (the cache is append-only). A check on the dump, not the engine: it is
      what caught a misaligned delta in `tools/reference_dump.py`'s first version of this code.
  (g) G5-STATE — the engine's own §08 state, D2H'd at the prompt end and the run end by
      `decode_run --state-fp`, against those same pinned fingerprints. What is gated is what §08.6
      makes exact — append-only Δ² of 0.0, one lane per token, a state that moved, a window that
      rolled, no family left at the §08.4 zero — plus every row's geometry and the shared helper
      proved against the pins themselves. The engine-vs-pin DISTANCES are printed and not gated: plan
      `progress/2026-10-12/plan-prefill-2026-10-12.md` §2.6 ships this as "a measurement and a tripwire, not a
      band", and the band is a main-agent decision with a `D-x` and a ledger line behind it.

  (h) THE BATCHED PATH (P4b) — the same prompt run twice, once at the wave scratch's full width and
      once with `decode_run --wave-rows 1`, judged two ways: the launchers' own counters (P4's
      batched glue/MLP forms must be the ones submitted, over the same total rows, in materially fewer
      launches) and the two runs' dumps (identical sampled ids, `cmp`-identical logits). The second
      half is what turns "P4's deficits were bit-identical" into "batching cost zero numerics"; the
      first half is what makes the second half mean anything, because a wave that never batched would
      print bit-identical outputs too. `--selftest` injects each half alone (`--break-wave`,
      `--break-ab-logits`) and requires the other to stay green.

  (i) THE PREFILL ROWS — every pinned row of the prompt wave against the engine's own per-row dump
      (`logits<i>.bin` ↔ row *i*, §06.5's pairing), judged by the floors section (b) already uses and by
      the same flip rule. Only `wide640f`/`wide1024f` carry the row block
      (`reference_dump.py --prefill-logits all`), and they are this tree's only ground truth above MAXSC
      lanes — which is exactly where §15's form dispute sits. COVERAGE is proved first and is not a
      numerics question: the manifest's `prefill_logit_rows` decides the row shape `cmd` is built with,
      the run's own `PREFILL logits=` echo is then checked against that ask, and a gap stops the gate
      before (a)/(b)/(g)/(h). A missing row dump has to read as a missing row dump — never as a floor
      verdict over whatever survived, which is what §17's shipped run did (`1023 missing` next to a green
      `every judged row is inside the floors` over the one row that existed).

SCOPE — `--scope full` (the default) counts every check that fails. `--scope form` is the attention-FORM
A/B of §16/§17: section (h)'s wave-width OUTPUT comparisons still run and still print, as
`[DEFERRED-PASS]`/`[DEFERRED-FAIL]`, and every one of them is recorded in `wave_carry_deferred.json`
beside the dumps, but they are not counted into that run's verdict, because their independent variable
is `--wave-rows` while the form is held identical on every side. Nothing is skipped or softened: the
summary line says how many were deferred, and `--scope full` counts them again.

THE THIRD BUCKET — ECHO. A FREE-RUNNING run that takes an admitted thin-margin token at step N (by the
tie policy, the quantization-slack rule, or the cascade room rule) is, from step N+1 on, answering a
different question than the pin, because the pin at step N+k was computed on the reference's own chain
and this run was computed on its own. `prefix24dec8 --gemm lt --attn-tile 1` printed 19 red lines for
ONE such flip at step 2, and 15 of them were that flip's downstream, not 15 defects. So a check on a
step downstream of an admitted flip prints as `[ECHO-PASS]`/`[ECHO-FAIL]`, is recorded with the flip,
the two tokens and the admitting rule in `echo_attribution.json`, is counted in the verdict line, and is
NOT counted as an independent failure. Nothing is skipped: the check runs and prints, a step with nothing
admitted upstream stays `[FAIL]` exactly as before, an admitted flip in a REPLAY opens no window at all
(there is no fork to attribute anything to), and `--selftest` proves both directions with the must-catch
and must-not-catch steps derived from the corpus's geometry. What this can no longer catch is stated in
`teacher_forced_checks`: a defect that perturbs only the continuation of the engine's own different token
and leaves the teacher-forced chain untouched.

The two phases, and the line protocol that shows them. Since P2 the prompt is fed as ONE prefill wave
(`engine/prefill.h`, §06.1) and its STEP lines say `kind=prefill`; generation steps say `kind=decode`.
§07.5's stop check is armed in the second phase only, and this gate asserts both halves of that rule:
no prefill row may carry `stop=1`, and the wave's LAST row must carry the fixture's first answer token
(`greedy_continuation[0]`, §06.5's `y0`). Under `--prefill-logits last` — §06.5's shape, and the default
— the wave argmaxes only that last row, so the earlier prefill rows print `sampled=-1`; `all` prices
every row and is the shape the wave's byte-identity comparison against the pre-P2 per-token prompt loop
is made in (`progress/2026-10-12/plan-prefill-2026-10-12.md` §4, row P2).

Why this file contains no engine arithmetic. Its value is independence. A Python mirror of the
kernels would check the engine against the engine, which is the failure `bench/gate-audit.md` exists
to find. Everything Python-side here is either a §10 metric, the §00/§08 state derivation, or the
§03.4 argmax used to prove the fixture pairs with itself. Nothing here recomputes a model output.

CPU-only in `--no-engine` mode; the default mode runs the device binary as a subprocess and reads its
dumps. It does not import torch (the reference's own dumps are read with numpy), so it runs under the
same `.venv` the rest of the gates use. The one exception is `(g)`'s fingerprint-identity case, which
lazy-imports torch because "recompute the pinned fingerprint with the dumper's own function" is the
only way to prove the helper still matches the pins; nothing else in this file touches torch, and
`--no-engine` on a fixture with no full state tensors never imports it.
"""
from __future__ import annotations

import argparse
import contextlib
import hashlib
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from tools import band_history    # the band ledger this gate refuses to run against otherwise

REFS = ROOT / "data" / "refs" / "dec8"   # replaced by main() from --fixture; see `set_fixture`
SHALLOW_REFS = ROOT / "data" / "refs" / "dec8"   # the cross-check target for a deep fixture
DEFAULT_BINARY = ROOT / "engine" / "build" / "decode_run"
WORK = ROOT / "tmp" / "engine_gate"

# §07.5 / A-10: the id greedy decoding stops on — the end-of-turn token. Spelled as an id, never as
# template text, and it is the same constant as `engine::kStopToken`. The `dec8` prompt is a rendered
# chat template, so this id occurs INSIDE the prompt (index 8) as well as in a continuation.
STOP_ID = 248046

# A-18's measured bound, used as a policy line and not as a guess: below this many steps the engine
# can be gated feeding ITSELF (dec8 does, and that is the stronger property — the chain is
# re-derived, not replayed); above it a pin has to be teacher-forced, because two legal bf16
# continuations of one prompt agree only 0.8776 of the way over 49 prompts x 24 tokens. So
# `n_decode > FREE_RUN_BOUND` is exactly when this gate switches to `decode_run --replay`, and a
# divergence then shows up at one step instead of cascading through the rest of the fixture.
FREE_RUN_BOUND = 8

# The step kinds `decode_run` emits. P2 gave the prompt phase its own entry (`engine/prefill.h`), so a
# prompt-phase step now says `kind=prefill` where it used to say `kind=prompt`; what this file judges is
# the PHASE (§07.5's rule is about the prompt, not about which loop ran it), and the check names keep
# saying "prompt" for that reason. One spelling per concept: the engine and the `--dry-run` stand-in
# below both emit `prefill`, and nothing emits the old spelling.
PREFILL_KIND = "prefill"

# §06.5's first sampled token (`y0`) comes from the row BEFORE generation step 0, and that row's pin is
# the BARE `logits_last.npy` — `reference_dump.py::Recorder.pin` never prefixes it and refuses to let
# `--logit-steps` drop it, so on every fixture the bare name IS the prefill row (§10.5's last-writer-wins
# trap belongs to `input_ids`/`pre_final_hidden`, which the hooks re-pin unprefixed, not to this one).
# D-8: the tie policy is a statement about a PIN that never chose a winner, not about a generation phase,
# so it governs this row exactly as it governs step 40. `-1` is how this file names that row in step
# vocabulary, so one place decides which pin a step number means.
PREFILL_ROW = -1
DECODE_KIND = "decode"

# §06.1/§06.5: the two shapes a prefill wave's logits can take, exactly as `decode_run
# --prefill-logits` spells them. `last` is the engine's default and §06.5's shape — only the wave's
# last row is argmaxed, so every earlier prefill STEP line carries `sampled=-1 top_logit=none` (-1 is
# the engine's "nothing pending" marker, not a token id). `all` prices every row, which is what the
# byte-identity comparison against the pre-P2 per-token prompt loop runs in, and it is the only shape
# in which a prompt-phase argmax exists at all to be misread as a stop.
PREFILL_LOGITS_LAST = "last"
PREFILL_LOGITS_ALL = "all"


# ---- carried records in `data/refs/` that are not pins: named and skipped, never silently ignored --
# `data/refs/` holds two kinds of directory. Most are pinned HF dumps — a `manifest.json` plus the
# `.npy` files this gate pairs against. Three are CORPUS records that other tools read instead:
# `speccheck/gates/run_g7.py` (G7's token agreement against the reference's own self-consistency
# floor, §10.5 rule 3), `tools/corpus_freerun.py` and `tools/corpus_margin.py`. A corpus record
# carries `kind: "corpus"` with `prompts[]` (text, ids, the reference's greedy continuation) and no
# `sources`, no `greedy_continuation`, no `pins` and no `.npy` at all, so `--fixture` has nothing to
# pair against: `pairing_checks` reached `man["sources"]` and died with `KeyError: 'sources'` on
# `corpus48engine` — one of the three gate-internal crashes §15.2 recorded, and one whose exit code
# an admission could not tell apart from a numerics result.
#
# The remedy is `tools/reference_dump.py::CARRIED_NOT_PINNED` applied one level up: a carried file is
# NAMED and skipped, with the reason printed, so a reader of the log can see that the directory was
# looked at and what it is. The list is DERIVED-checked by `refuse_non_pin_fixture` rather than
# remembered — every `kind`-carrying, `pins`-less manifest must appear here, and a listed name must
# still read that way — because a hand-typed fixture list is the exact mistake §15.2 and D40 recorded.
NOT_A_PIN_FIXTURE: dict[str, str] = {
    "corpus48": "the reference corpus (49 plain-text prompts with the HF greedy continuation of"
                " each). It is a token-agreement corpus, not a pin set: `speccheck/gates/run_g7.py`"
                " compares a free run's prefixes against it and `tools/corpus_margin.py` prices the"
                " floor, which is §10.5 rule 3's tripwire. It carries no `.npy`, so there is no"
                " step, state or logits to pair.",
    "corpus48sdpa": "the same corpus re-dumped with `--attn_implementation sdpa`, i.e. the"
                    " reference's own self-consistency floor (A-18: the 0.8776 token-equality"
                    " floor, sdpa against eager). Same shape as `corpus48`, same reason.",
    "corpus48engine": "the ENGINE free-run record over `corpus48`'s prompts — its own manifest"
                      " names `binary: engine/build/decode_run` and `reference_corpus: corpus48`,"
                      " so it records engine output, not reference output, and it carries no `.npy`"
                      " either. `tools/corpus_freerun.py` writes it; its `gen[i]` pairs with"
                      " `corpus48`'s `greedy_continuation[i]`, which is a corpus comparison, not a"
                      " gate pairing.",
}

# A refusal that is neither green nor red. An admission needs three outcomes, not two: a fixture this
# gate declines to adjudicate is a fact about the fixture, and folding it into exit 1 makes it read as
# a numerics failure — which is how `PASS=0` from an error message was tabled as a result (§15.2).
EXIT_SKIPPED = 3


def refuse_non_pin_fixture(name: str, man: dict) -> None:
    """Stop, naming the fixture and why, when its manifest is a carried record rather than a pin set.

    Two failure modes are refused and one is announced:
      * a `kind`-carrying manifest with no `pins` that is NOT listed → refuse, and say what to do.
        Guessing here is what would let a schema gap read as a numerics result.
      * a listed name whose manifest has since become a real pin set → refuse, because the carried
        reason printed beside it would be a lie.
      * a listed carried record → print `ENGINE GATE SKIPPED` with its reason and exit
        `EXIT_SKIPPED`, so no caller can mistake it for either verdict.
    """
    carried = man.get("kind") is not None and "pins" not in man
    listed = name in NOT_A_PIN_FIXTURE
    if listed and not carried:
        raise SystemExit(
            f"gate: {name!r} is on NOT_A_PIN_FIXTURE but its manifest now carries `pins` (or has"
            " dropped `kind`) — it has become a pinned fixture. Re-read why it was listed, then"
            " remove the entry; leaving it here would skip a fixture this gate can judge.")
    if not listed:
        if not carried:
            return
        raise SystemExit(
            f"gate: fixture {name!r} carries kind={man.get('kind')!r} and no `pins`, so it is a"
            " carried record rather than a pin set — and it is not on `engine/gate.py`"
            "::NOT_A_PIN_FIXTURE. Add it there with a printed reason (the"
            " `tools/reference_dump.py::CARRIED_NOT_PINNED` rule: named and skipped, never silently"
            " ignored), or point --fixture at a pinned dump. Refusing rather than guessing, so a"
            " schema gap cannot be mistaken for a numerics result.")
    print(f"ENGINE GATE SKIPPED — {name} is not a pinned fixture: {NOT_A_PIN_FIXTURE[name]}")
    raise SystemExit(EXIT_SKIPPED)


def selftest_non_pin_fixtures() -> list[str]:
    """Prove the carried-record list has teeth, in process — no engine, no device, nothing written.

    One case per way the list can rot, each naming what it must catch:
      * a LISTED carried record is skipped BY NAME, with its reason printed and `EXIT_SKIPPED` returned
        (a silent skip is §15.2's bug wearing a different hat);
      * an UNLISTED carried record is refused by name and prints no verdict at all — a refusal that
        reads as a result is exactly how a `PASS=0` from an error message got tabled as one;
      * a listed name whose manifest has since become a pin set is refused too, because a stale pardon
        would retire a fixture this gate can judge;
      * the population walk: every manifest under `data/refs/` that reads as a carried record is on the
        list. This is what makes the list DERIVED rather than remembered — delete an entry while its
        directory is still on disk and this case goes red, with no need to run that fixture at all;
      * and the must-not-catch side: a real pinned manifest, read from disk exactly as it is, passes
        through with nothing printed and nothing raised. If that case ever fired, the list would be
        eating fixtures, which is the opposite failure and a worse one.
    """
    bad: list[str] = []

    def attempt(name: str, man: dict) -> tuple[object, str]:
        """(exit code or None, everything printed) for one `refuse_non_pin_fixture` call."""
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                refuse_non_pin_fixture(name, man)
            return None, buf.getvalue()
        except SystemExit as exc:
            return exc.code, buf.getvalue()

    listed = sorted(NOT_A_PIN_FIXTURE)
    if listed:
        name = listed[0]
        code, out = attempt(name, {"kind": "corpus"})
        ok = code == EXIT_SKIPPED and out.startswith(f"ENGINE GATE SKIPPED — {name}")
        print(f"  [{'PASS' if ok else 'FAIL'}] a listed carried record is skipped by name, with its"
              f" reason printed and rc={EXIT_SKIPPED}"
              f" — rc={code!r} printed={out.strip()[:60]!r}")
        if not ok:
            bad.append(f"{name}: a listed carried record must print ENGINE GATE SKIPPED and exit"
                       f" {EXIT_SKIPPED}; got rc={code!r} printed={out.strip()[:60]!r}")
        code, out = attempt(name, {"pins": {"x": {"step": 0}}})
        ok = isinstance(code, str) and "become a pinned fixture" in code and "SKIPPED" not in out
        print(f"  [{'PASS' if ok else 'FAIL'}] a listed name whose manifest now carries `pins` is"
              f" refused, not skipped — because a stale pardon skips a judgeable fixture"
                           f" — rc={str(code)[:60]!r}")
        if not ok:
            bad.append(f"{name}: a listed manifest that became a pin set must be refused;"
                       f" got rc={str(code)[:80]!r}")
    else:
        bad.append("NOT_A_PIN_FIXTURE is empty while data/refs/ is expected to hold carried records")
    code, out = attempt("zz-unlisted-carried-record", {"kind": "corpus"})
    ok = isinstance(code, str) and "NOT_A_PIN_FIXTURE" in code and "ENGINE GATE" not in out
    print(f"  [{'PASS' if ok else 'FAIL'}] an unlisted carried record is refused by name and prints no"
          f" verdict line — rc={str(code)[:60]!r}")
    if not ok:
        bad.append("an unlisted carried record must be refused with no verdict line;"
                   f" got rc={str(code)[:80]!r} printed={out.strip()[:60]!r}")
    real = json.loads((REFS / "manifest.json").read_text())
    code, out = attempt(REFS.name, real)
    ok = code is None and out == ""
    print(f"  [{'PASS' if ok else 'FAIL'}] must-not-catch: the real pinned manifest of"
          f" {REFS.name!r} passes through untouched — rc={code!r} printed={out.strip()[:60]!r}")
    if not ok:
        bad.append(f"{REFS.name}: a pinned manifest must never be refused as a carried record")
    unclassified = []
    for mf in sorted((ROOT / "data" / "refs").glob("*/manifest.json")):
        man = json.loads(mf.read_text())
        if man.get("kind") is not None and "pins" not in man and mf.parent.name not in NOT_A_PIN_FIXTURE:
            unclassified.append(mf.parent.name)
    ok = not unclassified
    print(f"  [{'PASS' if ok else 'FAIL'}] the list matches the population: {len(listed)} listed,"
          f" and every carried manifest under data/refs/ is one of them"
          f" — unclassified: {unclassified or 'none'}")
    if unclassified:
        bad.append("carried manifests not on NOT_A_PIN_FIXTURE: " + ", ".join(unclassified)
                   + " — add them with a reason, or the list has become a silent hole")
    return bad


def set_fixture(name: str) -> None:
    """Point every pin read at `data/refs/<name>`. Module state rather than a parameter on nine
    functions because `load_pin` is called from the pairing, curve and integrity checks alike.

    The carried-record check belongs here rather than in `main` because this is the one choke point
    every mode passes through — `--selftest`, `--smoke`, `--dry-run` and a device run all call it
    before they read a pin, so a corpus directory is refused the same way in all four."""
    global REFS
    REFS = ROOT / "data" / "refs" / name
    manifest = REFS / "manifest.json"
    if not manifest.exists():
        raise SystemExit(f"gate: fixture {name!r} has no manifest at {REFS}/manifest.json"
                         " — make it with tools/reference_dump.py (see engine/results.md)")
    refuse_non_pin_fixture(name, json.loads(manifest.read_text()))


def lane_geometry() -> tuple[int, int, int]:
    """(lanes one block stages, max split blocks per query row, lane ceiling) — read out of the two
    files that enforce them and never retyped here.

    P7 turned the flat `MAXSC` wall into a product (§07.3). One block of the one-pass softmax stages
    `MAXSC` lanes and cannot be split at all; above that the engine cuts the lane axis into
    `ceil(lanes / MAXSC)` blocks of the streaming form, so the ceiling is `MAXSC × kAttnMaxSplit`
    and the `part` scratch is sized for exactly that many partials. `engine/kernels_attn.cpp` is the
    enforcer of both halves — it throws above the product rather than writing past a stage or past the
    scratch — which is why both constants are grepped out of the code that obeys them instead of being
    copied here: a number in this gate could disagree with the launcher, and two authorities that can
    drift is the failure this file refuses everywhere else. `prototype/attn_decode/run.py` reads its
    `TILE` the same way for the same reason.
    """
    kernel = ROOT / "prototype" / "attn_decode" / "attn_decode_kernels.h"
    m = re.search(r"#define\s+MAXSC\s+(\d+)", kernel.read_text())
    if not m:
        raise SystemExit(f"gate: cannot read MAXSC from {kernel} — the lane staging is the"
                         " kernel's to own, so this gate refuses to guess it")
    staging = int(m.group(1))
    launcher = ROOT / "engine" / "kernels_attn.cpp"
    s = re.search(r"constexpr int kAttnMaxSplit\s*=\s*(\d+)", launcher.read_text())
    if not s:
        raise SystemExit(f"gate: cannot read kAttnMaxSplit from {launcher} — that is the launcher's"
                         " half of the lane ceiling (it sizes `part` with it), and a ceiling this gate"
                         " invented for itself would be a band, not a check")
    splits = int(s.group(1))
    return staging, splits, staging * splits


def lane_ceiling() -> int:
    """The lane ceiling alone, for callers that only need the number. See `lane_geometry()`."""
    return lane_geometry()[2]


def lane_ceiling_note() -> str:
    """The ceiling with its two factors named, so a FAIL says which constant to go look at."""
    staging, splits, ceiling = lane_geometry()
    return (f"staging MAXSC {staging} x kAttnMaxSplit {splits} = {ceiling} lanes"
            " (`prototype/attn_decode/attn_decode_kernels.h` sizes a block,"
            " `engine/kernels_attn.cpp` caps the blocks per row and allocates the `part` partials for"
            " them; above it the launcher throws rather than truncating)")


# ---- the band ----------------------------------------------------------------------------------
# A token's logits vector is the end of a 24-layer cascade, so §10.1/§10.2 have already decided what
# it is allowed to look like, and this gate uses their published numbers instead of inventing one:
#
#   cos >= 0.9995   §10.1 G5, "incremental == batch": prefill T then decode D, compared with the
#                   reference at the same positions, is held to sampled ids exact + logits
#                   cos >= 0.9995. That is the closest published analogue of what this gate does, so
#                   it is the hard floor here.
#   meanrel <= 5 %  §10.2's cascade ceiling ("cos >= 0.999, meanrel <= 5 %"), the band G2/G3/G4 use
#                   for every whole-cascade tensor comparison.
#   cos >= 0.9999   §10.1 G6, decode logits under online softmax. REPORTED, not gated — DECIDED
#                   2026-10-11, do not promote it; the reasoning is below, not in a results file,
#                   because a band comment is the thing a future reader edits.
#
#   ndiff / max ULP REPORTED, NOT GATED. §10.1 F1-F4 is explicit about why: on a chained dot the
#   LEGAL ndiff is 29-47 %, and a max-ULP bound is blind to a whole-tensor 1-ULP shift. A threshold
#   on either would be theatre.
#
# Why G6's 0.9999 is advisory here and G5's 0.9995 is the floor — and why the three §10 cos figures
# are not interchangeable thresholds for the same quantity:
#
#   * G3i (cos >= 0.9999) measures an ISOLATED single layer fed the reference's own `s_in`, and it is
#     the tightest figure precisely because "no upstream cascade can enter" (§10.1). It is the right
#     bar for one kernel and the wrong bar for a token.
#   * G6's 0.9999 is a decode-step logits vector in the REFERENCE, i.e. the whole 24-layer chain
#     already upstream of it, with the softmax form as the only variable. It is a bf16-cascade figure,
#     not a per-op one.
#   * G5's 0.9995 is the incremental-vs-batch invariant: same model, same cascade, two ways of getting
#     to position T+D.
#   The rule, as decided: an end-to-end quantity must not be judged by a per-layer or per-operation
#   bar, because the error it is allowed to carry is the accumulated error of all 24 layers and the
#     whole token chain, not the error of one stage. G3i/G5/G6 therefore compare different things —
#     one stage, two routes to one position, and one form of one stage — and a threshold moved between
#     them is a different measurement, not a stricter or looser one.
#   What actually decides correctness on this gate is **token identity (G7, exact ids)**; the logits
#     band is supporting evidence about *how* the engine got there, which is why a step whose cos sits
#     between the two figures is reported (the G6 note below) and not failed. The first end-to-end
#     pass had exactly that profile: step 1 at cos 0.999886 — under G6, over the G5 floor, with every
#     one of the eight ids exact and the reference's own top-1/top-2 margin 7-23x the largest element
#     error at every step. It stays a documented observation. Two CPU probes price it, both with no
#     engine and no device: `engine/logits_floor.py` is the LAST-OP floor (legal reassociation of the
#     reference's own logits dot, its answer 0.99999862), and `engine/route_price.py` is the
#     END-TO-END floor (the same reference run through two legal GDN prompt forms; its spread is
#     1-cos 4.7e-5..1.9e-4 and it is wider than the engine's deviation at every step). The engine is
#     below the first and inside the second, which is why the flag is informational: a per-stage floor
#     applied to an end-to-end quantity will always fire, and the quantity that could really be wrong
#     is tokens, which is G7. engine/results.md records both tables and the reasoning.
#
# §10.2 also fixes the DEFINITIONS: cos is over the flattened tensor, meanrel = mean|Δ| / mean|ref|
# (a ratio of means, not a mean of ratios — the latter explodes on the near-zero logits that most of
# a 248 320-wide vocabulary is).
BAND = {
    "cos_hard": 0.9995,      # §10.1 G5 — the flat floor, in force where no route allowance is priced
    "meanrel_hard": 0.05,    # §10.2 cascade ceiling — same fallback rule
    "cos_report": 0.9999,    # §10.1 G6, decode logits
    "route_margin": 3.50,    # cos: FALLBACK margin, used only for a cached allowance that carries no
                             # margin of its own (the pre-D-2 global scheme). See ROUTE ALLOWANCE below.
    "route_margin_rel": 0.42,  # meanrel: the same fallback.
}

# ROUTE ALLOWANCE — the logits band is keyed on the reference's own measured route spread, per fixture
# and per step. DECIDED 2026-10-12 by the operator: option 1 of the three written up in
# `engine/results.md` ("Decision 2 of the deep gate"). The full write-up with its tables is there; what
# belongs here, next to the code that applies it, is the rule, where its numbers come from, and what was
# deliberately not chosen.
#
# Why the flat floors above had to go. They are absolute, and they were measured to be unreachable by the
# reference itself. `engine/route_price.py` runs one pinned prompt twice through `speccheck/specimpl.py`
# — once with the chunked GDN prompt form (§04.7, the one G3 canonicalises) and once with the recurrent
# one (§04.4, the form an engine with no prefill necessarily takes) — feeds the reference's own tokens,
# and measures how far apart those two LEGAL forms land on the logits vector at each step. On `dec64f`
# that spread is 1-cos 2.71e-5 .. 1.714e-3 and it goes under G5 at steps 27, 28, 29, 30, 31, 32 — six of
# the seven steps the engine goes under it, same window, same shape, no engine anywhere in the
# measurement — and its meanrel is over §10.2's 5 % at 44 of the 64 steps (worst 60.8 %, i.e. 12x the
# ceiling). An absolute floor there is not measuring the engine. `dec64t` (same pin policy, a prompt
# whose own chain never reaches its stop id) supplies the other half: there the spread never exceeds
# 1.258e-4 and the engine's worst deficit is 2.410e-4, so the same instrument says a well-conditioned
# step is entitled to be judged far more tightly than 5e-4.
#
# The rule, at step j:
#
#     1 - cos(engine_j, pin_j) <= margin     x  route_1cos(j)        (the "shape" half)
#     meanrel(engine_j, pin_j) <= margin_rel x  route_meanrel(j)     (the "scale" half)
#
# `route_*(j)` is the measurement above, cached per fixture in
# `data/refs/<fixture>/route_allowance.json` by `engine/route_allowance.py`. This gate READS that file
# and never re-prices it: re-pricing is ~5 s of torch CPU per fixture and a gate run must not silently
# re-run the reference, and a cached spread carries a provenance header (command, date, and the hashes
# `load_allowance` checks) that a freshly computed one would not.
#
# The margins are DERIVED, not chosen — that is the entire difference between this and a relabelled band.
# Definition, PER FIXTURE (D-2, 2026-10-12, which retired the single global constant): `margin` for one
# fixture = max over THAT fixture's steps of (engine deficit / route spread), i.e. the tightest constant
# that admits every step that fixture was measured on, rounded UP at two decimals by
# `engine/route_allowance.py` and stored in the fixture's own file. Why per-fixture: while the margin was
# one global max, `dec300f`'s pricing would have moved the constant every other fixture is judged by — its
# worst ratio is 7.98 at step 86 while its route spread is *tighter* than the 64-step fixtures' (median
# route 1-cos 4.07e-5 against `dec64t`'s 6.71e-5), so pricing it globally meant 3.50 → 7.99 for everybody.
# Unpriced it sat on the flat floor with 3.7e-6 of headroom at step 86, and the 4k work is a perturbation.
# Keying per fixture means the derivation cannot read another fixture's rows — `--selftest` in that tool
# asserts exactly that, and asserts it is falsifiable by checking a global max would have answered
# differently. Measured, and re-derived from the shipped allowance files rather than from memory:
#
#     fixture   margin / margin_rel   own worst cos ratio        own worst meanrel ratio   steps
#     dec64f    3.50 / 0.42           3.0368 at step 32          0.3344 at step 54         64
#     dec64t    3.50 / 0.42           3.4939 at step 54          0.4111 at step 4          64
#     dec300f   7.99 / 0.72           7.9837 at step 86          0.7198 at step 72         128
#     mt32f     2.93 / 0.24           2.9221 at step 19          0.2303 at step 3          32
#
# `dec64t`, `dec300f` and `mt32f` are carrying their own derivations. `dec64f` carries 3.50/0.42 while its
# own derivation is 3.04/0.34: those two files were written under the global rule and D-2 recorded keeping
# them, so their `margin_basis` field is absent and `load_allowance` says out loud that the margin in force
# is one the fixture did not set for itself. Re-pricing `dec64f` REFUSES until somebody deletes its file
# deliberately, which is how "keep 3.50" stays a decision on the record rather than drift.
#
# Say the inconvenient halves plainly. (1) A margin of 3.0 or less leaves `dec64f` step 32 (3.0368) and
# `dec64t` step 54 (3.4939) RED — the constant is a measured property of this engine, not a taste call.
# (2) A margin defined as `max(engine deficit / route spread)` puts that fixture's worst step exactly on
# the boundary BY CONSTRUCTION — `dec300f` step 86 now sits at 7.984 against a 7.99 cap, i.e. the rule is
# calibrated to the very errors it is meant to tolerate. That mild self-reference is known and NOT fixed
# here: D-3 (plan §3) / Q4b schedules an engine-independent multiplier from legal-route disagreement alone,
# and if that number comes out smaller than the engine needs, it is a finding about the engine, not a
# licence to loosen. Per-fixture keying removed the cross-fixture leak, not the self-reference.
#
# What it costs and what it buys, in both directions. Outside the excursion the allowance is TIGHTER than
# the floor it replaces — on `dec64t` all 64 steps get a cos cap under §10.1's 5e-4 (tightest 1.2274e-4,
# step 41), and on `dec64f` 54 of 64 do (tightest 9.4788e-05, step 24) — so this is not a loosening of
# the gate, it is a band that is tight where the reference is tight and loose only where the reference is
# provably loose. Inside the excursion it is looser: `dec64f` step 28's cap becomes 5.9983e-3, 12x the
# flat floor it replaces. What still binds there is unchanged and hard: the
# per-step argmax (the two legal routes agree with each other at 64/64 steps, and with the pin at every
# step of that window) plus the tie policy's proof.
#
# meanrel gets the SAME treatment rather than becoming reported-only, and the reason is what cos cannot
# see: cos is scale-invariant, so a mis-scaled final norm — exactly the class §10.2's meanrel ceiling was
# put in for — scores cos 1.000000 and only meanrel can catch it. Reported-only would have handed that
# failure class back, which is the hole option 2 below was rejected for; keyed on the measured spread it
# keeps the check and stops judging a step by a figure two legal reference forms exceed 12x over.
#
# What was NOT chosen:
#   * option 2, "gate the decision, not the vector": per-step argmax only, logits demoted to reported
#     evidence with the §10.2 margin statistic as the check. Cheapest and honest, and it says what the
#     gate really verifies — but it gives up any claim on the logits vector's shape, i.e. the whole
#     wrong-but-argmax-preserving class. The argmax rule is a hard gate HERE too, unchanged by this
#     decision, underneath the band in section (b), with decision 1's proved-tie exception as its only
#     qualification. It is just not the only thing this gate checks.
#   * option 3, "leave the bands and label the window": keep 0.9995 hard and carry a per-fixture list of
#     steps known to be inside legal spread, printed as `SPREAD` lines that are reported but not armed.
#     REJECTED, and the reason is on the record: an exempted step is judged by NOTHING, so a real bug
#     that lands on an exempted step passes silently — and the list is hand-maintained from a measurement
#     that can move. Under the allowance there is no exempted step and no list to keep in sync: every
#     step is judged, by a number measured at that step.
#
# Staleness is designed to be loud. Each cached file names the fixture it was priced for and records the
# prompt-id hash, the forced-feed hash, the `speccheck/specimpl.py` hash and the step count, because those
# are what the measurement was made from; `load_allowance` FAILS the gate if any of them disagrees with the
# fixture in front of it, so a re-pinned fixture or a changed reference cannot leave a stale band behind.
# It also checks the file against itself: every step's cap must equal the file's own margin times that step's
# spread, and the margin must admit the engine deficits recorded in the same file, so a hand-edited number
# fails instead of quietly becoming the band. A fixture with NO cached file — `dec8` today, whose route table
# `route_price.py` only prints — falls back to the flat published floors and prints `flat G5` on every step
# line, so the fallback is visible rather than assumed.

# TIE POLICY — an exact tie in the pin is a named, proved exception to "ids exact per step", and it
# is the strictest thing available, not a concession. DECIDED 2026-10-12 by the operator; the band
# above is untouched by it, and the two are not the same question.
#
# SCOPE, and it is the part that was wrong first (D-1, 2026-10-12): the policy is an EXCEPTION, so it
# applies only where the engine needs excusing — i.e. where the engine's id is NOT the pin's top-1 id.
# A step where engine and pin picked the same id is an ordinary exact match and must not consult the
# tie rule at all; `mt32f` step 11 proved why. There the pin ties 24.25 (payload 0x41c2) between ids
# [4016, 15048], the engine picked 4016 — the PIN'S OWN top-1 — and the step was nevertheless RED,
# because the predicate as first written also demanded the engine's F32 at the pin's top-1 round back
# to the pinned payload, and it rounded 1 ULP high (24.391083 -> 0x41c3). G7 was satisfied outright at
# that step; a tied step was being judged MORE strictly than an untied one, by a half-ULP band this
# file's own comment declines to invent. `tie_verdict` now returns three kinds — "no-tie", "match"
# (policy not consulted, the tie still noted on the log), "tie" (the clauses below armed) — and section
# (a) only arms a check on the third. `--selftest --fixture dec64f` (or `--dry-run --tie-same-ulp 17`) is
# the regression: same id, logit 1 ULP off the payload, must be GREEN.
#
# Why an exception is admissible at all: `argmax` over an exact tie is not defined by the mathematics
# (§03.4 pins ties to the lowest id, which is a *tie-BREAK* — an arbitrary choice among values the
# model called equal), so "engine's id == pin's id" cannot be demanded at such a step. Measured in
# `dec64f`: at step 17 the top two pinned logits are both 17.25 (ids 2018, 10407) and at step 18 both
# 19.75 (ids 1379, 64704) — 2 ways tied at both. That the engine's other choice is not a numerics
# failure is shown independently by `engine/route_price.py`: both legal specimpl routes also pick
# 10407 at step 17, i.e. two legal forms of the reference disagree with the pin's tie-break exactly
# as the engine does. Those two steps are what the policy is FOR, and they still resolve through it.
#
# Why the predicate is what it is, at a step the policy is actually invoked at. The exception is
# admissible only if the tie is PROVABLE FROM THE PIN, and the proof has to live in the space the tie
# exists in: the pin's BF16 payload. So:
#   1. >= 2 ids share the exact maximum **on the BF16 payload**, read through `load_pin`'s own
#      bit-shift widening (§10.5: a bitcast, never a numeric cast) — never by a float `isclose`;
#   2. the engine's sampled id is one of the tied ids;
#   3. RNE(the engine's F32 logit AT THE ENGINE'S OWN CHOSEN ID) equals the shared tied payload, i.e.
#      the engine's arithmetic kept the value the model called tied inside its own BF16 bin. All the
#      tied ids share one payload — that is what a payload tie IS — so this is one check, and it is
#      made at the id the engine actually stands on, not at the one the pin's tie-break happened to
#      name. (Checking at the pin's top-1 was D-1's other half of the same mistake: at `dec64f` step 17
#      the engine's value at 10407 is 17.237261 -> 0x418a, at 2018 it is 17.194519 -> 0x418a; both
#      hold there, which is why the old wording survived two fixtures.)
# Clause 3 in raw F32 space instead of payload space would be a category error, not strictness:
# `lm_head`'s FULL form writes an F32 accumulator with no BF16 rounding
# (`prototype/lm_head_gemv/lm_head_gemv_kernels.h:64`), so an F32-equals-BF16 test can only pass by
# luck. And clause 3 is not vacuous in payload space either: only 48 317 of 248 320 logits at step 17
# (19 %) land in their pinned BF16 bin, so it constrains real behaviour.
#
# What it deliberately does NOT do: it does not compare the engine's value at the *other* tied ids
# against a tolerance. At 17.25 one BF16 ULP is 0.125, so "within one ULP" would be a brand-new band
# with no floor probe behind it, which `AGENTS.md` forbids inventing. The engine's value at every tied
# id is therefore PRINTED (F32 and payload) and not gated, so a human can see whether the engine saw
# them as near-tied. If that ever looks load-bearing, size it with a floor probe and promote it then.
#
# A tie must never pass silently: every step the policy actually resolves is reported as TIE with the
# tied ids, the tied value and the engine's per-id values, and counted in the summary line; a tied step
# the engine matched outright is printed as an ordinary-match note, so the tie is still visible. An
# engine that picks a THIRD, untied id at a tied step fails loudly, and `--selftest` proves all three
# directions: the untied injection FAILS, the other-tied-id injection PASSES as TIE, and the
# same-id-with-a-1-ULP-off-logit injection PASSES as the ordinary match it is.
TIE_POLICY = ("exact tie in the pin (>= 2 ids share the max on the BF16 payload), the engine picked an"
              " id other than the pin's top-1, that id is one of the tied ones, and the engine's own"
              " logit at it rounds back to the shared tied payload")

# QUANTIZATION SLACK — the second named exception, and the one D-9 refused to invent until it was
# sized. DECIDED 2026-10-13; the bands are untouched by it, and it is NOT a widened tolerance: the
# vector band still binds at every step it excuses a token at, and the token exception has a number
# under it, not a word.
#
# The shape it covers is the step next door to D-1's tie. `chunk64b` step 0: the pin puts 8337 at
# 15.625 and 20619 at 15.500, so there IS a unique winner — by 0.125, which is 2 BF16 ULP at that
# magnitude. The engine's legal element error at that step is larger than the margin it would have to
# clear (maxabs 0.2509 measured, D-9), so "the engine must reproduce 8337" asks a bit-exact question
# of a bit-tolerant fixture: any legal bf16 route can flip it, and demanding the pin's id is demanding
# a coin-flip land a particular way. D-9 kept it RED rather than invent a threshold. This is the
# threshold, derived.
#
# THE BOUND, per judged step j:
#
#     bound(j) = meanrel cap in force at step j  x  mean|pin_j|
#
# That is the band the gate already polices, read out in logits units instead of as a ratio. meanrel is
# defined (§10.2) as mean|Δ| / mean|ref|, so `cap x mean|pin|` is the largest element error this step's
# own meanrel check would pass if the error were spread evenly over the vocabulary — the same cap
# `rule_for_step` prints on the step line, from this fixture's own `route_allowance.json` where one is
# priced and from §10.2's flat 5 % ceiling where it is not. No new constant, no new measurement, no new
# error model: the two inputs are a policed cap and a pinned tensor. `engine/slack_size.py` prints it
# for every fixture and every step, and re-derives it from `gate.py` rather than keeping a copy, in the
# spirit of `prototype/mlp_gemv/floor_probe.py` reading its band from `run.py`.
#
# Why the meanrel half and not the cos half: cos is scale-invariant, so it has no element-space reading
# at all. Turning a cos deficit cap into a max-element bound (‖Δ‖ <= 2·sqrt(cap)·‖ref‖) yields ~83 at
# `chunk64b` step 0 — a bound that excuses any token ever sampled. meanrel is the half of the policed
# band with a unit, and the only one that can be quoted in logits.
#
# Why it is not a tolerance fitted to the case. At `chunk64b` step 0 the bound comes out 0.16205 and
# the engine's own measured element error there is 0.2509: the bound is SMALLER than the error this
# engine demonstrably carries at that step. (Measured mean is not the licence: mean|Δ| at that step is
# 0.0389, well under the margin, which is exactly why the cap is used and the measurement is not.)
#
# One wrinkle, stated rather than hidden. meanrel is an AVERAGE, so read literally it licenses a single
# element to be wrong by cap x mean|pin| x N — ~4x10^5 logits at `chunk64b` step 0, the same degeneracy
# the cos half has. The reading used is the per-element average, cap x mean|pin|: it is the smaller figure
# of the two, and the smaller of the two against what this step is measured to carry (0.2509), which is
# what makes it defensible as the scale below which a pinned top-2 simply cannot be decided.
#
# What the bound moves with, because it will move: it is the licence IN FORCE, not a property of the pin.
# `chunk64b` is unpriced, so its cap is §10.2's flat 5 %. Pricing it — D-2's standing offer, and the note
# atop every `chunk64b` gate run says so — would put that fixture's own margin in place of 5 %, which at
# step 0 is nearer the 1.1990 % the run itself reports. The bound would fall to ~0.04, below this step's
# 0.125 margin, and step 0 would fall back OUT of the region and back to RED — correctly, because a tighter
# licence is a narrower permission to be wrong. Anyone who prices `chunk64b` re-runs `engine/slack_size.py`
# before concluding the slack rule broke: the licence changed, the rule did not.
#
# SCOPE, and it is deliberately tiny. Measured over all eight fixtures (`engine/slack_size.py`, the
# whole table is in `engine/results.md`): the only steps in the corpus where a UNIQUE winner's margin
# is inside the bound are four in `chunk64b` — steps 0, 1, 13 and 22, every one of them a 1-ULP pin
# margin — and none anywhere else. The nearest decidable step in the corpus is `dec300f` step 30, whose
# margin is 1.46x its own bound, and `dec64f` step 29 at 2.23x, so the line is not comfortably clear of
# the nearest green step and a bound 46 % larger would start excusing `dec300f`. That is why the
# injections are aimed at exactly those two steps. The exact-tie steps that also sit under the bound
# (`dec64f` 17 and 18, `mt32f` 11, `chunk64b` 11) belong to D-1's proved-tie predicate — a pin with no
# unique winner is a different question with a stricter answer (its leader set is payload-equality, and
# at `chunk64b` 11 that set is 2 of the 4 ids inside this bound) — so `step_slack` refuses any step
# whose pin does not have a single winner, and D-1 is never overridden silently.
#
# THE PREDICATE, admissible only if ALL of: the pin has a unique winner at the step; the engine's id is
# NOT that winner (D-1's scope rule — a step where the engine reproduced the pin's id consults no
# exception, and the step still gets a note when it sits inside the region); the pin's top-1/top-2
# margin is smaller than bound(j); the engine's id is one of the pinned ids lying inside that margin
# (printed, and at `chunk64b` step 0 it is exactly the pair 8337/20619); and every other judged step of
# the generation still matches, which is the aggregate id-list check doing its usual job — one excused
# token may not drag a changed chain with it. §06.5's `y0` row gets no slack: nothing polices a vector
# band on that row, so there is no cap to read a bound from and the rule fails closed there.
#
# Everything it used is printed on the line — both pinned values, the margin, that margin in ULP, the
# bound, the cap and mean|pin| it came from, which rule set the cap, the leader ids, and the engine's own
# values at each. The engine's own gap between the two is printed and NOT gated: it is the engine's
# business which way its coin landed, and gating it would be a new band. `report()` counts the uses.
SLACK_RULE = ("the pin's winner is unique but its top-1/top-2 margin is smaller than the element error"
              " this step's own meanrel band licenses (cap x mean|pin|), and the engine's id is one of"
              " the pinned ids inside that margin")

# THE CASCADE BAND — what a reassociating GEMM route is entitled to look like at the END of the
# cascade, and the fourth named exception. ADOPTED 2026-10-14 from `progress/2026-10-14/route-c-audit-2026-10-14.md`;
# the measurements are in that file and in `progress/bench/evidence/rc_a1/a1.json` (reproduced independently in
# `tmp/rcv_a1/a1.json`), and D-11 (`bench/build-queue.md`) is the decision that asks for them here.
#
# WHY A FOURTH POLICY WAS NEEDED. Everything this file policed before was measured either one op deep
# (§10.1 G5's cos, §10.2's meanrel, A1's per-matmul 1 ULP) or on one specific route (the D-2 allowance,
# whose margin is the HAND engine's own deficit against the reference's two legal GDN prompt forms).
# The audit's §0.1 is the finding that a per-matmul bar cannot be applied twenty-four layers downstream:
# applied literally at the logits it fails every fixture and every stage by 3-5 orders of magnitude
# while the stage ladder stays flat (×8.8 end to end, `embed` bit-identical) — the band, not the engine.
# So what is priced here is not a tighter op-level bound but the measured size of a legal cascade.
#
# THE FIVE NUMBERS, and what each one is a ceiling ON:
#
#   row_1cos_max   2e-3   one LOGITS ROW against another row of the same prompt, on the hand<->hipBLAS
#                         pairing. A2 measured the hand route against the library route over 1694
#                         dumped rows of all eight fixtures: max 1.889e-3 (dec64f), mean 5.2e-5..2.0e-4
#                         per fixture. The line is 6 % above the worst row ever seen, and A4's three
#                         injected bugs sit 2-3 orders worse (their 1-cos separations run 8e7-3e8 ×).
#                         It does double duty: section (h)'s wave-width band on a `library` run, and the
#                         floor under every step's own cap (rule 1 below). Both are the same pairing,
#                         hand-vs-hipBLAS, which is why one number served both until `lt` arrived.
#   row_1cos_max_lt 2.75e-3 the SAME statistic on the hand<->hipBLASLt pairing, i.e. what section (h)
#                         compares when the run is `--gemm lt`: `resolve_wave_gemm` forces the hand GEMV
#                         at one row, so the wide side of that pair is a hipBLASLt kernel, not a hipBLAS
#                         one, and a band sized on the old pairing is a claim about a pair this gate no
#                         longer runs. Measured, not adjusted, and re-measured on the algo-pin table
#                         the shipping binary actually carries: `engine/wave_band_floor_probe.py` over 9
#                         fixtures and 2951 compared rows reads a worst row of 2.204122e-03 (dec64f/
#                         logits45.bin — the same row that is the worst on the hipBLAS leg too, at
#                         1.888942e-03), and the line is that max x a stated 1.25 margin (2.755e-03,
#                         rounded down to 2.75e-3 rather than up). The SAME probe re-measured the
#                         hipBLAS leg at 1.8889e-03, i.e. `row_1cos_max` needed no move,
#                         and it is NOT loosened for `lt`: a `library` run is still judged at 2e-3. The
#                         two legs do not even order the same way per fixture — lt is the *lower* spread
#                         on dec8, chunk64b, p128 and both wide fixtures and the higher one on dec64f,
#                         dec64t, mt32f, wide520f — which is the reason this is a second number and not
#                         a bump of the first.
#                         WHY TRIMMING THE PIN TABLE DID NOT MOVE IT, and what it did move. A pin binds
#                         on (N, K, M) with M = `kMaxWaveRows` (engine/kernels_gemm.cpp `lt_plan_for`),
#                         so it can only ever serve a body of exactly 512 rows. The row that SETS this
#                         band lives in dec64f, whose widest body is 17 rows — no pin reaches it, on
#                         three entries or one — so when D28 cut the table from three pins to one
#                         (`down_proj` alone) the max came back the same number to every printed digit,
#                         and the band could not tighten. What did move is where the pins actually
#                         bound: wide640f/wide1024f, the two fixtures with 512-row waves, went from
#                         2.0330e-04 to 1.4517e-04 (-29 %). A cheaper corpus, same ceiling. Recorded
#                         here because the first version of this comment cited a three-entry table the
#                         binary no longer carries; the re-take is
#                         `engine/wave_band_floor_probe.py --gemm-pins all`, 2026-10-16, and the echo
#                         that proves which set a run ran is `pins=<in force>/<in the table>` on the
#                         `GEMM route=` line.
#                         DOES IT STILL BITE? Yes, and the proof is a command rather than an argument:
#                         the smallest defect this tree can inject at this check is `--break-ab-logits`
#                         (one row shifted 8 logits), which lands at 4.316e-03 of 1-cos on the stand-in
#                         corpus — 1.57x above the new line, and `--selftest` asserts it goes RED. So a
#                         future move of this number above ~4.3e-3 turns `--selftest` red by itself; the
#                         band cannot quietly grow until it cannot fail, which is the property
#                         `prototype/gdn_collapse/results.md` §1 demands of any driver that judges by a
#                         threshold it did not measure.
#   stage_meanrel  3e-2   one STAGE TILE (a `DecodeScratch` tile, not a logits row) against the same
#                         tile on the other route. A3's ladder spans meanrel 1.92e-3..1.70e-2 over 56
#                         comparisons; 3e-2 is that ladder's top with 1.8x of headroom, and A4's bugs
#                         are 10^5-10^6 above it.
#   stage_cos      0.9998  the same stage ladder's cos floor: A3's worst stage is 0.999848723 (`norm`),
#                         0.999833378 (`block_out`), so the floor is 1.7e-4 below the worst legal
#                         stage and the whole ladder is monotone to within the ×8.8 spread above.
#   flip_room_ulp  1.50   how much top-1/top-2 room a side had to have before a flipped token is a bug
#                         rather than a coin toss — measured IN BF16 STEPS AT THAT ROW'S OWN MAGNITUDE
#                         (`room_in_ulp`), not in logits. The unit is the whole point: the 0.10 this
#                         key used to carry was 0.8 of a step at logit 20 and 40 steps at 0.5, so it
#                         was a magnitude proxy masquerading as a decision rule, and `wide520f` row 180
#                         went RED at logit 20 for a pair one representable step apart (§3e of
#                         prototype/gemm_algo_sweep/results.md).
#                         WHERE 1.50 COMES FROM. The census that sizes it is
#                         `prototype/gemm_algo_sweep/results.md` §3f: both cross-width pairings
#                         (batched-vs-carry and batched-vs-one-row) over 9 fixtures x 2 routes x 2951
#                         rows, all 74 flips named with their room in logits AND in ULP. Worst flip
#                         anywhere in it: **0.9985 ULP** (`lt`, wide520f row 180, the red row), then
#                         0.7247 (`library`, p128 row 35), 0.6224 (`library`, batched-vs-one-row),
#                         0.4353 (`lt`, same). So 1.50 is 1.50x the worst legal flip in the corpus on
#                         the route that produced it, and 2.07x/2.41x/3.45x over the other three —
#                         printed on every run by `flip_room_line()`, because a threshold whose margin
#                         is not in the log is a threshold nobody can audit a year later.
#                         WHAT THE OLD LINE DID WHEN REWRITTEN IN THIS UNIT, both directions. 1.50 ULP
#                         is a SMALLER absolute gap than 0.10 for |top-1| < 16 (0.09375 at [8,16),
#                         0.0469 at [4,8), 0.0117 at [1,2)) and a LARGER one at or above 16 (0.1875 at
#                         [16,32), 0.375 at [32,64)), because `bf16_ulp` is a step function. So this
#                         is a trade, not a relaxation: it tightens everywhere the corpus's flips
#                         were NOT sitting and relaxes at the magnitudes they were. Honest half of that
#                         sentence: over the 5902 rows of the census exactly ONE row changes colour,
#                         wide520f row 180 on `lt` (RED -> admitted), and zero rows change colour the
#                         other way, because no flip in the corpus sits below |top-1| = 8. The
#                         tightening is therefore licensed by the unit, not demonstrated by this
#                         corpus, and it is witnessed by an injection instead —
#                         `--break-ab-room 3:2.0:1.0` (§3f, "Does it still bite").
#
# HOW THEY BIND, which is the whole of the design. Three rules, and the first is the reason none of
# these numbers is a loosening of anything that is already green:
#
#   1. A step's band is never TIGHTER than the cascade floor: on the `library` route the cap in force is
#      `max(the hand route's cap at that step, the cascade figure)` — cos deficit `max(rule, 2e-3)`,
#      meanrel `max(rule, 3e-2)`. A step the hand route is allowed to be sloppy at (dec64f's excursion
#      window, meanrel 7.09 %) stays exactly that sloppy; a step whose D-2 allowance demands 1.14e-4 of
#      a route whose own legal hand<->library delta is 1.9e-3 stops demanding the impossible. The hand
#      route is untouched: for `--gemm hand` every cap in this file is the number it was yesterday.
#   2. `stage_meanrel`/`stage_cos` are STAGE figures and they bind stage tiles only — the only stage
#      comparison in the tree is `tools/route_c_audit.py a3`, which now reads them from here instead of
#      carrying a locally computed band. They are NOT applied to a logits row: a row's meanrel between
#      two legal routes reaches 6.08e-02 (measured over the same 1694 A2 rows — `tmp/rc_a2`, priced by
#      `engine/cascade_floor_probe.py`), three times this ceiling, because a logits row's energy sits
#      in a few hundred large elements while the rest of a 248 320-wide vocabulary cancels. Gating a
#      row at 3e-2 would fail legal data, exactly as a 1-ULP gate on the MLP act would (A1: that stage
#      floors at 5-8 ULP because three chained BF16 roundings are not 1-Lipschitz in ULP units).
#   3. A flipped token is admissible only where a side's own room was < `flip_room_max_ulp` BF16 ULP at
#      that row's magnitude; a flip with >= 1.50 ULP of room on BOTH sides is RED, and the line prints
#      both sides' gaps so a reader can see the difference between "this decision was a coin toss" and
#      "this decision was decided and the engine disagrees". `room` here is the PIN's top-1/top-2 gap —
#      the hand route reproduces the pin's decision, so the pin's room IS the hand route's room.
#      `tools/route_c_audit.py a2` buckets by the SMALLER of the two sides' gaps, which is a slightly
#      more permissive reading: two of the 24 A2 flips (p128 row 27, chunk64b row 97) have >= 0.10 of
#      room on the hand side and < 0.10 on the library side. This gate takes the literal reading, prints
#      the other side's gap beside it, and names the disagreement in a note rather than resolving it
#      silently in either direction. Nothing about that min() reading changed with the unit; the census
#      that sized 1.50 uses the same quantity (`engine/wave_flip_census.py` imports it from there).
#
#
# THEY ARE POLICED, IN PLACE. `tools/band_history.py` reads every block in its `POLICED_BLOCKS` list, so
# these four are ledger keys named `CASCADE_BAND.<field>` (`BAND` and `CASCADE_BAND` share this file, so
# the block has to be in the key or a ledger line could not say which number it describes). Each of the
# four carries its own line in `bench/band-history.jsonl` under D-11, citing the A2/A3 measurement behind
# the number above; `stage_cos_min` is listed in that tool's `LOWER_IS_LOOSER` because it is a FLOOR, so a
# downward move there is the band widening. The lines spell their decision as the commit subject
# ("D-11 accepted (option A), commit f3b2f6c") rather than the bare `D-11`, because `record` only resolves
# a bare `D-x` against `progress/2026-10-12/plan-2026-10-12.md`'s log and D-11 lives in `bench/build-queue.md`. Moving
# a value here still takes a line first:
#
#     .venv/bin/python tools/band_history.py record --file engine/gate.py \
#         --key CASCADE_BAND.row_1cos_max --old 0.002 --new 0.003 --decision <D-x> \
#         --reason "<the new measurement, on which fixture, at which step>"
#
# STOP AND REPORT, do not widen. A fixture that lands outside one of these four gets its number printed
# and handed to the operator; the audit derived them before this task from measurements already on disk,
# and a band tuned to the fixture it is judging is the thing `AGENTS.md` forbids.
# STOP AND REPORT, do not widen. A fixture that lands outside one of these four gets its number printed
# and handed to the operator; the audit derived them before this task from measurements already on disk,
# and a band tuned to the fixture it is judging is the thing `AGENTS.md` forbids.
CASCADE_BAND = {
    "row_1cos_max": 2.0e-3,       # per logits row, hand<->hipBLAS (also the floor under a step's cap)
    "row_1cos_max_lt": 2.75e-3,   # per logits row, hand<->hipBLASLt — section (h) under `--gemm lt`
    "stage_meanrel_max": 3.0e-2,   # per stage tile (a3), NOT per logits row
    "stage_cos_min": 0.9998,       # ditto
    "flip_room_max_ulp": 1.5,    # top-1/top-2 room, IN BF16 STEPS AT THE ROW'S OWN top-1, a flip may have
}
CASCADE_RULE = ("a reassociating route is judged by the step's own band floored at the measured cascade"
                " (cos deficit >= 2e-3, meanrel >= 3e-2), and a flipped token is admissible only where"
                " the side that owns the decision — the reference in a pin-vs-engine pairing, the closer"
                " of the two engines in a cross-width one — had under 1.5 BF16 ULP of top-1/top-2 room")

# Where the flip line came from, in the gate's own output, in the same shape as `WAVE_ROW_BAND_SOURCE`
# below: a band that does not name its denominator is a band nobody can re-check. The three figures the
# audit needs are HERE, not in a reader's memory: the line, the worst legal flip the census measured on
# the route in force, and the ratio between them.
FLIP_ROOM_SOURCE = {
    "library": ("worst legal flip measured on this route 0.7247 ULP (p128 row 35, batched-vs-carry)"
                " over 2951 rows, so the line is 2.07x the worst flip in the corpus; the same pairing"
                " against the one-row side puts its worst at 0.6224 ULP (chunk64b/wide520f row 55),"
                " 2.41x"),
    "lt": ("worst legal flip measured on this route 0.9985 ULP (wide520f row 180, batched-vs-carry,"
           " logit 20.04 — one BF16 step of 0.125 apart, which is the row that was RED under the old"
           " magnitude-blind 0.10) over 2951 rows, so the line is 1.50x the worst flip in the corpus;"
           " the same pairing against the one-row side puts its worst at 0.4353 ULP (wide520f row 87),"
           " 3.45x"),
}
FLIP_ROOM_CENSUS = ("census: engine/wave_flip_census.py over data dumped by"
                    " engine/gate.py --gemm <route> for all nine fixtures, both cross-width pairings,"
                    " 74 flips in 11 804 row comparisons; numbers and commands in"
                    " prototype/gemm_algo_sweep/results.md §3f")
FLIP_ROOM_REFERENCE_SOURCE = (
    "this pairing has NO measured margin of its own: 0 flips were admitted against a pin across the"
    " nine fixtures x both routes of the 2026-10-16 device set (18 gate runs, all ENGINE GATE PASS),"
    " so on this path the line is held by injection evidence alone -- `--break-runnerup` (the pin"
    " decided by 11.00 ULP while the engine lifted its runner-up one BF16 step over the pin's max) is"
    " RED here, and `--selftest`'s reading-split case requires the two readings to disagree on that"
    " same row. A threshold that cannot name a measured margin has to say so; borrowing the"
    " cross-width margin below would be the same unaccountable number wearing a better coat")


def flip_room_line(pairing: str = "cross-width") -> str:
    """The flip line, its margin, and WHICH pairing the margin belongs to, as one sentence.

    Same purpose as `wave_row_1cos_band()`'s provenance clause — a band that does not name its
    denominator cannot be re-checked — with one addition the per-row band does not need: the two room
    readings are calibrated differently, so a flip line without its pairing would let the reference
    path borrow a margin it never earned. `pairing` is "cross-width" (two engine sides, no reference,
    margin measured by the census) or "reference" (pin vs engine, no real-corpus margin to quote).
    """
    val = f"{CASCADE_BAND['flip_room_max_ulp']:.2f} BF16 ULP"
    if pairing == "reference":
        return (f"{val}, measured on the REFERENCE's own top-1/top-2 gap (the min-over-sides reading"
                " belongs to pairings with no reference, and using it here would let an injected"
                " one-step lift read as a coin toss). " + FLIP_ROOM_REFERENCE_SOURCE)
    src = FLIP_ROOM_SOURCE.get(GEMM_ROUTE)
    return (f"{val} at the row's own top-1"
            + (f" — {src}" if src else
               " (not in force on the hand route; no route delta to size it on)")
            + ". " + FLIP_ROOM_CENSUS)

# Where each wave-width row band came from, in the gate's own output rather than in a reader's memory.
# A band that does not name its denominator is a band nobody can re-check — see `prototype/attn_tile_model.py`'s
# rule that a frozen form must be spelled with its own words, and `tools/route_c_audit.py`'s A2 for the
# leg that predates this file. The measured MAX lives here and the band lives in CASCADE_BAND, so the
# number that gates appears once.
WAVE_ROW_BAND_SOURCE = {
    "library": ("hand GEMV (--wave-rows 1) vs hipBLAS (--wave-rows 512): D-11's A2 measured max"
                " 1.8889e-03 over 1694 rows, line at +6 %; re-measured 2026-10-16 by"
                " engine/wave_band_floor_probe.py at 1.888942e-03 over 9 fixtures/2951 rows, so this"
                " leg is UNCHANGED — and it is pin-independent by construction: D27 keeps the whole"
                " `library` route on hipBLAS' default algorithm choice, so no `--gemm-pins` set can"
                " move a number on this leg"),
    "lt": ("hand GEMV (--wave-rows 1) vs hipBLASLt (--wave-rows 512): measured max 2.204122e-03"
           " (dec64f/logits45.bin) over 9 fixtures/2951 rows; the line 2.75e-03 is that max x 1.2476"
           " — the probe's stated 1.25 margin would owe 2.755e-03, and the line is kept at the tighter"
           " round number below it; engine/wave_band_floor_probe.py 2026-10-16 with `--gemm-pins all`"
           " against the SHIPPING algo table: engine/kernels_gemm.cpp `algo_pins()`, which held ONE"
           " entry then (`down_proj` 1024x3584 @ M=512, D28) and holds TWO since D35, restated by D36"
           " (`deep_gate_proj+up_proj` 3584x1024 and `down_proj` 1024x3584, both @ M=512). The"
           " first version of this line named the three-entry table the binary no longer carries;"
           " re-taken so provenance matches the build. No pin can reach the row that carries THIS"
           " number (dec64f's widest body is 17 rows, pins bind only at M=512), which is why the trim"
           " left the max where it was — and why the promotion does not either. What the promotion does"
           " move is the 512-row fixtures, and those were priced for exactly this two-pin set in"
           " §3e of prototype/gemm_algo_sweep/results.md at 1.2446e-04 on wide640f, i.e. 22x inside"
           " this line, so nothing here widens; a fresh"
           " `engine/wave_band_floor_probe.py --gemm-pins all` under the two shipped pins is the"
           " re-take that would restate the table itself)."),
}


def wave_row_1cos_band() -> tuple[float, str]:
    """(the per-row `1-cos` band for the pairing this route creates, the sentence naming its provenance).

    Section (h) compares two runs of one prompt at two widths, and on a library route the one-row side
    is the hand GEMV whatever `--gemm` says (`resolve_wave_gemm`). So the pairing — and therefore the
    band — is a function of the route, not a constant: hand-vs-hipBLAS under `library`,
    hand-vs-hipBLASLt under `lt`. `hand` has no band at all (byte-identity is what that route owes) and
    never reaches this function; section (h) runs its byte comparison instead.
    """
    if GEMM_ROUTE == "lt":
        return CASCADE_BAND["row_1cos_max_lt"], WAVE_ROW_BAND_SOURCE["lt"]
    return CASCADE_BAND["row_1cos_max"], WAVE_ROW_BAND_SOURCE["library"]

# Which GEMM route the run being judged used (`decode_run --gemm`). Module state for the same reason
# `REFS` is: five sections need it and none of them owns it. `library` is the default because it is the
# shipped route now — D-11 admitted it on the audit's terms, which are a band and not bit-identity — and
# the cascade floor below is armed for it and for nothing else. `hand` stays fully supported and stays
# judged the old way: with this set to `hand`, every band in this file means exactly what it meant before
# route (c). This default and `decode_run`'s move together, in one commit, because a gate that judged
# library output under hand bands would be judging the wrong rule.
GEMM_ROUTES = ("hand", "library", "lt")
GEMM_ROUTE = "library"


def set_route(name: str) -> None:
    """Declare which GEMM route the run being judged used. Refuses anything but the real ones,
    because a typo here would silently judge a library run by the hand bands."""
    global GEMM_ROUTE
    if name not in GEMM_ROUTES:
        raise SystemExit(f"gate: --gemm takes {'|'.join(GEMM_ROUTES)}, got {name!r}")
    GEMM_ROUTE = name


# Which hipBLASLt algo pins the run being judged was given (`decode_run --gemm-pins`), in the same
# module-state shape as `GEMM_ROUTE` and for the same reason: which algorithm served which shape is part
# of what its bits are, so a log that says `lt` but not WHICH `lt` cannot be cited for a pin claim. This
# is the attribution hole D28 left open: `wide520f --gemm lt` read one red row and the gate had no knob
# to ask whether the pinned kernel or the heuristic's rank-0 produced it. `all` is the default because
# that is `decode_run`'s own default (`kernels_gemm.cpp::gemm_pins_select`), and the two defaults move
# together for `--attn-tile`'s reason: a gate that judged a different config than the engine ships is
# judging the wrong binary.
GEMM_PINS = "all"


def parse_pins(spec: str) -> list[str]:
    """The shape a `--gemm-pins` spec selects, spelled the way the engine spells it: `all` is the `"*"`
    sentinel `gemm_pins_select` uses, `none` is the empty list, anything else is the shape names.

    Parsed here rather than only forwarded, for the engine's own reason: a spec that silently meant "no
    pins" would price rank-0 twice and report the pin as a null. Which names EXIST is the engine's
    question — it holds `algo_pins()` and refuses an unknown name before any launch — so this adjudicates
    the shape of the spec and the count it implies, which `gemm_pin_checks` then compares with the run's
    own echoed `pins=`.
    """
    if spec == "all":
        return ["*"]
    if spec == "none":
        return []
    names = spec.split(",")
    if any(n == "" for n in names):
        raise SystemExit(f"gate: --gemm-pins {spec!r} has an empty entry — use all, none, or a"
                         " comma-separated list of pinned shape names")
    return names


def pin_expectation(spec: str, table: int) -> tuple[int, str]:
    """(how many pins `spec` puts in force, the clause saying why) against a table `table` wide.

    The table size is an ARGUMENT because it comes from the run's own `pins=<allowed>/<total>` echo and
    not from a constant in this file. That is the whole point: with `--gemm-pins all` the only honest
    expectation is the relation `allowed == total`, and a hardcoded table size here would be a second
    authority over a table owned by `algo_pins()` — the exact drift that put a measured band on the
    wrong pin set (prototype/gemm_algo_sweep/results.md §3d, D28).
    """
    names = parse_pins(spec)
    if names == ["*"]:
        return table, "`all` asks for every SHIPPED entry the binary has"
    if not names:
        return 0, "`none` asks for the heuristic's rank-0 everywhere"
    return len(names), f"a list of {len(names)} name(s) asks for exactly those"


def set_pins(spec: str) -> None:
    """Declare the algo-pin set the run being judged was given. Runs after `set_route`, because the two
    are not independent: a pin IS a hipBLASLt algo record, so `--gemm-pins down_proj` on a `library` or
    `hand` run selects nothing, and a gate that let it through would spend the run claiming a pin set the
    binary cannot honour. Refused here, not judged silently."""
    global GEMM_PINS
    names = parse_pins(spec)
    if names != ["*"] and GEMM_ROUTE != "lt":
        raise SystemExit(f"gate: --gemm-pins {spec!r} names hipBLASLt algo pins, but --gemm"
                         f" {GEMM_ROUTE!r} runs no hipBLASLt at all. The knob would be a no-op and the"
                         " log would claim a pin set nothing honoured; leave `--gemm-pins all` on this"
                         " route, or run `--gemm lt`.")
    GEMM_PINS = spec


def route_is_library() -> bool:
    """Is the cascade floor in force? True for every library route and for nothing else.

    `lt` joins `library` because the band is sized by the same thing: both routes replace a hand GEMV's
    fixed reduction order with a library kernel's, so the wave-boundary check is inside a band on either
    and byte-identical on neither. `lt` is not simply `library` though — it is a DIFFERENT library, so a
    run under `lt` is a separate judgment with its own log, not a re-read of a `library` one. The hand
    route keeps every cap in this file exactly where it was before route (c)."""
    return GEMM_ROUTE in ("library", "lt")


fails: list[str] = []
fail_names: list[str] = []    # the check NAME of every failure, for the D-9 expected-red split below
notes: list[str] = []
ties: list[str] = []        # one entry per generation step resolved by the tie policy; counted in report()
tie_notes: list[str] = []   # tied steps the engine matched outright: printed, NOT policy decisions (D-1)
slack: list[str] = []       # steps admitted by the quantization-slack rule; counted in report()
slack_notes: list[str] = [] # steps inside the slack region the engine matched outright: notes, not uses
rooms: list[str] = []       # flips admitted by the cascade room rule (library route only); counted

# ---- the judging scope ------------------------------------------------------------------------
# What a run is being asked to adjudicate. `full` (the default) is the ordinary gate: a check that
# fails is a gate failure. `form` is §16/§17's attention-FORM A/B — one fixture, the same pinned rows,
# run once per `--attn-tile`, asking which form is right above MAXSC.
#
# Why a scope has to exist at all. §17's `--attn-tile 1` run came back with 26 failures headed by
# section (h)'s wave-width comparisons ("batched and carry ... 1032 rows, 0 of them still
# bit-identical"). Those checks vary the WAVE WIDTH; the A/B varies the attention FORM. A check whose
# independent variable is not the thing under test cannot answer the question being asked, and worse,
# it buries the two readings that could. Two ways out were on the table:
#
#   (a) run each wave-width comparison against the same form on both sides. This is ALREADY true —
#       `run_batched_ab` passes `--attn-tile` to every side precisely so that an A/B of widths cannot
#       smuggle a form difference in — so it cannot help: the defect is attribution, not a mismatched
#       pair of sides.
#   (b) DEFER them under `--scope form`: still run, still printed, still recorded in a file beside the
#       dumps — but not counted into that run's verdict, and labelled as belonging to neither form.
#
# (b) is what is implemented, and it is not a silencer: the checks run on every side, their lines print
# as `[DEFERRED-PASS]`/`[DEFERRED-FAIL]`, every one of them (pass or fail) goes to
# `wave_carry_deferred.json` beside the run's dumps so the two forms' carry results can be diffed
# mechanically, and the summary says how many were deferred rather than printing a bare PASS. A carry
# check that genuinely fails identically on both forms therefore shows up as what it is — a separate
# finding about the wave carry, not a verdict about either form. `--scope full`, the default and what
# every ordinary gate run uses, still counts every one of them as a failure.
SCOPES = ("full", "form")
SCOPE = "full"
# The reason attached to the `defer()` block currently open; None when nothing is being deferred.
DEFER_REASON: str | None = None
deferred: list[str] = []           # failed checks that ran, printed, and are attributed elsewhere
deferred_records: list[dict] = []  # every check inside a defer block, pass or fail, for the sidecar

# One spelling of the reason, so both deferred regions of section (h) say the same thing.
WAVE_CARRY_REASON = ("wave-width claim, not a form claim: this comparison holds the attention form "
                     "fixed on both sides and varies --wave-rows, so under --scope form it cannot "
                     "adjudicate which form is wrong above MAXSC")


def set_scope(name: str) -> None:
    """Declare what this run is adjudicating. Refuses anything else for `set_route`'s reason: a typo
    here would wave away a class of failures by spelling its scope wrong."""
    global SCOPE
    if name not in SCOPES:
        raise SystemExit(f"gate: --scope takes {'|'.join(SCOPES)}, got {name!r}")
    SCOPE = name


@contextlib.contextmanager
def defer(reason: str, *, on: bool = True):
    """Run the checks inside, but attribute a failure to `reason` instead of to this run's verdict.

    Only ever opened around checks whose independent variable is not the thing under test. It cannot
    make a check vacuous: the block runs the check, prints its line, records it in the sidecar, and
    counts it in the summary as deferred.
    """
    global DEFER_REASON
    prev = DEFER_REASON
    DEFER_REASON = reason if on else None
    try:
        yield
    finally:
        DEFER_REASON = prev


# ---- the echo attribution ------------------------------------------------------------------------
# A FREE-RUNNING run that takes an ADMITTED thin-margin token at step N is, from step N+1 on, answering
# a different question than the pin: the pin at step N+k was computed on the reference's own chain and
# this run was computed on its own. §10.5's alignment rule already says a comparison must name which
# alignment it used; past an admitted flip the two sides share no alignment at all. `prefix24dec8
# --gemm lt --attn-tile 1` was 19 red lines for ONE coin flip the slack rule admits at step 2: the flip,
# the id it caused, the fed-chain line, 11 band failures at steps 3..7 whose pins were computed on
# tokens this run was never fed, and 4 cross-width lines that compared a forked wave with an unforked
# one. A storm that reads as 19 defects is how a real defect hides, so the downstream lines are named
# ECHO: printed, counted, and attributed to the flip that caused them.
# An ECHO is NOT a waiver. The step is still judged, at identical conditioning, in section (b2), which
# feeds the reference's own tokens and has no echo window; that is where a real numeric defect has to be
# caught (see `--selftest`'s late-injection case).
echoes: list[str] = []                       # echoed checks that FAILED
echo_records: list[dict] = []                # every echoed check, pass or fail, for the sidecar
ECHO_REASON: str | None = None               # set by `echo()`; None means a failure is a failure
ECHO_WINDOW: str = ""                        # what this run's echo window is, for the verdict line
ECHO_FILE_NAME = "echo_attribution.json"     # the sidecar, beside `wave_carry_deferred.json`


@contextlib.contextmanager
def echo(reason: str, *, on: bool = True):
    """Attribute every check failure inside the block to an admitted divergence upstream, and say so.

    The sibling of `defer()`, same shape on purpose: the attribution must be impossible to lose at a
    call site, and the checks still run, still print, and still land in a sidecar. `on=False` is the
    ordinary path, which is what lets one call site serve both an echoed step and an un-echoed one.
    A block that is both deferred and echoed goes in the echo bucket — it names a concrete cause, where
    `defer` only names a scope — and its record is written to both sidecars so neither attribution loses
    the line."""
    global ECHO_REASON
    if not on:
        yield
        return
    ECHO_REASON = reason
    try:
        yield
    finally:
        ECHO_REASON = None


def echo_first_step(admitted) -> int | None:
    """The first generation step DOWNSTREAM of an explicitly admitted divergence, or None.

    Pure, so the classifier is testable in process. `admitted` holds the step indices whose engine-vs-pin
    divergence one of the three proved rules ADMITTED (tie policy, quantization slack, cascade room);
    `PREFILL_ROW` — §06.5's first answer token — counts as step -1 because everything after it is
    downstream. The answer is `min(admitted) + 1`: the flip's OWN step is still judged normally, because
    up to and including that step the run was fed the pin's own tokens. An empty list returns None, and
    None is the case that must keep every line red — "nothing was admitted" is the only state in which a
    divergence has no attribution and each band failure is a real defect."""
    real = [j for j in admitted if j is not None]
    return None if not real else min(real) + 1


def echo_reason(step: int, engine_id, pin_id, rule: str) -> str:
    """The sentence an echoed line must carry: which flip, which tokens, which rule admitted it."""
    where = "the prefill row (§06.5's y0)" if step == PREFILL_ROW else f"step {step}"
    return (f"downstream of the admitted flip at {where}: the engine was fed {engine_id} where the pin"
            f" was computed on {pin_id} (admitted by the {rule}). From the next step on the two sides"
            " are answering different questions (§10.5's alignment rule), so a divergence there is an"
            " ECHO of this flip, not a second defect — section (b2) judges the same steps with the"
            " reference's own tokens fed, and has no echo window")


def check(ok: bool, name: str, detail: object = "") -> bool:
    # `detail` is stringified rather than typed as str because several call sites pass a list of ids,
    # which is exactly the thing an operator needs to see when a step fails.
    text = detail if isinstance(detail, str) else repr(detail)
    ok = bool(ok)
    # Inside a `defer()` block the line still prints and is still recorded; only the bucket changes, and
    # the tag says so so that no reader can mistake a deferred line for a pass or quietly for a failure.
    # `ECHO` is the third bucket and means the same thing about the verdict with a different cause: the
    # check ran, printed, is recorded, and is NOT counted as an independent failure.
    tag = (("[ECHO-PASS] " if ok else "[ECHO-FAIL] ") if ECHO_REASON is not None else
           ("[DEFERRED-PASS] " if ok else "[DEFERRED-FAIL] ") if DEFER_REASON is not None
           else ("[PASS] " if ok else "[FAIL] "))
    print("  " + tag + name + ((" — " + text) if text else ""))
    if DEFER_REASON is not None:
        deferred_records.append({"check": name, "ok": ok, "reason": DEFER_REASON, "detail": text})
    if ECHO_REASON is not None:
        echo_records.append({"check": name, "ok": ok, "reason": ECHO_REASON, "detail": text})
        if not ok:
            print("           [ECHO CAUSE] " + ECHO_REASON)
    if not ok:
        line = name + (": " + text if text else "")
        if ECHO_REASON is not None:
            echoes.append(line)
        elif DEFER_REASON is None:
            fails.append(line)
            fail_names.append(name)
        else:
            deferred.append(line)
    return ok


def section(title: str) -> None:
    print("\n== " + title)


def load_pin(name: str) -> np.ndarray:
    """One pinned dump. numpy has no BF16, so a uint16 payload is widened by the bit shift (§10.5: a
    bitcast, never a numeric cast — `Tensor.to(uint16)`-style truncation is the bug that ate the first
    §04.5 device run)."""
    path = REFS / name
    if not path.exists():
        raise SystemExit(f"gate: pinned dump missing: {path}")
    return widen_pin(np.load(path, allow_pickle=False))


def widen_pin(a: np.ndarray) -> np.ndarray:
    """A pinned array as floats: a uint16 payload is BF16 bits and gets `load_pin`'s bit shift.

    One spelling of the widening because two would be two chances to get it wrong: the per-step logits
    pins and the [rows, vocab] `logits_rows` block are read through here, and a row block read with a
    numeric cast instead of a bitcast is the `gdn_gated_norm` garbage-memory trap (§10.5) at 485 MiB.
    """
    if a.dtype == np.uint16:
        return (a.astype(np.uint32) << 16).view(np.float32)
    return a


gapset: frozenset = frozenset()   # the FIXTURE's declared logit gaps, as step indices; set by
                                  # `pairing_checks` from the manifest, because section (f2) runs
                                  # later and must not re-read the manifest for it


def declared_logit_gaps(man: dict) -> list:
    """The steps this fixture DECLARES it did not pin logits for, as sorted step indices.

    `tools/reference_dump.py --logit-steps` keeps a long chain's logits at a chosen set of steps and
    writes every step it dropped into `logits_declared_absent`, one entry per step — never a count, a
    range or a ratio, because "keep 1 in 8" expressed as a policy is a weaker gate worn as a knob.
    A fixture made before the flag has no key and therefore declares nothing, which is the correct
    answer: it has no gaps either.
    """
    raw = man.get("logits_declared_absent", []) or []
    steps = []
    for tag in raw:
        try:
            steps.append(int(str(tag).removeprefix("decode")))
        except ValueError:
            raise SystemExit(f"gate: {REFS.name}: logits_declared_absent names {tag!r}, which is not"
                             " a step tag like 'decode7'. Every gap has to be named per step, so this"
                             " gate refuses to guess what the list means")
    return sorted(set(steps))


def first_pinned_logits(dec_logits: list):
    """The first step's pinned logits vector, or None if the fixture declared every step away. Only
    ever asked for a dtype/shape to print, so a fixture with no per-step logits gets a sentence here
    instead of an IndexError on an empty list."""
    return next((lg for lg in dec_logits if lg is not None), None)


def argmax_low(a: np.ndarray) -> int:
    """§03.4: greedy argmax, ties to the LOWEST id. numpy's argmax returns the first maximum, which
    is the lowest index; spelled out because the C++ side has to do the same with a strict `>`.
    Note what that means at a tie: the id it returns is a property of the tie-BREAK, not of the model
    — see the TIE POLICY block above and `pin_tie` below."""
    return int(np.argmax(a))


def bf16_payload(x) -> np.ndarray:
    """F32 -> BF16 payload, round-to-nearest-even, done on the bits (no torch, no numeric cast).

    On a value that already IS BF16 this is a bitcast in disguise: it returns the payload it went in
    with, which is what makes it usable on a pin stored as F32 (dec8) as well as on the engine's F32
    logits. `engine/pin_bits_probe.py` is the measurement that the pinned logits are BF16-exact (0
    values moved in either fixture); a fixture that stopped being so says so in `pin_tie`.
    """
    u = np.asarray(x, dtype=np.float32).reshape(-1).view(np.uint32)
    lsb = (u >> 16) & 1                       # RNE, not truncation: add half an ULP plus the lsb
    return ((u + 0x7FFF + lsb) >> 16).astype(np.uint16)


def bf16_bump(vec: np.ndarray, index: int) -> np.ndarray:
    """A copy of `vec` with element `index` moved to the NEXT BF16 value up, on the bits.

    The smallest change that leaves the argmax alone while moving the value out of its own BF16 bin,
    which is what the D-1 selftest injection needs: an engine whose F32 logit at the id it chose rounds
    1 ULP above the pinned payload (`mt32f` step 11). Steps up in payload space, and refuses a negative
    rather than silently stepping it DOWN — payload order is numeric order only for non-negatives.
    """
    out = np.array(vec, dtype=np.float32, copy=True).reshape(-1)
    x = float(out[index])
    if not np.isfinite(x) or x < 0:
        raise SystemExit(f"gate: the ULP bump only knows how to step a non-negative value up"
                         f" (asked for index {index}, value {x!r})")
    payload = int(bf16_payload(x)[0]) + 1
    out[index] = np.array([payload << 16], dtype=np.uint32).view(np.float32)[0]
    if not float(out[index]) > x:      # a bit trick that went sideways is worse than no injection
        raise SystemExit(f"gate: the ULP bump at index {index} moved {x!r} to"
                         f" {float(out[index])!r}, which is not one BF16 ULP up")
    return out.reshape(np.shape(vec))


def logits_pin(step: int) -> str:
    """The pin file carrying step `step`'s logits. `PREFILL_ROW` is §06.5's `y0` row, pinned bare."""
    return "logits_last.npy" if step == PREFILL_ROW else f"decode{step}.logits_last.npy"


def pin_tie(step: int):
    """The exact-maximum tie set of one pinned logits dump, or None if that step has a plain winner.

    Everything is on the PIN'S payload. The step's pin — `logits_pin(step)`, so `decode{j}.logits_last.npy`
    or the bare `logits_last.npy` at `PREFILL_ROW` — is read raw; the payload is the
    stored uint16 (or, for an older F32-stored fixture, the BF16 payload of a value the probe already
    proved is BF16-exact); the winner is located by widening that payload with `load_pin`'s bit shift,
    because BF16 payload order is not numeric order once the sign bit is set, so the MAX has to be
    taken on widened floats while MEMBERSHIP is proved on the payload. Ties are payload equality, never
    a float tolerance.
    """
    path = REFS / logits_pin(step)
    raw = np.load(path, allow_pickle=False).reshape(-1)
    stored = raw.dtype == np.uint16
    payload = raw if stored else bf16_payload(raw)
    wide = (payload.astype(np.uint32) << 16).view(np.float32)   # load_pin's bit-shift widening
    if not stored and not np.array_equal(wide, raw.astype(np.float32)):
        notes.append(f"{path.name}: its pinned F32 logits are not BF16-exact, so a payload-space tie"
                     " claim would not be sound at this step. Re-pin with --logits-bits bf16"
                     " (engine/pin_bits_probe.py is the probe that prices this).")
        return None
    top = wide.max()
    ids = [int(i) for i in np.flatnonzero(wide == top)]
    if len(ids) < 2:
        return None
    return {"ids": ids, "value": float(top), "payload": int(payload[ids[0]]),
            "top1": ids[0], "payload_is_stored": stored}


def tie_slack_bound(step: int, row=None):
    """`(bound, the sentence naming it)` for a step whose pin is an exact tie, or None.

    The same licensed element error D-9 sizes its slack with — that step's meanrel cap times
    `mean|pin|` — read here because `step_slack` deliberately refuses a step with no unique winner
    ("no unique winner: D-1's proved-tie predicate owns this step"), and the tie predicate therefore has
    no way to size slack on its own. One reader of the cap, two consumers of the number.
    """
    cap_src = meanrel_cap(step, row)
    if cap_src is None:
        return None
    cap, cap_src = cap_src
    wide = np.asarray(pin_row(step), dtype=np.float32).reshape(-1)
    meanabs = float(np.abs(wide).mean())
    label = ("the prefill row (§06.5's y0)" if step == PREFILL_ROW else f"step {step}")
    return (cap * meanabs, f"{label}: this step's licensed element error {cap * meanabs:.6g} ="
                           f" meanrel cap {cap:.4%} x mean|pin| {meanabs:.6g} ({cap_src})")


def tie_verdict(step: int, eng, sampled: int) -> tuple[str, bool, str]:
    """Decide one judged step against the tie policy: `(kind, admissible, the line to print)`.

    `step` may be `PREFILL_ROW` — §06.5's `y0` row goes through this same predicate (D-8), which is what
    `chunk64b` proved: its pinned prefill row ties ids 760 and 1596 at 19.875, and an engine landing on
    1596 with a logit that rounds back to that payload has done nothing illegal.

    The single predicate for the whole file — section (a) gates on it and section (b) reuses the answer
    for its argmax line — so exactly one place decides what an engine may do at a tied step. Three
    kinds, because the policy is an exception and an exception is only needed by an engine that deviated
    from the pin (D-1, see the TIE POLICY block):

      "no-tie" — the pin has a single winner here, so the policy has nothing to say and
                 section (a) does not even print; `admissible` is False because nothing was admitted.
      "match"  — the engine's id IS the pin's top-1 id. That is G7 satisfied outright: an ordinary
                 exact match, the rule is not consulted, and the line only records that the step
                 happened to be tied. Still not `admissible`, because there was nothing to admit.
      "tie"    — the engine picked some other id at a tied step. Only now do the clauses apply, and
                 the predicate fails closed: no engine logits vector, an id outside the tied set, or a
                 logit that does not round back to the shared tied payload all mean not admissible.
    """
    tie = pin_tie(step)
    if tie is None:
        return "no-tie", False, ""
    label = ("the prefill row (§06.5's y0)" if step == PREFILL_ROW else f"step {step}")
    if sampled == tie["top1"]:
        return "match", False, (
            f"{label}: the pin is an exact {len(tie['ids'])}-way tie at {tie['value']!r}"
            f" (payload 0x{tie['payload']:04x}) among ids {tie['ids']}, and the engine picked the"
            f" pinned top-1 {tie['top1']} — an ordinary id match, so the tie policy is not consulted")

    def shown(i: int) -> str:
        if eng is None:
            return f"{i} -> (no engine logits)"
        v = float(eng[i])
        return f"{i} -> {v:.8g} (bf16 0x{int(bf16_payload(v)[0]):04x})"

    head = (f"TIE {label}: pinned {tie['value']!r} (payload 0x{tie['payload']:04x}) tied"
            f" {len(tie['ids'])} ways among ids {tie['ids']}; engine's value at each tied id:"
            f" " + ", ".join(shown(i) for i in tie["ids"]))
    if eng is None:
        return "tie", False, head + " | NOT ADMISSIBLE: no engine logits vector, so the tie cannot be proved"
    if sampled not in tie["ids"]:
        return "tie", False, (head + f" | engine id {sampled} is NOT one of the tied ids"
                              " | NOT ADMISSIBLE")
    got = int(bf16_payload(float(eng[sampled]))[0])
    held = got == tie["payload"]
    if held:
        return "tie", True, (head + f" | engine id {sampled} is one of the tied ids; engine's logit at"
                             f" that id = {float(eng[sampled]):.8g} -> bf16 0x{got:04x} == the shared"
                             " tied payload")
    # The payload equality above was written when the engine and the pin were the same arithmetic, and it
    # is a stricter question than the policy is about: the clause exists to prove the engine really
    # landed on the tied pair, not to demand it reproduce the pinned BITS. `chunk64b` step 11 is where
    # that shows — pinned 14.75 (0x416c) tied between 17024 and 19177, a `--gemm library` engine sitting
    # at 14.6945 (0x416b) on the id it picked, one ULP below the payload for no reason except that a
    # different dot order summed to it. So: same id set, and the distance from the pinned maximum judged
    # by D-9's own measure — the element error this step's band already licenses. An id off that bound is
    # still refused, and `--tie-far` is the injection that proves the bound bites.
    bound = tie_slack_bound(step)
    if bound is None:
        return "tie", False, (head + f" | engine id {sampled} is one of the tied ids; engine's logit at"
                                     f" that id = {float(eng[sampled]):.8g} -> bf16 0x{got:04x} !="
                                     f" the shared tied payload 0x{tie['payload']:04x}, and this step"
                                     " has no priced meanrel band, so slack cannot be sized"
                                     " | NOT ADMISSIBLE")
    lim, src = bound
    off = abs(float(eng[sampled]) - float(tie["value"]))
    ok = off <= lim
    return ("tie", ok, head + f" | engine id {sampled} is one of the tied ids; engine's logit at"
                              f" that id = {float(eng[sampled]):.8g} -> bf16 0x{got:04x} !="
                              f" the shared tied payload 0x{tie['payload']:04x}, and it is {off:.6g}"
                              f" from the pinned tied value — {src}"
                              + (f" | ADMITTED as a tie within quantization slack (D-9's measure,"
                                 " generalised to an exact tie: the id is one the pin tied at its own"
                                 " maximum, and the engine's value there is inside the error this"
                                 " step's band licenses)" if ok else
                                 " | NOT ADMISSIBLE: further from the pinned maximum than this step's"
                                 " band licenses, so this is not the pinned pair with a different"
                                 " rounding, it is a different value"))


def meanrel_cap(step: int, row=None):
    """`(cap, the sentence naming where the cap comes from)` for step `step`, or None if there is none.

    The cap is the meanrel band this gate already polices at that step — `rule_for_step`'s second slot —
    read here so the slack rule can quote the same number the band line quotes. `row` is the fixture's
    cached allowance row when the caller already has one; when it is None the file is read quietly with
    `allowance_row`, which is what the CPU-side injections need (they run before section (b), and a
    stale allowance must refuse rather than double-report the same FAIL). `PREFILL_ROW` gets None on
    purpose: section (b) judges no vector band on §06.5's row, so there is no licensed element error
    there and the slack rule has nothing honest to compare a margin against.
    """
    if step == PREFILL_ROW:
        return None
    if row is None:
        row = allowance_row(step)
    if row is None:
        cap, src = (BAND["meanrel_hard"],
                    f"flat §10.2 cascade ceiling {BAND['meanrel_hard']:.2%}, because {REFS.name} carries"
                    f" no {ALLOWANCE_FILE}")
    else:
        cap, src = (float(row["allowance_meanrel"]),
                    f"{REFS.name}'s own meanrel allowance at step {step}")
    if route_is_library() and cap < CASCADE_BAND["stage_meanrel_max"]:
        # The same floor `rule_for_step` applied to the band line, applied here too, so the slack rule
        # can never be narrower than the band that licenses it: on the library route the licensed
        # element error at a step is never smaller than the measured cascade.
        cap = CASCADE_BAND["stage_meanrel_max"]
        src += (f", floored at the measured cascade {cap:.0%} because this run is --gemm library"
                " (CASCADE_BAND)")
    return (cap, src)


def step_slack(step: int, ref, row=None):
    """The quantization-slack facts for one judged step, or None when the rule does not apply.

    Returns `{top1, top2, v1, v2, margin, ulps, bound, cap, cap_src, meanabs, leaders, numbers}` for a
    step whose pin has a UNIQUE winner and a policed meanrel band, and None otherwise — None is the
    fail-closed answer, and it is what keeps D-1's proved-tie predicate (a pin with no winner at all) and
    §06.5's `y0` row out of this rule's reach instead of being quietly folded into it.

    `ref` is the step's PINNED logits as `load_pin` widens them. The margin is a question about values,
    not about payloads — payload equality is what a tie means, and that question is `pin_tie`'s — so
    widening is the right reading here and it avoids re-reading a 1 MB tensor the caller has already
    loaded. `numbers` is the sentence with every figure on it, built once so the admitting line and the
    refusing line cannot quote different numbers.
    """
    cap_src = meanrel_cap(step, row)
    if cap_src is None:
        return None
    cap, cap_src = cap_src
    wide = np.asarray(ref, dtype=np.float32).reshape(-1)
    order = np.argsort(wide)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    v1, v2 = float(wide[top1]), float(wide[top2])
    margin = v1 - v2
    if margin <= 0.0:
        return None                      # no unique winner: D-1's proved-tie predicate owns this step
    meanabs = float(np.abs(wide).mean())
    bound = cap * meanabs
    # `order` is the id axis sorted by value, so the mask has to be taken on the SAME axis: `wide[order]`
    # is the descending value list and `order[mask]` is the ids it belongs to. Masking `order` with a mask
    # computed on `wide` (the id axis) is the kind of one-line slip that prints a plausible pair of wrong
    # ids — at chunk64b step 0 it named 74123 and 57529 instead of 8337 and 20619.
    leaders = [int(i) for i in order[wide[order] >= v1 - bound]]
    label = ("the prefill row (§06.5's y0)" if step == PREFILL_ROW else f"step {step}")
    numbers = (f"{label}: pin top1 {top1} ({v1}) over top2 {top2} ({v2}) — margin {margin:.6g}"
               f" = {margin / bf16_ulp(v1):.1f} BF16 ULP at that magnitude, against this step's"
               f" licensed element error {bound:.6g} = meanrel cap {cap:.4%} x mean|pin|"
               f" {meanabs:.6g} ({cap_src})")
    return {"top1": top1, "top2": top2, "v1": v1, "v2": v2, "margin": margin,
            "ulps": margin / bf16_ulp(v1), "bound": bound, "cap": cap, "cap_src": cap_src,
            "meanabs": meanabs, "leaders": leaders, "numbers": numbers, "label": label}


def slack_verdict(sl: dict, eng, sampled: int) -> tuple[bool, str]:
    """`(admissible, the line to print)` for one step already known to sit inside the slack region.

    Fails closed on a missing engine logits vector (the choice cannot be attributed) and on an id outside
    the pin's leader set. The engine's own values at the leaders are printed either way, with its own
    gap between them, so a reader can see whether the engine saw the pair as near-tied; that gap is
    printed and not gated, exactly as D-1 prints the engine's value at every tied id without gating it.
    """
    shown = ("(no engine logits)" if eng is None else
             "; engine's values: " + ", ".join(f"{i} -> {float(eng[i]):.8g}" for i in sl["leaders"]))
    head = "SLACK " + sl["numbers"]
    if eng is None:
        return False, (head + " | NOT ADMISSIBLE: no engine logits vector, so the engine's choice here"
                       " cannot be checked against the margin")
    if sampled not in sl["leaders"]:
        return False, (head + f" | engine id {sampled} is NOT one of the pin's {len(sl['leaders'])}"
                       f" ids inside that margin {sl['leaders']}" + shown + " | NOT ADMISSIBLE")
    gap = float(eng[sampled]) - float(eng[sl["top1"]])
    return True, (head + f" | the pin could not decide this token: engine id {sampled} is one of the"
                  f" {len(sl['leaders'])} pinned ids inside the margin {sl['leaders']}" + shown
                  + f"; the engine's own gap at those two ids is {gap:.6g} (printed, not gated — which"
                    " way a legal route's coin landed is not a band)"
                  + " | ADMITTED as tie-by-quantization-slack")


def read_one_logits(work: Path, index: int, vocab: int) -> tuple:
    """One row's engine logits as `(array | None, why_not)`. Size-checked: a truncated dump read as a
    logits vector fails a step for the wrong reason."""
    path = work / f"logits{index}.bin"
    if not path.exists():
        return None, "file missing — check --out"
    a = np.fromfile(path, dtype="<f4")
    return (a, "") if a.size == vocab else (None, f"{a.size} floats, expected {vocab}")


def read_engine_logits(work: Path, n_decode: int, prompt_len: int, vocab: int) -> list:
    """Every generation step's full logits vector, read once, before the section that needs them.

    Section (a) needs them because the tie policy is proved against the engine's OWN numbers, not
    around them, and a tie that cannot be proved has to fail at (a) rather than be waived. Entries are
    `(array | None, why_not)`; (b) keeps naming its own reasons so a missing dump still reads as one.
    """
    return [read_one_logits(work, prompt_len + j, vocab) for j in range(n_decode)]


def tie_other(step: int, flag: str = "--tie-alt") -> int:
    """The pin's second tied id — the legitimate alternative the tie policy exists to admit. Used by
    `--tie-alt` (and `--y0-alt`, on `PREFILL_ROW`) so the SELFTEST can prove the policy admits, not only
    that it refuses."""
    tie = pin_tie(step)
    if tie is None:
        raise SystemExit(f"gate: {flag} {step} names a step with no exact tie in the pin"
                         f" ({logits_pin(step)}), so the injection would prove nothing (the"
                         " vacuous-negative trap `prototype/mlp_gemv/results.md` warns about)")
    return tie["ids"][1]


def untied_id(step: int) -> int:
    """An id that is NOT in the pin's tie set — what an engine must never produce at a tied step.
    Deterministic (the lowest id whose pinned value is not the tied maximum), and it refuses a step
    that is not tied, for the same reason as `tie_other`."""
    tie = pin_tie(step)
    if tie is None:
        raise SystemExit(f"gate: --break-tie {step} names a step with no exact tie in the pin; that"
                         " injection would just be the ordinary wrong-id test wearing a tie costume"
                         f" ({logits_pin(step)})")
    wide = load_pin(logits_pin(step))
    return int(np.flatnonzero(wide != np.float32(tie["value"]))[0])


def y0_wrong_id() -> int:
    """An id the pinned prefill row does NOT put at its maximum — `--break-y0`, D-8's negative. Unlike
    `untied_id` this one is legal on an untied pin (there the ordinary wrong-id rule is the whole rule);
    on a tied pin it is the case the policy must refuse: an id outside the tied set."""
    wide = load_pin(logits_pin(PREFILL_ROW))
    return int(np.flatnonzero(wide != np.float32(wide.max()))[0])


def pin_row(step: int) -> np.ndarray:
    """One step's pinned logits as a FLAT F32 vector — `load_pin` widened, then reshaped.

    The dumps come off disk in whatever shape the dumper wrote them, and ranking a 2-D row goes wrong in
    the loudest possible way: `np.argsort` on a (1, N) row, indexed back against that row, is a 230 GiB
    fancy-index (measured, not hypothetical — it was this file's first attempt at `pin_runnerup`). A
    logits vector is a vector; take the shape the code wants at the read.
    """
    return np.asarray(load_pin(logits_pin(step)), dtype=np.float32).reshape(-1)


def bf16_above(x: float) -> float:
    """The next BF16 value above `x`, on the payload bits — the smallest change that wins a near-tie.

    Same guard as `bf16_bump`: payload order is numeric order only for non-negatives, so stepping a
    negative would step it DOWN, and an injection that quietly moved the wrong way is worse than no
    injection. The pinned maxima this is used on are positive logits.
    """
    if not np.isfinite(x) or x < 0:
        raise SystemExit(f"gate: the lift only knows how to step a non-negative value up"
                         f" (asked for {x!r})")
    payload = int(bf16_payload(x)[0]) + 1
    out = float(np.array([payload << 16], dtype=np.uint32).view(np.float32)[0])
    if not out > x:
        raise SystemExit(f"gate: the lift from {x!r} produced {out!r}, which is not one BF16 ULP up")
    return out


def pin_runnerup(step: int, flag: str = "--slack-alt") -> int:
    """The pin's runner-up id — what an engine lands on when a near-tie goes the other way. Refuses a
    step whose pin has no unique winner: that is D-1's case, and an injection aimed at a tie would be
    caught by the tie policy no matter what the slack rule does (the vacuous-negative trap)."""
    wide = pin_row(step)
    order = np.argsort(wide)[::-1]
    if float(wide[order[0]]) == float(wide[order[1]]):
        raise SystemExit(f"gate: {flag} {step} names a step whose pin ties at its maximum"
                         f" ({logits_pin(step)}); that is the proved-tie policy's case, not the"
                         " quantization-slack rule's — pick a step with a unique winner")
    return int(order[1])


def outside_leader_id(step: int) -> int:
    """The highest-valued pinned id NOT inside this step's slack leader set — what the rule must refuse.

    The nearest miss rather than a random far-away id: an injection that lands just outside the line is
    the one that shows the line is where the rule says it is. Refuses a step whose margin is not inside
    the bound, for the same reason `--break-tie` refuses an untied step: outside the region every wrong
    id is caught by the ordinary rule, so the injection would prove nothing about this one.
    """
    ref = pin_row(step)
    sl = step_slack(step, ref)
    if sl is None:
        raise SystemExit(f"gate: --break-slack {step} names a step this rule does not reach"
                         f" ({logits_pin(step)} has no unique winner, or no policed meanrel band"
                         " there — §06.5's prefill row is judged by argmax alone and gets no slack)")
    inside = set(sl["leaders"])
    for i in np.argsort(ref)[::-1]:
        if int(i) not in inside:
            return int(i)
    raise SystemExit(f"gate: --break-slack {step}: every id of the vocabulary is inside the bound"
                     f" ({sl['bound']:.6g}), which means the bound is not a rule any more")


def slack_region_steps(n_steps: int) -> tuple:
    """`(steps inside the slack region, the narrowest step outside it)` for this fixture.

    Both are needed to aim the injections: the admitting one has to sit inside, and the refusing one is
    only worth anything at the DECIDABLE step closest to the line — an injection aimed at a step whose
    margin is 100x its bound would be caught by anything. Reads the same caps the gate polices, through
    the same `step_slack` the gate judges with, so the selftest aims at what the rule actually says.
    """
    inside, outside = [], []
    for j in range(n_steps):
        sl = step_slack(j, pin_row(j))
        if sl is None:
            continue                       # exact tie (D-1's) or no band: neither side of this rule
        (inside if sl["margin"] < sl["bound"] else outside).append((sl["margin"] / sl["bound"], j))
    return ([j for _, j in sorted(inside, key=lambda t: t[1])],
            min(outside)[1] if outside else None)


def slack_injection(step: int, mode: str, flag: str) -> int:
    """The id a slack injection lifts over the pin's maximum, refusing a step on the wrong side of it.

    `mode` is `"alt"` (the admitted case: a leader, at a step inside the region), `"outside"` (a
    non-leader, still inside the region — the rule must refuse it) or `"decidable"` (the pin's runner-up
    at a step the pin DID decide — the refusal that proves the bound is sized, not invented). Aiming any
    of them at the wrong kind of step would produce a negative every check catches, which is the trap
    `prototype/mlp_gemv/results.md` names, so it is an error rather than a silent no-op.
    """
    ref = pin_row(step)
    sl = step_slack(step, ref)
    inside = sl is not None and sl["margin"] < sl["bound"]
    want_inside = mode != "decidable"
    if inside != want_inside:
        where = ("is inside the quantization-slack region" if inside else
                 "is NOT inside the quantization-slack region"
                 + ("" if sl else " (this step has no unique winner in the pin, or no policed meanrel"
                              " band — §06.5's prefill row is the standing example)")
                 + (f": margin {sl['margin']:.6g} vs bound {sl['bound']:.6g}" if sl else ""))
        raise SystemExit(
            f"gate: {flag} {step} names a step that {where}. The three slack injections are one"
            " perturbation aimed at three different steps, and each only proves something on its own"
            " side of the line: --slack-alt and --break-slack need a step whose margin is under its"
            " bound (--slack-region prints them), --break-runnerup needs one over it (the narrowest"
            " such step is what `--selftest` picks).")
    return outside_leader_id(step) if mode == "outside" else pin_runnerup(step, flag)


def lift_over_pin(step: int, vec, index: int) -> np.ndarray:
    """A copy of `vec` with element `index` lifted one BF16 step above this step's pinned maximum.

    The smallest change that makes that id the engine's own argmax. All three slack injections use it,
    so they are one perturbation shape and the only thing that can change the verdict is the rule's own
    line — which is what makes the pair a measurement of where that line sits.
    """
    out = np.array(vec, dtype=np.float32, copy=True).reshape(-1)
    out[index] = bf16_above(float(pin_row(step).max()))
    return out



def lift_past_tie_slack(step: int, vec, index: int) -> np.ndarray:
    """A copy of `vec` with element `index` placed eight licensed element errors above the pinned tie.

    The needle that keeps the generalised tie clause from being a shrug: `--tie-alt` proves a tied id the
    engine reached by ordinary rounding is admitted, and this proves a tied id the engine reached by
    moving it further than the step's own band licenses is not. Eight times the bound rather than one
    over it, because a needle that sits one epsilon from the line fails for the wrong reason when the
    bound itself moves a little. Refuses a step with no tie or no priced band, so it cannot quietly
    degrade into the ordinary wrong-id test.
    """
    out = np.array(vec, dtype=np.float32, copy=True).reshape(-1)
    bound = tie_slack_bound(step)
    tie = pin_tie(step)
    if bound is None or tie is None:
        raise SystemExit(f"gate: --tie-far {step} names a step with no pinned tie or no priced meanrel"
                         " band; the slack bound cannot be sized there, so the injection would prove"
                         " nothing")
    out[index] = float(tie["value"]) + 8.0 * bound[0]
    return out


def print_slack_region() -> int:
    """`--slack-region`: which of this fixture's steps the pin could not decide, and how near the line
    the rest of them are. CPU-only — it reads pins and cached caps, starts no binary, touches no device.

    The refusal messages in `slack_injection` point here, because an injection refused for sitting on the
    wrong side of the line has to tell the operator where the line is. `engine/slack_size.py` is the same
    arithmetic over all eight fixtures at once, which is where the corpus-wide claim is made; this is the
    per-fixture view a gate log points at.
    """
    section("quantization slack — the steps this fixture's pin could not decide")
    n = fixture_step_count()
    inside, narrow_out, ties = [], None, []
    for j in range(n):
        sl = step_slack(j, pin_row(j))
        if sl is None:
            ties.append(j)
            continue
        if sl["margin"] < sl["bound"]:
            inside.append(sl)
        elif narrow_out is None or sl["margin"] / sl["bound"] < narrow_out["margin"] / narrow_out["bound"]:
            narrow_out = sl
    print(f"  {REFS.name}: {n} judged generation steps; {len(inside)} inside the slack region,"
          f" {len(ties)} with no unique winner in the pin (D-1's proved-tie policy, not this one)"
          f" and {n - len(inside) - len(ties)} decidable by the pin alone")
    for sl in inside:
        print("  [inside]  " + sl["numbers"] + f" — leaders inside the margin: {sl['leaders']}")
    if narrow_out is not None:
        print(f"  [closest-decidable] " + narrow_out["numbers"] + f" — the pin decides this one by"
              f" {narrow_out['margin'] / narrow_out['bound']:.2f}x, so a bound"
              f" {narrow_out['margin'] / narrow_out['bound']:.2f}x larger would start excusing it"
              " (`--break-runnerup " + str(narrow_out["label"].split()[-1]) + "` is that test)")
    else:
        print("  [closest-decidable] none — every step of this fixture is inside the region, which"
              " would mean the rule has stopped deciding anything")
    print("  rule: " + SLACK_RULE)
    return 0


def metrics(got: np.ndarray, ref: np.ndarray) -> dict:
    g, r = got.astype(np.float64), ref.astype(np.float64)
    d = g - r
    order = np.argsort(r)[::-1]
    maxabs = float(np.max(np.abs(d)))
    gap = float(r[order[0]] - r[order[1]])
    return {
        "cos": float(np.dot(g, r) / (np.linalg.norm(g) * np.linalg.norm(r))),
        "meanrel": float(np.mean(np.abs(d)) / np.mean(np.abs(r))),   # §10.2's definition
        "ndiff": int(np.count_nonzero(d)),
        "maxabs": maxabs,
        "gap12": gap,
        # How many times bigger the reference's own top-1/top-2 margin is than our worst element
        # error — i.e. how much further the error would have had to grow to flip the token. Guarded:
        # a bit-exact step has maxabs 0, and dividing by an epsilon there prints a ludicrous number.
        "margin": (float("inf") if maxabs == 0.0 else gap / maxabs),
    }


def top2(v: np.ndarray) -> tuple[int, float, float]:
    """(argmax id, its value, the gap to the runner-up) for one logits row, lowest id winning a tie.

    `np.partition` rather than a full `argsort`: this is called on 248 320-wide rows, sometimes per row
    of a 524-row prompt, and the whole ordering is never needed — only the top pair is.
    """
    a = np.asarray(v, dtype=np.float64).reshape(-1)
    i = int(np.argmax(a))                       # first maximum = lowest id, same rule as `argmax_low`
    top = float(a[i])
    rest = np.delete(a, i) if a.size > 1 else np.array([float("-inf")])
    return i, top, top - float(np.max(rest))


def room_in_ulp(top_val: float, gap: float) -> float:
    """One side's top-1/top-2 gap in BF16 steps at that side's own top-1 — the unit the flip line is in.

    A logit gap means something only at the magnitude it sits at (`bf16_ulp`'s own note: 0.125 is nothing
    at 40 and two whole ULP at 15.6), which is why the line moved into this unit out of absolute logits.
    Same step function section (b) states a near-tie in, so there is ONE definition of ULP in this tree.
    """
    step = bf16_ulp(top_val)
    return float("inf") if not step or step != step else float(gap) / step


def flip_room_ulp(*sides: tuple[float, float]) -> float:
    """The room of a comparison with NO reference in it, in ULP: the SMALLER of the sides' own gaps,
    each divided by that side's own step size. A side is `(top-1 value, top-1 minus top-2)` as `top2`
    returns it.

    The min() is §A2's reading, unchanged, and it is right for exactly the pairings §A2 measured: two
    legal widths of one route (batched vs one-row vs carried), where neither side is an answer sheet and
    "a row in the first bucket was a coin toss for one of the two routes and cannot be blamed on the
    other". For a comparison against the PIN there is an answer sheet, and this is the wrong quantity —
    see `reference_room_ulp`.
    """
    return min(room_in_ulp(v, g) for v, g in sides)


def reference_room_ulp(pin: np.ndarray) -> tuple[float, float, float, float]:
    """`(id, room_in_ulp, gap_logits, top1_value)` of the REFERENCE's own decision at one row.

    In a pin-vs-engine comparison the question the flip rule answers is "was the reference decided, and
    did the engine go elsewhere?". The room that bears on that is the reference's own: the engine's
    runner-up gap is not evidence about the reference, and it is actively misleading here — an injected
    wrong id lifted one BF16 step over the pin's maximum has an engine-side room of exactly 1.00 ULP at
    any magnitude, which is the narrowest illegal lift this tree knows how to build (`lift_over_pin`) and
    would read as a coin toss under a min() reading.
    """
    i, val, gap = top2(pin)
    return i, room_in_ulp(val, gap), gap, val


def room_is_thin(room_ulp: float) -> bool:
    """The line itself, in ONE place: is this much room (BF16 ULP) a coin toss? Both readings route
    through here, so the number appears once and only the CHOICE of quantity differs between them."""
    return room_ulp < CASCADE_BAND["flip_room_max_ulp"]


def flip_admissible(*sides: tuple[float, float]) -> bool:
    """Is this flip admissible? — the SYMMETRIC reading, for a comparison with no reference (section (h)'s
    cross-width pairings). Judged on `flip_room_ulp`: the smaller of the sides' gaps in BF16 ULP.

    One line, two READINGS, because the two pairings ask different questions (§3f of
    prototype/gemm_algo_sweep/results.md). Before 2026-10-16 this was the only reading and it was applied
    to the pin-vs-engine pairing too, which is how `wide520f` row 180 and `--break-runnerup` came to need
    opposite answers from one constant: the real cross-width row needs the line above 0.9985 ULP and the
    injected pin-vs-engine wrong-id sits at exactly 1.00 ULP of engine-side room, a 0.0015 ULP squeeze
    that no value of the constant can serve. Splitting the READING — not the constant — is the fix:
    reference comparisons ask about the reference (`reference_flip_admissible`), engine-vs-engine asks
    about the closer of the two engines. Merging them back together is the mistake that starts the same
    argument over again, and `--selftest`'s reading-split case fails if anyone does it.
    """
    return flip_room_ulp(*sides) < CASCADE_BAND["flip_room_max_ulp"]


def reference_flip_admissible(pin: np.ndarray) -> bool:
    """Is a flip against the PIN admissible? Only if the REFERENCE itself had under the line of room.

    Same number as `flip_admissible`, different side measured, for the reason in `reference_room_ulp`.
    §A2's min() rationale does not reach here: §A2 bucketed hand-vs-library rows, two engines and no
    reference, and the two corpus rows its min() reading turned on (p128 row 27, chunk64b row 97) are
    engine-vs-engine rows. The device set of 2026-10-16 reports zero real admissions on this pairing
    (9 fixtures × both routes), so what holds this reading up is injections, and `flip_room_line`
    says so instead of borrowing the cross-width margin.
    """
    return room_is_thin(reference_room_ulp(pin)[1])


def room_verdict(step_label: str, pin: np.ndarray, eng: np.ndarray | None,
                 sampled: int) -> tuple[bool, str]:
    """`(admissible, the line to print)` for one flipped token under the cascade room rule.

    This is the REFERENCE pairing, so the room judged is the PIN's own top-1/top-2 gap in BF16 ULP
    (`reference_room_ulp`, and the reason it is not `flip_room_ulp` is written there). The pin is the
    hand route's answer sheet as well: the hand route reproduces this fixture's pinned ids, so the room
    the pin had IS the room the hand route had.

    The ENGINE's own gap is printed beside it and NOT judged. That is the distinction a single symmetric
    line erased: for two legal widths of one route the closer side's uncertainty is the honest quantity,
    because neither side is an answer sheet; here one side is, and the other side's near-tie says nothing
    about whether the reference decided. `--selftest` runs one row under both readings and requires them
    to disagree, so this cannot be "simplified" back into one path unnoticed.

    Fails closed on a missing engine row: a flip that cannot be attributed to a row cannot be admitted.
    """
    p_id, p_val, p_gap = top2(pin)
    _pid, p_ulp, _pg, _pv = reference_room_ulp(pin)
    line = (f"ROOM {step_label}: pin top1 {p_id} ({p_val:.6g}) over its runner-up by {p_gap:.6g}"
            f" = {p_ulp:.2f} BF16 ULP at that magnitude,"
            f" against the flip line {flip_room_line('reference')} ({CASCADE_RULE})")
    if eng is None:
        return False, line + " | NOT ADMISSIBLE: no engine logits row, so the flip cannot be attributed"
    e_id, e_val, e_gap = top2(eng)
    e_ulp = room_in_ulp(e_val, e_gap)
    line += (f"; engine top1 {e_id} ({e_val:.6g}) over ITS runner-up by {e_gap:.6g}"
             f" = {e_ulp:.2f} ULP — printed, not judged: this pairing has a reference, and the"
             " engine's own uncertainty is not the reference's")
    if p_gap <= 0.0:
        # A zero gap in the pin is not a thin decision, it is no decision: that row is an exact tie, and
        # D-8's tie policy already owns it (it admits only an id the pin actually tied at its maximum,
        # proved from the payload). Letting the flip rule wave through an arbitrary id because "0.0 is
        # under 0.10" would silently widen D-8 through the back door, and `--break-y0` on `chunk64b` is
        # exactly that id.
        return False, (line + " | NOT ADMISSIBLE under the flip rule: the pin's gap here is 0, i.e. the"
                       " pin TIED at its maximum, and that question belongs to the tie policy above"
                       " (which admits only an id the pin tied), not to the thin-room rule")
    if not reference_flip_admissible(pin):
        return False, (line + f" | RED: the REFERENCE had {p_ulp:.3g} BF16 ULP of room here"
                       f" ({p_gap:.6g} at top-1 {p_val:.6g}), at or over"
                       f" {CASCADE_BAND['flip_room_max_ulp']:.2f}, so the pin decided this token and"
                       " the engine went elsewhere. That is a divergence, not the thin-room class §A2"
                       " of progress/2026-10-14/route-c-audit-2026-10-14.md measured (which had no reference on either side)")
    return True, (line + f" | ADMITTED as a thin-room flip: the reference's own room is"
                  f" {p_ulp:.4g} BF16 ULP, under {CASCADE_BAND['flip_room_max_ulp']:.2f}, which is"
                  " where A-18's chaos floor and the 24/24 thin-room flips of §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md live (printed, not gated:"
                  " which way this route's coin landed is not a band)")


def cascade_floor(cos_cap: float, rel_cap: float) -> tuple[float, float, str]:
    """(cos-deficit cap, meanrel cap, the clause naming the floor) after the cascade floor is applied.

    A FLOOR ON HOW TIGHTLY A STEP MAY BE JUDGED, never a ceiling: on the library route a step's cap is
    `max(that step's own cap, the measured cascade figure)`, because demanding a deficit smaller than
    the legal hand<->library delta (max 1.889e-3 per row over 1694 rows, §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md) from a route whose dots
    reassociate is asking it not to be that route. On the hand route this returns its arguments
    untouched, so every cap in this file means exactly what it meant before route (c).
    """
    if not route_is_library():
        return cos_cap, rel_cap, ""
    cos_f = max(cos_cap, CASCADE_BAND["row_1cos_max"])
    rel_f = max(rel_cap, CASCADE_BAND["stage_meanrel_max"])
    clause = (f"floored at the measured cascade (cos deficit >= {CASCADE_BAND['row_1cos_max']:.1e},"
              f" meanrel <= {CASCADE_BAND['stage_meanrel_max']:.0%}) because this run is"
              f" `--gemm library`: {CASCADE_RULE}")
    return cos_f, rel_f, clause


def load_fp(step: int, kind: str):
    """One step's state fingerprint: [layer, 4] F64 with columns n2, maxabs, mean, dn2 (see
    `tools/reference_dump.py:fp_row`). None when the fixture predates the pins — dec8 was made
    before there were fingerprints, and everything that reads them degrades to reporting that
    rather than failing."""
    path = REFS / f"decode{step}.{kind}_fp.npy"
    return np.load(path, allow_pickle=False) if path.exists() else None


def fp_total(step: int, kind: str, col: int):
    a = load_fp(step, kind)
    return None if a is None else float(a[:, col].sum())


# ---- the route allowance -----------------------------------------------------------------------
# The rule and its derivation live in the ROUTE ALLOWANCE block next to `BAND`, above. Everything here
# is a reader: no number in this section is computed by a gate run.
ALLOWANCE_FILE = "route_allowance.json"
SELFTEST_ALLOW_FACTOR = 1.5   # how far over its own allowance the selftest injection puts a step


def _ids_sha(ids) -> str:
    return hashlib.sha256(",".join(str(int(t)) for t in ids).encode()).hexdigest()[:16]


def allowance_inputs(n_steps: int) -> dict:
    """The identity of the measurement a cached allowance must have been priced from.

    Four fields, and each is there because changing it invalidates a cached spread:
      * `prefill_ids` / `feed_ids` — the route spread is a statement about ONE prompt fed ONE forced
        token chain; re-pin the fixture and the number belongs to a different prompt.
      * `n_steps` — an allowance priced with `--max-steps` (or against a shorter chain) must not be
        used for the steps it never priced.
      * `specimpl_sha256` — the spread is measured inside `speccheck/specimpl.py`, so a change to the
        reference's arithmetic is a change to the band. `engine/route_price.py`'s own hash is NOT in
        here on purpose: a doc-comment edit to the measurer is not a change to the measurement.
    """
    man = json.loads((REFS / "manifest.json").read_text())
    return {"prefill_ids": _ids_sha(man["prefill_ids"]),
            "feed_ids": _ids_sha(man["greedy_continuation"][:-1]),
            "n_steps": n_steps,
            "specimpl_sha256": hashlib.sha256(
                (ROOT / "speccheck" / "specimpl.py").read_bytes()).hexdigest()}


def allowance_margins(doc) -> tuple[float, float, str]:
    """(cos margin, meanrel margin, and where they came from) for one cached allowance file.

    PER-FIXTURE since D-2 (2026-10-12): the file states its own margins, because the derivation is a max
    over that fixture's own steps and nothing else — so a fixture that measures a wide engine/route ratio
    (`dec300f`: 7.98 at step 86, on a route spread *tighter* than the 64-step fixtures) cannot loosen the
    gate for anybody else. `BAND["route_margin"]`/`route_margin_rel` are the documented FALLBACK for a
    file written under the old global scheme, which carried no margins of its own; when they are used the
    caller says so out loud rather than applying them silently.
    """
    if doc.get("margin") is not None and doc.get("margin_rel") is not None:
        if doc.get("margin_basis"):
            return (float(doc["margin"]), float(doc["margin_rel"]),
                    "this fixture's own, derived from its own steps by engine/route_allowance.py")
        return (float(doc["margin"]), float(doc["margin_rel"]),
                "carried in this file from the retired global-max scheme (no `margin_basis`): the"
                " number binds, but it was a max over several fixtures, so it may be looser than this"
                " fixture's own derivation — see derivation.worst_ratio_by_fixture")
    return (BAND["route_margin"], BAND["route_margin_rel"],
            "the documented global fallback: this file carries no margins at all")


def read_allowance_doc():
    """The cached file's raw contents, or None if there is none. Reads only — `load_allowance` judges."""
    path = REFS / ALLOWANCE_FILE
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as e:
        check(False, f"{ALLOWANCE_FILE} is readable", f"{type(e).__name__}: {e}")
        return None


def allowance_problems(doc, n_steps: int) -> list:
    """Every reason this cached file must not be trusted for this fixture. Empty list = use it.

    Four groups, in the order a reader needs them: it is the right fixture's file; its provenance still
    matches the pins and the reference in front of it; its margins are self-consistent with the caps they
    produced and with the engine deficits recorded beside them; and it covers the steps being judged.
    """
    bad = []
    if doc.get("fixture") != REFS.name:
        bad.append(f"fixture={doc.get('fixture')!r} but this is {REFS.name!r} — a per-fixture margin read"
                   " by the wrong fixture is exactly the mix-up per-fixture keying makes possible, so a"
                   " file has to name the fixture it was priced for")
    margin, margin_rel, _whence = allowance_margins(doc)
    want_inputs = allowance_inputs(n_steps)
    got_inputs = doc.get("inputs", {})
    for field, want in want_inputs.items():
        if got_inputs.get(field) != want:
            bad.append(f"inputs.{field}={got_inputs.get(field)!r} but this fixture is {want!r} — the"
                       " pins or the reference moved under this cached spread; re-price it with"
                       " engine/route_allowance.py")
    steps = doc.get("steps", [])
    for r in steps:
        j = int(r["step"])
        for col, m in (("one_cos", margin), ("meanrel", margin_rel)):
            spread, cap = float(r[f"route_{col}"]), float(r[f"allowance_{col}"])
            if abs(cap - m * spread) > 1e-9 * max(abs(cap), 1e-30):
                bad.append(f"step {j}: allowance_{col}={cap:.6e} is not margin x route_{col}"
                           f" = {m:.2f} x {spread:.6e} — the margin in force and the caps it produced"
                           " have to be the same numbers, so a hand-edited cap is a band nobody measured")
            # The margin must at least admit the engine this same file was priced against. That is not a
            # claim about TODAY's engine — (b)'s band check judges that — only that the file cannot
            # contradict itself: a margin below its own recorded ratios was edited after the fact.
            eng = r.get(f"engine_{col}_when_priced")
            if eng is not None and float(eng) / spread > m:
                bad.append(f"step {j}: the engine measured when this file was priced was"
                           f" {float(eng) / spread:.3f}x the route spread, over this file's own margin"
                           f" {m:.2f} — it could not have passed the day it was written, so the margin"
                           " or the spread was edited after the fact")
    priced = {int(r["step"]) for r in steps}
    missing = [j for j in range(n_steps) if j not in priced]
    if missing:
        bad.append(f"{len(priced)} of {n_steps} steps priced; no allowance row for step(s)"
                   f" {missing[:6]}{' …' if len(missing) > 6 else ''}")
    return bad


def load_allowance(n_steps: int):
    """The per-step allowance for this fixture as `{margin, margin_rel, margin_source, steps}`, or None.

    None means "judge by the flat published floors", which is the STRICTER fallback and is always
    announced per step, never assumed. A file that is present but does not match what it is being read
    for is a FAIL, not a fallback: a stale spread is a band nobody measured, and the whole point of
    caching it is that its provenance is checkable.
    """
    doc = read_allowance_doc()
    if doc is None:
        notes.append(f"no {ALLOWANCE_FILE} next to {REFS.name}'s pins: every step is judged by the flat"
                     " §10.1 G5 / §10.2 floors. Price it (CPU, ~5 s per fixture, no device):"
                     " `engine/route_price.py --fixture " + REFS.name + " --engine-log <gate log>"
                     " --out tmp/route_" + REFS.name + ".json` then `engine/route_allowance.py --table"
                     " tmp/route_" + REFS.name + ".json` — one fixture per run is enough now that a"
                     " margin is derived from that fixture alone (D-2).")
        return None
    bad = allowance_problems(doc, n_steps)
    check(not bad, f"{ALLOWANCE_FILE} is valid for {REFS.name}", bad[:4])
    if bad:
        return None
    gen = doc.get("generated", {})
    margin, margin_rel, whence = allowance_margins(doc)
    print(f"  allowance: {ALLOWANCE_FILE} — priced {gen.get('date', 'date unknown')} by"
          f" `{gen.get('command', 'command unknown')}`; margins cos {margin:.2f} / meanrel"
          f" {margin_rel:.2f} ({whence})")
    return {"margin": margin, "margin_rel": margin_rel, "margin_source": whence,
            "steps": {int(r["step"]): r for r in doc["steps"]}}


def rule_for_step(allowance, j: int) -> tuple:
    """(cos cap, meanrel cap, the line naming the rule in force) for step j.

    Named on every step because a reader of a gate log must never have to guess why a step passed, and
    the two rules give different answers at the same step: the allowance is tighter than the flat floor
    most places and looser in the low-energy window. The margin quoted here is THIS FIXTURE's own, taken
    from its own allowance file (D-2), because that is the only margin whose derivation looked at these
    steps.

    On the `library` route both caps then pass through `cascade_floor`, which can only ever widen them
    (to the measured hand<->library cascade, §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md) and never narrow them — see CASCADE_BAND.
    """
    if allowance is None:
        cap, rel, floor_clause = cascade_floor(1.0 - BAND["cos_hard"], BAND["meanrel_hard"])
        return (cap, rel,
                f"flat G5 (cos deficit <= {cap:.4e}, meanrel <= {rel:.0%}) — no route allowance priced"
                f" for {REFS.name}" + (f"; {floor_clause}" if floor_clause else ""))
    row = allowance["steps"][j]
    spread = float(row["route_one_cos"])
    spread_rel = float(row["route_meanrel"])
    cap, rel, floor_clause = cascade_floor(float(row["allowance_one_cos"]),
                                           float(row["allowance_meanrel"]))
    return (cap, rel,
            f"route allowance for {REFS.name} (this step's measured spread={spread:.4e},"
            f" {REFS.name}'s own margin={allowance['margin']:.2f} -> cos deficit <="
            f" {float(row['allowance_one_cos']):.4e}; meanrel spread={spread_rel:.4%},"
            f" margin_rel={allowance['margin_rel']:.2f} -> <="
            f" {float(row['allowance_meanrel']):.4%})"
            + (f"; {floor_clause}" if floor_clause else ""))


def _counterfactual(value: float, flat_cap: float, flat_name: str) -> str:
    """What a failed band line says besides its own number: whether the FLAT published floor would have
    caught it too. A failure under the flat cap is exactly the failure the old band could not see, and
    that is the sentence a reader of a gate log needs; it is printed only on a failure, because a passing
    step does not need a counterfactual.
    """
    return (f"the flat {flat_name} ({flat_cap:.4e}) would have PASSED this step — the route allowance"
            " is what caught it" if value <= flat_cap else
            f"it is outside the flat {flat_name} ({flat_cap:.4e}) too")


def fixture_step_count() -> int:
    """How many generation steps this fixture gates — the number an allowance must cover."""
    return len(json.loads((REFS / "manifest.json").read_text())["greedy_continuation"]) - 1


def allowance_row(j: int):
    """Step j's cached row for a DRY RUN, quietly, or None. `dry_run` cannot use `load_allowance`: it
    runs before section (b), and an injection whose allowance is stale must refuse rather than double-
    report the same FAIL."""
    doc = read_allowance_doc()
    if doc is None or allowance_problems(doc, fixture_step_count()):
        return None
    for row in doc["steps"]:
        if int(row["step"]) == j:
            return row
    return None


def relative_noise(vec: np.ndarray, fraction: float = 0.05) -> np.ndarray:
    """Aggregate corruption that LEAVES the argmax alone: `fraction` multiplicative Gaussian noise.

    At 5 % this puts cos at ≈ 1 − σ²/2 ≈ 0.9987 — under G5's 0.9995 and over §10.2's 0.999 cascade
    floor. This is the case that asks whether the band has teeth of its own, separate from the token id.
    Shared by the injection and by `selftest`, which has to know what deficit the injection actually
    produces before it can claim the band will catch it.
    """
    rng = np.random.default_rng(4)          # fixed: the injection must be reproducible
    return (vec * (1.0 + fraction * rng.standard_normal(vec.shape))).astype(np.float32)


def allowance_noise(vec: np.ndarray, j: int, factor: float) -> np.ndarray:
    """Noise that puts step j's logits deficit at `factor` x its own cached route allowance.

    Sized, not guessed: for g = x + n with n ~ N(0, s²), 1-cos ≈ s²/(2·mean(x²)) to first order; s is
    then corrected once against the deficit actually achieved, so the injection lands on its target
    instead of near it. Constant amplitude rather than per-element proportional, because a proportional
    σ big enough to move cos by 1e-4 is a couple of percent of the LARGEST logits and can shuffle the
    top pair — that would make the argmax line the thing that caught it, which is a different test.
    """
    row = allowance_row(j)
    if row is None:
        raise SystemExit(f"gate: --break-allow {j} needs a valid {ALLOWANCE_FILE} for {REFS.name}:"
                         " the injection's whole meaning is 'above this step's measured allowance,"
                         " under the flat floor' (engine/route_allowance.py writes the file, CPU only)")
    target = factor * float(row["allowance_one_cos"])
    flat = 1.0 - BAND["cos_hard"]
    if target >= flat:
        raise SystemExit(f"gate: --break-allow {j}: 2x the allowance at this step is {target:.4e},"
                         f" which is at or over the flat §10.1 G5 floor {flat:.4e}. An injection the"
                         " old band would have caught anyway proves nothing"
                         " (`prototype/mlp_gemv/results.md`); aim it at a step with a tighter spread.")
    x = vec.astype(np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    rng = np.random.default_rng(7)          # fixed: the injection must be reproducible
    n = rng.standard_normal(x.shape) * (rms * np.sqrt(2.0 * target))
    achieved = 1.0 - metrics(x + n, x)["cos"]
    return (x + n * np.sqrt(target / max(achieved, 1e-30))).astype(np.float32)


def band_noise(step: int, vec) -> np.ndarray:
    """Noise sized to land four times outside whatever band this step is actually judged under.

    The route-independent aggregate negative. `--break-noise`'s fixed 5 % was sized against the flat §10.1
    G5 floor, and on the library route the cascade floor (2e-03 of 1-cos) is WIDER than what that noise
    produces (1.2e-03), so the fixed-size injection stops being a negative there — by policy, not by
    accident. This sizes itself against the cap in force instead: 4x the effective cos-deficit cap at that
    step, whichever rule produced it, so it fails on either route for the same reason (it is outside the
    band) rather than because the injection happens to be loud.

    Constant amplitude, one correction pass, exactly as `allowance_noise` sizes its own: a proportional σ
    this large would shuffle the top pair and make the argmax line the thing that caught it, which is a
    different test.
    """
    row = allowance_row(step)
    cap = (1.0 - BAND["cos_hard"]) if row is None else float(row["allowance_one_cos"])
    rel = BAND["meanrel_hard"] if row is None else float(row["allowance_meanrel"])
    cap_eff, _rel_eff, _clause = cascade_floor(cap, rel)
    target = 4.0 * cap_eff
    x = np.asarray(vec, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    rng = np.random.default_rng(11)          # fixed: the injection must be reproducible
    n = rng.standard_normal(x.shape) * (rms * np.sqrt(2.0 * target))
    achieved = 1.0 - metrics(x + n, x)["cos"]
    return (x + n * np.sqrt(target / max(achieved, 1e-30))).astype(np.float32)


def band_noise(step: int, vec) -> np.ndarray:
    """Noise sized to land four times outside whatever band this step is actually judged under.

    The route-independent aggregate negative. `--break-noise`'s fixed 5 % was sized against the flat §10.1
    G5 floor, and on the library route the cascade floor (2e-03 of 1-cos) is WIDER than what that noise
    produces (1.2e-03), so the fixed-size injection stops being a negative there — by policy, not by
    accident. This sizes itself against the cap in force instead: 4x the effective cos-deficit cap at that
    step, whichever rule produced it, so it fails on either route for the same reason (it is outside the
    band) rather than because the injection happens to be loud.

    Constant amplitude, one correction pass, exactly as `allowance_noise` sizes its own: a proportional σ
    this large would shuffle the top pair and make the argmax line the thing that caught it, which is a
    different test.
    """
    row = allowance_row(step)
    cap = (1.0 - BAND["cos_hard"]) if row is None else float(row["allowance_one_cos"])
    rel = BAND["meanrel_hard"] if row is None else float(row["allowance_meanrel"])
    cap_eff, _rel_eff, _clause = cascade_floor(cap, rel)
    target = 4.0 * cap_eff
    x = np.asarray(vec, dtype=np.float64)
    rms = float(np.sqrt(np.mean(x * x)))
    rng = np.random.default_rng(11)          # fixed: the injection must be reproducible
    n = rng.standard_normal(x.shape) * (rms * np.sqrt(2.0 * target))
    achieved = 1.0 - metrics(x + n, x)["cos"]
    return (x + n * np.sqrt(target / max(achieved, 1e-30))).astype(np.float32)


def break_allow_step():
    """The step `--break-allow` should be aimed at: the tightest priced allowance in this fixture whose
    injected deficit still fits under the flat G5 floor — i.e. the step where the old band was most
    nearly blind. None when no step qualifies, which `selftest` reports as a SKIP rather than quietly
    running a negative that any band would catch.

    Refused outright on the library route, and said why: that needle's whole meaning is "a deficit the flat
    band could not see", and the cascade floor is wider than the flat band, so on that route the deficit is
    admitted by policy at every step. `--break-band` is the aggregate negative that still means something
    there.
    """
    if route_is_library():
        return None
    doc = read_allowance_doc()
    if doc is None:
        return None
    if allowance_problems(doc, fixture_step_count()):
        return None
    flat = 1.0 - BAND["cos_hard"]
    cands = [(float(r["allowance_one_cos"]), int(r["step"])) for r in doc["steps"]
             if SELFTEST_ALLOW_FACTOR * float(r["allowance_one_cos"]) < flat]
    return min(cands)[1] if cands else None



# ---- (e) the curve ---------------------------------------------------------------------------
def curve_checks(per_step: list, n_decode: int) -> None:
    """The per-step table, and what it says about depth.

    Reports; does not gate on a new threshold. G5's floor is already applied per step in (b) and
    inventing a second figure for "how much drift is allowed by step 64" would be exactly the
    taste-based band §10.1 F1–F4 warns about. What the table is FOR is the shape: a flat curve
    says the extra depth bought nothing beyond dec8's 8 steps, a rising one says state carry-over
    is decaying and names the step where it started. Either is a result; both are printed.
    """
    section("(e) the curve — logits and reference state deltas against step count")
    have_fp = load_fp(0, "state") is not None
    print("  step  fed        sampled    cos        meanrel   |  S n2      S dn2     conv n2"
          "  conv dn2   kv dn2")
    for j, m in enumerate(per_step):
        if m is None:
            continue
        cells = []
        for kind, col in (("state", 0), ("state", 3), ("conv", 0), ("conv", 3), ("kv", 3)):
            v = fp_total(j, kind, col)
            cells.append("-" if v is None else f"{v:.4g}")
        print(f"  {j:4d}  {m['fed']:8d}  {m['sampled']:8d}  {m['cos']:.6f}  "
              f"{m['meanrel']:8.4%}  |  " + "  ".join(f"{c:>9}" for c in cells)
              + ("" if m["argmax_ok"] else "   <- ARGMAX DISAGREES"))
    ok = [m for m in per_step if m is not None]
    if len(ok) < 2:
        notes.append("(e) has fewer than two steps to compare, so there is no curve to read.")
        return
    om = [1.0 - m["cos"] for m in ok]          # the useful orientation: 0 is perfect
    head, tail = om[:FREE_RUN_BOUND], om[-FREE_RUN_BOUND:]
    slope = float(np.polyfit(np.arange(len(om), dtype=np.float64), om, 1)[0])
    worst = int(np.argmax(om))
    print(f"  drift reading: 1-cos median over steps 0..{len(head) - 1} is "
          f"{np.median(head):.3e}, over the last {len(tail)} steps {np.median(tail):.3e}"
          f" (ratio {np.median(tail) / max(np.median(head), 1e-300):.2f}); least-squares slope"
          f" {slope:+.3e} per step; worst step {worst} at cos {ok[worst]['cos']:.6f}"
          f" ({ok[worst]['ndiff']}/{ok[worst]['n']} elements differing).")
    print("  [info] read it as: flat means the depth bought nothing beyond dec8's 8 steps; a rise"
              " means state carry-over is decaying and the step index names where. Neither is"
              " gated — (b) already applies §10.1 G5 per step, and a second threshold for "
              "\"how much drift by step 64\" would be the taste-based band §10.1 F1–F4 forbids.")
    if not have_fp:
        notes.append("(e) ran without state fingerprints: this fixture predates them (dec8 was"
                     " pinned first). Re-make it with tools/reference_dump.py to get the state"
                     " columns.")


# ---- (f) fixture integrity of the state deltas ------------------------------------------------
def fingerprint_checks(man: dict, n_decode: int) -> None:
    """Are the pinned state deltas self-consistent? A check on the DUMP, not the engine.

    The kv row's dn2 must be exactly 0: the reference's KV is append-only, so nothing already
    written may move. That is not decoration — it is the check that caught a first version of
    `fp_row` comparing a flattened prefix, which lined head 1 of a grown cache against the tail of
    head 0 and reported dn2 3.0e3 on a cache that is bit-identical (verified separately).
    """
    section("(f) fixture integrity — the pinned state deltas must describe a real cache")
    if load_fp(0, "state") is None:
        notes.append("(f) skipped: this fixture carries no state fingerprints.")
        return
    cfg = man["text_config"]
    gdn = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
    attn = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
    rows = man.get("state_fingerprint", {}).get("rows_are_model_layers", {})
    check(rows.get("state") == gdn and rows.get("conv") == gdn and rows.get("kv") == attn,
          "fingerprint rows map to the §03.5 layer routing read from the manifest",
          f"state {len(rows.get('state', []))} conv {len(rows.get('conv', []))}"
          f" kv {len(rows.get('kv', []))} vs routing {len(gdn)}+{len(attn)}")
    bad_kv, bad_finite = [], []
    for s in range(n_decode):
        for kind in ("state", "conv", "kv"):
            a = load_fp(s, kind)
            if a is None:
                continue
            if not np.isfinite(a).all():
                bad_finite.append(f"decode{s}.{kind}")
            if kind == "kv" and np.any(a[:, 3] != 0.0):
                bad_kv.append(f"decode{s}: dn2 {a[:, 3].max():.3g}")
    check(not bad_finite, "every fingerprint value is finite", bad_kv[:4])
    check(not bad_kv, "the pinned KV delta is exactly 0 at every step (append-only cache)",
          bad_kv[:4])
    print(f"  [info] the S/conv delta columns are REPORTED, not gated: a moment-sum fingerprint"
          f" catches a state that decayed, was zeroed, was scaled or was left behind, and cannot"
          f" see a corruption that preserves the multiset of values. Full per-element state"
          f" gating would cost 18 MiB per step, which is the 1.7 GB this fixture exists to avoid.")


def deep_matches_shallow(n_decode: int) -> None:
    """If this is a deep fixture made from the same prompt, its first steps must be BIT-identical
    to dec8's. Two runs of the same reference on the same weights have one answer, so this is a
    free cross-check between fixtures — and it also proves the bf16-bit storage of the deep pins
    is lossless against dec8's f32 ones."""
    if REFS == SHALLOW_REFS or not (SHALLOW_REFS / "manifest.json").exists():
        return
    section("(f2) the deep fixture must extend the shallow one, not merely resemble it")
    deep_man = json.loads((REFS / "manifest.json").read_text())
    shallow_man = json.loads((SHALLOW_REFS / "manifest.json").read_text())
    if deep_man["prefill_ids"] != shallow_man["prefill_ids"]:
        notes.append(f"{REFS.name} has a different prompt from dec8, so the overlap check does"
                     " not apply.")
        return
    n_shallow = len(shallow_man["greedy_continuation"]) - 1
    n_steps = min(n_shallow, n_decode)
    if all(s in gapset for s in range(n_steps)):
        notes.append(f"(f2) did not run: every one of the {n_steps} overlapping steps is a declared"
                     " logit gap in this fixture, so there is no step left to compare with dec8."
                     " The gaps are declared per step by the pin, but this cross-check is one of the"
                     " things it gave up.")
        return
    agree = []
    for s in range(n_steps):
        if s in gapset:
            continue    # declared absent, already counted in section (0)'s pairing line
        deep = load_pin(f"decode{s}.logits_last.npy")
        shallow = np.load(SHALLOW_REFS / f"decode{s}.logits_last.npy", allow_pickle=False)
        if shallow.dtype == np.uint16:
            shallow = (shallow.astype(np.uint32) << 16).view(np.float32)
        agree.append(np.array_equal(deep, shallow))
    check(all(agree) and bool(agree),
          f"the first {len(agree)} steps of {REFS.name} are bit-identical to dec8's",
          f"{sum(agree)}/{len(agree)} steps bit-equal (ids also equal:"
          f" {deep_man['greedy_continuation'][:len(agree) + 1] == shallow_man['greedy_continuation'][:len(agree) + 1]})")



# ---- (g) the engine's own §08 state, against the pinned fingerprints ---------------------------
# The state half of G5 ("prefill `T` then decode `D` equals one prefill of `T+D` (states, logits,
# sampled ids", §10). Sections (a)/(b) have always judged the ids and the logits; the STATES were the
# half no gate could see, because nothing read the engine's §08.1 tuple. `decode_run --state-fp` now
# D2Hs it at the two end conditions a fixture has — after the last prompt token, after the last
# generation step — and this section compares those rows with what the dumper pinned.
#
# What gets GATED here is everything §08.6 settles with exact arithmetic rather than a tolerance: the
# KV may not be rewritten (inv 3: Δ² exactly 0.0), the cache must advance by exactly one lane per
# token (inv 4), a state that did not move while tokens were fed is a state that was left behind
# (inv 1), a conv window that did not roll is not a window (inv 2), and a state that is still all zero
# after 17+ tokens was never written (the §08.4 zeroed start, left behind). Everything mechanical is
# gated too: each row's element count, the row→layer map, the §08.4 no-previous convention on both
# sides of the comparison, and the helper identity below.
#
# What is NOT gated is how far the engine's state is ALLOWED to be from the dump's.
# `progress/2026-10-12/plan-prefill-2026-10-12.md` §2.6 assigns that in terms: "P1 ships a measurement and a
# tripwire, not a band", and whether the recurrent-route state is "close enough" to the chunk-route
# state is "decided by the main agent against those printed numbers" in the D-3/D-5 idiom, with the
# route spread at the same position as the natural denominator — which needs `route_price` extended
# to state fingerprints, a measurement that does not exist yet. Adopting a threshold here would be a
# band move without a `D-x` and a `bench/band-history.jsonl` line, and `tools/band_history.py` exists
# to refuse exactly that. So every distance below is printed at full precision and judged by nobody in
# this file. The nearest recorded yardstick for a state-level difference between two legal forms is
# G3's `cos >= 0.99999` per layer (`speccheck/gates/run_chunkcheck.py`, A-08's verification), which on
# a moment sum means an n2 ratio inside ~2e-5 for the same quantity under a different sum order. It
# is printed as a yardstick and never applied as a limit.
#
# THE CONV FAMILY IS NOT THE SAME EXTENT ON BOTH SIDES, and that is a finding, not an engine bug. The
# engine's window is `k_conv−1 = 3` pre-conv columns (§08.2, `state.h`'s `kConvWindow`); the pinned
# `conv_fp` row pools the REFERENCE's conv buffer, which is `k_conv = 4` columns wide because
# `causal_conv1d_update` cats `[conv_state, x]` before convolving (A-04). Verified on the pins, not
# argued: `dec8` and `dec64f` share one 17-token prompt, `dec8` pins the prefill raw columns
# `L{ll}.gdn.mixed_qkv_pre_conv`, and pooling dec8's newest 4 of them reproduces dec64f's pinned
# `conv_fp` row bit-for-bit for every GDN layer (newest 3 and newest 2 do not). So a correct engine
# and a correct pin differ on this family by the energy of one raw column — 19–31 % of n2 on the three
# layers checked by hand — and no engine change fixes that. The comparable quantity is the pinned
# buffer's newest 3 columns, which only the deep-step fixtures have; `conv_window_pin` derives it and
# its absence is reported, never skipped quietly.

FP_FAMILIES = ("state", "conv", "kv")
FP_COLS = ("n2", "maxabs", "mean", "dn2")


def prefill_fp(kind: str):
    """The untagged fingerprint arrays: the reference's state at the END OF THE PROMPT.

    The manifest records them with `step: "prefill"`, which is what makes them the pair for
    `decode_run --state-fp`'s `phase=prefill` rows. None when the fixture carries none: `dec8` was
    pinned before fingerprints existed, and plan §5.1 says plainly that it has no `(g)` rows.
    """
    path = REFS / f"{kind}_fp.npy"
    return np.load(path, allow_pickle=False) if path.exists() else None


def deep_state_step():
    """The deepest step whose FULL per-layer state tensors were pinned, or None.

    `--deep-steps` keeps them at one step (63 for `dec64f`/`dec64t`, 31 for `mt32f`) and
    `--deep-steps none` keeps none (`dec300f`, whose 128 steps are fingerprint-only). The full
    tensors are what both the helper-identity case and the conv-window comparison need, so this one
    question decides whether `(g)` can prove its own arithmetic here and whether it can compare a
    conv window at all — and both answers get printed, because a missing comparison must not be
    mistaken for a passing one.
    """
    steps = [int(p.name.split(".")[0][len("decode"):])
             for p in REFS.glob("decode*.L00.gdn.state_new.npy")]
    return max(steps) if steps else None


def index_state_fp(lines: list) -> tuple[dict, dict]:
    """Group the engine's `STATEFP` lines as {(phase, family, row): fields} plus {phase: prefix}.

    The prefix arrives on its own line (`STATEFP phase=prefill prefix=17`) because it is a property of
    the snapshot, not of a row; a `note=` line (a run that never left the prompt) carries neither and
    is kept under `note:<phase>` so a caller can say WHY a phase is missing instead of guessing.
    """
    rows: dict = {}
    meta: dict = {}
    for line in lines:
        phase = line.get("phase")
        if phase is None:
            continue
        if "family" not in line:
            if "prefix" in line:
                meta[phase] = int(line["prefix"])
            if "note" in line:
                meta["note:" + phase] = line["note"]
            continue
        fields = {"layer": int(line["layer"]), "n": int(line["n"]),
                  **{c: float(line[c]) for c in FP_COLS}}
        for extra in ("cols", "lanes"):
            if extra in line:
                fields[extra] = int(line[extra])
        rows.setdefault((phase, line["family"]), {})[int(line["row"])] = fields
    return rows, meta


def fp_expectations(man: dict) -> dict:
    """The row→layer map and per-row element counts, DERIVED from the manifest's `text_config`.

    Nothing here is copied from the engine or read off the pin files: the layer map comes from
    `layer_types`, the GDN row from `linear_num_value_heads × linear_key_head_dim ×
    linear_value_head_dim`, the conv row from `(linear_conv_kernel_dim − 1) ×` the q/k/v row width
    (§04.1's 6144), the pinned conv extent from `linear_conv_kernel_dim ×` it, and the KV row from
    `2 key/value heads × 2 tensors × lanes × head_dim`. So a row count or extent that stops matching
    fails here whether it moved in the engine, in the checkpoint, or in the dumper: the geometry is
    checked, not assumed.
    """
    cfg = man["text_config"]
    gdn = [i for i, t in enumerate(cfg["layer_types"]) if t == "linear_attention"]
    attn = [i for i, t in enumerate(cfg["layer_types"]) if t == "full_attention"]
    k_conv = cfg["linear_conv_kernel_dim"]
    qkv_rows = (2 * cfg["linear_num_value_heads"] * cfg["linear_value_head_dim"]
                + cfg["linear_num_key_heads"] * cfg["linear_key_head_dim"])
    return {
        "layers": {"state": gdn, "conv": gdn, "kv": attn},
        "n": {"state": (cfg["linear_num_value_heads"] * cfg["linear_key_head_dim"]
                        * cfg["linear_value_head_dim"]),
              "conv": (k_conv - 1) * qkv_rows},
        "pin_n": {"conv": k_conv * qkv_rows},
        "k_conv": k_conv,
        "head_dim": cfg["head_dim"],
    }


def conv_window_pin(step: int):
    """Fingerprint rows for the engine's conv WINDOW, derived from the pinned conv BUFFER.

    The newest `k_conv−1` columns of `decode{step}.L{ll}.gdn.conv_state` are what §08.2's window
    holds — A-04's `cat([conv_state, x])` is what makes the pinned buffer one column wider — so this
    is the only place the engine's conv row has something of its own extent to be compared against.
    Returns None when those tensors are not pinned, and the caller says so out loud.
    """
    from tools.state_fp import fp_row_np
    rows = json.loads((REFS / "manifest.json").read_text())[
        "state_fingerprint"]["rows_are_model_layers"]["conv"]
    paths = [REFS / f"decode{step}.L{layer:02d}.gdn.conv_state.npy" for layer in rows]
    if not all(p.exists() for p in paths):
        return None
    return np.array([fp_row_np([load_pin(p.name)[..., 1:]], None) for p in paths], dtype=np.float64)


def fp_helper_identity(man: dict) -> None:
    """The mechanical half of a `(g)` PASS: does the shared helper still reproduce the pinned
    fingerprint arrays from the pinned tensors, BIT FOR BIT?

    This is the check that the *gate's* arithmetic matches the *dumper's*, not the engine's. It matters
    because everything else in `(g)` rests on the claim that the engine's rows, the pinned rows, and
    any row this file derives all mean the same thing — and that claim is a piece of code shared
    between two programs, so it gets proved by importing the dumper's own function and running it on
    the pinned tensors. This is the only place in the file that imports torch.

    Bit-for-bit is a real constraint and a narrow one. `fp_row` sums in F32 because that is the
    arithmetic the pins were made with (§10.5: a fresh dump must reproduce them), and an F32 tree sum
    is a property of torch's reduction order, which follows the thread count: on this box the default
    reproduces all 42 rows of a deep step exactly, and `torch.set_num_threads(1)` does not (largest
    relative deviation 4.4e-7 on `mt32f`, measured while writing this). That asymmetry is exactly why
    the ENGINE's rows are compared with an F64 accumulation floor and reported, while this case can
    demand equality — this one compares two uses of one function, and the engine does not use torch.

    Column 3 (Δ²) is excluded wherever the predecessor step's tensors were not kept; `--deep-steps`
    keeps one step, so there is nothing on disk to difference against. The Δ² convention that IS
    checkable without them — §08.4's, that a first row's Δ² equals its n2 — is asserted against the
    pins in `state_fp_checks` below.
    """
    step = deep_state_step()
    if step is None:
        notes.append(f"(g) could NOT prove its own helper on {REFS.name}: this fixture keeps no full"
                     " per-layer state tensors (`--deep-steps none`), so the fp-of-pins identity has"
                     " nothing to recompute from, and the conv family cannot be priced at its own"
                     " extent either. The engine-vs-pin numbers printed for this fixture are still"
                     " real; what is missing is the proof of the ruler. Run this section on"
                     " dec64f/dec64t/mt32f, or re-make the fixture with --deep-steps (a re-pin, so a"
                     " main-agent call: plan §5.1 forbids touching the shipped fixtures).")
        return
    import torch     # the only torch this file imports; see the module header
    from tools.state_fp import fp_row

    rows = man.get("state_fingerprint", {}).get("rows_are_model_layers", {})
    if not all(v in rows for v in FP_FAMILIES):
        check(False, "the manifest states which model layer each fingerprint row is", str(rows))
        return
    total, ok = 0, True
    for family in FP_FAMILIES:
        pin = load_fp(step, family)
        if pin is None:
            check(False, f"fp-of-pins: decode{step}.{family}_fp.npy exists")
            ok = False
            continue
        if family == "kv":
            paths = [REFS / f"decode{step}.L{layer:02d}.cache_{w}.npy"
                     for layer in rows[family] for w in ("keys", "values")]
        else:
            which = "state_new" if family == "state" else "conv_state"
            paths = [REFS / f"decode{step}.L{layer:02d}.gdn.{which}.npy"
                     for layer in rows[family]]
        if not all(p.exists() for p in paths):
            check(False, f"fp-of-pins: {family}'s full tensors are pinned at decode{step}",
                  f"{sum(1 for p in paths if not p.exists())} of {len(paths)} missing")
            ok = False
            continue
        bad, worst = [], 0.0
        for i, layer in enumerate(rows[family]):
            parts = ([torch.from_numpy(load_pin(p.name)) for p in paths[2 * i:2 * i + 2]]
                     if family == "kv" else [torch.from_numpy(load_pin(paths[i].name))])
            got = fp_row(parts, None)
            total += 1
            if got[:3] != [float(x) for x in pin[i][:3]]:
                bad.append(f"{family} row {i} (layer {layer})")
                worst = max([worst] + [abs(g - p) / abs(p) if p else abs(g)
                                       for g, p in zip(got[:3], pin[i][:3])])
        check(not bad, f"fp-of-pins identity at decode{step}: {family} rows reproduce BIT-EXACTLY"
                       " with the dumper's own function (n2/maxabs/mean)",
              "" if not bad else f"{len(bad)} row(s) differ, largest relative deviation {worst:.3e};"
                                 f" first {bad[:3]} — if the deviation is at the 1e-7 level this is"
                                 " torch's reduction order (thread count), not the data")
        ok = ok and not bad
    if ok:
        print(f"  [info] {total} rows recomputed with `tools/state_fp.py::fp_row` at decode{step} and"
              " matched bit-for-bit: the pinned arrays, this file, and — up to the F64 accumulation"
              " floor `decode_run.cpp` states — the engine's rows are one definition. Whether a fresh"
              " HF run reproduces these pins byte-for-byte is §10.5's question, checked by"
              " `tools/reference_dump.py --verify`, not here.")


def fp_distance(eng: dict, pin: np.ndarray, pin_n: int) -> dict:
    """Engine-vs-pin distance over one family at one phase. Reports; never judges.

    `n2 ratio` is the summary a moment-sum fingerprint is actually for — two routes' energies divided.
    `maxabs` is an absolute deviation because it is itself a magnitude. `mean` is printed against the
    pin's own RMS (`sqrt(n2/n)`) because the mean of a state tensor is a small number whose relative
    deviation is meaningless on its own. `pin_n` is the pin's pooled extent, which differs from the
    engine's only for the conv family; where it differs, the ratio carries that mismatch and the
    caller says so rather than presenting it as a state error.
    """
    pin_rms = np.sqrt(pin[:, 0] / max(pin_n, 1))
    out = {"n2": (0.0, None), "maxabs": (0.0, None), "mean": (0.0, None), "ratios": [],
           "n2_worst": 1.0}
    for r, got in sorted(eng.items()):
        ratio = got["n2"] / pin[r][0] if pin[r][0] else float("nan")
        out["ratios"].append((r, got["layer"], ratio))
        for key, value in (("n2", abs(ratio - 1.0)),
                           ("maxabs", abs(got["maxabs"] - pin[r][1])),
                           ("mean", (abs(got["mean"] - pin[r][2]) / pin_rms[r]
                                     if pin_rms[r] else 0.0))):
            if value > out[key][0]:
                out[key] = (value, (r, got["layer"]))
    # The ratio is printed as itself, never as `1 + |deviation|`: a conv row that pools three columns
    # where the pin pooled four reads 0.72, and "1.28" would read as the engine carrying MORE energy
    # than the reference when it carries less. Sign is the finding.
    if out["ratios"] and out["n2"][1] is not None:
        out["n2_worst"] = next(r for rr, layer, r in out["ratios"]
                               if (rr, layer) == out["n2"][1])
    return out


def print_fp_distance(label: str, dist: dict, caveat: str = "") -> None:
    """One distance line, formatted so two fixtures can be diffed by eye.

    A zero worst-deviation prints as `no row differs` rather than naming row 0: a `--dry-run` prints
    all-zero lines by construction, and "worst at row 0" would read like a finding about row 0.
    """
    def at(value, cell):
        return f"row {cell[0]} (layer {cell[1]})" if value else "no row differs"
    print(f"  {label:<44} n2 ratio {dist['n2_worst']:.6f}"
          f" at {at(dist['n2'][0], dist['n2'][1])}; maxabs |Δ| {dist['maxabs'][0]:.4g}"
          f" at {at(dist['maxabs'][0], dist['maxabs'][1])}; mean |Δ|/RMS {dist['mean'][0]:.3e}"
          f" at {at(dist['mean'][0], dist['mean'][1])}"
          + ("" if not caveat else "\n" + " " * 46 + caveat))


def parse_state_fp_break(spec: str):
    """Validate `--break-state-fp WHAT:FAMILY:ROW` into `(what, family, row)`; None when unset.

    Validation lives here because an injection that silently does nothing is worse than no injection:
    it reads as a passing negative. So every impossible request is a refusal, in the same spirit as
    `pin_tie` refusing to inject a tie at an untied step. `frozen` on the KV is refused for the same
    reason — a KV row that stands still is the correct behaviour (§08.6 inv 3), so "freezing" it
    cannot be the bug; `rewrote` on `state`/`conv` is refused because those two are SUPPOSED to move,
    and the bug there is the opposite one, `frozen`.
    """
    if not spec:
        return None
    parts = spec.split(":")
    if len(parts) != 3:
        raise SystemExit(f"gate: --break-state-fp wants WHAT:FAMILY:ROW, got {spec!r}")
    what, family, row = parts
    if what not in ("zero", "frozen", "rewrote"):
        raise SystemExit(f"gate: --break-state-fp WHAT must be zero, frozen or rewrote, not"
                         f" {what!r}")
    if family not in FP_FAMILIES:
        raise SystemExit(f"gate: --break-state-fp FAMILY must be one of {FP_FAMILIES}, not"
                         f" {family!r}")
    if what == "frozen" and family == "kv":
        raise SystemExit("gate: --break-state-fp frozen:kv is not a bug — §08.6 inv 3 wants the KV"
                         " to stand still. Use rewrote:kv:ROW for the append-only failure.")
    if what == "rewrote" and family != "kv":
        raise SystemExit(f"gate: --break-state-fp rewrote:{family} is not a bug — a moving"
                         " state is the point (inv 1/2). Use frozen:"
                         + family + ":ROW for a state that was left behind.")
    try:
        return what, family, int(row)
    except ValueError:
        raise SystemExit(f"gate: --break-state-fp ROW must be an integer row index, not {row!r}")


def state_fp_stand_in_lines(man: dict, prompt_len: int, n_decode: int, break_spec: str = "") -> list:
    """`decode_run --state-fp`'s line protocol, filled with the pinned rows: a perfect engine.

    The rows ARE the pins — prompt-end rows from the untagged arrays, run-end rows from
    `decode{n_decode-1}` — so a clean `(g)` on this input proves plumbing and nothing about the
    engine, exactly like the rest of `--dry-run`. What it does prove is worth a CPU run: that the line
    protocol parses, that the geometry and maps agree with what `(g)` derives from the manifest, that
    the §08.4 and §08.6 convention lines fire on real numbers, and — with `--break-state-fp` — that
    each of the three state failures lights exactly the row it was aimed at.

    The conv rows carry the pinned FOUR-column pool under the engine's three-column label, so the conv
    distance a dry run prints is the extent mismatch and nothing else (block comment above `(g)`).
    A real engine's row adds its own arithmetic on top of that, which is precisely the difference
    between this stand-in and a device pass.
    """
    inj = parse_state_fp_break(break_spec)
    want = fp_expectations(man)
    lines = []
    phases = [("prefill", prompt_len, {k: prefill_fp(k) for k in FP_FAMILIES})]
    if n_decode > 0:
        phases.append(("end", prompt_len + n_decode,
                       {k: load_fp(n_decode - 1, k) for k in FP_FAMILIES}))
    if inj and inj[1] == "kv" and inj[2] >= len(want["layers"]["kv"]):
        raise SystemExit(f"gate: --break-state-fp {break_spec}: this fixture has"
                         f" {len(want['layers']['kv'])} KV rows")
    for family in FP_FAMILIES:
        if inj and inj[2] >= len(want["layers"][family]):
            raise SystemExit(f"gate: --break-state-fp {break_spec}: {family} has"
                             f" {len(want['layers'][family])} rows here")
    for phase, prefix, pin in phases:
        if not any(p is not None for p in pin.values()):
            continue                       # a fixture with no pins at all: (g) says so itself
        # A perfect engine's conv row is the window, i.e. the newest k_conv−1 columns of the pinned
        # buffer — so where those tensors exist at this phase's step, the stand-in uses them and the
        # window comparison reads as the identity it is. Where they do not (the prompt end of every
        # deep fixture, which keeps no prefill tensors), the stand-in falls back to the pinned
        # four-column row and the extent caveat in (g) is what carries it.
        if phase == "end" and deep_state_step() == n_decode - 1:
            window = conv_window_pin(n_decode - 1)
            if window is not None:
                pin = dict(pin, conv=window)
        lines.append(f"STATEFP phase={phase} prefix={prefix}")
        for family in FP_FAMILIES:
            rows = pin[family]
            if rows is None:
                continue
            for r in range(rows.shape[0]):
                vals = [float(x) for x in rows[r]]
                if family == "kv":
                    extent, tail = 4 * prefix * want["head_dim"], f" lanes={prefix}"
                elif family == "conv":
                    extent = want["n"]["conv"]
                    tail = f" cols={want['k_conv'] - 1}"
                else:
                    extent, tail = want["n"]["state"], ""
                # `frozen` and `rewrote` are run-end bugs by definition: at the prompt end there is no
                # earlier state to have been left behind, so the injection passes through untouched.
                if inj and inj[1] == family and inj[2] == r and (
                        inj[0] == "zero" or (inj[0] == "frozen" and phase == "end")
                        or (inj[0] == "rewrote" and phase == "end")):
                    if inj[0] == "zero":
                        vals = [0.0, 0.0, 0.0, 0.0]
                    elif inj[0] == "frozen":
                        vals[3] = 0.0
                    else:
                        vals[3] = 1.0
                lines.append(f"STATEFP phase={phase} family={family} row={r}"
                             f" layer={want['layers'][family][r]}{tail} n={extent}"
                             f" n2={vals[0]:.17g} maxabs={vals[1]:.17g} mean={vals[2]:.17g}"
                             f" dn2={vals[3]:.17g}")
    return lines


def state_fp_checks(man: dict, prompt_len: int, n_decode: int, ran_decode: int,
                    statefp: list | None, stand_in: bool = False) -> None:
    """(g) G5-STATE — the engine's §08 state against the pinned fingerprints (§08.6 inv 1–4, §10 G5).

    `statefp` is the run's `STATEFP` lines, or None when there was no engine run at all. The pin-side
    checks run either way: they are most of what P1 can prove before a device is touched, and
    `--state-fp` means the driver has been asked to look at the state at all.

    `stand_in` marks a `--dry-run`: the fingerprint rows came from this file, not from the engine, and
    the helper-identity case is skipped — not because it matters less, but because it is a property of
    the pins and the shared helper rather than of an engine, and it is the one thing in this file that
    imports torch. `--selftest` runs it once, in its own process, and says so.
    """
    section("(g) G5-STATE — the engine's §08 state against the pinned fingerprints (§08.6, §10 G5)")
    pins = {k: prefill_fp(k) for k in FP_FAMILIES}
    if all(p is None for p in pins.values()):
        notes.append(f"(g) has nothing to compare: {REFS.name} carries no state fingerprints at all"
                     " (`dec8` was pinned before there were any). Nothing in this section ran, and"
                     " that is not a pass on the state — plan §5.1 says this fixture simply has no"
                     " (g) rows.")
        return
    want = fp_expectations(man)
    fmap = man.get("state_fingerprint", {}).get("rows_are_model_layers", {})
    check(fmap == want["layers"],
          "the pinned fingerprint row map is the §03.5 routing, re-derived from the manifest's"
          " layer_types (rows are model layers, not call order)",
          f"manifest {len(fmap.get('state', []))}/{len(fmap.get('conv', []))}/"
          f"{len(fmap.get('kv', []))} vs re-derived {len(want['layers']['state'])}/"
          f"{len(want['layers']['conv'])}/{len(want['layers']['kv'])}")

    # ---- the pin side: conventions the comparisons below stand on --------------------------------
    for kind in FP_FAMILIES:
        pin = pins[kind]
        if pin is None:
            notes.append(f"(g): {kind} has no untagged (prompt-end) fingerprint in {REFS.name}, so"
                         " that family's prompt-end comparison did not run.")
            continue
        check(bool(np.isfinite(pin).all()),
              f"the pinned prompt-end {kind}_fp values are finite")
        same = bool(np.array_equal(pin[:, 3], pin[:, 0]))
        check(same, f"the pinned prompt-end {kind}_fp rows carry §08.4's convention: no predecessor,"
                    " so Δ² IS the state (dn2 == n2)",
              "" if same else f"largest |dn2−n2| {np.max(np.abs(pin[:, 3] - pin[:, 0])):.3g}")

    if stand_in:
        print("  [info] fp-of-pins identity not re-run here: this is a --dry-run, and the identity is"
              " a property of the pins and `tools/state_fp.py`, not of an engine. `--selftest` runs it"
              " once in its own process; a real gate run runs it every time.")
    else:
        fp_helper_identity(man)

    if statefp is None:
        notes.append("(g) ran the pin side only: no engine run (--no-engine), so the engine's own"
                     " state was never observed and none of the §08.6 invariants below were tested."
                     " Re-run with a device pass.")
        return
    if not statefp:
        check(False, "the run printed its state fingerprints",
              "no STATEFP line in the output, so the engine's §08 state is invisible and G5's state"
              " half is unjudged. Pass `--state-fp` to decode_run (this gate passes it by default);"
              " a silent skip here is exactly the gap P1 was written to close.")
        return

    rows, meta = index_state_fp(statefp)
    phases = [p for p in ("prefill", "end") if p in meta]
    missing = [p for p in (["prefill"] + (["end"] if ran_decode > 0 else [])) if p not in meta]
    check(not missing, "the run emitted every end condition it reached",
          "; ".join(f"{p}: " + (meta.get("note:" + p) or "no phase line — the run stopped before"
                                " it got there") for p in missing))

    # ---- the mechanical block: geometry of every row ---------------------------------------------
    geometry, finite = [], []
    for phase in phases:
        for family in FP_FAMILIES:
            got = rows.get((phase, family), {})
            if [got[r]["layer"] for r in sorted(got)] != want["layers"][family]:
                check(False, f"{phase}: {family} rows map to the §03.5 layer routing",
                      f"engine {[got[r]['layer'] for r in sorted(got)][:6]}… vs routing"
                      f" {want['layers'][family][:6]}…")
            for r in sorted(got):
                g = got[r]
                if not np.isfinite([g[c] for c in FP_COLS]).all():
                    finite.append(f"{phase} {family} row {r}")
                if family == "state" and g["n"] != want["n"]["state"]:
                    geometry.append(f"{phase} state row {r}: n={g['n']}, §08.1 says"
                                    f" {want['n']['state']}")
                if family == "conv":
                    if g["n"] != want["n"]["conv"]:
                        geometry.append(f"{phase} conv row {r}: n={g['n']}, §08.2's window is"
                                        f" (k_conv−1)×{want['n']['conv'] // (want['k_conv'] - 1)}"
                                        f" = {want['n']['conv']}")
                    if g.get("cols") not in (None, want["k_conv"] - 1):
                        geometry.append(f"{phase} conv row {r}: cols={g['cols']}, §08.2 keeps"
                                        f" k_conv−1 = {want['k_conv'] - 1}")
                if family == "kv":
                    lanes = g.get("lanes")
                    if g["n"] != 4 * (lanes or 0) * want["head_dim"]:
                        geometry.append(f"{phase} kv row {r}: n={g['n']} is not 2 heads × 2 tensors"
                                        f" × lanes {lanes} × head_dim {want['head_dim']} — a KV"
                                        " fingerprint over unwritten slots is meaningless")
                    if lanes is not None and phase in meta and lanes != meta[phase]:
                        geometry.append(f"{phase} kv row {r}: cache claims {lanes} lanes but the"
                                        f" snapshot's prefix is {meta[phase]}")
            check(bool(got), f"{phase}: the engine emitted the {family} rows",
                  f"{len(got)} of {len(want['layers'][family])}")
    check(not geometry, "every emitted row's extent matches the geometry derived from the checkpoint",
          geometry[:4])
    check(not finite, "every emitted fingerprint value is finite", finite[:4])

    # ---- §08.6, invariant by invariant, engine side ---------------------------------------------
    inv: dict[str, list[str]] = {"1": [], "2": [], "3": [], "4": []}
    if "prefill" in meta:
        ok = meta["prefill"] == prompt_len
        inv["4"].append(f"prompt end at prefix {meta['prefill']}"
                        + ("" if ok else f", expected {prompt_len}"))
        check(ok, f"inv 4: the prompt-end snapshot sits at the fixture's prefix ({prompt_len})")
    if "end" in meta:
        want_prefix = prompt_len + ran_decode
        ok = meta["end"] == want_prefix
        inv["4"].append(f"run end at prefix {meta['end']}")
        check(ok, f"inv 4: the run-end snapshot advanced the prefix by one per generated token"
                  f" ({want_prefix})", f"engine says {meta['end']}")
    for phase in phases:
        zeroed = [f"{family} row {r}" for family in FP_FAMILIES for r in rows.get((phase, family), {})
                  if rows[(phase, family)][r]["maxabs"] == 0.0]
        check(not zeroed, f"§08.4: no state family is still all zero at the {phase} end — a zeroed"
                          " moment sum after tokens were fed means the state was never written",
              "offending: [" + ", ".join(zeroed[:6]) + "]")
    if "end" in meta and ran_decode > 0:
        for family, number in (("state", "1"), ("conv", "2")):
            got = rows.get(("end", family), {})
            still = [f"{family} row {r}" for r in sorted(got) if got[r]["dn2"] == 0.0]
            check(not still,
                  f"inv {number}: every {family} row moved between the prompt end and the run end"
                  f" ({ran_decode} tokens) — a Δ² of exactly 0 is a state left behind",
                  "offending: [" + ", ".join(still[:6]) + "]")
            inv[number].append(f"{len(got) - len(still)}/{len(got)} rows moved")
        got = rows.get(("end", "kv"), {})
        rewrote = [f"kv row {r} dn2 {got[r]['dn2']:.6g}" for r in sorted(got)
                   if got[r]["dn2"] != 0.0]
        check(not rewrote,
              "inv 3: the KV's already-written lanes did not move — Δ² exactly 0.0 over the common"
              " lane extent, part-wise (K vs K, V vs V)",
              "offending: [" + ", ".join(rewrote[:6]) + "]")
        inv["3"].append(f"{len(got) - len(rewrote)}/{len(got)} rows bit-still")
    elif "end" in meta:
        notes.append("(g) inv 1/2 could not be tested: no generation step ran, so there is no pair of"
                     " states to difference.")

    # ---- the distances: printed, priced, judged by nobody in this file ---------------------------
    print("  engine-vs-pin distance, per family and end condition. REPORTED, not gated: no threshold"
          " is adopted here (plan §2.6 gives that call to the main agent in the D-3/D-5 idiom, and a"
          " band move needs a `D-x` plus a bench/band-history.jsonl line). Yardstick for scale:"
          " G3's measured chunk-vs-recurrent agreement, A-08's cos >= 0.99999/layer, which on a moment"
          " sum is an n2 ratio inside ~2e-5 for the SAME quantity under a different sum order.")
    last = max(ran_decode - 1, 0)
    step_pins = {k: (load_fp(last, k) if "end" in meta else None) for k in FP_FAMILIES}
    distances: dict = {}
    for phase, pins_here in (("prefill", pins), ("end", step_pins)):
        if phase not in meta:
            continue
        for family in FP_FAMILIES:
            got = rows.get((phase, family), {})
            pin = pins_here.get(family)
            if not got or pin is None:
                notes.append(f"(g): no {family} comparison at the {phase} end — "
                             + ("the engine emitted no rows" if not got
                                else f"the pin is missing ({family}_fp)"))
                continue
            pin_n = want["pin_n"].get(family, want["n"].get(family, 1))
            caveat = ""
            if family == "conv":
                caveat = (f"EXTENT MISMATCH, NOT A STATE ERROR: the engine pooled"
                          f" k_conv−1 = {want['k_conv'] - 1} raw columns"
                          f" ({want['n']['conv']} elements), the pin pooled k_conv ="
                          f" {want['k_conv']} ({pin_n}); see the block comment for the pin evidence"
                          f" and `conv_window_pin` for the comparable number.")
            distances[(phase, family)] = fp_distance(got, pin, pin_n)
            print_fp_distance(f"engine({phase}) vs pin({phase}) {family}",
                              distances[(phase, family)], caveat)
            if family == "conv" and phase == "end":
                win = conv_window_pin(deep_state_step() or -1)
                if win is None:
                    notes.append("(g) could NOT price the conv window on this fixture: the pinned"
                                 " conv buffer is not kept at a step whose newest 3 columns can be"
                                 " pooled (`--deep-steps none`), so inv 2 rests on the engine-side"
                                 " Δ² line above and the extent-mismatched distance, and nothing"
                                 " else. A re-pin with --deep-steps is the only way past that, and"
                                 " re-pinning a shipped fixture is a main-agent call (plan §5.1).")
                else:
                    win_d = fp_distance(got, win, want["n"]["conv"])
                    print_fp_distance("engine(end) vs pin conv WINDOW (newest k_conv−1 cols)",
                                      win_d)
                    inv["2"].append(f"window at its own extent: n2 ratio worst"
                                    f" {win_d['n2_worst']:.6f}"
                                    + (f" at row {win_d['n2'][1][0]}" if win_d["n2"][1]
                                       else " (every row identical)"))
    for family, number in (("state", "1"), ("conv", "2"), ("kv", "3")):
        for phase in ("prefill", "end"):
            d = distances.get((phase, family))
            if d and d["ratios"]:
                if d["n2"][0] == 0.0:
                    inv[number].append(f"{phase}-end n2 ratio: every row identical")
                    continue
                worst = max(d["ratios"], key=lambda t: abs(t[2] - 1.0))
                inv[number].append(f"{phase}-end n2 ratio worst {worst[2]:.6f}"
                                   f" (row {worst[0]}, layer {worst[1]})")
    print("  §08.6, invariant by invariant — what this fixture proves here and what it cannot:")
    for number, text in (("1", "S is updated by every token, in all 18 GDN layers"),
                         ("2", f"the conv window holds the last k_conv−1 raw columns (A-04: the pin"
                               f" buffers k_conv)"),
                         ("3", "the KV is append-only: already-written slots never move"),
                         ("4", "the cache advanced by one per token; prefix == position")):
        print(f"    inv {number} — {text}: " + ("; ".join(inv[number]) if inv[number]
                                                     else "not testable on this run"))
    print("    not proved by fingerprints, at any tolerance: a corruption that preserves the multiset"
          " of values (a rotation inside a tensor), the ORDER of anything, or the per-element state —"
          " moment sums cannot see them, and full per-element state pins cost 18 MiB per step, which"
          " is the 1.7 GB a deep fixture exists to avoid (§08.6's own note, restated in (f)).")
    notes.append("(g) printed engine-vs-pin state distances and gated none of them. That is the"
                 " package as planned (§2.6: 'P1 ships a measurement and a tripwire, not a band'); the"
                 " adopted-threshold decision needs the route spread at the same position, i.e."
                 " `engine/route_price.py` extended to state fingerprints, then a `D-x` and a"
                 " band_history line. Nothing in this run is a band.")

# ---- (0) the fixture pairing -------------------------------------------------------------------
def pairing_checks(man: dict) -> tuple[list, list, np.ndarray]:
    """Section (0): does this fixture pair with itself? Inputs, per-step logits, the prefill anchor,
    and the last-writer-wins trap that the bare names are the LAST step's, not the prompt's.

    A step whose logits the fixture declared absent is skipped for the logits half only — its feed line
    is still checked — and the accounting is closed with a printed `paired N of M steps, K declared
    gaps`, because "the pairing passed" means nothing if it quietly passed on two steps out of twenty.
    A missing logits file that was NOT declared still exits 1 from `load_pin`, and `--selftest` keeps an
    injection for exactly that."""
    section("(0) fixture pairing — proved from the dumps (§10.5)")
    print(f"  manifest: {REFS / 'manifest.json'}")
    prompt_ids = man["sources"][0]["ids"]
    greedy = man["greedy_continuation"]
    print(f"  sources[0].ids        = {prompt_ids}")
    print(f"  greedy_continuation   = {greedy}")

    global gapset
    gapset = frozenset(declared_logit_gaps(man))
    n_steps = len(greedy) - 1
    # A declared gap must name a step this fixture HAS. A declaration for step 40 of a 24-step chain is
    # a stale copy of a manifest, and a stale declaration is how a real loss hides.
    beyond = sorted(s for s in gapset if s >= n_steps)
    check(not beyond,
          "every declared logit gap names a step this fixture has, so a stale manifest cannot quietly"
          " stand in for a lean pin",
          f"{n_steps} steps in the chain, and "
          + ", ".join(f"decode{s}" for s in beyond) + " which is not a step this fixture has"
          if beyond else "no gaps declared")
    gapset = frozenset(s for s in gapset if s < n_steps)
    # A declared gap that is not actually missing is a stale declaration, and a stale declaration is how
    # a real loss hides: the pin says step 12 carries nothing, step 12's file is sitting there, and the
    # next re-pin will "conveniently" lose it. Refuse the disagreement in both directions.
    lying = sorted(s for s in gapset if (REFS / f"decode{s}.logits_last.npy").exists())
    check(not lying,
          "every declared logit gap is actually absent from the fixture (the declaration and the disk"
          " agree, in both directions)",
          "declares a step whose logits are present: "
          + ", ".join(f"decode{s}" for s in lying) if lying else "no gaps declared")
    dec_logits = [None if s in gapset else load_pin(f"decode{s}.logits_last.npy").reshape(-1)
                  for s in range(n_steps)]
    dec_inputs = [load_pin(f"decode{s}.input_ids.npy").reshape(-1) for s in range(n_steps)]
    prefill_logits = load_pin("logits_last.npy").reshape(-1)

    paired = 0
    for s, (lg, ip) in enumerate(zip(dec_logits, dec_inputs)):
        # The input line is checked whether or not the logits were pinned: what was fed is a fact about
        # the chain, and a step that gave up its logits does not get to give up its feed list too.
        check(len(ip) == 1 and int(ip[0]) == greedy[s],
              f"decode{s}.input_ids == greedy_continuation[{s}]", ip.astype(int).tolist())
        if lg is None:
            continue
        meta = man["pins"][f"decode{s}.logits_last"]
        check(meta["step"] == f"decode{s}", f"manifest declares decode{s}.logits_last step",
              f'{meta["step"]} / {meta["shape"]} / {meta["dtype"]}')
        check(argmax_low(lg) == greedy[s + 1],
              f"argmax(decode{s}.logits_last) == greedy_continuation[{s + 1}]",
              f"{argmax_low(lg)} vs {greedy[s + 1]}")
        paired += 1
    check(paired + len(gapset) == n_steps,
          "every step of the chain is either paired against its logits or declared absent — none"
          " silently unjudged",
          f"paired {paired} of {n_steps} steps, {len(gapset)} declared gaps"
          + (f" ({', '.join('decode' + str(s) for s in sorted(gapset))})" if gapset else ""))
    check(paired > 0 or n_steps == 0,
          "at least one step of the chain still carries logits to pair against: a fixture that"
          " declares EVERY step away is a manifest, not a pin (the dumper refuses `--logit-steps none`"
          " so this cannot be made by accident)",
          f"{len(gapset)} of {n_steps} steps declared away")
    check(argmax_low(prefill_logits) == greedy[0],
          "argmax(bare logits_last.npy) == greedy_continuation[0] — so the bare name is the PREFILL",
          f"{argmax_low(prefill_logits)} vs {greedy[0]}")
    bare_inputs = load_pin("input_ids.npy").reshape(-1)
    if n_steps == 0:
        # No decode step ran, so there was never a second writer of the bare names: `--decode 0`
        # fixtures (`p8`, `p64`, `p300` — see `tools/dump_fixtures.sh`) leave `input_ids.npy` holding
        # THE PROMPT, which is the pairing a reader wants anyway, and §10.5's last-writer-wins trap
        # has nothing to bite on. The line below cannot be evaluated at all here — `greedy[-2]` has no
        # referent when the chain is one token long — and it raised IndexError out of this section on
        # all three fixtures (§15.2's first crash). The GEOMETRY decides which of the two facts is
        # asserted, not a hand-listed fixture name, so a future `--decode 0` fixture is covered the
        # day it is dumped. And the replacement is a check, not a waiver: it compares every id, so a
        # bare `input_ids.npy` that came from anywhere other than the prompt still fails.
        check(bare_inputs.astype(int).tolist() == list(man["prefill_ids"]),
              "bare input_ids.npy IS the prompt: this fixture has no decode step, so nothing"
              " overwrote it (§10.5's last-writer-wins trap needs a decode phase, and `--decode 0`"
              " has none)",
              f"first 6 of {bare_inputs.size} ids {bare_inputs.astype(int).tolist()[:6]}"
              f" vs prefill_ids ({len(man['prefill_ids'])} ids); full-list equality required")
        notes.append(
            f"{REFS.name} has no decode step (`--decode 0`), so the per-step half of (a) and all of"
            " (b) judge zero steps and prove nothing about a decode chain. What this fixture DOES"
            " prove is named where it is proved: §06.5's prefill row against the bare"
            " `logits_last.npy` (section (0) proves the pin's argmax, section (a) re-judges the"
            " engine's y0 against it), the prompt wave's step count, §08.6's state pins for the"
            " prefill pass, and the chunk-64 padding §10.5 requires p64/p300 for. A PASS here is"
            " that claim, and it is not a decode claim.")
    else:
        check(int(bare_inputs[0]) == greedy[-2],
              f"bare input_ids.npy is decode{len(greedy) - 2}'s input, NOT the prompt — recorded so it"
              " is never used as one (§10.5's last-writer-wins trap)",
              f"{bare_inputs.astype(int).tolist()} == greedy_continuation[-2] == {greedy[-2]}")
    seen = first_pinned_logits(dec_logits)
    print(f"  [info] prompt ids come from the manifest's sources[0].ids; the prompt's own logits"
          f" come from the bare logits_last.npy, whose argmax was just proved to be"
          f" greedy_continuation[0]. "
          + (f"Pinned logits read as {seen.dtype}, shape {seen.shape}, stored as"
             f" {man.get('logits_storage', 'f32 (pre-dates --logits-bits)')}."
             if seen is not None else
             "No step's logits are pinned, so there is no per-step logits storage to report; the"
             " bare prefill logits are the only token evidence this fixture carries."))
    stop_at = [i for i, t in enumerate(prompt_ids) if t == STOP_ID]
    print(f"  [info] the prompt itself contains the stop id {STOP_ID} at index {stop_at}, so a"
          " prompt-phase step can legitimately sample it. §07.5 is a generation-phase rule and such"
          " a match must be ignored (section (a) asserts both halves of that).")
    return prompt_ids, dec_logits, dec_inputs


# ---- (i) the pinned PREFILL ROWS — every row of the wave, not only §06.5's last one --------------
# Why this section exists: `prototype/attn_decode/results.md` §15 is a dispute about ROW 1023 of a
# 1024-token prompt, and every fixture made before `reference_dump.py --prefill-logits all` carries one
# prefill logits vector — the last one. Above 512 lanes there was therefore no ground truth in this tree
# for ANY route: two engine forms could be compared with each other (`engine/attn_form_band.py`) and
# nothing else. `wide640f` (640 rows) and `wide1024f` (1024 rows) are that ground truth, and this is the
# section that judges an engine run against them.
#
# It judges with the floors that already exist and adds none. Which ones, and why those:
#
#   * `BAND["cos_hard"]` (§10.1 G5, "logits cos >= 0.9995") and `BAND["meanrel_hard"]` (§10.2 cascade
#     ceiling, meanrel <= 5 %) are the published floors for "one logits vector, engine against the
#     reference", and a prefill row IS one logits vector. They are what section (b) applies to a fixture
#     with no priced route allowance, and these two fixtures have none.
#   * On the library route both caps then pass through `cascade_floor`, which can only widen them (to
#     `CASCADE_BAND`'s measured hand<->library cascade). That is the correct floor here for a sharper
#     reason than usual: rows above MAXSC are the STREAMING form of §07.3, i.e. a deliberately
#     reassociated sum, and demanding a tighter deficit than a legal reassociation produces is demanding
#     the route not be itself.
#   * A moved argmax is judged by `flip_admissible`, the same room rule section (a)/(b) use, on the same
#     grounds (§A2 of progress/2026-10-14/route-c-audit-2026-10-14.md: its 1694 rows are prefill rows too).
#
# What a ROW cannot consult, and what happens instead — said here because a silent hole in an argmax rule
# is how a gate goes quietly wrong: the D-8 tie policy is keyed to a per-step pin FILE
# (`pin_tie(step)` opens `logits_pin(step)`), so a tie inside a row BLOCK has no predicate to consult.
# Rather than invent a second tie policy, a row whose pin ties is judged by the one clause that needs no
# slack bound — the engine's id must be one of the pinned tied ids — and anything else FAILS. That is
# fail-closed: the worst it can do is report a real-looking divergence on a row where the reference
# itself did not choose, and it prints the tie set so the reader can see that is what happened.
PREFILL_ROWS_PIN = "logits_rows.npy"


def prefill_row_pin_checks(man: dict, prompt_len: int, vocab: int):
    """Prove what the fixture's pinned prefill-row block claims, before anything is judged against it.

    Runs in BOTH modes (`--no-engine` included): it is a statement about the dump, not about the engine.
    The load-bearing one is that row -1 of `logits_rows` IS `logits_last`, bit for bit — the two come
    from one forward, so if they ever differ the block is not what its name says. §10.5 is written
    because a bare dump name once paired with the wrong thing and cost a debugging session; a block pin
    is the same failure with more rows, and the pairing must be proved from bytes rather than from the
    fact that both were written by one script.

    Returns the memmapped block for the engine section, or None when the fixture pins no rows.
    """
    declared = str(man.get("prefill_logit_rows", "last"))
    path = REFS / PREFILL_ROWS_PIN
    if declared != "all":
        check(not path.exists(), f"{REFS.name} pins prefill rows: {declared}",
              f"the manifest says prefill_logit_rows={declared!r} but {PREFILL_ROWS_PIN} sits beside"
              " it. Which half is the lie is not this gate's to guess: either the dumper wrote a"
              " manifest that does not describe its own output, or a file was copied in by hand."
              if path.exists() else
              f"no {PREFILL_ROWS_PIN}, as the manifest says — only §06.5's last row is judged for"
              " this fixture, which is every fixture made before the flag existed")
        return None
    if not check(path.exists(), f"{PREFILL_ROWS_PIN} is present, as the manifest claims",
                 f"{path.stat().st_size / (1024 ** 2):.1f} MiB beside the manifest"
                 if path.exists() else
                 "the manifest says every prefill row is pinned and the file is gone — a"
                 " half-installed fixture, and section (i) would have nothing to judge"):
        return None
    raw = np.load(path, allow_pickle=False)
    shape_ok = list(raw.shape) == [prompt_len, vocab]
    check(shape_ok,
          f"{PREFILL_ROWS_PIN} is [{prompt_len}, {vocab}] — one row per prompt token, one column per"
          " vocabulary id",
          f"{list(raw.shape)}" if shape_ok else
          f"{list(raw.shape)}; a row count that is not the prompt length means the block and the"
          " prompt ids came from different runs, which is §10.5's trap in block form")
    stored = str(man.get("logits_storage", "f32"))
    storage_ok = (raw.dtype == np.uint16) == (stored == "bf16")
    check(storage_ok, f"{PREFILL_ROWS_PIN} storage matches the manifest's logits_storage ({stored})",
          f"file dtype {raw.dtype}" if storage_ok else
          f"file dtype {raw.dtype} does not match the declared {stored!r}: the block is being widened"
          " by the wrong rule, which is how a BF16 pin ends up read as garbage (§10.5)")
    last = np.asarray(load_pin("logits_last.npy"), dtype=np.float32).reshape(-1)
    row_last = np.asarray(widen_pin(raw[-1]), dtype=np.float32).reshape(-1)
    same = np.array_equal(row_last, last)
    check(same, f"{PREFILL_ROWS_PIN} row -1 IS `logits_last`, bit for bit — the block's own pairing"
                " proof (§10.5)",
          "" if same else f"{int(np.count_nonzero(row_last != last))} of {row_last.size} elements"
          " differ, so the block was not produced by the forward that produced the pinned last row")
    gib = raw.nbytes / (1024 ** 2)
    print(f"  [info] {PREFILL_ROWS_PIN}: {raw.shape[0]} rows x {raw.shape[1]} ({raw.dtype},"
          f" a BF16 bitcast) = {gib:.1f} MiB. Row i pairs with the engine's logits<i>.bin, so"
          f" {raw.shape[0] - 1} rows above §06.5's are now adjudicable against the reference"
          " instead of only against another engine build.")
    return raw


def prefill_row_coverage(n_rows: int, work: Path, vocab: int, asked: str,
                         echoed: str | None) -> tuple[bool, str]:
    """Did the run deliver the row coverage this fixture's pins require? A coverage verdict, never a
    numerics one — the caller stops here rather than carrying a gap into a comparison.

    Three distinct ways to lose coverage, and they must not read as one another:

      * the gate asked for `last` while the fixture pins every row — a forcing/command-line gap, so the
        engine could only ever have written one row;
      * the gate asked for `all` and the run's own `PREFILL logits=` echo says `last` — the flag did not
        reach `decode_run`. That is exactly §17's defect: the forcing was written *after* the command
        line had been assembled, so it moved the gate's bookkeeping and left the engine running `last`.
        Nothing downstream could tell, and 1023 missing rows read as an engine failure;
      * the echo says `all` and the files are still absent or short — the engine priced rows it did not
        write, or died part-way through the prompt.

    Why this is a separate verdict instead of one more row inside section (i): with a coverage gap, every
    aggregate over the rows that DID arrive is a number over an arbitrary subset, and section (i)'s
    existing `check(not outside, "every judged prefill row is inside the floors")` passes when nothing
    was judged. §17's shipped run is that failure: `1023 missing` sat next to a green floor verdict, which
    is a numerics result made out of nothing. Pure (no globals, no prints) so `--selftest` can give it
    teeth at vocabulary 8.
    """
    if n_rows <= 0:
        return False, f"the row block pins {n_rows} rows — a block with no rows is not coverage"
    if asked != PREFILL_LOGITS_ALL:
        return False, (f"COVERAGE GAP: the gate asked `decode_run` for `--prefill-logits {asked}` but"
                       f" {REFS.name} pins all {n_rows} prefill rows, so at most row {n_rows - 1} can"
                       " exist. This is the instrument's command line, not an engine defect: the"
                       " fixture's `prefill_logit_rows` must drive the invocation.")
    if echoed is None:
        return False, (f"COVERAGE GAP: the run printed no `PREFILL logits=` field, so nothing in its"
                       " log says which row shape it ran (a pre-P2 binary, or a `--out` it could not"
                       " write). A run that cannot say what it dumped cannot be judged on what it"
                       " dumped")
    if echoed != PREFILL_LOGITS_ALL:
        return False, (f"COVERAGE GAP: the gate asked for `all` and the run's own PREFILL line says"
                       f" logits={echoed}, so the flag did not reach decode_run — §17's exact defect"
                       " (a forcing that fires after the command line is built changes nothing the"
                       " engine sees). No numerics verdict follows from a row shape this gate did not"
                       " run")
    missing, short = [], []
    for i in range(n_rows):
        path = work / f"logits{i}.bin"
        if not path.exists():
            missing.append(i)
        elif path.stat().st_size != vocab * 4:
            short.append((i, path.stat().st_size))
    if missing or short:
        parts = []
        if missing:
            parts.append(f"{len(missing)} missing, first at rows {missing[:8]}")
        if short:
            parts.append(f"{len(short)} truncated, first "
                         f"logits{short[0][0]}.bin={short[0][1]} B, expected {vocab * 4} B")
        return False, (f"COVERAGE GAP: {n_rows} row dumps were asked for and {n_rows - len(missing)}"
                       f" are readable — " + "; ".join(parts) + ". Nothing was compared, so no numerics"
                       " verdict is issued: a coverage gap that falls through into a row comparison"
                       " turns into a floor verdict over whatever survived")
    return True, (f"{n_rows}/{n_rows} row dumps present, one per pinned row"
                  f" ({n_rows * vocab * 4 / (1024 ** 2):.0f} MiB of logits)")


def prefill_row_checks(work: Path, vocab: int, rows_pin, asked: str,
                       echoed: str | None) -> None:
    """Every pinned prefill row against the engine's own dump, judged by the floors already in force.

    The row count comes from the pin block, not from a prompt length passed in: `prefill_row_pin_checks`
    has already proved that block's shape against the prompt ids, and a second source for the same number
    is a second chance to disagree.

    `asked`/`echoed` are the row shape this gate requested and the row shape the run's own PREFILL line
    says it ran; the coverage proof runs on them before a single row is read, and a gap ends the section.
    """
    section("(i) the prefill rows — every pinned row against the engine's dump (§10.1 G5, §07.3)")
    n_rows = rows_pin.shape[0]
    # Coverage first, and nothing else prints until it passes. A gap here is not a numerics question and
    # must not arrive dressed as one: §17's shipped run printed its floors/geometry preamble and a green
    # `every judged prefill row is inside the floors` on top of `1023 missing`.
    cov_ok, cov_why = prefill_row_coverage(n_rows, work, vocab, asked, echoed)
    if not check(cov_ok, f"coverage: the engine delivered all {n_rows} pinned prefill rows", cov_why):
        return
    staging, _, _ = lane_geometry()
    cos_cap, rel_cap, floor_clause = cascade_floor(1.0 - BAND["cos_hard"], BAND["meanrel_hard"])
    rule = (f"  rule: per row, cos deficit <= {cos_cap:.4e} and meanrel <= {rel_cap:.4%}"
            + (f" — floored there, not chosen: {floor_clause}" if floor_clause else
               " — the flat §10.1 G5 and §10.2 floors; no route allowance is priced for prefill rows"
                 " (route_allowance.json prices decode steps)"))
    print(rule)
    print("  a moved argmax is judged by "
          + (CASCADE_RULE if route_is_library() else
             "the room rule, which is not armed on the hand route: an argmax that moved there is a"
             " defect, because bit-identity is the claim on that route"))
    if n_rows > staging:
        print(f"  geometry: MAXSC {staging} lanes is the one-pass/streaming boundary, so rows"
              f" 0..{staging - 1} are one-pass and rows {staging}..{n_rows - 1} are the streaming"
              f" form §07.3 bands — {n_rows - staging} of the {n_rows} pinned rows")
    else:
        print(f"  geometry: MAXSC {staging} lanes, and every row of this {n_rows}-row prompt fits"
              " it — this fixture does not exercise the streaming form")

    # Streaming accumulators, one row in memory at a time: the block is 303-485 MiB of BF16 and the
    # engine's dumps are 968 KiB each, so holding both sides of the whole comparison would be a
    # gigabyte for no benefit. `why` from read_one_logits is not re-quoted per row: a missing row is
    # counted and named once, below, and 1024 identical reasons would hide that count.
    regions: dict[str, dict] = {}
    worst_cos, worst_rel = (0.0, -1), (0.0, -1)
    at_row: dict[int, float] = {}       # deficit at the boundary rows worth printing by name
    outside: list[str] = []
    missing: list[int] = []
    flips: list[str] = []
    tied_rows: list[str] = []
    pin_thin: list[int] = []      # pinned rows whose own top-1/top-2 room is under the flip threshold
    pin_tied: list[int] = []      # pinned rows that tie at their own maximum
    pooled = dict(n=0, dot=0.0, na=0.0, nb=0.0, sad=0.0, sar=0.0)
    for i in range(n_rows):
        eng, _why = read_one_logits(work, i, vocab)
        if eng is None:
            missing.append(i)
            continue
        ref = np.asarray(widen_pin(np.asarray(rows_pin[i])), dtype=np.float32).reshape(-1)
        m = metrics(eng, ref)
        region = "one-pass" if i + 1 <= staging else "streaming"
        r = regions.setdefault(region, dict(n=0, identical=0, over_cos=0, over_rel=0, moved=0))
        r["n"] += 1
        if m["ndiff"] == 0:
            r["identical"] += 1
        deficit = 1.0 - m["cos"]
        if i in (staging - 1, staging, n_rows - 1):
            at_row[i] = deficit
        if deficit > cos_cap:
            r["over_cos"] += 1
            outside.append(f"row {i} ({region}): cos deficit {deficit:.4e} > {cos_cap:.4e}"
                           f" (cos {m['cos']:.8f}), meanrel {m['meanrel']:.4%},"
                           f" ndiff {m['ndiff']}/{ref.size}, maxabs {m['maxabs']:.4g}")
        if m["meanrel"] > rel_cap:
            r["over_rel"] += 1
            outside.append(f"row {i} ({region}): meanrel {m['meanrel']:.4%} > {rel_cap:.4%}"
                           f" (cos deficit {deficit:.4e}), ndiff {m['ndiff']}/{ref.size}")
        if deficit > worst_cos[0]:
            worst_cos = (deficit, i)
        if m["meanrel"] > worst_rel[0]:
            worst_rel = (m["meanrel"], i)
        pin_id, pin_val, pin_room = top2(ref)
        eng_id, eng_val, eng_room = top2(eng)
        pin_ulp = room_in_ulp(pin_val, pin_room)
        # A census of the PIN alone, taken before anything is compared: how many of these rows is the
        # reference itself ill-equipped to decide. It is printed with the verdict because the flip rule
        # forgives a move on exactly this class of row, and a reader who cannot see how large that class
        # is cannot tell a green run from a green run that had a lot of room to be green.
        if pin_ulp < CASCADE_BAND["flip_room_max_ulp"]:
            pin_thin.append(i)
        if int(np.count_nonzero(ref == ref.max())) > 1:
            pin_tied.append(i)
        if eng_id != pin_id:
            r["moved"] += 1
            tied = np.flatnonzero(ref == ref.max())
            if len(tied) > 1:
                ok = int(eng_id) in {int(t) for t in tied}
                line = (f"row {i} ({region}): the PIN TIES {len(tied)} ways at {float(ref.max())!r}"
                        f" among {intlist(tied)}; the engine picked {eng_id}"
                        + (" — one of the tied ids, so the reference had not decided this row"
                           if ok else " — NOT one of the tied ids, so this is a divergence, not a"
                           " tie"))
                if ok:
                    tied_rows.append(line)
                check(ok, f"row {i}: the engine's argmax moved against a TIED pinned row", line)
            else:
                ok = room_is_thin(pin_ulp)      # the REFERENCE reading; `flip_admissible` is for pairings with no reference
                row_ulp = pin_ulp
                line = (f"row {i} ({region}): pin top1 {pin_id} ({pin_val:.6g}) over its runner-up by"
                        f" {pin_room:.6g} = {pin_ulp:.3g} ULP vs engine {eng_id} ({eng_val:.6g})"
                        f" over its own runner-up by {eng_room:.6g}"
                        + (f" — ADMITTED as a thin-room flip: the reference's own room is"
                           f" {row_ulp:.4g} BF16 ULP, under"
                           f" {CASCADE_BAND['flip_room_max_ulp']:.2f}, which is the class §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md measured"
                           if ok else
                           f" — RED: neither side had under"
                           f" {CASCADE_BAND['flip_room_max_ulp']:.2f} BF16 ULP of"
                           " room, so no band in force covers this flip. Stop and report; this"
                           " section does not widen one"))
                if ok:
                    rooms.append(f"prefill row {i}: {pin_id} -> {eng_id}"
                                 f" (room {row_ulp:.4g} BF16 ULP)")
                flips.append(line)
                check(ok, f"row {i}: the engine's argmax moved against the pinned row", line)
        e = eng.astype(np.float64)
        f64 = ref.astype(np.float64)
        pooled["n"] += f64.size
        pooled["dot"] += float(e @ f64)
        pooled["na"] += float(e @ e)
        pooled["nb"] += float(f64 @ f64)
        pooled["sad"] += float(np.abs(e - f64).sum())
        pooled["sar"] += float(np.abs(f64).sum())

    if missing:
        # Unreachable once `prefill_row_coverage` has passed — it counts the same files, same sizes — and
        # still wrong to let fall through: a row that went unread between the count and the read means
        # this section is about to issue a floor verdict over fewer rows than it claims to have judged.
        # Deliberately not a second copy of the coverage PASS line: one claim, one name.
        check(False, "the engine wrote every prefill row the coverage check counted",
              f"{len(missing)} became unreadable after the coverage pass, first at rows {missing[:8]}"
              " — the section stops here rather than judging a subset")
        return
    for name in ("one-pass", "streaming"):
        r = regions.get(name)
        if not r:
            continue
        print(f"  {name:9s} rows: {r['n']:5d} judged, bit-identical {r['identical']:5d},"
              f" outside the cos cap {r['over_cos']:4d}, outside the meanrel cap {r['over_rel']:4d},"
              f" argmax moved {r['moved']:4d}")
    for i in sorted(at_row):
        which = {staging - 1: "last one-pass row", staging: "first streaming row",
                 n_rows - 1: "last row of the prompt (the one §15 disputes)"}[i]
        print(f"  boundary row {i} ({which}): cos deficit {at_row[i]:.4e}")
    print(f"  the pin's own fragile rows, counted on the reference alone: {len(pin_tied)} tie at their"
          f" maximum {intlist(pin_tied)}, {len(pin_thin)} have top-1/top-2 room under"
          f" {CASCADE_BAND['flip_room_max_ulp']:.2f} BF16 ULP {intlist(pin_thin)} — on the other"
          f" {n_rows - len(set(pin_tied) | set(pin_thin))} of the {n_rows} pinned rows an argmax move is"
          " a defect by the rule already in force, which is what makes a green run here a claim about the"
          " engine rather than a tautology about the fixture (census: tools/prefill_row_rooms.py)")
    total = sum(r["n"] for r in regions.values())
    # `total == n_rows` is in the check on purpose: a verdict about "every judged row" must not be
    # issuable when the judging was empty. That is the vacuous-pass half of §17's artifact — 1 row of
    # 1024 judged, 0 outside, and a green line that read like a numerics result.
    check(total == n_rows and not outside,
          f"every one of the {n_rows} pinned prefill rows was judged and is inside the floors named"
          " above",
          ("; ".join(outside[:6]) + (f" … ({len(outside)} lines in all)" if len(outside) > 6 else "")
           if outside else f"{total} rows judged: worst cos deficit {worst_cos[0]:.4e} at row"
           f" {worst_cos[1]}, worst meanrel {worst_rel[0]:.4%} at row {worst_rel[1]}"))
    if pooled["n"] and pooled["na"] and pooled["nb"] and pooled["sar"]:
        pcos = pooled["dot"] / (pooled["na"] ** 0.5 * pooled["nb"] ** 0.5)
        prel = pooled["sad"] / pooled["sar"]
        print(f"  pooled over all {pooled['n']:,} compared elements — REPORTED, not judged, because"
              " the published floors are per logits vector: cos "
              f"{pcos:.8f} (1-cos {1 - pcos:.3e}), meanrel {prel:.3e}")
    for line in flips:
        print("  [flip] " + line)
    if tied_rows:
        print(f"  [ROW-TIE] {len(tied_rows)} row(s) where the pin itself tied and the engine picked"
              " one of the pinned tied ids. D-8's tie policy is keyed to a per-step pin file and was"
              " NOT generalised to a row block, so the narrower clause (the engine's id must be one of"
              " the tied ids, with no quantization-slack allowance) is what judged them: "
              + "; ".join(tied_rows))


def intlist(a) -> str:
    """An id list, capped: a tied row in a 248 320-wide vocabulary could print a thousand numbers."""
    ids = [int(x) for x in a]
    return str(ids[:8]) + (f" … {len(ids)} in all" if len(ids) > 8 else "")


def selftest_room_readings(fixture: str, step: int | None) -> list[str]:
    """One injected row, judged by BOTH room readings, and they must DISAGREE.

    This is the witness that the flip rule has two readings on purpose. The row is
    `--break-runnerup`'s: the pin decided step `step` by a wide margin, and the engine lifted the pin's
    runner-up exactly one BF16 step over the pin's maximum — the narrowest illegal choice this tree
    builds. Judged as a reference comparison the reference's own room says RED; judged with the
    symmetric min() the engine's own one-step gap says ADMITTED. Both are printed, with the ULP figures,
    and the case fails if either verdict moves:

      * merge the two pairings back into one path and one of the two flips colour — which is exactly how
        `wide520f` row 180 (cross-width, must-admit at 0.9985 ULP) and this row (reference, must-refuse
        at 1.00 ULP engine-side) came to need opposite answers from one constant;
      * loosen the slack bound, and the row stops being decidable for a different reason.

    Nothing here is a device result and nothing is timed: it is arithmetic on pinned rows.
    """
    if not route_is_library():
        print("  [SKIP] the two room readings: the flip rule is consulted on library routes only;"
              " `--gemm hand` never reaches it (its own ids check is the whole rule there).")
        return []
    if step is None:
        print(f"  [SKIP] the two room readings: {fixture} has no step whose margin clears its licensed"
              " element error, so there is no decidable row to aim the injection at.")
        return []
    pin = pin_row(step)
    eng = lift_over_pin(step, pin, pin_runnerup(step, "--break-runnerup"))
    _p_id, p_ulp, p_gap, p_val = reference_room_ulp(pin)
    e_id, e_val, e_gap = top2(eng)
    e_ulp = room_in_ulp(e_val, e_gap)
    ref_thin = room_is_thin(p_ulp)
    min_thin = room_is_thin(min(p_ulp, e_ulp))
    detail = (f"step {step}: pin top1 over its runner-up by {p_gap:.6g} = {p_ulp:.2f} ULP;"
              f" engine top1 {e_id} ({e_val:.6g}) over ITS runner-up by {e_gap:.6g} = {e_ulp:.2f} ULP"
              f" → reference reading {'ADMITTED' if ref_thin else 'RED'}"
              f" at {p_ulp:.2f} ULP, min()-over-sides reading"
              f" {'ADMITTED' if min_thin else 'RED'} at {min(p_ulp, e_ulp):.2f} ULP"
              f" (line {CASCADE_BAND['flip_room_max_ulp']:.2f})")
    ok = (not ref_thin) and min_thin
    print(f"  [{'PASS' if ok else 'FAIL'}] the same injected wrong-id, judged by both room readings,"
          f" must split: {detail}" + ("" if ok else
      " — the two readings must disagree in THIS direction: the reference decided (RED) while the"
        " engine's own near-tie would admit. If they agree, one pairing has been folded into the other"
        " and the distinction §3f of prototype/gemm_algo_sweep/results.md exists for is gone"))
    return [] if ok else [f"the two room readings must split at step {step}: {detail}"]


def selftest_prefill_rows() -> list[str]:
    """The teeth of section (i), on synthetic rows — no engine, no device, no fixture read.

    Section (i) judges 640-1024-row blocks, which is too big to inject through the stand-in (the
    dry-run stand-in dumps only §06.5's row, and a ~1 GiB stand-in dump would make every dry run a disk
    benchmark). So its negatives live here, at vocabulary 8: same predicates, same code path, and each
    case names the one thing it must and must not call a failure — the rule that a negative which catches
    everything measures nothing applies to a new section exactly as it applies to a new kernel.

    The second half of this function is the §17 witness, and it is the one this repo cannot skip: a
    coverage gap must fail, and it must fail WITHOUT issuing a numerics verdict. Three checkers on this
    box have already been able to pass while measuring nothing; a fourth is one too many. So the coverage
    cases assert on the printed output as well as on the failure list — a green `every one of the N
    pinned prefill rows was judged…` line while rows are missing is the specific thing that must be
    impossible, not merely accompanied by a warning.
    """
    bad: list[str] = []
    vocab = 8
    # Two pinned rows, built so each case isolates ONE predicate. `wide` is won by id 7 with 14 of room,
    # so anything that beats it beat it with room to spare — a wide-room flip. `near` is won by id 7 with
    # 0.05 of room, so a flip off it is a thin-room flip and NOTHING ELSE: swapping the two top values
    # moves cos by ~2e-5 and meanrel by 0.25 %, both well inside the floors, so if the thin case goes red
    # the flip rule is what broke, not the shape test.
    wide = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 20.0], dtype=np.float32)
    near = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 19.95, 20.0], dtype=np.float32)
    # The line that may never print over a coverage gap. Named once here rather than pasted per case.
    VERDICT = "was judged and is inside the floors"

    def run_case(name: str, pins: list[np.ndarray], rows: list[np.ndarray | None], *,
                 want_fail: bool, needle: str = "", forbid: tuple[str, ...] = (),
                 asked: str = PREFILL_LOGITS_ALL, echoed: str | None = PREFILL_LOGITS_ALL,
                 truncate: tuple[int, ...] = ()) -> None:
        """Judge `rows` against `pins` (None = the engine wrote no such file).

        `asked`/`echoed` are what the gate requested and what the run's PREFILL line claims it ran, so a
        case can reproduce §17 exactly: asked `all`, echoed `last`, i.e. a forcing that never reached the
        engine. `truncate` writes those rows short, which is a third way to lose a row and must read as
        coverage rather than as a numerics failure at that row.
        """
        before = list(fails)
        shown = io.StringIO()
        with tempfile.TemporaryDirectory(prefix="gate_rows_") as td:
            work = Path(td)
            for i, r in enumerate(rows):
                if r is not None:
                    raw = r.astype("<f4")
                    if i in truncate:
                        raw = raw[:max(1, raw.size // 2)]
                    (work / f"logits{i}.bin").write_bytes(raw.tobytes())
            pin = np.vstack(pins)
            keep = sys.stdout
            sys.stdout = shown
            try:
                prefill_row_checks(work, vocab, pin, asked, echoed)
            finally:
                sys.stdout = keep
            new = fails[len(before):]
            fails[:] = before
        text = shown.getvalue()
        got = bool(new)
        named = (not needle) or any(needle in f for f in new) or needle in text
        offenders = [f for f in forbid if f in text]
        ok = got == want_fail and named and not offenders
        print(f"  [{'PASS' if ok else 'FAIL'}] section (i) {name}"
              + (f" — {'failed' if got else 'clean'}, as it had to" if ok else
                 f" — wanted {'a failure' if want_fail else 'clean'}, got"
                 f" {(new[:1] or text.splitlines()[-2:])}"
                 + (f"; it printed a verdict it had no business printing: {offenders}"
                    if offenders else "")))
        if not ok:
            bad.append(f"section (i) {name}")

    run_case("judges a bit-identical row as clean", [wide, wide], [wide.copy(), wide.copy()],
             want_fail=False)
    moved = wide.copy()
    moved[6] = 21.0                       # wins, and leaves 1.0 of room on the engine side
    run_case("REDs a moved argmax whose room is wide", [wide, wide], [wide.copy(), moved],
             want_fail=True, needle="neither side had under")
    swapped = near.copy()
    swapped[6], swapped[7] = near[7], near[6]     # the engine's winner is the pin's runner-up
    run_case("admits a moved argmax whose room is thin", [near, near], [near.copy(), swapped],
             want_fail=False)
    loud = wide.copy()
    loud[0] = 200.0                       # argmax untouched, whole-vector shape destroyed
    run_case("REDs a row outside the cos/meanrel floors", [wide, wide], [wide.copy(), loud],
             want_fail=True, needle="cos deficit")

    # ---- the §17 witness: a coverage gap fails, and issues no numerics verdict ------------------
    run_case("reports a row the engine never dumped", [wide, wide], [wide.copy(), None],
             want_fail=True, needle="missing",
             forbid=(VERDICT, "rows:     1 judged"))
    run_case("reports a run that dumped NO rows at all", [wide, wide], [None, None],
             want_fail=True, needle="COVERAGE GAP", forbid=(VERDICT, "bit-identical"))
    run_case("reports a row-shaped mismatch with the engine's own echo (§17's defect: asked `all`,"
             " the run says `last`)", [wide, wide], [wide.copy(), wide.copy()],
             want_fail=True, needle="did not reach decode_run",
             echoed=PREFILL_LOGITS_LAST, forbid=(VERDICT,))
    run_case("names a gate that asked for the wrong row shape at all", [wide, wide],
             [wide.copy(), None], want_fail=True, needle="prefill_logit_rows",
             asked=PREFILL_LOGITS_LAST, forbid=(VERDICT,))
    run_case("reports a run whose log never said which row shape it ran", [wide, wide],
             [wide.copy(), wide.copy()], want_fail=True, needle="no `PREFILL logits=` field",
             echoed=None, forbid=(VERDICT,))
    run_case("treats a truncated row dump as coverage, not as a numerics failure at that row",
             [wide, wide], [wide.copy(), wide.copy()], truncate=(1,),
             want_fail=True, needle="truncated", forbid=(VERDICT, "cos deficit"))

    # The coverage predicate on its own, so a regression in it cannot hide behind section (i)'s prints.
    def cov(rows: list[np.ndarray | None], *, asked: str = PREFILL_LOGITS_ALL,
            echoed: str | None = PREFILL_LOGITS_ALL) -> tuple[bool, str]:
        with tempfile.TemporaryDirectory(prefix="gate_cov_") as td:
            work = Path(td)
            for i, r in enumerate(rows):
                if r is not None:
                    (work / f"logits{i}.bin").write_bytes(r.astype("<f4").tobytes())
            return prefill_row_coverage(len(rows), work, vocab, asked, echoed)

    for what, (ok, why) in {
            "full coverage passes": cov([wide.copy(), wide.copy()]),
            "a hole fails": cov([wide.copy(), None]),
            "asking `last` fails": cov([wide.copy(), None], asked=PREFILL_LOGITS_LAST),
            "an echo that disagrees with the ask fails": cov([wide.copy(), wide.copy()],
                                                            echoed=PREFILL_LOGITS_LAST),
            "an empty block fails rather than passing vacuously": cov([]),
    }.items():
        want_ok = what == "full coverage passes"
        print(f"  [{'PASS' if ok == want_ok else 'FAIL'}] prefill_row_coverage: {what}"
              + ("" if ok == want_ok else f" — got ok={ok}: {why}"))
        if ok != want_ok:
            bad.append(f"prefill_row_coverage: {what}")
    # A row whose dump is the wrong size is not coverage either, or a truncated corpus would pass the
    # count and then fail every floor for a reason that has nothing to do with the engine.
    with tempfile.TemporaryDirectory(prefix="gate_cov_") as td:
        work = Path(td)
        (work / "logits0.bin").write_bytes(wide.astype("<f4").tobytes()[:-4])
        (work / "logits1.bin").write_bytes(wide.astype("<f4").tobytes())
        short_ok, short_why = prefill_row_coverage(2, work, vocab, PREFILL_LOGITS_ALL,
                                                   PREFILL_LOGITS_ALL)
        print(f"  [{'FAIL' if short_ok else 'PASS'}] prefill_row_coverage: a short dump is not"
              + ("" if not short_ok else f" coverage — it passed: {short_why}"))
        if short_ok:
            bad.append("prefill_row_coverage: a short dump")
    return bad


# ---- (c) state sizes ---------------------------------------------------------------------------
def state_checks(state_line: dict | None) -> None:
    section("(c) state sizes — against engine/state_sizes.py, which is run rather than retyped")
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            from engine import state_sizes as ss
        ok = True
        detail = "imported clean; its own §00.5/spec-page checks passed"
    except SystemExit as e:      # state_sizes.py exits 1 when state.h stops matching §00.5
        ok, detail = False, f"state_sizes.py exited with {e.code}"
    except Exception as e:
        ok, detail = False, f"{type(e).__name__}: {e}"
    check(ok, "engine/state_sizes.py PASS", detail)
    if not ok:
        print(buf.getvalue().rstrip())
        return
    check(ss.gdn_state == 18 * 1024 * 1024, "GDN recurrent state is 18.0 MiB", f"{ss.gdn_state} B")
    check(ss.kv_per_token == 2 * 2 * 256 * 2 * 6,
          "KV is 12288 B/token = 2 caches × 2 kv-heads × 256 × 2 B × 6 layers",
          f"{ss.kv_per_token} B")
    if state_line is None:
        notes.append("(c) ran on the Python side only: --no-engine did not start decode_run.")
        return
    cap = int(state_line["cap"])
    want = {
        "gdn": ss.gdn_state,
        "conv": ss.conv_window,
        "kv": ss.kv_per_token * cap,
        "per_token": ss.kv_per_token,
        "total": ss.gdn_state + ss.conv_window + ss.kv_per_token * cap,
    }
    for k, v in sorted(want.items()):
        check(int(state_line[k]) == v, f"decode_run's STATE {k} matches state_sizes.py",
              f"engine {state_line[k]} vs python {v} (cap {cap})")


# ---- (d) the launch plan -----------------------------------------------------------------------
def launch_plan(n_layers: int, n_gdn: int, n_attn: int) -> dict:
    """One token's launch bill, counted from §07 — independent of the C++'s own arithmetic.

    glue   1 + 2·24. Layer 0's mixer norm has no add in front of it (§03.2's first call), and every
           other add is fused with the norm that follows it: mixer add + post-attention norm, MLP add
           + the next layer's input norm, and after layer 23 the MLP add + the FINAL norm. That last
           pairing is why the final norm is not a launch of its own, and why this is the same 49 that
           `prototype/glue/run.py` pins its token plan against — the number §07's audit in
           `bench/gate-audit.md` demanded.
    proj   GDN: in_proj_qkv, in_proj_z, in_proj_b, in_proj_a (§04.1) + out_proj = 5.
           attention: q_proj, k_proj, v_proj (§05.1) + o_proj = 4.
    gdn    conv (§04.2), gates (§04.3), recurrence (§04.4), gated norm (§04.5) = 4.
    attn   one fused §05.1-§05.7 launch per attention layer (§07.3).
    mlp    gate+up in one launch, down in another (§03.3) = 2 per layer.
    cast   the widenings the F32-input kernels force. Four per GDN layer — the mixer norm row, the
           gated-norm output, the MLP norm row, the MLP act — and four per attention layer (mixer norm
           row, the gated o_proj operand, MLP norm row, MLP act), plus ONE more for the final norm
           before lm_head. The four are exactly the sites whose reader is still a hand kernel that
           takes `const float*`: `proj_gemv` (PACK_X=false) reads the norm row, `mlp_gemv` reads the
           act, `lm_head_gemv` reads the final norm row. It used to be five per GDN layer: a packed
           `g_raw│g_a│g_b` widening fed `conv_kernel`/`gates_kernel`, and D30 1.7' had both kernels
           widen their own BF16 operands, so the widening lost its only reader at every width and on
           both routes and left this row with them. That is the whole of why this row can move while
           §07's own text — which names operations, not dtype adapters — does not.
           Every one of these is a dtype disagreement between graduated kernels, not model
           math — `kernels_cast.cpp` says why the rest of them cannot be dropped today.
           PER TOKEN, and that is what section (d) judges: a one-row body runs the hand GEMV whatever
           `--gemm` asked for (`resolve_wave_gemm`), so all of them arrive for a decode step at any
           setting and this row does not move with the route. A WIDE body on a library route submits
           fewer, because the widenings whose only reader is a hand kernel are guarded on that route;
           the deriver of THAT count is `engine/stage_clock.cpp`'s `wave_plan_bill`, which is the only
           reader of a per-body cast row (the wave report). Two tables, two shapes — do not "fix" one
           to look like the other.
    other  the embedding row read (§02.3), one conv-window roll-back per GDN layer (§08.2), lm_head.
    """
    return {
        "glue": 1 + 2 * n_layers,
        "proj": 5 * n_gdn + 4 * n_attn,
        "gdn": 4 * n_gdn,
        "attn": n_attn,
        "mlp": 2 * n_layers,
        "cast": 4 * n_gdn + 4 * n_attn + 1,
        "other": 1 + n_gdn + 1,
    }


def launch_plan_verdicts(want: dict, got: dict) -> list[tuple[str, bool, str]]:
    """The whole of section (d)'s arithmetic: one verdict per row of the bill, plus one for the total.

    It is a function rather than six inline `check` calls for one reason: `--selftest` has to aim
    injections at the SAME comparison section (d) runs with, or the selftest witnesses a second spelling
    of the rule and the one that ships goes untested (the trap `prototype/attn_tile_model.py` documents —
    two self-consistent spellings, each agreeing with itself). Section (d) prints these verdicts and
    nothing else, so there is exactly one comparison in the file.

    Exact integer equality, in both directions, and the selftest requires both: +1 on `cast` (the
    deleted widening re-added) and −1 on it (a different cast wrongly dropped) must each be caught. A
    bill check that fires one way only is half an equality.
    """
    out: list[tuple[str, bool, str]] = []
    for k in sorted(want):
        out.append((f"launch count {k}", got.get(k) == want[k],
                    f"engine {got.get(k)} vs §07 {want[k]}"))
    total = int(got.get("total", -1))
    out.append(("launch total", total == sum(want.values()), f"{total} vs {sum(want.values())}"))
    return out


def selftest_launch_bill() -> list[str]:
    """Section (d)'s teeth, in both directions, on the CPU — no engine, no device.

    The must-catch set is derived from what this row IS, not hand-listed: the bill is an exact sum of
    per-layer counts, so the two ways it can be wrong by one are a launch re-added and a launch lost,
    and each must be named at its own row. `total` gets its own case because a sum that no longer follows
    from its rows is a different defect from a row that moved — and the third case (a row the engine
    never printed at all) is the one a `dict.get()` would otherwise turn into a silent None.
    """
    bad: list[str] = []
    want = launch_plan(24, 18, 6)          # §03.5: 24 layers, 18 GDN, 6 attention
    clean = dict(want)
    clean["total"] = sum(want.values())

    def failing(got: dict) -> list[str]:
        return [name for name, ok, _ in launch_plan_verdicts(want, got) if not ok]

    def case(name: str, got: dict, must_name: str) -> None:
        hits = failing(got)
        ok = must_name in hits
        print(f"  [{'PASS' if ok else 'FAIL'}] {name} — caught by {hits or 'nothing'}")
        if not ok:
            bad.append(f"launch-bill selftest: {name} was not caught at {must_name!r}")

    agree = not failing(clean)
    print(f"  [{'PASS' if agree else 'FAIL'}] the clean bill reproduces §07 exactly "
          f"(cast={want['cast']}, total={clean['total']})")
    if not agree:
        bad.append("launch-bill selftest: the clean bill does not reproduce §07")
    case("the deleted GDN cast re-added (cast +1)", {**clean, "cast": want["cast"] + 1},
         "launch count cast")
    case("a different cast wrongly dropped (cast −1)", {**clean, "cast": want["cast"] - 1},
         "launch count cast")
    case("a total that no longer follows from its rows",
         {**clean, "total": clean["total"] + 1}, "launch total")
    case("a row the engine never printed", {k: v for k, v in clean.items() if k != "gdn"},
         "launch count gdn")
    return bad


def parse_run(text: str) -> dict:
    out = {"steps": [], "state": {}, "launch": {}, "launchstat": {}, "prefill": {},
           "result": {}, "statefp": []}
    for line in text.splitlines():
        if line.startswith("STEP "):
            out["steps"].append(dict(re.findall(r"(\w+)=([^\s]+)", line)))
        elif line.startswith("STATEFP "):
            # Its own key, and matched BEFORE `STATE ` would matter: these are the §08.6 state rows
            # section (g) judges, one dict per printed line (phase markers included). "STATEFP" does
            # not match "STATE " so the two readers cannot eat each other's lines.
            out["statefp"].append(dict(re.findall(r"(\w+)=([^\s]+)", line)))
        elif line.startswith("LAUNCHSTAT "):
            # Observed launches, counted by the launchers themselves (`engine/launch_stats.h`). A
            # different prefix from `LAUNCH ` on purpose: that line is the PLAN section (d) polices,
            # this one is the OBSERVATION section (h) judges, and a reader must not be able to
            # mistake one for the other. Same reason `STATEFP` is not `STATE`.
            d = dict(re.findall(r"(\w+)=([^\s]+)", line))
            family = d.pop("family", "?")
            out["launchstat"][family] = {k: int(v) for k, v in d.items()}
        elif line.startswith("PREFILL "):
            out["prefill"] = dict(re.findall(r"(\w+)=([^\s]+)", line))
        elif line.startswith("GEMM "):
            # The handle line: which GEMM a wide body ran, echoed by `decode_run` when `--gemm` was
            # named. NOT the route assertion — its `route=` field names the library ("route=hipblas
            # handle=1 workspace_mib=128 …") because it is a statement about the handle that got built,
            # and the hand side spells itself "route=hand". The route the run is judged under comes from
            # the NUMERICS line's `gemm=` field below, which spells `library|hand` the way `--gemm` does.
            out["gemm"] = dict(re.findall(r"(\w+)=([^\s]+)", line))
        elif line.startswith("NUMERICS "):
            # Pin provenance: the wave-row ceiling and the hipBLAS version this binary was built and
            # linked against (D-11). Read by `numerics_provenance_checks` against the recorded file, and
            # its `gemm=` field is ALSO the authoritative route echo (see the route check in `run_all`).
            out["numerics"] = dict(re.findall(r"(\w+)=([^\s]+)", line))
        elif line.startswith("STATE ") or line.startswith("LAUNCH "):
            key = line.split()[0].lower()
            out[key] = dict(re.findall(r"(\w+)=([^\s]+)", line))
        elif line.startswith("RESULT "):
            out["result"] = dict(re.findall(r'(\w+)=(".*?"|[^\s]+)', line))
    return out


def cmd_flag(cmd: list, flag: str) -> str | None:
    """The value a flag carries in an argv this gate is about to run, or None if it is not there.

    Exists so a requirement can be asserted on the argv rather than on the variable that decided it:
    §17's row-shape forcing mutated `args.prefill_logits` after `cmd` had been assembled, so the gate's
    own notes said `all`, the engine received `last`, and 1023 missing rows read as an engine defect.
    """
    i = cmd.index(flag) if flag in cmd else -1
    return cmd[i + 1] if 0 <= i + 1 < len(cmd) else None


# ---- (h) the batched-path proof (P4b) -----------------------------------------------------------
#
# WHY A LAUNCH COUNTER, WHEN THE OUTPUTS WERE ALREADY BIT-IDENTICAL. P4 batched glue and the MLP
# inside the prefill wave, and every fixture came back with its worst per-step logits deficit
# bit-identical (x1.000 on all four deep ones). That is the right numerical answer — a batched kernel
# reads a weight row once per tile and changes no per-element reduction order — and it is also the
# EXACT answer an engine that never reached the batched kernels would print. Identical arithmetic
# submitted one row at a time produces identical bits, so output equality cannot distinguish "P4 is
# free" from "P4 is dead code", and the fixtures could not tell those apart even in principle. Only
# the submissions can, so section (h) counts them:
#
#   * the same prompt runs twice, once at the scratch's full wave width and once at
#     `decode_run --wave-rows 1`, which makes `prefill_wave` split the prompt into waves of one token
#     and therefore makes every glue/MLP/proj/GDN/attention submission take the single-row form;
#   * the batched side must issue row/token-batched launches and the one-row side must not, while
#     BOTH cover exactly the same rows — same work, fewer launches, not less work;
#   * and the two runs must be byte-identical in their dumps, which is what converts the x1.000
#     observation from "suspicious" into "batching costs exactly zero numerics", proven by an A/B.
#
# The negative is built into the shape: the `--wave-rows 1` side IS the dead-code case, run for real.
# `--selftest` injects it as a bug (a multi-row wave whose counters say one row at a time) and requires
# the launch half to FAIL while the identity half stays green — the proof that the identity half alone
# cannot see this, which is the whole reason this section exists. A negative that every check catches
# proves nothing (`prototype/mlp_gemv/results.md`); this one is caught by exactly one of the halves.
# P5, P6 and P8 each added an aimed-at-one-family version of that injection (`--break-wave`,
# `--break-gdn-one-row`, `--break-attn-one-row`) for the same locality reason: if the projection, GDN
# and attention checks all fire together on every injection, no one of them is known to bite on its own.

_HIPBLAS_VERSION: str | None = None


def _hipblas_version_text() -> str:
    """hipBLAS's version as `a.b.c`, from the loaded library (`hipblasGetVersion`, NULL handle).

    Lives behind a lazy ctypes load so `--dry-run` and `--selftest` do not need the HIP toolchain
    present; the stand-in cannot ask the library at all, which is why it echoes `hipblas=dry-run` and
    why `numerics_provenance_checks` refuses to assert anything in dry-run mode.
    """
    global _HIPBLAS_VERSION
    if _HIPBLAS_VERSION is None:
        try:
            import ctypes  # noqa: PLC0415 - only needed when a real engine run is being read

            lib = ctypes.CDLL("libhipblas.so")
            v = ctypes.c_int(0)
            if lib.hipblasGetVersion(None, ctypes.byref(v)) != 0 or v.value <= 0:
                return "unknown"
            _HIPBLAS_VERSION = f"{v.value // 10000}.{v.value // 100 % 100}.{v.value % 100}"
        except OSError:
            return "unavailable"
    return _HIPBLAS_VERSION


def numerics_provenance_checks(run: dict, *, stand_in: bool, write: bool = False) -> None:
    """D-11's provenance half: the run has to say what produced the bits, and the tree has to have
    recorded what the pins were produced under.

    Two numbers, both emitted by `decode_run` on its NUMERICS line and both recorded in
    `engine/numerics_provenance.json` (tracked, because `data/` is gitignored — a file beside the pins
    would be invisible on a fresh clone, which is the exact failure this is here to catch):

    * `wave_rows_cap` — `kMaxWaveRows`. Under the row-count rule the wave width IS the numerics (it
      decides hand GEMV vs hipBLAS), so a corpus is only reproducible against the ceiling it was taken
      in. Cross-checked three ways: what the binary says, what `engine/decode_step.h` says now
      (`wave_row_ceiling()`), and what was recorded.
    * `hipblas_version` — read from the library by `hipblasGetVersion`, so it names the .so that ran.
      Required against the record on the library route, where the version is the numerics. On the hand
      route it is printed and not required: reddening a hand-route pin because the box's hipBLAS moved
      would be a false alarm about a library that was never called.
    """
    if stand_in:
        print("  [info] NUMERICS provenance is not asserted in dry-run mode: the stand-in echoes"
              " `hipblas=dry-run` because it is Python and has no library to ask. The check runs"
              " against a real `decode_run` (section (h) of `--selftest` proves it discriminates).")
        return
    num = run.get("numerics", {})
    cap = num.get("wave_rows_cap")
    ver = num.get("hipblas")
    check(bool(num), "the run printed a NUMERICS line (wave-row ceiling + hipBLAS version)",
          "no NUMERICS line: `decode_run` predates D-11 or stopped printing it, so the bits this gate"
          " is judging carry no record of what produced them" if not num
          else f"{num}")
    if not num:
        return
    if write:
        doc = {"wave_rows_cap": int(cap),
               "hipblas_version": ver,
               "recorded_on": date.today().isoformat(),
               "recorded_by": "engine/gate.py --write-provenance, from a real decode_run NUMERICS"
                              f" line (gemm={num.get('gemm')}, wave_rows={num.get('wave_rows')})",
               "provenance_of": "the pinned fixtures in data/refs/ as they stand. Captured on the hand"
                                " route, so hipblas_version describes the environment the pins live in"
                                " rather than a library any of them was produced by.",
               "why_these_two": "D-11: with two GEMMs in the tree, cross-wave bit-identity is a claim"
                                " about a specific library's solution per (m,n,k), not a law. See"
                                " numerics_provenance_checks in engine/gate.py.",
               "rerecord": "engine/gate.py --fixture <name> --write-provenance"}
        (ROOT / "engine" / "numerics_provenance.json").write_text(
            json.dumps(doc, indent=2) + "\n")
        print(f"  [info] wrote engine/numerics_provenance.json from this run: {doc}")
        return
    path = ROOT / "engine" / "numerics_provenance.json"
    if not path.exists():
        check(False, "the tree records what the pins were produced under",
              f"engine/numerics_provenance.json is missing. Record it from a run you have read:"
              f" `engine/gate.py --fixture <name> --write-provenance` (this run says cap={cap}"
              f" hipblas={ver})")
        return
    rec = json.loads(path.read_text())
    check(str(rec.get("wave_rows_cap")) == str(cap),
          "the run's wave-row ceiling is the one recorded for these pins",
          f"run says kMaxWaveRows={cap}, engine/numerics_provenance.json records"
          f" {rec.get('wave_rows_cap')} (recorded {rec.get('recorded_on','?')}) — the wave width is"
          " the numerics under the row-count rule, so a corpus taken under a different ceiling is a"
          " different corpus; re-record only after re-reading why it moved")
    check(str(rec.get("wave_rows_cap")) == str(wave_row_ceiling()),
          "the recorded ceiling is the one the source declares today",
          f"engine/numerics_provenance.json says {rec.get('wave_rows_cap')},"
          f" engine/decode_step.h says {wave_row_ceiling()} — if the ceiling really moved, every pin"
          " has to be re-taken under it before the record may")
    if route_is_library():
        check(str(rec.get("hipblas_version")) == str(ver),
              "the run's hipBLAS is the version recorded for these pins",
              f"hipblasGetVersion says {ver}, the record says {rec.get('hipblas_version')} —"
              " hipBLAS picks a different solution per (m,n,k) (A-32/§A1 of progress/2026-10-14/route-c-audit-2026-10-14.md), so a silent bump is a"
              " numerics change: re-take the pins, then re-record")
    else:
        print(f"  [info] hipBLAS {ver} printed, not gated on the hand route: nothing in this run"
              " called it (rows<=1 is the hand GEMV whatever the flag says).")


def gemm_pin_checks(run: dict, *, stand_in: bool) -> None:
    """Which algo pins the run SAYS it ran, asserted against the spec this gate asked for.

    The route echo's forcing, one knob down, and the reason it exists is the same: section (h)'s band,
    the `GEMM-LT source=` lines and every pin-facing claim in this tree are properties of the algorithm
    that served each shape, and `--gemm-pins` made that set a runtime choice. A log that cannot say which
    pin set ran therefore adjudicates neither — which is exactly how `wide520f --gemm lt`'s one red row
    ended up unattributable (D28: `library` did not flip it, and nothing in the log said whether the
    pinned `down_proj` kernel or the heuristic's rank-0 did).

    The engine prints `pins=<in force>/<in the table>`, both counted out of `algo_pins()` in the process
    that ran. Three equalities follow, and there are no other cases: `all` means allowed == total, `none`
    means allowed == 0, and a csv means allowed == its own length. A missing field is an error rather
    than a skip for `--attn-touch`'s reason — a binary that predates the knob cannot be judged as though
    it honoured it, and silence would let a pre-`--gemm-pins` build pass as a post- one.
    """
    if GEMM_ROUTE != "lt":
        # Nothing to assert. hipBLAS and the hand GEMV have no algo-pin concept, and `set_pins` has
        # already refused a spec that named one, so an absent `pins=` field on this route is correct
        # bookkeeping rather than a gap in it.
        return
    field = run.get("gemm", {}).get("pins")
    if field is None:
        check(False, "the run said which algo pins it ran (`--gemm-pins`)",
              "its GEMM line carried no `pins=` — the binary predates `--gemm-pins`, or the flag never"
              " reached it, so nothing in this log says whether a pinned kernel or the heuristic's"
              " rank-0 served these shapes and no claim about either can be made from these bits"
              + (" (dry-run: the stand-in does echo the field, and `--selftest`'s mismatch case is what"
                 " proves this check discriminates)" if stand_in else ""))
        return
    allowed, sep, total = str(field).partition("/")
    if not (allowed.isdigit() and sep and total.isdigit()):
        check(False, "the run's `pins=` field is readable",
              f"`pins={field}` does not read `<in force>/<in the table>` — refusing to guess which half"
              " is which, because the two readings give opposite answers about whether a pin ran")
        return
    got, table = int(allowed), int(total)
    want, why = pin_expectation(GEMM_PINS, table)
    verdict = ("and it did" if got == want else
               f"but {got} did — this log is a judgment about"
               + (" rank-0 everywhere" if got == 0 else " a different pin subset")
               + ", not about the config this gate was asked to judge")
    check(got == want,
          "the run ran the algo-pin set this gate is judging it under"
          f" (--gemm-pins {GEMM_PINS})",
          f"GEMM line echoed pins={got}/{table} — {why}, so {want} in force was owed, " + verdict
          + (" (stand-in echo: the table width is this file's arithmetic, not the engine's, so what"
             " is exercised here is the comparison — a real run counts its own table)" if stand_in else ""))

    # ...and it must say WHICH ones by name. The counts above can be equal while the two sides ran
    # different subsets (four names in the spec, four rows in force, one of them the wrong one), and a
    # candidate A/B is exactly the case that bites: `--gemm-pins deep_down_proj` and
    # `--gemm-pins down_proj` are both `pins=1/1` and are opposite claims about which kernel served the
    # shape. So the echoed `pin_names=` is compared as a set against the spec whenever the spec is a
    # list; for `all`/`none` the engine's own count already settles it and this only records the names.
    # A missing field is refused rather than skipped: a binary that predates candidates cannot be
    # judged to have honoured them (`pins=`'s reason, one knob down).
    echo_names = run.get("gemm", {}).get("pin_names")
    if echo_names is None:
        check(False, "the run named the algo pins it ran (`pin_names=`)",
              "its GEMM line carried `pins=` but no `pin_names=`, so this log can count in-force pins"
              " but cannot say WHICH ones — and two pin sets of equal size are indistinguishable in"
              " it, which is every candidate A/B this tree can run"
              + (" (dry-run: the stand-in does echo the field, and `--selftest`'s mismatch case is what"
                 " proves this check discriminates)" if stand_in else ""))
    else:
        listed = [n for n in str(echo_names).split(",") if n and n != "none"]
        spec_names = parse_pins(GEMM_PINS)
        if not spec_names:
            # `none` owes an empty list, so any name at all is a pin that ran while the gate believes
            # rank-0 ran everywhere — the same lie `pins=0/1` catches from the count side.
            check(not listed, "`--gemm-pins none` really ran no pin",
                  f"GEMM line echoed pin_names={echo_names} while the spec owes none"
                  + (" (stand-in echo)" if stand_in else ""))
        elif spec_names != ["*"]:
            # A csv: set equality against what was asked. This is the case a candidate A/B lives in, and
            # the one the count cannot decide - `down_proj` and `deep_down_proj` are both exactly one pin.
            ran, asked = set(listed), set(spec_names)
            check(ran == asked,
                  f"the run named exactly the pins this gate asked for (--gemm-pins {GEMM_PINS})",
                  f"GEMM line echoed pin_names={echo_names}"
                  + (" — it did" if ran == asked else
                     f"; the difference is {sorted(ran ^ asked)} ({sorted(ran - asked)} ran in place of"
                     f" {sorted(asked - ran)}), so this log is a verdict about some other subset")
                  + (" (stand-in echo)" if stand_in else ""))
        else:
            # `all`: this file holds no shipped-name list (the engine owns `algo_pins()`, and copying it
            # here is the second-authority drift D28 records), so what IS decidable is that the list the
            # run printed has as many entries as the run claims are in force. A one-name run that lists
            # two, or a two-pin run that lists one, is a line describing two different configs.
            check(len(listed) == got,
                  f"the run's pin name list agrees with its own count ({got} in force)",
                  f"GEMM line echoed pins={got}/{table} with pin_names={echo_names} — "
                  + ("consistent" if len(listed) == got else
                     f"{len(listed)} names for {got} pins in force, which is one line about two"
                     " different sets")
                  + (" (stand-in echo)" if stand_in else ""))

    # `cand=<in force>/<candidates in the table>` is the third half of the same line, and it is the one
    # that guards the promotion itself: `all` means SHIPPED, and the moment a candidate could bind under
    # it, every log this file ever produced would be ambiguous about which kernel served the shape. Under
    # `all` or `none` the owed answer is zero candidates in force, and that is decidable here without
    # knowing which rows are candidates — unlike a csv, where the gate cannot tell a candidate name from a
    # shipped one (the engine owns that, and `deep_pins.py`'s CHECK-TABLE is what adjudicates it against
    # `insitu_verdicts.tsv`), so a csv only owes `in force <= names given`.
    cand_field = run.get("gemm", {}).get("cand")
    if cand_field is None:
        check(False, "the run said whether any candidate pin was in force (`cand=`)",
              "its GEMM line carried `pins=` but no `cand=`, so this log cannot show that `all` meant"
              " the shipped set and nothing else" + (" (dry-run: the stand-in does echo the field, and"
                                                    " `--selftest`'s mismatch case is what proves this"
                                                    " check discriminates)" if stand_in else ""))
    else:
        cin, sep2, ctot = str(cand_field).partition("/")
        if not (cin.isdigit() and sep2 and ctot.isdigit()):
            check(False, "the run's `cand=` field is readable", f"`cand={cand_field}` does not read"
                  " `<in force>/<candidates in the table>` — the same refusal as `pins=`: guessing "
                  " which half is which is how a candidate gets read as shipped")
        else:
            cin, ctot = int(cin), int(ctot)
            if parse_pins(GEMM_PINS) in (["*"], []):
                check(cin == 0,
                      f"`--gemm-pins {GEMM_PINS}` bound no candidate pin"
                      " (candidates bind only when named, so `all` stays a synonym for shipped)",
                      f"GEMM line echoed cand={cand_field}" + (" — it did" if cin == 0 else
                       f": {cin} candidate(s) ran while the spec names none, which means `all` "
                       "has stopped meaning `the shipped set` and every band and verdict taken under"
                       " it is now a claim about an unknown kernel")
                      + (" (stand-in echo)" if stand_in else ""))
            else:
                check(cin <= len(parse_pins(GEMM_PINS)),
                      f"the run's candidate count is inside its own spec ({len(parse_pins(GEMM_PINS))}"
                      f" name(s) given)",
                      f"GEMM line echoed cand={cand_field} against --gemm-pins {GEMM_PINS}, which is"
                      " more candidates in force than names in the spec — one line, two configs"
                      + (" (stand-in echo)" if stand_in else ""))

    # The pool the run queried, asserted the same way. `--gemm-heur` is a runtime knob, so a log that does
    # not say its width cannot be read as "the shipped pool ran"; and this gate holds no width flag, so
    # the only admissible answer is the shipped default. The reason that matters is not the common case
    # (rank-0 is the same algorithm at width 8 and 64 on all seven shapes, so a wider pool is inert while
    # the top record is admissible) but the case where it is NOT: the engine takes the first ADMISSIBLE
    # record, so at an odd body width whose rank-0 turns out inadmissible, pool depth decides which
    # kernel serves the shape — and two depths would then be two different runs.
    width_field = run.get("gemm", {}).get("heur_width")
    want_width = heuristic_width_default()
    if width_field is None:
        check(False, "the run said how wide a heuristic pool it queried (`heur_width=`)",
              "its GEMM line carried no `heur_width=`, so this log cannot show the shipped pool was the"
              " one queried" + (" (dry-run: the stand-in does echo the field, and `--selftest`'s"
                                " mismatch case is what proves this check discriminates)"
                                if stand_in else ""))
    else:
        check(str(width_field) == str(want_width),
              f"the run queried the shipped heuristic pool (kHeuristicWidth={want_width})",
              f"GEMM line echoed heur_width={width_field}"
              + (" — it did" if str(width_field) == str(want_width) else
                 f", not {want_width}. This gate holds no --gemm-heur flag because a deeper pool has"
                 " nothing to adjudicate here yet, so a run at another width is a different config and"
                 " not a verdict this file can endorse")
              + (" (stand-in echo)" if stand_in else ""))


def heuristic_width_default() -> int:
    """`kHeuristicWidth` — how many candidates the lt heuristic query is asked for — read out of
    `engine/kernels_gemm.cpp`, never retyped here, for the same reason `wave_row_ceiling()` reads its
    constant from the code that obeys it.

    What it is for: the pool width is a property of the RUN, not of the build, and the only way this gate
    can tell "judged at the shipped pool" from "judged at some pool nobody priced" is to compare the run's
    own `heur_width=` echo against the shipped default. This file deliberately has no `--gemm-heur` flag:
    a deeper pool cannot move a pinned shape (a pin bypasses the query) and did not move an unpinned one
    at M=512 (rank-0 is the same algorithm at width 8 and at 64 for all seven shapes, and `RESULT
    default_seq=0` says rank-0 is what got taken), so there is nothing here for such a flag to
    adjudicate — and a run that arrived with one should read RED rather than be quietly judged as though
    it had shipped settings.
    """
    text = (ROOT / "engine" / "kernels_gemm.cpp").read_text()
    m = re.search(r"kHeuristicWidth\s*=\s*(\d+)\s*;", text)
    if not m:
        raise SystemExit("gate: cannot find kHeuristicWidth in engine/kernels_gemm.cpp — the pool width"
                         " this gate asserts runs used has to be read from the code that uses it, not"
                         " remembered")
    return int(m.group(1))


def shipped_pin_count() -> int:
    """How many rows `algo_pins()` marks `PinSet::kShipped`, counted out of the C++ source.

    Read, never remembered, for the standing reason (`heuristic_width_default()` below, `wave_row_ceiling()`
    above): a literal here would be a second authority over a table the engine owns, and this file has
    already been caught once carrying a band whose pin set the binary no longer had
    (`prototype/gemm_algo_sweep/results.md` §3d, D28). What it is for: the dry-run stand-in has no engine
    to ask, so it needs a shipped count to echo `pins=<i>/<n>` with, and the selftest's clean case must
    expect the same number the stand-in prints or the selftest is checking a spelling nobody emits. When D35
    promoted the set from one row to two, and when D36 swapped one of those rows back, every literal of
    \"one\" or \"two\" in this file would have had to be found by hand; counted this way, the stand-in and its
    clean case both moved on their own.
    """
    text = (ROOT / "engine" / "kernels_gemm.cpp").read_text()
    n = len(re.findall(r'"[0-9a-f]{48}",\s*PinSet::kShipped', text))
    if n == 0:
        raise SystemExit("gate: no `PinSet::kShipped` row in engine/kernels_gemm.cpp — either the shipped"
                         " set is genuinely empty (then `--gemm-pins all` means nothing and this gate"
                         " should say so loudly) or the row shape changed and the count must be read"
                         " from the real spelling, not guessed")
    return n


def standin_pin_counts(names: list[str]) -> tuple[int, int]:
    """(in force, denominator) for the dry-run stand-in's `pins=` echo, from the shipped count.

    One helper, because the stand-in and the selftest's clean case must not compute this twice and drift.
    The stand-in still has no table of its own: for a csv it echoes \"every name you gave me is in force\",
    which is stand-in arithmetic by design and is labelled as such where the check reads it.
    """
    if not names:
        return 0, shipped_pin_count()
    if names == ["*"]:
        n = shipped_pin_count()
        return n, n
    return len(names), shipped_pin_count()


def wave_row_ceiling() -> int:
    """`kMaxWaveRows` — how many rows the wave scratch is tall — read from `engine/decode_step.h`,
    never retyped here. Same reason `lane_geometry()` reads its two constants out of the code that
    enforces them: the number belongs to the code that obeys it, and a copy here would be a second
    authority that can drift. `decode_run --wave-rows` accepts at most this, so it is also the
    ceiling of the A/B."""
    path = ROOT / "engine" / "decode_step.h"
    m = re.search(r"constexpr int kMaxWaveRows\s*=\s*(\d+)", path.read_text())
    if not m:
        raise SystemExit("gate: cannot read kMaxWaveRows from engine/decode_step.h — the wave width"
                         " is the scratch's to own, so this gate refuses to guess it")
    return int(m.group(1))


def wave_row_sizes(prompt_len: int, wave_rows: int) -> list:
    """How `prefill_wave` splits `prompt_len` tokens over a scratch `wave_rows` rows tall (§2.1's
    split, `engine/prefill.cpp`). One wave of the whole prompt is the case every shipped fixture is
    in; the loop is here so a longer prompt is still counted rather than quietly assumed away."""
    sizes, left = [], int(prompt_len)
    while left > 0:
        n = min(int(wave_rows), left)
        sizes.append(n)
        left -= n
    return sizes


def wave_expect(counts: dict, prompt_len: int, decode_steps: int, wave_rows: int) -> dict:
    """The counters a run of this shape MUST print, family by family, derived rather than remembered.

    `counts` is the per-body launch bill from `launch_plan()` (§07's structure, section (d)'s source),
    so this is the same arithmetic §07 implies — which is the point: the engine's observed counters are
    compared against what §07 says a wave of that width owes. A refactor that loops the wave body per
    token changes `calls` and `rows_max` and leaves every output bit alone, and this is the table that
    notices.

    "Per body" means one `submit_wave_body` call, and for all three batchable families that number is
    the per-TOKEN bill from §07 whether the body was one token or 295: glue issues one launch per fused
    add+norm, the MLP one per half, and `proj` one per projection (§06.2's "one GEMM per projection"),
    each covering every row of the body — so 49/24/24/114 launches arrive per body at any width (49
    glue, 24+24 MLP halves, 114 proj: five projections for each of 18 GDN layers, four for each of 6
    attention layers), and what the width changes is `rows` on each of them. That is why the same table
    serves both sides. `cast` is NOT in this table and must not be added to it: it has no launcher
    counter (`engine/launch_stats.h`), and a wide body's cast count is route-dependent —
    `engine/stage_clock.cpp`'s `wave_plan_bill` prices it against the wave report, per body and per
    route, which is where a per-body cast row is read.

    A generation step is a body of one token (`submit_token_body` is `submit_wave_body` with n = 1),
    which is why the decode steps join the wave sizes below and why `rows_total` comes out
    independent of `wave_rows`: batching moves work between launches, it does not add or remove any.
    """
    sizes = wave_row_sizes(prompt_len, wave_rows) + [1] * int(decode_steps)
    out = {}
    for family, per_body in counts.items():
        out[family] = {
            "calls": per_body * len(sizes),
            "calls_batched": per_body * sum(1 for s in sizes if s > 1),
            "rows_total": per_body * sum(sizes),
            "rows_max": max(sizes) if sizes else 0,
        }
    return out


def wave_launchstat_lines(counts: dict, prompt_len: int, decode_steps: int, wave_rows: int,
                          family_rows: dict | None = None) -> list:
    """`wave_expect()` formatted as the `LAUNCHSTAT` lines the engine prints. Stand-in only — a real
    run's counters come from the launchers — but produced from the same function the check compares
    against, so the dry-run plumbing cannot drift from the device one.

    `family_rows` names families whose observed width differs from the side's wave width, which is how
    `--break-wave` and `--break-gdn-one-row` write the counters of a body that silently ran one row at
    a time while still asking for (and echoing) a wide wave."""
    out = []
    for fam, c in wave_expect(counts, prompt_len, decode_steps, wave_rows).items():
        if family_rows and fam in family_rows:
            c = wave_expect({fam: counts[fam]}, prompt_len, decode_steps,
                            family_rows[fam])[fam]
        out.append(f"LAUNCHSTAT family={fam} calls={c['calls']} calls_batched={c['calls_batched']}"
                   f" rows_total={c['rows_total']} rows_max={c['rows_max']}")
    return out


def ab_room_spec(spec: str) -> tuple[int, float, float] | None:
    """Parse `--break-ab-room POS[:ULP[:MAG]]` → `(position, room in BF16 ULP, top-1 magnitude)`.

    Empty means no injection. `MAG` defaults to 63.0 because that is the stand-in corpus's own row
    magnitude, and `ULP` to 2.0 — just above the line, which is the only margin at which a must-catch
    is a measurement of the line rather than of the bug's size. The two knobs together are the point:
    the same room in ULP is a different absolute gap at every magnitude, and that is exactly the axis
    the line was moved onto (§3f of prototype/gemm_algo_sweep/results.md).
    """
    if not spec.strip():
        return None
    parts = spec.split(":")
    if not 1 <= len(parts) <= 3:
        raise SystemExit(f"--break-ab-room takes POS[:ULP[:MAG]], got {spec!r}")
    try:
        pos = int(parts[0])
        ulp = float(parts[1]) if len(parts) > 1 else 2.0
        mag = float(parts[2]) if len(parts) > 2 else 63.0
    except ValueError:
        raise SystemExit(f"--break-ab-room takes POS[:ULP[:MAG]], got {spec!r}")
    if pos < 0 or ulp <= 0.0 or mag <= 0.0:
        raise SystemExit(f"--break-ab-room needs a position at or after 0 and positive ULP/MAG,"
                         f" got {spec!r}")
    return pos, ulp, mag


def ab_room_row(wide_side: bool, ulp: float, mag: float) -> tuple[np.ndarray, int]:
    """Row `pos` for ONE side of the stand-in, built so the two sides disagree at an EXACT room.

    Same shape on both sides — a floor at `-mag` across all 64 lanes with the top two ids at `mag` and
    `mag - gap`, the winner swapped — so each side's own top-1 is `mag` and its own gap is exactly
    `gap = ulp x bf16_ulp(mag)`. `bf16_ulp` returns powers of two and `ulp` is a small binary fraction
    in every case the selftest runs, so `mag - gap` is exact in F32 and the room the flip rule prices
    is the requested number of ULP, not an approximation of it.

    Two details that are not arbitrary. The flat floor matters: leaving the arange's other 62 lanes in
    place would bury a one-ULP tie at the top of a row whose energy sits far above it, so the
    injection would be judged on arithmetic it never intended. And only the two top lanes differ
    between the sides, so the row band and the pooled stage figures cannot see this injection at all —
    the flip half is the only half that can go red, which is what makes it aimed rather than
    indiscriminate (`prototype/mlp_gemv/results.md`'s locality rule).
    """
    win, lose = (63, 62) if wide_side else (62, 63)
    gap = ulp * bf16_ulp(mag)
    row = np.full(64, -mag, dtype=np.float32)
    row[win], row[lose] = mag, np.float32(mag - gap)
    return row, win


def batched_ab_stand_in(man: dict, prompt_ids: list, dec_logits: list, counts: dict,
                        work: Path, wave_rows: int, break_wave: bool = False,
                        break_ab_logits: int = -1, break_ab_ids: int = -1, carry_rows: int = 0,
                        break_gdn_one_row: bool = False,
                        break_attn_one_row: bool = False,
                        break_ab_room: str = "") -> dict:
    """The sides of section (h), on the CPU, for `--dry-run` / `--selftest`.

    Like section (d) in dry-run mode this is Python comparing with itself — said out loud in the note
    the caller appends — and its job is to prove the CHECKER discriminates, not the engine. The logits
    files are a shape (small, deterministic, same names and count a real run writes), not engine
    numbers; `--break-ab-logits` differs one file on the one-row side to prove the byte comparison
    bites, and `--break-wave` writes one-row counters for a multi-row wave to prove the launch
    comparison bites — each aimed at one half, so neither injection can satisfy itself.

    `carry_rows` > 0 adds P6's third side: the same prompt run in waves that small, so the prompt end
    falls mid-chunk and the conv window has to be carried across a wave boundary. Its logits are
    identical to the wide side's by construction here, which is why the injection that matters for it
    is `--break-gdn-one-row` (the wide GDN launches never happened), not a logits perturbation.

    `break_attn_one_row` aims the same bug at P8's site alone: the attention layer of a wide wave
    submitting one query row per launch. Same reasoning as the GDN injection — a wave that quietly ran
    one row at a time produces identical bits, so only the counters can see it.

    `break_ab_room` is the flip line's own negative: POS's row becomes a controlled near-tie on which
    the two sides pick different ids, at `ULP` BF16 ULP of room and a top-1 of `MAG` (see
    `ab_room_spec`/`ab_room_row`). Unlike the two injections above it, this one changes BOTH sides —
    the bug it models is two legal widths disagreeing, and a perturbation on one side alone could not
    state a room for the pair.
    """
    greedy = man["greedy_continuation"]
    room_spec = ab_room_spec(break_ab_room)

    def side(name: str, requested_rows: int, observed_rows: int, perturbs: bool,
             only_one_row: dict | None = None) -> dict:
        out_dir = work / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("logits*.bin"):
            stale.unlink()

        def perturb(pos: int) -> None:
            """One row, one file, on ONE side only (the caller decides which): the output comparison half
            must name this file and no other. Perturbing both sides would change nothing, which is the
            same mistake as a negative that is true on both runs. Aimed in the same index vocabulary as
            the STEP lines (`pos`).

            The perturbation is the whole row moved 8 logits, not a flipped bit, and that is deliberate:
            a 1-ULP bit flip is what the byte comparison used to catch, and on the library route the band is
            2e-03 of 1-cos, so a bit flip would be INSIDE the band and the negative would go extinct the
            moment `--gemm library` was gated. A constant offset breaks byte-identity (so the hand route's
            check still bites), puts that row's `1-cos` near 5e-03 and the corpus's pooled `1-cos` near
            1e-03, outside both cascade halves (so the library route's row and cosine checks bite), and
            leaves the row's ORDER untouched — the argmax and every sampled id are unchanged, so the id
            halves stay green and each injection still fails exactly one half of the section. It is one row
            out of the corpus, so the POOLED `meanrel` line deliberately does not bite here (one row of 25
            is ~1e-02 against a 3e-02 pooled band); that is the difference between a row rule and a stage
            rule, and the selftest case says so rather than pretending.
            """
            if not perturbs or pos != break_ab_logits:
                return
            path = out_dir / f"logits{pos}.bin"
            row = np.fromfile(path, dtype="<f4") + np.float32(8.0)
            row.astype("<f4").tofile(path)

        def perturb_id(pos: int) -> int:
            """The other half: one row's SAMPLED ID changed on one side, its logits untouched. Under
            `--gemm hand` this is the ordinary wrong-id failure; under `--gemm library` it is the flip
            rule being asked whether an id difference at a row with real room is admissible (it is not —
            the stand-in's rows are `arange`, so every gap there is 1.0 and no row of this corpus is a
            coin toss). Paired with `--break-ab-logits`, which moves the bits and must NOT move an id.
            """
            if not perturbs or pos != break_ab_ids:
                return -1
            return 999900001 + pos      # an id no row of this corpus argmaxes to; unique per position

        def room_case(pos: int):
            """`(row, sampled_id)` when POS's row is the controlled near-tie, else None.

            Each side writes its own winner into `sampled`, which is what makes the two sides disagree
            the way two legal widths disagree — an id choice that follows from that side's own bits.
            """
            if room_spec is None or pos != room_spec[0]:
                return None
            return ab_room_row(not perturbs, room_spec[1], room_spec[2])

        lines = [f"PREFILL tokens={len(prompt_ids)} logits={PREFILL_LOGITS_ALL}"
                 f" prefix_len={len(prompt_ids)} wave_rows={requested_rows}",
                 # Every row priced: the A/B compares every row of the wave, and under `last` there
                 # would be nothing but the last row's logits to compare.
                 ]
        for i, tok in enumerate(prompt_ids):
            sampled = greedy[0] if i == len(prompt_ids) - 1 else prompt_ids[i + 1]
            sampled = perturb_id(i) if perturb_id(i) >= 0 else sampled
            rc = room_case(i)
            if rc is not None:
                sampled = rc[1]
            lines.append(f"STEP kind={PREFILL_KIND} index={i} input={tok} sampled={sampled}"
                         f" top_logit=1 pos_before={i} pos_after={i + 1} stop=0")
            (out_dir / f"logits{i}.bin").write_bytes(
                (rc[0] if rc is not None else np.arange(64, dtype=np.float32)).tobytes())
            perturb(i)
        for j in range(len(dec_logits)):
            pos = len(prompt_ids) + j
            alt = perturb_id(pos)
            sampled = alt if alt >= 0 else greedy[j + 1]
            rc = room_case(pos)
            if rc is not None:
                sampled = rc[1]
            lines.append(f"STEP kind={DECODE_KIND} index={pos} input={greedy[j]}"
                         f" sampled={sampled} top_logit=1 pos_before={pos}"
                         f" pos_after={pos + 1} stop=0")
            (out_dir / f"logits{pos}.bin").write_bytes(
                (rc[0] if rc is not None else np.arange(64, dtype=np.float32)).tobytes())
            perturb(pos)
        lines += wave_launchstat_lines(counts, len(prompt_ids), len(dec_logits), observed_rows,
                                       only_one_row)
        lines.append(f"RESULT mode=run steps={len(prompt_ids) + len(dec_logits)} ms_total=0"
                     f" ms_per_step=0 last_sampled={greedy[-1]}"
                     f" prefix_len={len(prompt_ids) + len(dec_logits)} stopped=0")
        return {"requested_wave_rows": requested_rows, "dir": out_dir,
                "run": parse_run("\n".join(lines))}

    # `break_wave` is the refactor: the driver still asked for (and echoes) a wide wave, but the body
    # submitted one row at a time, so only the counters change. That is exactly the state of an engine
    # whose batched kernels are never reached. `break_gdn_one_row` is the same bug aimed at ONE family
    # — the three GDN sites — so the GDN checks are seen to bite on their own instead of hiding behind
    # glue's, which is the locality rule `prototype/mlp_gemv/results.md` insists on.
    gdn_one_row = {f: 1 for f in counts if f.startswith("gdn_")} if break_gdn_one_row else None
    if break_attn_one_row:
        # Merged rather than replacing: `--selftest` runs each injection alone, but nothing should make
        # the two mutually exclusive by accident.
        attn_one_row = {"attn": 1}
        gdn_one_row = {**(gdn_one_row or {}), **attn_one_row}
    sides = {"batched": side("ab_batched", wave_rows, 1 if break_wave else wave_rows,
                              perturbs=False,
                              only_one_row=None if break_wave else gdn_one_row),
             "perrow": side("ab_perrow", 1, 1, perturbs=True)}
    if carry_rows > 1:
        sides["carry"] = side("ab_carry", carry_rows, carry_rows, perturbs=False)
    return sides


def compare_dump_dirs(a: Path, b: Path) -> tuple:
    """Compare every `logits*.bin` in two run directories, byte for byte (the `cmp` of P2's
    byte-identity proof, done in the gate so it is permanent). Returns (files, bytes, differing)."""
    names_a = sorted(p.name for p in a.glob("logits*.bin"))
    names_b = sorted(p.name for p in b.glob("logits*.bin"))
    total = 0
    differing = [] if names_a == names_b else ["the file sets differ"]
    for name in names_a:
        pa, pb = a / name, b / name
        if not pb.exists():
            differing.append(name + " (missing on the other side)")
            continue
        ba, bb = pa.read_bytes(), pb.read_bytes()
        total += len(ba)
        if ba != bb or len(ba) != len(bb):
            differing.append(name)
    return len(names_a), total, differing


def corpus_drift(a_dir: Path, b_dir: Path) -> dict | None:
    """Row-by-row and pooled drift between two runs' logits corpora, F32 dumps against F32 dumps.

    Exists because section (h) on the library route compares two runs rather than a run and a pin, so
    it has no allowance file and no §10.1 floor to borrow: the only honest question is "is the drift
    between these two widths inside the band the audit measured for two GEMMs". Rows are streamed one at
    a time (a 524-row corpus is ~0.5 GB in F32 and ~1 GB per side in F64; holding both is a page-fault
    storm on this box, and `devices/` says first touch of fresh pages is 7x slow).

    Returns None when there is nothing to compare (no common `logitsN.bin`).
    """
    def indices(d: Path) -> dict:
        out = {}
        for p in d.glob("logits*.bin"):
            try:
                out[int(p.name[len("logits"):-len(".bin")])] = p
            except ValueError:
                continue
        return out

    ia, ib = indices(a_dir), indices(b_dir)
    common = sorted(set(ia) & set(ib))
    if not common:
        return None
    sad = saa = sab = naa = nbb = 0.0
    worst_1cos, worst_1cos_row = 0.0, None
    worst_row_rel, worst_rel_row = 0.0, None
    identical = 0
    for i in common:
        a = np.fromfile(ia[i], dtype="<f4").astype(np.float64)
        b = np.fromfile(ib[i], dtype="<f4").astype(np.float64)
        if a.shape != b.shape:
            return {"mismatch": f"logits{i}.bin is {a.shape} vs {b.shape}"}
        d = b - a
        sad += float(np.sum(np.abs(d)))
        saa += float(np.sum(np.abs(a)))
        sab += float(np.dot(b, a))
        naa += float(np.dot(a, a))
        nbb += float(np.dot(b, b))
        one = 1.0 - float(np.dot(b, a) / (np.linalg.norm(a) * np.linalg.norm(b)))
        if one >= worst_1cos:
            worst_1cos, worst_1cos_row = one, i
        rel = float(np.mean(np.abs(d)) / np.mean(np.abs(a))) if np.mean(np.abs(a)) else 0.0
        if rel > worst_row_rel:
            worst_row_rel, worst_rel_row = rel, i
        identical += int(ia[i].read_bytes() == ib[i].read_bytes())
    return {
        "rows": len(common), "only_a": len(ia) - len(common), "only_b": len(ib) - len(common),
        "identical": identical,
        "meanrel": (sad / saa) if saa else 0.0,
        "one_cos": (1.0 - sab / np.sqrt(naa * nbb)) if (naa and nbb) else 0.0,
        "worst_1cos": worst_1cos, "worst_1cos_row": worst_1cos_row,
        "worst_row_rel": worst_row_rel, "worst_rel_row": worst_rel_row,
    }


def wave_band_checks(a_dir: Path, b_dir: Path, a_label: str, b_label: str, what: str) -> None:
    """The library route's replacement for bit-identity between two widths (D-11).

    Two GEMMs of the same weights do not produce the same bits at different widths, and demanding they
    did was demanding the impossible: §A1 of progress/2026-10-14/route-c-audit-2026-10-14.md found hipBLAS' own solution differs at every N, so `M=1`, `M=5`
    and `M=512` are three different summation orders and §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md measured what that costs (worst per-row
    `1-cos` 1.9e-03, worst per-row `meanrel` 6.1e-02, pooled `meanrel` <= 1.56e-02 over 1694 rows).
    The band is that measurement, adopted as policy: per-row `1-cos <= 2e-03`, pooled `meanrel <= 3e-02`,
    pooled `cos >= 0.9998`. The POOLED two are route-independent — they price one cascade of stage work
    against another and the library on the wide side does not change what a stage tile is — but the
    per-row one is not: it is a property of the pairing, and the pairing is hand-vs-whichever-library.
    So the row line comes from `wave_row_1cos_band()` and prints its own provenance, and the two library
    routes are judged by two measured numbers rather than by one number inherited across a change of
    independent variable.

    What is LOST to bit-identity has to be said, because it is the reason this section was written the
    other way: a wrong-but-self-consistent conv-window carry can no longer be caught by a byte
    comparison, so the window is now proven only to within the band. It is still a real test (a window
    read from the wrong rows moves logits by orders more than 2e-03 of 1-cos), but it is a weaker one,
    and on a `--gemm hand` run the old byte-exact check is what runs.
    """
    drift = corpus_drift(a_dir, b_dir)
    if drift is None:
        check(False, f"{a_label} and {b_label} have logits to compare",
              "no `logitsN.bin` in common — one side wrote nothing, so there is nothing to band")
        return
    if "mismatch" in drift:
        check(False, f"{a_label} and {b_label} wrote the same-shaped corpus",
              drift["mismatch"])
        return
    row_band, row_source = wave_row_1cos_band()
    ok_1cos = drift["worst_1cos"] <= row_band
    ok_rel = drift["meanrel"] <= CASCADE_BAND["stage_meanrel_max"]
    ok_cos = drift["one_cos"] <= 1.0 - CASCADE_BAND["stage_cos_min"]
    line = (f"{drift['rows']} rows, {drift['identical']} of them still bit-identical;"
            f" pooled meanrel {drift['meanrel']:.3e} (band {CASCADE_BAND['stage_meanrel_max']:.0e})"
            f" [route-independent: a stage tile is the same tile whichever library made it],"
            f" pooled 1-cos {drift['one_cos']:.3e}"
            f" (band {1.0 - CASCADE_BAND['stage_cos_min']:.0e}); worst per-row 1-cos"
            f" {drift['worst_1cos']:.3e} at logits{drift['worst_1cos_row']}.bin"
            f" (band {row_band:.2e} — {row_source}); worst per-row meanrel"
            f" {drift['worst_row_rel']:.3e} at logits{drift['worst_rel_row']}.bin, which is printed"
            " and NOT gated — a logits row's own meanrel reaches 6.1e-02 across the audit's corpus"
            " because one small-magnitude row can carry it, and the adopted band puts meanrel at the"
            " stage level, where it is pooled and bounded")
    check(ok_1cos, f"{a_label} and {b_label} keep every logits row inside the cascade band"
                   f" (per-row 1-cos <= {row_band:.2e} for the `--gemm {GEMM_ROUTE}` pairing)"
                   f" — {what}", line)
    check(ok_rel, f"{a_label} and {b_label} pool to a stage meanrel inside the cascade band"
                  f" (<= {CASCADE_BAND['stage_meanrel_max']:.0%})", line)
    check(ok_cos, f"{a_label} and {b_label} pool to a cosine inside the cascade band"
                  f" (>= {CASCADE_BAND['stage_cos_min']})", line)
    check(drift["only_a"] == 0 and drift["only_b"] == 0,
          f"{a_label} and {b_label} wrote the same set of rows",
          f"only in {a_label}: {drift['only_a']}, only in {b_label}: {drift['only_b']}"
          " — a row present on one side only is a different corpus, not a wider band")


def cross_width_ids(sides: list[tuple[str, dict]], band_ref: str) -> None:
    """Cross-width token-id agreement on the library route, where bit-identity is gone (D-11).

    Two widths of a library route can sample different tokens at a row, and §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md says exactly where: at
    rows whose top-1/top-2 gap was thin. So a differing id is admissible when the row's room is under
    the flip line and red otherwise — the same rule section (a) applies to a pin, with `room` read as
    the smaller of the two sides' own gaps because in this comparison neither side is a reference (the
    audit defined its flip line the same way; `flip_admissible` is where that is written).

    Row files are mapped from the STEP lines rather than assumed, and the mapping is the trivial one:
    `decode_run` prints the row's ABSOLUTE position as `index=` for BOTH kinds (`kind=prefill index=0…`
    then `kind=decode index=147…`), and dumps that row as `logits{index}.bin`. There is no `prompt_len`
    to add — an earlier version added one and every flip then pointed at a file one prompt-length past
    the dump, which is a bug that could only ever be found on this route, and nothing had been on it.
    Both sides walked in lockstep, so a structural difference in the two runs is reported as one rather
    than silently pairing rows.
    """
    label_a, a = sides[0]
    for label_b, b in sides[1:]:
        sa, sb = a["run"]["steps"], b["run"]["steps"]
        if len(sa) != len(sb):
            check(False, f"{label_a} and {label_b} ran the same number of rows",
                  f"{len(sa)} STEP lines vs {len(sb)} — id comparison needs them position by position")
            continue
        flips, reds = [], []
        for i, (x, y) in enumerate(zip(sa, sb)):
            if x.get("kind") != y.get("kind") or x.get("index") != y.get("index"):
                check(False, f"{label_a} and {label_b} agree row {i} is the same row",
                      f"{x.get('kind')}/{x.get('index')} vs {y.get('kind')}/{y.get('index')}")
                continue
            xa, xb = x.get("sampled"), y.get("sampled")
            if xa == xb or xa in (None, -1) or xb in (None, -1):
                continue
            # `parse_run` keeps a STEP field as the string it was printed as, and the index is already
            # the absolute row position, so the only arithmetic needed here is the `int`.
            fidx = int(x["index"])
            pa, pb = a["dir"] / f"logits{fidx}.bin", b["dir"] / f"logits{fidx}.bin"
            if not pa.exists() or not pb.exists():
                check(False, f"{label_a} and {label_b} differ at row {i} and both rows are readable",
                      f"{pa if not pa.exists() else pb} is missing — with no rows there is no room to"
                      " price the flip against")
                continue
            ra = np.fromfile(pa, dtype="<f4").astype(np.float64)
            rb = np.fromfile(pb, dtype="<f4").astype(np.float64)
            if ra.shape != rb.shape:
                check(False, f"{label_a} and {label_b} row {i} is comparable",
                      f"logits{fidx}.bin is {ra.shape} floats on {label_a}, {rb.shape} on {label_b}")
                continue
            a_id, a_val, a_gap = top2(ra)
            b_id, b_val, b_gap = top2(rb)
            where = f"{x.get('kind')} row {x.get('index')}"
            pair_ulp = flip_room_ulp((a_val, a_gap), (b_val, b_gap))
            if flip_admissible((a_val, a_gap), (b_val, b_gap)):
                flips.append(f"{where}: {label_a} {xa} / {label_b} {xb}, room"
                             f" {pair_ulp:.4g} BF16 ULP"
                             f" (< {CASCADE_BAND['flip_room_max_ulp']:.2f};"
                             f" own gaps {a_gap:.6g} / {b_gap:.6g})")
            else:
                reds.append(f"{where}: {xa} vs {xb} with room {pair_ulp:.4g} BF16 ULP"
                            f" (both sides' own gaps {a_gap:.6g} / {b_gap:.6g})")
        check(not reds,
              f"{label_a} and {label_b} sample the same tokens at every row that was not a coin toss"
              f" — the cascade flip rule, {band_ref}",
              f"flip line: {flip_room_line()}. "
              f"{len(sa)} rows compared; {len(flips)} thin-room flip(s)"
              + (f": {'; '.join(flips)}" if flips else "")
              + (f" | {len(reds)} RED: {'; '.join(reds)}" if reds else ""))


def batched_proof_checks(sides: dict, counts: dict, prompt_len: int, decode_steps: int,
                         wave_rows: int, *, stand_in: bool) -> None:
    """Section (h). `sides` is what `batched_ab_stand_in` returns, or the same shape built from real
    `decode_run` runs; one judging function, so the negative exercises the code that judges the
    engine. Two sides (`batched`, `perrow`) is P4b/P5's shape; a third (`carry`, P6) is judged by the
    same rules and compared for bit-identity against the wide side."""
    section("(h) the batched path — counted launches, not inferred outputs (§08.2, P4b/P5/P6)")
    batched, perrow = sides["batched"], sides["perrow"]
    judged = [("batched", batched), ("one-row", perrow)]
    if "carry" in sides:
        judged.append(("carry", sides["carry"]))

    # The knob first: if `--wave-rows` were ignored, every count below would be an accident. The
    # engine echoes the scratch width it actually allocated, on the PREFILL line.
    for label, side in judged:
        echoed = side["run"].get("prefill", {}).get("wave_rows")
        check(echoed is not None and str(echoed) == str(side["requested_wave_rows"]),
              f"decode_run honoured --wave-rows on the {label} side",
              f"PREFILL wave_rows={echoed}, the command asked"
              f" {side['requested_wave_rows']}")

    want = {label: wave_expect(counts, prompt_len, decode_steps, side["requested_wave_rows"])
            for label, side in judged}
    for label, side in judged:
        got = side["run"].get("launchstat", {})
        check(bool(got), f"the {label} run reported observed launch counters",
              "no LAUNCHSTAT line: the counters are compiled out or the driver stopped printing"
              " them, and section (h) has nothing to judge" if not got else "")
        for family in sorted(counts):
            obs = got.get(family)
            exp = want[label][family]
            check(obs == exp, f"{label} side: {family} launches match the wave arithmetic",
                  f"observed {obs} vs §07's plan for a {prompt_len}-token prompt +"
                  f" {decode_steps} steps at wave_rows={side['requested_wave_rows']}: {exp}")

    # The two halves of the claim, spelled out so a FAIL names the thing that broke rather than a
    # dict comparison.
    gl = batched["run"].get("launchstat", {}).get("glue", {})
    check(gl.get("calls_batched", 0) > 0 and gl.get("rows_max", 0) == min(prompt_len, wave_rows),
          "the batched side issues ROW-BATCHED glue launches",
          f"glue calls={gl.get('calls')} calls_batched={gl.get('calls_batched')}"
          f" rows_max={gl.get('rows_max')}, expected rows_max={min(prompt_len, wave_rows)} for a"
          f" {prompt_len}-token prompt. rows_max=1 here means the wave is one row at a time and the"
          " batched glue kernel is dead code")
    mlp_gateup = batched["run"].get("launchstat", {}).get("mlp_gateup", {})
    mlp_down = batched["run"].get("launchstat", {}).get("mlp_down", {})
    check(mlp_gateup.get("calls_batched", 0) > 0 and mlp_down.get("calls_batched", 0) > 0
          and mlp_gateup.get("rows_max", 0) == min(prompt_len, wave_rows)
          and mlp_down.get("rows_max", 0) == min(prompt_len, wave_rows),
          "the batched side issues TOKEN-BATCHED MLP launches (gateup and down)",
          f"gateup {mlp_gateup} down {mlp_down}, expected rows_max="
          f"{min(prompt_len, wave_rows)} on both")
    rg = perrow["run"].get("launchstat", {}).get("glue", {})
    check(rg.get("calls_batched", -1) == 0 and rg.get("rows_max", -1) == 1,
          "the one-row side issues ONE-ROW launches (so the knob produces the shape it claims)",
          f"glue calls_batched={rg.get('calls_batched')} rows_max={rg.get('rows_max')}"
          " — both must be 0 and 1, or the run named `one-row` was not one row at a time and the"
          " comparison below proves nothing")

    # P5's half, spelled out separately rather than folded into the loop above: the projection family is
    # the one whose batching could legitimately have moved bits (a long dot over `hidden`/`heads`, not a
    # residual add), so "the wave batched the projections" is the claim the fixtures cannot make from
    # their outputs — and `--break-wave`'s one-row projection injection is what shows this check bites.
    pj = batched["run"].get("launchstat", {}).get("proj", {})
    check(pj.get("calls_batched", 0) > 0
          and pj.get("rows_max", 0) == min(prompt_len, wave_rows),
          "the batched side issues TOKEN-BATCHED projection launches",
          f"proj calls={pj.get('calls')} calls_batched={pj.get('calls_batched')}"
          f" rows_max={pj.get('rows_max')}, expected rows_max={min(prompt_len, wave_rows)} for a"
          f" {prompt_len}-token prompt. rows_max=1 here means every projection still ran one token per"
          " launch — the §06.2 \"one GEMM per projection\" form is dead code")
    rpj = perrow["run"].get("launchstat", {}).get("proj", {})
    check(rpj.get("calls_batched", -1) == 0 and rpj.get("rows_max", -1) == 1,
          "the one-row side issues ONE-ROW projection launches",
          f"proj calls_batched={rpj.get('calls_batched')} rows_max={rpj.get('rows_max')}"
          " — both must be 0 and 1, or the side named `one-row` was not one token at a time")

    # P6's half, in the same shape again: the three GDN sites the wave now issues once per layer
    # (conv §04.2, gates §04.3, gated norm §04.5). They are the launches whose batching could have
    # moved bits — the conv reads a carried window and the norm reduces over a head — so "the wave
    # batched the mixer" is again a claim the outputs cannot make. `gdn_rec` (§04.4's scan) is absent
    # from this family list on purpose: it has no width to report, so counting it would be a check that
    # cannot fail (`engine/launch_stats.h`).
    for fam, what in (("gdn_conv", "conv (§04.2)"), ("gdn_gates", "gates (§04.3)"),
                      ("gdn_gated_norm", "gated norm (§04.5)")):
        b = batched["run"].get("launchstat", {}).get(fam, {})
        check(b.get("calls_batched", 0) > 0 and b.get("rows_max", 0) == min(prompt_len, wave_rows),
              f"the batched side issues TOKEN-BATCHED {what} launches",
              f"{fam} calls={b.get('calls')} calls_batched={b.get('calls_batched')}"
              f" rows_max={b.get('rows_max')}, expected rows_max={min(prompt_len, wave_rows)} for a"
              f" {prompt_len}-token prompt. rows_max=1 here means the mixer still runs one token per"
              f" launch and P6's wide {fam} kernel is dead code")
        r = perrow["run"].get("launchstat", {}).get(fam, {})
        check(r.get("calls_batched", -1) == 0 and r.get("rows_max", -1) == 1,
              f"the one-row side issues ONE-ROW {what} launches",
              f"{fam} calls_batched={r.get('calls_batched')} rows_max={r.get('rows_max')}"
              " — both must be 0 and 1")

    # P8's half, in the same shape a fourth time: the attention site (§05.1-§05.7 over the whole wave).
    # It is the family whose batching had the most room to move bits — a wave's row t reads cache lanes
    # another row appended, so the ordering is a real dependency rather than an accident of layout — and
    # it is the family the prototype had to prove separately for that reason (`prototype/attn_decode`
    # §11: a wave is bit-identical to its rows run one at a time). Counted as SITES, one per attention
    # layer per body, because how many kernels a site submits is lane depth, not wave width: see
    # `engine/launch_stats.h`. `--break-attn-one-row` is the aimed negative.
    at = batched["run"].get("launchstat", {}).get("attn", {})
    check(at.get("calls_batched", 0) > 0
          and at.get("rows_max", 0) == min(prompt_len, wave_rows),
          "the batched side issues MULTI-ROW attention launches (one site per attention layer,"
          " not one per token)",
          f"attn calls={at.get('calls')} calls_batched={at.get('calls_batched')}"
          f" rows_max={at.get('rows_max')}, expected rows_max={min(prompt_len, wave_rows)} for a"
          f" {prompt_len}-token prompt. rows_max=1 here means the wave's queries still go through"
          " attention one row per launch, and P8's staged wave form is dead code")
    rat = perrow["run"].get("launchstat", {}).get("attn", {})
    check(rat.get("calls_batched", -1) == 0 and rat.get("rows_max", -1) == 1,
          "the one-row side issues ONE-ROW attention launches",
          f"attn calls_batched={rat.get('calls_batched')} rows_max={rat.get('rows_max')}"
          " — both must be 0 and 1")

    # Same work, fewer launches. `rows_total` wave-independent is the half that says the launch
    # delta is batching rather than skipped work. Every side is compared against the wide one, so a
    # third (--wave-rows 5) side is held to the same "not less work" rule as the one-row side.
    for family in sorted(counts):
        b_total = batched["run"].get("launchstat", {}).get(family, {}).get("rows_total")
        for label, side in judged:
            if label == "batched":
                continue
            o_total = side["run"].get("launchstat", {}).get(family, {}).get("rows_total")
            check(b_total == o_total and b_total is not None,
                  f"the {label} side covers the same {family} rows as the batched side"
                  " (fewer launches, not less work)",
                  f"rows_total {b_total} vs {o_total}")
    b_calls = sum(c.get("calls", 0) for c in batched["run"].get("launchstat", {}).values())
    r_calls = sum(c.get("calls", 0) for c in perrow["run"].get("launchstat", {}).values())
    # The delta belongs to the PROMPT wave, so that is what gets compared. Generation steps are one
    # token wide whatever the prompt does — `submit_token_body` is `submit_wave_body` with n = 1, and
    # batching the mixer is P5/P6/P7's job, not this one's — so on a fixture like `dec64t` (27 prompt
    # tokens, 64 steps) the whole-run ratio is only 1.4x even though the wave itself is 27x cheaper.
    # Quoting the whole-run ratio as "the batching win" would understate it and, as a threshold, would
    # fail a correct engine.
    decode_share = sum(counts.values()) * decode_steps
    b_wave, r_wave = b_calls - decode_share, r_calls - decode_share
    waves_b = len(wave_row_sizes(prompt_len, wave_rows))
    waves_r = len(wave_row_sizes(prompt_len, 1))
    wave_ratio = (r_wave / b_wave) if b_wave else 0.0
    check(b_wave > 0 and r_wave > b_wave and r_wave * waves_b == b_wave * waves_r,
          "the batched run issues materially fewer batchable launches for the prompt wave",
          f"prompt wave: {b_wave} glue+proj+MLP+GDN launches across {waves_b} wave(s) vs {r_wave}"
          f" across {waves_r} — {wave_ratio:.0f}x fewer for a {prompt_len}-token prompt. The whole"
          f" run is {b_calls} vs {r_calls}: {decode_steps} generation steps are one token wide"
          " however the prompt is batched")

    # And the outputs must not have moved — which means something different per route, and the route
    # has to be said out loud because the stronger claim is the one a reader assumes. On `--gemm hand`
    # the width cannot move a bit (measured, and the reason route (c) is landable at all), so the rule is
    # byte-identity and every id must agree. On `--gemm library` the widths are three different
    # summation orders (§A1 of progress/2026-10-14/route-c-audit-2026-10-14.md: hipBLAS picks a different solution at every N), so the rule is the measured
    # cascade band plus the flip rule, and the lines say so.
    #
    # WHY THIS BLOCK IS THE DEFERRED HALF OF SECTION (h), AND THE LAUNCH COUNTERS ABOVE IT ARE NOT. These
    # are the checks whose independent variable is the WAVE WIDTH: batched vs one-row vs carried, same
    # form on every side, different number of rows per launch. Under `--scope form` the question is which
    # attention FORM is right above MAXSC, and a wave-width verdict cannot answer it — §17 is the exhibit,
    # where 26 failures here outran the two HF-row readings that could. The launch counters above stay
    # armed in every scope: they count SITES, not lanes, so no attention form can move them, and they are
    # the only thing in the tree that can tell "batching cost nothing" from "batching never ran".
    # See the `SCOPES` comment for why deferral (with the results still printed and still written to
    # `wave_carry_deferred.json`) was chosen over re-running the same form on both sides.
    keys = lambda s: [(st.get("kind"), st.get("index"), st.get("input"), st.get("sampled"),
                       st.get("pos_before"), st.get("pos_after")) for st in s["run"]["steps"]]
    with defer(WAVE_CARRY_REASON, on=SCOPE == "form"):
        if route_is_library():
            cross_width_ids([("batched", batched), ("one-row", perrow)]
                            + ([("carry", sides["carry"])] if "carry" in sides else []),
                            "D-11, from the 24 flips in 1694 rows, all thin-room, that §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md counted")
            wave_band_checks(batched["dir"], perrow["dir"], "batched", "one-row",
                             "the route (c) wave-width band (D-11)")
        else:
            check(keys(batched) == keys(perrow) and bool(keys(batched)),
                  "both sides sample the same token ids",
                  f"{len(keys(batched))} prefill rows + generation steps compared"
                  + (" — 0 lines, which is a broken run, not a match" if not keys(batched) else ""))
            nfiles, nbytes, differing = compare_dump_dirs(batched["dir"], perrow["dir"])
            check(not differing and nfiles > 0, "both sides write bit-identical logits",
                  f"{nfiles} files, {nbytes} bytes compared, {len(differing)} differing"
                  + (f": {differing[:5]}" if differing else ""))

    # ---- P6: the conv window CARRIED across wave boundaries (§2.4's end condition, A-04's `T > 1`).
    # This is the check the wide run cannot make for itself: at `--wave-rows 512` a shipped fixture is
    # ONE wave, so the window is read once, at the zeros §08.2 starts a request with, and written once.
    # Every bug in the carry — zero-padding where the window belongs, a window read after the launch
    # instead of before, new_win taken from the post-conv `u` instead of the pre-conv `raw` — is
    # invisible on a single-wave prompt and blindingly visible here, because each of the
    # ceil(T/5)−1 interior boundaries re-reads what the previous wave left. These are the logits of a
    # prompt ended mid-chunk, so if this passes the window is right, not merely plausible.
    if "carry" in sides:
        carry = sides["carry"]
        c_rows = carry["requested_wave_rows"]
        sizes = wave_row_sizes(prompt_len, c_rows)
        check(len(sizes) > 1, "the carried-window side really splits the prompt into several waves",
              f"--wave-rows {c_rows} on a {prompt_len}-token prompt gives {len(sizes)} wave(s). One"
              " wave would make this whole side vacuous — there would be no interior boundary to carry"
              " across, which is the case the wide run already covers")
        cl = carry["run"].get("launchstat", {})
        for fam, what in (("gdn_conv", "conv (§04.2)"), ("gdn_gates", "gates (§04.3)"),
                          ("gdn_gated_norm", "gated norm (§04.5)")):
            c = cl.get(fam, {})
            check(c.get("calls_batched", 0) > 0
                  and c.get("rows_max", 0) == min(prompt_len, c_rows),
                  f"the carried-window side issues wide {what} launches (so a window really was"
                  " carried between waves)",
                  f"{fam} {c}, expected rows_max={min(prompt_len, c_rows)} on every interior wave")
        if route_is_library():
            with defer(WAVE_CARRY_REASON, on=SCOPE == "form"):
                wave_band_checks(batched["dir"], carry["dir"], "batched", "carry",
                                 "the carried conv window, A-04's T>1 end condition (D-11)")
            notes.append(f"conv window carried across {len(sizes) - 1} interior wave boundary(ies) at"
                         f" --wave-rows {c_rows}, logits inside the cascade band rather than"
                         " byte-identical to the single-wave run. That is a WEAKER claim than the"
                         " `--gemm hand` row above, and it has to be read as one: a window read from"
                         " the wrong rows still moves logits by orders more than the band allows, but"
                         " the band no longer proves the bytes are the same bytes, because on this"
                         " route they are not. This is A-04's `T > 1` clause tested, not assumed."
                         + (" DEFERRED under --scope form: a wave-width claim, attributed to neither"
                            " attention form." if SCOPE == "form" else ""))
        else:
            with defer(WAVE_CARRY_REASON, on=SCOPE == "form"):
                nk, nb, diff = compare_dump_dirs(batched["dir"], carry["dir"])
                check(not diff and nk > 0,
                      f"the {len(sizes)}-wave side writes bit-identical logits to the one-wave side"
                      " — the carried conv window is exact (A-04's T>1 end condition)",
                      f"{nk} files, {nb} bytes compared, {len(diff)} differing"
                      + (f": {diff[:5]}" if diff else "")
                      + ". A difference here is the window, not rounding: both sides read the same weights in"
                        " the same order and every other family is already proven bit-identical above")
            notes.append(f"conv window carried across {len(sizes) - 1} interior wave boundary(ies) at"
                         f" --wave-rows {c_rows}, logits bit-identical to the single-wave run"
                         f" ({nk} files). This is A-04's `T > 1` clause tested rather than assumed."
                         + (" DEFERRED under --scope form: a wave-width claim, attributed to neither"
                            " attention form." if SCOPE == "form" else ""))
    if stand_in:
        notes.append("section (h) in dry-run mode is the gate's own arithmetic compared with itself"
                     " (the stand-in emits both sides), exactly as section (d) is: it proves the"
                     " checker bites, never anything about the engine.")
    notes.append(
        f"P4b+P5+P6+P8 launch count on {REFS.name}: the {prompt_len}-token prompt wave issued {b_wave}"
        f" batchable launches vs {r_wave} with --wave-rows 1 ({wave_ratio:.0f}x fewer);"
        " the whole run, " f"{b_calls} vs {r_calls}, because {decode_steps} generation steps are one"
        " token wide either way. Counts, never durations — the timing pass is the main agent's"
        " (bench/performance-status.md §6) and the command is in engine/results.md.")




def run_batched_ab(args, prompt_ids: list, n_decode: int, greedy: list, replay: bool,
                   work: Path, wave_rows: int, carry_rows: int = 0) -> dict:
    """Run this fixture's prompt twice for section (h): once at the scratch's full wave width, once
    with `decode_run --wave-rows 1`. `carry_rows` (P6) adds a third run in small waves, which is the
    only shape in which the conv window has to be carried from one wave to the next.

    Same flags as the main run apart from `--wave-rows` and `--out`, plus `--prefill-logits all` on
    BOTH sides — under §06.5's default there would be one logits vector per side to compare instead of
    the whole wave, and comparing one row when the batch moved hundreds is not a comparison. This is a
    device path; the CPU stand-in for it is `batched_ab_stand_in`.
    """
    def side(name: str, rows: int) -> dict:
        out_dir = work / name
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("logits*.bin"):
            stale.unlink()      # section (h) must judge this run, not the last one's leftovers
        cmd = [args.binary, "--model", args.model, "--ids", ",".join(map(str, prompt_ids)),
               "--steps", str(n_decode), "--out", str(out_dir),
               "--prefill-logits", PREFILL_LOGITS_ALL, "--wave-rows", str(rows),
               # The same route on every side of the A/B, and said out loud rather than left to the
               # default: section (h) chooses bit-identity or the cascade band on this, and the three
               # sides have to be the same route for either reading to mean anything.
               "--gemm", GEMM_ROUTE,
               # The pin set too, and for the identical reason with a sharper edge: the pinned
               # `down_proj` algo and the heuristic's rank-0 are different kernels with different
               # reduction orders, so a width A/B whose sides ran different pin sets measures two things
               # at once — and section (h)'s band is keyed to the pairing, not to a route word.
               "--gemm-pins", GEMM_PINS,
               # Same argument as the main run, one level down: an A/B of WAVE WIDTHS in which the two
               # sides silently ran different attention FORMS compares two things at once. §13's device
               # pass is exactly the pass that was read while one side had silently thrown inside
               # `prefill_wave`, so the form is named on the command line here too.
               "--attn-tile", args.attn_tile,
               # The touch SPELLING, for the identical reason: a width A/B whose two sides ran different
               # inner spellings of the same form measures two things at once.
               "--attn-touch", args.attn_touch,
               # The GDN scan too, for the identical reason: the two sides of a width A/B must differ
               # in width and nothing else. And its SPELLING for the same reason again — section (j)
               # is the only place the two spellings are allowed to differ, and it differs them alone.
               "--gdn-scan", args.gdn_scan, "--gdn-chunk", str(args.gdn_chunk),
               "--gdn-spell", args.gdn_spell]
        if replay:
            cmd += ["--replay", ",".join(str(t) for t in greedy[:-1])]
        print("  $ " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        check(proc.returncode == 0, f"the {name} A/B run exits 0",
              f"rc={proc.returncode}" + (" " + proc.stderr.strip()[-300:] if proc.stderr.strip() else ""))
        return {"requested_wave_rows": rows, "dir": out_dir, "run": parse_run(proc.stdout)}

    out = {"batched": side("ab_batched", wave_rows), "perrow": side("ab_perrow", 1)}
    if carry_rows > 1:
        out["carry"] = side("ab_carry", carry_rows)
    return out


# ---- (j) the `gdn_rec` spellings, all run against each other --------------------------------------
#
# WHY A SECTION AND NOT A HOPE. `engine/gdn_rec_unroll4_kernels.h` and
# `engine/gdn_rec_fused_kernels.h` are each claimed BIT-identical to the shipped `rec_kernel` — that is
# their admission, not a band they get to fail into (§10.1 has no room for a recurrence change and
# neither D32 nor D50 licenses one). A per-fixture green at `--gdn-spell unroll4` or `fused` is
# NOT that proof: the pins are judged inside bands that would admit a re-associated dot, so "the gate
# stayed green" cannot distinguish "identical" from "inside the allowance". The only question that
# decides it is "do the spellings write the same bytes?", and that needs every run in one gate.
#
# WHY EVERY SIDE IS COMPARED AGAINST `ship` AND NOT PAIRWISE-CHAINED. `ship` is the form every pin in
# this tree is pinned through, so it is the reference the claim is written against; a chain
# (`unroll4` vs `ship`, then `fused` vs `unroll4`) would make a regression in one link read as a pass
# in the next. One reference and N-1 comparisons is also the shape that survives a fourth spelling
# being added: the loop grows, the claim does not change shape.
#
# WHAT THIS SECTION CAN SEE, stated honestly because it is not everything. It compares every byte the
# engine dumps: the whole logits corpus at `--prefill-logits all`, which is `o` several reductions
# downstream. It cannot see the state per element, because the engine dumps no state VALUES —
# `--state-fp` prints §08.6 moment sums, which are order-invariant by design and therefore cannot
# witness bit-identity of a buffer. The per-element `o`-and-state diff of the engine's own kernel is
# `prototype/gdn_rec_isa/gdn_rec_isa --modes receng,rec2c4` (`receng` launches
# `engine/gdn_rec_unroll4_kernels.h`, `rec2c4` is the same source the engine's fused header is
# transcribed from), and the two witnesses are complementary: this one runs the shipping launch path
# on pinned prompts, that one reads the state buffer itself.
# Cost: one device run per spelling per fixture (three today), at the scratch's full wave width.
GDN_SPELLS = ("ship", "unroll4", "fused")

# The reference every other side is byte-compared against: the spelling the pins themselves were
# pinned through, and the one the engine shipped before any spelling knob existed.
GDN_SPELL_REFERENCE = "ship"


def run_gdn_spell_ab(args, prompt_ids: list, n_decode: int, greedy: list, replay: bool,
                     work: Path) -> dict:
    """Run this fixture once per `--gdn-spell` value, at the scratch's full wave width.

    Every flag is identical on both sides except the spelling — same route, same pin set, same
    attention form and touch, same scan — because an A/B whose two sides differ in two things
    adjudicates nothing (`run_batched_ab`'s reason, one knob over). `--prefill-logits all` for the
    same reason section (h) uses it: at §06.5's default there would be one row per side to compare
    instead of the whole wave, and comparing one row when the recurrence ran thousands is not a
    comparison. This is a device path; `gdn_spell_stand_in` is its dry-run counterpart.
    """
    def side(spell: str) -> dict:
        out_dir = work / f"ab_spell_{spell}"
        out_dir.mkdir(parents=True, exist_ok=True)
        for stale in out_dir.glob("logits*.bin"):
            stale.unlink()      # the diff must judge this run, not the last one's leftovers
        cmd = [args.binary, "--model", args.model, "--ids", ",".join(map(str, prompt_ids)),
               "--steps", str(n_decode), "--out", str(out_dir),
               "--prefill-logits", PREFILL_LOGITS_ALL,
               "--gemm", GEMM_ROUTE, "--gemm-pins", GEMM_PINS,
               "--attn-tile", args.attn_tile, "--attn-touch", args.attn_touch,
               "--gdn-scan", args.gdn_scan, "--gdn-chunk", str(args.gdn_chunk),
               # THE ONE DIFFERENCE. Named explicitly on both sides rather than left to the binary's
               # default on one of them: a side that got its spelling from a default is a side whose
               # spelling the log cannot prove.
               "--gdn-spell", spell]
        if replay:
            cmd += ["--replay", ",".join(str(t) for t in greedy[:-1])]
        print("  $ " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        check(proc.returncode == 0, f"the {spell} gdn-spell A/B run exits 0",
              f"rc={proc.returncode}" + (" " + proc.stderr.strip()[-300:] if proc.stderr.strip() else ""))
        return {"dir": out_dir, "run": parse_run(proc.stdout)}

    return {s: side(s) for s in GDN_SPELLS}


def gdn_spell_stand_in(work: Path) -> dict:
    """Dry-run counterpart: replays the pins into EVERY side so the comparison machinery below runs on
    real bytes without a device. NOT A RESULT — `--dry-run` says so on its own face, and the stand-in
    copying one corpus to every side is exactly what the real run must prove is not cheating: the pins
    are the same bits on every side here by construction, and on a device only by admission."""
    out = {}
    for spell in GDN_SPELLS:
        d = work / f"ab_spell_{spell}"
        d.mkdir(parents=True, exist_ok=True)
        for stale in d.glob("logits*.bin"):
            stale.unlink()
        for src in sorted(work.glob("logits*.bin")):
            shutil.copyfile(src, d / src.name)
        out[spell] = {"dir": d, "run": {"numerics": {"gdn_spell": spell}}}
    return out


def gdn_spell_checks(sides: dict, *, stand_in: bool) -> None:
    """Every side must say which spelling it ran, and every side must have written `ship`'s bytes.

    `stand_in` marks a `--dry-run`: there the corpora are the same pinned bytes copied N times, so the
    identity rows CANNOT fail, and it says so out loud rather than printing a PASS that means nothing.
    The dry run exercises the comparison and the echo plumbing; only a real run adjudicates the port.
    Same `stand_in` discipline as `batched_proof_checks` and `gemm_pin_checks`.
    """
    for spell in GDN_SPELLS:
        side = sides.get(spell)
        if side is None:            # a side that failed to run already failed its own rc check
            continue
        echoed = side["run"].get("numerics", {}).get("gdn_spell")
        check(str(echoed) == spell,
              f"the `--gdn-spell {spell}` side ran the spelling it was given",
              ("its NUMERICS line carried no `gdn_spell=` — the binary predates --gdn-spell, so no "
               "claim about either spelling can be made from this log" if echoed is None else
               f"NUMERICS echoed gdn_spell={echoed}, this side was run with --gdn-spell {spell}"))
    ref = sides.get(GDN_SPELL_REFERENCE)
    compared = 0
    for spell in GDN_SPELLS:
        if spell == GDN_SPELL_REFERENCE:
            continue
        side = sides.get(spell)
        if ref is None or side is None:
            continue        # the rc check above already failed this gate; do not report it twice
        compared += 1
        n_files, n_bytes, differing = compare_dump_dirs(ref["dir"], side["dir"])
        # A comparison over an empty corpus cannot fail, which is the `gdn_collapse/run.py` class of
        # defect this tree has now hit four times. So the denominator is asserted before the verdict.
        check(n_files > 0,
              f"both the `{GDN_SPELL_REFERENCE}` and the `{spell}` gdn-spell sides wrote a logits "
              "corpus to compare",
              f"{n_files} `logitsN.bin` per side, {n_bytes / 1e6:.1f} MB compared"
              if n_files else
              "no `logitsN.bin` on either side — with nothing dumped, \"no differing file\" would be an"
              " absence of evidence rather than a proof of identity, so there is no verdict here")
        check(not differing,
              f"the `{spell}` gdn_rec spelling writes BIT-identical logits to `{GDN_SPELL_REFERENCE}` "
              f"(D32: engine/gdn_rec_unroll4_kernels.h vs prototype/gdn_decode's rec_kernel; D50: "
              f"engine/gdn_rec_fused_kernels.h, the fused prologue on those walks; §04.4)",
              f"{len(differing)} of {n_files} dumped logits rows ({n_bytes / 1e6:.1f} MB compared) "
              f"differ: {differing[:6]}" if differing else "")
        if not differing and not stand_in:
            notes.append(f"gdn-spell A/B: {n_files} logits rows, {n_bytes / 1e6:.1f} MB, "
                         f"bit-identical between `{GDN_SPELL_REFERENCE}` and `{spell}` — that spelling "
                         "moved no bits on this fixture")
    # And the denominator of the comparisons themselves: `GDN_SPELLS` is a three-tuple today, so a run
    # that only produced two sides must not read as "every comparison passed".
    check(compared == len(GDN_SPELLS) - 1,
          f"every non-`{GDN_SPELL_REFERENCE}` gdn-spell side was byte-compared against it",
          f"{compared} of {len(GDN_SPELLS) - 1} expected comparisons ran over the sides this gate "
          f"produced ({', '.join(sorted(sides))}) — a missing side is an absence of a verdict, not a "
          "pass")
    if stand_in:
        notes.append("gdn-spell A/B ran on the DRY-RUN stand-in: every side is the same pinned bytes "
                     "replayed, so the identity rows above are this file exercising its own comparison, "
                     "NOT a verdict about the port — only a real run (without --dry-run) can give one")



def dry_run(man: dict, prompt_ids: list, dec_logits: list, work: Path, break_ids: int = -1,
            break_logits: int = -1, break_noise: int = -1, break_prompt_stop: int = -1,
            replay: bool = False, break_feed: int = -1, break_tie: int = -1,
            tie_alt: int = -1, break_allow: int = -1, tie_same_ulp: int = -1,
            tie_far: int = -1,
            break_band: int = -1,
            break_state_fp: str = "", prefill_logits: str = PREFILL_LOGITS_LAST,
            break_y0: bool = False, y0_alt: bool = False,
            slack_alt: int = -1, break_slack: int = -1, break_runnerup: int = -1,
            break_pins: int = -1, break_pin_names: str = "",
            break_heur_width: int = -1, break_cand: str = "") -> tuple[
                str, int]:
    """A stand-in for `decode_run`, on the CPU, that replays the pins as if the engine were perfect.

    Why this is in a gate: sections (a)-(d) parse a line protocol, read binary dumps, and do float
    maths that has never once run against a real logits file. Discovering a typo there costs a
    quiescent-GPU window, which is the scarcest thing in this repo (`AGENTS.md`). So the plumbing gets
    exercised here, on the CPU, against input built from the same pins the checks will compare with —
    which is exactly why it cannot be read as a result: it passes by construction. Anything it fails
    is a bug in this file, not in the engine. (`--selftest` is what proves it can also FAIL.)

    `break_ids`/`break_logits` inject one deliberate error at that generation step, the way
    `prototype/*/run.py --negative` does: the injected bug has to make the gate fail *at that step*,
    which is the property worth testing — a negative that every check would catch proves nothing
    (`prototype/mlp_gemv/results.md` has that trap written down). `break_prompt_stop` injects the one
    bug the engine actually shipped with: the §07.5 stop check firing on a prompt-phase sample, which
    truncates the run there and leaves both the rest of the prompt and all generation steps unsaid.

    `prefill_logits` picks which row shape the stand-in prints, so that it matches the `decode_run`
    whose output this is a stand-in for. Under `last` (§06.5, the default) only the wave's LAST row has
    a sample and the earlier prefill rows print `sampled=-1`; under `all` every row is priced, which is
    the shape the byte-identity gate compares. It matters for `break_prompt_stop`: an intermediate
    prompt row that was never argmaxed cannot be the row that fires a stop check, so an injection aimed
    at one is refused rather than allowed to pass vacuously.

    `break_y0`/`y0_alt` are D-8's two directions on §06.5's first sampled token: a first answer token
    that is not the pinned one (must FAIL), and the pin's OWN other tied id at a tied prefill row (must
    be ADMITTED). The stand-in writes the pinned prefill logits as the engine's row for this, which is
    the honest shape of the second case — the policy asks whether the engine's logit at the id it chose
    keeps the pinned payload, and only a fixture whose pin ties can say yes.

    `slack_alt`/`break_slack`/`break_runnerup` are the three aims of one quantization-slack injection, and
    unlike the pins above they perturb the logits as well as the sample, because a step the pin could not
    decide is a statement about the engine's own numbers: the lifted id has to be the engine's argmax for
    the injection to look like the thing it is testing. All three refuse a step on the wrong side of the
    slack line, so none of them can quietly degenerate into the ordinary wrong-id test.
    """
    greedy = man["greedy_continuation"]
    cfg = man["text_config"]
    lines = []
    cap = len(prompt_ids) + len(dec_logits) + 1
    plan = launch_plan(cfg["num_hidden_layers"],
                       sum(1 for t in cfg["layer_types"] if t == "linear_attention"),
                       sum(1 for t in cfg["layer_types"] if t == "full_attention"))
    lines.append(f"STATE cap={cap} gdn={ss_sizes()[0]} conv={ss_sizes()[1]}"
                 f" kv={ss_sizes()[2] * cap} total={ss_sizes()[0] + ss_sizes()[1] + ss_sizes()[2] * cap}"
                 f" per_token={ss_sizes()[2]}")
    lines.append("LAUNCH " + " ".join(f"{k}={v}" for k, v in plan.items())
                 + f" total={sum(plan.values())}")
    # The stand-in echoes the lines the ENGINE really prints, not the lines this gate would like to
    # see: on the library route the real binary prints `GEMM route=hipblas handle=1 …` (the handle line
    # names the library, not the route word), and on the hand route `GEMM route=hand`. A stand-in that
    # prints the value a check is hunting for is a check that can never fail — this one printed
    # `route=library` while the engine printed `route=hipblas`, which is how every dry run was green and
    # every real library run was red. `--selftest` asserts this spelling.
    # The route echo has a sibling the real engine prints and this stand-in has to fake: `pins=`. This
    # file has no algo table to count — `algo_pins()` is C++, and the counts belong to the process that
    # ran — so the two numbers come from `standin_pin_counts()`, which reads the SHIPPED count out of the
    # source and otherwise echoes the spec back. That is enough to exercise the parser and the equality;
    # `--break-pins` is what proves the equality can FAIL, and a real run counts its own table. Same
    # standing limit as `hipblas=dry-run` below.
    # `pin_names=` is faked the same way, from the spec: the stand-in has no rows to name, so it names
    # what it was asked for (`--break-pin-names` overrides that to prove the name check discriminates).
    pins_names = parse_pins(GEMM_PINS)
    pins_in, pins_total = standin_pin_counts(pins_names)
    if break_pins >= 0:
        pins_in = break_pins
    # The names a stand-in prints are stand-in names: `shipped-1…N` for the shipped set, where N comes
    # from the source rather than from a remembered \"one\", the given names for a csv, `none` for none.
    pin_names_echo = break_pin_names if break_pin_names else (
        "none" if not pins_names else
        ",".join(f"shipped-{i + 1}" for i in range(pins_in)) if pins_names == ["*"]
        else ",".join(pins_names))
    # `cand=` is the candidate half of the same line, and the stand-in has no candidates to count either:
    # under `all`/`none` it owes zero in force, which is the only half of this field this file can decide.
    cand_echo = break_cand if break_cand else ("0/0" if pins_names == ["*"] or not pins_names
                                              else f"{len(pins_names)}/{len(pins_names)}")
    lines.append("GEMM route=" + ("hipblaslt" if GEMM_ROUTE == "lt" else "hipblas")
                 + " handle=1 workspace_mib=128 dtype=bf16-in/bf16-C/f32-compute"
                 + (f" algo_source=heuristic-rank-0-per-shape heur_width={break_heur_width if break_heur_width >= 0 else heuristic_width_default()}"
                    f" pins={pins_in}/{pins_total} pin_names={pin_names_echo} cand={cand_echo}"
                    if GEMM_ROUTE == "lt" else "")
                 + " stream=default" if route_is_library() else "GEMM route=hand")
    # Same reasoning as the GEMM line above, and the same limit: the stand-in has no hipBLAS to ask, so
    # it echoes `hipblas=dry-run` and `numerics_provenance_checks` refuses to assert anything from it.
    # `gemm=` here is the route the gate was invoked with, and it is the field the route check reads.
    lines.append(f"NUMERICS wave_rows_cap={wave_row_ceiling()} wave_rows={wave_row_ceiling()}"
                 f" gemm={GEMM_ROUTE} hipblas=dry-run")
    for i, t in enumerate(prompt_ids):
        is_last_row = i == len(prompt_ids) - 1
        # Only the LAST prefill row's sample is checked by section (a) — that is the one the fixture
        # pins and the one §06.5 says the wave is required to produce. Under `last` the earlier rows
        # were never argmaxed and say so with -1, exactly as `decode_run` does; under `all` every row is
        # priced and the earlier ones echo the next prompt token.
        if prefill_logits != PREFILL_LOGITS_ALL and not is_last_row:
            sampled, top = -1, "none"
        else:
            sampled = greedy[0] if is_last_row else (
                prompt_ids[i + 1] if i + 1 < len(prompt_ids) else greedy[0])
            top = "1"
        if is_last_row and (break_y0 or y0_alt):
            # D-8's two injections, and they are opposites on purpose: one id the pin never picked, and
            # the pin's own other tied id. `tie_other` refuses the second unless the prefill pin really
            # ties, so the admitting half cannot be exercised vacuously.
            sampled = y0_wrong_id() if break_y0 else tie_other(PREFILL_ROW, "--y0-alt")
        if is_last_row:
            # The engine dumps the row it argmaxed, and under `last` that is this one. Without it a tied
            # `y0` pin could never be proved on the CPU, and "cannot be proved" fails (D-8).
            (work / f"logits{i}.bin").write_bytes(
                load_pin(logits_pin(PREFILL_ROW)).astype(np.float32).tobytes())
        # `stop=` is the §07.5 CHECK being armed and matched — never the raw match. A prompt-phase step
        # reports stop=0 even if it sampled the stop id, because its sample is discarded; that is what
        # decode_run prints and what section (a) asserts. `--break-prompt-stop` is the one case where
        # it is 1, and it reproduces the first device run: the stop match ends the run AT the prompt,
        # so the remaining prompt tokens and all 8 generation steps are simply missing.
        if i == break_prompt_stop:
            if sampled < 0:
                # No argmax on this row, so there is no stop check to fire: the injection would be
                # caught by nothing except itself. Refuse it (the same refusal `--break-tie` and
                # `--break-allow` make when aimed at a step that cannot carry their bug) and name the
                # two ways to make it representable.
                raise SystemExit(
                    f"--break-prompt-stop {i}: prefill row {i} of {len(prompt_ids)} carries no"
                    " argmax under --prefill-logits last (§06.5 prices only the last row), so the bug"
                    f" cannot be injected there. Aim at the last row ({len(prompt_ids) - 1}), or"
                    " pass --prefill-logits all.")
            # The stop check fires BECAUSE the sample is the stop id, so the stand-in has to say both;
            # `decode_run` prints `stop=1` only under that condition.
            lines.append(f"STEP kind={PREFILL_KIND} index={i} input={t} sampled={STOP_ID}"
                         f" top_logit=1 pos_before={i} pos_after={i + 1} stop=1")
            lines.append(f"RESULT mode=run steps={cap - 1} ms_total=0 ms_per_step=0"
                         f" last_sampled={STOP_ID} prefix_len={i + 1} stopped=1")
            return "\n".join(lines), 0
        lines.append(f"STEP kind={PREFILL_KIND} index={i} input={t} sampled={sampled}"
                     f" top_logit={top} pos_before={i} pos_after={i + 1} stop=0")
    for j in range(len(dec_logits)):
        pos = len(prompt_ids) + j
        sampled = greedy[j + 1]
        logits = dec_logits[j]
        # What this step is fed. In a replay that is the caller's list (the reference's own token);
        # `--break-feed` is the driver feeding the wrong entry of it, an off-by-one in the replay
        # list being a completely plausible way to get this wrong.
        fed = greedy[j]
        if j == break_feed:
            fed = greedy[j] + 1
        if j == break_ids:
            sampled = greedy[j + 1] + 1        # one wrong id, exactly where a real divergence starts
        if j == break_tie:
            # The failure the tie policy exists to catch: at a step where the PIN is tied, the engine
            # produces an id that is not one of the tied ones. `untied_id` refuses a step that is not
            # tied, so this injection cannot silently degrade into the ordinary wrong-id test.
            sampled = untied_id(j)
        if j == tie_alt:
            # The other direction, which is the property the policy buys: the engine's id is the
            # pin's OTHER tied id, and the gate must admit that as a TIE rather than fail it.
            sampled = tie_other(j)
        if j == tie_same_ulp:
            # D-1's regression, in the direction the old predicate got wrong: the engine keeps the
            # pin's own top-1 id — so G7 is satisfied and the policy must not be consulted at all —
            # and its F32 logit there sits one BF16 ULP above the pinned payload, which is exactly
            # what the old clause-3-at-the-pin's-top-1 refused (`mt32f` step 11). `pin_tie` refusing an
            # untied step is what stops this degrading into a plain noise injection.
            logits = bf16_bump(logits, pin_tie(j)["top1"])
        if j == tie_far:
            # The bound on the generalised tie clause, on the other side of the line from --tie-alt: a
            # tied id, moved FURTHER from the pinned tie than this step's band licenses. `--tie-alt`
            # proves the clause admits what rounding can reach; this proves it does not admit what
            # rounding cannot. Same id set, so the only thing that can change the verdict is the bound.
            sampled = tie_other(j, flag="--tie-far")
            logits = lift_past_tie_slack(j, logits, sampled)
        if j == break_logits:
            # Flip the argmax: raise the runner-up past the top. This is the smallest logits-level
            # failure that matters — a token chosen wrongly, not a vector that is merely noisy.
            logits = logits.copy()
            order = np.argsort(logits)[::-1]
            logits[order[1]] = logits[order[0]] + 1.0
        if j in (slack_alt, break_slack, break_runnerup):
            # The quantization-slack trio, and they are deliberately the SAME perturbation: lift one id
            # one BF16 step above the pin's own maximum, so it really is the engine's argmax and the
            # engine's own numbers are as innocent as they can be. What differs is only the step it is
            # aimed at, and each needle refuses a step on the wrong side of the line (`slack_injection`):
            #   --slack-alt        a leader, at a step whose margin is under its bound   -> ADMITTED
            #   --break-slack      a non-leader, same kind of step                       -> must FAIL
            #   --break-runnerup   a leader, at a step whose margin is OVER its bound    -> must FAIL
            # The second and third are the two directions the brief asks for: a token the pin never
            # tied, and a token that looks tied but was decided. Because the perturbation is identical in
            # all three, the pair `--slack-alt` green / `--break-runnerup` red is a direct measurement of
            # where the bound sits, not a proof that the gate can fail.
            picked = (slack_injection(j, "alt", "--slack-alt") if j == slack_alt else
                      slack_injection(j, "outside", "--break-slack") if j == break_slack else
                      slack_injection(j, "decidable", "--break-runnerup"))
            logits = lift_over_pin(j, logits, picked)
            sampled = picked
        if j == break_noise:
            # Aggregate corruption that LEAVES the argmax alone: 5 % relative noise puts cos at
            # ≈ 1 − σ²/2 ≈ 0.9987, under G5's 0.9995 and over §10.2's 0.999 cascade floor. This is the
            # case that asks whether the band has teeth of its own, separate from the token id.
            logits = relative_noise(logits)
        if j == break_allow:            # The same class — argmax untouched, aggregate wrong — sized against THIS step's measured
            # route allowance instead of against the flat floor: a deficit over the allowance that still
            # fits under the flat §10.1 G5 floor, i.e. a failure the old band could not have seen. The
            # meanrel line usually goes with it (any noise that moves cos moves meanrel); the assertion
            # is about the cos line failing AT THIS STEP, which is what `selftest` checks.
            logits = allowance_noise(logits, j, SELFTEST_ALLOW_FACTOR)
        if j == break_band:
            # Sized against the cap in force rather than against a fixed percentage, so this is the one
            # aggregate negative that reads the same on either route: outside the band, whichever band the
            # route brought. `band_noise`'s docstring says why the fixed 5 % stopped being one here.
            logits = band_noise(j, logits)
        lines.append(f"STEP kind={DECODE_KIND} index={pos} input={fed} sampled={sampled}"
                     f" top_logit=1 pos_before={pos} pos_after={pos + 1} stop=0")
        (work / f"logits{pos}.bin").write_bytes(logits.astype(np.float32).tobytes())
    lines += state_fp_stand_in_lines(man, len(prompt_ids), len(dec_logits), break_state_fp)
    lines.append(f"RESULT mode={'replay' if replay else 'run'}"
                 f" steps={len(prompt_ids) + len(dec_logits)} ms_total=0 ms_per_step=0"
                 f" last_sampled={greedy[-1]} prefix_len={cap - 1} stopped=0")
    return "\n".join(lines), 0


def ss_sizes() -> list:
    """The three constant state figures, from engine/state_sizes.py (same source section (c) uses)."""
    with contextlib.redirect_stdout(io.StringIO()):
        from engine import state_sizes as ss
    return [ss.gdn_state, ss.conv_window, ss.kv_per_token]


def selftest_declared_gaps(fixture: str) -> list:
    """The declared-logit-gap rule, injected on both sides (`tools/reference_dump.py --logit-steps`).

    One sentence decides the rule: a step whose logits are missing may be skipped only if the FIXTURE
    NAMED that step. Get it wrong in the permissive direction and the gate quietly stops checking
    steps — the failure `prototype/mlp_gemv/results.md` names, a negative every check catches — and
    get it wrong in the strict direction and a legitimately lean fixture cannot be checked at all. So
    there is an injection on each side, plus the three ways a declaration can be a lie:

      * undeclared gap      — must still exit 1 with `pinned dump missing`, naming THAT step;
      * declared gap        — must exit 0 and print `paired N of M steps, K declared gaps`;
      * declaration lies    — a declared step whose logits are present: exit 1;
      * declaration is stale— a declared step past the end of the chain: exit 1;
      * every step declared — a manifest, not a pin: exit 1;
      * engine asked anyway — a gapped fixture must be refused before any run, `--dry-run` included.

    The scratch fixture is a copy of the section-(0) files only (manifest, logits, feed list): copying
    a whole 60-to-292 MB fixture to exercise a five-file code path would make every `--selftest` pay
    for tensors this section never reads. `route_allowance.json` is deliberately NOT copied: a fixture
    copy carrying a priced allowance reads as NEW band keys (`margin`, `margin_rel`) that no
    `bench/band-history.jsonl` line introduces, and the band check would stop the run before section
    (0) — which is that check working correctly, and the reason the three P3 fixtures ship unpriced.
    """
    section("(0) declared logit gaps — a step is skipped only because the pin said so")
    bad = []
    src = ROOT / "data" / "refs" / fixture
    n_steps = len(json.loads((src / "manifest.json").read_text())["greedy_continuation"]) - 1
    if n_steps < 2:
        print(f"  [SKIP] {fixture} pins {n_steps} decode step(s); the gap injections need a chain of"
              " at least 2 (one to drop, one left to pair).")
        return bad
    mid = n_steps // 2
    scratch = ROOT / "data" / "refs" / f"selftest-gap-{os.getpid()}"
    base = [sys.executable, str(Path(__file__).resolve()), "--fixture", scratch.name]

    def build(declare, remove=None):
        """A fresh scratch fixture: the section-(0) files, `remove`d steps deleted, `declare`d steps
        named in the manifest. Rebuilt per case so one case cannot leave state for the next."""
        if scratch.exists():
            shutil.rmtree(scratch)
        scratch.mkdir(parents=True)
        keep = re.compile(r"^(manifest\.json|logits_last\.npy|input_ids\.npy"
                          r"|decode\d+\.(logits_last|input_ids)\.npy)$")
        for entry in src.iterdir():
            if keep.match(entry.name):
                shutil.copyfile(entry, scratch / entry.name)
        for step in (remove if remove is not None else []):
            (scratch / f"decode{step}.logits_last.npy").unlink()
        man = json.loads((scratch / "manifest.json").read_text())
        man["logits_declared_absent"] = [f"decode{s}" for s in (declare or [])]
        (scratch / "manifest.json").write_text(json.dumps(man, indent=1, default=str))

    cases = [
        # (declare, remove)                 rc  needles                                                                                     what
        (None, [mid],                        1, [f"pinned dump missing: {scratch / f'decode{mid}.logits_last.npy'}"],
         "a logits file that is merely MISSING still exits 1, naming the step (no silent skip)"),
        ([mid], [mid],                       0, [f"paired {n_steps - 1} of {n_steps} steps, 1 declared gap",
                                                "!declares a step whose logits are present"],
         "the same step DECLARED in the manifest is skipped, and the pairing count says so"),
        ([mid], None,                        1, ["declares a step whose logits are present"],
         "a declaration naming a step whose logits are actually there is refused (both directions)"),
        ([n_steps + 5], None,                1, ["which is not a step this fixture has"],
         "a declaration past the end of the chain is refused as the stale copy it is"),
        (list(range(n_steps)), None,         1, ["is a manifest, not a pin"],
         "declaring EVERY step away is refused: a pin with no logits has nothing for §10 G7 to judge"),
    ]
    for declare, remove, want_rc, needles, what in cases:
        build(declare, remove)
        p = subprocess.run(base + ["--no-engine"], capture_output=True, text=True)
        bad += _needles(p, want_rc, needles, what)
    # The refusal to run an engine at all on a gapped fixture, aimed at `--dry-run` so it costs no
    # device window: the check sits before the binary is started, which is the only reason this is
    # testable here at all.
    build([mid], [mid])
    p = subprocess.run(base + ["--dry-run"], capture_output=True, text=True)
    bad += _needles(p, 1, ["cannot run the engine against it", f"decode{mid}"],
                    "a gapped fixture is refused a device run instead of quietly judging fewer steps")
    shutil.rmtree(scratch, ignore_errors=True)
    return bad


def _needles(p, want_rc, needles, what) -> list:
    """Same needle language as the main injection table, so a gap case reads the same way: a needle
    starting with `!` must be ABSENT, and the rc has to match. Returns the failures, if any."""
    missing = []
    for needle in needles:
        absent = needle.startswith("!")
        if (needle.lstrip("!") in p.stdout) == absent:
            missing.append(needle)
    ok = p.returncode == want_rc and not missing
    print(f"  [{'PASS' if ok else 'FAIL'}] {what} — rc={p.returncode} want {want_rc}"
          + ("" if not missing else " — missing: " + "; ".join(missing)))
    if not ok:
        print("      " + "\n      ".join(p.stdout.splitlines()[-6:])[:400])
    return [] if ok else [what]


def selftest_declared_gaps(fixture: str) -> list:
    """Injections for the declared-logit-gap rule (`tools/reference_dump.py --logit-steps`).

    One sentence is being tested: a step whose logits are missing may be skipped only if the FIXTURE
    NAMED that step. Get it wrong in the permissive direction and the gate quietly stops checking
    steps — the failure `prototype/mlp_gemv/results.md` names, a negative every check catches — and get
    it wrong in the strict direction and a legitimately lean fixture cannot be checked at all. So there
    is an injection on each side, plus the ways a declaration can be a lie:

      * undeclared gap     — must still exit 1 with `pinned dump missing`, naming THAT step;
      * declared gap       — must exit 0 and print `paired N of M steps, K declared gaps`;
      * declaration lies   — a declared step whose logits are present: exit 1;
      * declaration stale  — a declared step past the end of the chain: exit 1;
      * every step declared — a manifest, not a pin: exit 1;
      * engine asked anyway— a gapped fixture is refused before any run, `--dry-run` included.

    The scratch fixture copies the section-(0) files only (manifest, logits, feed list): copying a
    whole 60–292 MB fixture to exercise a five-file path would make every `--selftest` pay for tensors
    this section never reads. `route_allowance.json` is deliberately NOT among them — a copy of a
    priced fixture reads as NEW band keys (`margin`, `margin_rel`) that no `bench/band-history.jsonl`
    line introduces, and the band check stops the run before section (0) ever sees the injection.
    That is the band check working, and it is also why the three P3 fixtures ship unpriced.
    """
    section("(0) declared logit gaps — a step is skipped only because the pin said so")
    src = ROOT / "data" / "refs" / fixture
    n_steps = len(json.loads((src / "manifest.json").read_text())["greedy_continuation"]) - 1
    if n_steps < 2:
        print(f"  [SKIP] {fixture} pins {n_steps} decode step(s); the gap injections need a chain of"
              " at least 2 — one to drop and one left to pair.")
        return []
    mid = n_steps // 2
    scratch = ROOT / "data" / "refs" / f"selftest-gap-{os.getpid()}"
    keep = re.compile(r"^(manifest\.json|logits_last\.npy|input_ids\.npy"
                      r"|decode\d+\.(logits_last|input_ids)\.npy)$")

    def build(declare=(), remove=()):
        """A fresh scratch fixture: the section-(0) files, `remove`'d steps deleted, `declare`'d steps
        named in the manifest. Rebuilt per case so one case cannot leave state for the next."""
        shutil.rmtree(scratch, ignore_errors=True)
        scratch.mkdir(parents=True)
        for entry in src.iterdir():
            if keep.match(entry.name):
                shutil.copyfile(entry, scratch / entry.name)
        for step in remove:
            (scratch / f"decode{step}.logits_last.npy").unlink()
        man = json.loads((scratch / "manifest.json").read_text())
        man["logits_declared_absent"] = [f"decode{s}" for s in declare]
        (scratch / "manifest.json").write_text(json.dumps(man, indent=1, default=str))

    def run(extra, want_rc, needles, what):
        p = subprocess.run([sys.executable, str(Path(__file__).resolve()),
                            "--fixture", scratch.name] + extra, capture_output=True, text=True)
        # A hard refusal is a `raise SystemExit`, which lands on stderr; a gated check is a `check()`
        # line on stdout. The needles below span both, because the rule under test is exactly the
        # boundary between them — "refuse loudly" versus "skip and say so".
        out = p.stdout + p.stderr
        missing = []
        for needle in needles:
            absent = needle.startswith("!")
            if (needle.lstrip("!") in out) == absent:
                missing.append(needle)
        ok = p.returncode == want_rc and not missing
        print(f"  [{'PASS' if ok else 'FAIL'}] {what} — rc={p.returncode} want {want_rc}"
              + ("" if not missing else " — missing: " + "; ".join(missing)))
        if not ok:
            print("      " + "\n      ".join((p.stdout + p.stderr).splitlines()[-8:])[:500])
        return [] if ok else [what]

    bad = []
    try:
        build(remove=[mid])
        bad += run(["--no-engine"], 1, [f"pinned dump missing: {scratch / f'decode{mid}.logits_last.npy'}"],
                   "logits that are merely MISSING still exit 1, naming the step — no silent skip")
        build(declare=[mid], remove=[mid])
        bad += run(["--no-engine"], 0,
                   [f"paired {n_steps - 1} of {n_steps} steps, 1 declared gap",
                    "!declares a step whose logits are present"],
                   "that step DECLARED in the manifest is skipped, and the pairing line says so")
        build(declare=[mid])
        bad += run(["--no-engine"], 1, ["declares a step whose logits are present"],
                   "a declaration naming a step whose logits are actually there is refused")
        build(declare=[n_steps + 5])
        bad += run(["--no-engine"], 1, ["which is not a step this fixture has"],
                   "a declaration past the end of the chain is refused as the stale copy it is")
        build(declare=list(range(n_steps)))
        bad += run(["--no-engine"], 1, ["is a manifest, not a pin"],
                   "declaring EVERY step away is refused: with no logits there is nothing for §10 G7"
                   " to judge")
        build(declare=[mid], remove=[mid])
        bad += run(["--dry-run"], 1, ["cannot run the engine against it", f"decode{mid}"],
                   "a gapped fixture is refused a device run rather than quietly judging fewer steps"
                   " (aimed at --dry-run: the refusal fires before a binary is started)")
    finally:
        shutil.rmtree(scratch, ignore_errors=True)
    return bad


def echo_injection_target() -> dict:
    """Which fixture can carry the echo case, and the two steps to aim at — DERIVED, never typed.

    The case needs a step the pin could NOT decide (so a proved rule admits a flip there, which is the
    only thing that opens an echo window) and a DECIDABLE step downstream of it (so the injected defect
    is one the pin DID decide — a real defect, not a coin the rule already admitted). Both come from the
    same caps the gate polices (`slack_region_steps`, `step_slack`), walked over every fixture that has a
    manifest, so the must-catch and must-not-catch sets follow the corpus's geometry rather than a
    hand-listed name that rots the day a fixture is re-dumped — the `attn_tile_model.py` lesson, one layer
    over. Returns `{fixture, host, late, passed}`, with `fixture` None when nothing offers both (the
    caller reports that as a hole in the corpus, not a skip, because an unadjudicated echo allowance is
    the whole risk) and `passed` naming every fixture the walk priced and why it rejected it — a census
    that does not say what it read is the "0 of 55" mistake in `AGENTS.md`.
    """
    saved = REFS
    passed: list[str] = []      # the census: what the walk priced and rejected, and why
    try:
        for d in sorted((ROOT / "data" / "refs").iterdir()):
            if not (d / "manifest.json").exists():
                continue
            man = json.loads((d / "manifest.json").read_text())
            if man.get("kind") is not None and "pins" not in man:
                # A carried record, not a pin set — the same discriminator `refuse_non_pin_fixture`
                # uses to exit 3, applied BEFORE `set_fixture`, because a walk of the population must
                # not be able to end the selftest. It is named in the census, not silently passed over.
                passed.append(f"{d.name}: carried record, not a pin set")
                continue
            n_steps = len(man.get("greedy_continuation", [])) - 1
            if n_steps < 2:
                passed.append(f"{d.name}: {n_steps} decode step(s)")
                continue
            if n_steps > FREE_RUN_BOUND:
                passed.append(f"{d.name}: {n_steps} steps, replayed, so it cannot fork")
                continue
            set_fixture(d.name)
            try:
                inside, _narrowest = slack_region_steps(n_steps)
            except SystemExit as exc:
                passed.append(f"{d.name}: {str(exc).strip()[:70]}")
                continue
            if not inside:
                passed.append(f"{d.name}: no step inside the slack region")
                continue
            host = min(inside)
            decidable = [j for j in range(host + 1, n_steps)
                         if (sl := step_slack(j, pin_row(j))) is not None
                         and sl["margin"] >= sl["bound"]]
            if not decidable:
                passed.append(f"{d.name}: nothing decidable downstream of step {host}")
                continue
            return {"fixture": d.name, "host": host, "late": max(decidable), "passed": passed}
        return {"fixture": None, "passed": passed}
    finally:
        set_fixture(saved.name)


def selftest_echo_window() -> list[str]:
    """The echo attribution's own teeth, all on the CPU. Four things have to be proved.

      1. the classifier: nothing admitted opens nothing; `PREFILL_ROW` opens the window at step 0, since
         every decode step is downstream of the first answer token; a flip admitted at step j opens it
         at j+1 and never at j, because up to and including j the run was fed the pin's own tokens.
      2. a fixture where NOTHING is admitted: no line may read ECHO and no sidecar may be written. A
         bucket that fills when nothing was admitted is a bucket that swallows failures for free.
      3. MUST-CATCH — a real numeric defect at a LATE step, downstream of an admitted early one. The
         free-run sections must call it ECHO (printed, counted, attributed) and section (b2) — which has
         no echo window because both sides are fed the same tokens — must be the half that fails the run.
         If (b2) did not fail, the echo allowance would be a hiding place for a defect, and that is the
         only honest objection to this design.
      4. the sidecar's record count must equal the printed line count in both directions, and the
         summary's count must equal the sidecar's (the `gdn_collapse/run.py` trap: a driver whose parse
         matched nothing prints PASS over an empty list).
    """
    bad: list[str] = []

    def case(ok: bool, what: str, detail: str = "") -> None:
        print(f"  [{'PASS' if ok else 'FAIL'}] {what}" + (f" — {detail}" if detail else ""))
        if not ok:
            bad.append(f"{what}: {detail}")

    case(echo_first_step([]) is None, "nothing admitted opens no echo window",
         f"echo_first_step([]) -> {echo_first_step([])}")
    case(echo_first_step([PREFILL_ROW]) == 0,
         "an admitted §06.5 prefill row puts every generation step in the window",
         f"echo_first_step([PREFILL_ROW]) -> {echo_first_step([PREFILL_ROW])}")
    case(echo_first_step([5, 2]) == 3 and echo_first_step([0]) == 1,
         "a flip admitted at step j opens the window at j+1, at the EARLIEST admission, never at j",
         "[5,2] -> 3 and [0] -> 1")
    tgt = echo_injection_target()
    if tgt["fixture"] is None:
        case(False, "a fixture exists whose geometry can carry the echo case",
             "no fixture under data/refs has a step inside the quantization-slack region AND a"
             " decidable step downstream of it, so the echo allowance is unadjudicated. That is a hole"
             " in the corpus, not a pass. The walk priced: " + " | ".join(tgt["passed"]))
        return bad
    fx, host, late = tgt["fixture"], tgt["host"], tgt["late"]
    gate = str(Path(__file__).resolve())

    def run(extra: list, work: Path) -> tuple[int, str, dict | None]:
        p = subprocess.run([sys.executable, gate, "--fixture", fx, "--gemm", GEMM_ROUTE,
                            "--gemm-pins", GEMM_PINS, "--dry-run", "--work", str(work)] + extra,
                           capture_output=True, text=True)
        side = work / ECHO_FILE_NAME
        rec = json.loads(side.read_text()) if side.exists() else None
        return p.returncode, p.stdout, rec

    def tally(out: str) -> tuple[int, int]:
        return (len(re.findall(r"^  \[ECHO-", out, re.M)),
                len(re.findall(r"^  \[ECHO-FAIL\]", out, re.M)))

    print(f"  [info] aimed at {fx}: step {host} sits inside the quantization-slack region (a flip"
          f" there is admitted) and step {late} is the latest step downstream of it that the pin DID"
          " decide — both read from the caps the gate polices, not typed here (`--slack-region"
          f" --fixture {fx}` prints the same two lists). The walk priced "
          + f"{len(tgt['passed'])} other candidate(s): " + " | ".join(tgt["passed"]))
    with tempfile.TemporaryDirectory(prefix="gate_echo_") as td:
        work = Path(td)
        rc, out, rec = run([], work)
        n_all, n_fail = tally(out)
        case(rc == 0 and n_all == 0 and rec is None,
             "a clean run admits nothing, so no line is an ECHO and no sidecar is written",
             f"rc={rc} want 0; {n_all} ECHO lines want 0; sidecar"
             f" {'written' if rec is not None else 'absent'}")
        case("[echo]" not in out, "the nothing-admitted case prints no echo window at all",
             "a `[echo]` line appeared" if "[echo]" in out else "absent, as it must be")

        rc, out, rec = run(["--slack-alt", str(host)], work)
        case(rc == 0 and f"admitted flip at step {host}" in out and rec is not None
             and rec["n_failed"] == 0 and rec["admitted_step"] == host,
             "MUST-NOT-CATCH: an admitted flip at its own step, with nothing wrong downstream, stays"
             " green and names the flip it attributed to",
             f"rc={rc} want 0; window line"
             f" {'seen' if f'admitted flip at step {host}' in out else 'MISSING'}; sidecar"
             f" {rec['n_failed'] if rec else 'absent'} failed, want 0")

        rc, out, rec = run(["--slack-alt", str(host), "--break-runnerup", str(late)], work)
        n_all, n_fail = tally(out)
        echoed_argmax = f"[ECHO-FAIL] step {late}: argmax agrees with the pin" in out
        caught_b2 = f"[FAIL] b2 step {late}" in out
        counted_a = f"[FAIL] generation step {late} samples" in out
        case(rc == 1 and echoed_argmax and caught_b2 and not counted_a,
             "MUST-CATCH: a one-BF16-step lift at a LATE decidable step, downstream of an admitted"
             " flip, is echoed by the free-run sections and RED in section (b2) — the defect is moved,"
             " never lost",
             f"rc={rc} want 1; late step echoed {'yes' if echoed_argmax else 'NO'}; (b2) red"
             f" {'yes' if caught_b2 else 'NO'}; still counted as an ordinary (a) failure"
             f" {'YES — the echo did not apply' if counted_a else 'no'}")
        case(rec is not None and rec["n_checks"] == n_all and rec["n_failed"] == n_fail
             and f"{n_fail} check(s) failed" in out,
             "the sidecar records exactly the lines that printed, in both directions, and the summary"
             " counts the same number",
             f"sidecar {rec['n_checks'] if rec else '-'} records / {n_all} printed;"
             f" {rec['n_failed'] if rec else '-'} failed recorded / {n_fail} printed"
             if rec else "no sidecar was written at all")
        case(rec is not None and rec["echo_from"] == host + 1
             and rec["admitted_as"] == "quantization-slack rule",
             "the sidecar names its own cause: which step, which rule, and where the window starts",
             f"echo_from={rec['echo_from'] if rec else '-'}, admitted_as="
             f"{rec['admitted_as'] if rec else '-'}")

        rc, out, rec = run(["--break-runnerup", str(late)], work)
        n_all, n_fail = tally(out)
        case(rc == 1 and n_all == 0 and rec is None
             and f"[FAIL] step {late}: argmax agrees with the pin" in out,
             "MUST-CATCH, other half: the same late defect with NOTHING admitted upstream is an"
             " ordinary counted failure with no echo window at all",
             f"rc={rc} want 1; {n_all} ECHO lines want 0; the step's own argmax line"
             f" {'failed as usual' if f'[FAIL] step {late}: argmax agrees with the pin' in out else 'DID NOT FAIL'}")
    return bad


def selftest(fixture: str = "dec8") -> int:
    """Prove the gate discriminates, on the CPU, before anyone spends a GPU window on it.

    Runs: the clean dry run must pass; each injected bug must fail, and fail *at its own step*.
    A negative that any check would catch is worth nothing (`prototype/mlp_gemv/results.md`), so
    the assertion is about where the failure landed, not that one happened.

    The tie policy gets three cases, because a policy with only a refusal test is indistinguishable from
    the policy that refuses everything: the engine on the pin's OTHER tied id must PASS as a TIE, the
    engine on an untied third id must FAIL, and the engine that keeps the pin's own top-1 while its logit
    sits 1 BF16 ULP off the tied payload must PASS as the ordinary match it is (D-1, the `mt32f` step 11
    shape). All three are aimed at a step the fixture really has a tie at, and all are skipped out loud if
    this fixture has none.

    The route allowance gets one: a step's logits deficit pushed past its own measured route spread
    while still inside the flat §10.1 G5 floor — i.e. a failure the band this allowance replaced could
    not have seen. Skipped out loud on a fixture with no priced allowance (dec8 today). The plain 5 %
    noise case is likewise skipped out loud when a fixture's OWN allowance at that step is wider than
    the noise's deficit, because then the band admits the injection and expecting a FAIL there would be
    a claim about the flat floor, not about the rule in force.

    And the band history gets one: a scratch tree whose `engine/gate.py` loosens `BAND.cos_hard` with no
    line recording it, which must stop the gate at startup — before a fixture is read, a binary started,
    or a device touched. `tools/band_history.py --selftest` covers the checker's own behaviour; this
    case covers the wiring, because a checker nobody calls is the same as no checker.

    The state fingerprints get three, one per §08.6 failure a moment-sum fingerprint can see (a family
    left at the §08.4 zero, a state or window that never moved, KV lanes that moved when they should
    not have), each aimed at a single row and each required to name that row and no other — the
    locality is the assertion, because a negative every check catches proves nothing. They need a
    fixture that carries fingerprints, so `--selftest --fixture dec64f` is the run that exercises
    section (g); on `dec8` the skip is printed. The section's other mechanical case — recomputing a
    pinned fingerprint from pinned tensors with `tools/state_fp.py::fp_row` and requiring bit equality
    — runs once in this process, since it is a fact about the pins and not about an injection.

    The declared logit gaps get six (`selftest_declared_gaps`), because that rule is one sentence with a
    failure mode on each side: a logits file the fixture did not declare must still stop the gate, and a
    declared one must be skipped and PRINTED as skipped. The injections cover both directions of the
    declaration disagreeing with the disk, the case where every step is declared away, and the refusal
    to run an engine at all against a gapped fixture.
    """
    base = [sys.executable, str(Path(__file__).resolve()), "--fixture", fixture,
            # The route the selftest is run under is the route its children are judged under, otherwise
            # `--selftest --gemm library` would prove the hand route's rules bite and call it done.
            "--gemm", GEMM_ROUTE,
            # ...and so is the pin set, for the identical reason: `--selftest --gemm lt --gemm-pins
            # none` has to prove the NONE config's rules bite, not borrow the `all` config's verdict.
            "--gemm-pins", GEMM_PINS]
    # The cos band line is NAMED by the rule that produced it, so the needle has to know which rule this
    # fixture is under: with a priced allowance the same injected noise reads as
    # `step 5: cos deficit … inside the route allowance …`. `allow_step` is where the allowance
    # injection will be aimed, or None when there is nothing priced to aim it at.
    allow_step = break_allow_step()
    cos_fail_5 = (f"[FAIL] step 5: cos deficit" if allowance_row(5) is not None
                  else f"[FAIL] step 5: cos >= {BAND['cos_hard']} (§10.1 G5)")
    # The 5 % noise injection is sized against the FLAT §10.1 G5 floor, so on a fixture whose OWN
    # allowance at step 5 is wider than what that noise produces it is not a negative at all — the band
    # admits the deficit because two legal reference forms are further apart than that at the step. The
    # honest response is to say so and skip, exactly as `--break-allow` refuses a step where its own
    # injection would not discriminate; the allowance-sized case below asks the same question at the
    # right size for this fixture.
    noise_pin = load_pin("decode5.logits_last.npy").reshape(-1)
    noise_deficit = 1.0 - metrics(relative_noise(noise_pin), noise_pin)["cos"]
    noise_row = allowance_row(5)
    noise_cap = None if noise_row is None else float(noise_row["allowance_one_cos"])
    # The floor is part of "the cap in force" now, so the question "would this injection be caught?" has to
    # ask it. On the library route the cascade floor (2e-03) is wider than what 5 % noise produces
    # (1.2e-03), which makes the fixed-size injection not a negative there at all — the same honest skip
    # the allowance already gets, for the same reason, and `--break-band` is the case that asks the
    # question at the size the route in force actually polices.
    noise_eff, _noise_rel, _noise_clause = cascade_floor(
        noise_cap if noise_cap is not None else 1.0 - BAND["cos_hard"], BAND["meanrel_hard"])
    noise_caught = noise_deficit > noise_eff
    cases = [
        ([], 0, None, "clean dry run"),
        # The stand-in must print the engine's OWN spelling, not the value the route check is looking
        # for, or that check can never fail. The real binary writes `GEMM route=hipblas handle=…` on the
        # library route and `route=hand` on the hand route; the stand-in used to write `route=library`,
        # so the dry runs were green for a whole commit while every real library run was red. The route
        # assertion now reads NUMERICS's `gemm=`, and this case pins both halves of what the check
        # printed, so neither the checker nor its stand-in can drift back into agreeing with each other.
        ([], 0, [f"NUMERICS echoed gemm={GEMM_ROUTE}",
                 f"(GEMM line says {'hipblaslt' if GEMM_ROUTE == 'lt' else ('hipblas' if route_is_library() else 'hand')})"],
         "the route check reads NUMERICS's `gemm=` and the stand-in's GEMM line carries the spelling the"
         " real engine prints, so the check is exercised against something that could disagree"),
        (["--break-ids", "3"], 1, "[FAIL] generation step 3 samples greedy_continuation[4]",
         "wrong id injected at step 3"),
        (["--break-logits", "2"], 1,
         ["[FAIL] step 2: argmax agrees with the pin",
          # The old FAIL line printed "the pin is tied but the engine's id is not one of the tied ids"
          # for EVERY unadmitted argmax failure, and step 2 is untied in all eight fixtures (narrowest
          # gap 0.75, mt32f) — so before the fix this needle's negation was the line that lied. Now the
          # line states the pin's actual top1/top2, the gap in BF16 ULP, and the engine's element error.
          "the pin's winner is unique",
          "!the pin is tied but the engine's id"],
         "top two logits swapped at step 2 — and the FAIL line must give the real decision margin"
         " instead of asserting a tie that does not exist"),
        # The bug the engine actually shipped with, replayed in the stand-in: id 248046 sampled at
        # prompt index 6 ended a 17-token prompt after 7 steps. Both checks have to name it.
        #
        # It is aimed with `--prefill-logits all` on purpose. Since P2 the prompt is a wave and §06.5's
        # default prices only its last row, so row 6 has no argmax at all and there is nothing at that
        # position to fire a stop check on — `dry_run` refuses the injection rather than letting it pass
        # vacuously (the vacuous-negative trap `prototype/mlp_gemv/results.md` names). In the `all` shape
        # every row is argmaxed, the bug is representable again, and this is where the regression that
        # actually happened is caught. The `last` shape gets its own case below, aimed at the one row it
        # does price.
        (["--break-prompt-stop", "6", "--prefill-logits", PREFILL_LOGITS_ALL], 1,
         ["[FAIL] no prompt-phase step fires §07.5's stop check",
          "[FAIL] one step per prompt token",
          "that last step's §07.5 stop check HAD fired"],
         "the stop check fired at prompt index 6 (the first device run's bug: a 17-token prompt"
         " ended after 7 steps and generation never started), in the row shape where an intermediate"
         " prompt row is argmaxed at all"),
        # The forced driver's own failure mode: an off-by-one in the replay list. Everything else
        # can stay perfect, so only the fed-token check can catch it.
        (["--break-feed", "4"], 1,
         "[FAIL] the tokens the engine fed itself are the ones the reference fed",
         "generation step 4 was fed the wrong token (a replay list off by one)"),
    ]
    # The algo-pin echo, on the only route that has pins. Two directions, because a stand-in that can
    # only agree with the gate proves nothing (`prototype/gdn_collapse/results.md` §1: a check whose
    # input cannot be empty-and-wrong is decoration). The clean case pins the field's spelling — the
    # engine's, not this file's invention — and the second injects the count the current spec does NOT
    # imply, which is the failure the equality exists for: a run asked for `none` that quietly ran the
    # pin, or the reverse, reads as a pin verdict while pricing rank-0.
    if GEMM_ROUTE == "lt":
        names = parse_pins(GEMM_PINS)
        # Derive the stand-in's own arithmetic from the same helper the stand-in uses, or this "clean"
        # case would be pinning a spelling the stand-in no longer prints — which is how a selftest becomes
        # a second authority over the thing it is supposed to be checking.
        in_force, table = standin_pin_counts(names)
        wrong_pins = 0 if in_force else 1
        # The names check needs its own injection, and the wrong list depends on what the spec owes: a
        # short list where `all` owes the shipped set, one name where `none` owes none, and a substituted
        # name where a csv owes its own list. Deriving it per spec is what keeps the case meaningful for
        # whatever `--gemm-pins` the selftest is being run under.
        wrong_names = ("a" if names == ["*"] else
                       "someone" if not names else
                       ",".join(["not_a_pin_in_force"] + names[1:]))
        cases += [
            ([], 0, f"echoed pins={in_force}/{table}",
             "the clean stand-in carries the engine's `pins=<shipped in force>/<shipped rows>` field and"
             " the check read BOTH halves of it: a field this file could not parse prints no such line,"
             " and a binary that stopped printing it altogether hits the missing-field FAIL two sections"
             " up"),
            (["--break-pins", str(wrong_pins)], 1,
             "[FAIL] the run ran the algo-pin set this gate is judging it under",
             f"the stand-in echoed {wrong_pins} pin(s) in force while --gemm-pins {GEMM_PINS} owes"
             f" {in_force}. A log like that is a judgment about some other algo config and the gate"
             " must refuse to read it as one about this one — which is exactly how D28's red wide520f"
             " row reached this tree with no way to ask whether the pin or rank-0 produced it"),
            (["--break-pin-names", wrong_names], 1,
             ("the run's pin name list agrees with its own count" if names == ["*"] else
              "really ran no pin" if not names else
              "the run named exactly the pins this gate asked for"),
             f"the stand-in echoed pin_names={wrong_names} while --gemm-pins {GEMM_PINS} owes"
             + (" no pins at all" if not names else
                f" the {in_force}-row shipped set" if names == ["*"] else f" {names}")
             + ". The counts can agree while the SETS do not — `down_proj` (D28's pin, restored by D36)"
               " and `deep_gate_proj+up_proj` are each exactly one pin; this check is about the set"),
            (["--break-cand", "2/3"], 1,
             "bound no candidate pin",
             "the stand-in echoed cand=2/3 under `--gemm-pins all`, i.e. two CANDIDATE kernels served"
             " the wave while the spec names no candidate at all. That is the drift a promotion invites:"
             " if `all` could acquire a candidate, every band and token verdict taken under it would be"
             " a claim about an unknown kernel, and this line is what stops that being silent"),
            (["--break-heur-width", str(heuristic_width_default() + 8)], 1,
             "the run queried the shipped heuristic pool",
             f"the stand-in echoed heur_width={heuristic_width_default() + 8} while the shipped"
             f" kHeuristicWidth is {heuristic_width_default()}. A deeper pool is inert while rank-0 is"
             " admissible and decisive when it is not, which is exactly why a log has to say which one"
             " it ran under rather than leave a reader to infer it from the build date"),
        ]
    if noise_caught:
        size = ("" if noise_cap is None else
                f" — deficit {noise_deficit:.3e} over this step's own cap {noise_cap:.3e}")
        cases.append((["--break-noise", "5"], 1, cos_fail_5,
                      "5 % noise on step 5's logits (argmax untouched, aggregate wrong) — must FAIL"
                      f" under the rule in force{size}"))
    else:
        if noise_cap is not None and route_is_library():
            why = f"allowance {noise_cap:.3e}, floored at the cascade band's {noise_eff:.3e}"
        elif route_is_library():
            why = f"the cascade floor, {noise_eff:.3e}"
        else:
            why = f"this fixture's own allowance at that step, {noise_eff:.3e}"
        print(f"  [SKIP] 5 % noise at step 5: it produces a deficit of {noise_deficit:.3e}, inside the"
              f" cap in force there ({why}), so the band legitimately admits it — two legal routes are"
              " further apart than that. `--break-band` asks the same question at this route's size.")
    # The aggregate negative that means the same thing on either route: noise sized outside whatever band is
    # in force at the step, so the library route is not left with no test that its wider floor still has a
    # far side. Locality: the argmax is untouched by construction (constant-amplitude noise, sized below the
    # top-pair gap), so only the band lines may notice.
    cases.append((["--break-band", "5"], 1, cos_fail_5,
                  "noise sized 4x outside the cap in force at step 5 (argmax untouched) — must FAIL on"
                  " either route, because being floored is not being unlimited"))
    # The tie policy, both directions, at the first step this fixture actually has a tie at — an
    # injection aimed at a step that is not tied would be caught by every check no matter what the
    # policy does, which is the vacuous-negative trap `prototype/mlp_gemv/results.md` names. dec8 has
    # no tied step; dec64f has two (17, 18), so `--selftest --fixture dec64f` is the one that exercises
    # this pair.
    man = json.loads((REFS / "manifest.json").read_text())
    tie_step = next((j for j in range(len(man["greedy_continuation"]) - 1) if pin_tie(j)), None)
    # §07.5 under the DEFAULT row shape. The case above needs `all` because an unpriced row cannot fire a
    # stop check, but the default wave prices exactly one prompt row — the last — and that row's sample IS
    # §06.5's first answer token, so a stop check armed there would end a run before generation began.
    # This is the case that proves the rule is still testable without `all`, so the negative did not go
    # extinct when the intermediate argmaxes did.
    cases.append((
        ["--break-prompt-stop", str(len(man["sources"][0]["ids"]) - 1)], 1,
        ["[FAIL] no prompt-phase step fires §07.5's stop check",
         "[FAIL] the prompt's last token samples greedy_continuation[0]"],
        "the stop check fired on the wave's last row — the only row `--prefill-logits last` prices,"
        " and the one whose sample §06.5 turns into the first generated token; §07.5 still says a"
        " prompt-phase match does not end the run"))
    if tie_step is None:
        print(f"  [SKIP] tie injections: no exact tie anywhere in {fixture}'s pinned logits."
              " Run `--selftest --fixture dec64f` (ties at 17 and 18) or"
              " `--selftest --fixture mt32f` (ties at 11) to exercise the tie policy.")
    else:
        cases += [
            (["--tie-alt", str(tie_step)], 0,
             f"TIE step {tie_step}",
             f"engine sampled the pin's other tied id at tied step {tie_step} — must be ADMITTED"
             f" (the id is {tie_other(tie_step)}, the pinned top-1 is {pin_tie(tie_step)['top1']})"),
            (["--break-tie", str(tie_step)], 1,
             [f"[FAIL] generation step {tie_step}: exact tie in the pin",
              "is NOT one of the tied ids"],
             f"engine sampled an UNTIED id at tied step {tie_step} — must fail loudly"),
            # D-1's regression. The old predicate consulted the tie rule whenever the PIN was tied, so a
            # step where the engine reproduced the pin's own id could still go red on a half-ULP payload
            # question (`mt32f` step 11: engine 24.391083 rounds 0x41c3, pin 0x41c2, both ids 4016).
            # This injection reproduces that shape and requires GREEN.
            (["--tie-same-ulp", str(tie_step)], 0,
             "the tie policy is not consulted",
             f"engine kept the pin's own top-1 {pin_tie(tie_step)['top1']} at tied step {tie_step} with"
             " its logit one BF16 ULP above the tied payload (the mt32f step 11 shape) — must PASS as"
             " an ordinary id match, rule not consulted"),
            # The generalised clause (`tie_verdict`'s slack branch) needs both directions aimed at the
            # SAME id set, or the admitting case proves nothing: --tie-alt above picks a tied id the
            # engine reached by rounding, this one picks a tied id the engine reached by moving the value
            # eight licensed element errors. Only the bound can tell them apart.
            (["--tie-far", str(tie_step)], 1,
             [f"[FAIL] generation step {tie_step}: exact tie in the pin",
              "NOT ADMISSIBLE: further from the pinned maximum than this step's band licenses"],
             f"engine sampled the pin's other tied id at tied step {tie_step} with its logit 8x the"
             " licensed element error above the pinned tie — must FAIL: the id is one the pin tied, but"
             " that value is not the pinned pair with a different rounding"),
        ]
    # D-8's two directions, on §06.5's first sampled token rather than on a generation step. The
    # refusing one runs on every fixture: routing `y0` through the tie policy must not have cost the
    # exact-id rule its teeth. The admitting one only exists where the pinned prefill row actually
    # ties — `chunk64b`, ids 760 and 1596 at 19.875 — and is skipped out loud elsewhere, because an
    # injection the policy can never be asked about is the vacuous negative this file keeps refusing.
    cases.append((["--break-y0"], 1,
                  ["[FAIL] the prompt's last token samples greedy_continuation[0]"],
                  "the wave's last row sampled a different first answer token — D-8 widened the tie"
                  " policy to this row without weakening the exact-id rule"))
    if pin_tie(PREFILL_ROW) is None:
        print(f"  [SKIP] --y0-alt: {fixture}'s pinned prefill row has a single winner, so there is no"
              " tie to admit at §06.5's row. `--selftest --fixture chunk64b` exercises D-8's"
              " admitting half (it is the one fixture whose `y0` pin ties).")
    else:
        cases.append((["--y0-alt"], 0, ["TIE the prefill row (§06.5's y0)"],
                      "the wave's last row sampled the pinned prefill row's OTHER tied id — must be"
                      " ADMITTED as a proved tie (D-8). The id is"
                      f" {tie_other(PREFILL_ROW, '--y0-alt')}, the pinned top-1 is"
                      f" {pin_tie(PREFILL_ROW)['top1']}"))
    # The quantization-slack rule (D-9's sized rule), in three aims of ONE perturbation: lift a pinned id
    # one BF16 step over the pin's own maximum and sample it. The lift is identical in all three, so the
    # only thing that can change the verdict is where the rule's line sits — which is what makes the pair
    # `--slack-alt` green / `--break-runnerup` red a measurement of the bound rather than a demonstration
    # that the gate can fail. Each needle also refuses a step on the wrong side of that line, so none of
    # them can decay into the ordinary wrong-id test (`prototype/mlp_gemv/results.md` again).
    slack_in, slack_out = slack_region_steps(len(man["greedy_continuation"]) - 1)
    if not slack_in:
        print(f"  [SKIP] quantization-slack injections: no step of {fixture} has a unique pinned winner"
              " sitting inside its own licensed element error. `--selftest --fixture chunk64b` is the"
              " fixture that exercises the admitting half (its steps 0, 1, 13 and 22).")
    else:
        j_in = slack_in[0]
        cases += [
            (["--slack-alt", str(j_in)], 0,
             [f"SLACK step {j_in}", "ADMITTED as tie-by-quantization-slack"],
             f"the pin's runner-up at step {j_in}, lifted one BF16 step over the pinned maximum — a step"
             " whose margin is smaller than the element error its own band licenses, so the rule must"
             " ADMIT it. A slack rule that never fires would be the rubber stamp in reverse"),
            (["--break-slack", str(j_in)], 1,
             [f"[FAIL] generation step {j_in}: the pin's winner is unique but its margin is under",
              "is NOT one of the pin's"],
             f"the same lift at the same undecidable step {j_in}, aimed at the highest-valued pinned id"
             " OUTSIDE the margin — a token the pin never tied at all, so the rule must refuse it"),
        ]
    if slack_in and slack_out is None:
        print(f"  [SKIP] --break-runnerup: every step of {fixture} with a unique winner is inside the"
              " slack region, so there is no decidable step to aim it at.")
    elif slack_out is not None:
        cases.append((["--break-runnerup", str(slack_out)], 1,
                      [f"[FAIL] generation step {slack_out} samples greedy_continuation",
                       f"[FAIL] step {slack_out}: argmax agrees with the pin"],
                      f"the same one-ULP lift at step {slack_out}, the DECIDABLE step of this fixture"
                      " closest to the line — the pin's runner-up, but here the margin clears the"
                      " licensed element error, so the pin did decide this token and the rule must"
                      " refuse it. This is the negative that fails if the bound is ever sized too"
                      " generously, which is the failure D-9 was guarding against"))
    # The route allowance has to still bite, and bite where the old flat band could not: noise sized to
    # exceed this step's own measured spread while staying inside §10.1's flat 5e-4 deficit. Aimed at the
    # tightest priced allowance in the fixture (the step where the flat floor was most nearly blind), and
    # skipped out loud if this fixture has no priced allowance or no such step — a negative every band
    # would catch is the vacuous one `prototype/mlp_gemv/results.md` names.
    if allow_step is None:
        print(f"  [SKIP] route-allowance injection: "
              + ("this is a `--gemm library` run, where the cascade floor is wider than the flat §10.1 G5"
                 " floor this needle is bounded by — every deficit it can produce is admitted by policy"
                 " here, so the allowance rule is not the cap in force. `--break-band` is the aggregate"
                 " negative that means something on this route."
                 if route_is_library() else
                 f"{REFS.name} carries no valid {ALLOWANCE_FILE}, or no step whose allowance leaves room"
                 " for a deficit under the flat §10.1 G5 floor. Price it with"
                 " engine/route_allowance.py (CPU, no device) to exercise this case."))
    else:
        cases += [
            (["--break-allow", str(allow_step)], 1,
             [f"[FAIL] step {allow_step}: cos deficit", "would have PASSED this step"],
             f"step {allow_step}'s logits deficit pushed to {SELFTEST_ALLOW_FACTOR}x its own measured"
             " route allowance while still inside the flat G5 floor the old band used — must FAIL there"
             " (this is the failure class the allowance exists to make visible)"),
        ]
    # The state fingerprints get three injections, one per §08.6 failure a moment-sum fingerprint can
    # see, each aimed at ONE row and required to name only that row. They need a fixture that carries
    # fingerprints — `dec8` predates them — so on the default fixture this is skipped out loud rather
    # than quietly passing.
    if prefill_fp("state") is None:
        print(f"  [SKIP] state-fingerprint injections: {fixture} carries no state fingerprints at"
              " all (dec8 was pinned before there were any), so section (g) has nothing to perturb."
              " Run `--selftest --fixture dec64f` (or dec64t / mt32f) to exercise it.")
    else:
        cases += [
            (["--break-state-fp", "zero:state:5"], 1,
             ["[FAIL] §08.4: no state family is still all zero at the prefill end",
              "offending: [state row 5]",
              "![state row 4]", "![state row 6]", "![conv row 5]", "![kv row 5]"],
             "GDN state row 5 zeroed at both end conditions — the §08.4 zeroed start, left behind"
             " (and at the run end, a state that never moved: the same row named twice and no other"
             " row named)"),
            (["--break-state-fp", "frozen:conv:2"], 1,
             ["[FAIL] inv 2: every conv row moved", "offending: [conv row 2]",
              "![conv row 1]", "![conv row 3]", "![state row 2]"],
             "conv window row 2 did not roll across the whole run (§08.6 inv 2, the A-04 carry) —"
             " must fail at row 2 and nowhere else"),
            (["--break-state-fp", "rewrote:kv:1"], 1,
             ["[FAIL] inv 3: the KV's already-written lanes did not move",
              "offending: [kv row 1 dn2 1]",
              "![kv row 0", "![kv row 2"],
             "KV row 1's already-written lanes moved (§08.6 inv 3, append-only storage) — the one"
             " invariant whose correct value is exactly 0.0, so it needs no tolerance to violate"),
        ]
    # Section (h) gets two, and they are aimed at opposite halves on purpose. `--break-wave` is the
    # bug this section exists to catch — a multi-row wave whose launch counters say one row at a time,
    # which is exactly what an engine whose batched kernels are never reached looks like — and the
    # assertion that matters is as much about what must stay GREEN as what must go red: the ids and the
    # logits of such a run are identical to the correct one, so a gate that only compares outputs
    # cannot fail it (the x1.000 finding in `engine/results.md`'s P4 section). `--break-ab-logits` is
    # the mirror: identical launches, one differing logit, which the launch half must not care about.
    # Without the pair, either half could be vacuous and the other would hide it.
    ab_pos = 3
    cases += [
        (["--break-wave"], 1,
         ["[FAIL] batched side: glue launches match the wave arithmetic",
          "[FAIL] batched side: proj launches match the wave arithmetic",
          "[FAIL] batched side: mlp_gateup launches match the wave arithmetic",
          "[FAIL] batched side: gdn_conv launches match the wave arithmetic",
          "[FAIL] the batched side issues ROW-BATCHED glue launches",
          "[FAIL] the batched side issues TOKEN-BATCHED MLP launches",
          "[FAIL] the batched side issues TOKEN-BATCHED projection launches",
          "[FAIL] the batched side issues TOKEN-BATCHED conv (§04.2) launches",
          "[FAIL] the batched run issues materially fewer batchable launches",
          "![FAIL] one-row side: glue launches match the wave arithmetic",
          "![FAIL] one-row side: proj launches match the wave arithmetic",
          "![FAIL] the one-row side issues ONE-ROW projection launches",
          "![FAIL] both sides sample the same token ids",
          "![FAIL] both sides write bit-identical logits",
          "![FAIL] the one-row side covers the same glue rows as the batched side",
          "![FAIL] the one-row side covers the same proj rows as the batched side",
          "![FAIL] the carried-window side"],
         "a wide wave that submitted one row per launch (P4's, P5's and P6's batched kernels never"
         " reached). Only the launch half may notice: sampled ids and logits are bit-identical either"
         " way, so the identity half must stay green, `the one-row side covers the same proj rows as"
         " the batched side` must stay green (batching moves rows between launches, it does not add or"
         " remove work), the `--wave-rows 1` side must be untouched, and the carried-window side must"
         " be unaffected — it was never asked to batch at the wide width"),
        (["--break-gdn-one-row"], 1,
         ["[FAIL] batched side: gdn_conv launches match the wave arithmetic",
          "[FAIL] batched side: gdn_gates launches match the wave arithmetic",
          "[FAIL] batched side: gdn_gated_norm launches match the wave arithmetic",
          "[FAIL] the batched side issues TOKEN-BATCHED conv (§04.2) launches",
          "[FAIL] the batched side issues TOKEN-BATCHED gates (§04.3) launches",
          "[FAIL] the batched side issues TOKEN-BATCHED gated norm (§04.5) launches",
          "![FAIL] batched side: glue launches match the wave arithmetic",
          "![FAIL] batched side: proj launches match the wave arithmetic",
          "![FAIL] batched side: mlp_gateup launches match the wave arithmetic",
          "![FAIL] the batched side issues ROW-BATCHED glue launches",
          "![FAIL] the batched side issues TOKEN-BATCHED projection launches",
          "![FAIL] the one-row side: gdn_conv launches match the wave arithmetic",
          "![FAIL] the one-row side issues ONE-ROW conv (§04.2) launches",
          "![FAIL] both sides write bit-identical logits",
          "![FAIL] both sides sample the same token ids",
          "![FAIL] the one-row side covers the same gdn_conv rows as the batched side",
          "![FAIL] the carried-window side"],
         "P6's aimed negative: the three wide GDN launches silently degenerated to one row per token"
         " while glue/proj/MLP stayed batched. It must be caught by the launch half, by the GDN checks"
         " specifically — every other family's check stays green, so none of them is quietly carrying"
         " this proof — and by nothing on the one-row side, which is correct by construction. The ids"
         " and logits of such a run are identical to the correct one, which is the whole reason this"
         " section counts submissions instead of reading outputs"),
        (["--break-attn-one-row"], 1,
         ["[FAIL] batched side: attn launches match the wave arithmetic",
          "[FAIL] the batched side issues MULTI-ROW attention launches",
          "![FAIL] batched side: glue launches match the wave arithmetic",
          "![FAIL] batched side: proj launches match the wave arithmetic",
          "![FAIL] batched side: mlp_gateup launches match the wave arithmetic",
          "![FAIL] batched side: gdn_conv launches match the wave arithmetic",
          "![FAIL] the batched side issues ROW-BATCHED glue launches",
          "![FAIL] the batched side issues TOKEN-BATCHED projection launches",
          "![FAIL] the batched side issues TOKEN-BATCHED conv (§04.2) launches",
          "![FAIL] the one-row side: attn launches match the wave arithmetic",
          "![FAIL] the one-row side issues ONE-ROW attention launches",
          "![FAIL] both sides write bit-identical logits",
          "![FAIL] both sides sample the same token ids",
          "![FAIL] the one-row side covers the same attn rows as the batched side",
          "![FAIL] the carried-window side"],
         "P8's aimed negative: the attention layer of a wide wave went back to one query row per"
         " launch while every other family stayed batched. Only the launch half may notice, and only"
         " the attention checks may say so — glue/proj/MLP/GDN stay green, so the GDN injection is not"
         " quietly carrying this proof either (§10.1's locality rule). Bits cannot see it at all:"
         " `prototype/attn_decode` §11 measures a wave of queries to be bit-identical to those same"
         " rows launched singly, which is exactly why the identity half must stay green here"),
        (["--break-ab-logits", str(ab_pos)], 1,
         ([f"[FAIL] batched and one-row keep every logits row inside the cascade band",
           "[FAIL] batched and one-row pool to a cosine inside the cascade band",
           "![FAIL] batched and one-row sample the same tokens at every row that was not a coin toss",
           "![FAIL] batched and one-row pool to a stage meanrel inside the cascade band",
           "![FAIL] the batched side issues ROW-BATCHED glue launches"]
          if route_is_library() else
          [f"[FAIL] both sides write bit-identical logits", "1 differing: ['logits%d.bin']" % ab_pos,
           "![FAIL] the batched side issues ROW-BATCHED glue launches",
           "![FAIL] the batched side issues TOKEN-BATCHED projection launches",
           "![FAIL] the batched run issues materially fewer batchable launches",
           "![FAIL] both sides sample the same token ids"]),
         f"position {ab_pos}'s logits are shifted 8 logits on the one-row side: the output comparison must"
         " bite and name that one file, while every launch count stays correct (the locality rule"
         " `prototype/mlp_gemv/results.md` demands — a negative every check catches proves nothing)."
         " Sized 8 logits rather than one byte because the row band on either library route is ~2e-03"
         " of 1-cos and a 1-ULP flip would sit inside it: the negative would have gone extinct the"
         " moment the band arrived. The row's order is untouched, so no id moves here; and because it"
         " is ONE row of a 25-row stand-in corpus it does not move the POOLED meanrel past its 3e-02"
         " stage band either (asserted green above), which is what separates a row rule from a stage"
         " rule"),
        (["--break-ab-ids", str(ab_pos)], 1,
         ([f"[FAIL] batched and one-row sample the same tokens at every row that was not a coin toss",
           "![FAIL] batched and one-row keep every logits row inside the cascade band",
           "![FAIL] batched and one-row pool to a stage meanrel inside the cascade band"]
          if route_is_library() else
          ["[FAIL] both sides sample the same token ids",
           "![FAIL] both sides write bit-identical logits",
           "![FAIL] the batched side issues ROW-BATCHED glue launches"]),
         f"position {ab_pos}'s SAMPLED id differs on the one-row side while its logits are untouched:"
         " the other half of the locality pair with --break-ab-logits above. On the hand route this is"
         " the ordinary different-ids failure; on the library route it is the flip rule being asked"
         " whether an id difference at a row with real room is admissible, and the stand-in's rows are"
         " aranges whose every gap is 1.0 — no row of this corpus is a coin toss, so it must refuse"),
        # The flip line's own edges, MEASURED rather than asserted. `--break-ab-room POS:ULP[:MAG]`
        # builds a row whose room is exactly the ULP figure it is handed (§3f of
        # prototype/gemm_algo_sweep/results.md), so these three cases say where the 1.50 ULP line sits
        # in its own unit, on both sides of it, and the room is printed inside the FAIL line so a reader
        # can check the injection is above the line instead of trusting that it is:
        #   3:2.0      2.00 ULP at this corpus's own magnitude = 0.50 logits -> RED (both readings)
        #   3:2.0:1.0  2.00 ULP at top-1 1.0 = 0.0156 logits -> RED on the ULP line, and it is the row
        #              the retired magnitude-blind 0.10 ADMITTED. The tightening half, and the only
        #              witness that the new line is not purely a relaxation.
        #   3:1.0:63   1.00 ULP at top-1 63 = 0.25 logits -> ADMITTED now, RED under the retired line:
        #              wide520f row 180's shape, reproduced on the CPU rather than waited for on device.
        (["--break-ab-room", f"{ab_pos}:2.0"], 1,
         ([f"[FAIL] batched and one-row sample the same tokens at every row that was not a coin toss",
           f"prefill row {ab_pos}: 63 vs 62 with room 2 BF16 ULP"
           f" (both sides' own gaps 0.5 / 0.5)",
           "![FAIL] batched and one-row keep every logits row inside the cascade band",
           "![FAIL] batched and one-row pool to a stage meanrel inside the cascade band",
           "![FAIL] the batched side issues ROW-BATCHED glue launches"]
          if route_is_library() else
          ["[FAIL] both sides sample the same token ids",
           "[FAIL] both sides write bit-identical logits",
           "![FAIL] the batched side issues ROW-BATCHED glue launches"]),
         f"position {ab_pos} is a near-tie 2.00 BF16 ULP wide at top-1 63 (0.50 logits) on which the two"
         " sides pick different ids: above the 1.50 ULP line, so the flip rule must refuse it and name"
         " the room in ULP. Only two lanes differ between the sides, so the row band, the pooled stage"
         " figures and every launch count stay green — the flip rule is the only thing that may notice"),
        (["--break-ab-room", f"{ab_pos}:2.0:1.0"], 1,
         ([f"[FAIL] batched and one-row sample the same tokens at every row that was not a coin toss",
           f"prefill row {ab_pos}: 63 vs 62 with room 2 BF16 ULP"
           f" (both sides' own gaps 0.015625 / 0.015625)",
           "![FAIL] batched and one-row keep every logits row inside the cascade band"]
          if route_is_library() else
          ["[FAIL] both sides sample the same token ids",
           "[FAIL] both sides write bit-identical logits"]),
         f"position {ab_pos} is the SAME 2.00 ULP room at top-1 1.0, i.e. 0.015625 logits — a gap the"
         f" retired 0.10 line admitted (0.015625 < 0.10) and the ULP line refuses. This is the case"
         " that shows the change is a trade and not a relaxation: had it stayed green, the new line"
         " would have been a pure loosening and would have needed a different justification"),
        (["--break-ab-room", f"{ab_pos}:1.0:63"],
         0 if route_is_library() else 1,
         ([f"[PASS] batched and one-row sample the same tokens at every row that was not a coin toss",
           f"prefill row {ab_pos}: batched 63 / one-row 62, room 1 BF16 ULP (< 1.50;"
           f" own gaps 0.25 / 0.25)",
           "![FAIL] batched and one-row keep every logits row inside the cascade band"]
          if route_is_library() else
          ["[FAIL] both sides sample the same token ids",
           "[FAIL] both sides write bit-identical logits"]),
         f"position {ab_pos} at 1.00 BF16 ULP of room at top-1 63 = 0.25 logits: the wide520f row-180"
         " shape (one representable step apart at a large logit), which the retired 0.10 called RED and"
         " this line admits as the coin toss the census measured it to be"),
    ]
    bad = []
    for extra, want_rc, want_in, what in cases:
        p = subprocess.run(base + ["--dry-run"] + extra, capture_output=True, text=True)
        needles = [] if want_in is None else ([want_in] if isinstance(want_in, str) else list(want_in))
        missing = []
        for needle in needles:
            # A needle starting with "!" must be ABSENT. That is what makes an injection local: "the
            # gate failed" is cheap — a broken parser fails everything — while "the gate failed at
            # exactly the row I broke" is the property worth requiring. It is the trap
            # `prototype/mlp_gemv/results.md` names: a negative every check catches proves nothing.
            absent = needle.startswith("!")
            hit = needle.lstrip("!") in p.stdout
            if hit == absent:
                missing.append(needle)
        ok = p.returncode == want_rc and not missing
        where = "" if not needles else (" — every expected FAIL line present" if not missing
                                        else " — missing: " + "; ".join(missing))
        print(f"  [{'PASS' if ok else 'FAIL'}] {what} — rc={p.returncode} want {want_rc}{where}")
        if not ok:
            print("      " + "\n      ".join(l for l in p.stdout.splitlines() if "FAIL" in l)[:400])
            bad.append(what)
    # The refusal of a stand-in injection on a real run, asserted here rather than left to the reader.
    # This one runs WITHOUT `--dry-run`, which is the point; `--no-engine` is there so that a broken guard
    # fails this case on CPU instead of silently starting a device run inside `--selftest`. With the guard
    # intact nothing after argument parsing is reached at all. Without it, the same argv runs the engine,
    # discards the flag, and prints PASS over a bug that was never injected.
    section("(h0) a stand-in injection asked of a real run is refused, not ignored")
    for extra in (["--break-ab-room", f"{ab_pos}:2.0"], ["--break-ab-logits", "2"],
                  ["--break-ab-ids", "2"]):
        p = subprocess.run(base + ["--no-engine"] + extra, capture_output=True, text=True)
        ok = p.returncode != 0 and "only the CPU stand-in" in (p.stdout + p.stderr)
        print(f"  [{'PASS' if ok else 'FAIL'}] {extra[0]} without --dry-run refuses to run the engine"
              f" and names the flag — rc={p.returncode}")
        if not ok:
            bad.append(f"stand-in injection {extra[0]} not refused on a real run")
    # The fingerprint-identity case is a property of the pins and the shared helper rather than of an
    # injection, and it is the only thing in this file that imports torch — so it runs once here, in
    # this process, instead of once per dry-run subprocess. A fixture with no fingerprint pins has
    # nothing to prove it with, and says so in the SKIP line above.
    if prefill_fp("state") is not None:
        section("(g) fp-of-pins identity, run once in the selftest process")
        before = len(fails)
        fp_helper_identity(man)
        if len(fails) > before:
            bad.append("fp-of-pins identity (section (g)'s mechanical case): " + fails[-1][:160])
    bad += selftest_declared_gaps(fixture)
    section("(k) the carried-record list's own teeth — in process, no fixture written")
    bad += selftest_non_pin_fixtures()
    section("(d2) the launch bill's own teeth — both directions, in process")
    bad += selftest_launch_bill()
    section("(h2) the flip rule's TWO room readings, on one injected row — in process")
    bad += selftest_room_readings(fixture, slack_out)
    section("(b2) the echo window's own teeth — nothing admitted, an admitted flip, and a late defect")
    bad += selftest_echo_window()
    section("(i) the prefill-row section's own negatives — synthetic rows, in process")
    row_bad = selftest_prefill_rows()
    bad += row_bad
    # The band history has to bite here too, and bite before anything runs: an unrecorded edit to `BAND`
    # must stop the gate, not merely look suspicious in review. The injection is the real thing — a
    # scratch tree whose engine/gate.py carries a looser flat floor, sitting beside the shipped ledger —
    # so this exercises the startup check in the real startup path rather than a stand-in for it
    # (`tools/band_history.py --selftest` is where the checker's own cases live).
    with tempfile.TemporaryDirectory(prefix="gate_bandhist_") as td:
        tdp = Path(td)
        for entry in ROOT.iterdir():
            if entry.name not in ("engine", "bench", ".git"):
                os.symlink(entry.resolve(), tdp / entry.name)
        (tdp / "bench").mkdir()
        for entry in (ROOT / "bench").iterdir():
            if entry.name != Path(band_history.HISTORY_REL).name:
                os.symlink(entry, tdp / "bench" / entry.name)
        shutil.copyfile(ROOT / band_history.HISTORY_REL, tdp / band_history.HISTORY_REL)
        (tdp / "engine").mkdir()
        for entry in (ROOT / "engine").iterdir():
            if entry.name != "gate.py":
                os.symlink(entry, tdp / "engine" / entry.name)
        # Loosen the flat floor — a cos FLOOR going down is the band widening — and refuse to run the
        # case if the edit found nothing to edit, so it cannot pass by editing nothing.
        src = (ROOT / "engine" / "gate.py").read_text()
        hit = re.search(r'^(\s*"cos_hard":\s*)([0-9.eE+-]+)', src, re.M)
        injected = False
        if hit:
            looser = format(float(hit.group(2)) - 0.0001, "g")
            (tdp / "engine" / "gate.py").write_text(
                src[:hit.start()] + hit.group(1) + looser + src[hit.end():])
            injected = True
        p = subprocess.run([sys.executable, str(tdp / "engine" / "gate.py"),
                            "--fixture", fixture, "--no-engine"], capture_output=True, text=True)
        needles = ["BAND HISTORY FAIL", "BAND.cos_hard", "LOOSER than recorded",
                   "band_history.py record"]
        missing = [n for n in needles if n not in p.stdout]
        ok = injected and p.returncode == 1 and not missing
        print(f"  [{'PASS' if ok else 'FAIL'}] an unrecorded band edit stops the gate before any"
              " fixture read or device work — rc=" + str(p.returncode) + " want 1"
              + ("" if not missing else " — missing: " + "; ".join(missing))
              + ("" if injected else " — the injection could not find BAND.cos_hard to loosen"))
        if not ok:
            print("      " + "\n      ".join(p.stdout.splitlines()[:6])[:400])
            bad.append("unrecorded band edit must stop the gate")
    print("SELFTEST " + ("FAIL — " + "; ".join(bad) if bad else
                        "PASS (the gate passes clean input and catches each injected bug at its"
                        " injected step; this says nothing about the engine)"))
    return 1 if bad else 0


# ---- the smoke run ---------------------------------------------------------------------------
def smoke_run(n_steps: int, args) -> int:
    """`--smoke N`: a FREE run of N generated tokens. Not a parity claim, and not a timing.

    What "smoke" has to mean to be worth a device window: it did not crash, no step produced a
    NaN or an out-of-range id, the state stayed inside what was allocated, and the run ended by
    its length budget or by the stop id — one of those two, not anything else. The forced pins
    own depth and parity; this is the question they cannot ask, because a teacher-forced chain
    never lets the engine's own divergence happen and a long context is exactly where a
    carry-over bug that only shows up past the pin lives.
    """
    man = json.loads((REFS / "manifest.json").read_text())
    prompt_ids = man["sources"][0]["ids"]
    vocab = man["text_config"]["vocab_size"]
    ceiling = lane_ceiling()
    section(f"(smoke) free run of {n_steps} generated tokens on {REFS.name}'s prompt")
    lanes = len(prompt_ids) + n_steps
    check(lanes <= ceiling, f"the planned run fits the attention stage's lane ceiling",
          f"{len(prompt_ids)} prompt + {n_steps} generated = {lanes} lanes vs"
          f" {lane_ceiling_note()}")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    for stale in work.glob("logits*.bin"):
        stale.unlink()
    cmd = [args.binary, "--model", args.model, "--ids", ",".join(map(str, prompt_ids)),
           "--steps", str(n_steps), "--out", str(work)]
    if args.smoke_no_stop:
        # Without this the run stops where the model decides to, which on this prompt is 38 tokens
        # in — and a 300-token smoke that only grew the state 38 slots would be a 38-token smoke.
        cmd.append("--no-stop")
    print("  $ " + " ".join(cmd))
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        check(False, "decode_run exits 0", f"rc={proc.returncode}: "
              + (proc.stderr.strip().splitlines() or [""])[-1])
        return report()
    check(True, "decode_run exits 0")
    run = parse_run(proc.stdout)
    prompt_steps = [s for s in run["steps"] if s["kind"] == PREFILL_KIND]
    decode_steps = [s for s in run["steps"] if s["kind"] == DECODE_KIND]
    result = run["result"]
    stopped = result.get("stopped") == "1"
    check(len(prompt_steps) == len(prompt_ids), "one step per prompt token",
          f"{len(prompt_steps)} of {len(prompt_ids)}")
    check(not [s for s in prompt_steps if s.get("stop") == "1"],
          "no prompt-phase step fired §07.5", f"{len(prompt_steps)} prompt steps")
    check(len(decode_steps) == n_steps or stopped,
          "the run ended by its length budget or by the stop id, not by anything else",
          f"{len(decode_steps)} generation steps, stopped={stopped}, RESULT={result}")
    if args.smoke_no_stop:
        check(not stopped and len(decode_steps) == n_steps,
              "--no-stop: the run went the full length and §07.5 never fired",
              f"{len(decode_steps)} steps, stopped={stopped}")
        if stopped:
            notes.append("--no-stop was passed but the run stopped anyway: the depth this smoke"
                         " claims was not exercised.")
    if stopped:
        check(decode_steps and int(decode_steps[-1]["sampled"]) == STOP_ID
              and decode_steps[-1].get("stop") == "1",
              "a stopped run stopped ON the stop id, in the generation phase",
              f"last step sampled={decode_steps[-1]['sampled'] if decode_steps else 'none'}"
              f" stop={decode_steps[-1].get('stop') if decode_steps else '-'}")
    ids = [int(s["sampled"]) for s in decode_steps]
    bad = [i for i in ids if not 0 <= i < vocab]
    check(not bad, f"every sampled id is a vocabulary id (0..{vocab - 1})", bad[:6])
    top_bad = [s["top_logit"] for s in decode_steps if not re.fullmatch(r"-?\d+(\.\d+)?([eE][-+]?\d+)?", s["top_logit"])]
    check(not top_bad, f"every step printed a finite top logit (nan/inf would land here)",
          top_bad[:4])
    # The full logits vectors are the only place a NaN can hide: a step whose argmax looks sane can
    # still be one element away from an inf. ~1 MB per step, read then deleted.
    nonfinite, missing, worst = [], [], None
    for j in range(len(decode_steps)):
        path = work / f"logits{len(prompt_steps) + j}.bin"
        if not path.exists():
            missing.append(path.name)
            continue
        eng = np.fromfile(path, dtype="<f4")
        if eng.size != vocab:
            missing.append(f"{path.name} has {eng.size} floats")
            continue
        if not np.isfinite(eng).all():
            nonfinite.append(f"step {j}: {int((~np.isfinite(eng)).sum())} non-finite")
        else:
            mx = float(np.max(np.abs(eng)))
            worst = mx if worst is None else max(worst, mx)
    check(not missing, "one full logits dump per generation step", missing[:4])
    check(not nonfinite, "no NaN or infinity in any logits vector", nonfinite[:4])
    print(f"  [info] largest |logit| seen anywhere: {worst if worst is not None else 'n/a'}.")
    cap = int(run["state"].get("cap", -1)) if run["state"] else -1
    prefix = int(result.get("prefix_len", -1))
    check(0 <= prefix <= cap <= 10 ** 9, "the state ended inside its allocation",
          f"prefix_len={prefix} cap={cap} (§08.6 invariant: P must stay within the slots)")
    written = sum(p.stat().st_size for p in work.glob("logits*.bin"))
    for p in work.glob("logits*.bin"):
        p.unlink()
    notes.append(f"smoke run dumped and then deleted {written / (1024 * 1024):.0f} MiB of logits."
                 " decode_run also printed its own ms line: a smoke run's timing is NOT a"
                 " measurement and is deliberately not quoted here (bench/performance-status.md"
                 " §6 owns what a clean window takes).")
    return report()


def main() -> int:
    global ECHO_WINDOW
    ap = argparse.ArgumentParser()
    ap.add_argument("--binary", default=str(DEFAULT_BINARY))
    ap.add_argument("--model", default=str(ROOT / "models" / "Qwen3.5-0.8B"))
    ap.add_argument("--work", default=str(WORK))
    ap.add_argument("--fixture", default="dec8",
                    help="which data/refs/<name> to gate against: dec8 (8 free steps, the engine"
                         " feeds itself) or dec64f (64 forced steps, replayed — see the header)")
    ap.add_argument("--smoke", type=int, default=0, metavar="N",
                    help="free-run N generated tokens and check only that nothing broke: no crash,"
                         " no NaN, ids in range, state inside its allocation, ended by length or"
                         " the stop id. No parity claim, no timing.")
    ap.add_argument("--smoke-no-stop", action="store_true",
                    help="with --smoke: leave §07.5 unarmed so the run really reaches N tokens"
                         " (the model stops on this prompt after ~38, so without this a 300-token"
                         " smoke is a 38-token smoke)")
    ap.add_argument("--no-engine", action="store_true",
                    help="fixture pairing + state arithmetic only; starts no binary, touches no device")
    ap.add_argument("--dry-run", action="store_true",
                    help="run every section except the engine itself, against a CPU stand-in that"
                         " replays the pins; exercises this file's parsing and metrics. NOT a result.")
    ap.add_argument("--break-ids", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: make that generation step sample a wrong id, and require"
                         " the gate to catch it there")
    ap.add_argument("--break-logits", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: swap that step's top two logits, and require the gate to"
                         " catch it there")
    ap.add_argument("--break-noise", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: put 5 % relative noise on that step's logits, which moves"
                         " cos under the floor without changing the argmax")
    ap.add_argument("--break-allow", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: size that step's noise against ITS OWN cached route"
                         " allowance — a deficit over the allowance that still fits under the flat"
                         " §10.1 G5 floor, i.e. the failure the old flat band could not see. Refuses a"
                         " step with no priced allowance, so it cannot degrade into the --break-noise"
                         " test")
    ap.add_argument("--break-prompt-stop", type=int, default=-1, metavar="PROMPT_INDEX",
                    help="with --dry-run: fire the §07.5 stop check at that prefill row and end the"
                         " run there, which is the bug the first device run had. Under the default"
                         " --prefill-logits last only the wave's LAST row has an argmax to fire on"
                         " (§06.5), so an earlier index is refused unless --prefill-logits all is")
    ap.add_argument("--break-feed", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: feed that step the wrong token, which is an off-by-one"
                         " in the replay list — the plausible way a forced driver breaks")
    ap.add_argument("--break-tie", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: at a step where the PIN is an exact tie, sample an id that"
                         " is NOT one of the tied ids, and require the gate to fail there. Refuses a"
                         " step that is not tied, so the negative cannot quietly become the"
                         " ordinary wrong-id test")
    ap.add_argument("--tie-alt", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: at a tied step, sample the pin's OTHER tied id — what the"
                         " engine really does at dec64f steps 17/18 — and require the gate to admit"
                         " it as a TIE instead of failing it")
    ap.add_argument("--tie-same-ulp", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: at a step where the PIN is tied, keep the pin's own top-1 id"
                         " (so the ids match) but put the engine's F32 logit there one BF16 ULP above"
                         " the pinned payload — the `mt32f` step 11 shape. The step must PASS: an"
                         " engine that reproduced the pin's id is not invoking the tie exception, so"
                         " the rule must not be consulted (D-1). Refuses an untied step, for the same"
                         " reason as --break-tie")
    ap.add_argument("--break-band", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: put noise on that step's logits sized 4x OUTSIDE the cap this"
                         " run is actually judged under (the allowance, the flat floor, or the cascade"
                         " floor, whichever is widest), with the argmax left alone. Unlike --break-noise"
                         " this is a negative on either route: on `--gemm library` a fixed 5 % sits"
                         " inside the cascade floor by policy, so it proves nothing there")
    ap.add_argument("--tie-far", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: at a step where the PIN is tied, sample one of the OTHER tied"
                         " ids but put the engine's logit there eight licensed element errors above the"
                         " pinned tie — too far for rounding, so the generalised tie clause must still"
                         " FAIL it (the counterpart to --tie-alt). Refuses an untied step or one with no"
                         " priced meanrel band, so it cannot degrade into the ordinary wrong-id test")
    ap.add_argument("--gemm", choices=list(GEMM_ROUTES), default="lt",
                    help="which GEMM route the run being judged uses (route (c), `decode_run --gemm`)."
                         " `lt` — the shipped default since the in-situ A/B (+11.0%% / +8.4%% whole wave,"
                         " D28) and the default `decode_run` itself now carries; `library` remains a"
                         " first-class value so the claim stays A/B-able in one binary. Either one"
                         " arms the cascade floor (CASCADE_BAND): each step's cos/meanrel cap"
                         " is floored at the measured hand<->library cascade, a flipped token is"
                         " admissible where that row's top-1/top-2 room was under"
                         f" {CASCADE_BAND['flip_room_max_ulp']:.2f} BF16 ULP at the row's own"
                         " magnitude (the smaller of the two sides' gaps,"
                         " the definition §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md gives), and section (h) judges wave-width drift against that"
                         " band instead of bit-identity. `lt` is the hipBLASLt call path"
                         " (`--gemm lt`, `prototype/gemm_algo_sweep/results.md` §3a): a different"
                         " library, so a different kernel and a different accumulation order, and it is"
                         " judged by the SAME cascade floor because the band is sized by "
                         "\"a library chose the reduction\" rather than by which library did — but it is"
                         " a separate judgment with its own log, never a re-read of a `library` run."
                         " `hand` is still fully supported and leaves"
                         " every band in this file exactly where it was before route (c) — it is the"
                         " A/B side, and the route whose bits are byte-identical to a pre-route"
                         " binary. The flag is passed to every `decode_run` this gate starts and the"
                         " run's echoed route is checked against it, so a log cannot be judged under"
                         " the wrong rule.")
    ap.add_argument("--gemm-pins", default="all",
                    help="which entries of the hipBLASLt algo-pin table the run being judged was"
                         " given, passed straight to `decode_run --gemm-pins`: `all` (the default, and"
                         " `decode_run`'s own default), `none` (the heuristic's rank-0 everywhere), or"
                         " a comma-separated list of pin names. The table itself is"
                         " `engine/kernels_gemm.cpp::algo_pins()`, which holds TWO sets: the shipped rows"
                         " adjudicated in situ (two today, `down_proj` and `deep_gate_proj+up_proj`: D28's"
                         " row, restored by D36 when §9's same-day correction refuted the reason it had"
                         " been retired, and the gateup row §9 promoted) and the CANDIDATE rows, which"
                         " bind only when named — so `all` means the shipped set and never silently"
                         " acquires a candidate. And a named subset REPLACES the set: the §9 side that"
                         " named only the gateup row left `down_proj` unpinned, and an unpinned"
                         " `down_proj` costs about a fifth of that family, so read a spec from the run's"
                         " `pin_names=` echo, never from its name — a tie may not buy a pin either. What"
                         " decides a set is not this file: it is `deep_pins.py`'s CHECK-TABLE against"
                         " `insitu_verdicts.tsv`, so a pin moves only with a measurement behind it. Naming"
                         " a candidate here is how an in-situ A/B gets gated under the existing bands."
                         " This file never copies the table's size: the run echoes"
                         " `pins=<shipped in force>/<shipped rows>`, `pin_names=<list>` and"
                         " `cand=<in force>/<candidates>`, and the checks are equalities on all three, so"
                         " `all` means \"every shipped row the binary has\" and a same-size-but-wrong set"
                         " still gets caught — a count alone cannot tell `down_proj` from"
                         " `deep_down_proj`. Named on every run and asserted against those echoes,"
                         " exactly like `--gemm`: a log that says `lt` but not which algorithm ran cannot"
                         " be cited for a pin claim either way, and D28's red `wide520f` row is what that"
                         " blindness cost. Only meaningful with `--gemm lt`, and a spec that names pins on"
                         " another route is refused rather than judged.")
    ap.add_argument("--write-provenance", action="store_true",
                    help="with a real (not --dry-run) run: write engine/numerics_provenance.json from"
                         " this run's NUMERICS line (wave-row ceiling + hipBLAS version, D-11) and"
                         " skip the comparison. Re-record only after re-reading why the numbers moved.")
    ap.add_argument("--selftest", action="store_true",
                    help="re-run this file in subprocesses: clean dry run must pass, each injected"
                         " bug must fail at its own step")
    ap.add_argument("--prefill-logits", choices=[PREFILL_LOGITS_LAST, PREFILL_LOGITS_ALL],
                    default=PREFILL_LOGITS_LAST,
                    help="which logits the prompt wave is asked for: `last` (the default, §06.5 — only"
                         " the wave's last row is argmaxed, so earlier prefill STEP lines carry"
                         " `sampled=-1`) or `all` (every row priced, which is what the wave's"
                         " byte-identity comparison against the pre-P2 per-token prompt loop runs in,"
                         " and the only shape where a prompt-phase argmax exists mid-wave)")
    ap.add_argument("--scope", choices=list(SCOPES), default="full",
                    help="what this run is adjudicating. `full` (the default) counts every check that"
                         " fails. `form` is the §16/§17 attention-FORM A/B: the same fixture run once"
                         " per --attn-tile, asking which form is right above MAXSC. Under `form`,"
                         " section (h)'s wave-width OUTPUT comparisons (batched vs one-row vs carried"
                         " — cascade band / bit-identity / cross-width ids) still run, still print, and"
                         " are still written to `wave_carry_deferred.json` beside the dumps, but they"
                         " are labelled DEFERRED and not counted, because their independent variable"
                         " is --wave-rows, not the form: §17's --attn-tile 1 run reported 26 failures"
                         " of that kind ahead of the two readings that could decide anything. Nothing"
                         " is skipped or softened — `--scope full` counts them again, and a carry check"
                         " that fails on both forms is a separate finding, visible in both runs'"
                         " DEFERRED lines and in the two sidecar files.")
    ap.add_argument("--attn-tile", choices=("0", "1"), default="1",
                    help="which attention FORM a wave body runs, passed straight to `decode_run"
                         " --attn-tile`: `1`, the default since results.md §21, is the tiled"
                         " attn_tile_wave (the shipped path, 4.03x the old form and bit-identical to it"
                         " on every above-MAXSC row of both wide anchors); `0` selects the per-(row,head)"
                         " attn_core it was proved against (prototype/attn_decode/results.md §12-§21)."
                         " Named on every"
                         " run and asserted against the run's own NUMERICS echo, because the open"
                         " question was which form is wrong above MAXSC and a judged run that cannot"
                         " say which form it ran is evidence about neither. This default and"
                         " `decode_run.cpp`'s own are moved in one commit, or a gate would silently judge"
                         " a different binary than the one the engine ships.")
    ap.add_argument("--attn-touch", choices=("0", "1"), default="1",
                    help="which SPELLING of the tiled wave's per-touch inner work runs, passed"
                         " straight to `decode_run --attn-touch`: `1`, the default, is `attn_slim_wave`"
                         " (`engine/attn_slim_kernels.h`) — the score dot's trip count made"
                         " compile-time (it was runtime only because the `NEG_DIMHALF` injection asks for"
                         " 4 of 8 dims and LLVM rolled the hot dot to match) and the warp reduce folded"
                         " by hand with its own `s_waitcnt`, 45 issue slots per (row,lane,head) touch"
                         " instead of 116, measured 1.52-1.57x on the wave and owed BIT identity, not a"
                         " band (`prototype/attn_valu_slim/results.md` §2, §7). `0` selects the"
                         " prototype's `attn_tile_wave` unchanged — the frozen side the census and the"
                         " timing baseline are taken on, and the reason both spellings stay compiled"
                         " into one engine binary. Like `--attn-tile` it is wave-only, so a decode step"
                         " is the same bits under either value, and it is asserted against the run's own"
                         " NUMERICS echo rather than inferred: a judged run that cannot say which"
                         " spelling produced its bits is evidence about neither. This default and"
                         " `decode_run.cpp`'s own move in one commit, for the same reason"
                         " `--attn-tile`'s defaults do (`engine/gate.py`'s echo check fails a binary"
                         " that does not carry the field).")
    ap.add_argument("--gdn-scan", choices=("seq", "collapse"), default="seq",
                    help="which §04.4 scan a GDN body runs, passed straight to `decode_run"
                         " --gdn-scan` (P9, §04.7, route A is progress/2026-10-14/p9-phase1-2026-10-14.md D-12):"
                         " `seq` is the shipped recurrence (one launch per token per layer) and the"
                         " form every decode pin is pinned through; `collapse` is the same recurrence"
                         " held resident in LDS, one launch per layer per body, owed BIT identity"
                         " rather than a band (prototype/gdn_collapse/collapse_mirror.py is the CPU"
                         " proof, the byte-identical-logits diff is the device one). `chunk` is gone"
                         " from these choices because the binary refuses it: D-12 killed the chunked"
                         " scan on arithmetic and §1c.1 D6 found its carry form wrong (boundary decay"
                         " as a quotient of two exps, NaN at S=0), so the gate does not offer a route"
                         " that cannot run — `decode_run --gdn-scan chunk` prints the reason. Asserted"
                         " against the run's own NUMERICS echo, for the same reason as --attn-tile: a"
                         " run that cannot say which scan it ran is evidence about neither, and a"
                         " silent fall back to `seq` would price the shipped form twice and report it"
                         " as the other one")
    ap.add_argument("--gdn-spell", choices=("ship", "unroll4", "fused"), default="fused",
                    help="which SPELLING of the sequential recurrence's state walks a `gdn_rec` launch"
                         " runs, passed straight to `decode_run --gdn-spell`"
                         " (`engine/gdn_rec_unroll4_kernels.h`, `prototype/gdn_rec_isa/results.md` §10,"
                         " D32). `fused` is the default since D51; `unroll4` shipped from D33 to D51"
                         " and is the same operations in the same order with `#pragma"
                         " unroll 4` on both walks: −117 of 1,523 slots, and owed BIT identity, not a"
                         " band. `ship` is the pre-D33 `rec_kernel`, kept as a first-class side so the"
                         " claim stays A/B-able from one binary, and `fused` is the default — the unroll4"
                         " walks with the q/k norm prologue fused into one pass over shared memory"
                         " (`engine/gdn_rec_fused_kernels.h`, −32 of 1,406 slots, also owed BIT"
                         " identity), the default since its paired in-situ A/B read 1.0645 on the"
                         " `gdn_rec` family row at 4096 lanes (bench §20)."
                         " Section (j) is what actually checks that identity, by running every side and"
                         " comparing every dumped byte against `ship` rather than by watching this"
                         " flag's own run go green (a banded check cannot tell 'identical' from 'inside"
                         " the allowance'). Named on every run and asserted against the run's own"
                         " NUMERICS echo, and `--gdn-scan collapse` refuses the pair: that scan"
                         " launches a different kernel, so a run that echoed a spelling while running"
                         " it would price one kernel twice and call the second price the win")
    ap.add_argument("--gdn-chunk", type=int, default=64,
                    help="the chunk size in tokens, passed to `decode_run --gdn-chunk` and asserted"
                         " against its echo. 64 is the value Phase 0's budget selected; 16 is"
                         " measured and rejected (per-row 1-cos 3.388e-4 > the 3.2e-4 ceiling), so"
                         " the refusal text matters more than the number")
    ap.add_argument("--break-pins", type=int, default=-1, metavar="N",
                    help="with --dry-run: echo N as the in-force pin count instead of the one this"
                         " gate's own --gemm-pins spec implies. That is the bug the check exists to"
                         " catch — a run asked for `none` that quietly ran the pin, or the reverse,"
                         " which reads as a pin verdict while pricing rank-0 — and without an"
                         " injection the stand-in could only ever agree with itself. The selftest"
                         " derives N as the wrong value for the current spec and requires the gate to"
                         " go RED")
    ap.add_argument("--break-pin-names", default="", metavar="NAMES",
                    help="with --dry-run: echo this as the run's `pin_names=` instead of the list the"
                         " spec implies. The count equality cannot see a same-size wrong set —"
                         " `down_proj` (shipped) and `deep_down_proj` (candidate) are both one pin —"
                         " and telling those two apart is the entire candidate A/B, so the name check"
                         " needs its own injection or it is decoration.")
    ap.add_argument("--break-heur-width", type=int, default=-1, metavar="N",
                    help="with --dry-run: echo N as the run's `heur_width=` instead of the shipped"
                         " `kHeuristicWidth`. A run that queried a different pool than the gate"
                         " assumes is a different config, and the only witness is that echo, so the"
                         " check on it needs an injection to be worth anything.")
    ap.add_argument("--break-cand", default="", metavar="I/T",
                    help="with --dry-run: echo this as the run's `cand=` instead of what the spec owes."
                         " The interesting injection is a nonzero candidate count under `--gemm-pins all`:"
                         " `all` meaning `shipped and nothing else` is the property that makes every"
                         " pinned-set verdict in this tree citable, so it needs an injection of its own.")
    ap.add_argument("--break-wave", action="store_true",
                    help="with --dry-run: make the wide-wave side of section (h) report one-row-per-"
                         "token launch counters, which is what a refactor that silently drops back to "
                         "per-token glue/MLP looks like from the outside. The launch half must FAIL and "
                         "the bit-identity half must stay green — the outputs cannot see this bug, "
                         "which is the entire reason section (h) counts launches")
    ap.add_argument("--break-gdn-one-row", action="store_true",
                    help="with --dry-run: the wide GDN conv/gates/gated-norm launches never happened"
                         " — the run asks for (and echoes) a wide wave, but those three families"
                         " report one row per launch while glue/proj/MLP stay batched. Only the"
                         " launch half may notice: the ids and logits are identical either way. This"
                         " is P6's addition to the pair that keeps either half of section (h) from"
                         " going vacuous, aimed at one family so the GDN checks are seen to bite")
    ap.add_argument("--break-attn-one-row", action="store_true",
                    help="with --dry-run: the attention layer of a wide wave went back to one query"
                         " row per launch (P8's staged wave form never ran) while every other family"
                         " stays batched. The wave's rows produce identical bits either way — that is"
                         " what `prototype/attn_decode` §11 proves — so only the counters can see it,"
                         " and this is the injection that shows the attention half of section (h) is"
                         " not a check that cannot fail")
    ap.add_argument("--break-ab-logits", type=int, default=-1, metavar="POS",
                    help="with --dry-run: differ run position POS's logits on the one-row side of "
                         "section (h), so the byte-comparison half is proved to bite and to name the "
                         "one file it broke (the locality rule `prototype/mlp_gemv/results.md` "
                         "insists on). Aimed at a position both sides print")
    ap.add_argument("--break-ab-ids", type=int, default=-1, metavar="POS",
                    help="with --dry-run: change the SAMPLED id at section (h) position POS on the"
                         " one-row side and leave its logits alone — the other half of the locality pair"
                         " with --break-ab-logits. On `--gemm hand` that is the ordinary "
                         "different-ids failure; on `--gemm library` it is the flip rule being asked"
                         " whether an id difference at a row with real room is admissible (it is not)")
    ap.add_argument("--break-ab-room", dest="break_ab_room", default="", metavar="POS[:ULP[:MAG]]",
                    help="with --dry-run: make section (h) position POS a controlled near-tie on which"
                         " the two sides pick different ids, with a room of exactly ULP BF16 ULP (default"
                         " 2.0, just above the 1.50 line) at a top-1 of MAG (default 63, this corpus's"
                         " own magnitude). Only the two top lanes differ between the sides, so the row"
                         " band and the pooled stage figures cannot see it and the flip rule is the only"
                         " thing that can go red. ULP and MAG together are the point: 2.0 ULP at top-1 1"
                         " is 0.0156 of a logit, a gap the retired magnitude-blind 0.10 could never have"
                         " seen, and 1.0 ULP at top-1 63 is 0.25 logits, which the old line called RED"
                         " and this one admits (the wide520f row-180 shape, on the CPU)")
    ap.add_argument("--break-y0", dest="break_y0", action="store_true",
                    help="with --dry-run: the wave's LAST prefill row samples an id the pinned"
                         " prefill row does not put at its maximum, so §06.5's first answer token is"
                         " wrong. D-8's negative: routing `y0` through the tie policy must not have"
                         " made the exact-id rule vacuous")
    ap.add_argument("--y0-alt", dest="y0_alt", action="store_true",
                    help="with --dry-run: the wave's LAST prefill row samples the pinned prefill row's"
                         " other tied id — must be ADMITTED as a TIE (D-8). Refused unless this"
                         " fixture's prefill pin really ties")
    ap.add_argument("--slack-alt", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: at a step whose pin has a unique winner whose margin is"
                         " smaller than that step's licensed element error, lift the pin's runner-up"
                         " one BF16 step over the pinned maximum and sample it — the `chunk64b` step 0"
                         " shape, and the gate must ADMIT it as tie-by-quantization-slack. Refused on a"
                         " step the pin DID decide, so it cannot become the ordinary wrong-id test")
    ap.add_argument("--break-slack", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: the same lift at the same kind of step, aimed at the"
                         " highest-valued pinned id that is NOT inside the margin — a token the pin"
                         " never tied at all. Must FAIL. Refused unless the step is inside the slack"
                         " region (the vacuous-negative trap)")
    ap.add_argument("--break-runnerup", type=int, default=-1, metavar="STEP",
                    help="with --dry-run: the same lift, the pin's runner-up, at a step whose margin is"
                         " LARGER than its licensed element error — a token that looks tied but that"
                         " the pin did decide. Must FAIL, and this is the injection that catches a slack"
                         " bound sized too generously. Refused unless the step is outside the region")
    ap.add_argument("--slack-region", action="store_true",
                    help="print this fixture's quantization-slack region — every step whose unique"
                         " pinned winner sits inside its own licensed element error, with the margin,"
                         " the bound and what set the bound — and exit. CPU-only, no engine, no device")
    ap.add_argument("--state-fp", dest="state_fp", action="store_true", default=True,
                    help="ask decode_run for its §08 state fingerprints (the default, because a"
                         " fixture that carries fingerprint pins cannot judge G5's state half"
                         " without them). Accepted explicitly for the commands in the prefill plan's"
                         " §2.6; section (g) reads whatever arrived either way")
    ap.add_argument("--no-state-fp", dest="state_fp", action="store_false",
                    help="do not ask for them: section (g) then fails loudly on a fixture that has"
                         " pins, which is the honest reading of a gate that could not see the state")
    ap.add_argument("--break-state-fp", default="", metavar="WHAT:FAMILY:ROW",
                    help="with --dry-run: corrupt one emitted state-fingerprint row and require"
                         " section (g) to fail at exactly that row. WHAT is `zero` (a state that was"
                         " never written), `frozen` (a state or window that did not move over the"
                         " whole run — §08.6 inv 1/2) or `rewrote` (already-written KV lanes moved"
                         " — inv 3). Refuses an impossible target, so it cannot become a generic"
                         " perturbation test")
    ap.add_argument("--build", action="store_true", help="run engine/build.sh first")
    args = ap.parse_args()

    # (pre) INJECTION HYGIENE. Several `--break-*` flags are read ONLY by `batched_ab_stand_in`, the CPU
    # sides section (h) compares under `--dry-run`; on a real run `run_batched_ab` supplies the sides and
    # never looks at them. Passing one to a device run would therefore buy a green verdict for a bug that
    # was never injected — the empty-must-catch-set trap, and it happened: a `--break-ab-room` sweep run
    # without `--dry-run` printed nine quiet rc=0 lines that read as "the injection did not bite". Refuse
    # the combination and say which flag carries the injection instead of ignoring it.
    stand_in_only = {"--break-ab-logits": args.break_ab_logits,
                     "--break-ab-ids": args.break_ab_ids,
                     "--break-ab-room": args.break_ab_room,
                     "--break-wave": args.break_wave,
                     "--break-gdn-one-row": args.break_gdn_one_row,
                     "--break-attn-one-row": args.break_attn_one_row}
    asked = [flag for flag, val in stand_in_only.items() if val not in ("", -1, False)]
    if asked and not args.dry_run:
        raise SystemExit(
            f"gate: {', '.join(asked)} "
            + ("is" if len(asked) == 1 else "are") + " section (h) "
            + ("injection" if len(asked) == 1 else "injections") + " that only the CPU stand-in "
            "implements (`batched_ab_stand_in`); without --dry-run this gate would run the engine, "
            "silently discard "
            + ("it" if len(asked) == 1 else "them") + ", and report PASS. Add --dry-run "
            "(`--selftest` does so itself) to judge the checker, or drop "
            + ("the flag" if len(asked) == 1 else "the flags") + " to judge the engine.")

    # (pre) BAND HISTORY — checked before a fixture is read, before a binary is started, before a
    # device is touched. `tools/band_history.py` compares every number in `BAND` and in `CASCADE_BAND`,
    # and every priced fixture's route margin, against `bench/band-history.jsonl`; an unrecorded move
    # and this gate refuses to run, because a gate must not be judged by a band nobody agreed to
    # (`AGENTS.md`: bands are derived from measured floors and never loosened to pass). It reads the
    # same small files this gate reads anyway — the policed band blocks and one margin per priced
    # fixture.
    if (rc := band_history.for_gate(ROOT)):
        return rc

    set_fixture(args.fixture)
    set_route(args.gemm)
    # After `set_route` and not before: the pin set is a hipBLASLt concept, and the pair is adjudicated
    # together so a `--gemm library --gemm-pins down_proj` invocation is refused instead of spending a
    # whole device window asserting a pin that no code path could reach.
    set_pins(args.gemm_pins)
    set_scope(args.scope)

    if args.slack_region:
        return print_slack_region()

    if args.selftest:
        return selftest(args.fixture)

    if args.build:
        subprocess.run([str(ROOT / "engine" / "build.sh")], check=True)

    if args.smoke:
        if args.dry_run or args.no_engine:
            raise SystemExit("gate: --smoke runs the engine; it is not a --dry-run or --no-engine"
                             " thing (there is no stand-in for 'nothing crashed')")
        return smoke_run(args.smoke, args)

    man = json.loads((REFS / "manifest.json").read_text())
    prompt_ids, dec_logits, dec_inputs = pairing_checks(man)
    greedy = man["greedy_continuation"]
    n_decode = len(dec_logits)
    # The row block, if this fixture carries one, is proved here (in both modes) and read again by
    # section (i) after the run. Its vocabulary width is taken from the block's own shape when the block
    # exists, so a row-pinned fixture is never sized by a decode-step pin it might not carry.
    seen_pin = first_pinned_logits(dec_logits)
    vocab_pin = (int(man["text_config"]["vocab_size"]) if seen_pin is None else int(seen_pin.size))
    rows_pin = prefill_row_pin_checks(man, len(prompt_ids), vocab_pin)

    if args.no_engine:
        state_checks(None)
        fingerprint_checks(man, n_decode)
        deep_matches_shallow(n_decode)
        state_fp_checks(man, len(prompt_ids), n_decode, 0, None)
        return report()

    if gapset:
        # A declared gap is licensed for the sections that can live with it — (0) pairs what the pin
        # carries, (f)/(f2)/(g) read fingerprints and state, none of which needs the dropped logits —
        # and NOT for the ones that judge every step of the chain. Skipping the gapped steps from (b)
        # and (e) would be easy and would end in a gate that passes because it looked at less; the
        # refusal is the difference between a weaker fixture and a quiet gate. It fires before the
        # binary starts and before `--dry-run` too, so the injection can be tested on the CPU.
        raise SystemExit(
            f"gate: {REFS.name} declares {len(gapset)} step(s) with no pinned logits"
            f" ({', '.join('decode' + str(s) for s in sorted(gapset))}), so this gate cannot run the"
            " engine against it: sections (a)/(b)/(e) judge EVERY pinned step against that step's"
            " pin, and quietly judging fewer steps is how a gate goes vacuous. `--no-engine` is the"
            " mode a gapped fixture can be checked in (pairing, state integrity, fingerprints); a"
            " device run needs those steps pinned — see `tools/reference_dump.py --logit-steps`.")

    section("(run) decode_run — one prefill wave (§06.1), then one decode_step per generated token")
    work = Path(args.work)
    work.mkdir(parents=True, exist_ok=True)
    # Section (b) reads `logits{pos}.bin` by name, so a run that stops early is judged against whatever
    # the PREVIOUS run left behind. The first device run stopped at prompt index 6, so it wrote
    # logits0..6 — and still reported only six failures, all in section (a), which is only possible if
    # section (b) found logits17..24.bin that run cannot have written (its default `--work` is the same
    # directory `--dry-run` and `--selftest` write into). Clear them: (b) must judge the run it is
    # judging, and a missing dump has to read as missing.
    for stale in work.glob("logits*.bin"):
        stale.unlink()
    # The same argument for the echo sidecar: a summary that reads "3 check(s) echoed" must describe THIS
    # run, and a file left by a previous one would say the opposite — a green run leaving a red run's
    # echoes behind is the stale-dump trap section (b) already clears logits for.
    (work / ECHO_FILE_NAME).unlink(missing_ok=True)
    # A replay above A-18's bound, a free run below it: see FREE_RUN_BOUND's comment. `--replay`
    # also tells `decode_run` not to let §07.5 end the run, because dec64f's own chain samples the
    # stop id at generation step 24 and teacher forcing feeds it straight back in.
    replay = n_decode > FREE_RUN_BOUND
    lanes = len(prompt_ids) + n_decode
    ceiling = lane_ceiling()
    check(lanes <= ceiling, "the planned run fits the attention stage's lane ceiling",
          f"{len(prompt_ids)} prompt + {n_decode} generation = {lanes} lanes vs"
          f" {lane_ceiling_note()}")
    # THE ROW PINS DRIVE THE INVOCATION, and they have to be read before the command line is built, not
    # after. `prefill_logit_rows` in the manifest is the only sensible source for "which rows must
    # exist": the pins are what section (i) judges, so a fixture that pins every prefill row requires
    # `--prefill-logits all` and the caller is not permitted to forget it. §17 is what happens when the
    # requirement is written down one statement too late: this forcing sat *below* the `cmd = [...]`
    # it now precedes, so it raised `args.prefill_logits` after `--prefill-logits last` had already been
    # appended, the engine ran `last` and wrote one row, and the gate then reported `1023 missing` as
    # though the engine had failed to produce them. Deciding the shape here, and proving afterwards that
    # the run's own echo agrees, is what makes a gap read as a gap.
    rows_required = str(man.get("prefill_logit_rows", "last")) == PREFILL_LOGITS_ALL
    if rows_required and args.prefill_logits != PREFILL_LOGITS_ALL:
        args.prefill_logits = PREFILL_LOGITS_ALL
        print("  [note] the manifest pins every prefill row (prefill_logit_rows=all), so"
              " --prefill-logits was raised to `all` BEFORE the command line was built")
    cmd = [args.binary, "--model", args.model, "--ids", ",".join(map(str, prompt_ids)),
           "--steps", str(n_decode), "--out", str(work),
           # Named on every run rather than left to the binary's default, so the echoed command says
           # which row shape the dumps were taken in: `last` is §06.5's and the one the fixtures are
           # gated in, `all` is the one the wave's byte-identity comparison against the pre-P2 per-token
           # prompt loop is made from.
           "--prefill-logits", args.prefill_logits,
           # Named on every run too, for the same reason and a stronger one: this file judges a
           # different band per route (CASCADE_BAND), so a run whose route came from the binary's own
           # default could be judged by the wrong rule. The echo is checked right after the run.
           "--gemm", GEMM_ROUTE,
           # The algo-pin set, named here and asserted against the run's own `pins=` echo for the same
           # reason as the route: which algorithm served `down_proj` at the full wave width is a runtime
           # choice now, so a log that cannot say which one ran cannot be cited for a pin claim in
           # either direction — which is how D28's one red row arrived unattributable.
           "--gemm-pins", GEMM_PINS,
           # The attention form, named for the same reason and asserted against the same kind of echo:
           # section (i) exists to decide §15's question, and a log that cannot say which form produced
           # its rows is not evidence about either of them.
           "--attn-tile", args.attn_tile,
           # The touch SPELLING, named and asserted for the identical reason: the two spellings are
           # claimed bit-identical, so a log that cannot say which one ran cannot witness that claim
           # either way.
           "--attn-touch", args.attn_touch,
           # The GDN scan, named on every run and asserted below: P9's matrix is this flag twice over
           # the ten fixtures, and a judged run that cannot say which scan produced its rows is
           # evidence about neither.
           "--gdn-scan", args.gdn_scan, "--gdn-chunk", str(args.gdn_chunk),
           # The recurrence's SPELLING, named and asserted like the four above. It is claimed
           # bit-identical, so a log that cannot say which one ran cannot witness that claim in either
           # direction — and section (j) is the A/B this line makes attributable to a spelling rather
           # than to whichever default the binary happened to carry.
           "--gdn-spell", args.gdn_spell]
    if args.state_fp:
        # On by default: a fixture that carries fingerprint pins cannot judge G5's state half without
        # them, and a section that silently did not run is the gap P1 was written to close (§2.6). The
        # cost is two D2Hs of the §08.1 tuple per run — ~26 MB each at 63–70 GB/s, on the correctness
        # path, outside every timed region — which is why plan §2.6 gives it no timing significance.
        cmd.append("--state-fp")
    if replay:
        cmd += ["--replay", ",".join(str(t) for t in greedy[:-1])]
    # The argv is the contract, not the variable that decided it. Asserted on `cmd`, the object the
    # subprocess actually receives, because that is precisely where §17's bug lived: a forcing that fires
    # after the list is built changes the gate's bookkeeping and nothing the engine sees.
    check(cmd_flag(cmd, "--prefill-logits") == args.prefill_logits,
          "the command line carries the prefill row shape this gate decided on",
          f"argv carries --prefill-logits {cmd_flag(cmd, '--prefill-logits')!r} and the gate decided"
          f" {args.prefill_logits!r}" + (" — required by this fixture's row pins"
                                          if rows_required else ""))
    feeding = ("REPLAY — the reference's own token is fed at every step, because this fixture is"
               f" deeper than A-18's {FREE_RUN_BOUND}-token bound and a self-fed run past that"
               " could not be pinned (a divergence then shows at one step, not in all of them)"
               if replay else
               "SELF-FED — the engine feeds its own sample, which is the stronger claim: the"
               " chain is re-derived rather than replayed")
    print("  feeding: " + feeding)
    if args.dry_run:
        stdout, rc = dry_run(man, prompt_ids, dec_logits, work, args.break_ids,
                             args.break_logits, args.break_noise, args.break_prompt_stop,
                             replay, args.break_feed, args.break_tie, args.tie_alt,
                             args.break_allow, args.tie_same_ulp, args.tie_far, args.break_band,
                             args.break_state_fp,
                             args.prefill_logits, break_y0=args.break_y0, y0_alt=args.y0_alt,
                             slack_alt=args.slack_alt, break_slack=args.break_slack,
                             break_runnerup=args.break_runnerup,
                             break_pins=args.break_pins,
                             break_pin_names=args.break_pin_names,
                             break_heur_width=args.break_heur_width,
                             break_cand=args.break_cand)
        notes.append("DRY RUN: the gate's own plumbing was exercised against a stand-in that replays"
                     " the pins. NOTHING ENGINE-RELATED WAS MEASURED. It exists so the first device"
                     " run is not also the first time this file parses a STEP line. Section (d) is"
                     " Python comparing with itself in this mode; only a real decode_run makes it a"
                     " cross-check.")
        notes.append(f"prefill row shape in force: --prefill-logits {args.prefill_logits}"
                     + (" — only the wave's last row is argmaxed, so the earlier prefill rows print"
                        " `sampled=-1` (§06.5)" if args.prefill_logits == PREFILL_LOGITS_LAST else
                        " — every prefill row is argmaxed, which is the shape the wave's"
                        " byte-identity comparison runs in"))
        if args.break_ids >= 0:
            notes.append(f"injected: generation step {args.break_ids} sampled the wrong id")
        if args.break_logits >= 0:
            notes.append(f"injected: generation step {args.break_logits} had its top two logits"
                         " swapped")
        if args.break_noise >= 0:
            notes.append(f"injected: generation step {args.break_noise} got 5 % relative noise"
                         " (wrong in aggregate, argmax untouched)")
        if args.break_allow >= 0:
            notes.append(f"injected: generation step {args.break_allow} got noise sized to"
                         f" {SELFTEST_ALLOW_FACTOR}x its own cached route allowance — over the"
                         " allowance, under the flat §10.1 G5 floor")
        if args.break_prompt_stop >= 0:
            notes.append(f"injected: the §07.5 stop check fired at prefill row"
                         f" {args.break_prompt_stop} and ended the run there")
        if args.break_tie >= 0:
            notes.append(f"injected: generation step {args.break_tie} sampled an id that is not one"
                         " of the tied ids the pin offers at that step")
        if args.tie_alt >= 0:
            notes.append(f"injected: generation step {args.tie_alt} sampled the pin's other tied id"
                         " — the case the tie policy exists to admit")
        if args.tie_same_ulp >= 0:
            notes.append(f"injected: generation step {args.tie_same_ulp} kept the pin's own top-1 id"
                         " with its logit one BF16 ULP above the tied payload — an ordinary match, so"
                         " the tie policy must not be consulted (D-1)")
        if args.break_state_fp:
            notes.append(f"injected: state fingerprint {args.break_state_fp} — section (g) must fail"
                         " at exactly that family and row, and nowhere else")
        if args.break_y0:
            notes.append("injected: the wave's last row sampled an id the pinned prefill row does not"
                         " put at its maximum — §06.5's first answer token is wrong (D-8's must-FAIL)")
        if args.y0_alt:
            notes.append("injected: the wave's last row sampled the pinned prefill row's OTHER tied"
                         " id — the case D-8 says the tie policy must admit")
        for needle, what in ((args.slack_alt, "lifted the pin's runner-up one BF16 step over the pinned"
                                              " maximum and sampled it at a step the pin could not"
                                              " decide — the slack rule must ADMIT it"),
                             (args.break_slack, "lifted a pinned id that is NOT inside the step's"
                                                " margin, at a step inside the slack region — the slack"
                                                " rule must refuse it"),
                             (args.break_runnerup, "lifted the pin's runner-up at a step whose margin"
                                                   " is over its licensed element error — the pin DID"
                                                   " decide this token, so the slack rule must refuse")):
            if needle >= 0:
                notes.append(f"injected: generation step {needle} {what}")
    else:
        print("  $ " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout, rc = proc.stdout, proc.returncode
        for line in proc.stdout.splitlines():
            print("      " + line)
    if rc != 0:
        check(False, "decode_run exits 0", f"rc={rc}")
        return report()
    check(True, "decode_run exits 0")
    run = parse_run(stdout)
    # Which GEMM the run says it ran. Not a formality: section (b)'s caps, section (a)'s flip rule and
    # section (h)'s band-or-bit-identity choice are all keyed on it, and a wrapper binary that quietly
    # forced the other route would otherwise be judged by the wrong band. The authoritative echo is the
    # NUMERICS line's `gemm=` field: the engine prints it in every mode, it spells the two routes exactly
    # as `--gemm` does, and D-11 already made that line provenance-bearing. The `GEMM route=` line is
    # printed and NOT asserted on — its `route=` names the library ("hipblas") as a fact about the handle,
    # so comparing that spelling with the judging route made every genuine library run read as a route
    # mismatch, which nothing caught because the dry-run stand-in printed the route word instead.
    echoed_route = run.get("numerics", {}).get("gemm")
    check(echoed_route == GEMM_ROUTE, "the run ran the GEMM route this gate is judging it under",
          ("its NUMERICS line carried no `gemm=` — the binary did not get --gemm, or it predates route"
           " (c), so nothing in this log says which band these bits are owed"
           if echoed_route is None else
           f"NUMERICS echoed gemm={echoed_route}, this gate was run with --gemm {GEMM_ROUTE}"
           f" (GEMM line says {run.get('gemm', {}).get('route', 'nothing')})"))
    # Which ALGORITHM the route was allowed to use, asserted immediately after the route itself and for
    # the same reason: `--gemm lt` names a library, `--gemm-pins` names which of its algorithms ran, and
    # section (h)'s band plus every `source=pin` claim are statements about the second one.
    gemm_pin_checks(run, stand_in=args.dry_run)
    # D-11's provenance half, immediately after the route echo and for the same reason: with two GEMMs
    # in the tree, what produced the bits is part of what they mean.
    numerics_provenance_checks(run, stand_in=args.dry_run, write=args.write_provenance)
    # Which attention FORM produced these bits, read off the run's own NUMERICS line rather than from
    # the command line this gate printed, for the same reason the route is: the two forms are allowed to
    # differ only inside §07.3's band, so a log that cannot say which one ran adjudicates nothing.
    if not args.dry_run:
        echoed_tile = run.get("numerics", {}).get("attn_tile")
        check(str(echoed_tile) == str(args.attn_tile),
              "the run ran the attention form this gate is judging it under",
              ("its NUMERICS line carried no `attn_tile=` — the binary predates --attn-tile, so no"
               " claim about either form can be made from this log" if echoed_tile is None else
               f"NUMERICS echoed attn_tile={echoed_tile}, this gate was run with --attn-tile"
               f" {args.attn_tile}"))
        # The touch SPELLING, read off the same line for the same reason (engine/attn_slim_kernels.h).
        # MANDATORY, not optional: a binary that predates `--attn-touch` cannot be judged as though it
        # carried the flag, and the two spellings are claimed bit-identical, so silence here would let a
        # pre-port binary pass as a post-port one. Same shape as the row-coverage rule: a check that
        # cannot fail is decoration.
        echoed_touch = run.get("numerics", {}).get("attn_touch")
        check(str(echoed_touch) == str(args.attn_touch),
              "the run ran the attention touch spelling this gate is judging it under",
              ("its NUMERICS line carried no `attn_touch=` — the binary predates --attn-touch, so no"
               " claim about either spelling can be made from this log" if echoed_touch is None else
               f"NUMERICS echoed attn_touch={echoed_touch}, this gate was run with --attn-touch"
               f" {args.attn_touch}"))
        # The GDN scan, read off the same line for the same reason (P9). This is also the check that
        # catches a silent fall back: a `--gdn-scan chunk` run whose launcher decided to run `seq`
        # after all would echo `seq` here and fail, rather than pricing the shipped form twice.
        echoed_scan = run.get("numerics", {}).get("gdn_scan")
        check(str(echoed_scan) == str(args.gdn_scan),
              "the run ran the GDN scan this gate is judging it under",
              ("its NUMERICS line carried no `gdn_scan=` — the binary predates --gdn-scan, so no claim"
               " about either scan can be made from this log" if echoed_scan is None else
               f"NUMERICS echoed gdn_scan={echoed_scan}, this gate was run with --gdn-scan"
               f" {args.gdn_scan}"))
        echoed_chunk = run.get("numerics", {}).get("gdn_chunk")
        check(str(echoed_chunk) == str(args.gdn_chunk),
              "the run named the chunk size this gate is judging it under",
              ("its NUMERICS line carried no `gdn_chunk=`" if echoed_chunk is None else
               f"NUMERICS echoed gdn_chunk={echoed_chunk}, this gate was run with --gdn-chunk"
               f" {args.gdn_chunk}"))
        # The recurrence SPELLING, read off the same line for the same reason (D32). Mandatory, not
        # optional: a binary that predates `--gdn-spell` cannot be judged as though it carried the knob,
        # and the two spellings are claimed bit-identical, so silence would let a pre-port build pass as
        # a post-port one — the `--attn-touch` rule, and the row-coverage rule before it.
        echoed_spell = run.get("numerics", {}).get("gdn_spell")
        check(str(echoed_spell) == str(args.gdn_spell),
              "the run ran the gdn_rec spelling this gate is judging it under",
              ("its NUMERICS line carried no `gdn_spell=` — the binary predates --gdn-spell, so no"
               " claim about either spelling can be made from this log" if echoed_spell is None else
               f"NUMERICS echoed gdn_spell={echoed_spell}, this gate was run with --gdn-spell"
               f" {args.gdn_spell}"))
    check(bool(run["steps"]), "one STEP line per token", f"{len(run['steps'])} lines")
    if not run["steps"]:
        notes.append("no STEP lines at all: decode_run printed nothing the gate could parse — read"
                     " the raw log above (a load failure or a thrown HIP error exits before any step)")
        return report()
    # ---- the coverage the row pins asked for, proved before a single number is compared ----------
    # Section (i) re-proves this and would refuse there too; it is checked HERE because a run that did
    # not deliver the rows this fixture pins has no numerics for sections (a)/(b)/(g)/(h) to judge, and
    # §17 is the exhibit for what happens when they run anyway: one coverage failure surrounded by other
    # sections' red lines, none of which say anything about the question. A coverage gap stops the gate,
    # names the side that owes the rows, and prints no verdict of any other kind. (`--dry-run` is exempt
    # for the standing reason: the stand-in dumps §06.5's row only, by design.)
    if rows_pin is not None and not args.dry_run:
        cov_ok, cov_why = prefill_row_coverage(rows_pin.shape[0], work, vocab_pin,
                                               args.prefill_logits,
                                               run.get("prefill", {}).get("logits"))
        if not check(cov_ok, "coverage: every prefill row this fixture pins was dumped, before any"
                             " numerics section runs", cov_why):
            notes.append("COVERAGE GAP: the gate stops here on purpose. Sections (a)/(b)/(g)/(h) were"
                         " not run, because a run that did not produce the rows its fixture pins cannot"
                         " answer a numerics question about them. Nothing in this output is a numerics"
                         " result.")
            return report()

    # The route allowance is read HERE, before section (a), not in (b): the quantization-slack rule needs
    # each step's meanrel cap, and (b) judges its bands from the same read. One reader, one print, and the
    # two sections cannot disagree about which rule a step is under.
    allowance = load_allowance(n_decode)

    # ---- (a) sampled ids -----------------------------------------------------------------------
    section("(a) sampled token ids — §03.4 greedy, §10.1 G7 exactness")
    # The prompt phase is `kind=prefill` now (P2's wave entry); the check names keep saying "prompt"
    # because §07.5's rule belongs to the phase, not to whichever loop ran it.
    prompt_steps = [s for s in run["steps"] if s["kind"] == PREFILL_KIND]
    decode_steps = [s for s in run["steps"] if s["kind"] == DECODE_KIND]
    # WHY the prompt came up short, not just that it did. The run log already says what ended it, and
    # reading that log on a device window is exactly the round trip these two lines exist to skip.
    if len(prompt_steps) != len(prompt_ids):
        last = run["steps"][-1] if run["steps"] else {}
        fired = last.get("stop") == "1"
        why = (f"{len(prompt_steps)} of {len(prompt_ids)} prompt tokens fed — the run ended early."
               f" Last STEP: kind={last.get('kind', 'no STEP line at all')}"
               f" index={last.get('index', '-')} sampled={last.get('sampled', '-')}"
               f" stop={last.get('stop', '-')}; RESULT: {run['result'] or 'no RESULT line'} — "
               + ("that last step's §07.5 stop check HAD fired. A stop match during the prompt phase"
                  " is not a stop: the sample is discarded and the next fed token comes from the"
                  f" fixture, and a chat prompt legitimately contains the stop id {STOP_ID} (dec8's"
                  " does, at index 8). `decode_run.cpp` owns the generation-only rule; it is the"
                  " first device run's bug, reproduced here so it cannot come back quietly"
                  if fired else
                  "no stop check on the last step, so this is an early exit or a thrown error rather"
                  " than §07.5 — read the lines above for the HIP message"))
    else:
        why = f"{len(prompt_steps)} vs {len(prompt_ids)}"
    check(len(prompt_steps) == len(prompt_ids), "one step per prompt token", why)
    # The same bug seen from the other side, as a standing tripwire: no prompt-phase step may be the one
    # that fires §07.5. The raw match is NOT the failure here — prompt-phase steps sampling the stop id is
    # expected on a chat prompt and prints `stop=0` with that id in `sampled=`. `stop=` is what
    # decode_run's human-readable [STOP] marker means (same printf condition), so this covers it too.
    prompt_stop = [s for s in prompt_steps if s.get("stop") == "1"]
    check(not prompt_stop, "no prompt-phase step fires §07.5's stop check",
          "" if not prompt_stop else
                  f"kind=prefill index={prompt_stop[0].get('index')}"
          f" sampled={prompt_stop[0].get('sampled')} ended the run; §07.5 arms only during"
          f" generation (and {STOP_ID} inside a prompt is not news — see decode_run.cpp)")
    check(len(decode_steps) == n_decode, f"{n_decode} generation steps", f"{len(decode_steps)}")
    got_ids = [int(s["sampled"]) for s in decode_steps]
    # The engine's logits vectors, read here rather than in (b): the tie policy is proved against what
    # the engine actually computed, so (a) needs them, and a tie that cannot be proved must FAIL at (a)
    # rather than be waived.
    # The vocabulary width comes from the pins when the fixture carries any, which is what `vocab_pin`
    # above already decided — and it is reused rather than recomputed so the two cannot disagree about
    # how wide a logits row is. Reading `.size` off `first_pinned_logits` unguarded was an
    # `AttributeError: 'NoneType' has no attribute 'size'` on every `--decode 0` fixture (p8/p64/p300)
    # the moment section (0) stopped crashing first: such a fixture has no per-step pin at all, and its
    # width then comes from the manifest's own `text_config.vocab_size`.
    vocab = vocab_pin
    eng_logits = read_engine_logits(work, n_decode, len(prompt_ids), vocab)
    # ---- §06.5's `y0`: the wave's last row, judged by the same predicate as every other step (D-8) --
    # The tie policy is a statement about a PIN that never chose a winner, not about a generation phase,
    # so the first sampled token gets the same test as step 40. `chunk64b` is the exhibit: its pinned
    # prefill row ties ids 760 and 1596 at 19.875 (payload 0x419f) and the engine's logit at 1596 rounds
    # back to that payload, so the bare `== greedy[0]` this line used to be was failing a row on which the
    # reference itself had not decided. It still fails closed: no engine logits vector for the row means
    # the tie cannot be proved, and `tie_verdict` says so rather than waiving the step.
    y0 = int(prompt_steps[-1]["sampled"]) if prompt_steps else None
    y0_kind, y0_ok, y0_line = (("no-tie", False, "") if y0 is None else
                               tie_verdict(PREFILL_ROW, read_one_logits(work, len(prompt_ids) - 1,
                                                                        vocab)[0], y0))
    if y0_kind == "tie":
        ties.append(f"prefill row -> {y0}")
        check(y0_ok, "the prefill row: exact tie in the pin and the engine picked a different id"
                     " — tested against the tie policy", y0_line)
    elif y0_kind == "match":
        tie_notes.append(f"prefill row -> {y0}")
        print("  [info] " + y0_line)
    y0_matches = (y0 == greedy[0]) if y0 is not None else False
    # The same room rule, on the same grounds, at §06.5's row: A2's 1694 rows are prefill rows as well
    # as generation steps, and the flip table does not distinguish them. `y0_ok` (D-8's proved tie) is
    # tried first and wins when it applies, because it is the narrower claim.
    y0_room_ok = False
    if y0 is not None and not y0_matches and not y0_ok and route_is_library():
        y0_pin = np.asarray(load_pin(logits_pin(PREFILL_ROW)), dtype=np.float32).reshape(-1)
        y0_ok_room, y0_room_line = room_verdict("the prefill row (§06.5's y0)", y0_pin,
                                                read_one_logits(work, len(prompt_ids) - 1, vocab)[0],
                                                y0)
        check(y0_ok_room, "the prefill row: the engine's first answer token differs from the pinned"
                          " one — tested against the cascade room rule", y0_room_line)
        y0_room_ok = y0_ok_room
        if y0_ok_room:
            p_id, p_val, p_gap = top2(y0_pin)
            rooms.append(f"prefill row {p_id} -> {y0} (pin room {p_gap:.6g} ="
                         f" {room_in_ulp(p_val, p_gap):.4g} BF16 ULP <"
                         f" {CASCADE_BAND['flip_room_max_ulp']:.2f})")
    check(y0_matches or y0_ok or y0_room_ok,
          "the prompt's last token samples greedy_continuation[0]",
          (f"{y0 if y0 is not None else 'no prompt step ran'} vs {greedy[0]}"
           + (" — admitted as a proved tie (D-8), see the TIE line above" if y0_ok and
              not y0_matches else "")
           + (" — admitted as a thin-room flip, see the ROOM line above" if y0_room_ok and
              not y0_matches and not y0_ok else "")))
    # One tie decision per step, in the pin's payload space (see TIE POLICY). `tie_ok` is what (b)'s
    # argmax line reuses, so the two sections cannot disagree about what a tied step means.
    tie_ok: dict[int, bool] = {}
    allow_steps = (allowance or {}).get("steps", {})
    # ---- which steps are DOWNSTREAM of an admitted divergence ---------------------------------
    # Decided ONCE, before any of the three rules prints, because the attribution needs the whole
    # picture and the three rules print from three loops: a step is downstream of a flip only if some
    # rule admitted it, and the room rule — the last of the three — would otherwise be the one admit
    # nothing could have been attributed to. `admitted` says which engine-vs-pin divergences a proved
    # rule let through and by which rule; `echo_from` is the first step whose FED token therefore
    # differs from the pin's. Every check at or after that step is an ECHO: it ran, it printed, and it
    # is not counted as a second defect, because the two sides stopped answering the same question
    # (§10.5's alignment rule). The map is asserted at the end of this section against what the three
    # loops admit on their own lines, so the two cannot drift apart silently.
    admitted = admitted_divergences(n_decode, got_ids, greedy, eng_logits, dec_logits,
                                    allow_steps, y0, y0_matches, y0_ok, y0_room_ok)
    # An echo window needs a FORK to attribute a failure to, and a replayed run has none: `--replay`
    # feeds the reference's own token at every step, so a flip admitted at step 2 changes nothing about
    # what step 3 was fed, and every band failure downstream of it is a real failure. Section (a)'s own
    # `the tokens the engine fed itself are the ones the reference fed` line is what guarantees this,
    # three lines below, so the two cannot drift: if a replay ever failed to deliver the pinned chain, it
    # is that line that goes red.
    echo_from = None if replay else echo_first_step(admitted)
    if echo_from is not None:
        cause = min(admitted)
        e_id, p_id, rule = admitted[cause]
        ECHO_WINDOW = echo_reason(cause, e_id, p_id, rule)
        notes.append(f"ECHO window in force from generation step {echo_from}: " + ECHO_WINDOW
                     + ". Nothing was widened to get here — the same three proved rules admitted one"
                        " token, and this attribution only moves where the downstream failures are"
                        " counted. Section (b2) judges the same steps at identical conditioning, with"
                        " no echo window.")
        print("  [echo] " + ECHO_WINDOW)
    for j in range(min(n_decode, len(got_ids))):
        kind, admissible, line = tie_verdict(j, eng_logits[j][0], got_ids[j])
        if kind == "no-tie":
            continue                       # the ordinary case: the pin has a single winner here
        if kind == "match":
            # D-1: engine and pin picked the same id, so G7 is satisfied outright and there is no
            # exception to grant. The line is printed so a reader can still see the pin was tied here,
            # but nothing is armed and nothing is counted as a use of the policy.
            tie_notes.append(f"step {j} -> {got_ids[j]}")
            print("  [info] " + line)
            continue
        ties.append(f"step {j} -> {got_ids[j]}")
        with echo(ECHO_WINDOW, on=echo_from is not None and j >= echo_from):
            check(admissible,
                  f"generation step {j}: exact tie in the pin and the engine picked a different id"
                  " — tested against the tie policy", line)
        tie_ok[j] = admissible
    # One slack decision per step, on the pin's VALUES this time (see QUANTIZATION SLACK). The tie
    # policy already had its turn, and `step_slack` returns None on the steps it owns — a pin with no
    # unique winner is D-1's question and it has a stricter answer there, so this rule narrows to the
    # steps next door rather than overlapping. D-1's scope rule applies here too, for the same reason:
    # a step where the engine reproduced the pin's own id is an ordinary match and consults nothing,
    # even when the step sits inside the region — it gets a note, not a counted use.
    slack_ok: dict[int, bool] = {}
    for j in range(min(n_decode, len(got_ids))):
        sl = step_slack(j, dec_logits[j], allow_steps.get(j))
        if sl is None or sl["margin"] >= sl["bound"]:
            continue           # D-1's case, no band to read, or a margin the pin DID decide — in which
                               # case the ordinary exact-id rule below is the whole rule, and the honest
                               # place for this step to fail is there, not under a slack check name
        if got_ids[j] == sl["top1"]:
            if sl["margin"] < sl["bound"]:
                slack_notes.append(f"step {j} -> {got_ids[j]} (margin {sl['margin']:.6g}, bound"
                                   f" {sl['bound']:.6g})")
                print("  [info] " + sl["numbers"] + f" — and the engine reproduced the pin's own id"
                      f" {got_ids[j]}, so no exception is consulted here (D-1's scope rule: the rule"
                      " is an exception, and nothing is counted)")
            continue
        ok, line = slack_verdict(sl, eng_logits[j][0], got_ids[j])
        with echo(ECHO_WINDOW, on=echo_from is not None and j >= echo_from):
            check(ok,
                  f"generation step {j}: the pin's winner is unique but its margin is under the"
                  " element error this step's band licenses — tested against the quantization-slack"
                  " rule", line)
        if ok:
            # Only ADMITTED steps are counted and named: a refused one is a FAIL line that says NOT
            # ADMISSIBLE, and putting it in the same list as the admitted ones would make the summary
            # claim a use of the rule that this run actually refused.
            slack_ok[j] = True
            slack.append(f"step {j} -> {got_ids[j]} (pin margin {sl['margin']:.6g}"
                         f" = {sl['ulps']:.1f} ULP vs bound {sl['bound']:.6g})")
    # The cascade room rule, armed on the library route alone. A flipped token there is admissible
    # exactly where the hand route's own decision was a thin one: `room` is the PIN's top-1/top-2 gap —
    # the hand route reproduces the pin's ids, which is what that route's green gate means, so the pin's
    # room IS the hand route's room — and §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md measured 24 flips in 1694 hand<->library rows with every
    # one of them under 0.10 of it and none of the 1566 rows at or above it flipping. At or over the line
    # a flip is RED, which is the half that stops this being a shrug. `--gemm hand` never comes here: its
    # wave width cannot move a bit, so a flipped id on that route is a bug and the exact-id rule is the
    # whole rule.
    room_ok: dict[int, bool] = {}
    if route_is_library():
        for j in range(min(n_decode, len(got_ids))):
            if got_ids[j] == greedy[j + 1] or tie_ok.get(j) or slack_ok.get(j):
                continue      # not a flip, or a step an earlier exception already decided
            ok, line = room_verdict(f"step {j}", dec_logits[j], eng_logits[j][0], got_ids[j])
            with echo(ECHO_WINDOW, on=echo_from is not None and j >= echo_from):
                check(ok,
                      f"generation step {j}: the engine's id differs from the pin's at a step neither"
                      " the tie policy nor the slack rule covers — tested against the cascade room"
                      " rule", line)
            if ok:
                room_ok[j] = True
                p_id, p_val, p_gap = top2(dec_logits[j])
                rooms.append(f"step {j} {p_id} -> {got_ids[j]} (pin room {p_gap:.6g} ="
                             f" {room_in_ulp(p_val, p_gap):.4g} BF16 ULP <"
                             f" {CASCADE_BAND['flip_room_max_ulp']:.2f})")
    for j, (g, w) in enumerate(zip(got_ids, greedy[1:])):
        by_tie = g != w and tie_ok.get(j, False)
        by_slack = g != w and slack_ok.get(j, False)
        by_room = g != w and room_ok.get(j, False)
        with echo(ECHO_WINDOW, on=echo_from is not None and j >= echo_from):
            check(g == w or by_tie or by_slack or by_room,
                  f"generation step {j} samples greedy_continuation[{j + 1}]",
                  f"{g} vs {w}" + (" — admitted as a proved tie, see the TIE line above" if by_tie
                                   else " — admitted as tie-by-quantization-slack, see the SLACK line"
                                        " above" if by_slack else
                                   " — admitted as a thin-room flip, see the ROOM line above" if by_room
                                   else ""))
    # The two halves of the attribution must be the same set. `admitted_divergences` decided the echo
    # window before any of these three rules printed, using the same predicates and the same guards; if
    # a guard is ever added in one place and not the other, this line says so instead of letting a step
    # be attributed to a rule that refused it (or the reverse: a real failure silently echoed).
    loop_admitted = {j for j in tie_ok if tie_ok[j]} | set(slack_ok) | set(room_ok)
    if (y0_ok or y0_room_ok) and not y0_matches:
        loop_admitted.add(PREFILL_ROW)
    check(set(admitted) == loop_admitted,
          "the echo window was attributed to exactly the steps the three rules admitted on their own"
          " lines — one predicate decides an admission, not two",
          f"attributed {sorted(admitted)} vs admitted on the lines above {sorted(loop_admitted)}")
    unresolved = [j for j, (g, w) in enumerate(zip(got_ids, greedy[1:]))
                  if g != w and not tie_ok.get(j, False) and not slack_ok.get(j, False)
                  and not room_ok.get(j, False)]
    # The aggregate line follows the steps it aggregates. If every unmatched step is downstream of an
    # admitted flip there is nothing left for THIS line to fail on; leaving it red would be the storm
    # talking again, and leaving it silently green would hide the count, so it prints as an ECHO.
    hard_unresolved = [j for j in unresolved
                       if not (echo_from is not None and j >= echo_from)]
    with echo(ECHO_WINDOW, on=bool(unresolved) and not hard_unresolved):
        check(not hard_unresolved,
              f"all {n_decode} generated ids match, in order — exactly, or as a tie, a quantization"
              " slack, or a thin-room flip proved from the pin",
              f"{len(tie_ok)} step(s) resolved by the tie policy"
              f" ({', '.join(sorted(ties)) if ties else 'none'})"
              + (f"; {len(slack)} admitted by quantization slack"
                 f" ({'; '.join(slack) if slack else 'none'})" if slack_ok else "")
              + (f"; {len(rooms)} admitted as thin-room flips"
                 f" ({'; '.join(rooms) if rooms else 'none'})" if room_ok else "")
              + (f" | unmatched steps {unresolved}, all of them downstream of the admitted flip at"
                 f" step {min(admitted)} — see the [ECHO] lines" if unresolved and
                    not hard_unresolved else
                    f" | unmatched steps {unresolved}")
              + f" | engine {got_ids} / pinned {greedy[1:]}")
    fed = [int(s["input"]) for s in decode_steps]
    # The fed-chain line is the statement of the fork itself, so it is the one line that cannot be
    # dropped: it is where the log says which step the engine took its own road. Same rule as above —
    # every differing index downstream of the admitted flip is the flip's own consequence.
    fed_diff = [j for j, (f, w) in enumerate(zip(fed, greedy[:-1])) if f != w]
    with echo(ECHO_WINDOW, on=bool(fed_diff) and not
              [j for j in fed_diff if not (echo_from is not None and j >= echo_from)]):
        check(fed == greedy[:-1],
              "the tokens the engine fed itself are the ones the reference fed (the engine feeds"
              " itself, so this is the chain being re-derived, not replayed)",
              f"{fed}" + (f" | differs from the pinned chain at generation step(s) {fed_diff}, from"
                          " the admitted flip onward" if fed_diff else ""))
    if replay:
        # Same check, different meaning, so it is said out loud: here the fed list is what THIS gate
        # handed `decode_run --replay`, so it proves the delivery. What proves the model is the
        # `sampled=` values above — the engine's own argmax at each step, compared with the
        # reference's and deliberately not fed back.
        check(not [s for s in decode_steps if s.get("stop") == "1"],
              "no generation step ended a replay run (§07.5 is a free-run rule)",
              f"{len(decode_steps)} replayed steps")
        where = [j for j, s in enumerate(decode_steps) if int(s["sampled"]) == STOP_ID]
        print(f"  [info] the pinned chain samples the stop id {STOP_ID} at generation step(s)"
              f" {where if where else 'none'}; teacher forcing feeds that token straight back in,"
              " so the engine prints it and keeps going — that is why a replay does not arm §07.5.")
    positions = [int(s["pos_before"]) for s in decode_steps]
    check(positions == list(range(len(prompt_ids), len(prompt_ids) + n_decode)),
          f"positions run {len(prompt_ids)}..{len(prompt_ids) + n_decode - 1} (§02.4's P)",
          f"{positions}")
    if stdout.count("[STOP]") and greedy[-1] not in man["special_ids"].values():
        notes.append(f"engine stopped on id {STOP_ID} in the generation phase but the manifest's"
                     " continuation does not")

    # ---- (b) logits ------------------------------------------------------------------------------
    section("(b) logits vs the pinned dumps — the rule in force is named on every step")
    print(f"  route in force: --gemm {GEMM_ROUTE}."
          + ("" if not route_is_library() else
             f" Flip line in force, per pairing — cross-width (two engine sides, no reference):"
             f" {flip_room_line('cross-width')} | reference (pin vs engine):"
             f" {flip_room_line('reference')}.")
          + (f" Algo pins in force: --gemm-pins {GEMM_PINS}, the run echoed"
             f" pins={run.get('gemm', {}).get('pins', '?')}"
             f" pin_names={run.get('gemm', {}).get('pin_names', '?')}"
             f" cand={run.get('gemm', {}).get('cand', '?')} (all three asserted above) — the pinned algo"
             " and the heuristic's rank-0 are different kernels, and coverage is the axis that bites: an"
             " UNpinned shape costs about a fifth of its family once a neighbour changes (D36, on §9's"
             " correction), so a wave-width verdict here is a verdict about THIS pin set, named above."
             if GEMM_ROUTE == "lt" else "")
          + (" The cascade floor is ARMED: no step is judged by a cap tighter than the measured"
             f" hand<->library cascade (cos deficit {CASCADE_BAND['row_1cos_max']:.0e}, meanrel"
             f" {CASCADE_BAND['stage_meanrel_max']:.0%}), because a route whose dots reassociate cannot"
             " be held to a smaller deficit than its own reassociation (progress/2026-10-14/route-c-audit-2026-10-14.md"
             " §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md), and the per-matmul floor behind it is 1 ULP at every width (A1). Each step line"
             " names the floor when it is what made the cap wider." if route_is_library() else
             " The hand route is judged by exactly the caps it has always been judged by; the cascade"
             " floor is not in force."))
    print(f"  where an allowance is priced, the rule is cos deficit <= margin x the step's measured"
          f" route spread and meanrel <= margin_rel x it, and the margins are THIS FIXTURE's own — read"
          f" from its own {ALLOWANCE_FILE} and derived from its own steps only, so pricing another"
          f" fixture cannot move them (D-2; the fallback in BAND is"
          f" {BAND['route_margin']:.2f} / {BAND['route_margin_rel']:.2f}, used only by a file priced"
          f" before per-fixture keying). Where nothing is priced, the flat published floors apply:"
          f" cos >= {BAND['cos_hard']} (G5 incremental==batch), meanrel <="
          f" {BAND['meanrel_hard']:.0%} (§10.2 cascade ceiling).")
    print(f"  reported: cos vs {BAND['cos_report']} (G6 decode logits); ndiff and max ULP are "
          "reported but not gated (§10.1 F1–F4: a chained dot's legal ndiff is 29–47 %, and a "
          "max-ULP bound is blind to a whole-tensor 1-ULP shift)")
    per_step: list = [None] * n_decode
    under_g6: list = []
    ratios: list = []            # (engine deficit / that step's route spread), for the summary line
    for j in range(n_decode):
        path = work / f"logits{len(prompt_ids) + j}.bin"
        eng, why = eng_logits[j]
        if eng is None:
            check(False, f"engine wrote {path.name}", why + " — check --out")
            continue
        ref = dec_logits[j]
        if eng.shape != ref.shape:
            check(False, f"step {j}: vocabulary length", f"{eng.shape} vs {ref.shape}")
            continue
        m = metrics(eng, ref)
        deficit = 1.0 - m["cos"]
        gap_x = ("exact" if m["margin"] == float("inf") else f"{m['margin']:.0f}×")
        cos_cap, rel_cap, rule_line = rule_for_step(allowance, j)
        if allowance is not None:
            ratios.append((j, deficit / float(allowance["steps"][j]["route_one_cos"])))
        argmax_ok = argmax_low(eng) == argmax_low(ref)
        fed = int(decode_steps[j]["input"]) if j < len(decode_steps) else int(dec_inputs[j][0])
        sampled = (int(decode_steps[j]["sampled"]) if j < len(decode_steps)
                   else argmax_low(ref))
        per_step[j] = {"cos": m["cos"], "meanrel": m["meanrel"], "ndiff": m["ndiff"],
                       "n": ref.size, "argmax_ok": argmax_ok, "fed": fed, "sampled": sampled}
        # `deficit` is printed at full precision because it is what `engine/route_allowance.py` re-derives
        # the margins from; the 6-decimal `cos` is not enough (at a 2.3e-4 deficit it is ±0.4 %).
        check(True, f"step {j}: metrics",
              f"cos {m['cos']:.6f} deficit {deficit:.6e} meanrel {m['meanrel']:.4%}"
              f" ndiff {m['ndiff']}/{ref.size}"
              f" maxabs {m['maxabs']:.4g} | ref top1−top2 gap {m['gap12']:.3f}"
              f" = {gap_x} the largest element error | rule: {rule_line}")
        if allowance is None:
            with echo(ECHO_WINDOW, on=echo_from is not None and j >= echo_from):
                check(deficit <= cos_cap, f"step {j}: cos >= {BAND['cos_hard']} (§10.1 G5)")
                check(m["meanrel"] <= rel_cap,
                      f"step {j}: meanrel <= {BAND['meanrel_hard']:.0%} (§10.2)")
        else:
            ok_cos = deficit <= cos_cap
            with echo(ECHO_WINDOW, on=echo_from is not None and j >= echo_from):
                check(ok_cos,
                      f"step {j}: cos deficit {deficit:.4e} inside the route allowance"
                      f" {cos_cap:.4e}",
                      "" if ok_cos else _counterfactual(deficit, 1.0 - BAND["cos_hard"],
                                                        "§10.1 G5 floor"))
            ok_rel = m["meanrel"] <= rel_cap
            with echo(ECHO_WINDOW, on=echo_from is not None and j >= echo_from):
                check(ok_rel,
                      f"step {j}: meanrel {m['meanrel']:.4%} inside the route allowance"
                      f" {rel_cap:.4%}",
                      "" if ok_rel else _counterfactual(m["meanrel"], BAND["meanrel_hard"],
                                                        "§10.2 ceiling"))
        with echo(ECHO_WINDOW, on=echo_from is not None and j >= echo_from):
            check(argmax_ok or tie_ok.get(j, False) or slack_ok.get(j, False) or room_ok.get(j, False),
                  f"step {j}: argmax agrees with the pin",
                  f"{argmax_low(eng)}" + ("" if argmax_ok else
                                          (" — but the pin is tied here and the engine's id is one of"
                                           " the tied ids, so the tie policy admits it"
                                           if tie_ok.get(j, False) else
                                           " — but this is the step the quantization-slack rule"
                                           " admits: "
                                           + step_slack(j, ref, allow_steps.get(j))["numbers"]
                                           if slack_ok.get(j, False) else
                                           " — but this is the thin-room flip the cascade rule"
                                           " admits: "
                                           + room_verdict(f"step {j}", ref, eng, argmax_low(eng))[1]
                                           if room_ok.get(j, False)
                                           else decision_margin(ref, eng, m))))
        if m["cos"] < BAND["cos_report"]:
            under_g6.append(f"{j}:{m['cos']:.6f}")
    if ratios:
        worst_j, worst_r = max(ratios, key=lambda t: t[1])
        tighter = sum(1 for j in range(n_decode)
                      if allowance["steps"][j]["allowance_one_cos"] <= 1 - BAND["cos_hard"])
        notes.append(f"all {len(ratios)} of {n_decode} steps judged by the route allowance; the"
                     f" allowance is TIGHTER than the flat §10.1 G5 floor at {tighter} of them. Worst"
                     f" step {worst_j}: deficit = {worst_r:.3f}x its own measured route spread"
                     f" (this fixture's margin in force {allowance['margin']:.2f}, derived as the max"
                     f" ratio inside {REFS.name} alone — engine/route_allowance.py; no other fixture"
                     " can move it, and no step of this fixture sits outside it by construction).")
    if under_g6:
        notes.append(f"{len(under_g6)} of {n_decode} steps sit under G6's decode-logits figure "
                     f"{BAND['cos_report']} (step:cos — " + ", ".join(under_g6[:12])
                     + (" …" if len(under_g6) > 12 else "") + "). G6 is ADVISORY here by decision,"
                     " not an open question: an end-to-end token carries the whole 24-layer"
                     " cascade, so it is not judged by a figure belonging to a narrower"
                     " comparison — see the BAND comment above and the two measured floors in"
                     " engine/results.md: last-op reassociation 0.99999862"
                     " (engine/logits_floor.py) and end-to-end two-legal-routes 1-cos"
                     " 4.7e-5..1.9e-4 (engine/route_price.py).")

    state_checks(run["state"])
    curve_checks(per_step, n_decode)
    # (b2) runs the SAME fixture a second time with the reference's own tokens fed at every step, so a
    # defect that the echo window above legitimately stopped counting is still judged, at identical
    # conditioning, by the same bands. It costs one extra engine run on a free-running fixture and none
    # on a replayed one, where (a)/(b) were already teacher-forced.
    teacher_forced_checks(args, man, prompt_ids, dec_logits, greedy, n_decode, allowance, work,
                          vocab, replay)
    if rows_pin is not None:
        if args.dry_run:
            notes.append("section (i) did NOT run in this dry run: the stand-in dumps only §06.5's"
                         f" row, and a {rows_pin.shape[0]}-row stand-in dump would be a disk"
                         " benchmark rather than a cross-check. The section's own negatives are"
                         " `selftest_prefill_rows`, which --selftest runs in process, and they include"
                         " the coverage proof `prefill_row_coverage` that a real run passes through "
                         " here — so the gap §17 fell into is tested at vocabulary 8 instead.")
        else:
            prefill_row_checks(work, vocab_pin, rows_pin,
                               args.prefill_logits, run.get("prefill", {}).get("logits"))
    fingerprint_checks(man, n_decode)
    deep_matches_shallow(n_decode)
    state_fp_checks(man, len(prompt_ids), n_decode, len(decode_steps), run["statefp"],
                    stand_in=args.dry_run)

    # ---- (d) launch plan ------------------------------------------------------------------------
    section("(d) the launch plan — §07's structure, counted in Python")
    cfg = man["text_config"]
    n_layers = cfg["num_hidden_layers"]
    n_gdn = sum(1 for t in cfg["layer_types"] if t == "linear_attention")
    n_attn = sum(1 for t in cfg["layer_types"] if t == "full_attention")
    want = launch_plan(n_layers, n_gdn, n_attn)
    got = {k: int(v) for k, v in run["launch"].items()}
    for name, ok, detail in launch_plan_verdicts(want, got):
        check(ok, name, detail)
    check(want["glue"] == 49, "glue is the 49 `prototype/glue/run.py` pins its token plan against",
          f"{want['glue']}")
    check(n_gdn + n_attn == n_layers and n_attn == 6,
          "layer routing read from the manifest matches §03.5", f"{n_gdn}+{n_attn} of {n_layers}")
    if "ms_per_step" in run["result"]:
        notes.append(f"decode_run reported {run['result']['ms_per_step']} ms/step. Timing from a"
                     " gate run is a by-product, not a measurement: see bench/performance-status.md"
                     " §6 for what a clean window takes.")

    # ---- (h) the batched-path proof --------------------------------------------------------------
    # Runs on every device gate, because that is the only way a later refactor that quietly drops back
    # to per-row launches fails something. The counters come from the launchers (`engine/launch_stats.h`)
    # and the sides differ only in `--wave-rows`, so this is the one place in the tree that can tell
    # "batching cost nothing" from "batching never ran" — the outputs cannot.
    counts = {"glue": want["glue"], "proj": want["proj"],
              "mlp_gateup": want["mlp"] // 2, "mlp_down": want["mlp"] // 2,
              # P6's three token-parallel GDN sites. Each bills ONE launch per GDN layer per body at
              # any width — the same per-body shape as glue/proj/MLP — so `wave_expect` can price a
              # wave of them unchanged. `gdn_rec` (§04.4's scan) is absent on purpose: its launcher has
              # no width to report, and a family that can only ever read rows=1 is a check that cannot
              # fail (`engine/launch_stats.h` says the same).
              "gdn_conv": want["gdn"] // 4, "gdn_gates": want["gdn"] // 4,
              "gdn_gated_norm": want["gdn"] // 4,
              # P8's attention site. One per attention layer per body at any width — the same per-body
              # shape as the four above, which is why `wave_expect` prices it unchanged — where the body
              # is a wave of n rows or a single generation step. It counts SITES, not the 1-3 kernels a
              # site submits; `engine/launch_stats.h` explains why counting kernels would make the
              # expected number a function of context length instead of of wave width.
              "attn": want["attn"]}
    wave_rows = wave_row_ceiling()
    # P6's third side: a prompt cut into waves of 5, so a 46-token prompt crosses nine interior wave
    # boundaries and every one of them has to hand the conv window to the next. 5 is small enough to
    # force that on every shipped fixture and is not a divisor of any of their lengths (64, 27, 46,
    # 128, 300) — a side that divided evenly would put the prompt end on a boundary and test the case
    # the wide run already tests.
    carry_rows = 5
    # BOTH RUN-VS-RUN SECTIONS ARE TEACHER-FORCED WHENEVER THE FIXTURE HAS A DECODE PHASE, and the reason
    # is the same in (h) and (j): their independent variable is a width or a spelling, and on a
    # free-running fixture the two sides can each take their own thin-margin token and fork — after which
    # one side is answering a different prompt than the other and every row downstream compares two
    # different questions. `prefix24dec8 --gemm lt --attn-tile 1` is the exhibit: the batched side took
    # the coin flip the slack rule admits at step 2, the one-row side did not, and four cross-width lines
    # went red on rows that were no longer comparable (worst per-row 1-cos 1.79e-01 against a 2.75e-03
    # line, which is a fork, not a route). Replaying both sides with the reference's own tokens — the same
    # `--replay` the deep fixtures already run at — makes width the only difference, and the replayed pair
    # measures 9.4e-03 pooled meanrel and 4.5e-04 worst-row 1-cos where the forked pair measured
    # 4.6e-02 and 1.79e-01. Nothing that was being proved is lost: a width or spelling difference that
    # changes a decision still shows as a RED flip at that row, because both sides are now fed the same
    # tokens and have to agree on them. A fixture with no decode phase gets nothing: `--replay` with an
    # empty list is not a thing.
    ab_replay = replay or n_decode >= 1
    if args.dry_run:
        sides = batched_ab_stand_in(man, prompt_ids, dec_logits, counts, work, wave_rows,
                                    args.break_wave, args.break_ab_logits, args.break_ab_ids,
                                    carry_rows,
                                    args.break_gdn_one_row, args.break_attn_one_row,
                                    args.break_ab_room)
    else:
        sides = run_batched_ab(args, prompt_ids, n_decode, greedy, ab_replay, work, wave_rows,
                               carry_rows)
    batched_proof_checks(sides, counts, len(prompt_ids), n_decode, wave_rows,
                         stand_in=args.dry_run)
    # (j) the `gdn_rec` spelling A/B, and it runs here rather than next to section (b) because it is a
    # RUN-vs-RUN comparison like (h), not a run-vs-pin one. Every spelling of the recurrence is in this
    # binary whatever the flags say, so the sides are launches of one build and the claim under
    # test — D32's and D50's bit-identity — is checked rather than assumed.
    section("(j) the gdn_rec spellings — bit-identical, all run in this gate (§04.4, D32, D50)")
    spell_sides = (gdn_spell_stand_in(work) if args.dry_run else
                   run_gdn_spell_ab(args, prompt_ids, n_decode, greedy, ab_replay, work))
    gdn_spell_checks(spell_sides, stand_in=args.dry_run)
    # The deferred half, written where the two forms' dumps already are. `--scope form` produces one of
    # these per run, so the question "did the wave carry fail identically on both forms?" is a diff of
    # two small files rather than a reading of two long logs — which is how a carry failure stays a
    # separate finding instead of becoming a rumour about whichever form ran last.
    if deferred_records:
        record = work / "wave_carry_deferred.json"
        record.write_text(json.dumps({
            "fixture": REFS.name,
            "scope": SCOPE,
            "attn_tile": args.attn_tile,
            "gdn_scan": args.gdn_scan,
            "gdn_chunk": args.gdn_chunk,
            "gemm": GEMM_ROUTE,
            # Which algorithms that route was allowed to use. The sidecar exists so two runs can be
            # diffed without reading two logs, and a diff of two `lt` runs taken under different pin
            # sets would otherwise be a diff of two different numerics.
            "gemm_pins": GEMM_PINS,
            "reason": WAVE_CARRY_REASON,
            "n_checks": len(deferred_records),
            "n_failed": sum(1 for r in deferred_records if not r["ok"]),
            "checks": deferred_records,
        }, indent=1) + "\n")
        notes.append(f"{len(deferred)} wave-width check(s) DEFERRED under --scope {SCOPE}"
                     f" (of {len(deferred_records)} run) — attributed to neither attention form, and"
                     f" recorded with every pass beside them in {record.name}. Re-run with"
                     " `--scope full` to have them counted; compare the two forms' files to see whether"
                     " a carry failure is a form finding or a wave-carry finding.")
    # The echoed half, beside it, for the same reason: the question "was that a real defect or the"
    # downstream consequence of an admitted coin flip?" should be answerable from a small file, and a
    # deferral that only exists in a long log is a deferral nobody can audit.
    if echo_records:
        record = work / ECHO_FILE_NAME
        cause = min(admitted) if admitted else None
        record.write_text(json.dumps({
            "fixture": REFS.name,
            "gemm": GEMM_ROUTE,
            "gemm_pins": GEMM_PINS,
            "attn_tile": args.attn_tile,
            "gdn_scan": args.gdn_scan,
            "gdn_chunk": args.gdn_chunk,
            "free_running": not replay,
            # The flip every echoed line is attributed to, in the engine's own words and the pin's: which
            # step, which token each side took, and which proved rule admitted it. A reader of the sidecar
            # must be able to re-check the attribution without re-reading the log.
            "admitted_step": cause,
            "admitted_as": (admitted[cause][2] if cause is not None else None),
            "engine_token": (admitted[cause][0] if cause is not None else None),
            "pin_token": (admitted[cause][1] if cause is not None else None),
            "admitted_steps": sorted(admitted),
            "echo_from": echo_from,
            "reason": ECHO_WINDOW,
            "n_checks": len(echo_records),
            "n_failed": sum(1 for r in echo_records if not r["ok"]),
            "checks": echo_records,
        }, indent=1) + "\n")
        notes.append(f"{len(echoes)} check(s) ECHO of the admitted flip at"
                     f" {'the prefill row' if cause == PREFILL_ROW else 'step ' + str(cause)}"
                     f" (of {len(echo_records)} judged from generation step {echo_from} on) — they ran,"
                     " they are printed, and they are not counted as independent failures because the"
                     " two sides were not answering the same question there. All of them, passes"
                     f" included, are in {record.name} beside the dumps; the same steps at identical"
                     " conditioning are judged by the same bands in section (b2), which has no echo"
                     " window and did gate.")
    return report()


def bf16_ulp(x: float) -> float:
    """One BF16 step at |x| — 2**(exponent − 7). Used to state a gap in ULP, which is the unit a
    near-tie question has to be asked in: 0.125 is nothing at 40 and two whole ULP at 15.6."""
    a = abs(float(x))
    if a == 0.0 or not math.isfinite(a):
        return float("nan")
    return math.ldexp(1.0, math.floor(math.log2(a)) - 7)


def decision_margin(ref: np.ndarray, eng, m: dict) -> str:
    """Why an argmax disagreed, said on the pin's own numbers — for the FAIL line of section (b).

    A gate must not assert what it did not check. The text this replaced printed "the pin is tied but the
    engine's id is not one of the tied ids" for EVERY unadmitted argmax failure, tied or not, and that
    false claim is what made `chunk64b` step 0 read as a broken fixture instead of what it is: a pinned
    winner 2 BF16 ULP ahead of the runner-up against an engine element error of 0.2509 (D-9). So this
    prints the two ids, the gap in ULP, the engine's values at both, and whether the engine's error is
    big enough that a legal route could flip the token — which is a fact, not a verdict.
    """
    order = np.argsort(ref)[::-1]
    top1, top2 = int(order[0]), int(order[1])
    v1, v2 = float(ref[top1]), float(ref[top2])
    gap = v1 - v2
    tied = int(np.count_nonzero(ref == ref.max()))
    if tied > 1:
        return (f" — the pin ties {tied} ways at {v1} (ids"
                f" {sorted(int(i) for i in np.flatnonzero(ref == ref.max()))}), and the tie policy did"
                " not admit the engine's id here — it is not one of the tied ids, or the engine's logit"
                " at it does not round back to the shared payload")
    ulps = gap / bf16_ulp(v1)
    verdict = ("LARGER than that margin, so a legal bf16 route can flip this token (§10.4's chaos at a"
               " near-tie; the logits band above still judges the vector)"
               if m["maxabs"] > gap else
               "smaller than that margin, so this is a real disagreement, not a rounding coin-flip")
    return (f" — the pin's winner is unique: top1 {top1} ({v1}) over top2 {top2} ({v2}), a gap of"
            f" {gap:.4g} = {ulps:.1f} BF16 ULP. The engine's own values there are"
            f" {float(eng[top1]):.6g} and {float(eng[top2]):.6g}, and its largest element error at"
            f" this step is {m['maxabs']:.4g}, " + verdict)


# EXPECTED-RED (D-9) — the failures this tree already knows about, NAMED, so a NEW failure stays
# distinguishable on a fixture that is red for a recorded reason. This changes no verdict and pardons
# nothing: a fixture listed here is still FAIL and still exits 1. It exists because a red that is absorbed
# is a red nobody reads any more.
#
# The list is EMPTY as of 2026-10-13, and the way it emptied is the story. `chunk64b` step 0 was its only
# entry: the pinned winner 8337 sat 2 BF16 ULP ahead of the runner-up 20619 while the engine's legal
# element error there is larger (maxabs 0.2509), so no legal bf16 route could be held to that token and
# D-9 kept the fixture RED rather than invent a threshold. What D-9 asked for was a SIZED rule — "the
# legal element error at that position plus a route spread, the way D-3/D-5/D-6 sized their bands" — and
# that is what the QUANTIZATION SLACK block above derives: bound = the meanrel cap this step is already
# policed by x mean|pin|, which at that step is 0.16205 against a margin of 0.125. The three recorded red
# lines did not go green by loosening anything; they go green because the token they demanded was one the
# pin never decided, and each of the three now says so with the numbers on the line. `engine/slack_size.py`
# prints the bound for every step of every fixture and `engine/results.md` carries the table; the rule
# reaches no step outside `chunk64b`, and `--break-runnerup` at `dec300f` step 30 — the corpus's nearest
# decidable step, 1.46x its own bound — is the injection that proves it.
#
# `wide520f` was on this list for exactly one commit and is not any more, which is worth recording because
# the trip came out of the ceiling this package removed: 500 prompt + 24 generation = 524 lanes, the
# one-pass attention stage served 512 (`MAXSC`, `prototype/attn_decode/attn_decode_kernels.h:32`), so the
# lane preflight refused the run and `decode_run` exited 1 before a single STEP line — which is also why
# `wide520f`'s y0, id, logits, `(g)` and `(h)` rows had never been judged at all. P7 (§07.3) made the
# engine take the prototype's split-K online form above the staging, and the two lines healed whole,
# observed with the entry still in place: `[expected-red] wide520f: all 2 recorded line(s) did NOT fail
# this run — healed`, on a run of 320 checks with 0 failures. That reporting line is the reason a heal on
# a fully green fixture is visible at all: without it a healed fixture and an unread list look identical.
# The mechanism stays for that reason even with nothing on it: a recorded red that heals should still print
# `healed` when somebody removes its entry after seeing it, and a NEW failure on a fixture that goes red
# for a reason this list has not been told about still has to say so.
EXPECTED_RED: dict = {}


def admitted_divergences(n_decode: int, got_ids: list, greedy: list, eng_logits: list,
                         dec_logits: list, allow_steps: dict, y0, y0_matches: bool,
                         y0_ok: bool, y0_room_ok: bool) -> dict:
    """Which engine-vs-pin divergences this run EXPLICITLY ADMITTED, and by which rule.

    Returns `{step: (engine_id, pin_id, rule)}`, with `PREFILL_ROW` for §06.5's first answer token.
    One function because the echo attribution needs the whole picture at once and the three rules print
    from three different loops: an attribution computed inside the loop that finds it would classify a
    step against a half-filled map (the room rule runs last, so a slack line printed above it could not
    know about a flip admitted below). Pure — the same three predicates, the same guards, no printing —
    and `main` asserts the map against what the three loops admit on their own lines, so a guard added
    in one place and not the other cannot drift quietly. Precedence matches the loops: tie, then slack,
    then room, first match wins.
    """
    admitted: dict[int, tuple] = {}
    if y0 is not None and not y0_matches and (y0_ok or y0_room_ok):
        admitted[PREFILL_ROW] = (y0, greedy[0],
                                 "exact-tie policy" if y0_ok else "cascade room rule")
    for j in range(min(n_decode, len(got_ids))):
        eng = eng_logits[j][0]
        kind, admissible, _ = tie_verdict(j, eng, got_ids[j])
        if kind == "tie" and admissible:
            admitted[j] = (got_ids[j], greedy[j + 1], "exact-tie policy")
            continue
        sl = step_slack(j, dec_logits[j], allow_steps.get(j))
        if sl is not None and sl["margin"] < sl["bound"] and got_ids[j] != sl["top1"]:
            if slack_verdict(sl, eng, got_ids[j])[0]:
                admitted[j] = (got_ids[j], greedy[j + 1], "quantization-slack rule")
                continue
        if route_is_library() and got_ids[j] != greedy[j + 1]:
            if room_verdict(f"step {j}", dec_logits[j], eng, got_ids[j])[0]:
                admitted[j] = (got_ids[j], greedy[j + 1], "cascade room rule")
    return admitted


def teacher_forced_checks(args, man: dict, prompt_ids: list, dec_logits: list, greedy: list,
                         n_decode: int, allowance, work: Path, vocab: int, replay: bool) -> None:
    """Section (b2): the same decode steps judged with the REFERENCE's own tokens fed at every step.

    WHY A SECOND RUN EXISTS. Section (a)/(b) run the engine free-running — the engine feeds itself — and
    that is the stronger claim when it holds. But one admitted thin-margin token at step N and the run is
    answering a different question from step N+1 on, so every band failure there is a fact about a prompt
    nobody asked, and section (a)/(b) have to call them ECHO rather than pretend. A real numeric defect
    must not be able to hide in that window, and the only way to be sure is to judge the same steps with
    both sides fed the same tokens: `decode_run --replay` with the pinned fed sequence, which is exactly
    what the deep fixtures already run (§A-18's bound). Here conditioning is identical by construction,
    so the section has NO echo window and every failure it prints is a real one.

    NO BAND IS WIDER HERE. Same `rule_for_step` caps, same G5 §10.1 floor, same §10.2 meanrel ceiling,
    same 1.50-ULP flip line, same three admission rules. The only thing that changes is what the engine
    was fed, and that is asserted from the run's own STEP lines rather than assumed.

    WHAT IT CANNOT CATCH, in one sentence: a defect that perturbs only the continuation of the engine's
    OWN different token — a state-carry fault that fires only once the engine is fed its own flip, say —
    is ECHOed by (a)/(b) and invisible here too, because this section never runs that chain, and no pin
    describes it (§10.5 rule 2: there is no reference for a prompt nobody fed). The free-run's own lines
    still print it, labelled ECHO with the flip blamed, so it is visible in the log and in the sidecar —
    it is simply not adjudicable, and it is the price of not printing fifteen red lines for one coin.

    Returns nothing; its checks are ordinary `check()` calls and they gate.
    """
    section("(b2) the same steps at identical conditioning — fed the reference's own tokens (§10.5"
            " rule 2, A-18's replay)")
    if replay:
        print("  [info] nothing added here: this fixture is deeper than"
              f" A-18's {FREE_RUN_BOUND}-token bound, so the main run above was ALREADY run with"
              " `--replay`. Sections (a)/(b) judged it at identical conditioning, there is no echo"
              " window to adjudicate around, and a second identical run would buy nothing.")
        return
    if n_decode == 0:
        print("  [info] nothing added here: this fixture has no decode phase, so there is no"
              " self-fed chain to re-feed. §06.5's prefill row is fed by the prompt, not by the"
              " engine, so it is already at identical conditioning in section (a).")
        return

    tf_work = work / "tf"
    tf_work.mkdir(parents=True, exist_ok=True)
    for stale in tf_work.glob("logits*.bin"):
        stale.unlink()          # same rule as the main run: judge this run, not the last one's leftovers
    fed_list = ",".join(str(t) for t in greedy[:-1])
    if args.dry_run:
        stdout, rc = dry_run(man, prompt_ids, dec_logits, tf_work, replay=True,
                             prefill_logits=args.prefill_logits, break_ids=args.break_ids,
                             break_logits=args.break_logits, break_noise=args.break_noise,
                             break_prompt_stop=args.break_prompt_stop, break_feed=args.break_feed,
                             break_tie=args.break_tie, tie_alt=args.tie_alt,
                             break_allow=args.break_allow, tie_same_ulp=args.tie_same_ulp,
                             tie_far=args.tie_far, break_band=args.break_band,
                             break_state_fp=args.break_state_fp, break_y0=args.break_y0,
                             y0_alt=args.y0_alt, slack_alt=args.slack_alt,
                             break_slack=args.break_slack, break_runnerup=args.break_runnerup,
                             break_pins=args.break_pins, break_pin_names=args.break_pin_names,
                             break_heur_width=args.break_heur_width,
                             break_cand=args.break_cand)
        notes.append("section (b2) ran against the same CPU stand-in as the main run, replayed: it"
                     " proves this section's plumbing parses and can FAIL, and nothing about the engine.")
    else:
        cmd = [args.binary, "--model", args.model, "--ids", ",".join(map(str, prompt_ids)),
               "--steps", str(n_decode), "--out", str(tf_work),
               "--prefill-logits", args.prefill_logits,
               "--gemm", GEMM_ROUTE, "--gemm-pins", GEMM_PINS,
               "--attn-tile", args.attn_tile, "--attn-touch", args.attn_touch,
               "--gdn-scan", args.gdn_scan, "--gdn-chunk", str(args.gdn_chunk),
               "--gdn-spell", args.gdn_spell,
               # THE ONE DIFFERENCE FROM THE MAIN RUN, and it is the whole section: the fed token at step
               # j is the reference's own, so the engine's sample is never fed back and every row is
               # comparable with the pin that was computed on that same token. `--state-fp` is deliberately
               # NOT passed: this section judges logits and ids, and the §08.6 fingerprints are the main
               # run's claim, already gated in section (g).
               "--replay", fed_list]
        print("  $ " + " ".join(cmd))
        proc = subprocess.run(cmd, capture_output=True, text=True)
        stdout, rc = proc.stdout, proc.returncode
        for line in proc.stdout.splitlines():
            print("      " + line)
    if not check(rc == 0, "the teacher-forced run exits 0", f"rc={rc}"):
        return
    tf = parse_run(stdout)
    tf_steps = [s for s in tf["steps"] if s["kind"] == DECODE_KIND]
    # A driver that parses another program's output must show it read what the program printed: an empty
    # list here would judge nothing and print a PASS over it (the `gdn_collapse/run.py` trap). The count
    # comes from the run's own RESULT line, so the two independent readings have to agree with each
    # other and with the number of steps this fixture has.
    printed = (tf["result"] or {}).get("steps")
    check(printed is not None and int(printed) == len(tf["steps"])
          and len(tf_steps) == n_decode,
          "the teacher-forced run printed, and this gate parsed, every step it printed, and the"
          f" generation half of them is the fixture's {n_decode}",
          f"RESULT says steps={printed}, {len(tf['steps'])} STEP lines parsed of"
          f" {len(prompt_ids) + n_decode} expected, of which {len(tf_steps)} are"
          f" kind={DECODE_KIND}")
    if len(tf_steps) != n_decode:
        notes.append("section (b2) stopped after the count check: it will not judge a partial replay"
                     " and call the missing steps green.")
        return
    mode = (tf["result"] or {}).get("mode")
    check(mode == "replay",
          "the run this section judges really was teacher-forced (RESULT mode=replay)",
          f"mode={mode} — the fed sequence this section is owed is the reference's own, and a run that"
          " fed itself is section (a)/(b)'s question, not this one")
    if mode != "replay":
        return
    tf_fed = [int(s["input"]) for s in tf_steps]
    if not check(tf_fed == greedy[:-1],
                 "every token this run was fed is the reference's own — the identical conditioning"
                 " this section exists to create, asserted from the run's STEP lines rather than"
                 " assumed from the flag",
                 f"fed {tf_fed} vs the pinned chain {greedy[:-1]}"):
        notes.append("section (b2)'s premise failed: the replay did not deliver the pinned fed"
                     " sequence, so nothing below it is a like-for-like comparison and no band line"
                     " here should be read as a result.")
        return
    tf_ids = [int(s["sampled"]) for s in tf_steps]
    eng_tf = read_engine_logits(tf_work, n_decode, len(prompt_ids), vocab)
    allow_steps = (allowance or {}).get("steps", {})
    # The same three admission rules, applied in the same order, with NO echo window anywhere: at
    # identical conditioning a divergence has no upstream cause to blame, which is the whole point.
    ties_b2, slack_b2, room_b2 = [], [], []
    admit: dict[int, bool] = {}
    for j in range(n_decode):
        eng, why = eng_tf[j]
        ref = dec_logits[j]
        if eng is None:
            check(False, f"b2 step {j}: engine wrote {tf_work.name}/logits{len(prompt_ids) + j}.bin",
                  why + " — check --out")
            continue
        if eng.shape != ref.shape:
            check(False, f"b2 step {j}: vocabulary length", f"{eng.shape} vs {ref.shape}")
            continue
        kind, ok_tie, line = tie_verdict(j, eng, tf_ids[j])
        if kind == "tie":
            check(ok_tie, f"b2 step {j}: the pin is an exact tie and the engine picked another id"
                          " — tested against the tie policy at identical conditioning", line)
            if ok_tie:
                admit[j] = True
                ties_b2.append(f"step {j} -> {tf_ids[j]}")
        sl = step_slack(j, ref, allow_steps.get(j))
        if j not in admit and sl is not None and sl["margin"] < sl["bound"] \
                and tf_ids[j] != sl["top1"]:
            ok_sl, line = slack_verdict(sl, eng, tf_ids[j])
            check(ok_sl, f"b2 step {j}: the pin's winner is unique but its margin is under the element"
                         " error this step's band licenses — tested against the quantization-slack"
                         " rule at identical conditioning", line)
            if ok_sl:
                admit[j] = True
                slack_b2.append(f"step {j} -> {tf_ids[j]} (pin margin {sl['margin']:.6g}"
                                f" = {sl['ulps']:.1f} ULP vs bound {sl['bound']:.6g})")
        if j not in admit and route_is_library() and tf_ids[j] != greedy[j + 1]:
            ok_room, line = room_verdict(f"b2 step {j}", ref, eng, tf_ids[j])
            check(ok_room, f"b2 step {j}: the engine's id differs from the pin's where neither the tie"
                           " policy nor the slack rule covers it — tested against the cascade room"
                           " rule at identical conditioning", line)
            if ok_room:
                admit[j] = True
                room_b2.append(f"step {j} -> {tf_ids[j]}")
        check(tf_ids[j] == greedy[j + 1] or admit.get(j, False),
              f"b2 step {j} samples greedy_continuation[{j + 1}] with the reference's own token fed"
              " in", f"{tf_ids[j]} vs {greedy[j + 1]}"
              + ("" if tf_ids[j] == greedy[j + 1] else " — and nothing upstream admits it, because at"
                                                   " identical conditioning there is no upstream flip"
                                                   " to blame"))
        m = metrics(eng, ref)
        deficit = 1.0 - m["cos"]
        cos_cap, rel_cap, rule_line = rule_for_step(allowance, j)
        check(True, f"b2 step {j}: metrics at identical conditioning",
              f"cos {m['cos']:.6f} deficit {deficit:.6e} meanrel {m['meanrel']:.4%}"
              f" ndiff {m['ndiff']}/{ref.size} maxabs {m['maxabs']:.4g}"
              f" | fed {tf_fed[j]} (the pin's own) | rule: {rule_line}")
        ok_cos = deficit <= cos_cap
        check(ok_cos, f"b2 step {j}: cos deficit {deficit:.4e} inside the band in force"
                      f" {cos_cap:.4e} — at identical conditioning, so this is the number that"
                      " cannot be blamed on a forked chain",
              "" if ok_cos else _counterfactual(deficit, 1.0 - BAND["cos_hard"], "§10.1 G5 floor"))
        ok_rel = m["meanrel"] <= rel_cap
        check(ok_rel, f"b2 step {j}: meanrel {m['meanrel']:.4%} inside the band in force"
                      f" {rel_cap:.4%}",
              "" if ok_rel else _counterfactual(m["meanrel"], BAND["meanrel_hard"],
                                                "§10.2 ceiling"))
    for label, items in (("TIE", ties_b2), ("SLACK", slack_b2), ("ROOM", room_b2)):
        if items:
            notes.append(f"section (b2): {len(items)} step(s) needed the {label} rule even with the"
                         " reference's own tokens fed — the same rules as the free run, the same caps,"
                         " and no echo anywhere: " + "; ".join(items))
    if echoes:
        notes.append(f"section (b2) is the answer to the {len(echoes)} ECHO line(s) above: the same"
                     f" {n_decode} steps were re-run fed the reference's own tokens"
                     f" ({tf_work.name}/) and judged by the same bands with no echo window. If a"
                     " failure were a real numeric defect rather than the admitted flip's downstream"
                     " consequence, it is red here instead.")


def report() -> int:
    section("summary")
    for n in notes:
        print("  [note] " + n)
    if ties:
        print(f"  [TIE] {len(ties)} step(s) — generation steps and, since D-8, §06.5's prefill row"
              " among them — were exact ties in the pin and were decided by the tie policy: "
              + ", ".join(ties))
    if tie_notes:
        print(f"  [TIE-NOTE] {len(tie_notes)} step(s) had an exact tie in the pin that the engine"
              " matched outright — ordinary id matches, tie policy not consulted (D-1):"
              " " + ", ".join(tie_notes))
    if slack:
        print(f"  [SLACK] {len(slack)} step(s) — a unique winner in the pin, but its margin was smaller"
              " than the element error that step's own meanrel band licenses, so the pinned token was"
              " not the pin's to decide and the engine's id was one of the pin's ids inside the margin"
              " (D-9's sized rule; every figure it used is on the SLACK line itself): " + "; ".join(slack))
    if slack_notes:
        print(f"  [SLACK-NOTE] {len(slack_notes)} step(s) sat inside that region and the engine"
              " reproduced the pin's own id anyway — ordinary id matches, the slack rule was not"
              " consulted (D-1's scope rule): " + "; ".join(slack_notes))
    if rooms:
        print(f"  [ROOM] {len(rooms)} flipped token(s) admitted because that row's top-1/top-2 room was"
              f" under {CASCADE_BAND['flip_room_max_ulp']:.2f} BF16 ULP at its own magnitude — the"
              " thin-room class §A2 of progress/2026-10-14/route-c-audit-2026-10-14.md measured, where a legal bf16"
              " route may land either way (every figure is on the ROOM line, and only a `--gemm library`"
              " run can consult this rule): " + "; ".join(rooms))
    rec = EXPECTED_RED.get(REFS.name)
    if not fails and rec is not None:
        # The heal case D-9's reporting needs as much as the red case: a line goes on this list because it
        # is red for a named reason, and a fixture that becomes fully GREEN prints no label at all, so the
        # heal would be invisible — and an invisible heal is indistinguishable from a list that was never
        # read. Note only; it cannot make a red fixture green and it does not touch the exit code. When the
        # reason behind a heal is gone for good, the entry comes out of the list (P7 removed `wide520f`'
        # that way, after seeing this line print with the entry still in place).
        print(f"  [expected-red] {REFS.name}: all {len(rec['checks'])} recorded line(s) did NOT fail"
              " this run — healed. Re-read why they were red before believing green: " + rec["why"])
    # The deferred checks print here as well as in place, so a reader who jumps to the bottom cannot
    # mistake a deferred failure for a pass — and cannot mistake it for a form finding either.
    if deferred:
        print(f"  [DEFERRED] {len(deferred)} check(s) ran and are reported here but are NOT counted in"
              f" this verdict, because this gate was invoked with --scope {SCOPE}: "
              + WAVE_CARRY_REASON + ". They are neither form's result: `--scope full` counts them, and"
              " comparing the two forms' wave_carry_deferred.json files says whether a failure here is"
              " a form finding or a separate one about the wave carry.")
        for d in deferred:
            print("  - [DEFERRED] " + d)
    # The echo block is the same shape for the same reason: a reader who jumps to the bottom must not
    # read an echoed failure as a pass, and must not have to scroll to find out which flip caused it.
    if echoes:
        print(f"  [ECHO] {len(echoes)} check(s) failed but are NOT counted as independent failures:"
              f" they sit downstream of an explicitly admitted divergence, so the engine and the pin"
              f" were not answering the same question there. {len(echo_records)} check(s) in total"
              f" ran inside that window ({len(echo_records) - len(echoes)} of them passed). Every one"
              f" is listed in {ECHO_FILE_NAME} beside the dumps, and the same steps at identical"
              f" conditioning are judged in section (b2): {echoes[0].split(' — ')[0]}")
        for e in echoes:
            print("  - [ECHO] " + e)
    if fails:
        print(f"ENGINE GATE FAIL — {len(fails)} failed"
              + (f" ({len(ties)} step(s) reported TIE and not counted as failures"
                 if ties else "") + (f"{', ' if ties else '('}{len(slack)} step(s) admitted by the"
                                      " slack rule" if slack else "")
              + (f"{', ' if (ties or slack) else '('}{len(rooms)} flip(s) admitted as thin-room"
                 if rooms else "")
              + (")" if (ties or slack or rooms) else "")
              + (f"; {len(echoes)} more ECHO of an admitted flip and not counted" if echoes else ""))
        for f in fails:
            print("  - " + f)
        # D-9's reporting half: name the recorded failures so a NEW one on an already-red fixture stays
        # visible, and say it loudly enough that nobody reads "3 failed" on `chunk64b` and stops. Labels
        # only — none of this changes the exit code, and nothing here makes a red fixture green.
        recorded = EXPECTED_RED.get(REFS.name)
        if recorded is None:
            print(f"  [expected-red] none recorded for {REFS.name}, so all {len(fails)} failure(s)"
                  " above are new by construction")
        else:
            known = [n for n in fail_names if n in recorded["checks"]]
            unseen = [n for n in fail_names if n not in recorded["checks"]]
            healed = [c for c in recorded["checks"] if c not in fail_names]
            print(f"  [expected-red] {REFS.name}: {len(known)} of {len(fails)} — {recorded['why']}"
                  ". A label, not a pardon: this fixture is FAIL and exits 1.")
            if unseen:
                print(f"  [NEW-FAILURE] {len(unseen)} failure(s) are NOT in the recorded list: "
                      + "; ".join(unseen))
            if healed:
                print(f"  [expected-red] {len(healed)} recorded line(s) did NOT fail this run — the"
                      " fixture or the engine moved, so re-read D-9 before believing green here: "
                      + "; ".join(healed))
        return 1
    reported = ([f"{len(ties)} step(s) reported TIE"] if ties else []) \
        + ([f"{len(slack)} step(s) reported SLACK"] if slack else []) \
        + ([f"{len(rooms)} flip(s) reported ROOM"] if rooms else [])
    if deferred:
        # Never a bare PASS while a wave-width check went red: the count and the reason are on the
        # verdict line itself, so the exit code cannot be the only place the deferral is visible.
        reported.append(f"{len(deferred)} wave-width check(s) DEFERRED and belonging to neither form"
                        " (--scope form; see the [DEFERRED] lines above)")
    if echoes:
        # Same rule as the deferral above, and the same reason: an exit code of 0 must not be the only
        # place a whole class of failures is visible. The count names both halves so a reader can see the
        # size of the window, not just how much of it went red.
        reported.append(f"{len(echoes)} check(s) ECHO of an admitted flip ({len(echo_records)} judged"
                        " downstream of it; see the [ECHO] lines and section (b2)")
    print("ENGINE GATE PASS" + (f" — with {' and '.join(reported)}" if reported else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
